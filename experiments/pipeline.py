r"""A configurable re-implementation of the `ModeModel` reduction and
regression stages, so that design choices can be varied and scored.

The surrogate turns a residual vector
:math:`(r_j, \delta\phi_k) = (A_{\rm eob}/A_{\rm pn}, \phi_{\rm eob} - \phi_{\rm pn})`
into a small number of principal-component coefficients, and regresses
those against the intrinsic parameters. Everything between the residuals
and the mismatch is a modelling choice; this module makes each of them a
knob:

``weighting``
    How the residual vector is scaled before the PCA sees it. The
    production model concatenates the amplitude and phase blocks raw,
    which means the PCA optimizes an L2 norm in which a radian of phase
    and a unit of amplitude ratio count the same --- and at a 5 Hz
    starting frequency the phase block is thousands of radians wide, so
    it takes essentially the whole variance budget.
``detrend``
    How the linear-in-frequency part of the phase residual is removed
    before training.
``pc_scaling``
    How the principal-component coefficients are scaled to form the
    regression target. This sets the *relative weight each component
    carries in the network's loss*, which is the single most consequential
    choice in the pipeline and the least obviously parametrized one.
``regressor``
    What is fitted from parameters to component coefficients.
``n_components``, ``dtype``, ``features``
    The remaining structural choices.

Every variant is scored the same way: the PSD-weighted mismatch against
the EOB validation waveforms, marginalised over time shift and reference
phase, exactly as :class:`mlgw_bns.model_validation.ValidateModel` does.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
from sklearn.kernel_ridge import KernelRidge  # type: ignore
from sklearn.linear_model import Ridge  # type: ignore
from sklearn.neural_network import MLPRegressor  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from mlgw_bns.data_management import FDWaveforms, Residuals
from mlgw_bns.neural_network import Hyperparameters, TimeshiftsNN

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass
class Config:
    """One point in the design space. Defaults reproduce the production model."""

    #: "none" reproduces the production model; "block_std" equalises the
    #: amplitude and phase blocks; "mismatch" whitens each column by its
    #: contribution to the mismatch integral.
    weighting: str = "none"

    #: "predictor" reproduces the production model (subtract the learned
    #: merger time shift, then anchor at the first node); "wproject"
    #: removes the span of {1, f} in the mismatch-weighted inner product,
    #: which is exactly the subspace the mismatch marginalises over.
    detrend: str = "predictor"

    #: "eigen" reproduces the production model: divide by the largest
    #: absolute coefficient, then multiply by ``eigenvalue ** pc_exponent``.
    #: "uniform" scales every component by its standard deviation, which
    #: makes the regression loss proportional to the reconstruction error.
    #: "robust" is "eigen" with a high quantile in place of the maximum.
    pc_scaling: str = "eigen"
    pc_exponent: float = 0.01948145530324084

    #: "mlp" is the production network; "mlp_scaled" standardizes the
    #: regression targets; "krr" is an RBF kernel ridge regression;
    #: "rff_ridge" is the random-Fourier-feature approximation to it.
    regressor: str = "mlp"
    regressor_kwargs: dict = field(default_factory=dict)

    n_components: int = 30
    dtype: str = "float64"

    #: "raw" is (q, lambda_1, lambda_2, chi_1, chi_2), as used in
    #: production; "physical" reparametrizes to the combinations the
    #: waveform actually depends on at leading order.
    features: str = "raw"

    #: What the amplitude block actually holds. "pn_ratio" is production,
    #: :math:`A_{\rm eob} / A_{\rm pn}` with each waveform divided by its
    #: own PN baseline; "reference_ratio" divides every waveform by one
    #: fixed reference amplitude instead. The two agree wherever the PN
    #: amplitude is a smooth stand-in for the EOB one, and differ exactly
    #: where it is not --- for the (2,1) and (3,3) modes the PN amplitude
    #: has a deep minimum at a parameter-dependent frequency, and
    #: dividing by it there sends the ratio to twenty or sixty while the
    #: waveform itself does nothing remarkable.
    #:
    #: "softened_pn" is the one-parameter family joining the two. It keeps
    #: each waveform's own PN baseline but divides by a *floored* version
    #: of it,
    #:
    #: .. math::
    #:     D = \sqrt{A_{\rm pn}^2 + (\delta A_{\rm ref})^2}
    #:
    #: which equals :math:`|A_{\rm pn}|` wherever that is large and levels
    #: off at :math:`\delta A_{\rm ref}` where it is not. A divisor that
    #: is continuous and changes sign has to pass through zero, so the
    #: floor can only be put on the magnitude; the physical sign is
    #: carried by :math:`A_{\rm eob}`, which crosses zero smoothly on its
    #: own. :math:`\delta \to 0` recovers division by
    #: :math:`|A_{\rm pn}|`, and :math:`\delta \to \infty` recovers
    #: "reference_ratio".
    amplitude: str = "pn_ratio"

    #: :math:`\delta` above, as a fraction of the reference amplitude.
    #: Only read when ``amplitude`` is "softened_pn".
    amplitude_floor: float = 0.1

    n_train: Optional[int] = None

    def label(self) -> str:
        base = Config()
        parts = [
            f"{name}={getattr(self, name)}"
            for name in (
                "weighting",
                "detrend",
                "pc_scaling",
                "pc_exponent",
                "regressor",
                "n_components",
                "dtype",
                "features",
                "amplitude",
                "amplitude_floor",
                "n_train",
            )
            if getattr(self, name) != getattr(base, name)
        ]
        if self.regressor_kwargs:
            parts.append(
                "regressor_kwargs="
                + ",".join(f"{k}:{v}" for k, v in sorted(self.regressor_kwargs.items()))
            )
        return ", ".join(parts) if parts else "baseline"


# --------------------------------------------------------------------------
# feature maps
# --------------------------------------------------------------------------


def make_features(parameters: np.ndarray, kind: str) -> np.ndarray:
    r"""Map ``(q, lambda_1, lambda_2, chi_1, chi_2)`` to regression inputs.

    The ``"physical"`` map replaces the raw parameters with the
    combinations the residual actually depends on at leading order: the
    symmetric mass ratio, the effective spin :math:`\chi_{\rm eff}` and
    its antisymmetric partner, and the tidal parameters
    :math:`\tilde\Lambda`, :math:`\delta\tilde\Lambda`. The tidal
    deformabilities are also taken in the logarithm, since they are drawn
    uniformly over three decades while their effect on the waveform is
    far from linear in them.
    """
    q, lambda_1, lambda_2, chi_1, chi_2 = parameters.T

    if kind == "raw":
        return parameters

    eta = q / (1 + q) ** 2
    m_1 = q / (1 + q)
    m_2 = 1 / (1 + q)

    chi_effective = (m_1 * chi_1 + m_2 * chi_2)
    chi_antisymmetric = (chi_1 - chi_2) / 2

    lambda_tilde = (
        16
        / 13
        * (
            (12 * m_2 + m_1) * m_1**4 * lambda_1
            + (12 * m_1 + m_2) * m_2**4 * lambda_2
        )
    )
    delta_lambda = lambda_1 * m_1**4 - lambda_2 * m_2**4

    if kind == "physical":
        return np.column_stack(
            [
                eta,
                chi_effective,
                chi_antisymmetric,
                np.log(lambda_tilde),
                delta_lambda,
            ]
        )

    if kind == "physical_plus_raw":
        return np.column_stack(
            [
                eta,
                chi_effective,
                chi_antisymmetric,
                np.log(lambda_tilde),
                delta_lambda,
                np.log(lambda_1),
                np.log(lambda_2),
            ]
        )

    raise ValueError(f"Unknown feature map {kind!r}")


# --------------------------------------------------------------------------
# column weights
# --------------------------------------------------------------------------


def trapezoid_weights(frequencies: np.ndarray) -> np.ndarray:
    """Quadrature weights for a trapezoid rule on a non-uniform grid."""
    weights = np.zeros_like(frequencies)
    weights[1:-1] = (frequencies[2:] - frequencies[:-2]) / 2
    weights[0] = (frequencies[1] - frequencies[0]) / 2
    weights[-1] = (frequencies[-1] - frequencies[-2]) / 2
    return weights


def mismatch_column_weights(
    amplitude_frequencies_hz: np.ndarray,
    phase_frequencies_hz: np.ndarray,
    mean_eob_amplitude: np.ndarray,
    mean_pn_amplitude: np.ndarray,
    psd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Per-node weights making the L2 norm of the residual vector track
    the mismatch.

    To second order in small errors, the mismatch between a waveform and a
    perturbed copy of it is

    .. math::
        \mathcal{M} \simeq \tfrac12 \left[
            \operatorname{Var}_w(\delta \phi)
            + \operatorname{Var}_w(\delta \ln A) \right],
        \qquad
        w(f) \propto \frac{A^2(f)}{S_n(f)},

    so the natural inner product on the residual vector weights each node
    by :math:`w(f)\,\Delta f`. Two things follow, and both differ from
    what the production pipeline does.

    The phase block gets :math:`A_{\rm eob}^2 / S_n`, which concentrates
    the weight where the signal actually is rather than spreading it
    uniformly over nodes.

    The amplitude block is stored as the *ratio* :math:`r = A/A_{\rm pn}`
    rather than as :math:`\ln A`, and :math:`\delta \ln A = \delta r / r`,
    so its weight is :math:`w / r^2 \propto A_{\rm pn}^2 / S_n`. That is
    finite even where the EOB amplitude crosses zero --- which is exactly
    where the (2,1) and (3,3) residuals are worst behaved. The weighting
    de-emphasises those nodes on its own, without any special-casing.

    Each block is normalised to unit total weight, so that the squared
    distance between two residual vectors is approximately twice the
    mismatch between the waveforms they describe.
    """
    amplitude_weights = (
        mean_pn_amplitude**2
        / np.interp(amplitude_frequencies_hz, psd[:, 0], psd[:, 1])
        * trapezoid_weights(amplitude_frequencies_hz)
    )
    phase_weights = (
        mean_eob_amplitude**2
        / np.interp(phase_frequencies_hz, psd[:, 0], psd[:, 1])
        * trapezoid_weights(phase_frequencies_hz)
    )

    return (
        amplitude_weights / amplitude_weights.sum(),
        phase_weights / phase_weights.sum(),
    )


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------


class Surrogate:
    """Reduction + regression, assembled according to a :class:`Config`."""

    def __init__(
        self,
        config: Config,
        amplitude_frequencies_hz: np.ndarray,
        phase_frequencies_hz: np.ndarray,
        psd: np.ndarray,
        timeshifts_predictor: Optional[TimeshiftsNN] = None,
    ):
        self.config = config
        self.amplitude_frequencies_hz = amplitude_frequencies_hz
        self.phase_frequencies_hz = phase_frequencies_hz
        self.psd = psd
        self.timeshifts_predictor = timeshifts_predictor

    # -- residual vector assembly ------------------------------------------

    def _set_column_weights(
        self, amplitude_residuals: np.ndarray, phase_residuals: np.ndarray,
        mean_pn_amplitude: np.ndarray,
    ) -> None:
        n_amplitude = amplitude_residuals.shape[1]
        n_phase = phase_residuals.shape[1]

        if self.config.weighting == "none":
            self.amplitude_column_weights = np.ones(n_amplitude)
            self.phase_column_weights = np.ones(n_phase)
        elif self.config.weighting == "block_std":
            self.amplitude_column_weights = np.full(
                n_amplitude, 1 / max(amplitude_residuals.std(), 1e-300) ** 2
            )
            self.phase_column_weights = np.full(
                n_phase, 1 / max(phase_residuals.std(), 1e-300) ** 2
            )
        elif self.config.weighting == "mismatch":
            # The amplitude and phase blocks live on different node sets,
            # so the mean EOB amplitude is built where it is known --- on
            # the amplitude nodes --- and carried to the phase nodes by a
            # log-log interpolation, the amplitude spanning many decades.
            mean_eob_amplitude = (
                np.abs(np.mean(amplitude_residuals, axis=0)) * mean_pn_amplitude
            )
            mean_eob_amplitude_on_phase_nodes = np.exp(
                np.interp(
                    np.log(self.phase_frequencies_hz),
                    np.log(self.amplitude_frequencies_hz),
                    np.log(np.maximum(mean_eob_amplitude, 1e-300)),
                )
            )
            (
                self.amplitude_column_weights,
                self.phase_column_weights,
            ) = mismatch_column_weights(
                self.amplitude_frequencies_hz,
                self.phase_frequencies_hz,
                mean_eob_amplitude_on_phase_nodes,
                mean_pn_amplitude,
                self.psd,
            )
        else:
            raise ValueError(f"Unknown weighting {self.config.weighting!r}")

        self.column_scales = np.concatenate(
            [
                np.sqrt(self.amplitude_column_weights),
                np.sqrt(self.phase_column_weights),
            ]
        )

    @property
    def _linear_basis(self) -> np.ndarray:
        """The subspace of the phase residual the mismatch marginalises over.

        A constant phase and a term linear in frequency are exactly the
        two things :meth:`ValidateModel.mismatch` optimises away, so a
        model that reproduces the rest of the phase perfectly scores the
        same however badly it does on these two --- as long as the
        implied time shift stays inside the search window.
        """
        frequencies = self.phase_frequencies_hz
        return np.column_stack([np.ones_like(frequencies), frequencies])

    def _detrend(
        self, parameters: np.ndarray, phase_residuals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split the phase residual into a linear part and the rest.

        Returns the detrended residuals and the coefficients of whatever
        was taken out, so that :meth:`_retrend` can put back exactly the
        same thing (or a prediction of it).
        """
        phase_residuals = np.array(phase_residuals, copy=True)
        frequencies = self.phase_frequencies_hz

        if self.config.detrend == "predictor":
            assert self.timeshifts_predictor is not None
            time_shifts = self.timeshifts_predictor.predict(parameters).reshape(-1, 1)
            linear = 2 * np.pi * (frequencies - frequencies[0]) * time_shifts
            phase_residuals = phase_residuals - linear
            constants = phase_residuals[:, :1]
            coefficients = np.column_stack(
                [constants[:, 0], 2 * np.pi * time_shifts[:, 0]]
            )
            return phase_residuals - constants, coefficients

        if self.config.detrend == "wproject":
            # Weighted least squares in the same inner product the
            # mismatch uses, rather than a fit over the lowest fifth of
            # the nodes: the term to remove is the one the metric cannot
            # see, so it should be defined by the metric.
            basis = self._linear_basis
            weights = self.phase_column_weights
            gram = basis.T @ (weights[:, None] * basis)
            projector = np.linalg.solve(gram, basis.T * weights[None, :])
            coefficients = phase_residuals @ projector.T
            return phase_residuals - coefficients @ basis.T, coefficients

        raise ValueError(f"Unknown detrend {self.config.detrend!r}")

    def _softened_divisor(self, pn_amplitudes: np.ndarray) -> np.ndarray:
        r"""``sqrt(A_pn**2 + (delta * A_ref)**2)``, a floored ``|A_pn|``.

        The floor has to be frequency-dependent: the PN amplitude falls by
        orders of magnitude across the band, so a single number would
        either do nothing at low frequency or swamp the signal at high
        frequency. Taking it as a fraction of the reference amplitude ---
        which has the right shape and no zero --- makes it the same
        *relative* floor everywhere.
        """
        floor = self.config.amplitude_floor * self.reference_amplitude[None, :]
        return np.sqrt(pn_amplitudes**2 + floor**2)

    def _to_modelled_amplitude(
        self, amplitude_residuals: np.ndarray, pn_amplitudes: Optional[np.ndarray]
    ) -> np.ndarray:
        """Turn cached ``A_eob / A_pn`` into whatever is being modelled."""
        if self.config.amplitude == "pn_ratio":
            return amplitude_residuals
        assert pn_amplitudes is not None
        eob_amplitudes = amplitude_residuals * pn_amplitudes
        if self.config.amplitude == "reference_ratio":
            return eob_amplitudes / self.reference_amplitude[None, :]
        if self.config.amplitude == "softened_pn":
            return eob_amplitudes / self._softened_divisor(pn_amplitudes)
        raise ValueError(f"Unknown amplitude {self.config.amplitude!r}")

    def _from_modelled_amplitude(
        self, modelled: np.ndarray, pn_amplitudes: Optional[np.ndarray]
    ) -> np.ndarray:
        """Invert :meth:`_to_modelled_amplitude`, back to ``A_eob / A_pn``."""
        if self.config.amplitude == "pn_ratio":
            return modelled
        assert pn_amplitudes is not None
        if self.config.amplitude == "softened_pn":
            divisor = self._softened_divisor(pn_amplitudes)
        else:
            divisor = self.reference_amplitude[None, :]
        return modelled * divisor / pn_amplitudes

    def _combine(
        self, amplitude_residuals: np.ndarray, phase_residuals: np.ndarray
    ) -> np.ndarray:
        return (
            np.concatenate([amplitude_residuals, phase_residuals], axis=1)
            * self.column_scales[None, :]
        )

    def _split(self, combined: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        unscaled = combined / self.column_scales[None, :]
        n_amplitude = len(self.amplitude_column_weights)
        return unscaled[:, :n_amplitude], unscaled[:, n_amplitude:]

    # -- fitting -----------------------------------------------------------

    def fit(
        self,
        parameters: np.ndarray,
        amplitude_residuals: np.ndarray,
        phase_residuals: np.ndarray,
        mean_pn_amplitude: np.ndarray,
        fit_regressor: bool = True,
        pn_amplitudes: Optional[np.ndarray] = None,
    ) -> "Surrogate":
        dtype = np.dtype(self.config.dtype)
        amplitude_residuals = amplitude_residuals.astype(dtype)
        phase_residuals = phase_residuals.astype(dtype)

        self.reference_amplitude = mean_pn_amplitude
        amplitude_residuals = self._to_modelled_amplitude(
            amplitude_residuals, pn_amplitudes
        )

        self._set_column_weights(
            amplitude_residuals, phase_residuals, mean_pn_amplitude
        )
        phase_residuals, linear_coefficients = self._detrend(
            parameters, phase_residuals
        )
        data = self._combine(amplitude_residuals, phase_residuals).astype(dtype)

        self.mean = data.mean(axis=0)
        centred = data - self.mean
        _, singular_values, right = np.linalg.svd(centred, full_matrices=False)
        k = self.config.n_components
        self.eigenvectors = right[:k].T.astype(np.float64)
        self.eigenvalues = (singular_values[:k] ** 2).astype(np.float64)

        coefficients = (centred.astype(np.float64)) @ self.eigenvectors
        self.pc_scale = self._pc_scale(coefficients)

        # Under "wproject" the linear-in-frequency term is fitted per
        # waveform, so it has to be regressed alongside the principal
        # components for the decomposition and the reconstruction to be
        # exact inverses. Under "predictor" it is already a function of
        # the parameters --- the shared time-shift predictor --- and is
        # recovered from there instead, which is what production does.
        self.regress_linear = self.config.detrend == "wproject"
        self.linear_mean = linear_coefficients.mean(axis=0)
        self.linear_scale = np.where(
            linear_coefficients.std(axis=0) > 0, linear_coefficients.std(axis=0), 1.0
        )

        if fit_regressor:
            targets = coefficients / self.pc_scale[None, :]
            if self.regress_linear:
                targets = np.column_stack(
                    [
                        targets,
                        (linear_coefficients - self.linear_mean) / self.linear_scale,
                    ]
                )
            features = make_features(parameters, self.config.features)
            self.regressor_object = self._make_regressor(len(parameters))
            self.regressor_object.fit(features, targets)
        return self

    def _pc_scale(self, coefficients: np.ndarray) -> np.ndarray:
        r"""The divisor applied to each principal-component coefficient
        before it becomes a regression target.

        This is the pipeline's loss weighting in disguise. A regressor
        minimizing a plain mean squared error over the scaled targets is
        minimizing :math:`\sum_i (\Delta x_i / s_i)^2`, while the
        reconstruction error in residual space is :math:`\sum_i (\Delta
        x_i)^2` --- so the component-:math:`i` error is weighted by
        :math:`s_i^{-2}`, and the two agree only when :math:`s_i` is the
        same for every component.

        The options here span the range. ``"flat"`` is that agreement.
        ``"eigen"`` is production, :math:`s_i = \max|x_i| \lambda_i^{-\alpha}`;
        since :math:`\max|x_i| \propto \sqrt{\lambda_i}`, it lands on
        ``"flat"`` at :math:`\alpha = 1/2` and at :math:`\alpha \approx 0.02`
        it is nearly the opposite. ``"uniform"`` divides by the standard
        deviation, which is that opposite taken all the way: it asks for
        the same *relative* accuracy on the thirtieth component as on the
        first, though the thirtieth carries :math:`10^{-11}` of the
        variance.

        Note that none of this reaches a regressor that is linear in its
        targets. Kernel ridge solves :math:`(K + \alpha I)^{-1} y` per
        output, so scaling an output scales its solution exactly and
        changes no relative error. The weighting is a live issue for the
        network, which fits all thirty outputs jointly against one loss,
        and is a reason to prefer a per-output linear solver.
        """
        maximum = np.max(np.abs(coefficients), axis=0)
        if self.config.pc_scaling == "eigen":
            return maximum / self.eigenvalues**self.config.pc_exponent
        if self.config.pc_scaling == "robust":
            quantile = np.quantile(np.abs(coefficients), 0.99, axis=0)
            return quantile / self.eigenvalues**self.config.pc_exponent
        if self.config.pc_scaling == "uniform":
            return coefficients.std(axis=0)
        if self.config.pc_scaling == "flat":
            # One scale for every component, so that the mean squared
            # error over the targets *is* the reconstruction error in
            # residual space. This is the only choice under which the
            # thing the network minimizes and the thing that shows up in
            # the mismatch are the same quantity.
            return np.full(len(self.eigenvalues), maximum[0])
        if self.config.pc_scaling == "maxabs":
            return maximum
        raise ValueError(f"Unknown pc_scaling {self.config.pc_scaling!r}")

    def _make_regressor(self, n_train: int):
        kwargs = dict(self.config.regressor_kwargs)
        if self.config.regressor == "mlp":
            hyper = Hyperparameters.default(n_train)
            hyper.n_train = n_train
            hyper.max_iter *= 10
            params = hyper.nn_params
            params.update(kwargs)
            return _ScaledInput(MLPRegressor(**params), scale_targets=False)
        if self.config.regressor == "mlp_scaled":
            hyper = Hyperparameters.default(n_train)
            hyper.n_train = n_train
            hyper.max_iter *= 10
            params = hyper.nn_params
            params.update(kwargs)
            return _ScaledInput(MLPRegressor(**params), scale_targets=True)
        if self.config.regressor == "mlp_global":
            hyper = Hyperparameters.default(n_train)
            hyper.n_train = n_train
            hyper.max_iter *= 10
            params = hyper.nn_params
            params.update(kwargs)
            return _ScaledInput(MLPRegressor(**params), scale_targets="global")
        if self.config.regressor == "krr":
            params = dict(kernel="rbf", alpha=1e-8, gamma=0.2)
            params.update(kwargs)
            return _ScaledInput(KernelRidge(**params), scale_targets=True)
        if self.config.regressor == "rff_ridge":
            from sklearn.kernel_approximation import RBFSampler  # type: ignore
            from sklearn.pipeline import Pipeline  # type: ignore

            params = dict(n_components=2000, gamma=0.2, alpha=1e-8)
            params.update(kwargs)
            return _ScaledInput(
                Pipeline(
                    [
                        (
                            "rff",
                            RBFSampler(
                                n_components=params["n_components"],
                                gamma=params["gamma"],
                                random_state=42,
                            ),
                        ),
                        ("ridge", Ridge(alpha=params["alpha"])),
                    ]
                ),
                scale_targets=True,
            )
        raise ValueError(f"Unknown regressor {self.config.regressor!r}")

    # -- prediction --------------------------------------------------------

    def predict_residuals(
        self, parameters: np.ndarray, pn_amplitudes: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        features = make_features(parameters, self.config.features)
        prediction = self.regressor_object.predict(features)
        k = self.config.n_components
        coefficients = prediction[:, :k] * self.pc_scale[None, :]
        if self.regress_linear:
            linear_coefficients = (
                prediction[:, k:] * self.linear_scale + self.linear_mean
            )
        else:
            linear_coefficients = np.zeros((len(parameters), 2))

        data = coefficients @ self.eigenvectors.T + self.mean
        amplitude_residuals, phase_residuals = self._split(data)
        return (
            self._from_modelled_amplitude(amplitude_residuals, pn_amplitudes),
            self._retrend(parameters, phase_residuals, linear_coefficients),
        )

    def project_residuals(
        self,
        parameters: np.ndarray,
        amplitude_residuals: np.ndarray,
        phase_residuals: np.ndarray,
        pn_amplitudes: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Round-trip residuals through the PCA basis without the regressor.

        The gap between this and the true residuals is the floor the
        regression can never beat: whatever the network learns, the
        prediction lives in the span of the retained components.
        """
        phase_flat, linear_coefficients = self._detrend(parameters, phase_residuals)
        modelled = self._to_modelled_amplitude(amplitude_residuals, pn_amplitudes)
        data = self._combine(modelled, phase_flat)
        coefficients = (data - self.mean) @ self.eigenvectors
        reconstructed = coefficients @ self.eigenvectors.T + self.mean
        amplitude_out, phase_out = self._split(reconstructed)
        return (
            self._from_modelled_amplitude(amplitude_out, pn_amplitudes),
            self._retrend(parameters, phase_out, linear_coefficients),
        )

    def _retrend(
        self,
        parameters: np.ndarray,
        phase_residuals: np.ndarray,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        """Put the linear-in-frequency term back onto the phase residual."""
        if self.config.detrend == "predictor":
            assert self.timeshifts_predictor is not None
            frequencies = self.phase_frequencies_hz
            time_shifts = self.timeshifts_predictor.predict(parameters)
            return phase_residuals + np.outer(
                2 * np.pi * time_shifts, frequencies - frequencies[0]
            )
        return phase_residuals + coefficients @ self._linear_basis.T


class _ScaledInput:
    """Standardize the inputs (and optionally the targets) around a regressor.

    scikit-learn's :class:`MLPRegressor` standardizes neither, and the
    production pipeline only scales the inputs. Whether the targets
    should be standardized too is one of the things being tested, so it
    is a flag rather than a fixed choice.

    Note that standardizing the targets subsumes :attr:`Config.pc_scaling`:
    dividing ``coefficients / pc_scale`` by its own standard deviation
    leaves ``coefficients / std(coefficients)``, whatever ``pc_scale``
    was. So ``pc_scaling`` is only a live choice for the production
    ``"mlp"``, which is the one regressor here that does not scale what
    it is asked to predict.
    """

    def __init__(self, regressor, scale_targets):
        self.regressor = regressor
        #: ``False``/``"none"`` leaves the targets as ``pc_scale`` made
        #: them; ``True``/``"standard"`` standardizes each output
        #: separately, which overrides ``pc_scale`` with the
        #: ``"uniform"`` weighting; ``"global"`` centres each output but
        #: divides them all by one number, which preserves whatever
        #: relative weighting ``pc_scale`` set up.
        self.scale_targets = (
            "standard" if scale_targets is True else scale_targets or "none"
        )

    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> "_ScaledInput":
        self.input_scaler = StandardScaler().fit(x_data)
        if self.scale_targets == "standard":
            self.target_scaler = StandardScaler().fit(y_data)
            y_data = self.target_scaler.transform(y_data)
        elif self.scale_targets == "global":
            self.target_mean = y_data.mean(axis=0)
            self.target_std = float(np.std(y_data - self.target_mean))
            y_data = (y_data - self.target_mean) / self.target_std
        self.regressor.fit(self.input_scaler.transform(x_data), y_data)
        return self

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        prediction = self.regressor.predict(self.input_scaler.transform(x_data))
        if self.scale_targets == "standard":
            prediction = self.target_scaler.inverse_transform(prediction)
        elif self.scale_targets == "global":
            prediction = prediction * self.target_std + self.target_mean
        return prediction
