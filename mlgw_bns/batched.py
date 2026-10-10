r"""Batched, per-mode evaluation of the surrogate, on numpy or JAX.

:meth:`Model.predict <mlgw_bns.model.Model.predict>` and its relatives
evaluate one parameter set at a time, as a chain of scikit-learn, SciPy
and Numba calls. This module evaluates ``N`` parameter sets in one call
and returns each requested mode separately, as amplitude and phase. The
same code runs on :mod:`numpy` and on :mod:`jax.numpy`: every stage is
written against an array namespace ``xp``, as the TaylorF2 phase in
:mod:`mlgw_bns.taylorf2` already is.

Entry points
------------
* :class:`BatchedSurrogate` --- the frozen model, built once for a set
  of modes and a backend (``"numpy"`` or ``"jax"``) and then called as
  ``surrogate(intrinsic, total_mass, frequencies)``.
  :meth:`Model.predict_modes_amp_phase
  <mlgw_bns.model.Model.predict_modes_amp_phase>` (numpy) and
  :meth:`Model.jax_modes_amp_phase
  <mlgw_bns.model.Model.jax_modes_amp_phase>` (a pure JAX function)
  build and cache one.
* :func:`mode_polarizations` --- the spherical-harmonic projection of the
  modes onto :math:`h_+` and :math:`h_\times`, also batched and generic.

Conventions
-----------
Everything below is the convention of :meth:`Model.predict_modes_dict
<mlgw_bns.model.Model.predict_modes_dict>`, which the batched evaluation
reproduces.

* **Mode.** For each requested :math:`(\ell, m)`, with :math:`m > 0`,
  the returned ``amp`` and ``phase`` define the frequency-domain multipole

  .. math::
      \tilde{h}_{\ell m}(f) = A_{\ell m}(f)\, e^{+i \phi_{\ell m}(f)} ,

  in units of 1/Hz at the given luminosity distance. The negative-:math:`m`
  multipoles have their support at negative frequencies, and are not
  returned.
* **Projection.** With :math:`s = (-1)^\ell` and the spin-weighted
  harmonics :math:`{}_{-2}Y_{\ell m}(\iota, \varphi)`,

  .. math::
      h_+ = \tfrac{1}{2} \tilde{h}_{\ell m}
            \left(Y_{\ell m} + s\, Y^*_{\ell, -m}\right), \qquad
      h_\times = \tfrac{i}{2} \tilde{h}_{\ell m}
            \left(Y_{\ell m} - s\, Y^*_{\ell, -m}\right),

  so that :math:`h_+ - i h_\times = \tilde{h}_{\ell m} Y_{\ell m}`, summed
  over the modes. :func:`mode_polarizations` implements exactly this. No
  angular factor, reference phase or polarisation convention is applied
  to ``amp`` and ``phase`` themselves.
* **Time and phase.** The phase increases with frequency during the
  inspiral, and the time at which mode :math:`m` emits frequency
  :math:`f` is

  .. math::
      t_{\ell m}(f) = -\frac{1}{2\pi} \frac{\mathrm{d} \phi_{\ell m}}{\mathrm{d} f},

  negative before the merger (``return_tf=True`` returns it). The merger
  is at :math:`t = 0` with coalescence phase zero: after the merger the
  frequency-domain phase of the (2,2) mode is linear in :math:`f`, and
  every mode is shifted in time and rotated in orbital phase (by
  :math:`m/2` times the (2,2) rotation) so that this line, the tangent to
  the (2,2) phase at the top of the trained band, is
  :math:`\phi_{22} = 0`. The (2,2) is evaluated for this even when it is
  not requested. See :meth:`ModeModel.merger_reference
  <mlgw_bns.mode_model.ModeModel.merger_reference>`; a merger time
  :math:`t_c` and coalescence phase :math:`\phi_c` are applied by adding
  :math:`m \phi_c - 2\pi f t_c` to the phases.
* **Frequencies.** Below the trained band (``f M / M_ref`` under
  ``effective_initial_frequency_hz``) each mode is continued with its
  post-Newtonian amplitude and phase, blended in as in
  :meth:`ModeModel.predict_amplitude_phase
  <mlgw_bns.mode_model.ModeModel.predict_amplitude_phase>`; above
  it the amplitude is zero. Non-positive frequencies give zero amplitude
  and phase. The frequencies must increase along the last axis.
* **Out-of-range rows.** A row whose parameters fall outside the model's
  :attr:`parameter ranges <mlgw_bns.model.Model.parameter_ranges>`, or
  whose frequencies need an extension the model is configured not to
  make, is returned as NaN rather than raising, so that one bad row does
  not abort a batch; :meth:`BatchedSurrogate.valid` gives the mask.

Numerical accuracy
------------------
The kernel-ridge prediction ``K @ dual_coef`` is a cancelling sum: its
rounding error depends on the order in which it is carried out, and so on
the BLAS kernel, the batch size and the backend. This is the floor for the
agreement between numpy and JAX, or between batched and single
evaluations; it is far smaller for regressors fitted with the
leave-one-out penalty selection of
:class:`~mlgw_bns.neural_network.KernelRidgeNetwork` than for those with
a vanishing penalty. With ``N = 1`` on numpy, :class:`BatchedSurrogate`
performs the regressor evaluation with the same operations as
:meth:`KernelRidgeNetwork.predict
<mlgw_bns.neural_network.KernelRidgeNetwork.predict>`, and reproduces
:meth:`Model.predict_modes_dict <mlgw_bns.model.Model.predict_modes_dict>`
to rounding. JAX must run in double precision (this module enables
``jax_enable_x64`` when it builds a JAX surrogate).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np
from scipy.interpolate import CubicSpline  # type: ignore

from .dataset_generation import AMP_SI_BASE
from .jax_mlp import array_activation
from .pn_modes import H_21, H_22, H_31, H_32, H_33, H_43, H_44, Mode
from .special_func import wigner_d_function
from .taylorf2 import _make_taylorf2_psi

if TYPE_CHECKING:
    from .model import Model

__all__ = [
    "BatchedSurrogate",
    "mode_polarizations",
    "spin_weighted_spherical_harmonic",
]

_H_BY_MODE: dict[tuple[int, int], Callable] = {
    (2, 2): H_22,
    (2, 1): H_21,
    (3, 1): H_31,
    (3, 2): H_32,
    (3, 3): H_33,
    (4, 3): H_43,
    (4, 4): H_44,
}

#: Relative frequency step of the central difference giving the
#: post-Newtonian :math:`\mathrm{d}\phi/\mathrm{d}f` below the trained band
#: (``return_tf=True``). Near the optimum :math:`\epsilon^{1/3}` for a
#: phase of order :math:`10^6` rad: rounding and truncation errors are both
#: around :math:`10^{-11}` relative.
_TF_RELATIVE_STEP = 5e-6

_PSI_BY_BACKEND: dict[str, Callable] = {}


def _namespace(backend: str):
    """``(xp, taylorf2_psi)`` for ``"numpy"`` or ``"jax"``."""
    if backend == "numpy":
        xp = np
    elif backend == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as xp  # type: ignore[no-redef]
    else:
        raise ValueError(f"Unknown backend {backend!r}: expected 'numpy' or 'jax'.")
    if backend not in _PSI_BY_BACKEND:
        _PSI_BY_BACKEND[backend] = _make_taylorf2_psi(xp)
    return xp, _PSI_BY_BACKEND[backend]


# ====================================================================== #
# Post-Newtonian building blocks, generic over ``xp``
# ====================================================================== #


def _lead_argument(xp, lm, eta, delta, chi_a, chi_s):
    r"""Argument of the leading coefficient of :math:`H_{\ell m}`, snapped to
    :math:`\{0, \pi, \pm\pi/2\}` as in :func:`mlgw_bns.pn_modes.psi_lm`."""
    lead = _H_BY_MODE[lm](1e-8, eta, delta, chi_a, chi_s)
    lr, li = xp.real(lead), xp.imag(lead)
    return xp.where(
        lr >= xp.abs(li),
        0.0,
        xp.where(lr <= -xp.abs(li), math.pi, xp.copysign(math.pi / 2, li)),
    )


@dataclass
class _PNRequest:
    r"""One post-Newtonian evaluation of mode ``lm`` at ``f_natural``.

    ``f_natural`` broadcasts to ``(N, n)``. With ``unwrap=True`` the
    argument of :math:`H_{\ell m}` is unwrapped along the last axis and
    snapped to the leading coefficient's branch at the first element,
    exactly as :func:`mlgw_bns.pn_modes.psi_lm` does; with ``unwrap=False``
    each element is snapped to that branch on its own, which is equivalent
    wherever :math:`\arg H_{\ell m}` stays within :math:`\pi` of its
    leading value --- in particular below the trained band, where each
    row's grid is different.
    """

    lm: tuple
    f_natural: Any
    unwrap: bool = False
    amplitude: bool = False


def _post_newtonian(xp, psi, x, requests: list) -> list:
    r"""Per-mode post-Newtonian phases (and amplitudes) for ``N`` rows.

    :func:`mlgw_bns.pn_modes.psi_lm` and :func:`~mlgw_bns.pn_modes.amp_lm`,
    for the parameter rows ``x`` of shape ``(N, 5)``. All the TaylorF2
    phases are computed in a single call and all the :math:`H_{\ell m}`
    of one mode in another, which is what keeps small batches cheap.

    Returns one ``(amplitude or None, phase)`` pair of ``(N, n)`` arrays
    per request.
    """
    n_rows = x.shape[0]
    q, lam1, lam2, chi1, chi2 = (x[:, i : i + 1] for i in range(5))
    eta = q / (1.0 + q) ** 2
    delta = (q - 1) / (q + 1)
    chi_a = (chi1 - chi2) / 2
    chi_s = (chi1 + chi2) / 2

    freqs = [
        xp.broadcast_to(r.f_natural, (n_rows, r.f_natural.shape[-1])) for r in requests
    ]
    sizes = [f.shape[1] for f in freqs]
    splits = list(np.cumsum(sizes)[:-1])

    mode_freq = xp.concatenate(
        [2 * f / r.lm[1] for r, f in zip(requests, freqs)], axis=1
    )
    orbital = xp.split(psi(mode_freq, eta, chi1, chi2, lam1, lam2), splits, axis=1)

    results: list = [None] * len(requests)
    for lm in dict.fromkeys(r.lm for r in requests):
        indices = [i for i, r in enumerate(requests) if r.lm == lm]
        m = lm[1]
        v_all = xp.concatenate(
            [xp.abs(2 * np.pi * freqs[i] / m) ** (1.0 / 3.0) for i in indices], axis=1
        )
        h_all = _H_BY_MODE[lm](v_all, eta, delta, chi_a, chi_s)
        lead_arg = _lead_argument(xp, lm, eta, delta, chi_a, chi_s)
        mode_splits = list(np.cumsum([sizes[i] for i in indices])[:-1])
        for i, v, h in zip(
            indices,
            xp.split(v_all, mode_splits, axis=1),
            xp.split(h_all, mode_splits, axis=1),
        ):
            arg_h = xp.angle(h)
            if requests[i].unwrap:
                arg_h = xp.unwrap(arg_h, axis=-1)
                arg_h = arg_h + 2 * np.pi * xp.round(
                    (lead_arg - arg_h[:, :1]) / (2 * np.pi)
                )
            else:
                arg_h = arg_h + 2 * np.pi * xp.round((lead_arg - arg_h) / (2 * np.pi))
            phase = orbital[i] * (m / 2) + arg_h
            amplitude = None
            if requests[i].amplitude:
                prefactor = np.pi * xp.sqrt(2 * eta / 3) * v ** (-7 / 2)
                amplitude = (
                    prefactor * xp.real(h) if lm == (4, 4) else xp.abs(prefactor * h)
                )
            results[i] = (amplitude, phase)
    return results


# ====================================================================== #
# Spherical harmonics and the projection onto the polarizations
# ====================================================================== #


def spin_weighted_spherical_harmonic(ell, emm, inclination, azimuth, xp=np):
    r""":math:`{}_{-2}Y_{\ell m}(\iota, \varphi)`, as a complex array.

    The convention of :func:`mlgw_bns.special_func.spinsphericalharm`
    (with spin weight :math:`s = -2`). ``inclination`` and ``azimuth``
    broadcast against each other.
    """
    d = math.sqrt((2 * ell + 1) / (4 * math.pi)) * wigner_d_function(
        ell, emm, 2, inclination, xp=xp
    )
    return d * xp.exp(1j * emm * xp.asarray(azimuth))


def mode_polarizations(amp, phase, modes, inclination, azimuth=0.0, xp=np):
    r"""Project the modes onto :math:`h_+, h_\times`, mode by mode.

    Parameters
    ----------
    amp, phase : array
        Shape ``(N, n_modes, k)``, as returned by :class:`BatchedSurrogate`.
    modes : sequence of (l, m)
        The modes along the second axis, with :math:`m > 0`.
    inclination : float or array of shape ``(N,)``
        Inclination :math:`\iota`, in radians.
    azimuth : float or array of shape ``(N,)``
        Azimuthal angle :math:`\varphi` of the observer, in radians.
        :meth:`Model.predict_modes_dict
        <mlgw_bns.model.Model.predict_modes_dict>` uses 0.
    xp : module
        ``numpy`` (default) or ``jax.numpy``.

    Returns
    -------
    tuple[array, array]
        ``(h_plus, h_cross)``, complex, each of shape ``(N, n_modes, k)``;
        see the module docstring for the convention. Summing over the
        second axis gives the polarizations of :meth:`Model.predict
        <mlgw_bns.model.Model.predict>`.
    """
    inclination = xp.reshape(xp.asarray(inclination, dtype=float), (-1, 1))
    azimuth = xp.reshape(xp.asarray(azimuth, dtype=float), (-1, 1))
    y_sum, y_diff = [], []
    for ell, emm in modes:
        ell, emm = int(ell), int(emm)
        sign = -1.0 if ell % 2 else 1.0
        c = math.sqrt((2 * ell + 1) / (4 * math.pi))
        d_pos = wigner_d_function(ell, emm, 2, inclination, xp=xp)
        d_neg = wigner_d_function(ell, -emm, 2, inclination, xp=xp)
        # Y_lm = c d_{m,2} e^{i m phi};  Y*_{l,-m} = c d_{-m,2} e^{i m phi}
        rotation = xp.exp(1j * emm * azimuth)
        y_sum.append(c * (d_pos + sign * d_neg) * rotation)
        y_diff.append(c * (d_pos - sign * d_neg) * rotation)
    y_sum = xp.stack(y_sum, axis=1)  # (N or 1, n_modes, 1)
    y_diff = xp.stack(y_diff, axis=1)
    h = amp * xp.exp(1j * phase) / 2
    return h * y_sum, 1j * h * y_diff


# ====================================================================== #
# Frozen model
# ====================================================================== #


def _slope_operator(x: np.ndarray) -> np.ndarray:
    """Linear map from knot values to knot slopes of SciPy's not-a-knot
    :class:`~scipy.interpolate.CubicSpline` on the knots ``x``."""
    n = len(x)
    spline = CubicSpline(x, np.eye(n), axis=0)
    slopes = np.empty((n, n))
    slopes[:-1] = spline.c[2]
    slopes[-1] = spline(x[-1], 1)
    return slopes


@dataclass
class _Spline:
    """Not-a-knot cubic spline on fixed knots, for batched values."""

    knots: Any
    widths: Any
    slope_operator_t: Any  # transposed, so that slopes = y @ slope_operator_t


@dataclass
class _KernelRidge:
    """RBF kernel ridge regression, as plain arrays."""

    x_fit: Any
    x_fit_sq_norm: Any
    dual_coef: Any
    gamma: float


@dataclass
class _ModeArrays:
    mode: tuple
    regressor_kind: str
    # "kernel_ridge": shared inputs (scaler, training points) plus per-mode
    # gamma and dual coefficients; "mlp": weights and biases.
    param_mean: Any
    param_scale: Any
    kernel: Optional[_KernelRidge]
    mlp_weights: Optional[list]
    mlp_biases: Optional[list]
    mlp_activation: Optional[str]
    target_mean: Any
    target_scale: Any
    # principal components
    eig_scaling: Any
    pcs_scaling: Any
    eigenvectors_t: Any
    pca_mean: Any
    n_amp: int
    # downsampling nodes
    amp_spline: _Spline
    phase_spline: _Spline
    amp_nodes_natural: Any
    phase_nodes_natural: Any
    pn_amp_nodes: Any  # frozen reference amplitude, or None
    trained_fmax_hz: float


@dataclass
class _Frozen:
    modes: list  # the requested modes, the first entries of `modes_arrays`
    reference_mass: float
    mass_sum_seconds: float
    eff_fmin_hz: float
    connection_natural: float
    extend_low: bool
    extend_high: bool
    # parameter ranges: (6, 2), rows total_mass, q, lambda_1, lambda_2, chi_1, chi_2
    ranges: Any
    safe_row: Any
    modes_arrays: list
    # position of the (2,2), which sets the merger reference, in `modes_arrays`
    reference_index: int


def _convert(obj, xp, memo=None):
    """Recursively convert the numpy arrays of a frozen dataclass to ``xp``.

    Arrays shared between regressors (e.g. the training points) stay
    shared: ``memo`` maps the ``id`` of each converted numpy array to its
    conversion.
    """
    if memo is None:
        memo = {}
    if isinstance(obj, np.ndarray):
        if id(obj) not in memo:
            memo[id(obj)] = (obj, xp.asarray(obj))
        return memo[id(obj)][1]
    if isinstance(obj, list):
        return [_convert(item, xp, memo) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_convert(item, xp, memo) for item in obj)
    if hasattr(obj, "__dataclass_fields__"):
        return replace(
            obj,
            **{f.name: _convert(getattr(obj, f.name), xp, memo) for f in fields(obj)},
        )
    return obj


def _shared(array, cache: list) -> np.ndarray:
    """``array`` as float, or an equal array already in ``cache``."""
    array = np.asarray(array, dtype=float)
    for known in cache:
        if known.shape == array.shape and np.array_equal(known, array):
            return known
    cache.append(array)
    return array


def _freeze_kernel_ridge(x_fit, dual_coef, gamma, cache) -> _KernelRidge:
    """Share the training points (and their norms) between regressors."""
    x_fit = np.asarray(x_fit, dtype=float)
    for known in cache:
        if known.x_fit.shape == x_fit.shape and np.array_equal(known.x_fit, x_fit):
            x_fit, sq_norm = known.x_fit, known.x_fit_sq_norm
            break
    else:
        sq_norm = np.einsum("ij,ij->i", x_fit, x_fit)
    kernel = _KernelRidge(
        x_fit=x_fit,
        x_fit_sq_norm=sq_norm,
        # as stored (Fortran order): a different layout takes a different
        # BLAS path, whose rounding this cancelling sum amplifies
        dual_coef=np.asarray(dual_coef, dtype=float),
        gamma=float(1.0 / x_fit.shape[1] if gamma is None else gamma),
    )
    cache.append(kernel)
    return kernel


def _freeze_spline(knots: np.ndarray) -> _Spline:
    knots = np.asarray(knots, dtype=float)
    return _Spline(
        knots=knots,
        widths=np.diff(knots),
        slope_operator_t=_slope_operator(knots).T.copy(),
    )


def _freeze(model: "Model", modes: Sequence[Mode]) -> _Frozen:
    from .neural_network import JaxMLPNetwork, KernelRidgeNetwork, SklearnNetwork

    dataset = model.dataset
    ranges = model.parameter_ranges
    kernel_cache: list = []
    scaler_cache: list = []

    # the (2,2) sets the merger reference: evaluated even if not requested
    requested = list(modes)
    modes = requested + ([Mode(2, 2)] if Mode(2, 2) not in requested else [])

    modes_arrays = []
    for mode in modes:
        mm = model.mode_models[mode]
        nn = mm.nn
        pca = mm.pca_data
        di = mm.downsampling_indices
        if nn is None or pca is None or di is None:
            raise ValueError(f"The model of mode {mode} is not trained.")
        f_hz = np.asarray(mm.dataset.frequencies_hz)
        f_nat = np.asarray(mm.dataset.frequencies)
        amp_idx = np.asarray(di.amplitude_indices)
        phase_idx = np.asarray(di.phase_indices)

        reference = mm.dataset.amplitude_reference_parameters
        pn_amp_nodes = (
            None
            if reference is None
            else np.asarray(
                mm.dataset.waveform_generator.post_newtonian_amplitude(
                    reference, f_nat[amp_idx]
                ),
                dtype=float,
            )
        )

        common = dict(
            mode=(mode.l, mode.m),
            param_mean=_shared(nn.param_scaler.mean_, scaler_cache),  # type: ignore[attr-defined]
            param_scale=_shared(nn.param_scaler.scale_, scaler_cache),  # type: ignore[attr-defined]
            eig_scaling=np.asarray(pca.eigenvalues**nn.hyper.pc_exponent, dtype=float),
            pcs_scaling=np.asarray(pca.principal_components_scaling, dtype=float),
            eigenvectors_t=np.ascontiguousarray(
                np.asarray(pca.eigenvectors, dtype=float).T
            ),
            pca_mean=np.asarray(pca.mean, dtype=float),
            n_amp=int(di.amp_length),
            amp_spline=_freeze_spline(f_hz[amp_idx]),
            phase_spline=_freeze_spline(f_hz[phase_idx]),
            amp_nodes_natural=f_nat[amp_idx],
            phase_nodes_natural=f_nat[phase_idx],
            pn_amp_nodes=pn_amp_nodes,
            trained_fmax_hz=float(f_hz[-1]),
        )
        if isinstance(nn, KernelRidgeNetwork):
            regressor = nn.regressor
            if getattr(regressor, "kernel", "rbf") != "rbf":
                raise NotImplementedError(
                    "Only RBF kernel ridge regressors are supported."
                )
            modes_arrays.append(
                _ModeArrays(
                    regressor_kind="kernel_ridge",
                    kernel=_freeze_kernel_ridge(
                        regressor.X_fit_,
                        regressor.dual_coef_,
                        regressor.gamma,
                        kernel_cache,
                    ),
                    mlp_weights=None,
                    mlp_biases=None,
                    mlp_activation=None,
                    target_mean=np.asarray(nn.target_scaler.mean_, dtype=float),
                    target_scale=np.asarray(nn.target_scaler.scale_, dtype=float),
                    **common,
                )
            )
        elif isinstance(nn, SklearnNetwork):
            modes_arrays.append(
                _ModeArrays(
                    regressor_kind="mlp",
                    kernel=None,
                    mlp_weights=[np.asarray(w, dtype=float) for w in nn.nn.coefs_],
                    mlp_biases=[np.asarray(b, dtype=float) for b in nn.nn.intercepts_],
                    mlp_activation=nn.nn.activation,
                    target_mean=None,
                    target_scale=None,
                    **common,
                )
            )
        elif isinstance(nn, JaxMLPNetwork):
            network = nn.network
            modes_arrays.append(
                _ModeArrays(
                    regressor_kind="mlp",
                    kernel=None,
                    mlp_weights=[np.asarray(w, dtype=float) for w, _ in network.weights],
                    mlp_biases=[np.asarray(b, dtype=float) for _, b in network.weights],
                    mlp_activation=network.config.activation,
                    target_mean=np.asarray(network.output_mean, dtype=float),
                    target_scale=np.asarray(network.output_scale, dtype=float),
                    **common,
                )
            )
        else:
            raise NotImplementedError(
                f"No batched evaluation for regressor {type(nn).__name__}."
            )

    range_array = np.array(
        [
            ranges.mass_range,
            ranges.q_range,
            ranges.lambda1_range,
            ranges.lambda2_range,
            ranges.chi1_range,
            ranges.chi2_range,
        ],
        dtype=float,
    )
    first = model.mode_models[modes[0]]
    return _Frozen(
        modes=[(mode.l, mode.m) for mode in requested],
        reference_mass=float(dataset.total_mass),
        mass_sum_seconds=float(dataset.mass_sum_seconds),
        eff_fmin_hz=float(dataset.effective_initial_frequency_hz),
        connection_natural=float(
            dataset.hz_to_natural_units(dataset.effective_initial_frequency_hz)
        ),
        extend_low=bool(first.extend_with_post_newtonian),
        extend_high=bool(first.extend_with_zeros_at_high_frequency),
        ranges=range_array,
        safe_row=range_array[1:].mean(axis=1),
        modes_arrays=modes_arrays,
        reference_index=modes.index(Mode(2, 2)),
    )


# ====================================================================== #
# Evaluation
# ====================================================================== #


#: Rows per chunk of the numpy kernel evaluation: large enough for BLAS,
#: small enough to keep the ``(rows, n_train)`` kernel block in memory.
_NUMPY_CHUNK_ROWS = 128


def _kernel_ridge(xp, scaled_x, kernels: list) -> list:
    r"""Predictions of RBF kernel ridge regressors sharing training points.

    The operation order is that of scikit-learn's ``euclidean_distances``
    and ``rbf_kernel`` (and of
    :meth:`~mlgw_bns.neural_network.KernelRidgeNetwork.predict`): with a
    single row on numpy the result is bit-for-bit the same as theirs.
    ``K @ dual_coef`` is a heavily cancelling sum for the shipped models,
    see the module docstring.

    Returns one ``(N, n_outputs)`` array per kernel.
    """
    x_fit = kernels[0].x_fit
    sq_norm = kernels[0].x_fit_sq_norm
    if xp is not np:
        distances = -2.0 * (scaled_x @ x_fit.T)
        distances = distances + xp.einsum("ij,ij->i", scaled_x, scaled_x)[:, None]
        distances = xp.maximum(distances + sq_norm[None, :], 0.0)
        return [xp.exp(distances * -k.gamma) @ k.dual_coef for k in kernels]

    n_rows = scaled_x.shape[0]
    outputs = [np.empty((n_rows,) + k.dual_coef.shape[1:]) for k in kernels]
    by_gamma: dict = {}
    for i, k in enumerate(kernels):
        by_gamma.setdefault(k.gamma, []).append(i)
    # Two reused buffers: fresh (rows, n_train) temporaries cost more in page
    # faults than the arithmetic done on them.
    rows_max = min(n_rows, _NUMPY_CHUNK_ROWS)
    distance_buffer = np.empty((rows_max, x_fit.shape[0]))
    block_buffer = np.empty_like(distance_buffer)
    for start in range(0, n_rows, rows_max):
        chunk = scaled_x[start : start + rows_max]
        rows = chunk.shape[0]
        distances, block = distance_buffer[:rows], block_buffer[:rows]
        np.matmul(chunk, x_fit.T, out=distances)
        distances *= -2.0
        distances += np.einsum("ij,ij->i", chunk, chunk)[:, None]
        distances += sq_norm[None, :]
        np.maximum(distances, 0.0, out=distances)
        for gamma, indices in by_gamma.items():
            np.multiply(distances, -gamma, out=block)
            np.exp(block, out=block)
            for i in indices:
                np.matmul(
                    block, kernels[i].dual_coef, out=outputs[i][start : start + rows]
                )
    return outputs


_ACTIVATIONS = {
    "relu": lambda xp, x: xp.maximum(x, 0.0),
    "tanh": lambda xp, x: xp.tanh(x),
    "logistic": lambda xp, x: 1.0 / (1.0 + xp.exp(-x)),
    "identity": lambda xp, x: x,
    # those of `JaxMLPNetwork`
    **{
        name: (lambda name: lambda xp, x: array_activation(xp, name)(x))(name)
        for name in ("gelu", "silu", "softplus", "elu")
    },
}


def _spline_setup(xp, spline: _Spline, y):
    """Knot slopes for a batch of knot values ``y`` of shape ``(N, n)``."""
    return y @ spline.slope_operator_t


def _spline_eval(xp, spline: _Spline, y, slopes, x, derivative: bool):
    """Values (and optionally derivatives) of the splines at ``x`` ``(N, k)``.

    The Hermite form and the order of operations are those of SciPy's
    :class:`~scipy.interpolate.CubicSpline` / ``PPoly``. Outside the knots
    the value is held at the end point (derivative zero), as in
    :meth:`~mlgw_bns.downsampling_interpolation.DownsamplingTraining.resample`.
    """
    knots = spline.knots
    n = knots.shape[0] - 1
    idx = xp.clip(xp.searchsorted(knots, x, side="right") - 1, 0, n - 1)
    take = lambda arr: xp.take_along_axis(arr, idx, axis=1)  # noqa: E731
    y0, y1 = take(y), take(y[:, 1:])
    s0, s1 = take(slopes), take(slopes[:, 1:])
    dx = spline.widths[idx]
    slope = (y1 - y0) / dx
    tt = (s0 + s1 - 2 * slope) / dx
    c0 = tt / dx
    c1 = (slope - s0) / dx - tt
    t = x - knots[idx]
    t2 = t * t
    value = ((y0 + s0 * t) + c1 * t2) + c0 * (t2 * t)
    below, above = x < knots[0], x > knots[n]
    value = xp.where(below, y[:, :1], xp.where(above, y[:, n : n + 1], value))
    if not derivative:
        return value, None
    deriv = (s0 + 2 * c1 * t) + 3 * c0 * t2
    return value, xp.where(below | above, 0.0, deriv)


class BatchedSurrogate:
    r"""The surrogate, frozen into plain arrays for batched evaluation.

    Parameters
    ----------
    model : Model
        A trained :class:`~mlgw_bns.model.Model`.
    modes : sequence of (l, m), optional
        Modes to evaluate, in the order in which they are returned.
        Defaults to all of ``model.modes``. Only these modes' regressors
        are evaluated.
    backend : str
        ``"numpy"`` (default) or ``"jax"``. With ``"jax"``, calling the
        object is a pure JAX function of its arguments, which can be
        wrapped in :func:`jax.jit`, differentiated and ``vmap``-ed; every
        new input shape compiles again, so callers should pad batches to a
        few fixed sizes.
    n_threads : int, optional
        numpy only: batches of more than :attr:`chunk_rows` rows are
        evaluated in chunks on this many threads. Defaults to one per CPU.

    Notes
    -----
    The arrays are copied out of ``model`` at construction time: later
    changes to the model (including its parameter ranges) are not seen.
    See the :mod:`module documentation <mlgw_bns.batched>` for the
    conventions and the numerical accuracy.
    """

    def __init__(
        self,
        model: "Model",
        modes: Optional[Sequence] = None,
        backend: str = "numpy",
        n_threads: Optional[int] = None,
    ):
        #: numpy only: batches larger than this many rows are split into
        #: chunks evaluated on :attr:`n_threads` threads.
        self.chunk_rows = 128
        #: numpy only: threads for large batches; ``None`` means one per CPU.
        self.n_threads = n_threads
        if modes is None:
            modes = list(model.modes)
        modes = [Mode(int(lm[0]), int(lm[1])) for lm in modes]
        missing = [mode for mode in modes if mode not in model.modes]
        if missing:
            raise ValueError(f"Modes {missing} are not in this model ({model.modes}).")
        if not modes:
            raise ValueError("At least one mode must be requested.")
        self.backend = backend
        self._xp, self._psi = _namespace(backend)
        self.modes = [(mode.l, mode.m) for mode in modes]
        self._frozen = _convert(_freeze(model, modes), self._xp)

    def __repr__(self) -> str:
        return f"BatchedSurrogate(modes={self.modes}, backend={self.backend!r})"

    # ---------------------------------------------------------------- #

    def valid(self, intrinsic, total_mass, frequencies=None):
        """Boolean mask ``(N,)`` of the rows the surrogate can evaluate.

        A row is valid when its total mass and intrinsic parameters are
        within the model's parameter ranges and, if ``frequencies`` is
        given, when every positive frequency is either in the trained band
        or covered by an extension the model is configured to make.
        """
        xp = self._xp
        x = xp.atleast_2d(xp.asarray(intrinsic, dtype=float))
        total_mass = xp.broadcast_to(xp.asarray(total_mass, dtype=float), (x.shape[0],))
        return self._valid(xp, x, total_mass, frequencies)

    def _valid(self, xp, x, total_mass, frequencies):
        fr = self._frozen
        values = xp.concatenate([total_mass[:, None], x], axis=1)
        ok = xp.all((values >= fr.ranges[:, 0]) & (values <= fr.ranges[:, 1]), axis=1)
        if frequencies is not None and not (fr.extend_low and fr.extend_high):
            f = xp.asarray(frequencies, dtype=float)
            rescaled = f * (total_mass / fr.reference_mass)[:, None]
            positive = f > 0
            if not fr.extend_low:
                ok = ok & ~xp.any(positive & (rescaled < fr.eff_fmin_hz), axis=1)
            if not fr.extend_high:
                fmax = min(ma.trained_fmax_hz for ma in fr.modes_arrays)
                ok = ok & ~xp.any(positive & (rescaled > fmax), axis=1)
        return ok

    # ---------------------------------------------------------------- #

    def _residuals(self, xp, x):
        """Per mode, the amplitude and phase residuals at the nodes."""
        modes_arrays = self._frozen.modes_arrays
        components: list = [None] * len(modes_arrays)

        kernel_groups: dict = {}
        for i, ma in enumerate(modes_arrays):
            if ma.regressor_kind == "kernel_ridge":
                key = (id(ma.kernel.x_fit), id(ma.param_mean), id(ma.param_scale))
                kernel_groups.setdefault(key, []).append(i)
            else:
                activation = (x - ma.param_mean) / ma.param_scale
                last = len(ma.mlp_weights) - 1
                for j, (w, b) in enumerate(zip(ma.mlp_weights, ma.mlp_biases)):
                    activation = activation @ w + b
                    if j != last:
                        activation = _ACTIVATIONS[ma.mlp_activation](xp, activation)
                if ma.target_scale is not None:
                    activation = activation * ma.target_scale + ma.target_mean
                components[i] = activation
        for indices in kernel_groups.values():
            first = modes_arrays[indices[0]]
            scaled = (x - first.param_mean) / first.param_scale
            predictions = _kernel_ridge(
                xp, scaled, [modes_arrays[i].kernel for i in indices]
            )
            for i, prediction in zip(indices, predictions):
                ma = modes_arrays[i]
                components[i] = prediction * ma.target_scale + ma.target_mean

        out = []
        for ma, comps in zip(modes_arrays, components):
            combined = ((comps / ma.eig_scaling) * ma.pcs_scaling) @ ma.eigenvectors_t
            combined = combined + ma.pca_mean
            out.append((combined[:, : ma.n_amp], combined[:, ma.n_amp :]))
        return out

    # ---------------------------------------------------------------- #

    def __call__(
        self,
        intrinsic,
        total_mass,
        frequencies,
        distance_mpc=1.0,
        return_tf: bool = False,
    ):
        r"""Amplitude and phase of the requested modes for ``N`` binaries.

        Parameters
        ----------
        intrinsic : array, shape ``(N, 5)``
            Rows of :math:`[q \geq 1, \Lambda_1, \Lambda_2, \chi_1, \chi_2]`.
            A single row of shape ``(5,)`` is treated as ``(1, 5)``.
        total_mass : array, shape ``(N,)`` or scalar
            Total (detector-frame) mass, in solar masses.
        frequencies : array, shape ``(k,)`` or ``(N, k)``
            Increasing frequencies in Hz, shared by all rows or one grid
            per row.
        distance_mpc : array, shape ``(N,)`` or scalar
            Luminosity distance in Mpc. Defaults to 1.
        return_tf : bool
            Whether to also return :math:`t_{\ell m}(f)`.

        Returns
        -------
        amp, phase : array, shape ``(N, n_modes, k)``
            See the module docstring for the convention. Rows outside the
            model's range are NaN.
        tf : array, shape ``(N, n_modes, k)``
            Only if ``return_tf``: the time, in seconds relative to the
            merger, at which each mode emits each frequency.
        """
        xp = self._xp
        x = xp.atleast_2d(xp.asarray(intrinsic, dtype=float))
        n_rows = x.shape[0]
        total_mass = xp.broadcast_to(xp.asarray(total_mass, dtype=float), (n_rows,))
        distance = xp.broadcast_to(xp.asarray(distance_mpc, dtype=float), (n_rows,))
        f = xp.asarray(frequencies, dtype=float)
        f = xp.broadcast_to(f if f.ndim == 2 else f[None, :], (n_rows, f.shape[-1]))

        chunk = self.chunk_rows
        if xp is not np or n_rows <= chunk:
            return self._evaluate(xp, x, total_mass, f, distance, return_tf)

        # numpy: independent row chunks, on a thread pool (numpy and BLAS
        # release the GIL); BLAS itself is kept single-threaded meanwhile.
        from concurrent.futures import ThreadPoolExecutor

        from threadpoolctl import threadpool_limits  # type: ignore

        starts = range(0, n_rows, chunk)

        def run(start):
            rows = slice(start, start + chunk)
            return self._evaluate(
                xp, x[rows], total_mass[rows], f[rows], distance[rows], return_tf
            )

        workers = self.n_threads or os.cpu_count() or 1
        with threadpool_limits(limits=1, user_api="blas"):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                parts = list(pool.map(run, starts))
        return tuple(np.concatenate(pieces, axis=0) for pieces in zip(*parts))

    def _evaluate(self, xp, x, total_mass, f, distance, return_tf):
        fr = self._frozen
        n_rows = x.shape[0]

        valid = self._valid(xp, x, total_mass, f)
        # Invalid rows are evaluated at a harmless point and blanked out
        # at the end, so that they cannot raise or warn along the way.
        x = xp.where(valid[:, None], x, fr.safe_row[None, :])
        mass = xp.where(valid, total_mass, fr.reference_mass)

        mass_ratio = (mass / fr.reference_mass)[:, None]  # (N, 1)
        rescaled = f * mass_ratio  # (N, k): frequencies at the reference mass
        positive = f > 0
        low = positive & (rescaled < fr.eff_fmin_hz)

        # Below the band, the post-Newtonian continuation. The frequencies
        # increase, so the points below the band are a prefix of each row:
        # numpy only evaluates the columns up to the last of them, JAX
        # (fixed shapes) the whole grid.
        if xp is np:
            n_low = int(np.max(np.nonzero(low)[1], initial=-1)) + 1
        else:
            n_low = f.shape[1]
        if n_low:
            # natural units, in the operation order of
            # `ModeModel.predict_amplitude_phase`; the points not
            # below the band are moved to the connection, harmlessly (they
            # are discarded, but must not produce NaNs, nor NaN gradients)
            low_natural = xp.where(
                low[:, :n_low],
                rescaled[:, :n_low] * fr.mass_sum_seconds,
                fr.connection_natural,
            )
            f_min_connection = fr.connection_natural / 2.0
            zero_to_one = (low_natural - f_min_connection) / (
                fr.connection_natural - f_min_connection
            )
            blend = xp.where(
                low_natural > f_min_connection,
                (1 - xp.cos(zero_to_one * np.pi)) / 2,
                0.0,
            )
            connection = xp.full((1, 1), fr.connection_natural)
            edge = xp.full((n_rows, 1), fr.eff_fmin_hz)

        # --- all the post-Newtonian evaluations at once: at the nodes for
        # every mode (the phase only for the (2,2) when it is just the
        # merger reference), then below the band for the requested modes
        n_out = len(fr.modes)
        requests = []
        for i, ma in enumerate(fr.modes_arrays):
            requests.append(
                _PNRequest(ma.mode, ma.phase_nodes_natural[None, :], unwrap=True)
            )
            if i < n_out and ma.pn_amp_nodes is None:
                requests.append(
                    _PNRequest(ma.mode, ma.amp_nodes_natural[None, :], amplitude=True)
                )
        for ma in fr.modes_arrays[:n_out]:
            lm = ma.mode
            if n_low:
                requests.append(_PNRequest(lm, connection, amplitude=True))
                requests.append(_PNRequest(lm, low_natural, amplitude=True))
                if return_tf:
                    requests.append(
                        _PNRequest(lm, low_natural * (1 + _TF_RELATIVE_STEP))
                    )
                    requests.append(
                        _PNRequest(lm, low_natural * (1 - _TF_RELATIVE_STEP))
                    )
        pn = iter(_post_newtonian(xp, self._psi, x, requests))

        residuals = self._residuals(xp, x)
        nodes = []
        for i, (ma, (amp_res, phase_res)) in enumerate(zip(fr.modes_arrays, residuals)):
            _, pn_phase = next(pn)
            amp_nodes = None
            if i < n_out:
                pn_amp = (
                    ma.pn_amp_nodes[None, :]
                    if ma.pn_amp_nodes is not None
                    else next(pn)[0]
                )
                amp_nodes = pn_amp * amp_res
            phase_nodes = pn_phase + phase_res
            nodes.append(
                (amp_nodes, phase_nodes, _spline_setup(xp, ma.phase_spline, phase_nodes))
            )

        # --- the merger reference: the tangent to the (2,2) phase at the
        # top knot, as a function of the frequency at the reference mass
        reference = fr.modes_arrays[fr.reference_index]
        _, reference_phase, reference_slopes = nodes[fr.reference_index]
        merger_slope = reference_slopes[:, -1:]
        merger_intercept = (
            reference_phase[:, -1:] - merger_slope * reference.phase_spline.knots[-1]
        )

        amps, phases, tfs = [], [], []
        for ma, (amp_nodes, phase_nodes, phase_slopes) in zip(
            fr.modes_arrays[:n_out], nodes
        ):
            # --- in the band
            amp_slopes = _spline_setup(xp, ma.amp_spline, amp_nodes)
            amp, _ = _spline_eval(
                xp, ma.amp_spline, amp_nodes, amp_slopes, rescaled, False
            )
            phase, dphase = _spline_eval(
                xp, ma.phase_spline, phase_nodes, phase_slopes, rescaled, return_tf
            )
            if return_tf:
                dphase = dphase * mass_ratio  # d phase / d f, observer frame

            # --- below the band: post-Newtonian, glued on at the band's edge
            if n_low:
                pn_amp_connection, pn_phase_connection = next(pn)
                low_amp, low_phase = next(pn)
                amp_edge, _ = _spline_eval(
                    xp, ma.amp_spline, amp_nodes, amp_slopes, edge, False
                )
                phase_edge, _ = _spline_eval(
                    xp, ma.phase_spline, phase_nodes, phase_slopes, edge, False
                )
                low_amp = low_amp + blend * (amp_edge - pn_amp_connection)
                low_phase = (low_phase - pn_phase_connection) + phase_edge
                amp = _replace_prefix(xp, amp, low[:, :n_low], low_amp)
                phase = _replace_prefix(xp, phase, low[:, :n_low], low_phase)
                if return_tf:
                    _, up = next(pn)
                    _, down = next(pn)
                    # d phase / d f_natural * d f_natural / d f
                    low_f = xp.where(low[:, :n_low], f[:, :n_low], 1.0)
                    low_dphase = (up - down) / (2 * _TF_RELATIVE_STEP * low_f)
                    dphase = _replace_prefix(xp, dphase, low[:, :n_low], low_dphase)

            # --- merger reference
            phase = phase - merger_slope * rescaled - ma.mode[1] / 2 * merger_intercept

            # --- above the band, and non-positive frequencies: zero
            outside = (~positive) | (rescaled > ma.trained_fmax_hz)
            amp = xp.where(outside, 0.0, amp)
            phase = xp.where(outside, 0.0, phase)

            amp = amp * (mass**2 / AMP_SI_BASE)[:, None] / distance[:, None]
            amps.append(amp)
            phases.append(phase)
            if return_tf:
                dphase = xp.where(outside, 0.0, dphase - merger_slope * mass_ratio)
                tfs.append(-dphase / (2 * np.pi))

        blank = ~valid[:, None, None]
        out = [
            xp.where(blank, np.nan, xp.stack(amps, axis=1)),
            xp.where(blank, np.nan, xp.stack(phases, axis=1)),
        ]
        if return_tf:
            out.append(xp.where(blank, np.nan, xp.stack(tfs, axis=1)))
        return tuple(out)


def _replace_prefix(xp, full, mask, values):
    """``full`` with its first ``values.shape[1]`` columns replaced by
    ``values`` where ``mask`` is set."""
    n = values.shape[1]
    if n == full.shape[1]:
        return xp.where(mask, values, full)
    return xp.concatenate([xp.where(mask, values, full[:, :n]), full[:, n:]], axis=1)
