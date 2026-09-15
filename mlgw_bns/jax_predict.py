r"""JAX reimplementation of the surrogate's prediction pipeline.

Adapted from the JAX port of ``mlgw_bns`` written by **Saulo Albuquerque**
(``saulo-albuquerque-phys``) --- see ``jax_reference/`` for the verbatim
upstream files and their provenance. That work targets the single-mode
0.12.1 interface (a five-parameter ``SklearnNetwork`` MLP feeding a PCA
reconstruction, one ``(2, 2)`` polarisation); this module carries the
idea to the current multi-mode model, whose per-mode residual regressor
is a :class:`~mlgw_bns.neural_network.KernelRidgeNetwork` and whose output
is the observer-frame :math:`(h_+, h_\times)` summed over four modes.

The Post-Newtonian expansions are not reimplemented here: the per-mode
``H_lm`` coefficients are pure polynomials in ``v`` with scalar (not
array-valued) constants, so :mod:`mlgw_bns.pn_modes`'s numpy functions
run unmodified under ``jax.numpy`` arrays; the TaylorF2 phase itself is
built from :mod:`mlgw_bns.taylorf2`'s ``_make_*`` factories instantiated
with ``xp=jax.numpy`` (the same factories the numba-jitted numpy path
instantiates with ``xp=numpy``). Only the JIT-able cubic-spline evaluator
here is a genuine JAX-specific port.

Public entry points
-------------------
* :func:`mode_model_to_jax_residuals` --- one mode's
  ``predict_residuals_bulk`` (RBF kernel ridge + PCA reconstruction).
* :func:`model_to_jax_waveform` --- the full :meth:`Model.predict`:
  ``(params, frequencies_hz, total_mass, distance_mpc, inclination,
  reference_phase) -> (h_plus, h_cross)``. Pure JAX --- use ``jax.jit``
  and ``jax.vmap`` (over the leading axis of ``params``) freely.

Scope / accuracy
----------------
* The kernel-ridge ``K @ dual_coef_`` is a cancelling sum (small
  ``kernel_alpha``); XLA's reduction order differs from BLAS, so the
  output tracks the numpy pipeline to ~1e-4 relative, not bit-for-bit.
* The library resamples with a *not-a-knot* spline; this uses a frozen
  linear solve for the not-a-knot second derivatives, matching it.
* The low-frequency TaylorF2 splice (only below
  ``effective_initial_frequency_hz`` ~ 3.57 Hz, i.e. ``total_mass``
  below the dataset reference or a query grid starting that low) and the
  high-frequency zero-padding are **not** ported; keep the query grid
  inside ``[f_min_effective * M_ref / total_mass, f_max_trained]``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .pn_modes import H_21, H_22, H_33, H_44
from .taylorf2 import _make_taylorf2_psi

jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from .mode_model import ModeModel
    from .model import Model
    from .neural_network import (
        KernelRidgeNetwork,
        ModePhasesNN,
        NeuralNetwork,
        SklearnNetwork,
        TimeshiftsNN,
    )

# Physical constants (mlgw_bns.taylorf2 / dataset_generation).
_SUN_MASS_SECONDS = 4.92549094830932e-6
_EULER_GAMMA = 0.57721566490153286060
_AMP_SI_BASE = 4.2425873413901263e24

_ACTIVATIONS: dict[str, Callable] = {
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "logistic": jax.nn.sigmoid,
    "identity": lambda x: x,
}


# ====================================================================== #
# Regressor backends: parameters -> scaled PCA coefficients
# ====================================================================== #

def kernel_ridge_network_to_jax(
    nn: "KernelRidgeNetwork",
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """JAX version of :meth:`KernelRidgeNetwork.predict` (RBF kernel ridge).

    Matches scikit-learn's floating-point operation order (accumulate
    ``-2 X Y^T``, add the squared norms, ``exp``); see the note in
    :meth:`KernelRidgeNetwork.predict`.
    """
    regressor = nn.regressor
    x_fit = jnp.asarray(regressor.X_fit_, dtype=jnp.float64)
    dual_coef = jnp.asarray(regressor.dual_coef_, dtype=jnp.float64)
    sq_norm = jnp.asarray(np.einsum("ij,ij->i", regressor.X_fit_, regressor.X_fit_))
    gamma = regressor.gamma
    gamma = float(1.0 / x_fit.shape[1] if gamma is None else gamma)

    param_mean = jnp.asarray(nn.param_scaler.mean_, dtype=jnp.float64)
    param_scale = jnp.asarray(nn.param_scaler.scale_, dtype=jnp.float64)
    target_mean = jnp.asarray(nn.target_scaler.mean_, dtype=jnp.float64)
    target_scale = jnp.asarray(nn.target_scaler.scale_, dtype=jnp.float64)

    def predict(params: jnp.ndarray) -> jnp.ndarray:
        scaled_x = (jnp.asarray(params, jnp.float64) - param_mean) / param_scale
        sq_dist = -2.0 * (scaled_x @ x_fit.T)
        sq_dist += jnp.einsum("ij,ij->i", scaled_x, scaled_x)[:, None]
        sq_dist += sq_norm[None, :]
        kernel = jnp.exp(-gamma * jnp.maximum(sq_dist, 0.0))
        return (kernel @ dual_coef) * target_scale + target_mean

    return predict


def sklearn_mlp_to_jax(
    nn: "SklearnNetwork",
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """JAX version of the legacy ``SklearnNetwork`` MLP forward pass."""
    scaler_mean = jnp.asarray(nn.param_scaler.mean_, dtype=jnp.float64)
    scaler_scale = jnp.asarray(nn.param_scaler.scale_, dtype=jnp.float64)
    coefs = [jnp.asarray(w, dtype=jnp.float64) for w in nn.nn.coefs_]
    intercepts = [jnp.asarray(b, dtype=jnp.float64) for b in nn.nn.intercepts_]
    activation = _ACTIVATIONS[nn.nn.activation]

    def predict(params: jnp.ndarray) -> jnp.ndarray:
        x = (jnp.asarray(params, jnp.float64) - scaler_mean) / scaler_scale
        for w, b in zip(coefs[:-1], intercepts[:-1]):
            x = activation(x @ w + b)
        return x @ coefs[-1] + intercepts[-1]

    return predict


def neural_network_to_jax(nn: "NeuralNetwork") -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Dispatch to the right backend port for ``nn``."""
    from .neural_network import KernelRidgeNetwork, SklearnNetwork

    if isinstance(nn, KernelRidgeNetwork):
        return kernel_ridge_network_to_jax(nn)
    if isinstance(nn, SklearnNetwork):
        return sklearn_mlp_to_jax(nn)
    raise TypeError(f"No JAX port for regressor backend {type(nn).__name__!r}.")


def mode_model_to_jax_residuals(
    mode_model: "ModeModel",
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """JAX version of :meth:`ModeModel.predict_residuals_bulk`.

    ``params (n, 5) -> combined (amp | phi) residuals (n, n_amp + n_phi)``.
    """
    regressor_predict = neural_network_to_jax(mode_model.nn)
    pca = mode_model.pca_data
    pc_exponent = float(mode_model.nn.hyper.pc_exponent)
    eig_scaling = jnp.asarray(pca.eigenvalues**pc_exponent, dtype=jnp.float64)
    pcs_scaling = jnp.asarray(pca.principal_components_scaling, dtype=jnp.float64)
    eigenvectors = jnp.asarray(pca.eigenvectors, dtype=jnp.float64)
    pca_mean = jnp.asarray(pca.mean, dtype=jnp.float64)

    def predict(params: jnp.ndarray) -> jnp.ndarray:
        params = jnp.atleast_2d(jnp.asarray(params, jnp.float64))
        components = regressor_predict(params) / eig_scaling
        return (components * pcs_scaling) @ eigenvectors.T + pca_mean

    return predict


# ====================================================================== #
# TaylorF2 phase (natural units), mass-independent, and per-mode PN
# amplitude/phase. Both are shared with the numpy path (jax.numpy
# instantiation of mlgw_bns.taylorf2's `_make_*` factories, and
# mlgw_bns.pn_modes's H_lm functions used directly on jnp arrays).
# ====================================================================== #

_taylorf2_psi = _make_taylorf2_psi(jnp)

_H_BY_MODE = {(2, 2): H_22, (2, 1): H_21, (3, 3): H_33, (4, 4): H_44}


def _mode_pn_amp(lm, f_natural, eta, chi_a, chi_s):
    """``amp_lm``: ``|pi sqrt(2 eta/3) (2 pi f/m)^{-7/2} H_lm(v)|`` (``.real`` for (4,4))."""
    m = lm[1]
    v = jnp.abs(2 * math.pi * f_natural / m) ** (1.0 / 3.0)
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    prefactor = math.pi * jnp.sqrt(2 * eta / 3) * v ** (-7.0 / 2.0)
    h = _H_BY_MODE[tuple(lm)](v, eta, delta, chi_a, chi_s)
    if tuple(lm) == (4, 4):
        return prefactor * jnp.real(h)
    return jnp.abs(prefactor * h)


def _mode_pn_phase(lm, f_natural, eta, chi1, chi2, chi_a, chi_s, lam1, lam2):
    """``psi_lm``: ``(m/2) psi22(2f/m) + unwrap(arg H_lm)`` with the
    parameter-stable branch snap."""
    m = lm[1]
    mode_freq = 2 * f_natural / m
    orbital = _taylorf2_psi(mode_freq, eta, chi1, chi2, lam1, lam2) * (m / 2)

    v = jnp.abs(2 * math.pi * f_natural / m) ** (1.0 / 3.0)
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    h = _H_BY_MODE[tuple(lm)](v, eta, delta, chi_a, chi_s)
    arg_h = jnp.unwrap(jnp.angle(h))
    lead = _H_BY_MODE[tuple(lm)](jnp.asarray(1e-8), eta, delta, chi_a, chi_s)
    lr, li = jnp.real(lead), jnp.imag(lead)
    lead_arg = jnp.where(
        lr >= jnp.abs(li),
        0.0,
        jnp.where(lr <= -jnp.abs(li), math.pi, jnp.sign(li) * (math.pi / 2)),
    )
    arg_h = arg_h + 2 * math.pi * jnp.round((lead_arg - arg_h[0]) / (2 * math.pi))
    return orbital + arg_h


def reference_phase_backbone_jax(params, f0_natural, lm, rel_step=1e-4):
    """``M_lm = f0 dPsi/df|f0`` by central difference. ``params`` is ``(n, 5)``."""
    m = lm[1]
    q, lam1, lam2, chi1, chi2 = (params[:, k] for k in range(5))
    eta = q / (1.0 + q) ** 2
    h = f0_natural * rel_step
    lo = 2 * (f0_natural - h) / m
    hi = 2 * (f0_natural + h) / m
    psi_lo = _taylorf2_psi(lo, eta, chi1, chi2, lam1, lam2) * (m / 2)
    psi_hi = _taylorf2_psi(hi, eta, chi1, chi2, lam1, lam2) * (m / 2)
    return f0_natural * (psi_hi - psi_lo) / (2 * h)


# ====================================================================== #
# Reference predictors (ModePhasesNN, TimeshiftsNN)
# ====================================================================== #

def _minmax(scaler):
    lo = jnp.asarray(scaler.data_min_, dtype=jnp.float64)
    rng = jnp.asarray(scaler.data_range_, dtype=jnp.float64)
    return lambda x: (jnp.asarray(x, jnp.float64) - lo) / rng


def _kernel_ridge_pipeline_to_jax(pipeline):
    """Bare RBF ``KernelRidge`` inside a one-step ``Pipeline``."""
    krr = pipeline.named_steps["kernel_ridge"] if hasattr(
        pipeline, "named_steps"
    ) else pipeline
    x_fit = jnp.asarray(krr.X_fit_, dtype=jnp.float64)
    dual = jnp.asarray(krr.dual_coef_, dtype=jnp.float64)
    sq_norm = jnp.asarray(np.einsum("ij,ij->i", krr.X_fit_, krr.X_fit_))
    gamma = float(1.0 / x_fit.shape[1] if krr.gamma is None else krr.gamma)

    def predict(scaled_x):
        d = -2.0 * (scaled_x @ x_fit.T)
        d += jnp.einsum("ij,ij->i", scaled_x, scaled_x)[:, None]
        d += sq_norm[None, :]
        k = jnp.exp(-gamma * jnp.maximum(d, 0.0))
        out = k @ dual
        return out

    return predict


def mode_phases_nn_to_jax(
    nn: "ModePhasesNN", modes: list
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """JAX version of :meth:`ModePhasesNN.predict` -> ``(n, n_modes)``."""
    scale = _minmax(nn.scaler)
    regressor = _kernel_ridge_pipeline_to_jax(nn.regressor)
    f0 = float(next(iter(nn.analytic_coeffs.values()))[1]) if nn.analytic_coeffs else None
    ref = nn._ref_column()

    lr_by_mode = {}
    for j, lm in enumerate(nn.modes):
        lr = nn.analytic_coeffs[lm][0]
        lr_by_mode[j] = (
            jnp.asarray(lr.coef_, dtype=jnp.float64),
            float(lr.intercept_),
        )

    ncols = list(nn.modes)

    def predict(params: jnp.ndarray) -> jnp.ndarray:
        params = jnp.atleast_2d(jnp.asarray(params, jnp.float64))
        leftover = regressor(scale(params))  # (n, n_modes)

        q, chi1, chi2 = params[:, 0], params[:, 3], params[:, 4]
        chi_eff = (chi1 + q * chi2) / (1.0 + q)
        m_ref = None
        if ref is not None:
            m_ref = reference_phase_backbone_jax(params, f0, ncols[ref])

        cols = []
        for j, lm in enumerate(ncols):
            m_lm = reference_phase_backbone_jax(params, f0, lm)
            if ref is not None and j != ref:
                m_col = m_lm - m_ref
            else:
                m_col = m_lm
            design = jnp.stack(
                [m_col, params[:, 0], params[:, 1], params[:, 2],
                 params[:, 3], params[:, 4], chi_eff],
                axis=1,
            )
            coef, intercept = lr_by_mode[j]
            cols.append(design @ coef + intercept)
        analytic = jnp.stack(cols, axis=1)

        prediction = leftover + analytic
        if ref is not None:
            phi22 = prediction[:, ref]
            add = jnp.where(
                jnp.arange(prediction.shape[1])[None, :] == ref, 0.0, phi22[:, None]
            )
            prediction = prediction + add
        return prediction

    return predict


def timeshifts_nn_to_jax(nn: "TimeshiftsNN") -> Callable[[jnp.ndarray], jnp.ndarray]:
    """JAX version of :meth:`TimeshiftsNN.predict` (RBFSampler + Ridge) -> ``(n,)``."""
    scale = _minmax(nn.scaler)
    pipe = nn.regressor
    steps = dict(pipe.named_steps) if hasattr(pipe, "named_steps") else {}
    rff = next(v for v in steps.values() if hasattr(v, "random_weights_"))
    ridge = next(v for v in steps.values() if hasattr(v, "coef_"))
    weights = jnp.asarray(rff.random_weights_, dtype=jnp.float64)
    offset = jnp.asarray(rff.random_offset_, dtype=jnp.float64)
    n_comp = weights.shape[1]
    coef = jnp.asarray(np.ravel(ridge.coef_), dtype=jnp.float64)
    intercept = float(np.ravel(ridge.intercept_)[0])

    def predict(params: jnp.ndarray) -> jnp.ndarray:
        x = scale(jnp.atleast_2d(jnp.asarray(params, jnp.float64)))
        z = jnp.cos(x @ weights + offset) * jnp.sqrt(2.0 / n_comp)
        return z @ coef + intercept

    return predict


# ====================================================================== #
# not-a-knot cubic spline with frozen knots  (matches scipy.CubicSpline)
# ====================================================================== #

def make_not_a_knot_spline_jax(
    x_knots: np.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """``eval(y_knots, x_query) -> values``, matching
    ``scipy.interpolate.CubicSpline(x, y, extrapolate=False)`` in the
    interior with endpoint-clamped values outside ``[x[0], x[-1]]``
    (as :meth:`DownsamplingTraining.resample` does). JIT/``vmap``-able.

    The second-derivative solve ``M = D @ y`` is a frozen ``(n+1, n+1)``
    linear operator (``D`` depends only on the knots).
    """
    x = np.asarray(x_knots, dtype=np.float64)
    n = len(x) - 1
    h = np.diff(x)

    if n < 2:
        x_j = jnp.asarray(x)
        return lambda y, xq: jnp.interp(xq, x_j, y)

    # A @ M = B @ y, with not-a-knot boundary rows.
    A = np.zeros((n + 1, n + 1))
    B = np.zeros((n + 1, n + 1))
    for r in range(1, n):
        A[r, r - 1] = h[r - 1]
        A[r, r] = 2.0 * (h[r - 1] + h[r])
        A[r, r + 1] = h[r]
        B[r, r - 1] = 6.0 / h[r - 1]
        B[r, r] = -6.0 / h[r - 1] - 6.0 / h[r]
        B[r, r + 1] = 6.0 / h[r]
    # not-a-knot: h[1] M0 - (h0 + h1) M1 + h0 M2 = 0  (and mirror at the end)
    A[0, 0] = h[1]
    A[0, 1] = -(h[0] + h[1])
    A[0, 2] = h[0]
    A[n, n - 2] = h[n - 1]
    A[n, n - 1] = -(h[n - 2] + h[n - 1])
    A[n, n] = h[n - 2]

    D = jnp.asarray(np.linalg.solve(A, B), dtype=jnp.float64)
    x_j = jnp.asarray(x)
    h_j = jnp.asarray(h)
    x0, x1 = float(x[0]), float(x[-1])

    def _eval(y_knots: jnp.ndarray, x_query: jnp.ndarray) -> jnp.ndarray:
        M = D @ y_knots
        idx = jnp.clip(jnp.searchsorted(x_j, x_query, side="right") - 1, 0, n - 1)
        hi = jnp.take(h_j, idx)
        yi = jnp.take(y_knots, idx)
        yi1 = jnp.take(y_knots, idx + 1)
        Mi = jnp.take(M, idx)
        Mi1 = jnp.take(M, idx + 1)
        t = x_query - jnp.take(x_j, idx)
        b = (yi1 - yi) / hi - hi * (2.0 * Mi + Mi1) / 6.0
        cubic = yi + b * t + (Mi / 2.0) * t**2 + (Mi1 - Mi) / (6.0 * hi) * t**3
        cubic = jnp.where(x_query < x0, y_knots[0], cubic)
        cubic = jnp.where(x_query > x1, y_knots[-1], cubic)
        return cubic

    return _eval


# ====================================================================== #
# Full waveform
# ====================================================================== #

def model_to_jax_waveform(model: "Model") -> Callable:
    r"""Build a JAX function reproducing :meth:`Model.predict`.

    Returns ``predict(params, frequencies_hz, total_mass, distance_mpc,
    inclination, reference_phase=0.0) -> (h_plus, h_cross)`` where
    ``params`` is ``[q, lambda_1, lambda_2, chi_1, chi_2]`` (shape
    ``(5,)``; use ``jax.vmap`` over a leading batch axis). Pure JAX.
    """
    from .pn_modes import Mode as _PMode

    dataset = model.dataset
    modes = [tuple(m) for m in model.modes]
    m_ref = float(dataset.total_mass)
    mass_sum_seconds = float(dataset.mass_sum_seconds)
    eff_fmin_hz = float(dataset.effective_initial_frequency_hz)

    ref_params = dataset.amplitude_reference_parameters
    if ref_params is None:
        raise NotImplementedError(
            "JAX port currently assumes reference_amplitude=True "
            "(frozen per-mode PN amplitude); the shipped default_hom has it."
        )
    ref_q = float(ref_params.mass_ratio)
    ref_eta = ref_q / (1.0 + ref_q) ** 2
    ref_chi_a = (float(ref_params.chi_1) - float(ref_params.chi_2)) / 2.0
    ref_chi_s = (float(ref_params.chi_1) + float(ref_params.chi_2)) / 2.0

    residual_fns = []
    amp_splines = []
    phi_splines = []
    amp_freqs_nat = []
    phi_freqs_nat = []
    n_amps = []
    pn_amp_frozen = []
    amp_knots_hz = []
    phi_knots_hz = []

    for lm in modes:
        mm = model.mode_models[_PMode(*lm)]
        di = mm.downsampling_indices
        fz = np.asarray(mm.dataset.frequencies_hz)
        fn = np.asarray(mm.dataset.frequencies)
        residual_fns.append(mode_model_to_jax_residuals(mm))
        n_amps.append(di.amp_length)
        amp_knots_hz.append(fz[di.amplitude_indices])
        phi_knots_hz.append(fz[di.phase_indices])
        amp_splines.append(make_not_a_knot_spline_jax(fz[di.amplitude_indices]))
        phi_splines.append(make_not_a_knot_spline_jax(fz[di.phase_indices]))
        af = jnp.asarray(fn[di.amplitude_indices], dtype=jnp.float64)
        pf = jnp.asarray(fn[di.phase_indices], dtype=jnp.float64)
        amp_freqs_nat.append(af)
        phi_freqs_nat.append(pf)
        pn_amp_frozen.append(
            _mode_pn_amp(lm, af, ref_eta, ref_chi_a, ref_chi_s)
        )

    mode_phases_fn = mode_phases_nn_to_jax(model.mode_phases_predictor, model.modes)
    timeshift_fn = timeshifts_nn_to_jax(model.time_shifts_predictor)
    mode_phase_index = {lm: i for i, lm in enumerate(modes)}

    # spherical harmonics: azimuth 0 -> Y_lm real. Precompute per-mode
    # (A, B) = (Y_lm + Y_{l,-m}, Y_lm - Y_{l,-m}) as functions of iota.
    def _wigner_d(l, m, iota):
        cos_h = jnp.cos(iota / 2)
        sin_h = jnp.sin(iota / 2)
        ki = max(0, m - 2)
        kf = min(l + m, l - 2)
        acc = 0.0
        for k in range(ki, kf + 1):
            norm = (
                math.factorial(k)
                * math.factorial(l + m - k)
                * math.factorial(l - 2 - k)
                * math.factorial(k + 2 - m)
            )
            acc = acc + (
                (-1) ** k
                * cos_h ** (2 * l + m - 2 - 2 * k)
                * sin_h ** (2 * k + 2 - m)
            ) / norm
        const = math.sqrt(
            math.factorial(l + m)
            * math.factorial(l - m)
            * math.factorial(l + 2)
            * math.factorial(l - 2)
        )
        return const * acc

    def _ylm_real(l, m, iota):
        return math.sqrt((2 * l + 1) / (4 * math.pi)) * _wigner_d(l, m, iota)

    def predict(
        params,
        frequencies_hz,
        total_mass,
        distance_mpc,
        inclination,
        reference_phase=0.0,
    ):
        params = jnp.atleast_2d(jnp.asarray(params, jnp.float64))
        freqs = jnp.asarray(frequencies_hz, jnp.float64)
        q = params[:, 0]
        eta = q / (1.0 + q) ** 2
        chi1, chi2 = params[:, 3], params[:, 4]
        chi_a = (chi1 - chi2) / 2.0
        chi_s = (chi1 + chi2) / 2.0

        # tidal parameters (compute_lambda_tilde / compute_delta_lambda)
        m1 = m_ref / (1.0 + 1.0 / q)
        m2 = m_ref / (1.0 + q)
        Mtot = m1 + m2
        lam1_r, lam2_r = params[:, 1], params[:, 2]
        lambdatilde = (16.0 / 13.0) * (
            (m1 + 12.0 * m2) * m1**4 * lam1_r + (m2 + 12.0 * m1) * m2**4 * lam2_r
        ) / Mtot**5
        Xd = jnp.sqrt(1.0 - 4.0 * eta)
        dlambda = (
            (1690.0 * eta / 1319.0 - 4843.0 / 1319.0)
            * (m1**4 * lam1_r - m2**4 * lam2_r)
            / Mtot**4
            + (6162.0 * Xd / 1319.0)
            * (m1**4 * lam1_r + m2**4 * lam2_r)
            / Mtot**4
        )

        time_shift = timeshift_fn(params)  # (n,)
        ts_scaled = time_shift * (total_mass / m_ref)
        mode_phase0 = mode_phases_fn(params)  # (n, n_modes)
        rescaled = freqs[None, :] * (total_mass / m_ref)  # (n, k)
        # Fixed anchor for the time-shift phase trend below (see the
        # matching fix/comment in Model._hpc_waveform_per_mode): using
        # freqs[0] made the phase depend on wherever the caller's query
        # grid started, which only matched Model.predict's convention by
        # accident when frequencies[0] == effective_initial_frequency_hz.
        reference_frequency_hz = eff_fmin_hz * (m_ref / total_mass)

        pre = total_mass**2 / _AMP_SI_BASE * eta / distance_mpc  # (n,)

        hp = jnp.zeros((params.shape[0], freqs.shape[0]), dtype=jnp.complex128)
        hc = jnp.zeros_like(hp)

        for i, lm in enumerate(modes):
            l, m = lm
            residuals = residual_fns[i](params)  # (n, n_amp + n_phi)
            na = n_amps[i]
            amp_res = residuals[:, :na]
            phi_res = residuals[:, na:]

            def _phi_one(row, _lm=lm, _i=i):
                e = row[0] / (1.0 + row[0]) ** 2
                return _mode_pn_phase(
                    _lm, phi_freqs_nat[_i], e, row[3], row[4],
                    (row[3] - row[4]) / 2, (row[3] + row[4]) / 2, row[1], row[2],
                )

            pn_phi = jax.vmap(_phi_one)(params)  # (n, n_phi)

            amp_ds = pn_amp_frozen[i][None, :] * amp_res  # (n, n_amp)
            phi_ds = pn_phi + phi_res + mode_phase0[:, i][:, None]

            amp_rs = jax.vmap(amp_splines[i], in_axes=(0, 0))(amp_ds, rescaled)
            phi_rs = jax.vmap(phi_splines[i], in_axes=(0, 0))(phi_ds, rescaled)

            amp = amp_rs * pre[:, None]
            phi = (
                phi_rs
                + m * reference_phase
                + 2 * math.pi * (freqs[None, :] - reference_frequency_hz) * ts_scaled[:, None]
            )

            yr = _ylm_real(l, m, inclination)
            yr_m = _ylm_real(l, -m, inclination)
            A = yr + yr_m
            Bc = yr - yr_m
            if l % 2:
                c0, c5, c6 = Bc, -A, A
            else:
                c0, c5, c6 = A, -Bc, Bc
            cos_p = jnp.cos(phi)
            sin_p = jnp.sin(phi)
            hp = hp + amp * (c0 * cos_p + 1j * c0 * sin_p)
            hc = hc + amp * (c5 * sin_p + 1j * c6 * cos_p)

        hp = hp / eta[:, None] / 2.0
        hc = hc / eta[:, None] / 2.0
        if params.shape[0] == 1:
            return hp[0], hc[0]
        return hp, hc

    return predict
