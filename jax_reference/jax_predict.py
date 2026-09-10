"""Translation of a trained :class:`Model` inference pipeline into a
JAX-jittable function.

The full prediction pipeline — parameter scaling, MLP forward pass,
PCA eigenvalue un-scaling, and PCA reconstruction — is extracted from a
(fully loaded) :class:`Model` instance and frozen into a pure JAX function
that can be JIT-compiled, differentiated, or batched with the standard JAX
transforms (``jax.jit``, ``jax.grad``, ``jax.vmap``, …).

The returned function maps a 1-D parameter vector of shape ``(n_params,)``
(or a batch ``(n, n_params)`` when used with ``jax.vmap``) to the combined
amplitude+phase residuals array of shape ``(n_amp + n_phi,)``.

Example
-------
>>> import mlgw_bns
>>> import jax
>>> model = mlgw_bns.Model.default()
>>> predict_jax = model_to_jax_predict(model)
>>> import jax.numpy as jnp
>>> params = jnp.array([[0.6, 1.5, 300.0, 300.0, 0.0, 0.0]])
>>> residuals = predict_jax(params)          # plain call
>>> residuals_jit = jax.jit(predict_jax)(params)  # JIT-compiled
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np

# Enable 64-bit precision globally so that weights extracted from sklearn
# (which are float64) are not silently downcast to float32.
jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from .model import Model

# ------------------------------------------------------------------ #
# Physical constants (from taylorf2.py / dataset_generation.py)
# ------------------------------------------------------------------ #
_SUN_MASS_SECONDS: float = 4.92549094830932e-6  # M_sun * G / c^3
_EULER_GAMMA: float = 0.57721566490153286060
_TF2_BASE: float = 3.668693487138444e-19
_AMP_SI_BASE: float = 4.2425873413901263e24

# Map sklearn activation names to JAX functions.
_ACTIVATIONS: dict[str, Callable] = {
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "logistic": jax.nn.sigmoid,
    "identity": lambda x: x,
}


def sklearn_mlp_to_jax(
    nn,  # SklearnNetwork
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Convert a fitted :class:`SklearnNetwork` into a pure JAX function.

    Only the MLP forward pass (including input scaling via the
    :class:`~sklearn.preprocessing.StandardScaler`) is captured; the PCA
    reconstruction is *not* included here (see :func:`model_to_jax_predict`
    for the full pipeline).

    Parameters
    ----------
    nn : SklearnNetwork
        A trained network with ``nn.param_scaler`` and ``nn.nn`` attributes
        already fitted.

    Returns
    -------
    Callable
        A function ``predict(x) -> y`` where ``x`` has shape
        ``(n_samples, n_params)`` and ``y`` has shape
        ``(n_samples, n_pca_components)``.
        The function is JIT-compatible via ``jax.jit``.
    """
    # Freeze scaler parameters as JAX arrays.
    scaler_mean = jnp.array(nn.param_scaler.mean_, dtype=jnp.float64)
    scaler_scale = jnp.array(nn.param_scaler.scale_, dtype=jnp.float64)

    # Freeze MLP weights as JAX arrays.
    # sklearn stores coefs_ as (n_in, n_out) matrices — the forward pass is
    # x = activation(x @ W + b).
    coefs = [jnp.array(w, dtype=jnp.float64) for w in nn.nn.coefs_]
    intercepts = [jnp.array(b, dtype=jnp.float64) for b in nn.nn.intercepts_]

    activation_name = nn.nn.activation
    if activation_name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{activation_name}'. "
            f"Supported: {list(_ACTIVATIONS)}"
        )
    activation = _ACTIVATIONS[activation_name]

    def predict(x: jnp.ndarray) -> jnp.ndarray:
        """MLP forward pass with input scaling.

        Parameters
        ----------
        x : jnp.ndarray
            Shape ``(n_samples, n_params)``.

        Returns
        -------
        jnp.ndarray
            Shape ``(n_samples, n_pca_components)``.
        """
        x = (x - scaler_mean) / scaler_scale

        # Hidden layers (all but the last weight matrix).
        for W, b in zip(coefs[:-1], intercepts[:-1]):
            x = activation(x @ W + b)

        # Output layer: linear (no activation).
        x = x @ coefs[-1] + intercepts[-1]

        return x

    return predict


def model_to_jax_predict(
    model: "Model",
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a JAX-jittable function that reproduces
    :meth:`Model.predict_residuals_bulk` for a single waveform.

    The returned function implements, in order:

    1. Input-parameter scaling (``StandardScaler``).
    2. MLP forward pass.
    3. PCA eigenvalue un-scaling.
    4. PCA reconstruction back to the full residual space.

    All intermediate values are JAX arrays so the function is fully
    differentiable and JIT-compilable.

    Parameters
    ----------
    model : Model
        A fully loaded model (``model.nn`` and ``model.pca_data`` must not be
        ``None``).

    Returns
    -------
    Callable
        ``predict(x) -> combined_residuals`` where

        * ``x`` has shape ``(n_samples, n_params)``,
        * ``combined_residuals`` has shape
          ``(n_samples, n_amp_points + n_phi_points)``.

        The first ``n_amp_points`` columns are amplitude residuals;
        the remaining ``n_phi_points`` columns are phase residuals
        (matching the layout of :attr:`Residuals.combined`).

    Raises
    ------
    AssertionError
        If ``model.nn`` or ``model.pca_data`` is ``None``.
    """
    assert model.nn is not None, "model.nn is None — load or train the model first."
    assert model.pca_data is not None, (
        "model.pca_data is None — load or train the model first."
    )

    # ------------------------------------------------------------------ #
    # 1. Build the standalone MLP predict function.
    # ------------------------------------------------------------------ #
    mlp_predict = sklearn_mlp_to_jax(model.nn)

    # ------------------------------------------------------------------ #
    # 2. Freeze PCA reconstruction constants.
    # ------------------------------------------------------------------ #
    pc_exponent = float(model.nn.hyper.pc_exponent)
    eigenvalue_scaling = jnp.array(
        model.pca_data.eigenvalues ** pc_exponent, dtype=jnp.float64
    )
    pca_scaling = jnp.array(
        model.pca_data.principal_components_scaling, dtype=jnp.float64
    )
    eigenvectors = jnp.array(model.pca_data.eigenvectors, dtype=jnp.float64)
    pca_mean = jnp.array(model.pca_data.mean, dtype=jnp.float64)

    def predict(x: jnp.ndarray) -> jnp.ndarray:
        """Full NN + PCA pipeline.

        Parameters
        ----------
        x : jnp.ndarray
            Parameter array with shape ``(n_samples, n_params)``.

        Returns
        -------
        jnp.ndarray
            Combined residuals with shape
            ``(n_samples, n_amp_points + n_phi_points)``.
        """
        # NN forward pass → scaled PCA components
        scaled_pca = mlp_predict(x)

        # Undo eigenvalue scaling applied during training
        pca_components = scaled_pca / eigenvalue_scaling

        # PCA reconstruction:
        #   scaled_data = pca_components * principal_components_scaling
        #   zero_mean   = scaled_data @ eigenvectors.T
        #   result      = zero_mean + mean
        scaled_data = pca_components * pca_scaling
        zero_mean = scaled_data @ eigenvectors.T
        combined_residuals = zero_mean + pca_mean

        return combined_residuals

    return predict


# ================================================================== #
# Natural cubic spline interpolation (JAX-traceable)
#
# The spline knots (x_ds) are frozen at factory time.  Only the
# values (y_ds) change at inference time, so we precompute the
# Thomas-algorithm sweep constants from x_ds in numpy and store
# them as frozen JAX arrays.  At inference time, only O(n) JAX
# operations need to run (scan + polynomial evaluation), keeping
# the full function JIT/grad/vmap-compatible.
# ================================================================== #


def _make_cubic_spline_jax(
    x_ds_np: np.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Build a JAX-jittable natural cubic spline evaluator for a fixed knot grid.

    Implements the same interpolant as
    ``scipy.interpolate.splrep(x_ds, y_ds, s=0, k=3)`` /
    ``splev(x_new, tck)``.

    The knot positions ``x_ds_np`` are frozen at construction time
    (converted to JAX arrays); the values ``y_ds`` and the query
    points ``x_new`` are JAX arrays provided at call time.

    Parameters
    ----------
    x_ds_np : np.ndarray
        Sorted knot positions, shape ``(m,)``.

    Returns
    -------
    Callable
        ``eval_spline(y_ds, x_new) -> np.ndarray``
        where ``y_ds`` has shape ``(m,)`` and ``x_new`` has shape ``(k,)``.
        Fully JIT/grad/vmap-compatible.
    """
    n = len(x_ds_np) - 1  # number of intervals

    if n < 2:
        # Degenerate case: fall back to linear interpolation.
        x_jax = jnp.array(x_ds_np, dtype=jnp.float64)

        def _eval_linear(y_ds: jnp.ndarray, x_new: jnp.ndarray) -> jnp.ndarray:
            return jnp.interp(x_new, x_jax, y_ds)

        return _eval_linear

    h_np = np.diff(x_ds_np)  # (n,) interval widths

    # -------------------------------------------------------------- #
    # Tridiagonal system for interior second derivatives m[1..n-1]:
    #   A @ m_int = rhs(y_ds)
    # where A has:
    #   main diagonal  d_i = 2*(h[i-1] + h[i])   i=1..n-1  (size n-1)
    #   off-diagonals  e_i = h[i]                 i=1..n-2  (size n-2)
    # -------------------------------------------------------------- #
    sz = n - 1  # system size
    d_np = 2.0 * (h_np[:-1] + h_np[1:])  # main diagonal   (n-1,)
    e_np = h_np[1:-1]                      # off-diagonals   (n-2,)

    # ---- Thomas algorithm forward sweep: precompute from x_ds only ----
    # c_star[i] = e[i] / (d[i] - e[i-1] * c_star[i-1])   (modified upper diag)
    # inv_d[i]  = 1 / (d[i] - e[i-1] * c_star[i-1])     (reciprocal mod diag)
    c_star_np = np.zeros(sz - 1)
    inv_d_np = np.zeros(sz)

    inv_d_np[0] = 1.0 / d_np[0]
    if sz > 1:
        c_star_np[0] = e_np[0] * inv_d_np[0]
    for i in range(1, sz):
        denom = d_np[i] - e_np[i - 1] * c_star_np[i - 1]
        inv_d_np[i] = 1.0 / denom
        if i < sz - 1:
            c_star_np[i] = e_np[i] * inv_d_np[i]

    # ---- Freeze as JAX constants ----
    h_jax = jnp.array(h_np, dtype=jnp.float64)
    e_jax = jnp.array(e_np, dtype=jnp.float64)          # lower diag (n-2,)
    c_star_jax = jnp.array(c_star_np, dtype=jnp.float64)  # (sz-1,)
    inv_d_jax = jnp.array(inv_d_np, dtype=jnp.float64)    # (sz,)
    x_jax = jnp.array(x_ds_np, dtype=jnp.float64)

    def _eval_spline(y_ds: jnp.ndarray, x_new: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the natural cubic spline at query positions.

        Parameters
        ----------
        y_ds : jnp.ndarray
            Spline values at the knot positions, shape ``(n+1,)``.
        x_new : jnp.ndarray
            Query positions, shape ``(k,)``.

        Returns
        -------
        jnp.ndarray
            Interpolated values, shape ``(k,)``.
        """
        # ---- Build RHS vector (depends on y_ds) ----
        dy = jnp.diff(y_ds)  # (n,)
        # rhs[i] = 6*(dy[i+1]/h[i+1] - dy[i]/h[i])  for i=0..sz-2
        rhs = 6.0 * (dy[1:] / h_jax[1:] - dy[:-1] / h_jax[:-1])  # (sz,)

        # ---- Thomas forward sweep on RHS ----
        # d0 → first modified value
        d0 = rhs[0] * inv_d_jax[0]

        def fwd_step(
            d_prev: jnp.ndarray,
            inputs: tuple,
        ) -> tuple:
            e_i, inv_di, r_i = inputs
            d_new = (r_i - e_i * d_prev) * inv_di
            return d_new, d_new

        _, d_rest = jax.lax.scan(
            fwd_step,
            d0,
            (e_jax, inv_d_jax[1:], rhs[1:]),
        )
        d_star = jnp.concatenate([jnp.array([d0]), d_rest])  # (sz,)

        # ---- Thomas back substitution ----
        x_last = d_star[-1]

        def back_step(
            x_next: jnp.ndarray,
            inputs: tuple,
        ) -> tuple:
            c_i, d_i = inputs
            x_i = d_i - c_i * x_next
            return x_i, x_i

        _, xs_rev = jax.lax.scan(
            back_step,
            x_last,
            (c_star_jax[::-1], d_star[:-1][::-1]),
        )
        m_interior = jnp.concatenate([xs_rev[::-1], jnp.array([x_last])])  # (sz,)

        # Full second-derivative vector: natural B.C. → m[0]=m[n]=0
        m = jnp.concatenate(
            [jnp.array([0.0]), m_interior, jnp.array([0.0])]
        )  # (n+1,)

        # ---- Evaluate cubic polynomial in each query interval ----
        idx = jnp.searchsorted(x_jax, x_new, side="right") - 1
        idx = jnp.clip(idx, 0, n - 1)

        hi  = jnp.take(h_jax, idx)
        yi  = jnp.take(y_ds, idx)
        yi1 = jnp.take(y_ds, idx + 1)
        mi  = jnp.take(m, idx)
        mi1 = jnp.take(m, idx + 1)
        t   = x_new - jnp.take(x_jax, idx)

        b_i = (yi1 - yi) / hi - hi / 6.0 * (2.0 * mi + mi1)
        c_i = mi / 2.0
        d_i = (mi1 - mi) / (6.0 * hi)

        return yi + b_i * t + c_i * t ** 2 + d_i * t ** 3

    return _eval_spline


# ================================================================== #
# Complete waveform factory (NN + PCA + TaylorF2 + cubic spline)
# ================================================================== #


def model_to_jax_waveform(
    model: "Model",
) -> Callable:
    """Build a JAX-jittable function that returns the plus and cross
    polarizations at an arbitrary frequency grid, matching the output
    of ``Model.predict``.

    This is the final step of the full JAX pipeline:

    ::

        params (5,) + frequencies_hz (k,) + total_mass + distance_mpc + inclination
            → NN+PCA residuals (downsampled)
            → TaylorF2 PN amplitude + phase (downsampled)
            → combine residuals × PN
            → cubic-spline interpolation to query grid
            → h = amp * exp(i*phi)
            → hp, hc polarizations

    Parameters
    ----------
    model : Model
        A fully loaded ``Model`` instance.

    Returns
    -------
    Callable
        ``predict(params, frequencies_hz, total_mass, distance_mpc, inclination) -> (hp, hc)``

        * ``params``: shape ``(5,)`` = ``[q, λ₁, λ₂, χ₁, χ₂]``
        * ``frequencies_hz``: shape ``(k,)`` — query frequencies in Hz
        * ``total_mass``: scalar, total binary mass in solar masses
        * ``distance_mpc``: scalar, luminosity distance in Mpc
        * ``inclination``: scalar, inclination angle in radians
        * ``hp``: shape ``(k,)`` complex — plus polarization
        * ``hc``: shape ``(k,)`` complex — cross polarization

        The function is JIT/grad/vmap-compatible via JAX transforms.
    """
    assert model.nn is not None
    assert model.pca_data is not None
    assert model.downsampling_indices is not None

    # -- Downsampled-grid waveform --
    predict_ds = model_to_jax_waveform_ds(model)

    # -- Cubic spline interpolators built from fixed knot grids --
    amp_freqs_hz_np = np.asarray(
        model.dataset.frequencies_hz[model.downsampling_indices.amplitude_indices]
    )
    phi_freqs_hz_np = np.asarray(
        model.dataset.frequencies_hz[model.downsampling_indices.phase_indices]
    )
    amp_spline = _make_cubic_spline_jax(amp_freqs_hz_np)
    phi_spline = _make_cubic_spline_jax(phi_freqs_hz_np)

    M_ref: float = float(model.dataset.total_mass)

    def predict(
        params: jnp.ndarray,
        frequencies_hz: jnp.ndarray,
        total_mass: jnp.ndarray,
        distance_mpc: jnp.ndarray,
        inclination: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate plus and cross polarizations at arbitrary frequencies.

        Parameters
        ----------
        params : jnp.ndarray
            Shape ``(5,)`` = ``[q, λ₁, λ₂, χ₁, χ₂]``.
        frequencies_hz : jnp.ndarray
            Query frequencies in Hz, shape ``(k,)``.
        total_mass : scalar jnp.ndarray
            Total binary mass in solar masses.
        distance_mpc : scalar jnp.ndarray
            Luminosity distance in Mpc.
        inclination : scalar jnp.ndarray
            Inclination angle in radians.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            ``(hp, hc)`` — complex plus and cross polarizations.
        """
        # 1. Downsampled amplitude and phase (at reference mass grid)
        amp_ds, phi_ds = predict_ds(params)

        # 2. Rescale query frequencies to the reference-mass frame.
        rescaled_freqs = frequencies_hz * (total_mass / M_ref)

        # 3. Cubic-spline interpolation
        amp = amp_spline(amp_ds, rescaled_freqs)
        phi = phi_spline(phi_ds, rescaled_freqs)

        # 4. Apply the mlgw_bns_prefactor and distance:
        #    pre = total_mass² / AMP_SI_BASE * eta / distance_mpc
        eta = params[0] / (1.0 + params[0]) ** 2
        pre = total_mass ** 2 / _AMP_SI_BASE * eta / distance_mpc
        amp = amp * pre

        # 5. Convert to complex Cartesian form: h = A * exp(i*phi)
        h_real = amp * jnp.cos(phi)
        h_imag = amp * jnp.sin(phi)

        # 6. Apply inclination-dependent polarization prefactors
        cosi = jnp.cos(inclination)
        pre_plus = (1.0 + cosi ** 2) / 2.0
        pre_cross = cosi

        hp = pre_plus * h_real + 1j * pre_plus * h_imag
        hc = pre_cross * h_imag - 1j * pre_cross * h_real

        return hp, hc

    return predict

#
# These are direct ports of the @njit functions in taylorf2.py.
# All np.* calls are replaced with jnp.* so the functions are
# fully traceable by jax.jit, jax.grad, and jax.vmap.
# Conditional branches on traced values are replaced with jnp.where.
# ================================================================== #


def _compute_quadrupole_yy_jax(lam: jnp.ndarray) -> jnp.ndarray:
    """Quadrupole coefficient from Lambda (Yunes-Yagi Love-Q relation).
    Replaces the numba version which had a Python branch on lam <= 0.
    """
    loglam = jnp.log(jnp.where(lam > 0.0, lam, 1.0))  # avoid log(0)
    logCQ = (
        0.194
        + 0.0936 * loglam
        + 0.0474 * loglam ** 2
        - 4.21e-3 * loglam ** 3
        + 1.23e-4 * loglam ** 4
    )
    return jnp.where(lam <= 0.0, 1.0, jnp.exp(logCQ))


def _compute_lambda_tilde_jax(
    m1: jnp.ndarray,
    m2: jnp.ndarray,
    l1: jnp.ndarray,
    l2: jnp.ndarray,
) -> jnp.ndarray:
    """Reduced tidal deformability Lambda-tilde."""
    M = m1 + m2
    return (16.0 / 13.0) * (
        (m1 + 12.0 * m2) * m1 ** 4 * l1
        + (m2 + 12.0 * m1) * m2 ** 4 * l2
    ) / M ** 5


def _compute_delta_lambda_jax(
    m1: jnp.ndarray,
    m2: jnp.ndarray,
    l1: jnp.ndarray,
    l2: jnp.ndarray,
) -> jnp.ndarray:
    """Asymmetric tidal parameter delta-Lambda-tilde."""
    M = m1 + m2
    eta = (m1 * m2) / M ** 2
    X = jnp.sqrt(1.0 - 4.0 * eta)
    comb1 = (1690.0 * eta / 1319.0 - 4843.0 / 1319.0) * (m1 ** 4 * l1 - m2 ** 4 * l2) / M ** 4
    comb2 = (6162.0 * X / 1319.0) * (m1 ** 4 * l1 + m2 ** 4 * l2) / M ** 4
    return comb1 + comb2


def _PhifT7hPNComplete_jax(
    f: jnp.ndarray,
    M: float,
    eta: jnp.ndarray,
    Lama: jnp.ndarray,
    Lamb: jnp.ndarray,
) -> jnp.ndarray:
    """7.5PN tidal phase from https://arxiv.org/abs/2005.13367 ."""
    v = jnp.power(jnp.abs(jnp.pi * M * f * _SUN_MASS_SECONDS), 1.0 / 3.0)
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    Xa = 0.5 * (1.0 + delta)
    Xb = 0.5 * (1.0 - delta)
    Xa2 = Xa * Xa; Xa3 = Xa2 * Xa; Xa4 = Xa3 * Xa; Xa5 = Xa4 * Xa
    Xb2 = Xb * Xb; Xb3 = Xb2 * Xb; Xb4 = Xb3 * Xb; Xb5 = Xb4 * Xb
    v2 = v * v; v3 = v2 * v; v4 = v3 * v; v5 = v4 * v
    kapa = 3.0 * Lama * Xa4 * Xb
    kapb = 3.0 * Lamb * Xb4 * Xa
    pNa = -3.0 / (16.0 * eta) * (12.0 + Xa / Xb)
    pNb = -3.0 / (16.0 * eta) * (12.0 + Xb / Xa)
    p1a = 5.0 * (3179.0 - 919.0 * Xa - 2286.0 * Xa2 + 260.0 * Xa3) / (672.0 * (12.0 - 11.0 * Xa))
    p1b = 5.0 * (3179.0 - 919.0 * Xb - 2286.0 * Xb2 + 260.0 * Xb3) / (672.0 * (12.0 - 11.0 * Xb))
    p2a = -jnp.pi
    p2b = -jnp.pi
    p3a = (
        -5 * (
            -387973870.0
            + 43246839.0 * Xa + 174965616.0 * Xa2 + 158378220.0 * Xa3
            - 20427120.0 * Xa4 + 4572288.0 * Xa5
        ) / 27433728.0
    ) / (12.0 - 11.0 * Xa)
    p3b = (
        -5 * (
            -387973870.0
            + 43246839.0 * Xb + 174965616.0 * Xb2 + 158378220.0 * Xb3
            - 20427120.0 * Xb4 + 4572288.0 * Xb5
        ) / 27433728.0
    ) / (12.0 - 11.0 * Xb)
    p4a = -jnp.pi * (27719.0 - 22415.0 * Xa + 7598.0 * Xa2 - 10520.0 * Xa3) / (672.0 * (12.0 - 11.0 * Xa))
    p4b = -jnp.pi * (27719.0 - 22127.0 * Xb + 7022.0 * Xb2 - 10232.0 * Xb3) / (672.0 * (12.0 - 11.0 * Xb))
    return v5 * (
        kapa * pNa * (1.0 + p1a * v2 + p2a * v3 + p3a * v4 + p4a * v5)
        + kapb * pNb * (1.0 + p1b * v2 + p2b * v3 + p3b * v4 + p4b * v5)
    )


def _PhifQM3hPN_jax(
    f: jnp.ndarray,
    M: float,
    eta: jnp.ndarray,
    s1z: jnp.ndarray,
    s2z: jnp.ndarray,
    Lam1: jnp.ndarray,
    Lam2: jnp.ndarray,
) -> jnp.ndarray:
    """QM self-spin 3.5PN phase correction (Eq.50-52 of arXiv:1812.07923)."""
    v = jnp.power(jnp.abs(jnp.pi * M * f * _SUN_MASS_SECONDS), 1.0 / 3.0)
    v2 = v * v
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    X1 = 0.5 * (1.0 + delta)
    X2 = 0.5 * (1.0 - delta)
    at1 = X1 * s1z
    at2 = X2 * s2z
    at1_2 = at1 * at1
    at2_2 = at2 * at2
    CQ1 = _compute_quadrupole_yy_jax(Lam1) - 1.0
    CQ2 = _compute_quadrupole_yy_jax(Lam2) - 1.0
    a2CQ_p = at1_2 * CQ1 + at2_2 * CQ2
    a2CQ_m = at1_2 * CQ1 - at2_2 * CQ2
    PhifQM = -75.0 / (64.0 * eta) * a2CQ_p / v
    PhifQM += ((45.0 / 16.0 * eta + 15635.0 / 896.0) * a2CQ_p + 2215.0 / 512.0 * delta * a2CQ_m) * v / eta
    PhifQM += -75.0 / (8.0 * eta) * a2CQ_p * v2 * jnp.pi
    return PhifQM


def _Phif3hPN_jax(
    f: jnp.ndarray,
    M: float,
    eta: jnp.ndarray,
    s1z: jnp.ndarray = 0.0,
    s2z: jnp.ndarray = 0.0,
    Lam: jnp.ndarray = 0.0,
    dLam: jnp.ndarray = 0.0,
) -> jnp.ndarray:
    """3.5PN phase including spins and tidal at 6PN.

    The tidal branch is always computed (safe for BNS where Lam != 0);
    when Lam=dLam=0 the tidal term evaluates to zero.
    """
    vlso = 1.0 / jnp.sqrt(6.0)
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    v = jnp.abs(jnp.pi * M * f * _SUN_MASS_SECONDS) ** (1.0 / 3.0)
    v2 = v * v;  v3 = v2 * v;  v4 = v2 * v2
    v5 = v4 * v; v6 = v3 * v3; v7 = v3 * v4
    v10 = v5 * v5; v12 = v10 * v2
    eta2 = eta ** 2; eta3 = eta ** 3

    m1M = 0.5 * (1.0 + delta)
    m2M = 0.5 * (1.0 - delta)
    chi1L = s1z
    chi2L = s2z
    # aligned-spin only (s1x=s1y=s2x=s2y=0)
    chi1sq = s1z * s1z
    chi2sq = s2z * s2z
    chi1dotchi2 = s1z * s2z
    SL = m1M * m1M * chi1L + m2M * m2M * chi2L
    dSigmaL = delta * (m2M * chi2L - m1M * chi1L)

    sigma = eta * (721.0 / 48.0 * chi1L * chi2L - 247.0 / 48.0 * chi1dotchi2)
    sigma += 719.0 / 96.0 * (m1M * m1M * chi1L * chi1L + m2M * m2M * chi2L * chi2L)
    sigma -= 233.0 / 96.0 * (m1M * m1M * chi1sq + m2M * m2M * chi2sq)
    phis_15PN = 188.0 * SL / 3.0 + 25.0 * dSigmaL
    ga = (554345.0 / 1134.0 + 110.0 * eta / 9.0) * SL + (13915.0 / 84.0 - 10.0 * eta / 3.0) * dSigmaL
    pn_ss3 = (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta * chi1L * chi2L
    pn_ss3 += (
        (4703.5 / 8.4 + 2935.0 / 6.0 * m1M - 120.0 * m1M * m1M)
        + (-4108.25 / 6.72 - 108.5 / 1.2 * m1M + 125.5 / 3.6 * m1M * m1M)
    ) * m1M * m1M * chi1sq
    pn_ss3 += (
        (4703.5 / 8.4 + 2935.0 / 6.0 * m2M - 120.0 * m2M * m2M)
        + (-4108.25 / 6.72 - 108.5 / 1.2 * m2M + 125.5 / 3.6 * m2M * m2M)
    ) * m2M * m2M * chi2sq
    phis_3PN = jnp.pi * (3760.0 * SL + 1490.0 * dSigmaL) / 3.0 + pn_ss3
    phis_35PN = (
        -8980424995.0 / 762048.0 + 6586595.0 * eta / 756.0 - 305.0 * eta2 / 36.0
    ) * SL - (
        170978035.0 / 48384.0 - 2876425.0 * eta / 672.0 - 4735.0 * eta2 / 144.0
    ) * dSigmaL

    LO = 3.0 / 128.0 / eta / v5
    pointmass = (
        1
        + 20.0 / 9.0 * (743.0 / 336.0 + 11.0 / 4.0 * eta) * v2
        + (phis_15PN - 16.0 * jnp.pi) * v3
        + 10.0 * (3058673.0 / 1016064.0 + 5429.0 / 1008.0 * eta + 617.0 / 144.0 * eta2 - sigma) * v4
        + (38645.0 / 756.0 * jnp.pi - 65.0 / 9.0 * eta * jnp.pi - ga) * (1.0 + 3.0 * jnp.log(v / vlso)) * v5
        + (
            11583231236531.0 / 4694215680.0
            - 640.0 / 3.0 * jnp.pi ** 2
            - 6848.0 / 21.0 * (_EULER_GAMMA + jnp.log(4.0 * v))
            + (-15737765635.0 / 3048192.0 + 2255.0 * jnp.pi ** 2 / 12.0) * eta
            + 76055.0 / 1728.0 * eta2
            - 127825.0 / 1296.0 * eta3
            + phis_3PN
        ) * v6
        + (
            jnp.pi * (77096675.0 / 254016.0 + 378515.0 / 1512.0 * eta - 74045.0 / 756.0 * eta2)
            + phis_35PN
        ) * v7
    )

    # Tidal correction at 6PN — always computed; zero when Lam=dLam=0
    tidal = Lam * v10 * (-39.0 / 2.0 - 3115.0 / 64.0 * v2) + dLam * 6595.0 / 364.0 * v12

    return LO * (pointmass + tidal)


def _Phif5hPN_jax(
    f: jnp.ndarray,
    M: float,
    eta: jnp.ndarray,
    s1z: jnp.ndarray = 0.0,
    s2z: jnp.ndarray = 0.0,
) -> jnp.ndarray:
    """5.5PN phase (point-mass + spins, no tidal — tidal handled separately).

    Calls ``_Phif3hPN_jax`` with Lam=dLam=0 for the 3.5PN base, then adds
    the 4PN–5.5PN corrections.
    """
    phi_35pn = _Phif3hPN_jax(f, M, eta, s1z, s2z, 0.0, 0.0)

    v = (jnp.pi * M * f * _SUN_MASS_SECONDS) ** (1.0 / 3.0)
    v2 = v * v;  v3 = v2 * v;  v4 = v2 * v2
    v5 = v4 * v; v6 = v3 * v3; v7 = v3 * v4
    v8 = v7 * v; v9 = v8 * v; v10 = v5 * v5; v11 = v10 * v
    logv = jnp.log(v)
    eta2 = eta ** 2; eta3 = eta ** 3
    log2 = 0.69314718055994528623
    log3 = 1.0986122886681097821

    # Coefficients with c_21_3PN=c_22_4PN=c_22_5PN=a_6_c=0
    coef_8pn = (
        - 36946947827.5 / 1601901100.8 * eta ** 4
        + 51004148102.5 / 1310646355.2 * eta3
        + (30060067316599.7 / 57668439628.8 - 39954.5 / 2721.6 * jnp.pi ** 2) * eta2
        + (
            -567987228950352.7 / 128152088064.0
            - 532292.8 / 396.9 * _EULER_GAMMA
            + 930221.5 / 5443.2 * jnp.pi ** 2
            - 142068.8 / 44.1 * log2
            + 2632.5 / 4.9 * log3
        ) * eta
        - 9049.0 / 56.7 * jnp.pi ** 2
        - 3681.2 / 18.9 * _EULER_GAMMA
        + 255071384399888515.3 / 83042553065472.0
        - 2632.5 / 19.6 * log3
        - 101102.0 / 396.9 * log2
    )
    coef_log8pn = -3 * (
        - 36946947827.5 / 1601901100.8 * eta ** 4
        + 51004148102.5 / 1310646355.2 * eta3
        + (30060067316599.7 / 57668439628.8 - 39954.5 / 2721.6 * jnp.pi ** 2) * eta2
        + (
            -567987228950352.7 / 128152088064.0
            - 532292.8 / 396.9 * _EULER_GAMMA
            + 930221.5 / 5443.2 * jnp.pi ** 2
            - 142068.8 / 44.1 * log2
            + 2632.5 / 4.9 * log3
        ) * eta
        - 9049.0 / 56.7 * jnp.pi ** 2
        - 3681.2 / 18.9 * _EULER_GAMMA
        + 255071384399888515.3 / 83042553065472.0
        - 2632.5 / 19.6 * log3
        - 101102.0 / 396.9 * log2
    )
    coef_loglog8pn = 9 * (266146.4 / 1190.7 * eta + 1840.6 / 56.7)
    coef_9pn = jnp.pi * (
        1032375.5 / 19958.4 * eta3
        + 4529333.5 / 12700.8 * eta2
        + (2255.0 / 6.0 * jnp.pi ** 2 - 149291726073.5 / 13412044.8) * eta
        - 640.0 / 3.0 * jnp.pi ** 2
        - 1369.6 / 2.1 * _EULER_GAMMA
        + 10534427947316.3 / 1877686272.0
        - 2739.2 / 2.1 * log2
    )
    coef_log9pn = -3 * 1369.6 / 6.3 * jnp.pi
    coef_10pn = (
        1.0 / (1.0 - 3.0 * eta) * (
            (242506658510205297979.7 / 85723270481616768.0) * eta ** 6
            - (1272143474037195162.1 / 67631771583129.6) * eta ** 5
            + (1116081080066315514991.3 / 27213736660830720.0 - 943479.7 / 1881.6 * jnp.pi ** 2) * eta ** 4
            + (
                -85710407655931086054085.1 / 3428930819264670720.0
                - 614779314.2 / 152806.5 * _EULER_GAMMA
                - 46051.9 / 153.6 * jnp.pi ** 2
                - 4311179766.8 / 152806.5 * log2
                + 127939.5 / 9.8 * log3
            ) * eta3
            + (
                -1873639936380505730110521.7 / 36575262072156487680.0
                - 9923919211.9 / 458419.5 * _EULER_GAMMA
                + 41579551.7 / 90316.8 * jnp.pi ** 2
                - 11734037971.3 / 458419.5 * log2
                - 5833093.5 / 548.8 * log3
            ) * eta2
            + (
                56993518125966874478111.3 / 1083711468804636672.0
                + 6378740752.7 / 916839.0 * _EULER_GAMMA
                - 545142954.7 / 812851.2 * jnp.pi ** 2
                + 15994339707.7 / 1833678.0 * log2
                + 892417.5 / 313.6 * log3
            ) * eta
            + 57822311.5 / 304819.2 * jnp.pi ** 2
            + 647058264.7 / 2750517.0 * _EULER_GAMMA
            - 143300652329540712655.9 / 12630669799587840.0
            - 551245.5 / 2195.2 * log3
            + 5399283943.1 / 5501034.0 * log2
        )
    )
    coef_log10pn = (
        3.0 / (1.0 - 3.0 * eta) * (
            1286378036.2 / 458419.5 * eta3
            + 1384949312.9 / 1375258.5 * eta2
            - 2427943164.1 / 2750517.0 * eta
            + 647058264.7 / 8251551.0
        )
    )
    coef_11pn = jnp.pi * (
        65762707344.5 / 14417109907.2 * eta ** 4
        - 108059782847.5 / 2621292710.4 * eta3
        + 512031495514639.7 / 62911025049.6 * eta2
        + (-1064790.5 / 3628.8 * eta2 + 4501578.5 / 14515.2 * eta - 9439.0 / 56.7) * jnp.pi ** 2
        + (
            -134666.2 / 56.7 * _EULER_GAMMA
            - 43038370739839704.7 / 3460106377728.0
            + 2632.5 / 4.9 * log3
            - 2100962.6 / 396.9 * log2
        ) * eta
        - 355801.1 / 793.8 * _EULER_GAMMA
        + 185754140723659441.1 / 27680851021824.0
        - 2632.5 / 19.6 * log3
        - 86254.9 / 113.4 * log2
    )
    coef_log11pn = -3 * jnp.pi * (134666.2 / 170.1 * eta + 355801.1 / 2381.4)

    return phi_35pn + (3.0 / 128.0 / eta / v5) * (
        (coef_8pn + coef_log8pn * logv + coef_loglog8pn * logv * logv) * v8
        + (coef_9pn + coef_log9pn * logv) * v9
        + (coef_10pn + coef_log10pn * logv) * v10
        + (coef_11pn + coef_log11pn * logv) * v11
    )


def _Af3hPN_jax(
    f: jnp.ndarray,
    M: float,
    eta: jnp.ndarray,
    s1z: jnp.ndarray = 0.0,
    s2z: jnp.ndarray = 0.0,
    Lam: jnp.ndarray = 0.0,
    dLam: jnp.ndarray = 0.0,
    Deff: float = 1.0,
) -> jnp.ndarray:
    """3.5PN amplitude for aligned-spin CBCs."""
    Mchirp = M * jnp.abs(eta) ** (3.0 / 5.0)
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    v = jnp.power(jnp.abs(jnp.pi * M * f * _SUN_MASS_SECONDS), 1.0 / 3.0)
    v2 = v * v;  v3 = v2 * v;  v4 = v2 * v2
    v5 = v4 * v; v6 = v3 * v3; v7 = v3 * v4
    eta2 = eta ** 2; eta3 = eta ** 3

    A0 = (
        jnp.abs(Mchirp) ** (5.0 / 6.0)
        / jnp.abs(f) ** (7.0 / 6.0)
        / Deff
        / jnp.abs(jnp.pi) ** (2.0 / 3.0)
        * jnp.sqrt(5.0 / 24.0)
    )

    chis = 0.5 * (s1z + s2z)
    chia = 0.5 * (s1z - s2z)
    be = 113.0 / 12.0 * (chis + delta * chia - 76.0 / 113.0 * eta * chis)
    sigma = (
        chia ** 2 * (81.0 / 16.0 - 20.0 * eta)
        + 81.0 / 8.0 * chia * chis * delta
        + chis ** 2 * (81.0 / 16.0 - eta / 4.0)
    )
    eps = delta * chia * (502429.0 / 16128.0 - 907.0 / 192.0 * eta) + chis * (
        5.0 / 48.0 * eta2 - 73921.0 / 2016.0 * eta + 502429.0 / 16128.0
    )

    return A0 * (
        1.0
        + v2 * (11.0 / 8.0 * eta + 743.0 / 672.0)
        + v3 * (be / 2.0 - 2.0 * jnp.pi)
        + v4 * (
            1379.0 / 1152.0 * eta2
            + 18913.0 / 16128.0 * eta
            + 7266251.0 / 8128512.0
            - sigma / 2.0
        )
        + v5 * (57.0 / 16.0 * jnp.pi * eta - 4757.0 * jnp.pi / 1344.0 + eps)
        + v6 * (
            856.0 / 105.0 * _EULER_GAMMA
            + 67999.0 / 82944.0 * eta3
            - 1041557.0 / 258048.0 * eta2
            - 451.0 / 96.0 * jnp.pi ** 2 * eta
            + 10.0 * jnp.pi ** 2 / 3.0
            + 3526813753.0 / 27869184.0 * eta
            - 29342493702821.0 / 500716339200.0
            + 856.0 / 105.0 * jnp.log(4.0 * v)
        )
        + v7 * (-1349.0 / 24192.0 * eta2 - 72221.0 / 24192.0 * eta - 5111593.0 / 2709504.0) * jnp.pi
    )


def _smoothly_connect_with_zero_jax(
    f_natural: jnp.ndarray,
    pn_amp: jnp.ndarray,
    pivot_1: float = 0.01,
    pivot_2: float = 0.02,
) -> jnp.ndarray:
    """Blend the PN amplitude to the constant value 20 above ``pivot_2``.

    Replaces the numpy boolean-mask version with ``jnp.where`` so the
    function is JAX-traceable.  The target value of 20 matches
    ``decreasing_function`` in the original code.
    """
    t = (f_natural - pivot_1) / (pivot_2 - pivot_1)
    smooth = (1.0 - jnp.cos(t * jnp.pi)) / 2.0        # 0 at t=0, 1 at t=1
    blended = pn_amp * (1.0 - smooth) + 20.0 * smooth  # transition region
    return jnp.where(
        f_natural < pivot_1,
        pn_amp,
        jnp.where(f_natural < pivot_2, blended, 20.0),
    )


# ================================================================== #
# Full waveform factory (NN + PCA + TaylorF2)
# ================================================================== #


def model_to_jax_waveform_ds(
    model: "Model",
) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """Build a JAX-jittable function that returns the amplitude and phase at
    the downsampled frequency grids, combining the NN+PCA residuals with
    TaylorF2 post-Newtonian results.

    The returned function mirrors what :meth:`Model._predict_amplitude_phase`
    does internally, but is fully JAX-traceable (``jax.jit``, ``jax.grad``,
    ``jax.vmap`` all work).

    Parameters
    ----------
    model : Model
        A fully loaded ``Model`` instance.

    Returns
    -------
    Callable
        ``predict(params) -> (amp_ds, phi_ds)`` where

        * ``params`` has shape ``(5,)`` = ``[q, λ₁, λ₂, χ₁, χ₂]``,
        * ``amp_ds`` has shape ``(n_amp,)`` — amplitude at the downsampled
          amplitude frequency grid,
        * ``phi_ds`` has shape ``(n_phi,)`` — phase at the downsampled phase
          frequency grid.

        The function is JIT-compatible via ``jax.jit``.

    Notes
    -----
    The output *does not* include the ``mlgw_bns_prefactor`` (which depends
    on the variable ``total_mass`` and the caller's choice of distance).
    Apply ``total_mass**2 / AMP_SI_BASE * eta`` separately if you need SI
    units.
    """
    assert model.nn is not None, "model.nn is None — load or train the model first."
    assert model.pca_data is not None, "model.pca_data is None."
    assert model.downsampling_indices is not None, "model.downsampling_indices is None."

    # -- NN+PCA residual function (accepts (1, 5), returns (1, n_amp+n_phi)) --
    predict_residuals = model_to_jax_predict(model)

    # -- Split index between amplitude and phase residuals --
    n_amp = model.downsampling_indices.amp_length

    # -- Frozen frequency grids --
    amp_freqs_hz = jnp.array(
        model.dataset.frequencies_hz[model.downsampling_indices.amplitude_indices],
        dtype=jnp.float64,
    )
    amp_freqs_natural = jnp.array(
        model.dataset.frequencies[model.downsampling_indices.amplitude_indices],
        dtype=jnp.float64,
    )
    phi_freqs_hz = jnp.array(
        model.dataset.frequencies_hz[model.downsampling_indices.phase_indices],
        dtype=jnp.float64,
    )

    # -- Reference mass used when training the dataset --
    M_ref: float = float(model.dataset.total_mass)

    def predict(
        params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Predict waveform amplitude and phase at the downsampled grids.

        Parameters
        ----------
        params : jnp.ndarray
            Shape ``(5,)`` = ``[q, λ₁, λ₂, χ₁, χ₂]``.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            ``(amp_ds, phi_ds)`` at the downsampled frequency grids.
        """
        q    = params[0]
        lam1 = params[1]
        lam2 = params[2]
        chi1 = params[3]
        chi2 = params[4]

        # Symmetric mass ratio and component masses (at reference total mass)
        eta = q / (1.0 + q) ** 2
        m1  = M_ref / (1.0 + 1.0 / q)
        m2  = M_ref / (1.0 + q)

        lambdatilde = _compute_lambda_tilde_jax(m1, m2, lam1, lam2)
        dlambda     = _compute_delta_lambda_jax(m1, m2, lam1, lam2)

        # NN+PCA → combined residuals
        combined      = predict_residuals(jnp.expand_dims(params, 0))  # (1, n_amp+n_phi)
        amp_residuals = combined[0, :n_amp]
        phi_residuals = combined[0, n_amp:]

        # TaylorF2 amplitude at the downsampled amplitude grid
        pn_amp = _Af3hPN_jax(amp_freqs_hz, M_ref, eta, chi1, chi2, lambdatilde, dlambda)
        pn_amp = pn_amp * (_TF2_BASE * _AMP_SI_BASE / eta / M_ref ** 2)
        pn_amp = _smoothly_connect_with_zero_jax(amp_freqs_natural, pn_amp)

        # TaylorF2 phase at the downsampled phase grid
        phi_5pn   = _Phif5hPN_jax(phi_freqs_hz, M_ref, eta, chi1, chi2)
        phi_tidal = _PhifT7hPNComplete_jax(phi_freqs_hz, M_ref, eta, lam1, lam2)
        phi_qm    = _PhifQM3hPN_jax(phi_freqs_hz, M_ref, eta, chi1, chi2, lam1, lam2)
        pn_phase  = -phi_5pn - phi_tidal - phi_qm
        pn_phase  = pn_phase - pn_phase[0]  # zero at first frequency point

        # Combine residuals with PN baseline
        amp_ds = jnp.exp(amp_residuals) * pn_amp
        phi_ds = phi_residuals + pn_phase

        return amp_ds, phi_ds

    return predict
