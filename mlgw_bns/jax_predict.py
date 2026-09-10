r"""JAX reimplementation of the surrogate's parameters-to-residuals map.

Adapted from the JAX port of ``mlgw_bns`` written by **Saulo Albuquerque**
(``saulo-albuquerque-phys``) --- see ``jax_reference/`` for the verbatim
upstream files and their provenance. That work targets the single-mode
0.12.1 interface (a five-parameter ``SklearnNetwork`` MLP feeding a PCA
reconstruction); this module carries the same idea over to the current
multi-mode model, whose per-mode residual regressor is a
:class:`~mlgw_bns.neural_network.KernelRidgeNetwork`.

What is ported so far
---------------------
* :func:`neural_network_to_jax` --- the parameters -> scaled-PCA-coefficient
  map, for both regressor backends (``KernelRidgeNetwork`` and the legacy
  ``SklearnNetwork`` MLP), as a pure JAX function.
* :func:`mode_model_to_jax_residuals` --- the full
  :meth:`~mlgw_bns.mode_model.ModeModel.predict_residuals_bulk`: the
  regression above followed by the PCA eigenvalue un-scaling and
  reconstruction, returning the combined ``(amp | phi)`` residual vector
  at the mode's downsampling nodes.
* :func:`make_cubic_spline_jax` --- a JIT/``vmap``-able natural cubic
  spline evaluator with frozen knots, from the upstream code. NB: the
  library resamples with a *not-a-knot* spline
  (:class:`scipy.interpolate.CubicSpline`), so this matches it only in the
  interior; the boundary intervals differ at the ~1e-3 level.

Still on numpy (the per-mode PN mode expansions
:mod:`~mlgw_bns.pn_modes`, the ``ModePhasesNN`` / ``TimeshiftsNN``
reference predictors, the :math:`{}_{-2}Y_{\ell m}` projection) --- so
this is not yet a full ``Model.predict`` replacement, only the regression
core that :mod:`experiments.timing` isolates.

Every function returns a pure ``jax`` callable that accepts a parameter
row of shape ``(n_params,)`` or a batch ``(n, n_params)``; use
``jax.jit`` / ``jax.vmap`` freely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np

# The sklearn weights are float64; keep JAX in x64 or they are silently
# downcast to float32 on the first operation.
jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from .mode_model import ModeModel
    from .neural_network import KernelRidgeNetwork, NeuralNetwork, SklearnNetwork

_ACTIVATIONS: dict[str, Callable] = {
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "logistic": jax.nn.sigmoid,
    "identity": lambda x: x,
}


def _as2d(x: jnp.ndarray) -> tuple[jnp.ndarray, bool]:
    """Promote a 1-D parameter row to ``(1, n)``; report whether it was 1-D."""
    x = jnp.asarray(x, dtype=jnp.float64)
    if x.ndim == 1:
        return x[None, :], True
    return x, False


def kernel_ridge_network_to_jax(
    nn: "KernelRidgeNetwork",
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    r"""JAX version of :meth:`KernelRidgeNetwork.predict`.

    Reproduces ``target_scaler.inverse_transform(K @ dual_coef_)`` with
    ``K_ij = exp(-gamma ||x_i - X_fit_j||^2)`` on standardized inputs, in
    scikit-learn's exact floating-point order (see the note in
    :meth:`KernelRidgeNetwork.predict`).

    Returns
    -------
    Callable
        ``predict(params) -> scaled_pca_components`` for ``params`` of
        shape ``(n, n_params)``, output shape ``(n, n_components)``.
    """
    regressor = nn.regressor
    x_fit = jnp.asarray(regressor.X_fit_, dtype=jnp.float64)
    dual_coef = jnp.asarray(regressor.dual_coef_, dtype=jnp.float64)
    sq_norm = jnp.asarray(np.einsum("ij,ij->i", regressor.X_fit_, regressor.X_fit_))
    gamma = regressor.gamma
    if gamma is None:
        gamma = 1.0 / x_fit.shape[1]
    gamma = float(gamma)

    param_mean = jnp.asarray(nn.param_scaler.mean_, dtype=jnp.float64)
    param_scale = jnp.asarray(nn.param_scaler.scale_, dtype=jnp.float64)
    target_mean = jnp.asarray(nn.target_scaler.mean_, dtype=jnp.float64)
    target_scale = jnp.asarray(nn.target_scaler.scale_, dtype=jnp.float64)

    def predict(params: jnp.ndarray) -> jnp.ndarray:
        params = jnp.asarray(params, dtype=jnp.float64)
        scaled_x = (params - param_mean) / param_scale
        sq_dist = -2.0 * (scaled_x @ x_fit.T)
        sq_dist += jnp.einsum("ij,ij->i", scaled_x, scaled_x)[:, None]
        sq_dist += sq_norm[None, :]
        sq_dist = jnp.maximum(sq_dist, 0.0)
        kernel = jnp.exp(-gamma * sq_dist)
        return (kernel @ dual_coef) * target_scale + target_mean

    return predict


def sklearn_mlp_to_jax(
    nn: "SklearnNetwork",
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """JAX version of the legacy ``SklearnNetwork`` MLP forward pass.

    Kept for models trained with the multi-layer-perceptron backend;
    the shipped ``default_hom`` uses :class:`KernelRidgeNetwork`.
    """
    scaler_mean = jnp.asarray(nn.param_scaler.mean_, dtype=jnp.float64)
    scaler_scale = jnp.asarray(nn.param_scaler.scale_, dtype=jnp.float64)
    coefs = [jnp.asarray(w, dtype=jnp.float64) for w in nn.nn.coefs_]
    intercepts = [jnp.asarray(b, dtype=jnp.float64) for b in nn.nn.intercepts_]
    activation = _ACTIVATIONS[nn.nn.activation]

    def predict(params: jnp.ndarray) -> jnp.ndarray:
        x = (jnp.asarray(params, dtype=jnp.float64) - scaler_mean) / scaler_scale
        for w, b in zip(coefs[:-1], intercepts[:-1]):
            x = activation(x @ w + b)
        return x @ coefs[-1] + intercepts[-1]

    return predict


def neural_network_to_jax(
    nn: "NeuralNetwork",
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Dispatch to the right backend port for ``nn``."""
    from .neural_network import KernelRidgeNetwork, SklearnNetwork

    if isinstance(nn, KernelRidgeNetwork):
        return kernel_ridge_network_to_jax(nn)
    if isinstance(nn, SklearnNetwork):
        return sklearn_mlp_to_jax(nn)
    raise TypeError(
        f"No JAX port for regressor backend {type(nn).__name__!r}; "
        "supported: KernelRidgeNetwork, SklearnNetwork."
    )


def mode_model_to_jax_residuals(
    mode_model: "ModeModel",
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    r"""JAX version of :meth:`ModeModel.predict_residuals_bulk`.

    ``params -> combined_residuals``: the parameters-to-coefficients
    regression, divided by ``eigenvalues ** pc_exponent``, then the PCA
    reconstruction ``(c * pcs_scaling) @ eigenvectors.T + mean``. The
    output is the concatenated ``(amp_residual | phi_residual)`` vector at
    the mode's downsampling nodes, shape ``(n, n_amp + n_phi)``.

    Parameters
    ----------
    mode_model : ModeModel
        A loaded mode model (``nn`` and ``pca_data`` present).
    """
    regressor_predict = neural_network_to_jax(mode_model.nn)

    pca = mode_model.pca_data
    pc_exponent = float(mode_model.nn.hyper.pc_exponent)
    eig_scaling = jnp.asarray(pca.eigenvalues ** pc_exponent, dtype=jnp.float64)
    pcs_scaling = jnp.asarray(pca.principal_components_scaling, dtype=jnp.float64)
    eigenvectors = jnp.asarray(pca.eigenvectors, dtype=jnp.float64)
    pca_mean = jnp.asarray(pca.mean, dtype=jnp.float64)

    def predict(params: jnp.ndarray) -> jnp.ndarray:
        params2d, was_1d = _as2d(params)
        scaled_components = regressor_predict(params2d)
        components = scaled_components / eig_scaling
        reconstructed = (components * pcs_scaling) @ eigenvectors.T + pca_mean
        return reconstructed[0] if was_1d else reconstructed

    return predict


def make_cubic_spline_jax(
    x_knots: np.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Natural cubic spline evaluator with frozen knots (from the upstream
    JAX port). ``eval(y_knots, x_query) -> values``; JIT/``vmap``-able.

    The library uses a *not-a-knot* spline, so this agrees with it in the
    interior but differs on the two boundary intervals.
    """
    x_np = np.asarray(x_knots, dtype=np.float64)
    n = len(x_np) - 1

    if n < 2:
        x_j = jnp.asarray(x_np)

        def _linear(y_knots: jnp.ndarray, x_query: jnp.ndarray) -> jnp.ndarray:
            return jnp.interp(x_query, x_j, y_knots)

        return _linear

    h_np = np.diff(x_np)
    sz = n - 1
    d_np = 2.0 * (h_np[:-1] + h_np[1:])
    e_np = h_np[1:-1]

    c_star_np = np.zeros(max(sz - 1, 0))
    inv_d_np = np.zeros(sz)
    inv_d_np[0] = 1.0 / d_np[0]
    if sz > 1:
        c_star_np[0] = e_np[0] * inv_d_np[0]
    for i in range(1, sz):
        denom = d_np[i] - e_np[i - 1] * c_star_np[i - 1]
        inv_d_np[i] = 1.0 / denom
        if i < sz - 1:
            c_star_np[i] = e_np[i] * inv_d_np[i]

    h_j = jnp.asarray(h_np)
    e_j = jnp.asarray(e_np)
    c_star_j = jnp.asarray(c_star_np)
    inv_d_j = jnp.asarray(inv_d_np)
    x_j = jnp.asarray(x_np)

    def _eval(y_knots: jnp.ndarray, x_query: jnp.ndarray) -> jnp.ndarray:
        dy = jnp.diff(y_knots)
        rhs = 6.0 * (dy[1:] / h_j[1:] - dy[:-1] / h_j[:-1])
        d0 = rhs[0] * inv_d_j[0]

        def fwd(d_prev, ins):
            e_i, inv_di, r_i = ins
            d_new = (r_i - e_i * d_prev) * inv_di
            return d_new, d_new

        _, d_rest = jax.lax.scan(fwd, d0, (e_j, inv_d_j[1:], rhs[1:]))
        d_star = jnp.concatenate([jnp.array([d0]), d_rest])

        x_last = d_star[-1]

        def back(x_next, ins):
            c_i, d_i = ins
            x_i = d_i - c_i * x_next
            return x_i, x_i

        _, xs_rev = jax.lax.scan(back, x_last, (c_star_j[::-1], d_star[:-1][::-1]))
        m_int = jnp.concatenate([xs_rev[::-1], jnp.array([x_last])])
        m = jnp.concatenate([jnp.array([0.0]), m_int, jnp.array([0.0])])

        idx = jnp.clip(jnp.searchsorted(x_j, x_query, side="right") - 1, 0, n - 1)
        hi = jnp.take(h_j, idx)
        yi = jnp.take(y_knots, idx)
        yi1 = jnp.take(y_knots, idx + 1)
        mi = jnp.take(m, idx)
        mi1 = jnp.take(m, idx + 1)
        t = x_query - jnp.take(x_j, idx)
        b_i = (yi1 - yi) / hi - hi / 6.0 * (2.0 * mi + mi1)
        c_i = mi / 2.0
        d_i = (mi1 - mi) / (6.0 * hi)
        return yi + b_i * t + c_i * t**2 + d_i * t**3

    return _eval
