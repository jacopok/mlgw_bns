r"""JAX reimplementation of the surrogate's prediction pipeline.

Adapted from the JAX port of ``mlgw_bns`` written by **Saulo Albuquerque**
(``saulo-albuquerque-phys``) --- see ``jax_reference/`` for the verbatim
upstream files and their provenance. That work targets the single-mode
0.12.1 interface (a five-parameter ``SklearnNetwork`` MLP feeding a PCA
reconstruction, one ``(2, 2)`` polarisation); this module carries the
idea to the current multi-mode model, whose per-mode residual regressor
is a :class:`~mlgw_bns.neural_network.KernelRidgeNetwork` and whose output
is the observer-frame :math:`(h_+, h_\times)` summed over four modes.

The Post-Newtonian expansions (:func:`_taylorf2_psi` and the ``_H_2*`` /
``_H_3*`` / ``_H_4*`` mode coefficients) and the JIT-able cubic-spline
evaluator are ports of the corresponding upstream code /
:mod:`mlgw_bns.taylorf2` / :mod:`mlgw_bns.pn_modes`.

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
_LOG2 = 0.69314718055994528623
_LOG3 = 1.0986122886681097821

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
# TaylorF2 phase (natural units), mass-independent
# ====================================================================== #

def _phif3hpn(v, eta, s1z, s2z):
    """3.5PN point-mass + spin phase, tidal terms omitted (as called from
    :func:`Phif5hPN` inside ``phase_5h_post_newtonian_tidal``)."""
    vlso = 1.0 / math.sqrt(6.0)
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    v2, v3, v4, v5, v6, v7 = v**2, v**3, v**4, v**5, v**6, v**7
    eta2, eta3 = eta**2, eta**3
    m1M = 0.5 * (1.0 + delta)
    m2M = 0.5 * (1.0 - delta)
    chi1sq, chi2sq = s1z * s1z, s2z * s2z
    chi1dotchi2 = s1z * s2z
    SL = m1M * m1M * s1z + m2M * m2M * s2z
    dSigmaL = delta * (m2M * s2z - m1M * s1z)

    sigma = eta * (721.0 / 48.0 * s1z * s2z - 247.0 / 48.0 * chi1dotchi2)
    sigma += 719.0 / 96.0 * (m1M * m1M * chi1sq + m2M * m2M * chi2sq)
    sigma -= 233.0 / 96.0 * (m1M * m1M * chi1sq + m2M * m2M * chi2sq)
    phis_15PN = 188.0 * SL / 3.0 + 25.0 * dSigmaL
    ga = (554345.0 / 1134.0 + 110.0 * eta / 9.0) * SL + (
        13915.0 / 84.0 - 10.0 * eta / 3.0
    ) * dSigmaL
    pn_ss3 = (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta * s1z * s2z
    pn_ss3 += (
        (4703.5 / 8.4 + 2935.0 / 6.0 * m1M - 120.0 * m1M * m1M)
        + (-4108.25 / 6.72 - 108.5 / 1.2 * m1M + 125.5 / 3.6 * m1M * m1M)
    ) * m1M * m1M * chi1sq
    pn_ss3 += (
        (4703.5 / 8.4 + 2935.0 / 6.0 * m2M - 120.0 * m2M * m2M)
        + (-4108.25 / 6.72 - 108.5 / 1.2 * m2M + 125.5 / 3.6 * m2M * m2M)
    ) * m2M * m2M * chi2sq
    phis_3PN = math.pi * (3760.0 * SL + 1490.0 * dSigmaL) / 3.0 + pn_ss3
    phis_35PN = (
        -8980424995.0 / 762048.0 + 6586595.0 * eta / 756.0 - 305.0 * eta2 / 36.0
    ) * SL - (
        170978035.0 / 48384.0 - 2876425.0 * eta / 672.0 - 4735.0 * eta2 / 144.0
    ) * dSigmaL

    LO = 3.0 / 128.0 / eta / v5
    pointmass = (
        1
        + 20.0 / 9.0 * (743.0 / 336.0 + 11.0 / 4.0 * eta) * v2
        + (phis_15PN - 16.0 * math.pi) * v3
        + 10.0
        * (3058673.0 / 1016064.0 + 5429.0 / 1008.0 * eta + 617.0 / 144.0 * eta2 - sigma)
        * v4
        + (38645.0 / 756.0 * math.pi - 65.0 / 9.0 * eta * math.pi - ga)
        * (1.0 + 3.0 * jnp.log(v / vlso))
        * v5
        + (
            11583231236531.0 / 4694215680.0
            - 640.0 / 3.0 * math.pi**2
            - 6848.0 / 21.0 * (_EULER_GAMMA + jnp.log(4.0 * v))
            + (-15737765635.0 / 3048192.0 + 2255.0 * math.pi**2 / 12.0) * eta
            + 76055.0 / 1728.0 * eta2
            - 127825.0 / 1296.0 * eta3
            + phis_3PN
        )
        * v6
        + (
            math.pi
            * (77096675.0 / 254016.0 + 378515.0 / 1512.0 * eta - 74045.0 / 756.0 * eta2)
            + phis_35PN
        )
        * v7
    )
    return LO * pointmass


def _phif5hpn(v, eta, s1z, s2z):
    """5.5PN point-mass + spin phase (``Phif5hPN`` with Lam=dLam=0)."""
    phi_35pn = _phif3hpn(v, eta, s1z, s2z)
    v5, v8, v9, v10, v11 = v**5, v**8, v**9, v**10, v**11
    logv = jnp.log(v)
    eta2, eta3 = eta**2, eta**3
    pi2 = math.pi**2

    base8 = (
        -36946947827.5 / 1601901100.8 * eta**4
        + 51004148102.5 / 1310646355.2 * eta3
        + (30060067316599.7 / 57668439628.8 - 39954.5 / 2721.6 * pi2) * eta2
        + (
            -567987228950352.7 / 128152088064.0
            - 532292.8 / 396.9 * _EULER_GAMMA
            + 930221.5 / 5443.2 * pi2
            - 142068.8 / 44.1 * _LOG2
            + 2632.5 / 4.9 * _LOG3
        )
        * eta
        - 9049.0 / 56.7 * pi2
        - 3681.2 / 18.9 * _EULER_GAMMA
        + 255071384399888515.3 / 83042553065472.0
        - 2632.5 / 19.6 * _LOG3
        - 101102.0 / 396.9 * _LOG2
    )
    coef_8pn = base8
    coef_log8pn = -3.0 * base8
    coef_loglog8pn = 9.0 * (266146.4 / 1190.7 * eta + 1840.6 / 56.7)
    coef_9pn = math.pi * (
        1032375.5 / 19958.4 * eta3
        + 4529333.5 / 12700.8 * eta2
        + (2255.0 / 6.0 * pi2 - 149291726073.5 / 13412044.8) * eta
        - 640.0 / 3.0 * pi2
        - 1369.6 / 2.1 * _EULER_GAMMA
        + 10534427947316.3 / 1877686272.0
        - 2739.2 / 2.1 * _LOG2
    )
    coef_log9pn = -3.0 * 1369.6 / 6.3 * math.pi
    coef_10pn = (
        1.0
        / (1.0 - 3.0 * eta)
        * (
            (242506658510205297979.7 / 85723270481616768.0) * eta**6
            - (1272143474037195162.1 / 67631771583129.6) * eta**5
            + (
                1116081080066315514991.3 / 27213736660830720.0
                - 943479.7 / 1881.6 * pi2
            )
            * eta**4
            + (
                -85710407655931086054085.1 / 3428930819264670720.0
                - 614779314.2 / 152806.5 * _EULER_GAMMA
                - 46051.9 / 153.6 * pi2
                - 4311179766.8 / 152806.5 * _LOG2
                + 127939.5 / 9.8 * _LOG3
            )
            * eta3
            + (
                -1873639936380505730110521.7 / 36575262072156487680.0
                - (9923919211.9 / 458419.5) * _EULER_GAMMA
                + 41579551.7 / 90316.8 * pi2
                - 11734037971.3 / 458419.5 * _LOG2
                - 5833093.5 / 548.8 * _LOG3
            )
            * eta2
            + (
                56993518125966874478111.3 / 1083711468804636672.0
                + 6378740752.7 / 916839.0 * _EULER_GAMMA
                - 545142954.7 / 812851.2 * pi2
                + 15994339707.7 / 1833678.0 * _LOG2
                + (892417.5 / 313.6) * _LOG3
            )
            * eta
            + (57822311.5 / 304819.2) * pi2
            + (647058264.7 / 2750517.0) * _EULER_GAMMA
            - 143300652329540712655.9 / 12630669799587840.0
            - 551245.5 / 2195.2 * _LOG3
            + 5399283943.1 / 5501034.0 * _LOG2
        )
    )
    coef_log10pn = (
        3.0
        / (1.0 - 3.0 * eta)
        * (
            1286378036.2 / 458419.5 * eta3
            + 1384949312.9 / 1375258.5 * eta2
            - 2427943164.1 / 2750517.0 * eta
            + 647058264.7 / 8251551.0
        )
    )
    coef_11pn = math.pi * (
        65762707344.5 / 14417109907.2 * eta**4
        - 108059782847.5 / 2621292710.4 * eta3
        + 512031495514639.7 / 62911025049.6 * eta2
        + (-1064790.5 / 3628.8 * eta2 + 4501578.5 / 14515.2 * eta - 9439.0 / 56.7) * pi2
        + (
            -134666.2 / 56.7 * _EULER_GAMMA
            - 43038370739839704.7 / 3460106377728.0
            + 2632.5 / 4.9 * _LOG3
            - 2100962.6 / 396.9 * _LOG2
        )
        * eta
        - 355801.1 / 793.8 * _EULER_GAMMA
        + 185754140723659441.1 / 27680851021824.0
        - 2632.5 / 19.6 * _LOG3
        - 86254.9 / 113.4 * _LOG2
    )
    coef_log11pn = -3.0 * math.pi * (134666.2 / 170.1 * eta + 355801.1 / 2381.4)

    return phi_35pn + (3.0 / 128.0 / eta / v5) * (
        (coef_8pn + coef_log8pn * logv + coef_loglog8pn * logv * logv) * v8
        + (coef_9pn + coef_log9pn * logv) * v9
        + (coef_10pn + coef_log10pn * logv) * v10
        + (coef_11pn + coef_log11pn * logv) * v11
    )


def _phift7hpn(v, eta, lam1, lam2):
    """7.5PN tidal phase (``PhifT7hPNComplete``)."""
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    Xa = 0.5 * (1.0 + delta)
    Xb = 0.5 * (1.0 - delta)
    Xa2, Xa3, Xa4, Xa5 = Xa**2, Xa**3, Xa**4, Xa**5
    Xb2, Xb3, Xb4, Xb5 = Xb**2, Xb**3, Xb**4, Xb**5
    v2, v3, v4, v5 = v**2, v**3, v**4, v**5
    kapa = 3.0 * lam1 * Xa4 * Xb
    kapb = 3.0 * lam2 * Xb4 * Xa
    pNa = -3.0 / (16.0 * eta) * (12.0 + Xa / Xb)
    pNb = -3.0 / (16.0 * eta) * (12.0 + Xb / Xa)
    p1a = 5.0 * (3179.0 - 919.0 * Xa - 2286.0 * Xa2 + 260.0 * Xa3) / (
        672.0 * (12.0 - 11.0 * Xa)
    )
    p1b = 5.0 * (3179.0 - 919.0 * Xb - 2286.0 * Xb2 + 260.0 * Xb3) / (
        672.0 * (12.0 - 11.0 * Xb)
    )
    p2a = p2b = -math.pi
    p3a = (
        -5.0
        * (
            -387973870.0
            + 43246839.0 * Xa
            + 174965616.0 * Xa2
            + 158378220.0 * Xa3
            - 20427120.0 * Xa4
            + 4572288.0 * Xa5
        )
        / 27433728.0
    ) / (12.0 - 11.0 * Xa)
    p3b = (
        -5.0
        * (
            -387973870.0
            + 43246839.0 * Xb
            + 174965616.0 * Xb2
            + 158378220.0 * Xb3
            - 20427120.0 * Xb4
            + 4572288.0 * Xb5
        )
        / 27433728.0
    ) / (12.0 - 11.0 * Xb)
    p4a = (
        -math.pi
        * (27719.0 - 22415.0 * Xa + 7598.0 * Xa2 - 10520.0 * Xa3)
        / (672.0 * (12.0 - 11.0 * Xa))
    )
    p4b = (
        -math.pi
        * (27719.0 - 22127.0 * Xb + 7022.0 * Xb2 - 10232.0 * Xb3)
        / (672.0 * (12.0 - 11.0 * Xb))
    )
    return v5 * (
        kapa * pNa * (1.0 + p1a * v2 + p2a * v3 + p3a * v4 + p4a * v5)
        + kapb * pNb * (1.0 + p1b * v2 + p2b * v3 + p3b * v4 + p4b * v5)
    )


def _compute_quadrupole_yy(lam):
    loglam = jnp.log(jnp.where(lam > 0.0, lam, 1.0))
    logCQ = (
        0.194
        + 0.0936 * loglam
        + 0.0474 * loglam**2
        - 4.21e-3 * loglam**3
        + 1.23e-4 * loglam**4
    )
    return jnp.where(lam <= 0.0, 1.0, jnp.exp(logCQ))


def _phifqm3hpn(v, eta, s1z, s2z, lam1, lam2):
    """3.5PN quadrupole-monopole self-spin phase (``PhifQM3hPN``)."""
    v2 = v * v
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    X1 = 0.5 * (1.0 + delta)
    X2 = 0.5 * (1.0 - delta)
    at1_2 = (X1 * s1z) ** 2
    at2_2 = (X2 * s2z) ** 2
    CQ1 = _compute_quadrupole_yy(lam1) - 1.0
    CQ2 = _compute_quadrupole_yy(lam2) - 1.0
    a2CQ_p = at1_2 * CQ1 + at2_2 * CQ2
    a2CQ_m = at1_2 * CQ1 - at2_2 * CQ2
    out = -75.0 / (64.0 * eta) * a2CQ_p / v
    out += (
        (45.0 / 16.0 * eta + 15635.0 / 896.0) * a2CQ_p + 2215.0 / 512.0 * delta * a2CQ_m
    ) * v / eta
    out += -75.0 / (8.0 * eta) * a2CQ_p * v2 * math.pi
    return out


def _taylorf2_psi(f_natural, eta, chi1, chi2, lam1, lam2):
    r"""``phase_5h_post_newtonian_tidal`` in natural units (``M f``).

    ``v = (pi M f_Hz T_sun)^{1/3} = (pi f_natural)^{1/3}``, so the total
    mass drops out.
    """
    v = jnp.abs(math.pi * f_natural) ** (1.0 / 3.0)
    phi5 = _phif5hpn(v, eta, chi1, chi2)
    phit = _phift7hpn(v, eta, lam1, lam2)
    phiqm = _phifqm3hpn(v, eta, chi1, chi2, lam1, lam2)
    return -phi5 - phit - phiqm


# ====================================================================== #
# Per-mode dimensionless PN amplitude coefficients H_lm(v)
# ====================================================================== #

def _H_22(v, eta, delta, chi_a, chi_s):
    v2, v3, v4, v6 = v**2, v**3, v**4, v**6
    c2 = 451 * eta / 168 - 323 / 224
    c3 = 27 * delta * chi_a / 8 - 11 * eta * chi_s / 6 + 27 * chi_s / 8
    c4 = (
        -49 * delta * chi_a * chi_s / 16
        + 105271 * eta**2 / 24192
        + 6 * eta * chi_a**2
        + eta * chi_s**2 / 8
        - 1975055 * eta / 338688
        - 49 * chi_a**2 / 32
        - 49 * chi_s**2 / 32
        - 27312085 / 8128512
    )
    c6 = (
        107291 * delta * eta * chi_a * chi_s / 2688
        - 875047 * delta * chi_a * chi_s / 32256
        + 31 * math.pi * delta * chi_a / 12
        + 34473079 * eta**3 / 6386688
        + 491 * eta**2 * chi_a**2 / 84
        - 51329 * eta**2 * chi_s**2 / 4032
        - 3248849057 * eta**2 / 178827264
        + 129367 * eta * chi_a**2 / 2304
        + 8517 * eta * chi_s**2 / 224
        - 7 * math.pi * eta * chi_s / 3
        - 205 * math.pi**2 * eta / 48
        + 545384828789 * eta / 5007163392
        - 875047 * chi_a**2 / 64512
        - 875047 * chi_s**2 / 64512
        + 31 * math.pi * chi_s / 12
        + 428j * math.pi / 105
        - 177520268561 / 8583708672
    )
    return 1 + v2 * c2 + v3 * c3 + v4 * c4 + v6 * c6


def _H_21(v, eta, delta, chi_a, chi_s):
    i = 1j
    v1, v2, v3, v4, v5, v6 = v, v**2, v**3, v**4, v**5, v**6
    coef = i * math.sqrt(2) / 3
    c1 = delta
    c2 = -1.5 * delta * chi_s - 1.5 * chi_a
    c3 = 117 / 56 * delta * eta + 335 / 672 * delta
    c4 = (
        -965 / 336 * delta * eta * chi_s
        + 3427 / 1344 * delta * chi_s
        - math.pi * delta
        - i / 2 * delta
        - i / 2 * delta * math.log(16)
        - 2101 / 336 * eta * chi_a
        + 3427 / 1344 * chi_a
    )
    c5 = (
        21365 / 8064 * delta * eta**2
        + 10 * delta * eta * chi_a**2
        + 39 / 8 * delta * eta * chi_s**2
        - 36529 / 12544 * delta * eta
        - 307 / 32 * delta * chi_a**2
        - 307 / 32 * delta * chi_s**2
        + 3 * math.pi * delta * chi_s
        - 964357 / 8128512 * delta
        + 213 / 4 * eta * chi_a * chi_s
        - 307 / 16 * chi_a * chi_s
        + 3 * math.pi * chi_a
    )
    c6 = (
        -547 / 768 * delta * eta**2 * chi_s
        - 15 * delta * eta * chi_a**2 * chi_s
        - 3 / 16 * delta * eta * chi_s**3
        - 7049629 / 225792 * delta * eta * chi_s
        + 417 / 112 * math.pi * delta * eta
        - 1489 / 112 * i * delta * eta
        - 89 / 28 * i * delta * eta * math.log(2)
        + 729 / 64 * delta * chi_a**2 * chi_s
        + 243 / 64 * delta * chi_s**3
        + 143063173 / 5419008 * delta * chi_s
        - 2455 / 1344 * math.pi * delta
        - 335 / 1344 * i * delta
        - 335 / 336 * i * delta * math.log(2)
        + 42617 / 1792 * eta**2 * chi_a
        - 15 * eta * chi_a**3
        - 489 / 16 * eta * chi_a * chi_s**2
        - 22758317 / 225792 * eta * chi_a
        + 243 / 64 * chi_a**3
        + 729 / 64 * chi_a * chi_s**2
        + 143063173 / 5419008 * chi_a
    )
    return coef * (v1 * c1 + v2 * c2 + v3 * c3 + v4 * c4 + v5 * c5 + v6 * c6)


def _H_33(v, eta, delta, chi_a, chi_s):
    i = 1j
    v1, v3, v4, v5, v6 = v, v**3, v**4, v**5, v**6
    coef = -0.75 * i * math.sqrt(5 / 7)
    c1 = delta
    c3 = delta * (27 * eta / 8 - 1945 / 672)
    c4 = (
        -2 * delta * eta * chi_s / 3
        + 65 * delta * chi_s / 24
        + math.pi * delta
        - 21 * i * delta / 5
        + 6 * i * delta * math.log(3 / 2)
        - 28 * eta * chi_a / 3
        + 65 * chi_a / 24
    )
    c5 = (
        420389 * delta * eta**2 / 63360
        + 10 * delta * eta * chi_a**2
        + delta * eta * chi_s**2 / 8
        - 11758073 * delta * eta / 887040
        - 81 * eta * chi_a**2 / 32
        - 81 * eta * chi_s**2 / 32
        - 1077664867 * delta / 447068160
        + 81 * eta * chi_a * chi_s / 4
        - 81 * chi_a * chi_s / 16
    )
    c6 = (
        -67 * delta * eta**2 * chi_s / 72
        - 58745 * delta * eta * chi_s / 4032
        + 131 * math.pi * delta * eta / 16
        - 440957 * i * delta * eta / 9720
        + 69 * i * delta * eta * math.log(3 / 2) / 4
        + 163021 * delta * chi_s / 16128
        - 5675 * math.pi * delta / 1344
        + 389 * i * delta / 32
        - 1945 * i * delta * math.log(3 / 2) / 112
        - 137 * eta**2 * chi_a / 24
        - 148501 * eta * chi_a / 4032
        + 163021 * chi_a / 16128
    )
    return coef * (v1 * c1 + v3 * c3 + v4 * c4 + v5 * c5 + v6 * c6)


def _H_44(v, eta, delta, chi_a, chi_s):
    i = 1j
    v2, v4, v5, v6 = v**2, v**4, v**5, v**6
    coef = math.sqrt(10 / 7) * 4 / 9
    c2 = 3 * eta - 1
    c4 = 1063 * eta**2 / 88 - 128221 * eta / 7392 + 158383 / 36960
    c5 = (
        math.pi * (2 - 6 * eta)
        - eta
        * (1695 * eta * chi_a + 2075 * chi_s - 3579 * i + 2880 * i * math.log(2))
        / 120
        + (
            565 * delta * chi_a
            + 1140 * eta**2 * chi_s
            + 565 * chi_s
            - 1008 * i
            + 960 * i * math.log(2)
        )
        / 120
    )
    c6 = (
        eta
        * (
            243 * delta * chi_a * chi_s / 16
            + 563 * chi_a**2 / 32
            + 247 * chi_s**2 / 32
            - 22580029007 / 880588800
        )
        - 81 * delta * chi_a * chi_s / 16
        - 7606537 * eta**3 / 274560
        + eta**2 * (-30 * chi_a**2 - 3 * chi_s**2 / 8 + 901461137 / 11531520)
        - 81 * chi_a**2 / 32
        - 81 * chi_s**2 / 32
        + 7888301437 / 29059430400
    )
    return coef * (v2 * c2 + v4 * c4 + v5 * c5 + v6 * c6)


_H_BY_MODE = {(2, 2): _H_22, (2, 1): _H_21, (3, 3): _H_33, (4, 4): _H_44}


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
    chi_a = (chi1 - chi2) / 2.0
    chi_s = (chi1 + chi2) / 2.0
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
                + 2 * math.pi * (freqs[None, :] - freqs[0]) * ts_scaled[:, None]
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
