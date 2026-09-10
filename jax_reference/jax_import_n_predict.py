"""Standalone JAX waveform predictor — no mlgw_bns dependency.

This module loads the HDF5 file produced by ``jax_export.py`` and
builds a JIT-compilable, differentiable, vmap-compatible function
that generates gravitational-wave polarizations (hp, hc).

Dependencies: jax, numpy, h5py  (nothing from mlgw_bns).

Usage
-----
>>> from jax_import_n_predict import load_predict
>>> predict = load_predict("mlgw_bns_jax_model.h5")
>>>
>>> import jax.numpy as jnp
>>> params   = jnp.array([1.1, 300.0, 300.0, 0.1, 0.1])  # q, λ₁, λ₂, χ₁, χ₂
>>> freqs    = jnp.linspace(20., 2048., 2000)
>>> hp, hc   = predict(params, freqs,
...                     total_mass=jnp.array(2.8),
...                     distance_mpc=jnp.array(100.0),
...                     inclination=jnp.array(0.0))
"""

from __future__ import annotations

from typing import Callable

import h5py
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ------------------------------------------------------------------ #
# Physical constants (same values used in mlgw_bns)
# ------------------------------------------------------------------ #
_SUN_MASS_SECONDS: float = 4.92549094830932e-6
_EULER_GAMMA: float = 0.57721566490153286060
_TF2_BASE: float = 3.668693487138444e-19
_AMP_SI_BASE: float = 4.2425873413901263e24

# ------------------------------------------------------------------ #
# Activation map
# ------------------------------------------------------------------ #
_ACTIVATIONS: dict[str, Callable] = {
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "logistic": jax.nn.sigmoid,
    "identity": lambda x: x,
}


# ================================================================== #
# Natural cubic spline interpolation (JAX-traceable)
# ================================================================== #

def _make_cubic_spline_jax(
    x_ds_np: np.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Build a JAX-jittable natural cubic spline evaluator for a fixed
    knot grid.  Knot positions are frozen; values and query points are
    provided at call time.
    """
    n = len(x_ds_np) - 1

    if n < 2:
        x_jax = jnp.array(x_ds_np, dtype=jnp.float64)

        def _eval_linear(y_ds: jnp.ndarray, x_new: jnp.ndarray) -> jnp.ndarray:
            return jnp.interp(x_new, x_jax, y_ds)

        return _eval_linear

    h_np = np.diff(x_ds_np)
    sz = n - 1
    d_np = 2.0 * (h_np[:-1] + h_np[1:])
    e_np = h_np[1:-1]

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

    h_jax = jnp.array(h_np, dtype=jnp.float64)
    e_jax = jnp.array(e_np, dtype=jnp.float64)
    c_star_jax = jnp.array(c_star_np, dtype=jnp.float64)
    inv_d_jax = jnp.array(inv_d_np, dtype=jnp.float64)
    x_jax = jnp.array(x_ds_np, dtype=jnp.float64)

    def _eval_spline(y_ds: jnp.ndarray, x_new: jnp.ndarray) -> jnp.ndarray:
        dy = jnp.diff(y_ds)
        rhs = 6.0 * (dy[1:] / h_jax[1:] - dy[:-1] / h_jax[:-1])

        d0 = rhs[0] * inv_d_jax[0]

        def fwd_step(d_prev, inputs):
            e_i, inv_di, r_i = inputs
            d_new = (r_i - e_i * d_prev) * inv_di
            return d_new, d_new

        _, d_rest = jax.lax.scan(fwd_step, d0, (e_jax, inv_d_jax[1:], rhs[1:]))
        d_star = jnp.concatenate([jnp.array([d0]), d_rest])

        x_last = d_star[-1]

        def back_step(x_next, inputs):
            c_i, d_i = inputs
            x_i = d_i - c_i * x_next
            return x_i, x_i

        _, xs_rev = jax.lax.scan(back_step, x_last, (c_star_jax[::-1], d_star[:-1][::-1]))
        m_interior = jnp.concatenate([xs_rev[::-1], jnp.array([x_last])])
        m = jnp.concatenate([jnp.array([0.0]), m_interior, jnp.array([0.0])])

        idx = jnp.searchsorted(x_jax, x_new, side="right") - 1
        idx = jnp.clip(idx, 0, n - 1)

        hi = jnp.take(h_jax, idx)
        yi = jnp.take(y_ds, idx)
        yi1 = jnp.take(y_ds, idx + 1)
        mi = jnp.take(m, idx)
        mi1 = jnp.take(m, idx + 1)
        t = x_new - jnp.take(x_jax, idx)

        b_i = (yi1 - yi) / hi - hi / 6.0 * (2.0 * mi + mi1)
        c_i = mi / 2.0
        d_i = (mi1 - mi) / (6.0 * hi)

        return yi + b_i * t + c_i * t ** 2 + d_i * t ** 3

    return _eval_spline


# ================================================================== #
# Post-Newtonian helper functions (pure JAX, no external deps)
# ================================================================== #

def _compute_quadrupole_yy_jax(lam):
    loglam = jnp.log(jnp.where(lam > 0.0, lam, 1.0))
    logCQ = (
        0.194
        + 0.0936 * loglam
        + 0.0474 * loglam ** 2
        - 4.21e-3 * loglam ** 3
        + 1.23e-4 * loglam ** 4
    )
    return jnp.where(lam <= 0.0, 1.0, jnp.exp(logCQ))


def _compute_lambda_tilde_jax(m1, m2, l1, l2):
    M = m1 + m2
    return (16.0 / 13.0) * (
        (m1 + 12.0 * m2) * m1 ** 4 * l1
        + (m2 + 12.0 * m1) * m2 ** 4 * l2
    ) / M ** 5


def _compute_delta_lambda_jax(m1, m2, l1, l2):
    M = m1 + m2
    eta = (m1 * m2) / M ** 2
    X = jnp.sqrt(1.0 - 4.0 * eta)
    comb1 = (1690.0 * eta / 1319.0 - 4843.0 / 1319.0) * (m1 ** 4 * l1 - m2 ** 4 * l2) / M ** 4
    comb2 = (6162.0 * X / 1319.0) * (m1 ** 4 * l1 + m2 ** 4 * l2) / M ** 4
    return comb1 + comb2


def _PhifT7hPNComplete_jax(f, M, eta, Lama, Lamb):
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
            -387973870.0 + 43246839.0 * Xa + 174965616.0 * Xa2 + 158378220.0 * Xa3
            - 20427120.0 * Xa4 + 4572288.0 * Xa5
        ) / 27433728.0
    ) / (12.0 - 11.0 * Xa)
    p3b = (
        -5 * (
            -387973870.0 + 43246839.0 * Xb + 174965616.0 * Xb2 + 158378220.0 * Xb3
            - 20427120.0 * Xb4 + 4572288.0 * Xb5
        ) / 27433728.0
    ) / (12.0 - 11.0 * Xb)
    p4a = -jnp.pi * (27719.0 - 22415.0 * Xa + 7598.0 * Xa2 - 10520.0 * Xa3) / (672.0 * (12.0 - 11.0 * Xa))
    p4b = -jnp.pi * (27719.0 - 22127.0 * Xb + 7022.0 * Xb2 - 10232.0 * Xb3) / (672.0 * (12.0 - 11.0 * Xb))
    return v5 * (
        kapa * pNa * (1.0 + p1a * v2 + p2a * v3 + p3a * v4 + p4a * v5)
        + kapb * pNb * (1.0 + p1b * v2 + p2b * v3 + p3b * v4 + p4b * v5)
    )


def _PhifQM3hPN_jax(f, M, eta, s1z, s2z, Lam1, Lam2):
    v = jnp.power(jnp.abs(jnp.pi * M * f * _SUN_MASS_SECONDS), 1.0 / 3.0)
    v2 = v * v
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    X1 = 0.5 * (1.0 + delta)
    X2 = 0.5 * (1.0 - delta)
    at1 = X1 * s1z; at2 = X2 * s2z
    at1_2 = at1 * at1; at2_2 = at2 * at2
    CQ1 = _compute_quadrupole_yy_jax(Lam1) - 1.0
    CQ2 = _compute_quadrupole_yy_jax(Lam2) - 1.0
    a2CQ_p = at1_2 * CQ1 + at2_2 * CQ2
    a2CQ_m = at1_2 * CQ1 - at2_2 * CQ2
    PhifQM = -75.0 / (64.0 * eta) * a2CQ_p / v
    PhifQM += ((45.0 / 16.0 * eta + 15635.0 / 896.0) * a2CQ_p + 2215.0 / 512.0 * delta * a2CQ_m) * v / eta
    PhifQM += -75.0 / (8.0 * eta) * a2CQ_p * v2 * jnp.pi
    return PhifQM


def _Phif3hPN_jax(f, M, eta, s1z=0.0, s2z=0.0, Lam=0.0, dLam=0.0):
    vlso = 1.0 / jnp.sqrt(6.0)
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    v = jnp.abs(jnp.pi * M * f * _SUN_MASS_SECONDS) ** (1.0 / 3.0)
    v2 = v * v; v3 = v2 * v; v4 = v2 * v2
    v5 = v4 * v; v6 = v3 * v3; v7 = v3 * v4
    v10 = v5 * v5; v12 = v10 * v2
    eta2 = eta ** 2; eta3 = eta ** 3

    m1M = 0.5 * (1.0 + delta)
    m2M = 0.5 * (1.0 - delta)
    chi1L = s1z; chi2L = s2z
    chi1sq = s1z * s1z; chi2sq = s2z * s2z
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

    tidal = Lam * v10 * (-39.0 / 2.0 - 3115.0 / 64.0 * v2) + dLam * 6595.0 / 364.0 * v12

    return LO * (pointmass + tidal)


def _Phif5hPN_jax(f, M, eta, s1z=0.0, s2z=0.0):
    phi_35pn = _Phif3hPN_jax(f, M, eta, s1z, s2z, 0.0, 0.0)

    v = (jnp.pi * M * f * _SUN_MASS_SECONDS) ** (1.0 / 3.0)
    v2 = v * v; v3 = v2 * v; v4 = v2 * v2
    v5 = v4 * v; v6 = v3 * v3; v7 = v3 * v4
    v8 = v7 * v; v9 = v8 * v; v10 = v5 * v5; v11 = v10 * v
    logv = jnp.log(v)
    eta2 = eta ** 2; eta3 = eta ** 3
    log2 = 0.69314718055994528623
    log3 = 1.0986122886681097821

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


def _Af3hPN_jax(f, M, eta, s1z=0.0, s2z=0.0, Lam=0.0, dLam=0.0, Deff=1.0):
    Mchirp = M * jnp.abs(eta) ** (3.0 / 5.0)
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    v = jnp.power(jnp.abs(jnp.pi * M * f * _SUN_MASS_SECONDS), 1.0 / 3.0)
    v2 = v * v; v3 = v2 * v; v4 = v2 * v2
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


def _smoothly_connect_with_zero_jax(f_natural, pn_amp, pivot_1=0.01, pivot_2=0.02):
    t = (f_natural - pivot_1) / (pivot_2 - pivot_1)
    smooth = (1.0 - jnp.cos(t * jnp.pi)) / 2.0
    blended = pn_amp * (1.0 - smooth) + 20.0 * smooth
    return jnp.where(
        f_natural < pivot_1,
        pn_amp,
        jnp.where(f_natural < pivot_2, blended, 20.0),
    )


# ================================================================== #
# Main entry point: load model + build predictor
# ================================================================== #

def load_predict(path: str) -> Callable:
    """Load an exported HDF5 model and return a JAX-jittable
    ``predict(params, frequencies_hz, total_mass, distance_mpc, inclination) -> (hp, hc)``
    function.

    Parameters
    ----------
    path : str
        Path to the HDF5 file produced by ``jax_export.py``.

    Returns
    -------
    Callable
        A pure function suitable for ``jax.jit``, ``jax.grad``, ``jax.vmap``.
    """
    with h5py.File(path, "r") as f:
        # ── MLP ──────────────────────────────────────────────────────
        activation_name = f["mlp"].attrs["activation"]
        n_layers = int(f["mlp"].attrs["n_layers"])
        coefs = [jnp.array(f[f"mlp/coef_{i}"][...], dtype=jnp.float64) for i in range(n_layers)]
        intercepts = [jnp.array(f[f"mlp/intercept_{i}"][...], dtype=jnp.float64) for i in range(n_layers)]

        # ── Scaler ───────────────────────────────────────────────────
        scaler_mean = jnp.array(f["scaler/mean"][...], dtype=jnp.float64)
        scaler_scale = jnp.array(f["scaler/scale"][...], dtype=jnp.float64)

        # ── PCA ──────────────────────────────────────────────────────
        eigenvectors = jnp.array(f["pca/eigenvectors"][...], dtype=jnp.float64)
        eigenvalues = jnp.array(f["pca/eigenvalues"][...], dtype=jnp.float64)
        pca_mean = jnp.array(f["pca/mean"][...], dtype=jnp.float64)
        pca_scaling = jnp.array(f["pca/principal_components_scaling"][...], dtype=jnp.float64)
        pc_exponent = float(f["pca"].attrs["pc_exponent"])

        # ── Grid ─────────────────────────────────────────────────────
        frequencies_hz_np = f["grid/frequencies_hz"][...]
        frequencies_natural_np = f["grid/frequencies_natural"][...]
        amp_idx = f["grid/amplitude_indices"][...]
        phi_idx = f["grid/phase_indices"][...]
        M_ref = float(f["grid"].attrs["total_mass"])

    # Derived constants
    activation = _ACTIVATIONS[activation_name]
    eigenvalue_scaling = eigenvalues ** pc_exponent
    n_amp = len(amp_idx)

    amp_freqs_hz_np = frequencies_hz_np[amp_idx]
    amp_freqs_natural = jnp.array(frequencies_natural_np[amp_idx], dtype=jnp.float64)
    phi_freqs_hz_np = frequencies_hz_np[phi_idx]

    amp_freqs_hz_jax = jnp.array(amp_freqs_hz_np, dtype=jnp.float64)
    phi_freqs_hz_jax = jnp.array(phi_freqs_hz_np, dtype=jnp.float64)

    # Build cubic-spline evaluators (frozen knots)
    amp_spline = _make_cubic_spline_jax(amp_freqs_hz_np)
    phi_spline = _make_cubic_spline_jax(phi_freqs_hz_np)

    # ────────────────────────────────────────────────────────────── #
    # Internal functions (closures over frozen data)
    # ────────────────────────────────────────────────────────────── #

    def _mlp_forward(x: jnp.ndarray) -> jnp.ndarray:
        x = (x - scaler_mean) / scaler_scale
        for W, b in zip(coefs[:-1], intercepts[:-1]):
            x = activation(x @ W + b)
        x = x @ coefs[-1] + intercepts[-1]
        return x

    def _nn_pca_predict(x: jnp.ndarray) -> jnp.ndarray:
        scaled_pca = _mlp_forward(x)
        pca_components = scaled_pca / eigenvalue_scaling
        scaled_data = pca_components * pca_scaling
        zero_mean = scaled_data @ eigenvectors.T
        return zero_mean + pca_mean

    def _predict_ds(params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        q = params[0]; lam1 = params[1]; lam2 = params[2]
        chi1 = params[3]; chi2 = params[4]

        eta = q / (1.0 + q) ** 2
        m1 = M_ref / (1.0 + 1.0 / q)
        m2 = M_ref / (1.0 + q)

        lambdatilde = _compute_lambda_tilde_jax(m1, m2, lam1, lam2)
        dlambda = _compute_delta_lambda_jax(m1, m2, lam1, lam2)

        combined = _nn_pca_predict(jnp.expand_dims(params, 0))
        amp_residuals = combined[0, :n_amp]
        phi_residuals = combined[0, n_amp:]

        pn_amp = _Af3hPN_jax(amp_freqs_hz_jax, M_ref, eta, chi1, chi2, lambdatilde, dlambda)
        pn_amp = pn_amp * (_TF2_BASE * _AMP_SI_BASE / eta / M_ref ** 2)
        pn_amp = _smoothly_connect_with_zero_jax(amp_freqs_natural, pn_amp)

        phi_5pn = _Phif5hPN_jax(phi_freqs_hz_jax, M_ref, eta, chi1, chi2)
        phi_tidal = _PhifT7hPNComplete_jax(phi_freqs_hz_jax, M_ref, eta, lam1, lam2)
        phi_qm = _PhifQM3hPN_jax(phi_freqs_hz_jax, M_ref, eta, chi1, chi2, lam1, lam2)
        pn_phase = -phi_5pn - phi_tidal - phi_qm
        pn_phase = pn_phase - pn_phase[0]

        amp_ds = jnp.exp(amp_residuals) * pn_amp
        phi_ds = phi_residuals + pn_phase

        return amp_ds, phi_ds

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
            Shape ``(5,)`` = ``[q, lambda_1, lambda_2, chi_1, chi_2]``.
        frequencies_hz : jnp.ndarray
            Query frequencies in Hz, shape ``(k,)``.
        total_mass : scalar
            Total binary mass in solar masses.
        distance_mpc : scalar
            Luminosity distance in Mpc.
        inclination : scalar
            Inclination angle in radians.

        Returns
        -------
        (hp, hc) : tuple of jnp.ndarray
            Complex plus and cross polarizations, shape ``(k,)``.
        """
        amp_ds, phi_ds = _predict_ds(params)

        rescaled_freqs = frequencies_hz * (total_mass / M_ref)

        amp = amp_spline(amp_ds, rescaled_freqs)
        phi = phi_spline(phi_ds, rescaled_freqs)

        eta = params[0] / (1.0 + params[0]) ** 2
        pre = total_mass ** 2 / _AMP_SI_BASE * eta / distance_mpc
        amp = amp * pre

        h_real = amp * jnp.cos(phi)
        h_imag = amp * jnp.sin(phi)

        cosi = jnp.cos(inclination)
        pre_plus = (1.0 + cosi ** 2) / 2.0
        pre_cross = cosi

        hp = pre_plus * h_real + 1j * pre_plus * h_imag
        hc = pre_cross * h_imag - 1j * pre_cross * h_real

        return hp, hc

    return predict
