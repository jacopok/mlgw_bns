"""TaylorF2 Post-Newtonian functions. 

Code adapted from `bajes <https://arxiv.org/abs/2102.00017>`_, 
which can be found in `this repo <https://github.com/matteobreschi/bajes>`_.
"""

from typing import TYPE_CHECKING, Callable

import numpy as np
from numba import njit  # type: ignore

from .frequency_of_merger import frequency_of_merger

if TYPE_CHECKING:
    from .dataset_generation import WaveformParameters

SUN_MASS_SECONDS: float = 4.92549094830932e-6  # M_sun * G / c**3
EULER_GAMMA = 0.57721566490153286060


def smoothing_func(x: np.ndarray) -> np.ndarray:
    """A function [0, 1] -> [0, 1]
    with zero derivative at the edges.
    """

    return (1 - np.cos(x * np.pi)) / 2


@njit(cache=True)
def compute_quadrupole_yy(lam: float) -> float:
    """Compute quadrupole coefficient from Lambda
    using the chi precessing spin parameter (for given 3-dim spin vectors)"""

    if lam <= 0.0:
        return 1.0
    loglam = np.log(lam)
    logCQ = (
        0.194
        + 0.0936 * loglam
        + 0.0474 * loglam ** 2
        - 4.21e-3 * loglam ** 3
        + 1.23e-4 * loglam ** 4
    )
    return np.exp(logCQ)


@njit(cache=True)
def compute_lambda_tilde(m1: float, m2: float, l1: float, l2: float) -> float:
    """Compute Lambda Tilde from masses and tides components
    --------
    m1 = primary mass component [solar masses]
    m2 = secondary mass component [solar masses]
    l1 = primary tidal component [dimensionless]
    l2 = secondary tidal component [dimensionless]
    """
    M = m1 + m2
    m1_4 = m1 ** 4.0
    m2_4 = m2 ** 4.0
    M5 = M ** 5.0
    comb1 = m1 + 12.0 * m2
    comb2 = m2 + 12.0 * m1
    return (16.0 / 13.0) * (comb1 * m1_4 * l1 + comb2 * m2_4 * l2) / M5


@njit(cache=True)
def compute_delta_lambda(m1: float, m2: float, l1: float, l2: float):
    """Compute delta Lambda Tilde from masses and tides components
    --------
    m1 = primary mass component [solar masses]
    m2 = secondary mass component [solar masses]
    l1 = primary tidal component [dimensionless]
    l2 = secondary tidal component [dimensionless]
    """
    M = m1 + m2
    q = m1 / m2
    eta = q / ((1.0 + q) * (1.0 + q))
    X = np.sqrt(1.0 - 4.0 * eta)
    m1_4 = m1 ** 4.0
    m2_4 = m2 ** 4.0
    M4 = M ** 4.0
    comb1 = (1690.0 * eta / 1319.0 - 4843.0 / 1319.0) * (m1_4 * l1 - m2_4 * l2) / M4
    comb2 = (6162.0 * X / 1319.0) * (m1_4 * l1 + m2_4 * l2) / M4
    return comb1 + comb2


@njit(cache=True)
def PhifT7hPN(
    f: np.ndarray, M: float, eta: float, Lama: float, Lamb: float
) -> np.ndarray:
    """Compute 7.5PN tidal phase correction
    Appendix B [https://arxiv.org/abs/1203.4352]
    """
    v = np.power(np.abs(np.pi * M * f * SUN_MASS_SECONDS), 1.0 / 3.0)
    delta = np.sqrt(1.0 - 4.0 * eta)
    Xa = 0.5 * (1.0 + delta)
    Xb = 0.5 * (1.0 - delta)
    Xa2 = Xa * Xa
    Xa3 = Xa2 * Xa
    Xa4 = Xa3 * Xa
    Xa5 = Xa4 * Xa
    Xb2 = Xb * Xb
    Xb3 = Xb2 * Xb
    Xb4 = Xb3 * Xb
    Xb5 = Xb4 * Xb
    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v5 = v4 * v
    # v10 = v**10
    beta221a, beta222a, beta311a, beta331a = 0.0, 0.0, 0.0, 0.0
    beta221b, beta222b, beta311b, beta331b = 0.0, 0.0, 0.0, 0.0
    kapa = 3.0 * Lama * Xa4 * Xb
    kapb = 3.0 * Lamb * Xb4 * Xa
    pNa = (
        -3.0 / (16.0 * eta) * (12.0 + Xa / Xb)
    )  # -4.*(12. - 11.*Xa)*Xa4 # w\ LO term 3./(128.*eta) factored out
    pNb = (
        -3.0 / (16.0 * eta) * (12.0 + Xb / Xa)
    )  # -4.*(12. - 11.*Xb)*Xb4 # w\ LO term 3./(128.*eta) factored out
    p1a = (
        5.0
        * (3179.0 - 919.0 * Xa - 2286.0 * Xa2 + 260.0 * Xa3)
        / (672.0 * (12.0 - 11.0 * Xa))
    )
    p1b = (
        5.0
        * (3179.0 - 919.0 * Xb - 2286.0 * Xb2 + 260.0 * Xb3)
        / (672.0 * (12.0 - 11.0 * Xb))
    )
    p2a = -np.pi
    p2b = -np.pi
    p3a = (
        39927845.0 / 508032.0
        - 480043345.0 / 9144576.0 * Xa
        + 9860575.0 / 127008.0 * Xa2
        - 421821905.0 / 2286144.0 * Xa3
        + 4359700.0 / 35721.0 * Xa4
        - 10578445.0 / 285768.0 * Xa5
    )
    p3a += (
        5.0 / 9.0 * (1.0 - 2.0 / 3.0 * Xa) * beta222a
        + 5.0 / 684.0 * (3.0 - 13.0 * Xa + 18.0 * Xa2 - 8.0 * Xa3) * beta221a
        + Xb2 * (1.0 - 2.0 * Xa) * (5.0 / 36288.0 * beta311a + 675.0 / 448.0 * beta331a)
    )
    p3a = p3a / (12.0 - 11.0 * Xa)
    p3b = (
        39927845.0 / 508032.0
        - 480043345.0 / 9144576.0 * Xb
        + 9860575.0 / 127008.0 * Xb2
        - 421821905.0 / 2286144.0 * Xb3
        + 4359700.0 / 35721.0 * Xb4
        - 10578445.0 / 285768.0 * Xb5
    )
    p3b += (
        5.0 / 9.0 * (1.0 - 2.0 / 3.0 * Xb) * beta222b
        + 5.0 / 684.0 * (3.0 - 13.0 * Xb + 18.0 * Xb2 - 8.0 * Xb3) * beta221b
        + Xa2 * (1.0 - 2.0 * Xb) * (5.0 / 36288.0 * beta311b + 675.0 / 448.0 * beta331b)
    )
    p3b = p3b / (12.0 - 11.0 * Xb)
    p4a = (
        -np.pi
        * (27719.0 - 22127.0 * Xa + 7022.0 * Xa2 - 10232.0 * Xa3)
        / (672.0 * (12.0 - 11.0 * Xa))
    )
    p4b = (
        -np.pi
        * (27719.0 - 22127.0 * Xb + 7022.0 * Xb2 - 10232.0 * Xb3)
        / (672.0 * (12.0 - 11.0 * Xb))
    )
    # LO = 3.0/128.0/eta/v5
    return v5 * (
        kapa * pNa * (1.0 + p1a * v2 + p2a * v3 + p3a * v4 + p4a * v5)
        + kapb * pNb * (1.0 + p1b * v2 + p2b * v3 + p3b * v4 + p4b * v5)
    )


@njit(cache=True)
def PhifT7hPNComplete(
    f: np.ndarray, M: float, eta: float, Lama: float, Lamb: float
) -> np.ndarray:
    """Compute 7.5PN tidal phase correction
    https://arxiv.org/abs/2005.13367
    """
    v = np.power(np.abs(np.pi * M * f * SUN_MASS_SECONDS), 1.0 / 3.0)
    delta = np.sqrt(1.0 - 4.0 * eta)
    Xa = 0.5 * (1.0 + delta)
    Xb = 0.5 * (1.0 - delta)
    Xa2 = Xa * Xa
    Xa3 = Xa2 * Xa
    Xa4 = Xa3 * Xa
    Xa5 = Xa4 * Xa
    Xb2 = Xb * Xb
    Xb3 = Xb2 * Xb
    Xb4 = Xb3 * Xb
    Xb5 = Xb4 * Xb
    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v5 = v4 * v
    # v10 = v**10
    kapa = 3.0 * Lama * Xa4 * Xb
    kapb = 3.0 * Lamb * Xb4 * Xa
    pNa = (
        -3.0 / (16.0 * eta) * (12.0 + Xa / Xb)
    )  # -4.*(12. - 11.*Xa)*Xa4 # w\ LO term 3./(128.*eta) factored out
    pNb = (
        -3.0 / (16.0 * eta) * (12.0 + Xb / Xa)
    )  # -4.*(12. - 11.*Xb)*Xb4 # w\ LO term 3./(128.*eta) factored out
    p1a = (
        5.0
        * (3179.0 - 919.0 * Xa - 2286.0 * Xa2 + 260.0 * Xa3)
        / (672.0 * (12.0 - 11.0 * Xa))
    )
    p1b = (
        5.0
        * (3179.0 - 919.0 * Xb - 2286.0 * Xb2 + 260.0 * Xb3)
        / (672.0 * (12.0 - 11.0 * Xb))
    )
    p2a = -np.pi
    p2b = -np.pi
    p3a = (
        -5
        * (
            -387973870.0
            + 43246839.0 * Xa
            + 174965616.0 * Xa2
            + 158378220.0 * Xa3
            - 20427120.0 * Xa4
            + 4572288.0 * Xa5
        )
        / 27433728.0
    )
    p3a = p3a / (12.0 - 11.0 * Xa)
    p3b = (
        -5
        * (
            -387973870.0
            + 43246839.0 * Xb
            + 174965616.0 * Xb2
            + 158378220.0 * Xb3
            - 20427120.0 * Xb4
            + 4572288.0 * Xb5
        )
        / 27433728.0
    )
    p3b = p3b / (12.0 - 11.0 * Xb)
    p4a = (
        -np.pi
        * (27719.0 - 22415.0 * Xa + 7598.0 * Xa2 - 10520.0 * Xa3)
        / (672.0 * (12.0 - 11.0 * Xa))
    )
    p4b = (
        -np.pi
        * (27719.0 - 22127.0 * Xb + 7022.0 * Xb2 - 10232.0 * Xb3)
        / (672.0 * (12.0 - 11.0 * Xb))
    )
    # LO = 3.0/128.0/eta/v5
    return v5 * (
        kapa * pNa * (1.0 + p1a * v2 + p2a * v3 + p3a * v4 + p4a * v5)
        + kapb * pNb * (1.0 + p1b * v2 + p2b * v3 + p3b * v4 + p4b * v5)
    )


@njit(cache=True)
def PhifQM3hPN(
    f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam1=0.0, Lam2=0.0
):
    """Compute post-Newtonian EOS-dependent self-spin term @ 3.5PN for compact binary coalescences
    Eq.(50-52) [https://arxiv.org/abs/1812.07923]
    Uses Love-Q relation of Yunes-Yagi, Eq.(41) [https://arxiv.org/abs/1806.01772]
    --------
    f = frequency series [Hz]
    M = total mass [solar masses]
    eta = symmetric mass ratio [dimensionless]
    s1x = primary spin component along x axis [dimensionless]
    s1y = primary spin component along y axis [dimensionless]
    s1z = primary spin component along z axis [dimensionless]
    s2x = secondary spin component along x axis [dimensionless]
    s2y = secondary spin component along y axis [dimensionless]
    s2z = secondary spin component along z axis [dimensionless]
    Lam1 = primary tidal parameter ell=2 [dimensionless]
    Lam2 = secondary tidal parameter ell=2 [dimensionless]
    """
    # TODO: this implementation needs a check
    # -> in MLGW_BNS L1=L2=0 will not happen
    # if Lam1 == 0. and Lam2 == 0. :  return 0.
    v = np.power(np.abs(np.pi * M * f * SUN_MASS_SECONDS), 1.0 / 3.0)
    v2 = v * v
    delta = np.sqrt(1.0 - 4.0 * eta)
    X1 = 0.5 * (1.0 + delta)
    X2 = 0.5 * (1.0 - delta)
    at1 = X1 * s1z
    at2 = X2 * s2z
    at1_2 = at1 * at1
    at2_2 = at2 * at2
    CQ1 = compute_quadrupole_yy(Lam1) - 1.0  # remove BBH contrib.
    CQ2 = compute_quadrupole_yy(Lam2) - 1.0  # remove BBH contrib.
    a2CQ_p_a2CQ = at1_2 * CQ1 + at2_2 * CQ2
    a2CQ_m_a2CQ = at1_2 * CQ1 - at2_2 * CQ2
    PhifQM = -75.0 / (64.0 * eta) * a2CQ_p_a2CQ / v  # LO
    PhifQM += (
        (
            (45.0 / 16.0 * eta + 15635.0 / 896.0) * a2CQ_p_a2CQ
            + 2215.0 / 512 * delta * a2CQ_m_a2CQ
        )
        * v
        / eta
    )  # NLO
    PhifQM += -75.0 / (8.0 * eta) * a2CQ_p_a2CQ * v2 * np.pi  # Tail
    return PhifQM


@njit(cache=True)
def Phif5hPN(
    f: np.ndarray,
    M: float,
    eta: float,
    s1x: float = 0.0,
    s1y: float = 0.0,
    s1z: float = 0.0,
    s2x: float = 0.0,
    s2y: float = 0.0,
    s2z: float = 0.0,
    Lam: float = 0.0,
    dLam: float = 0.0,
) -> np.ndarray:
    """Compute post-Newtonian phase @ 5.5PN for compact binary coalescences
    including spins contributions and tidal effects @ 6PN (if Lam or dLam != 0)
    [https://arxiv.org/abs/1904.09558]
    --------
    f = frequency series [Hz]
    M = binary mass [solar masses]
    s1x = primary spin component along x axis [dimensionless]
    s1y = primary spin component along y axis [dimensionless]
    s1z = primary spin component along z axis [dimensionless]
    s2x = secondary spin component along x axis [dimensionless]
    s2y = secondary spin component along y axis [dimensionless]
    s2z = secondary spin component along z axis [dimensionless]
    Lam = reduced tidal deformability parameter [dimensionless]
    dLam = asymmetric reduced tidal deformation parameter [dimensionless]
    """

    vlso = 1.0 / np.sqrt(6.0)
    delta = np.sqrt(1.0 - 4.0 * eta)
    v = (np.pi * M * f * SUN_MASS_SECONDS) ** (1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3
    v7 = v3 * v4
    v8 = v7 * v
    v9 = v8 * v
    v10 = v5 * v5
    v11 = v10 * v
    logv = np.log(v)
    eta2 = eta ** 2
    eta3 = eta ** 3
    log2 = 0.69314718055994528623
    log3 = 1.0986122886681097821

    phi_35pn = Phif3hPN(f, M, eta, s1x, s1y, s1z, s2x, s2y, s2z, Lam, dLam)

    c_21_3PN = 0.0
    c_22_4PN = 0.0
    c_22_5PN = 0.0
    a_6_c = 0.0

    coef_8pn = (
        c_21_3PN * (4.0 / 8.1 * eta - 16.0 / 8.1 * eta2)
        + c_22_4PN * 160.0 / 9.0 * eta
        - 36946947827.5 / 1601901100.8 * eta * eta * eta * eta
        + 51004148102.5 / 1310646355.2 * eta3
        + (30060067316599.7 / 57668439628.8 - 39954.5 / 2721.6 * np.pi * np.pi)
        * eta
        * eta
        + (
            -567987228950352.7 / 128152088064.0
            - 532292.8 / 396.9 * EULER_GAMMA
            + 930221.5 / 5443.2 * np.pi * np.pi
            - 142068.8 / 44.1 * log2
            + 2632.5 / 4.9 * log3
        )
        * eta
        - 9049.0 / 56.7 * np.pi * np.pi
        - 3681.2 / 18.9 * EULER_GAMMA
        + 255071384399888515.3 / 83042553065472.0
        - 2632.5 / 19.6 * log3
        - 101102.0 / 396.9 * log2
    )

    coef_log8pn = -3 * (
        c_21_3PN * (4.0 / 8.1 * eta - 16.0 / 8.1 * eta * eta)
        + c_22_4PN * 160.0 / 9.0 * eta
        - 36946947827.5 / 1601901100.8 * eta * eta * eta * eta
        + 51004148102.5 / 1310646355.2 * eta * eta * eta
        + (30060067316599.7 / 57668439628.8 - 39954.5 / 2721.6 * np.pi * np.pi)
        * eta
        * eta
        + (
            -567987228950352.7 / 128152088064.0
            - 532292.8 / 396.9 * EULER_GAMMA
            + 930221.5 / 5443.2 * np.pi * np.pi
            - 142068.8 / 44.1 * log2
            + 2632.5 / 4.9 * log3
        )
        * eta
        - 9049.0 / 56.7 * np.pi * np.pi
        - 3681.2 / 18.9 * EULER_GAMMA
        + 255071384399888515.3 / 83042553065472.0
        - 2632.5 / 19.6 * log3
        - 101102.0 / 396.9 * log2
    )

    coef_loglog8pn = 9 * (266146.4 / 1190.7 * eta + 1840.6 / 56.7)

    coef_9pn = np.pi * (
        1032375.5 / 19958.4 * eta * eta * eta
        + 4529333.5 / 12700.8 * eta * eta
        + (2255.0 / 6.0 * np.pi * np.pi - 149291726073.5 / 13412044.8) * eta
        - 640.0 / 3.0 * np.pi * np.pi
        - 1369.6 / 2.1 * EULER_GAMMA
        + 10534427947316.3 / 1877686272.0
        - 2739.2 / 2.1 * log2
    )

    coef_log9pn = -3 * 1369.6 / 6.3 * np.pi

    coef_10pn = (
        1.0
        / (1.0 - 3.0 * eta)
        * (
            a_6_c * (72.0 * eta - 216.0 * eta * eta)
            + c_21_3PN
            * (
                -76.4 / 2.1 * eta * eta * eta * eta
                - 59.9 / 6.3 * eta * eta * eta
                + 281.5 / 18.9 * eta * eta
                - 48.4 / 18.9 * eta
            )
            + c_22_4PN
            * (
                2564.0 / 7.0 * eta * eta * eta
                - 69.8 / 2.1 * eta * eta
                - 62.2 / 2.1 * eta
            )
            + c_22_5PN * (48.0 * eta * eta - 16.0 * eta)
            + (242506658510205297979.7 / 85723270481616768.0)
            * eta
            * eta
            * eta
            * eta
            * eta
            * eta
            - (1272143474037195162.1 / 67631771583129.6) * eta * eta * eta * eta * eta
            + (
                1116081080066315514991.3 / 27213736660830720.0
                - 943479.7 / 1881.6 * np.pi * np.pi
            )
            * eta
            * eta
            * eta
            * eta
            + (
                -85710407655931086054085.1 / 3428930819264670720.0
                - 614779314.2 / 152806.5 * EULER_GAMMA
                - 46051.9 / 153.6 * np.pi * np.pi
                - 4311179766.8 / 152806.5 * log2
                + 127939.5 / 9.8 * log3
            )
            * eta
            * eta
            * eta
            + (
                -1873639936380505730110521.7 / 36575262072156487680.0
                - (9923919211.9 / 458419.5) * EULER_GAMMA
                + 41579551.7 / 90316.8 * np.pi * np.pi
                - 11734037971.3 / 458419.5 * log2
                - 5833093.5 / 548.8 * log3
            )
            * eta
            * eta
            + (
                56993518125966874478111.3 / 1083711468804636672.0
                + 6378740752.7 / 916839.0 * EULER_GAMMA
                - 545142954.7 / 812851.2 * np.pi * np.pi
                + 15994339707.7 / 1833678.0 * log2
                + (892417.5 / 313.6) * log3
            )
            * eta
            + (57822311.5 / 304819.2) * np.pi * np.pi
            + (647058264.7 / 2750517.0) * EULER_GAMMA
            - 143300652329540712655.9 / 12630669799587840.0
            - 551245.5 / 2195.2 * log3
            + 5399283943.1 / 5501034.0 * log2
        )
    )

    coef_log10pn = (
        3.0
        / (1.0 - 3.0 * eta)
        * (
            1286378036.2 / 458419.5 * eta * eta * eta
            + 1384949312.9 / 1375258.5 * eta * eta
            - 2427943164.1 / 2750517.0 * eta
            + 647058264.7 / 8251551.0
        )
    )

    coef_11pn = np.pi * (
        c_21_3PN * (-16.0 / 2.7 * eta * eta + 4.0 / 2.7 * eta)
        + 32.0 / 9.0 * eta * c_22_4PN
        + 65762707344.5 / 14417109907.2 * eta * eta * eta * eta
        - 108059782847.5 / 2621292710.4 * eta * eta * eta
        + 512031495514639.7 / 62911025049.6 * eta * eta
        + (-1064790.5 / 3628.8 * eta * eta + 4501578.5 / 14515.2 * eta - 9439.0 / 56.7)
        * np.pi
        * np.pi
        + (
            -134666.2 / 56.7 * EULER_GAMMA
            - 43038370739839704.7 / 3460106377728.0
            + 2632.5 / 4.9 * log3
            - 2100962.6 / 396.9 * log2
        )
        * eta
        - 355801.1 / 793.8 * EULER_GAMMA
        + 185754140723659441.1 / 27680851021824.0
        - 2632.5 / 19.6 * log3
        - 86254.9 / 113.4 * log2
    )

    coef_log11pn = -3 * np.pi * (134666.2 / 170.1 * eta + 355801.1 / 2381.4)

    return phi_35pn + (3.0 / 128.0 / eta / v5) * (
        (coef_8pn + coef_log8pn * logv + coef_loglog8pn * logv * logv) * v8
        + (coef_9pn + coef_log9pn * logv) * v9
        + (coef_10pn + coef_log10pn * logv) * v10
        + (coef_11pn + coef_log11pn * logv) * v11
    )


@njit(cache=True)
def Af3hPN(
    f: np.ndarray,
    M: float,
    eta: float,
    s1x: float = 0.0,
    s1y: float = 0.0,
    s1z: float = 0.0,
    s2x: float = 0.0,
    s2y: float = 0.0,
    s2z: float = 0.0,
    Lam: float = 0.0,
    dLam: float = 0.0,
    Deff: float = 1.0,
) -> np.ndarray:
    """Compute post-Newtonian amplitude @ 3.5PN for compact binary coalescences
    --------
    f = frequency series [Hz]
    M = binary mass [solar masses]
    s1x = primary spin component along x axis [dimensionless]
    s1y = primary spin component along y axis [dimensionless]
    s1z = primary spin component along z axis [dimensionless]
    s2x = secondary spin component along x axis [dimensionless]
    s2y = secondary spin component along y axis [dimensionless]
    s2z = secondary spin component along z axis [dimensionless]
    Lam = reduced tidal deformability parameter [dimensionless] (not used)
    dLam = asymmetric reduced tidal deformation parameter [dimensionless] (not used)
    Deff = luminosity distance
    --------
    Adapted from
    https://bitbucket.org/dailiang8/gwbinning/src/master/
    """

    Mchirp = M * np.power(np.abs(eta), 3.0 / 5.0)
    delta = np.sqrt(1.0 - 4.0 * eta)
    v = np.power(np.abs(np.pi * M * f * SUN_MASS_SECONDS), 1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3
    v7 = v3 * v4
    eta2 = eta ** 2
    eta3 = eta ** 3

    # 0PN order
    A0 = (
        np.power(np.abs(Mchirp), 5.0 / 6.0)
        / np.power(np.abs(f), 7.0 / 6.0)
        / Deff
        / np.abs(np.pi) ** (2.0 / 3.0)
        * np.sqrt(5.0 / 24.0)
    )

    # Modulus correction due to aligned spins
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
        + v3 * (be / 2.0 - 2.0 * np.pi)
        + v4
        * (
            1379.0 / 1152.0 * eta2
            + 18913.0 / 16128.0 * eta
            + 7266251.0 / 8128512.0
            - sigma / 2.0
        )
        + v5 * (57.0 / 16.0 * np.pi * eta - 4757.0 * np.pi / 1344.0 + eps)
        + v6
        * (
            856.0 / 105.0 * EULER_GAMMA
            + 67999.0 / 82944.0 * eta3
            - 1041557.0 / 258048.0 * eta2
            - 451.0 / 96.0 * np.pi ** 2 * eta
            + 10.0 * np.pi ** 2 / 3.0
            + 3526813753.0 / 27869184.0 * eta
            - 29342493702821.0 / 500716339200.0
            + 856.0 / 105.0 * np.log(4.0 * v)
        )
        + v7
        * (-1349.0 / 24192.0 * eta2 - 72221.0 / 24192.0 * eta - 5111593.0 / 2709504.0)
        * np.pi
    )


@njit(cache=True)
def Phif3hPN(
    f: np.ndarray,
    M: float,
    eta: float,
    s1x: float = 0.0,
    s1y: float = 0.0,
    s1z: float = 0.0,
    s2x: float = 0.0,
    s2y: float = 0.0,
    s2z: float = 0.0,
    Lam: float = 0.0,
    dLam: float = 0.0,
) -> np.ndarray:
    """Compute post-Newtonian phase @ 3.5PN for compact binary coalescences
    including spins contributions and tidal effects @ 6PN (if Lam or dLam != 0)
    --------
    f = frequency series [Hz]
    M = binary mass [solar masses]
    s1x = primary spin component along x axis [dimensionless]
    s1y = primary spin component along y axis [dimensionless]
    s1z = primary spin component along z axis [dimensionless]
    s2x = secondary spin component along x axis [dimensionless]
    s2y = secondary spin component along y axis [dimensionless]
    s2z = secondary spin component along z axis [dimensionless]
    Lam = reduced tidal deformability parameter [dimensionless]
    dLam = asymmetric reduced tidal deformation parameter [dimensionless]
    --------
    Adapted from
    https://bitbucket.org/dailiang8/gwbinning/src/master/
    """
    vlso = 1.0 / np.sqrt(6.0)
    delta = np.sqrt(1.0 - 4.0 * eta)
    v = np.abs(np.pi * M * f * SUN_MASS_SECONDS) ** (1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3
    v7 = v3 * v4
    v10 = v5 * v5
    v12 = v10 * v2
    eta2 = eta ** 2
    eta3 = eta ** 3

    m1M = 0.5 * (1.0 + delta)
    m2M = 0.5 * (1.0 - delta)
    chi1L = s1z
    chi2L = s2z
    chi1sq = s1x * s1x + s1y * s1y + s1z * s1z
    chi2sq = s2x * s2x + s2y * s2y + s2z * s2z
    chi1dotchi2 = s1x * s2x + s1y * s2y + s1z * s2z
    SL = m1M * m1M * chi1L + m2M * m2M * chi2L
    dSigmaL = delta * (m2M * chi2L - m1M * chi1L)

    # Phase correction due to spins
    sigma = eta * (721.0 / 48.0 * chi1L * chi2L - 247.0 / 48.0 * chi1dotchi2)
    sigma += 719.0 / 96.0 * (m1M * m1M * chi1L * chi1L + m2M * m2M * chi2L * chi2L)
    sigma -= 233.0 / 96.0 * (m1M * m1M * chi1sq + m2M * m2M * chi2sq)
    phis_15PN = 188.0 * SL / 3.0 + 25.0 * dSigmaL
    ga = (554345.0 / 1134.0 + 110.0 * eta / 9.0) * SL + (
        13915.0 / 84.0 - 10.0 * eta / 3.0
    ) * dSigmaL
    pn_ss3 = (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta * chi1L * chi2L
    pn_ss3 += (
        (
            (4703.5 / 8.4 + 2935.0 / 6.0 * m1M - 120.0 * m1M * m1M)
            + (-4108.25 / 6.72 - 108.5 / 1.2 * m1M + 125.5 / 3.6 * m1M * m1M)
        )
        * m1M
        * m1M
        * chi1sq
    )
    pn_ss3 += (
        (
            (4703.5 / 8.4 + 2935.0 / 6.0 * m2M - 120.0 * m2M * m2M)
            + (-4108.25 / 6.72 - 108.5 / 1.2 * m2M + 125.5 / 3.6 * m2M * m2M)
        )
        * m2M
        * m2M
        * chi2sq
    )
    phis_3PN = np.pi * (3760.0 * SL + 1490.0 * dSigmaL) / 3.0 + pn_ss3
    phis_35PN = (
        -8980424995.0 / 762048.0 + 6586595.0 * eta / 756.0 - 305.0 * eta2 / 36.0
    ) * SL - (
        170978035.0 / 48384.0 - 2876425.0 * eta / 672.0 - 4735.0 * eta2 / 144.0
    ) * dSigmaL

    # Point mass
    LO = 3.0 / 128.0 / eta / v5
    # pointmass = 1.0
    pointmass = (
        1
        + 20.0 / 9.0 * (743.0 / 336.0 + 11.0 / 4.0 * eta) * v2
        + (phis_15PN - 16.0 * np.pi) * v3
        + 10.0
        * (3058673.0 / 1016064.0 + 5429.0 / 1008.0 * eta + 617.0 / 144.0 * eta2 - sigma)
        * v4
        + (38645.0 / 756.0 * np.pi - 65.0 / 9.0 * eta * np.pi - ga)
        * (1.0 + 3.0 * np.log(v / vlso))
        * v5
        + (
            11583231236531.0 / 4694215680.0
            - 640.0 / 3.0 * np.pi ** 2
            - 6848.0 / 21.0 * (EULER_GAMMA + np.log(4.0 * v))
            + (-15737765635.0 / 3048192.0 + 2255.0 * np.pi ** 2 / 12.0) * eta
            + 76055.0 / 1728.0 * eta2
            - 127825.0 / 1296.0 * eta3
            + phis_3PN
        )
        * v6
        + (
            np.pi
            * (
                77096675.0 / 254016.0
                + 378515.0 / 1512.0 * eta
                - 74045.0 / 756.0 * eta ** 2
            )
            + phis_35PN
        )
        * v7
    )

    # Tidal correction to phase at 6PN
    # Eq.(1,4) [https://arxiv.org/abs/1310.8288]
    # Lam is the reduced tidal deformation parameter (\tilde\Lambda)
    # dLam is the asymmetric reduced tidal deformation parameter (\delta\tilde\Lambda)
    if Lam != 0.0 or dLam != 0.0:
        tidal = (
            Lam * v10 * (-39.0 / 2.0 - 3115.0 / 64.0 * v2) + dLam * 6595.0 / 364.0 * v12
        )
    else:
        tidal = 0.0 * v

    return LO * (pointmass + tidal)


def decreasing_function(frequencies: np.ndarray, merger_freq: float) -> np.ndarray:
    """A function which is equal to 1 when the input frequency is
    equal to the merger frequency, and which then decreases.

    Parameters
    ----------
    frequencies : np.ndarray
    merger_freq : float
    """

    return 20 * np.ones_like(frequencies)


def smoothly_connect_with_zero(
    frequencies: np.ndarray,
    pn_amp: np.ndarray,
    pivot_1: float,
    pivot_2: float,
    merger_freq: float,
    smoothing_func: Callable[[np.ndarray], np.ndarray] = smoothing_func,
):

    mask_mid = np.logical_and(pivot_1 <= frequencies, frequencies < pivot_2)
    mask_end = pivot_2 <= frequencies

    connecting_coefficient = smoothing_func(
        (frequencies[mask_mid] - pivot_1) / (pivot_2 - pivot_1)
    )

    pn_amp[mask_mid] = (
        pn_amp[mask_mid] * (1 - connecting_coefficient)
        + decreasing_function(frequencies[mask_mid], merger_freq)
        * connecting_coefficient
    )

    pn_amp[mask_end] = decreasing_function(frequencies[mask_end], merger_freq)

    return pn_amp


def amplitude_3h_post_newtonian(
    params: "WaveformParameters", frequencies: np.ndarray
) -> np.ndarray:
    par_dict = params.taylor_f2(frequencies)

    pn_amp = (
        Af3hPN(
            par_dict["f"],
            par_dict["mtot"],
            params.eta,
            par_dict["s1x"],
            par_dict["s1y"],
            par_dict["s1z"],
            par_dict["s2x"],
            par_dict["s2y"],
            par_dict["s2z"],
            Lam=params.lambdatilde,
            dLam=params.dlambda,
            Deff=par_dict["Deff"],
        )
        * params.dataset.taylor_f2_prefactor(params.eta)
    )

    # merger_freq = frequency_of_merger(params)

    return smoothly_connect_with_zero(frequencies, pn_amp, 0.01, 0.02, 0.02)


def phase_5h_post_newtonian_tidal(
    params: "WaveformParameters", frequencies: np.ndarray
) -> np.ndarray:

    par_dict = params.taylor_f2(frequencies)

    phi_5pn = Phif5hPN(
        par_dict["f"],
        par_dict["mtot"],
        params.eta,
        par_dict["s1x"],
        par_dict["s1y"],
        par_dict["s1z"],
        par_dict["s2x"],
        par_dict["s2y"],
        par_dict["s2z"],
    )

    # Tidal and QM contributions
    phi_tidal = PhifT7hPNComplete(
        par_dict["f"],
        par_dict["mtot"],
        params.eta,
        par_dict["lambda1"],
        par_dict["lambda2"],
    )
    # Quadrupole-monopole term
    # [https://arxiv.org/abs/gr-qc/9709032]
    phi_qm = PhifQM3hPN(
        par_dict["f"],
        par_dict["mtot"],
        params.eta,
        par_dict["s1x"],
        par_dict["s1y"],
        par_dict["s1z"],
        par_dict["s2x"],
        par_dict["s2y"],
        par_dict["s2z"],
        par_dict["lambda1"],
        par_dict["lambda2"],
    )

    # I use the convention h = h+ + i hx
    phase = -phi_5pn - phi_tidal - phi_qm

    return phase - phase[0]
