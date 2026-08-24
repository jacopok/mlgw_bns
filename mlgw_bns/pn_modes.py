r"""Post-Newtonian (PN) expressions for individual gravitational-wave modes.

This module provides closed-form Post-Newtonian expressions for the
amplitude and phase of each spherical-harmonic mode :math:`(\ell, m)`
emitted by a quasi-circular compact binary. They are used as a low-cost
analytical fallback (and as a Taylor-F2 cross-check) for the
neural-network surrogate that lives in
:mod:`~mlgw_bns.modes_model`.

Two main outputs are exposed:

* :data:`_post_newtonian_amplitudes_by_mode` --- mapping ``Mode -> A(f)``
* :data:`_post_newtonian_phases_by_mode` --- mapping ``Mode -> phi(f)``

Both are produced by composing the per-mode coefficient functions
``H_lm(v, eta, delta, chi_a, chi_s)`` defined below with the generic
factories :func:`amp_lm`, :func:`phi_lm` and :func:`psi_lm`.

Conventions
-----------
The expressions follow Appendix E of `arXiv:2001.10914
<http://arxiv.org/abs/2001.10914>`_, with the spin/mass-asymmetry
conventions of `arXiv:1601.05588 <http://arxiv.org/abs/1601.05588>`_.
For the phases, we use the approximation (eq. 4.8 of arXiv:2001.10914)

.. math::
    \phi_{\ell m}(f) \approx \frac{m}{2}\, \phi_{22}\!\left(\frac{2f}{m}\right),

i.e. the higher-mode phase is obtained by frequency-rescaling the
(2, 2) phase.

Symbols used in the per-mode coefficient functions:

* :math:`\eta = m_1 m_2 / M^2` is the symmetric mass ratio,
* :math:`\delta = (m_1 - m_2) / M = (q - 1)/(q + 1)` is the mass asymmetry,
* :math:`\chi_a^z = \tfrac{1}{2}(\chi_1 - \chi_2)` is the antisymmetric
  spin component,
* :math:`\chi_s^z = \tfrac{1}{2}(\chi_1 + \chi_2)` is the symmetric
  spin component,
* :math:`v = (2\pi f / m)^{1/3}` is the per-mode PN expansion velocity.
"""

from typing import Callable, NamedTuple

import numpy as np

from .dataset_generation import WaveformParameters
from .taylorf2 import phase_5h_post_newtonian_tidal

#: Signature of the per-mode dimensionless coefficient functions
#: ``H_lm(v, eta, delta, chi_a_z, chi_s_z) -> np.ndarray``.
H_callable = Callable[[np.ndarray, float, float, float, float], np.ndarray]

#: Signature of a fully-built per-mode waveform component
#: (amplitude or phase) as a function of ``(params, frequencies)``.
Callable_Waveform = Callable[[WaveformParameters, np.ndarray], np.ndarray]


class Mode(NamedTuple):
    r"""A single :math:`(\ell, m)` mode of the spherical-harmonic decomposition.

    Used throughout the codebase as a hashable identifier for the modes
    that the surrogate supports (e.g. as keys in
    :data:`_post_newtonian_amplitudes_by_mode`).

    Attributes
    ----------
    l : int
        Polar index :math:`\ell`.
    m : int
        Azimuthal index :math:`m`. Positive for the modes physically
        stored; the negative-:math:`m` counterpart is obtained via
        :meth:`opposite`.
    """

    l: int
    m: int

    def opposite(self) -> "Mode":
        r"""Return the mode :math:`(\ell, -m)`.

        The negative-:math:`m` modes are not stored independently; they
        are reconstructed from the positive-:math:`m` ones using the
        standard parity relation
        :math:`h_{\ell,-m} = (-1)^\ell\, h_{\ell m}^*`.
        """
        return self.__class__(self.l, -self.m)


# ---------------------------------------------------------------------------
# Per-mode dimensionless PN amplitude coefficients H_lm(v).
#
# Each function returns the dimensionless complex expansion
#
#     H_lm(v) = sum_n c_n(eta, delta, chi_a, chi_s) v^n
#
# from Appendix E of arXiv:2001.10914 truncated at the order kept in the
# original implementation. The amplitudes of the mode in the frequency
# domain are then obtained by combining these coefficients with the
# leading SPA prefactor; see :func:`amp_lm`.
# ---------------------------------------------------------------------------


def H_22(
    v: np.ndarray,
    eta: float,
    delta: float,
    chi_a_z: float,
    chi_s_z: float,
) -> np.ndarray:
    r"""Dimensionless PN coefficient :math:`H_{22}(v)` of the leading (2, 2) mode.

    See Appendix E of `arXiv:2001.10914
    <http://arxiv.org/abs/2001.10914>`_. The series is kept up to
    :math:`v^6` and includes spin-orbit, spin-spin and tail
    contributions.
    """
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v6 = v3 * v3

    v2_coefficient = (
        451 * eta / 168
        - 323 / 224
    )

    v3_coefficient = (
        27 * delta * chi_a_z / 8
        - 11 * eta * chi_s_z / 6
        + 27 * chi_s_z / 8
    )

    v4_coefficient = (
        -49 * delta * chi_a_z * chi_s_z / 16
        + 105271 * eta ** 2 / 24192
        + 6 * eta * chi_a_z ** 2
        + eta * chi_s_z ** 2 / 8
        - 1975055 * eta / 338688
        - 49 * chi_a_z ** 2 / 32
        - 49 * chi_s_z ** 2 / 32
        - 27312085 / 8128512
    )

    v6_coefficient = (
        107291 * delta * eta * chi_a_z * chi_s_z / 2688
        - 875047 * delta * chi_a_z * chi_s_z / 32256
        + 31 * np.pi * delta * chi_a_z / 12
        + 34473079 * eta ** 3 / 6386688
        + 491 * eta ** 2 * chi_a_z ** 2 / 84
        - 51329 * eta ** 2 * chi_s_z ** 2 / 4032
        - 3248849057 * eta ** 2 / 178827264
        + 129367 * eta * chi_a_z ** 2 / 2304
        + 8517 * eta * chi_s_z ** 2 / 224
        - 7 * np.pi * eta * chi_s_z / 3
        - 205 * np.pi ** 2 * eta / 48
        + 545384828789 * eta / 5007163392
        - 875047 * chi_a_z ** 2 / 64512
        - 875047 * chi_s_z ** 2 / 64512
        + 31 * np.pi * chi_s_z / 12
        + 428 * 1j * np.pi / 105
        - 177520268561 / 8583708672
    )

    return (
        1
        + v2 * v2_coefficient
        + v3 * v3_coefficient
        + v4 * v4_coefficient
        + v6 * v6_coefficient
    )


def H_21(
    v: np.ndarray,
    eta: float,
    delta: float,
    chi_a_z: float,
    chi_s_z: float,
) -> np.ndarray:
    r"""Dimensionless PN coefficient :math:`H_{21}(v)`."""
    i = 0 + 1j

    v1 = v
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3

    H21_coef = i * np.sqrt(2) * 1 / 3

    v1_coef = delta

    v2_coef = (
        - 3 / 2 * delta * chi_s_z
        - 3 / 2 * chi_a_z
    )

    v3_coef = (
        117 / 56 * delta * eta
        + 335 / 672 * delta
    )

    v4_coef = (
        - 965 / 336 * delta * eta * chi_s_z
        + 3427 / 1344 * delta * chi_s_z
        - np.pi * delta
        - i / 2 * delta
        - i / 2 * delta * np.log(16)
        - 2101 / 336 * eta * chi_a_z
        + 3427 / 1344 * chi_a_z
    )

    v5_coef = (
        21365 / 8064 * delta * eta ** 2
        + 10 * delta * eta * chi_a_z ** 2
        + 39 / 8 * delta * eta * chi_s_z ** 2
        - 36529 / 12544 * delta * eta
        - 307 / 32 * delta * chi_a_z ** 2
        - 307 / 32 * delta * chi_s_z ** 2
        + 3 * np.pi * delta * chi_s_z
        - 964357 / 8128512 * delta
        + 213 / 4 * eta * chi_a_z * chi_s_z
        - 307 / 16 * chi_a_z * chi_s_z
        + 3 * np.pi * chi_a_z
    )

    v6_coef = (
        - 547 / 768 * delta * eta ** 2 * chi_s_z
        - 15 * delta * eta * chi_a_z ** 2 * chi_s_z
        - 3 / 16 * delta * eta * chi_s_z ** 3
        - 7049629 / 225792 * delta * eta * chi_s_z
        + 417 / 112 * np.pi * delta * eta
        - 1489 / 112 * i * delta * eta
        - 89 / 28 * i * delta * eta * np.log(2)
        + 729 / 64 * delta * chi_a_z ** 2 * chi_s_z
        + 243 / 64 * delta * chi_s_z ** 3
        + 143063173 / 5419008 * delta * chi_s_z
        - 2455 / 1344 * np.pi * delta
        - 335 / 1344 * i * delta
        - 335 / 336 * i * delta * np.log(2)
        + 42617 / 1792 * eta ** 2 * chi_a_z
        - 15 * eta * chi_a_z ** 3
        - 489 / 16 * eta * chi_a_z * chi_s_z ** 2
        - 22758317 / 225792 * eta * chi_a_z
        + 243 / 64 * chi_a_z ** 3
        + 729 / 64 * chi_a_z * chi_s_z ** 2
        + 143063173 / 5419008 * chi_a_z
    )

    return H21_coef * (
        v1 * v1_coef
        + v2 * v2_coef
        + v3 * v3_coef
        + v4 * v4_coef
        + v5 * v5_coef
        + v6 * v6_coef
    )


def H_31(
    v: np.ndarray,
    eta: float,
    delta: float,
    chi_a_z: float,
    chi_s_z: float,
) -> np.ndarray:
    r"""Dimensionless PN coefficient :math:`H_{31}(v)`."""
    i = 0 + 1j

    v1 = v
    v3 = v * v * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v

    H31_coef = i / (12 * np.sqrt(7))

    v1_coef = delta

    v3_coef = (
        17 * delta * eta / 24
        - 1049 * delta / 672
    )

    v4_coef = (
        10 * delta * eta * chi_s_z / 3
        + 65 * delta * chi_s_z / 24
        - np.pi * delta
        - 7 * i * delta / 5
        - i * delta * np.log(1024) / 5
        - 40 * eta * chi_a_z / 3
        + 65 * chi_a_z / 24
    )

    v5_coef = (
        - 4085 * delta * eta ** 2 / 4224
        + 10 * delta * eta * chi_a_z ** 2
        + delta * eta * chi_s_z ** 2 / 8
        - 272311 * delta * eta / 59136
        - 81 * delta * chi_a_z ** 2 / 32
        - 81 * delta * chi_s_z ** 2 / 32
        + 90411961 * delta / 89413632
        + 81 * eta * chi_a_z * chi_s_z / 4
        - 81 * chi_a_z * chi_s_z / 16
    )

    v6_coef = (
        803 * delta * eta ** 2 * chi_s_z / 72
        - 36187 * delta * eta * chi_s_z / 1344
        + 245 * np.pi * delta * eta / 48
        - 239 * i * delta * eta / 120
        - 5 * i * delta * eta * np.log(2) / 12
        + 264269 * delta * chi_s_z / 16128
        + 313 * np.pi * delta / 1344
        + 1049 * i * delta / 480
        + 1049 * i * delta * np.log(2) / 336
        + 2809 * eta ** 2 * chi_a_z / 72
        - 318205 * eta * chi_a_z / 4032
        + 264269 * chi_a_z / 16128
    )

    return H31_coef * (
        v1 * v1_coef
        + v3 * v3_coef
        + v4 * v4_coef
        + v5 * v5_coef
        + v6 * v6_coef
    )


def H_32(
    v: np.ndarray,
    eta: float,
    delta: float,
    chi_a_z: float,
    chi_s_z: float,
) -> np.ndarray:
    r"""Dimensionless PN coefficient :math:`H_{32}(v)`."""
    i = 0 + 1j

    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3

    H32_coef = 1 / 3 * np.sqrt(5 / 7)

    v2_coef = (
        1 - 3 * eta
    )

    v3_coef = (
        4 * eta * chi_s_z
    )

    v4_coef = (
        - 589 * eta ** 2 / 72
        + 12325 * eta / 2016
        - 10471 / 10080
    )

    v5_coef = (
        eta * (113 * delta * chi_a_z / 8 + 1081 * chi_s_z / 84 - 66 * i / 5)
        + (-113 * delta * chi_a_z - 113 * chi_s_z + 72 * i) / 24
        - 115 * eta ** 2 * chi_s_z
    )

    v6_coef = (
        eta * (- 1633 * delta * chi_a_z * chi_s_z / 48
               - 563 * chi_a_z ** 2 / 32
               - 2549 * chi_s_z ** 2 / 96
               + 8 * np.pi * chi_s_z
               - 8689883 / 149022720)
        + 81 * delta * chi_a_z * chi_s_z / 16
        + 837223 * eta ** 3 / 63360
        + eta ** 2 * (30 * chi_a_z ** 2
                      + 313 * chi_s_z ** 2 / 24
                      - 78584047 / 2661120)
        + 81 * chi_a_z ** 2 / 32
        + 81 * chi_s_z ** 2 / 32
        + 824173699 / 447068160
    )

    return H32_coef * (
        v2 * v2_coef
        + v3 * v3_coef
        + v4 * v4_coef
        + v5 * v5_coef
        + v6 * v6_coef
    )


def H_33(
    v: np.ndarray,
    eta: float,
    delta: float,
    chi_a_z: float,
    chi_s_z: float,
) -> np.ndarray:
    r"""Dimensionless PN coefficient :math:`H_{33}(v)`."""
    i = 0 + 1j

    v1 = v
    v3 = v * v * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v

    H33_coefficient = -3 / 4 * i * np.sqrt(5 / 7)

    v1_coefficient = delta

    v3_coefficient = delta * (27 * eta / 8 - 1945 / 672)

    v4_coefficient = (
        -2 * delta * eta * chi_s_z / 3
        + 65 * delta * chi_s_z / 24
        + np.pi * delta
        - 21 * i * delta / 5
        + 6 * i * delta * np.log(3 / 2)
        - 28 * eta * chi_a_z / 3
        + 65 * chi_a_z / 24
    )

    v5_coefficient = (
        420389 * delta * eta ** 2 / 63360
        + 10 * delta * eta * chi_a_z ** 2
        + delta * eta * chi_s_z ** 2 / 8
        - 11758073 * delta * eta / 887040
        - 81 * eta * chi_a_z ** 2 / 32
        - 81 * eta * chi_s_z ** 2 / 32
        - 1077664867 * delta / 447068160
        + 81 * eta * chi_a_z * chi_s_z / 4
        - 81 * chi_a_z * chi_s_z / 16
    )

    v6_coefficient = (
        - 67 * delta * eta ** 2 * chi_s_z / 72
        - 58745 * delta * eta * chi_s_z / 4032
        + 131 * np.pi * delta * eta / 16
        - 440957 * i * delta * eta / 9720
        + 69 * i * delta * eta * np.log(3 / 2) / 4
        + 163021 * delta * chi_s_z / 16128
        - 5675 * np.pi * delta / 1344
        + 389 * i * delta / 32
        - 1945 * i * delta * np.log(3 / 2) / 112
        - 137 * eta ** 2 * chi_a_z / 24
        - 148501 * eta * chi_a_z / 4032
        + 163021 * chi_a_z / 16128
    )

    h33 = (
        v1_coefficient * v1
        + v3_coefficient * v3
        + v4_coefficient * v4
        + v5_coefficient * v5
        + v6_coefficient * v6
    )

    return H33_coefficient * h33


def H_43(
    v: np.ndarray,
    eta: float,
    delta: float,
    chi_a_z: float,
    chi_s_z: float,
) -> np.ndarray:
    r"""Dimensionless PN coefficient :math:`H_{43}(v)`."""
    i = 0 + 1j

    v3 = v * v * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v

    H43_coef = 3 * i * np.sqrt(3 / 35) / 4

    v3_coef = (
        2 * delta * eta
        - delta
    )

    v4_coef = (
        5 * eta * chi_a_z / 2
        - 5 * delta * eta * chi_s_z / 2
    )

    v5_coef = (
        887 * delta * eta ** 2 / 132
        - 10795 * delta * eta / 1232
        + 18035 * delta / 7392
    )

    v6_coef = (
        - 469 * delta * eta ** 2 * chi_s_z / 48
        + 4399 * delta * eta * chi_s_z / 448
        + 2 * np.pi * delta * eta
        - 16301 * i * delta * eta / 810
        + 12 * i * delta * eta * np.log(3 / 2)
        - 113 * delta * chi_s_z / 24
        - np.pi * delta
        + 32 * i * delta / 5
        - 6 * i * delta * np.log(3 / 2)
        - 1642 * eta ** 2 * chi_a_z / 48
        + 41683 * eta * chi_a_z / 1344
        - 113 * chi_a_z / 24
    )

    return H43_coef * (
        v3 * v3_coef
        + v4 * v4_coef
        + v5 * v5_coef
        + v6 * v6_coef
    )


def H_44(
    v: np.ndarray,
    eta: float,
    delta: float,
    chi_a_z: float,
    chi_s_z: float,
) -> np.ndarray:
    r"""Dimensionless PN coefficient :math:`H_{44}(v)`."""
    i = 0 + 1j

    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3

    H44_coef = np.sqrt(10 / 7) * 4 / 9

    v2_coef = (
        3 * eta - 1
    )

    v4_coef = (
        1063 * eta ** 2 / 88
        - 128221 * eta / 7392
        + 158383 / 36960
    )

    v5_coef = (
        np.pi * (2 - 6 * eta)
        - eta * (1695 * eta * chi_a_z
                 + 2075 * chi_s_z - 3579 * i
                 + 2880 * i * np.log(2)) / 120
        + (565 * delta * chi_a_z
           + 1140 * eta ** 2 * chi_s_z
           + 565 * chi_s_z - 1008 * i
           + 960 * i * np.log(2)) / 120
    )

    v6_coef = (
        eta * (243 * delta * chi_a_z * chi_s_z / 16
               + 563 * chi_a_z ** 2 / 32
               + 247 * chi_s_z ** 2 / 32
               - 22580029007 / 880588800)
        - 81 * delta * chi_a_z * chi_s_z / 16
        - 7606537 * eta ** 3 / 274560
        + eta ** 2 * (-30 * chi_a_z ** 2
                      - 3 * chi_s_z ** 2 / 8
                      + 901461137 / 11531520)
        - 81 * chi_a_z ** 2 / 32
        - 81 * chi_s_z ** 2 / 32
        + 7888301437 / 29059430400
    )

    return H44_coef * (
        v2 * v2_coef
        + v4 * v4_coef
        + v5 * v5_coef
        + v6 * v6_coef
    )


# ---------------------------------------------------------------------------
# Auxiliary PN expansion helpers (currently unused at runtime but kept for
# future amplitude refinements based on the resummed "rho_lm" parametrization
# of the EOB literature; see e.g. arXiv:0811.2069).
# ---------------------------------------------------------------------------


def eulerlog(l: int, v: np.ndarray) -> np.ndarray:
    r"""``eulerlog`` factor used in resummed PN amplitudes.

    Defined as :math:`\gamma_E + \log(2\, l\, v)`, where :math:`\gamma_E`
    is the Euler--Mascheroni constant.

    Parameters
    ----------
    l : int
        Multipolar index :math:`\ell`.
    v : np.ndarray
        PN expansion velocity.

    Returns
    -------
    np.ndarray
        :math:`\mathrm{eulerlog}_l(v)`.
    """
    gamma_E = 0.5772156649015328606
    return gamma_E + np.log(2 * l * v)


def rho_21_NS(v: np.ndarray, nu: float) -> np.ndarray:
    r"""Resummed (2, 1) NS amplitude factor :math:`\rho_{21}(v, \nu)`.

    Currently unused by the surrogate; provided as a reference for
    higher-order amplitude refinements.

    Parameters
    ----------
    v : np.ndarray
        PN expansion velocity.
    nu : float
        Symmetric mass ratio :math:`\eta = m_1 m_2 / M^2`.
    """
    return (
        1
        + (-59 / 56 + 23 * nu / 84) * v ** 2
        + (-47009 / 56448 - 10993 * nu / 14112 + 617 * nu ** 2 / 4704) * v ** 4
        + (7613184941 / 2607897600 - (107 / 105) * eulerlog(1, v)) * v ** 6
        + (-1168617463883 / 911303737344 + (6313 / 5880) * eulerlog(1, v)) * v ** 8
        + ((-63735873771463 + 14061362165760 * eulerlog(1, v)) / 16569158860800) * v ** 10
    )


# ---------------------------------------------------------------------------
# Factories that combine an `H_lm` coefficient function with the leading
# SPA prefactor to produce concrete amplitude/phase callables.
# ---------------------------------------------------------------------------


def amp_lm(H_lm_callable: H_callable, mode: Mode) -> Callable_Waveform:
    r"""Build the frequency-domain amplitude callable for one mode.

    The amplitude is the leading stationary-phase-approximation
    prefactor times the dimensionless coefficient ``H_lm(v)`` returned by
    ``H_lm_callable``:

    .. math::
        A_{\ell m}(f) = \pi \sqrt{\tfrac{2\eta}{3}}\, v^{-7/2}\,
                        H_{\ell m}(v)\quad\text{with } v = (2\pi f/m)^{1/3}.

    For the (4, 4) mode the real part is taken explicitly (matching the
    convention used by the original implementation); all other modes use
    ``np.abs``. The intent of the original conditional appears to also
    cover the (2, 1) and (2, 2) modes --- see the inline TODO below ---
    but the behavior preserved here is the one actually observed at
    runtime.

    Parameters
    ----------
    H_lm_callable : H_callable
        One of :func:`H_22`, :func:`H_21`, ..., :func:`H_44`.
    mode : Mode
        Mode :math:`(\ell, m)` that ``H_lm_callable`` describes; only
        :attr:`Mode.m` is used here, to compute the PN velocity.

    Returns
    -------
    Callable_Waveform
        Function ``(params, frequencies) -> amplitude`` ready to be
        stored in :data:`_post_newtonian_amplitudes_by_mode`.
    """

    def function(params: "WaveformParameters", frequencies: np.ndarray) -> np.ndarray:
        v = np.abs(2 * np.pi * frequencies / mode.m) ** (1.0 / 3.0)
        delta = (params.mass_ratio - 1) / (params.mass_ratio + 1)
        chi_a_z = (params.chi_1 - params.chi_2) / 2
        chi_s_z = (params.chi_1 + params.chi_2) / 2

        prefactor = (
            np.pi
            * np.sqrt(2 * params.eta / 3)
            * v ** (-7 / 2)
        )
        H_val = H_lm_callable(v, params.eta, delta, chi_a_z, chi_s_z)

        # NOTE: preserves the original behavior of `H_lm_callable == H_44
        # and H_22 and H_21` (which reduces to `is H_44`, since H_22/H_21
        # are always-truthy function objects). The conditional structure
        # suggests the original author may have intended to include
        # H_22 and H_21 as well; revisit if amplitudes for those modes
        # need to be taken from `.real` rather than `np.abs`.
        if H_lm_callable is H_44:
            return prefactor * H_val.real
        return np.abs(prefactor * H_val)

    return function


def phi_lm(mode: Mode) -> Callable_Waveform:
    r"""Build the frequency-domain phase callable for one mode (rescaled (2,2) phase).

    Uses the approximation
    :math:`\phi_{\ell m}(f) \approx \tfrac{m}{2}\,\phi_{22}(2f/m)`
    (eq. 4.8 of arXiv:2001.10914), where :math:`\phi_{22}` is the
    Taylor-F2 phase up to 3.5 PN with tidal corrections, provided by
    :func:`~mlgw_bns.taylorf2.phase_5h_post_newtonian_tidal`.

    Note: the frequency rescaling :math:`f \to 2f/m` is folded into the
    final returned callable :func:`psi_lm`; this lighter version only
    applies the :math:`m/2` factor.

    Parameters
    ----------
    mode : Mode
        Mode :math:`(\ell, m)`. Only ``mode.m`` is used.

    Returns
    -------
    Callable_Waveform
        Function ``(params, frequencies) -> phase``.
    """

    def function(params: "WaveformParameters", frequencies: np.ndarray) -> np.ndarray:
        return phase_5h_post_newtonian_tidal(params, frequencies) * (mode.m / 2)

    return function


def psi_lm(H_lm_callable: H_callable, mode: Mode) -> Callable_Waveform:
    r"""Build the full per-mode phase callable, with frequency rescaling and amplitude phase.

    Combines the orbital phase
    :math:`\tfrac{m}{2}\,\phi_{22}(2f/m)` with the phase of the
    dimensionless complex amplitude coefficient :math:`H_{\ell m}(v)`:

    .. math::
        \phi_{\ell m}(f) = \tfrac{m}{2}\,\phi_{22}(2f/m)
                          + \arg H_{\ell m}(v).

    The result is unwrapped so that it is a continuous function of
    frequency.

    Parameters
    ----------
    H_lm_callable : H_callable
        Dimensionless PN amplitude coefficient function for this mode.
    mode : Mode
        Mode :math:`(\ell, m)`.

    Returns
    -------
    Callable_Waveform
        Function ``(params, frequencies) -> phase``.
    """

    def function(params: "WaveformParameters", frequencies: np.ndarray) -> np.ndarray:
        mode_freq = 2 * frequencies / mode.m
        orbital_phase = phase_5h_post_newtonian_tidal(params, mode_freq) * (mode.m / 2)

        v = np.abs(2 * np.pi * frequencies / mode.m) ** (1.0 / 3.0)
        delta = (params.mass_ratio - 1) / (params.mass_ratio + 1)
        chi_a_z = (params.chi_1 - params.chi_2) / 2
        chi_s_z = (params.chi_1 + params.chi_2) / 2
        H = H_lm_callable(v, params.eta, delta, chi_a_z, chi_s_z)
        return orbital_phase + np.unwrap(np.angle(H))

    return function


# ---------------------------------------------------------------------------
# Public registries used by `higher_order_modes` and `modes_model`.
#
# Each maps a `Mode` to a ready-to-call function
# ``(params, frequencies) -> array``. Adding support for a new mode is a
# matter of providing the corresponding `H_lm` and a row in both dicts.
# ---------------------------------------------------------------------------

#: Per-mode frequency-domain amplitude callables built via :func:`amp_lm`.
_post_newtonian_amplitudes_by_mode: dict[Mode, Callable_Waveform] = {
    Mode(2, 1): amp_lm(H_21, Mode(2, 1)),
    Mode(2, 2): amp_lm(H_22, Mode(2, 2)),
    Mode(3, 1): amp_lm(H_31, Mode(3, 1)),
    Mode(3, 2): amp_lm(H_32, Mode(3, 2)),
    Mode(3, 3): amp_lm(H_33, Mode(3, 3)),
    Mode(4, 3): amp_lm(H_43, Mode(4, 3)),
    Mode(4, 4): amp_lm(H_44, Mode(4, 4)),
}

#: Per-mode frequency-domain phase callables built via :func:`psi_lm`.
_post_newtonian_phases_by_mode: dict[Mode, Callable_Waveform] = {
    Mode(2, 1): psi_lm(H_21, Mode(2, 1)),
    Mode(2, 2): psi_lm(H_22, Mode(2, 2)),
    Mode(3, 1): psi_lm(H_31, Mode(3, 1)),
    Mode(3, 2): psi_lm(H_32, Mode(3, 2)),
    Mode(3, 3): psi_lm(H_33, Mode(3, 3)),
    Mode(4, 3): psi_lm(H_43, Mode(4, 3)),
    Mode(4, 4): psi_lm(H_44, Mode(4, 4)),
}