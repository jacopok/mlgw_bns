r"""Shared special functions and small numerical helpers.

The Wigner :math:`d` and spin-weighted spherical harmonic
implementations here are the single source of truth for the whole
package: :mod:`~mlgw_bns.model` uses them to project the co-precessing
modes onto the observer's sky, :mod:`~mlgw_bns.higher_order_modes` uses
them for the :math:`s=-2` specialisation, and
:mod:`~mlgw_bns.twist_waveform` uses them to build the Wigner-:math:`D`
matrices of the precessing twist. They mirror ``wigner_d_function()``
and ``spinsphericalharm()`` in TEOBResumS' ``C/src/SpecialFuns.c``.

Both accept either a scalar angle or a numpy array of angles.
"""

import math
from typing import Tuple, Union

import numpy as np

__all__ = [
    "factorial",
    "wigner_d_function",
    "spinsphericalharm",
    "unwrap_euler",
    "dynamic_then_uniform_grid",
    "reduced_tidal_parameter",
    "effective_spin_parameter",
]

#: Either a single angle or an array of them.
AngleLike = Union[float, np.ndarray]


def factorial(n: int) -> int:
    """Factorial :math:`n!`, mirroring ``fact()`` in TEOBResumS.

    Parameters
    ----------
    n : int
        Non-negative integer.

    Returns
    -------
    int
        :math:`n!`.

    Raises
    ------
    ValueError
        If ``n`` is negative.
    """
    if n < 0:
        raise ValueError("computing a negative factorial")
    return math.factorial(n)


def wigner_d_function(l: int, m: int, s: int, i: AngleLike) -> AngleLike:
    r"""Wigner :math:`d`-function :math:`d^{\ell}_{m,s}(\iota)`.

    As given in Eq. (II.8) of `arXiv:0709.0093
    <https://arxiv.org/pdf/0709.0093.pdf>`_.

    Parameters
    ----------
    l : int
        Upper index :math:`\ell`.
    m : int
        First lower index.
    s : int
        Second lower index.
    i : float or np.ndarray
        Argument of the Wigner :math:`d`-function, in radians. May be an
        array, in which case an array of the same shape is returned.

    Returns
    -------
    float or np.ndarray
        The value of :math:`d^{\ell}_{m,s}(\iota)`.
    """

    angle = np.asarray(i, dtype=float)
    costheta = np.cos(angle * 0.5)
    sintheta = np.sin(angle * 0.5)

    norm = math.sqrt(
        factorial(l + m) * factorial(l - m) * factorial(l + s) * factorial(l - s)
    )

    # Bounds of the summation over k.
    ki = max(0, m - s)
    kf = min(l + m, l - s)

    d_wigner = np.zeros_like(costheta)
    for k in range(ki, kf + 1):
        div = 1.0 / (
            factorial(k) * factorial(l + m - k) * factorial(l - s - k) * factorial(s - m + k)
        )
        d_wigner = d_wigner + div * (
            (-1.0) ** k
            * costheta ** (2 * l + m - s - 2 * k)
            * sintheta ** (2 * k + s - m)
        )

    result = norm * d_wigner
    return result if result.ndim else float(result)


def spinsphericalharm(
    s: int, l: int, m: int, phi: AngleLike, i: AngleLike
) -> Tuple[AngleLike, AngleLike]:
    r"""Spin-weighted spherical harmonic :math:`{}_{s}Y_{\ell m}(\varphi, \iota)`.

    As given in Eq. (II.7) of `arXiv:0709.0093
    <https://arxiv.org/pdf/0709.0093.pdf>`_.

    Parameters
    ----------
    s : int
        Spin weight.
    l : int
        Multipolar index :math:`\ell`.
    m : int
        Multipolar index :math:`m`.
    phi : float or np.ndarray
        Azimuthal angle, in radians.
    i : float or np.ndarray
        Polar angle, in radians.

    Returns
    -------
    tuple
        ``(rY, iY)``: the real and imaginary parts of
        :math:`{}_{s}Y_{\ell m}`.

    Raises
    ------
    ValueError
        If ``(l, m)`` are not a valid pair of indices.
    """

    if l < 0 or m < -l or m > l:
        raise ValueError("Invalid (l,m) values in spinsphericalharm")

    c = (-1.0) ** (-s) * math.sqrt((2.0 * l + 1.0) / (4.0 * math.pi))
    d_wigner = c * wigner_d_function(l, m, -s, i)

    return np.cos(m * np.asarray(phi)) * d_wigner, np.sin(m * np.asarray(phi)) * d_wigner


def unwrap_euler(p):
    size = len(p)
    if size < 1:
        return p  # Return the original array if size is less than 1

    dphi = 0.0
    corr = 0.0

    prev = p[0]
    delta = p[1] - p[0]

    for j in range(1, size):
        # Setting current data point
        p[j] += corr
        curr = p[j]

        # Check if decreasing too much - adding 2Pi
        if (curr < prev - np.pi) and (curr - prev < delta - np.pi):
            dphi = 2 * np.pi

        # Check if increasing too much - removing 2Pi
        if (curr > prev + np.pi) and (curr - prev > delta + np.pi):
            dphi = -2 * np.pi

        # Adding corrections
        corr += dphi
        p[j] += dphi

        # Resetting for next iteration
        prev = p[j]
        delta = p[j] - p[j - 1]
        dphi = 0.0

    return p  # Return the unwrapped array

def dynamic_then_uniform_grid(
        f_min: float,
        f_switch: float,
        f_max: float,
        alpha: float = 1e-6,
        beta: float = 1e-4,
        uniform_step: float = 1e-4
    ) -> np.ndarray:
    
    # Build non-uniform grid up to f_switch
    freqs = [f_min]
    while freqs[-1] < f_switch:
        step = alpha * freqs[-1] + beta
        next_freq = freqs[-1] + step
        if next_freq > f_switch:
            break
        freqs.append(next_freq)
    non_uniform_part = np.array(freqs)
    
    # Build uniform grid from f_switch to f_max
    # Include f_switch if it is not already in non_uniform_part
    if not np.isclose(non_uniform_part[-1], f_switch):
        uniform_start = f_switch
    else:
        uniform_start = non_uniform_part[-1]
    
    uniform_part = np.arange(uniform_start, f_max + uniform_step, uniform_step)
    
    # Concatenate both parts, avoid duplicates at the boundary
    if np.isclose(non_uniform_part[-1], uniform_part[0]):
        combined = np.concatenate([non_uniform_part, uniform_part[1:]])
    else:
        combined = np.concatenate([non_uniform_part, uniform_part])
    
    return combined

def reduced_tidal_parameter(lambda1, lambda2, mass_ratio):
    """
    Compute the reduced tidal parameter for a given lambda1, lambda2, and total mass.
    """
    total_mass = 2.8
    m1 = total_mass / (1 + mass_ratio)
    m2 = total_mass - m1
    lam = 16 / (13 * total_mass ** 5) * ((m1 + 12 * m2) * m1 ** 4 * lambda1 + (m2 + 12 * m1) * m2 ** 4 * lambda2)
    return lam

def effective_spin_parameter(chi1, chi2, mass_ratio):
    """
    Effective spin parameter (chi_tilde).
    mass_ratio = q = m2 / m1  (with m1 >= m2)
    """
    return (chi1 + mass_ratio * chi2) / (1 + mass_ratio)