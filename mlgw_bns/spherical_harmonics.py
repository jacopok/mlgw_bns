r"""Closed-form spin-weighted spherical harmonics for selected GW modes.

These functions evaluate :math:`^{-2}Y_{\ell m}(\iota, \phi_c)` for the
:math:`(\ell, m)` pairs used when assembling multi-mode waveforms in
scripts such as ``full_waveform_mismatch.py``. Each name encodes the
indices, e.g. ``Y_2_pos_2`` is :math:`(\ell, m) = (2, 2)`.
"""

from __future__ import annotations

import numpy as np

#: Normalization prefactor shared by all :math:`\ell=2` harmonics here.
_NORM_ELL2: float = 0.5 * np.sqrt(5.0 / np.pi)

#: Normalization prefactor shared by all :math:`\ell=3` harmonics here.
_NORM_ELL3: float = 0.5 * np.sqrt(21.0 / (2.0 * np.pi))

#: Normalization prefactor shared by all :math:`\ell=4` harmonics here.
_NORM_ELL4: float = 0.75 * np.sqrt(7.0 / np.pi)

__all__ = [
    "Y_2_neg_1",
    "Y_2_neg_2",
    "Y_2_pos_1",
    "Y_2_pos_2",
    "Y_3_neg_3",
    "Y_3_pos_3",
    "Y_4_neg_4",
    "Y_4_pos_4",
]


def Y_2_pos_2(inclination: float, phi_c: float) -> complex:
    r"""Spin-weighted harmonic :math:`^{-2}Y_{2,2}(\iota, \phi_c)`.

    Parameters
    ----------
    inclination : float
        Inclination :math:`\iota`.
    phi_c : float
        Coalescence phase :math:`\phi_c`.

    Returns
    -------
    complex
        Value of :math:`^{-2}Y_{2,2}`.
    """
    return (
        _NORM_ELL2
        * np.cos(inclination / 2) ** 4
        * np.exp(2.0j * phi_c)
    )


def Y_2_neg_2(inclination: float, phi_c: float) -> complex:
    r"""Spin-weighted harmonic :math:`^{-2}Y_{2,-2}(\iota, \phi_c)`.

    Parameters
    ----------
    inclination : float
        Inclination :math:`\iota`.
    phi_c : float
        Coalescence phase :math:`\phi_c`.

    Returns
    -------
    complex
        Value of :math:`^{-2}Y_{2,-2}`.
    """
    return (
        _NORM_ELL2
        * np.sin(inclination / 2) ** 4
        * np.exp(-2.0j * phi_c)
    )


def Y_2_pos_1(inclination: float, phi_c: float) -> complex:
    r"""Spin-weighted harmonic :math:`^{-2}Y_{2,1}(\iota, \phi_c)`.

    Parameters
    ----------
    inclination : float
        Inclination :math:`\iota`.
    phi_c : float
        Coalescence phase :math:`\phi_c`.

    Returns
    -------
    complex
        Value of :math:`^{-2}Y_{2,1}`.
    """
    return (
        _NORM_ELL2
        * np.cos(inclination / 2) ** 2
        * np.sin(inclination)
        * np.exp(1.0j * phi_c)
    )


def Y_2_neg_1(inclination: float, phi_c: float) -> complex:
    r"""Spin-weighted harmonic :math:`^{-2}Y_{2,-1}(\iota, \phi_c)`.

    Parameters
    ----------
    inclination : float
        Inclination :math:`\iota`.
    phi_c : float
        Coalescence phase :math:`\phi_c`.

    Returns
    -------
    complex
        Value of :math:`^{-2}Y_{2,-1}`.
    """
    return (
        _NORM_ELL2
        * np.sin(inclination / 2) ** 2
        * np.sin(inclination)
        * np.exp(-1.0j * phi_c)
    )


def Y_3_pos_3(inclination: float, phi_c: float) -> complex:
    r"""Spin-weighted harmonic :math:`^{-2}Y_{3,3}(\iota, \phi_c)`.

    Parameters
    ----------
    inclination : float
        Inclination :math:`\iota`.
    phi_c : float
        Coalescence phase :math:`\phi_c`.

    Returns
    -------
    complex
        Value of :math:`^{-2}Y_{3,3}`.
    """
    return (
        _NORM_ELL3
        * np.cos(inclination / 2) ** 4
        * np.sin(inclination)
        * (-np.exp(3.0j * phi_c))
    )


def Y_3_neg_3(inclination: float, phi_c: float) -> complex:
    r"""Spin-weighted harmonic :math:`^{-2}Y_{3,-3}(\iota, \phi_c)`.

    Parameters
    ----------
    inclination : float
        Inclination :math:`\iota`.
    phi_c : float
        Coalescence phase :math:`\phi_c`.

    Returns
    -------
    complex
        Value of :math:`^{-2}Y_{3,-3}`.
    """
    return (
        _NORM_ELL3
        * np.sin(inclination / 2) ** 4
        * np.sin(inclination)
        * np.exp(-3.0j * phi_c)
    )


def Y_4_pos_4(inclination: float, phi_c: float) -> complex:
    r"""Spin-weighted harmonic :math:`^{-2}Y_{4,4}(\iota, \phi_c)`.

    Parameters
    ----------
    inclination : float
        Inclination :math:`\iota`.
    phi_c : float
        Coalescence phase :math:`\phi_c`.

    Returns
    -------
    complex
        Value of :math:`^{-2}Y_{4,4}`.
    """
    return (
        _NORM_ELL4
        * np.cos(inclination / 2) ** 4
        * np.sin(inclination) ** 2
        * np.exp(4.0j * phi_c)
    )


def Y_4_neg_4(inclination: float, phi_c: float) -> complex:
    r"""Spin-weighted harmonic :math:`^{-2}Y_{4,-4}(\iota, \phi_c)`.

    Parameters
    ----------
    inclination : float
        Inclination :math:`\iota`.
    phi_c : float
        Coalescence phase :math:`\phi_c`.

    Returns
    -------
    complex
        Value of :math:`^{-2}Y_{4,-4}`.
    """
    return (
        _NORM_ELL4
        * np.sin(inclination / 2) ** 4
        * np.sin(inclination) ** 2
        * np.exp(-4.0j * phi_c)
    )
