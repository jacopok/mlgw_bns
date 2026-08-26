r"""Named spin-weighted spherical harmonics for selected GW modes.

These are convenience aliases evaluating
:math:`{}^{-2}Y_{\ell m}(\iota, \varphi_c)` for the :math:`(\ell, m)`
pairs used when assembling multi-mode waveforms in the validation
scripts. Each name encodes the indices, e.g. :func:`Y_2_pos_2` is
:math:`(\ell, m) = (2, 2)`.

The values come from :func:`~mlgw_bns.special_func.spinsphericalharm`,
which is the package's single implementation of the harmonics; these
wrappers only fix ``s = -2`` and the indices, and repackage the
``(real, imaginary)`` pair into a complex number.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import numpy as np

from .special_func import spinsphericalharm

__all__ = [
    "Y_2_neg_1",
    "Y_2_neg_2",
    "Y_2_pos_1",
    "Y_2_pos_2",
    "Y_3_neg_3",
    "Y_3_pos_3",
    "Y_4_neg_4",
    "Y_4_pos_4",
    "spin_weighted_harmonic",
]

#: The :math:`(\ell, m)` pairs for which a named alias is provided.
_NAMED_MODES: Tuple[Tuple[int, int], ...] = (
    (2, 2),
    (2, 1),
    (2, -1),
    (2, -2),
    (3, 3),
    (3, -3),
    (4, 4),
    (4, -4),
)


def spin_weighted_harmonic(l: int, m: int, inclination: float, phi_c: float) -> complex:
    r"""Spin-weighted harmonic :math:`{}^{-2}Y_{\ell m}(\iota, \varphi_c)`.

    Parameters
    ----------
    l : int
        Multipolar index :math:`\ell`.
    m : int
        Multipolar index :math:`m`.
    inclination : float
        Inclination :math:`\iota`.
    phi_c : float
        Coalescence phase :math:`\varphi_c`.

    Returns
    -------
    complex
        Value of :math:`{}^{-2}Y_{\ell m}`.
    """
    real, imaginary = spinsphericalharm(-2, l, m, phi_c, inclination)
    return real + 1j * imaginary


def _name(l: int, m: int) -> str:
    """Alias name for a mode, e.g. ``(2, -1) -> "Y_2_neg_1"``."""
    return f"Y_{l}_{'neg' if m < 0 else 'pos'}_{abs(m)}"


def _make_alias(l: int, m: int) -> Callable[[float, float], complex]:
    """Build the ``Y_l_sign_m(inclination, phi_c)`` alias for one mode."""
    alias = partial(spin_weighted_harmonic, l, m)
    alias.__doc__ = (
        rf"Spin-weighted harmonic :math:`{{}}^{{-2}}Y_{{{l},{m}}}(\iota, \varphi_c)`. "
        "See :func:`spin_weighted_harmonic`."
    )
    return alias


for _l, _m in _NAMED_MODES:
    globals()[_name(_l, _m)] = _make_alias(_l, _m)

del _l, _m
