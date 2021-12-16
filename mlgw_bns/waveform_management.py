"""Some utility functions."""
from __future__ import annotations

import numpy as np


def phase_unwrapping(
    waveform_cartesian: np.ndarray, eps: float = 1e-2, set_zero_at_start: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Starting from an array of cartesian-form complex numbers,
    returns two real arrays: amplitude and phase.
    """

    phidot = (
        (np.diff(np.angle(waveform_cartesian), axis=-1, prepend=0) + eps) % (2 * np.pi)
    ) - eps

    phase = np.cumsum(phidot, axis=-1)
    if set_zero_at_start:
        phase -= phase[0]

    return (np.abs(waveform_cartesian), phase)
