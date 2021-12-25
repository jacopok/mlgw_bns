"""Some utility functions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Iterator

import h5py
import numpy as np


class Data:
    @property
    @abstractmethod
    def group_name(self) -> str:
        """Name of the group this data should be saved as."""

    @property
    @abstractmethod
    def _arrays_list(self) -> list[str]:
        """List of arrays contained in this object.

        These will be accessed when saving the data
        to file.
        """

    def __iter__(self) -> Iterator[Any]:
        r"""The implementation of this method allows for
        the unpacking of a subclass of this object.

        Examples
        --------

        >>> @dataclass
        ... class Example(Data):
        ...    _arrays_list = ['item1', 'item2']
        ...    item1 : int
        ...    item2 : int
        >>>
        >>> ex = Example(1, 2)
        >>> item1, item2 = ex
        >>> print(item1)
        1

        Here we are giving these items a datatype of int for simplicity,
        while in real applications they will be ``np.ndarray``s.
        """

        for array_name in self._arrays_list:
            yield getattr(self, array_name)

    def save_to_file(
        self,
        file: h5py.File,
    ) -> None:
        """Save the data to a group in an h5 file."""
        if self.group_name not in file:
            file.create_group(self.group_name)

        for array_name in self._arrays_list:
            array_path = f"{self.group_name}/{array_name}"
            array = getattr(self, array_name)
            if array_path not in file:
                file.create_dataset(array_path, data=array)
            else:
                file[array_path][:] = array


@dataclass
class DownsamplingIndices(Data):

    _arrays_list: ClassVar[list[str]] = ["amplitude_indices", "phase_indices"]
    group_name: ClassVar[str] = "downsampling"

    amplitude_indices: list[int]
    phase_indices: list[int]


@dataclass
class Residuals(Data):
    """Dataclass which contains a set of sample frequencies
    as well as amplitude and phase residuals.


    Parameters
    ----------
    amplitude_residuals: np.ndarray
            Amplitude residuals. This array should have shape
            ``(number_of_waveforms, number_of_sample_points)``.
    phase_residuals: np.ndarray
            Phase residuals. This array should have shape
            ``(number_of_waveforms, number_of_sample_points)``.

    Class Attributes
    ----------------
    """

    _arrays_list: ClassVar[list[str]] = [
        "amplitude_residuals",
        "phase_residuals",
    ]
    group_name: ClassVar[str] = "residuals"

    amplitude_residuals: np.ndarray
    phase_residuals: np.ndarray


@dataclass
class FDWaveform(Data):
    """Dataclass which contains a set of sample frequencies
    as well as the amplitude and phase of frequency-domain waveforms.

    Parameters
    ----------
    amplitudes: np.ndarray
            Amplitude residuals.
    phases: np.ndarray
            Phase residuals.

    Class Attributes
    ----------------
    """

    _arrays_list: ClassVar[list[str]] = [
        "amplitudes",
        "phases",
    ]
    group_name: ClassVar[str] = "residuals"

    amplitudes: np.ndarray
    phases: np.ndarray


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
