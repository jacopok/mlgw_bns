"""Management of data which can be saved 
in an h5 file.

The idea of these data structures is to """
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Iterator, Optional, Type, TypeVar

import h5py
import numpy as np

# make type hinting available for
TYPE_DATA = TypeVar("TYPE_DATA", bound="SavableData")

# mypy does not like abstract dataclasses, see https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class SavableData:
    """Generic container for data which might need to be saved to file.

    Subclasses should also be decorated with ``@dataclass``.


    """

    @property
    @abstractmethod
    def group_name(self) -> str:
        """Name of the group this data should be saved as."""

    @classmethod
    def _arrays_list(cls) -> list[str]:
        """A list of the names of the arrays contained within
        a class.
        """
        return [field.name for field in fields(cls)]

    def __iter__(self) -> Iterator[Any]:
        r"""The implementation of this method allows for
        the unpacking of a subclass of this object.

        Examples
        --------

        >>> @dataclass
        ... class ExampleData(SavableData):
        ...    item1 : int
        ...    item2 : int
        >>> ex = ExampleData(1, 2)
        >>> item1, item2 = ex
        >>> print(item1)
        1

        Here we are giving these items a datatype of int for simplicity,
        while in real applications they will typically be ``np.ndarray``s.
        """

        for array_name in self._arrays_list():
            yield getattr(self, array_name)

    def save_to_file(
        self,
        file: h5py.File,
    ) -> None:
        """Save the data to a group in an h5 file."""
        if self.group_name not in file:
            file.create_group(self.group_name)

        for array_name in self._arrays_list():
            array_path = f"{self.group_name}/{array_name}"
            array = getattr(self, array_name)
            if array_path not in file:
                file.create_dataset(array_path, data=array)
            else:
                file[array_path][:] = array

    @classmethod
    def from_file(
        cls: Type[TYPE_DATA],
        file: h5py.File,
        group_name: Optional[str] = None,
    ) -> TYPE_DATA:

        if group_name is None:
            group_name = str(cls.group_name)

        def arrays_in_file():
            for array_name in cls._arrays_list():
                array_path = f"{group_name}/{array_name}"
                yield file[array_path][...]

        return cls(*arrays_in_file())


@dataclass
class DownsamplingIndices(SavableData):
    amplitude_indices: list[int]
    phase_indices: list[int]

    group_name: ClassVar[str] = "downsampling"

    @property
    def amp_length(self):
        return len(self.amplitude_indices)

    @property
    def phi_length(self):
        return len(self.phase_indices)

    @property
    def length(self):
        return self.amp_length + self.phi_length


@dataclass
class Residuals(SavableData):
    """Dataclass which contains a set of sample frequencies
    as well as amplitude and phase residuals.


    Parameters
    ----------
    amplitude_residuals: np.ndarray
            Amplitude residuals. This array should have shape
            ``(number_of_waveforms, number_of_amplitude_sample_points)``.
    phase_residuals: np.ndarray
            Phase residuals. This array should have shape
            ``(number_of_waveforms, number_of_phase_sample_points)``.

    Class Attributes
    ----------------
    """

    amplitude_residuals: np.ndarray
    phase_residuals: np.ndarray

    group_name: ClassVar[str] = "residuals"

    @property
    def combined(self) -> np.ndarray:
        """Combine the amplitude and phase residuals
        into a single array, with shape
        ``(number_of_waveforms, number_of_amplitude_sample_points+number_of_phase_sample_points)``.

        Returns
        -------
        np.ndarray
            Combined residuals.
        """

        return np.concatenate((self.amplitude_residuals, self.phase_residuals), axis=1)

    @classmethod
    def from_combined_residuals(
        cls, combined_residuals: np.ndarray, numbers_of_points: tuple[int, int]
    ) -> "Residuals":

        amp_points, phase_points = numbers_of_points

        return cls(
            combined_residuals[:, :amp_points], combined_residuals[:, -phase_points:]
        )


@dataclass
class PrincipalComponentData(SavableData):
    r"""Dataclass which contains a set of sample frequencies
    as well as amplitude and phase residuals.

    In the parameter definitions, the number of dimensions is the
    :math:`N` such that each data point belongs to :math:`\mathbb{R}^N`,
    while the number of components, typically denoted as :math:`K`, is the
    number of the principal components we choose to keep when
    reducing the dimensionality of the data.

    Parameters
    ----------
    eigenvectors: np.ndarray
            Eigenvectors from the PCA. This array should have shape
            ``(number_of_dimensions, number_of_components)``.
    eigenvalues: np.ndarray
            Eigenvalues from the PCA. This array should have shape
            ``(number_of_components, )``.
    mean: np.ndarray
            Mean subtracted from the data before decomposing
            the covariance matrix. This array should have shape
            ``(number_of_dimensions, )``.
    principal_components_scaling: np.ndarray
            Scale by which to divide the principal components,
            typically computed as the maximum of each in the training.
            Dividing the eigenvalues by this allows for the scale of the
            principal components to always be between 0 and 1.
            This array should have shape
            ``(number_of_components, )``.

    """

    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    mean: np.ndarray
    principal_components_scaling: np.ndarray

    group_name: ClassVar[str] = "pca"


@dataclass
class FDWaveforms(SavableData):
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

    amplitudes: np.ndarray
    phases: np.ndarray

    group_name: ClassVar[str] = "waveforms"


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
