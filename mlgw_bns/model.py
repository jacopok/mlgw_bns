from abc import ABC, abstractmethod

import h5py
from numba import njit  # type: ignore

from .dataset_generation import Dataset, TEOBResumSGenerator, WaveformGenerator
from .downsampling_interpolation import DownsamplingTraining


class Model:
    """Generic ``mlgw_bns`` model.

    Functionality:

    * neural network training
    * fast surrogate waveform generation
    * saving the model to an h5 file

    """

    def __init__(
        self,
        filename: str,
        initial_frequency_hz: float = 20.0,
        srate_hz: float = 4096.0,
        waveform_generator: WaveformGenerator = TEOBResumSGenerator(),
    ):

        self.filename = filename
        self.dataset = Dataset(initial_frequency_hz, srate_hz)
        self.waveform_generator = waveform_generator

    @property
    def file(self) -> h5py.File:
        """File object in which to save datasets.

        Returns
        -------
        h5py.File
            To be used as a context manager.

        Examples
        --------
        >>> model = Model('test')
        >>> with model.file as file:
        ...     if 'new_group' not in file:
        ...         file.create_group('new_group')
        ...     print(file.keys())
        <KeysViewHDF5 ['new_group']>
        """
        return h5py.File(self.filename, mode="a")

    def generate(self):
        training_downsampling = DownsamplingTraining(self.dataset)
        training_downsampling.save_downsampling(128, self.file)


class HyperparameterOptimization:
    pass
