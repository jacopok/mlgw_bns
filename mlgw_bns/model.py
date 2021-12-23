from abc import ABC, abstractmethod
from typing import Optional

import h5py
from numba import njit  # type: ignore

from .dataset_generation import Dataset, TEOBResumSGenerator, WaveformGenerator
from .downsampling_interpolation import DownsamplingTraining, GreedyDownsamplingTraining


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
        downsampling_training: Optional[DownsamplingTraining] = None,
    ):

        self.filename = filename
        self.dataset = Dataset(initial_frequency_hz, srate_hz)
        self.waveform_generator = waveform_generator
        self.downsampling_training = (
            GreedyDownsamplingTraining(self.dataset)
            if downsampling_training is None
            else downsampling_training
        )

    @property
    def file(self) -> h5py.File:
        """File object in which to save datasets.

        Returns
        -------
        file : h5py.File
            To be used as a context manager.
        """
        return h5py.File(f"{self.filename}.h5", mode="a")

    def generate(self, training_downsampling_dataset_size: int = 64):
        self.downsampling_training.save_downsampling(
            training_downsampling_dataset_size, self.file
        )


class HyperparameterOptimization:
    pass
