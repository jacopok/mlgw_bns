from abc import ABC, abstractmethod
from typing import Optional

import h5py
from numba import njit  # type: ignore

from .data_management import DownsamplingIndices, PrincipalComponentData
from .dataset_generation import Dataset, TEOBResumSGenerator, WaveformGenerator
from .downsampling_interpolation import DownsamplingTraining, GreedyDownsamplingTraining
from .principal_component_analysis import PrincipalComponentTraining


class Model:
    """Generic ``mlgw_bns`` model.
    This class implements little functionality by itself,
    acting instead as a container and wrapper around the different
    moving parts inside ``mlgw_bns``.

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
        pca_components: int = 30,
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

        self.pca_components = pca_components

    @property
    def file(self) -> h5py.File:
        """File object in which to save datasets.

        Returns
        -------
        file : h5py.File
            To be used as a context manager.
        """
        return h5py.File(f"{self.filename}.h5", mode="a")

    def generate(
        self,
        training_downsampling_dataset_size: int = 64,
        training_pca_dataset_size: int = 1024,
    ):
        """Generate a new model from scratch.


        Parameters
        ----------
        training_downsampling_dataset_size : int, optional
            By default 64.
        training_pca_dataset_size : int, optional
            By default 1024.
        """

        self.downsampling_indices: DownsamplingIndices = (
            self.downsampling_training.train(training_downsampling_dataset_size)
        )

        self.pca_training = PrincipalComponentTraining(
            self.dataset, self.downsampling_indices, self.pca_components
        )

        self.pca_data: PrincipalComponentData = self.pca_training.train(
            training_pca_dataset_size
        )

    def save(self):
        for arr in [self.downsampling_indices, self.pca_data]:
            arr.save_to_file(self.file)


class HyperparameterOptimization:
    pass
