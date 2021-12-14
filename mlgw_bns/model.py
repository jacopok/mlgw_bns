from abc import ABC, abstractmethod

from numba import njit  # type: ignore

from .dataset_generation import TEOBResumSGenerator, WaveformGenerator


class Model(ABC):
    """Generic ``mlgw_bns`` model.

    Functionality:

    * neural network training
    * fast surrogate waveform generation

    """

    @property
    @abstractmethod
    def waveform_generator(self) -> WaveformGenerator:
        pass


class TEOBResumSModel(Model):
    def __init__(self):
        self._waveform_generator = TEOBResumSGenerator()

    @property
    def waveform_generator(self):
        return self._waveform_generator


class HyperparameterOptimization:
    pass
