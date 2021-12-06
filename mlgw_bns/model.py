from abc import ABC, abstractmethod
from .dataset_generation import SlowWaveformGenerator, TEOBResumSGenerator


class Model(ABC):
    @property
    @abstractmethod
    def waveform_generator(self) -> SlowWaveformGenerator:
        pass


class TEOBResumSModel(Model):
    def __init__(self):
        self._waveform_generator = TEOBResumSGenerator()

    @property
    def waveform_generator(self):
        return self._waveform_generator


class HyperparameterOptimization:
    pass
