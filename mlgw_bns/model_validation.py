from abc import ABC
from typing import Type

from .dataset_generation import TEOBResumSGenerator, WaveformGenerator
from .model import Model, TEOBResumSModel


class ValidatingModel(Model):
    """Functionality for the validation of a model.

    Includes:
    - mismatch computation between model and reference
    - noise psd management
    - computation of reconstruction residuals
    """

    def __init__(
        self, waveform_generator: Type[WaveformGenerator] = TEOBResumSGenerator
    ):
        self._waveform_generator = TEOBResumSGenerator()

    @property
    def waveform_generator(self):
        return self._waveform_generator

    def compute_mismatch(self):
        pass
