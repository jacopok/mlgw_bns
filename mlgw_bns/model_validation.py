from abc import ABC
from typing import Type

from .dataset_generation import WaveformGenerator
from .model import Model


class ValidatingModel(Model):
    """Functionality for the validation of a model.

    Includes:
    - mismatch computation between model and reference
    - noise psd management
    - computation of reconstruction residuals
    """

    def compute_mismatch(self):
        pass
