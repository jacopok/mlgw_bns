from abc import ABC
from typing import Type

from .dataset_generation import WaveformGenerator
from .model import Model


class ValidateModel:
    """Functionality for the validation of a model.

    Includes:
    - mismatch computation between model and reference
    - noise psd management
    - computation of reconstruction residuals
    """

    def __init__(self, model: Model):
        self.model = model

    def compute_mismatch(self):
        pass
