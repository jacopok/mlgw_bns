from dataclasses import dataclass
from time import perf_counter
from typing import Callable, ClassVar, Optional

import numpy as np
import optuna
from sklearn.neural_network import MLPRegressor  # type: ignore

from .data_management import Residuals
from .model import Hyperparameters, Model


class HyperparameterOptimization:

    # reference generation time for a single waveform,
    # to be used in the computation of the effective time
    waveform_gen_time: float = 0.1

    def __init__(
        self,
        model: Model,
        optimization_seed: int = 42,
        hyper_validation_fraction: float = 0.1,
    ):

        assert model.is_loaded

        self.model = model
        self.rng = np.random.default_rng(seed=optimization_seed)
        self.hyper_validation_fraction = hyper_validation_fraction

    @property
    def training_data_number(self):
        return len(self.model.training_dataset)

    def objective(
        self,
        trial: optuna.Trial,
    ) -> tuple[float, float]:

        hyper = Hyperparameters.from_trial(trial, n_train_max=self.training_data_number)

        # train network on a subset of the data
        validation_data_number = int(
            self.hyper_validation_fraction * self.training_data_number
        )

        shuffled_indices = self.rng.choice(self.training_data_number)
        training_indices = shuffled_indices[-validation_data_number:]
        validation_indices = shuffled_indices[:-validation_data_number]

        start_time = perf_counter()
        nn = self.model.train_nn(hyper, training_indices)
        end_time = perf_counter()

        effective_time = (
            end_time - start_time
        ) + self.waveform_gen_time * self.training_data_number

        # validate on another subset of the data
        self.model.predict_residuals_bulk(
            self.model.training_parameters[validation_indices], nn, hyper
        )

        # return training time and accuracy

        accuracy = 0.0

        return accuracy, effective_time

    @staticmethod
    def residuals_difference(residuals_1: Residuals, residuals_2: Residuals) -> float:
        """Compare two sets of :class:`Residuals`.

        Parameters
        ----------
        residuals_1 : Residuals
            First set of residuals to be compared.
        residuals_2 : Residuals
            Second set of residuals to be compared.

        Returns
        -------
        float
            The average square-difference between the two residual sets.
        """

        amp_square_differences = (
            np.abs(residuals_1.amplitude_residuals - residuals_2.amplitude_residuals)
            ** 2
        )
        phi_square_differences = (
            np.abs(residuals_1.phase_residuals - residuals_2.phase_residuals) ** 2
        )

        return np.average(amp_square_differences)
