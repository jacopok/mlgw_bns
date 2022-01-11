import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, ClassVar, Optional

import joblib  # type: ignore
import numpy as np
import optuna
from optuna.visualization import (
    plot_parallel_coordinate,
    plot_param_importances,
    plot_pareto_front,
)
from sklearn.neural_network import MLPRegressor  # type: ignore

from .data_management import Residuals
from .model import Hyperparameters, Model


class HyperparameterOptimization:
    """Manager for the optimization of the hyperparameters
    corresponding to a certain :class:`Model`.

    Parameters
    ----------
    model: Model
            Reference model for the optimization.
    optimization_seed: int, optional
            Seed for the random number to be used in the optimization.
            Defaults to 42.
    hyper_validation_fraction: float
            Fraction of the data to be used in validation
            during the optimization.
    study: optuna.Study, optional
            Pre-made study to use.
            Defaults to None; if not provided,
            the initializer looks for a file with the correct name
            in the local directory and uses it,
            and it creates a new study if it cannot find it.
    """

    # reference generation time for a single waveform,
    # to be used in the computation of the effective time
    waveform_gen_time: float = 0.1

    save_every_n_minutes: float = 10.0

    def __init__(
        self,
        model: Model,
        optimization_seed: int = 42,
        hyper_validation_fraction: float = 0.1,
        study: Optional[optuna.Study] = None,
    ):

        assert model.is_loaded

        self.model = model
        self.rng = np.random.default_rng(seed=optimization_seed)
        self.hyper_validation_fraction = hyper_validation_fraction

        if study is None:
            try:
                self.study = joblib.load(self.study_filename)
                logging.info("Loading study from %s", self.study_filename)
            except FileNotFoundError:
                self.study = optuna.create_study(
                    directions=["minimize", "minimize"], study_name=self.model.filename
                )
                logging.info("Creating new study")
        else:
            self.study = study

    @property
    def training_data_number(self) -> int:
        """Number of available training waveforms."""
        return len(self.model.training_dataset)

    @property
    def study_filename(self) -> str:
        """Name of the file to save the study to."""
        return f"study_{self.model.filename}.pkl"

    def objective(
        self,
        trial: optuna.Trial,
    ) -> tuple[float, float]:
        """Objective function to be used when optimizing the hyperparameters
        for the neural network and PCA.

        Parameters
        ----------
        trial : optuna.Trial
                This object is required to generate the parameters
                according to the rules of the :module:``optuna`` optimizer used.

        Returns
        -------
        tuple[float, float]
                Base-10 logarithm of the accuracy and training time, respectively.

                The accuracy is defined as the average of the square differences between
                the true and estimated residuals.

                The training time includes both the training of the network and,
                roughly, the generation of the waveforms used for training.
        """

        hyper = Hyperparameters.from_trial(trial, n_train_max=self.training_data_number)

        # train network on a subset of the data
        validation_data_number = int(
            self.hyper_validation_fraction * self.training_data_number
        )

        shuffled_indices = self.rng.choice(
            self.training_data_number, self.training_data_number, replace=False
        )
        training_indices = shuffled_indices[: hyper.n_train]
        validation_indices = shuffled_indices[-validation_data_number:]

        start_time = perf_counter()
        nn = self.model.train_nn(hyper, training_indices)
        end_time = perf_counter()

        effective_time = (
            end_time - start_time
        ) + self.waveform_gen_time * hyper.n_train

        # validate on another subset of the data
        predicted_residuals = self.model.predict_residuals_bulk(
            self.model.training_parameters[validation_indices], nn, hyper
        )

        accuracy = self.residuals_difference(
            self.model.training_dataset[validation_indices], predicted_residuals
        )

        return np.log10(accuracy), np.log10(effective_time)

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

        return (
            np.average(amp_square_differences) + np.average(phi_square_differences)
        ) / 2.0

    def optimize(self, timeout_min: float = 10.0, save_to_file=True):
        """Run the optimization ---
        this is a wrapper around :meth:`optuna.Study.optimize`).
        This command can take an arbitrary amount of time, therefore
        its timeout is provided as a parameter.
        Typically, good results can be achieved within a few hours.

        Parameters
        ----------
        timeout_min : float, optional
            Number of minutes to run for, by default 10.0.
        save_to_file : bool, optional
            Whether to save the study every so often ---
            the interval between which to save is determined by the class attribute
            :attr:`save_every_n_minutes`.
            By default True.
        """

        obj = lambda trial: self.objective(trial)

        iterations: int = max(int(timeout_min / self.save_every_n_minutes), 1)

        timeout_per_iter = 60 * timeout_min / iterations

        for _ in range(iterations):

            self.study.optimize(obj, timeout=timeout_per_iter)

            if save_to_file:
                joblib.dump(self.study, self.study_filename)
                logging.info("Saved to file.")

    def plot_pareto(self) -> None:
        """Plot the Pareto front of the bivariate optimization."""
        fig = plot_pareto_front(
            self.study,
            target_names=[
                "Error [log10(average square error)] ",
                "time [log10(time in seconds)]",
            ],
        )
        fig.show()

    def plot_parallel(self, **kwargs):
        to_plot = lambda trial: trial.values[0]
        fig = plot_parallel_coordinate(self.study, target_name=to_plot, **kwargs)
        fig.show()

    def plot_param_importance(self):
        fig = plot_param_importances(
            self.study, target=lambda t: t.values[0], target_name="Error"
        )
        fig.show()
