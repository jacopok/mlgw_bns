from __future__ import annotations

import datetime
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

from .data_management import Residuals
from .model import Hyperparameters, Model
from .neural_network import best_trial_under_n


class HyperparameterOptimization:
    """Manager for the optimization of the hyperparameters
    corresponding to a certain :class:`Model`.

    The optimization performed is over two variables:
    the reconstruction accuracy and the training time.

    **Reconstruction accuracy** is quantified by looking at the average
    square error in the reconstruction of the residuals;
    spefifically, the average is taken over both amplitude and phase residuals,
    and the value returned by the :meth:`objective` function is the base-10
    logarithm of this.

    **Training time** accounts for both the time required to train the
    neural network and the estimated time required to generate the waveforms
    needed for the training.
    This can vary, since one of the hyperparameters varied in the training
    is the number of waveforms in the training dataset.

    Including it is convenient since having more waveforms ---
    a finer sampling of the waveform space ---
    means the optimal network might be different.

    However, only ever training networks with as large a number of waveforms
    as we might wish to use in the end gets expensive;
    therefore, we vary the number of training waveforms in the optimization,
    so that the optuna study is able to learn the basic region of parameter space
    which it is best to explore, and then extend that knowledge to the
    new region of the parameter space with more training waveforms.

    The inclusion of this cost term is needed since, typically,
    using more waveforms will yield a better fit.
    So, we do multi-parameter optimization: see, for example,
    `Multiobjective tree-structured parzen estimator
    for computationally expensive optimization problems <https://doi.org/10.1145/3377930.3389817>`_
    by Ozaki et al.

    To visualize the Pareto front of the optimization, one can use the
    :meth:`plot_pareto` method after an optimization run.

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

    Class Attributes
    waveform_gen_time: float
            Reference generation time for a single waveform,
            to be used in the computation of the effective time
            in the :meth:`objective`.
            Defaults to 0.1.
    save_every_n_minutes: float
            When running the optimization through :meth:`optimize`,
            every how many minutes to save the study.
            Defaults to 30.
    """

    waveform_gen_time: float = 0.1

    save_every_n_minutes: float = 30.0

    def __init__(
        self,
        model: Model,
        optimization_seed: int = 42,
        hyper_validation_fraction: float = 0.1,
        study: Optional[optuna.Study] = None,
    ):

        assert model.auxiliary_data_available
        assert model.training_dataset_available

        self.model = model
        self.rng = np.random.default_rng(seed=optimization_seed)
        self.hyper_validation_fraction = hyper_validation_fraction

        if study is None:
            try:
                self.study: optuna.Study = joblib.load(self.study_filename)
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
        assert self.model.training_dataset is not None
        return len(self.model.training_dataset)

    @property
    def study_filename(self) -> str:
        """Name of the file to save the study to."""
        return f"{self.model.filename}_study.pkl"

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
        assert self.model.training_dataset is not None
        assert self.model.training_parameters is not None

        # train network on a subset of the data
        validation_data_number = int(
            self.hyper_validation_fraction * self.training_data_number
        )

        hyper = Hyperparameters.from_trial(
            trial, n_train_max=self.training_data_number - validation_data_number
        )

        assert hyper.n_train + validation_data_number <= self.training_data_number

        shuffled_indices = self.rng.choice(
            self.training_data_number, self.training_data_number, replace=False
        )
        training_indices = shuffled_indices[: hyper.n_train]
        validation_indices = shuffled_indices[-validation_data_number:]

        start_time = perf_counter()
        nn = self.model.train_nn(hyper, list(training_indices))
        end_time = perf_counter()

        effective_time = (
            end_time - start_time
        ) + self.waveform_gen_time * hyper.n_train

        # validate on another subset of the data
        predicted_residuals = self.model.predict_residuals_bulk(
            self.model.training_parameters[validation_indices], nn
        )

        accuracy = self.residuals_difference(
            self.model.training_dataset[list(validation_indices)], predicted_residuals
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

    def optimize(self, timeout_min: float = 5.0) -> None:
        """Run the optimization ---
        this is a wrapper around :meth:`optuna.Study.optimize` ---
        for a certain amount of minutes.

        Parameters
        ----------
        timeout_min : float, optional
            Number of minutes to run for, by default 5
        """

        obj = lambda trial: self.objective(trial)
        self.study.optimize(obj, timeout=timeout_min * 60)

    def optimize_and_save(self, timeout_hr: float = 1.0) -> None:
        """Run the optimization ---
        this is a wrapper around :meth:`optuna.Study.optimize`.
        This command can take an arbitrary amount of time, therefore
        its timeout is provided as a parameter.
        Typically, good results can be achieved within a few hours.

        The interval between which to save is determined by the class attribute
        :attr:`save_every_n_minutes`.

        Parameters
        ----------
        timeout_hr : float, optional
            Number of hours to run for, by default 1.
        """

        iterations: int = max(int(timeout_hr * 60 / self.save_every_n_minutes), 1)

        expected_datetime_end = datetime.datetime.now() + datetime.timedelta(
            hours=timeout_hr
        )
        logging.info(
            "Starting to train at %s, will end at %s",
            (datetime.datetime.now(), expected_datetime_end.isoformat()),
        )

        for n in range(iterations):

            remaining_minutes: float = (
                expected_datetime_end - datetime.datetime.now()
            ) / datetime.timedelta(minutes=1)

            if remaining_minutes <= 0:
                return

            iterations_left: int = iterations - n

            timeout_min: float = remaining_minutes / iterations_left

            self.optimize(timeout_min=timeout_min)

            joblib.dump(self.study, self.study_filename)
            logging.info("Saved to file.")

    def plot_pareto(self) -> None:
        """Plot the Pareto front of the bivariate optimization,
        making use of the function :func:`optuna.visualization.plot_pareto_front`."""

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

    def best_hyperparameters(
        self, training_number: Optional[int] = None
    ) -> Hyperparameters:
        """Return the best hyperparameters found using less than
        a certain number of training waveforms.

        Parameters
        ----------
        training_number : int, optional
            Number of training waveforms; by default None,
            in which case return the hyperparameters
            for as many waveforms as the current model has available.

        Returns
        -------
        Hyperparameters
        """

        best_trials = self.study.best_trials

        if training_number is None:
            training_number = self.training_data_number

        return best_trial_under_n(best_trials, training_number)

    def save_best_trials_to_file(self, filename: str = "best_trials") -> None:
        """Save the best trials obtained so far in the optimization to the file
        "filename".pkl.

        The best trials are obtained as ``self.study.best_trials``.

        Parameters
        ----------
        filename : str, optional
            Filename to save to, by default "best_trials"
        """
        joblib.dump(self.study.best_trials, f"{filename}.pkl")

    def total_training_time(self) -> datetime.timedelta:
        return sum(
            ((t.datetime_complete - t.datetime_start) for t in self.study.trials),  # type: ignore
            datetime.timedelta(),
        )
        # Trial.datetime_complete (and _start) are defined as optional in the
        # FrozenTrial type, but here they will always be set
