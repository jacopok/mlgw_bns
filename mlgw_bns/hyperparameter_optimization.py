from __future__ import annotations

import datetime
import logging
from typing import Callable, ClassVar, Optional

import joblib  # type: ignore
import numpy as np
import optuna
from optuna.visualization import plot_parallel_coordinate, plot_param_importances

from .data_management import Residuals
from .mode_model import Hyperparameters, ModeModel
from .neural_network import KernelRidgeNetwork, mode_key, save_kernel_ridge_default


class HyperparameterOptimization:
    """Manager for the optimization of :class:`KernelRidgeNetwork`'s
    hyperparameters (``kernel_gamma``, ``kernel_alpha``) for a certain
    :class:`ModeModel`, over one variable: **reconstruction accuracy**.

    Reconstruction accuracy is quantified directly in residual space,
    not through the PSD-weighted mismatch: the network's PCA-space
    prediction on a held-out validation set is expanded back to
    amplitude/phase residuals and compared to the true ones with
    :meth:`residuals_difference`. This is cheaper than a mismatch (no
    per-waveform optimization over a time shift) and, since it does not
    involve a PSD, is not implicitly weighted towards the frequencies a
    given detector is most sensitive to.

    The number of training waveforms is fixed (default :math:`2^{13} = 8192`)
    so that trials are only ever compared at the same training-set size;
    with a single objective and no architecture search (kernel ridge has
    only the two hyperparameters above), this makes the optimization
    single-objective, unlike the bi-objective (accuracy vs. training
    time) search used for the MLP.

    Once :meth:`optimize` or :meth:`optimize_and_save` has produced a
    study, :meth:`save_best_as_default` writes the best trial's
    ``(kernel_gamma, kernel_alpha)`` out as the default for this model's
    mode, read back by :meth:`~mlgw_bns.neural_network.Hyperparameters.default_kernel_ridge`.

    Parameters
    ----------
    model: ModeModel
            Reference model for the optimization. Its ``nn_kind`` is set
            to :class:`KernelRidgeNetwork`.
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
    n_train_fixed: int, optional
            Fixed number of training waveforms, for fair comparison
            across trials. Defaults to None; if not provided, uses the
            class default (:attr:`n_train_fixed` = 8192).

    Class Attributes
    save_every_n_minutes: float
            When running the optimization through :meth:`optimize`,
            every how many minutes to save the study.
            Defaults to 30.
    """

    save_every_n_minutes: float = 30.0

    #: Fixed n_train for fair comparison across trials, at 8192 waveforms.
    n_train_fixed: int = 2**13

    def __init__(
        self,
        model: ModeModel,
        optimization_seed: int = 42,
        hyper_validation_fraction: float = 0.01,
        study: Optional[optuna.Study] = None,
        n_train_fixed: Optional[int] = None,
    ):

        assert model.auxiliary_data_available
        assert model.training_dataset_available

        self.model = model
        self.model.nn_kind = KernelRidgeNetwork
        if n_train_fixed is not None:
            self.n_train_fixed = n_train_fixed
        self.rng = np.random.default_rng(seed=optimization_seed)
        self.hyper_validation_fraction = hyper_validation_fraction

        if study is None:
            try:
                self.study: optuna.Study = joblib.load(self.study_filename)
                logging.info("Loading study from %s", self.study_filename)
            except FileNotFoundError:
                self.study = optuna.create_study(
                    direction="minimize", study_name=self.model.filename
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
    ) -> float:
        """Objective function used when optimizing :class:`KernelRidgeNetwork`'s
        ``kernel_gamma`` and ``kernel_alpha``.

        Parameters
        ----------
        trial : optuna.Trial
                This object is required to generate the parameters
                according to the rules of the :module:``optuna`` optimizer used.

        Returns
        -------
        float
                Base-10 logarithm of :meth:`residuals_difference` between
                the true and predicted amplitude/phase residuals on a
                held-out validation set, at the fixed training-set size
                :attr:`n_train_fixed`.
        """
        assert self.model.training_dataset is not None
        assert self.model.training_parameters is not None

        # train network on a subset of the data
        validation_data_number = int(
            self.hyper_validation_fraction * self.training_data_number
        )
        max_n_train = self.training_data_number - validation_data_number

        # Use fixed n_train for fair comparison across trials
        n_train = min(self.n_train_fixed, max_n_train)

        hyper = Hyperparameters.from_trial_kernel_ridge(trial, n_train=n_train)

        assert hyper.n_train + validation_data_number <= self.training_data_number

        shuffled_indices = self.rng.choice(
            self.training_data_number, self.training_data_number, replace=False
        )
        training_indices = shuffled_indices[: hyper.n_train]
        validation_indices = shuffled_indices[-validation_data_number:]

        nn = self.model.train_nn(hyper, list(training_indices))

        predicted_residuals = self.model.predict_residuals_bulk(
            self.model.training_parameters[validation_indices], nn
        )
        true_residuals = self.model.training_dataset[validation_indices]

        accuracy = self.residuals_difference(true_residuals, predicted_residuals)

        return float(np.log10(accuracy))

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

    def plot_parallel(self, **kwargs):
        fig = plot_parallel_coordinate(self.study, target_name="Accuracy", **kwargs)
        fig.show()

    def plot_param_importance(self, filename: str = "param_importance.png"):
        fig = plot_param_importances(self.study, target_name="Accuracy")
        fig.write_image(filename, format="png")
        # fig.show()

    def best_hyperparameters(self) -> Hyperparameters:
        """Return the hyperparameters of the best trial found so far.

        Returns
        -------
        Hyperparameters
        """
        return Hyperparameters.from_frozen_trial(self.study.best_trial)

    def save_best_as_default(self) -> None:
        """Write the best trial's ``(kernel_gamma, kernel_alpha)`` out as
        the default for this model's mode, via
        :func:`~mlgw_bns.neural_network.save_kernel_ridge_default`.

        Read back by
        :meth:`~mlgw_bns.neural_network.Hyperparameters.default_kernel_ridge`,
        which is what :meth:`~mlgw_bns.mode_model.ModeModel.set_hyper_and_train_nn`
        falls back to when trained with :class:`KernelRidgeNetwork` and no
        explicit hyperparameters.
        """
        best = self.best_hyperparameters()
        save_kernel_ridge_default(self.model.mode, best.kernel_gamma, best.kernel_alpha)
        logging.info(
            "Saved best trial for mode %s as default: kernel_gamma=%.3g kernel_alpha=%.3g",
            mode_key(self.model.mode),
            best.kernel_gamma,
            best.kernel_alpha,
        )

    def save_best_trials_to_file(self, filename: str = "best_trials_modes") -> None:
        """Save the best trial found so far in the optimization to the
        file "filename".pkl, as a single-element list, for compatibility
        with :func:`~mlgw_bns.neural_network.retrieve_best_trials_list`.

        Parameters
        ----------
        filename : str, optional
            Filename to save to, by default "best_trials"
        """
        joblib.dump([self.study.best_trial], f"{filename}.pkl")

    def total_training_time(self) -> datetime.timedelta:
        return sum(
            ((t.datetime_complete - t.datetime_start) for t in self.study.trials),  # type: ignore
            datetime.timedelta(),
        )
        # Trial.datetime_complete (and _start) are defined as optional in the
        # FrozenTrial type, but here they will always be set
