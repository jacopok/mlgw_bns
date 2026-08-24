from __future__ import annotations

import datetime
import logging
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
import plotly.graph_objects as go

from .data_management import Residuals
from .model import Hyperparameters, Model
from .model_validation import ValidateModel
from .neural_network import best_trial_under_n


class HyperparameterOptimization:
    """Manager for the optimization of the hyperparameters
    corresponding to a certain :class:`Model`.

    The optimization performed is over two variables:
    the reconstruction accuracy and the training time.

    **Reconstruction accuracy** is quantified by the average validation mismatch
    between EOB waveforms and model-predicted waveforms (with time-shift
    correction). The value returned by the :meth:`objective` function is the
    base-10 logarithm of this average mismatch.

    **Training time** accounts for both the time required to train the
    neural network and the estimated time required to generate the waveforms
    needed for the training.

    The number of training waveforms is fixed (default :math:`2^{14}`) to enable
    fair comparison of architectures without bias from varying training set size.
    So, we do multi-objective optimization: see, for example,
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
    n_train_fixed: int, optional
            Fixed number of training waveforms for fair architecture comparison.
            Defaults to None; if not provided, uses the class default
            (:attr:`n_train_fixed` = 2**14).

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

    # Fixed n_train for fair architecture comparison (no bias from varying training set size)
    n_train_fixed: int = 2**14

    def __init__(
        self,
        model: Model,
        optimization_seed: int = 42,
        hyper_validation_fraction: float = 0.01,
        study: Optional[optuna.Study] = None,
        n_train_fixed: Optional[int] = None,
    ):

        assert model.auxiliary_data_available
        assert model.training_dataset_available

        self.model = model
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
                Base-10 logarithm of the average validation mismatch and training
                time, respectively.

                The mismatch is the average over the validation set of the
                waveform mismatch (EOB vs model prediction, with time-shift
                correction).

                The training time includes both the training of the network and,
                roughly, the generation of the waveforms used for training.
        """
        assert self.model.training_dataset is not None
        assert self.model.training_parameters is not None

        # train network on a subset of the data
        validation_data_number = int(
            self.hyper_validation_fraction * self.training_data_number
        )
        max_n_train = self.training_data_number - validation_data_number

        # Use fixed n_train for fair architecture comparison
        n_train = min(self.n_train_fixed, max_n_train)

        hyper = Hyperparameters.from_trial(
            trial,
            n_train_max=max_n_train,
            n_train_fixed=n_train,
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

        # validate using average mismatch (GW-relevant metric)
        validation_param_set = self.model.training_parameters[validation_indices]
        validator = ValidateModel(self.model)
        try:
            mismatches = validator.mismatches_for_params(
                validation_param_set, nn, include_time_shifts=True, disable_tqdm=True
            )
            avg_mismatch = np.mean(mismatches)
        except ValueError as e:
            # Mismatch optimization failed (e.g. overflow from poorly trained model)
            raise optuna.TrialPruned from e

        return np.log10(avg_mismatch), np.log10(effective_time)

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
                "Mismatch [log10(average validation mismatch)]",
                "Time [log10(time in seconds)]",
            ],
        )
        
        # Adjust figure size and font size
        fig.update_layout(
            width=800, height=600,  # Adjust figure size
            font=dict(size=14),  # Adjust font size
            title=dict(font=dict(size=20)),  # Adjust title font size
            xaxis=dict(title_font=dict(size=16)),  # X-axis label font size
            yaxis=dict(title_font=dict(size=16))   # Y-axis label font size
        )

        # Show plot
        fig.show()

    def plot_pareto_with_selection(self, filename: str = "pareto_front.pdf") -> None:
        """
        Publication-quality Pareto plot with highlighted selected hyperparameters.
        """

        study = self.study
        trials = [t for t in study.trials if t.values is not None]

        # Extract all objectives
        errors = np.array([t.values[0] for t in trials])
        times = np.array([t.values[1] for t in trials])

        # Pareto optimal trials
        pareto_trials = study.best_trials
        pareto_errors = np.array([t.values[0] for t in pareto_trials])
        pareto_times = np.array([t.values[1] for t in pareto_trials])

        # Selected best hyperparameters
        best_hp = self.best_hyperparameters()

        # --- Robust matching ---
        selected_trial = None
        for t in pareto_trials:
            match = True
            for key, value in t.params.items():

                hp_value = getattr(best_hp, key)

                # Handle tuples vs lists
                if isinstance(hp_value, tuple):
                    if list(hp_value) != value:
                        match = False
                        break

                # Handle floats with tolerance
                elif isinstance(hp_value, float):
                    if not np.isclose(hp_value, value, rtol=1e-6):
                        match = False
                        break

                else:
                    if hp_value != value:
                        match = False
                        break

            if match:
                selected_trial = t
                break

        if selected_trial is None:
            raise RuntimeError("Could not match best hyperparameters to Pareto trial.")

        sel_error, sel_time = selected_trial.values

        # Sort Pareto front for nice connection
        sorted_idx = np.argsort(pareto_errors)
        pareto_errors = pareto_errors[sorted_idx]
        pareto_times = pareto_times[sorted_idx]

        # ---- Create Figure ----
        fig = go.Figure()

        # All trials
        fig.add_trace(go.Scatter(
            x=errors,
            y=times,
            mode="markers",
            marker=dict(size=6),
            opacity=0.3,
            name="All trials"
        ))

        # Pareto front
        fig.add_trace(go.Scatter(
            x=pareto_errors,
            y=pareto_times,
            mode="lines+markers",
            line=dict(width=2),
            marker=dict(size=8),
            name="Pareto front"
        ))

        # Selected best model
        fig.add_trace(go.Scatter(
            x=[sel_error],
            y=[sel_time],
            mode="markers",
            marker=dict(size=16, symbol="star"),
            name="Selected model"
        ))

        fig.update_layout(
            width=800,
            height=600,
            template="simple_white",
            font=dict(family="Times New Roman", size=16),
            xaxis=dict(
                title="log10 Average Validation Mismatch",
                title_font=dict(size=18),
                showgrid=True
            ),
            yaxis=dict(
                title="log10 Runtime (s)",
                title_font=dict(size=18),
                showgrid=True
            ),
            legend=dict(font=dict(size=14))
        )

        fig.write_image(filename, format="pdf")

        print(f"Saved Pareto plot to {filename}")

    def plot_parallel(self, **kwargs):
        to_plot = lambda trial: trial.values[0]
        fig = plot_parallel_coordinate(self.study, target_name=to_plot, **kwargs)
        fig.show()

    def plot_param_importance(self, filename: str = "param_importance.png"):
        fig = plot_param_importances(
            self.study, target=lambda t: t.values[0], target_name="Mismatch"
        )
        fig.write_image(filename, format="png")
        # fig.show()

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

    def save_best_trials_to_file(self, filename: str = "best_trials_modes") -> None:
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
