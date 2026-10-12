from __future__ import annotations

import dataclasses
import datetime
import hashlib
import logging
import os
import signal
import socket
import subprocess
import threading
import time
import warnings
from typing import Callable, ClassVar, Optional, Sequence

import joblib  # type: ignore
import numpy as np
import optuna
from optuna.visualization import plot_parallel_coordinate, plot_param_importances

from .data_management import Residuals
from .jax_mlp import MLPConfig, TrainingInterrupted
from .mode_model import Hyperparameters, ModeModel
from .neural_network import KernelRidgeNetwork, mode_key, save_kernel_ridge_default
from .sharding import EXIT_REQUEUE, Stopped


class HyperparameterOptimization:
    """Manager for the optimization of :class:`KernelRidgeNetwork`'s
    hyperparameters for a certain :class:`ModeModel`, over one variable:
    **reconstruction accuracy**.

    With ``kernel_alpha_selection="loo"`` (the default) only the kernel
    width ``kernel_gamma`` is searched: the fit then chooses one ridge
    penalty per principal component itself, by exact leave-one-out
    cross-validation together with the rounding error of the resulting
    predictions (see
    :func:`~mlgw_bns.neural_network.kernel_ridge_leave_one_out`). With
    ``"fixed"``, one ``kernel_alpha`` shared by all the components is
    searched as well, as the packaged models were tuned; that search
    drove the (2,2) penalty down to 4e-14, where the fit is numerically
    unregularized and its predictions carry ~1e-2 rad of rounding noise.

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
    kernel_alpha_selection: str, optional
            ``"loo"`` (default) or ``"fixed"``; see above.

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
        kernel_alpha_selection: str = "loo",
    ):

        assert model.auxiliary_data_available
        assert model.training_dataset_available

        self.model = model
        self.model.nn_kind = KernelRidgeNetwork
        self.kernel_alpha_selection = kernel_alpha_selection
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
        """Name of the file to save the study to.

        Studies with per-output leave-one-out penalties are kept apart from
        the ones which searched a single ``kernel_alpha``: their trials
        sample different parameters.
        """
        if self.kernel_alpha_selection == "loo":
            return f"{self.model.filename}_loo_study.pkl"
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

        hyper = Hyperparameters.from_trial_kernel_ridge(
            trial, n_train=n_train, kernel_alpha_selection=self.kernel_alpha_selection
        )

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
    def residuals_difference(
        residuals_1: Residuals, residuals_2: Residuals, weights: Optional[np.ndarray] = None
    ) -> float:
        """Compare two sets of :class:`Residuals`.

        Parameters
        ----------
        residuals_1 : Residuals
            First set of residuals to be compared.
        residuals_2 : Residuals
            Second set of residuals to be compared.
        weights : np.ndarray, optional
            Of each waveform in the average (uniform by default), such as
            the power weights the regressors of some modes are trained with
            (:meth:`ModeModel._training_power_weights
            <mlgw_bns.mode_model.ModeModel._training_power_weights>`).

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
        if weights is not None:
            weights = np.asarray(weights, dtype=float) / np.mean(weights)
            amp_square_differences = amp_square_differences * weights[:, np.newaxis]
            phi_square_differences = phi_square_differences * weights[:, np.newaxis]

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
            datetime.datetime.now(),
            expected_datetime_end.isoformat(),
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
        kernel_alpha = best.kernel_alpha if best.kernel_alpha_selection == "fixed" else None
        save_kernel_ridge_default(self.model.mode, best.kernel_gamma, kernel_alpha)
        logging.info(
            "Saved best trial for mode %s as default: kernel_gamma=%.3g kernel_alpha=%s",
            mode_key(self.model.mode),
            best.kernel_gamma,
            "per-component (leave-one-out)" if kernel_alpha is None else f"{kernel_alpha:.3g}",
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


# -- studies shared by many workers (SLURM) -------------------------------------
#
# The optimization of regressors trained on large sharded training sets
# (:mod:`mlgw_bns.modes_dataset`, :mod:`mlgw_bns.precession_dataset`), by any
# number of worker processes on any number of nodes: the study is kept in an
# optuna journal file on a filesystem they all see (no database server), each
# worker reads the training and validation sets once and then asks the study
# for trials until it is stopped (or a budget of trial-hours is spent); the
# workers of a node are started together by :func:`run_in_parallel`. A worker asked to stop (the
# walltime near) abandons the trial it is running to a retry, which resumes
# from the checkpoint of its perceptron; a worker killed outright leaves a
# trial whose heartbeat stops, and which the next worker to start retries.

#: The perceptrons' hyperparameters searched by :func:`mlp_config_from_trial`:
#: name -> (low, high, log) of a float or int, or the choices of a categorical.
MLP_SEARCH_SPACE: dict = {
    "depth": (2, 8, False),
    "width": (64, 1024, True),
    "activation": ("gelu", "silu", "tanh"),
    "learning_rate": (1e-4, 3e-3, True),
    # the final learning rate, as a fraction of the first
    "final_rate_ratio": (1e-4, 1e-1, True),
    "warmup": (0.0, 0.1, False),
    "clip_norm": (1e-2, 1e2, True),
    "log2_batch": (7, 12, False),
    "weight_decay": (1e-8, 1e-2, True),
}


def _suggest(trial: optuna.Trial, name: str):
    space = MLP_SEARCH_SPACE[name]
    if isinstance(space[0], str):
        return trial.suggest_categorical(name, list(space))
    low, high, log = space
    if isinstance(low, int) and isinstance(high, int):
        return trial.suggest_int(name, low, high, log=log)
    return trial.suggest_float(name, low, high, log=log)


def mlp_config_from_trial(
    trial: optuna.Trial, steps: int, n_train: int, base: Optional[MLPConfig] = None
) -> MLPConfig:
    """A :class:`~mlgw_bns.jax_mlp.MLPConfig` sampled from ``trial`` over
    :data:`MLP_SEARCH_SPACE` (layers of equal width), the others of ``base``
    (the defaults), trained for about ``steps`` steps on ``n_train`` samples
    (:meth:`~mlgw_bns.jax_mlp.MLPConfig.with_steps`): the number of steps
    is fixed in a study, since it sets the cost of a trial and the
    schedule of the learning rate.

    The seed (of the initialization, the held-out split and the batches) is
    a function of the parameters (:func:`trial_seed`): it differs from
    trial to trial, as it must, since how well a perceptron trains varies
    with its seed by as much as with its hyperparameters (a study with one
    seed would favour the configurations lucky with it); and it is the same
    for a retry of a trial, which resumes from its checkpoint."""
    base = MLPConfig() if base is None else base
    rate = _suggest(trial, "learning_rate")
    config = dataclasses.replace(
        base,
        hidden=(_suggest(trial, "width"),) * _suggest(trial, "depth"),
        activation=_suggest(trial, "activation"),
        learning_rate=rate,
        final_learning_rate=rate * _suggest(trial, "final_rate_ratio"),
        warmup=_suggest(trial, "warmup"),
        clip_norm=_suggest(trial, "clip_norm"),
        batch_size=2 ** _suggest(trial, "log2_batch"),
        weight_decay=_suggest(trial, "weight_decay"),
        seed=trial_seed(trial),
    )
    trial.set_user_attr("seed", config.seed)
    return config.with_steps(steps, n_train)


def _params_digest(trial: optuna.Trial) -> bytes:
    return hashlib.blake2b(repr(sorted(trial.params.items())).encode(), digest_size=8).digest()


def trial_seed(trial: optuna.Trial) -> int:
    """A seed for ``trial``, from the parameters it has sampled so far."""
    return int.from_bytes(_params_digest(trial)[:4], "little") & 0x7FFFFFFF


def mlp_command_line(config: MLPConfig) -> str:
    """The options of ``precession_scale.py train`` or ``modes_scale.py
    train`` training a perceptron of this configuration (with
    ``--mlp-steps`` for its number of epochs)."""
    return " ".join([
        "--mlp --mlp-hidden", " ".join(str(h) for h in config.hidden),
        f"--mlp-activation {config.activation} --mlp-rate {config.learning_rate:.3g}",
        f"--mlp-final-rate {config.final_learning_rate:.3g} --mlp-batch {config.batch_size}",
        f"--mlp-warmup {config.warmup:.3g} --mlp-clip-norm {config.clip_norm:.3g}",
        f"--mlp-weight-decay {config.weight_decay:.3g} --mlp-seed {config.seed}",
    ])


def open_study(
    filename: str,
    study_name: str = "study",
    pruner: Optional[optuna.pruners.BasePruner] = None,
    fixed: Optional[dict] = None,
    seed: Optional[int] = None,
    sampler: str = "tpe",
    metric: Optional[str] = None,
) -> optuna.Study:
    """The study ``study_name`` (minimizing one objective, named ``metric``)
    kept in the optuna journal ``filename``, made if not there. Its trials
    are sampled by multivariate TPE (``sampler="tpe"``; with the "constant
    liar", so that workers running at once do not all try the same) or by a
    Gaussian process (``"gp"``: its acquisition integrating over the
    outcomes of the trials running; it learns from the completed trials
    only, not the pruned ones, and needs torch), with the parameters in
    ``fixed`` held to their values."""
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        if sampler == "tpe":
            chosen: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(
                multivariate=True, constant_liar=True, seed=seed
            )
        elif sampler == "gp":
            chosen = optuna.samplers.GPSampler(seed=seed)
        else:
            raise ValueError(f"unknown sampler {sampler!r}: tpe or gp")
        if fixed:
            chosen = optuna.samplers.PartialFixedSampler(fixed, chosen)
    study = optuna.create_study(
        storage=JournalStorage(JournalFileBackend(filename)),
        study_name=study_name,
        direction="minimize",
        pruner=pruner if pruner is not None else optuna.pruners.NopPruner(),
        sampler=chosen,
        load_if_exists=True,
    )
    if metric is not None and study.metric_names != [metric]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            study.set_metric_names([metric])
    return study


def read_study(filename: str) -> optuna.Study:
    """The study kept in the optuna journal ``filename`` (its only one), read
    without writing to it or taking its lock: of studies running, or copied
    from the cluster at any time (the journal is appended to, line by line,
    and a last line not written in full is left out)."""
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    storage = JournalStorage(JournalFileBackend(filename))
    (summary,) = storage.get_all_studies()
    return optuna.load_study(study_name=summary.study_name, storage=storage)


def pruner_settings(kind: str, start: Optional[float] = None, reduction: int = 3) -> dict:
    """The settings of :func:`make_pruner`, with the defaults filled in."""
    if kind == "none":
        return {"kind": kind}
    if kind == "median":
        return {"kind": kind, "start": 0.2 if start is None else start}
    if kind == "halving":
        return {"kind": kind, "start": 0.1 if start is None else start, "reduction": reduction}
    raise ValueError(f"unknown pruner {kind!r}: none, median or halving")


def make_pruner(kind: str, steps: int, start: Optional[float] = None, reduction: int = 3) -> optuna.pruners.BasePruner:
    """``"none"``; ``"median"``: a trial whose best held-out loss is worse
    than the median of the others' at the same step, after the first
    ``start`` (default a fifth) of the steps; ``"halving"``: successive
    halving, the trials compared after ``start`` (default a tenth) of the
    steps, then ``reduction`` times as many and so on, the best
    1 / ``reduction`` kept each time."""
    settings = pruner_settings(kind, start, reduction)
    if kind == "none":
        return optuna.pruners.NopPruner()
    first = max(int(settings["start"] * steps), 1)
    if kind == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=first)
    return optuna.pruners.SuccessiveHalvingPruner(min_resource=first, reduction_factor=reduction)


# successive halving keeps the values of the trials at each rung (as system
# attributes "completed_rung_<i>"), which before the settings were kept were
# those of its only settings then
_LEGACY_HALVING = {"kind": "halving", "start": 0.1, "reduction": 3}


def check_pruner_settings(study: optuna.Study, settings: dict) -> None:
    """Keep ``settings`` (:func:`pruner_settings`) in ``study``, or raise
    ValueError if it was run with others: successive halving compares a
    trial with the others at the same rung, which with other settings are
    at other steps."""
    kept = study.user_attrs.get("pruner")
    if kept is None and any("completed_rung_0" in t.system_attrs for t in study.get_trials(deepcopy=False)):
        kept = _LEGACY_HALVING
    if kept is None:
        study.set_user_attr("pruner", settings)
    elif kept != settings:
        raise ValueError(
            f"{study.study_name} was run with the pruner {kept}, not {settings}: use these, or another study"
        )


def _value(trial: optuna.trial.FrozenTrial) -> float:
    return np.inf if trial.value is None else trial.value


def enqueue_best(study: optuna.Study, source: optuna.Study, number: int) -> int:
    """Enqueue the parameters of the ``number`` best trials of ``source``
    (another study, e.g. on fewer training samples) in ``study``, for its
    workers to try first; how many were not tried or enqueued there yet."""
    tried = {tuple(sorted(t.params.items())) for t in study.get_trials(deepcopy=False)}
    tried |= {
        tuple(sorted(_enqueued(t).items()))
        for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.WAITING,))
    }
    done = [t for t in source.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))]
    added = 0
    for trial in sorted(done, key=_value)[:number]:
        if tuple(sorted(trial.params.items())) not in tried:
            _enqueue(study, trial.params, enqueued_from=f"{source.study_name}#{trial.number}")
            added += 1
    return added


def trial_hours(study: optuna.Study) -> float:
    """The hours the finished trials of ``study`` took, summed: what the
    budget of :func:`run_worker` counts."""
    seconds = sum(
        (t.datetime_complete - t.datetime_start).total_seconds()
        for t in study.get_trials(deepcopy=False)
        if t.state.is_finished() and t.datetime_start is not None and t.datetime_complete is not None
    )
    return seconds / 3600.0


def checkpoint_name(directory: str, trial: optuna.Trial) -> str:
    """The checkpoint of a trial's perceptron: named by its parameters, so
    that a retry of the trial (same parameters) resumes from it."""
    return os.path.join(directory, f"trial-{_params_digest(trial).hex()}.joblib")


class StopRequested:
    """``SIGTERM`` and ``SIGUSR1`` (which SLURM sends before the walltime)
    noted rather than acted on at once, since the study's journal must not
    be interrupted mid-write: the worker stops between trials, or inside
    one through :meth:`check` (a perceptron with a checkpoint handles them
    itself while it trains)."""

    def __init__(self):
        self.signal: Optional[int] = None
        for number in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(number, self._note)

    def _note(self, signum, _frame) -> None:
        self.signal = signum

    def __bool__(self) -> bool:
        return self.signal is not None

    def check(self) -> None:
        if self.signal is not None:
            raise Stopped(f"signal {self.signal}")


def pruning_progress(
    trial: optuna.Trial, stop: StopRequested, steps: Optional[int] = None
) -> Callable[[int, float], None]:
    """A ``progress`` of :meth:`JaxMLP.fit <mlgw_bns.jax_mlp.JaxMLP.fit>`
    reporting the best held-out loss to ``trial`` (in log10), pruning it if
    its study's pruner says so (not in the last 5% of its ``steps``, its
    training as good as done), and stopping on a signal."""
    last = np.inf if steps is None else 0.95 * steps

    def progress(step: int, best_loss: float) -> None:
        stop.check()
        trial.report(float(np.log10(best_loss)), step)
        if step < last and trial.should_prune():
            raise optuna.TrialPruned(f"pruned at step {step}")

    return progress


def _enqueue(study: optuna.Study, params: dict, **user_attrs) -> None:
    """Enqueue a trial of ``params`` in ``study``, with ``user_attrs``, and
    the parameters themselves (as the user attribute ``enqueued``: optuna
    keeps them where only it can read them)."""
    study.enqueue_trial(params, user_attrs={**user_attrs, "enqueued": params})


def _enqueued(trial: optuna.trial.FrozenTrial) -> dict:
    """The parameters ``trial`` was enqueued with (:func:`_enqueue`)."""
    return dict(trial.user_attrs.get("enqueued", {}))


def _retry(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Enqueue ``trial`` again: the parameters it sampled, and those it was
    enqueued with (if it stopped before sampling them)."""
    _enqueue(study, {**_enqueued(trial), **trial.params}, retry_of=trial.number)


def recover_stale_trials(study: optuna.Study, stale_seconds: float) -> int:
    """Fail the running trials of ``study`` whose heartbeat stopped more
    than ``stale_seconds`` ago (their worker died), and enqueue them again;
    how many."""
    now = time.time()
    recovered = 0
    for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        beat = trial.user_attrs.get("heartbeat")
        if beat is None and trial.datetime_start is not None:
            beat = trial.datetime_start.timestamp()
        if beat is None or now - beat < stale_seconds:
            continue
        try:
            study._storage.set_trial_state_values(trial._trial_id, optuna.trial.TrialState.FAIL)
        except optuna.exceptions.UpdateFinishedTrialError:
            continue  # another worker got there first
        _retry(study, trial)
        recovered += 1
        logging.info("trial %i of %s: no heartbeat for %.0f s, enqueued again", trial.number, study.study_name, now - beat)
    return recovered


class _Heartbeat:
    """Sets the user attribute ``heartbeat`` (the time) of a running trial
    every ``seconds``, in a thread, until stopped."""

    def __init__(self, trial: optuna.Trial, seconds: float):
        self.trial, self.seconds = trial, seconds
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while True:
            try:
                self.trial.set_user_attr("heartbeat", time.time())
            except Exception as error:  # the trial finished meanwhile
                logging.debug("heartbeat: %s", error)
            if self.done.wait(self.seconds):
                return

    def __enter__(self) -> "_Heartbeat":
        self.thread.start()
        return self

    def __exit__(self, *_) -> None:
        self.done.set()
        self.thread.join()


def run_worker(
    study: optuna.Study,
    objective: Callable[[optuna.Trial, StopRequested], float],
    hours: Optional[float] = None,
    stop: Optional[StopRequested] = None,
    stale_minutes: float = 30.0,
    heartbeat_seconds: float = 60.0,
    max_failures: int = 3,
) -> bool:
    """Run trials of ``study`` until the trials of all its workers have
    taken ``hours`` (:func:`trial_hours`; ``None``: until stopped); whether
    that budget is spent (otherwise the worker was asked to stop, and the
    trial it was running enqueued again).

    ``objective(trial, stop)`` gives the value of a trial, or raises
    :class:`optuna.TrialPruned`; :class:`~mlgw_bns.sharding.Stopped` or
    :class:`~mlgw_bns.jax_mlp.TrainingInterrupted` when stopped (see
    :class:`StopRequested`). A trial raising anything else fails; after
    ``max_failures`` in a row the worker gives up (raises it).
    """
    stop = StopRequested() if stop is None else stop
    recover_stale_trials(study, 60.0 * stale_minutes)
    worker = f"{socket.gethostname()}:{os.getpid()}"
    job = os.environ.get("SLURM_JOB_ID")
    failures = 0
    while True:
        spent = trial_hours(study)
        if hours is not None and spent >= hours:
            logging.info("%s: %.3g of %.3g hours of trials spent", study.study_name, spent, hours)
            return True
        if stop:
            return False
        trial = study.ask()
        trial.set_user_attr("worker", worker if job is None else f"{worker} job {job}")
        start = time.time()
        try:
            with _Heartbeat(trial, heartbeat_seconds):
                value = objective(trial, stop)
        except optuna.TrialPruned as pruned:
            logging.info("trial %i: %s after %.0f s", trial.number, pruned, time.time() - start)
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            failures = 0
        except (Stopped, TrainingInterrupted) as error:
            logging.info("trial %i stopped (%s): enqueued again", trial.number, error)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            _retry(study, study.get_trials(deepcopy=False)[trial.number])
            return False
        except Exception:
            logging.exception("trial %i failed", trial.number)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            failures += 1
            if failures >= max_failures:
                raise
        else:
            study.tell(trial, value)
            failures = 0
            best = study.best_trial
            logging.info(
                "trial %i: %.4g in %.0f s (best: trial %i, %.4g)",
                trial.number, value, time.time() - start, best.number, best.value,
            )


def run_in_parallel(command: Sequence[str], workers: int) -> int:
    """Run ``workers`` copies of ``command`` at once, each on its share of
    the CPUs of this process (its affinity, and as many threads), with
    ``SIGUSR1`` and ``SIGTERM`` forwarded to them; the exit status: the first
    of theirs other than 0 and ``EXIT_REQUEUE``, else ``EXIT_REQUEUE`` if one
    stopped so, else 0."""
    pinned = hasattr(os, "sched_getaffinity")
    cpus = sorted(os.sched_getaffinity(0)) if pinned else list(range(os.cpu_count() or 1))
    shares = [[int(c) for c in share] for share in np.array_split(cpus, workers)]
    processes: list = []
    received: list = []

    def forward(signum, _frame) -> None:
        received.append(signum)
        for process in processes:
            if process.poll() is None:
                process.send_signal(signum)

    previous = {s: signal.signal(s, forward) for s in (signal.SIGUSR1, signal.SIGTERM)}
    try:
        for share in shares:
            threads = str(max(len(share), 1))
            env = dict(
                os.environ, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
                XLA_PYTHON_CLIENT_PREALLOCATE="false",  # workers sharing a GPU
            )
            setaffinity = (lambda share=share: os.sched_setaffinity(0, share)) if pinned and share else None
            processes.append(subprocess.Popen(list(command), env=env, preexec_fn=setaffinity))
            for signum in received:  # arrived while the others started
                processes[-1].send_signal(signum)
        codes = [process.wait() for process in processes]
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    logging.info("workers exited with %s", codes)
    return next((c for c in codes if c not in (0, EXIT_REQUEUE)), EXIT_REQUEUE if EXIT_REQUEUE in codes else 0)


def study_report(study: optuna.Study, top: int = 10) -> str:
    """The best ``top`` trials of ``study``, their parameters and user
    attributes, how many trials ended how, and the importance of each
    parameter (PED-ANOVA: how much more often its values among the best
    trials fall where they do than among all)."""
    trials = study.get_trials(deepcopy=False)
    counts = {state.name.lower(): 0 for state in optuna.trial.TrialState}
    for trial in trials:
        counts[trial.state.name.lower()] += 1
    lines = [
        f"{study.study_name}: {len(trials)} trials ("
        + ", ".join(f"{n} {state}" for state, n in counts.items() if n)
        + f"), {trial_hours(study):.3g} hours"
    ]
    done = sorted((t for t in trials if t.state == optuna.trial.TrialState.COMPLETE), key=_value)
    for trial in done[:top]:
        attrs = {k: v for k, v in trial.user_attrs.items() if k not in ("heartbeat", "worker", "enqueued")}
        lines.append(f"  #{trial.number}: {trial.value:.4g}  {trial.params}  {attrs}")
    if len(done) >= 10:
        importances = optuna.importance.get_param_importances(study)
        lines.append("  importance: " + ", ".join(f"{name} {value:.2f}" for name, value in importances.items()))
    return "\n".join(lines)
