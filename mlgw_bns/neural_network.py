r"""Neural-network and Gaussian-Process wrappers used by the surrogate.

This module collects the small set of regressors that the surrogate
relies on at training and prediction time:

* :class:`Hyperparameters` --- a dataclass holding every hyperparameter
  needed to instantiate the main scikit-learn :class:`MLPRegressor`,
  plus a handful of "extras" specific to the surrogate
  (``pc_exponent``, ``n_train``). It also provides constructors that
  read the parameters from an Optuna trial, which is how the
  hyperparameter-optimization pipeline talks to the surrogate.

* :class:`NeuralNetwork` --- a thin abstract base class defining the
  fit/predict/save/load interface used by the rest of the codebase, so
  that different backends can be swapped in transparently.

* :class:`SklearnNetwork` --- the concrete implementation used in
  production. Wraps an :class:`MLPRegressor` together with a
  :class:`StandardScaler` for the input features.

* :class:`TimeshiftsGPR` and :class:`TimeshiftsNN` --- two surrogates
  for the merger time shifts between higher-order modes.
  :class:`TimeshiftsNN` trains an RFF + Ridge model (fast, compact) and
  is the primary trainer and predictor; :class:`TimeshiftsGPR` remains
  available as a fallback reference implementation.

* :func:`retrieve_best_trials_list` and :func:`best_trial_under_n`
  --- helpers for fetching the pretrained Pareto front of best
  hyperparameter trials shipped with the package.

The optional PyTorch backend that used to live in this module has been
removed; only the scikit-learn backend is now supported.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING, Optional, Union

import joblib  # type: ignore
import numpy as np
import pkg_resources
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
from sklearn.kernel_approximation import RBFSampler  # type: ignore
from sklearn.linear_model import Ridge  # type: ignore
from sklearn.neural_network import MLPRegressor  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore

if TYPE_CHECKING:
    import optuna

#: Location, relative to the package, of the joblib-pickled Pareto front
#: produced by the hyperparameter-optimization pipeline.
TRIALS_FILE = "data/best_trials.pkl"


@dataclass
class Hyperparameters:
    r"""All hyperparameters needed to train one per-mode surrogate.

    The bulk of the attributes are passed verbatim to
    :class:`~sklearn.neural_network.MLPRegressor` via :attr:`nn_params`.
    Two additional attributes, :attr:`pc_exponent` and :attr:`n_train`,
    control the surrogate-specific preprocessing rather than the network
    itself.

    Parameters
    ----------
    pc_exponent : float
            Exponent used in the normalization of the principal
            components: the network learns to reconstruct
            :math:`x_i\, \lambda_i^\alpha`, where :math:`x_i` are the
            principal-component coordinates of a waveform,
            :math:`\lambda_i` are the corresponding PCA eigenvalues, and
            :math:`\alpha` is this parameter. Larger values give more
            weight to higher-order components.
    n_train : int
            Number of waveforms used during training.
    hidden_layer_sizes : tuple[int, ...]
            Sizes of the hidden layers of the MLP.
            See :class:`MLPRegressor` for details.
    activation : str
            Activation function (e.g. ``"relu"``, ``"tanh"``,
            ``"logistic"``). See :class:`MLPRegressor`.
    alpha : float
            L2 regularization strength. See :class:`MLPRegressor`.
    batch_size : int
            Mini-batch size. Capped at training-set size in
            :attr:`nn_params`. See :class:`MLPRegressor`.
    learning_rate_init : float
            Initial learning rate for the Adam optimizer.
            See :class:`MLPRegressor`.
    tol : float
            Tolerance for the optimizer's convergence criterion.
            See :class:`MLPRegressor`.
    validation_fraction : float
            Fraction of the training set held out for validation
            inside :class:`MLPRegressor`.
    n_iter_no_change : int
            Number of iterations with no improvement before optimization
            is stopped. See :class:`MLPRegressor`.
    max_iter : int, optional
            Hard upper bound on the number of training iterations.
            Defaults to 1000.
    """

    pc_exponent: float
    n_train: int

    hidden_layer_sizes: tuple[int, ...]
    activation: str
    alpha: float
    batch_size: int
    learning_rate_init: float
    tol: float
    validation_fraction: float
    n_iter_no_change: float

    max_iter: int = field(default=1000)

    @property
    def n_layers(self) -> int:
        """Number of hidden layers in the network."""
        return len(self.hidden_layer_sizes)

    @property
    def nn_params(self) -> dict[str, Union[int, float, str, bool, tuple[int, ...]]]:
        """Keyword arguments suitable for instantiating an :class:`MLPRegressor`.

        The ``batch_size`` is clipped to the actual training set size so
        that :class:`MLPRegressor` does not silently fall back to the
        full-batch mode, and ``early_stopping`` is disabled because the
        surrogate handles its own early stopping logic via ``tol`` and
        ``n_iter_no_change``.
        """
        return {
            "max_iter": self.max_iter,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "tol": self.tol,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "early_stopping": False,
            "shuffle": True,
            "batch_size": min(
                self.batch_size, int(self.n_train * (1 - self.validation_fraction))
            ),
        }

    @classmethod
    def from_trial(
        cls,
        trial: "optuna.Trial",
        n_train_max: int,
        n_train_fixed: Optional[int] = None,
    ) -> "Hyperparameters":
        """Sample a :class:`Hyperparameters` from an :class:`optuna.Trial`.

        Used by :mod:`~mlgw_bns.hyperparameter_optimization` to drive a
        bi-objective (accuracy vs. training-set size) optimization.

        Parameters
        ----------
        trial : optuna.Trial
                Trial object used to draw each hyperparameter value.
        n_train_max : int
                Upper bound for :attr:`n_train` when it is sampled.
        n_train_fixed : int, optional
                If provided, fix :attr:`n_train` to this value (the
                attribute is still registered on the trial, as a
                degenerate ``[fixed, fixed]`` interval). This enables a
                fair architecture comparison at a fixed dataset size.

        Returns
        -------
        Hyperparameters
                A freshly sampled hyperparameter set.
        """
        n_layers = trial.suggest_int("n_layers", 1, 4)

        layers = tuple(
            trial.suggest_int(f"size_layer_{i}", 10, 200) for i in range(n_layers)
        )

        if n_train_fixed is not None:
            n_train = n_train_fixed
            trial.suggest_int("n_train", n_train_fixed, n_train_fixed)
        else:
            n_train = trial.suggest_int("n_train", 50, n_train_max)

        return cls(
            hidden_layer_sizes=layers,
            activation=str(
                trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])
            ),
            alpha=trial.suggest_loguniform("alpha", 1e-6, 1e-1),
            batch_size=trial.suggest_int("batch_size", 100, 200),
            learning_rate_init=trial.suggest_loguniform(
                "learning_rate_init", 2e-4, 5e-2
            ),
            tol=trial.suggest_loguniform("tol", 1e-15, 1e-7),
            validation_fraction=trial.suggest_uniform("validation_fraction", 0.05, 0.2),
            n_iter_no_change=trial.suggest_int(
                "n_iter_no_change", 40, 100, log=True
            ),
            pc_exponent=trial.suggest_loguniform("pc_exponent", 1e-3, 1),
            n_train=n_train,
        )

    @classmethod
    def from_frozen_trial(
        cls, frozen_trial: "optuna.trial.FrozenTrial"
    ) -> "Hyperparameters":
        """Reconstruct a :class:`Hyperparameters` from a frozen Optuna trial.

        This is the inverse of :meth:`from_trial` for already-completed
        trials, used when reading the pretrained Pareto front shipped
        with the package.

        Parameters
        ----------
        frozen_trial : optuna.trial.FrozenTrial
                Completed trial whose params dictionary contains the
                values produced by :meth:`from_trial`.

        Returns
        -------
        Hyperparameters
                Hyperparameter set corresponding to the trial.
        """
        params = frozen_trial.params
        n_layers = params.pop("n_layers")
        layers = [params.pop(f"size_layer_{i}") for i in range(n_layers)]
        params["hidden_layer_sizes"] = tuple(layers)

        return cls(**params)

    @classmethod
    def default(cls, training_waveform_number: Optional[int] = None) -> "Hyperparameters":
        """Hand-tuned defaults used for the ``(3, 3)`` mode surrogate.

        These values come from an Optuna optimization run and are kept
        here as a checked-in fallback so that the surrogate can be
        trained without first running a full hyperparameter search.

        Parameters
        ----------
        training_waveform_number : int
                Number of training waveforms. Must be provided
                explicitly --- this is asserted at runtime.

        Returns
        -------
        Hyperparameters
                Hyperparameter set with the hard-coded default values.
        """
        assert training_waveform_number is not None

        return cls(
            hidden_layer_sizes=(169, 71),
            activation="relu",
            alpha=0.0008918136131265236,
            batch_size=160,
            learning_rate_init=0.0002353780383291372,
            tol=4.6659267067767714e-14,
            n_iter_no_change=74,
            validation_fraction=0.07405053167928363,
            pc_exponent=0.01948145530324084,
            n_train=861,
        )


class NeuralNetwork(ABC):
    """Abstract base class for a neural-network wrapper.

    Concrete subclasses must implement :meth:`fit`, :meth:`predict`,
    :meth:`save` and :meth:`from_file`. The class is intentionally
    minimal so that backends other than scikit-learn could be added
    without changing the rest of the codebase.

    Parameters
    ----------
    hyper : Hyperparameters
            Hyperparameters used to configure the underlying network.
    """

    def __init__(self, hyper: Hyperparameters):
        self.hyper = hyper

    @abstractmethod
    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """Fit the network to the given training data.

        Parameters
        ----------
        x_data : np.ndarray
                Input features, shape ``(n_samples, n_features)``.
        y_data : np.ndarray
                Targets, shape ``(n_samples, n_outputs)`` (or
                ``(n_samples,)`` for a scalar target).
        """

    @abstractmethod
    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """Evaluate the network on new inputs.

        Parameters
        ----------
        x_data : np.ndarray
                Input features, shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
                Predicted outputs with the same leading dimension as
                ``x_data``.
        """

    @abstractmethod
    def save(self, filename: str) -> None:
        """Persist the network to disk.

        Parameters
        ----------
        filename : str
                Path of the file to write to.
        """

    @classmethod
    @abstractmethod
    def from_file(cls, filename: Union[IO[bytes], str]) -> "NeuralNetwork":
        """Load a previously saved network.

        Parameters
        ----------
        filename : str or IO[bytes]
                Either a path to read from or an open binary stream.

        Returns
        -------
        NeuralNetwork
                Reconstructed instance.
        """


class SklearnNetwork(NeuralNetwork):
    """Wrapper around an :class:`sklearn.neural_network.MLPRegressor`.

    Inputs are standardized with a :class:`StandardScaler` before being
    passed to the regressor; the scaler is fitted once at :meth:`fit`
    time and stored alongside the network so that :meth:`predict` can
    apply the same transform at inference time.

    Parameters
    ----------
    hyper : Hyperparameters
            Hyperparameters configuring the network. The actual
            scikit-learn keyword arguments are produced by
            :attr:`Hyperparameters.nn_params`.
    nn : MLPRegressor, optional
            Pre-built regressor to wrap. If ``None`` (the default), a
            fresh one is instantiated from ``hyper``.
    param_scaler : StandardScaler, optional
            Pre-fitted scaler for the inputs. If ``None``, the scaler
            will be fitted the first time :meth:`fit` is called.
    """

    def __init__(
        self,
        hyper: Hyperparameters,
        nn: Optional[MLPRegressor] = None,
        param_scaler: Optional[StandardScaler] = None,
    ):
        super().__init__(hyper=hyper)
        self.nn = nn if nn is not None else MLPRegressor(**hyper.nn_params)
        if param_scaler is not None:
            self.param_scaler: StandardScaler = param_scaler

    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """Fit the scaler and the underlying :class:`MLPRegressor`.

        The mini-batch size is temporarily clipped to the input feature
        count to avoid scikit-learn's "batch_size larger than data" warning;
        it is restored to the configured value once training completes.
        """
        self.param_scaler = StandardScaler().fit(x_data)

        old_batch_size = self.nn.batch_size
        self.nn.batch_size = min(self.nn.batch_size, x_data.shape[1])

        scaled_x = self.param_scaler.transform(x_data)
        self.nn.fit(scaled_x, y_data)

        self.nn.batch_size = old_batch_size

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        scaled_x = self.param_scaler.transform(x_data)
        return self.nn.predict(scaled_x)

    def get_loss_over_epochs(self) -> list[float]:
        """Return the training loss recorded at each epoch.

        Returns
        -------
        list[float]
                Loss values from :attr:`MLPRegressor.loss_curve_`,
                one per completed iteration.
        """
        return [self.nn.loss_curve_[epoch] for epoch in range(self.nn.n_iter_)]

    def save(self, filename: str) -> None:
        """Pickle ``(hyper, nn, param_scaler)`` to ``filename`` via joblib."""
        joblib.dump((self.hyper, self.nn, self.param_scaler), filename)

    @classmethod
    def from_file(cls, filename: Union[IO[bytes], str]) -> "SklearnNetwork":
        """Inverse of :meth:`save`. The tuple is unpacked into the constructor."""
        return cls(*joblib.load(filename))


class TimeshiftsGPR:
    """Gaussian-Process regressor for merger time shifts between modes.

    Used by :class:`~mlgw_bns.modes_model.ModesModel` to align the
    mergers of different higher-order modes in the time domain before
    they are combined into the polarizations. The inputs (intrinsic
    parameters) are min-max scaled to ``[0, 1]`` before being passed to
    the underlying scikit-learn
    :class:`GaussianProcessRegressor`.

    The fitted version of this model is heavy (~3 GB on disk). For
    training and inference, :class:`TimeshiftsNN` (RFF + Ridge) is the
    primary model; this GPR class is kept as a fallback reference.

    Parameters
    ----------
    training_params : np.ndarray, optional
            Training feature matrix, shape ``(n_samples, n_features)``.
    training_timeshifts : np.ndarray, optional
            Training targets, shape ``(n_samples,)``.

    Attributes
    ----------
    regressor : GaussianProcessRegressor
            Underlying scikit-learn GPR.
    scaler : MinMaxScaler
            Scaler used to normalize input parameters to ``[0, 1]``.
    is_fitted : bool
            ``True`` once :meth:`fit` has been called successfully.
    """

    def __init__(
        self,
        training_params: Optional[np.ndarray] = None,
        training_timeshifts: Optional[np.ndarray] = None,
    ):
        self.training_params = training_params
        self.training_timeshifts = training_timeshifts

        self.regressor = GaussianProcessRegressor(
            n_restarts_optimizer=3,
            random_state=3,
        )
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def fit(self) -> "TimeshiftsGPR":
        """Fit the GPR on the stored training data.

        The training parameters are min-max scaled before the GPR is
        trained, so any future call to :meth:`predict` must apply the
        same scaler.

        Returns
        -------
        TimeshiftsGPR
                ``self``, for chained calls.

        Raises
        ------
        ValueError
                If either ``training_params`` or ``training_timeshifts``
                was not provided at construction time.
        """
        if self.training_params is None or self.training_timeshifts is None:
            raise ValueError("Training data not provided.")

        scaled_params = self.scaler.fit_transform(self.training_params)
        self.regressor.fit(scaled_params, self.training_timeshifts)
        self.is_fitted = True

        return self

    def predict(self, params: np.ndarray) -> np.ndarray:
        """Predict time shifts for new input parameters.

        Parameters
        ----------
        params : np.ndarray
                Input feature matrix, shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
                Predicted time shifts, shape ``(n_samples,)`` (or
                ``(n_samples, n_modes)`` depending on how the regressor
                was trained).

        Raises
        ------
        ValueError
                If :meth:`fit` has not yet been called.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")

        scaled_params = self.scaler.transform(params)
        return self.regressor.predict(scaled_params)

    def save_model(self, filename: str) -> None:
        """Persist the entire object to ``filename`` via joblib.

        Parameters
        ----------
        filename : str
                Destination path.
        """
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename: str) -> "TimeshiftsGPR":
        """Load a previously saved :class:`TimeshiftsGPR`.

        Parameters
        ----------
        filename : str
                Path to a file written by :meth:`save_model`.

        Returns
        -------
        TimeshiftsGPR
                Loaded instance.

        Raises
        ------
        ValueError
                If the pickled object is not a :class:`TimeshiftsGPR`.
        """
        model = joblib.load(filename)
        if not isinstance(model, cls):
            raise ValueError("Loaded model is not of the correct type.")
        return model


class TimeshiftsNN:
    """RFF + Ridge surrogate for merger time shifts (primary model).

    Primary drop-in replacement for :class:`TimeshiftsGPR`. Training fits
    a :class:`~sklearn.pipeline.Pipeline` of
    :class:`~sklearn.kernel_approximation.RBFSampler` followed by
    :class:`~sklearn.linear_model.Ridge` on min-max scaled intrinsic
    parameters. Compared to the GPR, the on-disk footprint is orders of
    magnitude smaller with similar accuracy and much faster inference.

    Pre-trained instances can also be constructed by passing a fitted
    ``regressor`` and ``scaler`` directly (e.g. when unpickling or
    wrapping an existing pipeline).

    Parameters
    ----------
    regressor : object, optional
            Fitted scikit-learn regressor or pipeline with ``predict``.
            If provided together with a fitted ``scaler``, the instance
            is ready for inference and :meth:`fit` must not be called.
    scaler : MinMaxScaler, optional
            Fitted input scaler. Defaults to a fresh :class:`MinMaxScaler`
            when training from data.
    training_params : np.ndarray, optional
            Training feature matrix, shape ``(n_samples, n_features)``.
            Required for :meth:`fit` unless ``regressor`` is pre-fitted.
    training_timeshifts : np.ndarray, optional
            Training targets, shape ``(n_samples,)``.
    n_components : int, optional
            Number of random Fourier features. Default 1000.
    gamma : float, optional
            RBF kernel scale for :class:`RBFSampler`. Default 1.0.
    ridge_alpha : float, optional
            L2 strength for :class:`Ridge`. Default ``1e-6``.
    random_state : int, optional
            Random seed for the RFF projection. Default 42.

    Attributes
    ----------
    is_fitted : bool
            ``True`` once the model is ready for :meth:`predict`.
    """

    DEFAULT_N_COMPONENTS = 1000
    DEFAULT_GAMMA = 1.0
    DEFAULT_RIDGE_ALPHA = 1e-6
    DEFAULT_RANDOM_STATE = 42

    def __init__(
        self,
        regressor=None,
        scaler: Optional[MinMaxScaler] = None,
        *,
        training_params: Optional[np.ndarray] = None,
        training_timeshifts: Optional[np.ndarray] = None,
        n_components: int = DEFAULT_N_COMPONENTS,
        gamma: float = DEFAULT_GAMMA,
        ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        self.regressor = regressor
        self.scaler = scaler if scaler is not None else MinMaxScaler()
        self.training_params = training_params
        self.training_timeshifts = training_timeshifts
        self.n_components = n_components
        self.gamma = gamma
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        self.is_fitted = regressor is not None

    @staticmethod
    def make_rff_ridge_pipeline(
        n_components: int = DEFAULT_N_COMPONENTS,
        gamma: float = DEFAULT_GAMMA,
        ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> Pipeline:
        """Build the default RFF + Ridge training pipeline."""
        return Pipeline(
            [
                (
                    "rff",
                    RBFSampler(
                        n_components=n_components,
                        gamma=gamma,
                        random_state=random_state,
                    ),
                ),
                ("ridge", Ridge(alpha=ridge_alpha)),
            ]
        )

    def fit(self) -> "TimeshiftsNN":
        """Fit the RFF + Ridge model on stored training data.

        Input parameters are min-max scaled to ``[0, 1]`` before the
        RFF projection and Ridge regression, matching :class:`TimeshiftsGPR`.

        Returns
        -------
        TimeshiftsNN
                ``self``, for chained calls.

        Raises
        ------
        ValueError
                If either ``training_params`` or ``training_timeshifts``
                was not provided at construction time.
        """
        if self.training_params is None or self.training_timeshifts is None:
            raise ValueError("Training data not provided.")

        scaled_params = self.scaler.fit_transform(self.training_params)
        self.regressor = self.make_rff_ridge_pipeline(
            n_components=self.n_components,
            gamma=self.gamma,
            ridge_alpha=self.ridge_alpha,
            random_state=self.random_state,
        )
        self.regressor.fit(scaled_params, self.training_timeshifts)
        self.is_fitted = True
        return self

    def predict(self, params: np.ndarray) -> np.ndarray:
        """Predict time shifts for new input parameters.

        Parameters
        ----------
        params : np.ndarray
                Input feature matrix, shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
                Predicted time shifts.

        Raises
        ------
        ValueError
                If :meth:`fit` has not yet been called and no pre-fitted
                regressor was supplied at construction time.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")

        scaled_params = self.scaler.transform(params)
        return self.regressor.predict(scaled_params)

    def save_model(self, filename: str) -> None:
        """Persist the entire object to ``filename`` via joblib."""
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename: str) -> "TimeshiftsNN":
        """Load a previously saved :class:`TimeshiftsNN`.

        Raises
        ------
        ValueError
                If the pickled object is not a :class:`TimeshiftsNN`.
        """
        model = joblib.load(filename)
        if not isinstance(model, cls):
            raise ValueError("Loaded model is not of the correct type.")
        return model


def load_timeshifts_predictor(
    nn_path: str,
    gpr_path: str,
) -> Union[TimeshiftsNN, TimeshiftsGPR]:
    """Load the time-shift predictor, preferring RFF+Ridge over GPR.

    Parameters
    ----------
    nn_path : str
            Path to a :class:`TimeshiftsNN` checkpoint (RFF + Ridge).
    gpr_path : str
            Path to a :class:`TimeshiftsGPR` checkpoint used if the NN
            file cannot be loaded.

    Returns
    -------
    TimeshiftsNN or TimeshiftsGPR
            Loaded predictor.

    Raises
    ------
    ValueError
            If neither checkpoint can be loaded.
    """
    try:
        return TimeshiftsNN.load_model(nn_path)
    except Exception as nn_err:
        try:
            return TimeshiftsGPR.load_model(gpr_path)
        except Exception as gpr_err:
            raise ValueError(
                f"Could not load TimeshiftsNN from {nn_path!r} ({nn_err}) "
                f"or TimeshiftsGPR from {gpr_path!r} ({gpr_err})."
            ) from gpr_err


def retrieve_best_trials_list() -> "list[optuna.trial.FrozenTrial]":
    """Return the pretrained Pareto front of best hyperparameter trials.

    The pickled trial list is shipped with the package at the path
    given by :data:`TRIALS_FILE`. It can be regenerated with
    :meth:`mlgw_bns.hyperparameter_optimization.HyperparameterOptimization.save_best_trials_to_file`
    after running an optimization job (the shipped file is the result of
    roughly 30 hours of optimization on a laptop).

    Returns
    -------
    list[optuna.trial.FrozenTrial]
            Trials lying on the Pareto front of the bi-objective
            (accuracy vs. ``n_train``) optimization.
    """
    stream = pkg_resources.resource_stream(__name__, TRIALS_FILE)
    return joblib.load(stream)


def best_trial_under_n(
    best_trials: "list[optuna.trial.FrozenTrial]",
    training_number: int,
) -> Hyperparameters:
    """Pick the most accurate trial that used no more than ``training_number`` waveforms.

    Convenience helper used by :class:`~mlgw_bns.hyperparameter_optimization.HyperparameterOptimization`
    to extract a concrete :class:`Hyperparameters` instance from the
    Pareto front, given an upper bound on the allowed training-set size.

    Parameters
    ----------
    best_trials : list[optuna.trial.FrozenTrial]
            Pareto front of completed trials; typically the result of
            :func:`retrieve_best_trials_list`.
    training_number : int
            Maximum allowed value of ``n_train``. The returned trial
            satisfies ``trial.params["n_train"] <= training_number``.

    Returns
    -------
    Hyperparameters
            Hyperparameters from the most-accurate qualifying trial.
            Its :attr:`n_train` is overwritten with ``training_number``
            so that the caller trains on exactly the requested number
            of waveforms.
    """
    accuracy = lambda trial: trial.values[0]

    # Sort by accuracy (lower is better) among the trials that fit the
    # training-size budget, and pick the best one.
    best_trial = sorted(
        [trial for trial in best_trials if trial.params["n_train"] <= training_number],
        key=accuracy,
    )[0]

    hyper = Hyperparameters.from_frozen_trial(best_trial)
    hyper.n_train = training_number
    return hyper
