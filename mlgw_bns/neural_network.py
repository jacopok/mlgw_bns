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

* :func:`load_kernel_ridge_defaults` and :func:`save_kernel_ridge_default`
  --- the per-mode ``(kernel_gamma, kernel_alpha)`` counterpart for
  :class:`KernelRidgeNetwork`, written by
  :meth:`~mlgw_bns.hyperparameter_optimization.HyperparameterOptimization.save_best_as_default`.

The optional PyTorch backend that used to live in this module has been
removed; only the scikit-learn backend is now supported.
"""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Optional, Union

import joblib  # type: ignore
import numpy as np
import scipy.linalg  # type: ignore
from importlib.resources import files
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
from sklearn.kernel_approximation import RBFSampler  # type: ignore
from sklearn.kernel_ridge import KernelRidge  # type: ignore
from sklearn.linear_model import Ridge  # type: ignore
from sklearn.neural_network import MLPRegressor  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore

if TYPE_CHECKING:
    import optuna

    from .pn_modes import Mode

#: Location, relative to the package, of the joblib-pickled Pareto front
#: produced by the hyperparameter-optimization pipeline.
TRIALS_FILE = "data/best_trials.pkl"

#: Where the per-mode :class:`KernelRidgeNetwork` defaults, tuned by
#: :class:`~mlgw_bns.hyperparameter_optimization.HyperparameterOptimization`,
#: are read from and written to. Kept as plain JSON, rather than joblib
#: like :data:`TRIALS_FILE`, since it is meant to be hand-edited and
#: diffed as easily as the code that produces it.
KERNEL_DEFAULTS_PATH = Path(__file__).parent / "data" / "kernel_ridge_defaults.json"


def mode_key(mode: "Optional[Mode]") -> str:
    """Turn a :class:`~mlgw_bns.pn_modes.Mode` into the string key used to
    index the per-mode kernel-ridge defaults. ``None`` is the (2,2) mode,
    the convention used throughout :class:`~mlgw_bns.mode_model.ModeModel`.
    """
    l, m = (2, 2) if mode is None else mode
    return f"{l}{m}"


def load_kernel_ridge_defaults() -> "dict[str, tuple[float, float]]":
    """Read the per-mode ``(kernel_gamma, kernel_alpha)`` defaults.

    Returns an empty mapping if the file does not exist yet, i.e. before
    any mode has been optimized.
    """
    try:
        with open(KERNEL_DEFAULTS_PATH) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {}
    return {key: (v["kernel_gamma"], v["kernel_alpha"]) for key, v in raw.items()}


def save_kernel_ridge_default(mode: "Optional[Mode]", kernel_gamma: float, kernel_alpha: float) -> None:
    """Persist ``(kernel_gamma, kernel_alpha)`` as the default for ``mode``,
    merging into whatever is already on disk for the other modes.
    """
    try:
        with open(KERNEL_DEFAULTS_PATH) as f:
            raw = json.load(f)
    except FileNotFoundError:
        raw = {}

    raw[mode_key(mode)] = {"kernel_gamma": kernel_gamma, "kernel_alpha": kernel_alpha}

    KERNEL_DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(KERNEL_DEFAULTS_PATH, "w") as f:
        json.dump(raw, f, indent=2, sort_keys=True)


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
    legacy_batch_size_clip : bool, optional
            Reproduce the mini-batch clipping of models packaged before
            this flag existed, which clipped ``batch_size`` to the number
            of input features (five) rather than the number of training
            samples. Defaults to ``False``. See :meth:`SklearnNetwork.fit`.
    kernel_gamma : float, optional
            Width of the RBF kernel used by :class:`KernelRidgeNetwork`,
            on standardized inputs. Ignored by :class:`SklearnNetwork`.
    kernel_alpha : float, optional
            Ridge regularization used by :class:`KernelRidgeNetwork`.
            Small values give the best median accuracy; larger ones trade
            that against the worst case. Ignored by :class:`SklearnNetwork`.
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

    #: Kept so that the packaged models, all of which were trained with the
    #: mini-batch clipped to the feature count, can be reproduced exactly.
    legacy_batch_size_clip: bool = field(default=False)

    #: Defaults for :class:`KernelRidgeNetwork`, from a scan over gamma and
    #: alpha on the (2,2) mode with 8192 training waveforms. Unused by the
    #: network backend, which is why they carry defaults rather than being
    #: required like the rest.
    kernel_gamma: float = field(default=0.1)
    kernel_alpha: float = field(default=1e-10)

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

        This is the inverse of :meth:`from_trial` (or, for a trial
        produced by :meth:`from_trial_kernel_ridge`, of that instead) for
        already-completed trials, used when reading the pretrained
        Pareto front shipped with the package.

        Parameters
        ----------
        frozen_trial : optuna.trial.FrozenTrial
                Completed trial whose params dictionary contains the
                values produced by :meth:`from_trial` or
                :meth:`from_trial_kernel_ridge`.

        Returns
        -------
        Hyperparameters
                Hyperparameter set corresponding to the trial.
        """
        params = dict(frozen_trial.params)

        if "n_layers" not in params:
            # A trial from from_trial_kernel_ridge: only n_train,
            # kernel_gamma and kernel_alpha were ever sampled.
            return cls.default_kernel_ridge(
                n_train=params["n_train"],
                kernel_gamma=params["kernel_gamma"],
                kernel_alpha=params["kernel_alpha"],
            )

        n_layers = params.pop("n_layers")
        layers = [params.pop(f"size_layer_{i}") for i in range(n_layers)]
        params["hidden_layer_sizes"] = tuple(layers)

        return cls(**params)

    @classmethod
    def from_trial_kernel_ridge(
        cls, trial: "optuna.Trial", n_train: int
    ) -> "Hyperparameters":
        """Sample a :class:`Hyperparameters` for :class:`KernelRidgeNetwork`
        from an :class:`optuna.Trial`.

        Unlike :meth:`from_trial`, this only samples the two parameters
        :class:`KernelRidgeNetwork` actually reads, :attr:`kernel_gamma`
        and :attr:`kernel_alpha`; ``n_train`` is fixed rather than
        sampled, since the accuracy comparison across trials is only
        fair at a fixed training-set size, and the MLP-specific fields
        are filled with placeholders :class:`KernelRidgeNetwork` ignores.

        Parameters
        ----------
        trial : optuna.Trial
                Trial object used to draw ``kernel_gamma`` and ``kernel_alpha``.
        n_train : int
                Fixed number of training waveforms.

        Returns
        -------
        Hyperparameters
        """
        trial.suggest_int("n_train", n_train, n_train)

        return cls.default_kernel_ridge(
            n_train=n_train,
            kernel_gamma=trial.suggest_float("kernel_gamma", 1e-3, 30.0, log=True),
            kernel_alpha=trial.suggest_float("kernel_alpha", 1e-14, 1e-2, log=True),
        )

    @classmethod
    def default_kernel_ridge(
        cls,
        n_train: int,
        mode: "Optional[Mode]" = None,
        kernel_gamma: Optional[float] = None,
        kernel_alpha: Optional[float] = None,
    ) -> "Hyperparameters":
        """Build a :class:`Hyperparameters` for :class:`KernelRidgeNetwork`.

        The MLP-specific fields are filled with placeholders, since
        :class:`KernelRidgeNetwork` never reads them. If ``kernel_gamma``
        or ``kernel_alpha`` are not given explicitly, they are looked up
        in :func:`load_kernel_ridge_defaults` for ``mode``, falling back
        to the class-level defaults if that mode has not been optimized
        yet.

        Parameters
        ----------
        n_train : int
                Number of training waveforms.
        mode : Mode, optional
                Mode whose tuned defaults to use, when ``kernel_gamma``
                or ``kernel_alpha`` are not given explicitly. Defaults to
                the (2,2) mode.
        kernel_gamma : float, optional
        kernel_alpha : float, optional

        Returns
        -------
        Hyperparameters
        """
        if kernel_gamma is None or kernel_alpha is None:
            tuned = load_kernel_ridge_defaults().get(mode_key(mode))
            if tuned is not None:
                kernel_gamma = kernel_gamma if kernel_gamma is not None else tuned[0]
                kernel_alpha = kernel_alpha if kernel_alpha is not None else tuned[1]

        return cls(
            pc_exponent=1.0,
            n_train=n_train,
            hidden_layer_sizes=(1,),
            activation="relu",
            alpha=0.0,
            batch_size=1,
            learning_rate_init=0.0,
            tol=0.0,
            validation_fraction=0.1,
            n_iter_no_change=1,
            kernel_gamma=kernel_gamma if kernel_gamma is not None else cls.kernel_gamma,
            kernel_alpha=kernel_alpha if kernel_alpha is not None else cls.kernel_alpha,
        )

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

        The mini-batch size is temporarily clipped to the number of
        training samples, to avoid scikit-learn's "batch_size larger than
        data" warning, and restored once training completes.

        Every model packaged before this was written was trained with
        :attr:`Hyperparameters.legacy_batch_size_clip` behaviour, in which
        the clip used ``x_data.shape[1]`` --- the number of *features*,
        which is five --- so the configured ``batch_size`` never survived
        at any training-set size. That is preserved here as an option, for
        reproducing those models exactly; it is not the default, because
        it makes any tuning of ``batch_size`` meaningless.
        """
        self.param_scaler = StandardScaler().fit(x_data)

        old_batch_size = self.nn.batch_size
        # A dataclass default is a class attribute, so this resolves even
        # on a `Hyperparameters` unpickled from before the field existed
        # --- such an instance picks up the new default, i.e. the repaired
        # clip. That is harmless for the packaged models: the flag is only
        # read here, and loading one of them to predict never fits. It
        # only means that *re-fitting* with old hyperparameters uses the
        # corrected mini-batch, which is what one would want anyway.
        clip_to = (
            x_data.shape[1] if self.hyper.legacy_batch_size_clip else x_data.shape[0]
        )
        self.nn.batch_size = min(self.nn.batch_size, clip_to)

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


class KernelRidgeNetwork(NeuralNetwork):
    r"""Kernel ridge regression from parameters to component coefficients.

    A drop-in alternative to :class:`SklearnNetwork`, selected by passing
    it as ``nn_kind`` to :class:`~mlgw_bns.mode_model.ModeModel`. On the
    (2,2) mode with 8192 training waveforms it reaches a median mismatch
    of :math:`3.4 \times 10^{-9}` against the network's
    :math:`7 \times 10^{-6}`, and fits in twenty seconds rather than ten
    minutes.

    Two things make the difference. The first is that this map is smooth
    and low-dimensional --- five parameters to a few tens of coefficients
    --- which is the regime kernel methods are good at, and there is no
    stochastic optimizer to converge. The second is subtler: the network
    minimizes an unweighted mean squared error over targets that have
    been divided by :math:`\max_j |x_{ji}|` per component, which weights
    component :math:`i`'s contribution to the *residual* by
    :math:`s_i^{-2}`, running some nine orders of magnitude in favour of
    the least important component. Kernel ridge solves
    :math:`(K + \alpha I)^{-1} y` separately for each output, which is
    equivariant under rescaling each output, so that weighting --- and
    hence :attr:`Hyperparameters.pc_exponent` --- cannot affect it at all.

    Inputs are standardized, as for the network. Outputs are standardized
    too, which is what makes a single :attr:`Hyperparameters.kernel_alpha`
    meaningful across components that span ten orders of magnitude.

    The cost of a prediction grows with the training set, since it
    evaluates one kernel per training point: about 260 microseconds per
    waveform per mode at 8192 training waveforms, against 35 for the
    network. Set against the ~25 ms a full four-mode waveform takes end
    to end, that is a few per cent.

    Parameters
    ----------
    hyper : Hyperparameters
            Only :attr:`~Hyperparameters.kernel_gamma` and
            :attr:`~Hyperparameters.kernel_alpha` are read; the network
            attributes are ignored, but the object is kept whole so that
            the rest of the codebase can treat the two backends alike.
    regressor : KernelRidge, optional
            Pre-built regressor to wrap.
    param_scaler : StandardScaler, optional
            Pre-fitted scaler for the inputs.
    target_scaler : StandardScaler, optional
            Pre-fitted scaler for the outputs.
    """

    def __init__(
        self,
        hyper: Hyperparameters,
        regressor: "Optional[KernelRidge]" = None,
        param_scaler: Optional[StandardScaler] = None,
        target_scaler: Optional[StandardScaler] = None,
    ):
        super().__init__(hyper=hyper)
        if regressor is None:
            regressor = KernelRidge(
                kernel="rbf",
                gamma=getattr(hyper, "kernel_gamma", 0.1),
                alpha=getattr(hyper, "kernel_alpha", 1e-10),
            )
        self.regressor: KernelRidge = regressor
        if param_scaler is not None:
            self.param_scaler: StandardScaler = param_scaler
        if target_scaler is not None:
            self.target_scaler: StandardScaler = target_scaler

    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """Fit the two scalers and solve the kernel system.

        For small ``kernel_gamma`` the RBF Gram matrix is close to
        low-rank --- broad kernels make training points look alike to
        each other --- so its Cholesky solve routinely reports a
        singular matrix regardless of ``kernel_alpha``. scikit-learn
        already falls back to an exact least-squares solve (slower, but
        not wrong) and warns every time it does; that warning is
        expected here rather than a sign of a bad fit, so it is
        silenced.
        """
        self.param_scaler = StandardScaler().fit(x_data)
        self.target_scaler = StandardScaler().fit(y_data)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Singular matrix in solving dual problem.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Ill-conditioned matrix.*",
                category=scipy.linalg.LinAlgWarning,
            )
            self.regressor.fit(
                self.param_scaler.transform(x_data),
                self.target_scaler.transform(y_data),
            )

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        scaled_x = self.param_scaler.transform(x_data)
        return self.target_scaler.inverse_transform(self.regressor.predict(scaled_x))

    def save(self, filename: str) -> None:
        """Pickle ``(hyper, regressor, param_scaler, target_scaler)`` via joblib."""
        joblib.dump(
            (self.hyper, self.regressor, self.param_scaler, self.target_scaler),
            filename,
        )

    @classmethod
    def from_file(cls, filename: Union[IO[bytes], str]) -> "KernelRidgeNetwork":
        """Inverse of :meth:`save`. The tuple is unpacked into the constructor."""
        return cls(*joblib.load(filename))


class TimeshiftsGPR:
    """Gaussian-Process regressor for merger time shifts between modes.

    Used by :class:`~mlgw_bns.model.Model` to align the
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


class ModePhasesNN:
    """RFF + Ridge surrogate for per-mode reference phases.

    Analogous to :class:`TimeshiftsNN`, but a *multi-output* regressor:
    for a given set of intrinsic parameters it predicts the vector
    ``[phi_lm[f0] for lm in modes]`` --- the phase of each spherical
    harmonic mode at the lowest frequency node of the training grid.

    The relative phases between modes are hard for the per-mode PCA +
    network to learn from the residuals directly; pulling the node-0
    per-mode phase constant out into this dedicated regressor gives the
    per-mode networks cleaner training data while still allowing the
    mode phases (and hence their relative phases) to be reconstructed at
    predict time.

    Same construction contract as :class:`TimeshiftsNN`: pass
    ``training_params`` and ``training_mode_phases`` to fit from data, or
    a fitted ``regressor`` and ``scaler`` to wrap an existing pipeline.

    Parameters
    ----------
    regressor : object, optional
            Fitted scikit-learn regressor or pipeline with ``predict``.
    scaler : MinMaxScaler, optional
            Fitted input scaler. Defaults to a fresh :class:`MinMaxScaler`.
    modes : list[tuple[int, int]], optional
            The ``(l, m)`` modes, in the column order of the targets.
    training_params : np.ndarray, optional
            Training feature matrix, shape ``(n_samples, n_features)``.
    training_mode_phases : np.ndarray, optional
            Training targets, shape ``(n_samples, n_modes)``.
    n_components, gamma, ridge_alpha, random_state
            As in :class:`TimeshiftsNN`.

    Attributes
    ----------
    is_fitted : bool
            ``True`` once the model is ready for :meth:`predict`.
    """

    DEFAULT_N_COMPONENTS = TimeshiftsNN.DEFAULT_N_COMPONENTS
    DEFAULT_GAMMA = TimeshiftsNN.DEFAULT_GAMMA
    DEFAULT_RIDGE_ALPHA = TimeshiftsNN.DEFAULT_RIDGE_ALPHA
    DEFAULT_RANDOM_STATE = TimeshiftsNN.DEFAULT_RANDOM_STATE

    def __init__(
        self,
        regressor=None,
        scaler: Optional[MinMaxScaler] = None,
        *,
        modes: Optional[list] = None,
        training_params: Optional[np.ndarray] = None,
        training_mode_phases: Optional[np.ndarray] = None,
        f0_natural: Optional[float] = None,
        n_components: int = DEFAULT_N_COMPONENTS,
        gamma: float = DEFAULT_GAMMA,
        ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        self.regressor = regressor
        self.scaler = scaler if scaler is not None else MinMaxScaler()
        self.modes = modes
        self.training_params = training_params
        self.training_mode_phases = training_mode_phases
        #: Frequency (natural units) at which the reference phase is
        #: anchored. When set, the huge stationary-phase backbone
        #: ``a * M_lm + b`` is subtracted analytically per mode and only
        #: the smooth leftover is regressed. ``None`` -> plain regressor
        #: on the raw targets (legacy behaviour / old pickles).
        self.f0_natural = f0_natural
        #: ``(l, m) -> (a, b, f0_natural)`` calibration of the analytic
        #: backbone, populated by :meth:`fit` when ``f0_natural`` is set.
        self.analytic_coeffs: dict = {}
        self.n_components = n_components
        self.gamma = gamma
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        self.is_fitted = regressor is not None

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__.setdefault("f0_natural", None)
        self.__dict__.setdefault("analytic_coeffs", {})

    def _analytic_prediction(self, params: np.ndarray) -> np.ndarray:
        """``a * M_lm(params) + b`` per mode, shape ``(n_samples, n_modes)``."""
        from .pn_modes import Mode, reference_phase_backbone

        params = np.asarray(params, dtype=float)
        columns = []
        for lm in self.modes:
            a, b, f0 = self.analytic_coeffs[lm]
            m = reference_phase_backbone(params, f0, Mode(*lm))
            columns.append(a * m + b)
        return np.stack(columns, axis=1)

    def fit(self) -> "ModePhasesNN":
        """Fit the RFF + Ridge model on stored training data.

        When ``f0_natural`` was given, each mode's target is first
        reduced by an analytically-calibrated stationary-phase backbone
        (``a * M_lm + b``, see
        :func:`~mlgw_bns.pn_modes.reference_phase_backbone`) and the
        regressor only learns the smooth ``O(10-100 rad)`` leftover.

        Raises
        ------
        ValueError
                If either ``training_params`` or ``training_mode_phases``
                was not provided at construction time.
        """
        if self.training_params is None or self.training_mode_phases is None:
            raise ValueError("Training data not provided.")

        targets = np.asarray(self.training_mode_phases, dtype=float)

        if self.f0_natural is not None:
            from .pn_modes import Mode, reference_phase_backbone

            params = np.asarray(self.training_params, dtype=float)
            self.analytic_coeffs = {}
            leftover = np.empty_like(targets)
            for j, lm in enumerate(self.modes):
                m = reference_phase_backbone(params, self.f0_natural, Mode(*lm))
                a, b = np.polyfit(m, targets[:, j], 1)
                self.analytic_coeffs[lm] = (float(a), float(b), float(self.f0_natural))
                leftover[:, j] = targets[:, j] - (a * m + b)
            targets = leftover

        scaled_params = self.scaler.fit_transform(self.training_params)
        self.regressor = TimeshiftsNN.make_rff_ridge_pipeline(
            n_components=self.n_components,
            gamma=self.gamma,
            ridge_alpha=self.ridge_alpha,
            random_state=self.random_state,
        )
        self.regressor.fit(scaled_params, targets)
        self.is_fitted = True
        return self

    def predict(self, params: np.ndarray) -> np.ndarray:
        """Predict per-mode reference phases, shape ``(n_samples, n_modes)``."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")

        scaled_params = self.scaler.transform(params)
        prediction = self.regressor.predict(scaled_params)
        if self.analytic_coeffs:
            prediction = prediction + self._analytic_prediction(params)
        return prediction

    def save_model(self, filename: str) -> None:
        """Persist the entire object to ``filename`` via joblib."""
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename: str) -> "ModePhasesNN":
        """Load a previously saved :class:`ModePhasesNN`."""
        model = joblib.load(filename)
        if not isinstance(model, cls):
            raise ValueError("Loaded model is not of the correct type.")
        return model


def load_mode_phases_predictor_from_file(
    filename: Union[IO[bytes], str]
) -> ModePhasesNN:
    """Load a :class:`ModePhasesNN` checkpoint."""
    model = joblib.load(filename)
    if not isinstance(model, ModePhasesNN):
        raise ValueError("Loaded object is not a ModePhasesNN.")
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


def load_timeshifts_predictor_from_file(
    filename: Union[IO[bytes], str]
) -> Union[TimeshiftsNN, TimeshiftsGPR]:
    """Load a single time-shifts predictor checkpoint of either kind.

    Unlike :func:`load_timeshifts_predictor`, this does not try two
    separate paths: it loads whichever of :class:`TimeshiftsNN` or
    :class:`TimeshiftsGPR` was pickled at ``filename``.

    Parameters
    ----------
    filename : IO[bytes] or str
            Path (or open stream) to a checkpoint written by either
            :meth:`TimeshiftsNN.save_model` or
            :meth:`TimeshiftsGPR.save_model`.

    Returns
    -------
    TimeshiftsNN or TimeshiftsGPR
            Loaded predictor.

    Raises
    ------
    ValueError
            If the loaded object is not a :class:`TimeshiftsNN` or
            :class:`TimeshiftsGPR`.
    """
    model = joblib.load(filename)
    if not isinstance(model, (TimeshiftsNN, TimeshiftsGPR)):
        raise ValueError("Loaded object is not a TimeshiftsNN or TimeshiftsGPR.")
    return model


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

    stream = files(__name__).joinpath(TRIALS_FILE).open("rb")
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
