from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING, Optional, Union

import joblib  # type: ignore
import numpy as np
import pkg_resources
from sklearn.neural_network import MLPRegressor  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

if TYPE_CHECKING:
    import optuna
    import torch  # type: ignore
    import torch.utils.data as Data  # type: ignore
    from torch.autograd import Variable  # type: ignore

TRIALS_FILE = "data/best_trials.pkl"


@dataclass
class Hyperparameters:
    r"""Dataclass containing the parameters which are passed to
    the neural network for training, as well as a few more
    (:attr:`pc_exponent`) and (:attr:`n_train`).

    Parameters
    ----------
    pc_exponent: float
            Exponent to be used in the normalization of the
            principal components: the network learns to reconstruct
            :math:`x_i \lambda_i^\alpha`, where
            :math:`x_i` are the principal-component
            representation of a waveform, while
            :math:`\lambda_i` are the eigenvalues of the PCA
            and finally :math:`\alpha` is this parameter.
    n_train: float
            Number of waveforms to use in the training.
    hidden_layer_sizes: tuple[int, ...]
            Sizes of the layers in the neural network.
            For more details, refer to the documentation
            of the :class:`MLPRegressor`.
    activation: str
            Activation function.
            For more details, refer to the documentation
            of the :class:`MLPRegressor`.
    alpha: float
            Regularization parameter.
            For more details, refer to the documentation
            of the :class:`MLPRegressor`.
    batch_size: int
            For more details, refer to the documentation
            of the :class:`MLPRegressor`.
    learning_rate_init: float
            For more details, refer to the documentation
            of the :class:`MLPRegressor`.
    tol: float
            For more details, refer to the documentation
            of the :class:`MLPRegressor`.
    validation_fraction: float
            For more details, refer to the documentation
            of the :class:`MLPRegressor`.
    n_iter_no_change: float
            For more details, refer to the documentation
            of the :class:`MLPRegressor`.
    """

    # controls how much weight is give to higher principal components
    pc_exponent: float

    # number of training data points to use
    n_train: int

    # parameters for the sklearn neural network
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
    def nn_params(self) -> dict[str, Union[int, float, str, bool, tuple[int, ...]]]:
        """Return a dictionary which can be readily unpacked
        and used to initialize a :class:`MLPRegressor`.
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
    def from_trial(cls, trial: "optuna.Trial", n_train_max: int):
        """Generate the hyperparameter set starting from an
        :class:`optuna.Trial`.

        Parameters
        ----------
        trial : optuna.Trial
                Used to generate the parameters.
        n_train_max : int
                Upper bound for the attribute :attr:`n_train`.
        """

        n_layers = trial.suggest_int("n_layers", 1, 4)

        layers = tuple(
            trial.suggest_int(f"size_layer_{i}", 10, 200) for i in range(n_layers)
        )

        return cls(
            hidden_layer_sizes=layers,
            activation=str(
                trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])
            ),
            alpha=trial.suggest_loguniform("alpha", 1e-6, 1e-1),  # Default: 1e-4
            batch_size=trial.suggest_int("batch_size", 100, 200),  # Default: 200
            learning_rate_init=trial.suggest_loguniform(
                "learning_rate_init", 2e-4, 5e-2
            ),  # Default: 1e-3
            tol=trial.suggest_loguniform("tol", 1e-15, 1e-7),  # default: 1e-4
            validation_fraction=trial.suggest_uniform("validation_fraction", 0.05, 0.2),
            n_iter_no_change=trial.suggest_int(
                "n_iter_no_change", 40, 100, log=True
            ),  # default: 10
            pc_exponent=trial.suggest_loguniform("pc_exponent", 1e-3, 1),
            n_train=trial.suggest_int("n_train", 50, n_train_max),
        )

    @classmethod
    def from_frozen_trial(cls, frozen_trial: "optuna.trial.FrozenTrial"):

        params = frozen_trial.params
        n_layers = params.pop("n_layers")

        layers = [params.pop(f"size_layer_{i}") for i in range(n_layers)]
        params["hidden_layer_sizes"] = tuple(layers)

        return cls(**params)

    @classmethod
    def default(cls, training_waveform_number: Optional[int] = None):

        try:
            if training_waveform_number is not None:
                best_trials = retrieve_best_trials_list()
                return best_trial_under_n(best_trials, training_waveform_number)
        except (FileNotFoundError, IndexError):
            pass

        return cls(
            hidden_layer_sizes=(50, 50),
            activation="relu",
            alpha=1e-4,
            batch_size=200,
            learning_rate_init=1e-3,
            tol=1e-9,
            n_iter_no_change=50,
            validation_fraction=0.1,
            pc_exponent=0.2,
            n_train=200,
        )


class NeuralNetwork(ABC):
    """Abstract class for a wrapper around
    a generic neural network.
    """

    def __init__(self, hyper: Hyperparameters):
        self.hyper = hyper

    @abstractmethod
    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """Fit the network to the data.

        Parameters
        ----------
        x_data : np.ndarray
        y_data : np.ndarray
        """

    @abstractmethod
    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """Make a prediction for a new set of data.

        Parameters
        ----------
        x_data : np.ndarray

        Returns
        -------
        np.ndarray
        """

    @abstractmethod
    def save(self, filename: str) -> None:
        """Save self to file.

        Parameters
        ----------
        filename : str
            File to save to.
        """

    @classmethod
    @abstractmethod
    def from_file(self, filename: Union[IO[bytes], str]):
        """Load object from file.

        Parameters
        ----------
        filename : str
            File to read

        Returns
        -------
        NeuralNetwork
        """


class SklearnNetwork(NeuralNetwork):
    """Wrapper for a MLPRegressor, the regressor provided by
    the library `scikit-learn <https://scikit-learn.org/stable/>`_.
    """

    def __init__(
        self,
        hyper: Hyperparameters,
        nn: Optional[MLPRegressor] = None,
        param_scaler: Optional[StandardScaler] = None,
    ):
        super().__init__(hyper=hyper)
        self.nn = MLPRegressor(**hyper.nn_params) if nn is None else nn
        if param_scaler is not None:
            self.param_scaler: StandardScaler = param_scaler

    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        self.param_scaler = StandardScaler().fit(x_data)

        old_batch_size = self.nn.batch_size
        self.nn.batch_size = min(self.nn.batch_size, x_data.shape[1])

        scaled_x = self.param_scaler.transform(x_data)
        self.nn.fit(scaled_x, y_data)

        self.nn.batch_size = old_batch_size

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        scaled_x = self.param_scaler.transform(x_data)
        return self.nn.predict(scaled_x)

    def save(self, filename: str):
        joblib.dump((self.hyper, self.nn, self.param_scaler), filename)

    @classmethod
    def from_file(cls, filename: Union[IO[bytes], str]):
        return cls(*joblib.load(filename))


class TorchNetwork(NeuralNetwork):
    """Wrapper for a network using pytorch."""

    def __init__(
        self,
        hyper: Hyperparameters,
        nn: Optional["torch.nn.modules.container.Sequential"] = None,
    ) -> None:
        import torch  # type : ignore
        import torch.utils.data as Data  # type : ignore
        from torch.autograd import Variable  # type : ignore

        super().__init__(hyper=hyper)

        self.nn = nn if nn is not None else self.make_nn()

    def make_nn(
        self, size_in: int = 5, size_out: int = 30
    ) -> "torch.nn.modules.container.Sequential":
        activations = {
            "logistic": torch.nn.LogSigmoid(),
            "tanh": torch.nn.Tanh(),
            "relu": torch.nn.ReLU(),
        }
        activation = activations[self.hyper.activation]

        sizes_in = (size_in,) + self.hyper.hidden_layer_sizes
        sizes_out = self.hyper.hidden_layer_sizes + (size_out,)

        return torch.nn.Sequential(
            *(
                layer
                for s_in, s_out in zip(sizes_in, sizes_out)
                for layer in (activation, torch.nn.Linear(s_in, s_out))
            )
        )

    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        # TODO: add validation
        # from pytorch ignite maybe?
        # https://pytorch.org/ignite/generated/ignite.handlers.early_stopping.EarlyStopping.html

        x, y = Variable(x_data), Variable(y_data)

        torch_dataset = Data.TensorDataset(x, y)

        loader_iterable = iter(
            Data.DataLoader(
                dataset=torch_dataset, batch_size=self.hyper.batch_size, shuffle=True
            )
        )

        optimizer = torch.optim.Adam(
            self.nn.parameters(),
            lr=self.hyper.learning_rate_init,
            weight_decay=self.hyper.alpha,
        )

        loss_func = torch.nn.MSELoss()

        n_iters_no_improvement = 0
        previous_loss = np.inf

        while n_iters_no_improvement < self.hyper.n_iter_no_change:
            batch_x, batch_y = next(loader_iterable)

            b_x = Variable(batch_x)
            b_y = Variable(batch_y)

            prediction = self.nn(b_x)

            loss = loss_func(prediction, b_y)

            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            loss_improvement = previous_loss - loss.item()
            if abs(loss_improvement) > self.hyper.tol:
                n_iters_no_improvement = 0
            else:
                n_iters_no_improvement += 1

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        x = Variable(x_data)

        return self.nn(x).numpy()

    def save(self, filename: str):
        joblib.dump((self.hyper, self.nn), filename)

    @classmethod
    def from_file(cls, filename: Union[IO[bytes], str]):
        return cls(*joblib.load(filename))


def retrieve_best_trials_list() -> "list[optuna.trial.FrozenTrial]":
    """Read the list of best trials which is provided
    with the package.

    This list's location can be found at the location
    defined by ``TRIALS_FILE``;
    if one wishes to modify it, a new one can be generated
    with the method :meth:`save_best_trials_to_file` of the
    class :class:`mlgw_bns.hyperparameter_optimization.HyperparameterOptimization`
    after an optimization job has been run.

    The current trials provided are the result of about 30 hours
    of optimization on a laptop.

    Returns
    -------
    list[optuna.trial.FrozenTrial]
        List of the trials in the Pareto front of an optimization.
    """

    stream = pkg_resources.resource_stream(__name__, TRIALS_FILE)
    return joblib.load(stream)


def best_trial_under_n(
    best_trials: "list[optuna.trial.FrozenTrial]", training_number: int
) -> Hyperparameters:
    """Utility function to retrieve
    a set of hyperparameters starting from a list of optimization trials.
    The best trial in terms of accuracy is returned.

    Parameters
    ----------
    best_trials : list[optuna.trial.FrozenTrial]
        List of trials in the Pareto front of an optimization run.
        A default value for such a list is the one provided by
        :func:`retrieve_best_trials_list`.
    training_number : int
        Return the best trial obtained while using less than this number
        of training waveforms.

    Returns
    -------
    Hyperparameters
        Hyperparameters corresponding to the best trial found.
    """

    accuracy = lambda trial: trial.values[0]

    # take the most accurate trial
    # which used less training data than the given
    # training number
    best_trial = sorted(
        [trial for trial in best_trials if trial.params["n_train"] <= training_number],
        key=accuracy,
    )[0]

    hyper = Hyperparameters.from_frozen_trial(best_trial)
    hyper.n_train = training_number
    return hyper
