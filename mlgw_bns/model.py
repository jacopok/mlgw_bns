import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar, Optional, Union

import h5py
import joblib  # type: ignore
import numpy as np
import optuna
import pkg_resources
from numba import njit  # type: ignore
from sklearn.neural_network import MLPRegressor  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from .data_management import (
    DownsamplingIndices,
    FDWaveforms,
    PrincipalComponentData,
    Residuals,
    SavableData,
)
from .dataset_generation import (
    Dataset,
    ParameterSet,
    TEOBResumSGenerator,
    WaveformGenerator,
)
from .downsampling_interpolation import DownsamplingTraining, GreedyDownsamplingTraining
from .principal_component_analysis import (
    PrincipalComponentAnalysisModel,
    PrincipalComponentTraining,
)

TRIALS_FILE = "data/best_trials.pkl"


@dataclass
class Hyperparameters:
    r"""Dataclass containing the parameters which are passed to
    the neural network for training, as well as a few more.

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

    max_iter: ClassVar[int] = 2000

    group_name: ClassVar[str] = "hyperparameters"

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
            "batch_size": self.batch_size,
            "learning_rate_init": self.learning_rate_init,
            "tol": self.tol,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "early_stopping": True,
            "shuffle": True,
        }

    @classmethod
    def from_trial(cls, trial: optuna.Trial, n_train_max: int):
        """Generate the hyperparameter set starting from an
        :class:`optuna.Trial`.

        Parameters
        ----------
        trial : optuna.Trial
                Used to generate the parameters.
        n_train_max : int
                Upper bound for the attribute :attr:`n_train`.
        """

        n_layers = trial.suggest_int("n_layers", 2, 4)

        layers = tuple(
            trial.suggest_int(f"size_layer_{i}", 10, 100) for i in range(n_layers)
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
            n_train=trial.suggest_int("n_train", 200, n_train_max),
        )

    @classmethod
    def from_frozen_trial(cls, frozen_trial: optuna.trial.FrozenTrial):

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
        except FileNotFoundError:
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


class Model:
    """``mlgw_bns`` model.
    This class incorporates all the functionality required to
    compute the downsampling indices, train a PCA model,
    train a neural network and predict new waveforms.


    Parameters
    ----------
    filename : str
            Name for the model. Saved data will be saved under this name.
    initial_frequency_hz : float, optional
            Initial frequency for the waveforms, by default 20.0
    srate_hz : float, optional
            Time-domain signal rate for the waveforms,
            which is twice the maximum frequency of
            their frequency-domain version.
            By default 4096.0
    pca_components : int, optional
            Number of PCA components to use when reducing
            the dimensionality of the dataset.
            By default 30, which is high enough to reach extremely good
            reconstruction accuracy (mismatches smaller than :math:`10^{-8}`).
    waveform_generator : WaveformGenerator, optional
            Generator for the waveforms to be used in the training,
            by default TEOBResumSGenerator().
    downsampling_training : DownsamplingTraining, optional
            Training algorithm for the downsampling;
            by default None, which means the greedy algorithm
            implemented in :class:`GreedyDownsamplingTraining` is used.
    """

    def __init__(
        self,
        filename: str,
        initial_frequency_hz: float = 20.0,
        srate_hz: float = 4096.0,
        pca_components: int = 30,
        waveform_generator: WaveformGenerator = TEOBResumSGenerator(),
        downsampling_training: Optional[DownsamplingTraining] = None,
    ):

        self.filename = filename
        self.dataset = Dataset(initial_frequency_hz, srate_hz)
        self.waveform_generator = waveform_generator
        self.downsampling_training = (
            GreedyDownsamplingTraining(self.dataset)
            if downsampling_training is None
            else downsampling_training
        )

        self.pca_components = pca_components

        self.nn: Optional[MLPRegressor] = None
        self.hyper: Hyperparameters = Hyperparameters.default()

        self.is_loaded: bool = False

    @property
    def file_arrays(self) -> h5py.File:
        """File object in which to save datasets.

        Returns
        -------
        file : h5py.File
            To be used as a context manager.
        """
        return h5py.File(f"{self.filename}_arrays.h5", mode="a")

    @property
    def filename_nn(self) -> str:
        """File name in which to save the neural network."""
        return f"{self.filename}_nn.pkl"

    @property
    def filename_hyper(self) -> str:
        """File name in which to save the hyperparameters."""
        return f"{self.filename}_hyper.pkl"

    def generate(
        self,
        training_downsampling_dataset_size: Optional[int] = 64,
        training_pca_dataset_size: Optional[int] = 1024,
        training_nn_dataset_size: Optional[int] = 1024,
    ) -> None:
        """Generate a new model from scratch.

        The parameters are the sizes of the three datasets to be used when training,
        if they are set to None they are not computed and the pre-existing
        values are used instead.

        Parameters
        ----------
        training_downsampling_dataset_size : int, optional
                By default 64.
        training_pca_dataset_size : int, optional
                By default 1024.
        training_nn_dataset_size : int, optional
                By default 1024.

        """

        if training_downsampling_dataset_size is not None:
            self.downsampling_indices: DownsamplingIndices = (
                self.downsampling_training.train(training_downsampling_dataset_size)
            )

        if training_pca_dataset_size is not None:
            self.pca_training = PrincipalComponentTraining(
                self.dataset, self.downsampling_indices, self.pca_components
            )

            self.pca_data: PrincipalComponentData = self.pca_training.train(
                training_pca_dataset_size
            )

        if training_nn_dataset_size is not None:
            _, parameters, residuals = self.dataset.generate_residuals(
                training_nn_dataset_size, self.downsampling_indices
            )

            self.training_dataset: Residuals = residuals
            self.training_parameters: ParameterSet = parameters

        self.train_parameter_scaler()

        self.is_loaded = True

    def save_arrays(self) -> None:
        """Save all big arrays contained in this object to the file
        defined as ``{filename}.h5``.
        """

        for arr in [
            self.downsampling_indices,
            self.pca_data,
            self.training_dataset,
            self.training_parameters,
        ]:
            arr.save_to_file(self.file_arrays)

    def save(self) -> None:
        self.save_arrays()
        joblib.dump(self.hyper, self.filename_hyper)
        if self.nn is not None:
            joblib.dump(self.nn, self.filename_nn)

    def load(self) -> None:
        """Load model from the files present in the current folder."""

        self.downsampling_indices = DownsamplingIndices.from_file(self.file_arrays)
        self.pca_data = PrincipalComponentData.from_file(self.file_arrays)
        self.training_dataset = Residuals.from_file(self.file_arrays)
        self.training_parameters = ParameterSet.from_file(self.file_arrays)

        try:
            self.hyper = joblib.load(self.filename_hyper)
            self.nn = joblib.load(self.filename_nn)
        except FileNotFoundError:
            logging.info("No trained network nor hyperparmeters found.")

        self.train_parameter_scaler()

        self.is_loaded = True

    def train_parameter_scaler(self) -> None:
        """Train the parameter scaler, which takes the
        waveform parameters and makes their magnitudes similar
        in order to aid the training of the network.

        The parameter scaler is always trained on all the available dataset.
        """

        self.param_scaler: StandardScaler = StandardScaler().fit(
            self.training_parameters.parameter_array
        )

    @property
    def reduced_residuals(self) -> np.ndarray:
        """Reduced-dimensionality residuals
        --- in other words, PCA components ---
        corresponding to :attr:`training_dataset`.

        This attribute is cached.
        """

        return self._reduced_residuals(self.training_dataset)

    @lru_cache(maxsize=1)
    def _reduced_residuals(self, dataset: Residuals):
        return self.pca_model.reduce_data(dataset.combined, self.pca_data)

    @property
    def pca_model(self) -> PrincipalComponentAnalysisModel:
        """PCA model to be used for dimensionality reduction.

        Returns
        -------
        PrincipalComponentAnalysisModel
        """
        return PrincipalComponentAnalysisModel(self.pca_components)

    def train_nn(
        self, hyper: Hyperparameters, indices: Union[list[int], slice] = slice(None)
    ) -> MLPRegressor:
        """Train a

        Parameters
        ----------
        hyper : Hyperparameters
            Hyperparameters to be used in the initialization
            of the network.
        indices : Union[list[int], slice], optional
            Indices used to perform a selection of a subsection
            of the training data; by default ``slice(None)``
            which means all available training data is used.

        Returns
        -------
        MLPRegressor
            Trained network.
        """

        scaled_params: np.ndarray = self.param_scaler.transform(
            self.training_parameters.parameter_array[indices]
        )

        training_residuals = (
            self.reduced_residuals
            * (self.pca_data.eigenvalues ** hyper.pc_exponent)[np.newaxis, :]
        )

        return MLPRegressor(**hyper.nn_params).fit(
            scaled_params, training_residuals[indices]
        )

    def set_hyper_and_train_nn(self, hyper: Optional[Hyperparameters] = None) -> None:
        if hyper is None:
            hyper = Hyperparameters.default(len(self.training_dataset))

        self.nn = self.train_nn(hyper)
        self.hyper = hyper

    def predict_residuals_bulk(
        self, params: ParameterSet, nn: MLPRegressor, hyper: Hyperparameters
    ) -> Residuals:
        """Make a prediction for a set of different parameters,
        using a network provided as a parameter.

        Parameters
        ----------
        params : ParameterSet
            Parameters of the residuals to reconstruct.
        nn : MLPRegressor
            Neural network to use for the reconstruction
        hyper : Hyperparameters
            Used just for the :attr:`Hyperparameters.pc_exponent` attribute.

        Returns
        -------
        Residuals
            Prediction through the model plus PCA.
        """

        scaled_params = self.param_scaler.transform(params.parameter_array)

        scaled_pca_components = nn.predict(scaled_params)

        combined_residuals = self.pca_model.reconstruct_data(
            scaled_pca_components / (self.pca_data.eigenvalues ** hyper.pc_exponent),
            self.pca_data,
        )

        return Residuals.from_combined_residuals(
            combined_residuals, self.downsampling_indices.numbers_of_points
        )

    def predict_waveforms_bulk(
        self, params: ParameterSet, nn: MLPRegressor, hyper: Hyperparameters
    ) -> FDWaveforms:

        residuals = self.predict_residuals_bulk(params, nn, hyper)

        return self.dataset.recompose_residuals(
            residuals, params, self.downsampling_indices
        )


#     def predict(self, frequencies: np.ndarray, params: dict[str, float]):
#         """Calculate the waveforms in the plus and cross polarizations,
#         accounting for extrinsic parameters

#         Parameters
#         ----------
#         f (np.ndarray)
#                 Frequencies where to compute the waveform, in SI units.
#         params (dict)
#                 Dictionary of parameters, both intrinsic and extrinsic.

#         Returns
#         -------
#         hp, hc (complex np.ndarray)
#                 Plus and cross-polarized waveforms.

#         """

#         q = params.pop('q', 1)
#         eta = q / (1+q)**2

#         p = [q,
#             params.pop('lambda1', 0),
#             params.pop('lambda2', 0),
#             params.pop('s1z', 0),
#             params.pop('s2z', 0)]

#         distance = params.pop('distance', 1)
#         iota = params.pop('iota', 0)
#         phi_ref = params.pop('phi_ref', 0)
#         time_shift  = params.pop('time_shift', 0)
#         mtot = params.pop('mtot', const.MASS_SUM)

#         inv_mass_sum_hz = const.INV_SUN_MASS_HZ / mtot

#         _, amp, phi = self.predict(
#             freqs = f / inv_mass_sum_hz,
#             params=p, one_wf=True)

#         phase_shift = phi_ref + (2 * np.pi * time_shift) * f
#         pre = self.dataset.mlgw_bns_prefactor(eta)
#         cosi = np.cos(iota)
#         pre_plus = (1 + cosi ** 2) / 2 * pre / distance
#         pre_cross = cosi * pre * (-1j) / distance

#         return include_extrinsic_params(amp, phi, phase_shift, pre_plus, pre_cross)

# @njit
# def include_extrinsic_params(amp: np.ndarray, phi: np.ndarray, phase_shift: float, pre_plus: float, pre_cross: float) -> tuple[np.ndarray, np.ndarray]:
#     wf = amp * np.exp(1j * (phi + phase_shift))
#     hp = pre_plus * wf
#     hc = pre_cross * wf
#     return hp, hc


def retrieve_best_trials_list() -> list[optuna.trial.FrozenTrial]:

    stream = pkg_resources.resource_stream(__name__, TRIALS_FILE)
    return joblib.load(stream)


def best_trial_under_n(
    best_trials: list[optuna.trial.FrozenTrial], training_number: int
) -> Hyperparameters:

    accuracy = lambda trial: trial.values[0]

    # take the most accurate trial
    # which used less training data than the given
    # training number
    best_trial = sorted(
        [trial for trial in best_trials if trial.params["n_train"] <= training_number],
        key=accuracy,
    )[0]

    return Hyperparameters.from_frozen_trial(best_trial)
