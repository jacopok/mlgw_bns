from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import IO, ClassVar, Optional, Type, Union

import h5py
import joblib  # type: ignore
import numpy as np
import optuna
import pkg_resources
from numba import njit  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from .data_management import (
    DownsamplingIndices,
    FDWaveforms,
    PrincipalComponentData,
    Residuals,
    SavableData,
)
from .dataset_generation import (
    BarePostNewtonianGenerator,
    Dataset,
    ParameterGenerator,
    ParameterSet,
    TEOBResumSGenerator,
    WaveformGenerator,
    WaveformParameters,
)
from .downsampling_interpolation import DownsamplingTraining, GreedyDownsamplingTraining
from .neural_network import Hyperparameters, NeuralNetwork, SklearnNetwork
from .principal_component_analysis import (
    PrincipalComponentAnalysisModel,
    PrincipalComponentTraining,
)
from .taylorf2 import SUN_MASS_SECONDS

DEFAULT_DATASET_BASENAME = "data/default_dataset"


@dataclass
class ParametersWithExtrinsic:
    r"""Parameters for the generation of a single waveform,
    including extrinsic parameters.

    Parameters
    ----------
    mass_ratio : float
            Mass ratio of the system, :math:`q = m_1 / m_2`,
            where :math:`m_1 \geq m_2`, so :math:`q \geq 1`.
    lambda_1 : float
            Tidal polarizability of the larger star.
            In papers it is typically denoted as :math:`\Lambda_1`;
            for a definition see for example section D of
            `this paper <http://arxiv.org/abs/1805.11579>`_.
    lambda_2 : float
            Tidal polarizability of the smaller star.
    chi_1 : float
            Aligned dimensionless spin component of the larger star.
            The dimensionless spin is defined as
            :math:`\chi_i = S_i / m_i^2` in
            :math:`c = G = 1` natural units, where
            :math:`S_i` is the :math:`z` component
            of the dimensionful spin vector.
            The :math:`z` axis is defined as the one which is
            parallel to the orbital angular momentum of the binary.
    chi_2 : float
            Aligned spin component of the smaller star.
    distance_mpc : float
            Distance to the binary system, in Megaparsecs.
    inclination : float
            Inclination --- angle between the binary system's
            angular momentum and the observation direction, in radians.
    total_mass : float
            Total mass of the binary system, in solar masses.
    reference_phase : float
            This will be set as the phase of the first point of the waveform.
            Defaults to 0.
    time_shift : float
            The waveform will be shifted in the time domain
            by this amount (measured in seconds).
            In the frequency domain, this means adding a linear
            term to the phase.
            Defaults to 0, which by convention means a configuration
            in which the merger happens at the right edge of the
            timeseries. This also means that, in the frequency domain,
            the phase at high frequencies is roughly constant.

    """

    mass_ratio: float
    lambda_1: float
    lambda_2: float
    chi_1: float
    chi_2: float
    distance_mpc: float
    inclination: float
    total_mass: float
    reference_phase: float = 0.0
    time_shift: float = 0.0

    def intrinsic(self, dataset: Dataset) -> WaveformParameters:
        return WaveformParameters(
            mass_ratio=self.mass_ratio,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            chi_1=self.chi_1,
            chi_2=self.chi_2,
            dataset=dataset,
        )

    @property
    def mass_sum_seconds(self) -> float:
        return self.total_mass * SUN_MASS_SECONDS

    def teobresums_dict(self, dataset: Dataset) -> dict[str, Union[float, int, str]]:
        """Parameter dictionary in a format compatible with
        TEOBResumS.

        The parameters are all converted to natural units.
        """
        base_dict = self.intrinsic(dataset).teobresums

        return {
            **base_dict,
            **{
                "M": self.total_mass,
                "distance": self.distance_mpc,
                "inclination": self.inclination,
            },
        }
        # TODO figure out if it is possible to also pass
        # the phase and the time shift to TEOB.


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
    pca_components_number : int, optional
            Number of PCA components to use when reducing
            the dimensionality of the dataset.
            By default 30, which is high enough to reach extremely good
            reconstruction accuracy (mismatches smaller than :math:`10^{-8}`).
    waveform_generator : WaveformGenerator, optional
            Generator for the waveforms to be used in the training;
            by default None, in which case the system attempts to import
            the Python wrapper for TEOBResumS, failing which a :class:`BareBarePostNewtonianGenerator`
            is used, which is unable to generate effective-one-body waveforms.
    downsampling_training : DownsamplingTraining, optional
            Training algorithm for the downsampling;
            by default None, which means the greedy algorithm
            implemented in :class:`GreedyDownsamplingTraining` is used.
    nn_kind : Type[NeuralNetwork]
            Neural network implementation to use,
            defaults to :class:`SklearnNetwork`.
    parameter_generator_kwargs: dict[str, tuple[float, float]], optional
            Dictionary of keyword arguments to be used when initializing
            a :class:`ParameterGenerator`.
            It should be a map of strings to tuples, in the form
            `{'q_range': (1., 3.)}`.

            Options include
            ``q_range``,
            ``lambda1_range``,
            ``lambda2_range``,
            ``chi1_range``,
            ``chi2_range``.
    mass_range: tuple[float, float]
            Range of total masses :math:`M` (in solar masses)
            that the model should be able to reconstruct.
            The reconstruction works with the frequency expressed as :math:`Mf`,
            but changing the total mass changes the range of relevant frequencies.

            Defaults to (2.8, 2.8).
    """

    def __init__(
        self,
        filename: str,
        initial_frequency_hz: float = 20.0,
        srate_hz: float = 4096.0,
        pca_components_number: int = 30,
        multibanding: bool = True,
        waveform_generator: Optional[WaveformGenerator] = None,
        downsampling_training: Optional[DownsamplingTraining] = None,
        nn_kind: Type[NeuralNetwork] = SklearnNetwork,
        parameter_generator_kwargs: Optional[dict[str, tuple[float, float]]] = None,
        mass_range: tuple[float, float] = (2.8, 2.8),
    ):

        self.filename = filename

        if waveform_generator is None:
            try:
                from EOBRun_module import EOBRunPy  # type: ignore

                self.waveform_generator: WaveformGenerator = TEOBResumSGenerator(
                    EOBRunPy
                )
            except ModuleNotFoundError:
                self.waveform_generator = BarePostNewtonianGenerator()
        else:
            self.waveform_generator = waveform_generator

        effective_initial_frequency, effective_srate = expand_frequency_range(
            initial_frequency_hz, srate_hz, mass_range, Dataset.total_mass
        )

        self.dataset = Dataset(
            effective_initial_frequency,
            effective_srate,
            waveform_generator=self.waveform_generator,
            multibanding=multibanding,
            parameter_generator_kwargs=parameter_generator_kwargs,
        )

        if downsampling_training is None:
            self.downsampling_training: DownsamplingTraining = (
                GreedyDownsamplingTraining(self.dataset)
            )
        else:
            self.downsampling_training = downsampling_training

        self.pca_components_number = pca_components_number

        self.nn: Optional[NeuralNetwork] = None

        self.training_dataset: Optional[Residuals] = None
        self.training_parameters: Optional[ParameterSet] = None

        self.pca_data: Optional[PrincipalComponentData] = None
        self.downsampling_indices: Optional[DownsamplingIndices] = None

        self.nn_kind = nn_kind

    @classmethod
    def default(cls, filename: str):
        model = cls(DEFAULT_DATASET_BASENAME)

        stream_arrays = pkg_resources.resource_stream(__name__, model.filename_arrays)
        stream_nn = pkg_resources.resource_stream(__name__, model.filename_nn)

        model.load(streams=(stream_arrays, stream_nn))

        model.filename = filename

        return model

    @property
    def auxiliary_data_available(self) -> bool:
        return self.pca_data is not None and self.downsampling_indices is not None

    @property
    def nn_available(self) -> bool:
        return self.nn is not None and self.auxiliary_data_available

    @property
    def training_dataset_available(self) -> bool:
        return (
            self.training_dataset is not None and self.training_parameters is not None
        )

    @property
    def filename_arrays(self) -> str:
        return f"{self.filename}_arrays.h5"

    @property
    def file_arrays(self) -> h5py.File:
        """File object in which to save datasets.

        Returns
        -------
        file : h5py.File
            To be used as a context manager.
        """
        return h5py.File(self.filename_arrays, mode="a")

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
        training_pca_dataset_size: Optional[int] = 256,
        training_nn_dataset_size: Optional[int] = 256,
    ) -> None:
        """Generate a new model from scratch.

        The parameters are the sizes of the three datasets to be used when training,
        if they are set to None they are not computed and the pre-existing values are used instead.

        Raises
        ------
        AssertionError
                If one of the parameters is set to None but no
                pre-existing data is availabele for it.


        Parameters
        ----------
        training_downsampling_dataset_size : int, optional
                By default 64.
        training_pca_dataset_size : int, optional
                By default 256.
        training_nn_dataset_size : int, optional
                By default 256.

        """

        if training_downsampling_dataset_size is not None:
            self.downsampling_indices = self.downsampling_training.train(
                training_downsampling_dataset_size
            )
        else:
            assert self.downsampling_indices is not None

        if training_pca_dataset_size is not None:
            self.pca_training = PrincipalComponentTraining(
                self.dataset, self.downsampling_indices, self.pca_components_number
            )

            self.pca_data = self.pca_training.train(training_pca_dataset_size)
        else:
            assert self.pca_data is not None

        if training_nn_dataset_size is not None:
            _, parameters, residuals = self.dataset.generate_residuals(
                training_nn_dataset_size, self.downsampling_indices
            )

            self.training_dataset = residuals
            self.training_parameters = parameters
        else:
            assert self.training_dataset is not None
            assert self.training_parameters is not None

        self.train_parameter_scaler()

    def save_arrays(self, include_training_data: bool = True) -> None:
        """Save all big arrays contained in this object to the file
        defined as ``{filename}.h5``.
        """

        assert self.pca_data is not None
        assert self.downsampling_indices is not None
        assert self.training_parameters is not None

        arr_list: list[SavableData] = [
            self.downsampling_indices,
            self.pca_data,
            self.training_parameters,
        ]

        if include_training_data:
            assert self.training_dataset is not None

            arr_list += [
                self.training_dataset,
            ]

        for arr in arr_list:
            arr.save_to_file(self.file_arrays)

    def save(self, include_training_data: bool = True) -> None:
        self.save_arrays(include_training_data)
        if self.nn is not None:
            self.nn.save(self.filename_nn)

    def load(self, streams: Optional[tuple[IO[bytes], IO[bytes]]] = None) -> None:
        """Load model from the files present in the current folder.

        Parameters
        ----------
        streams: tuple[IO[bytes], IO[bytes], IO[bytes]], optional
                For internal use (specifically, loading the default model).
                Defaults to None (look in the current folder).
        """

        if streams is not None:
            filename_arrays: Union[IO[bytes], str]
            filename_nn: Union[IO[bytes], str]

            filename_arrays, filename_nn = streams
            file_arrays = h5py.File(filename_arrays, mode="r")
            ignore_warnings = True
        else:
            file_arrays = self.file_arrays
            filename_nn = self.filename_nn
            ignore_warnings = False

        self.downsampling_indices = DownsamplingIndices.from_file(file_arrays)
        self.pca_data = PrincipalComponentData.from_file(file_arrays)
        self.training_parameters = ParameterSet.from_file(file_arrays)

        if (
            self.downsampling_indices is None
            or self.pca_data is None
            or self.training_parameters is None
        ):
            raise FileNotFoundError

        self.training_dataset = Residuals.from_file(
            file_arrays, ignore_warnings=ignore_warnings
        )

        try:
            self.nn = self.nn_kind.from_file(filename_nn)
        except FileNotFoundError:
            logging.warn("No trained network or hyperparmeters found.")

        self.train_parameter_scaler()

    def train_parameter_scaler(self) -> None:
        """Train the parameter scaler, which takes the
        waveform parameters and makes their magnitudes similar
        in order to aid the training of the network.

        The parameter scaler is always trained on all the available dataset.
        """
        assert self.training_parameters is not None

        self.param_scaler: StandardScaler = StandardScaler().fit(
            self.training_parameters.parameter_array
        )

    @property
    def reduced_residuals(self) -> np.ndarray:
        """Reduced-dimensionality residuals
        --- in other words, PCA components ---
        corresponding to the :attr:`training_dataset`.

        This attribute is cached.
        """

        assert self.training_dataset is not None

        return self._reduced_residuals(self.training_dataset)

    @lru_cache(maxsize=1)
    def _reduced_residuals(self, residuals: Residuals):

        assert self.pca_data is not None

        return self.pca_model.reduce_data(residuals.combined, self.pca_data)

    @property
    def pca_model(self) -> PrincipalComponentAnalysisModel:
        """PCA model to be used for dimensionality reduction.

        Returns
        -------
        PrincipalComponentAnalysisModel
        """
        return PrincipalComponentAnalysisModel(self.pca_components_number)

    def train_nn(
        self, hyper: Hyperparameters, indices: Union[list[int], slice] = slice(None)
    ) -> NeuralNetwork:
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
        NeuralNetwork
            Trained network.
        """
        assert self.training_parameters is not None
        assert self.pca_data is not None

        scaled_params: np.ndarray = self.param_scaler.transform(
            self.training_parameters.parameter_array[indices]
        )

        training_residuals = (
            self.reduced_residuals
            * (self.pca_data.eigenvalues ** hyper.pc_exponent)[np.newaxis, :]
        )

        nn = self.nn_kind(hyper)
        nn.fit(scaled_params, training_residuals[indices])
        return nn

    def set_hyper_and_train_nn(self, hyper: Optional[Hyperparameters] = None) -> None:
        """Train the network according to the hyperparameters given,
        and set it as a class attribute

        Parameters
        ----------
        hyper : Hyperparameters, optional
            Hyperparameters to use when training the network, by default None.
            If not given, the default is to fall back to the standard set of hyperparameters
            provided with the module.
        """

        if hyper is None:
            assert self.training_dataset is not None
            hyper = Hyperparameters.default(len(self.training_dataset))

        # increase the number of maximum iterations by a lot:
        # here we do not want to stop the training early.
        hyper.max_iter *= 10

        self.nn = self.train_nn(hyper)

    def predict_residuals_bulk(
        self, params: ParameterSet, nn: NeuralNetwork
    ) -> Residuals:
        """Make a prediction for a set of different parameters,
        using a network provided as a parameter.

        Parameters
        ----------
        params : ParameterSet
            Parameters of the residuals to reconstruct.
        nn : NeuralNetwork
            Neural network to use for the reconstruction

        Returns
        -------
        Residuals
            Prediction through the model plus PCA.
        """

        assert self.pca_data is not None
        assert self.downsampling_indices is not None

        scaled_params = self.param_scaler.transform(params.parameter_array)

        scaled_pca_components = nn.predict(scaled_params)

        combined_residuals = self.pca_model.reconstruct_data(
            scaled_pca_components / (self.pca_data.eigenvalues ** nn.hyper.pc_exponent),
            self.pca_data,
        )

        return Residuals.from_combined_residuals(
            combined_residuals, self.downsampling_indices.numbers_of_points
        )

    def predict_waveforms_bulk(
        self,
        params: ParameterSet,
        nn: Optional[NeuralNetwork] = None,
    ) -> FDWaveforms:

        if nn is None:
            nn = self.nn
        assert nn is not None

        residuals = self.predict_residuals_bulk(params, nn)

        return self.dataset.recompose_residuals(
            residuals, params, self.downsampling_indices
        )

    def predict(self, frequencies: np.ndarray, params: ParametersWithExtrinsic):
        r"""Calculate the waveforms in the plus and cross polarizations,
        accounting for extrinsic parameters

        Parameters
        ----------
        frequencies : np.ndarray
                Frequencies where to compute the waveform, in Hz.

                These should always be within the range in which the
                model has been trained, and be careful!
                The model is always trained with a specific initial frequency
                :math:`f_0`, and a final frequency :math:`f_1`,
                and it is trained to reconstruct the dependence
                of the waveform on :math:`M_0 f`, where :math:`M_0` is
                some standard mass, typically :math:`2.8M_{\odot}`.

                Now, this means that the model can only predict in the range
                :math:`M_0 f_0 \leq M f \leq M_0 f_1`;
                when :math:`M` differs significantly from :math:`M_0`
                this will be quite a different range from :math:`[f_0, f_1]`.


        params : ParametersWithExtrinsic
                Parameters for the waveform, both intrinsic and extrinsic.

        Raises
        ------
        AssertionError
                When the frequencies given are either too high or too low.

        Returns
        -------
        hp, hc (complex np.ndarray)
                Cartesian plus and cross-polarized waveforms, computed
                at the given frequencies, measured in 1/Hz.

        """
        assert self.downsampling_indices is not None
        assert self.nn is not None

        rescaled_frequencies = frequencies * (
            params.total_mass / self.dataset.total_mass
        )

        assert min(rescaled_frequencies) >= self.dataset.initial_frequency_hz
        assert max(rescaled_frequencies) <= self.dataset.srate_hz / 2.0

        intrinsic_params = params.intrinsic(self.dataset)

        residuals = self.predict_residuals_bulk(
            ParameterSet.from_list_of_waveform_parameters([intrinsic_params]), self.nn
        )

        pn_amplitude = self.dataset.waveform_generator.post_newtonian_amplitude(
            intrinsic_params,
            self.dataset.frequencies[self.downsampling_indices.amplitude_indices],
        )
        pn_phase = self.dataset.waveform_generator.post_newtonian_phase(
            intrinsic_params,
            self.dataset.frequencies[self.downsampling_indices.phase_indices],
        )

        # downsampled amplitude array
        amp_ds = combine_residuals_amp(residuals.amplitude_residuals[0], pn_amplitude)
        phi_ds = combine_residuals_phi(residuals.phase_residuals[0], pn_phase)

        amp = self.downsampling_training.resample(
            self.dataset.frequencies_hz[self.downsampling_indices.amplitude_indices],
            rescaled_frequencies,
            amp_ds,
        )

        phi = (
            self.downsampling_training.resample(
                self.dataset.frequencies_hz[self.downsampling_indices.phase_indices],
                rescaled_frequencies,
                phi_ds,
            )
            + params.reference_phase
            + (2 * np.pi * params.time_shift) * frequencies
        )

        cartesian_waveform = combine_amp_phase(amp, phi)

        pre = self.dataset.mlgw_bns_prefactor(intrinsic_params.eta, params.total_mass)
        cosi = np.cos(params.inclination)
        pre_plus = (1 + cosi ** 2) / 2 * pre / params.distance_mpc
        pre_cross = cosi * pre * (-1j) / params.distance_mpc

        return compute_polarizations(cartesian_waveform, pre_plus, pre_cross)


@njit
def combine_amp_phase(amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Combine amplitude and phase arrays into a Cartesian waveform,
    according to
    :math:`h = A e^{i \phi}`.

    This function is separated out just so that it can be decorated with ``@njit``.

    Parameters
    ----------
    amp : np.ndarray
    phase : np.ndarray

    Returns
    -------
    np.ndarray
    """
    return amp * np.exp(1j * phase)


@njit
def combine_residuals_amp(amp: np.ndarray, amp_pn: np.ndarray) -> np.ndarray:
    r"""Combine amplitude residuals with their Post-Newtonian counterparts,
    according to
    :math:`A = A_{PN} e^{\Delta A}`.

    This function is separated out just so that it can be decorated with ``@njit``.

    Parameters
    ----------
    amp : np.ndarray
    amp_pn : np.ndarray

    Returns
    -------
    np.ndarray
    """
    return amp_pn * np.exp(amp)


@njit
def combine_residuals_phi(phi: np.ndarray, phi_pn: np.ndarray) -> np.ndarray:
    """Combine amplitude residuals with their Post-Newtonian counterparts,
    according to
    :math:`\phi = \phi_{PN} + \Delta \phi`.

    This function is separated out just so that it can be decorated with ``@njit``.

    Parameters
    ----------
    phi : np.ndarray
    phi_pn : np.ndarray

    Returns
    -------
    np.ndarray
    """
    return phi_pn + phi


@njit
def compute_polarizations(
    waveform: np.ndarray,
    pre_plus: Union[complex, float],
    pre_cross: Union[complex, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the two polarizations of the waveform,
    assuming they are the same but for a differerent prefactor
    (which is the case for compact binary coalescences).

    This function is separated out so that it can be decorated with
    `numba.njit <https://numba.pydata.org/numba-doc/latest/reference/jit-compilation.html>`_
    which allows it to be compiled --- this can speed up the computation somewhat.

    Parameters
    ----------
    waveform : np.ndarray
        Cartesian complex-valued waveform.
    pre_plus : complex
        Complex-valued prefactor for the plus polarization of the waveform.
    pre_cross : complex
        Complex-valued prefactor for the cross polarization of the waveform.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Plus and cross polarizations: complex-valued arrays.
    """
    hp = pre_plus * waveform
    hc = pre_cross * waveform
    return hp, hc


def expand_frequency_range(
    initial_frequency: float,
    final_frequency: float,
    mass_range: tuple[float, float],
    reference_mass: float,
) -> tuple[float, float]:
    """Widen the frequency range to account for the
    different masses the user requires.

    Parameters
    ----------
    initial_frequency : float
        Lower bound for the frequency.
        Typically in Hz, but this function just requires it
        to be consistent with the other parameters.
    final_frequency : float
        Upper bound for the frequency.
        It can also be given as the time-domain
        signal rate :math:`r = 1 / \Delta t`, which is
        twice che maximum frequency because of the Nyquist bound.

        Since all this function does is multiply it by a certain factor,
        the formulations can be exchanged.
    mass_range : tuple[float, float]
        Range of allowed masses, in the same unit as the
        reference mass (typically, solar masses).
    reference_mass : float
        Reference mass the model uses to convert frequencies
        to the dimensionless :math:`Mf`.

    Returns
    -------
    tuple[float, float]
        New lower and upper bounds for the frequency range.
    """

    m_min, m_max = mass_range
    assert m_min <= m_max

    return (
        initial_frequency * (reference_mass / m_max),
        final_frequency * (reference_mass / m_min),
    )
