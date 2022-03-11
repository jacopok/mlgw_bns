from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import IO, ClassVar, Optional, Type, Union

import h5py
import joblib  # type: ignore
import numpy as np
import pkg_resources
from numba import njit  # type: ignore

from .data_management import (
    DownsamplingIndices,
    FDWaveforms,
    ParameterRanges,
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


class FrequencyTooLowError(ValueError):
    """Raised when the frequency given to the predictor is too low."""


class FrequencyTooHighError(ValueError):
    """Raised when the frequency given to the predictor is too high."""


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

    def teobresums_dict(
        self, dataset: Dataset, use_effective_frequencies: bool = True
    ) -> dict[str, Union[float, int, str]]:
        """Parameter dictionary in a format compatible with
        TEOBResumS.

        The parameters are all converted to natural units.
        """
        base_dict = self.intrinsic(dataset).teobresums(use_effective_frequencies)

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
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        initial_frequency_hz: float = 20.0,
        srate_hz: float = 4096.0,
        pca_components_number: int = 30,
        multibanding: bool = True,
        waveform_generator: Optional[WaveformGenerator] = None,
        downsampling_training: Optional[DownsamplingTraining] = None,
        nn_kind: Type[NeuralNetwork] = SklearnNetwork,
        parameter_ranges: ParameterRanges = ParameterRanges(),
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

        self.parameter_ranges = parameter_ranges
        self.initial_frequency_hz = initial_frequency_hz
        self.srate_hz = srate_hz
        self.multibanding = multibanding

        self.dataset = self._make_dataset()

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

    def __str__(self):

        n_waveforms = (
            f"waveforms_available = {len(self.training_dataset)}"
            if self.training_dataset_available
            else ""
        )

        return (
            "Model("
            f"filename={self.filename}, "
            f"auxiliary_data_available={self.auxiliary_data_available}, "
            f"nn_available={self.nn_available}, "
            f"training_dataset_available={self.training_dataset_available}, "
            + n_waveforms
            + f"parameter_ranges={self.parameter_ranges})"
        )

    @classmethod
    def default(cls, filename: Optional[str] = None):
        model = cls(DEFAULT_DATASET_BASENAME)

        stream_arrays = pkg_resources.resource_stream(__name__, model.filename_arrays)
        stream_nn = pkg_resources.resource_stream(__name__, model.filename_nn)

        model.load(streams=(stream_arrays, stream_nn))

        model.filename = filename

        return model

    def _make_dataset(self) -> Dataset:

        return Dataset(
            self.initial_frequency_hz,
            self.srate_hz,
            waveform_generator=self.waveform_generator,
            multibanding=self.multibanding,
            parameter_ranges=self.parameter_ranges,
        )

    def _handle_missing_filename(self) -> None:
        raise ValueError('Please set the "filename" attribute of this object')

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
        if self.filename is None:
            self._handle_missing_filename()

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

        if self.filename is None:
            self._handle_missing_filename()

        return f"{self.filename}_nn.pkl"

    @property
    def filename_hyper(self) -> str:
        """File name in which to save the hyperparameters."""

        if self.filename is None:
            self._handle_missing_filename()

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
            self.parameter_ranges,
        ]

        if include_training_data:
            assert self.training_parameters is not None
            assert self.training_dataset is not None

            arr_list += [
                self.training_parameters,
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
        if self.downsampling_indices is None or self.pca_data is None:
            raise FileNotFoundError

        parameter_ranges = ParameterRanges.from_file(file_arrays)
        assert parameter_ranges is not None
        self.parameter_ranges = parameter_ranges

        self.dataset = self._make_dataset()

        self.training_dataset = Residuals.from_file(
            file_arrays, ignore_warnings=ignore_warnings
        )

        try:
            self.nn = self.nn_kind.from_file(filename_nn)
        except FileNotFoundError:
            logging.warn("No trained network or hyperparameters found.")

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

        training_residuals = (
            self.reduced_residuals
            * (self.pca_data.eigenvalues ** hyper.pc_exponent)[np.newaxis, :]
        )

        nn = self.nn_kind(hyper)
        nn.fit(
            self.training_parameters.parameter_array[indices],
            training_residuals[indices],
        )
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

        scaled_pca_components = nn.predict(params.parameter_array)

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

    def _predict_amplitude_phase(
        self, frequencies: np.ndarray, params: ParametersWithExtrinsic
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the amplitude and phase of a waveform.
        This function is basically the same as :method:`predict`,
        with the difference that it does not compute the
        Cartesian waveform.

        Also, it only gives one polarization
        and does not account for the distance

        Parameters
        ----------
        frequencies : np.ndarray
        params : ParametersWithExtrinsic

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Amplitude and phase.
        """

        assert self.downsampling_indices is not None
        assert self.nn is not None

        rescaled_frequencies = frequencies * (
            params.total_mass / self.dataset.total_mass
        )

        try:
            assert (
                rescaled_frequencies[0] >= self.dataset.effective_initial_frequency_hz
            )
        except AssertionError as e:
            raise FrequencyTooLowError() from e

        try:
            assert rescaled_frequencies[-1] <= self.dataset.effective_srate_hz / 2.0
        except AssertionError as e:
            raise FrequencyTooHighError() from e

        self.parameter_ranges.check_parameters_in_ranges(params)

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

        pre = self.dataset.mlgw_bns_prefactor(intrinsic_params.eta, params.total_mass)

        amp = (
            self.downsampling_training.resample(
                self.dataset.frequencies_hz[
                    self.downsampling_indices.amplitude_indices
                ],
                rescaled_frequencies,
                amp_ds,
            )
            * pre
            / params.distance_mpc
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

        return amp, phi

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
                For speed, this is only checked against the first and last elements
                of the array, assuming that it is sorted.

        Returns
        -------
        hp, hc (complex np.ndarray)
                Cartesian plus and cross-polarized waveforms, computed
                at the given frequencies, measured in 1/Hz.

        """

        amp, phi = self._predict_amplitude_phase(frequencies, params)

        cartesian_waveform_real, cartesian_waveform_imag = combine_amp_phase(amp, phi)

        cosi = np.cos(params.inclination)
        pre_plus = (1 + cosi ** 2) / 2
        pre_cross = cosi

        return compute_polarizations(
            cartesian_waveform_real, cartesian_waveform_imag, pre_plus, pre_cross
        )

    def time_until_merger(
        self,
        frequency: float,
        params: ParametersWithExtrinsic,
        delta_f: Optional[float] = None,
    ) -> float:
        r"""Approximate the time left until merger for a wavorm starting at a given frequency,
        using the approximate Stationary Phase Approximation expression
        given in `Marsat and Baker 2018 <https://arxiv.org/abs/1806.10734>`_ (eq. 20):

        :math:`t = - \frac{1}{2 \pi} \frac{\mathrm{d} \phi}{\mathrm{d} f}`

        The derivative is computed with ninth-order central differences,
        because why not.

        Parameters
        ----------
        frequencies : Union[float, np.ndarray]
            One frequency or an array of them, for which to compute
            the time to merger.
        params : ParametersWithExtrinsic
            Parameters of the CBC.
        delta_f: float, optional
            delta_f for the numerical calculation of the derivative.
            If None (default), it is computed internally as f/1000.

        Returns
        -------
        Union[float, np.ndarray]
            Time or times left until merger.
        """

        df = frequency / 1000
        freqs = frequency + df * np.arange(-4, 5)
        weights = np.array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840.0

        try:
            _, phis = self._predict_amplitude_phase(freqs, params)
            logging.info("Derivative coming from mlgw_bns")
        except FrequencyTooLowError:
            logging.info("Derivative coming from the PN approximant")
            phis = self.waveform_generator.post_newtonian_phase(
                params.intrinsic(self.dataset), freqs * params.mass_sum_seconds
            )

        derivative = np.sum(phis * weights) / df

        return derivative / (2 * np.pi)


@njit
def combine_amp_phase(
    amp: np.ndarray, phase: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Combine amplitude and phase arrays into a Cartesian waveform,
    according to
    :math:`h = A e^{i \phi}`.

    This function is separated out just so that it can be decorated with ``@njit``.

    Parameters
    ----------
    amp : np.ndarray
    phase : np.ndarray

    Returns
    -------
    tuple[np.ndarray, np.ndarray]:
        Real and imaginary parts of the waveform, respectively.
    """
    return (amp * np.cos(phase), amp * np.sin(phase))


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
    r"""Combine amplitude residuals with their Post-Newtonian counterparts,
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
    waveform_real: np.ndarray,
    waveform_imag: np.ndarray,
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
    waveform_real : np.ndarray
        Real part of the cartesian complex-valued waveform.
    waveform_imag : np.ndarray
        Imaginary part of the cartesian complex-valued waveform.
    pre_plus : complex
        Real-valued prefactor for the plus polarization of the waveform.
    pre_cross : complex
        Real-valued prefactor for the cross polarization of the waveform.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Plus and cross polarizations: complex-valued arrays.
    """

    hp = pre_plus * waveform_real + 1j * pre_plus * waveform_imag
    hc = pre_cross * waveform_imag - 1j * pre_cross * waveform_real

    return hp, hc
