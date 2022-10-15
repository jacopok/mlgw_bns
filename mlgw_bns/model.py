from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import IO, ClassVar, Optional, Type, Union

import h5py
import joblib  # type: ignore
import numpy as np
import pkg_resources
import yaml
from dacite import from_dict
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
from .taylorf2 import SUN_MASS_SECONDS, smoothing_func

PRETRAINED_MODEL_FOLDER = "data/"
MODELS_AVAILABLE = ["default", "fast"]


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

    @classmethod
    def gw170817(cls) -> ParametersWithExtrinsic:
        """Convenience method: an easy-to-access
        set of parameters, roughly corresponding to the
        best-fit values for GW170817.
        """
        
        return cls(
            mass_ratio=1.,
            lambda_1=400.,
            lambda_2=400.,
            chi_1=0.,
            chi_2=0.,
            distance_mpc=40.,
            inclination=5/6*np.pi,
            total_mass=2.8,
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
            Initial frequency for the waveforms.
    srate_hz : float, optional
            Time-domain signal rate for the waveforms,
            which is twice the maximum frequency of
            their frequency-domain version.
    pca_components_number : int, optional
            Number of PCA components to use when reducing
            the dimensionality of the dataset.
            By default 30, which is high enough to reach extremely good
            reconstruction accuracy (mismatches smaller than :math:`10^{-8}`).
    multibanding : bool
            Whether to use a multibanded frequency array. 
            See the multibanding module for more details.
    parameter_ranges : ParameterRanges
            Ranges for the parameters to pass to the parameter generator.
    extend_with_post_newtonian: bool
            Whether to accept frequencies lower than the minimum training frequency,
            providing a hybrid post-newtonian / EOB surrogate waveform.
            If this is False, an error will be raised if the frequencies
            given include ones that are too low.
    extend_with_zeros_at_high_frequency: bool
            Whether to accept frequencies higher than the maximum training frequency,
            padding the returned waveform with zeros.
            If this is False, an error will be raised if the frequencies
            given include ones that are too high.
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
    parameter_generator : Optional[ParameterGenerator]
            Certain parameter generators should not be regenerated each time;
            if this is the case, then pass the parameter generator here.
            Defaults to None.
    """
    

    def __init__(
        self,
        filename: Optional[str] = None,
        initial_frequency_hz: float = 10.0,
        srate_hz: float = 4096.0,
        pca_components_number: int = 30,
        multibanding: bool = True,
        parameter_ranges: ParameterRanges = ParameterRanges(),
        extend_with_post_newtonian = True,
        extend_with_zeros_at_high_frequency = False,
        waveform_generator: Optional[WaveformGenerator] = None,
        downsampling_training: Optional[DownsamplingTraining] = None,
        nn_kind: Type[NeuralNetwork] = SklearnNetwork,
        parameter_generator : Optional[ParameterGenerator] = None,
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
        self.parameter_generator = parameter_generator
        self.extend_with_post_newtonian = extend_with_post_newtonian
        self.extend_with_zeros_at_high_frequency = extend_with_zeros_at_high_frequency
        

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
            f"waveforms_available = {len(self.training_dataset)}, "
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

    @property    
    def metadata_dict(self) -> dict:
        return {
            'initial_frequency_hz': self.initial_frequency_hz,
            'srate_hz': self.srate_hz,
            'pca_components_number': self.pca_components_number,
            'multibanding': self.multibanding,
            'parameter_ranges': asdict(self.parameter_ranges),
            'extend_with_post_newtonian': self.extend_with_post_newtonian,
            'extend_with_zeros_at_high_frequency': self.extend_with_zeros_at_high_frequency,
        }

    @classmethod
    def default(cls, model_name: Optional[str]=None, **kwargs):
        
        if model_name is None:
            model_name = MODELS_AVAILABLE[0]

        if model_name not in MODELS_AVAILABLE:
            raise(ValueError(f'Model {model_name} not available!'))
        
        given_filename = kwargs.pop('filename', None)
        
        model = cls(filename=PRETRAINED_MODEL_FOLDER + model_name, **kwargs)

        stream_meta = pkg_resources.resource_stream(__name__, model.filename_metadata)
        stream_arrays = pkg_resources.resource_stream(__name__, model.filename_arrays)
        stream_nn = pkg_resources.resource_stream(__name__, model.filename_nn)

        model.load(streams=(stream_meta, stream_arrays, stream_nn))

        model.filename = given_filename

        return model

    def _make_dataset(self) -> Dataset:

        return Dataset(
            self.initial_frequency_hz,
            self.srate_hz,
            waveform_generator=self.waveform_generator,
            multibanding=self.multibanding,
            parameter_ranges=self.parameter_ranges,
            parameter_generator=self.parameter_generator
        )
    
    @property
    def parameter_generator(self):
        return self._parameter_generator

    @parameter_generator.setter
    def parameter_generator(self, val):
        self._parameter_generator = val
        try:
            self.dataset.parameter_generator = val
        except AttributeError:
            pass

    @property
    def waveform_generator(self):
        return self._waveform_generator

    @waveform_generator.setter
    def waveform_generator(self, val):
        self._waveform_generator = val
        try:
            self.dataset.waveform_generator = val
        except AttributeError:
            pass


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
    def filename_metadata(self) -> str:
        if self.filename is None:
            self._handle_missing_filename()

        return f"{self.filename}.yaml"

    def save_metadata(self):
        
        with open(self.filename_metadata, 'w') as f:
            yaml.dump(self.metadata_dict, f)
    
    def load_metadata(self, stream: Optional[IO[bytes]] = None) -> dict:
        
        if stream is None:
            with open(self.filename_metadata, 'r') as f:
                return yaml.load(f, Loader=yaml.FullLoader)
        
        else:
            return yaml.load(stream, Loader=yaml.FullLoader)
        

    def set_metadata(self, meta_dict: dict) -> None:
        
        for key, value in meta_dict.items():
            if key == 'parameter_ranges':
                value = from_dict(data_class=ParameterRanges, data=value)
            setattr(self, key, value)

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
        self.save_metadata()
        self.save_arrays(include_training_data)
        if self.nn is not None:
            self.nn.save(self.filename_nn)

    def load(self, streams: Optional[tuple[IO[bytes], IO[bytes], IO[bytes]]] = None) -> None:
        """Load model from the files present in the current folder.

        Parameters
        ----------
        streams: tuple[IO[bytes], IO[bytes], IO[bytes]], optional
                For internal use (specifically, loading the default model).
                Defaults to None (look in the current folder).
        """

        if streams is not None:
            stream_meta: Union[IO[bytes], None]
            filename_arrays: Union[IO[bytes], str]
            filename_nn: Union[IO[bytes], str]

            stream_meta, filename_arrays, filename_nn = streams
            file_arrays = h5py.File(filename_arrays, mode="r")
            ignore_warnings = True
        else:
            stream_meta = None
            file_arrays = self.file_arrays
            filename_nn = self.filename_nn
            ignore_warnings = False


        self.set_metadata(self.load_metadata(stream_meta))
        self.downsampling_indices = DownsamplingIndices.from_file(file_arrays)
        self.pca_data = PrincipalComponentData.from_file(file_arrays)
        self.training_parameters = ParameterSet.from_file(
            file_arrays, ignore_warnings=ignore_warnings
        )
        if self.downsampling_indices is None or self.pca_data is None:
            raise FileNotFoundError

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
        hyper.max_iter *= 100

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

        if rescaled_frequencies[0] < self.dataset.effective_initial_frequency_hz:

            if not self.extend_with_post_newtonian:
                raise FrequencyTooLowError(
                    "This model is not configured to be extended with a post-newtonian"
                    "waveform. Set the 'extend_with_post_newtonian' attribute of the model to True"
                    "if that is what you want."
                )
            
            extend_with_pn = True
            limit_index = np.searchsorted(rescaled_frequencies, self.dataset.effective_initial_frequency_hz)
            
            # if we're extending downwards, then we need to also compute the PN phase 
            # at the very end of the low-frequency bit (which might not be in the given array)
            # in order to connect with the high-frequency bit without any discontinuity in phase.
            
            low_freqs_hz = np.append(rescaled_frequencies[:limit_index], self.dataset.effective_initial_frequency_hz) # type: ignore
            rescaled_frequencies = np.append(self.dataset.effective_initial_frequency_hz, rescaled_frequencies[limit_index:]) # type: ignore
            
            low_freqs = self.dataset.hz_to_natural_units(low_freqs_hz)
            connection_f = self.dataset.hz_to_natural_units(self.dataset.effective_initial_frequency_hz)
            
        else:
            extend_with_pn = False

        if len(rescaled_frequencies) < 1:
            # this should never happen! 
            raise ValueError('At least one point should be in the model band')

        if rescaled_frequencies[-1] > self.dataset.effective_srate_hz / 2.0:
            if not self.extend_with_zeros_at_high_frequency:
                raise FrequencyTooHighError(
                    "This model is not configured to be extended with zeros at high frequency."
                    "Set the 'extend_with_zeros_at_high_frequency' attribute of the model to True"
                    "if that is what you want."
                )
            else:
                extend_hf = True
                high_frequency_index = int(np.searchsorted(rescaled_frequencies, self.dataset.effective_srate_hz / 2.0))
                hf_segment_length = len(rescaled_frequencies) - high_frequency_index
                rescaled_frequencies = rescaled_frequencies[:high_frequency_index]


        else:
            extend_hf = False

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

        resampled_amp = self.downsampling_training.resample(
                self.dataset.frequencies_hz[
                    self.downsampling_indices.amplitude_indices
                ],
                rescaled_frequencies,
                amp_ds,
            )
        
        
        resampled_phi = self.downsampling_training.resample(
                self.dataset.frequencies_hz[self.downsampling_indices.phase_indices],
                rescaled_frequencies,
                phi_ds,
        )

        if extend_with_pn:
            
            eob_amplitude_at_connection = resampled_amp[0]
            f_min_connection = connection_f / 2.0
            connecting_mask = np.where(
                low_freqs > f_min_connection,
            )
            
            zero_to_one = (
                (low_freqs[connecting_mask] - f_min_connection) / 
                (connection_f - f_min_connection)
            )
            
            low_freq_amp = (
                self.dataset.waveform_generator.post_newtonian_amplitude(
                intrinsic_params,
                low_freqs,
                )
            )
            pn_amplitude_at_connection = low_freq_amp[-1]
            
            low_freq_amp[connecting_mask] += (
                smoothing_func(zero_to_one) 
                * (eob_amplitude_at_connection - pn_amplitude_at_connection)
            )
            
            resampled_amp = np.concatenate((low_freq_amp[:-1], resampled_amp[1:]))
            
            low_f_phi = self.dataset.waveform_generator.post_newtonian_phase(
                intrinsic_params,
                low_freqs,
            )
            
            resampled_phi = np.concatenate((
                low_f_phi[:-1],
                resampled_phi[1:] + low_f_phi[-1]
            ))

        if extend_hf:
            resampled_amp = np.concatenate((resampled_amp, np.zeros(hf_segment_length)))
            resampled_phi = np.concatenate((resampled_phi, np.zeros(hf_segment_length)))

        amp = (
            resampled_amp
            * pre
            / params.distance_mpc
        )

        phi = (
            resampled_phi
            + params.reference_phase
            + (2 * np.pi * params.time_shift) * frequencies
        )
        
        return amp, phi

    def predict(self, frequencies: np.ndarray, params: ParametersWithExtrinsic):
        r"""Calculate the waveforms in the plus and cross polarizations,
        accounting for extrinsic parameters.
        
        This function is able to yield a sensible waveform at arbitrarily 
        low frequencies, by hybridizing the EOB-trained high-frequency part
        with a Post-Newtonian approximant. 
        This feature can be turned off with the :attr:`extend_with_post_newtonian`
        parameter of the :class:`Model` object.

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
        FrequencyTooLowError
                When the frequencies given are too low, below the training range.
                For speed, this is only checked against the first and last elements
                of the array, assuming that it is sorted.

                This is raised only if the PN extension of the waveform is
                disabled by setting :attr:`extend_with_post_newtonian`
                to False.

        Raises
        ------
        FrequencyTooHighError
                When the frequencies given are too high.
                For speed, this is only checked against the first and last elements
                of the array, assuming that it is sorted.

                This is raised only if the extension of the waveform with zeroes is
                disabled by setting :attr:`extend_with_zeros_at_high_frequency`
                to False.


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
        frequency : float
            frequency for which to compute the time to merger.
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

        if delta_f is None:
            delta_f = frequency / 1000
        freqs = frequency + delta_f * np.arange(-4, 5)
        weights = np.array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840.0

        try:
            _, phis = self._predict_amplitude_phase(freqs, params)
            logging.info("Derivative coming from mlgw_bns")
        except FrequencyTooLowError:
            logging.info("Derivative coming from the PN approximant")
            phis = self.waveform_generator.post_newtonian_phase(
                params.intrinsic(self.dataset), freqs * params.mass_sum_seconds
            )

        derivative = np.sum(phis * weights) / delta_f

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

