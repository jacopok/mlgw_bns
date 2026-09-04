from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import IO, ClassVar, Optional, Type, Union

import time
import h5py
import joblib  # type: ignore
import numpy as np
from numpy.ma import indices
import yaml
from dacite import from_dict
from numba import njit  # type: ignore
from scipy.interpolate import interp1d

from .data_management import (
    array_memory,
    format_bytes,
    peak_memory_usage,
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
    UniformParameterGenerator,
    TEOBResumSGenerator,
    WaveformGenerator,
    WaveformParameters,
    AMP_SI_BASE,
)
from .higher_order_modes import (
    BarePostNewtonianModeGenerator,
    Mode,
    ModeGenerator,
)
from .downsampling_interpolation import (
    DownsamplingTraining,
    GreedyDownsamplingTraining,
    RDPDownsamplingTraining,
)
from .neural_network import (
    Hyperparameters,
    KernelRidgeNetwork,
    NeuralNetwork,
    SklearnNetwork,
    TimeshiftsGPR,
    TimeshiftsNN,
    load_timeshifts_predictor_from_file,
)
from .principal_component_analysis import (
    PrincipalComponentAnalysisModel,
    PrincipalComponentTraining,
    remove_linear_trend,
)
from .taylorf2 import SUN_MASS_SECONDS, smoothing_func
from .higher_order_modes import mode_to_k


#: The regressor backends a saved model may name in its metadata. The
#: network is the historical default; the kernel is far more accurate on
#: the same training data, at the cost of a prediction time that grows
#: with the training set. See :class:`~mlgw_bns.neural_network.KernelRidgeNetwork`.
NN_KINDS: dict[str, Type[NeuralNetwork]] = {
    "SklearnNetwork": SklearnNetwork,
    "KernelRidgeNetwork": KernelRidgeNetwork,
}


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


class ModeModel:
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
    downsampling_training : DownsamplingTraining or type, optional
            Training algorithm for the downsampling. Can be an instance or a class
            (e.g. :class:`RDPDownsamplingTrainingWithResiduals` for faster RDP-based
            downsampling). By default None, which means the greedy algorithm
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
        initial_frequency_hz: float = 20.0,
        srate_hz: float = 4096.0,
        pca_components_number: int = 30,
        multibanding: bool = True,
        extend_with_post_newtonian = True,
        extend_with_zeros_at_high_frequency = True,
        waveform_generator: Optional[WaveformGenerator] = None,
        downsampling_training: Optional[DownsamplingTraining] = None,
        nn_kind: Type[NeuralNetwork] = SklearnNetwork,
        parameter_ranges: ParameterRanges = ParameterRanges(),
        parameter_generator : Optional[ParameterGenerator] = None,
        parameter_generator_class: Optional[Type[ParameterGenerator]] = None,
        mode: Optional[Mode] = None,
        reference_amplitude: bool = False,
    ):

        self.reference_amplitude = reference_amplitude
        self.filename = filename

        if waveform_generator is None:
            try:
                from EOBRun_module import EOBRunPy  # type: ignore

                logging.info("Using EOBRunPy as a waveform generator")
                self.waveform_generator: WaveformGenerator = TEOBResumSGenerator(
                    EOBRunPy
                )
            except ModuleNotFoundError:
                logging.info(
                    "EOBRun_module not found, "
                    "using BarePostNewtonianGenerator as a waveform generator"
                )
                self.waveform_generator = BarePostNewtonianGenerator()
        else:
            self.waveform_generator = waveform_generator

        self.parameter_ranges = parameter_ranges
        self.initial_frequency_hz = initial_frequency_hz
        self.srate_hz = srate_hz
        self.multibanding = multibanding
        self.parameter_generator_class = (
            parameter_generator_class
            if parameter_generator_class is not None
            else UniformParameterGenerator
        )
        self.parameter_generator = parameter_generator
        self.extend_with_post_newtonian = extend_with_post_newtonian
        self.extend_with_zeros_at_high_frequency = extend_with_zeros_at_high_frequency 

        self.dataset = self._make_dataset()

        if downsampling_training is None:
            # For the higher-order modes, cap the frequency ratio between
            # adjacent phase nodes so the high-frequency band (where the
            # (4,4) waveform phase looks locally linear but its residual
            # does not) stays populated.
            self.downsampling_training: DownsamplingTraining = (
                GreedyDownsamplingTraining(
                    self.dataset,
                    max_phi_gap_ratio=1.03 if mode is not None else None,
                )
            )
        elif isinstance(downsampling_training, type) and issubclass(
            downsampling_training, DownsamplingTraining
        ):
            self.downsampling_training = downsampling_training(self.dataset)
        else:
            self.downsampling_training = downsampling_training

        self.pca_components_number = pca_components_number

        self.nn: Optional[NeuralNetwork] = None
        self.timeshifts_predictor: Optional[Union[TimeshiftsGPR, TimeshiftsNN]] = None

        # Shared per-mode reference-phase predictor and this mode's column
        # in its output. Set by `Model` for HOM models; when None the
        # phase reconstruction adds no per-mode constant.
        self.mode_phases_predictor = None
        self.mode_phases_index: Optional[int] = None

        self.training_dataset: Optional[Residuals] = None
        self.training_parameters: Optional[ParameterSet] = None

        self.pca_data: Optional[PrincipalComponentData] = None
        self.downsampling_indices: Optional[DownsamplingIndices] = None

        self.nn_kind = nn_kind
        self.mode = mode

    def __str__(self):

        n_waveforms = (
            f"waveforms_available = {len(self.training_dataset)}, "
            if self.training_dataset_available
            else ""
        )

        return (
            "ModeModel("
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
            'nn_kind': self.nn_kind.__name__,
            'reference_amplitude': self.reference_amplitude,
        }

    def _make_dataset(self) -> Dataset:

        return Dataset(
            self.initial_frequency_hz,
            self.srate_hz,
            waveform_generator=self.waveform_generator,
            multibanding=self.multibanding,
            parameter_ranges=self.parameter_ranges,
            parameter_generator=self.parameter_generator,
            parameter_generator_class=self.parameter_generator_class,
            reference_amplitude=self.reference_amplitude,
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
        """Apply a metadata dictionary read back from the YAML sidecar.

        Two keys need decoding rather than a plain ``setattr``: the
        parameter ranges, which are a nested dataclass, and the regressor
        backend, which is stored by name so that the YAML stays readable
        and free of Python references. A file written before ``nn_kind``
        was recorded simply does not carry the key, which leaves the
        constructor default in place --- and that default is the network,
        which is what those models were trained with.
        """

        for key, value in meta_dict.items():
            if key == 'parameter_ranges':
                value = from_dict(data_class=ParameterRanges, data=value)
            elif key == 'nn_kind':
                value = NN_KINDS[value]
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

    @property
    def filename_timeshifts(self) -> str:
        """File name in which to save the mode time-shifts predictor."""

        if self.filename is None:
            self._handle_missing_filename()

        return f"{self.filename}_timeshifts.pkl"

    def _predicted_mode_phase0(self, intrinsic_params) -> float:
        """Per-mode reference phase to restore at the anchor node.

        Non-zero only for HOM models, where ``remove_linear_trend``
        subtracted the shared
        :class:`~mlgw_bns.neural_network.ModePhasesNN` prediction of
        :math:`\\phi_{\\ell m}(f_0)` from the training residuals; this
        returns the very same prediction so it cancels.
        """
        if self.mode_phases_predictor is None or self.mode_phases_index is None:
            return 0.0
        return float(
            self.mode_phases_predictor.predict([intrinsic_params.array])[0][
                self.mode_phases_index
            ]
        )

    def generate(
        self,
        training_downsampling_dataset_size: Optional[int] = 64,
        training_pca_dataset_size: Optional[int] = 256,
        training_nn_dataset_size: Optional[int] = 256,
        timeshifts_predictor: Optional[Union[TimeshiftsGPR, TimeshiftsNN]] = None,
        precomputed_residuals: Optional[tuple] = None,
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
        timeshifts_predictor : TimeshiftsGPR or TimeshiftsNN, optional
                If given, used as :attr:`timeshifts_predictor` instead of
                fitting a new one from this model's own residuals. Used by
                :class:`~mlgw_bns.model.Model` to share a single
                predictor, trained on the (2,2) mode, across every mode.
        precomputed_residuals : tuple, optional
                ``(freq_downsampled_natural, ParameterSet, Residuals)`` for
                this mode, already downsampled to
                :attr:`downsampling_indices`, sized
                ``max(training_pca_dataset_size, training_nn_dataset_size)``.
                Supplied by :meth:`~mlgw_bns.model.Model.generate` from one
                shared multi-mode EOB sweep; when given, the per-mode
                ``Dataset.generate_residuals`` calls for the PCA and NN
                training sets are skipped.

        """

        logging.info(
            "Generating a new model for %s, with %s waveforms for the "
            "downsampling, %s for the PCA and %s for the network",
            "the (2,2) mode" if self.mode is None else f"mode {self.mode}",
            training_downsampling_dataset_size,
            training_pca_dataset_size,
            training_nn_dataset_size,
        )

        if training_downsampling_dataset_size is not None:
            logging.info("Training the downsampling")
            self.downsampling_indices = self.downsampling_training.train(
                training_downsampling_dataset_size
            )
        else:
            assert self.downsampling_indices is not None

        self.log_expected_training_memory(
            training_pca_dataset_size, training_nn_dataset_size
        )

        if training_nn_dataset_size is not None:
            # A single dataset serves both the time-shift predictor and the
            # network: the former only needs one number per waveform, read
            # off the phase residuals which the latter is trained on anyway.
            #
            # The residuals are generated at the downsampled frequencies:
            # the time shift is a chord of the phase residual, so the
            # full-resolution grid buys nothing here while costing a factor
            # `waveform_length / (amp_length + phi_length)` --- of order a
            # thousand --- in memory.
            if precomputed_residuals is not None:
                freq_downsampled, all_parameters, all_residuals = precomputed_residuals
                parameters = self.dataset.parameter_set_cls(
                    all_parameters.parameter_array[:training_nn_dataset_size]
                )
                residuals = all_residuals[:training_nn_dataset_size]
            else:
                logging.info("Generating the training dataset")
                freq_downsampled, parameters, residuals = (
                    self.dataset.generate_residuals(
                        training_nn_dataset_size,
                        self.downsampling_indices,
                        flatten_phase=False,
                    )
                )
            frequencies_hz = self.dataset.natural_units_to_hz(freq_downsampled)

        # LEARN Δt(θ), needed below to remove the linear trend
        # from the phase residuals before PCA and NN training.
        # `TimeshiftsNN` (RFF + Ridge) rather than `TimeshiftsGPR`: the
        # latter's `GaussianProcessRegressor` defaults to
        # `normalize_y=False`, so with a zero-mean prior and a
        # unit-amplitude RBF kernel it collapses to predicting 0 for
        # timeshift targets whose magnitude is far from unity.
        if timeshifts_predictor is not None:
            self.timeshifts_predictor = timeshifts_predictor
        elif training_nn_dataset_size is not None:
            logging.info("Training the time-shifts predictor")

            # `phase_timeshifts` rather than `flatten_phase`, since the
            # residuals are needed in their raw form further down.
            self.training_timeshifts_data = residuals.phase_timeshifts(
                frequencies=frequencies_hz
            )
            self.timeshifts_predictor = TimeshiftsNN(
                training_params=parameters.parameter_array,
                training_timeshifts=self.training_timeshifts_data
            ).fit()
        else:
            assert self.timeshifts_predictor is not None

        if training_pca_dataset_size is not None:
            logging.info("Training the PCA")
            self.pca_training = PrincipalComponentTraining(
                self.dataset,
                self.downsampling_indices,
                self.pca_components_number,
                self.timeshifts_predictor,
                subtract_mode_phase_anchor=self.mode is not None,
                mode_phases_predictor=self.mode_phases_predictor,
                mode_index=self.mode_phases_index,
            )

            if precomputed_residuals is not None:
                freq_ds_pca, all_parameters, all_residuals = precomputed_residuals
                self.pca_data = self.pca_training.train_on(
                    self.dataset.parameter_set_cls(
                        all_parameters.parameter_array[:training_pca_dataset_size]
                    ),
                    all_residuals[:training_pca_dataset_size],
                    self.dataset.natural_units_to_hz(freq_ds_pca),
                )
            else:
                self.pca_data = self.pca_training.train(training_pca_dataset_size)
        else:
            assert self.pca_data is not None

        if training_nn_dataset_size is not None:
            logging.info("Removing the linear trend from the training residuals")
            residuals.phase_residuals = remove_linear_trend(
                parameters=parameters,
                phi_diff=residuals.phase_residuals,
                frq=frequencies_hz,
                timeshifts_predictor=self.timeshifts_predictor,
                subtract_mode_phase_anchor=self.mode is not None,
                mode_phases_predictor=self.mode_phases_predictor,
                mode_index=self.mode_phases_index,
            )

            self.training_dataset = residuals
            self.training_parameters = parameters
        else:
            assert self.training_dataset is not None
            assert self.training_parameters is not None

        logging.info(
            "ModeModel generation done (peak memory usage: %s)",
            format_bytes(peak_memory_usage()),
        )

    def log_expected_training_memory(
        self,
        training_pca_dataset_size: Optional[int],
        training_nn_dataset_size: Optional[int],
    ) -> None:
        """Log the memory the training datasets are expected to take up.

        This is only knowable after the downsampling training has run, since
        the number of sample points it keeps per waveform is decided by the
        greedy (or RDP) algorithm and depends on the tolerances, the mode and
        the frequency grid.

        Parameters
        ----------
        training_pca_dataset_size : int, optional
            Number of waveforms which will be used to train the PCA.
        training_nn_dataset_size : int, optional
            Number of waveforms which will be used to train the network.
        """

        assert self.downsampling_indices is not None

        points_per_waveform = (
            self.downsampling_indices.amp_length + self.downsampling_indices.phi_length
        )

        for name, size in [
            ("PCA", training_pca_dataset_size),
            ("network", training_nn_dataset_size),
        ]:
            if size is None:
                continue

            logging.info(
                "The %s training dataset (%i waveforms x %i sample points) "
                "will take up about %s, "
                "with a transient peak of about %s while it is generated",
                name,
                size,
                points_per_waveform,
                format_bytes(array_memory((size, points_per_waveform), np.float32)),
                format_bytes(array_memory((size, points_per_waveform), np.float64) * 2),
            )

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
            self.parameter_ranges
        ]

        if include_training_data:
            assert self.training_parameters is not None
            assert self.training_dataset is not None

            arr_list += [
                self.training_parameters,
                self.training_dataset,
            ]

        # Open file once for all arrays (avoids repeated open/close overhead)
        with h5py.File(self.filename_arrays, mode="a") as f:
            for arr in arr_list:
                arr.save_to_file(f)

    def save(
        self,
        include_training_data: bool = True,
        include_timeshifts_predictor: bool = True,
    ) -> None:
        """Save this model to the files derived from :attr:`filename`.

        Parameters
        ----------
        include_training_data : bool, optional
                Whether to also persist the training residuals and
                parameters. Defaults to ``True``.
        include_timeshifts_predictor : bool, optional
                Whether to write :attr:`timeshifts_predictor` to
                ``{filename}_timeshifts.pkl``. Defaults to ``True``.
                :meth:`Model.save` passes ``False``: every mode
                shares a single predictor, which that class saves once
                under its own base filename rather than once per mode.
        """
        self.save_metadata()
        self.save_arrays(include_training_data)
        if self.nn is not None:
            self.nn.save(self.filename_nn)
        if include_timeshifts_predictor and self.timeshifts_predictor is not None:
            self.timeshifts_predictor.save_model(self.filename_timeshifts)

    def load(
        self,
        streams: Optional[
            tuple[IO[bytes], IO[bytes], IO[bytes], Optional[IO[bytes]]]
        ] = None,
    ) -> None:
        """Load model from the files present in the current folder.

        Parameters
        ----------
        streams: tuple[IO[bytes], IO[bytes], IO[bytes], Optional[IO[bytes]]], optional
                For internal use (specifically, loading the default model).
                The fourth element (time-shifts predictor) may be ``None``
                if the packaged model does not ship one.
                Defaults to None (look in the current folder).
        """

        if streams is not None:
            stream_meta: Union[IO[bytes], None]
            h5_source: Union[IO[bytes], str]
            filename_nn: Union[IO[bytes], str]
            filename_timeshifts: Union[IO[bytes], str, None]

            stream_meta, h5_source, filename_nn, filename_timeshifts = streams
            ignore_warnings = True
        else:
            stream_meta = None
            h5_source = self.filename_arrays
            filename_nn = self.filename_nn
            filename_timeshifts = self.filename_timeshifts
            ignore_warnings = False

        # Read-only open: supports many parallel workers (ProcessPool) on the same
        # file. Append mode ("a") takes a write lock and fails on NFS / multi-proc.
        with h5py.File(h5_source, mode="r") as file_arrays:
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

        if filename_timeshifts is not None:
            try:
                self.timeshifts_predictor = load_timeshifts_predictor_from_file(
                    filename_timeshifts
                )
            except FileNotFoundError:
                logging.info("No time-shifts predictor found.")

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

        # print(len(self.training_parameters.parameter_array))

        training_residuals = (
            self.reduced_residuals
            * (self.pca_data.eigenvalues ** hyper.pc_exponent)[np.newaxis, :]
        )

        nn = self.nn_kind(hyper)

        start_time = time.time()  # Record the start time

        nn.fit(
            self.training_parameters.parameter_array[indices],
            training_residuals[indices],
        )

        end_time = time.time()  # Record the end time

        training_duration = end_time - start_time  # Compute the duration
        logging.info(
            "Training the network on %i waveforms took %.2f seconds "
            "(peak memory usage so far: %s)",
            len(training_residuals[indices]),
            training_duration,
            format_bytes(peak_memory_usage()),
        )

        # loss_over_epochs = nn.get_loss_over_epochs()

        # print(f"Loss over epochs: {loss_over_epochs}")
        
        return nn

    def set_hyper_and_train_nn(self, hyper: Optional[Hyperparameters] = None, idxs: Union[list[int], slice] = slice(None)) -> None:
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
            if self.nn_kind is KernelRidgeNetwork:
                hyper = Hyperparameters.default_kernel_ridge(
                    len(self.training_dataset), mode=self.mode
                )
            else:
                hyper = Hyperparameters.default(len(self.training_dataset))

        logging.info("Training the network with hyperparameters %s", hyper)

        # increase the number of maximum iterations by a lot:
        # here we do not want to stop the training early.
        hyper.max_iter *= 10

        self.nn = self.train_nn(hyper, indices=idxs)


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

    def plot_pca_cumulative_variance(self) -> tuple[np.ndarray, np.ndarray]:
        """Plot cumulative explained variance of the PCA basis.

        Parameters
        ----------
        save_path : str, optional
            If provided, the plot will be saved to this file.
        """

        assert self.pca_data is not None

        eigenvalues = self.pca_data.eigenvalues

        explained_variance = eigenvalues / np.sum(eigenvalues)
        cumulative_variance = np.cumsum(explained_variance)

        pcs = np.arange(1, len(cumulative_variance) + 1)

        return pcs, cumulative_variance

    def predict_waveforms_bulk(
        self,
        params: ParameterSet,
        nn: Optional[NeuralNetwork] = None,
    ) -> FDWaveforms:

        if nn is None:
            nn = self.nn
        assert nn is not None

        residuals = self.predict_residuals_bulk(params, nn)

        waveforms = self.dataset.recompose_residuals(
            residuals, params, self.downsampling_indices
        )

        return waveforms

    def predict_amplitude_phase(
        self, frequencies: np.ndarray, params: ParametersWithExtrinsic
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the amplitude and phase of a waveform.
        This function is basically the same as :meth:`predict`,
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

        # The trained band's own top edge, not the theoretical
        # `effective_srate_hz / 2`: the dataset's frequency grid can land a
        # hair past that nominal value by construction, which would
        # otherwise make this model's own last trained point count as
        # "out of band" and get zeroed out (see `[[predict-amplitude-phase-hf-edge]]`).
        trained_fmax_hz = self.dataset.frequencies_hz[-1]
        if rescaled_frequencies[-1] > trained_fmax_hz:
            if not self.extend_with_zeros_at_high_frequency:
                raise FrequencyTooHighError(
                    "This model is not configured to be extended with zeros at high frequency."
                    "Set the 'extend_with_zeros_at_high_frequency' attribute of the model to True"
                    "if that is what you want."
                )
            else:
                extend_hf = True
                high_frequency_index = int(np.searchsorted(rescaled_frequencies, trained_fmax_hz))
                hf_segment_length = len(rescaled_frequencies) - high_frequency_index
                rescaled_frequencies = rescaled_frequencies[:high_frequency_index]


        else:
            extend_hf = False

        self.parameter_ranges.check_parameters_in_ranges(params)

        intrinsic_params = params.intrinsic(self.dataset)

        residuals = self.predict_residuals_bulk(
            ParameterSet.from_list_of_waveform_parameters([intrinsic_params]), self.nn
        )

        # None unless this model was trained against a fixed reference
        # amplitude, in which case the same divisor has to be put back
        # here; see `WaveformGenerator.generate_residuals`.
        reference = self.dataset.amplitude_reference_parameters
        pn_amplitude = self.dataset.waveform_generator.post_newtonian_amplitude(
            intrinsic_params if reference is None else reference,
            self.dataset.frequencies[self.downsampling_indices.amplitude_indices],
        )
        pn_phase = self.dataset.waveform_generator.post_newtonian_phase(
            intrinsic_params,
            self.dataset.frequencies[self.downsampling_indices.phase_indices],
        )

        # downsampled amplitude array
        amp_ds = combine_residuals_amp(residuals.amplitude_residuals[0], pn_amplitude)
        phi_ds = combine_residuals_phi(residuals.phase_residuals[0], pn_phase)

        if self.timeshifts_predictor is not None:
            # add back the linear-in-frequency phase trend that
            # `remove_linear_trend` subtracted from the training residuals
            phase_freqs_hz = self.dataset.frequencies_hz[
                self.downsampling_indices.phase_indices
            ]
            time_shift = self.timeshifts_predictor.predict(
                [intrinsic_params.array]
            )[0]
            phi_ds = phi_ds + 2 * np.pi * (phase_freqs_hz - phase_freqs_hz[0]) * time_shift

        phi_ds = phi_ds + self._predicted_mode_phase0(intrinsic_params)

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

            # Glue the model-band phase onto the end of the low-frequency PN
            # segment by matching the value at the connection frequency. The
            # explicit `- resampled_phi[0]` makes this correct whether the PN
            # phase is anchored at f0 or carries the absolute stationary-phase
            # backbone (in which case a bare `+ low_f_phi[-1]` would double it).
            resampled_phi = np.concatenate((
                low_f_phi[:-1],
                resampled_phi[1:] - resampled_phi[0] + low_f_phi[-1]
            ))

        # Anchor the phase to zero at the first node so that `reference_phase`
        # continues to set the phase there. HOM models keep the per-mode
        # constant restored by `_predicted_mode_phase0`, which carries the
        # inter-mode alignment.
        if self.mode_phases_predictor is None:
            resampled_phi = resampled_phi - resampled_phi[0]

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
            + (2 * np.pi * params.time_shift) * frequencies # TODO: changed `+` to `-`
        )
        
        return amp, phi

    def predict_amplitude_phase_optimized(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
        apply_time_shift: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Amplitude and phase for one mode.

        ``apply_time_shift`` (default ``True``) adds back the
        linear-in-frequency phase trend that ``remove_linear_trend``
        stripped from the training residuals, using this model's
        :attr:`timeshifts_predictor`. :meth:`Model._hpc_waveform` and
        :meth:`Model._hpc_waveform_per_mode` pass ``False`` because they
        apply that shift themselves (with the requested time shift, which
        may be user-supplied, and the total-mass rescaling); passing
        ``True`` there would apply it twice.
        """
        # from time import perf_counter
        # t0 = perf_counter()

        assert self.downsampling_indices is not None
        assert self.nn is not None

        # t1 = perf_counter()

        # Rescale frequencies early
        rescaled_frequencies = frequencies * (params.total_mass / self.dataset.total_mass)
        eff_fmin_hz = self.dataset.effective_initial_frequency_hz
        rescaled_f_min = rescaled_frequencies[0]
        rescaled_f_max = rescaled_frequencies[-1]

        # t2 = perf_counter()

        # ----------------------------
        # Low-frequency extension
        # ----------------------------
        extend_with_pn = rescaled_f_min < eff_fmin_hz
        if extend_with_pn:
            if not self.extend_with_post_newtonian:
                raise FrequencyTooLowError("ModeModel not configured to extend with post-Newtonian waveform.")

            limit_index = np.searchsorted(rescaled_frequencies, eff_fmin_hz)
            low_freqs_hz = np.append(rescaled_frequencies[:limit_index], eff_fmin_hz)
            rescaled_frequencies = np.append(eff_fmin_hz, rescaled_frequencies[limit_index:])

            low_freqs = self.dataset.hz_to_natural_units(low_freqs_hz)
            connection_f = self.dataset.hz_to_natural_units(eff_fmin_hz)

        # t3 = perf_counter()

        # ----------------------------
        # High-frequency extension
        # ----------------------------
        # The trained band's own top edge, not the theoretical
        # `eff_srate_hz / 2`: the dataset's frequency grid can land a hair
        # past that nominal value by construction, which would otherwise
        # make this model's own last trained point count as "out of band"
        # and get zeroed out (see `[[predict-amplitude-phase-hf-edge]]`).
        trained_fmax_hz = self.dataset.frequencies_hz[-1]
        extend_hf = rescaled_f_max > trained_fmax_hz
        if extend_hf:
            if not self.extend_with_zeros_at_high_frequency:
                raise FrequencyTooHighError("ModeModel not configured to extend with zeros at high frequency.")
            high_frequency_index = np.searchsorted(rescaled_frequencies, trained_fmax_hz)
            hf_segment_length = len(rescaled_frequencies) - high_frequency_index
            rescaled_frequencies = rescaled_frequencies[:high_frequency_index]

        # t4 = perf_counter()

        # ----------------------------
        # NN Prediction & Residual Combination
        # ----------------------------
        self.parameter_ranges.check_parameters_in_ranges(params)
        intrinsic_params = params.intrinsic(self.dataset)

        residuals = self.predict_residuals_bulk(
            ParameterSet.from_list_of_waveform_parameters([intrinsic_params]), self.nn
        )
        
        # t5 = perf_counter()

        ds = self.downsampling_indices
        freqs_hz = self.dataset.frequencies_hz

        # See the note in `predict`: mirrors `generate_residuals`.
        reference = self.dataset.amplitude_reference_parameters
        pn_amp = self.dataset.waveform_generator.post_newtonian_amplitude(
            intrinsic_params if reference is None else reference,
            self.dataset.frequencies[ds.amplitude_indices],
        )
        pn_phi = self.dataset.waveform_generator.post_newtonian_phase(
            intrinsic_params, self.dataset.frequencies[ds.phase_indices]
        )

        amp_ds = combine_residuals_amp(residuals.amplitude_residuals[0], pn_amp)
        phi_ds = combine_residuals_phi(residuals.phase_residuals[0], pn_phi)

        if self.timeshifts_predictor is not None and apply_time_shift:
            # add back the linear-in-frequency phase trend that
            # `remove_linear_trend` subtracted from the training residuals
            phase_freqs_hz = freqs_hz[ds.phase_indices]
            time_shift = self.timeshifts_predictor.predict(
                [intrinsic_params.array]
            )[0]
            phi_ds = phi_ds + 2 * np.pi * (phase_freqs_hz - phase_freqs_hz[0]) * time_shift

        phi_ds = phi_ds + self._predicted_mode_phase0(intrinsic_params)

        # t6 = perf_counter()

        # ----------------------------
        # Resample to full frequency resolution
        # ----------------------------
        resample = self.downsampling_training.resample
        resampled_amp = resample(freqs_hz[ds.amplitude_indices], rescaled_frequencies, amp_ds)
        resampled_phi = resample(freqs_hz[ds.phase_indices], rescaled_frequencies, phi_ds)

        # t7 = perf_counter()

        # ----------------------------
        # Low-frequency smoothing
        # ----------------------------
        if extend_with_pn:
            f_min_connection = connection_f / 2.0
            mask = low_freqs > f_min_connection
            zero_to_one = (low_freqs[mask] - f_min_connection) / (connection_f - f_min_connection)

            low_amp = self.dataset.waveform_generator.post_newtonian_amplitude(intrinsic_params, low_freqs)
            low_phi = self.dataset.waveform_generator.post_newtonian_phase(intrinsic_params, low_freqs)

            amp_diff = resampled_amp[0] - low_amp[-1]
            low_amp[mask] += smoothing_func(zero_to_one) * amp_diff

            resampled_amp = np.concatenate((low_amp[:-1], resampled_amp[1:]))
            # See the note in `predict_amplitude_phase`: match at the connection
            # frequency so this is correct for an absolute PN phase too.
            resampled_phi = np.concatenate(
                (low_phi[:-1], resampled_phi[1:] - resampled_phi[0] + low_phi[-1])
            )

        # Anchor to zero at the first node for non-HOM models so that
        # `reference_phase` sets the phase there; HOM per-mode constants are
        # kept (they carry the inter-mode alignment).
        if self.mode_phases_predictor is None:
            resampled_phi = resampled_phi - resampled_phi[0]

        # t8 = perf_counter()

        # ----------------------------
        # High-frequency zero-padding
        # ----------------------------
        if extend_hf:
            zeros = np.zeros(hf_segment_length)
            resampled_amp = np.concatenate((resampled_amp, zeros))
            resampled_phi = np.concatenate((resampled_phi, zeros))

        # t9 = perf_counter()

        # ----------------------------
        # Final amplitude and phase
        # ----------------------------
        pre = self.dataset.mlgw_bns_prefactor(intrinsic_params.eta, params.total_mass)
        amp = resampled_amp * pre / params.distance_mpc

        phi = (
            resampled_phi
            + params.reference_phase
            + (2 * np.pi * params.time_shift) * frequencies
        )

        # t10 = perf_counter()

        # print(f"🔍 Profiling `predict_amplitude_phase_optimized`")
        # print(f"  Frequency rescaling        : {t2 - t1:.6f}s")
        # print(f"  NN + PCA                   : {t5 - t4:.6f}s")
        # print(f"  Residual + PN              : {t6 - t5:.6f}s")
        # print(f"  Resampling                 : {t7 - t6:.6f}s")
        # print(f"  Final amplitude + phase    : {t10 - t9:.6f}s")
        # print(f"  TOTAL                      : {t10 - t0:.6f}s")

        return amp, phi



    def predict(self, frequencies: np.ndarray, params: ParametersWithExtrinsic):
        r"""Calculate the waveforms in the plus and cross polarizations,
        accounting for extrinsic parameters.
        
        This function is able to yield a sensible waveform at arbitrarily 
        low frequencies, by hybridizing the EOB-trained high-frequency part
        with a Post-Newtonian approximant. 
        This feature can be turned off with the :attr:`extend_with_post_newtonian`
        parameter of the :class:`ModeModel` object.

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

        amp, phi = self.predict_amplitude_phase(frequencies, params)

        cartesian_waveform_real, cartesian_waveform_imag = combine_amp_phase(amp, phi)

        cosi = np.cos(params.inclination)
        pre_plus = (1 + cosi ** 2) / 2
        pre_cross = cosi

        # take Δt (θ) and re-add it to the phase

        return compute_polarizations(
            cartesian_waveform_real, cartesian_waveform_imag, pre_plus, pre_cross
        )
        
    def generate_teob_amp_phase(
        self,
        params: "WaveformParameters", 
        frequencies: Optional[np.ndarray] = None,
        downsampling_indices: Optional[DownsamplingIndices] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns Amplitude and Phase using TEOBResumS.

        Parameters
        ----------
        parameters : ParameterSet
            Parameters of the waveforms to generate
        downsampling_indices : DownsamplingIndices, optional
            Indices to downsample the waveforms at, by default None

        Returns
        -------
         tuple[np.ndarray, np.ndarray]
            Amplitude and phase.       
        """

        if downsampling_indices is None:
            amp_indices: Union[slice, list[int]] = slice(None)
            phi_indices: Union[slice, list[int]] = slice(None)
        else:
            amp_indices = self.downsampling_indices.amplitude_indices
            phi_indices = self.downsampling_indices.phase_indices

        amps = []
        phis = []

        _, amp, phi = self.dataset.waveform_generator.effective_one_body_waveform(
            params, frequencies
        )

        amps.append(amp[amp_indices])
        phis.append(phi[phi_indices])

        amps_reshape = amps.flatten()
        phi_reshape = phis.flatten()
        
        return amps_reshape, phi_reshape

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
            _, phis = self.predict_amplitude_phase(freqs, params)
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
    :math:`A = A_{PN} \, \Delta A`.

    The residual is the plain ratio :math:`A / A_{PN}`, not its logarithm:
    a logarithm cannot represent the sign, and the EOB mode amplitude does
    change sign (the (2,1) and (3,3) modes cross zero within the band),
    which is a physical :math:`\pi` phase flip rather than something to
    discard.

    This function is separated out just so that it can be decorated with ``@njit``.

    Parameters
    ----------
    amp : np.ndarray
    amp_pn : np.ndarray

    Returns
    -------
    np.ndarray
    """
    return amp_pn * amp


@njit
def combine_residuals_phi(phi: np.ndarray, phi_pn: np.ndarray) -> np.ndarray:
    r"""Combine amplitude residuals with their Post-Newtonian counterparts,
    according tos
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