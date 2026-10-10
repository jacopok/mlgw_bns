from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import IO, ClassVar, Optional, Type, Union

import time
import h5py
import joblib  # type: ignore
import numpy as np
import sklearn
from numpy.ma import indices
import yaml
from dacite import from_dict
from numba import njit  # type: ignore
from scipy.interpolate import CubicSpline, interp1d


def _with_fast_sklearn_config(func):
    """Run ``func`` with scikit-learn's per-call input validation disabled.

    Every ``predict`` here feeds a single, already-clean parameter row to
    fitted ``KernelRidge`` / ``MLPRegressor`` estimators. scikit-learn's
    default ``check_array`` finiteness scan and ``@validate_params``
    introspection then cost more than the linear algebra they guard. The
    context manager is scoped to the call, so it never leaks to a caller
    that uses scikit-learn itself.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with sklearn.config_context(
            assume_finite=True, skip_parameter_validation=True
        ):
            return func(*args, **kwargs)

    return wrapper

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
    JaxMLPNetwork,
    KernelRidgeNetwork,
    NeuralNetwork,
    SklearnNetwork,
)
from .principal_component_analysis import (
    PrincipalComponentAnalysisModel,
    PrincipalComponentTraining,
)
from .taylorf2 import SUN_MASS_SECONDS, smoothing_func
from .higher_order_modes import mode_to_k


#: The regressor backends a saved model may name in its metadata. The
#: network is the historical default; the kernel is far more accurate on
#: the same training data, at the cost of a prediction time that grows
#: with the training set. See :class:`~mlgw_bns.neural_network.KernelRidgeNetwork`
#: and, for a perceptron whose cost does not,
#: :class:`~mlgw_bns.neural_network.JaxMLPNetwork`.
NN_KINDS: dict[str, Type[NeuralNetwork]] = {
    "SklearnNetwork": SklearnNetwork,
    "KernelRidgeNetwork": KernelRidgeNetwork,
    "JaxMLPNetwork": JaxMLPNetwork,
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
    coalescence_phase : float
            Orbital phase at the merger, in radians: the :math:`(\ell, m)`
            mode is rotated by :math:`e^{i m \phi_c}`. Defaults to 0.
    merger_time : float
            Time of the merger, in seconds. Defaults to 0.

    Notes
    -----
    The merger is where the frequency-domain phase of the (2,2) mode
    becomes linear, at the top of the trained band; see
    :meth:`ModeModel.predict_amplitude_phase`.
    """

    mass_ratio: float
    lambda_1: float
    lambda_2: float
    chi_1: float
    chi_2: float
    distance_mpc: float
    inclination: float
    total_mass: float
    coalescence_phase: float = 0.0
    merger_time: float = 0.0

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


def mode_power_weights(
    amplitude_residuals: np.ndarray,
    frequencies_hz: np.ndarray,
    pn_amplitude: Optional[np.ndarray] = None,
    exponent: float = 1.0,
) -> np.ndarray:
    r"""Per-waveform regression weights from the integrated mode power.

    :math:`\omega_i = (P_i / \max_j P_j)^{\text{exponent}}` with
    :math:`P_i = \int A_i(f)^2 \mathrm{d}f`, so :math:`\omega \in (0, 1]`
    with a maximum of exactly 1.

    The point is the odd-:math:`m` modes. Their amplitude vanishes
    identically on the equal-mass, equal-spin locus --- for (2,1) the
    residual is exactly zero at :math:`q = 1` and grows linearly in
    :math:`q - 1` --- so the extracted phase there is
    :math:`\arg(0)`: undefined, and poorly conditioned in a whole
    neighbourhood of it. A smooth global regressor forced to fit that
    boundary layer rings across the low-:math:`q` region that does carry
    power. Weighting by the mode's own power lets the fit relax exactly
    where the target is ill-conditioned *and* contributes nothing to the
    summed waveform, without dropping any training data. See
    `arXiv:2609.03025 <https://arxiv.org/abs/2609.03025>`_, which
    introduces the same weighting for the same reason.

    Parameters
    ----------
    amplitude_residuals : np.ndarray
        Shape ``(n_waveforms, n_amplitude_nodes)``; the modelled
        amplitude quantity :math:`A_{\rm EOB} / A_{\rm PN}`.
    frequencies_hz : np.ndarray
        The amplitude nodes, in Hz, matching the second axis.
    pn_amplitude : np.ndarray, optional
        The fixed :math:`A_{\rm PN}(\theta_{\rm ref})` divisor, which
        turns the residual back into the physical mode amplitude. This
        is available whenever the dataset was built with
        ``reference_amplitude=True`` (every shipped model), since the
        divisor is then parameter-independent. If ``None``, the power is
        computed from the bare residual instead --- a proxy, since the
        per-waveform divisor has not been undone.
    exponent : float, optional
        Applied to the normalised power. ``1.0`` (the default) is the
        plain power weighting; smaller values soften it.

    Returns
    -------
    np.ndarray
        Shape ``(n_waveforms,)``, maximum 1.
    """

    amplitudes = np.asarray(amplitude_residuals, dtype=np.float64)

    if pn_amplitude is None:
        logging.warning(
            "No fixed reference amplitude available (reference_amplitude=False): "
            "weighting on the bare residual power, which does not undo the "
            "per-waveform Post-Newtonian divisor."
        )
    else:
        amplitudes = amplitudes * np.asarray(pn_amplitude, dtype=np.float64)[np.newaxis, :]

    power = np.trapezoid(amplitudes ** 2, np.asarray(frequencies_hz), axis=1)

    maximum = np.max(power)
    if not np.isfinite(maximum) or maximum <= 0.0:
        logging.warning(
            "Mode power is degenerate (max = %s); falling back to uniform weights",
            maximum,
        )
        return np.ones(len(power))

    return (power / maximum) ** exponent


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
        power_weighting: bool = True,
        power_weight_exponent: float = 1.0,
    ):

        self.reference_amplitude = reference_amplitude
        #: Whether :meth:`train_nn` weights each training waveform by its
        #: integrated mode power (see :func:`mode_power_weights`). Only
        #: odd-``m`` modes are affected --- their amplitude vanishes on
        #: the equal-mass, equal-spin locus, which is what makes the
        #: weighting necessary; even-``m`` modes are fit unweighted, and
        #: so is the (2,2)-only model.
        self.power_weighting = power_weighting
        #: Exponent applied to the normalised power in
        #: :func:`mode_power_weights`.
        self.power_weight_exponent = power_weight_exponent
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
            'power_weighting': self.power_weighting,
            'power_weight_exponent': self.power_weight_exponent,
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

    def generate(
        self,
        training_downsampling_dataset_size: Optional[int] = 64,
        training_pca_dataset_size: Optional[int] = 256,
        training_nn_dataset_size: Optional[int] = 256,
        precomputed_residuals: Optional[tuple] = None,
        n_jobs: int = 1,
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
        precomputed_residuals : tuple, optional
                ``(freq_downsampled_natural, ParameterSet, Residuals)`` for
                this mode, already downsampled to
                :attr:`downsampling_indices`, sized
                ``max(training_pca_dataset_size, training_nn_dataset_size)``.
                Supplied by :meth:`~mlgw_bns.model.Model.generate` from one
                shared multi-mode EOB sweep; when given, the per-mode
                ``Dataset.generate_residuals`` calls for the PCA and NN
                training sets are skipped.
        n_jobs : int, optional
                Number of parallel worker processes for every EOB sweep in
                this call. Sequential (``1``) by default -- parallelism is
                opt-in; pass a higher value explicitly to use multiple
                workers.

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
            self.downsampling_training.n_jobs = n_jobs
            self.downsampling_indices = self.downsampling_training.train(
                training_downsampling_dataset_size
            )
        else:
            assert self.downsampling_indices is not None

        self.log_expected_training_memory(
            training_pca_dataset_size, training_nn_dataset_size
        )

        if training_pca_dataset_size is not None:
            logging.info("Training the PCA")
            self.pca_training = PrincipalComponentTraining(
                self.dataset, self.downsampling_indices, self.pca_components_number
            )
            if precomputed_residuals is not None:
                _, all_parameters, all_residuals = precomputed_residuals
                self.pca_data = self.pca_training.train_on(
                    all_residuals[:training_pca_dataset_size]
                )
            else:
                self.pca_data = self.pca_training.train(
                    training_pca_dataset_size, n_jobs=n_jobs
                )
        else:
            assert self.pca_data is not None

        if training_nn_dataset_size is not None:
            if precomputed_residuals is not None:
                _, all_parameters, all_residuals = precomputed_residuals
                parameters = self.dataset.parameter_set_cls(
                    all_parameters.parameter_array[:training_nn_dataset_size]
                )
                residuals = all_residuals[:training_nn_dataset_size]
            else:
                logging.info("Generating the training dataset")
                _, parameters, residuals = self.dataset.generate_residuals(
                    training_nn_dataset_size, self.downsampling_indices, n_jobs=n_jobs
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

    def save(self, include_training_data: bool = True) -> None:
        """Save this model to the files derived from :attr:`filename`.

        Parameters
        ----------
        include_training_data : bool, optional
                Whether to also persist the training residuals and
                parameters. Defaults to ``True``.
        """
        self.save_metadata()
        self.save_arrays(include_training_data)
        if self.nn is not None:
            self.nn.save(self.filename_nn)

    def load(
        self,
        streams: Optional[tuple[IO[bytes], IO[bytes], IO[bytes]]] = None,
    ) -> None:
        """Load model from the files present in the current folder.

        Parameters
        ----------
        streams: tuple[IO[bytes], IO[bytes], IO[bytes]], optional
                For internal use (specifically, loading the default model):
                the metadata, arrays and network streams.
                Defaults to None (look in the current folder).
        """

        if streams is not None:
            stream_meta: Union[IO[bytes], None]
            h5_source: Union[IO[bytes], str]
            filename_nn: Union[IO[bytes], str]

            stream_meta, h5_source, filename_nn = streams
            ignore_warnings = True
        else:
            stream_meta = None
            h5_source = self.filename_arrays
            filename_nn = self.filename_nn
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


    def _training_power_weights(
        self, amplitude_residuals: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Power weights for the training set (or for the waveforms of these
        ``amplitude_residuals``), or ``None`` for a flat fit.

        Returns ``None`` --- meaning "fit unweighted", byte-identical to
        the behaviour before power weighting existed --- unless
        :attr:`power_weighting` is set *and* this is an odd-``m`` mode.
        Even-``m`` modes have no vanishing-amplitude locus, and measuring
        their weights on the shipped model gives an almost flat
        distribution (0.42-0.99 for (4,4), 0.73-0.99 for (2,2)), so
        weighting them would perturb two modes that are already accurate
        for no benefit. See :func:`mode_power_weights`.
        """

        if not self.power_weighting or self.mode is None or self.mode.m % 2 == 0:
            return None

        if amplitude_residuals is None:
            assert self.training_dataset is not None
            amplitude_residuals = self.training_dataset.amplitude_residuals
        assert self.downsampling_indices is not None

        amplitude_indices = self.downsampling_indices.amplitude_indices

        # Mirrors `WaveformGenerator.generate_residuals`: with a fixed
        # amplitude reference the divisor is parameter-independent, so
        # multiplying it back in recovers the physical mode amplitude.
        reference = self.dataset.amplitude_reference_parameters
        pn_amplitude = (
            None
            if reference is None
            else self.dataset.waveform_generator.post_newtonian_amplitude(
                reference, self.dataset.frequencies[amplitude_indices]
            )
        )

        return mode_power_weights(
            amplitude_residuals,
            self.dataset.frequencies_hz[amplitude_indices],
            pn_amplitude=pn_amplitude,
            exponent=self.power_weight_exponent,
        )

    def train_nn(
        self,
        hyper: Hyperparameters,
        indices: Union[list[int], slice] = slice(None),
        checkpoint: Optional[str] = None,
    ) -> NeuralNetwork:
        """Train a network on the training dataset (see :meth:`fit_nn`).

        Parameters
        ----------
        hyper : Hyperparameters
            Hyperparameters to be used in the initialization
            of the network.
        indices : Union[list[int], slice], optional
            Indices used to perform a selection of a subsection
            of the training data; by default ``slice(None)``
            which means all available training data is used.
        checkpoint : str, optional
            Where a :class:`~mlgw_bns.neural_network.JaxMLPNetwork` keeps
            the state of its training; see :meth:`fit_nn`.

        Notes
        -----
        For odd-``m`` modes the fit is weighted by each waveform's
        integrated mode power (:meth:`_training_power_weights`), which
        keeps the near-vanishing-amplitude waveforms around equal mass
        from distorting it. Every other case fits unweighted.

        Returns
        -------
        NeuralNetwork
            Trained network.
        """
        assert self.training_parameters is not None

        sample_weight = self._training_power_weights()
        if sample_weight is not None:
            sample_weight = sample_weight[indices]

        return self.fit_nn(
            hyper,
            self.training_parameters.parameter_array[indices],
            self.reduced_residuals[indices],
            sample_weight,
            checkpoint=checkpoint,
        )

    def fit_nn(
        self,
        hyper: Hyperparameters,
        parameters: np.ndarray,
        reduced_residuals: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        checkpoint: Optional[str] = None,
    ) -> NeuralNetwork:
        """Fit a network of kind :attr:`nn_kind` from ``parameters``
        ``(N, 5)`` to the principal components of their residuals
        ``(N, pca_components_number)`` (as
        :meth:`PrincipalComponentAnalysisModel.reduce_data
        <mlgw_bns.principal_component_analysis.PrincipalComponentAnalysisModel.reduce_data>`
        gives them), with these power weights (see
        :meth:`_training_power_weights`).

        This is :meth:`train_nn` without the training dataset, for training
        sets reduced a piece at a time (as
        :class:`~mlgw_bns.modes_dataset.ShardedModesDataset` does).

        Parameters
        ----------
        checkpoint : str, optional
            Where a :class:`~mlgw_bns.neural_network.JaxMLPNetwork` keeps
            the state of its training, to resume it after an interruption
            (other kinds take none).

        Returns
        -------
        NeuralNetwork
            Trained network.
        """
        assert self.pca_data is not None

        training_residuals = (
            reduced_residuals
            * (self.pca_data.eigenvalues ** hyper.pc_exponent)[np.newaxis, :]
        )

        nn = self.nn_kind(hyper)
        fit_options: dict = {}
        if isinstance(nn, JaxMLPNetwork):
            # what a unit of each output is in the residuals, for its loss
            fit_options["output_units"] = self.pca_data.principal_components_scaling / (
                self.pca_data.eigenvalues ** hyper.pc_exponent
            )
            fit_options["checkpoint"] = checkpoint
        elif checkpoint is not None:
            raise ValueError(f"{type(nn).__name__} trainings are not checkpointed")

        start_time = time.time()

        nn.fit(parameters, training_residuals, sample_weight=sample_weight, **fit_options)

        logging.info(
            "Training the network on %i %s waveforms took %.2f seconds "
            "(peak memory usage so far: %s)",
            len(training_residuals),
            "unweighted"
            if sample_weight is None
            else f"power-weighted (median omega {np.median(sample_weight):.3g})",
            time.time() - start_time,
            format_bytes(peak_memory_usage()),
        )

        return nn

    def default_hyperparameters(self, n_train: int) -> Hyperparameters:
        """The hyperparameters :meth:`set_hyper_and_train_nn` uses unless
        given others, for :attr:`nn_kind` and this mode."""
        if self.nn_kind is KernelRidgeNetwork:
            return Hyperparameters.default_kernel_ridge(n_train, mode=self.mode)
        if self.nn_kind is JaxMLPNetwork:
            return Hyperparameters.default_jax_mlp(n_train)
        return Hyperparameters.default(n_train)

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
            hyper = self.default_hyperparameters(len(self.training_dataset))

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

    @property
    def m(self) -> int:
        """Azimuthal number of this mode; a mode-less model is the (2,2)."""
        return 2 if self.mode is None else self.mode.m

    def _nodes(self, intrinsic_params) -> tuple[np.ndarray, np.ndarray]:
        """Amplitude and phase at the downsampling nodes, at the reference mass.

        The phase is referenced at :math:`f_0` (see
        :func:`~mlgw_bns.data_management.re_reference`), as in training.
        """
        assert self.downsampling_indices is not None
        assert self.nn is not None
        residuals = self.predict_residuals_bulk(
            ParameterSet.from_list_of_waveform_parameters([intrinsic_params]), self.nn
        )
        ds = self.downsampling_indices
        # None unless this model was trained against a fixed reference
        # amplitude, in which case the same divisor has to be put back
        # here; see `WaveformGenerator.generate_residuals`.
        reference = self.dataset.amplitude_reference_parameters
        pn_amp = self.dataset.waveform_generator.post_newtonian_amplitude(
            intrinsic_params if reference is None else reference,
            self.dataset.frequencies[ds.amplitude_indices],
        )
        pn_phi = self.dataset.waveform_generator.post_newtonian_phase(
            intrinsic_params, self.dataset.frequencies[ds.phase_indices]
        )
        return (
            combine_residuals_amp(residuals.amplitude_residuals[0], pn_amp),
            combine_residuals_phi(residuals.phase_residuals[0], pn_phi),
        )

    def merger_reference(
        self, intrinsic_params, phase_nodes: Optional[np.ndarray] = None
    ) -> tuple[float, float]:
        r"""The line :math:`s f + b` this mode's phase tends to at the merger.

        :math:`s` and :math:`b` are the slope and intercept of the tangent
        to the phase (the not-a-knot spline through the phase nodes) at the
        top of the trained band, in Hz at the reference mass. For the
        (2,2), whose frequency-domain phase is linear after the merger,
        :math:`-s/2\pi` is the merger time and :math:`b` the (2,2) phase
        there; :class:`~mlgw_bns.model.Model` hands the (2,2) reference to
        every mode, see :meth:`predict_amplitude_phase`.

        Parameters
        ----------
        intrinsic_params : WaveformParameters
        phase_nodes : np.ndarray, optional
            The phase at the nodes, if already computed.

        Returns
        -------
        tuple[float, float]
            ``(slope, intercept)``.
        """
        assert self.downsampling_indices is not None
        if phase_nodes is None:
            _, phase_nodes = self._nodes(intrinsic_params)
        knots = self.dataset.frequencies_hz[self.downsampling_indices.phase_indices]
        slope = float(CubicSpline(knots, phase_nodes)(knots[-1], 1))
        return slope, float(phase_nodes[-1] - slope * knots[-1])

    @_with_fast_sklearn_config
    def predict_amplitude_phase(
        self,
        frequencies: np.ndarray,
        params: ParametersWithExtrinsic,
        merger_reference: Optional[tuple[float, float]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Amplitude and phase of this mode.

        The multipole is :math:`\tilde{h}_{\ell m} = A e^{i \phi}`. The
        phase is referenced to the merger: with the (2,2) reference line
        :math:`s f + b` (:meth:`merger_reference`, of the rescaled
        frequency), the mode's phase is

        .. math::
            \phi_{\ell m}(f) - s f - \frac{m}{2} b
            + m \phi_c - 2 \pi f t_c,

        that is, a time shift and an orbital phase rotation, the same for
        all the modes, which put the merger of the (2,2) at
        :math:`t_c` = ``params.merger_time`` with a phase
        :math:`2\phi_c` (``params.coalescence_phase``) there.

        Parameters
        ----------
        frequencies : np.ndarray
            Increasing frequencies, in Hz.
        params : ParametersWithExtrinsic
        merger_reference : tuple[float, float], optional
            ``(slope, intercept)`` of the (2,2) of the same parameters, from
            its :meth:`merger_reference`. Defaults to this mode's own, which
            is right for the (2,2) itself (and for a model trained on a
            single mode, since its residuals are referenced to itself).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Amplitude and phase.

        Raises
        ------
        FrequencyTooLowError, FrequencyTooHighError
            If ``frequencies`` extend below (above) the trained band and
            the model is not configured to extend it there.
        """

        assert self.downsampling_indices is not None

        ratio = params.total_mass / self.dataset.total_mass
        rescaled_all = frequencies * ratio
        rescaled_frequencies = rescaled_all
        eff_fmin_hz = self.dataset.effective_initial_frequency_hz

        # ----------------------------
        # Low-frequency extension
        # ----------------------------
        extend_with_pn = rescaled_frequencies[0] < eff_fmin_hz
        if extend_with_pn:
            if not self.extend_with_post_newtonian:
                raise FrequencyTooLowError(
                    "ModeModel not configured to extend with post-Newtonian waveform."
                )
            limit_index = np.searchsorted(rescaled_frequencies, eff_fmin_hz)
            # the connection point is added to both pieces, so that the PN
            # one can be glued on without a discontinuity in phase
            low_freqs_hz = np.append(rescaled_frequencies[:limit_index], eff_fmin_hz)
            rescaled_frequencies = np.append(
                eff_fmin_hz, rescaled_frequencies[limit_index:]
            )
            low_freqs = self.dataset.hz_to_natural_units(low_freqs_hz)
            connection_f = self.dataset.hz_to_natural_units(eff_fmin_hz)

        # ----------------------------
        # High-frequency extension
        # ----------------------------
        # The trained band's own top edge, not the theoretical
        # `eff_srate_hz / 2`: the dataset's frequency grid can land a hair
        # past that nominal value by construction, which would otherwise
        # make this model's own last trained point count as "out of band"
        # and get zeroed out.
        trained_fmax_hz = self.dataset.frequencies_hz[-1]
        hf_segment_length = 0
        if rescaled_frequencies[-1] > trained_fmax_hz:
            if not self.extend_with_zeros_at_high_frequency:
                raise FrequencyTooHighError(
                    "ModeModel not configured to extend with zeros at high frequency."
                )
            high_frequency_index = np.searchsorted(rescaled_frequencies, trained_fmax_hz)
            hf_segment_length = len(rescaled_frequencies) - high_frequency_index
            rescaled_frequencies = rescaled_frequencies[:high_frequency_index]

        # ----------------------------
        # Nodes, and resampling
        # ----------------------------
        self.parameter_ranges.check_parameters_in_ranges(params)
        intrinsic_params = params.intrinsic(self.dataset)
        amp_ds, phi_ds = self._nodes(intrinsic_params)
        if merger_reference is None:
            merger_reference = self.merger_reference(intrinsic_params, phi_ds)
        slope, intercept = merger_reference

        ds = self.downsampling_indices
        freqs_hz = self.dataset.frequencies_hz
        resample = self.downsampling_training.resample
        resampled_amp = resample(freqs_hz[ds.amplitude_indices], rescaled_frequencies, amp_ds)
        resampled_phi = resample(freqs_hz[ds.phase_indices], rescaled_frequencies, phi_ds)

        if extend_with_pn:
            f_min_connection = connection_f / 2.0
            mask = low_freqs > f_min_connection
            zero_to_one = (low_freqs[mask] - f_min_connection) / (connection_f - f_min_connection)

            low_amp = self.dataset.waveform_generator.post_newtonian_amplitude(intrinsic_params, low_freqs)
            low_phi = self.dataset.waveform_generator.post_newtonian_phase(intrinsic_params, low_freqs)

            amp_diff = resampled_amp[0] - low_amp[-1]
            low_amp[mask] += smoothing_func(zero_to_one) * amp_diff

            resampled_amp = np.concatenate((low_amp[:-1], resampled_amp[1:]))
            # the PN segment is shifted to meet the band at the connection
            resampled_phi = np.concatenate(
                (low_phi[:-1] - low_phi[-1] + resampled_phi[0], resampled_phi[1:])
            )

        # ----------------------------
        # Merger reference, high-frequency zeros, extrinsic parameters
        # ----------------------------
        resampled_phi = (
            resampled_phi
            - slope * rescaled_all[: len(resampled_phi)]
            - self.m / 2 * intercept
        )
        if hf_segment_length:
            zeros = np.zeros(hf_segment_length)
            resampled_amp = np.concatenate((resampled_amp, zeros))
            resampled_phi = np.concatenate((resampled_phi, zeros))

        pre = self.dataset.mlgw_bns_prefactor(intrinsic_params.eta, params.total_mass)
        amp = resampled_amp * pre / params.distance_mpc
        phi = (
            resampled_phi
            + self.m * params.coalescence_phase
            - (2 * np.pi * params.merger_time) * frequencies
        )
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