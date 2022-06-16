"""Functionality for the generation of a training dataset.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, ClassVar, Optional, Type, Union

import h5py
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm  # type: ignore

from .data_management import (
    DownsamplingIndices,
    FDWaveforms,
    ParameterRanges,
    Residuals,
    SavableData,
    phase_unwrapping,
)
from .multibanding import reduced_frequency_array

# from .downsampling_interpolation import DownsamplingTraining
from .taylorf2 import (
    SUN_MASS_SECONDS,
    amplitude_3h_post_newtonian,
    compute_delta_lambda,
    compute_lambda_tilde,
    phase_5h_post_newtonian_tidal,
)

TF2_BASE: float = 3.668693487138444e-19
# ( Msun * G / c**3)**(5/6) * Hz**(-7/6) * c / Mpc / s
AMP_SI_BASE: float = 4.2425873413901263e24
# Mpc / Msun**2 / Hz


class WaveformGenerator(ABC):
    """Generator of theoretical waveforms
    according to some pre-existing model.

    This is an abstract class: users may extend ``mlgw_bns`` by
    subclassing this and training a networks using that new waveform generator.

    This can be accomplished by implementing the methods
    :meth:`post_newtonian_amplitude`,
    :meth:`post_newtonian_phase` and
    :meth:`effective_one_body_waveform`.

    Users may wish to leave the Post-Newtonian model already implemented here
    and only switch TEOBResumS to another waveform template:
    the easiest way to accomplish this is to subclass :class:`BarePostNewtonianGenerator`
    and only override its :meth:`effective_one_body_waveform` method.
    """

    def __init__(self):
        self.frequencies: Optional[np.ndarray] = None

    @abstractmethod
    def post_newtonian_amplitude(
        self, params: "WaveformParameters", frequencies: np.ndarray
    ) -> np.ndarray:
        r"""Amplitude of the Fourier transform of the waveform computed
        at arbitrary frequencies.
        This should be implemented in some fast, closed-form way.
        The speed of the overall model relies on the evaluation
        of this function not taking too long.

        Parameters
        ----------
        params : WaveformParameters
                Parameters of the binary system for which to generate the waveform.
        frequencies : np.ndarray
                Array of frequencies at which to compute the amplitude.
                Should be given in mass-rescaled natural units; they will be
                passed to :func:`WaveformParameters.taylor_f2`.

        Returns
        -------
        amplitude : np.ndarray
                Amplitude of the Fourier transform of the waveform,
                given with the natural-units convention
                :math:`|\widetilde{h}_+(f)| r \eta / M^2`,
                where we are using :math:`c=  G = 1` natural units,
                :math:`r` is the distance to the binary,
                :math:`\eta` is the symmetric mass ratio,
                :math:`M` is the total mass of the binary.
        """

    @abstractmethod
    def post_newtonian_phase(
        self, params: "WaveformParameters", frequencies: np.ndarray
    ) -> np.ndarray:
        r"""Phase of the Fourier transform of the waveform computed
        at arbitrary frequencies.
        This should be implemented in some fast, closed-form way.
        The speed of the overall model relies on the evaluation
        of this not taking too long.

        Parameters
        ----------
        params : WaveformParameters
                Parameters of the binary system for which to generate the waveform.
        frequencies : np.ndarray
                Array of frequencies at which to compute the phase.
                Should be given in mass-rescaled natural units; they will be
                passed to :func:`WaveformParameters.taylor_f2`.

        Returns
        -------
        phase : np.ndarray
                Phase of the Fourier transform of the waveform,
                specifically the phase of the plus polarization in radians.
                At the :math:`(\ell = 2, m =2)` multipole,
                the phase of the cross-polarization will simply by :math:`\pi/2`
                plus this one.
        """

    @abstractmethod
    def effective_one_body_waveform(
        self, params: "WaveformParameters", frequencies: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Waveform computed according to the comparatively slower
        effective-one-body method.

        Parameters
        ----------
        params : WaveformParameters
                Parameters of the binary system for which to generate the waveform.
        frequencies : np.ndarray, optional
                Frequencies at which to compute the waveform, in natural units.
                Defaults to None, which means the EOB generator will choose
                the frequencies at which to compute the waveform.

        Returns
        -------
        frequencies : np.ndarray
                Frequencies at which the waveform is given, in natural units:
                the quantity here is :math:`Mf` (with :math:`G = c = 1`).
        amplitude : np.ndarray
                Amplitude of the plus-polarized waveform.
                The normalization for the amplitude is the same as discussed
                in :func:`post_newtonian_amplitude`.
        phase : np.ndarray
                Phase of the plus-polarized waveform,
                in radians, given as a continuously-varying array
                (so, not constrained between 0 and 2pi).
        """

    def generate_residuals(
        self,
        params: "WaveformParameters",
        frequencies: Optional[np.ndarray] = None,
        downsampling_indices: Optional[DownsamplingIndices] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the residuals of the :func:`effective_one_body_waveform`
        from the Post-Newtonian one computed with
        :func:`post_newtonian_amplitude` and
        :func:`post_newtonian_phase`.

        Residuals are defined as discussed in :class:`Dataset`.

        Parameters
        ----------
        params : WaveformParameters
                Parameters for which to compute the residuals.
        frequencies : np.ndarray, optional
                Frequencies at which to compute the residuals,
                in natural units.
                If this parameter is given, the downsampling_indices
                should index this frequency array.
                Defaults to None, meaning that the frequencies computed
                by the :meth:`effective_one_body_waveform` method
                are used.
        downsampling_indices : Optional[DownsamplingIndices]
                Indices at which to compute the residuals.
                If not provided (default) the waveform is given at
                all indices corresponding to the default FFT grid.
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
                Amplitude residuals and phase residuals.
        """

        (frequencies_eob, amplitude_eob, phase_eob) = self.effective_one_body_waveform(
            params, frequencies
        )

        if downsampling_indices:
            amp_indices, phi_indices = downsampling_indices
            amp_frequencies = frequencies_eob[amp_indices]
            phi_frequencies = frequencies_eob[phi_indices]

            amplitude_eob = amplitude_eob[amp_indices]
            phase_eob = phase_eob[phi_indices]
        else:
            amp_frequencies = frequencies_eob
            phi_frequencies = frequencies_eob

        amplitude_pn = self.post_newtonian_amplitude(params, amp_frequencies)
        phase_pn = self.post_newtonian_phase(params, phi_frequencies)

        assert np.all(amplitude_pn > 0)
        assert np.all(amplitude_eob > 0)

        return (np.log(amplitude_eob / amplitude_pn), phase_eob - phase_pn)


class BarePostNewtonianGenerator(WaveformGenerator):
    """Generate waveforms with
    3.5PN-accurate post-newtonian amplitude,
    and 5.5PN-accurate post-newtonian phase
    with 7.5PN-accurate tidal terms.

    This classes' :meth:`effective_one_body_waveform` method
    is not implemented: it is used as a fallback when
    the ``EOBRun_module`` python wrapper for TEOBResumS
    cannot be imported.
    """

    def post_newtonian_amplitude(
        self, params: "WaveformParameters", frequencies: np.ndarray
    ) -> np.ndarray:
        return amplitude_3h_post_newtonian(params, frequencies)

    def post_newtonian_phase(
        self, params: "WaveformParameters", frequencies: np.ndarray
    ) -> np.ndarray:
        return phase_5h_post_newtonian_tidal(params, frequencies)

    def effective_one_body_waveform(
        self, params: "WaveformParameters", frequencies: Optional[np.ndarray] = None
    ):
        raise NotImplementedError(
            "This generator does not include the possibility "
            "to generate effective one body waveforms"
        )


class TEOBResumSGenerator(BarePostNewtonianGenerator):
    """Generate waveforms using the
    TEOBResumS effective-one-body code"""

    def __init__(self, eobrun_callable: Callable[[dict], tuple[np.ndarray, ...]]):
        super().__init__()
        self.eobrun_callable = eobrun_callable

    def effective_one_body_waveform(
        self, params: "WaveformParameters", frequencies: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Generate an EOB waveform with TEOB.

        Examples
        --------
        >>> from EOBRun_module import EOBRunPy
        >>> tg = TEOBResumSGenerator(EOBRunPy)
        >>> p = WaveformParameters(1, 300, 300, .3, -.3, Dataset(20., 4096.))
        >>> f, amp, phi = tg.effective_one_body_waveform(p)
        """

        par_dict: dict = params.teobresums()

        # tweak initial frequency backward by a few samples
        # this is needed because of a bug in TEOBResumS
        # causing the phase evolution not to behave properly
        # at the beginning of integration
        # TODO remove this once the TEOB bug is fixed

        n_additional = 256
        f_0 = par_dict["initial_frequency"]
        delta_f = par_dict["df"]
        new_f0 = f_0 - delta_f * n_additional
        par_dict["initial_frequency"] = new_f0

        to_slice = (
            slice(-len(frequencies), None)
            if frequencies is not None
            else slice(n_additional, None)
        )

        if frequencies is not None:
            frequencies_list = list(
                np.insert(
                    frequencies,
                    0,
                    np.arange(f_0 - delta_f * n_additional, f_0, step=delta_f),
                )
            )
            par_dict.pop("df")
            par_dict["interp_freqs"] = "yes"
            par_dict["freqs"] = frequencies_list

        f_spa, rhpf, ihpf, _, _ = self.eobrun_callable(par_dict)

        f_spa = f_spa[to_slice]
        # f_spa = f_spa

        waveform = (rhpf - 1j * ihpf)[to_slice]
        # waveform = rhpf - 1j * ihpf

        amplitude, phase = phase_unwrapping(waveform)

        return (f_spa, amplitude, phase)


@dataclass
class WaveformParameters:

    r"""Parameters for a single waveform.

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
    dataset : Dataset
            Reference dataset, which includes information
            required for the generation of the waveform,
            such as the initial frequency or the reference total mass.

    Class Attributes
    ----------------
    number_of_parameters: int
            How many intrinsic parameters are modelled.
            This class variable should equal the number of other
            floating-point attributes the class has,
            it is included for convenience.
    """

    mass_ratio: float
    lambda_1: float
    lambda_2: float
    chi_1: float
    chi_2: float
    dataset: "Dataset"

    number_of_parameters: ClassVar[int] = 5

    def almost_equal_to(self, other: object):
        """Check for equality with another set of parameters,
        accounting for imprecise floats.
        """
        if not isinstance(other, WaveformParameters):
            return NotImplemented
        return self.dataset is other.dataset and all(
            np.isclose(getattr(self, param), getattr(other, param), atol=0.0, rtol=1e-6)
            for param in ["mass_ratio", "lambda_1", "lambda_2", "chi_1", "chi_1"]
        )

    @property
    def eta(self):
        r"""Symmetric mass ratio of the binary.

        It is defined as :math:`\eta = \mu / M`, where
        :math:`\mu  = (1 / m_1 + 1/ m_2)^{-1}`
        and :math:`M = m_1 + m_2`.

        It can also be expressed as
        :math:`\eta = m_1 m_2 / (m_1 + m_2)^2 = q / (1+q)^2`,
        where :math:`q = m_1 / m_2` is the mass ratio.

        It is also sometimes denoted as :math:`\nu`.
        It goes from 0 in the test-mass limit (one mass vanishing)
        to :math:`1/4` in the equal-mass limit.
        """
        return self.mass_ratio / (1.0 + self.mass_ratio) ** 2

    @property
    def m_1(self):
        """Mass of the heavier star in the system, in solar masses."""
        return self.dataset.total_mass / (1 + 1 / self.mass_ratio)

    @property
    def m_2(self):
        """Mass of the lighter star in the system, in solar masses."""
        return self.dataset.total_mass / (1 + self.mass_ratio)

    @property
    def lambdatilde(self):
        r"""Symmetrized tidal deformability parameter :math:`\widetilde\Lambda`,
        which gives the largest contribution to the waveform phase.
        For the precise definition see equation 5 of `this paper <http://arxiv.org/abs/1805.11579>`__."""
        return compute_lambda_tilde(self.m_1, self.m_2, self.lambda_1, self.lambda_2)

    @property
    def dlambda(self):
        r"""Antisymmetrized tidal deformability parameter :math:`\delta \widetilde\Lambda`,
        which gives the next-to-largest contribution to the waveform phase.
        For the precise definition see equation 27 of `this paper <http://arxiv.org/abs/2102.00017>`__."""
        return compute_delta_lambda(self.m_1, self.m_2, self.lambda_1, self.lambda_2)

    def teobresums(
        self, use_effective_frequencies: bool = True
    ) -> dict[str, Union[float, int, str]]:
        """Parameter dictionary in a format compatible with
        TEOBResumS.

        The parameters are all converted to natural units.
        """

        if use_effective_frequencies:
            initial_freq = (
                self.dataset.effective_initial_frequency_hz
                * self.dataset.mass_sum_seconds
            )
            srate = self.dataset.effective_srate_hz * self.dataset.mass_sum_seconds
        else:
            initial_freq = (
                self.dataset.initial_frequency_hz * self.dataset.mass_sum_seconds
            )
            srate = self.dataset.srate_hz * self.dataset.mass_sum_seconds

        return {
            "q": self.mass_ratio,
            "LambdaAl2": self.lambda_1,
            "LambdaBl2": self.lambda_2,
            "chi1": self.chi_1,
            "chi2": self.chi_2,
            "M": self.dataset.total_mass,
            "distance": 1.0,
            "initial_frequency": initial_freq,
            "use_geometric_units": "yes",
            "interp_uniform_grid": "no",
            "domain": 1,  # Fourier domain
            "srate_interp": srate,
            "df": self.dataset.delta_f_hz * self.dataset.mass_sum_seconds,
            "inclination": 0.0,
            "output_hpc": "no",
            "time_shift_FD": "yes",
            "ode_tmax": 1e12,
        }

    def taylor_f2(
        self, frequencies: np.ndarray
    ) -> dict[str, Union[float, int, np.ndarray]]:
        """Parameter dictionary in a format compatible with
        the custom implemnentation of TaylorF2 implemented within ``mlgw_bns``.

        Parameters
        ----------
        frequencies : np.ndarray
                The frequencies where to compute the
                waveform, to be given in natural units
        """

        return {
            "f": frequencies / self.dataset.mass_sum_seconds,
            "q": self.mass_ratio,
            "s1x": 0,
            "s1y": 0,
            "s1z": self.chi_1,
            "s2y": 0,
            "s2x": 0,
            "s2z": self.chi_2,
            "lambda1": self.lambda_1,
            "lambda2": self.lambda_2,
            "f_min": self.dataset.effective_initial_frequency_hz,
            "phi_ref": 0,
            "phaseorder": 11,
            "tidalorder": 15,
            "usenewtides": 1,
            "usequadrupolemonopole": 1,
            "mtot": self.dataset.total_mass,
            "s1x": 0,
            "s1y": 0,
            "s2x": 0,
            "s2y": 0,
            "Deff": 1.0,
            "phiRef": 0.0,
            "timeShift": 0.0,
            "iota": 0.0,
        }

    @property
    def array(self) -> np.ndarray:
        r"""Represent the parameters as a numpy array.

        Returns
        -------
        np.ndarray
            Array representation of the parameters, specifically
            :math:`[q, \Lambda_1, \Lambda_2, \chi_1, \chi_2]`.
        """
        return np.array(
            [self.mass_ratio, self.lambda_1, self.lambda_2, self.chi_1, self.chi_2]
        )


@dataclass
class ParameterSet(SavableData):
    """Dataclass which contains an array of parameters for waveform generation.

    The meaning of each row of parameters is the same which is described
    in :meth:`WaveformParameters.array`.

    Parameters
    ----------
    parameter_array: np.ndarray
            Array with shape
            ``(number_of_parameter_tuples, number_of_parameters)``,
            where ``number_of_parameters==5`` currently.
    """

    parameter_array: np.ndarray

    group_name: ClassVar[str] = "training_parameters"

    def __post_init__(self):
        assert self.parameter_array.shape[1] == WaveformParameters.number_of_parameters

    def __getitem__(self, key):
        """Allow for the slicing of this object to be a closed operation."""
        return self.__class__(self.parameter_array[key])

    def waveform_parameters(self, dataset: Dataset) -> list[WaveformParameters]:
        """Return a list of WaveformParameters.

        Parameters
        ----------
        dataset : Dataset
            Dataset, required for the initialization of :class:`WaveformParameters`.

        Returns
        -------
        list[WaveformParameters]

        Examples
        --------

        We generate a :class:`ParameterSet` with a single array of parameters ,

        >>> param_set = ParameterSet(np.array([[1, 2, 3, 4, 5]]))
        >>> dataset = Dataset(initial_frequency_hz=20., srate_hz=4096.)
        >>> wp_list = param_set.waveform_parameters(dataset)
        >>> print(wp_list[0].array)
        [1 2 3 4 5]
        """

        return [WaveformParameters(*params, dataset) for params in self.parameter_array]  # type: ignore

    @classmethod
    def from_parameter_generator(
        cls, parameter_generator: "ParameterGenerator", number_of_parameter_tuples: int
    ):
        """Make a set of new parameter tuples
        by randomly generating them with a :class:`ParameterGenerator`.

        Parameters
        ----------
        parameter_generator : ParameterGenerator
            To generate the tuples.
        number_of_parameter_tuples : int
            How many tuples to generate.
        """

        param_array = np.array(
            [next(parameter_generator).array for _ in range(number_of_parameter_tuples)]
        )

        return cls(param_array)

    @classmethod
    def from_list_of_waveform_parameters(cls, wf_params_list: list[WaveformParameters]):
        return cls(np.array([params.array for params in wf_params_list]))


class ParameterGenerator(ABC, Iterator):
    """Generic generator of parameters for new waveforms
    to be used for training.

    Parameters
    ----------
    dataset: Dataset
            Dataset to which the generated parameters will refer.
            This parameter is required because the parameters must include
            things such as the initial frequency, which are properties of the dataset.
    seed: Optional[int]
            Seed for the random number generator, optional.
            If it is not given, the :attr:`Dataset.seed_sequence` of the
            dataset is used.

    Class Attributes
    ----------------
    number_of_free_parameters: int
            Number of parameter which will vary during the random parameter generation.
    """

    number_of_free_parameters: int = 5
    parameter_set_cls = ParameterSet

    def __init__(self, dataset: "Dataset", seed: Optional[int] = None, **kwargs):

        self.dataset = dataset

        if seed is None:
            self.rng = default_rng(
                self.dataset.seed_sequence.integers(low=0, high=2 ** 63 - 1)
            )
        else:
            self.rng = default_rng(seed)

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> WaveformParameters:
        """Provide the next set of parameters."""


class UniformParameterGenerator(ParameterGenerator):
    r"""Generator of parameters according to a uniform distribution
    over their allowed ranges.

    # TODO update docs here!

    Parameters
    ----------
    dataset: Dataset
            See the documentation for the initialization of a
            :class:`ParameterGenerator`.
    parameter_ranges: ParameterRanges
    seed: Optional[int]
            See the documentation for the initialization of a
            :class:`ParameterGenerator`.


    Examples
    --------
    >>> generator = UniformParameterGenerator(
    ...    dataset=Dataset(20., 4096.),
    ...    parameter_ranges=ParameterRanges(q_range=(1., 2.)))
    >>> params = next(generator)
    >>> print(type(params))
    <class 'mlgw_bns.dataset_generation.WaveformParameters'>
    >>> print(params.mass_ratio) # doctest: +NUMBER
    1.306
    """

    def __init__(
        self,
        dataset: Dataset,
        parameter_ranges: ParameterRanges,
        seed: Optional[int] = None,
    ):

        self.q_range: tuple[float, float] = parameter_ranges.q_range
        self.lambda1_range: tuple[float, float] = parameter_ranges.lambda1_range
        self.lambda2_range: tuple[float, float] = parameter_ranges.lambda2_range
        self.chi1_range: tuple[float, float] = parameter_ranges.chi1_range
        self.chi2_range: tuple[float, float] = parameter_ranges.chi2_range
        super().__init__(dataset=dataset, seed=seed)

    def __next__(self) -> WaveformParameters:
        mass_ratio = self.rng.uniform(*self.q_range)
        lambda_1 = self.rng.uniform(*self.lambda1_range)
        lambda_2 = self.rng.uniform(*self.lambda2_range)
        chi_1 = self.rng.uniform(*self.chi1_range)
        chi_2 = self.rng.uniform(*self.chi2_range)

        return WaveformParameters(
            mass_ratio, lambda_1, lambda_2, chi_1, chi_2, self.dataset
        )


class Dataset:
    r"""Metadata for a dataset.

    # TODO: the name of this class is misleading,
    as it contains all information contained to generate
    the dataset but not the data itself.
    (but I cannot think of a better one, maybe DatasetMeta?)

    The amplitude residuals are defined as
    :math:`\log(A _{\text{EOB}} / A_{\text{PN}})`,
    while the phase residuals are defined as
    :math:`\phi _{\text{EOB}} - \phi_{\text{PN}}`.

    Parameters
    ----------
    initial_frequency_hz : float
            Initial frequency from which the waveforms in this dataset
            should be generated by the effective one body model.
    srate_hz : float
            Sampling rate in the time domain.
            The maximum frequency of the generated time-domain waveforms will be
            half of this value (see
            `Nyquist frequency <https://en.wikipedia.org/wiki/Nyquist_frequency>`_).
    delta_f_hz : Optional[float]
            Frequency spacing for the generated waveforms.
            If it is not given, it defaults to the one computed through
            :func:`Dataset.optimal_df_hz`.
    waveform_generator : WaveformGenerator
            Waveform generator to be used.
            Defaults to TEOBResumSGenerator, which uses TEOB for
            the EOB waveform an a TaylorF2 approximant, with 3.5PN-correct
            amplitude and 5.5PN-correct phase.
    parameter_generator_class : Type[ParameterGenerator]
            Parameter generator class to be used.
            Should be a subclass of ParameterGenerator; the argument is
            the class as opposed to an instance since the parameter generator
            needs to reference the dataset and therefure must be created after it.
            Defaults to UniformParameterGenerator.
    parameter_ranges: ParameterRanges
            Ranges for the parameters to be generated.
            Defaults to ParameterRanges(), which will use the
            parameters defined as defaults in that class.
    parameter_generator : Optional[ParameterGenerator]
            Certain parameter generators should not be regenerated each time;
            if this is the case, then pass the parameter generator here.
            Defaults to None.
    seed : int
            Seed for the random number generator used when generating
            waveforms for the training.
            Defaults to 42.
    multibanding: bool
            Whether to use multibanding for the default frequency array.
            If True, the frequency array is computed according to
            :func:`reduced_frequency_array`;
            if False, the frequency array is the "default FFT" one
            with spacing :attr:`delta_f_hz`.
            Defaults to False.
    f_pivot_hz: float
            Pivot frequency for the multibanding in Hz, only used if
            :attr:`multibanding` is True.
            Defaults to 40.

    Examples
    --------
    >>> dataset = Dataset(initial_frequency_hz=20., srate_hz=4096.)
    >>> print(dataset.delta_f_hz) # should be 1/256 Hz, doctest: +NUMBER
    0.001953125

    Class Attributes
    ----------------
    total_mass : float
            Total mass of the reference binary, in solar masses (class attribute).
            Defaults to 2.8; this does not typically need to be changed.
    """

    total_mass: float = 2.8

    def __init__(
        self,
        initial_frequency_hz: float,
        srate_hz: float,
        delta_f_hz: Optional[float] = None,
        waveform_generator: WaveformGenerator = BarePostNewtonianGenerator(),
        parameter_generator_class: Type[ParameterGenerator] = UniformParameterGenerator,
        parameter_ranges: ParameterRanges = ParameterRanges(),
        parameter_generator: Optional[ParameterGenerator] = None,
        seed: int = 42,
        multibanding: bool = True,
        f_pivot_hz: float = 40.0,
    ):

        self.initial_frequency_hz = initial_frequency_hz
        self.srate_hz = srate_hz

        (
            self.effective_initial_frequency_hz,
            self.effective_srate_hz,
        ) = expand_frequency_range(
            initial_frequency_hz,
            srate_hz,
            parameter_ranges.mass_range,
            self.total_mass,
        )

        self.delta_f_hz = self.optimal_df_hz() if delta_f_hz is None else delta_f_hz
        self.waveform_generator = waveform_generator
        self.parameter_generator_class = parameter_generator_class
        self.parameter_ranges = parameter_ranges

        self.parameter_generator = parameter_generator

        self.seed_sequence = np.random.default_rng(seed=seed)

        self.residuals_amp: list[np.ndarray] = []
        self.residuals_phi: list[np.ndarray] = []

        self.multibanding = multibanding
        self.f_pivot_hz = f_pivot_hz

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"initial_frequency_hz={self.initial_frequency_hz}, "
            f"srate_hz={self.srate_hz}, "
            f"effective_initial_frequency_hz={self.effective_initial_frequency_hz}, "
            f"effective_srate_hz={self.effective_srate_hz}, "
            f"delta_f_hz={self.delta_f_hz}"
            ")"
        )

    @property
    def waveform_length(self) -> int:
        if self.multibanding:
            return len(self.frequencies)
        else:
            return (
                int(
                    (self.effective_srate_hz / 2 - self.effective_initial_frequency_hz)
                    / self.delta_f_hz
                )
                + 1
            )

    @property
    def frequencies(self) -> np.ndarray:
        """Frequency array corresponding to this dataset,
        in natural units.
        """

        return self._frequencies()

    @lru_cache(maxsize=1)
    def _frequencies(self):
        if self.waveform_generator.frequencies is not None:
            return self.waveform_generator.frequencies

        return self.hz_to_natural_units(self.frequencies_hz)

    @property
    def frequencies_hz(self) -> np.ndarray:
        """Frequency array corresponding to this dataset,
        in Hz.
        """
        return self._frequencies_hz()

    @lru_cache(maxsize=1)
    def _frequencies_hz(self):

        if self.waveform_generator.frequencies is not None:
            return self.natural_units_to_hz(self.waveform_generator.frequencies)

        if self.multibanding:
            return reduced_frequency_array(
                self.effective_initial_frequency_hz,
                self.effective_srate_hz / 2,
                self.f_pivot_hz,
            )
        else:
            return np.arange(
                self.effective_initial_frequency_hz,
                self.effective_srate_hz / 2 + self.delta_f_hz,
                self.delta_f_hz,
            )

    @property
    def parameter_set_cls(self):
        if self.parameter_generator is None:
            return ParameterGenerator.parameter_set_cls
        return self.parameter_generator.parameter_set_cls

    def optimal_df_hz(
        self, power_of_two: bool = True, margin_percent: float = 8.0
    ) -> float:
        r"""Frequency spacing required for the condition
        :math:`\Delta f < 1/T`, where :math:`T` is the seglen
        (length of the signal).

        The optimal frequency spacing `df` is the inverse of the seglen
        (length of the signal) rounded up to the next power of 2.

        The seglen has a closed-form expression in the Newtonian limit,
        see e.g. Maggiore (2007), eq. 4.21:

        :math:`t = 5/256 (\pi f)^{-8/3} (G M \eta^{3/5} / c^3)^{-5/3}`

        The symmetric mass ratio :math:`\eta` changes across our dataset,
        so we take the upper limit with :math:`\eta = 1/4`.

        Parameters
        ----------
        power_of_two : bool
            whether to return a frequency spacing which is a round power of two.
            Defaults to True.
        margin_percent : float
            percent of margin to be added to the seglen, so that
            :math:`\Delta f < 1 / (T + \delta T)` holds for
            :math:`\delta T \leq T (\text{margin} / 100)`.

            This should not be too low, since varying the waveform parameters
            can perturb the seglen and make it a bit higher than the
            Newtonian approximation used in this formula.

        Returns
        -------
        delta_f_hz : float
            Frequency spacing, in Hz.
        """

        seglen = (
            5
            / 256
            * (np.pi * self.effective_initial_frequency_hz) ** (-8 / 3)
            * (self.mass_sum_seconds * (1 / 4) ** (3 / 5)) ** (-5 / 3)
        )

        if power_of_two:
            return 2 ** (-np.ceil(np.log2(seglen * (1 + margin_percent / 100))))
        else:
            return 1 / seglen

    @property
    def mass_sum_seconds(self) -> float:
        """Reference total mass expressed in seconds, :math:`GM / c^3`.

        Returns
        -------
        float
        """
        return self.total_mass * SUN_MASS_SECONDS

    def hz_to_natural_units(self, frequency_hz: Union[float, np.ndarray]):
        """Utility function: convert Hz to natural units,
        using the reference total mass of the dataset.

        Parameters
        ----------
        frequency_hz : Union[float, np.ndarray]

        Returns
        -------
        frequency_nu
            Frequency in natural units.
        """
        return frequency_hz * self.mass_sum_seconds

    def natural_units_to_hz(self, frequency: Union[float, np.ndarray]):
        """Utility function: convert Hz to natural units,
        using the reference total mass of the dataset.

        Parameters
        ----------
        frequency : Union[float, np.ndarray]

        Returns
        -------
        frequency_hz
            Frequency in Hz.
        """
        return frequency / self.mass_sum_seconds

    def taylor_f2_prefactor(self, eta: float) -> float:
        """Prefactor by which to multiply the waveform
        generated by TaylorF2.

        Parameters
        ----------
        eta : float
                Mass ratio of the binary
        """
        return TF2_BASE * AMP_SI_BASE / eta / self.total_mass ** 2

    def mlgw_bns_prefactor(
        self, eta: float, total_mass: Optional[float] = None
    ) -> float:
        """Prefactor by which to multiply the waveform
        generated by `mlgw_bns`.

        Parameters
        ----------
        eta : float
                Mass ratio of the binary
        total_mass : Optional[float]
                Total mass of the binary.
                Defaults to None, in which case the `total_mass`
                attribute of the Dataset will be used.
        """

        if total_mass is None:
            total_mass = self.total_mass

        return total_mass ** 2 / AMP_SI_BASE * eta

    def make_parameter_generator(
        self, seed: Optional[int] = None
    ) -> ParameterGenerator:
        """Make a new parameter generator,
        of the type determined by :attr:`parameter_generator_class`.

        Parameters
        ----------
        seed : int, optional
            Seed for the RNG inside the parameter generator, by default None

        Returns
        -------
        ParameterGenerators
        """
        if seed is None:
            seed = self.seed_sequence.integers(low=0, high=1 << 63 - 1)

        return self.parameter_generator_class(
            parameter_ranges=self.parameter_ranges,
            dataset=self,
            seed=seed,
        )

    def generate_residuals(
        self,
        size: int,
        downsampling_indices: Optional[DownsamplingIndices] = None,
        flatten_phase: bool = True,
    ) -> tuple[np.ndarray, ParameterSet, Residuals]:
        """Generate a set of waveform residuals.

        Parameters
        ----------
        size : int
                Number of waveforms to generate.
        downsampling_indices: Optional[DownsamplingIndices]
                If provided, return the waveform only at these indices,
                which can be different between phase and amplitude.
                Defaults to None.
        flatten_phase: bool
                Whether to subtract a linear term from the phase
                such that it is roughly constant in its first section
                (through the method :func:`Residuals.flatten_phase`).
                Defaults to True,
                but it is always set to False if the downsampling indices
                are not provided.

        Returns
        -------
        frequencies: np.ndarray,
            Frequencies at which the waveforms are computed,
            in natural units. This array should have shape
            ``(number_of_sample_points, )``.
        parameters: ParameterSet
        residuals: Residuals
        """
        if downsampling_indices is None:
            amp_length = self.waveform_length
            phi_length = self.waveform_length
            flatten_phase = False
        else:
            amp_length = downsampling_indices.amp_length
            phi_length = downsampling_indices.phi_length

        amp_residuals = np.empty((size, amp_length))
        phi_residuals = np.empty((size, phi_length))
        parameter_array = np.empty((size, WaveformParameters.number_of_parameters))

        if self.parameter_generator is None:
            parameter_generator = self.make_parameter_generator()
        else:
            parameter_generator = self.parameter_generator

        for i in tqdm(range(size), unit="residuals"):
            params = next(parameter_generator)

            (
                amp_residuals[i],
                phi_residuals[i],
            ) = self.waveform_generator.generate_residuals(
                params, self.frequencies, downsampling_indices
            )

            parameter_array[i] = params.array

        residuals = Residuals(amp_residuals, phi_residuals)

        if flatten_phase:
            if downsampling_indices is None:
                indices: Union[slice, list[int]] = slice(None)
            else:
                indices = downsampling_indices.phase_indices

            residuals.flatten_phase(self.frequencies[indices])

        return (
            self.frequencies,
            self.parameter_set_cls(parameter_array),
            residuals,
        )

    def recompose_residuals(
        self,
        residuals: Residuals,
        params: ParameterSet,
        downsampling_indices: Optional["DownsamplingIndices"] = None,
    ) -> FDWaveforms:
        """Recompose a set of residuals into true waveforms.

        Parameters
        ----------
        residuals : Residuals
            Residuals to recompose.
        params : ParameterSet
            Parameters of the waveforms corresponding to the residuals.
        downsampling_indices: DownsamplingIndices, optional
            Indices at which to sample the waveforms.
            Defaults to None, which means to use the whole sampling

        Returns
        -------
        FDWaveforms
            Reconstructed waveforms; these may differ from the original ones
            by a linear phase term (corresponding to a time shift) even if no manipulation
            has been done, because of how the :class:`Residuals` are stored.
        """

        amp_residuals, phi_residuals = residuals

        waveform_param_list = params.waveform_parameters(self)

        if downsampling_indices is None:
            amp_indices: Union[slice, list[int]] = slice(None)
            phi_indices: Union[slice, list[int]] = slice(None)
        else:
            amp_indices = downsampling_indices.amplitude_indices
            phi_indices = downsampling_indices.phase_indices

        pn_amps = np.array(
            [
                self.waveform_generator.post_newtonian_amplitude(
                    par, self.frequencies[amp_indices]
                )
                for par in waveform_param_list
            ]
        )
        pn_phis = np.array(
            [
                self.waveform_generator.post_newtonian_phase(
                    par, self.frequencies[phi_indices]
                )
                for par in waveform_param_list
            ]
        )

        return FDWaveforms(
            amplitudes=np.exp(amp_residuals) * pn_amps,
            phases=phi_residuals + pn_phis,
        )

    def generate_waveforms_from_params(
        self,
        parameters: ParameterSet,
        downsampling_indices: Optional[DownsamplingIndices] = None,
    ) -> FDWaveforms:
        """Generate full effective-one-body waveforms
        at each of the parameters in the given parameter set.

        Parameters
        ----------
        parameters : ParameterSet
            Parameters of the waveforms to generate
        downsampling_indices : DownsamplingIndices, optional
            Indices to downsample the waveforms at, by default None

        Returns
        -------
        FDWaveforms
        """

        if downsampling_indices is None:
            amp_indices: Union[slice, list[int]] = slice(None)
            phi_indices: Union[slice, list[int]] = slice(None)
        else:
            amp_indices = downsampling_indices.amplitude_indices
            phi_indices = downsampling_indices.phase_indices

        waveform_param_list = parameters.waveform_parameters(self)

        amps = []
        phis = []

        for par in tqdm(waveform_param_list, unit="waveforms"):
            _, amp, phi = self.waveform_generator.effective_one_body_waveform(
                par, self.frequencies
            )
            amps.append(amp[amp_indices])
            phis.append(phi[phi_indices])

        return FDWaveforms(np.array(amps), np.array(phis))


def expand_frequency_range(
    initial_frequency: float,
    final_frequency: float,
    mass_range: tuple[float, float],
    reference_mass: float,
) -> tuple[float, float]:
    r"""Widen the frequency range to account for the
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
        initial_frequency * (m_min / reference_mass),
        final_frequency * (m_max / reference_mass),
    )
