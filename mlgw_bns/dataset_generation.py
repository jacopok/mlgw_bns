"""Functionality for the generation of a training dataset.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Type, Union

import EOBRun_module  # type: ignore
import h5py
import numpy as np
from numpy.random import SeedSequence, default_rng
from tqdm import tqdm  # type: ignore

from .data_management import (
    DownsamplingIndices,
    FDWaveforms,
    Residuals,
    SavableData,
    phase_unwrapping,
)

# from .downsampling_interpolation import DownsamplingTraining
from .taylorf2 import (
    SUN_MASS_SECONDS,
    Af3hPN,
    Phif5hPN,
    PhifQM3hPN,
    PhifT7hPNComplete,
    compute_delta_lambda,
    compute_lambda_tilde,
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
    the easiest way to accomplish this is to subclass TEOBResumSGenerator
    and only override its :meth:`effective_one_body_waveform` method.


    """

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
        self, params: "WaveformParameters"
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Waveform

        Parameters
        ----------
        params : WaveformParameters
                Parameters of the binary system for which to generate the waveform.

        Returns
        -------
        frequencies : np.ndarray
                Frequencies at which the waveform is given, in natural units:
                the quantity here is :math:`Mf` (with :math:`G = c = 1`).
        cartesian_waveform : np.ndarray
                Cartesian form of the plus-polarized waveform.
                The normalization for the amplitude is the same as discussed
                in :func:`post_newtonian_amplitude`.
        """

    def generate_residuals(
        self,
        params: "WaveformParameters",
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
        downsampling_indices : Optional[DownsamplingIndices]
                Indices at which to compute the residuals.
                If not provided (default) the waveform is given at
                all indices corresponding to the default FFT grid.
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
                Amplitude residuals and phase residuals.
        """

        (
            frequencies_eob,
            waveform_eob,
        ) = self.effective_one_body_waveform(params)

        amplitude_eob, phase_eob = phase_unwrapping(waveform_eob)

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

        return (np.log(amplitude_eob / amplitude_pn), phase_eob - phase_pn)


class TEOBResumSGenerator(WaveformGenerator):
    """Generate waveforms using the
    TEOBResumS effective-one-body code,
    3.5PN-accurate post-newtonian amplitude,
    and 5.5PN-accurate post-newtonian phase
    with 7.5PN-accurate tidal terms."""

    def post_newtonian_amplitude(
        self, params: "WaveformParameters", frequencies: np.ndarray
    ) -> np.ndarray:
        par_dict = params.taylor_f2(frequencies)

        return (
            Af3hPN(
                par_dict["f"],
                par_dict["mtot"],
                params.eta,
                par_dict["s1x"],
                par_dict["s1y"],
                par_dict["s1z"],
                par_dict["s2x"],
                par_dict["s2y"],
                par_dict["s2z"],
                Lam=params.lambdatilde,
                dLam=params.dlambda,
                Deff=par_dict["Deff"],
            )
            * params.dataset.taylor_f2_prefactor(params.eta)
        )

    def post_newtonian_phase(
        self, params: "WaveformParameters", frequencies: np.ndarray
    ) -> np.ndarray:

        par_dict = params.taylor_f2(frequencies)

        phi_5pn = Phif5hPN(
            par_dict["f"],
            par_dict["mtot"],
            params.eta,
            par_dict["s1x"],
            par_dict["s1y"],
            par_dict["s1z"],
            par_dict["s2x"],
            par_dict["s2y"],
            par_dict["s2z"],
        )

        # Tidal and QM contributions
        phi_tidal = PhifT7hPNComplete(
            par_dict["f"],
            par_dict["mtot"],
            params.eta,
            par_dict["lambda1"],
            par_dict["lambda2"],
        )
        # Quadrupole-monopole term
        # [https://arxiv.org/abs/gr-qc/9709032]
        phi_qm = PhifQM3hPN(
            par_dict["f"],
            par_dict["mtot"],
            params.eta,
            par_dict["s1x"],
            par_dict["s1y"],
            par_dict["s1z"],
            par_dict["s2x"],
            par_dict["s2y"],
            par_dict["s2z"],
            par_dict["lambda1"],
            par_dict["lambda2"],
        )

        # I use the convention h = h+ + i hx
        phase = -phi_5pn - phi_tidal - phi_qm

        return phase - phase[0]

    def effective_one_body_waveform(self, params: "WaveformParameters"):
        r"""Generate an EOB waveform with TEOB.

        Examples
        --------
        >>> tg = TEOBResumSGenerator()
        >>> p = WaveformParameters(1, 300, 300, .3, -.3, Dataset(20., 4096.))
        >>> f, waveform = tg.effective_one_body_waveform(p)
        >>> print(len(waveform))
        519169
        >>> print(waveform[0]) # doctest: +NUMBER
        (4679.9+3758.8j)
        """

        par_dict = params.teobresums

        n_additional = 256

        # tweak initial frequency backward by a few samples
        # this is needed because of a bug in TEOBResumS
        # causing the phase evolution not to behave properly
        # at the beginning of integration
        # TODO remove this once the TEOB bug is fixed

        f_0 = par_dict["initial_frequency"]
        delta_f = par_dict["df"]
        new_f0 = f_0 - delta_f * n_additional
        par_dict["initial_frequency"] = new_f0

        f_spa, rhpf, ihpf, _, _ = EOBRun_module.EOBRunPy(par_dict)

        f_spa = f_spa[n_additional:]

        waveform = (rhpf - 1j * ihpf)[n_additional:]

        return (f_spa, waveform)


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

    @property
    def teobresums(self) -> dict[str, Union[float, int]]:
        """Parameter dictionary in a format compatible with
        TEOBResumS.

        The parameters are all converted to natural units.
        """
        return {
            "q": self.mass_ratio,
            "Lambda1": self.lambda_1,
            "Lambda2": self.lambda_2,
            "chi1": self.chi_1,
            "chi2": self.chi_2,
            "M": self.dataset.total_mass,
            "distance": 1.0,
            "initial_frequency": self.dataset.initial_frequency_hz
            * self.dataset.mass_sum_seconds,
            "use_geometric_units": 1,
            "interp_uniform_grid": 0,
            "domain": 1,  # Fourier domain
            "srate_interp": self.dataset.srate_hz * self.dataset.mass_sum_seconds,
            "df": self.dataset.delta_f_hz * self.dataset.mass_sum_seconds,
            "interp_FD_waveform": 1,
            "inclination": 0.0,
            "output_hpc": 0,
            "output_dynamics": 0,
            "time_shift_FD": 1,
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
            "f_min": self.dataset.initial_frequency_hz,
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
    def array(self):
        return np.array(
            [self.mass_ratio, self.lambda_1, self.lambda_2, self.chi_1, self.chi_2]
        )


@dataclass
class ParameterSet(SavableData):
    """Dataclass which contains an array of parameters for waveform generation.

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
            How many tuples to generate
        """

        param_array = np.array(
            [next(parameter_generator).array for _ in range(number_of_parameter_tuples)]
        )

        return cls(param_array)


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

    def __init__(self, dataset: "Dataset", seed: Optional[int] = None, **kwargs):

        self.dataset = dataset

        if seed is None:
            self.rng = default_rng(self.dataset.seed_sequence.generate_state(1)[0])
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

    Parameters
    ----------
    q_range : tuple[float, float]
            Range of valid mass ratios.
            Defaults to (1., 2.).
    lambda1_range : tuple[float, float]
            Range of valid tidal deformabilities parameters for the larger star.
            Defaults to (5., 5000.): the lower bound is not zero because that
            may create some issues with TEOB crashing.
    lambda2_range : tuple[float, float]
            Range of valid tidal deformabilities parameters for the smaller star.
            Defaults to (5., 5000.).
    chi1_range : tuple[float, float]
            Range of valid dimensionless aligned spins for the larger star.
            Defaults to (-.5, .5).
    chi2_range : tuple[float, float]
            Range of valid dimensionless aligned spins for the smaller star.
            Defaults to (-.5, .5).

    Keyword Arguments
    -----------------
    dataset: Dataset
            See the documentation for the initialization of a
            :class:`ParameterGenerator`.
    seed: Optional[int]
            See the documentation for the initialization of a
            :class:`ParameterGenerator`.


    Examples
    --------
    >>> generator = UniformParameterGenerator(dataset=Dataset(20., 4096.))
    >>> params = next(generator)
    >>> print(type(params))
    <class 'mlgw_bns.dataset_generation.WaveformParameters'>
    >>> print(params.mass_ratio) # doctest: +NUMBER
    1.96
    """

    def __init__(
        self,
        q_range: tuple[float, float] = (1.0, 2.0),
        lambda1_range: tuple[float, float] = (5.0, 5000.0),
        lambda2_range: tuple[float, float] = (5.0, 5000.0),
        chi1_range: tuple[float, float] = (-0.5, 0.5),
        chi2_range: tuple[float, float] = (-0.5, 0.5),
        **kwargs,
    ):

        self.q_range = q_range
        self.lambda1_range = lambda1_range
        self.lambda2_range = lambda2_range
        self.chi1_range = chi1_range
        self.chi2_range = chi2_range
        super().__init__(**kwargs)

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
    parameter_generator_kwargs : dict[str, Any]

    seed : int
            Seed for the random number generator used when generating
            waveforms for the training.
            Defaults to 42.

    Examples
    --------
    >>> dataset = Dataset(initial_frequency_hz=20., srate_hz=4096.)
    >>> print(dataset.delta_f_hz) # should be 1/256 Hz, doctest: +NUMBER
    0.00390625

    Class Attributes
    ----------------
    total_mass : float
            Total mass of the reference binary, in solar masses (class attribute).
            Defaults to 2.8; this does not typically need to be changed.
    """

    # TODO
    # saving to file to be managed with https://docs.h5py.org/en/stable/quick.html

    # total mass of the binary, in solar masses
    total_mass: float = 2.8

    arrays_to_save: list[str] = [
        "amplitude_residuals",
        "phase_residuals",
        "frequencies",
    ]
    # TODO document this

    def __init__(
        self,
        initial_frequency_hz: float,
        srate_hz: float,
        delta_f_hz: Optional[float] = None,
        waveform_generator: WaveformGenerator = TEOBResumSGenerator(),
        parameter_generator_class: Type[ParameterGenerator] = UniformParameterGenerator,
        parameter_generator_kwargs: Optional[dict[str, Any]] = None,
        seed: int = 42,
    ):

        self.initial_frequency_hz = initial_frequency_hz
        self.srate_hz = srate_hz
        self.delta_f_hz = self.optimal_df_hz() if delta_f_hz is None else delta_f_hz
        self.waveform_generator = waveform_generator
        self.parameter_generator_class = parameter_generator_class
        self.parameter_generator_kwargs = (
            {} if parameter_generator_kwargs is None else parameter_generator_kwargs
        )

        self.seed_sequence = SeedSequence(seed)

        self.residuals_amp: list[np.ndarray] = []
        self.residuals_phi: list[np.ndarray] = []

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"initial_frequency_hz={self.initial_frequency_hz}, "
            f"delta_f_hz={self.delta_f_hz}, "
            f"srate_hz={self.srate_hz}"
            ")"
        )

    @property
    def waveform_length(self) -> int:
        return (
            int((self.srate_hz / 2 - self.initial_frequency_hz) / self.delta_f_hz) + 1
        )

    @property
    def frequencies(self) -> np.ndarray:
        """Frequency array corresponding to this dataset,
        in natural units.
        """
        return self.hz_to_natural_units(self.frequencies_hz)

    @property
    def frequencies_hz(self) -> np.ndarray:
        """Frequency array corresponding to this dataset,
        in Hz.
        """
        return np.arange(
            self.initial_frequency_hz,
            self.srate_hz / 2 + self.delta_f_hz,
            self.delta_f_hz,
        )

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
            * (np.pi * self.initial_frequency_hz) ** (-8 / 3)
            * (self.mass_sum_seconds * (1 / 4) ** (3 / 5)) ** (-5 / 3)
        )

        if power_of_two:
            return 2 ** (-np.ceil(np.log2(seglen * (1 + margin_percent / 100))))
        else:
            return 1 / seglen

    @property
    def mass_sum_seconds(self) -> float:
        return self.total_mass * SUN_MASS_SECONDS

    def hz_to_natural_units(self, frequency_hz: Union[float, np.ndarray]):
        return frequency_hz * self.mass_sum_seconds

    def taylor_f2_prefactor(self, eta: float) -> float:
        """Prefactor by which to multiply the waveform
        generated by TaylorF2.

        Parameters
        ----------
        eta : float
                Mass ratio of the binary
        """
        return TF2_BASE * AMP_SI_BASE / eta / self.total_mass ** 2

    def mlgw_bns_prefactor(self, eta: float) -> float:
        """Prefactor by which to multiply the waveform
        generated by `mlgw_bns`.

        Parameters
        ----------
        eta : float
                Mass ratio of the binary
        """
        return self.total_mass ** 2 / AMP_SI_BASE * eta

    def make_parameter_generator(self) -> ParameterGenerator:
        return self.parameter_generator_class(
            dataset=self,
            seed=self.seed_sequence.generate_state(1)[0],
            **self.parameter_generator_kwargs,
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

        parameter_generator = self.make_parameter_generator()

        for i in tqdm(range(size), unit="residuals computed"):
            params = next(parameter_generator)

            (
                amp_residuals[i],
                phi_residuals[i],
            ) = self.waveform_generator.generate_residuals(params, downsampling_indices)

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
            ParameterSet(parameter_array),
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
            _, cartesian_wf = self.waveform_generator.effective_one_body_waveform(par)
            amp, phi = phase_unwrapping(cartesian_wf)
            amps.append(amp[amp_indices])
            phis.append(phi[phi_indices])

        return FDWaveforms(np.array(amps), np.array(phis))
