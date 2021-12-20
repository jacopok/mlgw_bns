"""Functionality for the generation of a training dataset.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional, Type, Union

import EOBRun_module  # type: ignore
import h5py
import numpy as np
from numpy.random import SeedSequence, default_rng

from .taylorf2 import (
    SUN_MASS_SECONDS,
    Af3hPN,
    Phif5hPN,
    PhifQM3hPN,
    PhifT7hPNComplete,
    compute_delta_lambda,
    compute_lambda_tilde,
)
from .waveform_management import phase_unwrapping

TF2_BASE: float = 3.668693487138444e-19
# ( Msun * G / c**3)**(5/6) * Hz**(-7/6) * c / Mpc / s
AMP_SI_BASE: float = 4.2425873413901263e24
# Mpc / Msun**2 / Hz


class WaveformGenerator(ABC):
    """Generator of theoretical waveforms
    according to some pre-existing model.
    """

    @abstractmethod
    def post_newtonian_amplitude(
        self, params: "WaveformParameters", frequencies: np.ndarray
    ) -> np.ndarray:
        pass

    @abstractmethod
    def post_newtonian_phase(
        self, params: "WaveformParameters", frequencies: np.ndarray
    ) -> np.ndarray:
        pass

    @abstractmethod
    def effective_one_body_waveform(self, params: "WaveformParameters") -> np.ndarray:
        pass


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
        >>> print(waveform[0])
        (4679.915419630735+3758.8458103665052j)
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
    """

    mass_ratio: float
    lambda_1: float
    lambda_2: float
    chi_1: float
    chi_2: float
    dataset: "Dataset"

    def __eq__(self, other: object):
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


class ParameterGenerator(ABC, Iterator):
    """Generic generator of parameters for new waveforms
    to be used for training.
    """

    def __init__(self, dataset: "Dataset", seed: Optional[int] = None):

        self.dataset = dataset

        if seed is None:
            self.rng = default_rng(self.dataset.seed_sequence.generate_state(1)[0])
        else:
            self.rng = default_rng(seed)

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> WaveformParameters:
        pass


class UniformParameterGenerator(ParameterGenerator):
    """Generator of parameters according to a uniform distribution
    over their allowed ranges.
    """

    def __next__(self) -> WaveformParameters:
        mass_ratio = self.rng.uniform(*self.dataset.q_range)
        lambda_1 = self.rng.uniform(*self.dataset.lambda1_range)
        lambda_2 = self.rng.uniform(*self.dataset.lambda2_range)
        chi_1 = self.rng.uniform(*self.dataset.chi1_range)
        chi_2 = self.rng.uniform(*self.dataset.chi2_range)

        return WaveformParameters(
            mass_ratio, lambda_1, lambda_2, chi_1, chi_2, self.dataset
        )


class Dataset:
    r"""A dataset of data generated with some model.

    Includes:

    * frequencies at which the data are sampled
    * frequency indices
    * amplitude and phase residuals for all modes

    The amplitude residuals are defined as
    :math:`\log(A _{\text{EOB}} / A_{\text{PN}})`,
    while the phase residuals are defined as
    :math:`\phi _{\text{EOB}} - \phi_{\text{PN}}`.

    Attributes
    ----------
    total_mass : float
            Total mass of the reference binary, in solar masses.
    q_range : tuple[float, float]
            Range of valid mass ratios.
    lambda1_range : tuple[float, float]
            Range of valid tidal deformabilities parameters for the larger star.
    lambda2_range : tuple[float, float]
            Range of valid tidal deformabilities parameters for the smaller star.
    chi1_range : tuple[float, float]
            Range of valid dimensionless aligned spins for the larger star.
    chi2_range : tuple[float, float]
            Range of valid dimensionless aligned spins for the smaller star.
    waveform_generator_class : Type[WaveformGenerator]
            Waveform generator class to be used, should be a
            subclass of WaveformGenerator.
            Defaults to TEOBResumSGenerator, which uses TEOB for
            the EOB waveform an a TaylorF2 approximant, with 3.5PN-correct
            amplitude and 5.5PN-correct phase.
    parameter_generator_class : Type[ParameterGenerator]
            Parameter generator class to be used, should be a
            subclass of ParameterGenerator.
            Defaults to UniformParameterGenerator.
    """

    # TODO
    # saving to file to be managed with https://docs.h5py.org/en/stable/quick.html

    # total mass of the binary, in solar masses
    total_mass: float = 2.8

    q_range: tuple[float, float] = (1.0, 2.0)
    lambda1_range: tuple[float, float] = (5.0, 5000.0)
    lambda2_range: tuple[float, float] = (5.0, 5000.0)
    chi1_range: tuple[float, float] = (-0.5, 0.5)
    chi2_range: tuple[float, float] = (-0.5, 0.5)

    waveform_generator_class: Type["WaveformGenerator"] = TEOBResumSGenerator
    parameter_generator_class: Type["ParameterGenerator"] = UniformParameterGenerator

    def __init__(
        self,
        initial_frequency_hz: float,
        srate_hz: float,
        delta_f_hz: Optional[float] = None,
        seed: int = 42,
    ):
        r"""
        Initialize dataset.

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
        delta_f_hz : float
                Frequency spacing for the generated waveforms.
        seed : int
                Seed for the random number generator used when generating
                waveforms for the training.
                Defaults to 42.

        Examples
        --------
        >>> dataset = Dataset(initial_frequency_hz=20., srate_hz=4096.)
        >>> print(dataset.delta_f_hz) # should be 1/256 Hz
        0.00390625
        """

        self.initial_frequency_hz = initial_frequency_hz
        self.srate_hz = srate_hz
        self.delta_f_hz = self.optimal_df_hz() if delta_f_hz is None else delta_f_hz
        self.waveform_generator = self.waveform_generator_class()

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

    def save(self, file) -> None:
        """Save the data to a h5 file."""
        pass
        # f = h5py.File(self.filename, 'w')
        # dset = f.create_dataset(self.filename)

    def load(self) -> None:
        """Load the data from a h5 file."""
        # with h5py.File(self.filename, "r") as file:
        #     self.frequencies = file["data"]
        pass

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

    def generate_residuals(
        self, size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        parameter_generator = self.parameter_generator_class(
            dataset=self, seed=self.seed_sequence.generate_state(1)[0]
        )

        amp_residuals = np.empty((size, self.waveform_length))
        phi_residuals = np.empty((size, self.waveform_length))

        for i in range(size):
            params = next(parameter_generator)
            (
                frequencies_eob,
                waveform_eob,
            ) = self.waveform_generator.effective_one_body_waveform(params)

            amplitude_eob, phase_eob = phase_unwrapping(waveform_eob)

            amplitude_pn = self.waveform_generator.post_newtonian_amplitude(
                params, frequencies_eob
            )
            phase_pn = self.waveform_generator.post_newtonian_phase(
                params, frequencies_eob
            )

            amp_residuals[i] = np.log(amplitude_eob / amplitude_pn)
            phi_residuals[i] = phase_eob - phase_pn

        return frequencies_eob, amp_residuals, phi_residuals
