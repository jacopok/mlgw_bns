r"""Full precessing waveforms from the aligned-spin ``mlgw_bns`` surrogate.

The surrogate of :class:`~mlgw_bns.model.Model` is aligned-spin: it
predicts the frequency-domain multipoles :math:`\tilde{h}_{\ell m}(f)` of
a binary whose spins are parallel to the orbital angular momentum. A
precessing binary, though, is well described --- this is the observation
underlying every "twisted" approximant, TEOBResumS included --- by
*those same multipoles*, taken to live in the co-precessing frame that
follows :math:`\hat{L}(t)`, and then rotated into the inertial frame by
three time-dependent Euler angles:

.. math::
    h^{\rm inertial}_{\ell m} = \sum_{n} D^{\ell}_{m n}(\alpha, \beta, \gamma)
                                 \, h^{\rm co-prec}_{\ell n} \,.

This module puts the two halves together: the Euler angles come from the
PN spin-precession dynamics of :mod:`~mlgw_bns.twist_waveform` (a
reproduction of what TEOBResumS itself integrates), the co-precessing
multipoles from :meth:`~mlgw_bns.model.Model.coprecessing_modes_dict`,
and the rotation from :func:`~mlgw_bns.twist_waveform.twist_modes`.

Twisting in the frequency domain
--------------------------------

The rotation above is a statement about the multipoles at a given
*time*, whereas the surrogate lives in the frequency domain. In the
stationary-phase approximation each multipole's Fourier transform at a
frequency :math:`f` is dominated by the single time :math:`t` at which
the multipole's instantaneous frequency equals :math:`f`; since the
Euler angles vary on the (much longer) precession timescale, the twist
carries over to the frequency domain by evaluating the angles at that
stationary time. This is the standard frequency-domain twisting-up
procedure, as used by e.g. ``IMRPhenomXPHM`` (`arXiv:2004.06503
<https://arxiv.org/abs/2004.06503>`_).

Two consequences shape the implementation:

* the stationary time differs from multipole to multipole. The
  co-precessing multipole :math:`n` has phase :math:`\simeq n
  \Phi_{\rm orb}`, so at frequency :math:`f` it is stationary where
  :math:`M \Omega_{\rm orb} = 2 \pi f M / n`. The angles are therefore
  looked up separately for each :math:`n`, against the orbital
  frequency that the PN integration provides alongside them.
* with the :math:`\tilde{h}(f) = \int h(t) e^{2 \pi i f t} \mathrm{d}t`
  convention used throughout ``mlgw_bns``, the :math:`n > 0`
  co-precessing multipoles have their support at :math:`f > 0` and the
  :math:`n < 0` ones at :math:`f < 0`. The two halves of the sum over
  :math:`n` must then be twisted separately: the :math:`n > 0` half
  builds :math:`\tilde{h}^{\rm inertial}_{\ell m}(f)`, the :math:`n < 0`
  half builds :math:`\tilde{h}^{\rm inertial}_{\ell m}(-f)`. Both are
  needed, because precession breaks the equatorial symmetry that would
  otherwise let :math:`h_\times` be recovered from :math:`h_+` alone.

The whole construction reduces, identically, to
:meth:`~mlgw_bns.model.Model.predict` when the in-plane spin components
vanish; :func:`check_aligned_spin_limit` asserts exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .higher_order_modes import Mode
from .model import Model
from .mode_model import ParametersWithExtrinsic
from .special_func import spinsphericalharm
from .taylorf2 import SUN_MASS_SECONDS
from .twist_waveform import integrate_pn_spin_precession, twist_modes

#: Key of a spherical-harmonic multipole, ``(l, m)``.
ModeKey = Tuple[int, int]


@dataclass
class PrecessingParametersWithExtrinsic:
    r"""Parameters of a single precessing waveform.

    The aligned-spin counterpart is
    :class:`~mlgw_bns.mode_model.ParametersWithExtrinsic`, of which this
    is the natural generalisation: the two scalar spins are promoted to
    full three-vectors, and an azimuthal angle is added to the
    inclination, since for a precessing source the two are no longer
    degenerate with a global phase.

    The frame is the usual one at the reference frequency: :math:`\hat{z}`
    along the initial orbital angular momentum :math:`\hat{L}_0`.

    Parameters
    ----------
    mass_ratio : float
        Mass ratio :math:`q = m_1 / m_2 \geq 1`.
    lambda_1, lambda_2 : float
        Tidal polarizabilities of the larger and smaller star.
    chi_1, chi_2 : Sequence[float]
        Dimensionless spin *vectors* :math:`\vec{\chi}_i = \vec{S}_i /
        m_i^2` of the two stars, as ``(x, y, z)`` triples.
    distance_mpc : float
        Distance to the source, in Megaparsecs.
    inclination : float
        Polar angle :math:`\iota` of the line of sight, in radians,
        measured from :math:`\hat{L}_0`.
    azimuth : float
        Azimuthal angle :math:`\varphi` of the line of sight, in radians.
        Defaults to 0.
    total_mass : float
        Total mass of the binary, in solar masses.
    reference_phase : float
        Phase of the first point of the waveform. Defaults to 0.
    time_shift : float
        Time shift applied to the waveform, in seconds. Defaults to 0.
    """

    mass_ratio: float
    lambda_1: float
    lambda_2: float
    chi_1: Sequence[float]
    chi_2: Sequence[float]
    distance_mpc: float
    inclination: float
    total_mass: float
    azimuth: float = 0.0
    reference_phase: float = 0.0
    time_shift: float = 0.0

    @property
    def chi_1_vector(self) -> np.ndarray:
        """Spin of the larger star as a length-3 array."""
        return np.asarray(self.chi_1, dtype=float)

    @property
    def chi_2_vector(self) -> np.ndarray:
        """Spin of the smaller star as a length-3 array."""
        return np.asarray(self.chi_2, dtype=float)

    @property
    def eta(self) -> float:
        r"""Symmetric mass ratio :math:`\eta = q / (1+q)^2`."""
        return self.mass_ratio / (1.0 + self.mass_ratio) ** 2

    def aligned(self) -> ParametersWithExtrinsic:
        r"""The aligned-spin parameters describing the co-precessing frame.

        Only the spin components along :math:`\hat{L}_0` survive; these
        are the ones the aligned-spin surrogate is trained on. The
        inclination is set to zero, since the co-precessing multipoles
        returned by
        :meth:`~mlgw_bns.model.Model.coprecessing_modes_dict` carry no
        sky projection --- the projection happens after the twist.

        Returns
        -------
        ParametersWithExtrinsic
            Parameters to hand to the aligned-spin :class:`Model`.
        """
        return ParametersWithExtrinsic(
            mass_ratio=self.mass_ratio,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            chi_1=float(self.chi_1_vector[2]),
            chi_2=float(self.chi_2_vector[2]),
            distance_mpc=self.distance_mpc,
            inclination=0.0,
            total_mass=self.total_mass,
            reference_phase=self.reference_phase,
            time_shift=self.time_shift,
        )


@dataclass
class EulerAngles:
    r"""The precession Euler angles, as functions of the orbital frequency.

    :func:`~mlgw_bns.twist_waveform.integrate_pn_spin_precession` returns
    :math:`\alpha, \beta, \gamma` and :math:`M \Omega_{\rm orb}` sampled
    along the PN inspiral; since :math:`M \Omega_{\rm orb}` increases
    monotonically it can be used as the independent variable, which is
    what the stationary-phase lookup needs.

    Attributes
    ----------
    momega : np.ndarray
        Orbital frequency :math:`M \Omega_{\rm orb}`, increasing.
    alpha, beta, gamma : np.ndarray
        The three Euler angles at those frequencies, in radians.
    time : np.ndarray, optional
        The integration time, in units of the total mass, at each sample.
        Kept so that :meth:`reanchored` can re-tabulate the angles
        against a more accurate time--frequency relation than the
        3.5PN one the integration itself marches with.
    """

    momega: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    time: Optional[np.ndarray] = None

    def at_momega(
        self, momega: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Interpolate the three angles at the given orbital frequencies.

        Outside the integrated range the angles are held at their
        boundary values: below, the PN integration simply started later
        than the requested frequency; above, it stopped at merger, past
        which the twist is in any case only a crude extrapolation.

        Parameters
        ----------
        momega : np.ndarray
            Orbital frequencies :math:`M \Omega_{\rm orb}` at which the
            angles are wanted.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(alpha, beta, gamma)``, each of the shape of ``momega``.

        Notes
        -----
        :math:`\alpha` and :math:`\gamma` are unwrapped before
        interpolating: they wind through many turns over an inspiral, and
        interpolating the wrapped values would smear each :math:`2\pi`
        jump across a whole interval.
        """
        return (
            np.interp(momega, self.momega, np.unwrap(self.alpha)),
            np.interp(momega, self.momega, self.beta),
            np.interp(momega, self.momega, np.unwrap(self.gamma)),
        )

    def reanchored(
        self,
        frequencies_hz: np.ndarray,
        reference_mode: np.ndarray,
        mass_sum_seconds: float,
        reference_frequency_hz: float,
    ) -> "EulerAngles":
        r"""Re-tabulate the angles against a better orbital-frequency axis.

        :func:`~mlgw_bns.twist_waveform.integrate_pn_spin_precession`
        advances :math:`M \Omega_{\rm orb}` along the inspiral with a
        3.5PN energy-balance :math:`\dot\Omega`, which runs fast through
        the late inspiral; the angles are accurate as *functions of
        time*, but the frequency they are labelled with drifts from the
        true one. Given a frequency-domain :math:`(2, 2)` multipole
        whose phase carries an accurate time--frequency relation --- the
        surrogate's, or TEOBResumS' --- this returns a copy of the angles
        whose ``momega`` axis is rebuilt from that relation, so that the
        stationary-phase lookup in
        :func:`twist_modes_frequency_domain` reads them at the right
        point of the precession cycle. The angle values themselves are
        untouched; only the frequency they sit at changes.

        Below ``reference_frequency_hz`` --- where the reference multipole
        has little power and its phase derivative is unreliable --- the
        original PN ``momega`` is kept, and the two are matched there.

        Parameters
        ----------
        frequencies_hz : np.ndarray
            The (positive, increasing) frequency grid of
            ``reference_mode``, in Hz.
        reference_mode : np.ndarray
            The complex :math:`(2, 2)` co-precessing multipole
            :math:`\tilde{h}_{22}(f)` on that grid.
        mass_sum_seconds : float
            Total mass of the binary in seconds, converting
            :math:`M \Omega_{\rm orb} = \pi M f_{22}`.
        reference_frequency_hz : float
            Frequency below which the reference phase is not trusted and
            the PN ``momega`` is kept; the two axes are matched here.

        Returns
        -------
        EulerAngles
            A copy with the re-anchored ``momega``; ``self`` unchanged
            (and returned as-is if it carries no ``time``).
        """
        if self.time is None:
            return self

        phase = np.unwrap(np.angle(reference_mode))
        # Stationary-phase time of the (2,2): it reaches GW frequency f at
        # t(f) = (1 / 2 pi) dphi/df. The sign of the phase convention is
        # absorbed by requiring t to increase with f (an inspiral chirps
        # up).
        spa_time = np.gradient(phase, frequencies_hz) / (2.0 * np.pi)
        if spa_time[-1] < spa_time[0]:
            spa_time = -spa_time
        spa_time = np.maximum.accumulate(spa_time)

        # PN (2,2) GW frequency along its own integration time.
        pn_f22 = self.momega / (np.pi * mass_sum_seconds)
        anchor = max(reference_frequency_hz, frequencies_hz[0], pn_f22[0])

        spa_time = spa_time - np.interp(anchor, frequencies_hz, spa_time)
        pn_time = self.time - np.interp(anchor, pn_f22, self.time)

        # true (2,2) frequency at each PN sample: from the reference phase
        # above the anchor, from the PN integration below it
        f22 = np.where(
            pn_time >= 0.0,
            np.interp(pn_time, spa_time, frequencies_hz,
                      left=frequencies_hz[0], right=frequencies_hz[-1]),
            pn_f22,
        )
        momega = np.maximum.accumulate(np.pi * mass_sum_seconds * f22)
        return replace(self, momega=momega)


def newtonian_time_to_merger(eta: float, momega: float) -> float:
    r"""Leading-order time left to merger, in units of the total mass.

    .. math::
        \frac{t_{\rm c}}{M} = \frac{5}{256\, \eta\, v^8} \,,
        \qquad v = (M \Omega_{\rm orb})^{1/3}

    Used only to bound the PN spin-precession integration: it runs until
    the merger fit stops it, and a cutoff comfortably past the Newtonian
    estimate keeps the integrator from being the thing that stops it.
    For a binary neutron star entering the band at a few tens of Hz this
    is of order :math:`10^8 M`, several orders of magnitude beyond the
    default cutoff of
    :func:`~mlgw_bns.twist_waveform.integrate_pn_spin_precession`.

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.
    momega : float
        Orbital frequency :math:`M \Omega_{\rm orb}`.

    Returns
    -------
    float
        Time to merger, in units of the total mass.
    """
    return 5.0 / (256.0 * eta * momega ** (8.0 / 3.0))


def eob_orbital_frequency_rate(
    frequencies_hz: np.ndarray,
    reference_mode: np.ndarray,
    mass_sum_seconds: float,
) -> Callable[[float], float]:
    r"""Build :math:`M\Omega_{\rm orb} \mapsto \mathrm{d}(M\Omega_{\rm orb})
    /\mathrm{d}t` from an accurate :math:`(2, 2)` phase.

    In the stationary-phase approximation the :math:`(2, 2)` multipole
    reaches GW frequency :math:`f` at :math:`t(f) = (2\pi)^{-1}
    \mathrm{d}\phi_{22}/\mathrm{d}f`, so :math:`\mathrm{d}f/\mathrm{d}t =
    2\pi / \phi_{22}''(f)`; with :math:`M\Omega_{\rm orb} = \pi M f` the
    orbital-frequency rate follows. Feeding this to
    :func:`euler_angles` marches the precession along the surrogate's
    (EOB-accurate) orbital-frequency track instead of the PN flux.

    Outside the span of ``frequencies_hz`` the rate is continued as
    :math:`\propto \Omega^{11/3}` (Newtonian energy balance), matched at
    the nearer edge: precession barely accumulates below the band, and
    above merger the twist is only an extrapolation regardless.

    Parameters
    ----------
    frequencies_hz : np.ndarray
        Increasing, positive frequency grid of ``reference_mode``, in Hz.
    reference_mode : np.ndarray
        Complex :math:`(2, 2)` multipole on that grid.
    mass_sum_seconds : float
        Total mass in seconds.

    Returns
    -------
    callable
        ``omega_dot(Momega) -> dMomega/dt``, geometric units.
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    phase = np.unwrap(np.angle(reference_mode))
    momega = np.pi * mass_sum_seconds * frequencies_hz

    # SPA time of the (2,2): |t(f)| = (1 / 2 pi) |dphi/df|, the time from
    # frequency f to coalescence. It shrinks monotonically as f grows
    # through the inspiral, so its f-derivative keeps a constant sign
    # there; where that sign flips is the noisy low-frequency edge of the
    # model, or the post-merger, and is dropped.
    spa_time = np.abs(np.gradient(phase, frequencies_hz)) / (2.0 * np.pi)
    d_spa = np.gradient(spa_time, frequencies_hz)
    inspiral_sign = np.sign(np.median(d_spa[: max(4, d_spa.size // 8)]))
    trusted = np.sign(d_spa) == inspiral_sign
    turnover = np.argmax(~trusted & (frequencies_hz > frequencies_hz[0] * 2.0))
    if turnover > 0:
        trusted[turnover:] = False
    if trusted.sum() < 8:
        trusted[:] = True

    # d(M Omega)/dt in *geometric* units (t in units of the total mass):
    #   M Omega = pi M_sec f,   d(M Omega)/dt = pi M_sec^2 |df/dt_sec|
    #           = pi M_sec^2 / |dt/df|.
    rate = np.pi * mass_sum_seconds ** 2 / np.abs(d_spa)
    # fit log(rate) against log(M Omega), where it is close to the
    # Newtonian straight line of slope 11/3, so a low-order polynomial
    # de-noises the twice-differentiated phase without distorting it.
    coeffs = np.polynomial.polynomial.Polynomial.fit(
        np.log(momega[trusted]), np.log(rate[trusted]), deg=4
    )

    lo_o, hi_o = momega[trusted][0], momega[trusted][-1]
    lo_r, hi_r = np.exp(coeffs(np.log(lo_o))), np.exp(coeffs(np.log(hi_o)))

    def omega_dot(omg: float) -> float:
        omg = float(omg)
        if omg <= lo_o:
            return lo_r * (omg / lo_o) ** (11.0 / 3.0)
        if omg >= hi_o:
            return hi_r * (omg / hi_o) ** (11.0 / 3.0)
        return float(np.exp(coeffs(np.log(omg))))

    return omega_dot


def euler_angles(
    params: PrecessingParametersWithExtrinsic,
    initial_frequency_hz: float,
    largest_mode_m: int = 4,
    omega_dot: Optional["Callable[[float], float]"] = None,
) -> EulerAngles:
    r"""Integrate the PN spin-precession dynamics for a given source.

    Thin wrapper around
    :func:`~mlgw_bns.twist_waveform.integrate_pn_spin_precession` that
    takes care of the units and of choosing a low enough starting
    frequency.

    Parameters
    ----------
    params : PrecessingParametersWithExtrinsic
        Source parameters; only the intrinsic ones matter here.
    initial_frequency_hz : float
        Lowest frequency, in Hz, at which the waveform will be
        evaluated.
    largest_mode_m : int
        Largest :math:`|m|` among the multipoles that will be twisted.
        The multipole :math:`m` reaches the frequency
        ``initial_frequency_hz`` when the orbit is only at
        :math:`2/m` of the corresponding :math:`(2,2)` frequency, so the
        integration has to start that much earlier. Defaults to 4.
    omega_dot : callable, optional
        ``M \Omega_{\rm orb} \mapsto \mathrm{d}(M\Omega_{\rm orb})/
        \mathrm{d}t`` in geometric units. When given, the precession is
        integrated *against* orbital frequency, marched along this rate
        instead of the PN flux -- reproducing the ``SPIN_FLX_EOB``
        hand-off TEOBResumS does once the spin dynamics reaches the EOB
        band. Build it from an accurate :math:`(2, 2)` phase with
        :func:`eob_orbital_frequency_rate`. When ``None`` (default), the
        integration marches in time with the 3.5PN flux, exactly as
        TEOBResumS' ``SPIN_FLX_PN`` branch.

    Returns
    -------
    EulerAngles
        The angles, tabulated against :math:`M \Omega_{\rm orb}`.
    """
    mass_sum_seconds = params.total_mass * SUN_MASS_SECONDS

    # The (2,2) GW frequency, in geometric units, at which the earliest
    # multipole we care about starts contributing.
    initial_frequency_22 = (
        2.0 * initial_frequency_hz * mass_sum_seconds / largest_mode_m
    )

    solution = integrate_pn_spin_precession(
        nu=params.eta,
        chi1vec=params.chi_1_vector,
        chi2vec=params.chi_2_vector,
        f0=initial_frequency_22,
        t_max=2.0 * newtonian_time_to_merger(params.eta, np.pi * initial_frequency_22),
        independent_variable="time" if omega_dot is None else "orbital_frequency",
        omega_dot=omega_dot,
    )

    return EulerAngles(
        momega=solution["Momega"],
        alpha=solution["alpha"],
        beta=solution["beta"],
        gamma=solution["gamma"],
        time=solution["t"],
    )


def twist_modes_frequency_domain(
    coprecessing_modes: Dict[ModeKey, np.ndarray],
    frequencies: np.ndarray,
    angles: EulerAngles,
    mass_sum_seconds: float,
) -> Tuple[Dict[ModeKey, np.ndarray], Dict[ModeKey, np.ndarray]]:
    r"""Twist frequency-domain co-precessing multipoles into the inertial frame.

    Applies the Wigner-:math:`D` rotation of
    :func:`~mlgw_bns.twist_waveform.twist_modes` frequency by frequency,
    with the Euler angles evaluated at each co-precessing multipole's own
    stationary-phase point (see the module docstring for why).

    Parameters
    ----------
    coprecessing_modes : dict[tuple[int, int], np.ndarray]
        Co-precessing multipoles :math:`\tilde{h}^{\rm co-prec}_{\ell n}(f)`
        for :math:`n > 0`, as returned by
        :meth:`~mlgw_bns.model.Model.coprecessing_modes_dict`.
    frequencies : np.ndarray
        The (positive) frequency grid, in Hz.
    angles : EulerAngles
        Precession angles, tabulated against orbital frequency.
    mass_sum_seconds : float
        Total mass of the binary, in seconds, used to convert the
        frequencies to geometric units.

    Returns
    -------
    tuple[dict, dict]
        ``(modes_positive_f, modes_negative_f)``: the inertial-frame
        multipoles :math:`\tilde{h}^{\rm inertial}_{\ell m}(f)` and
        :math:`\tilde{h}^{\rm inertial}_{\ell m}(-f)`, both keyed by
        :math:`(\ell, m)` and both spanning all :math:`-\ell \leq m \leq
        \ell` for every :math:`\ell` present in the input.
    """
    positive_f: Dict[ModeKey, np.ndarray] = {}
    negative_f: Dict[ModeKey, np.ndarray] = {}

    # Every co-precessing multipole is twisted on its own, because each
    # gets its own set of Euler angles; the results are then summed.
    for (ell, n), mode_array in coprecessing_modes.items():
        # Stationary-phase orbital frequency of this multipole: its GW
        # frequency is n / (2 pi) times the orbital one.
        momega = 2.0 * np.pi * frequencies * mass_sum_seconds / n
        alpha, beta, gamma = angles.at_momega(momega)

        wanted = [(ell, m) for m in range(1, ell + 1)]
        for target, include_positive_n in ((positive_f, True), (negative_f, False)):
            twisted, twisted_negative_m, twisted_m0 = twist_modes(
                {(ell, n): mode_array},
                alpha,
                beta,
                gamma,
                lm_inertial=wanted,
                include_positive_n=include_positive_n,
                include_negative_n=not include_positive_n,
            )
            for contribution in (twisted, twisted_negative_m, twisted_m0):
                for key, value in contribution.items():
                    if key in target:
                        target[key] = target[key] + value
                    else:
                        target[key] = value

    return positive_f, negative_f


def polarizations_from_inertial_modes(
    modes_positive_f: Dict[ModeKey, np.ndarray],
    modes_negative_f: Dict[ModeKey, np.ndarray],
    inclination: float,
    azimuth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Project the inertial-frame multipoles onto the observer's sky.

    On the positive-frequency grid the two independent combinations of
    the polarizations are

    .. math::
        \tilde{h}_+ - i \tilde{h}_\times &= \sum_{\ell m}
            \tilde{h}_{\ell m}(f)\; {}_{-2}Y_{\ell m}(\iota, \varphi) \\
        \tilde{h}_+ + i \tilde{h}_\times &= \sum_{\ell m}
            \tilde{h}_{\ell m}^*(-f)\; {}_{-2}Y_{\ell m}^*(\iota, \varphi)

    the second following from the reality of the time-domain strain.
    For an aligned-spin source the two are related by the equatorial
    symmetry of the multipoles and only the first is needed; under
    precession that symmetry is broken and both are required.

    Parameters
    ----------
    modes_positive_f, modes_negative_f : dict[tuple[int, int], np.ndarray]
        Inertial-frame multipoles at :math:`+f` and :math:`-f`, as
        returned by :func:`twist_modes_frequency_domain`.
    inclination : float
        Polar angle :math:`\iota` of the line of sight, in radians.
    azimuth : float
        Azimuthal angle :math:`\varphi` of the line of sight, in radians.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The complex polarizations ``(h_plus, h_cross)``, in the same
        convention as :meth:`~mlgw_bns.model.Model.predict`.
    """
    def harmonic(key: ModeKey) -> complex:
        real, imaginary = spinsphericalharm(-2, key[0], key[1], azimuth, inclination)
        return complex(real, imaginary)

    harmonics = {
        key: harmonic(key) for key in set(modes_positive_f) | set(modes_negative_f)
    }

    h_minus = sum(
        mode_array * harmonics[key] for key, mode_array in modes_positive_f.items()
    )
    h_plus_combination = sum(
        np.conj(mode_array) * np.conj(harmonics[key])
        for key, mode_array in modes_negative_f.items()
    )

    h_plus = (h_minus + h_plus_combination) / 2.0
    h_cross = 1j * (h_minus - h_plus_combination) / 2.0
    return h_plus, h_cross


class PrecessingModel:
    r"""A precessing waveform model built on an aligned-spin :class:`Model`.

    Parameters
    ----------
    model : Model
        The aligned-spin surrogate providing the co-precessing
        multipoles.

    Examples
    --------
    >>> from mlgw_bns.model import Model                     # doctest: +SKIP
    >>> precessing = PrecessingModel(Model.default_for_testing())  # doctest: +SKIP
    >>> hp, hc = precessing.predict(frequencies, params)     # doctest: +SKIP
    """

    def __init__(self, model: Model) -> None:
        self.model = model

    @property
    def largest_mode_m(self) -> int:
        r"""The largest :math:`m` among the surrogate's modes."""
        return max(mode.m for mode in self.model.modes)

    def euler_angles(
        self,
        params: PrecessingParametersWithExtrinsic,
        initial_frequency_hz: float,
        omega_dot: Optional[Callable[[float], float]] = None,
        anchor_to_reference_phase: bool = False,
    ) -> EulerAngles:
        """The precession angles for a source, from the PN dynamics.

        See :func:`euler_angles`, of which this is the bound version.

        Parameters
        ----------
        params : PrecessingParametersWithExtrinsic
            Source parameters.
        initial_frequency_hz : float
            Lowest frequency, in Hz, at which the waveform is wanted.
        omega_dot : callable, optional
            Orbital-frequency rate to march the precession along; see
            :func:`euler_angles`.
        anchor_to_reference_phase : bool
            If ``True`` (and ``omega_dot`` is not given), build
            ``omega_dot`` from this model's own aligned-spin :math:`(2,
            2)` phase for ``params`` -- i.e. integrate the precession
            along the surrogate's EOB-accurate orbital-frequency track
            rather than the PN flux. This is the surrogate analogue of
            TEOBResumS' ``SPIN_FLX_EOB`` hand-off.

        Returns
        -------
        EulerAngles
            The angles, tabulated against :math:`M \\Omega_{\\rm orb}`.
        """
        if omega_dot is None and anchor_to_reference_phase:
            reference_frequencies = np.geomspace(
                max(initial_frequency_hz, self.model.dataset.initial_frequency_hz),
                self.model.dataset.effective_srate_hz / 2.0,
                1024,
            )
            amplitude, phase = self.model.predict_amplitude_phase_mode(
                Mode(2, 2), reference_frequencies, params.aligned()
            )
            omega_dot = eob_orbital_frequency_rate(
                reference_frequencies,
                amplitude * np.exp(1j * phase),
                params.aligned().mass_sum_seconds,
            )
        return euler_angles(
            params,
            initial_frequency_hz,
            largest_mode_m=self.largest_mode_m,
            omega_dot=omega_dot,
        )

    def predict_modes_dict(
        self,
        frequencies: np.ndarray,
        params: PrecessingParametersWithExtrinsic,
        source: str = "surrogate",
        angles: Optional[EulerAngles] = None,
        reanchor: bool = True,
    ) -> Tuple[Dict[ModeKey, np.ndarray], Dict[ModeKey, np.ndarray]]:
        r"""Inertial-frame multipoles of the precessing waveform.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies at which to evaluate the waveform, in Hz.
        params : PrecessingParametersWithExtrinsic
            Source parameters.
        source : str
            Where the co-precessing multipoles come from:
            ``"surrogate"`` (default), ``"post_newtonian"`` or
            ``"eob"``. See
            :meth:`~mlgw_bns.model.Model.coprecessing_modes_dict`.
        angles : EulerAngles, optional
            Precomputed Euler angles, to save re-integrating the PN
            dynamics when comparing two sources of co-precessing
            multipoles for the same binary. Computed from ``params`` if
            ``None``.
        reanchor : bool
            Whether to re-tabulate the Euler angles against the
            time--frequency relation carried by the :math:`(2, 2)`
            co-precessing phase, rather than the 3.5PN one the PN
            integration marches with. On by default; see
            :meth:`EulerAngles.reanchored`. Halves the mismatch against
            TEOBResumS at moderate opening angles.

        Returns
        -------
        tuple[dict, dict]
            ``(modes_positive_f, modes_negative_f)``; see
            :func:`twist_modes_frequency_domain`.
        """
        if angles is None:
            angles = self.euler_angles(params, float(frequencies[0]))

        coprecessing = self.model.coprecessing_modes_dict(
            frequencies, params.aligned(), source=source
        )

        if reanchor and (2, 2) in coprecessing:
            angles = angles.reanchored(
                frequencies,
                coprecessing[(2, 2)],
                params.aligned().mass_sum_seconds,
                reference_frequency_hz=max(
                    float(frequencies[0]), self.model.dataset.initial_frequency_hz
                ),
            )

        return twist_modes_frequency_domain(
            coprecessing,
            frequencies,
            angles,
            mass_sum_seconds=params.aligned().mass_sum_seconds,
        )

    def predict(
        self,
        frequencies: np.ndarray,
        params: PrecessingParametersWithExtrinsic,
        source: str = "surrogate",
        angles: Optional[EulerAngles] = None,
        reanchor: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Predict the two polarizations of a precessing waveform.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies at which to evaluate the waveform, in Hz.
        params : PrecessingParametersWithExtrinsic
            Source parameters.
        source : str
            Where the co-precessing multipoles come from; see
            :meth:`predict_modes_dict`.
        angles : EulerAngles, optional
            Precomputed Euler angles; see :meth:`predict_modes_dict`.
        reanchor : bool
            Whether to re-anchor the Euler angles to the :math:`(2, 2)`
            phase; see :meth:`predict_modes_dict`. On by default.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The complex polarizations ``(h_plus, h_cross)``, in the same
            convention as :meth:`~mlgw_bns.model.Model.predict`.
        """
        modes_positive_f, modes_negative_f = self.predict_modes_dict(
            frequencies, params, source=source, angles=angles, reanchor=reanchor
        )
        return polarizations_from_inertial_modes(
            modes_positive_f,
            modes_negative_f,
            inclination=params.inclination,
            azimuth=params.azimuth,
        )


def check_aligned_spin_limit(
    model: Model,
    frequencies: np.ndarray,
    params: PrecessingParametersWithExtrinsic,
    rtol: float = 1e-8,
) -> None:
    r"""Assert that the twist is the identity for aligned spins.

    With no in-plane spin the orbital angular momentum never tilts, so
    :math:`\beta \equiv 0` and the Wigner-:math:`D` matrices collapse to
    the identity: the precessing pipeline must then return exactly what
    :meth:`~mlgw_bns.model.Model.predict` returns. This exercises every
    convention in the chain at once --- the sign of the rotation, the
    split between the :math:`\pm f` halves of the twist, and the
    normalisation of the multipoles --- so it is the check worth running
    after touching any of them.

    Parameters
    ----------
    model : Model
        The aligned-spin surrogate.
    frequencies : np.ndarray
        Frequencies at which to compare, in Hz.
    params : PrecessingParametersWithExtrinsic
        Source parameters. Its in-plane spin components are ignored: the
        comparison is made for the aligned-spin source with the same
        :math:`\chi_z`.
    rtol : float
        Relative tolerance of the comparison. Defaults to ``1e-8``.

    Raises
    ------
    AssertionError
        If the two waveforms differ by more than ``rtol``.
    """
    aligned_params = PrecessingParametersWithExtrinsic(
        mass_ratio=params.mass_ratio,
        lambda_1=params.lambda_1,
        lambda_2=params.lambda_2,
        chi_1=(0.0, 0.0, params.chi_1_vector[2]),
        chi_2=(0.0, 0.0, params.chi_2_vector[2]),
        distance_mpc=params.distance_mpc,
        inclination=params.inclination,
        total_mass=params.total_mass,
        azimuth=0.0,
    )

    reference = aligned_params.aligned()
    reference.inclination = params.inclination
    expected_plus, expected_cross = model.predict(frequencies, reference)

    predicted_plus, predicted_cross = PrecessingModel(model).predict(
        frequencies, aligned_params
    )

    for expected, predicted, name in (
        (expected_plus, predicted_plus, "h_plus"),
        (expected_cross, predicted_cross, "h_cross"),
    ):
        scale = np.max(np.abs(expected))
        assert np.allclose(expected, predicted, rtol=rtol, atol=rtol * scale), (
            f"the aligned-spin limit of the precessing twist does not reproduce "
            f"{name}: largest deviation "
            f"{np.max(np.abs(expected - predicted)) / scale:.3e} (relative)"
        )
