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

The orbital phase at the reference frequency
--------------------------------------------

With precession the orbital phase is no longer a choice of observer: it
sets the angle between the orbital separation and the in-plane spins. As
in LALSuite (``f_ref``, ``phiRef``, LIGO-T1500606), NRSur7dq4, SEOBNR and
TEOBResumS (at the start of its integration), it is fixed at the
frequency where the spins are given, ``reference_frequency_hz``; the
aligned-spin surrogate instead puts it to zero at the merger. The
stationary-phase transform :math:`X_{\ell m}(f) = \Psi_{\ell m} - f
\Psi'_{\ell m}` of a frequency-domain multipole's phase does not change
under time shifts, and at :math:`f = m F` equals

.. math::
    X_{\ell m}(m F) = m \phi_{\rm orb} + \delta_{\ell m} + \pi / 4 \,,

with :math:`\phi_{\rm orb}` the orbital phase when the orbital frequency
is :math:`F`, and :math:`\delta_{\ell m}` the multipole's phase at zero
orbital phase. :meth:`PrecessingModel.reference_orbital_phase` reads
:math:`\phi_{\rm orb}` off the surrogate's own :math:`(2, 2)` at the
reference frequency (modulo :math:`\pi`, the branch from the
:math:`(2, 1)` or :math:`(3, 3)`), taking for :math:`\delta_{\ell m}`
its leading-order value (:data:`LEADING_ORDER_MODE_PHASES`), and the
multipoles are rotated so that it equals
:attr:`PrecessingParametersWithExtrinsic.reference_phase`. Like
NRSur7dq4's, this orbital phase is defined by the waveform: TEOBResumS'
dynamical one differs from it by its multipoles' higher-order phase
corrections, ~5e-3 m rad at 10 Hz for a binary neutron star. The spins,
on the other hand, are placed on the PN orbital-frequency track the Euler
angles are integrated (and looked up) along, as in TEOBResumS; the two
read the same instant slightly differently (TEOBResumS' EOB and PN spin
dynamics start at 9.506 and 9.5 Hz), which a comparison with it has to
account for (:meth:`PrecessingModel.teob_reference_phase`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator

from .higher_order_modes import Mode
from .model import Model
from .mode_model import ParametersWithExtrinsic
from .special_func import spinsphericalharm
from .taylorf2 import SUN_MASS_SECONDS
from .twist_waveform import integrate_pn_spin_precession, twist_modes

#: Key of a spherical-harmonic multipole, ``(l, m)``.
ModeKey = Tuple[int, int]

#: Phase of a co-precessing multipole at zero orbital phase, to leading
#: (Newtonian) order: :math:`\delta_{\ell m}` in
#: :math:`X_{\ell m}(m F) = m \phi_{\rm orb} + \delta_{\ell m} + \pi / 4`
#: (see the module docstring). TEOBResumS' multipoles at the first sample
#: of its integration have these plus ~5e-3 m rad at 10 Hz (a binary
#: neutron star), the same for every binary.
LEADING_ORDER_MODE_PHASES: Dict[ModeKey, float] = {
    (2, 2): -np.pi,
    (2, 1): np.pi / 2.0,
    (3, 3): -np.pi / 2.0,
}

#: Half-width, in Hz, of the window :func:`stationary_phase_transform` is
#: fitted on: narrow enough for a cubic to describe the phase (which at
#: 10 Hz advances by ~2 pi per mHz for a binary neutron star), wide enough
#: to be insensitive to its rounding.
STATIONARY_PHASE_HALF_WIDTH_HZ = 2e-3


def stationary_phase_window(frequency: float) -> np.ndarray:
    """The frequencies :func:`stationary_phase_transform` is fitted on, in Hz."""
    return np.linspace(
        frequency - STATIONARY_PHASE_HALF_WIDTH_HZ,
        frequency + STATIONARY_PHASE_HALF_WIDTH_HZ,
        401,
    )


def stationary_phase_transform(
    frequencies: np.ndarray, phase: np.ndarray, frequency: float
) -> float:
    r""":math:`X = \Psi - f \Psi'` at ``frequency``.

    From a cubic fit of the continuous phase :math:`\Psi` sampled at
    ``frequencies`` (:func:`stationary_phase_window`). It is unchanged by
    a time shift, which adds :math:`2 \pi f t` to :math:`\Psi`, and equals
    the time-domain phase plus :math:`\pi / 4` at the stationary time.
    It agrees to ~1e-6 rad with :math:`\Psi - f \Psi'` from the
    surrogate's analytic :math:`\Psi'`, which
    :mod:`~mlgw_bns.batched_precession` uses. A quadratic fit was biased
    by :math:`f \Psi''' w^2 / 10` (:math:`w` the half-width): -3e-3 rad
    for the :math:`(2, 2)` of a binary neutron star at 9.5 Hz.
    """
    c3, c2, c1, c0 = np.polyfit(frequencies - frequency, phase, 3)
    return float(c0 - frequency * c1)


def reference_phase_keys(modes: Sequence[Mode]) -> list:
    r"""The multipoles the orbital phase at the reference frequency is read
    from: the :math:`(2, 2)` and, for the branch, the :math:`(2, 1)` (else the
    :math:`(3, 3)`) if ``modes`` has it; in increasing :math:`m`."""
    keys = [(2, 2)] + [key for key in ((2, 1), (3, 3)) if Mode(*key) in modes][:1]
    return sorted(keys, key=lambda key: key[1])


def orbital_phase_from_transforms(transforms: Dict[ModeKey, float], xp=np):
    r"""The orbital phase given the stationary-phase transforms
    :math:`X_{\ell m}` of the :func:`reference_phase_keys` multipoles.

    From the :math:`(2, 2)`, modulo :math:`\pi`; the branch is the one
    closer to the estimate of the odd-:math:`m` multipole, if there is one.
    In :math:`(-\pi, \pi]`. ``xp`` is ``numpy`` (default) or
    ``jax.numpy``, and the transforms may then be arrays.
    """
    # modulo 2 pi / m
    estimates = {
        key[1]: (transform - LEADING_ORDER_MODE_PHASES[key] - np.pi / 4.0) / key[1]
        for key, transform in transforms.items()
    }
    phase = estimates[2]
    odd = [m for m in estimates if m % 2]
    if odd:
        (m_odd,) = odd

        def distance(candidate):
            return xp.abs(xp.angle(xp.exp(1j * m_odd * (candidate - estimates[m_odd]))))

        phase = xp.where(
            distance(phase + np.pi) < distance(phase), phase + np.pi, phase
        )
    return xp.angle(xp.exp(1j * phase))


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
    along the orbital angular momentum :math:`\hat{L}_0` there, and the
    spin vectors are the ones at that frequency.

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
        Orbital phase at ``reference_frequency_hz``, in radians (see the
        module docstring), which for a precessing binary is the angle
        between the orbital separation and the in-plane spins; at the
        merger, as for the aligned-spin model, if
        ``reference_frequency_hz`` is ``None``. Defaults to 0.
    merger_time : float
        Time of the merger, in seconds. Defaults to 0.
    reference_frequency_hz : float, optional
        The :math:`(2, 2)` gravitational-wave frequency, in Hz, at which
        ``chi_1``, ``chi_2``, :math:`\hat{L}_0 = \hat{z}` and
        ``reference_phase`` are given ---
        TEOBResumS' ``initial_frequency``, LALSuite's ``f_ref``. Precession
        rotates both the in-plane spins and :math:`\hat{L}` between any
        two frequencies, so the same vectors given at two different
        frequencies are two different binaries. ``None`` (default) means
        wherever the PN spin-precession integration starts, which is
        well below the requested band (see :func:`euler_angles`).
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
    merger_time: float = 0.0
    reference_frequency_hz: Optional[float] = None

    @classmethod
    def from_lvk(
        cls,
        mass_1: float,
        mass_2: float,
        theta_jn: float,
        phi_jl: float,
        tilt_1: float,
        tilt_2: float,
        phi_12: float,
        a_1: float,
        a_2: float,
        phase: float,
        reference_frequency_hz: float,
        lambda_1: float,
        lambda_2: float,
        distance_mpc: float,
        merger_time: float = 0.0,
    ) -> "PrecessingParametersWithExtrinsic":
        r"""A binary given by the LVK spin angles, at ``reference_frequency_hz``.

        Masses in solar masses, the heavier first; see
        :func:`mlgw_bns.spin_conversion.lvk_to_precessing`.
        """
        from .spin_conversion import lvk_to_precessing

        orientation = lvk_to_precessing(
            theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
            reference_frequency_hz, phase,
        )
        return cls(
            mass_ratio=mass_1 / mass_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            chi_1=tuple(float(c) for c in orientation.chi_1),
            chi_2=tuple(float(c) for c in orientation.chi_2),
            distance_mpc=distance_mpc,
            inclination=float(orientation.inclination),
            total_mass=mass_1 + mass_2,
            azimuth=float(orientation.azimuth),
            reference_phase=float(orientation.reference_phase),
            merger_time=merger_time,
            reference_frequency_hz=reference_frequency_hz,
        )

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
        sky projection --- the projection happens after the twist. Its
        coalescence phase is ``reference_phase`` only if
        ``reference_frequency_hz`` is ``None``: otherwise
        :class:`PrecessingModel` sets the orbital phase at the reference
        frequency itself.

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
            coalescence_phase=(
                self.reference_phase if self.reference_frequency_hz is None else 0.0
            ),
            merger_time=self.merger_time,
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
    """

    momega: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray

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

        A monotone cubic Hermite interpolant (PCHIP) is used in place of
        the plain linear one used before, closer in spirit to
        TEOBResumS' own ``gsl_interp_cspline`` lookup of these angles
        (``twist_hlm_FD``/``twist_hlm_TD`` in ``TEOBResumSWaveform.c``):
        the PN integration returns only a sparse set of accepted ODE
        steps (see
        :func:`~mlgw_bns.twist_waveform.integrate_pn_spin_precession`),
        and linear interpolation between them is a coarser approximation
        of the (smooth) angle trajectory. PCHIP is shape-preserving (no
        overshoot between knots, whatever their spacing, which spans
        orders of magnitude here), and it still reproduces the ODE
        solution exactly at every returned node.
        """
        momega = np.asarray(momega)
        clipped = np.clip(momega, self.momega[0], self.momega[-1])

        # the interpolator needs strictly increasing knots
        unique_momega, unique_index = np.unique(self.momega, return_index=True)

        def spline(values: np.ndarray) -> np.ndarray:
            return PchipInterpolator(unique_momega, values[unique_index])(clipped)

        return (
            spline(np.unwrap(self.alpha)),
            spline(self.beta),
            spline(np.unwrap(self.gamma)),
        )


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
    # Where the spins are given: the PN state starts there, and is also
    # integrated backwards if the waveform needs lower frequencies.
    reference_frequency_22 = (
        initial_frequency_22
        if params.reference_frequency_hz is None
        else params.reference_frequency_hz * mass_sum_seconds
    )

    solution = integrate_pn_spin_precession(
        nu=params.eta,
        chi1vec=params.chi_1_vector,
        chi2vec=params.chi_2_vector,
        f0=reference_frequency_22,
        f_start=initial_frequency_22,
        lambdas=(params.lambda_1, params.lambda_2),
        t_max=2.0 * newtonian_time_to_merger(params.eta, np.pi * initial_frequency_22),
        independent_variable="time" if omega_dot is None else "orbital_frequency",
        omega_dot=omega_dot,
    )

    return EulerAngles(
        momega=solution["Momega"],
        alpha=solution["alpha"],
        beta=solution["beta"],
        gamma=backward_gamma(
            solution["gamma"],
            solution["Momega"],
            np.hypot(solution["Lh"][:, 0], solution["Lh"][:, 1]),
            np.pi * reference_frequency_22,
        ),
    )


def backward_gamma(gamma, momega, in_plane_l, reference_momega, xp=np):
    r""":math:`\gamma`, continued through the reference point.

    At the reference point :math:`\hat{L} = \hat{z}`, so the in-plane part of
    :math:`\hat{L}` passes through zero there and changes direction:
    :math:`\alpha = \arctan(L_y / L_x)` jumps by :math:`\pi`, while
    :math:`\gamma`, integrated, does not. The rotation
    :math:`R(\alpha + \pi, \beta, \gamma) = R(\alpha, -\beta, \gamma + \pi)`
    then jumps by :math:`\pi` about :math:`\hat{L}`, flipping the sign of every
    odd-:math:`m` co-precessing multipole below the reference frequency.
    TEOBResumS' backward integration does the same, invisibly: its multipoles
    start at the reference point. Adding :math:`\pi` to :math:`\gamma` where
    :math:`\alpha` is flipped (below the reference, wherever
    :math:`\hat{L} \neq \hat{z}`) makes the rotation continuous; with no
    in-plane spin nothing changes.
    """
    flipped = (momega < reference_momega) & (in_plane_l > 0)
    return gamma + xp.where(flipped, np.pi, 0.0)


def twist_modes_frequency_domain(
    coprecessing_modes: Dict[ModeKey, np.ndarray],
    frequencies: np.ndarray,
    angles: EulerAngles,
    mass_sum_seconds: float,
    xp=np,
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
        Precession angles, tabulated against orbital frequency: anything
        with an ``at_momega`` method, such as
        :class:`~mlgw_bns.batched_precession.TabulatedAngles`.
    mass_sum_seconds : float
        Total mass of the binary, in seconds, used to convert the
        frequencies to geometric units.
    xp : module
        Array namespace, ``numpy`` (default) or ``jax.numpy``; with the
        latter, ``frequencies`` and ``mass_sum_seconds`` may carry a batch
        axis, ``(N, k)`` and ``(N, 1)``.

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

        # twist_modes applies the time-domain rotation D(alpha, beta,
        # gamma), for multipoles in the h = A exp(-i phi(t)) convention.
        # The surrogate's frequency-domain multipoles are the *complex
        # conjugates* of those multipoles' transforms (their phase falls
        # with f: d phi / df = +2 pi x time to merger), and since the
        # Wigner d-matrix is real, the rotation acting on them is D* =
        # D(-alpha, beta, -gamma). Using D instead mirrors the precession,
        # invisibly in the aligned-spin limit (alpha = gamma = 0): against
        # TEOBResumS' own h+, hx that was a mismatch of ~9e-2 at beta ~
        # 0.27, ~5e-4 with the sign fixed; TEOBResumS' frequency-domain
        # twist (twist_hlm_FD) was in turn checked against the Fourier
        # transform of its time-domain one to ~2e-4.
        wanted = [(ell, m) for m in range(1, ell + 1)]
        for target, include_positive_n in ((positive_f, True), (negative_f, False)):
            twisted, twisted_negative_m, twisted_m0 = twist_modes(
                {(ell, n): mode_array},
                -alpha,
                beta,
                -gamma,
                lm_inertial=wanted,
                include_positive_n=include_positive_n,
                include_negative_n=not include_positive_n,
                xp=xp,
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
    xp=np,
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
    xp : module
        Array namespace, ``numpy`` (default) or ``jax.numpy``; the angles
        may then be arrays broadcasting against the multipoles.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The complex polarizations ``(h_plus, h_cross)``, in the same
        convention as :meth:`~mlgw_bns.model.Model.predict`.
    """
    def harmonic(key: ModeKey):
        real, imaginary = spinsphericalharm(
            -2, key[0], key[1], azimuth, inclination, xp=xp
        )
        return real + 1j * imaginary

    harmonics = {
        key: harmonic(key) for key in set(modes_positive_f) | set(modes_negative_f)
    }

    h_minus = sum(
        mode_array * harmonics[key] for key, mode_array in modes_positive_f.items()
    )
    h_plus_combination = sum(
        xp.conj(mode_array) * xp.conj(harmonics[key])
        for key, mode_array in modes_negative_f.items()
    )

    h_plus = (h_minus + h_plus_combination) / 2.0
    h_cross = 1j * (h_minus - h_plus_combination) / 2.0
    return h_plus, h_cross


def twist_coefficients(
    keys: Sequence[ModeKey],
    frequencies: np.ndarray,
    angles: EulerAngles,
    mass_sum_seconds: float,
    inclination: float,
    azimuth: float,
    xp=np,
) -> Dict[ModeKey, Tuple[np.ndarray, np.ndarray]]:
    r"""The twist and sky projection of each co-precessing multipole.

    The polarizations are linear in the co-precessing multipoles:

    .. math::
        \tilde{h}_{+, \times}(f) = \sum_{\ell m} c^{+, \times}_{\ell m}(f)
        \, \tilde{h}^{\rm co}_{\ell m}(f) \,.

    :func:`twist_modes_frequency_domain` builds the inertial multipoles at
    :math:`-f` from the complex conjugates of the co-precessing ones, but
    :func:`polarizations_from_inertial_modes` conjugates those again, so
    the coefficients are those of a unit multipole twisted and projected.
    They vary only on the precession time scale --- what mode-by-mode
    relative binning interpolates (Leslie, Dai and Pratten,
    `arXiv:2109.09872 <https://arxiv.org/abs/2109.09872>`_, eq. 3).

    Parameters
    ----------
    keys : sequence of (l, m)
        Co-precessing multipoles, :math:`m > 0`.
    frequencies : np.ndarray
        The (positive) frequency grid, in Hz.
    angles : EulerAngles
        Precession angles; anything with an ``at_momega`` method.
    mass_sum_seconds : float
        Total mass of the binary, in seconds.
    inclination, azimuth : float
        Line of sight, in radians; see
        :class:`PrecessingParametersWithExtrinsic`.
    xp : module
        Array namespace, ``numpy`` (default) or ``jax.numpy``.

    Returns
    -------
    dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
        Mapping ``(l, m) -> (c_plus, c_cross)``, on ``frequencies``.
    """
    unit = xp.ones_like(xp.asarray(frequencies, dtype=float), dtype=complex)
    coefficients = {}
    for key in keys:
        key = (int(key[0]), int(key[1]))
        positive_f, negative_f = twist_modes_frequency_domain(
            {key: unit}, frequencies, angles, mass_sum_seconds, xp=xp
        )
        coefficients[key] = polarizations_from_inertial_modes(
            positive_f, negative_f, inclination, azimuth, xp=xp
        )
    return coefficients


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

    def jax_predict(
        self, modes: Optional[Sequence] = None, n_steps: Optional[int] = None
    ) -> Callable:
        r"""A pure JAX function computing :meth:`predict` for a batch of binaries.

        See :func:`mlgw_bns.batched_precession.precessing_waveform` for its
        signature. Wrap it in :func:`jax.jit`; every new input shape compiles
        again. Requires the ``jax`` extra.

        Parameters
        ----------
        modes : sequence of (l, m), optional
            Co-precessing multipoles to twist; defaults to all of the model's.
        n_steps : int, optional
            Steps of each leg of the precession integration; defaults to
            :data:`mlgw_bns.batched_precession.N_STEPS`.
        """
        from .batched_precession import N_STEPS, precessing_waveform

        return precessing_waveform(
            self.model, modes, N_STEPS if n_steps is None else n_steps
        )

    def jax_predict_modes(
        self, modes: Optional[Sequence] = None, n_steps: Optional[int] = None
    ) -> Callable:
        r"""A pure JAX function giving each co-precessing multipole and its
        twist, for a batch of binaries.

        See :func:`mlgw_bns.batched_precession.precessing_mode_components`;
        the arguments are those of :meth:`jax_predict`.
        """
        from .batched_precession import N_STEPS, precessing_mode_components

        return precessing_mode_components(
            self.model, modes, N_STEPS if n_steps is None else n_steps
        )

    def mode_components(
        self,
        frequencies: np.ndarray,
        params: PrecessingParametersWithExtrinsic,
        source: str = "surrogate",
        angles: Optional[EulerAngles] = None,
    ) -> Dict[ModeKey, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        r"""Each co-precessing multipole and its twist coefficients.

        The numpy counterpart of :meth:`jax_predict_modes`: :meth:`predict` is
        ``sum(c_plus * coprecessing), sum(c_cross * coprecessing)`` over the
        multipoles (see :func:`twist_coefficients`).

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

        Returns
        -------
        dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]
            Mapping ``(l, m) -> (coprecessing, c_plus, c_cross)``.
        """
        if angles is None:
            angles = self.euler_angles(params, float(frequencies[0]))
        amplitudes_and_phases = self.coprecessing_amplitudes_and_phases(
            frequencies, params, source=source
        )
        coefficients = twist_coefficients(
            list(amplitudes_and_phases),
            frequencies,
            angles,
            params.aligned().mass_sum_seconds,
            params.inclination,
            params.azimuth,
        )
        return {
            key: (amplitude * np.exp(1j * phase), *coefficients[key])
            for key, (amplitude, phase) in amplitudes_and_phases.items()
        }

    def reference_orbital_phase(
        self,
        params: PrecessingParametersWithExtrinsic,
        source: str = "surrogate",
    ) -> float:
        r"""Orbital phase of the bare co-precessing multipoles at the reference frequency.

        The multipoles as :meth:`~mlgw_bns.model.Model.coprecessing_modes_dict`
        returns them for ``params.aligned()``, at the time their orbital
        frequency is half of ``params.reference_frequency_hz``. From the
        :math:`(2, 2)`, modulo :math:`\pi`; the branch is the one closer to
        the estimate of the :math:`(2, 1)` (else the :math:`(3, 3)`), and
        irrelevant if the model has no odd-:math:`m` multipole. See the
        module docstring.

        Parameters
        ----------
        params : PrecessingParametersWithExtrinsic
            Source parameters; ``reference_frequency_hz`` must be set.
        source : str
            Where the co-precessing multipoles come from; see
            :meth:`predict_modes_dict`.

        Returns
        -------
        float
            The orbital phase, in radians, in :math:`(-\pi, \pi]`.
        """
        if params.reference_frequency_hz is None:
            raise ValueError("the parameters have no reference frequency")
        keys = reference_phase_keys(self.model.modes)
        windows = [
            stationary_phase_window(key[1] / 2.0 * params.reference_frequency_hz)
            for key in keys
        ]
        # one evaluation for every window; the phase is continuous within
        # each, which is all the fit needs
        phases = self.model.coprecessing_amplitudes_and_phases(
            np.concatenate(windows), params.aligned(), source=source
        )
        transforms = {}
        start = 0
        for key, window in zip(keys, windows):
            transforms[key] = stationary_phase_transform(
                window,
                phases[key][1][start : start + window.size],
                key[1] / 2.0 * params.reference_frequency_hz,
            )
            start += window.size
        return float(orbital_phase_from_transforms(transforms))

    def teob_reference_phase(
        self,
        params: PrecessingParametersWithExtrinsic,
        start_momega: float,
        start_orbital_phase: float,
        start_phase_22: float,
    ) -> float:
        r"""The ``reference_phase`` that reproduces a TEOBResumS precessing run.

        TEOBResumS imposes the spins, and puts its orbital phase to zero, at
        the first sample of its integration, which two of its clocks read
        differently: its PN spin dynamics, along whose orbital frequency the
        Euler angles are read, starts at 0.95 ``initial_frequency`` (in the
        frequency domain); its EOB dynamics, which the waveform follows, at a
        slightly different frequency (~9.506 Hz for 9.5). ``params`` must give
        the spins at the former, ``reference_frequency_hz = 0.95 *
        initial_frequency``. The orbital phase is then carried there from the
        latter along this model's own phase (a ~0.4 rad advance in those 6 mHz
        for a binary neutron star). At the start of the integration it is not
        zero in the sense of the module docstring, but the offset of
        TEOBResumS' :math:`(2, 2)` from its leading-order phase (~5e-3 rad,
        the multipole's higher-order phase corrections).

        Parameters
        ----------
        params : PrecessingParametersWithExtrinsic
            The binary, with ``reference_frequency_hz`` where TEOBResumS
            imposes the spins.
        start_momega : float
            :math:`M \Omega` of TEOBResumS' EOB dynamics at its first sample
            (``MOmega[0]`` of the dynamics ``EOBRunPy`` returns with
            ``arg_out``).
        start_orbital_phase : float
            Its orbital phase there (``Phi[0]``, zero).
        start_phase_22 : float
            The phase of its time-domain :math:`(2, 2)` multipole there, in
            its own convention :math:`h_{22} = A e^{-i \phi}`.

        Returns
        -------
        float
            The orbital phase at ``params.reference_frequency_hz``, in
            :math:`(-\pi, \pi]`.
        """
        offset = start_phase_22 - 2.0 * start_orbital_phase - LEADING_ORDER_MODE_PHASES[(2, 2)]
        start = replace(params, reference_frequency_hz=(
            start_momega / (np.pi * params.aligned().mass_sum_seconds)))
        phase = (
            np.angle(np.exp(1j * offset)) / 2.0
            - self.reference_orbital_phase(start)
            + self.reference_orbital_phase(params)
        )
        return float(np.angle(np.exp(1j * phase)))

    def coprecessing_amplitudes_and_phases(
        self,
        frequencies: np.ndarray,
        params: PrecessingParametersWithExtrinsic,
        source: str = "surrogate",
    ) -> Dict[ModeKey, Tuple[np.ndarray, np.ndarray]]:
        r"""The co-precessing multipoles, before the twist.

        Those of :meth:`~mlgw_bns.model.Model.coprecessing_amplitudes_and_phases`
        for ``params.aligned()``, with the orbital phase at the reference
        frequency set to ``params.reference_phase`` if
        ``params.reference_frequency_hz`` is given (see the module
        docstring). :meth:`predict_modes_dict` twists these.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies at which to evaluate the multipoles, in Hz.
        params : PrecessingParametersWithExtrinsic
            Source parameters.
        source : str
            Where the co-precessing multipoles come from; see
            :meth:`predict_modes_dict`.

        Returns
        -------
        dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
            Mapping ``(l, m) -> (amplitude, phase)``.
        """
        amplitudes_and_phases = self.model.coprecessing_amplitudes_and_phases(
            frequencies, params.aligned(), source=source
        )
        if params.reference_frequency_hz is None:
            return amplitudes_and_phases
        rotation = params.reference_phase - self.reference_orbital_phase(
            params, source=source
        )
        return {
            (l, m): (amplitude, phase + m * rotation)
            for (l, m), (amplitude, phase) in amplitudes_and_phases.items()
        }

    def predict_modes_dict(
        self,
        frequencies: np.ndarray,
        params: PrecessingParametersWithExtrinsic,
        source: str = "surrogate",
        angles: Optional[EulerAngles] = None,
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

        Returns
        -------
        tuple[dict, dict]
            ``(modes_positive_f, modes_negative_f)``; see
            :func:`twist_modes_frequency_domain`.
        """
        if angles is None:
            angles = self.euler_angles(params, float(frequencies[0]))

        amplitudes_and_phases = self.coprecessing_amplitudes_and_phases(
            frequencies, params, source=source
        )
        coprecessing = {
            key: amplitude * np.exp(1j * phase)
            for key, (amplitude, phase) in amplitudes_and_phases.items()
        }

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

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The complex polarizations ``(h_plus, h_cross)``, in the same
            convention as :meth:`~mlgw_bns.model.Model.predict`.
        """
        modes_positive_f, modes_negative_f = self.predict_modes_dict(
            frequencies, params, source=source, angles=angles
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
