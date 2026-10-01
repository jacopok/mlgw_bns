r"""TEOBResumS' precessing waveforms, in ``mlgw_bns``' conventions.

Helpers shared by the precessing validation in ``validate_model.py`` and
``benchmark_evaluation_time.py``:

* :func:`teob_run`, one frequency-domain TEOBResumS call for a precessing
  binary, with the polarizations in the ``mlgw_bns`` convention and, on
  request, its co-precessing multipoles and dynamics (the conventions are
  those of ``docs/explanation/precession.md``);
* the random draws of in-plane spins and of a detector's orientation;
* :func:`alpha_glitched`, the frequencies at which TEOBResumS' own
  frequency-domain twist reads its precession angle :math:`\alpha` across
  one of its spurious :math:`2\pi` steps, to leave out of a comparison.
  ``prolong_euler_angles_FD`` unwraps :math:`\alpha` with ``unwrap_HM``,
  which misses the wraps of :math:`\alpha` running backwards through
  :math:`\pm\pi` (reported upstream to the TEOBResumS developers).
"""

from __future__ import annotations

import os

import numpy as np

from mlgw_bns.higher_order_modes import Mode, mode_to_k

#: The co-precessing multipoles :func:`teob_run` asks for by default.
MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]

#: Largest in-plane spin magnitude for either body. Neutron-star spins in
#: a BNS are small; the range is pushed a little past the astrophysical
#: expectation on purpose, to see where the precession model breaks down.
MAX_IN_PLANE_SPIN = 0.4

TOTAL_MASS = 2.8
DISTANCE_MPC = 100.0

#: TEOBResumS frequency-domain resolution, in Hz. Fine enough that
#: ``1 / DF`` exceeds the full inspiral length from ``F0_TEOB``.
DF = 1.0 / 2048.0
#: Where to start the TEOBResumS integration, in Hz --- low enough that
#: every multipole is present throughout the comparison band. TEOBResumS'
#: (l, m) multipole is identically zero below (m / 2) * 0.95 * F0_TEOB
#: (see SPIN_REFERENCE_FREQUENCY_HZ for the 0.95): starting at 15 Hz would
#: leave no (4, 4) below 28.5 Hz and no (3, 3) below 21.4 Hz, against a
#: surrogate that has both --- a ~1e-2 mismatch on its own for edge-on
#: views, where those multipoles weigh most.
F0_TEOB = 10.0
#: Comparison band, in Hz.
BAND_LO = 20.0
BAND_HI = 2048.0
#: Frequency at which PrecessingModel takes the spin vectors to be given.
#: In the frequency domain (``domain = 1``) TEOBResumS starts both its EOB
#: and its PN spin dynamics at 0.95 * initial_frequency, and imposes the
#: input spins (and Lhat = z) *there* -- read off its own ``dynspin.txt``
#: (``output_dynamics``); in the time domain it is initial_frequency
#: itself. ``None`` is PrecessingModel's own default, the start of its PN
#: integration, a few Hz. Either of those is a different binary as far as
#: the precession goes, by an amount that grows with the opening angle.
SPIN_REFERENCE_FREQUENCY_HZ = 0.95 * F0_TEOB

#: TEOBResumS sampling rate, in Hz; set it to the model's band before calling
#: :func:`teob_run`.
SRATE_HZ = 4096.0


#: Spline intervals of TEOBResumS' spin dynamics masked on either side of each
#: of its alpha jumps: the cubic spline's error decays by ~2 + sqrt(3) per
#: interval away from a 2 pi step, so one interval leaves ~1e-5 in some
#: mismatches, six leave nothing measurable.
JUMP_MARGIN = 6


def teob_dynspin(directory: str) -> np.ndarray:
    """TEOBResumS' spin dynamics as written to ``dynspin.txt``: columns t,
    SA (3), SB (3), Lhat (3), alpha, beta, gamma, M Omega."""
    return np.loadtxt(os.path.join(directory, "dynspin.txt"))


def alpha_jumps(dynspin: np.ndarray) -> np.ndarray:
    r"""Indices ``k`` where TEOBResumS' stored :math:`\alpha` steps by more than
    :math:`\pi` from sample ``k`` to ``k + 1``.

    These are spurious ~2 pi steps (within Delta M Omega ~ 1e-6, far less
    than a precession cycle): harmless in :math:`e^{i n \alpha}` at the
    samples, but ``twist_hlm_FD`` splines :math:`\alpha` through them, so
    between those samples (and, by the spline's ringing, next to them) its
    twist uses a wrong angle. They are the wraps of :math:`\alpha` running
    backwards through :math:`\pm\pi`, which the frequency-domain path's
    ``unwrap_HM`` misses (reported upstream to the TEOBResumS developers).
    """
    return np.flatnonzero(np.abs(np.diff(dynspin[:, 10])) > np.pi)


def alpha_glitched(dynspin, frequencies, orders, mass_sum_seconds, margin):
    """Where TEOBResumS' own twist read alpha across one of its jumps.

    Boolean mask over ``frequencies`` (Hz): the multipole ``m`` (for each
    ``m`` in ``orders``) is stationary at ``M Omega = 2 pi f M / m``, and the
    frequency is flagged if that falls in a spline interval of the spin
    dynamics within ``margin`` intervals of an :func:`alpha_jumps` step.
    """
    jumps = alpha_jumps(dynspin)
    near = np.concatenate([jumps + k for k in range(-margin, margin + 1)])
    glitched = np.zeros(frequencies.size, bool)
    for m in orders:
        sample = np.searchsorted(
            dynspin[:, 13], 2 * np.pi * frequencies * mass_sum_seconds / m) - 1
        glitched |= np.isin(sample, near)
    return glitched


def antenna_patterns(theta: float, phi: float, psi: float) -> tuple[float, float]:
    r"""Antenna patterns :math:`(F_+, F_\times)` of an L-shaped detector."""
    cos_theta = np.cos(theta)
    a = 0.5 * (1.0 + cos_theta**2) * np.cos(2.0 * phi)
    b = cos_theta * np.sin(2.0 * phi)
    f_plus = a * np.cos(2.0 * psi) - b * np.sin(2.0 * psi)
    f_cross = a * np.sin(2.0 * psi) + b * np.cos(2.0 * psi)
    return f_plus, f_cross


def random_orientation(rng: np.random.Generator) -> dict:
    """Isotropic inclination and sky position, uniform azimuth and polarization."""
    return {
        "inclination": np.arccos(rng.uniform(-1.0, 1.0)),
        "azimuth": rng.uniform(0.0, 2.0 * np.pi),
        "theta": np.arccos(rng.uniform(-1.0, 1.0)),
        "phi": rng.uniform(0.0, 2.0 * np.pi),
        "psi": rng.uniform(0.0, np.pi),
    }


def random_spin_vector(rng: np.random.Generator, chi_z: float) -> tuple:
    """An aligned spin with a random in-plane component bolted on."""
    magnitude = rng.uniform(0.0, MAX_IN_PLANE_SPIN)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return (magnitude * np.cos(angle), magnitude * np.sin(angle), chi_z)


def teob_run(params, chi_1, chi_2, inclination, azimuth, multipoles=False,
             overrides=None, modes=None, frequencies=None):
    r"""One frequency-domain TEOBResumS call for a precessing binary.

    Returns ``(f, h_plus, h_cross)`` on TEOBResumS' uniform grid, or on
    ``frequencies`` (in Hz, above ``F0_TEOB``) if given, in the
    ``mlgw_bns`` Fourier convention (``h_+ - i h_\times`` is the multipole
    sum). With ``multipoles``, also the co-precessing multipoles this very
    call twisted (``hflm``, which ``twist_hlm_FD`` only reads), as
    ``{(l, m): (amplitude, phase)}`` with the multipole ``A e^{i phi}`` in
    the ``mlgw_bns`` convention, in TEOBResumS' own time and orbital-phase
    origin (the start of its integration), and its EOB dynamics (the
    ``dyn`` dict: ``t`` in units of the total mass, ``phi``, ``MOmega``,
    ..., plus the time-domain multipoles on the same samples as
    ``multipoles``: ``{(l, m): (amplitude, phase)}``). The phase is continuous
    as returned: it advances by up to ~2 pi t df per sample, more than pi
    for t > 1 / (2 df), so it must not be unwrapped. ``overrides`` are
    extra TEOBResumS parameters, applied last. ``modes`` are the co-precessing
    multipoles, :data:`MODES` by default.

    ``frequencies`` is ~500 times faster than the uniform grid, and the same
    there to ~1e-10: TEOBResumS twists every frequency it outputs, ~6e6 of
    them on the uniform grid up to ``SRATE_HZ / 2``, and with
    ``output_dynamics`` writes its dynamics resampled to ``1 / SRATE_HZ``.
    """
    from EOBRun_module import EOBRunPy

    modes = MODES if modes is None else list(modes)
    largest_l = max(mode.l for mode in modes)

    par = dict(
        q=params.mass_ratio,
        LambdaAl2=params.lambda_1,
        LambdaBl2=params.lambda_2,
        M=TOTAL_MASS,
        distance=DISTANCE_MPC,
        initial_frequency=F0_TEOB,
        srate_interp=SRATE_HZ,
        use_geometric_units="no",
        interp_uniform_grid="yes",
        domain=1,
        df=DF,
        inclination=inclination,
        # PrecessingModel at azimuth phi is TEOBResumS at coalescence_angle
        # pi/2 + phi with h_x negated -- the mirror image of the pi/2 - phi
        # of twist_waveform.compute_hpc, which the multipoles' conjugate
        # Fourier convention introduces. Invisible without precession
        # (an aligned binary is symmetric under it), unambiguous with it:
        # checked against the Fourier transform of TEOBResumS' time-domain
        # h+, hx (1e-3 at beta ~ 0.36, against 0.16 with pi/2 - phi).
        coalescence_angle=np.pi / 2.0 + azimuth,
        output_hpc="no",
        arg_out="yes" if multipoles else "no",
        use_spins=2,
        chi1x=chi_1[0], chi1y=chi_1[1], chi1z=chi_1[2],
        chi2x=chi_2[0], chi2y=chi_2[1], chi2z=chi_2[2],
        use_mode_lm=sorted({mode_to_k(mode) for mode in modes}),
        # Every inertial multipole the twist of those feeds: by default
        # TEOBResumS keeps only the inertial (l, m) listed in use_mode_lm,
        # dropping e.g. the (3, 1), (3, 2), (4, 1)-(4, 3) that precession
        # mixes out of the co-precessing (3, 3), (4, 4). Without them its h+,
        # hx differ from the full twist of its own multipoles by 1e-5 - 1e-4,
        # growing with the opening angle.
        use_mode_lm_inertial=list(range(mode_to_k(Mode(largest_l, largest_l)) + 1)),
    )
    # The twist takes the lowest output frequency as the start of the (2, 2),
    # and zeroes each (l, m) below m / 2 times it: F0_TEOB leads the list
    # (2e-4 mismatches without it), and is dropped from the output.
    first = 0
    if frequencies is not None:
        par.update(interp_freqs="yes", freqs=[F0_TEOB, *frequencies], interp_uniform_grid="no")
        first = 1
    par.update(overrides or {})
    f, real_hp, imag_hp, real_hc, imag_hc, *extra = EOBRunPy(par)
    f = np.asarray(f)[first:]
    hp = np.conj(np.asarray(real_hp) + 1j * np.asarray(imag_hp))[first:]
    hc = -np.conj(np.asarray(real_hc) + 1j * np.asarray(imag_hc))[first:]
    if not multipoles:
        return f, hp, hc
    hflm, htlm, dynamics = extra
    coprecessing = {}
    for mode in modes:
        amplitude, phase = hflm[str(mode_to_k(mode))][:2]
        coprecessing[(mode.l, mode.m)] = (
            np.asarray(amplitude)[first:], -np.asarray(phase)[first:]
        )
    dynamics = {k: np.asarray(x) for k, x in dynamics.items()}
    dynamics["multipoles"] = {
        (mode.l, mode.m): tuple(np.asarray(x) for x in htlm[str(mode_to_k(mode))][:2])
        for mode in modes
    }
    return f, hp, hc, coprecessing, dynamics
