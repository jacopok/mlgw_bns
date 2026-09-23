r"""What frequency-to-orbital-frequency remapping does the FD twist need?

TEOBResumS' own ``domain = 1`` twist (``twist_hlm_FD`` in
``TEOBResumSWaveform.c``) always looks up the Euler angles at exactly
:math:`M\Omega_{\rm orb} = 2\pi f / n` for a co-precessing mode :math:`n`,
splined against its own *raw*, un-reanchored PN-integrated
:math:`M\Omega_{\rm orb}(t)` track -- there is no remapping step in the
C code at all; the formula is hard-coded.

:mod:`mlgw_bns.precessing_model` cannot use that literal formula as well
as TEOB does, because its own PN precession trajectory
(:math:`\alpha(M\Omega_{\rm orb})` etc., from an independent Python
integration) does not reach a given physical state at exactly the same
:math:`M\Omega_{\rm orb}` value TEOB's own integration would -- validated
directly (`validate_twist_against_teob.py`) to differ by ~1%/3e-4 rad,
small but non-zero, and TEOB's own raw :math:`M\Omega_{\rm orb}(t)` is
also known to drift from the true orbital frequency late in the
inspiral. ``EulerAngles.reanchored`` is a heuristic fix for this: it
relabels the lookup axis using the *stationary-phase* time-frequency
relation of an accurate (2,2) phase, instead of the raw PN one.

This script checks that heuristic directly against data, entirely in the
frequency domain (no TD waveform anywhere): for a single reference
binary, it feeds TEOBResumS' own co-precessing FD (2,2) multipole into
*our* twist code, and at each output frequency solves for the single
scalar :math:`M\Omega_{\rm orb}` value (looked up on *our own* PN
alpha/beta/gamma-vs-:math:`M\Omega_{\rm orb}` track) that best reproduces
TEOBResumS' own twisted :math:`h_+, h_\times(f)` (also computed with only
the (2,2) co-precessing mode active, via ``use_mode_lm =
[mode_to_k(Mode(2,2))]``, ``use_mode_lm_inertial = [k(2,1), k(2,2)]`` --
this isolates the (2,2)-sourced twist from any (3,3)/(4,4) cross-talk).

The recovered :math:`M\Omega_{\rm orb}^\star(f)` is then compared with:

* the raw formula :math:`2\pi f / n` (TEOB's own, but read off *our* PN
  track rather than TEOB's);
* :meth:`~mlgw_bns.precessing_model.EulerAngles.reanchored`'s own
  remapping.

If the recovered curve tracks the reanchored one, the heuristic already
captures the needed correction (and the residual mismatch lives
elsewhere -- in the ~1%/3e-4 rad angle-trajectory offset, or in the
per-frequency fit residual itself, i.e. genuinely in the twist/SPA
machinery). If it does not, the difference is a direct measurement of
what :meth:`reanchored` is still missing.

Run with: python visualization/recover_frequency_remapping.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from mlgw_bns.higher_order_modes import Mode, mode_to_k
from mlgw_bns.precessing_model import (
    EulerAngles,
    newtonian_time_to_merger,
    polarizations_from_inertial_modes,
)
from mlgw_bns.taylorf2 import SUN_MASS_SECONDS
from mlgw_bns.twist_waveform import (
    amp_phase_to_complex,
    integrate_pn_spin_precession,
    twist_modes,
)

MASS_RATIO = 1.2
LAMBDA_1 = 300.0
LAMBDA_2 = 300.0
CHI_1 = (0.05, 0.0, 0.1)
CHI_2 = (0.0, -0.05, 0.05)
TOTAL_MASS = 2.8
DISTANCE_MPC = 100.0

INCLINATION = 0.5
AZIMUTH = np.pi / 2.0  # coalescence_angle = 0, per compute_hpc's convention

# TEOBResumS.h constants, so the physical amplitude prefactor matches
# exactly what TEOBResumS itself applies (TEOBResumS.c, domain=FD,
# physical units): nu * M^2 * MSUN_M * MSUN_S / (distance * MPC_M).
# compute_hpc() reproduces the *unscaled* (geometric-unit) rotation only.
MSUN_M = 1.476625061404649406193430731479084713e3
MPC_M = 3.085677581491367278913937957796471611e22

F0_HZ = 15.0
BAND_LO, BAND_HI = 20.0, 1024.0
DF = 1.0 / 512.0
SRATE_HZ = 4096.0

N_FIT_POINTS = 250

K22 = mode_to_k(Mode(2, 2))
K21 = mode_to_k(Mode(2, 1))


def teob_fd_run(use_spins, **spins):
    from EOBRun_module import EOBRunPy

    par = dict(
        q=MASS_RATIO,
        LambdaAl2=LAMBDA_1,
        LambdaBl2=LAMBDA_2,
        M=TOTAL_MASS,
        distance=DISTANCE_MPC,
        initial_frequency=F0_HZ,
        srate_interp=SRATE_HZ,
        use_geometric_units="no",
        interp_uniform_grid="yes",
        domain=1,
        df=DF,
        inclination=INCLINATION,
        coalescence_angle=0.0,
        output_hpc="no",
        arg_out="yes",
        use_spins=use_spins,
        **spins,
    )
    return EOBRunPy(par)


def get_coprecessing_and_ground_truth():
    """TEOB's own (2,2) co-precessing FD mode, and its (2,2)-only twist."""
    f_co, *_rest, hflm, _hfTlm, _dyn = teob_fd_run(
        1, chi1=CHI_1[2], chi2=CHI_2[2], use_mode_lm=[K22]
    )
    f_co = np.asarray(f_co)
    coprecessing_22 = amp_phase_to_complex(*hflm[str(K22)])

    f_gt, hpr, hpi, hcr, hci, *_ = teob_fd_run(
        2,
        chi1x=CHI_1[0], chi1y=CHI_1[1], chi1z=CHI_1[2],
        chi2x=CHI_2[0], chi2y=CHI_2[1], chi2z=CHI_2[2],
        use_mode_lm=[K22], use_mode_lm_inertial=[K21, K22],
    )
    f_gt = np.asarray(f_gt)
    # TEOBResumS' raw h_+, h_x use the opposite Fourier-sign convention
    # from mlgw_bns' h~(f) = int h(t) exp(2*pi*i*f*t) dt (see
    # validate_precessing_against_teob.teob_polarizations, which
    # conjugates for the same reason); the amp/phase-pair outputs
    # (hflm, via amp_phase_to_complex) do not need this.
    hp_gt = np.conj(np.asarray(hpr) + 1j * np.asarray(hpi))
    # Empirically, h_x additionally needs an overall sign flip beyond the
    # shared time/phase calibration to match polarizations_from_inertial_
    # modes' convention here: phase(h_x model / h_x TEOB) - phase(h_+
    # model / h_+ TEOB) = -pi to ~1e-4 rad at every frequency tested, not
    # explained further (possibly specific to the use_mode_lm_inertial
    # restriction to a single co-precessing mode used in this script) --
    # flagged rather than silently trusted.
    hc_gt = -np.conj(np.asarray(hcr) + 1j * np.asarray(hci))

    # both runs share the same interp_uniform_grid/df, so the frequency
    # axes should already match; assert rather than silently resample
    np.testing.assert_allclose(f_co, f_gt, rtol=0, atol=1e-9)
    return f_co, coprecessing_22, hp_gt, hc_gt


def model_hpc(coprecessing_value, alpha, beta, gamma):
    """Our twist of a single (2,2) co-precessing FD sample.

    Mirrors :func:`~mlgw_bns.precessing_model.twist_modes_frequency_domain`
    (a single co-precessing mode's contribution to ``modes_positive_f``
    and ``modes_negative_f``) plus
    :func:`~mlgw_bns.precessing_model.polarizations_from_inertial_modes`,
    rather than reusing the *time-domain* ``compute_hpc``: in the
    frequency domain, h_+(f) and h_x(f) each need both the +f and -f
    content of every inertial mode to reconstruct (a single complex
    h = h_+ - i*h_x at one frequency, as ``compute_hpc`` builds for a
    time sample, is not separable into h_+(f), h_x(f) on its own).

    ``polarizations_from_inertial_modes`` also reproduces TEOBResumS'
    geometric-unit (unscaled) rotation only; TEOBResumS' own domain=1,
    physical-units output additionally multiplies by ``nu * M^2 * MSUN_M
    * MSUN_S / (distance * MPC_M)`` (``TEOBResumS.c``, the ``DOMAIN_FD``
    branch) after twisting, applied here so the two are on the same
    scale.
    """
    positive_f: dict = {}
    negative_f: dict = {}
    for target, include_positive_n in ((positive_f, True), (negative_f, False)):
        twisted, twisted_negative_m, twisted_m0 = twist_modes(
            {(2, 2): np.array([coprecessing_value])},
            np.array([alpha]), np.array([beta]), np.array([gamma]),
            lm_inertial=[(2, 1), (2, 2)],
            include_positive_n=include_positive_n,
            include_negative_n=not include_positive_n,
        )
        for contribution in (twisted, twisted_negative_m, twisted_m0):
            target.update(contribution)

    hp, hc = polarizations_from_inertial_modes(
        positive_f, negative_f, INCLINATION, AZIMUTH
    )
    nu = MASS_RATIO / (1.0 + MASS_RATIO) ** 2
    prefactor = (nu * TOTAL_MASS ** 2 * MSUN_M * SUN_MASS_SECONDS
                 / (DISTANCE_MPC * MPC_M))
    return prefactor * hp[0], prefactor * hc[0]


def calibrate_time_phase(frequencies, coprecessing, hp_gt, hc_gt, angles: EulerAngles):
    r"""Fit a global time/phase offset between the two independent
    EOBRunPy calls (co-precessing + its own twist) using the raw,
    un-anchored angle lookup as the baseline model.

    This offset is a benign artefact of the two runs not sharing an
    exact internal time/phase reference (residual ~0.3-0.5s implied
    time-shift observed, comparable to the extra head-start FD runs take
    per ``initial_frequency_evolve`` in ``TEOBResumSPars.c``) -- not a
    physics discrepancy, and not something any Euler-angle choice can
    absorb (it is common to h_+ and h_x, whereas the angles only rotate
    between multipoles). It must be removed before per-frequency angle
    recovery means anything, exactly as ``full_waveform_mismatch``
    marginalises over an analogous time/phase shift when comparing whole
    waveforms.
    """
    mass_sum_seconds = TOTAL_MASS * SUN_MASS_SECONDS
    momega = np.pi * frequencies * mass_sum_seconds
    alpha, beta, gamma = angles.at_momega(momega)
    model = np.array([
        model_hpc(coprecessing[i], alpha[i], beta[i], gamma[i])[0]
        for i in range(frequencies.size)
    ])
    ratio = model / hp_gt
    phase = np.unwrap(np.angle(ratio))
    slope, intercept = np.polyfit(frequencies, phase, 1)
    dt = -slope / (2.0 * np.pi)
    print(f"  calibrated time offset: {dt * 1e3:+.3f} ms, "
          f"phase offset: {-intercept:+.3f} rad "
          f"(residual after removing: max {np.max(np.abs(phase - np.polyval((slope, intercept), frequencies))):.3e} rad)")
    return dt, -intercept


def recover_momega(frequencies, coprecessing, hp_gt, hc_gt, angles: EulerAngles,
                    dt: float, phase0: float):
    momega_lo, momega_hi = angles.momega[0], angles.momega[-1]
    recovered = np.full(frequencies.size, np.nan)
    residual_norm = np.full(frequencies.size, np.nan)

    current = np.clip(np.pi * frequencies[0] * TOTAL_MASS * SUN_MASS_SECONDS,
                       momega_lo, momega_hi)
    for i, f in enumerate(frequencies):
        calibration = np.exp(1j * (2.0 * np.pi * f * dt + phase0))
        target = np.array([hp_gt[i].real, hp_gt[i].imag,
                            hc_gt[i].real, hc_gt[i].imag])
        scale = max(np.max(np.abs(target)), 1e-300)

        def residual(m):
            alpha, beta, gamma = angles.at_momega(np.clip(m, momega_lo, momega_hi))
            hp, hc = model_hpc(coprecessing[i], alpha[0], beta[0], gamma[0])
            hp, hc = hp * calibration, hc * calibration
            model = np.array([hp.real, hp.imag, hc.real, hc.imag])
            return (model - target) / scale

        solution = least_squares(residual, [current], xtol=1e-14, ftol=1e-14, gtol=1e-14)
        recovered[i] = current = float(solution.x[0])
        residual_norm[i] = np.max(np.abs(solution.fun))

    return recovered, residual_norm


def main() -> None:
    print("Running TEOBResumS (co-precessing (2,2) + its own twist, FD)...")
    frequencies, coprecessing, hp_gt, hc_gt = get_coprecessing_and_ground_truth()

    eta = MASS_RATIO / (1.0 + MASS_RATIO) ** 2
    mass_sum_seconds = TOTAL_MASS * SUN_MASS_SECONDS
    f0_geom = F0_HZ * mass_sum_seconds
    solution = integrate_pn_spin_precession(
        eta, CHI_1, CHI_2, f0_geom,
        # default t_max=1e6 is far shorter than this system's ~2.5e7 M
        # inspiral from f0=15 Hz and silently truncates the integration
        # almost immediately (mirrors precessing_model.euler_angles()'s
        # own t_max choice).
        t_max=2.0 * newtonian_time_to_merger(eta, np.pi * f0_geom),
    )
    angles = EulerAngles(
        momega=solution["Momega"], alpha=solution["alpha"],
        beta=solution["beta"], gamma=solution["gamma"], time=solution["t"],
    )
    print(f"max beta along the PN track: {np.max(angles.beta):.4e} rad")

    band = (frequencies >= BAND_LO) & (frequencies <= BAND_HI)
    idx = np.where(band)[0]
    stride = max(1, idx.size // N_FIT_POINTS)
    idx = idx[::stride]
    fb = frequencies[idx]

    print("Calibrating the residual time/phase offset between the two runs...")
    dt, phase0 = calibrate_time_phase(fb, coprecessing[idx], hp_gt[idx], hc_gt[idx], angles)

    print(f"Recovering M*Omega_orb at {fb.size} frequencies "
          f"({BAND_LO}-{BAND_HI} Hz)...")
    recovered, residual_norm = recover_momega(
        fb, coprecessing[idx], hp_gt[idx], hc_gt[idx], angles, dt, phase0
    )
    print(f"fit residual (normalised): max {np.nanmax(residual_norm):.3e}, "
          f"median {np.nanmedian(residual_norm):.3e}")

    # The lookup formula M*Omega_orb = 2*pi*f/n is unchanged by
    # reanchoring -- reanchoring re-labels the alpha/beta/gamma-vs-Momega
    # *curve*, not the query. So "does reanchoring supply the momega the
    # fit needs" has to be read off through the *angle values* the two
    # curves give at that same formula value, not by comparing momega
    # numbers (which would be identical by construction).
    raw_momega = np.pi * fb * mass_sum_seconds
    # TEOB's native FD grid is uniform and fine (df = 1/512 Hz), so its
    # phase can be unwrapped here, but it runs to its Nyquist (~2000+ Hz
    # beyond real (2,2) signal, all numerical noise), and unwrapping
    # across that corrupts the reconstructed SPA time: restrict to a
    # comparable band to the surrogate's.
    anchor_band = (frequencies > 0) & (frequencies <= 2.0 * BAND_HI)
    anchored = angles.reanchored(
        frequencies[anchor_band], np.unwrap(np.angle(coprecessing[anchor_band])),
        mass_sum_seconds, BAND_LO,
    )

    _, raw_beta, _ = angles.at_momega(raw_momega)
    _, anchored_beta, _ = anchored.at_momega(raw_momega)
    _, recovered_beta, _ = angles.at_momega(recovered)

    print()
    print(f"{'f [Hz]':>10}  {'beta: raw':>11}  {'reanchored':>11}  "
          f"{'recovered':>11}  {'rec-raw':>10}  {'rec-reanchored':>15}")
    for i in range(0, fb.size, max(1, fb.size // 25)):
        print(f"{fb[i]:10.2f}  {raw_beta[i]:11.4e}  {anchored_beta[i]:11.4e}  "
              f"{recovered_beta[i]:11.4e}  {recovered_beta[i]-raw_beta[i]:+10.3e}  "
              f"{recovered_beta[i]-anchored_beta[i]:+15.3e}")

    print()
    print("summary over the whole band (|recovered - X|, in beta [rad]):")
    print(f"  vs raw (un-anchored) lookup:   "
          f"median {np.median(np.abs(recovered_beta - raw_beta)):.3e}, "
          f"max {np.max(np.abs(recovered_beta - raw_beta)):.3e}")
    print(f"  vs reanchored lookup:          "
          f"median {np.median(np.abs(recovered_beta - anchored_beta)):.3e}, "
          f"max {np.max(np.abs(recovered_beta - anchored_beta)):.3e}")

    np.savez(
        "visualization/recover_frequency_remapping.npz",
        frequencies=fb, recovered_momega=recovered, raw_momega=raw_momega,
        raw_beta=raw_beta, anchored_beta=anchored_beta,
        recovered_beta=recovered_beta, residual_norm=residual_norm,
    )
    print("\nsaved to visualization/recover_frequency_remapping.npz")


if __name__ == "__main__":
    main()
