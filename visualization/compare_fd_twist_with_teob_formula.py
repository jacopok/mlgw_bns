"""Our frequency-domain twist vs a verbatim port of TEOBResumS' twist_hlm_FD.

Both twists are fed identical inputs: TEOBResumS' own FD co-precessing
multipoles (``arg_out``) and one set of PN Euler angles tabulated against
the PN orbital frequency, as TEOBResumS' FD path does (``spin_flx = PN``).
Any difference is in the twist formula or its conventions, not in the
physics being modelled. The port also gets compared against TEOBResumS'
own FD h+, hx, which tells how closely our PN angles reproduce its own.

Run with: python visualization/compare_fd_twist_with_teob_formula.py [factor]
where ``factor`` (default 0.95) sets the spin reference frequency as a
fraction of TEOBResumS' initial_frequency.
"""
import numpy as np
from EOBRun_module import EOBRunPy

from mlgw_bns.precessing_model import (
    EulerAngles,
    newtonian_time_to_merger,
    polarizations_from_inertial_modes,
    twist_modes_frequency_domain,
)
from mlgw_bns.special_func import spinsphericalharm, wigner_d_function
from mlgw_bns.taylorf2 import SUN_MASS_SECONDS
from mlgw_bns.twist_waveform import integrate_pn_spin_precession, nu_to_X1

K_TO_MODE = {0: (2, 1), 1: (2, 2), 4: (3, 3), 8: (4, 4)}
M_TOTAL, F0, DF = 2.8, 15.0, 1.0 / 512.0
SPIN_REFERENCE_FACTOR = 0.95


def ylm(ell, m, phi, iota):
    real, imag = spinsphericalharm(-2, ell, m, phi, iota)
    return complex(real, imag)


def teob_twist_fd(f_geom, modes_amp_phase, angles, phi, iota):
    """twist_hlm_FD (TEOBResumSWaveform.c), vectorised over frequency.

    ``modes_amp_phase``: {(l, m): (A, P)} in TEOBResumS' FD convention.
    Returns (hp, hc) up to the overall amplitude prefactor.
    """
    hp = np.zeros(f_geom.size, complex)
    hc = np.zeros(f_geom.size, complex)
    for (ell, emm), (amp, phase) in modes_amp_phase.items():
        omg = 2.0 * np.pi * f_geom / emm
        alpha, beta, gamma = angles.at_momega(omg)
        eps = (-1.0) ** ell
        s_p = np.zeros(f_geom.size, complex)
        s_c = np.zeros(f_geom.size, complex)
        for n in range(-ell, ell + 1):
            a_ln = ylm(ell, n, phi, iota) * np.exp(-1j * n * alpha)
            d_mn = wigner_d_function(ell, emm, n, -beta)
            d_mnn = wigner_d_function(ell, -emm, n, -beta)
            s_p += a_ln.real * (d_mn + eps * d_mnn) + 1j * a_ln.imag * (d_mn - eps * d_mnn)
            s_c += a_ln.real * (d_mn - eps * d_mnn) + 1j * a_ln.imag * (d_mn + eps * d_mnn)
        t = np.exp(1j * (phase + emm * gamma))
        hp += amp * s_p * t
        hc += 1j * amp * s_c * t
    # modes are zero below their own start (f < f0lm) in the C code
    return 0.5 * hp, 0.5 * hc


def overlap(a, b, frequencies=None):
    """Normalised overlap, maximised over phase and (given ``frequencies``) time.

    Flat PSD: this compares waveform shapes, not detection statistics.
    """
    norm = np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
    if frequencies is None:
        return abs(np.vdot(a, b)) / norm
    integrand = np.conj(a) * b

    def match(t):
        return abs(np.sum(integrand * np.exp(2j * np.pi * frequencies * t))) / norm

    # coarse FFT search, then a local refinement
    df = frequencies[1] - frequencies[0]
    n = 1 << 22
    coarse = np.abs(np.fft.fft(integrand, n))
    k = int(np.argmax(coarse))
    t0 = -(k if k < n // 2 else k - n) / (n * df)
    dt = 1.0 / (n * df)
    ts = np.linspace(t0 - 2 * dt, t0 + 2 * dt, 81)
    t1 = ts[np.argmax([match(t) for t in ts])]
    ts = np.linspace(t1 - dt / 20, t1 + dt / 20, 81)
    return max(match(t) for t in ts)


def run(chi1, chi2, q=1.3, iota=0.9, azimuth=0.4, fmin=20.0, fmax=1024.0,
        lambdas=(400.0, 600.0), antenna=(0.6, 0.8)):
    par = dict(q=q, LambdaAl2=lambdas[0], LambdaBl2=lambdas[1], M=M_TOTAL, distance=100.,
               initial_frequency=F0, srate_interp=8192., use_geometric_units="no",
               interp_uniform_grid="yes", domain=1, df=DF, inclination=iota,
               coalescence_angle=np.pi / 2 - azimuth, output_hpc="no", arg_out="yes",
               use_spins=2, chi1x=chi1[0], chi1y=chi1[1], chi1z=chi1[2],
               chi2x=chi2[0], chi2y=chi2[1], chi2z=chi2[2],
               use_mode_lm=sorted(K_TO_MODE))
    f, rp, ip, rc, ic, hflm, _, _ = EOBRunPy(par)
    f = np.asarray(f)
    band = (f >= fmin) & (f <= fmax)
    fb = f[band]
    hp_teob = (np.asarray(rp) + 1j * np.asarray(ip))[band]
    hc_teob = (np.asarray(rc) + 1j * np.asarray(ic))[band]
    modes = {K_TO_MODE[int(k)]: (np.asarray(v[0])[band], np.asarray(v[1])[band])
             for k, v in hflm.items()}

    msun = M_TOTAL * SUN_MASS_SECONDS
    nu = q / (1 + q) ** 2
    # TEOBResumS' FD path imposes the spins at 0.95 * initial_frequency
    # (see validate_precessing_against_teob.SPIN_REFERENCE_FREQUENCY_HZ)
    sol = integrate_pn_spin_precession(
        nu, chi1, chi2, SPIN_REFERENCE_FACTOR * F0 * msun,
        t_max=2.0 * newtonian_time_to_merger(nu, np.pi * F0 * msun),
        lambdas=lambdas)
    angles = EulerAngles(momega=sol["Momega"], alpha=sol["alpha"], beta=sol["beta"],
                         gamma=sol["gamma"], time=sol["t"])

    hp_port, hc_port = teob_twist_fd(fb * msun, modes, angles, np.pi / 2 - azimuth, iota)

    # ours, in the mlgw_bns convention: the conjugate of TEOBResumS' FD
    coprec = {key: amp * np.exp(-1j * phase) for key, (amp, phase) in modes.items()}
    pos, neg = twist_modes_frequency_domain(coprec, fb, angles, msun)
    hp_ours, hc_ours = polarizations_from_inertial_modes(pos, neg, iota, azimuth)
    hp_ours, hc_ours = np.conj(hp_ours), np.conj(hc_ours)

    return {
        "max beta": sol["beta"].max(),
        "port vs TEOB h+": 1 - overlap(hp_port, hp_teob, fb),
        "port vs TEOB hx": 1 - overlap(hc_port, hc_teob, fb),
        "ours vs port h+": 1 - overlap(hp_ours, hp_port),
        "ours vs port hx": 1 - overlap(hc_ours, hc_port),
        "ours vs TEOB h+": 1 - overlap(hp_ours, hp_teob, fb),
        "ours vs TEOB hx": 1 - overlap(hc_ours, hc_teob, fb),
        # a detector strain: h+ and hx with their relative phase fixed
        "ours vs TEOB strain": 1 - overlap(antenna[0] * hp_ours + antenna[1] * hc_ours,
                                           antenna[0] * hp_teob + antenna[1] * hc_teob, fb),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        SPIN_REFERENCE_FACTOR = float(sys.argv[1])
    for label, chi1, chi2 in (
        ("aligned", (0, 0, 0.05), (0, 0, 0.0)),
        ("chi_perp 0.1", (0.1, 0.0, 0.05), (0.0, 0.05, 0.0)),
        ("chi_perp 0.3", (0.25, 0.15, 0.05), (-0.1, 0.2, 0.0)),
        ("chi_perp 0.5", (0.4, 0.3, 0.05), (-0.2, 0.3, 0.0)),
    ):
        result = run(chi1, chi2)
        print(f"{label:>13}: " + "  ".join(f"{k} {v:.2e}" for k, v in result.items()),
              flush=True)
