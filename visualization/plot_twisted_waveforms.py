"""Exercise the twisting machinery in :mod:`mlgw_bns.twist_waveform` and
plot what it produces.

The script does two things.

First, it runs three analytic sanity checks on the twist and on the PN
spin-precession dynamics:

1. with :math:`\\beta = 0` the twist must reduce to a pure rotation about
   :math:`\\hat{L}`, leaving every amplitude untouched and shifting each
   phase by :math:`m (\\alpha - \\gamma)`;
2. for a generic precession the Wigner-D matrices are unitary, so the
   power :math:`\\sum_m |h_{\\ell m}|^2` summed over a given
   :math:`\\ell` must be the same before and after the twist, at every
   time;
3. a non-spinning binary must not precess at all: :math:`\\hat{L}` stays
   at :math:`(0, 0, 1)`, :math:`\\beta` vanishes and :math:`\\gamma` is
   constant, while the orbital frequency increases monotonically.

Second, it integrates the PN dynamics for a generic precessing
configuration and plots the resulting Euler angles together with the
polarizations obtained by twisting a leading-order quadrupole waveform.
The co-precessing multipole used for the figure is a toy Newtonian one,
not TEOBResumS: the point is to show the effect of the twist, not to
produce an accurate waveform. For the latter, see
:mod:`mlgw_bns.precessing_model`, which twists the surrogate's
co-precessing multipoles in the frequency domain.

Run with: python visualization/plot_twisted_waveforms.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid

from mlgw_bns.twist_waveform import (
    amp_phase_to_complex,
    complex_to_amp_phase,
    compute_hpc,
    integrate_pn_spin_precession,
    twist_modes,
)

NU = 0.2222222222222222  # symmetric mass ratio for q = 2
F0 = 0.002  # initial M * f
CHI_1 = (0.5, 0.3, 0.2)
CHI_2 = (-0.2, 0.1, 0.4)
INCLINATION = 0.5
AZIMUTH = 0.0

PLOT_DURATION = 25_000.0  # in units of M
N_PLOT_SAMPLES = 20_000

FIGURE_PATH = "visualization/twisted_waveform.png"


def toy_coprecessing_modes(t):
    """A pair of synthetic co-precessing multipoles, for the sanity checks."""

    amp_22 = 1.0 - 0.2 * np.exp(-t / 50.0)
    phase_22 = 0.05 * t + 0.01 * t**2
    amp_21 = 0.1 * amp_22
    phase_21 = phase_22  # arbitrary toy relation

    modes = {
        (2, 2): amp_phase_to_complex(amp_22, phase_22),
        (2, 1): amp_phase_to_complex(amp_21, phase_21),
    }
    amplitudes = {(2, 2): amp_22, (2, 1): amp_21}
    phases = {(2, 2): phase_22, (2, 1): phase_21}
    return modes, amplitudes, phases


def check_zero_opening_angle_is_a_z_rotation(t, modes, amplitudes, phases):
    """With ``beta = 0`` the twist is a rotation about the orbital angular momentum."""

    alpha = 0.3 * np.sin(0.02 * t)
    gamma = 0.2 * np.cos(0.03 * t)
    beta = np.zeros_like(t)

    twisted, _, _ = twist_modes(
        modes, alpha, beta, gamma, lm_inertial=list(modes.keys())
    )

    for (ell, m), twisted_mode in twisted.items():
        amplitude, phase = complex_to_amp_phase(twisted_mode)
        expected_phase = phases[(ell, m)] + m * (alpha - gamma)
        assert np.allclose(
            amplitude, amplitudes[(ell, m)], atol=1e-10
        ), f"amplitude mismatch for mode {(ell, m)}"
        assert np.allclose(
            np.angle(np.exp(1j * (phase - expected_phase))), 0.0, atol=1e-8
        ), f"phase mismatch for mode {(ell, m)}"

    print("beta = 0 sanity check passed.")


def check_power_is_conserved(t, modes, amplitudes):
    """Wigner-D rotations are unitary, so they preserve the power in each :math:`\\ell`."""

    alpha = 0.3 * np.sin(0.02 * t)
    gamma = 0.2 * np.cos(0.03 * t)
    beta = 0.4 + 0.1 * np.sin(0.05 * t)

    positive_m, negative_m, zero_m = twist_modes(
        modes, alpha, beta, gamma, lm_inertial=list(modes.keys())
    )

    # each co-precessing mode has a partner at -m of equal magnitude
    power_in = 2.0 * sum(amplitude**2 for amplitude in amplitudes.values())
    power_out = sum(
        np.abs(mode) ** 2
        for twisted in (positive_m, negative_m, zero_m)
        for mode in twisted.values()
    )

    assert np.allclose(power_in, power_out, rtol=1e-8), "power not conserved by the twist"
    print("power-conservation sanity check passed.")


def check_non_spinning_binary_does_not_precess():
    """Without spins there is no torque on :math:`\\hat{L}`, so the angles are frozen."""

    solution = integrate_pn_spin_precession(NU, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), F0)

    assert np.allclose(
        solution["beta"], 0.0, atol=1e-12
    ), "beta != 0 for a non-spinning binary"
    assert np.allclose(
        solution["Lh"], np.array([0.0, 0.0, 1.0]), atol=1e-12
    ), "Lhat drifted for a non-spinning binary"
    assert np.allclose(
        solution["gamma"], solution["gamma"][0], atol=1e-12
    ), "gamma is not constant for a non-spinning binary"
    assert np.all(
        np.diff(solution["Momega"]) > 0
    ), "Momega is not monotonically increasing"

    print(
        f"non-spinning PN sanity check passed (integrated to "
        f"M Omega = {solution['Momega'][-1]:.4f} at t = {solution['t'][-1]:.1f} M)."
    )


def resample(solution, times):
    """Interpolate a PN solution onto a denser time grid.

    The ODE solver only outputs as many points as it needs to represent
    the *angles* accurately; those are far too few to resolve the
    waveform's own oscillation, which is faster by the ratio between the
    precession and orbital timescales.
    """

    resampled = {"t": times, "Momega": np.interp(times, solution["t"], solution["Momega"])}
    for angle in ("alpha", "beta", "gamma"):
        unwrapped = np.unwrap(solution[angle])
        resampled[angle] = np.interp(times, solution["t"], unwrapped)
    return resampled


def twisted_toy_waveform(solution):
    """Twist a leading-order quadrupole along the PN precession solution."""

    amplitude = 4.0 * NU * (2.0 * solution["Momega"]) ** (2.0 / 3.0)
    phase = 2.0 * cumulative_trapezoid(solution["Momega"], solution["t"], initial=0.0)
    modes = {(2, 2): amp_phase_to_complex(amplitude, phase)}

    positive_m, negative_m, zero_m = twist_modes(
        modes,
        solution["alpha"],
        solution["beta"],
        solution["gamma"],
        lm_inertial=[(2, 2), (2, 1)],
    )
    return compute_hpc(positive_m, negative_m, zero_m, phi=AZIMUTH, iota=INCLINATION)


def plot(solution, dense, strain):
    time = solution["t"] / 1e3

    fig, axes = plt.subplots(
        3, 1, figsize=(9, 9), gridspec_kw={"height_ratios": [1, 1, 1.4]}
    )

    axes[0].plot(time, np.unwrap(solution["alpha"]), label=r"$\alpha$")
    axes[0].plot(time, np.unwrap(solution["gamma"]), label=r"$\gamma$")
    axes[0].set_ylabel("angle [rad]")
    axes[0].legend()

    axes[1].plot(time, solution["beta"], c="black")
    axes[1].set_ylabel(r"$\beta$ [rad]")
    axes[1].set_xlabel(r"$t / (10^3 M)$")

    for axis in axes[:2]:
        axis.set_xlim(time[0], time[-1])
        axis.axvspan(
            dense["t"][0] / 1e3, dense["t"][-1] / 1e3, color="gray", alpha=0.15
        )

    dense_time = dense["t"] / 1e3
    axes[2].plot(dense_time, strain.real, lw=0.7, label=r"$h_+$")
    axes[2].plot(dense_time, -strain.imag, lw=0.7, label=r"$h_\times$")
    axes[2].set_xlim(dense_time[0], dense_time[-1])
    axes[2].set_title("early inspiral (shaded above)", fontsize="medium")
    axes[2].set_ylabel("strain [arbitrary units]")
    axes[2].set_xlabel(r"$t / (10^3 M)$")
    axes[2].legend()

    fig.suptitle(
        rf"PN precession and twisted quadrupole, "
        rf"$q = 2$, $\iota = {INCLINATION}$"
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"figure written to {FIGURE_PATH}")


if __name__ == "__main__":
    t = np.linspace(0.0, 100.0, 2000)
    modes, amplitudes, phases = toy_coprecessing_modes(t)

    check_zero_opening_angle_is_a_z_rotation(t, modes, amplitudes, phases)
    check_power_is_conserved(t, modes, amplitudes)
    check_non_spinning_binary_does_not_precess()

    solution = integrate_pn_spin_precession(NU, CHI_1, CHI_2, F0)
    print(
        f"precessing PN dynamics: {solution['t'].size} steps to "
        f"M Omega = {solution['Momega'][-1]:.4f} at t = {solution['t'][-1]:.1f} M, "
        f"beta in [{solution['beta'].min():.3f}, {solution['beta'].max():.3f}] rad."
    )

    # the waveform panel only covers the early inspiral, where the
    # orbital period is long enough for the individual cycles --- and
    # the modulation the precession imprints on them --- to be visible
    dense = resample(
        solution, np.linspace(solution["t"][0], PLOT_DURATION, N_PLOT_SAMPLES)
    )
    plot(solution, dense, twisted_toy_waveform(dense))
