r"""Can the surrogate predict TEOBResumS' orbital-phase origin?

:mod:`precession_floor_anatomy` finds that TEOBResumS' co-precessing
multipoles are the surrogate's up to a time shift and one orbital-phase
rotation, :math:`h^{\rm TEOB}_{\ell m} = h^{\rm sur}_{\ell m}
e^{i (2 \pi f T + m \varphi_0)}`, and that this rotation is most of the
precessing mismatch. TEOBResumS puts the orbital phase to zero where it
starts integrating (and imposes the spins there); the surrogate at the
merger. This script predicts :math:`\varphi_0` from that alone.

The stationary-phase transform :math:`X(f) = \Psi(f) - f \Psi'(f)` of a
multipole's frequency-domain phase does not change under time shifts, and
equals :math:`\phi^{\rm TD}_{\ell m}(t_f) + \pi / 4` at its stationary
time (checked against TEOBResumS' own time-domain output at its ODE
samples, to 1e-2 rad below 50 Hz; the rest is using :math:`m \Omega`
for the multipole's frequency). At TEOBResumS' first instant the orbital
phase is zero and :math:`\phi^{\rm TD}_{\ell m}(0) = \delta_{\ell m}`, the
multipole's phase offset (:math:`-\pi + 0.011` for the (2,2)). So

.. math::
    m \varphi_0 = \delta_{\ell m} + \pi / 4
        - X^{\rm sur}_{\ell m}(m F_{\rm orb}(0)) \pmod{2 \pi},

from the surrogate's own phase at TEOBResumS' starting orbital frequency
:math:`F_{\rm orb}(0)`. For each binary it compares this with the
:math:`\varphi_0` fitted on the multipoles, and with the nominal
starting frequency ``0.95 * initial_frequency`` in place of the actual
one. It then rotates
the surrogate by the predicted :math:`\varphi_0` and recomputes the
precessing mismatch (first line of sight of each binary), also without
the frequency nodes where TEOBResumS' own twist reads its Euler angle
:math:`\alpha` across one of the spurious 2 pi steps of its spin dynamics
(see :func:`precession_angle_residual.alpha_jumps`).

Run with: python visualization/precession_orbital_phase_origin.py
    [--n-binaries N] [--processes P]
"""

from __future__ import annotations

import argparse
import multiprocessing
import tempfile
from pathlib import Path

import numpy as np

import precession_error_vs_inplane_spin as scaling
import precession_floor_anatomy as anatomy
from precession_angle_residual import alpha_glitched, alpha_jumps, teob_dynspin
import validate_precessing_against_teob as v
from mlgw_bns import precessing_model
from mlgw_bns.precessing_model import PrecessingParametersWithExtrinsic

DATA_PATH = Path(__file__).with_name("precession_orbital_phase_origin.npz")
#: Spline intervals of TEOBResumS' spin dynamics masked on either side of each
#: of its alpha jumps: the cubic spline's error decays by ~2 + sqrt(3) per
#: interval away from a 2 pi step, so one interval leaves ~1e-5 in some
#: mismatches, six leave nothing measurable.
JUMP_MARGIN = 6


def wrap(angle):
    """Angle to (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(angle)))


def stationary_phase_transform(phase_function, frequency):
    r""":math:`X = \Psi - f \Psi'` at ``frequency``; see
    :func:`mlgw_bns.precessing_model.stationary_phase_transform`."""
    window = precessing_model.stationary_phase_window(frequency)
    return precessing_model.stationary_phase_transform(
        window, phase_function(window), frequency)


def evaluate_binary(case):
    model, precessing, validator = anatomy._setup()
    index, intrinsic, chi_1, chi_2, orientation = case
    params = PrecessingParametersWithExtrinsic(
        mass_ratio=intrinsic.mass_ratio, lambda_1=intrinsic.lambda_1,
        lambda_2=intrinsic.lambda_2, chi_1=chi_1, chi_2=chi_2,
        distance_mpc=v.DISTANCE_MPC, inclination=orientation["inclination"],
        azimuth=orientation["azimuth"], total_mass=v.TOTAL_MASS,
        reference_frequency_hz=v.SPIN_REFERENCE_FREQUENCY_HZ,
    )
    try:
        with tempfile.TemporaryDirectory() as directory:
            f_native, hp, hc, raw, dynamics = v.teob_run(
                intrinsic, chi_1, chi_2, params.inclination, params.azimuth,
                multipoles=True,
                overrides=dict(output_dynamics="yes", output_dir=directory),
            )
            dynspin = teob_dynspin(directory)
    except RuntimeError as error:
        print(f"binary {index}: TEOBResumS failed ({error})", flush=True)
        return None
    nodes, fb, surrogate, _, teob_raw, fit = anatomy.aligned_fit(
        model, validator, params, f_native, raw)
    mass_seconds = params.aligned().mass_sum_seconds
    orbital_frequency = dynamics["MOmega"][0] / (2.0 * np.pi * mass_seconds)
    nominal_orbital_frequency = v.SPIN_REFERENCE_FREQUENCY_HZ / 2.0

    def phase(source, key):
        def function(frequencies):
            return model.coprecessing_amplitudes_and_phases(
                frequencies, params.aligned(), source=source)[key][1]
        return function

    row = dict(binary=index, phi0=fit["phi0"], aligned_mismatch=fit["aligned_mismatch"],
               start_frequency_22=2.0 * orbital_frequency,
               inclination=params.inclination)
    for (l, m), (_, td_phase) in dynamics["multipoles"].items():
        delta = td_phase[0] - m * dynamics["phi"][0]
        row[f"delta_{l}{m}"] = delta
        for label, source, frequency in (
            ("surrogate", "surrogate", orbital_frequency),
            ("nominal", "surrogate", nominal_orbital_frequency),
        ):
            transform = stationary_phase_transform(phase(source, (l, m)), m * frequency)
            predicted = delta + np.pi / 4.0 - transform  # = m phi0, mod 2 pi
            row[f"error_{label}_{l}{m}"] = float(wrap(predicted - m * fit["phi0"]))
            row[f"predicted_{label}_{l}{m}"] = float(wrap(predicted))

    # phi0 from the (2,2), mod pi; the branch from the (2,1), whose m = 1
    # has no ambiguity.
    half = row["predicted_surrogate_22"] / 2.0
    candidates = np.array([half, half + np.pi])
    predicted_phi0 = candidates[np.argmin(np.abs(wrap(
        candidates - row["predicted_surrogate_21"])))]
    row["predicted_phi0"] = float(predicted_phi0)

    antenna = v.antenna_patterns(orientation["theta"], orientation["phi"], orientation["psi"])
    target = (antenna[0] * hp + antenna[1] * hc)[nodes]
    angles = precessing.euler_angles(params, float(validator.frequencies[0]))

    # The nodes where TEOBResumS' own twist read alpha across one of its
    # spurious 2 pi steps, for any multipole (its stationary-phase orbital
    # frequency is f / m), with JUMP_MARGIN spline intervals either side.
    glitched = alpha_glitched(
        dynspin, fb, sorted({m for (_, m) in raw}), mass_seconds, JUMP_MARGIN)
    clean = ~glitched
    row["alpha_jumps"] = alpha_jumps(dynspin).size
    row["glitched_fraction"] = glitched.mean()

    def mismatch(modes, keep=slice(None)):
        return validator.full_waveform_mismatch(
            {(2, 2): target[keep]},
            {(2, 2): anatomy.strain({k: a[keep] for k, a in modes.items()}, fb[keep],
                                    angles, params, antenna)},
            frequencies=fb[keep],
        )

    variants = {
        "surrogate": surrogate,
        "fitted_phi0": anatomy.rotated(surrogate, fb, 0.0, fit["phi0"]),
        "predicted_phi0": anatomy.rotated(surrogate, fb, 0.0, predicted_phi0),
        "teob_raw": teob_raw,
    }
    for name, modes in variants.items():
        row[f"mismatch_{name}"] = mismatch(modes)
        row[f"mismatch_{name}_clean"] = mismatch(modes, clean)
    print(
        f"binary {index}: phi0 fitted {fit['phi0']:+.3f} predicted {predicted_phi0:+.3f}  "
        + "  ".join(f"{k[6:]} {row[k]:+.3f}" for k in row if k.startswith("error_"))
        + f"  mismatch {row['mismatch_surrogate']:.1e} -> fitted "
        f"{row['mismatch_fitted_phi0']:.1e}, predicted {row['mismatch_predicted_phi0']:.1e}"
        f"; without the {row['glitched_fraction']:.1%} of nodes on TEOBResumS' alpha "
        f"jumps: {row['mismatch_surrogate_clean']:.1e} -> predicted "
        f"{row['mismatch_predicted_phi0_clean']:.1e} (TEOBResumS' own multipoles "
        f"{row['mismatch_teob_raw_clean']:.1e})",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-binaries", type=int, default=48)
    parser.add_argument("--processes", type=int, default=7)
    args = parser.parse_args()

    # the binaries of precession_floor_anatomy, first line of sight of each;
    # TEOBResumS' h+, hx now with every inertial multipole (see teob_run)
    scaling.N_BINARIES = args.n_binaries
    scaling.N_ORIENTATIONS = 4
    cases = [case for i, case in enumerate(scaling.draw_cases()) if i % 4 == 0]
    with multiprocessing.Pool(args.processes) as pool:
        rows = [row for row in pool.imap_unordered(evaluate_binary, cases) if row]

    records = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    np.savez(DATA_PATH, **records)
    print(f"\n{len(rows)} binaries written to {DATA_PATH}")
    for key in sorted(k for k in records if k.startswith("error_")):
        print(f"{key:>22}: median |error| {np.median(np.abs(records[key])):.3f} rad, "
              f"90th pct {np.percentile(np.abs(records[key]), 90):.3f} rad")
    for key in sorted(k for k in records if k.startswith("mismatch_")):
        print(f"{key:>24}: median {np.median(records[key]):.2e}, "
              f"90th pct {np.percentile(records[key], 90):.2e}")


if __name__ == "__main__":
    main()
