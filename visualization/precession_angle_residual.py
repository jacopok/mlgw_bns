r"""TEOBResumS' own co-precessing multipoles, twisted by us, against its h+, hx.

:mod:`precession_floor_anatomy` twists TEOBResumS' own co-precessing
multipoles along *our* Euler angles (``teob_raw``); for most binaries that
reproduced its precessing h+, hx to 1e-6--1e-5, for some only to 1e-3.
This isolates the reference: the same twist along TEOBResumS' *own* angles
(its ``dynspin.txt``, ``output_dynamics``), against its h+, hx computed
with its default inertial-mode set (only the (l, m) of ``use_mode_lm``)
and with every inertial multipole, masking 0--6 spline intervals of its
spin dynamics around each spurious 2 pi step of its alpha
(:func:`alpha_jumps`).

Findings (6 binaries): the angles make no difference; the truncated
inertial sum leaves 1e-5--7e-5, growing with the opening angle; each alpha
step leaves glitches that need ~6 intervals of margin. With the complete
sum and that margin our twist reproduces TEOBResumS to 1e-10--1e-14.

Run with: python visualization/precession_angle_residual.py BINARY [BINARY ...]
(indices of the binaries of precession_floor_anatomy)
"""

from __future__ import annotations

import multiprocessing
import sys
import tempfile
from pathlib import Path

import numpy as np

import precession_error_vs_inplane_spin as scaling
import precession_floor_anatomy as anatomy
import validate_precessing_against_teob as v
from mlgw_bns.higher_order_modes import mode_to_k
from mlgw_bns.precessing_model import EulerAngles, PrecessingParametersWithExtrinsic

DATA_PATH = Path(__file__).with_name("precession_angle_residual.npz")
#: Spline intervals masked either side of TEOBResumS' alpha jumps (None: no mask).
MARGINS = (None, 1, 3, 6)


def teob_dynspin(directory: str) -> np.ndarray:
    """TEOBResumS' spin dynamics as written to ``dynspin.txt``: columns t,
    SA (3), SB (3), Lhat (3), alpha, beta, gamma, M Omega."""
    return np.loadtxt(Path(directory) / "dynspin.txt")


def alpha_jumps(dynspin: np.ndarray) -> np.ndarray:
    r"""Indices ``k`` where TEOBResumS' stored :math:`\alpha` steps by more than
    :math:`\pi` from sample ``k`` to ``k + 1``.

    These are spurious ~2 pi steps (within Delta M Omega ~ 1e-6, far less
    than a precession cycle): harmless in :math:`e^{i n \alpha}` at the
    samples, but ``twist_hlm_FD`` splines :math:`\alpha` through them, so
    between those samples (and, by the spline's ringing, next to them) its
    twist uses a wrong angle.
    """
    return np.flatnonzero(np.abs(np.diff(dynspin[:, 10])) > np.pi)


def teob_angles(directory: str) -> EulerAngles:
    """TEOBResumS' Euler angles against M Omega, from its ``dynspin.txt``."""
    data = teob_dynspin(directory)
    momega, keep = np.unique(data[:, 13], return_index=True)
    return EulerAngles(momega=momega, alpha=data[keep, 10], beta=data[keep, 11],
                       gamma=data[keep, 12], time=data[keep, 0])


def evaluate_binary(case):
    """TEOBResumS' own multipoles twisted by us, against its h+, hx.

    For both its truncated and its complete inertial-mode sums, for a range
    of margins around its alpha jumps, and with our Euler angles or its own.
    """
    model, precessing, validator = anatomy._setup()
    index, intrinsic, chi_1, chi_2, orientation = case
    params = PrecessingParametersWithExtrinsic(
        mass_ratio=intrinsic.mass_ratio, lambda_1=intrinsic.lambda_1,
        lambda_2=intrinsic.lambda_2, chi_1=chi_1, chi_2=chi_2,
        distance_mpc=v.DISTANCE_MPC, inclination=orientation["inclination"],
        azimuth=orientation["azimuth"], total_mass=v.TOTAL_MASS,
        reference_frequency_hz=v.SPIN_REFERENCE_FREQUENCY_HZ,
    )
    ours = precessing.euler_angles(params, float(validator.frequencies[0]))
    antenna = v.antenna_patterns(orientation["theta"], orientation["phi"], orientation["psi"])
    mass_seconds = params.aligned().mass_sum_seconds
    row = dict(binary=index, opening_angle=float(ours.beta.max()))
    for reference, extra in (
        # TEOBResumS' default: only the inertial (l, m) of use_mode_lm
        ("truncated", dict(use_mode_lm_inertial=sorted({mode_to_k(m) for m in v.MODES}))),
        ("complete", {}),  # teob_run's own: every inertial multipole
    ):
        with tempfile.TemporaryDirectory() as directory:
            f_native, hp, hc, raw, _ = v.teob_run(
                intrinsic, chi_1, chi_2, params.inclination, params.azimuth,
                multipoles=True,
                overrides=dict(output_dynamics="yes", output_dir=directory, **extra),
            )
            dynspin = teob_dynspin(directory)
            theirs = teob_angles(directory)
        nodes, fb, _, _, teob_raw, _ = anatomy.aligned_fit(
            model, validator, params, f_native, raw)
        target = (antenna[0] * hp + antenna[1] * hc)[nodes]
        jumps = alpha_jumps(dynspin)
        row["alpha_jumps"] = jumps.size
        for margin in MARGINS:
            glitched = np.zeros(fb.size, bool)
            if margin is not None:
                near = np.concatenate([jumps + k for k in range(-margin, margin + 1)])
                for m in sorted({m for (_, m) in raw}):
                    sample = np.searchsorted(
                        dynspin[:, 13], 2 * np.pi * fb * mass_seconds / m) - 1
                    glitched |= np.isin(sample, near)
            keep = ~glitched
            for label, angles in (("ours", ours), ("teob", theirs)):
                row[f"{reference}_{margin}_{label}"] = validator.full_waveform_mismatch(
                    {(2, 2): target[keep]},
                    {(2, 2): anatomy.strain({k: a[keep] for k, a in teob_raw.items()},
                                            fb[keep], angles, params, antenna)},
                    frequencies=fb[keep],
                )
            row[f"masked_{margin}"] = glitched.mean()
    print(f"binary {index} (beta {row['opening_angle']:.3f}, {row['alpha_jumps']} alpha jumps): "
          + " | ".join(
              f"{reference} +-{margin}: {row[f'{reference}_{margin}_ours']:.1e} "
              f"(its angles {row[f'{reference}_{margin}_teob']:.1e})"
              for reference in ("truncated", "complete") for margin in MARGINS),
          flush=True)
    return row


def main() -> None:
    wanted = [int(x) for x in sys.argv[1:]]
    scaling.N_BINARIES = max(wanted) + 1
    scaling.N_ORIENTATIONS = 4
    cases = [c for i, c in enumerate(scaling.draw_cases()) if i % 4 == 0 and c[0] in wanted]
    with multiprocessing.Pool(min(len(cases), 6)) as pool:
        rows = pool.map(evaluate_binary, cases)
    records = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    np.savez(DATA_PATH, **records)
    print(f"written to {DATA_PATH}")


if __name__ == "__main__":
    main()
