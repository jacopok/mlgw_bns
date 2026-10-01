r"""Mismatch against TEOBResumS: the precessing model next to the aligned-spin one.

For each binary and line of sight of
:func:`precession_error_vs_inplane_spin.draw_cases`, two comparisons made
in exactly the same way:

* ``precessing`` --- :meth:`~mlgw_bns.precessing_model.PrecessingModel.predict`
  against TEOBResumS' :math:`h_+, h_\times` for the binary as drawn;
* ``aligned`` --- the same with the in-plane spins zeroed on both sides,
  where :class:`~mlgw_bns.precessing_model.PrecessingModel` is
  :meth:`~mlgw_bns.model.Model.predict` (with the orbital phase set at the
  reference frequency rather than at the merger).

Both put the spins, and the orbital phase, at TEOBResumS' own reference
point, the first sample of its integration
(:func:`validate_precessing_against_teob.teob_reference`, which also
reconciles the frequencies its spin and its orbital dynamics read there):
the information a validation has and a user does not. Both are evaluated on
TEOBResumS' own frequency nodes
(:func:`precession_floor_anatomy.comparison_nodes`), projected on one
detector, and maximised over time and one global phase only. Two more
versions of the precessing mismatch:

* ``precessing_clean`` --- without the frequencies where TEOBResumS' own
  twist reads its Euler angle :math:`\alpha` across one of its spurious
  2 pi steps (:func:`precession_angle_residual.alpha_glitched`), a defect
  of the reference;
* ``precessing_merger`` --- the co-precessing multipoles with their
  orbital phase at the merger, as before the reference-phase convention.

Run with: python visualization/precessing_vs_aligned_mismatch.py
    [--n-binaries N] [--n-orientations K] [--processes P] [--plot-only]
"""

from __future__ import annotations

import argparse
import multiprocessing
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import precession_error_vs_inplane_spin as scaling
import precession_floor_anatomy as anatomy
from precession_angle_residual import alpha_glitched, teob_dynspin
from precession_orbital_phase_origin import JUMP_MARGIN
import validate_precessing_against_teob as v
from mlgw_bns.precessing_model import PrecessingParametersWithExtrinsic

DATA_PATH = Path(__file__).with_name("precessing_vs_aligned_mismatch.npz")
FIGURE_PATH = Path(__file__).with_name("precessing_vs_aligned_mismatch.png")
#: Width of the density estimates, in decades of mismatch: one fixed width
#: for every curve, rather than one scaled to each curve's spread (which a
#: bimodal curve would blur).
BANDWIDTH_DEX = 0.15


def compare(case, aligned_spins):
    """One TEOBResumS call and the model's prediction for one line of sight."""
    model, precessing, validator = anatomy._setup()
    index, intrinsic, chi_1, chi_2, orientation = case
    if aligned_spins:
        chi_1, chi_2 = (0.0, 0.0, chi_1[2]), (0.0, 0.0, chi_2[2])
    with tempfile.TemporaryDirectory() as directory:
        f_native, hp, hc, raw, dynamics = v.teob_run(
            intrinsic, chi_1, chi_2, orientation["inclination"], orientation["azimuth"],
            multipoles=True,
            overrides=dict(output_dynamics="yes", output_dir=directory),
        )
        dynspin = None if aligned_spins else teob_dynspin(directory)
    params = PrecessingParametersWithExtrinsic(
        mass_ratio=intrinsic.mass_ratio, lambda_1=intrinsic.lambda_1,
        lambda_2=intrinsic.lambda_2, chi_1=chi_1, chi_2=chi_2,
        distance_mpc=v.DISTANCE_MPC, inclination=orientation["inclination"],
        azimuth=orientation["azimuth"], total_mass=v.TOTAL_MASS,
    )
    mass_seconds = params.aligned().mass_sum_seconds
    params.reference_frequency_hz, params.reference_phase = v.teob_reference(
        precessing, params, dynamics)

    nodes, fb = anatomy.comparison_nodes(validator, f_native)
    antenna = v.antenna_patterns(orientation["theta"], orientation["phi"], orientation["psi"])
    target = (antenna[0] * hp + antenna[1] * hc)[nodes]
    angles = precessing.euler_angles(params, float(fb[0]))
    hp_model, hc_model = precessing.predict(fb, params, angles=angles)

    def mismatch(strain, keep=slice(None)):
        return validator.full_waveform_mismatch(
            {(2, 2): target[keep]}, {(2, 2): strain[keep]}, frequencies=fb[keep])

    row = dict(mismatch=mismatch(antenna[0] * hp_model + antenna[1] * hc_model),
               reference_frequency=params.reference_frequency_hz,
               reference_phase=params.reference_phase,
               opening_angle=float(angles.beta.max()))
    if not aligned_spins:
        keep = ~alpha_glitched(dynspin, fb, sorted({m for (_, m) in raw}),
                               mass_seconds, JUMP_MARGIN)
        row["clean"] = mismatch(antenna[0] * hp_model + antenna[1] * hc_model, keep)
        row["glitched_fraction"] = 1.0 - keep.mean()
        merger = anatomy.strain(model.coprecessing_modes_dict(fb, params.aligned()),
                                fb, angles, params, antenna)
        row["merger"] = mismatch(merger)
    return row


def evaluate_case(case):
    index, intrinsic, chi_1, chi_2, orientation = case
    try:
        precessing = compare(case, aligned_spins=False)
        aligned = compare(case, aligned_spins=True)
    except RuntimeError as error:  # TEOBResumS' root finder, occasionally
        print(f"binary {index}: TEOBResumS failed ({error})", flush=True)
        return None
    row = dict(
        binary=index, mass_ratio=intrinsic.mass_ratio,
        inclination=orientation["inclination"],
        in_plane_spin=max(np.hypot(*chi_1[:2]), np.hypot(*chi_2[:2])),
        opening_angle=precessing["opening_angle"],
        reference_frequency=precessing["reference_frequency"],
        reference_phase=precessing["reference_phase"],
        aligned=aligned["mismatch"],
        precessing=precessing["mismatch"],
        precessing_clean=precessing["clean"],
        precessing_merger=precessing["merger"],
        glitched_fraction=precessing["glitched_fraction"],
    )
    print(f"binary {index} (beta {row['opening_angle']:.3f}, iota {row['inclination']:.2f}): "
          f"aligned {row['aligned']:.1e}, precessing {row['precessing']:.1e} "
          f"({row['precessing_clean']:.1e} without the {row['glitched_fraction']:.1%} of "
          f"nodes on alpha jumps; {row['precessing_merger']:.1e} merger-referenced)",
          flush=True)
    return row


#: label, key, colour, line style
CURVES = (
    ("aligned spins", "aligned", "#2a6fdb", "-"),
    ("precessing", "precessing", "#d9480f", "-"),
    ("precessing, without TEOBResumS' alpha-jump bins", "precessing_clean", "#d9480f", "--"),
    ("precessing, orbital phase at the merger", "precessing_merger", "#868e96", ":"),
)


def plot(records):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    logs = {key: np.log10(records[key][np.isfinite(records[key]) & (records[key] > 0)])
            for _, key, _, _ in CURVES}
    every = np.concatenate(list(logs.values()))
    grid = np.linspace(every.min() - 0.5, every.max() + 0.5, 600)
    for label, key, colour, style in CURVES:
        values = logs[key]
        density = gaussian_kde(values, bw_method=BANDWIDTH_DEX / values.std(ddof=1))(grid)
        median = np.median(values)
        ax.plot(grid, density, color=colour, ls=style, lw=2,
                label=f"{label}: median {10 ** median:.1e}")
        ax.axvline(median, color=colour, ls=style, lw=0.8, alpha=0.7)
    ax.set_xlabel(r"$\log_{10}$ mismatch against TEOBResumS $h_+, h_\times$")
    ax.set_ylabel("density")
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    n_binaries = np.unique(records["binary"]).size
    ax.set_title(
        f"{records['aligned'].size} lines of sight of {n_binaries} binaries "
        f"(total mass {v.TOTAL_MASS}, $|\\chi_\\perp| < {v.MAX_IN_PLANE_SPIN}$), "
        "orbital phase at TEOBResumS' first sample",
        fontsize=9,
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"figure written to {FIGURE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-binaries", type=int, default=48)
    parser.add_argument("--n-orientations", type=int, default=4)
    parser.add_argument("--processes", type=int, default=7)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if not args.plot_only:
        scaling.N_BINARIES = args.n_binaries
        scaling.N_ORIENTATIONS = args.n_orientations
        cases = scaling.draw_cases()
        with multiprocessing.Pool(args.processes) as pool:
            rows = [row for row in pool.imap_unordered(evaluate_case, cases) if row]
        records = {key: np.array([row[key] for row in rows]) for key in rows[0]}
        np.savez(DATA_PATH, **records)
        print(f"\n{len(rows)} lines of sight written to {DATA_PATH}")
    records = dict(np.load(DATA_PATH))
    for label, key, _, _ in CURVES:
        print(f"{label:>48}: median {np.median(records[key]):.2e}, "
              f"90th pct {np.percentile(records[key], 90):.2e}, "
              f"worst {np.max(records[key]):.2e}")
    plot(records)


if __name__ == "__main__":
    main()
