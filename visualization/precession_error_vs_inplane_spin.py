r"""How the precessing mismatch against TEOBResumS scales towards aligned spins.

For a fixed set of binaries (drawn as in
:mod:`visualization.validate_precessing_against_teob`) and lines of sight,
the in-plane spin components are scaled by a factor ``s`` from 0 (aligned)
to 1, and the full detector-strain mismatch against TEOBResumS' own
precessing :math:`h_+, h_\times` is computed twice: with the surrogate's
co-precessing multipoles, and with TEOBResumS' (``source="eob"``). The
second isolates the precession modelling from the network reconstruction;
at ``s = 0`` both reduce to the aligned-spin comparison.

Result (8 binaries x 2 lines of sight, 2026-09): the surrogate and the
TEOBResumS multipoles give the same mismatch throughout, so the network
error plays no part. Most cases are flat in the opening angle, at a level
already set in the aligned limit (7e-5 to 7e-3, worst edge-on); a few vary
non-monotonically, by up to ~4x, with no systematic growth towards large
beta. The floor is the co-precessing orbital phase at the reference
frequency, which differs between our EOB call and TEOBResumS' precessing
run by a binary-dependent n * phi0 (see the docstring of
validate_precessing_against_teob).

Run with: python visualization/precession_error_vs_inplane_spin.py [--plot-only]
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import validate_precessing_against_teob as v
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.precessing_model import PrecessingModel, PrecessingParametersWithExtrinsic

SCALES = np.array([0.0, 0.03, 0.1, 0.2, 0.5, 1.0])
N_BINARIES = 8
N_ORIENTATIONS = 2
SOURCES = ("surrogate", "eob")

HERE = Path(__file__).parent
DATA_PATH = HERE / "precession_error_vs_inplane_spin.npz"
FIGURE_PATH = HERE / "precession_error_vs_inplane_spin.png"

#: Categorical slots 1 and 2 of the reference palette.
COLORS = {"surrogate": "#2a78d6", "eob": "#eb6834"}


def draw_cases():
    """The binaries and orientations, reproducibly, as plain tuples."""
    model = Model.default_for_testing(modes=v.MODES)
    rng = np.random.default_rng(v.SEED)
    generator = model.dataset.make_parameter_generator(v.SEED)
    cases = []
    for index in range(N_BINARIES):
        intrinsic = next(generator)
        chi_1 = v.random_spin_vector(rng, intrinsic.chi_1)
        chi_2 = v.random_spin_vector(rng, intrinsic.chi_2)
        for _ in range(N_ORIENTATIONS):
            cases.append((index, intrinsic, chi_1, chi_2, v.random_orientation(rng)))
    return cases


_WORKER = {}


def evaluate(task):
    """Mismatches for one (case, scale) pair; NaN where TEOBResumS fails."""
    if not _WORKER:
        model = Model.default_for_testing(modes=v.MODES)
        v.SRATE_HZ = model.dataset.effective_srate_hz
        _WORKER["precessing"] = PrecessingModel(model)
        _WORKER["validator"] = ValidateModel(model.mode_models[Mode(2, 2)])
    precessing, validator = _WORKER["precessing"], _WORKER["validator"]
    frequencies = validator.frequencies

    (index, intrinsic, chi_1, chi_2, orientation), scale = task
    chi_1 = (scale * chi_1[0], scale * chi_1[1], chi_1[2])
    chi_2 = (scale * chi_2[0], scale * chi_2[1], chi_2[2])
    params = PrecessingParametersWithExtrinsic(
        mass_ratio=intrinsic.mass_ratio, lambda_1=intrinsic.lambda_1,
        lambda_2=intrinsic.lambda_2, chi_1=chi_1, chi_2=chi_2,
        distance_mpc=v.DISTANCE_MPC, inclination=orientation["inclination"],
        azimuth=orientation["azimuth"], total_mass=v.TOTAL_MASS,
        reference_frequency_hz=v.SPIN_REFERENCE_FREQUENCY_HZ,
    )
    f_plus, f_cross = v.antenna_patterns(orientation["theta"], orientation["phi"],
                                         orientation["psi"])
    angles = precessing.euler_angles(params, float(frequencies[0]))
    try:
        hp_t, hc_t, inside = v.teob_polarizations(
            intrinsic, chi_1, chi_2, params.inclination, params.azimuth, frequencies)
    except RuntimeError:  # TEOBResumS' root finder, occasionally
        return index, scale, float(angles.beta.max()), [np.nan] * len(SOURCES)
    teob = f_plus * hp_t + f_cross * hc_t
    mismatches = []
    for source in SOURCES:
        hp, hc = precessing.predict(frequencies, params, source=source, angles=angles)
        mismatches.append(validator.full_waveform_mismatch(
            {(2, 2): teob}, {(2, 2): (f_plus * hp + f_cross * hc)[inside]},
            frequencies=frequencies[inside]))
    return index, scale, float(angles.beta.max()), mismatches


def compute() -> dict:
    cases = draw_cases()
    tasks = [(case, scale) for case in cases for scale in SCALES]
    with multiprocessing.Pool(6) as pool:
        results = pool.map(evaluate, tasks, chunksize=1)
    shape = (len(cases), len(SCALES))
    records = {
        "scale": SCALES,
        "binary": np.array([case[0] for case in cases]),
        "inclination": np.array([case[4]["inclination"] for case in cases]),
        "opening_angle": np.array([r[2] for r in results]).reshape(shape),
    }
    for k, source in enumerate(SOURCES):
        records[source] = np.array([r[3][k] for r in results]).reshape(shape)
    np.savez(DATA_PATH, **records)
    return records


def plot(records: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    beta = records["opening_angle"]
    # s = 0 has beta = 0: draw it at a floor so the aligned limit shows
    floor = 0.5 * np.nanmin(beta[beta > 0])
    x = np.where(beta > 0, beta, floor)
    # One line per (binary, line of sight): a median across cases would mix
    # different binaries at each scale and fake a trend none of them has.
    for source, label in (("surrogate", "surrogate co-precessing modes"),
                          ("eob", "TEOBResumS co-precessing modes")):
        values = records[source]
        for row in range(values.shape[0]):
            ax.plot(x[row], values[row], color=COLORS[source], lw=1.2, alpha=0.7,
                    marker="o", ms=3, label=label if row == 0 else None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(floor, color="0.6", lw=0.8, ls=":")
    ax.annotate("aligned (s = 0)", (floor, ax.get_ylim()[1]), rotation=90,
                va="top", ha="right", color="0.35", fontsize=9)
    ax.set_xlabel(r"peak opening angle $\max_t \beta$ [rad]")
    ax.set_ylabel(r"mismatch vs TEOBResumS $h_+, h_\times$ (detector strain)")
    ax.set_title("Precessing mismatch as the in-plane spins are scaled to zero")
    ax.grid(True, which="major", color="0.9", lw=0.6)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"figure written to {FIGURE_PATH}")


def main() -> None:
    import sys

    if "--plot-only" in sys.argv:
        plot(dict(np.load(DATA_PATH)))
        return
    records = compute()
    print(f"{'s':>5}  {'median beta':>11}  " + "  ".join(f"{s:>10}" for s in SOURCES))
    for j, scale in enumerate(SCALES):
        print(f"{scale:5.2f}  {np.nanmedian(records['opening_angle'][:, j]):11.3f}  "
              + "  ".join(f"{np.nanmedian(records[s][:, j]):10.2e}" for s in SOURCES))
    plot(records)


if __name__ == "__main__":
    main()
