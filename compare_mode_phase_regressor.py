r"""Nystroem vs. compact ridge regressors for the ModePhasesNN leftover,
with the *current* pipeline (relative-to-(2,2) target + linear analytic
backbone + psi_lm 2*pi fix).

The reference-phase predictor's Nystroem(2500) pipeline stores an
``n_components**2`` normalization matrix (~50 MB), which would dominate the
~2 MB shipped package. This script asks whether that footprint buys enough
accuracy over:

* ``nystroem_1000`` / ``nystroem_500``  -- fewer landmarks (~8 MB / ~2 MB)
* ``rff_4000``   -- RBFSampler(4000) + Ridge, the previous default (~0.25 MB)
* ``kernel_ridge`` -- exact RBF KernelRidge (footprint ~ n_train, ~1 MB)

For each, a full :class:`ModePhasesNN` (``relative_to_22=True``,
``f0_natural`` set) is fitted on nested subsets of one EOB reference sweep
and scored on a held-out draw: per-mode phase error wrapped to
``(-pi, pi]`` (what the waveform sees) and raw RMS. The fitted predictor is
also pickled three ways -- as-is, with the training arrays dropped, and the
bare regressor -- to size the real on-disk cost.

Run::

    uv run python compare_mode_phase_regressor.py           # 16k @ 5 Hz
    uv run python compare_mode_phase_regressor.py --train-size 8000
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile

import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline

from mlgw_bns.model import DEFAULT_MODES, Model
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.neural_network import ModePhasesNN, TimeshiftsNN

from validate_regressor_training_curve import make_targets, wrapped

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

GAMMA = ModePhasesNN.DEFAULT_GAMMA
RANDOM_STATE = ModePhasesNN.DEFAULT_RANDOM_STATE


def factories() -> dict:
    def nystroem(nc):
        return lambda: TimeshiftsNN.make_nystroem_ridge_pipeline(
            n_components=nc, gamma=GAMMA, random_state=RANDOM_STATE
        )

    def rff(nc):
        return lambda: TimeshiftsNN.make_rff_ridge_pipeline(
            n_components=nc, gamma=GAMMA, ridge_alpha=1e-9, random_state=RANDOM_STATE
        )

    def kernel_ridge():
        return Pipeline([("kr", KernelRidge(kernel="rbf", gamma=GAMMA, alpha=1e-8))])

    return {
        "nystroem_2500": nystroem(2500),
        "nystroem_1000": nystroem(1000),
        "nystroem_500": nystroem(500),
        "rff_4000": rff(4000),
        "kernel_ridge": kernel_ridge,
    }


def pkl_sizes(mp: ModePhasesNN) -> tuple[float, float, float]:
    """(full, training-data-stripped, bare-regressor) pickle size in MB."""
    with tempfile.TemporaryDirectory() as d:
        full = os.path.join(d, "full.pkl")
        joblib.dump(mp, full)
        size_full = os.path.getsize(full) / 1e6

        tp, tmp = mp.training_params, mp.training_mode_phases
        mp.training_params = mp.training_mode_phases = None
        stripped = os.path.join(d, "stripped.pkl")
        joblib.dump(mp, stripped)
        size_stripped = os.path.getsize(stripped) / 1e6
        mp.training_params, mp.training_mode_phases = tp, tmp

        reg = os.path.join(d, "reg.pkl")
        joblib.dump(mp.regressor, reg)
        size_reg = os.path.getsize(reg) / 1e6
    return size_full, size_stripped, size_reg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-size", type=int, default=16000)
    ap.add_argument("--val-size", type=int, default=3000)
    ap.add_argument("--subsets", type=str, default="2000,4000,8000,16000")
    ap.add_argument("--grid-points", type=int, default=64)
    ap.add_argument("--fmax-hz", type=float, default=512.0)
    ap.add_argument("--initial-frequency-hz", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--regressors", type=str,
                    default="nystroem_2500,nystroem_1000,nystroem_500,rff_4000,kernel_ridge")
    ap.add_argument("--cache", type=str, default="mode_phase_regressor_data.npz")
    ap.add_argument("--out", type=str, default="compare_mode_phase_regressor.png")
    args = ap.parse_args()

    model = Model(modes=list(DEFAULT_MODES),
                  initial_frequency_hz=args.initial_frequency_hz,
                  reference_amplitude=True)
    modes = list(model.modes)
    mode_tuples = [(m.l, m.m) for m in modes]
    grid_hz, f_ref_natural, f0 = model._reference_grid(args.grid_points, args.fmax_hz)

    try:
        d = dict(np.load(args.cache))
        assert len(d["p_tr"]) >= args.train_size and len(d["p_val"]) >= args.val_size
        assert float(d["initial_frequency_hz"]) == args.initial_frequency_hz
        logging.info("loaded cache %s (%d train)", args.cache, len(d["p_tr"]))
        p_tr, phi_tr = d["p_tr"], d["phi_tr"]
        p_val, phi_val = d["p_val"], d["phi_val"]
    except (FileNotFoundError, KeyError, AssertionError):
        logging.info("generating %d train / %d val @ %g Hz",
                     args.train_size, args.val_size, args.initial_frequency_hz)
        p_tr, _dt, phi_tr = make_targets(
            model, grid_hz, f_ref_natural, args.train_size, 1, args.batch_size, "train")
        p_val, _dt, phi_val = make_targets(
            model, grid_hz, f_ref_natural, args.val_size, 2, args.batch_size, "val")
        np.savez(args.cache, p_tr=p_tr, phi_tr=phi_tr, p_val=p_val, phi_val=phi_val,
                 initial_frequency_hz=np.float64(args.initial_frequency_hz),
                 modes=np.array(mode_tuples))
        logging.info("wrote %s", args.cache)

    subsets = [n for n in sorted(int(s) for s in args.subsets.split(",")) if n <= len(p_tr)]
    regressors = args.regressors.split(",")
    facs = factories()

    # results[(mode_idx, regressor)][n] = (wrapped_median, wrapped_90, raw_rms)
    results: dict = {}
    sizes: dict = {}
    for rname in regressors:
        for n in subsets:
            mp = ModePhasesNN(
                modes=mode_tuples, f0_natural=f0, relative_to_22=True,
                training_params=p_tr[:n], training_mode_phases=phi_tr[:n],
                pipeline_factory=facs[rname],
            ).fit()
            pred = mp.predict(p_val)
            werr = np.abs(wrapped(pred - phi_val))   # (n_val, n_modes)
            raw = pred - phi_val
            for j in range(len(modes)):
                results[(j, rname, n)] = (
                    float(np.median(werr[:, j])),
                    float(np.percentile(werr[:, j], 90)),
                    float(np.sqrt(np.mean(raw[:, j] ** 2))),
                )
            if n == subsets[-1]:
                sizes[rname] = pkl_sizes(mp)
            logging.info(
                "%-13s n=%-6d  wrapped med per mode %s rad",
                rname, n,
                np.array2string(np.median(werr, axis=0), precision=3, separator=","),
            )

    # ---- tables ----
    print("\n=== on-disk size at largest subset [MB] ===")
    print(f"  {'regressor':14s} {'full pkl':>10s} {'no train-data':>14s} {'regressor only':>15s}")
    for rname in regressors:
        f, s, r = sizes[rname]
        print(f"  {rname:14s} {f:10.2f} {s:14.2f} {r:15.2f}")

    print("\n=== held-out per-mode phase error: wrapped median (90th) [rad] | raw RMS [rad] ===")
    for j, m in enumerate(modes):
        print(f"\nmode ({m.l},{m.m}):")
        hdr = f"  {'regressor':14s} " + " ".join(f"n={n}".ljust(24) for n in subsets)
        print(hdr)
        for rname in regressors:
            cells = []
            for n in subsets:
                med, p90, rms = results[(j, rname, n)]
                cells.append(f"{med:.3f} ({p90:.2f}) | {rms:.2f}".ljust(24))
            print(f"  {rname:14s} " + " ".join(cells))

    # ---- plot ----
    fig, axes = plt.subplots(2, len(modes), figsize=(5 * len(modes), 9), squeeze=False)
    for j, m in enumerate(modes):
        for row, (idx, ylab) in enumerate(
            [(0, "wrapped median phase error [rad]"), (2, "raw RMS leftover [rad]")]
        ):
            ax = axes[row][j]
            for rname in regressors:
                ax.plot(subsets, [results[(j, rname, n)][idx] for n in subsets],
                        "o-", label=rname)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("training-set size")
            ax.set_ylabel(ylab)
            ax.set_title(f"mode ({m.l},{m.m})")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize="small")
    fig.suptitle(
        f"ModePhasesNN regressor comparison, current pipeline "
        f"(relative-to-(2,2), {args.initial_frequency_hz:g} Hz)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
