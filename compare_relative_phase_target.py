r"""Does regressing ``phi_lm(f0) - phi_22(f0)`` beat ``phi_lm(f0)`` directly?

The reference-phase kink hunt showed the mode phases share a large,
strongly curved common term -- the merger-time-alignment phase
``2 pi f0 tc(theta)``, mode-independent -- that a smooth regressor fits
only slowly (``~n^-0.25``). ``ModePhasesNN.relative_to_22`` regresses the
difference ``phi_lm(f0) - phi_22(f0)`` for every higher-order mode, which
cancels that common term exactly, leaving only per-mode PN structure.

This fits :class:`ModePhasesNN` both ways on the same EOB draw over nested
training-set sizes and reports the held-out per-mode error (wrapped to
``(-pi, pi]`` -- the reconstructed waveform only cares mod ``2 pi`` -- and
raw RMS), plus a boxplot.

Run::

    python compare_relative_phase_target.py --train-size 12000 --val-size 2000 \
        --subsets 1000,2000,4000,8000,12000 --initial-frequency-hz 5
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

from mlgw_bns.model import DEFAULT_MODES, Model
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.neural_network import ModePhasesNN

from validate_regressor_training_curve import make_targets, wrapped

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def fit_and_score(subsets, train, val, mode_tuples, f0, relative):
    p_tr, _dt_tr, phi_tr = train
    p_val, _dt_val, phi_val = val
    out = {}
    for n in subsets:
        mp = ModePhasesNN(
            modes=mode_tuples, f0_natural=f0,
            training_params=p_tr[:n], training_mode_phases=phi_tr[:n],
            relative_to_22=relative,
        ).fit()
        pred = mp.predict(p_val)
        wrap_err = np.abs(wrapped(pred - phi_val))          # (n_val, n_modes)
        raw_err = pred - phi_val
        out[n] = (wrap_err, raw_err)
        logging.info(
            "relative=%-5s n=%-6d  wrapped median %s rad",
            relative, n,
            np.array2string(np.median(wrap_err, axis=0), precision=3, separator=","),
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-size", type=int, default=12000)
    ap.add_argument("--val-size", type=int, default=2000)
    ap.add_argument("--subsets", type=str, default="1000,2000,4000,8000,12000")
    ap.add_argument("--grid-points", type=int, default=64)
    ap.add_argument("--fmax-hz", type=float, default=512.0)
    ap.add_argument("--initial-frequency-hz", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--seed-train", type=int, default=1)
    ap.add_argument("--seed-val", type=int, default=2)
    ap.add_argument("--out", type=str, default="compare_relative_phase_target.png")
    args = ap.parse_args()

    subsets = sorted(int(s) for s in args.subsets.split(","))
    model = Model(modes=list(DEFAULT_MODES),
                  initial_frequency_hz=args.initial_frequency_hz,
                  reference_amplitude=True)
    modes = list(model.modes)
    mode_tuples = [(m.l, m.m) for m in modes]
    grid_hz, f_ref_natural, f0 = model._reference_grid(args.grid_points, args.fmax_hz)

    train = make_targets(model, grid_hz, f_ref_natural, args.train_size,
                         args.seed_train, args.batch_size, "train sweep")
    val = make_targets(model, grid_hz, f_ref_natural, args.val_size,
                       args.seed_val, args.batch_size, "val sweep")
    subsets = [n for n in subsets if n <= len(train[0])]
    logging.info("valid: %d train, %d val", len(train[0]), len(val[0]))

    res = {
        False: fit_and_score(subsets, train, val, mode_tuples, f0, False),
        True: fit_and_score(subsets, train, val, mode_tuples, f0, True),
    }

    print("\n=== held-out per-mode error: wrapped median / wrapped 90pct / raw RMS [rad] ===")
    for j, m in enumerate(modes):
        print(f"\nmode ({m.l},{m.m})")
        print(f"  {'n_train':>8s} | {'absolute':^34s} | {'relative to (2,2)':^34s}")
        for n in subsets:
            a_w, a_r = res[False][n]
            r_w, r_r = res[True][n]
            def fmt(w, r, jj):
                return (f"{np.median(w[:, jj]):.3e} "
                        f"{np.percentile(w[:, jj], 90):.3e} "
                        f"{np.sqrt(np.mean(r[:, jj] ** 2)):.3e}")
            print(f"  {n:8d} | {fmt(a_w, a_r, j):^34s} | {fmt(r_w, r_r, j):^34s}")

    # plot: wrapped |error| boxplots, absolute vs relative, per mode, at max subset
    n = subsets[-1]
    ncols = 2
    nrows = int(np.ceil(len(modes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for j, m in enumerate(modes):
        ax = axes[j // ncols][j % ncols]
        data, labels = [], []
        for nn in subsets:
            data.append(res[False][nn][0][:, j]); labels.append(f"{nn}\nabs")
            data.append(res[True][nn][0][:, j]); labels.append(f"{nn}\nrel")
        bp = ax.boxplot(data, showfliers=False, patch_artist=True)
        for k, box in enumerate(bp["boxes"]):
            box.set(facecolor="tab:blue" if k % 2 else "tab:gray", alpha=0.5)
        ax.set_yscale("log")
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(f"mode ({m.l},{m.m})")
        ax.set_ylabel(r"$|{\rm wrapped}(\phi^{\rm pred}-\phi^{\rm true})|$ [rad]")
        ax.grid(True, which="both", axis="y", alpha=0.3)
    for k in range(len(modes), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("reference phase: absolute target (grey) vs relative to (2,2) (blue)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
