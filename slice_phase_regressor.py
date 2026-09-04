"""1-D parameter-space slices: true reference-phase leftover vs regressor.

Fits the chosen regressor (default Nystroem+RidgeCV, the
``compare_phase_regressors.py`` winner) per mode on the cached training
set, then walks a straight line through parameter space varying one
intrinsic parameter at a time (others held at a base point), generates
fresh EOB waveforms along it, and overplots

* the true calibration leftover  ``phi_lm(f0) - (linear backbone fit)``
* the regressor's prediction of it

so you can see *where* in parameter space the regressor tracks the target
and where it doesn't.

Run with::

    python slice_phase_regressor.py                 # uses phase_regressor_data_5hz.npz
    python slice_phase_regressor.py --regressor rff --n-fit 32000
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from mlgw_bns.model import DEFAULT_MODES, Model
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.neural_network import ModePhasesNN
from mlgw_bns.dataset_generation import WaveformParameters

from compare_phase_regressors import build

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

#: parameter -> (array column, axis label, (lo, hi) slice range)
SLICES = {
    "q":       (0, r"$q$",         (1.0, 3.0)),
    "chi_1":   (3, r"$\chi_1$",    (-0.5, 0.5)),
    "chi_2":   (4, r"$\chi_2$",    (-0.5, 0.5)),
    "lambda_1":(1, r"$\Lambda_1$", (5.0, 5000.0)),
}
#: base point (q, lambda_1, lambda_2, chi_1, chi_2) the slices pass through
BASE = np.array([1.8, 500.0, 500.0, 0.0, 0.0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=str, default="phase_regressor_data_5hz.npz")
    parser.add_argument("--regressor", type=str, default="nystroem")
    parser.add_argument("--n-fit", type=int, default=100_000,
                        help="training points used to fit the regressor")
    parser.add_argument("--initial-frequency-hz", type=float, default=5.0)
    parser.add_argument("--grid-points", type=int, default=64)
    parser.add_argument("--fmax-hz", type=float, default=512.0)
    parser.add_argument("--slice-points", type=int, default=60)
    parser.add_argument("--modes", type=str, default="2,1;3,3;4,4")
    parser.add_argument("--out", type=str, default="slice_phase_regressor.png")
    args = parser.parse_args()

    d = dict(np.load(args.cache))
    f0 = float(d["f0"])
    p_tr, phi_tr = d["p_tr"][: args.n_fit], d["phi_tr"][: args.n_fit]
    mode_rows = {tuple(r): i for i, r in enumerate(d["modes"].tolist())}
    want_modes = [tuple(int(x) for x in m.split(",")) for m in args.modes.split(";")]

    model = Model(
        modes=list(DEFAULT_MODES),
        initial_frequency_hz=args.initial_frequency_hz,
        reference_amplitude=True,
    )
    grid_hz, f_ref_natural, f0_model = model._reference_grid(
        args.grid_points, args.fmax_hz
    )
    assert abs(f0_model - f0) < 1e-9 * f0, "cache f0 != model f0; check frequency"
    ds = model.mode_models[Mode(2, 2)].dataset

    # per-mode: linear backbone calibration + fitted regressor on the leftover
    calib, regr = {}, {}
    for lm in want_modes:
        j = mode_rows[lm]
        design = ModePhasesNN._backbone_design(p_tr, lm, f0)
        c = LinearRegression().fit(design, phi_tr[:, j])
        left = phi_tr[:, j] - c.predict(design)
        est = build(args.regressor)
        est.fit(p_tr, left)
        calib[lm], regr[lm] = c, est
        logging.info("mode %s: leftover std %.2f rad, regressor %s fitted on %d pts",
                     lm, left.std(), args.regressor, len(p_tr))

    # generate EOB targets along every slice
    slice_data = {}
    for name, (col, _label, (lo, hi)) in SLICES.items():
        grid = np.linspace(lo, hi, args.slice_points)
        params = []
        for v in grid:
            row = BASE.copy()
            row[col] = v
            params.append(WaveformParameters(*row, dataset=ds))
        p_s, _dt, phi_s = model._reference_sweep_targets(
            iter(params), len(params), grid_hz, f_ref_natural,
            batch_size=len(params), progress_label=f"slice {name}",
        )
        slice_data[name] = (p_s, phi_s)
        logging.info("slice %-9s: %d/%d valid EOB waveforms", name, len(p_s), len(grid))

    # plot: rows = modes, cols = slices
    nrow, ncol = len(want_modes), len(SLICES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                             squeeze=False)
    for r, lm in enumerate(want_modes):
        j = mode_rows[lm]
        for cix, (name, (col, label, _rng)) in enumerate(SLICES.items()):
            ax = axes[r][cix]
            p_s, phi_s = slice_data[name]
            x = p_s[:, col]
            true_left = phi_s[:, j] - calib[lm].predict(
                ModePhasesNN._backbone_design(p_s, lm, f0)
            )
            pred_left = regr[lm].predict(p_s)
            order = np.argsort(x)
            ax.plot(x[order], true_left[order], "o-", ms=3, lw=1,
                    color="tab:blue", label="true leftover")
            ax.plot(x[order], pred_left[order], "-", lw=1.8,
                    color="tab:red", label="regressor")
            ax.set_xlabel(label)
            if cix == 0:
                ax.set_ylabel(f"mode {lm}\nleftover [rad]")
            ax.grid(True, alpha=0.3)
            if r == 0 and cix == 0:
                ax.legend(fontsize=8)

    base_txt = (f"base: q={BASE[0]:g}, $\\Lambda_1$={BASE[1]:g}, "
                f"$\\Lambda_2$={BASE[2]:g}, $\\chi_1$={BASE[3]:g}, $\\chi_2$={BASE[4]:g}")
    fig.suptitle(
        f"reference-phase leftover along parameter slices "
        f"({args.regressor}, {args.initial_frequency_hz:g} Hz)\n{base_txt}"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
