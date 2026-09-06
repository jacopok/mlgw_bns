"""1-D parameter-space slices of the per-mode PCA components.

For the *current trained model* this walks straight lines through
parameter space (one intrinsic parameter varied, the rest held at a base
point), generates fresh multi-mode EOB waveforms along each line, forms
each mode's ``combined`` residual exactly as the training pipeline does
(amplitude ratio + linear-trend-and-anchor-removed phase), projects it
onto that mode's stored PCA basis, and plots

* the true PCA component along the slice (blue dots) -- is it a smooth
  function of the parameter, or does it have kinks / jumps?
* the trained regressor's prediction of the same component (red line) --
  does the NN track it?

The motivating question is whether the (2,1) reconstruction error is
driven by a discontinuity in parameter space (a non-smooth target the
smooth regressor cannot follow) or is just regression bias on a smooth
but high-curvature target.

Run with::

    python slice_mode_pca.py                       # (2,1) vs (2,2), default_hom
    python slice_mode_pca.py --modes 2,1 --n-components 10 --n-points 120
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

from mlgw_bns.model import Model
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.dataset_generation import WaveformParameters
from mlgw_bns.principal_component_analysis import (
    PrincipalComponentAnalysisModel,
    remove_linear_trend,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ALL_MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]

#: parameter -> (WaveformParameters.array column, axis label, (lo, hi))
SLICES = {
    "q":        (0, r"$q$",         (1.02, 2.98)),
    "chi_1":    (3, r"$\chi_1$",    (-0.48, 0.48)),
    "chi_2":    (4, r"$\chi_2$",    (-0.48, 0.48)),
    "lambda_1": (1, r"$\Lambda_1$", (10.0, 4900.0)),
}
#: base point (q, lambda_1, lambda_2, chi_1, chi_2) every slice passes through
BASE = np.array([1.6, 400.0, 400.0, 0.0, 0.0])


def mode_components(model, mode, parameter_array, amp_res, phase_res):
    """True (EOB) and NN-predicted PCA components for one mode along a slice.

    Returns ``(comps_true, comps_pred)``, both ``(n_points, K)`` in the
    same normalisation that :meth:`PrincipalComponentAnalysisModel.reduce_data`
    produces (the regressor target divided back by ``eigenvalues**pc_exponent``).
    """
    mm = model.mode_models[mode]
    ds = model.dataset
    phase_indices = mm.downsampling_indices.phase_indices
    freqs_hz = ds.natural_units_to_hz(np.asarray(ds.frequencies)[phase_indices])

    pset = ds.parameter_set_cls(np.asarray(parameter_array, dtype=np.float64))
    flattened_phase = remove_linear_trend(
        parameters=pset,
        phi_diff=phase_res[mode],
        frq=freqs_hz,
        timeshifts_predictor=model.time_shifts_predictor,
        subtract_mode_phase_anchor=True,
        mode_phases_predictor=model.mode_phases_predictor,
        mode_index=model.modes.index(mode),
    )
    combined = np.concatenate(
        (np.asarray(amp_res[mode], dtype=np.float64), flattened_phase), axis=1
    )
    comps_true = PrincipalComponentAnalysisModel.reduce_data(combined, mm.pca_data)

    scaled = mm.nn.predict(np.asarray(parameter_array, dtype=np.float64))
    comps_pred = scaled / (mm.pca_data.eigenvalues ** mm.nn.hyper.pc_exponent)
    return comps_true, comps_pred


def roughness(x, y):
    """Scale-free kink detector: max |2nd difference| / median |1st difference|.

    ~1 for a smooth curve sampled finely; >> 1 where there is a kink/jump.
    """
    order = np.argsort(x)
    y = y[order]
    d1 = np.abs(np.diff(y))
    d2 = np.abs(np.diff(y, 2))
    scale = np.median(d1)
    if scale == 0:
        return np.nan
    return float(np.max(d2) / scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="default_hom",
                        help="base filename, or several separated by ';' to "
                             "overlay their regressors against one truth "
                             "(only meaningful when they share a PCA basis)")
    parser.add_argument("--modes", type=str, default="2,1;2,2")
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--n-points", type=int, default=90)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--out-prefix", type=str, default="slice_mode_pca")
    parser.add_argument("--base", type=str, default=None,
                        help="override base point as 'q,l1,l2,chi1,chi2'")
    parser.add_argument("--range", action="append", default=[],
                        metavar="NAME,LO,HI",
                        help="override a slice range, e.g. --range q,1.0,1.4 "
                             "(repeatable); only the named slices are drawn")
    args = parser.parse_args()

    global BASE, SLICES
    if args.base is not None:
        BASE = np.array([float(x) for x in args.base.split(",")])
    if args.range:
        chosen = {}
        for spec in args.range:
            name, lo, hi = spec.split(",")
            col, label, _ = SLICES[name]
            chosen[name] = (col, label, (float(lo), float(hi)))
        SLICES = chosen

    want_modes = [Mode(*(int(x) for x in m.split(","))) for m in args.modes.split(";")]

    names = args.model.split(";")
    models = []
    for name in names:
        loaded = Model(modes=ALL_MODES, filename=name)
        loaded.load()
        if not loaded.nn_available:
            raise SystemExit(f"no trained network for {name!r}")
        models.append(loaded)

    # The first model supplies the EOB sweep, the PCA basis and the shared
    # predictors; the others contribute only their regressor prediction.
    model = models[0]
    ds = model.dataset
    freqs_natural = np.asarray(ds.frequencies, dtype=float)

    ds_idx = {m: model.mode_models[m].downsampling_indices for m in model.modes}
    amp_ref = {
        m: model.mode_models[m].dataset.amplitude_reference_parameters
        for m in model.modes
    }

    # EOB sweep along every slice
    slice_data = {}
    for name, (col, _lbl, (lo, hi)) in SLICES.items():
        grid = np.linspace(lo, hi, args.n_points)
        params = []
        for v in grid:
            row = BASE.copy()
            row[col] = v
            params.append(WaveformParameters(*row, dataset=ds))
        p_arr, amp_res, phase_res = model._multimode_mode_residuals(
            params, freqs_natural, ds_idx, amp_ref,
            progress_desc=f"slice {name}", n_jobs=args.n_jobs,
        )
        slice_data[name] = (p_arr, amp_res, phase_res)
        logging.info("slice %-9s: %d/%d valid EOB waveforms",
                     name, len(p_arr), len(grid))

    K = args.n_components
    for mode in want_modes:
        fig, axes = plt.subplots(
            K, len(SLICES), figsize=(max(3.6 * len(SLICES), 11), 2.1 * K),
            squeeze=False, sharex="col",
        )
        worst = []
        colours = ["tab:red", "tab:green", "tab:purple", "tab:orange"]
        for cix, (name, (col, label, _rng)) in enumerate(SLICES.items()):
            p_arr, amp_res, phase_res = slice_data[name]
            x = p_arr[:, col]
            comps_true, _ = mode_components(model, mode, p_arr, amp_res, phase_res)
            predictions = [
                mode_components(other, mode, p_arr, amp_res, phase_res)[1]
                for other in models
            ]
            order = np.argsort(x)
            for k in range(K):
                ax = axes[k][cix]
                ax.plot(x[order], comps_true[order, k], "o-", ms=3, lw=0.8,
                        color="tab:blue", label="true (EOB)")
                for label_name, comps_pred, colour in zip(names, predictions, colours):
                    ax.plot(x[order], comps_pred[order, k], "-", lw=1.6,
                            color=colour, label=label_name)
                r = roughness(x, comps_true[:, k])
                worst.append((r, name, k))
                ax.set_title(f"PC {k}   (roughness {r:.1f})", fontsize=8)
                ax.grid(True, alpha=0.3)
                if cix == 0:
                    ax.set_ylabel(f"PC {k}")
                if k == K - 1:
                    ax.set_xlabel(label)
                if k == 0 and cix == 0:
                    ax.legend(fontsize=7)

            # Quantify what the eye sees: component error over the whole
            # slice, and over the part of it that carries mode power.
            print(f"\nmode {mode.l}{mode.m}, slice {name}: "
                  f"RMS |predicted - true| over PC 0-{K - 1}")
            interesting = x > 1.2 if name == "q" else np.ones_like(x, dtype=bool)
            for label_name, comps_pred in zip(names, predictions):
                error = comps_pred[:, :K] - comps_true[:, :K]
                print(f"  {label_name:<12} all: {np.sqrt(np.mean(error ** 2)):.4f}"
                      f"   {'q>1.2' if name == 'q' else 'same'}: "
                      f"{np.sqrt(np.mean(error[interesting] ** 2)):.4f}")

        worst.sort(reverse=True)
        top = ", ".join(f"{n}/PC{k}:{r:.0f}" for r, n, k in worst[:5])
        fig.suptitle(
            f"mode {mode.l}{mode.m}: PCA components along parameter slices "
            f"({', '.join(names)})\nbase q={BASE[0]:g} $\\Lambda_1$={BASE[1]:g} "
            f"$\\Lambda_2$={BASE[2]:g} $\\chi_1$={BASE[3]:g} $\\chi_2$={BASE[4]:g}"
            f"   |   roughest: {top}"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = f"{args.out_prefix}_{mode.l}{mode.m}.png"
        fig.savefig(out, dpi=140)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
