r"""Look for residual kinks in the reference pre-pass regression targets.

The shared predictors are trained on two per-point targets produced by
:meth:`Model._reference_sweep_targets`:

* the cross-mode time shift ``Delta t`` -- the low-frequency least-squares
  slope of the (2,2) phase residual (``Residuals.phase_timeshifts``);
* each mode's reference phase ``phi_lm(f0)`` -- the phase residual at the
  first grid node.

This walks fine 1-D slices in ``q``, ``chi_1``, ``chi_2``, ``Lambda_1``
(others held at a base point), regenerates the EOB targets along each
slice, and flags isolated spikes in the discrete second difference -- a
kink an ordinary smooth regressor cannot represent.

For every kink found it also re-derives the (2,2) time shift and each
mode phase from a bare per-point chain so the pathology can be localised
to the raw TEOBResumS phase, the merger-time alignment, or the mlgw_bns
Taylor-F2 (l,m) PN subtraction.

Run with::

    python slice_reference_targets.py                       # default base point
    python slice_reference_targets.py --n 121 --base-chi1 0.3 --base-lambda 1000
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

from mlgw_bns.model import DEFAULT_MODES, Model
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.neural_network import ModePhasesNN
from mlgw_bns.dataset_generation import WaveformParameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

#: parameter -> (array column, axis label, (lo, hi))
SLICES = {
    "q":        (0, r"$q$",         (1.02, 2.98)),
    "chi_1":    (3, r"$\chi_1$",    (-0.49, 0.49)),
    "chi_2":    (4, r"$\chi_2$",    (-0.49, 0.49)),
    "lambda_1": (1, r"$\Lambda_1$", (10.0, 4900.0)),
}


def kink_indices(y: np.ndarray, k: float = 8.0, min_abs: float = 0.0):
    """Indices i (into y) where the second difference spikes in isolation."""
    d2 = np.diff(y, 2)
    scale = 1.4826 * np.median(np.abs(d2 - np.median(d2))) + 1e-30
    flag = (np.abs(d2) > k * scale) & (np.abs(d2) > min_abs)
    # d2[i] involves y[i], y[i+1], y[i+2]; attribute the kink to y[i+1]
    return np.where(flag)[0] + 1, d2, scale


def smooth_detrend(x: np.ndarray, y: np.ndarray, m_lm: np.ndarray | None) -> np.ndarray:
    """Remove a smooth trend for visualisation: M_lm (if given) + cubic in x."""
    xn = (x - x.mean()) / (x.std() + 1e-30)
    cols = [np.ones_like(xn), xn, xn ** 2, xn ** 3]
    if m_lm is not None:
        cols.insert(0, (m_lm - m_lm.mean()) / (m_lm.std() + 1e-30))
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=101, help="points per slice")
    parser.add_argument("--base-q", type=float, default=1.8)
    parser.add_argument("--base-lambda", type=float, default=500.0)
    parser.add_argument("--base-chi1", type=float, default=0.0)
    parser.add_argument("--base-chi2", type=float, default=0.0)
    parser.add_argument("--initial-frequency-hz", type=float, default=5.0)
    parser.add_argument("--grid-points", type=int, default=64)
    parser.add_argument("--fmax-hz", type=float, default=512.0)
    parser.add_argument("--k", type=float, default=8.0, help="MAD multiple for a kink")
    parser.add_argument("--out", type=str, default="slice_reference_targets.png")
    args = parser.parse_args()

    base = np.array([args.base_q, args.base_lambda, args.base_lambda,
                     args.base_chi1, args.base_chi2])

    model = Model(
        modes=list(DEFAULT_MODES),
        initial_frequency_hz=args.initial_frequency_hz,
        reference_amplitude=True,
    )
    grid_hz, f_ref_natural, f0 = model._reference_grid(args.grid_points, args.fmax_hz)
    ds = model.mode_models[Mode(2, 2)].dataset
    modes = list(model.modes)

    # rows of the figure
    row_labels = [r"$\Delta t$ [s]"] + [f"$\\phi_{{{m.l}{m.m}}}(f_0)$ [rad]" for m in modes]

    slice_results = {}
    for name, (col, _label, (lo, hi)) in SLICES.items():
        grid = np.linspace(lo, hi, args.n)
        params = []
        for v in grid:
            row = base.copy()
            row[col] = v
            params.append(WaveformParameters(*row, dataset=ds))
        p_s, dt_s, phi_s = model._reference_sweep_targets(
            iter(params), len(params), grid_hz, f_ref_natural,
            batch_size=len(params), progress_label=f"slice {name}",
        )
        slice_results[name] = (p_s, dt_s, phi_s)
        logging.info("slice %-9s: %d/%d valid EOB waveforms", name, len(p_s), len(grid))

    # ---- kink report + figure -------------------------------------------------
    nrow, ncol = 1 + len(modes), len(SLICES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 2.9 * nrow),
                             squeeze=False)

    print("\n=== kink report (isolated second-difference spikes) ===")
    for cix, (name, (col, label, _rng)) in enumerate(SLICES.items()):
        p_s, dt_s, phi_s = slice_results[name]
        x = p_s[:, col]
        order = np.argsort(x)
        x = x[order]

        targets = [dt_s[order]] + [phi_s[order, j] for j in range(len(modes))]
        for r, (y, rlabel) in enumerate(zip(targets, row_labels)):
            ax = axes[r][cix]
            if r == 0:
                m_lm = None
            else:
                m_lm = ModePhasesNN._backbone_design(
                    p_s[order], (modes[r - 1].l, modes[r - 1].m), f0
                )[:, 0]
            det = smooth_detrend(x, y, m_lm)
            idx, d2, scale = kink_indices(y, k=args.k)

            ax.plot(x, det, ".-", ms=3, lw=0.9, color="tab:blue")
            for i in idx:
                ax.axvline(x[i], color="tab:red", lw=1.0, alpha=0.7)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="k", lw=0.5)
            if cix == 0:
                ax.set_ylabel(rlabel)
            if r == nrow - 1:
                ax.set_xlabel(label)

            if len(idx):
                tag = "dt" if r == 0 else f"({modes[r-1].l},{modes[r-1].m})"
                for i in idx:
                    step = y[i] - y[i - 1] if i > 0 else np.nan
                    extra = ""
                    if r > 0:
                        extra = f"  = {step / (2 * np.pi):+.3f} x 2pi"
                    print(f"  {tag:7s} vs {name:9s}  x={x[i]:9.4f}  "
                          f"jump={step:+.5g}{extra}")

    fig.suptitle(
        f"reference pre-pass targets along parameter slices "
        f"({args.initial_frequency_hz:g} Hz)\n"
        f"base: q={base[0]:g}, $\\Lambda$={base[1]:g}, "
        f"$\\chi_1$={base[3]:g}, $\\chi_2$={base[4]:g}   "
        f"(smooth trend removed; red = isolated $\\Delta^2$ spike)"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
