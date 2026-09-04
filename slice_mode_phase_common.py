r"""Is the non-smoothness in phi_lm(f0) common across modes (EOB jitter) or
per-mode (PN)?

Same slices as ``slice_reference_targets.py`` but focuses on the mode
reference phases and, per slice, plots three things stacked:

  1. phi_lm(f0) detrended            -- what the ModePhasesNN regressor sees
  2. phi_lm(f0) - (m/2) phi_22(f0)   -- mode phase relative to (2,2)
  3. the common part  phi_22(f0) detrended, overlaid on each

If the jagged ~1e-4 rad wiggle cancels in row 2, the regression target
should be redefined relative to (2,2): the shared EOB phase jitter drops
out and only the smooth per-mode PN structure remains.
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

SLICES = {
    "q":        (0, r"$q$",         (1.05, 2.98)),
    "chi_1":    (3, r"$\chi_1$",    (-0.49, 0.49)),
    "chi_2":    (4, r"$\chi_2$",    (-0.49, 0.49)),
    "lambda_1": (1, r"$\Lambda_1$", (10.0, 4900.0)),
}


def detrend(x, y, extra=None, deg=5):
    xn = (x - x.mean()) / (x.std() + 1e-30)
    cols = [xn ** p for p in range(deg + 1)]
    if extra is not None:
        cols.append((extra - extra.mean()) / (extra.std() + 1e-30))
    d = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(d, y, rcond=None)
    return y - d @ c


def roughness(y):
    """RMS of the second difference -- a scale-free non-smoothness measure."""
    return float(np.sqrt(np.mean(np.diff(y, 2) ** 2)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=161)
    p.add_argument("--base-q", type=float, default=1.8)
    p.add_argument("--base-lambda", type=float, default=500.0)
    p.add_argument("--base-chi1", type=float, default=0.0)
    p.add_argument("--base-chi2", type=float, default=0.0)
    p.add_argument("--initial-frequency-hz", type=float, default=5.0)
    p.add_argument("--grid-points", type=int, default=64)
    p.add_argument("--fmax-hz", type=float, default=512.0)
    p.add_argument("--out", type=str, default="slice_mode_phase_common.png")
    args = p.parse_args()

    base = np.array([args.base_q, args.base_lambda, args.base_lambda,
                     args.base_chi1, args.base_chi2])

    model = Model(modes=list(DEFAULT_MODES),
                  initial_frequency_hz=args.initial_frequency_hz,
                  reference_amplitude=True)
    grid_hz, f_ref_natural, f0 = model._reference_grid(args.grid_points, args.fmax_hz)
    ds = model.mode_models[Mode(2, 2)].dataset
    modes = list(model.modes)
    i22 = modes.index(Mode(2, 2))

    nrow, ncol = len(modes), len(SLICES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 2.8 * nrow),
                             squeeze=False)

    print("\n=== roughness (RMS of 2nd difference) ===")
    print(f"{'slice':10s} {'mode':6s} {'phi_lm':>12s} {'phi_lm-(m/2)phi22':>18s} "
          f"{'ratio':>8s}")

    for cix, (name, (col, label, (lo, hi))) in enumerate(SLICES.items()):
        grid = np.linspace(lo, hi, args.n)
        params = []
        for v in grid:
            row = base.copy(); row[col] = v
            params.append(WaveformParameters(*row, dataset=ds))
        p_s, _dt, phi_s = model._reference_sweep_targets(
            iter(params), len(params), grid_hz, f_ref_natural,
            batch_size=len(params), progress_label=f"slice {name}")
        x = p_s[:, col]
        o = np.argsort(x); x = x[o]; phi_s = phi_s[o]
        phi22 = phi_s[:, i22]

        for r, m in enumerate(modes):
            ax = axes[r][cix]
            phi_lm = phi_s[:, r]
            rel = phi_lm - (m.m / 2.0) * phi22

            m_lm = ModePhasesNN._backbone_design(p_s[o], (m.l, m.m), f0)[:, 0]
            d_abs = detrend(x, phi_lm, extra=m_lm)
            d_rel = detrend(x, rel, extra=m_lm)
            d_22 = detrend(x, phi22)

            ax.plot(x, d_abs, ".-", ms=2.5, lw=0.8, color="tab:blue",
                    label=r"$\phi_{\ell m}$")
            ax.plot(x, d_rel, ".-", ms=2.5, lw=0.8, color="tab:red",
                    label=r"$\phi_{\ell m}-\frac{m}{2}\phi_{22}$")
            ax.plot(x, d_22, "-", lw=0.7, color="0.6", alpha=0.8,
                    label=r"$\phi_{22}$")
            ax.grid(True, alpha=0.3); ax.axhline(0, color="k", lw=0.5)
            if cix == 0:
                ax.set_ylabel(f"({m.l},{m.m})\ndetrended [rad]")
            if r == nrow - 1:
                ax.set_xlabel(label)
            if r == 0 and cix == 0:
                ax.legend(fontsize=7)

            r_abs, r_rel = roughness(d_abs), roughness(d_rel)
            print(f"{name:10s} ({m.l},{m.m})  {r_abs:12.3e} {r_rel:18.3e} "
                  f"{r_rel / (r_abs + 1e-30):8.3f}")

    fig.suptitle(
        f"mode reference phase: absolute vs relative to (2,2) "
        f"({args.initial_frequency_hz:g} Hz)\n"
        f"base q={base[0]:g}, $\\Lambda$={base[1]:g}, "
        f"$\\chi_1$={base[3]:g}, $\\chi_2$={base[4]:g}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
