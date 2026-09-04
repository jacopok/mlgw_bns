"""Zoom on the localized (4,4) q-kink and the (3,3) q->1 blow-up.

For a fine 1-D q scan (other params at a base point) it plots, per mode:
the reference-phase residual at f0 (the regression target), its first
finite difference, and the amplitude residual at f0 -- to tell a 2*pi
phase-unwrap artefact from a real mode-amplitude feature.

Run with::

    python zoom_mode_kink.py --mode 4,4 --q-lo 1.8 --q-hi 2.6
    python zoom_mode_kink.py --mode 3,3 --q-lo 1.0 --q-hi 1.2 --base-chi1 0
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

from mlgw_bns.model import DEFAULT_MODES, Model
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.dataset_generation import WaveformParameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=str, default="4,4")
    parser.add_argument("--q-lo", type=float, default=1.8)
    parser.add_argument("--q-hi", type=float, default=2.6)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--base-lambda", type=float, default=500.0)
    parser.add_argument("--base-chi1", type=float, default=0.0)
    parser.add_argument("--base-chi2", type=float, default=0.0)
    parser.add_argument("--initial-frequency-hz", type=float, default=5.0)
    parser.add_argument("--grid-points", type=int, default=64)
    parser.add_argument("--fmax-hz", type=float, default=512.0)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    l, m = (int(x) for x in args.mode.split(","))
    mode = Mode(l, m)
    out = args.out or f"zoom_kink_{l}{m}.png"

    model = Model(
        modes=list(DEFAULT_MODES),
        initial_frequency_hz=args.initial_frequency_hz,
        reference_amplitude=True,
    )
    _grid_hz, f_ref_natural, _f0 = model._reference_grid(args.grid_points, args.fmax_hz)
    ds = model.mode_models[Mode(2, 2)].dataset

    q = np.linspace(args.q_lo, args.q_hi, args.n)
    params = [
        WaveformParameters(qi, args.base_lambda, args.base_lambda,
                           args.base_chi1, args.base_chi2, dataset=ds)
        for qi in q
    ]
    parameter_array, amp_res, phase_res = model._multimode_mode_residuals(
        params, f_ref_natural, progress_desc=f"q scan ({l},{m})",
    )
    qv = parameter_array[:, 0]
    order = np.argsort(qv)
    qv = qv[order]
    phi0 = phase_res[mode][order, 0]          # phase residual at f0 (the target)
    amp0 = amp_res[mode][order, 0]            # amplitude residual at f0

    dphi = np.diff(phi0)
    dq = np.diff(qv)
    jump_ix = int(np.argmax(np.abs(dphi)))
    jump = float(dphi[jump_ix])
    logging.info(
        "mode (%d,%d): largest step d(phi0) = %.4f rad at q=%.4f "
        "(= %.3f x 2pi); neighbouring dq = %.2e",
        l, m, jump, qv[jump_ix], jump / (2 * np.pi), dq[jump_ix],
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    axes[0].plot(qv, phi0, ".-", ms=4)
    axes[0].set_ylabel(r"$\phi_{\ell m}(f_0)$ residual [rad]")
    axes[0].set_title(
        f"mode ({l},{m}) at $\\Lambda$={args.base_lambda:g}, "
        f"$\\chi_1$={args.base_chi1:g}, $\\chi_2$={args.base_chi2:g}"
    )

    axes[1].plot(0.5 * (qv[1:] + qv[:-1]), dphi / dq, ".-", ms=4)
    axes[1].axhline(0, color="k", lw=0.6)
    for k in (-2, -1, 1, 2):
        axes[1].axhline(k * 2 * np.pi, color="tab:red", lw=0.6, ls=":")
    axes[1].set_ylabel(r"$\Delta\phi_0 / \Delta q$ [rad]")
    axes[1].set_title("finite difference (red dotted = multiples of $2\\pi$ per unit q)")

    axes[2].plot(qv, amp0, ".-", ms=4)
    axes[2].set_ylabel(r"amplitude residual at $f_0$  ($A_{\rm EOB}/A_{\rm PN}$)")
    axes[2].set_xlabel(r"$q$")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
