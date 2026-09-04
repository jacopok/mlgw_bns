r"""Localise the residual non-smoothness in the reference-phase targets.

Three probes, all on a fine 1-D ``q`` (or other) slice:

1. **common vs per-mode.** The mode reference phases ``phi_lm(f0)`` are
   split into the cross-mode mean ``eps = <phi_lm(f0)>`` and the residual
   per-mode part. ``slice_mode_phase_common.py`` showed the jagged
   ~1e-5 rad wiggle is mode-independent, i.e. a common additive term.

2. **is the common term the EOB merger time?** A bare ``EOBRunPy`` call
   per point returns ``tc``; ``-2*pi*tc*f0`` is the merger-time-alignment
   contribution that lands identically in every mode's phase at ``f0``.
   Overlay its detrended form on ``eps``.

3. **q -> 1 edge.** Zoom the lowest-``q`` decade; odd-``m`` amplitudes
   vanish at ``q = 1`` so ``arg H_lm`` is ill-conditioned there.

Run::

    python probe_reference_kink_source.py --slice q --n 240
    python probe_reference_kink_source.py --slice chi_1 --base-q 1.1 --base-chi2 -0.4
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from mlgw_bns.model import DEFAULT_MODES, Model
from mlgw_bns.higher_order_modes import (
    Mode, mode_to_k, start_integration_early,
)
from mlgw_bns.dataset_generation import WaveformParameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
TWO_PI = 2.0 * np.pi

COL = {"q": 0, "lambda_1": 1, "lambda_2": 2, "chi_1": 3, "chi_2": 4}


def bare_tc(q, lam1, lam2, chi1, chi2, f_ref_natural, ds):
    """Merger time tc from a bare EOB call (2,2 mode only, fast)."""
    from EOBRun_module import EOBRunPy  # type: ignore

    params = WaveformParameters(q, lam1, lam2, chi1, chi2, dataset=ds)
    par_dict = params.teobresums()
    start_integration_early(par_dict, f_ref_natural, [Mode(2, 2)])
    par_dict["arg_out"] = "yes"
    par_dict["use_mode_lm"] = [mode_to_k(Mode(2, 2))]
    try:
        _f, _, _, _, _, _hflm, htlm, dyn = EOBRunPy(par_dict)
    except Exception:
        return np.nan
    if "tc" in dyn:
        return float(dyn["tc"])
    return float(np.asarray(htlm["t"])[-1])


def poly_detrend(x, y, deg):
    xn = (x - x.mean()) / (x.std() + 1e-30)
    d = np.column_stack([xn ** p for p in range(deg + 1)])
    c, *_ = np.linalg.lstsq(d, y, rcond=None)
    return y - d @ c


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slice", type=str, default="q", choices=list(COL))
    p.add_argument("--n", type=int, default=240)
    p.add_argument("--lo", type=float, default=None)
    p.add_argument("--hi", type=float, default=None)
    p.add_argument("--base-q", type=float, default=1.8)
    p.add_argument("--base-lambda", type=float, default=500.0)
    p.add_argument("--base-chi1", type=float, default=0.0)
    p.add_argument("--base-chi2", type=float, default=0.0)
    p.add_argument("--initial-frequency-hz", type=float, default=5.0)
    p.add_argument("--grid-points", type=int, default=64)
    p.add_argument("--fmax-hz", type=float, default=512.0)
    p.add_argument("--deg", type=int, default=7)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    col = COL[args.slice]
    default_rng = {"q": (1.02, 3.0), "chi_1": (-0.49, 0.49), "chi_2": (-0.49, 0.49),
                   "lambda_1": (5.0, 5000.0), "lambda_2": (5.0, 5000.0)}[args.slice]
    lo = args.lo if args.lo is not None else default_rng[0]
    hi = args.hi if args.hi is not None else default_rng[1]
    out = args.out or f"probe_reference_kink_{args.slice}.png"

    base = np.array([args.base_q, args.base_lambda, args.base_lambda,
                     args.base_chi1, args.base_chi2])

    model = Model(modes=list(DEFAULT_MODES),
                  initial_frequency_hz=args.initial_frequency_hz,
                  reference_amplitude=True)
    grid_hz, f_ref_natural, f0_nat = model._reference_grid(
        args.grid_points, args.fmax_hz)
    ds = model.mode_models[Mode(2, 2)].dataset
    modes = list(model.modes)
    f0_hz = grid_hz[0]

    grid = np.linspace(lo, hi, args.n)
    params = []
    for v in grid:
        row = base.copy(); row[col] = v
        params.append(WaveformParameters(*row, dataset=ds))

    p_s, dt_s, phi_s = model._reference_sweep_targets(
        iter(params), len(params), grid_hz, f_ref_natural,
        batch_size=len(params), progress_label=f"slice {args.slice}")
    x = p_s[:, col]
    o = np.argsort(x)
    x, dt_s, phi_s = x[o], dt_s[o], phi_s[o]

    # bare EOB merger time along the same slice
    tc = np.array(Parallel(n_jobs=16)(
        delayed(bare_tc)(*p_s[o][i], f_ref_natural, ds) for i in range(len(x))
    ))
    tc_ok = np.isfinite(tc)

    eps = phi_s.mean(axis=1)                       # cross-mode mean  (common part)
    per_mode = phi_s - eps[:, None]                # per-mode residual

    eps_d = poly_detrend(x, eps, args.deg)
    # -2 pi tc f is the merger-time-alignment term (_align_mode_phase_to_merger)
    # common to every mode; evaluated at f0 it is -2 pi f0 tc, all in the
    # natural (mass) units the phase residual is built in.
    align = -TWO_PI * f0_nat * tc
    align_d = np.full_like(x, np.nan)
    if tc_ok.sum() > args.deg + 2:
        align_d[tc_ok] = poly_detrend(x[tc_ok], align[tc_ok], args.deg)

    fig, axes = plt.subplots(3, 1, figsize=(11, 12))

    ax = axes[0]
    ax.plot(x, eps_d, ".-", ms=3, lw=0.8, color="tab:blue",
            label=r"$\langle\phi_{\ell m}(f_0)\rangle$ detrended (common part)")
    if np.isfinite(align_d).any():
        s = np.nanstd(eps_d) / (np.nanstd(align_d) + 1e-30)
        ax.plot(x, align_d * s, "-", lw=1.0, color="tab:red", alpha=0.8,
                label=rf"$-2\pi f_0 t_c$ detrended $\times${s:.2g} (merger time)")
    ax.set_ylabel("common phase [rad]")
    ax.set_title(
        f"common (cross-mode-mean) reference phase vs EOB merger time  "
        f"[{args.slice} slice, base q={base[0]:g} $\\Lambda$={base[1]:g} "
        f"$\\chi$=({base[3]:g},{base[4]:g})]")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for j, m in enumerate(modes):
        ax.plot(x, poly_detrend(x, per_mode[:, j], args.deg), ".-", ms=2.5, lw=0.7,
                label=f"({m.l},{m.m})")
    ax.set_ylabel(r"$\phi_{\ell m}-\langle\phi\rangle$ detrended [rad]")
    ax.set_title("per-mode residual after removing the common part")
    ax.legend(fontsize=8, ncol=4); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(x, dt_s, ".-", ms=3, lw=0.8, color="tab:green")
    ax.set_ylabel(r"$\Delta t$ [s]")
    ax.set_xlabel(args.slice)
    ax.set_title(r"cross-mode time shift $\Delta t$ (raw target)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    # numeric summary
    def rough(y):
        return float(np.sqrt(np.nanmean(np.diff(y, 2) ** 2)))
    print(f"\nslice {args.slice}: n={len(x)}  f0={f0_hz:.4f} Hz")
    print(f"  common part   roughness  {rough(eps_d):.3e} rad")
    for j, m in enumerate(modes):
        print(f"  ({m.l},{m.m}) per-mode  roughness  "
              f"{rough(poly_detrend(x, per_mode[:, j], args.deg)):.3e} rad")
    print(f"  Delta t       roughness  {rough(dt_s):.3e} s")
    if tc_ok.sum():
        print(f"  tc range {np.nanmin(tc):.4f}..{np.nanmax(tc):.4f} (nat), "
              f"d(tc) max {np.nanmax(np.abs(np.diff(tc[tc_ok]))):.3e}")
        if np.isfinite(align_d).any():
            m_ = tc_ok.copy()
            c = np.corrcoef(eps_d[m_], align_d[m_])[0, 1]
            print(f"  corr( common , -2pi f0 tc )  = {c:+.3f}")


if __name__ == "__main__":
    main()
