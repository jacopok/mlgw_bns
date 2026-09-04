r"""Self-convergence of the geometric training grid.

!!! CAVEAT (added after the fact) !!!
Like `explore_multiband_grid.py`, this measures cubic-spline reconstruction of
*merger-time-aligned* EOB output and so is far too optimistic about the
geometric grid. The real training pipeline unwraps TEOBResumS's wrapped FD
phase along the requested grid, and a log-spaced grid slips 2*pi per point at
low frequency. The geometric grid was investigated and rejected; keep the
multiband grid. See the memory note `geometric-training-grid`.

Companion to ``explore_multiband_grid.py``. That script established that a
plain geometric grid can replace the multiband grid; this one measures how
fast it converges as the point count grows from 1e3 to 1e5.

For each candidate ``np.geomspace(f0, fmax, N)`` and a handful of EOB
waveforms:

1. generate the EOB modes on ``N`` points, cubic-spline resample up to a
   fixed, very fine geometric reference grid (``REFERENCE_POINTS``);
2. compare to the EOB modes generated directly on the reference grid:

   * PSD-weighted mismatch (time-shift marginalised),
   * max and median :math:`|\Delta\phi(f)|` across the band,
   * max and median fractional :math:`|\Delta A(f)| / A`.

Because every grid in the sweep is drawn from the same geometric family and
compared to a finer member of it, this is a self-convergence test: the
curves show the discretisation error of the grid choice alone, and a
Richardson reading (error at ``N`` vs error at ``2N``) recovers the
convergence order. Thin dashed guides mark ``N^-4`` (cubic-spline pointwise
error, local spacing ``h ~ 1/N``) and ``N^-8`` (its square, the mismatch).

Output
------
``geometric_grid_convergence.png``   the five error measures vs ``N``
``geometric_grid_convergence.txt``   the same numbers as a table

Run with:  uv run python geometric_grid_convergence.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlgw_bns.downsampling_interpolation import DownsamplingTraining
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import Model

from explore_multiband_grid import Tee, eob_modes, load_psd, mismatch

INITIAL_FREQUENCY_HZ = 5.0
MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]

#: geometric point counts to test
N_SWEEP = [1000, 1500, 2200, 3300, 5000, 7500, 10000,
           15000, 22000, 33000, 50000, 75000, 100000]

#: fixed fine geometric reference grid (~5x the largest candidate)
REFERENCE_POINTS = 500_000
#: decimate the reference for the mismatch integral only
MM_STRIDE = 4

N_WAVEFORMS = 8
SEED = 5150


def main() -> None:
    out = open("geometric_grid_convergence.txt", "w")
    sys.stdout = Tee(sys.__stdout__, out)

    model = Model(
        modes=MODES,
        filename="geometric_grid_convergence_tmp",
        initial_frequency_hz=INITIAL_FREQUENCY_HZ,
        reference_amplitude=True,
    )
    mode_model = model.mode_models[Mode(2, 2)]
    dataset = mode_model.dataset
    generator = mode_model.waveform_generator

    f0 = dataset.effective_initial_frequency_hz
    fmax = dataset.effective_srate_hz / 2
    print(f"band {f0:.3f} - {fmax:.1f} Hz;  reference: geomspace with "
          f"{REFERENCE_POINTS} points")

    ref_grid = np.geomspace(f0, fmax, REFERENCE_POINTS)
    ref_nat = dataset.hz_to_natural_units(ref_grid)
    psd_ref = load_psd(ref_grid)
    mm_sl = slice(5, -5, MM_STRIDE)
    mm_freqs, mm_psd = ref_grid[mm_sl], psd_ref[mm_sl]
    edge = slice(5, -5)

    grids = {n: np.geomspace(f0, fmax, n) for n in N_SWEEP}

    pg = dataset.make_parameter_generator(seed=SEED)
    params_list = [next(pg) for _ in range(N_WAVEFORMS)]

    # metrics[n][mode] -> list over waveforms of
    #   (mismatch, max|dphi|, med|dphi|, max fracamp, med fracamp)
    metrics = {n: {m: [] for m in MODES} for n in N_SWEEP}

    t0 = time.time()
    for i, params in enumerate(params_list):
        try:
            ref_modes = eob_modes(generator, params, ref_grid, dataset, MODES)
        except Exception as exc:  # pragma: no cover
            print(f"  waveform {i}: reference EOB failed ({exc}); skipping")
            continue
        ref_cart = {m: a * np.exp(1j * p) for m, (a, p) in ref_modes.items()}

        for n, grid in grids.items():
            grid_nat = dataset.hz_to_natural_units(grid)
            try:
                cand = eob_modes(generator, params, grid, dataset, MODES)
            except Exception:
                continue
            for m in MODES:
                a_c, p_c = cand[m]
                a_r, p_r = ref_modes[m]
                a_up = DownsamplingTraining.resample(grid_nat, ref_nat, a_c)
                p_up = DownsamplingTraining.resample(grid_nat, ref_nat, p_c)

                h_r = ref_cart[m][mm_sl]
                h_c = a_up[mm_sl] * np.exp(1j * p_up[mm_sl])
                mm = mismatch(h_r, h_c, mm_freqs, mm_psd)

                dphi = np.abs(p_up - p_r)[edge]
                # fractional amplitude error, ignoring the odd-m nodes where
                # A_ref passes through zero
                scale = np.abs(a_r)[edge]
                good = scale > 1e-3 * np.max(scale)
                frac = np.abs(a_up - a_r)[edge][good] / scale[good]

                metrics[n][m].append(
                    (mm, dphi.max(), np.median(dphi), frac.max(), np.median(frac))
                )
        print(f"  waveform {i + 1}/{len(params_list)} "
              f"({time.time() - t0:.0f} s)")

    _tables(metrics)
    _plot(grids, metrics)
    print("\nSaved geometric_grid_convergence.png")
    sys.stdout = sys.__stdout__
    out.close()


def _median_stack(rows):
    arr = np.array(rows)
    return np.median(arr, axis=0) if len(arr) else np.full(5, np.nan)


LABELS = ["mismatch", "max |dphi| [rad]", "median |dphi| [rad]",
          "max fractional dA/A", "median fractional dA/A"]


def _tables(metrics):
    for k, label in enumerate(LABELS):
        print(f"\n  --- {label} (median over waveforms) ---")
        header = "  " + f"{'N':>8s}" + "".join(
            f"{f'({m.l},{m.m})':>12s}" for m in MODES)
        print(header)
        for n in N_SWEEP:
            vals = [_median_stack(metrics[n][m])[k] for m in MODES]
            print("  " + f"{n:>8d}" + "".join(f"{v:12.2e}" for v in vals))


#: y-axis floor per panel (machine precision / reference-grid noise), and the
#: reference power-law slope for the guide line
PANEL_FLOOR = [1e-16, 3e-16, 3e-16, 1e-16, 1e-16]
PANEL_SLOPE = [-8.0, -4.0, -2.0, -4.0, -4.0]
#: greedy-downsampling tolerances, drawn as a horizontal line where relevant
PANEL_TOL = [None, 3e-4, 3e-4, 8e-4, 8e-4]  # tol_phi / tol_amp


def _plot(grids, metrics):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.ravel()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    Ns = np.array(N_SWEEP, dtype=float)
    anchor_i = 1  # anchor the slope guide at N_SWEEP[1], still pre-convergence

    for k, (ax, label) in enumerate(zip(axes, LABELS)):
        for m, color in zip(MODES, colors):
            y = np.array([_median_stack(metrics[n][m])[k] for n in N_SWEEP])
            y = np.maximum(y, PANEL_FLOOR[k])  # 0 == "below double precision"
            ax.loglog(Ns, y, "o-", color=color, label=f"({m.l},{m.m})")

        anchor_y = max(
            _median_stack(metrics[N_SWEEP[anchor_i]][m])[k] for m in MODES
        )
        if np.isfinite(anchor_y) and anchor_y > 0:
            guide = anchor_y * (Ns / Ns[anchor_i]) ** PANEL_SLOPE[k]
            guide[guide < PANEL_FLOOR[k] * 0.5] = np.nan  # keep the axis sane
            ax.loglog(Ns, guide, "k--", lw=0.8,
                      label=rf"$N^{{{PANEL_SLOPE[k]:.0f}}}$")
        ax.set_ylim(bottom=PANEL_FLOOR[k] * 0.3)

        if PANEL_TOL[k] is not None:
            ax.axhline(PANEL_TOL[k], color="0.4", ls=":", lw=1.2,
                       label="downsampling tol")

        ax.set_xlabel("geometric grid points $N$")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize="small", ncol=2)

    axes[-1].axis("off")
    axes[-1].text(
        0.02, 0.95,
        "Reference is itself geometric (500k pts): the flat floors are the\n"
        "reference's own discretisation / double-precision noise, not the\n"
        "candidate grid.\n\n"
        "Everything except the (4,4) worst-point phase error is converged\n"
        "by N ~ 5000.  The (4,4) mode has a localised phase feature near its\n"
        "merger that needs N ~ 30-50k for the *max* error to drop well below\n"
        "tol_phi; its median error is fine much sooner.\n\n"
        "N ~ 20000 (26x smaller than the 519k multiband grid) keeps every\n"
        "mode comfortably inside both downsampling tolerances.",
        transform=axes[-1].transAxes, va="top", fontsize="medium", family="monospace",
    )
    fig.suptitle(
        f"Geometric training-grid self-convergence "
        f"(reference: geomspace, {REFERENCE_POINTS} points; "
        f"{N_WAVEFORMS} waveforms)")
    fig.tight_layout()
    fig.savefig("geometric_grid_convergence.png", dpi=150)


if __name__ == "__main__":
    main()
