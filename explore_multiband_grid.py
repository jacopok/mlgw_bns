r"""Exploratory comparison of frequency-grid choices for the training grid.

!!! CAVEAT (added after the fact) !!!
This script's optimistic conclusion (a geometric grid is ~lossless at ~50x
fewer points) is WRONG for the actual training pipeline. It compares
merger-time-aligned multi-mode EOB output, whose dominant linear phase slope
is removed, so a cubic spline through a sparse log-spaced grid looks fine.
The real `Dataset.generate_residuals` path uses single-mode, non-aligned
`phase_eob - phase_pn`: there the low-frequency phase slope is ~2500 rad/Hz,
TEOBResumS unwraps its wrapped FD phase along the *requested* grid, and a
log-spaced grid (dphi/step ~10 rad >> pi at 14 Hz) makes that unwrap slip a
full 2*pi per grid point. The multiband grid's dN/df ~ T(f) low-f density is
what prevents this. Keep the multiband grid. See the memory note
`geometric-training-grid`.

Background
----------
The greedy downsampling training (``GreedyDownsamplingTraining``) and the
dataset generation that follows it both work on the *multiband* grid
``mlgw_bns.multibanding.reduced_frequency_array``. At
``initial_frequency_hz = 5`` that grid is ~5x10^5 points; a training
dataset of 2**15 waveforms is then a (32768, ~520000) float array, which
is where the memory trouble comes from.

The multiband grid follows the Vinciguerra et al. rule ``dN/df ~ T(f)``
(seglen), i.e. it is roughly uniform in ``f^{-5/3}``. This script asks how
much accuracy is lost if it is replaced by:

* ``geom``    -- a plain geometric grid, ``np.geomspace`` (uniform in log f);
* ``invf53``  -- uniform in ``f^{-5/3}`` (the multiband low-f rule, applied
  across the whole band, with a tunable point count);
* ``hybrid``  -- geometric below ``f_pivot``, uniform above it.

Method
------
For each candidate grid ``G`` and each of a handful of EOB waveforms:

1. generate the EOB modes directly on a dense reference grid (a 4x-refined
   multiband grid) -> the "truth";
2. generate the EOB modes on ``G``, then cubic-spline resample up to the
   reference grid -- exactly the reconstruction path
   ``DownsamplingTraining.resample`` uses at inference;
3. compare: PSD-weighted mismatch (time-shift marginalised), and the
   L-infinity / median phase and fractional-amplitude errors across the band.

Part B additionally runs the real greedy downsampling on top of each
candidate grid and reports the resulting node count and the L-infinity
phase reconstruction error on fresh waveforms, versus the same greedy run
on the dense reference grid.

Outputs
-------
``explore_multiband_grid_mismatch.png``  mismatch & phase error vs point count
``explore_multiband_grid_spectrum.png``  per-frequency phase / amplitude error
``explore_multiband_grid.txt``           the printed tables

Run with:  uv run python explore_multiband_grid.py
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from mlgw_bns import multibanding
from mlgw_bns.multibanding import COMMON_GRID_MODE, reduced_frequency_array
from mlgw_bns.downsampling_interpolation import (
    DownsamplingTraining,
    GreedyDownsamplingTraining,
)
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import Model

logging.basicConfig(level=logging.WARNING)

INITIAL_FREQUENCY_HZ = 5.0
MODES = [Mode(2, 2), Mode(3, 3), Mode(4, 4)]
REPORT_MODES = [Mode(2, 2), Mode(4, 4)]

#: reference grid = multiband grid refined by this factor (power of two, so
#: the high-frequency ``df`` is divided exactly). Cubic-spline error scales
#: as ``df**4``, so at 4x the reference's own discretisation error is ~0.4%
#: of the standard multiband grid's -- fine for ranking grids at or below
#: the multiband grid's density.
REFERENCE_REFINEMENT = 4

N_WAVEFORMS = 10
SEED = 20259

#: point counts for the geometric / inv-f^5/3 sweeps
N_SWEEP = [1000, 3000, 10000, 30000, 100000]

#: decimate the (huge) reference grid by this stride for the mismatch
#: integral only -- the L-inf phase / amplitude errors stay on the full grid
MM_STRIDE = 4

GREEDY_TRAIN_WAVEFORMS = 30
GREEDY_VAL_WAVEFORMS = 20
TOL_PHI = 3e-4  # GreedyDownsamplingTraining default
TOL_AMP = 8e-4


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            if not getattr(s, "closed", False):
                s.write(data)

    def flush(self):
        for s in self.streams:
            if not getattr(s, "closed", False):
                s.flush()


@contextmanager
def refined_seglen(factor: float):
    """Temporarily scale the multibanding seglen so grids come out denser."""
    original = multibanding.SEGLEN_20_HZ
    multibanding.SEGLEN_20_HZ = original * factor
    try:
        yield
    finally:
        multibanding.SEGLEN_20_HZ = original


def build_candidate_grids(f0: float, fmax: float, fpivot: float) -> dict:
    """Candidate frequency grids over ``[f0, fmax]`` (Hz), keyed by name."""
    grids: dict[str, np.ndarray] = {}

    multiband = reduced_frequency_array(f0, fmax, fpivot, mode=COMMON_GRID_MODE)
    multiband = np.asarray(multiband)
    grids["multiband"] = multiband
    n_mb = len(multiband)

    def _finish(g: np.ndarray) -> np.ndarray:
        g = np.unique(np.concatenate([[f0], np.asarray(g, dtype=float), [fmax]]))
        return g[(g >= f0) & (g <= fmax)]

    for n in N_SWEEP:
        grids[f"geom_{n}"] = _finish(np.geomspace(f0, fmax, n))
        x = np.linspace(f0 ** (-5 / 3), fmax ** (-5 / 3), n)
        grids[f"invf53_{n}"] = _finish(np.sort(x ** (-3 / 5)))

    # inv-f^5/3 and geometric matched to the multiband point count
    x = np.linspace(f0 ** (-5 / 3), fmax ** (-5 / 3), n_mb)
    grids["invf53_matched"] = _finish(np.sort(x ** (-3 / 5)))
    grids["geom_matched"] = _finish(np.geomspace(f0, fmax, n_mb))

    # hybrid: geometric below the pivot, uniform (seglen-spaced) above it
    df_pivot = 1.0 / multibanding.seglen_from_freq(fpivot, mode=COMMON_GRID_MODE)
    for n_low in (2000, 5000, 10000):
        low = np.geomspace(f0, fpivot, n_low)
        high = np.arange(fpivot, fmax + df_pivot, df_pivot)
        grids[f"hybrid_{n_low}"] = _finish(np.concatenate([low, high]))

    return grids


def eob_modes(generator, params, freqs_hz, dataset, modes):
    """``{mode: (amp, phase)}`` on ``freqs_hz`` from one batched EOB call."""
    freqs_nat = dataset.hz_to_natural_units(np.asarray(freqs_hz, dtype=float))
    out = generator.all_modes_amplitude_phase(params, modes, freqs_nat)
    result = {}
    for mode in modes:
        _, amp, phase = out[mode]
        if len(amp) != len(freqs_nat) or not (
            np.all(np.isfinite(amp)) and np.all(np.isfinite(phase))
        ):
            raise RuntimeError("EOB call returned bad data")
        result[mode] = (np.asarray(amp), np.asarray(phase))
    return result


def load_psd(frequencies_hz: np.ndarray) -> np.ndarray:
    data = np.loadtxt("mlgw_bns/data/ET_psd.txt")
    log_psd = np.interp(
        np.log(frequencies_hz), np.log(data[:, 0]), np.log(data[:, 1])
    )
    return np.exp(log_psd)


def mismatch(h1, h2, frequencies, psd_values, max_delta_t=0.05):
    """1 - overlap, phase marginalised analytically, time shift numerically.

    Same definition as ``ValidateModel.mismatch``.
    """

    def product(a, b):
        return abs(np.trapezoid(np.conj(a) * b / psd_values, x=frequencies))

    norm = np.sqrt(product(h1, h1) * product(h2, h2))

    def to_minimize(t_c):
        offset = np.exp(2j * np.pi * frequencies * t_c)
        return -product(h1, h2 * offset)

    res = minimize_scalar(
        to_minimize,
        method="bounded",
        bounds=(-max_delta_t, max_delta_t),
        options={"maxiter": 1000, "xatol": 1e-13},
    )
    return 1 - (-res.fun) / norm


def main() -> None:
    out = open("explore_multiband_grid.txt", "w")
    sys.stdout = Tee(sys.__stdout__, out)

    print("Building reference model / dataset ...")
    model = Model(
        modes=MODES,
        filename="explore_multiband_grid_tmp",
        initial_frequency_hz=INITIAL_FREQUENCY_HZ,
        reference_amplitude=True,
    )
    mode_model = model.mode_models[Mode(2, 2)]
    dataset = mode_model.dataset
    generator = mode_model.waveform_generator

    f0 = dataset.effective_initial_frequency_hz
    fmax = dataset.effective_srate_hz / 2
    fpivot = dataset.f_pivot_hz
    print(
        f"  effective band: {f0:.3f} - {fmax:.1f} Hz, pivot {fpivot:.1f} Hz, "
        f"mass_sum_seconds {dataset.mass_sum_seconds:.6e}"
    )

    with refined_seglen(REFERENCE_REFINEMENT):
        ref_grid = np.asarray(
            reduced_frequency_array(f0, fmax, fpivot, mode=COMMON_GRID_MODE)
        )
    ref_grid = ref_grid[(ref_grid >= f0) & (ref_grid <= fmax)]
    ref_nat = dataset.hz_to_natural_units(ref_grid)
    psd_ref = load_psd(ref_grid)
    mm_sl = slice(5, -5, MM_STRIDE)  # decimated + edge-trimmed, for the mismatch integral
    mm_freqs = ref_grid[mm_sl]
    mm_psd = psd_ref[mm_sl]
    print(f"  reference grid ({REFERENCE_REFINEMENT}x): {len(ref_grid)} points "
          f"({len(mm_freqs)} used for the mismatch integral)")

    grids = build_candidate_grids(f0, fmax, fpivot)
    n_multiband = len(grids["multiband"])
    print(f"  multiband grid: {n_multiband} points")
    print("  candidate grids:")
    for name, g in grids.items():
        print(f"    {name:16s} {len(g):8d} points  ({len(g) / n_multiband:6.3f} x multiband)")

    param_generator = dataset.make_parameter_generator(seed=SEED)
    params_list = [next(param_generator) for _ in range(N_WAVEFORMS)]

    # ---- Part A: grid faithfulness -------------------------------------
    print("\n" + "=" * 78)
    print("PART A -- can a cubic spline through the candidate grid reproduce the")
    print("         EOB waveform? (mismatch vs 4x-refined multiband reference)")
    print("=" * 78)

    # results[grid][mode] -> list of (mismatch, max_phase_err, med_phase_err,
    #                                 max_amp_fracerr)
    results: dict = {name: {m: [] for m in MODES} for name in grids}
    # per-frequency accumulators for the spectrum plot (matched grids only)
    spectrum_grids = ["multiband", "geom_matched", "invf53_matched",
                      "geom_10000", "invf53_10000", "hybrid_5000"]
    phase_err_spec: dict = {n: {m: [] for m in MODES} for n in spectrum_grids}
    amp_err_spec: dict = {n: {m: [] for m in MODES} for n in spectrum_grids}

    t_start = time.time()
    for i, params in enumerate(params_list):
        try:
            ref_modes = eob_modes(generator, params, ref_grid, dataset, MODES)
        except Exception as exc:  # pragma: no cover
            print(f"  waveform {i}: reference EOB failed ({exc}); skipping")
            continue

        ref_cart = {
            m: a * np.exp(1j * p) for m, (a, p) in ref_modes.items()
        }

        for name, grid in grids.items():
            try:
                cand_modes = eob_modes(generator, params, grid, dataset, MODES)
            except Exception:
                continue
            grid_nat = dataset.hz_to_natural_units(grid)
            for m in MODES:
                a_c, p_c = cand_modes[m]
                a_r, p_r = ref_modes[m]
                a_up = DownsamplingTraining.resample(grid_nat, ref_nat, a_c)
                p_up = DownsamplingTraining.resample(grid_nat, ref_nat, p_c)

                h_r = ref_cart[m][mm_sl]
                h_c = (a_up[mm_sl] * np.exp(1j * p_up[mm_sl]))
                mm = mismatch(h_r, h_c, mm_freqs, mm_psd)

                edge = slice(5, -5)
                phase_err = np.abs(p_up - p_r)[edge]
                amp_frac = np.abs(a_up - a_r)[edge] / (np.abs(a_r)[edge] + 1e-300)
                results[name][m].append(
                    (mm, phase_err.max(), np.median(phase_err), np.nanmedian(amp_frac))
                )
                if name in spectrum_grids:
                    phase_err_spec[name][m].append(np.abs(p_up - p_r))
                    amp_err_spec[name][m].append(
                        np.abs(a_up - a_r) / (np.abs(a_r) + 1e-300)
                    )
        print(f"  waveform {i + 1}/{len(params_list)} done "
              f"({time.time() - t_start:.0f} s elapsed)")

    _print_part_a_tables(results, grids, n_multiband)
    _plot_part_a(results, grids, n_multiband)
    _plot_spectrum(ref_grid, phase_err_spec, amp_err_spec, spectrum_grids)

    # ---- Part B: greedy downsampling on top of each candidate grid -----
    print("\n" + "=" * 78)
    print("PART B -- greedy phase downsampling on top of each candidate grid:")
    print(f"         node count and L-inf phase error on {GREEDY_VAL_WAVEFORMS} fresh")
    print(f"         waveforms (tol_phi = {TOL_PHI:g})")
    print("=" * 78)
    _run_part_b(generator, dataset, ref_grid, ref_nat, grids, n_multiband)

    print("\nDone. Figures: explore_multiband_grid_mismatch.png, "
          "explore_multiband_grid_spectrum.png")
    sys.stdout = sys.__stdout__
    out.close()


def _agg(rows):
    """median & p90 of a list of tuples -> arrays."""
    arr = np.array(rows)  # (n_wave, 4)
    if len(arr) == 0:
        return None
    return {
        "mm_med": np.median(arr[:, 0]),
        "mm_p90": np.percentile(arr[:, 0], 90),
        "ph_max": np.median(arr[:, 1]),
        "ph_med": np.median(arr[:, 2]),
        "amp_med": np.median(arr[:, 3]),
    }


def _print_part_a_tables(results, grids, n_multiband):
    for m in REPORT_MODES:
        print(f"\n  --- mode ({m.l},{m.m}) ---")
        print(f"  {'grid':16s} {'points':>8s} {'x_mb':>6s} "
              f"{'mm_med':>10s} {'mm_p90':>10s} "
              f"{'ph_max[rad]':>12s} {'ph_med[rad]':>12s} {'amp_med':>10s}")
        for name, grid in grids.items():
            a = _agg(results[name][m])
            if a is None:
                continue
            print(f"  {name:16s} {len(grid):8d} {len(grid) / n_multiband:6.3f} "
                  f"{a['mm_med']:10.2e} {a['mm_p90']:10.2e} "
                  f"{a['ph_max']:12.2e} {a['ph_med']:12.2e} {a['amp_med']:10.2e}")


def _plot_part_a(results, grids, n_multiband):
    fig, axes = plt.subplots(2, len(REPORT_MODES), figsize=(6 * len(REPORT_MODES), 9),
                             squeeze=False)
    families = {
        "geom": ("geom_", "tab:blue", "o"),
        "invf53": ("invf53_", "tab:green", "s"),
        "hybrid": ("hybrid_", "tab:orange", "^"),
    }
    for col, m in enumerate(REPORT_MODES):
        ax_mm, ax_ph = axes[0][col], axes[1][col]
        for fam, (prefix, color, marker) in families.items():
            pts, mm, ph = [], [], []
            for name, grid in grids.items():
                if not name.startswith(prefix) or "matched" in name:
                    continue
                a = _agg(results[name][m])
                if a is None:
                    continue
                pts.append(len(grid))
                mm.append(a["mm_med"])
                ph.append(a["ph_max"])
            order = np.argsort(pts)
            pts = np.array(pts)[order]
            ax_mm.plot(pts, np.array(mm)[order], marker=marker, color=color, label=fam)
            ax_ph.plot(pts, np.array(ph)[order], marker=marker, color=color, label=fam)

        mb = _agg(results["multiband"][m])
        for ax, key in ((ax_mm, "mm_med"), (ax_ph, "ph_max")):
            ax.axhline(mb[key], color="k", ls="--", lw=1,
                       label=f"multiband ({n_multiband} pts)")
            ax.axvline(n_multiband, color="k", ls=":", lw=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize="small")
        ax_mm.set_title(rf"$(\ell,m)=({m.l},{m.m})$")
        ax_mm.set_ylabel("median mismatch vs reference")
        ax_ph.set_ylabel(r"median of per-waveform max $|\Delta\phi|$ [rad]")
        ax_ph.set_xlabel("grid point count")
    fig.suptitle("Grid faithfulness: cubic-spline reconstruction error vs point count")
    fig.tight_layout()
    fig.savefig("explore_multiband_grid_mismatch.png", dpi=150)
    print("\n  saved explore_multiband_grid_mismatch.png")


def _plot_spectrum(ref_grid, phase_err_spec, amp_err_spec, spectrum_grids):
    fig, axes = plt.subplots(2, len(REPORT_MODES), figsize=(6 * len(REPORT_MODES), 9),
                             squeeze=False)
    for col, m in enumerate(REPORT_MODES):
        ax_ph, ax_amp = axes[0][col], axes[1][col]
        for name in spectrum_grids:
            if not phase_err_spec[name][m]:
                continue
            ph = np.median(np.array(phase_err_spec[name][m]), axis=0)
            amp = np.median(np.array(amp_err_spec[name][m]), axis=0)
            ax_ph.loglog(ref_grid, ph, lw=1, label=name)
            ax_amp.loglog(ref_grid, amp, lw=1, label=name)
        ax_ph.set_title(rf"$(\ell,m)=({m.l},{m.m})$")
        ax_ph.set_ylabel(r"median $|\Delta\phi(f)|$ [rad]")
        ax_amp.set_ylabel(r"median fractional $|\Delta A(f)| / A$")
        ax_amp.set_xlabel("$f$ [Hz]")
        for ax in (ax_ph, ax_amp):
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize="small")
    fig.suptitle("Where each grid loses accuracy (reconstruction error vs frequency)")
    fig.tight_layout()
    fig.savefig("explore_multiband_grid_spectrum.png", dpi=150)
    print("  saved explore_multiband_grid_spectrum.png")


def _run_part_b(generator, dataset, ref_grid, ref_nat, grids, n_multiband):
    greedy = GreedyDownsamplingTraining(dataset, max_phi_gap_ratio=None)

    train_pg = dataset.make_parameter_generator(seed=SEED + 1)
    val_pg = dataset.make_parameter_generator(seed=SEED + 2)
    train_params = [next(train_pg) for _ in range(GREEDY_TRAIN_WAVEFORMS)]
    val_params = [next(val_pg) for _ in range(GREEDY_VAL_WAVEFORMS)]

    subset = {
        k: grids[k]
        for k in ["multiband", "geom_matched", "invf53_matched",
                  "geom_10000", "invf53_10000", "geom_20000", "invf53_20000",
                  "hybrid_5000"]
        if k in grids
    }

    # reference greedy: train & validate on the dense reference grid
    print("\n  reference greedy (dense grid) ...")
    ref_train = {m: [] for m in REPORT_MODES}
    for p in train_params:
        mm = eob_modes(generator, p, ref_grid, dataset, MODES)
        for m in REPORT_MODES:
            ref_train[m].append(mm[m][1])
    ref_val = {m: [] for m in REPORT_MODES}
    for p in val_params:
        mm = eob_modes(generator, p, ref_grid, dataset, MODES)
        for m in REPORT_MODES:
            ref_val[m].append(mm[m][1])

    ref_nodes = {}
    for m in REPORT_MODES:
        idx = greedy.find_indices(ref_grid, ref_train[m], tol=TOL_PHI)
        ref_nodes[m] = ref_grid[idx]
        errs = [
            np.max(np.abs(v - DownsamplingTraining.resample(ref_grid[idx], ref_grid, v[idx])))
            for v in ref_val[m]
        ]
        print(f"    ({m.l},{m.m}): {len(idx):4d} nodes, "
              f"val L-inf phase median {np.median(errs):.2e} rad")

    print(f"\n  {'grid':16s} {'points':>8s} | "
          + " | ".join(f"({m.l},{m.m}) nodes  Linf[rad]" for m in REPORT_MODES))
    for name, grid in subset.items():
        grid_nat = dataset.hz_to_natural_units(grid)
        row = f"  {name:16s} {len(grid):8d} | "
        for m in REPORT_MODES:
            train_phases = [
                DownsamplingTraining.resample(ref_nat, grid_nat, v) for v in ref_train[m]
            ]
            idx = greedy.find_indices(grid, train_phases, tol=TOL_PHI)
            # validate against the dense reference truth
            errs = []
            for v_ref in ref_val[m]:
                v_grid = DownsamplingTraining.resample(ref_nat, grid_nat, v_ref)
                recon = DownsamplingTraining.resample(
                    grid[idx], ref_grid, v_grid[idx]
                )
                errs.append(np.max(np.abs(v_ref - recon)))
            row += f"{len(idx):6d}  {np.median(errs):9.2e}  | "
        print(row)


if __name__ == "__main__":
    main()
