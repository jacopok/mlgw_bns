r"""How many greedy downsampling nodes does the algorithm keep, as a function
of the number of training waveforms it sees?

The production pipeline (``make_default_dataset.py``) trains the greedy
downsampling on 2**10 = 1024 waveforms per mode. This script sweeps that
number and records, per mode, on the real multiband grid:

* the number of amplitude / phase nodes the greedy keeps
  (the real ``GreedyDownsamplingTraining.find_indices``, run from scratch
  on the first ``n`` waveforms at each checkpoint --- not an online
  approximation);
* the L-infinity reconstruction error of those node sets on a fixed
  held-out set of waveforms the greedy never saw.

Production tolerances: ``tol_amp = 8e-4``, ``tol_phi = 3e-4``, and the phase
nodes get the same ``max_phi_gap_ratio = 1.03`` gap fill as a HOM ModeModel.

Output
------
``downsampling_vs_ntrain.png`` / ``.txt``

Run with:  uv run python downsampling_vs_ntrain.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlgw_bns.downsampling_interpolation import (
    DownsamplingTraining,
    GreedyDownsamplingTraining,
)
from mlgw_bns.dataset_generation import ParameterSet
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import Model

from explore_multiband_grid import Tee

INITIAL_FREQUENCY_HZ = 5.0
MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]

N_MAX = 512
CHECKPOINTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
N_HELDOUT = 40

TOL_AMP = 8e-4
TOL_PHI = 3e-4
MAX_PHI_GAP_RATIO = 1.03
SEED = 88


def heldout_linf(x, node_idx, x_val, ys_val):
    return float(np.median([
        np.max(np.abs(v - DownsamplingTraining.resample(x[node_idx], x_val, v[node_idx])))
        for v in ys_val
    ]))


def main() -> None:
    out = open("downsampling_vs_ntrain.txt", "w")
    sys.stdout = Tee(sys.__stdout__, out)

    model = Model(
        modes=MODES,
        filename="downsampling_vs_ntrain_tmp",
        initial_frequency_hz=INITIAL_FREQUENCY_HZ,
        reference_amplitude=True,
    )

    results: dict = {}
    for mode in MODES:
        mm = model.mode_models[mode]
        ds = mm.dataset
        x = ds.frequencies
        greedy = GreedyDownsamplingTraining(
            ds, tol_amp=TOL_AMP, tol_phi=TOL_PHI, max_phi_gap_ratio=MAX_PHI_GAP_RATIO
        )
        print(f"\n=== mode ({mode.l},{mode.m}) : grid {len(x)} points ===")

        t0 = time.time()
        train_wf, _ = ds.generate_waveforms_from_params(
            ParameterSet.from_parameter_generator(
                ds.make_parameter_generator(seed=SEED), N_MAX
            ), n_jobs=16,
        )
        val_wf, _ = ds.generate_waveforms_from_params(
            ParameterSet.from_parameter_generator(
                ds.make_parameter_generator(seed=SEED + 1000), N_HELDOUT
            ), n_jobs=16,
        )
        amps, phis = list(train_wf.amplitudes), list(train_wf.phases)
        val_amps, val_phis = list(val_wf.amplitudes), list(val_wf.phases)
        print(f"  generated {N_MAX}+{N_HELDOUT} waveforms in {time.time() - t0:.0f} s")

        rec = {k: [] for k in
               ("n", "amp_nodes", "phi_nodes", "amp_linf", "phi_linf")}
        for n in CHECKPOINTS:
            t0 = time.time()
            amp_idx = greedy.find_indices(x, amps[:n], tol=TOL_AMP)
            phi_idx = greedy.fill_phi_gaps(
                x, greedy.find_indices(x, phis[:n], tol=TOL_PHI)
            )
            rec["n"].append(n)
            rec["amp_nodes"].append(len(amp_idx))
            rec["phi_nodes"].append(len(phi_idx))
            rec["amp_linf"].append(heldout_linf(x, amp_idx, x, val_amps))
            rec["phi_linf"].append(heldout_linf(x, phi_idx, x, val_phis))
            print(f"  n={n:4d}: amp {len(amp_idx):4d} nodes  phi {len(phi_idx):4d} nodes  "
                  f"| held-out Linf amp {rec['amp_linf'][-1]:.2e}  phi {rec['phi_linf'][-1]:.2e}  "
                  f"({time.time() - t0:.0f} s)")

        results[mode] = {k: np.array(v) for k, v in rec.items()}
        del train_wf, val_wf, amps, phis, val_amps, val_phis

    _plot(results)
    print("\nSaved downsampling_vs_ntrain.png")
    sys.stdout = sys.__stdout__
    out.close()


def _plot(results: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for mode, color in zip(MODES, colors):
        r = results[mode]
        lbl = f"({mode.l},{mode.m})"
        axes[0][0].plot(r["n"], r["amp_nodes"], "o-", color=color, label=lbl)
        axes[0][1].plot(r["n"], r["phi_nodes"], "o-", color=color, label=lbl)
        axes[1][0].plot(r["n"], r["amp_linf"], "o-", color=color, label=lbl)
        axes[1][1].plot(r["n"], r["phi_linf"], "o-", color=color, label=lbl)

    axes[0][0].set_title("amplitude nodes kept  (find_indices, from scratch)")
    axes[0][1].set_title("phase nodes kept  (with max_phi_gap_ratio fill)")
    axes[1][0].set_title(r"held-out $L_\infty$ amplitude error")
    axes[1][1].set_title(r"held-out $L_\infty$ phase error")
    axes[1][0].axhline(TOL_AMP, color="0.4", ls=":", label="tol_amp")
    axes[1][1].axhline(TOL_PHI, color="0.4", ls=":", label="tol_phi")

    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_xlabel("number of training waveforms")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize="small")
    for ax in axes[1]:
        ax.set_yscale("log")
    for ax in axes[0]:
        ax.set_ylabel("node count")

    fig.suptitle(
        "Greedy downsampling: nodes kept and held-out accuracy vs. "
        f"training-set size (multiband grid, {INITIAL_FREQUENCY_HZ} Hz; "
        "production uses 1024)")
    fig.tight_layout()
    fig.savefig("downsampling_vs_ntrain.png", dpi=150)


if __name__ == "__main__":
    main()
