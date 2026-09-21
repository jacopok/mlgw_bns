"""Per-mode TEOBResumS/PN training residuals for a trained HOM ``Model``.

One EOB call per parameter point (all modes at once, parallelised) on the
model's own frequency grid, downsampled to each mode's phase indices, so
the fourth row is exactly what the PCA and the network are trained on.

Rows
----
1. amplitude residual A_eob / A_pn(theta)               (own PN divisor)
2. amplitude residual A_eob / A_pn(theta_ref)           (reference_amplitude)
3. phase residual phi_eob - phi_pn, re-anchored to 0 at f0 (the raw
   residual carries the ~1e5-1e6 rad arg H_lm(f0) constant)
4. the exact training target: ``remove_linear_trend`` with the model's
   shared time-shift predictor and (for the HOM) its ``ModePhasesNN``

Run: python visualization/plot_teob_pn_residuals.py [--n N] [--model BASE]
"""

import argparse

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import Model
from mlgw_bns.principal_component_analysis import remove_linear_trend

MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--model", default="mlgw_bns/data/default_hom")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = Model(modes=MODES, filename=args.model)
    model.load()
    assert model.time_shifts_predictor is not None
    assert model.mode_phases_predictor is not None

    dataset = model.mode_models[Mode(2, 2)].dataset
    pgen = dataset.make_parameter_generator(seed=args.seed)
    params_list = [next(pgen) for _ in range(args.n)]

    ds_idx = {m: model.mode_models[m].downsampling_indices for m in MODES}

    # One batched EOB call per parameter point for all four modes at once
    # (all at initial_frequency factor 0.5) -- exactly the convention the
    # batched Model.generate trains on, so row 4 is the real training target.
    parameter_array_full, amp_res_full, phi_res_full = model._multimode_mode_residuals(
        params_list,
        dataset.frequencies,
        downsampling_indices_by_mode=ds_idx,
        amplitude_reference_by_mode={
            m: model.mode_models[m].dataset.amplitude_reference_parameters for m in MODES
        },
    )
    param_array = np.asarray(parameter_array_full, dtype=float)
    amp_res = {m: np.asarray(amp_res_full[m], float) for m in MODES}
    phi_res = {m: np.asarray(phi_res_full[m], float) for m in MODES}
    param_set = dataset.parameter_set_cls(param_array)
    print(f"{len(param_array)}/{args.n} valid")

    cmap = matplotlib.colormaps["viridis"]
    q_min, q_max = dataset.parameter_ranges.q_range
    colors = [cmap((q - q_min) / (q_max - q_min)) for q in param_array[:, 0]]

    fig, axes = plt.subplots(4, len(MODES), figsize=(16, 12), sharex=True, squeeze=False)

    for i, mode in enumerate(MODES):
        phi_idx = ds_idx[mode].phase_indices
        f_nat = dataset.frequencies[phi_idx]
        f_hz = dataset.frequencies_hz[phi_idx]
        amp_f_nat = dataset.frequencies[ds_idx[mode].amplitude_indices]

        a = amp_res[mode]                    # A_eob / A_pn(ref) if amp_ref set
        phi = phi_res[mode]                  # phi_eob - phi_pn (keeps arg H_lm)

        target = remove_linear_trend(
            parameters=param_set,
            phi_diff=phi,
            frq=f_hz,
            timeshifts_predictor=model.time_shifts_predictor,
            subtract_mode_phase_anchor=(mode != Mode(2, 2)),
            mode_phases_predictor=model.mode_phases_predictor,
            mode_index=model.modes.index(mode) if mode != Mode(2, 2) else None,
        )

        for j in range(len(param_array)):
            c = colors[j]
            axes[0, i].plot(amp_f_nat, a[j], color=c, alpha=0.5, lw=0.7)
            axes[1, i].plot(amp_f_nat, a[j], color=c, alpha=0.5, lw=0.7)
            axes[2, i].plot(f_nat, phi[j] - phi[j, 0], color=c, alpha=0.5, lw=0.7)
            axes[3, i].plot(f_nat, target[j], color=c, alpha=0.5, lw=0.7)

        axes[0, i].set_title(rf"$(\ell,m)=({mode.l},{mode.m})$")
        axes[3, i].set_xlabel(r"$Mf$")
        tp90 = np.quantile(np.abs(target).max(1), 0.9)
        axes[3, i].set_title(f"target |max| p90 = {tp90:.3f} rad", fontsize=9)

    axes[0, 0].set_ylabel(r"$A_{\rm EOB}/A_{\rm PN}$")
    axes[1, 0].set_ylabel(r"$A_{\rm EOB}/A_{\rm PN}$ (same, zoom)")
    axes[2, 0].set_ylabel(r"$\phi_{\rm EOB}-\phi_{\rm PN}$, re-anchored [rad]")
    axes[3, 0].set_ylabel("training target [rad]")

    for row in axes:
        for ax in row:
            ax.grid(True)
            ax.set_xscale("log")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=q_min, vmax=q_max))
    fig.colorbar(sm, ax=axes, label="Mass ratio $q$", pad=0.01)
    fig.suptitle(
        f"Per-mode training residuals, {len(param_array)} waveforms ({args.model})"
    )
    out = "teob_pn_residuals_21_22_33_44.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
