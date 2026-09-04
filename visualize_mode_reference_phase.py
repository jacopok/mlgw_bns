"""Visualise the reference-phase training target of one mode.

For the mode picked with ``--mode`` (default ``4,4``) this draws the same
sweep ``ModePhasesNN`` trains on --- ``phi_lm(f0)`` for a parameter
sample --- subtracts the analytic backbone calibration ``a * M_lm + b``
(degree-1 fit, exactly as :meth:`ModePhasesNN.fit`), and plots the
leftover the ridge regressor actually has to learn against each intrinsic
parameter.

Run with::

    python visualize_mode_reference_phase.py --n 4000 --mode 4,4
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

from mlgw_bns.model import DEFAULT_MODES, Model
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.pn_modes import Mode as PNMode, reference_phase_backbone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PARAM_LABELS = [r"$q$", r"$\Lambda_1$", r"$\Lambda_2$", r"$\chi_1$", r"$\chi_2$"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--mode", type=str, default="4,4", help="l,m")
    parser.add_argument("--grid-points", type=int, default=64)
    parser.add_argument("--fmax-hz", type=float, default=512.0)
    parser.add_argument("--initial-frequency-hz", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--degree", type=int, default=1, help="backbone calibration degree")
    parser.add_argument("--gw-prior", action="store_true")
    parser.add_argument("--out", type=str, default="mode_reference_phase.png")
    args = parser.parse_args()

    l, m = (int(x) for x in args.mode.split(","))
    mode = Mode(l, m)

    kwargs = dict(
        modes=list(DEFAULT_MODES),
        initial_frequency_hz=args.initial_frequency_hz,
        reference_amplitude=True,
    )
    if args.gw_prior:
        from mlgw_bns.dataset_generation import GWPriorUniformSpinParameterGenerator

        kwargs["parameter_generator_class"] = GWPriorUniformSpinParameterGenerator
    model = Model(**kwargs)
    if mode not in model.modes:
        parser.error(f"mode {mode} not in {model.modes}")
    j = list(model.modes).index(mode)

    grid_hz, f_ref_natural, f0_natural = model._reference_grid(
        args.grid_points, args.fmax_hz
    )
    generator = model.mode_models[Mode(2, 2)].dataset.make_parameter_generator(
        seed=args.seed
    )
    p, _dt, phi = model._reference_sweep_targets(
        generator, args.n, grid_hz, f_ref_natural,
        batch_size=args.batch_size, progress_label=f"({l},{m}) sweep",
    )
    target = phi[:, j]

    from sklearn.linear_model import LinearRegression
    from mlgw_bns.neural_network import ModePhasesNN

    backbone = reference_phase_backbone(p, f0_natural, PNMode(l, m))

    # "old" calibration: degree-1 fit in the backbone scalar only.
    coeffs = np.polyfit(backbone, target, args.degree)
    fit = np.polyval(coeffs, backbone)
    leftover = target - fit

    # "new" (production) calibration: ModePhasesNN's linear backbone design.
    design = ModePhasesNN._backbone_design(p, (l, m), f0_natural)
    lr = LinearRegression().fit(design, target)
    leftover_params = target - lr.predict(design)

    print(f"\nmode ({l},{m}): {len(p)} points, seed {args.seed}")
    print(f"  target  phi_lm(f0):  span {target.min():.3e} .. {target.max():.3e} rad")
    print(f"  backbone M_lm:       span {backbone.min():.3e} .. {backbone.max():.3e}")
    print(
        f"  backbone-only (deg {args.degree}) leftover: std {leftover.std():.3f} rad, "
        f"|.| median {np.median(np.abs(leftover)):.3f}, 90th {np.percentile(np.abs(leftover), 90):.3f}"
    )
    print(
        f"  + linear in params:    leftover: std {leftover_params.std():.3f} rad, "
        f"|.| median {np.median(np.abs(leftover_params)):.3f}, "
        f"90th {np.percentile(np.abs(leftover_params), 90):.3f}"
    )
    print("  Pearson r of backbone-only leftover vs parameter:")
    for lab, col in zip(["q", "L1", "L2", "chi1", "chi2"], p.T):
        print(f"    {lab:5s} {np.corrcoef(leftover, col)[0, 1]:+.3f}")

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))

    ax = axes[0, 0]
    order = np.argsort(backbone)
    ax.scatter(backbone, target, s=6, alpha=0.3)
    ax.plot(backbone[order], fit[order], "r-", lw=1.5, label=f"degree-{args.degree} fit")
    ax.set_xlabel(r"backbone $M_{\ell m} = f_0\, d\Psi/df$")
    ax.set_ylabel(r"$\phi_{\ell m}(f_0)$ [rad]")
    ax.set_title("target vs analytic backbone")
    ax.legend()

    ax = axes[0, 1]
    ax.scatter(backbone, leftover, s=6, alpha=0.3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"backbone $M_{\ell m}$")
    ax.set_ylabel("leftover [rad]")
    ax.set_title("leftover vs backbone (what the ridge sees)")

    ax = axes[0, 2]
    ax.hist(leftover, bins=60)
    ax.set_xlabel("leftover [rad]")
    ax.set_title(f"leftover distribution (std {leftover.std():.3f} rad)")

    for k, (lab, col) in enumerate(zip(PARAM_LABELS, p.T)):
        ax = axes.flat[3 + k]
        ax.scatter(col, leftover, s=6, alpha=0.25, c="tab:blue", label="backbone only")
        ax.scatter(col, leftover_params, s=6, alpha=0.25, c="tab:orange",
                   label="+ linear in params")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(lab)
        ax.set_ylabel("leftover [rad]")
        if k == 0:
            ax.legend(markerscale=2, fontsize=8)

    ax = axes[2, 2]
    chi_eff = (p[:, 3] + p[:, 0] * p[:, 4]) / (1 + p[:, 0])
    sc = ax.scatter(p[:, 0], chi_eff, c=leftover, cmap="coolwarm", s=10)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$\chi_{\rm eff}$")
    ax.set_title("leftover on the $(q, \\chi_{\\rm eff})$ plane")
    fig.colorbar(sc, ax=ax, label="leftover [rad]")

    fig.suptitle(
        f"mode ({l},{m}) reference-phase training target, "
        f"{len(p)} points, {args.initial_frequency_hz:g} Hz"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
