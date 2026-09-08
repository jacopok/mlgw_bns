r"""Multi-mode mismatch against TEOBResumS as a function of total mass.

The surrogate is trained at one reference total mass
(``dataset.total_mass``, 2.8 Msun); :meth:`~mlgw_bns.model.Model.predict`
serves any other mass by rescaling the frequency axis. Below the
reference the rescaled grid dips under
``effective_initial_frequency_hz`` and the post-Newtonian low-frequency
extension switches on --- so the total-mass axis is exactly where a bug
in that extension shows up.

For a set of binaries this sweeps the total mass across the training
range and plots, against the per-mode TEOBResumS ground truth on the
model grid (no frequency-domain TEOBResumS call), both the full
higher-order-mode mismatch and the :math:`(2,2)`-only mismatch, each
with the per-mode time/phase marginalisation of
:meth:`~mlgw_bns.model_validation.ValidateModel.full_waveform_mismatch`.

A correct implementation gives two smooth curves that rise gently with
mass (fewer in-band cycles at low mass, so easier to model); the fixed
`predict_amplitude_phase` extension does. The version before commit
528c828 put a ~1000x step in the full-HOM curve exactly at 2.8 Msun,
from the extension overwriting each mode's inter-mode phase constant.

Run with: python visualization/mismatch_vs_total_mass.py
"""

from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.mode_model import ParametersWithExtrinsic

logging.basicConfig(level=logging.WARNING)

MODES = [Mode(2, 2), Mode(2, 1), Mode(3, 3), Mode(4, 4)]
N_BINARIES = 8
N_MASSES = 25
TOTAL_MASS_RANGE = (2.0, 4.0)
BAND_LO, BAND_HI = 20.0, 2048.0
SEED = 11

FIGURE_PATH = "visualization/mismatch_vs_total_mass.png"


def sweep(model: Model, n_binaries: int, n_masses: int):
    validator = ValidateModel(model.mode_models[Mode(2, 2)])
    frequencies = validator.frequencies
    band = (frequencies >= BAND_LO) & (frequencies <= BAND_HI)
    fb = frequencies[band]
    masses = np.linspace(*TOTAL_MASS_RANGE, n_masses)

    generator = model.dataset.make_parameter_generator(SEED)
    rng = np.random.default_rng(SEED)

    full = np.full((n_binaries, n_masses), np.nan)
    only22 = np.full((n_binaries, n_masses), np.nan)

    for b in range(n_binaries):
        intrinsic = next(generator)
        inclination = np.arccos(rng.uniform(-1.0, 1.0))
        for i, mass in enumerate(masses):
            params = ParametersWithExtrinsic(
                mass_ratio=intrinsic.mass_ratio,
                lambda_1=intrinsic.lambda_1,
                lambda_2=intrinsic.lambda_2,
                chi_1=intrinsic.chi_1,
                chi_2=intrinsic.chi_2,
                distance_mpc=100.0,
                inclination=inclination,
                total_mass=float(mass),
            )
            surrogate = model.predict_modes_dict(frequencies, params)
            truth = model.get_teob_modes_dict(frequencies, params)
            full[b, i] = validator.full_waveform_mismatch(
                {k: v[band] for k, v in truth.items()},
                {k: v[band] for k, v in surrogate.items()},
                frequencies=fb,
            )
            only22[b, i] = validator.full_waveform_mismatch(
                {(2, 2): truth[(2, 2)][band]},
                {(2, 2): surrogate[(2, 2)][band]},
                frequencies=fb,
            )
        print(f"  binary {b + 1}/{n_binaries} "
              f"(q={intrinsic.mass_ratio:.2f}): "
              f"full-HOM {np.nanmin(full[b]):.1e}--{np.nanmax(full[b]):.1e}",
              flush=True)

    return masses, full, only22


def plot(masses, full, only22, reference_mass, baseline=None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    panels = (
        (axes[0], full, baseline["full"] if baseline else None,
         "full higher-order-mode waveform"),
        (axes[1], only22, baseline["only22"] if baseline else None,
         r"$(2,2)$ mode only"),
    )
    for ax, data, base, title in panels:
        if base is not None:
            for row in base:
                ax.plot(masses, row, lw=1, alpha=0.3, color="tab:red")
            ax.plot(masses, np.nanmedian(base, axis=0), lw=2.5, color="tab:red",
                    label="before fix")
        for row in data:
            ax.plot(masses, row, lw=1, alpha=0.45, color="tab:blue")
        ax.plot(masses, np.nanmedian(data, axis=0), lw=2.5, color="tab:blue",
                label="median" if base is None else "after fix")
        ax.axvline(reference_mass, color="black", ls="--", lw=1,
                   label=f"dataset reference ({reference_mass} $M_\\odot$)")
        ax.set_yscale("log")
        ax.set_xlabel(r"total mass [$M_\odot$]")
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylabel("mismatch vs TEOBResumS (per-mode marginalised)")

    fig.suptitle("Surrogate mismatch across the total-mass axis")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"figure written to {FIGURE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-binaries", type=int, default=N_BINARIES)
    parser.add_argument("--n-masses", type=int, default=N_MASSES)
    parser.add_argument("--baseline", type=str, default=None,
                        help="a .npz from an earlier run to overlay (e.g. a "
                             "pre-fix checkout), for a before/after figure")
    args = parser.parse_args()

    model = Model(modes=MODES, filename="mlgw_bns/data/default_hom")
    model.load()

    masses, full, only22 = sweep(model, args.n_binaries, args.n_masses)
    np.savez("visualization/mismatch_vs_total_mass.npz",
             masses=masses, full=full, only22=only22)
    baseline = np.load(args.baseline) if args.baseline else None
    plot(masses, full, only22, model.dataset.total_mass, baseline)


if __name__ == "__main__":
    main()
