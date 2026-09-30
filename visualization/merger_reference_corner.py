"""Where in parameter space does the merger reference go wrong?

Draws binaries uniformly over the default model's training ranges (total
mass 2.8, the dataset reference, as in
``tests/test_model.py::test_default_model_merger_reference_matches_teobresums``)
and, for each, computes

- the unmaximised mismatch of the summed multi-mode waveform against
  TEOBResumS (:meth:`ValidateModel.unmaximised_mismatch`);
- the error of the model's merger time against TEOBResumS's, both the
  tangent to the (2,2) phase at the top node
  (:meth:`ValidateModel.merger_reference_errors`).

Two corner plots over (q, Lambda_1, Lambda_2, chi_1, chi_2) are colored
by either. The data are cached in ``merger_reference_corner.npz``; pass
``--recompute`` to regenerate them.

Run with: python visualization/merger_reference_corner.py [-n 1000]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from tqdm import tqdm

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.mode_model import ParametersWithExtrinsic
from mlgw_bns.model import Model
from mlgw_bns.model_validation import ValidateModel

HERE = Path(__file__).resolve().parent
PARAM_LABELS = [r"$q$", r"$\Lambda_1$", r"$\Lambda_2$", r"$\chi_1$", r"$\chi_2$"]
# Single-hue sequential ramp, starting from a visible light step so the
# small values do not vanish against the background.
CMAP = LinearSegmentedColormap.from_list(
    "sequential_blue", plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256))
)


def compute(n: int, seed: int) -> dict:
    model = Model.default_for_testing()
    validator = ValidateModel(model.mode_models[Mode(2, 2)])

    time_errors, phase_errors, param_set = validator.merger_reference_errors(
        validator.param_set(n, seed=seed)
    )

    mismatches = []
    for intrinsic in tqdm(param_set.waveform_parameters(model.dataset), desc="mismatches"):
        params = ParametersWithExtrinsic(
            mass_ratio=intrinsic.mass_ratio,
            lambda_1=intrinsic.lambda_1,
            lambda_2=intrinsic.lambda_2,
            chi_1=intrinsic.chi_1,
            chi_2=intrinsic.chi_2,
            distance_mpc=100.0,
            inclination=1.0,
            total_mass=2.8,
        )
        predicted = sum(model.predict_modes_dict(validator.frequencies, params).values())
        true = sum(model.get_teob_modes_dict(validator.frequencies, params).values())
        mismatches.append(validator.unmaximised_mismatch(true, predicted))

    return dict(
        parameters=param_set.parameter_array,
        mismatches=np.array(mismatches),
        time_errors=time_errors,
        phase_errors=phase_errors,
    )


def corner_plot(samples, values, norm, label, title, outfile):
    n_params = samples.shape[1]
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    # Worst points last, so they are drawn on top.
    order = np.argsort(values)
    samples, values = samples[order], values[order]

    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]
            if col > row:
                ax.axis("off")
                continue

            if row == col:
                # Median of the colored quantity in bins of this parameter.
                edges = np.histogram_bin_edges(samples[:, col], bins=12)
                index = np.clip(np.digitize(samples[:, col], edges) - 1, 0, len(edges) - 2)
                centers = (edges[1:] + edges[:-1]) / 2
                for q, style in ((0.5, "-"), (0.9, "--")):
                    stat = [
                        np.quantile(values[index == i], q) if np.any(index == i) else np.nan
                        for i in range(len(centers))
                    ]
                    ax.plot(centers, stat, style, color=CMAP(0.8), lw=2, label=f"{q:.0%} quantile")
                ax.set_yscale("log")
                ax.yaxis.tick_right()
            else:
                scatter = ax.scatter(
                    samples[:, col],
                    samples[:, row],
                    c=values,
                    cmap=CMAP,
                    norm=norm,
                    s=10,
                    linewidths=0,
                )

            if row == n_params - 1:
                ax.set_xlabel(PARAM_LABELS[col])
            else:
                ax.set_xticklabels([])
            if col == 0 and row != 0:
                ax.set_ylabel(PARAM_LABELS[row])
            elif col != row:
                ax.set_yticklabels([])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 2].legend(handles, labels, loc="center", frameon=False, title="diagonal: binned")
    fig.colorbar(scatter, ax=axes[0:2, 3:5], orientation="horizontal", label=label, fraction=0.3)
    fig.suptitle(title)
    fig.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=1000, help="number of binaries")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    cache = HERE / "merger_reference_corner.npz"
    if args.recompute or not cache.exists():
        np.savez(cache, **compute(args.n, args.seed))
    data = np.load(cache)
    samples = data["parameters"]
    mismatches = data["mismatches"]
    abs_time_errors = np.abs(data["time_errors"]) * 1e6

    print(
        f"{len(mismatches)} binaries; unmaximised mismatch median {np.median(mismatches):.2e}, "
        f"max {np.max(mismatches):.2e}; |dt| median {np.median(abs_time_errors):.1f} us, "
        f"99% {np.quantile(abs_time_errors, 0.99):.1f} us, max {np.max(abs_time_errors):.1f} us"
    )
    for i in np.argsort(abs_time_errors)[::-1][:8]:
        q, l1, l2, c1, c2 = samples[i]
        print(
            f"  q={q:.3f} L1={l1:6.0f} L2={l2:6.0f} chi1={c1:+.3f} chi2={c2:+.3f}: "
            f"|dt|={abs_time_errors[i]:6.1f} us, mismatch {mismatches[i]:.1e}"
        )

    corner_plot(
        samples,
        mismatches,
        LogNorm(vmin=np.quantile(mismatches, 0.02), vmax=np.max(mismatches)),
        "unmaximised mismatch against TEOBResumS",
        "Unmaximised summed-waveform mismatch (M = 2.8, inclination 1)",
        HERE / "merger_reference_corner_mismatch.png",
    )
    corner_plot(
        samples,
        abs_time_errors,
        LogNorm(vmin=np.quantile(abs_time_errors, 0.02), vmax=np.max(abs_time_errors)),
        r"$|\Delta t_{\rm merger}|$ [$\mu$s]",
        "Merger-time error of the (2,2) reference against TEOBResumS (M = 2.8)",
        HERE / "merger_reference_corner_time_error.png",
    )


if __name__ == "__main__":
    main()
