"""Plot the space of `KernelRidgeNetwork` hyperparameters explored by
`HyperparameterOptimization`, for all modes at once.

Loads each mode's optuna study from `optimization_dataset_<l><m>_study.pkl`
(saved next to the training dataset by `optimize_n_hours.py`) and scatters
every trial's `(kernel_gamma, kernel_alpha)` on log-log axes, one color per
mode. Marker size encodes the objective (log10 of the validation residual
difference, lower is better): bigger markers are better trials, scaled
relative to the best and worst trial seen across all modes. The best trial
per mode --- the one written to `kernel_ridge_defaults.json` by
`HyperparameterOptimization.save_best_as_default` --- is outlined in black.

Run with: python visualization/plot_krr_hyperparams.py
"""

from pathlib import Path

import joblib  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

MODES = ["21", "22", "33", "44"]
COLORS = {
    "21": "#4c72b0",
    "22": "#dd8452",
    "33": "#55a868",
    "44": "#c44e52",
}

MIN_MARKER_SIZE = 20
MAX_MARKER_SIZE = 400

if __name__ == "__main__":

    per_mode_trials = {}
    all_values = []

    for mode in MODES:
        study_file = REPOSITORY_ROOT / f"optimization_dataset_{mode}_study.pkl"
        if not study_file.exists():
            continue

        study = joblib.load(study_file)
        trials = [t for t in study.trials if t.value is not None]

        gammas = np.array([t.params["kernel_gamma"] for t in trials])
        alphas = np.array([t.params["kernel_alpha"] for t in trials])
        values = np.array([t.value for t in trials])

        per_mode_trials[mode] = (gammas, alphas, values)
        all_values.append(values)

    all_values = np.concatenate(all_values)
    worst, best = all_values.max(), all_values.min()

    def sizes_from_values(values: np.ndarray) -> np.ndarray:
        # values are log10(loss); lower is better, so flip before scaling.
        normalized = (worst - values) / (worst - best)
        return MIN_MARKER_SIZE + normalized * (MAX_MARKER_SIZE - MIN_MARKER_SIZE)

    fig, ax = plt.subplots(figsize=(7, 6))

    for mode, (gammas, alphas, values) in per_mode_trials.items():
        sizes = sizes_from_values(values)

        ax.scatter(
            gammas,
            alphas,
            s=sizes,
            color=COLORS[mode],
            alpha=0.7,
            edgecolors="none",
            label=f"({mode[0]}, {mode[1]})",
        )

        best_index = np.argmin(values)
        ax.scatter(
            gammas[best_index],
            alphas[best_index],
            s=sizes[best_index],
            facecolors=COLORS[mode],
            edgecolors="black",
            linewidths=1.5,
            zorder=10,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\gamma$ (kernel_gamma)")
    ax.set_ylabel(r"$\alpha$ (kernel_alpha)")
    ax.set_title(
        "KernelRidgeNetwork hyperparameter trials\n"
        "(marker size: bigger = lower validation loss; outlined: best per mode)"
    )
    legend = ax.legend(title="mode $(l, m)$")

    fig.tight_layout()
    fig.savefig(REPOSITORY_ROOT / "visualization" / "krr_hyperparams.pdf")
    plt.show()
