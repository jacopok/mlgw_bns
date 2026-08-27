r"""Two pictures of why the production surrogate stops where it does.

The first is about *which components of the residual the network is asked
to get right*. Principal-component coefficients are handed to the network
after being divided by their largest absolute value and multiplied by
``eigenvalue ** pc_exponent``. A regressor minimizing a plain mean
squared error over those scaled targets is minimizing
:math:`\sum_i (\Delta x_i / s_i)^2` while the error it actually causes in
the residual is :math:`\sum_i (\Delta x_i)^2`, so component :math:`i`
enters the loss with weight :math:`s_i^{-2}`. Since the largest
coefficient of a component scales like the square root of its eigenvalue,
and ``pc_exponent`` is about 0.02, that weight runs like
:math:`\lambda_i^{-0.96}` --- and the eigenvalues span ten orders of
magnitude.

The second is about *which parts of the frequency band the PCA is asked
to get right*. Concatenating the amplitude and phase residuals raw and
taking their PCA optimizes an unweighted L2 norm over the nodes, which
treats a radian of phase error at 5 Hz --- where the detector has no
sensitivity and the waveform has almost no power --- exactly like a
radian at 500 Hz.

Run with: python -m experiments.diagnose
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlgw_bns.higher_order_modes import Mode

from .evaluate import Experiment
from .pipeline import Config, mismatch_column_weights, trapezoid_weights

MODE_BY_NAME = {"22": Mode(2, 2), "21": Mode(2, 1), "33": Mode(3, 3), "44": Mode(4, 4)}
FIGURE_PATH = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="22")
    parser.add_argument("--n-train", type=int, default=8192)
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    mode = MODE_BY_NAME[args.mode]
    experiment = Experiment(mode, args.n_train)

    baseline = experiment.make_surrogate(Config(n_components=60), fit_regressor=False)
    improved = experiment.make_surrogate(
        Config(n_components=60, weighting="mismatch", detrend="wproject"),
        fit_regressor=False,
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # -- 1. the loss weighting across principal components ------------------
    eigenvalues = baseline.eigenvalues
    components = np.arange(1, len(eigenvalues) + 1)
    production_weight = 1 / baseline.pc_scale**2
    uniform = experiment.make_surrogate(
        Config(n_components=60, pc_scaling="uniform"), fit_regressor=False
    )

    axes[0].plot(
        components,
        production_weight / production_weight[0],
        "o-",
        markersize=3,
        label="production: $\\max|x_i|^{-2}\\lambda_i^{2\\alpha}$",
    )
    uniform_weight = 1 / uniform.pc_scale**2
    axes[0].plot(
        components,
        uniform_weight / uniform_weight[0] * (eigenvalues[0] / eigenvalues[0]),
        "s-",
        markersize=3,
        label="uniform: $\\sigma_i^{-2}$, i.e. flat in residual space",
    )
    axes[0].plot(
        components,
        eigenvalues / eigenvalues[0],
        "--",
        color="grey",
        label="eigenvalue $\\lambda_i / \\lambda_1$ (for scale)",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("principal component $i$")
    axes[0].set_ylabel("weight in the network's loss, relative to $i=1$")
    axes[0].set_title("what the network is asked to fit")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize="small")

    # -- 2. where the mismatch actually lives -------------------------------
    amplitude_weights = improved.amplitude_column_weights
    phase_weights = improved.phase_column_weights
    flat_amplitude = trapezoid_weights(experiment.amplitude_frequencies_hz)
    flat_phase = trapezoid_weights(experiment.phase_frequencies_hz)

    axes[1].plot(
        experiment.phase_frequencies_hz,
        phase_weights / phase_weights.max(),
        label="phase: $A^2/S_n$, what the mismatch weights by",
    )
    axes[1].plot(
        experiment.amplitude_frequencies_hz,
        amplitude_weights / amplitude_weights.max(),
        label="amplitude: $A_{\\rm pn}^2/S_n$",
    )
    axes[1].plot(
        experiment.phase_frequencies_hz,
        flat_phase / flat_phase.max(),
        "--",
        color="grey",
        label="what the production PCA weights by (flat per node)",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-8, 3)
    axes[1].set_xlabel("$f$ [Hz]")
    axes[1].set_ylabel("relative weight per node")
    axes[1].set_title("where the mismatch lives")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize="small")

    # -- 3. the truncation floor -------------------------------------------
    floors_file = FIGURE_PATH / f"floors_l{mode.l}_m{mode.m}_n{args.n_train}.json"
    if floors_file.exists():
        entries = json.loads(floors_file.read_text())
        series: dict[str, list[tuple[int, float]]] = {}
        for entry in entries:
            label = entry["label"]
            if "n_components" not in label:
                continue
            n = int(label.split("n_components=")[1].split(",")[0].split(" ")[0])
            key = label.replace(f"n_components={n}", "").replace(" [PCA floor]", "")
            key = key.strip().strip(",").strip() or "production"
            series.setdefault(key, []).append((n, entry["median"]))
        for key, points in sorted(series.items()):
            points.sort()
            axes[2].plot(
                [p[0] for p in points],
                [p[1] for p in points],
                "o-",
                markersize=4,
                label=key,
            )
        axes[2].set_yscale("log")
        axes[2].set_xlabel("principal components retained")
        axes[2].set_ylabel("median mismatch of the projection")
        axes[2].set_title("truncation floor: the best the basis can do")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize="small")
    else:
        axes[2].text(
            0.5,
            0.5,
            "run `--stage floors` first",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )

    figure.suptitle(
        f"mlgw_bns $({mode.l},{mode.m})$ surrogate, {args.n_train} training waveforms, "
        "5 Hz starting frequency"
    )
    figure.tight_layout()
    FIGURE_PATH.mkdir(exist_ok=True)
    out = FIGURE_PATH / f"diagnosis_l{mode.l}_m{mode.m}.png"
    figure.savefig(out, dpi=140)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
