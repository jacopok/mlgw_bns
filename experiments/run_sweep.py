"""Score a set of pipeline variants against the production baseline.

The sweep is staged rather than a grid. First the *floors*: how well the
retained principal components can represent the validation residuals at
all, which no amount of regression can beat. Then one-at-a-time changes
from the production configuration, so that each effect is attributable.
Then the combination of whatever won.

Run with: python -m experiments.run_sweep --stage floors
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from joblib import Parallel, delayed  # type: ignore

from mlgw_bns.higher_order_modes import Mode

from .evaluate import Experiment, Scores
from .pipeline import Config

#: One :class:`Experiment` per worker process, built on first use: it
#: holds the cached residuals and the resampled ground-truth waveforms,
#: which are the same for every variant.
_EXPERIMENTS: dict[tuple, Experiment] = {}


def score_one(
    mode: Mode,
    n_train: int,
    config: Config,
    projection_only: bool,
    sampling: str = "uniform",
):
    """Train and score one variant. Runs in a worker process."""
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    logging.basicConfig(level=logging.ERROR)

    key = (mode, n_train, sampling)
    if key not in _EXPERIMENTS:
        _EXPERIMENTS[key] = Experiment(mode, n_train, sampling)

    start = time.time()
    try:
        scores = _EXPERIMENTS[key].score(config, projection_only=projection_only)
    except Exception as error:  # noqa: BLE001 - a failed variant is a result
        return {
            "label": config.label() + " FAILED",
            "median": float("inf"),
            "percentile_90": float("inf"),
            "worst": float("inf"),
            "error": repr(error),
            "seconds": time.time() - start,
            "mismatches": [],
        }
    return {
        "label": scores.label,
        "median": scores.median,
        "percentile_90": scores.percentile_90,
        "worst": scores.worst,
        "seconds": time.time() - start,
        "mismatches": scores.mismatches.tolist(),
    }

MODE_BY_NAME = {"22": Mode(2, 2), "21": Mode(2, 1), "33": Mode(3, 3), "44": Mode(4, 4)}

RESULTS_PATH = Path(__file__).parent / "results"


def floors(n_train: int) -> list[tuple[Config, bool]]:
    """Truncation floors: what the retained components can represent.

    Scored by projecting the true validation residuals onto the basis and
    reconstructing, with no regressor in the loop. This is the ceiling on
    the accuracy of everything downstream.
    """
    jobs: list[tuple[Config, bool]] = []
    for n_components in (10, 20, 30, 60, 100):
        jobs.append((Config(n_components=n_components), True))
        jobs.append((Config(n_components=n_components, weighting="mismatch"), True))
        jobs.append((Config(n_components=n_components, detrend="wproject"), True))
        jobs.append(
            (
                Config(
                    n_components=n_components,
                    weighting="mismatch",
                    detrend="wproject",
                ),
                True,
            )
        )
    jobs.append((Config(dtype="float32"), True))
    jobs.append(
        (
            Config(dtype="float32", weighting="mismatch", detrend="wproject"),
            True,
        )
    )
    return jobs


BEST = Config(
    weighting="mismatch",
    detrend="wproject",
    pc_scaling="flat",
    regressor="krr",
    n_components=60,
)


def replace_(config: Config, **changes) -> Config:
    from dataclasses import replace

    return replace(config, **changes)


def one_at_a_time(n_train: int) -> list[tuple[Config, bool]]:
    """Single departures from the production configuration."""
    variants = [
        Config(),
        # -- how much of the network's loss each component gets
        Config(pc_scaling="uniform"),
        Config(pc_exponent=0.5),
        Config(pc_scaling="robust"),
        # -- what the PCA is asked to optimize
        Config(weighting="block_std"),
        Config(weighting="mismatch"),
        Config(detrend="wproject"),
        Config(weighting="mismatch", detrend="wproject"),
        # -- the regressor
        Config(regressor="mlp_scaled"),
        Config(regressor="rff_ridge"),
        Config(regressor="krr"),
        Config(regressor_kwargs={"max_iter": 1000}),
        # -- structural
        Config(n_components=60),
        Config(dtype="float32"),
        Config(features="physical"),
    ]
    return [(config, False) for config in variants]


def combined(n_train: int) -> list[tuple[Config, bool]]:
    """The changes that helped, together, and each one taken back out."""
    return [
        (Config(), False),
        (BEST, False),
        (BEST, True),
        (replace_(BEST, regressor="mlp_scaled"), False),
        (replace_(BEST, regressor="mlp"), False),
        (replace_(BEST, regressor="rff_ridge"), False),
        (replace_(BEST, weighting="none"), False),
        (replace_(BEST, pc_scaling="eigen"), False),
        (replace_(BEST, detrend="predictor"), False),
        (replace_(BEST, n_components=30), False),
        (replace_(BEST, n_components=100), False),
        (replace_(BEST, features="physical"), False),
        (replace_(BEST, dtype="float32"), False),
    ]


def training_size_scan(n_train: int) -> list[tuple[Config, bool]]:
    """How each configuration pays off as the training budget grows.

    This is the curve that answers "how far can a <10k-waveform budget be
    pushed": the shipped pipeline, the same pipeline with the mini-batch
    slip repaired, and the tuned kernel, all as a function of the number
    of training waveforms. `PRODUCTION` and `TUNED` are defined below
    `BEST`, so the lookup happens at call time rather than at import time.
    """
    jobs = []
    for size in (512, 1024, 2048, 4096, n_train):
        jobs.append((replace_(PRODUCTION, n_train=size), False))
        jobs.append((replace_(Config(), n_train=size), False))
        jobs.append((replace_(TUNED, n_train=size), False))
    return jobs


def regressor_tuning(n_train: int) -> list[tuple[Config, bool]]:
    """Tune the parameters -> component-coefficients map.

    The truncation floor turns out to sit far below any accuracy the
    model reaches, so this map --- not the basis --- is what limits the
    surrogate. All of these share the preprocessing that produced the
    best floor, so that only the regressor varies.
    """
    base = replace_(BEST, regressor="krr")
    jobs = []
    for gamma in (0.02, 0.05, 0.1, 0.2, 0.5):
        for alpha in (1e-10, 1e-8, 1e-6):
            jobs.append(
                (
                    replace_(
                        base, regressor_kwargs={"gamma": gamma, "alpha": alpha}
                    ),
                    False,
                )
            )
    for gamma in (0.05, 0.1, 0.2):
        jobs.append(
            (
                replace_(
                    base,
                    features="physical",
                    regressor_kwargs={"gamma": gamma, "alpha": 1e-8},
                ),
                False,
            )
        )
    return jobs


def network_tuning(n_train: int) -> list[tuple[Config, bool]]:
    """Vary the network, keeping everything else at the best settings.

    The production hyperparameters were tuned by Optuna in 2022 against
    861 training waveforms and an unscaled 30-output target; none of
    those three things still holds.
    """
    base = replace_(BEST, regressor="mlp_global")
    jobs = [(base, False)]
    for layers in ((256, 256), (400, 200, 100), (128, 128, 128, 128)):
        jobs.append(
            (
                replace_(
                    base,
                    regressor_kwargs={
                        "hidden_layer_sizes": layers,
                        "max_iter": 2000,
                    },
                ),
                False,
            )
        )
    for activation in ("tanh",):
        jobs.append(
            (
                replace_(
                    base,
                    regressor_kwargs={
                        "activation": activation,
                        "hidden_layer_sizes": (256, 256),
                        "max_iter": 2000,
                    },
                ),
                False,
            )
        )
    for learning_rate in (1e-3, 3e-3):
        jobs.append(
            (
                replace_(
                    base,
                    regressor_kwargs={
                        "learning_rate_init": learning_rate,
                        "hidden_layer_sizes": (256, 256),
                        "max_iter": 2000,
                    },
                ),
                False,
            )
        )
    return jobs


#: Reproduces what the shipped networks were actually trained with.
#: `SklearnNetwork.fit` clips the mini-batch size to ``x_data.shape[1]``,
#: the number of *features*, rather than ``shape[0]``, the number of
#: samples --- so the configured 160 becomes 5, whatever the training set
#: size. The docstring's stated intent ("avoid scikit-learn's `batch_size`
#: larger than data warning") is about samples, so this is a slip, but it
#: is what every packaged model was fitted with and so it is the honest
#: baseline to measure against.
PRODUCTION = Config(regressor_kwargs={"batch_size": 5})


def production_baseline(n_train: int) -> list[tuple[Config, bool]]:
    """What the shipped pipeline scores, and what each fix is worth."""
    return [
        (PRODUCTION, False),
        (Config(), False),
        (replace_(PRODUCTION, pc_scaling="flat"), False),
        (Config(pc_scaling="flat"), False),
        (Config(pc_scaling="flat", regressor="mlp_global"), False),
        (BEST, False),
        (replace_(BEST, regressor="mlp_global"), False),
    ]


#: The best kernel found by `regressor_tuning`.
TUNED = replace_(BEST, regressor_kwargs={"gamma": 0.1, "alpha": 1e-10})


def per_mode(n_train: int) -> list[tuple[Config, bool]]:
    """A compact comparison, meant to be run for each mode in turn.

    For the (2,1) and (3,3) modes this is also the test of whether the
    amplitude sign change needs special handling: the mismatch weighting
    de-emphasises the nodes around the zero crossing on its own, because
    the mode carries no power there, so if it is going to help anywhere
    it will help here.
    """
    return [
        (PRODUCTION, False),
        (Config(), False),
        (Config(regressor="krr"), False),
        (replace_(TUNED, weighting="none"), False),
        (TUNED, False),
        (replace_(TUNED, amplitude="reference_ratio"), False),
        (replace_(TUNED, amplitude="reference_ratio", weighting="none"), False),
        (replace_(TUNED, amplitude="reference_ratio", pc_scaling="eigen"), False),
        (TUNED, True),
        (replace_(TUNED, amplitude="reference_ratio"), True),
    ]


def tail(n_train: int) -> list[tuple[Config, bool]]:
    """Trade median accuracy against the worst case.

    The worst validation mismatches all sit in the corners of the
    parameter box, where the kernel has training points on one side only.
    More regularisation tames that at the cost of the median; so, in
    principle, does a training design that puts points nearer the faces
    of the box, which is what `--sampling sobol` provides.
    """
    jobs = []
    for alpha in (1e-10, 1e-8, 1e-7, 1e-6):
        jobs.append((replace_(TUNED, regressor_kwargs={"gamma": 0.1, "alpha": alpha}), False))
    for gamma in (0.1, 0.2):
        jobs.append(
            (
                replace_(
                    TUNED,
                    regressor_kwargs={"gamma": gamma, "alpha": 1e-8},
                    n_components=100,
                ),
                False,
            )
        )
    return jobs


STAGES = {
    "floors": floors,
    "one_at_a_time": one_at_a_time,
    "combined": combined,
    "sizes": training_size_scan,
    "regressor_tuning": regressor_tuning,
    "network_tuning": network_tuning,
    "production_baseline": production_baseline,
    "per_mode": per_mode,
    "tail": tail,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="floors", choices=sorted(STAGES))
    parser.add_argument("--mode", default="22")
    parser.add_argument("--n-train", type=int, default=8192)
    parser.add_argument("--n-jobs", type=int, default=7)
    parser.add_argument("--sampling", default="uniform", choices=["uniform", "sobol"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    mode = MODE_BY_NAME[args.mode]
    jobs = STAGES[args.stage](args.n_train)

    # One variant per worker process, each single-threaded: the fits are
    # small matrix products that gain little from threading but a lot
    # from being run at the same time as thirteen others.
    scored = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(score_one)(
            mode, args.n_train, config, projection_only, args.sampling
        )
        for config, projection_only in jobs
    )

    results = []
    for entry in sorted(
        (entry for entry in scored if entry is not None), key=lambda e: e["median"]
    ):
        print(
            f"median {entry['median']:.3e}  p90 {entry['percentile_90']:.3e}  "
            f"worst {entry['worst']:.3e}   {entry['label']}   "
            f"[{entry['seconds']:.0f} s]",
            flush=True,
        )
        results.append(entry)

    RESULTS_PATH.mkdir(exist_ok=True)
    suffix = "" if args.sampling == "uniform" else f"_{args.sampling}"
    out = (
        RESULTS_PATH
        / f"{args.stage}_l{mode.l}_m{mode.m}_n{args.n_train}{suffix}.json"
    )
    out.write_text(json.dumps(results, indent=1))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
