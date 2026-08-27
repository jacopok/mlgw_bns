"""What the accuracy gain costs at evaluation time.

The whole point of a surrogate is that it is fast, so a regressor that
buys accuracy with prediction time may not be a good trade. A kernel
ridge regression is the case to worry about: its prediction evaluates
one kernel per training point, so unlike a network its cost grows with
the training set. This measures the parameters-to-coefficients map on
its own --- not the resampling and recomposition around it, which every
variant shares --- for the configurations the sweeps compare.

Run with: python -m experiments.timing
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from mlgw_bns.higher_order_modes import Mode

from .evaluate import Experiment
from .pipeline import Config
from .run_sweep import PRODUCTION, TUNED


def time_prediction(surrogate, parameters: np.ndarray, repeats: int) -> float:
    """Seconds per waveform for the regression step, best of `repeats`."""
    from .pipeline import make_features

    features = make_features(parameters, surrogate.config.features)
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        surrogate.regressor_object.predict(features)
        best = min(best, time.perf_counter() - start)
    return best / len(parameters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    experiment = Experiment(Mode(2, 2), args.n_train)
    parameters = experiment.validation_parameters

    print(
        f"regression step only, {args.n_train} training waveforms, batches of "
        f"{len(parameters)}"
    )
    print(f"{'configuration':<20}  {'us / waveform':>14}  {'fit (s)':>9}")
    for label, config in (
        ("shipped (MLP)", PRODUCTION),
        ("tuned kernel", TUNED),
        ("random features", Config(regressor="rff_ridge")),
    ):
        start = time.perf_counter()
        surrogate = experiment.make_surrogate(config)
        fit_seconds = time.perf_counter() - start
        per_waveform = time_prediction(surrogate, parameters, args.repeats)
        print(f"{label:<20}  {per_waveform * 1e6:14.1f}  {fit_seconds:9.1f}")


if __name__ == "__main__":
    main()
