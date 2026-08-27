r"""Characterise the sign change in the (2,1) and (3,3) amplitude residuals.

The EOB amplitude of these modes passes through zero inside the band ---
physically, a :math:`\pi` phase flip --- so the residual
:math:`r = A_{\rm eob} / A_{\rm pn}` changes sign. This asks three
questions of the cached training data:

* how many training waveforms are affected, and where the crossing sits;
* how large the residual gets near the crossing, which is what decides
  whether a handful of waveforms can dominate a max-based normalisation;
* how much of the mode's PSD-weighted power sits near the crossing,
  which is what decides whether any of it matters.

Run with: python -m experiments.amplitude_zeros --mode 21
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from mlgw_bns.higher_order_modes import Mode

from . import cache as cache_module
from .evaluate import Experiment
from .pipeline import Config

MODE_BY_NAME = {"22": Mode(2, 2), "21": Mode(2, 1), "33": Mode(3, 3), "44": Mode(4, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="21")
    parser.add_argument("--n-train", type=int, default=8192)
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    mode = MODE_BY_NAME[args.mode]
    experiment = Experiment(mode, args.n_train)

    ratio = experiment.train_amplitude_residuals
    frequencies = experiment.amplitude_frequencies_hz

    sign_changes = np.diff(np.sign(ratio), axis=1) != 0
    affected = sign_changes.any(axis=1)
    print(f"mode ({mode.l},{mode.m}), {len(ratio)} training waveforms")
    print(f"  sign change somewhere in band: {affected.sum()} ({affected.mean():.1%})")

    if affected.any():
        first_crossing = np.array(
            [frequencies[np.argmax(row)] for row in sign_changes[affected]]
        )
        print(
            "  crossing frequency [Hz]: "
            f"min {first_crossing.min():.1f}, median {np.median(first_crossing):.1f}, "
            f"max {first_crossing.max():.1f}"
        )

    print(
        f"  |ratio|: median {np.median(np.abs(ratio)):.3f}, "
        f"p99 {np.percentile(np.abs(ratio), 99):.3f}, max {np.abs(ratio).max():.3f}"
    )

    # How concentrated is a max-based normalisation? If one waveform sets
    # the maximum for a principal component, every other waveform's target
    # for that component is squashed towards zero.
    for label, config in (
        ("production", Config(n_components=30)),
        (
            "mismatch-weighted",
            Config(n_components=30, weighting="mismatch", detrend="wproject"),
        ),
    ):
        surrogate = experiment.make_surrogate(config, fit_regressor=False)
        data = surrogate
        coefficients = None
        # Recompute the coefficients the same way `fit` does.
        amplitude = experiment.train_amplitude_residuals
        phase, _ = surrogate._detrend(
            experiment.train_parameters, experiment.train_phase_residuals
        )
        combined = surrogate._combine(amplitude, phase)
        coefficients = (combined - surrogate.mean) @ surrogate.eigenvectors
        peak = np.max(np.abs(coefficients), axis=0)
        typical = np.std(coefficients, axis=0)
        print(
            f"  {label}: max/std of the PC coefficients --- "
            f"median {np.median(peak / typical):.1f}, worst {np.max(peak / typical):.1f}"
        )


if __name__ == "__main__":
    main()
