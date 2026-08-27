r"""The error the downsampling alone contributes.

Every mismatch in :mod:`experiments.run_sweep` compares two waveforms
that were both reconstructed from the same set of downsampling nodes, so
the spline interpolation between those nodes cancels out. That isolates
the reduction and regression stages, which is what the sweep is about ---
but it also hides a floor that a finished model cannot escape: the
difference between the true EOB waveform and its own node-resampled
version.

This measures that floor directly, by generating full-resolution EOB
waveforms and comparing each against its resampling from the nodes.

Run with: python -m experiments.downsampling_floor --mode 22 --n 32
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
from joblib import Parallel, delayed  # type: ignore

from mlgw_bns.data_management import FDWaveforms
from mlgw_bns.dataset_generation import ParameterSet
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.resample_residuals import cartesian_waveforms_at_frequencies

from .evaluate import Experiment

MODE_BY_NAME = {"22": Mode(2, 2), "21": Mode(2, 1), "33": Mode(3, 3), "44": Mode(4, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="22")
    parser.add_argument("--n-train", type=int, default=8192)
    parser.add_argument("--n", type=int, default=32)
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    mode = MODE_BY_NAME[args.mode]
    experiment = Experiment(mode, args.n_train)
    dataset = experiment.model.dataset
    indices = experiment.downsampling_indices

    parameters = ParameterSet(experiment.validation_parameters[: args.n])
    waveform_parameters = parameters.waveform_parameters(dataset)

    def full_resolution(par):
        _, amplitude, phase = dataset.waveform_generator.effective_one_body_waveform(
            par, dataset.frequencies
        )
        return amplitude, phase

    results = Parallel(n_jobs=8)(
        delayed(full_resolution)(par) for par in waveform_parameters
    )

    validator = experiment.validator
    natural = dataset.hz_to_natural_units(validator.frequencies)

    # The exact waveform on the validator grid, interpolated from the full
    # FFT grid, which is dense enough that this step is not the error.
    exact = np.array(
        [
            np.interp(natural, dataset.frequencies, amplitude)
            * np.exp(
                1j
                * (
                    np.interp(natural, dataset.frequencies, phase)
                    - np.interp(natural[0], dataset.frequencies, phase)
                )
            )
            for amplitude, phase in results
        ]
    )

    downsampled = FDWaveforms(
        np.array([amplitude[indices.amplitude_indices] for amplitude, _ in results]),
        np.array([phase[indices.phase_indices] for _, phase in results]),
    )
    downsampled.phases = downsampled.phases - downsampled.phases[:, :1]
    resampled = cartesian_waveforms_at_frequencies(
        downsampled,
        natural,
        dataset,
        experiment.model.downsampling_training,
        indices,
    )

    mismatches = np.array(
        [validator.mismatch(a, b) for a, b in zip(exact, resampled)]
    )
    print(
        f"mode ({mode.l},{mode.m}), {len(indices.amplitude_indices)} amplitude "
        f"and {len(indices.phase_indices)} phase nodes"
    )
    print(
        f"  downsampling-only mismatch: median {np.median(mismatches):.3e}, "
        f"p90 {np.percentile(mismatches, 90):.3e}, worst {mismatches.max():.3e}"
    )


if __name__ == "__main__":
    main()
