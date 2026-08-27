"""Generate and cache the expensive part of a `ModeModel` training run.

Waveform generation dominates the cost of building a surrogate, but every
question about *how* the residuals are reduced and regressed --- the PCA
preprocessing, the scaling of the principal components, the choice of
regressor --- is downstream of it. This module generates one training set
and one validation set per mode, once, and stores them so that those
questions can be answered in seconds rather than minutes.

What is cached, per mode:

* the downsampling nodes (trained on a small set of waveforms),
* the *raw* residuals for the training set: ``A_eob / A_pn`` and
  ``phi_eob - phi_pn`` at those nodes, before any linear-trend removal,
  so that the time-shift handling can itself be varied,
* the same for a validation set drawn from a different seed,
* the EOB amplitude and phase of the validation waveforms, which are the
  ground truth the mismatches are computed against,
* the PN amplitude and phase at the nodes for both sets, needed to
  recompose a predicted residual back into a waveform.

Run with: python experiments/cache.py [--modes 22,21,33] [--n-train 8192]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np

from mlgw_bns.data_management import DownsamplingIndices, ParameterRanges, Residuals
from mlgw_bns.dataset_generation import ParameterSet
from mlgw_bns.higher_order_modes import Mode, teob_mode_generator_factory
from mlgw_bns.mode_model import ModeModel

#: Where the caches are written. Kept out of the repository: these are
#: hundreds of megabytes of regenerable intermediate data.
CACHE_DIR = Path(
    os.environ.get(
        "MLGW_EXPERIMENT_CACHE",
        "/tmp/claude-1000/-home-jacopo-Documents-masters-mlgw-bns/"
        "31a8f396-1902-43f7-978f-0cbf888155c6/scratchpad/cache",
    )
)

#: Matches the shipped `default_hom` model.
INITIAL_FREQUENCY_HZ = 5.0
SRATE_HZ = 4096.0

#: Seeds for the two parameter draws. `Dataset.generate_residuals` uses
#: seed 2 internally when no generator is set, so the training seed is
#: kept at 2 to reproduce the shipped model's draw exactly.
TRAIN_SEED = 2
VALIDATION_SEED = 1729


def cache_path(mode: Mode, n_train: int, sampling: str = "uniform") -> Path:
    suffix = "" if sampling == "uniform" else f"_{sampling}"
    return CACHE_DIR / f"cache_l{mode.l}_m{mode.m}_n{n_train}{suffix}.npz"


def make_mode_model(mode: Mode) -> ModeModel:
    """A `ModeModel` wired up the way `Model` wires up its per-mode models."""
    return ModeModel(
        filename=None,
        initial_frequency_hz=INITIAL_FREQUENCY_HZ,
        srate_hz=SRATE_HZ,
        mode=mode,
        waveform_generator=teob_mode_generator_factory(mode),
        parameter_ranges=ParameterRanges(),
    )


def build(
    mode: Mode,
    n_train: int,
    n_validation: int,
    n_downsampling: int,
    sampling: str = "uniform",
) -> Path:
    """Generate one cache.

    ``sampling`` selects how the *training* parameters are drawn --- the
    validation set is always the same i.i.d. uniform draw, so that two
    training strategies are scored against identical waveforms.
    """
    path = cache_path(mode, n_train, sampling)
    if path.exists():
        logging.info("Cache already present at %s", path)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    model = make_mode_model(mode)
    dataset = model.dataset

    logging.info("[%s] training the downsampling on %i waveforms", mode, n_downsampling)
    indices = model.downsampling_training.train(n_downsampling)

    logging.info(
        "[%s] generating %i training residuals (%s sampling)", mode, n_train, sampling
    )
    if sampling == "sobol":
        from .sampling import SobolParameterGenerator

        dataset.parameter_generator = SobolParameterGenerator(
            dataset, model.parameter_ranges, seed=TRAIN_SEED, n_points=n_train
        )
    else:
        dataset.parameter_generator = dataset.make_parameter_generator(TRAIN_SEED)
    frequencies_natural, train_params, train_residuals = dataset.generate_residuals(
        n_train, indices, flatten_phase=False, n_jobs=8
    )

    logging.info("[%s] generating %i validation residuals", mode, n_validation)
    dataset.parameter_generator = dataset.make_parameter_generator(VALIDATION_SEED)
    _, validation_params, validation_residuals = dataset.generate_residuals(
        n_validation, indices, flatten_phase=False, n_jobs=8
    )
    dataset.parameter_generator = None

    # The PN baseline at the nodes, so that a residual can be turned back
    # into a waveform without re-entering the waveform generator.
    def pn_baseline(params: ParameterSet):
        waveform_params = params.waveform_parameters(dataset)
        amplitudes = np.array(
            [
                dataset.waveform_generator.post_newtonian_amplitude(
                    par, dataset.frequencies[indices.amplitude_indices]
                )
                for par in waveform_params
            ]
        )
        phases = np.array(
            [
                dataset.waveform_generator.post_newtonian_phase(
                    par, dataset.frequencies[indices.phase_indices]
                )
                for par in waveform_params
            ]
        )
        return amplitudes, phases

    logging.info("[%s] evaluating the PN baseline at the nodes", mode)
    train_pn_amplitudes, train_pn_phases = pn_baseline(train_params)
    validation_pn_amplitudes, validation_pn_phases = pn_baseline(validation_params)

    np.savez_compressed(
        path,
        amplitude_indices=np.asarray(indices.amplitude_indices),
        phase_indices=np.asarray(indices.phase_indices),
        frequencies_natural=np.asarray(frequencies_natural),
        train_parameters=train_params.parameter_array,
        train_amplitude_residuals=train_residuals.amplitude_residuals,
        train_phase_residuals=train_residuals.phase_residuals,
        train_pn_amplitudes=train_pn_amplitudes,
        train_pn_phases=train_pn_phases,
        validation_parameters=validation_params.parameter_array,
        validation_amplitude_residuals=validation_residuals.amplitude_residuals,
        validation_phase_residuals=validation_residuals.phase_residuals,
        validation_pn_amplitudes=validation_pn_amplitudes,
        validation_pn_phases=validation_pn_phases,
    )
    logging.info("[%s] wrote %s (%.1f MB)", mode, path, path.stat().st_size / 1e6)
    return path


def load(mode: Mode, n_train: int, sampling: str = "uniform") -> dict:
    """Load a cache into a plain dict of float64 arrays.

    Everything is promoted to float64 here: `Dataset.generate_residuals`
    stores float32, which is only ~7 digits and so sits uncomfortably
    close to the dynamic range the PCA of these residuals spans.
    Promoting on load makes it possible to ask whether that matters
    without regenerating anything.
    """
    path = cache_path(mode, n_train, sampling)
    if not path.exists():
        raise FileNotFoundError(f"No cache at {path}; run experiments/cache.py first.")
    with np.load(path) as data:
        out = {key: data[key] for key in data.files}
    for key, value in out.items():
        if value.dtype == np.float32:
            out[key] = value.astype(np.float64)
    return out


def downsampling_indices(cache: dict) -> DownsamplingIndices:
    return DownsamplingIndices(
        list(cache["amplitude_indices"]), list(cache["phase_indices"])
    )


MODE_BY_NAME = {"22": Mode(2, 2), "21": Mode(2, 1), "33": Mode(3, 3), "44": Mode(4, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="22")
    parser.add_argument("--n-train", type=int, default=8192)
    parser.add_argument("--n-validation", type=int, default=256)
    parser.add_argument("--n-downsampling", type=int, default=256)
    parser.add_argument("--sampling", default="uniform", choices=["uniform", "sobol"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    for name in args.modes.split(","):
        build(
            MODE_BY_NAME[name.strip()],
            args.n_train,
            args.n_validation,
            args.n_downsampling,
            args.sampling,
        )


if __name__ == "__main__":
    main()
