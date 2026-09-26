"""Compare kernel-ridge regularizations on freshly generated EOB data.

Usage::

    python visualization/kernel_regularization_study.py generate DIR N [SEED]
    python visualization/kernel_regularization_study.py compare DIR N_TRAIN N_VAL MODES [VARIANTS]

``generate`` runs TEOBResumS (the public release: tidal deformabilities
capped at 5000, since it fails on some higher ones) on the packaged
model's own downsampling nodes, for all of its modes, and saves the raw
residuals in batches of 256 under ``DIR``.

``compare`` builds the regression targets as :meth:`Model.generate
<mlgw_bns.model.Model.generate>` does --- with the packaged PCA bases,
and with time-shift / mode-phase predictors retrained on this data, since
the public TEOBResumS aligns the merger differently from the version the
packaged ones were trained on --- then fits
:class:`~mlgw_bns.neural_network.KernelRidgeNetwork` with each variant:

* ``fixed``: the packaged per-mode ``kernel_alpha``, shared by all
  components;
* ``loo0``: one penalty per component, by leave-one-out error only;
* ``loo``: the same, plus the rounding-error term (the default).

It reports held-out errors in residual space (for odd-``m`` modes also
weighted by mode power, which is what those fits minimize) and the
spread of the predicted phase between a batch and single-row
evaluations. ``MODES`` is e.g. ``22,44,21,33``.

The result on 8192 waveforms (7168 training, 1024 held out) is in the
CHANGELOG. The absolute errors are well above the packaged model's,
trained on three times the data: only the comparison between the
variants is meaningful.
"""

import dataclasses
import glob
import logging
import os
import sys
import time
import warnings

import numpy as np

from mlgw_bns.data_management import ParameterRanges, Residuals
from mlgw_bns.dataset_generation import ParameterSet
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.mode_model import mode_power_weights
from mlgw_bns.model import Model
from mlgw_bns.neural_network import Hyperparameters, ModePhasesNN, TimeshiftsNN
from mlgw_bns.principal_component_analysis import remove_linear_trend

BATCH = 256


def generate(directory, total, seed):
    model = Model.default_for_testing()
    dataset = model.dataset
    ranges = ParameterRanges(lambda1_range=(5.0, 5000.0), lambda2_range=(5.0, 5000.0))
    generator = dataset.parameter_generator_class(
        parameter_ranges=ranges, dataset=dataset, seed=seed
    )
    indices = {mode: model.mode_models[mode].downsampling_indices for mode in model.modes}
    references = {
        mode: model.mode_models[mode].dataset.amplitude_reference_parameters
        for mode in model.modes
    }
    for batch in range(total // BATCH):
        params = [next(generator) for _ in range(BATCH)]
        filename = os.path.join(directory, f"seed{seed}_batch{batch:03d}.npz")
        if os.path.exists(filename):
            continue
        array, amplitudes, phases = model._multimode_mode_residuals(
            params, dataset.frequencies, indices, references, n_jobs=os.cpu_count()
        )
        np.savez(
            filename,
            params=array,
            **{f"amp_{m.l}{m.m}": amplitudes[m] for m in model.modes},
            **{f"phase_{m.l}{m.m}": phases[m] for m in model.modes},
        )
        print("saved", filename, len(array), flush=True)


def phase_spread(network, mode_model, x):
    """Largest phase difference between a batch and single rows."""
    batch = mode_model.predict_residuals_bulk(ParameterSet(x), network).phase_residuals
    rows = np.vstack(
        [
            mode_model.predict_residuals_bulk(ParameterSet(x[i : i + 1]), network).phase_residuals
            for i in range(len(x))
        ]
    )
    return np.max(np.abs(batch - rows))


def compare(directory, n_train, n_val, modes, variants):
    data = [np.load(f) for f in sorted(glob.glob(os.path.join(directory, "*.npz")))]
    params = np.concatenate([d["params"] for d in data])
    if len(params) < n_train + n_val:
        raise ValueError(f"Only {len(params)} waveforms in {directory}")
    model = Model.default_for_testing()
    train = slice(0, n_train)
    validation = list(range(len(params) - n_val, len(params)))

    # The shared reference predictors, retrained on the training part.
    all_modes = [(m.l, m.m) for m in model.modes]
    reference = model.mode_models[Mode(2, 2)]
    phase_22 = np.concatenate([d["phase_22"] for d in data])[train]
    f_22 = reference.dataset.frequencies_hz[reference.downsampling_indices.phase_indices]
    time_shifts = TimeshiftsNN(
        training_params=params[train],
        training_timeshifts=Residuals(np.zeros_like(phase_22), phase_22).phase_timeshifts(
            frequencies=f_22
        ),
    ).fit()
    mode_phases = ModePhasesNN(
        modes=all_modes,
        f0_natural=float(model.dataset.frequencies[0]),
        training_params=params[train],
        training_mode_phases=np.stack(
            [np.concatenate([d[f"phase_{l}{m}"] for d in data])[train][:, 0] for l, m in all_modes],
            axis=1,
        ),
    ).fit()

    for mode in [Mode(int(k[0]), int(k[1])) for k in modes.split(",")]:
        key = f"{mode.l}{mode.m}"
        mode_model = model.mode_models[mode]
        indices = mode_model.downsampling_indices
        amplitude = np.concatenate([d[f"amp_{key}"] for d in data])
        phase = remove_linear_trend(
            ParameterSet(params),
            np.concatenate([d[f"phase_{key}"] for d in data]),
            mode_model.dataset.frequencies_hz[indices.phase_indices],
            time_shifts,
            True,
            mode_phases,
            all_modes.index((mode.l, mode.m)),
        )
        residuals = Residuals(amplitude.astype(np.float32), phase)
        mode_model.training_dataset = residuals[list(range(n_train))]
        mode_model.training_parameters = ParameterSet(params[train])
        mode_model._reduced_residuals.cache_clear()
        truth = residuals[validation]

        base = Hyperparameters.default_kernel_ridge(n_train, mode=mode)
        for variant in variants:
            hyper = dataclasses.replace(
                base,
                kernel_alpha_selection="fixed" if variant == "fixed" else "loo",
                kernel_rounding_factor=0.0 if variant == "loo0" else 0.1,
            )
            start = time.perf_counter()
            network = mode_model.train_nn(hyper)
            fit_s = time.perf_counter() - start

            predicted = mode_model.predict_residuals_bulk(ParameterSet(params[validation]), network)
            dphi = predicted.phase_residuals - truth.phase_residuals
            damp = predicted.amplitude_residuals - truth.amplitude_residuals
            print(
                f"{key} {variant:5s} fit {fit_s:4.0f}s | phase rms {np.sqrt(np.mean(dphi**2)):.3e} "
                f"p99 {np.percentile(np.abs(dphi), 99):.3e} max {np.abs(dphi).max():.3e} | "
                f"amp rms {np.sqrt(np.mean(damp**2)):.3e} | batch-vs-row phase spread "
                f"{phase_spread(network, mode_model, params[validation][:48]):.1e} | "
                f"max|dual| {np.abs(network.regressor.dual_coef_).max():.1e}",
                flush=True,
            )
            if mode.m % 2:
                pn = mode_model.dataset.waveform_generator.post_newtonian_amplitude(
                    mode_model.dataset.amplitude_reference_parameters,
                    mode_model.dataset.frequencies[indices.amplitude_indices],
                )
                weights = mode_power_weights(
                    truth.amplitude_residuals,
                    mode_model.dataset.frequencies_hz[indices.amplitude_indices],
                    pn_amplitude=pn,
                )
                print(
                    f"      power-weighted: phase rms "
                    f"{np.sqrt(np.average(np.mean(dphi**2, axis=1), weights=weights)):.3e}, "
                    f"amp rms {np.sqrt(np.average(np.mean(damp**2, axis=1), weights=weights)):.3e}",
                    flush=True,
                )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    if sys.argv[1] == "generate":
        os.makedirs(sys.argv[2], exist_ok=True)
        generate(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 11)
    else:
        compare(
            sys.argv[2],
            int(sys.argv[3]),
            int(sys.argv[4]),
            sys.argv[5],
            sys.argv[6].split(",") if len(sys.argv) > 6 else ["fixed", "loo0", "loo"],
        )
