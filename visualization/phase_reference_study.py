"""How much the starting phases of the modes vary, and how well a trained
model finds the merger.

Usage::

    python visualization/phase_reference_study.py anchors N [SEED]
    python visualization/phase_reference_study.py merger MODEL_BASENAME N [SEED]

``anchors`` runs TEOBResumS (the public release: tidal deformabilities
capped at 5000) for ``N`` binaries and prints, per mode, the spread of the
phase residual (EOB minus post-Newtonian) at the first frequency of the
band in two references:

* TEOBResumS's own, with the time and the orbital phase set at the start
  of its integration (``raw``);
* the one the model is trained in, with every mode referenced to the
  (2,2) of the same waveform at that frequency
  (:func:`~mlgw_bns.data_management.re_reference`).

In the second the starting phases are constants, up to the difference
between the EOB and post-Newtonian phases at orbital frequencies
``f0/m`` and ``f0/2``, which is what makes a regressor for them
unnecessary.

``merger`` compares, for ``N`` binaries, the merger time and coalescence
phase of a trained model (the tangent to its (2,2) phase at the top of
the band, :meth:`~mlgw_bns.mode_model.ModeModel.merger_reference`) with
those of TEOBResumS, from the same tangent to its own (2,2) phase.
"""

import logging
import os
import sys
import warnings

import numpy as np

from mlgw_bns.data_management import ParameterRanges, re_reference, reference_gauge
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.model import DEFAULT_MODES, Model


def parameters(model, n, seed):
    dataset = model.dataset
    generator = dataset.parameter_generator_class(
        parameter_ranges=ParameterRanges(lambda1_range=(5.0, 5000.0), lambda2_range=(5.0, 5000.0)),
        dataset=dataset,
        seed=seed,
    )
    return [next(generator) for _ in range(n)]


def anchors(n, seed):
    model = Model(modes=list(DEFAULT_MODES), initial_frequency_hz=5.0, reference_amplitude=True)
    dataset = model.dataset
    # a coarse grid is enough: only the start of the band matters here
    grid = np.geomspace(dataset.frequencies[0], dataset.frequencies[-1], 400)
    generator = model.mode_models[Mode(2, 2)].waveform_generator
    raw = {mode: [] for mode in model.modes}
    for params in parameters(model, n, seed):
        waveforms = generator.all_modes_amplitude_phase(params, model.modes, grid)
        for mode in model.modes:
            f, _, phase = waveforms[mode]
            pn = model.mode_models[mode].waveform_generator.post_newtonian_phase(params, f)
            raw[mode].append(phase - pn)
    value, slope = reference_gauge(grid, np.array(raw[Mode(2, 2)]))
    print(f"{'mode':>6} {'raw std':>10} {'raw range':>10} | {'f0 mean':>9} {'f0 std':>9} {'f0 range':>9}")
    for mode in model.modes:
        residual = np.array(raw[mode])
        referenced = re_reference(grid, residual, mode.m, value, slope)[:, 0]
        print(
            f"({mode.l},{mode.m}) {residual[:, 0].std():10.3e} {np.ptp(residual[:, 0]):10.3e} | "
            f"{referenced.mean():+9.4f} {referenced.std():9.2e} {np.ptp(referenced):9.2e}"
        )


def merger(basename, n, seed):
    model = Model(modes=list(DEFAULT_MODES), filename=basename)
    model.load()
    reference = model.mode_models[Mode(2, 2)]
    dataset = reference.dataset
    top = dataset.frequencies[reference.downsampling_indices.phase_indices[-1]]
    # natural units, from the start of the band to its top, where the two
    # last points give the tangent
    grid = np.append(
        np.geomspace(dataset.frequencies[0], top * (1 - 1e-3), 400), top
    )
    generator = reference.waveform_generator
    times, phases = [], []
    for params in parameters(model, n, seed):
        # Both in the reference the model is trained in (the (2,2) matched
        # to its post-Newtonian phase, with the slope, at f0): the
        # surrogate's tangent at the top of the band ...
        slope, intercept = reference.merger_reference(params)
        slope = slope / dataset.mass_sum_seconds  # per natural frequency unit
        # ... and TEOBResumS's
        # all the modes, as in training: see `initial_frequency_scaling`
        _, _, phase = generator.all_modes_amplitude_phase(params, model.modes, grid)[
            Mode(2, 2)
        ]
        pn = generator.post_newtonian_phase(params, grid)
        value, residual_slope = reference_gauge(grid, phase - pn)
        referenced = pn + re_reference(grid, phase - pn, 2, value, residual_slope)
        eob_slope = (referenced[-1] - referenced[-2]) / (grid[-1] - grid[-2])
        eob_intercept = referenced[-1] - eob_slope * top
        # t = -(1 / 2 pi) d phase / d f
        times.append(-(slope - eob_slope) / (2 * np.pi) * dataset.mass_sum_seconds)
        phases.append((intercept - eob_intercept) / 2)
    times, phases = np.array(times), np.array(phases)
    print(
        f"merger time, surrogate - TEOBResumS: median {np.median(times):+.2e} s, "
        f"90% within {np.percentile(np.abs(times), 90):.2e} s "
        f"(at {dataset.total_mass} Msun)"
    )
    print(
        f"coalescence phase, surrogate - TEOBResumS: median {np.median(phases):+.3f} rad, "
        f"90% within {np.percentile(np.abs(phases), 90):.3f} rad"
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if sys.argv[1] == "anchors":
        anchors(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    else:
        merger(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 3)
