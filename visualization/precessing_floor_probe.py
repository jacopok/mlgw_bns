"""Where does the ~3e-7--8e-7 floor of validate_model.precessing_mismatches
come from, when the aligned full-waveform mismatch is ~1e-8 and the floor
shows no trend with the opening angle beta?

For each seed, on validate_model's precessing grid (f >= 21 Hz), h_+ only,
optimised mismatch as in precessing_mismatches:

  aligned limit (in-plane spins zeroed, so beta = 0 and the model's
  precessing prediction is the aligned one):
    model vs TEOB precessing run, ODE start 10 Hz      (what validation does)
    model vs TEOB precessing run, ODE start in START_FREQUENCIES
    model vs TEOB, 10 Hz, srate_interp doubled again (validation uses the
    training data's, dataset.srate_hz * srate_interp_scaling(modes))
  precessing (the validation's own binary):
    model vs TEOB at 10 Hz and each of START_FREQUENCIES (training starts its ODE at 1.79 Hz) (spins given at 0.95 times that)
    JAX predictor vs predict_numpy                      (with --numpy; ~20 s each)

  --anchor: aligned and precessing rows also with the orbital-phase anchor optimised

Usage: precessing_floor_probe.py MODEL N_SEEDS [--numpy] [--anchor]
"""
import dataclasses
import sys

import numpy as np

import validate_model as vm
import mlgw_bns.dataset_generation as dg
from mlgw_bns.higher_order_modes import Mode, srate_interp_scaling
from mlgw_bns.model import MODELS_AVAILABLE, Model
from mlgw_bns.precessing_model import PrecessingModel

name, n_seeds = sys.argv[1], int(sys.argv[2])
use_numpy = "--numpy" in sys.argv
#: TEOBResumS ODE starts (Hz) besides validate_model's 10 Hz; spins given at 0.95 times each
START_FREQUENCIES = (5.0, 4.0, 3.0, 2.4, 1.8)
model = Model.default_for_testing(name) if name in MODELS_AVAILABLE else vm.load_model(name)
precessing = PrecessingModel(model)
validator = vm.ValidateModel(model.mode_models[Mode(2, 2)])
grid = validator.frequencies
grid = grid[(grid >= vm.PRECESSING_MIN_FREQUENCY_HZ) & (grid < grid[-1])]

# teob_precessing_polarizations imports eobrun at call time: wrap it to
# override TEOBResumS settings without duplicating the call
original_eobrun = dg.eobrun
overrides: dict = {}
dg.eobrun = lambda pars: original_eobrun({**pars, **overrides})


def teob_plus(params, reference_frequency_hz=vm.PRECESSING_REFERENCE_FREQUENCY_HZ, **settings):
    overrides.clear()
    overrides.update(settings)
    params = dataclasses.replace(params, reference_frequency_hz=reference_frequency_hz)
    return vm.teob_precessing_polarizations(model, precessing, params, grid)[0], params


def mismatch(a, b):
    return validator.mismatch(a, b, frequencies=grid, max_delta_t=1e-3)


def best_anchor_mismatch(truth, params):
    """The mismatch with the orbital-phase anchor (reference_phase) also
    optimised: what is left is the waveform's shape, not the anchor."""
    from scipy.optimize import minimize_scalar

    def at(delta):
        return mismatch(truth, precessing.predict(
            grid, dataclasses.replace(params, reference_phase=params.reference_phase + delta))[0])

    best = minimize_scalar(at, bounds=(-0.1, 0.1), method="bounded", options={"xatol": 1e-6})
    print(f"    anchor offset {best.x:+.2e} rad", flush=True)
    return best.fun


columns = {}
for seed in range(n_seeds):
    params = vm.random_precessing_parameters(model, seed)
    aligned = dataclasses.replace(
        params,
        chi_1=(0.0, 0.0, params.chi_1[2]),
        chi_2=(0.0, 0.0, params.chi_2[2]),
    )
    row = {}
    # every TEOB run fixes its own orbital phase (teob_precessing_polarizations
    # sets reference_phase to it): re-predict for each
    t10, p = teob_plus(aligned)
    model_aligned = precessing.predict(grid, p)[0]
    row["aligned: model vs TEOB 10Hz"] = mismatch(t10, model_aligned)
    for f0 in START_FREQUENCIES:
        t, p = teob_plus(aligned, 0.95 * f0)
        row[f"aligned: model vs TEOB {f0:g}Hz"] = mismatch(t, precessing.predict(grid, p)[0])
        if "--anchor" in sys.argv:
            row[f"aligned: ... best anchor {f0:g}Hz"] = best_anchor_mismatch(t, p)
    ts, p = teob_plus(aligned, srate_interp=2 * model.dataset.srate_hz * srate_interp_scaling(model.modes))
    row["aligned: model vs TEOB 10Hz srate x2"] = mismatch(ts, precessing.predict(grid, p)[0])

    t10, p10 = teob_plus(params)
    jax10 = precessing.predict(grid, p10)[0]
    row["precessing: model vs TEOB 10Hz"] = mismatch(t10, jax10)
    if "--anchor" in sys.argv:
        row["precessing: ... best anchor 10Hz"] = best_anchor_mismatch(t10, p10)
    for f0 in START_FREQUENCIES:
        t, p = teob_plus(params, 0.95 * f0)
        row[f"precessing: model vs TEOB {f0:g}Hz"] = mismatch(t, precessing.predict(grid, p)[0])
        if "--anchor" in sys.argv:
            row[f"precessing: ... best anchor {f0:g}Hz"] = best_anchor_mismatch(t, p)
    if use_numpy:
        row["precessing: JAX vs numpy predictor"] = mismatch(
            precessing.predict_numpy(grid, p10)[0], jax10
        )
    print(f"seed {seed}: q {params.mass_ratio:.2f} L1 {params.lambda_1:.0f} "
          f"L2 {params.lambda_2:.0f}", flush=True)
    for key, value in row.items():
        print(f"  {key:40s} {value:.2e}", flush=True)
        columns.setdefault(key, []).append(value)

print("\nmedians over", n_seeds, "seeds:")
for key, values in columns.items():
    print(f"  {key:40s} {np.median(values):.2e}")
