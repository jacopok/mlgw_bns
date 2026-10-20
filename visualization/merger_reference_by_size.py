"""Merger-reference errors (time and phase at the merger, model minus EOB)
of several trained models on the same binaries.

Run with: python merger_reference_by_size.py N_BINARIES LAMBDA_MAX MODEL [MODEL ...]
"""
import dataclasses
import sys

import numpy as np

import validate_model as v
from mlgw_bns.dataset_generation import ParameterSet
from mlgw_bns.model_validation import ValidateModel

n_binaries, lambda_max, paths = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3:]
models = [v.load_model(path) for path in paths]
dataset = models[0].dataset
generator = dataset.parameter_generator_class(
    parameter_ranges=dataclasses.replace(
        models[0].parameter_ranges,
        lambda1_range=(5.0, lambda_max), lambda2_range=(5.0, lambda_max),
    ),
    dataset=dataset,
    seed=7,
)
intrinsics = ParameterSet.from_list_of_waveform_parameters(
    [next(generator) for _ in range(n_binaries)]
)
rows = []
for path, model in zip(paths, models):
    dt, dphi, valid = ValidateModel(model.mode_models[v.Mode(2, 2)]).merger_reference_errors(intrinsics)
    rows.append((dt, dphi))
    q = np.quantile
    print(
        f"{path}: |dt| median {np.median(abs(dt))*1e6:.1f} us, 90% {q(abs(dt), .9)*1e6:.1f}, "
        f"max {abs(dt).max()*1e6:.0f}; |dphi| median {np.median(abs(dphi)):.3f}, "
        f"90% {q(abs(dphi), .9):.3f}", flush=True,
    )
np.savez("merger_reference_by_size.npz", paths=paths, parameters=valid.parameter_array,
         dt=[r[0] for r in rows], dphi=[r[1] for r in rows])
