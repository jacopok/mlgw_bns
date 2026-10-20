"""Is the (2,2) merger reference harder to learn over a wider range of
tidal deformabilities at the same density of training points?

From the first N waveforms of a sharded dataset, fits the (2,2) regressor
of a model on (a) those with both deformabilities below LAMBDA_MAX and (b)
as many drawn from the whole set, and prints the merger-reference errors
(:meth:`ValidateModel.merger_reference_errors`) of each on binaries below
LAMBDA_MAX.

Run with: python merger_reference_range.py DATASET MODEL N LAMBDA_MAX N_BINARIES
"""
import dataclasses
import sys

import numpy as np

import validate_model as v
from mlgw_bns.dataset_generation import ParameterSet
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.modes_dataset import ShardedModesDataset
from mlgw_bns.neural_network import Hyperparameters

dataset_path, model_path = sys.argv[1], sys.argv[2]
n, lambda_max, n_binaries = int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
MODE = v.Mode(2, 2)

model = v.load_model(model_path)
mode_model = model.mode_models[MODE]
parameters, reduced, weights = ShardedModesDataset(dataset_path).reduced(model, MODE, n)
inside = np.flatnonzero(np.all(parameters[:, 1:3] <= lambda_max, axis=1))
anywhere = np.random.default_rng(0).choice(len(parameters), size=len(inside), replace=False)
generator = model.dataset.parameter_generator_class(
    parameter_ranges=dataclasses.replace(
        model.parameter_ranges, lambda1_range=(5.0, lambda_max), lambda2_range=(5.0, lambda_max)
    ),
    dataset=model.dataset,
    seed=7,
)
intrinsics = ParameterSet.from_list_of_waveform_parameters([next(generator) for _ in range(n_binaries)])

for name, rows in ((f"Lambda <= {lambda_max:g}", inside), ("whole range", anywhere)):
    mode_model.nn = mode_model.fit_nn(
        Hyperparameters.default_kernel_ridge(len(rows), mode=MODE), parameters[rows], reduced[rows],
        None if weights is None else weights[rows],
    )
    dt, dphi, _ = ValidateModel(mode_model).merger_reference_errors(intrinsics)
    print(
        f"{name:>16} ({len(rows)} waveforms): |dt| median {np.median(abs(dt))*1e6:6.1f} us, 90% "
        f"{np.quantile(abs(dt), .9)*1e6:6.1f}, max {abs(dt).max()*1e6:6.0f}; |dphi| median "
        f"{np.median(abs(dphi)):.3f}, 90% {np.quantile(abs(dphi), .9):.3f}", flush=True,
    )
