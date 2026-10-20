"""Merger-reference errors of the (2,2) regressor refitted with different
representations of the tidal deformabilities, the rest of the model
unchanged.

Refits the (2,2) of a model trained by ``modes_scale.py train`` on the
first N waveforms of its sharded dataset, once per representation, and prints the
merger-time and phase errors (:meth:`ValidateModel.merger_reference_errors`)
on the same binaries.

Run with: python merger_reference_experiments.py DATASET MODEL N N_BINARIES
"""
import sys

import numpy as np

import validate_model as v
from mlgw_bns.dataset_generation import ParameterSet
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.modes_dataset import ShardedModesDataset
from mlgw_bns.neural_network import Hyperparameters, KernelRidgeNetwork

dataset_path, model_path, n, n_binaries = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
MODE = v.Mode(2, 2)

#: transformations of the tidal deformabilities (columns 1 and 2) the
#: regressor sees instead of their values
TRANSFORMS = {
    "linear Lambda": None,
    "log Lambda": np.log,
    "Lambda^(1/5)": lambda x: x ** 0.2,
}


class TransformedKernelRidge(KernelRidgeNetwork):
    """Kernel ridge on transformed tidal deformabilities."""

    transform = None

    def _inputs(self, x):
        if self.transform is None:
            return x
        x = np.array(x, dtype=float)
        x[:, 1:3] = self.transform(x[:, 1:3])
        return x

    def fit(self, x_data, y_data, sample_weight=None):
        super().fit(self._inputs(x_data), y_data, sample_weight=sample_weight)

    def predict(self, x_data):
        return super().predict(self._inputs(x_data))

model = v.load_model(model_path)
mode_model = model.mode_models[MODE]
parameters, reduced, weights = ShardedModesDataset(dataset_path).reduced(model, MODE, n)
generator = model.dataset.parameter_generator_class(
    parameter_ranges=model.parameter_ranges, dataset=model.dataset, seed=7
)
intrinsics = ParameterSet.from_list_of_waveform_parameters([next(generator) for _ in range(n_binaries)])

mode_model.nn_kind = TransformedKernelRidge
for name, transform in TRANSFORMS.items():
    TransformedKernelRidge.transform = staticmethod(transform) if transform else None
    mode_model.nn = mode_model.fit_nn(Hyperparameters.default_kernel_ridge(n, mode=MODE), parameters, reduced, weights)
    model._batched_cache.clear()
    dt, dphi, _ = ValidateModel(mode_model).merger_reference_errors(intrinsics)
    alpha = np.atleast_1d(mode_model.nn.regressor.alpha)
    print(
        f"{name:>24}: |dt| median {np.median(abs(dt))*1e6:6.1f} us, 90% "
        f"{np.quantile(abs(dt), .9)*1e6:6.1f}, max {abs(dt).max()*1e6:6.0f}; |dphi| median "
        f"{np.median(abs(dphi)):.3f}, 90% {np.quantile(abs(dphi), .9):.3f}; "
        f"alpha {alpha.min():.1e}..{alpha.max():.1e}", flush=True,
    )
