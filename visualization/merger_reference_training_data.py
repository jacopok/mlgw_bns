"""Does the (2,2) merger tangent of the stored training residuals (the
multi-mode generation of ``ShardedModesDataset``) agree with that of the
single-mode generator the validation compares against?

Prints, for the first N waveforms of the dataset, the merger-time and phase
differences against the regenerated residuals of the tangent at the top
phase knot (:meth:`ValidateModel._merger_lines`), for the stored residuals
and for the regenerated ones projected onto the model's principal
components (the floor of a perfect regressor).

Run with: python merger_reference_training_data.py DATASET MODEL N
"""
import sys

import numpy as np

import validate_model as v
from mlgw_bns.data_management import FDWaveforms, Residuals
from mlgw_bns.dataset_generation import ParameterSet
from mlgw_bns.model_validation import ValidateModel
from mlgw_bns.modes_dataset import ShardedModesDataset

dataset_path, model_path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
model = v.load_model(model_path)
mode_model = model.mode_models[v.Mode(2, 2)]
dataset = mode_model.dataset
parameters, stored = next(ShardedModesDataset(dataset_path).residuals(v.Mode(2, 2), n))
parameters, stored = parameters[:n], np.asarray(stored.phase_residuals[:n], dtype=float)
print("stored phase dtype", stored.dtype)
amplitudes, fresh = (np.array(x) for x in zip(*(
    dataset.waveform_generator.generate_residuals(
        params, dataset.frequencies, mode_model.downsampling_indices,
        dataset.amplitude_reference_parameters,
    )
    for params in ParameterSet(parameters).waveform_parameters(dataset)
)))
# the principal-component projection alone, as a perfect regressor would give it
pca = mode_model.pca_model
projected = Residuals.from_combined_residuals(
    pca.reconstruct_data(pca.reduce_data(Residuals(amplitudes, fresh).combined, mode_model.pca_data), mode_model.pca_data),
    mode_model.downsampling_indices.numbers_of_points,
).phase_residuals


def report(name, phases):
    _, slope, intercept = ValidateModel(mode_model)._merger_lines(FDWaveforms(phases, phases - fresh))
    dt, dphi = -slope / (2 * np.pi), np.angle(np.exp(1j * intercept))
    print(f"{name}: |dt| median {np.median(abs(dt))*1e6:.2f} us, 90% {np.quantile(abs(dt), .9)*1e6:.2f}, "
          f"max {abs(dt).max()*1e6:.2f}; |dphi| median {np.median(abs(dphi)):.4f}, max {abs(dphi).max():.4f}")


report("stored training residuals", stored)
report("PCA projection", np.asarray(projected))
