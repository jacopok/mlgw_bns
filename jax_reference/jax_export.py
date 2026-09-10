"""Export a trained mlgw_bns Model to a standalone HDF5 file.

The exported file contains *all* the data that
``jax_import_n_predict.py`` needs to reconstruct the full
JAX waveform pipeline — no ``mlgw_bns`` dependency is required
at prediction time.

Usage
-----
>>> import mlgw_bns
>>> from jax_export import export_model
>>> model = mlgw_bns.Model.default()
>>> export_model(model, "mlgw_bns_jax_model.h5")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from mlgw_bns.model import Model


def export_model(model: "Model", path: str) -> None:
    """Dump everything jax_import_n_predict needs into *path*.

    Parameters
    ----------
    model : mlgw_bns.Model
        A fully loaded model (``model.nn``, ``model.pca_data``,
        ``model.downsampling_indices`` must not be ``None``).
    path : str
        Output HDF5 file path.
    """
    assert model.nn is not None, "model.nn is None — load the model first."
    assert model.pca_data is not None, "model.pca_data is None."
    assert model.downsampling_indices is not None, "model.downsampling_indices is None."

    nn = model.nn  # SklearnNetwork

    with h5py.File(path, "w") as f:
        # ── MLP weights ──────────────────────────────────────────────
        mlp = f.create_group("mlp")
        mlp.attrs["activation"] = nn.nn.activation
        mlp.attrs["n_layers"] = len(nn.nn.coefs_)
        for i, (w, b) in enumerate(zip(nn.nn.coefs_, nn.nn.intercepts_)):
            mlp.create_dataset(f"coef_{i}", data=np.asarray(w, dtype=np.float64))
            mlp.create_dataset(f"intercept_{i}", data=np.asarray(b, dtype=np.float64))

        # ── StandardScaler ───────────────────────────────────────────
        scaler = f.create_group("scaler")
        scaler.create_dataset("mean", data=np.asarray(nn.param_scaler.mean_, dtype=np.float64))
        scaler.create_dataset("scale", data=np.asarray(nn.param_scaler.scale_, dtype=np.float64))

        # ── PCA ──────────────────────────────────────────────────────
        pca = f.create_group("pca")
        pca.create_dataset("eigenvectors", data=np.asarray(model.pca_data.eigenvectors, dtype=np.float64))
        pca.create_dataset("eigenvalues", data=np.asarray(model.pca_data.eigenvalues, dtype=np.float64))
        pca.create_dataset("mean", data=np.asarray(model.pca_data.mean, dtype=np.float64))
        pca.create_dataset(
            "principal_components_scaling",
            data=np.asarray(model.pca_data.principal_components_scaling, dtype=np.float64),
        )
        pca.attrs["pc_exponent"] = float(nn.hyper.pc_exponent)

        # ── Frequency grids & downsampling ───────────────────────────
        grid = f.create_group("grid")
        grid.create_dataset(
            "frequencies_hz",
            data=np.asarray(model.dataset.frequencies_hz, dtype=np.float64),
        )
        grid.create_dataset(
            "frequencies_natural",
            data=np.asarray(model.dataset.frequencies, dtype=np.float64),
        )
        grid.create_dataset(
            "amplitude_indices",
            data=np.asarray(model.downsampling_indices.amplitude_indices, dtype=np.int64),
        )
        grid.create_dataset(
            "phase_indices",
            data=np.asarray(model.downsampling_indices.phase_indices, dtype=np.int64),
        )
        grid.attrs["total_mass"] = float(model.dataset.total_mass)

    print(f"Model exported to {path}")
