"""Tests for the sharded training sets of the co-precessing modes model
(:mod:`mlgw_bns.modes_dataset`), and for what they rely on: the residuals of
one EOB call per waveform, the perceptron backend, and the validation
against stored waveforms."""

import dataclasses
import glob
import os

import numpy as np
import pytest

from mlgw_bns.dataset_generation import ParameterSet
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.jax_mlp import MLPConfig
from mlgw_bns.model_validation import stored_waveform_mismatches
from mlgw_bns.modes_dataset import ModesDatasetConfig, ShardedModesDataset, load_model, stored_size
from mlgw_bns.neural_network import Hyperparameters, JaxMLPNetwork

#: two modes from 40 Hz, three shards (the last partial)
CONFIG = ModesDatasetConfig(n_binaries=40, shard_size=16, seed=3, modes=((2, 2), (3, 3)), initial_frequency_hz=40.0)

PERCEPTRON = MLPConfig(hidden=(16, 16), epochs=20, batch_size=8)


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    root = tmp_path_factory.mktemp("modes")
    train = ShardedModesDataset.create(str(root / "train"), CONFIG)
    train.train_downsampling(8, n_jobs=1)
    validation = ShardedModesDataset.create(
        str(root / "validation"), dataclasses.replace(CONFIG, n_binaries=12, seed=4)
    )
    validation.copy_downsampling(train)
    assert validation.generate(n_jobs=1) and train.generate(n_jobs=1)
    return train, validation


@pytest.fixture(scope="module")
def models(datasets, tmp_path_factory):
    train, _ = datasets
    root = tmp_path_factory.mktemp("runs")
    krr = train.train(40, str(root / "krr" / "model"))
    mlp = train.train(
        40, str(root / "mlp" / "model"), nn_kind=JaxMLPNetwork,
        hyperparameters=lambda mode, n: Hyperparameters.default_jax_mlp(n, PERCEPTRON),
        pca_size=40, checkpoint=str(root / "mlp" / "checkpoint"),
    )
    return krr, mlp


def test_baselines_only_where_the_residuals_are_kept(datasets):
    """The post-Newtonian baselines evaluated only at the points kept (and in
    the window of the reference) give the residuals of the whole grid, to
    the bit."""
    train, _ = datasets
    model = train.model()
    params = train.parameters(0)[:3]
    indices = {mode: model.mode_models[mode].downsampling_indices for mode in model.modes}
    references = {mode: model.mode_models[mode].dataset.amplitude_reference_parameters for mode in model.modes}
    _, amplitudes, phases = model._multimode_mode_residuals(params, model.dataset.frequencies, indices, references)
    _, full_amplitudes, full_phases = model._multimode_mode_residuals(
        params, model.dataset.frequencies, None, references
    )
    for mode in model.modes:
        assert np.array_equal(amplitudes[mode], full_amplitudes[mode][:, indices[mode].amplitude_indices])
        assert np.array_equal(phases[mode], full_phases[mode][:, indices[mode].phase_indices])


def test_shards_depend_on_the_seed_and_their_index_alone(datasets, tmp_path):
    train, _ = datasets
    copy = ShardedModesDataset.create(str(tmp_path / "copy"), CONFIG)
    copy.copy_downsampling(train)
    copy._timed_make(2, n_jobs=1)
    with np.load(train.path(2)) as a, np.load(copy.path(2)) as b:
        assert a.files == b.files
        for key in a.files:
            assert np.array_equal(a[key], b[key], equal_nan=True), key


def test_the_first_waveforms_are_the_same_whatever_is_read(datasets):
    train, _ = datasets
    parameters, residuals = train.load_residuals(20)
    all_parameters, all_residuals = train.load_residuals()
    assert len(all_parameters) == CONFIG.n_binaries and np.array_equal(parameters, all_parameters[:20])
    pieces = list(train.residuals(Mode(3, 3), 20))
    assert [len(p) for p, _ in pieces] == [16, 4]
    assert np.array_equal(
        np.concatenate([r.phase_residuals for _, r in pieces]), all_residuals[Mode(3, 3)].phase_residuals[:20]
    )


def test_downsampling_only_from_the_same_model(datasets, tmp_path):
    train, _ = datasets
    other = ShardedModesDataset.create(str(tmp_path / "other"), dataclasses.replace(CONFIG, initial_frequency_hz=30.0))
    with pytest.raises(ValueError):
        other.copy_downsampling(train)


def test_trained_models_validate(datasets, models):
    train, validation = datasets
    parameters, residuals = validation.load_residuals()
    for model in models:
        loaded = load_model(train, model.base_filename)
        result = stored_waveform_mismatches(loaded, parameters, residuals, chunk=5)
        # (models of 40 waveforms: the merger, which each mode's mismatch
        # is referenced to, is not always within its time window)
        for key in ("mode l2_m2", "mode l3_m3"):
            assert np.all((result[key] >= 0) & (result[key] <= 1)), key
        assert np.all((result["full"] >= 0) & (result["full"] < 0.1))
        assert np.allclose(result["power l2_m2"] + result["power l3_m3"], 1.0, atol=0.2)
        assert stored_size(loaded)["regressors"] > 0
    # the perceptrons' checkpoints are gone once their regressors are saved
    assert not glob.glob(os.path.join(os.path.dirname(models[1].base_filename), "checkpoint*"))


def test_a_training_resumes_from_the_modes_it_had_trained(datasets, models):
    train, _ = datasets
    krr = models[0]
    filename = krr.mode_models[Mode(3, 3)].filename_nn
    before = os.path.getmtime(krr.mode_models[Mode(2, 2)].filename_nn)
    os.remove(filename)
    assert train.train(40, krr.base_filename, modes=[(3, 3)]) is not None
    assert os.path.exists(filename)
    assert os.path.getmtime(krr.mode_models[Mode(2, 2)].filename_nn) == before


def test_perceptron_batched_as_per_mode(datasets, models):
    """The batched evaluation of a model of perceptrons gives the residuals
    of each mode's own."""
    _, mlp = models
    x = np.asarray(datasets[1].load(5, ("parameters",))["parameters"])
    batched = mlp.batched_surrogate(backend="numpy")._residuals(np, x)
    for mode, (amplitude, phase) in zip(mlp.modes, batched):
        mode_model = mlp.mode_models[mode]
        expected = mode_model.predict_residuals_bulk(ParameterSet(x), mode_model.nn)
        assert np.allclose(amplitude, expected.amplitude_residuals, rtol=1e-10, atol=1e-12)
        assert np.allclose(phase, expected.phase_residuals, rtol=1e-10, atol=1e-10)
