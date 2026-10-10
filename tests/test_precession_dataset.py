"""Tests for the sharded training sets of the regressed precession angles
(:mod:`mlgw_bns.precession_dataset`)."""

import os
import time

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from mlgw_bns.precession_dataset import DatasetConfig, ShardedDataset  # noqa: E402
from mlgw_bns.sharding import ShardLock  # noqa: E402
from mlgw_bns.precession_regression import (  # noqa: E402
    _penalized_least_squares,
    _bspline,
    AngleGrid,
    PRIOR_PENALTY,
    RIDGE_PENALTY,
    generate_training_set,
)

#: three shards, the last partial; a short integration
CONFIG = DatasetConfig(n_binaries=12, shard_size=5, seed=3, n_steps=256)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    dataset = ShardedDataset.create(str(tmp_path_factory.mktemp("precession") / "train"), CONFIG)
    assert dataset.generate(n_jobs=1, batch=4)
    return dataset


def test_shards_depend_on_the_seed_and_their_index_alone(dataset, tmp_path):
    copy = ShardedDataset.create(str(tmp_path / "copy"), CONFIG)
    copy._timed_make(1, n_jobs=1, batch=4)  # (another batch changes the angles by ~1e-17)
    with np.load(dataset.path(1)) as a, np.load(copy.path(1)) as b:
        assert sorted(a.files) == sorted(b.files)
        for key in a.files:
            assert np.array_equal(a[key], b[key]), key
    assert np.array_equal(dataset.load(fields=("intrinsic",))["intrinsic"][5:10], dataset.parameters(1)[0])


def test_the_first_n_binaries_and_selections(dataset):
    chunks = list(dataset.chunks(7, fields=("intrinsic", "coefficients")))
    assert [len(c["index"]) for c in chunks] == [5, 2]
    assert np.array_equal(np.concatenate([c["index"] for c in chunks]), np.arange(7))
    data = dataset.load(fields=("omega_reference",), select=lambda index: index % 2 == 0)
    assert np.array_equal(data["index"], np.arange(0, 12, 2))
    with pytest.raises(ValueError):
        dataset.load(13)


def test_the_nodes_follow_from_the_reference_frequency(dataset):
    data = dataset.load(2)
    generated = generate_training_set(
        CONFIG.grid, data["intrinsic"], data["omega_reference"], CONFIG.n_steps, CONFIG.stride, n_jobs=1
    )
    assert np.allclose(dataset.x(data["omega_reference"]), generated["x"], rtol=0, atol=1e-13)
    assert np.allclose(generated["coefficients"], data["coefficients"], rtol=1e-12, atol=1e-14)


def test_a_dataset_grows_but_does_not_change(dataset, tmp_path):
    with pytest.raises(ValueError):  # its last shard is partial
        ShardedDataset.create(dataset.directory, DatasetConfig(n_binaries=15, shard_size=5, seed=3, n_steps=256))
    directory = str(tmp_path / "grown")
    ShardedDataset.create(directory, DatasetConfig(n_binaries=10, shard_size=5, seed=3))
    assert ShardedDataset.create(directory, DatasetConfig(n_binaries=20, shard_size=5, seed=3)).n_shards == 4
    with pytest.raises(ValueError):
        ShardedDataset.create(directory, DatasetConfig(n_binaries=20, shard_size=5, seed=4))


def test_locks_are_exclusive_until_stale(tmp_path):
    filename = str(tmp_path / "locks" / "00000.lock")
    first, second = ShardLock(filename, 60.0), ShardLock(filename, 60.0)
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert not os.path.exists(filename)
    assert second.acquire()
    second._stop.set()  # its owner dies: the lock is no longer refreshed
    second._thread.join()
    os.utime(filename, (time.time() - 120, time.time() - 120))
    third = ShardLock(filename, 60.0)
    assert third.acquire()
    second.release()  # not its lock anymore: left alone
    assert os.path.exists(filename)
    third.release()


def test_refinement_and_training_a_shard_at_a_time(dataset):
    dataset.refinement(1, folds=2, prior_size=6, n_components=(4, 4), kernel_gamma=0.02)
    for fold in range(2):
        dataset.train_prior(1, fold)
    assert dataset.refine(1, n_jobs=1)
    refined = dataset.load(fields=("coefficients",), source="refine1")["coefficients"]
    assert refined.shape == (12, CONFIG.grid.n_envelopes, CONFIG.grid.n_coefficients)
    assert np.all(np.isfinite(refined))
    regressor = dataset.train(10, "refine1", n_components=(4, 4))
    data = dataset.load(fields=("intrinsic", "omega_reference"))
    coefficients, switch = regressor.predict(data["intrinsic"][10:], data["omega_reference"][10:])
    assert coefficients.shape == (2,) + refined.shape[1:] and switch.shape == (2, 14)


@pytest.mark.parametrize("blocks, prior", [(3, False), (3, True), (7, False)])
def test_banded_least_squares_solve_the_normal_equations(blocks, prior):
    """The banded assembly and solve of the envelope fits give the solution
    of the dense normal equations, complex or real, with a prior or not, for
    one or several right-hand sides."""
    rng = np.random.default_rng(blocks)
    grid = AngleGrid(n_cells=9)
    x = np.sort(rng.uniform(*grid.x_range, 300))
    cell, weights = _bspline(np, grid, x)
    n = grid.n_coefficients
    complex_ = blocks == 3
    factors = rng.normal(size=(300, blocks)) + (1j * rng.normal(size=(300, blocks)) if complex_ else 0)
    data = rng.normal(size=(300, 2)) + (1j * rng.normal(size=(300, 2)) if complex_ else 0)
    prior_values = rng.normal(size=(blocks, n)) if prior else None
    design = np.zeros((300, blocks * n), factors.dtype)
    for r in range(4):
        for b in range(blocks):
            design[np.arange(300), b * n + cell + r] += factors[:, b] * weights[:, r]
    gram = design.conj().T @ design
    scale = np.trace(gram).real / gram.shape[0]
    second = np.diff(np.eye(n), 2, axis=0)
    penalty = np.kron(np.eye(blocks), 1e-3 * second.T @ second + RIDGE_PENALTY * np.eye(n))
    rhs = design.conj().T @ data
    if prior:
        penalty += PRIOR_PENALTY * np.eye(blocks * n)
        rhs += scale * PRIOR_PENALTY * prior_values.reshape(-1, 1)
    expected = np.linalg.solve(gram + scale * penalty, rhs).T.reshape(2, blocks, n)
    both = _penalized_least_squares(cell, weights, factors, data, n, prior_values, 1e-3)
    one = _penalized_least_squares(cell, weights, factors, data[:, 1], n, prior_values, 1e-3)
    assert np.allclose(both, expected, rtol=0, atol=1e-9 * np.max(np.abs(expected)))
    assert np.allclose(one, both[1], rtol=0, atol=1e-12 * np.max(np.abs(expected)))
