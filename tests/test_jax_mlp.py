"""Tests for the JAX perceptron (:mod:`mlgw_bns.jax_mlp`)."""

import dataclasses
import os
import signal
import threading
import time

import numpy as np
import pytest

jax = pytest.importorskip("jax")
# as the precession code has it: the network is evaluated in double precision
jax.config.update("jax_enable_x64", True)

from mlgw_bns.jax_mlp import JaxMLP, MLPConfig, TrainingInterrupted  # noqa: E402


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (256, 3))
    return x, np.stack([np.sin(3 * x[:, 0]) + x[:, 1], x[:, 2] ** 2 - x[:, 0]], axis=1)


CONFIG = MLPConfig(hidden=(16, 16), epochs=3000, batch_size=32, check_every=5)


def test_a_training_stopped_by_a_signal_resumes_where_it_stopped(data, tmp_path):
    """SIGUSR1 (SLURM's warning before the walltime) makes the training save
    its checkpoint and stop; resumed, it ends where an uninterrupted training
    does, to the bit."""
    x, y = data
    straight = JaxMLP(CONFIG).fit(x, y)
    checkpoint = tmp_path / "mlp.checkpoint"

    def interrupt():
        while not checkpoint.exists():
            time.sleep(1e-3)
        os.kill(os.getpid(), signal.SIGUSR1)

    threading.Thread(target=interrupt, daemon=True).start()
    with pytest.raises(TrainingInterrupted):
        JaxMLP(CONFIG).fit(x, y, checkpoint=str(checkpoint), checkpoint_seconds=0.0)
    assert signal.getsignal(signal.SIGUSR1) is signal.SIG_DFL
    resumed = JaxMLP(CONFIG).fit(x, y, checkpoint=str(checkpoint))
    for (w_1, b_1), (w_2, b_2) in zip(straight.weights, resumed.weights):
        assert np.array_equal(w_1, w_2) and np.array_equal(b_1, b_2)
    assert np.array_equal(straight.history, resumed.history)
    # the checkpoint is of this training: another is refused
    with pytest.raises(ValueError):
        JaxMLP(dataclasses.replace(CONFIG, seed=1)).fit(x, y, checkpoint=str(checkpoint))


def test_patience_stops_early_and_keeps_the_best(data):
    x, y = data
    mlp = JaxMLP(dataclasses.replace(CONFIG, patience=3)).fit(x, y)
    assert mlp.history[-1, 0] < CONFIG.epochs
    assert np.argmin(mlp.history[:, 1]) == len(mlp.history) - 4
    assert np.allclose(mlp.predict(x[:4]), np.asarray(mlp.jax_function()(x[:4])), rtol=1e-12, atol=1e-12)


def test_uniform_sample_weights_are_no_weights(data):
    x, y = data
    config = dataclasses.replace(CONFIG, epochs=20)
    plain = JaxMLP(config).fit(x, y)
    uniform = JaxMLP(config).fit(x, y, sample_weight=np.full(len(x), 3.0))
    for (w_1, b_1), (w_2, b_2) in zip(plain.weights, uniform.weights):
        assert np.array_equal(w_1, w_2) and np.array_equal(b_1, b_2)
    weighted = JaxMLP(config).fit(x, y, sample_weight=np.linspace(0.1, 2.0, len(x)))
    assert not np.array_equal(plain.weights[0][0], weighted.weights[0][0])


def test_pickled_in_the_precision_of_the_training(data, tmp_path):
    """Trained in single precision, the weights are pickled so (they are
    those numbers) and evaluated in double."""
    import joblib

    x, y = data
    mlp = JaxMLP(dataclasses.replace(CONFIG, epochs=5)).fit(x, y)
    filename = tmp_path / "mlp.joblib"
    joblib.dump(mlp, filename)
    assert mlp.__getstate__()["weights"][0][0].dtype == np.float32
    loaded = joblib.load(filename)
    assert loaded.weights[0][0].dtype == np.float64
    assert np.array_equal(mlp.predict(x), loaded.predict(x))


def test_steps_set_the_epochs():
    config = MLPConfig(batch_size=128).with_steps(200000, 1024)
    # 922 of the 1024 kept (as in fit): 7 batches an epoch
    assert config.epochs == round(200000 / 7) and config.check_every == config.epochs // 200


def test_warmup_and_clipping_train_and_report_progress(data):
    """The warmup and the clipping of the gradients are options of the same
    training (off by default, which they leave as it was); ``progress`` sees
    the best held-out loss at every check, and an exception it raises ends
    the training."""
    x, y = data
    config = dataclasses.replace(CONFIG, epochs=200)
    reports = []
    mlp = JaxMLP(dataclasses.replace(config, warmup=0.1, clip_norm=0.5)).fit(
        x, y, progress=lambda steps, best: reports.append((steps, best))
    )
    steps_per_epoch = (len(x) - int(round(0.1 * len(x)))) // config.batch_size
    assert [s for s, _ in reports] == [int(e) * steps_per_epoch for e in mlp.history[:, 0]]
    assert np.all(np.diff([b for _, b in reports]) <= 0)
    assert min(b for _, b in reports) < 0.1 * reports[0][1]
    default = JaxMLP(config).fit(x, y)
    unchanged = JaxMLP(dataclasses.replace(config, warmup=0.0, clip_norm=None)).fit(x, y)
    assert np.array_equal(default.history, unchanged.history)
    assert not np.array_equal(default.history, mlp.history)

    class Stop(Exception):
        pass

    def stop(steps, best):
        raise Stop

    with pytest.raises(Stop):
        JaxMLP(config).fit(x, y, progress=stop)
