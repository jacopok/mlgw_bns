r"""A multi-layer perceptron trained and evaluated with JAX.

An alternative to :class:`~mlgw_bns.neural_network.KernelRidgeNetwork` for
regressors that are evaluated inside JAX functions (such as
:class:`~mlgw_bns.precession_regression.PrecessionRegressor`): the trained
network is a pure function of its inputs, :meth:`JaxMLP.jax_function`.

Training is plain Adam (with optional decoupled weight decay) on minibatches,
the learning rate decaying as a cosine, on a weighted mean squared error of
the standardized outputs; a fraction of the training set is held out, and
the parameters with the lowest loss on it are kept.

Long trainings (large training sets, batch jobs with a walltime) keep a
checkpoint (:meth:`JaxMLP.fit`): the whole state of the optimizer, saved
every few minutes and when the process is asked to stop (``SIGTERM`` or
``SIGUSR1``, which SLURM sends before the walltime), after which
:class:`TrainingInterrupted` is raised; the same call resumes from it,
and gives the parameters an uninterrupted training would have.
"""

from __future__ import annotations

import hashlib
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Tuple

import numpy as np

ACTIVATIONS = ("gelu", "silu", "tanh", "softplus", "elu")


def array_activation(xp, name: str) -> Callable:
    """The activation ``name`` for arrays of the module ``xp`` (numpy or
    ``jax.numpy``), as :func:`jax.nn.gelu` (tanh approximation) and the
    others compute it."""
    return {
        "gelu": lambda a: 0.5 * a * (1 + xp.tanh(np.sqrt(2 / np.pi) * (a + 0.044715 * (a * a * a)))),
        "silu": lambda a: a / (1 + xp.exp(-a)),
        "tanh": xp.tanh,
        "softplus": lambda a: xp.logaddexp(0.0, a),
        "elu": lambda a: xp.where(a > 0, a, xp.expm1(xp.minimum(a, 0.0))),
    }[name]


def _activation(jnp, name: str) -> Callable:
    import jax

    return {
        "gelu": jax.nn.gelu,
        "silu": jax.nn.silu,
        "tanh": jnp.tanh,
        "softplus": jax.nn.softplus,
        "elu": jax.nn.elu,
    }[name]


@dataclass(frozen=True)
class MLPConfig:
    """Architecture and training of a :class:`JaxMLP`."""

    hidden: Tuple[int, ...] = (256, 256, 256, 256)
    activation: str = "gelu"
    learning_rate: float = 2e-3
    final_learning_rate: float = 2e-5
    epochs: int = 3000
    batch_size: int = 128
    weight_decay: float = 0.0
    #: fraction of the training set held out to pick the best epoch
    validation_fraction: float = 0.1
    #: how often (in epochs) the held-out loss is computed
    check_every: int = 10
    seed: int = 0
    #: stop when the held-out loss has not improved for this many checks
    patience: Optional[int] = None
    #: of the parameters and the arithmetic of the training (the fitted
    #: network is evaluated in double precision)
    dtype: str = "float32"

    def with_steps(self, steps: int, n_train: int) -> "MLPConfig":
        """This configuration, with as many epochs as make about ``steps``
        steps on ``n_train`` samples (of which :attr:`validation_fraction`
        held out), and the held-out loss evaluated ~200 times."""
        kept = n_train - int(round(self.validation_fraction * n_train))  # as in fit
        per_epoch = max(kept // self.batch_size, 1)
        epochs = max(int(round(steps / per_epoch)), 1)
        return replace(self, epochs=epochs, check_every=max(epochs // 200, 1))


class TrainingInterrupted(Exception):
    """:meth:`JaxMLP.fit` was asked to stop by a signal, and saved its
    checkpoint first: call it again to resume."""


@dataclass
class JaxMLP:
    """A fitted perceptron: standardization of the inputs and outputs and
    the weights, as numpy arrays (so it pickles without JAX)."""

    config: MLPConfig = field(default_factory=MLPConfig)
    input_mean: Optional[np.ndarray] = None
    input_scale: Optional[np.ndarray] = None
    output_mean: Optional[np.ndarray] = None
    output_scale: Optional[np.ndarray] = None
    weights: Optional[list] = None
    history: Optional[np.ndarray] = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        loss_weights: Optional[np.ndarray] = None,
        checkpoint: Optional[str] = None,
        checkpoint_seconds: float = 600.0,
        log_seconds: float = 60.0,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "JaxMLP":
        """Train on inputs ``x`` ``(N, d_in)`` and targets ``y`` ``(N,
        d_out)``. ``loss_weights`` ``(d_out,)`` weigh the squared errors of
        the standardized outputs (uniform by default), ``sample_weight``
        ``(N,)`` those of each sample (uniform by default; the
        standardization is not weighted).

        With a ``checkpoint`` file, the state of the training is saved there
        every ``checkpoint_seconds``, at the end, and on ``SIGTERM`` or
        ``SIGUSR1`` (then :class:`TrainingInterrupted` is raised); a
        checkpoint of the same training (configuration and data) is resumed
        from, one of another refused. Progress is logged every
        ``log_seconds``.
        """
        import jax
        import jax.numpy as jnp

        config = self.config
        dtype = jnp.dtype(config.dtype)
        rng = np.random.default_rng(config.seed)
        self.input_mean, self.input_scale = x.mean(axis=0), x.std(axis=0)
        self.input_scale[self.input_scale == 0] = 1.0
        self.output_mean, self.output_scale = y.mean(axis=0), y.std(axis=0)
        self.output_scale[self.output_scale == 0] = 1.0
        xs = (x - self.input_mean) / self.input_scale
        ys = (y - self.output_mean) / self.output_scale
        weights = np.ones(y.shape[1]) if loss_weights is None else np.asarray(loss_weights, float)
        weights = jnp.asarray(weights / weights.mean(), dtype)
        samples = np.ones(len(x)) if sample_weight is None else np.asarray(sample_weight, float)
        samples = samples / samples.mean()

        order = rng.permutation(len(x))
        n_held = int(round(config.validation_fraction * len(x)))
        held, kept = order[:n_held], order[n_held:]
        x_train, y_train = jnp.asarray(xs[kept], dtype), jnp.asarray(ys[kept], dtype)
        x_held, y_held = jnp.asarray(xs[held], dtype), jnp.asarray(ys[held], dtype)
        s_train, s_held = jnp.asarray(samples[kept], dtype), jnp.asarray(samples[held], dtype)

        sizes = (x.shape[1],) + tuple(config.hidden) + (y.shape[1],)
        params = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            params.append((
                jnp.asarray(rng.normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)), (fan_in, fan_out)), dtype),
                jnp.zeros(fan_out, dtype),
            ))
        activation = _activation(jnp, config.activation)

        def forward(p, inputs):
            for w, b in p[:-1]:
                inputs = activation(inputs @ w + b)
            w, b = p[-1]
            return inputs @ w + b

        def loss(p, inputs, targets, sample):
            errors = jnp.sum(weights * (forward(p, inputs) - targets) ** 2, axis=-1)
            return jnp.mean(sample * errors) / weights.size

        batch_size = min(config.batch_size, len(kept))
        n_batches = max(len(kept) // batch_size, 1)
        total_steps = config.epochs * n_batches
        b1, b2, eps = 0.9, 0.999, 1e-8

        def step(carry, batch):
            p, m, v, t = carry
            grads = jax.grad(loss)(p, *batch)
            t = t + 1
            tf = t.astype(dtype)
            progress = jnp.minimum(tf / total_steps, 1.0)
            rate = config.final_learning_rate + 0.5 * (config.learning_rate - config.final_learning_rate) * (
                1.0 + jnp.cos(jnp.pi * progress)
            )
            m = jax.tree_util.tree_map(lambda a, g: b1 * a + (1 - b1) * g, m, grads)
            v = jax.tree_util.tree_map(lambda a, g: b2 * a + (1 - b2) * g * g, v, grads)
            p = jax.tree_util.tree_map(
                lambda w, a, s: w - rate * (
                    a / (1 - b1**tf) / (jnp.sqrt(s / (1 - b2**tf)) + eps) + config.weight_decay * w
                ),
                p, m, v,
            )
            return (p, m, v, t), None

        @jax.jit
        def epoch(carry, key, inputs, targets, sample):
            # the data are arguments, not constants baked into the program
            batches = jax.random.permutation(key, inputs.shape[0])[: n_batches * batch_size]
            batches = batches.reshape(n_batches, batch_size)
            carry, _ = jax.lax.scan(lambda c, i: step(c, (inputs[i], targets[i], sample[i])), carry, batches)
            return carry

        held_loss = jax.jit(loss)
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        state = {
            "carry": (params, zeros, zeros, jnp.asarray(0, jnp.int32)),
            "key": jax.random.PRNGKey(config.seed),
            "best": params, "best_loss": np.inf, "best_epoch": 0,
            "history": [], "epoch": 0, "stale": 0, "finished": False,
        }
        fingerprint = _fingerprint(config, x, y, loss_weights, sample_weight)
        if checkpoint is not None and os.path.exists(checkpoint):
            state = _load_checkpoint(checkpoint, fingerprint, dtype)
            logging.info(
                "MLP: resuming from %s at epoch %i/%i (best held-out loss %.3g at %i)",
                checkpoint, state["epoch"], config.epochs, state["best_loss"], state["best_epoch"],
            )

        def save():
            if checkpoint is not None:
                _save_checkpoint(checkpoint, state, fingerprint)

        stop = []
        previous = {}
        if checkpoint is not None and threading.current_thread() is threading.main_thread():
            for number in (signal.SIGTERM, signal.SIGUSR1):
                previous[number] = signal.signal(number, lambda signum, _: stop.append(signum))
        start = last_log = last_save = time.perf_counter()
        first_epoch = state["epoch"]
        try:
            while not state["finished"]:
                i = state["epoch"]
                state["key"], subkey = jax.random.split(state["key"])
                state["carry"] = epoch(state["carry"], subkey, x_train, y_train, s_train)
                state["epoch"] = i + 1
                if (i + 1) % config.check_every == 0 or i + 1 == config.epochs:
                    current = float(held_loss(state["carry"][0], x_held, y_held, s_held) if n_held
                                    else held_loss(state["carry"][0], x_train, y_train, s_train))
                    state["history"].append((i + 1, current))
                    if current < state["best_loss"]:
                        state["best"], state["best_loss"], state["best_epoch"] = state["carry"][0], current, i + 1
                        state["stale"] = 0
                    else:
                        state["stale"] += 1
                state["finished"] = i + 1 == config.epochs or (
                    config.patience is not None and state["stale"] >= config.patience
                )
                now = time.perf_counter()
                if now - last_log > log_seconds or state["finished"]:
                    per_epoch = (now - start) / (i + 1 - first_epoch)
                    loss_now = state["history"][-1][1] if state["history"] else float("nan")
                    logging.info(
                        "MLP epoch %i/%i: held-out loss %.3g (best %.3g at %i), %.2f s an epoch, %.0f s to go",
                        i + 1, config.epochs, loss_now, state["best_loss"], state["best_epoch"],
                        per_epoch, per_epoch * (config.epochs - i - 1),
                    )
                    last_log = now
                if state["finished"] or now - last_save > checkpoint_seconds:
                    save()
                    last_save = now
                if stop and not state["finished"]:
                    save()
                    raise TrainingInterrupted(
                        f"signal {stop[0]} at epoch {i + 1}/{config.epochs}; resume from {checkpoint}"
                    )
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
        if config.patience is not None and state["epoch"] < config.epochs:
            logging.info("MLP: no improvement in %i checks, stopped at epoch %i", config.patience, state["epoch"])
        self.weights = [(np.asarray(w, np.float64), np.asarray(b, np.float64)) for w, b in state["best"]]
        self.history = np.array(state["history"])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """The outputs at inputs ``x`` ``(N, d_in)`` (numpy)."""
        return self._evaluate(np, x)

    def __getstate__(self) -> dict:
        # the weights pickled in the precision they were trained in (they
        # are those numbers: nothing is lost), evaluated in double
        state = dict(self.__dict__)
        if state.get("weights") is not None:
            dtype = np.dtype(self.config.dtype)
            state["weights"] = [(w.astype(dtype), b.astype(dtype)) for w, b in state["weights"]]
        return state

    def __setstate__(self, state: dict) -> None:
        if state.get("weights") is not None:
            state["weights"] = [(np.asarray(w, np.float64), np.asarray(b, np.float64)) for w, b in state["weights"]]
        self.__dict__.update(state)

    def jax_function(self) -> Callable:
        """The network as a JAX function of the (unstandardized) inputs."""
        import jax.numpy as jnp

        frozen = JaxMLP(
            self.config,
            *(jnp.asarray(a) for a in (self.input_mean, self.input_scale, self.output_mean, self.output_scale)),
            [(jnp.asarray(w), jnp.asarray(b)) for w, b in self.weights],
        )
        return lambda x: frozen._evaluate(jnp, x)

    def _evaluate(self, xp, x):
        if xp is np:
            activation = array_activation(np, self.config.activation)
        else:
            activation = _activation(xp, self.config.activation)
        h = (x - self.input_mean) / self.input_scale
        for w, b in self.weights[:-1]:
            h = activation(h @ w + b)
        w, b = self.weights[-1]
        return (h @ w + b) * self.output_scale + self.output_mean


def _fingerprint(config: MLPConfig, x: np.ndarray, y: np.ndarray, loss_weights, sample_weight=None) -> str:
    """What identifies a training: the configuration and a hash of the data."""
    digest = hashlib.blake2b(repr(config).encode(), digest_size=16)
    weights = tuple(np.asarray(w, float) for w in (loss_weights, sample_weight) if w is not None)
    if sample_weight is not None and loss_weights is None:
        digest.update(b"sample weights only")
    for array in (x, y) + weights:
        array = np.ascontiguousarray(array)
        digest.update(repr((array.shape, array.dtype.str)).encode())
        digest.update(array.data)
    return digest.hexdigest()


def _save_checkpoint(filename: str, state: dict, fingerprint: str) -> None:
    """Write the training ``state`` (as numpy arrays) atomically."""
    import jax
    import joblib

    saved = dict(state, fingerprint=fingerprint)
    for key in ("carry", "best", "key"):
        saved[key] = jax.tree_util.tree_map(np.asarray, state[key])
    temporary = f"{filename}.tmp{os.getpid()}"
    with open(temporary, "wb") as file:
        joblib.dump(saved, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, filename)


def _load_checkpoint(filename: str, fingerprint: str, dtype) -> dict:
    import jax
    import jax.numpy as jnp
    import joblib

    state = joblib.load(filename)
    if state.pop("fingerprint") != fingerprint:
        raise ValueError(
            f"{filename} is the checkpoint of another training (configuration or data): "
            "remove it to start afresh"
        )
    params, m, v, t = state["carry"]
    floats = lambda tree: jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype), tree)  # noqa: E731
    state["carry"] = (floats(params), floats(m), floats(v), jnp.asarray(t, jnp.int32))
    state["best"] = floats(state["best"])
    state["key"] = jnp.asarray(state["key"])
    return state
