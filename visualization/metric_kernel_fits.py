r"""Stage 2 of the metric study (``metric_kernel_study.py``): kernel ridge on
fixed training sets, with distances from the metric of the waveforms.

::

    F=visualization/metric_kernel_fits.py
    python $F prepare CACHE.npz --dataset DIR --modes l2_m2 l2_m1 l3_m3   # components, validation errors' operators
    python $F fit CACHE.npz --metric METRIC.npz --out DIR --n 1024 4096 --specs linear sqrt ...
    python $F table DIR

A *spec* is how the parameters become the inputs of the kernel:

* ``linear``, ``log``, ``sqrt``, ``cbrt``: the coordinates of Lambda, standardized;
* ``constant:COORD:KIND:SCOPE:BETA``: then a linear map whose Gram matrix is the
  mean metric over the box, tempered (its eigenvalues, relative to the largest
  one, to the power BETA, plus a floor: BETA=0 is ``COORD``);
* ``embed:COORD:KIND:SCOPE:BETA``: then a small network whose pullback metric
  approximates the (tempered) metric point by point;
* ``gibbs:COORD:KIND:SCOPE:BETA``: no map; the nonstationary Gaussian kernel of
  Paciorek and Schervish, with the inverse of the local metric;
* ``learned:COORD``: a linear map, triangular, minimizing the leave-one-out
  error of the (2,2) on 1024 training waveforms.

KIND is ``residual`` or ``residual_shifted``, SCOPE ``sum`` or ``mode`` (the
metric of the mode being regressed). All maps are normalized to the total
variance of standardized inputs, so that the kernel widths compare.

The width is selected, for each regressor, on the first quarter of the validation
waveforms, and errors are those of the others. Errors are of the regression alone
(no truncation of the principal components), per waveform, in the (2,2)-like
mismatch of the mode with ET at 2.8 solar masses: ``mismatch`` optimized over
time and phase (at leading order in the error), ``unaligned`` not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

import metric_kernel_study as study  # noqa: E402
from modes_scale import parse_mode  # noqa: E402

from mlgw_bns.higher_order_modes import Mode  # noqa: E402
from mlgw_bns.modes_dataset import ShardedModesDataset  # noqa: E402
from mlgw_bns.neural_network import kernel_ridge_leave_one_out  # noqa: E402

GAMMAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
SELECT_FRACTION = 4  # one validation waveform in this many selects the width
FLOOR = 1e-3  # of the tempered eigenvalues, relative to the largest
LN10 = np.log(10)


# -- data ---------------------------------------------------------------------


def prepare(args):
    train, validation = ShardedModesDataset(args.dataset + "/train"), ShardedModesDataset(args.dataset + "/validation")
    pca = train.principal_components(args.pca_size, args.components)
    model = train.model()
    for mode in train.modes:
        model.mode_models[mode].pca_data = pca[mode]
    f, weights = study.quadrature(None, model)
    w = weights["ET_5_M2.8"][0]
    arrays = {"frequencies": f, "weights": w, "modes": np.array(args.modes)}
    power = {}
    for mode in train.modes:
        mode_model = model.mode_models[mode]
        di = mode_model.downsampling_indices
        hz = np.asarray(mode_model.dataset.frequencies_hz)
        reference = mode_model.dataset.waveform_generator.post_newtonian_amplitude(
            mode_model.dataset.amplitude_reference_parameters, mode_model.dataset.frequencies[di.amplitude_indices])
        knots = [hz[np.asarray(i)] for i in (di.amplitude_indices, di.phase_indices)]
        to_grid = [CubicSpline(k, np.eye(len(k)), axis=0)(f) for k in knots]
        data = pca[mode]
        name = f"l{mode.l}_m{mode.m}"
        xv, cv, weights_v = validation.reduced(model, mode, args.n_validation)
        residual = cv @ (data.eigenvectors * data.principal_components_scaling).T + data.mean
        amplitude = (residual[:, : len(knots[0])] * reference) @ to_grid[0].T
        power[name] = np.sum(amplitude**2 * w, axis=1)
        if tuple(mode) not in [tuple(m) for m in args.modes]:
            continue
        arrays["x_validation"] = xv
        basis = (data.eigenvectors * data.principal_components_scaling).T  # (components, nodes)
        arrays[f"{name}_to_amplitude"] = (basis[:, : len(knots[0])] * reference) @ to_grid[0].T
        arrays[f"{name}_to_phase"] = basis[:, len(knots[0]):] @ to_grid[1].T
        # the slope of the phase at the top knot: minus 2 pi times the merger time
        slope = CubicSpline(knots[1], np.eye(len(knots[1])), axis=0)(knots[1][-1], 1)
        arrays[f"{name}_to_time"] = -(basis[:, len(knots[0]):] @ slope) / (2 * np.pi)
        arrays[f"{name}_amplitude"] = amplitude.astype(np.float32)
        arrays[f"{name}_validation"] = cv
        x, c, weight = train.reduced(model, mode, args.n_train)
        arrays["x_train"], arrays[f"{name}_train"] = x, c
        if weight is not None:
            arrays[f"{name}_weight"] = weight
        print(name, c.shape, cv.shape, flush=True)
    total = sum(power.values())
    for name in power:
        arrays[f"{name}_share"] = power[name] / total
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.cache, **arrays)


def errors(cache, name, m, predicted):
    """Per validation waveform: mismatches (aligned and not) and the merger
    time error (s)."""
    f, w = cache["frequencies"], cache["weights"]
    out = {"mismatch": [], "unaligned": [], "time": []}
    to_a, to_p, to_t = (cache[f"{name}_to_{k}"] for k in ("amplitude", "phase", "time"))
    truth, amplitude = cache[f"{name}_validation"], cache[f"{name}_amplitude"]
    for start in range(0, len(truth), 512):
        rows = slice(start, start + 512)
        d = predicted[rows] - truth[rows]
        a = amplitude[rows].astype(float)
        da, dp = d @ to_a, a * (d @ to_p)
        power = np.sum(w * a**2, axis=1)
        # the amplitude mismatch does not see an overall factor
        da = da - a * (np.sum(w * a * da, axis=1) / power)[:, None]
        square = np.sum(w * (da**2 + dp**2), axis=1)
        columns = np.stack([a * m, a * f * 2 * np.pi * 1e-3], axis=1)  # phase, time (ms)
        gram = np.einsum("nik,njk,k->nij", columns, columns, w)
        rhs = np.einsum("nik,nk,k->ni", columns, dp, w)
        explained = np.einsum("ni,ni->n", rhs, np.linalg.solve(gram, rhs[..., None])[..., 0])
        out["mismatch"].append(0.5 * np.maximum(square - explained, 0) / power)
        out["unaligned"].append(0.5 * square / power)
        out["time"].append(np.abs(d @ to_t))
    return {k: np.concatenate(v) for k, v in out.items()}


# -- kernels ------------------------------------------------------------------


LAMBDA_POWERS = {"linear": (1.0, 1.0), "sqrt": (1 / 2, 1 / 2), "cbrt": (1 / 3, 1 / 3)}
SPIN_WARP = ((-0.13, 0.4), (-0.05, 0.5))  # center and width of the metric's trend along each spin


def warp(x, name):
    """The coordinates ``name`` of the parameters, and their derivatives: that
    of the Lambdas (``log``, a power, or ``pA,B`` for two exponents), then any
    of ``+logq``, ``+delta`` ((q-1)/(q+1)), ``+eta``, ``+asinh`` (the spins)."""
    first, *rest = name.split("+")
    x = np.array(x, dtype=float)
    u, j = x.copy(), np.ones_like(x)
    if first == "log":
        u[:, 1:3], j[:, 1:3] = np.log1p(x[:, 1:3]), 1 / (1 + x[:, 1:3])
    else:
        powers = LAMBDA_POWERS.get(first) or tuple(float(v) for v in first[1:].split(","))
        for k, a in zip((1, 2), powers):
            u[:, k], j[:, k] = x[:, k] ** a, a * x[:, k] ** (a - 1)
    q = x[:, 0]
    for part in rest:
        if part == "logq":
            u[:, 0], j[:, 0] = np.log(q), 1 / q
        elif part == "delta":
            u[:, 0], j[:, 0] = (q - 1) / (q + 1), 2 / (q + 1) ** 2
        elif part == "eta":
            u[:, 0], j[:, 0] = q / (1 + q) ** 2, (1 - q) / (1 + q) ** 3
        elif part == "asinh":
            for k, (center, width) in zip((3, 4), SPIN_WARP):
                t = (x[:, k] - center) / width
                u[:, k], j[:, k] = np.arcsinh(t), 1 / (width * np.sqrt(1 + t**2))
        else:
            raise ValueError(name)
    return u, j


def coordinates(x, name):
    return warp(x, name)[0]


def jacobian(x, name):
    """d(coordinate)/d(parameter), (n, 5)."""
    return warp(x, name)[1]


def tempered(g, beta, floor=FLOOR):
    values, vectors = np.linalg.eigh(g)
    values = np.maximum(values, 0) / values[..., -1:]
    return np.einsum("...ij,...j,...kj->...ik", vectors, values**beta + floor, vectors)


class Metric:
    """The metric of the mode (or of all) at the points where it was computed,
    in the standardized coordinates of the training set, tempered.
    Tempering is of each point's matrix relative to its largest eigenvalue; the
    scale of the metric, to the power BETA too, is put back."""

    def __init__(self, data, scaler, coord, kind, scope, beta, name):
        x = data["parameters"]
        modes = [f"l{l}_m{m}" for l, m in data["modes"]]
        index = 0 if scope == "sum" else 1 + modes.index(name)
        g = data[f"ET_5 {kind}"][:, index]
        if scope == "mode":  # per unit power of the mode
            g = g / data["ET_5 power"][:, modes.index(name)][:, None, None]
        scale = scaler.scale_ / jacobian(x, coord)
        g = g * scale[:, :, None] * scale[:, None, :]
        top = np.linalg.eigvalsh(g)[:, -1]
        self.z = scaler.transform(coordinates(x, coord))
        relative = (top / np.median(top)) ** beta
        self.g = tempered(g, beta) * relative[:, None, None]
        # bounded by 30 of the median scale either way: the odd modes vanish at q = 1
        bound = np.clip(relative, 1 / 30, 30) / relative
        self.g *= bound[:, None, None]


def normalize(a, b):
    """The inputs ``a`` (training) and ``b`` to the total variance of 5
    standardized ones."""
    scale = np.sqrt(5 / a.var(axis=0).sum())
    return a * scale, b * scale


def linear_map(gram):
    values, vectors = np.linalg.eigh(gram)
    return vectors * np.sqrt(np.maximum(values, 0))


def fit_embedding(metric, width=48, dimension=8, steps=6000, seed=0, decay=1e-4):
    """A network R^5 -> R^dimension (linear plus a perceptron) whose pullback
    metric J^T J approximates ``metric.g``: affine-invariant loss. Returns
    the map and its relative error on a held out eighth of the points."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    z, g = jnp.asarray(metric.z), np.asarray(metric.g)
    root = jnp.asarray(study.matrix_function(g, lambda v: np.maximum(v, 1e-12) ** -0.5))
    held = np.arange(len(g)) % 8 == 0
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    mean = study.matrix_function(study.matrix_function(g, np.log).mean(0), np.exp)
    base = np.zeros((5, dimension))
    base[:, :5] = linear_map(mean)
    params = {
        "linear": jnp.asarray(base),
        "w1": jax.random.normal(keys[0], (5, width)) / 5**0.5, "b1": jax.random.normal(keys[1], (width,)),
        "w2": jax.random.normal(keys[2], (width, width)) / width**0.5, "b2": jnp.zeros(width),
        "w3": 1e-2 * jax.random.normal(keys[3], (width, dimension)),
    }

    def phi(p, x):
        h = jnp.tanh(x @ p["w1"] + p["b1"])
        h = jnp.tanh(h @ p["w2"] + p["b2"])
        return x @ p["linear"] + h @ p["w3"]

    def mismatch(p, x, r):
        j = jax.jacfwd(lambda y: phi(p, y))(x)
        s = r @ (j.T @ j) @ r
        # symmetric in a factor up or down: s + s^-1 - 2, with a little ridge
        return jnp.trace(s) + jnp.trace(jnp.linalg.inv(s + 1e-6 * jnp.eye(5))) - 10

    def loss(p, x, r):
        return jnp.mean(jnp.log1p(jax.vmap(lambda a, b: mismatch(p, a, b))(x, r))) + decay * sum(
            jnp.sum(p[k] ** 2) for k in ("w1", "w2", "w3")) / width

    tree = jax.tree_util.tree_map

    @jax.jit
    def step(p, first, second, i):  # Adam, with a cosine schedule
        grad = jax.grad(loss)(p, z[~held], root[~held])
        first = tree(lambda a, b: 0.9 * a + 0.1 * b, first, grad)
        second = tree(lambda a, b: 0.999 * a + 0.001 * b**2, second, grad)
        rate = 3e-3 * 0.5 * (1 + jnp.cos(jnp.pi * i / steps))
        p = tree(lambda v, a, b: v - rate * (a / (1 - 0.9 ** (i + 1))) / (jnp.sqrt(b / (1 - 0.999 ** (i + 1))) + 1e-8),
                 p, first, second)
        return p, first, second

    first, second = tree(jnp.zeros_like, params), tree(jnp.zeros_like, params)
    for i in range(steps):
        params, first, second = step(params, first, second, i)
    ratio = lambda which: np.sqrt(np.mean([  # noqa: E731
        np.sum((0.5 * np.log(np.linalg.eigvalsh(np.asarray(r @ (j.T @ j) @ r)))) ** 2) / 5
        for j, r in zip(np.asarray(jax.vmap(jax.jacfwd(lambda y: phi(params, y)))(z[which])), np.asarray(root[which]))
    ]))
    print(f"    embedding: rms log length ratio {ratio(~held):.3f}, held out {ratio(held):.3f}", flush=True)
    return lambda x: np.asarray(phi(params, jnp.asarray(x)))


def polynomial(z, degree=3):
    from sklearn.preprocessing import PolynomialFeatures

    return PolynomialFeatures(degree).fit_transform(z)


def fit_local(metric):
    """log of the metric as cubic polynomials: ``z -> Sigma = g^-1``. Returns
    it, printing the rms log length ratio of a fit without an eighth."""
    logs = study.matrix_function(metric.g, np.log)
    iu = np.triu_indices(5)
    ridge = 1e-3

    def solve(rows):
        p = polynomial(metric.z[rows])
        return np.linalg.solve(p.T @ p + ridge * np.eye(p.shape[1]), p.T @ logs[rows][:, iu[0], iu[1]])

    def evaluate(coefficients, z, sign):
        values = np.clip(polynomial(z) @ coefficients, -20, 20)
        m = np.zeros((len(z), 5, 5))
        m[:, iu[0], iu[1]] = m[:, iu[1], iu[0]] = values
        return study.matrix_function(m, lambda v: np.exp(sign * np.clip(v, -14, 14)))

    held = np.arange(len(logs)) % 8 == 0
    check = evaluate(solve(~held), metric.z[held], 1)
    print(f"    local metric: rms log length ratio held out {np.sqrt(np.mean(study.log_ratios(check, metric.g[held])**2)):.3f}", flush=True)
    coefficients = solve(np.ones(len(logs), bool))
    return lambda z: evaluate(coefficients, z, -1)


def gibbs_parts(za, sa, zb, sb, block=256):
    """Squared distance and log of the prefactor of the nonstationary kernel."""
    q = np.empty((len(za), len(zb)))
    log_prefactor = np.empty_like(q)
    log_a, log_b = np.linalg.slogdet(sa)[1], np.linalg.slogdet(sb)[1]

    def one(start):
        rows = slice(start, start + block)
        mean = 0.5 * (sa[rows][:, None] + sb[None])
        difference = za[rows][:, None] - zb[None]
        lower = np.linalg.cholesky(mean)
        # forward substitution, vectorized
        u = np.empty_like(difference)
        for k in range(5):
            u[..., k] = (difference[..., k] - np.einsum("...j,...j->...", lower[..., k, :k], u[..., :k])) / lower[..., k, k]
        q[rows] = np.einsum("...k,...k->...", u, u)
        log_det = 2 * np.log(np.diagonal(lower, axis1=-2, axis2=-1)).sum(-1)
        log_prefactor[rows] = 0.25 * log_a[rows][:, None] + 0.25 * log_b[None] - 0.5 * log_det

    from concurrent.futures import ThreadPoolExecutor

    from threadpoolctl import threadpool_limits

    with threadpool_limits(1), ThreadPoolExecutor() as pool:
        list(pool.map(one, range(0, len(za), block)))
    return q, log_prefactor


def learn_linear(z, targets, steps=150, n=1024, components=20):
    """Triangular L minimizing the mean log leave-one-out error of ``targets``
    for inputs ``z L``, kernel exp(-|.|^2) and penalty 1e-9."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    jax.config.update("jax_enable_x64", True)
    z, y = jnp.asarray(z[:n]), jnp.asarray(targets[:n, :components])
    il = np.tril_indices(5)

    def loss(theta):
        lower = jnp.zeros((5, 5)).at[il].set(theta)
        u = z @ lower
        squares = jnp.sum(u * u, axis=1)
        k = jnp.exp(-jnp.maximum(squares[:, None] + squares[None] - 2 * u @ u.T, 0)) + 1e-9 * jnp.eye(len(z))
        inverse = jnp.linalg.inv(k)
        residual = (inverse @ y) / jnp.diagonal(inverse)[:, None]
        return jnp.mean(jnp.log(jnp.mean(residual**2, axis=0)))

    value = jax.jit(jax.value_and_grad(loss))
    start = np.zeros(len(il[0]))
    start[il[0] == il[1]] = 0.1**0.5
    result = minimize(lambda t: tuple(np.asarray(v, float) for v in value(t)), start, jac=True, method="L-BFGS-B",
                      options=dict(maxiter=steps))
    print(f"    learned map: loss {float(value(start)[0]):.2f} -> {result.fun:.2f} in {result.nit} steps", flush=True)
    lower = np.zeros((5, 5))
    lower[il] = result.x
    return lower


# -- fits ---------------------------------------------------------------------


def standardized(coord, x, xv):
    scaler = StandardScaler().fit(coordinates(x, coord))
    return scaler, scaler.transform(coordinates(x, coord)), scaler.transform(coordinates(xv, coord))


def split(spec):
    parts = spec.split(":")
    return parts if len(parts) > 1 else ["stationary", parts[0]]


def mapped(spec, cache, metric_data, name, x, xv, cached):
    """The inputs (training, validation) of the stationary kernel of a spec."""
    parts = split(spec)
    kind, coord = parts[:2]
    scaler, z, zv = standardized(coord, x, xv)
    if kind == "stationary":
        return z, zv
    if kind == "learned":
        key = (coord, len(x))
        if key not in cached:
            # on the (2,2), whatever the mode regressed
            targets = StandardScaler().fit_transform(cache["l2_m2_train"][: len(x)])
            cached[key] = learn_linear(z, targets)
        return normalize(z @ cached[key], zv @ cached[key])
    metric = Metric(metric_data, scaler, coord, parts[2], parts[3], float(parts[4]), name)
    if kind == "constant":
        lower = linear_map(metric.g.mean(axis=0))
        return normalize(z @ lower, zv @ lower)
    if kind == "embed":
        key = (spec, len(x)) if parts[3] == "sum" else (spec, len(x), name)
        if key not in cached:
            cached[key] = fit_embedding(metric)
        return normalize(cached[key](z), cached[key](zv))
    raise ValueError(spec)


def kernels(spec, cache, metric_data, name, x, xv, cached):
    """``gamma -> (training kernel, validation kernel)`` of a spec."""
    parts = split(spec)
    if parts[0] == "gibbs":
        scaler, z, zv = standardized(parts[1], x, xv)
        sigma = fit_local(Metric(metric_data, scaler, parts[1], parts[2], parts[3], float(parts[4]), name))
        sa, sb = sigma(z), sigma(zv)
        q, p = gibbs_parts(z, sa, z, sa)
        qv, pv = gibbs_parts(zv, sb, z, sa)
        # mean squared distance as that of standardized inputs (10)
        scale = 10 / np.mean(q[: 1024, : 1024])
        return lambda gamma: (np.exp(p - gamma * scale * q), np.exp(pv - gamma * scale * qv))
    a, b = mapped(spec, cache, metric_data, name, x, xv, cached)
    dt, dv = euclidean_distances(a, squared=True), euclidean_distances(b, a, squared=True)
    return lambda gamma: (np.exp(-gamma * dt), np.exp(-gamma * dv))


def farthest_points(points, n, seed):
    """Indices of ``n`` of the ``points``, each the farthest from those before
    it (the first one drawn), in random order."""
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(points)))]
    distance = np.full(len(points), np.inf)
    for _ in range(n - 1):
        distance = np.minimum(distance, np.sum((points - points[chosen[-1]]) ** 2, axis=1))
        chosen.append(int(np.argmax(distance)))
    return rng.permutation(chosen)


def boxes(x):
    """Regions of the parameters where errors are given apart."""
    q, l1, l2, s1, s2 = x.T
    return {
        "q_near_1": q < 1.15,
        "corner": (q > 2) & (l1 > 1000) & (l2 < 500),
        "small_lambda": np.minimum(l1, l2) < 50,
        "spin_edge": np.maximum(abs(s1), abs(s2)) > 0.45,
        "high_q": q > 2.5,
    }


def summary(e, share, x):
    n_select = len(share) // SELECT_FRACTION
    rest = slice(n_select, None)
    out = {}
    for key, v in e.items():
        out.update({f"{key}_{q}": float(np.quantile(v[rest], p)) for q, p in (("50", .5), ("90", .9), ("99", .99), ("max", 1.0))})
    out["weighted_99"] = float(np.quantile((e["mismatch"] * share)[rest], .99))
    out["weighted_mean"] = float(np.mean((e["mismatch"] * share)[rest]))
    for box, inside in boxes(x[rest]).items():
        out[f"{box}_n"] = int(inside.sum())
        for q, p in (("50", .5), ("99", .99)):
            out[f"{box}_mismatch_{q}"] = float(np.quantile(e["mismatch"][rest][inside], p))
            out[f"{box}_weighted_{q}"] = float(np.quantile((e["mismatch"] * share)[rest][inside], p))
            out[f"{box}_time_{q}"] = float(np.quantile(e["time"][rest][inside], p))
    # what the width is chosen on: the mean log mismatch, and its tail
    out["select"] = float(np.mean(np.log10(e["mismatch"][:n_select] + 1e-18)) + 0.5 * np.log10(np.quantile(e["mismatch"][:n_select], .99)))
    return out


def fit(args):
    cache = dict(np.load(args.cache))
    if args.also:
        with np.load(args.also) as more:  # of the same training set
            assert np.array_equal(more["x_train"], cache["x_train"])
            cache["l2_m2_train"] = more["l2_m2_train"]
    metric_data = dict(np.load(args.metric)) if args.metric else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modes = [tuple(m) for m in (args.modes or cache["modes"])]
    cached: dict = {}
    for n in args.n:
        for spec in args.specs:
            for offset in args.offsets:
                for l, m in modes:
                    name = f"l{l}_m{m}"
                    path = out / f"{spec.replace(':', '_')}__{name}__{n}__{offset}{'__fps' if args.placement else ''}.json"
                    if path.exists():
                        continue
                    start = time.time()
                    if args.placement:  # the offset is the seed
                        pool = cache["x_train"]
                        key = ("placement", n)
                        if key not in (store := cached.setdefault(offset, {})):
                            points, _ = mapped(args.placement, cache, metric_data, name, pool, pool[:1], {})
                            store[key] = farthest_points(points, n, offset)
                        rows = store[key]
                    else:
                        rows = slice(offset, offset + n)
                    x, xv = cache["x_train"][rows], cache["x_validation"]
                    sub = {**cache, "l2_m2_train": cache["l2_m2_train"][rows]} if "l2_m2_train" in cache else cache
                    kernel = kernels(spec, sub, metric_data, name, x, xv, cached.setdefault(offset, {}))
                    scaler = StandardScaler().fit(cache[f"{name}_train"][rows])
                    targets = scaler.transform(cache[f"{name}_train"][rows])
                    weight = cache.get(f"{name}_weight")
                    weight = None if weight is None else weight[rows]
                    rng = np.random.default_rng(0).choice(n, 256, replace=False)
                    results, predictions = [], {}
                    for gamma in args.gammas:
                        k, kv = kernel(gamma)
                        noise_rows = k[rng].copy()
                        dual, selection = kernel_ridge_leave_one_out(
                            k, targets, sample_weight=weight, noise_rows=noise_rows)
                        e = errors(cache, name, m, predictions.setdefault(gamma, scaler.inverse_transform(kv @ dual)))
                        results.append({"gamma": gamma, "dual": float(np.abs(dual).sum(axis=0).max()),
                                        "loo": float(np.median(selection.expected_error)), **summary(e, cache[f"{name}_share"], xv)})
                        del k, kv
                    best = min(results, key=lambda r: r["select"])
                    if args.save_predictions:
                        np.save(path.with_suffix(".npy"), predictions[best["gamma"]])
                    path.write_text(json.dumps({"spec": spec + ("@fps" if args.placement else ""), "mode": name, "n": n, "offset": offset, "best": best, "all": results}))
                    print(f"{spec:>38} {name} {n:6d} ({time.time() - start:.0f} s): gamma {best['gamma']}, mismatch median "
                          f"{best['mismatch_50']:.2e} 99% {best['mismatch_99']:.2e}, unaligned {best['unaligned_50']:.2e}, "
                          f"time {best['time_50'] * 1e6:.2f} us 99% {best['time_99'] * 1e6:.1f}, dual {best['dual']:.1e}", flush=True)


def table(args):
    rows = [json.loads(p.read_text()) for p in sorted(Path(args.out).glob("*.json"))]
    rows = [r for r in rows if "mode" in r]  # (not the full waveforms)
    for mode in sorted({r["mode"] for r in rows}):
        for n in sorted({r["n"] for r in rows}):
            chosen = [r for r in rows if r["mode"] == mode and r["n"] == n]
            if not chosen:
                continue
            print(f"\n== {mode}, N = {n}:  gamma | mismatch 50% 90% 99% max | unaligned 50% 99% | |dt| us 50% 99% | dual  (geometric mean over {len({r['offset'] for r in chosen})} subsets)")
            specs = {}
            for r in chosen:
                specs.setdefault(r["spec"], []).append(r["best"])
            order = sorted(specs.items(), key=lambda kv: np.mean([np.log(b["mismatch_50"]) for b in kv[1]]))
            for spec, b in order:
                g = lambda key, s=1.0: s * np.exp(np.mean([np.log(v[key]) for v in b]))  # noqa: E731
                print(f"{spec:>40}  {g('gamma'):6.3f} | {g('mismatch_50'):.1e} {g('mismatch_90'):.1e} {g('mismatch_99'):.1e} {g('mismatch_max'):.1e} | "
                      f"{g('unaligned_50'):.1e} {g('unaligned_99'):.1e} | {g('time_50', 1e6):7.2f} {g('time_99', 1e6):7.1f} | {g('dual'):.0e}")


class StoredPredictions:
    """In place of a regressor: its predictions, by the parameters."""

    class hyper:
        pc_exponent = 0

    def __init__(self, x, predictions):
        self.rows = {row.tobytes(): i for i, row in enumerate(x)}
        self.predictions = predictions

    def predict(self, x):
        return self.predictions[[self.rows[row.tobytes()] for row in np.asarray(x, dtype=float)]]


def full(args):
    """Full waveform mismatches (``stored_waveform_mismatches``) of every
    regressor of which all modes were predicted (``fit --save-predictions``), on
    the validation waveforms that did not select the width; first the
    references (validations of models on the same set)."""
    from mlgw_bns.model_validation import stored_waveform_mismatches

    train, validation = ShardedModesDataset(args.dataset + "/train"), ShardedModesDataset(args.dataset + "/validation")
    parameters, residuals = validation.load_residuals(args.n_validation)
    start = args.n_validation // SELECT_FRACTION
    score = slice(start, args.n_validation)
    regions = boxes(parameters[score])

    def report(label, result, file):
        out = {"label": label}
        for region, inside in {"all": np.ones(len(parameters[score]), bool), **regions}.items():
            v = result["full"][inside]
            out[region] = {"n": len(v), **{k: float(np.quantile(v, p)) for k, p in (("50", .5), ("90", .9), ("99", .99), ("max", 1))}}
        Path(file).write_text(json.dumps(out))
        print(label, {k: f"{v['50']:.1e}/{v['99']:.1e}/{v['max']:.1e}" for k, v in out.items() if k != "label"}, flush=True)

    out = Path(args.out)
    for reference in args.reference:
        with np.load(reference) as data:
            assert np.array_equal(data["parameters"][: len(parameters)], parameters)
            # (isotropic inclinations are drawn for the whole set)
            report(Path(reference).stem, {"full": data["mismatches"][score]}, out / f"full_{Path(reference).stem}.json")
            inclinations = data["inclination"][score]
    model = train.model()
    pca = train.principal_components(args.pca_size, args.components)
    for mode in train.modes:
        model.mode_models[mode].pca_data = pca[mode]
    names = [f"l{m.l}_m{m.m}" for m in train.modes]
    stems = sorted(p.stem for p in out.glob("*__l2_m2__*.npy"))
    for i, stem in enumerate(stems):
        file = out / f"full_{stem.replace('__l2_m2', '')}.json"
        paths = [out / f"{stem.replace('l2_m2', name)}.npy" for name in names]
        if i % args.shards != args.shard or file.exists() or not all(p.exists() for p in paths):
            continue
        for mode, path in zip(train.modes, paths):
            model.mode_models[mode].nn = StoredPredictions(np.asarray(parameters, dtype=float), np.load(path))
        result = stored_waveform_mismatches(
            model, parameters[score], {m: v[score] for m, v in residuals.items()}, inclinations, n_jobs=args.jobs)
        report(stem.replace("__l2_m2", ""), result, file)


COORDINATES = {"cbrt": "cbrt_lambda", "cbrt+logq": "cbrt_lambda_log_q"}


class SelectedRows(ShardedModesDataset):
    """A training set of which the regressors are fitted on ``rows`` only."""

    rows = slice(None)

    def reduced(self, model, mode, n):
        return tuple(None if v is None else v[self.rows] for v in super().reduced(model, mode, n))


def constant_map(spec, metric_data, pool, n, seed=0):
    """Of a ``constant`` spec: ``n`` farthest points (their rows in the
    ``pool``), the scaler of their coordinates and the linear map."""
    kind, coord, metric_kind, scope, beta = split(spec)
    assert kind == "constant" and scope == "sum"
    points, _ = mapped(spec, None, metric_data, None, pool, pool[:1], {})
    rows = farthest_points(points, n, seed) if n < len(pool) else np.arange(len(pool))
    scaler, z, _ = standardized(coord, pool[rows], pool[:1])
    lower = linear_map(Metric(metric_data, scaler, coord, metric_kind, scope, float(beta), None).g.mean(axis=0))
    return rows, scaler, lower * np.sqrt(5 / (z @ lower).var(axis=0).sum())


def save_maps(args):
    """The maps of some specs, for ``plot_metric_map.py``: their Gram matrices."""
    metric_data = dict(np.load(args.metric))
    pool = ShardedModesDataset(args.dataset + "/train").load(args.pool, ("parameters",))["parameters"]
    out = {}
    for spec in args.specs:
        _, scaler, lower = constant_map(spec, metric_data, np.asarray(pool, dtype=float), args.n)
        name = split(spec)[1]
        out.update({f"{name}/gram": lower @ lower.T, f"{name}/mean": scaler.mean_, f"{name}/scale": scaler.scale_})
        print(spec, np.round(lower @ lower.T, 3), np.linalg.eigvalsh(lower @ lower.T), sep="\n", flush=True)
    np.savez(args.out, **out)


def model(args):
    """A model of the library of which every mode regresses in the inputs of
    a ``constant`` spec, with one width, trained on ``--n`` farthest points."""
    from mlgw_bns.neural_network import Hyperparameters

    train = SelectedRows(args.dataset + "/train")
    pool = np.load(args.cache)["x_train"]
    train.rows, _, lower = constant_map(args.spec, dict(np.load(args.metric)), pool, args.n, args.seed)
    coord, x = split(args.spec)[1], pool[train.rows]

    def hyperparameters(mode, n):
        hyper = Hyperparameters.default_kernel_ridge(
            len(x), mode=mode, kernel_gamma=args.gamma, input_coordinates=COORDINATES[coord])
        hyper.input_map = lower
        return hyper

    train.train(len(pool), args.out, hyperparameters=hyperparameters, pca_size=args.pca_size, n_components=args.components)
    print(f"{args.out}: {len(x)} points, gamma {args.gamma}", flush=True)


def truncate(args):
    """Copies of a shipped model with its kernel ridge cut to the first N
    training points, and nothing else changed: their predictions are garbage,
    their cost is that of models trained on N waveforms (the coordinates of
    the inputs, and a 5 x 5 map of them, cost nothing next to the kernel)."""
    import shutil

    import mlgw_bns
    from mlgw_bns.model import Model

    data = Path(mlgw_bns.__file__).parent / "data"
    for n in args.n:
        model = Model.default_for_testing(args.model)
        base = Path(args.out) / f"n{n}" / "model"
        base.parent.mkdir(parents=True, exist_ok=True)
        for mode, mode_model in model.mode_models.items():
            name = f"l{mode.l}_m{mode.m}"
            for end in (".yaml", "_arrays.h5"):
                shutil.copy(data / f"{args.model}_{name}{end}", f"{base}_{name}{end}")
            regressor = mode_model.nn.regressor
            # (Fortran order, as the fits leave it)
            regressor.X_fit_ = np.array(regressor.X_fit_[:n])
            regressor.dual_coef_ = np.asfortranarray(regressor.dual_coef_[:n])
            mode_model.nn.save(f"{base}_{name}_nn.pkl")
        size = sum(f.stat().st_size for f in base.parent.glob("*_nn.pkl"))
        print(f"{base}: regressors {size / 1e6:.1f} MB", flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(required=True)
    c = commands.add_parser("prepare")
    c.add_argument("cache")
    c.add_argument("--dataset", required=True)
    c.add_argument("--modes", type=parse_mode, nargs="+", default=[(2, 2), (2, 1), (3, 3)])
    c.add_argument("--n-train", type=int, default=32768)
    c.add_argument("--n-validation", type=int, default=8192)
    c.add_argument("--pca-size", type=int, default=32768)
    c.add_argument("--components", type=int, default=60)
    c.set_defaults(run=prepare)
    c = commands.add_parser("fit")
    c.add_argument("cache")
    c.add_argument("--metric")
    c.add_argument("--out", required=True)
    c.add_argument("--n", type=int, nargs="+", default=[1024, 4096])
    c.add_argument("--offsets", type=int, nargs="+", default=[0])
    c.add_argument("--specs", nargs="+", required=True)
    c.add_argument("--gammas", type=float, nargs="+", default=list(GAMMAS))
    c.add_argument("--modes", type=parse_mode, nargs="+")
    c.add_argument("--also", help="a cache of the same training set with the (2,2) (for learned maps)")
    c.add_argument("--placement", metavar="SPEC", help="train on farthest points of the whole set in this map (offsets are seeds)")
    c.add_argument("--save-predictions", action="store_true")
    c.set_defaults(run=fit)
    c = commands.add_parser("full")
    c.add_argument("out")
    c.add_argument("--dataset", required=True)
    c.add_argument("--reference", nargs="+", required=True, help="validate_*.npz of models on the same validation set")
    c.add_argument("--n-validation", type=int, default=8192)
    c.add_argument("--pca-size", type=int, default=32768)
    c.add_argument("--components", type=int, default=60)
    c.add_argument("--shard", type=int, default=0)
    c.add_argument("--shards", type=int, default=1)
    c.add_argument("--jobs", type=int, default=16)
    c.set_defaults(run=full)
    c = commands.add_parser("truncate")
    c.add_argument("out")
    c.add_argument("--model", default="default_hom_lambda2000")
    c.add_argument("--n", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    c.set_defaults(run=truncate)
    c = commands.add_parser("model", help=model.__doc__)
    c.add_argument("cache")
    c.add_argument("--dataset", required=True)
    c.add_argument("--metric", required=True)
    c.add_argument("--spec", required=True)
    c.add_argument("--n", type=int, required=True)
    c.add_argument("--gamma", type=float, default=0.3)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    c.add_argument("--pca-size", type=int, default=32768)
    c.add_argument("--components", type=int, default=60)
    c.set_defaults(run=model)
    c = commands.add_parser("maps", help=save_maps.__doc__)
    c.add_argument("out")
    c.add_argument("--dataset", required=True)
    c.add_argument("--metric", required=True)
    c.add_argument("--specs", nargs="+", required=True)
    c.add_argument("--pool", type=int, default=32768)
    c.add_argument("--n", type=int, default=8192)
    c.set_defaults(run=save_maps)
    c = commands.add_parser("table")
    c.add_argument("out")
    c.set_defaults(run=table)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
