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


def coordinates(x, name):
    x = np.array(x, dtype=float)
    if name == "log":
        x[:, 1:3] = np.log1p(x[:, 1:3])
    elif name == "sqrt":
        x[:, 1:3] = np.sqrt(x[:, 1:3])
    elif name == "cbrt":
        x[:, 1:3] = np.cbrt(x[:, 1:3])
    elif name != "linear":
        raise ValueError(name)
    return x


def jacobian(x, name):
    """d(coordinate)/d(parameter), (n, 5)."""
    j = np.ones_like(x)
    if name == "log":
        j[:, 1:3] = 1 / (1 + x[:, 1:3])
    elif name == "sqrt":
        j[:, 1:3] = 0.5 / np.sqrt(x[:, 1:3])
    elif name == "cbrt":
        j[:, 1:3] = np.cbrt(x[:, 1:3]) ** -2 / 3
    return j


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


def kernels(spec, cache, metric_data, name, x, xv, cached):
    """``gamma -> (training kernel, validation kernel)`` of a spec, and a label."""
    parts = spec.split(":")
    kind, coord = (parts[0], parts[1]) if len(parts) > 1 else ("stationary", parts[0])
    scaler = StandardScaler().fit(coordinates(x, coord))
    z, zv = (scaler.transform(coordinates(v, coord)) for v in (x, xv))
    if kind == "stationary":
        a, b = z, zv
    elif kind == "learned":
        key = (coord, len(x))
        if key not in cached:
            # on the (2,2), whatever the mode regressed
            targets = StandardScaler().fit_transform(cache["l2_m2_train"][: len(x)])
            cached[key] = learn_linear(z, targets)
        a, b = normalize(z @ cached[key], zv @ cached[key])
    else:
        metric = Metric(metric_data, scaler, coord, parts[2], parts[3], float(parts[4]), name)
        if kind == "constant":
            lower = linear_map(metric.g.mean(axis=0))
            a, b = normalize(z @ lower, zv @ lower)
        elif kind == "embed":
            key = (spec, len(x)) if parts[3] == "sum" else (spec, len(x), name)
            if key not in cached:
                cached[key] = fit_embedding(metric)
            phi = cached[key]
            a, b = normalize(phi(z), phi(zv))
        elif kind == "gibbs":
            sigma = fit_local(metric)
            sa, sb = sigma(z), sigma(zv)
            q, p = gibbs_parts(z, sa, z, sa)
            qv, pv = gibbs_parts(zv, sb, z, sa)
            # mean squared distance as that of standardized inputs (10)
            scale = 10 / np.mean(q[: 1024, : 1024])
            return lambda gamma: (np.exp(p - gamma * scale * q), np.exp(pv - gamma * scale * qv))
        else:
            raise ValueError(spec)
    dt, dv = euclidean_distances(a, squared=True), euclidean_distances(b, a, squared=True)
    return lambda gamma: (np.exp(-gamma * dt), np.exp(-gamma * dv))


def summary(e, share):
    n_select = len(share) // SELECT_FRACTION
    rest = slice(n_select, None)
    out = {}
    for key, v in e.items():
        out.update({f"{key}_{q}": float(np.quantile(v[rest], p)) for q, p in (("50", .5), ("90", .9), ("99", .99), ("max", 1.0))})
    out["weighted_99"] = float(np.quantile((e["mismatch"] * share)[rest], .99))
    out["weighted_mean"] = float(np.mean((e["mismatch"] * share)[rest]))
    # what the width is chosen on: the mean log mismatch, and its tail
    out["select"] = float(np.mean(np.log10(e["mismatch"][:n_select] + 1e-18)) + 0.5 * np.log10(np.quantile(e["mismatch"][:n_select], .99)))
    return out


def fit(args):
    cache = dict(np.load(args.cache))
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
                    path = out / f"{spec.replace(':', '_')}__{name}__{n}__{offset}.json"
                    if path.exists():
                        continue
                    start = time.time()
                    rows = slice(offset, offset + n)
                    x, xv = cache["x_train"][rows], cache["x_validation"]
                    sub = {**cache, "l2_m2_train": cache["l2_m2_train"][rows]} if "l2_m2_train" in cache else cache
                    kernel = kernels(spec, sub, metric_data, name, x, xv, cached.setdefault(offset, {}))
                    scaler = StandardScaler().fit(cache[f"{name}_train"][rows])
                    targets = scaler.transform(cache[f"{name}_train"][rows])
                    weight = cache.get(f"{name}_weight")
                    weight = None if weight is None else weight[rows]
                    rng = np.random.default_rng(0).choice(n, 256, replace=False)
                    results = []
                    for gamma in args.gammas:
                        k, kv = kernel(gamma)
                        noise_rows = k[rng].copy()
                        dual, selection = kernel_ridge_leave_one_out(
                            k, targets, sample_weight=weight, noise_rows=noise_rows)
                        e = errors(cache, name, m, scaler.inverse_transform(kv @ dual))
                        results.append({"gamma": gamma, "dual": float(np.abs(dual).sum(axis=0).max()),
                                        "loo": float(np.median(selection.expected_error)), **summary(e, cache[f"{name}_share"])})
                        del k, kv
                    best = min(results, key=lambda r: r["select"])
                    path.write_text(json.dumps({"spec": spec, "mode": name, "n": n, "offset": offset, "best": best, "all": results}))
                    print(f"{spec:>38} {name} {n:6d} ({time.time() - start:.0f} s): gamma {best['gamma']}, mismatch median "
                          f"{best['mismatch_50']:.2e} 99% {best['mismatch_99']:.2e}, unaligned {best['unaligned_50']:.2e}, "
                          f"time {best['time_50'] * 1e6:.2f} us 99% {best['time_99'] * 1e6:.1f}, dual {best['dual']:.1e}", flush=True)


def table(args):
    rows = [json.loads(p.read_text()) for p in sorted(Path(args.out).glob("*.json"))]
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
    c.set_defaults(run=fit)
    c = commands.add_parser("truncate")
    c.add_argument("out")
    c.add_argument("--model", default="default_hom_lambda2000")
    c.add_argument("--n", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    c.set_defaults(run=truncate)
    c = commands.add_parser("table")
    c.add_argument("out")
    c.set_defaults(run=table)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
