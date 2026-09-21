r"""Does a JAX port of the surrogate's inner loop accelerate prediction?

Compares the numpy pipeline against the JAX port in
:mod:`mlgw_bns.jax_predict` on the two stages that have been ported so
far --- the per-mode parameters-to-residuals regression
(:meth:`ModeModel.predict_residuals_bulk`: RBF kernel ridge + PCA
reconstruction) and the cubic-spline resampling --- for a single
parameter point and for batches of increasing size evaluated with
``jax.vmap``.

The point of the batch axis: parameter estimation evaluates the waveform
many thousands of times, and JAX's win (if any) is in amortising the
dispatch / compilation and letting XLA batch the linear algebra, not in a
single call.

Run with::

    python visualization/benchmark_jax_prediction.py [--batches 1 8 64 512 4096]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import jax
import jax.numpy as jnp
from scipy.interpolate import CubicSpline

from mlgw_bns.dataset_generation import ParameterSet
from mlgw_bns.jax_predict import make_cubic_spline_jax, mode_model_to_jax_residuals
from mlgw_bns.model import Model


def _best(fn, reps: int) -> float:
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.min(times)) * 1e3  # ms


def bench_residuals(model: Model, batches: list[int], reps: int) -> None:
    modes = model.modes
    jax_fns = [mode_model_to_jax_residuals(model.mode_models[m]) for m in modes]

    @jax.jit
    def jax_all(arr):
        return [jax.vmap(f)(arr) for f in jax_fns]

    def numpy_all(arr):
        return [
            model.mode_models[m]
            .predict_residuals_bulk(ParameterSet(arr), model.mode_models[m].nn)
            .combined
            for m in modes
        ]

    rng = np.random.default_rng(0)
    centre = next(model.dataset.make_parameter_generator(1)).array

    # accuracy check on a single point
    arr1 = centre[None, :] * (1 + 0.03 * rng.standard_normal((1, 5)))
    ref = np.concatenate([c[0] for c in numpy_all(arr1)])
    got = np.concatenate([np.asarray(c[0]) for c in jax_all(jnp.asarray(arr1))])
    print(
        f"residuals: max relative JAX-vs-numpy difference "
        f"{np.abs(got - ref).max() / np.abs(ref).max():.1e} "
        "(kernel-ridge conditioning; see module docstring)\n"
    )

    print(f"{'batch':>7}  {'numpy':>12}  {'jax.vmap':>12}  {'speedup':>8}")
    for b in batches:
        arr = centre[None, :] * (1 + 0.05 * rng.standard_normal((b, 5)))
        ja = jnp.asarray(arr)
        jax.block_until_ready(jax_all(ja))  # compile for this shape
        numpy_all(arr)
        tn = _best(lambda: numpy_all(arr), reps)
        tj = _best(lambda: jax.block_until_ready(jax_all(ja)), reps)
        print(
            f"{b:>7}  {tn:>9.2f} ms  {tj:>9.2f} ms  {tn / tj:>7.1f}x"
            f"   ({tn / b * 1e3:.1f} vs {tj / b * 1e3:.1f} us/pt)"
        )


def bench_spline(model: Model, batches: list[int], reps: int) -> None:
    mode_model = model.mode_models[model.modes[0]]
    knots = np.asarray(
        mode_model.dataset.frequencies_hz[
            mode_model.downsampling_indices.phase_indices
        ]
    )
    query = np.linspace(knots[0], knots[-1], 2000)
    query_j = jnp.asarray(query)

    spline = make_cubic_spline_jax(knots)
    jit_spline = jax.jit(spline)
    vmap_spline = jax.jit(jax.vmap(spline, in_axes=(0, None)))

    rng = np.random.default_rng(1)

    def values(n):
        return np.cumsum(np.abs(rng.standard_normal((n, len(knots)))), axis=1)

    y1 = values(1)[0]
    got = np.asarray(jit_spline(jnp.asarray(y1), query_j))
    ref = CubicSpline(knots, y1)(query)
    print(
        f"\nspline: max |JAX(natural) - scipy(not-a-knot)| = "
        f"{np.abs(got - ref).max():.1e} (boundary-condition gap)\n"
    )

    print(f"{'batch':>7}  {'scipy':>12}  {'jax.vmap':>12}  {'speedup':>8}")
    for b in batches:
        yv = values(b)
        yj = jnp.asarray(yv)
        jax.block_until_ready(vmap_spline(yj, query_j))
        ts = _best(
            lambda: [
                CubicSpline(knots, yv[i], extrapolate=False)(query) for i in range(b)
            ],
            max(reps // 2, 2),
        )
        tj = _best(lambda: jax.block_until_ready(vmap_spline(yj, query_j)), reps)
        print(
            f"{b:>7}  {ts:>9.2f} ms  {tj:>9.2f} ms  {ts / tj:>7.1f}x"
            f"   ({ts / b * 1e3:.1f} vs {tj / b * 1e3:.1f} us/pt)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64, 512, 4096])
    parser.add_argument("--reps", type=int, default=7)
    args = parser.parse_args()

    model = Model.default_for_testing("default_hom")

    print("=== parameters -> residuals (kernel ridge + PCA), 4 modes ===")
    bench_residuals(model, args.batches, args.reps)

    print("\n=== cubic-spline resampling (one mode's phase grid -> 2000 points) ===")
    bench_spline(model, args.batches, args.reps)


if __name__ == "__main__":
    main()
