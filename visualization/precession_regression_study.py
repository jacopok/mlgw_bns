r"""Train and assess the regressed precession angles (a prototype).

:mod:`mlgw_bns.precession_regression` replaces the integration of the PN
spin-precession equations in the JAX precessing waveform by a regressor;
this script builds one and measures it against the integration::

    # integrate the training and validation binaries, fit their envelopes
    python visualization/precession_regression_study.py generate --n 4096 --seed 1 --out train.npz
    python visualization/precession_regression_study.py generate --n 512 --seed 2 --out validation.npz
    # PCA + kernel ridge, optionally refining the envelopes first
    python visualization/precession_regression_study.py train --data train.npz --out regressor.joblib --refine 2
    # errors of zeta and G, and mismatches of the waveforms, against the integration
    python visualization/precession_regression_study.py validate --regressor regressor.joblib --data validation.npz
    # cost of the angles and of the whole waveform, single and batched
    python visualization/precession_regression_study.py benchmark --regressor regressor.joblib

``generate`` takes ~0.15 s a binary on four cores (the integration and two
least-squares fits), ``train`` a few minutes for 4096 binaries, each
``--refine`` iteration ~10 minutes more.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from mlgw_bns.precession_regression import (
    AngleGrid,
    PrecessionRegressor,
    TrainingRanges,
    _carriers,
    _evaluate,
    carrier_table,
    fit_all_envelopes,
    reference_frame,
    refine_envelopes,
    training_data,
)


def generate(args) -> None:
    grid = AngleGrid()
    intrinsic, omega_reference = TrainingRanges().sample(args.n, args.seed)
    start = time.perf_counter()
    x, zeta, g = training_data(grid, intrinsic, omega_reference)
    coefficients, residuals = fit_all_envelopes(grid, intrinsic, omega_reference, x, zeta, g)
    print(
        f"{args.n} binaries in {time.perf_counter() - start:.0f} s; largest fit "
        f"residuals {residuals[:, 0].max():.1e} (zeta), {residuals[:, 1].max():.1e} (G)"
    )
    np.savez(
        args.out, intrinsic=intrinsic, omega_reference=omega_reference,
        coefficients=coefficients, x=x, zeta=zeta, g=g,
    )


def train(args) -> None:
    data = np.load(args.data)
    kwargs = dict(n_components=tuple(args.components), kernel_gamma=args.gamma)
    coefficients = data["coefficients"]
    if args.refine:
        coefficients, _ = refine_envelopes(
            AngleGrid(), data["intrinsic"], data["omega_reference"], data["x"],
            data["zeta"], data["g"], coefficients, iterations=args.refine, **kwargs,
        )
    PrecessionRegressor.train(
        data["intrinsic"], data["omega_reference"], coefficients, **kwargs
    ).save(args.out)


def function_errors(regressor: PrecessionRegressor, data, n: int) -> np.ndarray:
    r"""Largest errors of :math:`\zeta` and :math:`G` of the first ``n``
    binaries of ``data``, below and above :math:`x = -4.5` (a (2, 2)
    frequency of ~250 Hz for a total mass of 2.8): ``(n, 4)``."""
    grid = regressor.grid
    predicted = regressor.predict_coefficients(data["intrinsic"][:n], data["omega_reference"][:n])
    errors = []
    for i in range(n):
        frame = reference_frame(np, grid, data["intrinsic"][i], data["omega_reference"][i])
        x = data["x"][i]
        zeta, g = _evaluate(
            np, grid, predicted[i], _carriers(np, grid, *carrier_table(np, grid, frame), x), x
        )
        d_zeta, d_g = np.abs(zeta - data["zeta"][i]), np.abs(g - data["g"][i])
        high = x > -4.5
        errors.append((d_zeta[~high].max(), d_zeta[high].max(), d_g[~high].max(), d_g[high].max()))
    return np.array(errors)


def waveform_mismatches(regressor: PrecessionRegressor, data, n: int, seed: int = 7):
    """Mismatches, with nothing optimized, of the JAX precessing waveforms
    with the regressed angles against those with the integrated ones, for the
    first ``n`` binaries of ``data`` at random total masses and orientations
    (ET PSD, 20--2048 Hz): the larger of the two polarizations', and the
    parameters."""
    import jax

    from mlgw_bns.batched_precession import batch_arguments
    from mlgw_bns.higher_order_modes import Mode
    from mlgw_bns.model import Model
    from mlgw_bns.model_validation import ValidateModel
    from mlgw_bns.precessing_model import (
        PrecessingModel,
        PrecessingParametersWithExtrinsic,
    )
    from mlgw_bns.taylorf2 import SUN_MASS_SECONDS

    model = Model.default_for_testing()
    precessing = PrecessingModel(model)
    frequencies = np.geomspace(20.0, 2048.0, 1500)
    rng = np.random.default_rng(seed)
    params = []
    for i in range(n):
        row = data["intrinsic"][i]
        total_mass = rng.uniform(2.4, 3.2)
        params.append(PrecessingParametersWithExtrinsic(
            mass_ratio=row[0], lambda_1=row[1], lambda_2=row[2],
            chi_1=tuple(row[3:6]), chi_2=tuple(row[6:9]), distance_mpc=100.0,
            inclination=np.arccos(rng.uniform(-1, 1)), total_mass=total_mass,
            azimuth=rng.uniform(0, 2 * np.pi), reference_phase=rng.uniform(0, 2 * np.pi),
            reference_frequency_hz=data["omega_reference"][i] / (np.pi * total_mass * SUN_MASS_SECONDS),
        ))
    arguments = batch_arguments(params, frequencies)
    exact = jax.jit(precessing.jax_predict())(*arguments)
    regressed = jax.jit(precessing.jax_predict(precession_regressor=regressor))(*arguments)
    psd = ValidateModel(model.mode_models[Mode(2, 2)]).psd_at_frequencies(frequencies)

    def product(a, b):
        return np.trapezoid(np.conj(a) * b / psd, frequencies)

    mismatches = np.array([
        max(
            1 - np.real(product(a[i], b[i])) / np.sqrt(np.real(product(a[i], a[i]) * product(b[i], b[i])))
            for a, b in ((np.asarray(e), np.asarray(r)) for e, r in zip(exact, regressed))
        )
        for i in range(n)
    ])
    return mismatches, params


def validate(args) -> None:
    regressor = PrecessionRegressor.load(args.regressor)
    data = np.load(args.data)
    errors = function_errors(regressor, data, args.n)
    for j, name in enumerate(("zeta, x < -4.5", "zeta, x > -4.5", "G, x < -4.5", "G, x > -4.5")):
        print(f"{name:>15}: median {np.median(errors[:, j]):.1e}, 90% {np.percentile(errors[:, j], 90):.1e}, max {errors[:, j].max():.1e}")
    mismatches, params = waveform_mismatches(regressor, data, args.n)
    print(
        f"waveform mismatch: median {np.median(mismatches):.1e}, 90% "
        f"{np.percentile(mismatches, 90):.1e}, max {mismatches.max():.1e}"
    )
    for i in np.argsort(mismatches)[::-1][:5]:
        p = params[i]
        print(f"  {mismatches[i]:.1e}: q {p.mass_ratio:.2f}, chi_1 {np.round(p.chi_1, 2)}, "
              f"chi_2 {np.round(p.chi_2, 2)}, M {p.total_mass:.2f}, f_ref {p.reference_frequency_hz:.1f} Hz, "
              f"inclination {p.inclination:.2f}")


def benchmark(args) -> None:
    import jax

    from mlgw_bns.batched_precession import batch_arguments, precession_angles
    from mlgw_bns.model import Model
    from mlgw_bns.precessing_model import PrecessingModel
    from tests.test_batched_precession import BINARIES

    regressor = PrecessionRegressor.load(args.regressor)
    model = Model.default_for_testing()
    precessing = PrecessingModel(model)
    frequencies = np.geomspace(20.0, 2048.0, args.points)

    def best(function, repeats=7):
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            jax.block_until_ready(function())
            times.append(time.perf_counter() - start)
        return min(times) * 1e3

    waveforms = {
        "waveform, integrated": jax.jit(precessing.jax_predict()),
        "waveform, regressed": jax.jit(precessing.jax_predict(precession_regressor=regressor)),
    }
    angles = {
        "angles, integrated": jax.jit(precession_angles(model)),
        "angles, regressed": jax.jit(regressor.jax_angles()),
    }
    print(f"{'batch':>5} " + "".join(f"{name:>24}" for name in list(waveforms) + list(angles)))
    print("      (ms per call, and per binary)")
    for batch in args.batches:
        arguments = batch_arguments([BINARIES[i % 2] for i in range(batch)], frequencies)
        intrinsic, _, total_mass, *_, reference_frequency, _ = arguments
        row = []
        for function in waveforms.values():
            jax.block_until_ready(function(*arguments))
            row.append(best(lambda: function(*arguments)))
        for function in angles.values():
            call = lambda: function(intrinsic, total_mass, reference_frequency, frequencies[0])  # noqa: E731
            jax.block_until_ready(call())
            row.append(best(call))
        print(f"{batch:>5} " + "".join(f"{t:>14.2f} [{t / batch:>6.2f}]" for t in row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("generate")
    command.add_argument("--n", type=int, required=True)
    command.add_argument("--seed", type=int, default=1)
    command.add_argument("--out", required=True)
    command.set_defaults(run=generate)
    command = commands.add_parser("train")
    command.add_argument("--data", required=True)
    command.add_argument("--out", required=True)
    command.add_argument("--components", type=int, nargs=2, default=[64, 64])
    command.add_argument("--gamma", type=float, default=0.02)
    command.add_argument("--refine", type=int, default=0)
    command.set_defaults(run=train)
    command = commands.add_parser("validate")
    command.add_argument("--regressor", required=True)
    command.add_argument("--data", required=True)
    command.add_argument("--n", type=int, default=128)
    command.set_defaults(run=validate)
    command = commands.add_parser("benchmark")
    command.add_argument("--regressor", required=True)
    command.add_argument("--points", type=int, default=1500)
    command.add_argument("--batches", type=int, nargs="+", default=[1, 16, 128])
    command.set_defaults(run=benchmark)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.run(args)


if __name__ == "__main__":
    main()
