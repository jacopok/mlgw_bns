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
    # plots of the worst binaries, ranked by the first regressor
    python visualization/precession_regression_study.py worst --regressor a.joblib b.joblib --data validation.npz --out worst
    # cost of the angles and of the whole waveform, single and batched
    python visualization/precession_regression_study.py benchmark --regressor regressor.joblib

``generate`` takes ~0.1 s a binary on four cores (the integration and the
least-squares fits); ``train`` takes ~2.5 minutes for 4096 binaries and ~15
for 16384 (kernel ridge, cubic in their number; several ``--data`` files
are concatenated), each ``--refine`` iteration ~10 minutes more for 4096.
``--mass-ratio-exponent`` above 1 puts more binaries near equal masses; it
did not pay with 16384.
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
    grid = AngleGrid(elliptic_beat=not args.linear_beat)
    intrinsic, omega_reference = TrainingRanges(
        mass_ratio=tuple(args.mass_ratio), mass_ratio_exponent=args.mass_ratio_exponent
    ).sample(args.n, args.seed)
    start = time.perf_counter()
    x, zeta, g, switch = training_data(grid, intrinsic, omega_reference)
    coefficients, residuals = fit_all_envelopes(grid, intrinsic, omega_reference, x, zeta, g)
    print(
        f"{args.n} binaries in {time.perf_counter() - start:.0f} s; largest fit "
        f"residuals {residuals[:, 0].max():.1e} (zeta), {residuals[:, 1].max():.1e} (G)"
    )
    np.savez(
        args.out, intrinsic=intrinsic, omega_reference=omega_reference,
        coefficients=coefficients, switch=switch, x=x, zeta=zeta, g=g,
        elliptic_beat=grid.elliptic_beat,
    )


def load(filenames):
    """The training data of one or more ``generate`` files, concatenated."""
    files = [np.load(name) for name in filenames]
    data = {key: np.concatenate([f[key] for f in files]) for key in files[0].files if key != "elliptic_beat"}
    # the carriers the envelopes were fitted on; files from before the
    # option have the linear beat
    beats = {bool(f["elliptic_beat"]) if "elliptic_beat" in f.files else False for f in files}
    if len(beats) > 1:
        raise ValueError("these files were fitted on different carriers")
    data["grid"] = AngleGrid(elliptic_beat=beats.pop())
    return data


def train(args) -> None:
    data = load(args.data)
    kwargs = dict(n_components=tuple(args.components), kernel_gamma=args.gamma)
    coefficients = data["coefficients"]
    if args.refine:
        coefficients, _ = refine_envelopes(
            data["grid"], data["intrinsic"], data["omega_reference"], data["x"],
            data["zeta"], data["g"], coefficients, data["switch"],
            iterations=args.refine, **kwargs,
        )
        if args.save_refined:
            np.save(args.save_refined, coefficients)
    start = time.perf_counter()
    PrecessionRegressor.train(
        data["intrinsic"], data["omega_reference"], coefficients, data["switch"],
        grid=data["grid"], **kwargs,
    ).save(args.out)
    print(f"trained on {len(coefficients)} binaries in {time.perf_counter() - start:.0f} s")


def reconstruct(grid: AngleGrid, coefficients, data, i: int):
    r""":math:`(\zeta, G)` of binary ``i`` of ``data`` from the envelope
    ``coefficients``, on its ``x`` up to the switch to the integration."""
    frame = reference_frame(np, grid, data["intrinsic"][i], data["omega_reference"][i])
    x = data["x"][i]
    return _evaluate(np, grid, coefficients, _carriers(np, grid, *carrier_table(np, grid, frame), x), x)


def function_errors(regressor: PrecessionRegressor, data, n: int) -> np.ndarray:
    r"""Largest errors of :math:`\zeta` and :math:`G` of the first ``n``
    binaries of ``data``, below and above :math:`x = -10` (a (2, 2)
    frequency of ~25 Hz for a total mass of 2.8), up to the switch to the
    integration: ``(n, 4)``."""
    predicted = regressor.predict_coefficients(data["intrinsic"][:n], data["omega_reference"][:n])
    errors = []
    for i in range(n):
        zeta, g = reconstruct(regressor.grid, predicted[i], data, i)
        d_zeta, d_g = np.abs(zeta - data["zeta"][i]), np.abs(g - data["g"][i])
        high = data["x"][i] > -10.0
        errors.append((d_zeta[~high].max(), d_zeta[high].max(), d_g[~high].max(), d_g[high].max()))
    return np.array(errors)


def waveform_mismatches(regressor: PrecessionRegressor, data, n: int, seed: int = 7):
    """Mismatches, with nothing optimized, of the JAX precessing waveforms
    with the regressed angles against those with the integrated ones, for the
    first ``n`` binaries of ``data`` at random total masses and orientations
    (ET PSD, 20--2048 Hz): the larger of the two polarizations', the
    parameters, and ``(frequencies, psd, exact, regressed)``, the last two
    the polarizations ``(2, n, n_frequencies)``."""
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

    exact, regressed = np.asarray(exact), np.asarray(regressed)
    mismatches = np.array([
        max(
            1 - np.real(product(a[i], b[i])) / np.sqrt(np.real(product(a[i], a[i]) * product(b[i], b[i])))
            for a, b in zip(exact, regressed)
        )
        for i in range(n)
    ])
    return mismatches, params, (frequencies, psd, exact, regressed)


def validate(args) -> None:
    regressor = PrecessionRegressor.load(args.regressor)
    data = np.load(args.data)
    errors = function_errors(regressor, data, args.n)
    for j, name in enumerate(("zeta, x < -10", "zeta, x > -10", "G, x < -10", "G, x > -10")):
        print(f"{name:>15}: median {np.median(errors[:, j]):.1e}, 90% {np.percentile(errors[:, j], 90):.1e}, max {errors[:, j].max():.1e}")
    mismatches, params, _ = waveform_mismatches(regressor, data, args.n)
    print(
        f"waveform mismatch: median {np.median(mismatches):.1e}, 90% "
        f"{np.percentile(mismatches, 90):.1e}, max {mismatches.max():.1e}"
    )
    for i in np.argsort(mismatches)[::-1][:5]:
        p = params[i]
        print(f"  {mismatches[i]:.1e}: q {p.mass_ratio:.2f}, chi_1 {np.round(p.chi_1, 2)}, "
              f"chi_2 {np.round(p.chi_2, 2)}, M {p.total_mass:.2f}, f_ref {p.reference_frequency_hz:.1f} Hz, "
              f"inclination {p.inclination:.2f}")


def chi_p(intrinsic) -> np.ndarray:
    r"""The effective precession spin of the rows ``[q, \Lambda_1,
    \Lambda_2, \vec{\chi}_1, \vec{\chi}_2]``, :math:`q \geq 1`."""
    inverse = 1.0 / intrinsic[:, 0]
    in_plane_1 = np.hypot(intrinsic[:, 3], intrinsic[:, 4])
    in_plane_2 = np.hypot(intrinsic[:, 6], intrinsic[:, 7])
    return np.maximum(in_plane_1, inverse * (4 * inverse + 3) / (4 + 3 * inverse) * in_plane_2)


def worst(args) -> None:
    """Plot the binaries with the largest waveform mismatches with the first
    regressor: the errors of zeta and G of each regressor and of the
    envelopes fitted to the integration (the floor of the representation),
    and where in frequency the mismatches accumulate."""
    import matplotlib.pyplot as plt

    from mlgw_bns.taylorf2 import SUN_MASS_SECONDS

    data = load([args.data])
    n = args.n
    names = [name.rsplit("/", 1)[-1].removesuffix(".joblib") for name in args.regressor]
    regressors = [PrecessionRegressor.load(name) for name in args.regressor]
    runs = [waveform_mismatches(regressor, data, n) for regressor in regressors]
    params = runs[0][1]
    frequencies, psd, exact, _ = runs[0][2]
    order = np.argsort(runs[0][0])[::-1][: args.worst]
    colors = [f"C{j}" for j in range(len(regressors))]

    figure, axes = plt.subplots(1, len(regressors), figsize=(5 * len(regressors), 4), sharey=True, squeeze=False)
    spin = chi_p(data["intrinsic"][:n])
    for ax, name, (mismatches, *_) in zip(axes[0], names, runs):
        points = ax.scatter(data["intrinsic"][:n, 0], mismatches, c=spin, s=10, cmap="viridis")
        ax.scatter(data["intrinsic"][order, 0], mismatches[order], s=80, facecolors="none", edgecolors="r")
        for rank, i in enumerate(order):
            ax.annotate(str(rank), (data["intrinsic"][i, 0], mismatches[i]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points", color="r")
        ax.set(xlabel="$q$", yscale="log", title=f"{name}: median {np.median(mismatches):.1e}")
    axes[0, 0].set_ylabel("waveform mismatch")
    figure.colorbar(points, ax=axes[0], label=r"$\chi_p$")
    figure.savefig(f"{args.out}_overview.png", dpi=150, bbox_inches="tight")

    predicted = [regressor.predict_coefficients(data["intrinsic"][order], data["omega_reference"][order])
                 for regressor in regressors]
    figure, axes = plt.subplots(len(order), 4, figsize=(20, 3.2 * len(order)), squeeze=False)
    for row, i in zip(axes, order):
        p = params[i]
        to_hz = 1.0 / (np.pi * p.total_mass * SUN_MASS_SECONDS)
        f = data["grid"].omega_of(data["x"][i]) * to_hz
        zeta, g = data["zeta"][i], data["g"][i]
        floor = reconstruct(data["grid"], data["coefficients"][i], data, i)
        row[0].plot(f, zeta.real, "k", lw=1.5, label="integrated")
        row[1].plot(f, np.abs(floor[0] - zeta), color="0.6", label="envelope fit")
        row[2].plot(f, np.abs(floor[1] - g), color="0.6", label="envelope fit")
        for j, (regressor, name, color) in enumerate(zip(regressors, names, colors)):
            r_zeta, r_g = reconstruct(regressor.grid, predicted[j][list(order).index(i)], data, i)
            row[0].plot(f, r_zeta.real, color=color, lw=0.8, label=name)
            row[1].plot(f, np.abs(r_zeta - zeta), color=color, lw=0.8, label=name)
            row[2].plot(f, np.abs(r_g - g), color=color, lw=0.8, label=name)
            # where the mismatch of the worse polarization accumulates:
            # 1 - Re<a, b> = <a - b, a - b> / 2 for normalized a, b
            regressed = runs[j][2][3]
            worse = []
            for a, b in zip(exact[:, i], regressed[:, i]):
                a = a / np.sqrt(np.trapezoid(np.abs(a) ** 2 / psd, frequencies))
                b = b / np.sqrt(np.trapezoid(np.abs(b) ** 2 / psd, frequencies))
                integrand = np.abs(a - b) ** 2 / psd / 2
                worse.append(np.concatenate([[0.0], np.cumsum(np.diff(frequencies) * (integrand[1:] + integrand[:-1]) / 2)]))
            row[3].plot(frequencies, max(worse, key=lambda c: c[-1]), color=color, label=f"{name}: {runs[j][0][i]:.1e}")
        for ax in row[:3]:
            ax.axvspan(f[0], 20.0, color="0.92", zorder=0)
            ax.set_xscale("log")
        row[0].set(ylabel=r"Re $\zeta$")
        row[1].set(ylabel=r"$|\Delta \zeta|$", yscale="log")
        row[2].set(ylabel=r"$|\Delta G|$", yscale="log")
        row[3].set(ylabel="cumulative mismatch", xscale="log", yscale="log", ylim=(1e-7, None))
        row[0].set_title(
            f"q {p.mass_ratio:.2f}, $\\chi_1$ {np.round(p.chi_1, 2)}, $\\chi_2$ {np.round(p.chi_2, 2)}, "
            f"M {p.total_mass:.2f}, $f_{{\\rm ref}}$ {p.reference_frequency_hz:.0f} Hz, "
            f"$\\iota$ {p.inclination:.2f}", fontsize=9, loc="left",
        )
        row[1].legend(fontsize=7)
        row[3].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("(2, 2) frequency [Hz]")
    figure.savefig(f"{args.out}_cases.png", dpi=110, bbox_inches="tight")
    print(f"saved {args.out}_overview.png, {args.out}_cases.png")


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
    command.add_argument("--mass-ratio-exponent", type=float, default=1.0)
    command.add_argument("--mass-ratio", type=float, nargs=2, default=list(TrainingRanges().mass_ratio))
    command.add_argument(
        "--linear-beat", action="store_true",
        help="carriers beating at the linearized normal-mode splitting, as before the elliptic beat",
    )
    command.set_defaults(run=generate)
    command = commands.add_parser("train")
    command.add_argument("--data", required=True, nargs="+")
    command.add_argument("--save-refined")
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
    command = commands.add_parser("worst")
    command.add_argument("--regressor", required=True, nargs="+", help="the first ranks the binaries")
    command.add_argument("--data", required=True)
    command.add_argument("--n", type=int, default=128)
    command.add_argument("--worst", type=int, default=5)
    command.add_argument("--out", required=True, help="prefix of the figures")
    command.set_defaults(run=worst)
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
