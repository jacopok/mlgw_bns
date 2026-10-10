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

``generate`` takes ~25 ms a binary a core (the integration, ~19, and the
least-squares fits); ``train`` takes ~20 seconds for 4096 binaries and ~15
minutes for 16384 (kernel ridge, cubic in their number; several ``--data``
files are concatenated), each ``--refine`` iteration ~2 minutes more for
4096. ``--mass-ratio-exponent`` above 1 puts more binaries near equal
masses; it did not pay with 16384. ``--data`` also takes a
:class:`~mlgw_bns.precession_dataset.ShardedDataset` directory (for larger
training sets, see ``precession_scale.py``).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import time

import numpy as np

from mlgw_bns.jax_mlp import MLPConfig
from mlgw_bns.precession_regression import (
    AngleGrid,
    PrecessionRegressor,
    TrainingRanges,
    _carriers,
    _evaluate,
    _g_baselines,
    carrier_table,
    carrier_tables,
    fit_all_envelopes,
    generate_training_set,
    nutation_frequency,
    reference_frame,
    refine_envelopes,
)


def oversample_anharmonic(grid, ranges, n, seed, fraction, threshold):
    """``n`` binaries of ``ranges``, of which a ``fraction`` are drawn among
    those whose nutation reaches an elliptic parameter of ``threshold``
    (the rest uniformly)."""
    from joblib import Parallel, delayed

    n_high = int(round(fraction * n))
    candidates = n - n_high + int(np.ceil(n_high / 0.12))
    intrinsic, omega_reference = ranges.sample(candidates, seed)
    chunks = np.array_split(np.arange(candidates), 64)
    max_m = np.concatenate(Parallel(n_jobs=-1)(
        delayed(lambda c: nutation_parameters(grid, intrinsic[c], omega_reference[c]).max(axis=1))(c)
        for c in chunks
    ))
    uniform = np.arange(n - n_high)
    rest = np.arange(n - n_high, candidates)
    high = rest[max_m[rest] >= threshold][:n_high]
    if len(high) < n_high:
        raise ValueError(f"only {len(high)} of {n_high} candidates reach m = {threshold}")
    chosen = np.concatenate([uniform, high])
    print(f"{n_high} of {n} binaries with max m >= {threshold} (base rate {np.mean(max_m >= threshold):.0%})")
    return intrinsic[chosen], omega_reference[chosen]


def generate(args) -> None:
    # the switch targets are the spins' envelopes there, on the carriers of
    # this grid: a grid with other carriers needs its own integration
    grid = AngleGrid(
        elliptic_beat=not args.linear_beat, averaged_precession=args.averaged_precession,
        **({} if args.smoothing is None else {"smoothing": args.smoothing}),
    )
    ranges = TrainingRanges(mass_ratio=tuple(args.mass_ratio), mass_ratio_exponent=args.mass_ratio_exponent)
    if args.high_m_fraction:
        intrinsic, omega_reference = oversample_anharmonic(
            grid, ranges, args.n, args.seed, args.high_m_fraction, args.m_threshold
        )
    else:
        intrinsic, omega_reference = ranges.sample(args.n, args.seed)
    start = time.perf_counter()
    data = generate_training_set(grid, intrinsic, omega_reference)
    residuals = data.pop("residuals")
    print(
        f"{args.n} binaries in {time.perf_counter() - start:.0f} s; largest fit "
        f"residuals {residuals[:, 0].max():.1e} (zeta), {residuals[:, 1].max():.1e} (G)"
    )
    np.savez(
        args.out, intrinsic=intrinsic, omega_reference=omega_reference, **data,
        elliptic_beat=grid.elliptic_beat, averaged_precession=grid.averaged_precession,
        smoothing=grid.smoothing,
    )


def load(filenames):
    """The training data of one or more ``generate`` files, concatenated, or
    of a :class:`~mlgw_bns.precession_dataset.ShardedDataset` directory."""
    if len(filenames) == 1 and os.path.isdir(filenames[0]):
        from mlgw_bns.precession_dataset import ShardedDataset

        dataset = ShardedDataset(filenames[0])
        data = dataset.load()
        data["x"] = dataset.x(data["omega_reference"])
        data["grid"] = dataset.config.grid
        return data
    files = [np.load(name) for name in filenames]
    options = ("elliptic_beat", "averaged_precession", "smoothing")
    # files from before the baselines were kept have them computed when needed
    keys = [key for key in files[0].files if key not in options and all(key in f.files for f in files)]
    data = {key: np.concatenate([f[key] for f in files]) for key in keys}
    # the grid the envelopes were fitted on; files from before an option
    # have its old default (the linear beat, the symmetric carriers)
    defaults = {"elliptic_beat": False, "averaged_precession": False, "smoothing": AngleGrid().smoothing}
    grids = {
        AngleGrid(**{k: (f[k].item() if k in f.files else defaults[k]) for k in options}) for f in files
    }
    if len(grids) > 1:
        raise ValueError("these files were fitted on different grids")
    data["grid"] = grids.pop()
    return data


def check_carriers(data, grid: AngleGrid) -> None:
    """The switch targets of ``data`` are the spins' envelopes at the switch
    on the carriers of the grid it was generated with: refuse a grid with
    other carriers."""
    carriers = lambda g: (g.elliptic_beat, g.averaged_precession)  # noqa: E731
    if carriers(grid) != carriers(data["grid"]):
        raise ValueError(
            f"the data were generated on the carriers {carriers(data['grid'])}, not {carriers(grid)}: "
            "generate it with this grid"
        )


def envelopes_on(data, grid: AngleGrid, cache=None) -> np.ndarray:
    """The envelope coefficients of the binaries of ``data`` on ``grid``:
    those of the files if they were fitted on it, else refitted to their
    integrated angles (and kept in, or taken from, the ``.npy`` ``cache``)."""
    if grid == data["grid"]:
        return data["coefficients"]
    if cache is not None and os.path.exists(cache):
        coefficients = np.load(cache)
        if len(coefficients) == len(data["intrinsic"]):
            return coefficients
    start = time.perf_counter()
    coefficients, residuals = fit_all_envelopes(
        grid, data["intrinsic"], data["omega_reference"], data["x"], data["zeta"], data["g"]
    )
    print(
        f"refitted {len(coefficients)} binaries on {grid} in {time.perf_counter() - start:.0f} s; "
        f"largest fit residuals {residuals[:, 0].max():.1e} (zeta), {residuals[:, 1].max():.1e} (G)"
    )
    if cache is not None:
        np.save(cache, coefficients)
    return coefficients


def nutation_parameters(grid: AngleGrid, intrinsic, omega_reference) -> np.ndarray:
    r"""The elliptic parameter :math:`m` of the nutation of each binary at
    the carrier quadrature nodes below the switch to the integration,
    ``(n, n_nodes)``: close to one near a separatrix between precession
    morphologies, where the nutation is most anharmonic."""
    x, omega, _ = grid.quadrature_nodes()
    below = x <= grid.x_switch
    return np.array([
        nutation_frequency(np, reference_frame(np, grid, row, omega_ref), omega, return_parameter=True)[1][below]
        for row, omega_ref in zip(intrinsic, omega_reference)
    ])


def train(args) -> None:
    data = load(args.data)
    grid = dataclasses.replace(data["grid"], sidebands=args.sidebands)
    if args.cells is not None:
        grid = dataclasses.replace(grid, n_cells=args.cells)
    if args.smoothing is not None:
        grid = dataclasses.replace(grid, smoothing=args.smoothing)
    if args.averaged_precession:
        grid = dataclasses.replace(grid, averaged_precession=True)
    check_carriers(data, grid)
    kwargs = dict(n_components=tuple(args.components), kernel_gamma=args.gamma)
    coefficients = envelopes_on(data, grid, args.refit_cache)
    n = len(coefficients) if args.n_train is None else args.n_train
    data = {key: value[:n] for key, value in data.items() if key != "grid"}
    coefficients = coefficients[:n] if args.coefficients is None else np.load(args.coefficients)[:n]
    if args.refine:
        coefficients, _ = refine_envelopes(
            grid, data["intrinsic"], data["omega_reference"], data["x"],
            data["zeta"], data["g"], coefficients, data["switch"],
            iterations=args.refine, **kwargs,
        )
        if args.save_refined:
            np.save(args.save_refined, coefficients)
    if args.mlp:
        kwargs["mlp"] = mlp_config(args)
        kwargs["mlp_loss"] = args.mlp_loss
    start = time.perf_counter()
    regressor = PrecessionRegressor.train(
        data["intrinsic"], data["omega_reference"], coefficients, data["switch"],
        grid=grid, **kwargs,
    )
    regressor.save(args.out)
    print(f"trained on {len(coefficients)} binaries in {time.perf_counter() - start:.0f} s")
    print_unexplained(regressor)


def add_mlp_arguments(command) -> None:
    """The options of a :class:`~mlgw_bns.jax_mlp.JaxMLP` regressor."""
    default = MLPConfig()
    command.add_argument("--mlp", action="store_true", help="a JAX perceptron instead of kernel ridge")
    command.add_argument("--mlp-hidden", type=int, nargs="+", default=list(default.hidden))
    command.add_argument("--mlp-activation", default=default.activation)
    command.add_argument("--mlp-rate", type=float, default=default.learning_rate)
    command.add_argument("--mlp-final-rate", type=float, default=default.final_learning_rate)
    command.add_argument("--mlp-epochs", type=int, default=default.epochs)
    command.add_argument("--mlp-batch", type=int, default=default.batch_size)
    command.add_argument("--mlp-weight-decay", type=float, default=default.weight_decay)
    command.add_argument("--mlp-check-every", type=int, default=default.check_every,
                         help="epochs between evaluations of the held-out loss")
    command.add_argument("--mlp-patience", type=int, default=default.patience,
                         help="stop after this many evaluations without improvement")
    command.add_argument("--mlp-dtype", default=default.dtype, choices=["float32", "float64"])
    command.add_argument("--mlp-seed", type=int, default=default.seed)
    command.add_argument("--mlp-loss", default="natural", choices=["natural", "uniform"])


def mlp_config(args) -> MLPConfig:
    return MLPConfig(
        hidden=tuple(args.mlp_hidden), activation=args.mlp_activation,
        learning_rate=args.mlp_rate, final_learning_rate=args.mlp_final_rate, epochs=args.mlp_epochs,
        batch_size=args.mlp_batch, weight_decay=args.mlp_weight_decay, check_every=args.mlp_check_every,
        patience=args.mlp_patience, dtype=args.mlp_dtype, seed=args.mlp_seed,
    )


def print_unexplained(regressor: PrecessionRegressor) -> None:
    """How much of the variance of the zeta and G envelopes and of the
    switch targets kernel ridge cannot predict, by leave-one-out: each
    output's expected error weighed by its variance in the units of the
    data."""
    selection = getattr(regressor.network, "alpha_selection", None)
    if selection is None:
        return
    error = selection.expected_error
    scale = regressor.network.target_scaler.scale_
    k, kk = regressor._split
    blocks = [
        ("zeta", slice(0, k), regressor.pca_zeta.principal_components_scaling),
        ("G", slice(k, kk), regressor.pca_g.principal_components_scaling),
        ("switch", slice(kk, None), 1.0),
    ]
    fractions = []
    for name, part, unit in blocks:
        variance = (scale[part] * unit) ** 2
        fractions.append(f"{name} {np.sum(variance * error[part] ** 2) / np.sum(variance):.3f}")
    print("variance unexplained by leave-one-out: " + ", ".join(fractions))


def reconstruct(grid: AngleGrid, coefficients, data, i: int, table=None):
    r""":math:`(\zeta, G)` of binary ``i`` of ``data`` from the envelope
    ``coefficients``, on its ``x`` up to the switch to the integration
    (``table``, its :func:`carrier_table` if known)."""
    if table is None:
        table = carrier_table(np, grid, reference_frame(np, grid, data["intrinsic"][i], data["omega_reference"][i]))
    x = data["x"][i]
    return _evaluate(np, grid, coefficients, _carriers(np, grid, *table, x), x)


def function_errors(regressor: PrecessionRegressor, data, n: int) -> np.ndarray:
    r"""Largest errors of :math:`\zeta` and :math:`G` of the first ``n``
    binaries of ``data``, below and above :math:`x = -10` (a (2, 2)
    frequency of ~25 Hz for a total mass of 2.8), up to the switch to the
    integration: ``(n, 4)``."""
    predicted = regressor.predict_coefficients(data["intrinsic"][:n], data["omega_reference"][:n])
    errors = []
    for start in range(0, n, 1024):
        stop = min(start + 1024, n)
        _, phases, rates = carrier_tables(
            regressor.grid, data["intrinsic"][start:stop], data["omega_reference"][start:stop], keep=(0, 1, 2)
        )
        for i in range(start, stop):
            zeta, g = reconstruct(regressor.grid, predicted[i], data, i, (phases[i - start], rates[i - start]))
            d_zeta, d_g = np.abs(zeta - data["zeta"][i]), np.abs(g - data["g"][i])
            high = data["x"][i] > -10.0
            errors.append((d_zeta[~high].max(), d_zeta[high].max(), d_g[~high].max(), d_g[high].max()))
    return np.array(errors)


def validation_binaries(data, n: int, seed: int = 7):
    """The first ``n`` binaries of ``data`` at random total masses and
    orientations (``seed``; the first ``n`` are the same for any ``n``), as
    :class:`~mlgw_bns.precessing_model.PrecessingParametersWithExtrinsic`."""
    from mlgw_bns.precessing_model import PrecessingParametersWithExtrinsic
    from mlgw_bns.taylorf2 import SUN_MASS_SECONDS

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
    return params


#: Binaries a call of the jitted waveforms: memory bounded, one compilation.
WAVEFORM_BATCH = 256


def _waveforms(function, params, frequencies) -> np.ndarray:
    """``function`` (a jitted precessing waveform) of ``params``, in batches
    of :data:`WAVEFORM_BATCH`: the polarizations ``(2, n, n_frequencies)``."""
    from mlgw_bns.batched_precession import batch_arguments

    parts = []
    for start in range(0, len(params), WAVEFORM_BATCH):
        batch = params[start:start + WAVEFORM_BATCH]
        padded = batch + [batch[0]] * (min(WAVEFORM_BATCH, len(params)) - len(batch))
        polarizations = np.asarray(function(*batch_arguments(padded, frequencies)))
        parts.append(polarizations[:, :len(batch)])
        logging.info("waveforms: %i/%i", start + len(batch), len(params))
    return np.concatenate(parts, axis=1)


def exact_waveforms(data, n: int, cache=None, seed: int = 7):
    """The frequencies, the PSD and the polarizations ``(2, n,
    n_frequencies)`` of :func:`validation_binaries` with the integrated
    angles; kept in (or taken from) the ``.npz`` ``cache``, if given, made
    for as many binaries or more."""
    import jax

    from mlgw_bns.higher_order_modes import Mode
    from mlgw_bns.model import Model
    from mlgw_bns.model_validation import ValidateModel
    from mlgw_bns.precessing_model import PrecessingModel

    frequencies = np.geomspace(20.0, 2048.0, 1500)
    if cache is not None and os.path.exists(cache):
        with np.load(cache) as stored:
            if (stored["n"] >= n and stored["seed"] == seed and np.array_equal(stored["frequencies"], frequencies)
                    and np.array_equal(stored["omega_reference"][:n], data["omega_reference"][:n])):
                return frequencies, stored["psd"], stored["exact"][:, :n]
    model = Model.default_for_testing()
    psd = ValidateModel(model.mode_models[Mode(2, 2)]).psd_at_frequencies(frequencies)
    exact = _waveforms(jax.jit(PrecessingModel(model).jax_predict()), validation_binaries(data, n, seed), frequencies)
    if cache is not None:
        from mlgw_bns.sharding import save_arrays

        save_arrays(cache, n=n, seed=seed, frequencies=frequencies, psd=psd, exact=exact,
                    omega_reference=data["omega_reference"][:n])
    return frequencies, psd, exact


def waveform_mismatches(regressor: PrecessionRegressor, data, n: int, seed: int = 7, cache=None):
    """Mismatches, with nothing optimized, of the JAX precessing waveforms
    with the regressed angles against those with the integrated ones, for the
    first ``n`` binaries of ``data`` at random total masses and orientations
    (ET PSD, 20--2048 Hz): the larger of the two polarizations', the
    parameters, and ``(frequencies, psd, exact, regressed)``, the last two
    the polarizations ``(2, n, n_frequencies)``; the exact ones from
    :func:`exact_waveforms` (and its ``cache``)."""
    import jax

    from mlgw_bns.model import Model
    from mlgw_bns.precessing_model import PrecessingModel

    frequencies, psd, exact = exact_waveforms(data, n, cache, seed)
    params = validation_binaries(data, n, seed)
    precessing = PrecessingModel(Model.default_for_testing())
    regressed = _waveforms(jax.jit(precessing.jax_predict(precession_regressor=regressor)), params, frequencies)

    def product(a, b):
        return np.trapezoid(np.conj(a) * b / psd, frequencies, axis=-1)

    mismatches = np.max([
        1 - np.real(product(a, b)) / np.sqrt(np.real(product(a, a) * product(b, b)))
        for a, b in zip(exact, regressed)
    ], axis=0)
    return mismatches, params, (frequencies, psd, exact, regressed)


class OracleRegressor(PrecessionRegressor):
    """Stands in for ``regressor`` with given envelope ``coefficients`` and
    ``switch`` targets of the binaries it is asked about, in order: those
    they were fitted to, or those through its PCA (:meth:`project`), the
    floors set by the representation and by the compression."""

    def __init__(self, regressor: PrecessionRegressor, intrinsic, omega_reference, coefficients, switch):
        super().__init__(regressor.grid, regressor.ranges, regressor.pca_zeta, regressor.pca_g, None)
        self.stored = coefficients, switch
        without = coefficients.copy()
        without[:, self.grid.n_zeta] -= _g_baselines(self.grid, intrinsic, omega_reference)
        self.without_baselines = without

    def predict(self, intrinsic, omega_reference):
        return tuple(a[: len(intrinsic)] for a in self.stored)

    def jax_coefficients(self):
        import jax.numpy as jnp

        coefficients, switch = jnp.asarray(self.without_baselines), jnp.asarray(self.stored[1])
        return lambda intrinsic, omega_reference: (
            coefficients[: intrinsic.shape[0]], switch[: intrinsic.shape[0]]
        )


def validate(args) -> None:
    regressor = PrecessionRegressor.load(args.regressor)
    data = load(args.data)
    n = len(data["intrinsic"]) if args.n is None else args.n
    if args.oracle:
        check_carriers(data, regressor.grid)
        intrinsic, omega_reference = data["intrinsic"][:n], data["omega_reference"][:n]
        coefficients = envelopes_on(data, regressor.grid, args.refit_cache)[:n]
        if args.oracle == "pca":
            coefficients = regressor.project(intrinsic, omega_reference, coefficients)
        regressor = OracleRegressor(regressor, intrinsic, omega_reference, coefficients, data["switch"][:n])
        print(f"oracle: the {args.oracle} envelopes of the validation binaries")
    errors = function_errors(regressor, data, n)
    for j, name in enumerate(("zeta, x < -10", "zeta, x > -10", "G, x < -10", "G, x > -10")):
        print(f"{name:>15}: median {np.median(errors[:, j]):.1e}, 90% {np.percentile(errors[:, j], 90):.1e}, max {errors[:, j].max():.1e}")
    mismatches, params, _ = waveform_mismatches(regressor, data, n, cache=args.exact_cache)
    print(
        f"waveform mismatch: median {np.median(mismatches):.1e}, 90% "
        f"{np.percentile(mismatches, 90):.1e}, 99% {np.percentile(mismatches, 99):.1e}, "
        f"max {mismatches.max():.1e}, above 1e-2: {np.mean(mismatches > 1e-2):.1%}"
    )
    # by how anharmonic the nutation is: the tail is near the separatrices
    max_m = nutation_parameters(regressor.grid, data["intrinsic"][:n], data["omega_reference"][:n]).max(axis=1)
    for lo, hi in [(0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        cell = (max_m >= lo) & (max_m < hi)
        if cell.any():
            print(f"  max m {lo}-{hi} ({cell.sum():4d}): median {np.median(mismatches[cell]):.1e}, "
                  f"90% {np.percentile(mismatches[cell], 90):.1e}")
    if args.out:
        from mlgw_bns.sharding import save_arrays

        save_arrays(args.out, mismatches=mismatches, errors=errors, max_m=max_m, intrinsic=data["intrinsic"][:n])
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
    command.add_argument("--averaged-precession", action="store_true", help="see AngleGrid.averaged_precession")
    command.add_argument("--smoothing", type=float, help="see AngleGrid.smoothing")
    command.add_argument("--high-m-fraction", type=float, default=0.0,
                         help="this fraction of the binaries drawn among those with an anharmonic nutation")
    command.add_argument("--m-threshold", type=float, default=0.3)
    command.set_defaults(run=generate)
    command = commands.add_parser("train")
    command.add_argument("--data", required=True, nargs="+")
    command.add_argument("--save-refined")
    command.add_argument("--out", required=True)
    command.add_argument("--components", type=int, nargs=2, default=[64, 64])
    command.add_argument("--gamma", type=float, default=0.02)
    command.add_argument("--refine", type=int, default=0)
    command.add_argument("--sidebands", type=int, default=0, help="see AngleGrid.sidebands")
    command.add_argument("--cells", type=int, help="see AngleGrid.n_cells")
    command.add_argument("--smoothing", type=float, help="see AngleGrid.smoothing")
    command.add_argument("--averaged-precession", action="store_true", help="see AngleGrid.averaged_precession")
    command.add_argument("--n-train", type=int, help="train on the first this many binaries")
    command.add_argument("--refit-cache", help=".npy keeping the envelopes refitted on another grid")
    command.add_argument("--coefficients", help=".npy of envelopes to train on, such as --save-refined wrote")
    add_mlp_arguments(command)
    command.set_defaults(run=train)
    command = commands.add_parser("validate")
    command.add_argument("--regressor", required=True)
    command.add_argument("--data", required=True, nargs="+")
    command.add_argument("--n", type=int, help="the first this many binaries (default: all)")
    command.add_argument("--out", help="save the mismatches and errors here (.npz)")
    command.add_argument("--oracle", choices=["fit", "pca"],
                         help="the validation binaries' own envelopes instead of the regression")
    command.add_argument("--refit-cache", help=".npy keeping the validation envelopes refitted on the regressor's grid")
    command.add_argument("--exact-cache", help=".npz keeping the waveforms with the integrated angles")
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
