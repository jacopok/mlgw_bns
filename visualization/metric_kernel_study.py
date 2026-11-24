r"""Is the Fisher metric of the waveforms a good distance for the kernels of the
regressors? Stage 1: the metric itself, and how much the PSD matters.

::

    M=visualization/metric_kernel_study.py
    python $M metric OUT.npz --n 2048 --psd-dir DIR   # the metrics at N points (JAX)
    # from TEOBResumS instead (the surrogate's derivatives are wrong at small
    # Lambda): stencils of waveforms around N points (resumable), their metrics
    python $M stencils DIR --n 1024 --lambda-max 2000 --jobs 32 --oracle default_hom_lambda2000
    python $M metric OUT.npz --stencils DIR --lambda-max 2000 --oracle default_hom_lambda2000 --psd-dir DIR
    python $M report OUT.npz                          # spectra, variation, PSD dependence
    python $M check-mismatch                          # quadratic form against mismatches
    python $M check-teob --n 20 --jobs 32             # derivatives against TEOBResumS

The modes are taken as orthogonal (their average over the sphere is), so the
inner product is the sum over modes of 4 int da db* / S_n. With
:math:`dh = (dA + i A d\phi) e^{i\phi}`, the metrics (all divided by the
:math:`\rho^2` of the whole waveform) are:

* ``full``: of the waveform, projecting out time, orbital phase, amplitude and
  total mass (the model gets the latter by rescaling, exactly);
* ``full_fixed_mass``: the same, not projecting out the mass;
* ``residual``: :math:`d\phi` is of the phase residual (what is regressed);
* ``residual_shifted``: the same, projecting out time and orbital phase.

The model is evaluated at a reference total mass, and rescaled: at total mass
M (detector frame) its frequency f is seen by the detector at f M_ref / M, so
that the PSD and the edges of the band shift under it. Every metric (after the
projections and the normalization, which depend on M) is the average over
masses uniform in 2--4 solar masses; those of single masses (``ET_5_M2`` ...)
are for comparison.

Each is computed for the sum of the modes and for each mode alone. They are in
the coordinates of the parameter array (q, Lambda_1, Lambda_2, chi_1, chi_2).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from mlgw_bns.dataset_generation import ParameterSet
from mlgw_bns.model import Model
from mlgw_bns.model_validation import PSD_PATH, _trapezoid_weights

ORACLE = "default_hom_lambda8000"
KINDS = ("full", "full_fixed_mass", "residual", "residual_shifted")
# name -> (file, lowest frequency): bilby's noise curves, and the packaged one
PSDS = {
    "ET_5": ("ET", 5.0), "ET_10": ("ET", 10.0), "ET_20": ("ET", 20.0),
    "CE_10": ("CE", 10.0), "CE_20": ("CE", 20.0),
    "aLIGO_10": ("aLIGO_ZERO_DET_high_P", 10.0), "aLIGO_20": ("aLIGO_ZERO_DET_high_P", 20.0),
    "flat_5": (None, 5.0),
}
LAMBDA_MAX = 5000.0  # of the data of stage 2
BOX = np.array([[1.0, 3.0], [1.001, LAMBDA_MAX], [1.001, LAMBDA_MAX], [-0.5, 0.5], [-0.5, 0.5]])
F_MAX = 2048.0
MASS_RANGE, N_MASSES = (2.0, 4.0), 8
SINGLE_MASSES = (2.0, 2.8, 4.0)


def quadrature(psd_dir, model, n_grid=1024, n_masses=N_MASSES):
    """Frequencies of the model (at its reference mass), and ``name -> (masses,
    k)`` weights of the inner product at each of the masses to average over."""
    reference = float(model.dataset.total_mass)
    all_masses = MASS_RANGE[0] + np.ptp(MASS_RANGE) * (np.arange(n_masses) + 0.5) / n_masses
    knots = model.dataset.frequencies_hz
    f = np.geomspace(
        max(5.0 * MASS_RANGE[0] / reference, knots[0]), min(F_MAX * MASS_RANGE[1] / reference, knots[-1]), n_grid
    )
    f[0], f[-1] = f[0] * (1 + 1e-9), f[-1] * (1 - 1e-9)  # inside the trained band, whatever the rounding
    trapezoid = _trapezoid_weights(f)

    def at(inverse, f_min, masses):
        # the frequencies seen by the detector, for each mass; edges within
        # rounding of the band's are in it
        seen = f * reference / np.asarray(masses)[:, None]
        inside = (seen >= f_min * (1 - 1e-6)) & (seen <= F_MAX * (1 + 1e-6))
        return 4 * trapezoid * inverse(seen) * inside

    weights = {}
    for name, (file, f_min) in PSDS.items():
        if file is None:
            inverse = np.ones_like
        else:
            path = PSD_PATH / "ET_psd.txt" if file == "ET" else Path(psd_dir or ".") / f"{file}_psd.txt"
            if not path.exists():
                print(f"no {path}: skipping {name}", file=sys.stderr)
                continue
            table = np.log(np.loadtxt(path))
            inverse = lambda x, t=table: np.exp(-np.interp(np.log(x), t[:, 0], t[:, 1]))  # noqa: E731
        weights[name] = at(inverse, f_min, all_masses)
        if name in ("ET_5", "aLIGO_20"):
            for mass in SINGLE_MASSES:
                weights[f"{name}_M{mass:g}"] = at(inverse, f_min, [mass])
    return f, weights


def sample(n, seed=0, lower=None):
    """Half uniform in Lambda, half in log(1 + Lambda), in the box (above ``lower``)."""
    from scipy.stats import qmc

    u = qmc.Sobol(6, seed=seed).random(n)
    box = BOX if lower is None else np.column_stack([lower, BOX[:, 1]])
    x = box[:, 0] + u[:, :5] * (box[:, 1] - box[:, 0])
    log = np.expm1(np.log1p(box[1:3, 0]) + u[:, 1:3] * np.diff(np.log1p(box[1:3]), axis=1)[:, 0])
    x[:, 1:3] = np.where(u[:, 5:] < 0.5, log, x[:, 1:3])
    return x


class Derivatives:
    """The amplitudes, and the derivatives of amplitudes, phases and phase
    residuals, of the modes on a frequency grid, from a trained model."""

    def __init__(self, model, f):
        import jax

        self.model, self.f = model, f
        self.mass = float(model.dataset.total_mass)
        self.surrogate = model.batched_surrogate(backend="jax")
        self.modes = list(model.modes)
        frozen = self.surrogate._frozen
        # from the phase nodes to the grid
        self.resample = [
            CubicSpline(np.asarray(a.phase_spline.knots), np.eye(len(a.phase_spline.knots)), axis=0)(f)
            for a in frozen.modes_arrays[: len(self.modes)]
        ]

        def waveform(x, log_mass):
            amp, phase = self.surrogate(x[None], self.mass * jax.numpy.exp(log_mass), f)
            return amp[0], phase[0]

        def residual(x):
            parts = self.surrogate._residuals(jax.numpy, x[None])
            return [r @ p[0] for r, (_, p) in zip(self.resample, parts)]

        def everything(x):
            def one(tangent):
                _, d_wave = jax.jvp(waveform, (x, 0.0), (tangent[:5], tangent[5]))
                _, d_res = jax.jvp(residual, (x,), (tangent[:5],))
                return d_wave, jax.numpy.stack(d_res)
            (d_amp, d_phase), d_res = jax.vmap(one)(jax.numpy.eye(6))
            amp, _ = waveform(x, 0.0)
            return amp, d_amp, d_phase, d_res[:5]

        self._function = jax.jit(jax.vmap(everything))
        self._waveform = jax.jit(lambda x: waveform(x, 0.0))

    def __call__(self, x):
        """``A (N, modes, k)``, ``dA`` and ``dphase (N, 6, modes, k)`` (the
        last of the 6 is ln M), ``dphase_residual (N, 5, modes, k)``."""
        return tuple(np.asarray(a) for a in self._function(x))


def project(vectors, columns):
    """``vectors (d, n)`` without their components along ``columns (c, n)``."""
    norms = np.linalg.norm(columns, axis=1, keepdims=True)
    q, _ = np.linalg.qr((columns / np.where(norms > 0, norms, 1.0)).T)
    return vectors - (vectors @ q) @ q.T


def metrics(f, weights, m, amp, d_amp, d_phase, d_res):
    """``kind -> (scopes, 5, 5)`` of one waveform: the sum of the modes, then
    each alone; and each mode's share of rho^2. ``weights`` of one mass, or
    ``(masses, k)`` to average over."""
    if weights.ndim == 2:
        parts = [metrics(f, w, m, amp, d_amp, d_phase, d_res) for w in weights]
        return {k: np.mean([g[k] for g, _ in parts], axis=0) for k in parts[0][0]}, np.mean([r for _, r in parts], axis=0)
    root = np.sqrt(weights)  # (k,)
    rho2 = np.sum(amp**2 * weights, axis=1)
    flat = lambda re, im: np.concatenate([re * root, im * root], axis=-1)  # noqa: E731
    full = d_phase is not None
    if full:
        dh = flat(d_amp, amp * d_phase)            # (6, modes, 2k)
    dr = flat(d_amp[:5], amp * d_res)              # (5, modes, 2k)
    zero = np.zeros_like(amp)
    t = flat(zero, amp * 2 * np.pi * f)            # (modes, 2k)
    phi = flat(zero, amp * m[:, None])
    size = flat(amp, zero)
    out = {kind: [] for kind in KINDS if full or kind.startswith("residual")}
    for scope in [slice(None)] + [slice(i, i + 1) for i in range(len(m))]:
        pick = lambda v: v[..., scope, :].reshape(*v.shape[:-2], -1)  # noqa: E731
        shifts = [pick(t), pick(phi)]
        kinds = [("residual", pick(dr)), ("residual_shifted", project(pick(dr), np.array(shifts)))]
        if full:
            wave, mass = pick(dh)[:5], pick(dh)[5]
            kinds += [
                ("full", project(wave, np.array(shifts + [pick(size), mass]))),
                ("full_fixed_mass", project(wave, np.array(shifts + [pick(size)]))),
            ]
        for kind, vectors in kinds:
            out[kind].append(vectors @ vectors.T / rho2.sum())
    return {k: np.array(v) for k, v in out.items()}, rho2 / rho2.sum()


def open_oracle(name=ORACLE):
    return Model.default_for_testing(name)


STENCIL_STEPS = np.array([4e-3, 0.02, 0.02, 4e-3, 4e-3])  # those of Lambda: relative to 20 + Lambda


def stencils_command(args):
    """Residuals of the modes at each of N points (the first of 11 waveforms)
    and at a step above, and below, along each parameter, in shards."""
    model = open_oracle(args.oracle)
    x = sample(args.n, lower=[1.005, 2.0, 2.0, -0.5, -0.5])
    step = np.tile(STENCIL_STEPS, (len(x), 1))
    step[:, 1:3] *= 20 + x[:, 1:3]
    unit = np.concatenate([np.zeros((1, 5)), np.eye(5), -np.eye(5)])
    points = x[:, None] + step[:, None] * unit
    indices = {mode: model.mode_models[mode].downsampling_indices for mode in model.modes}
    references = {mode: model.mode_models[mode].dataset.amplitude_reference_parameters for mode in model.modes}
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    for k, start in enumerate(range(0, len(x), args.shard)):
        rows = slice(start, start + args.shard)
        path = directory / f"{k:04d}.npz"
        if path.exists():
            continue
        began = time.time()
        params = ParameterSet(points[rows].reshape(-1, 5)).waveform_parameters(model.dataset)
        _, amplitudes, phases = model._multimode_mode_residuals(
            params, model.dataset.frequencies, indices, references, n_jobs=args.jobs, keep_invalid=True)
        n = len(x[rows])
        arrays = {}
        for mode in model.modes:
            arrays[f"amplitude_l{mode.l}_m{mode.m}"] = np.asarray(amplitudes[mode]).reshape(n, 11, -1)
            arrays[f"phase_l{mode.l}_m{mode.m}"] = np.asarray(phases[mode]).reshape(n, 11, -1)
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, parameters=x[rows], steps=step[rows], **arrays)
        temporary.rename(path)
        print(f"{path}: {time.time() - began:.0f} s", flush=True)


def stencil_derivatives(directory, model, f):
    """As :class:`Derivatives` gives them (but for the phases), shard by shard:
    ``x, steps, A, dA, None, dphase_residual``; points that failed left out."""
    frozen = model.batched_surrogate(backend="numpy")._frozen.modes_arrays
    for path in sorted(Path(directory).glob("[0-9][0-9][0-9][0-9].npz")):
        with np.load(path) as data:
            x, steps = data["parameters"], data["steps"]
            amp, d_amp, d_res, ok = [], [], [], np.ones(len(x), bool)
            for mode, arrays in zip(model.modes, frozen):
                name = f"l{mode.l}_m{mode.m}"
                values = []
                for key, spline, factor in (("amplitude", arrays.amp_spline, arrays.pn_amp_nodes), ("phase", arrays.phase_spline, 1.0)):
                    raw = np.asarray(data[f"{key}_{name}"], dtype=float) * factor
                    ok &= np.isfinite(raw).all(axis=(1, 2))
                    grid = CubicSpline(spline.knots, np.eye(len(spline.knots)), axis=0)(f)
                    values.append((raw[:, 0] @ grid.T, (raw[:, 1:6] - raw[:, 6:]) / (2 * steps[:, :, None]) @ grid.T))
                amp.append(values[0][0])
                d_amp.append(values[0][1])
                d_res.append(values[1][1])
        yield x[ok], np.stack(amp, 1)[ok], np.stack(d_amp, 2)[ok], np.stack(d_res, 2)[ok]


def metric_command(args):
    model = open_oracle(args.oracle)
    f, weights = quadrature(args.psd_dir, model, n_masses=args.masses)
    modes = list(model.modes)
    m = np.array([mode.m for mode in modes], dtype=float)

    def rows():
        if args.stencils:
            for x, amp, d_amp, d_res in stencil_derivatives(args.stencils, model, f):
                for i in range(len(x)):
                    yield x[i], (amp[i], d_amp[i], None, d_res[i])
        else:
            derivatives = Derivatives(model, f)
            x = sample(args.n)
            for i in range(0, len(x), args.batch):
                yield from zip(x[i : i + args.batch], zip(*derivatives(x[i : i + args.batch])))

    parameters, results, powers = [], {}, {p: [] for p in weights}
    start = time.time()
    for x, row in rows():
        parameters.append(x)
        for p, w in weights.items():
            g, share = metrics(f, w, m, *row)
            powers[p].append(share)
            for k in g:
                results.setdefault(f"{p} {k}", []).append(g[k])
        if len(parameters) % 64 == 0:
            print(f"{len(parameters)}: {time.time() - start:.0f} s", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out, parameters=np.array(parameters), modes=np.array([(q.l, q.m) for q in modes]), box=BOX,
        psds=list(weights), **{k: np.array(v) for k, v in results.items()},
        **{f"{p} power": np.array(v) for p, v in powers.items()},
    )


# -- report -------------------------------------------------------------------


def in_coordinates(g, x, coordinates):
    """``g (n, ..., 5, 5)`` in the coordinates (linear or log_lambda) rescaled
    so that the box is the unit cube."""
    scale = np.tile(BOX[:, 1] - BOX[:, 0], (len(x), 1))
    if coordinates == "log_lambda":
        scale[:, 1:3] = (1 + x[:, 1:3]) * np.diff(np.log1p(BOX[1:3]), axis=1)[:, 0]
    scale = scale.reshape(len(x), *[1] * (g.ndim - 3), 5)
    return g * scale[..., :, None] * scale[..., None, :]


def matrix_function(g, function):
    values, vectors = np.linalg.eigh(g)
    return np.einsum("...ij,...j,...kj->...ik", vectors, function(values), vectors)


def floored(g, floor):
    """``g`` plus ``floor`` times the median of its largest eigenvalue."""
    return g + floor * np.median(np.linalg.eigvalsh(g)[..., -1]) * np.eye(g.shape[-1])


def log_ratios(a, b):
    """Half the logarithm of the generalized eigenvalues ``(n, 5)`` of ``a``
    with respect to ``b``: of the ratio of their distances."""
    inverse_root = matrix_function(b, lambda v: v**-0.5)
    return 0.5 * np.log(np.linalg.eigvalsh(inverse_root @ a @ np.swapaxes(inverse_root, -1, -2)))


def report_command(args):
    data = np.load(args.out)
    x, psds = data["parameters"], [str(p) for p in data["psds"]]
    if "box" in data:
        BOX[:] = data["box"]
    kinds = [k for k in KINDS if f"ET_5 {k}" in data]
    for floor in args.floors:
        print(f"\n##### eigenvalue floor {floor:g} of the median largest")
        print("\n== spectrum (median over the box of sqrt(eigenvalue / largest), ET_5, all modes) and "
              "variation over the box (std of the log ratio of distances to the log-Euclidean mean; "
              "0 is a constant metric), by coordinates")
        for kind in kinds:
            for coordinates in ("linear", "log_lambda"):
                g = floored(in_coordinates(data[f"ET_5 {kind}"][:, 0], x, coordinates), floor)
                values = np.linalg.eigvalsh(g)[:, ::-1]
                mean = matrix_function(matrix_function(g, np.log).mean(0), np.exp)
                ratio = log_ratios(g, np.broadcast_to(mean, g.shape))
                print(f"{kind:>17} {coordinates:>10}: largest sqrt(eig) 5/50/95% "
                      + "/".join(f"{q:8.2e}" for q in np.quantile(np.sqrt(values[:, 0]), [.05, .5, .95]))
                      + "  spectrum " + " ".join(f"{v:.1e}" for v in np.median(np.sqrt(values / values[:, :1]), axis=0))
                      + f"  variation {ratio.std():.2f} (of the overall scale {ratio.mean(1).std():.2f})")
        print("\n== PSD dependence: std, and 1/99% range, of the log ratio of distances to ET_5's, "
              "a constant overall scale removed (linear coordinates, all modes)")
        for kind in kinds:
            reference = floored(in_coordinates(data[f"ET_5 {kind}"][:, 0], x, "linear"), floor)
            for p in psds:
                ratio = log_ratios(floored(in_coordinates(data[f"{p} {kind}"][:, 0], x, "linear"), floor), reference)
                ratio -= ratio.mean()
                print(f"{kind:>17} {p:>13}: std {ratio.std():.2f}  1-99% [{np.quantile(ratio, .01):+.2f}, {np.quantile(ratio, .99):+.2f}]"
                      f"  per point scale {ratio.mean(1).std():.2f}")
    print("\n== each mode's metric against the sum's (residual, ET_5, log ratio std; normalized by the mode's own power)")
    reference = floored(in_coordinates(data["ET_5 residual"][:, 0], x, "linear"), 1e-4)
    power = data["ET_5 power"]
    for i, mode in enumerate(data["modes"]):
        share = power[:, i] / power.sum(1)
        ok = share > 1e-12
        g = floored(in_coordinates(data["ET_5 residual"][:, i + 1] / share[:, None, None], x, "linear")[ok], 1e-4)
        ratio = log_ratios(g, reference[ok])
        print(f"{tuple(mode)}: std {(ratio - ratio.mean()).std():.2f}")


# -- checks -------------------------------------------------------------------


def check_mismatch_command(args):
    """1 - overlap, maximized over time and phase, against d^T g d / 2 for small
    displacements, (2,2), at a single mass."""
    from scipy.optimize import minimize_scalar

    model = open_oracle(args.oracle)
    f, weights = quadrature(None, model)
    w = weights[args.weights][0]
    derivatives = Derivatives(model, f)
    m = np.array([q.m for q in derivatives.modes], dtype=float)
    i22 = [(q.l, q.m) for q in derivatives.modes].index((2, 2))
    rng = np.random.default_rng(1)
    for x in sample(8, seed=3):
        x = np.clip(x, BOX[:, 0] + 0.05 * np.ptp(BOX, axis=1), BOX[:, 1] - 0.05 * np.ptp(BOX, axis=1))
        g, rho2 = metrics(f, w, m, *(a[0] for a in derivatives(x[None])))
        g = g["full_fixed_mass"][1 + i22] * rho2.sum() / rho2[i22]
        direction = rng.normal(size=5) * np.ptp(BOX, axis=1)
        ratios = []
        for target in (1e-3, 1e-4, 1e-5):
            d = direction * np.sqrt(2 * target / (direction @ g @ direction))
            (a0, p0), (a1, p1) = (
                tuple(np.asarray(v)[i22] for v in derivatives._waveform(y)) for y in (x, x + d)
            )
            norm = np.sqrt(np.sum(a0**2 * w) * np.sum(a1**2 * w))
            def loss(t):
                return 1 - abs(np.sum(a0 * a1 * w * np.exp(1j * (p1 - p0 - 2 * np.pi * f * t)))) / norm
            # time that flattens the phase difference, then refine around it
            slope = np.polyfit(f, p1 - p0, 1, w=a0 * np.sqrt(w))[0] / (2 * np.pi)
            best = minimize_scalar(loss, bounds=(slope - 2e-4, slope + 2e-4), method="bounded", options=dict(xatol=1e-12))
            ratios.append(best.fun / target)
        print(np.round(x, 3), "mismatch / quadratic form at 1e-3, 1e-4, 1e-5:", np.round(ratios, 3), flush=True)


def lengths(g):
    return " ".join(f"{v:9.2e}" for v in np.sqrt(np.diagonal(g)))


def check_teob_command(args):
    """The ``residual`` metric from central differences of TEOBResumS waveforms,
    against the model's, ET from 5 Hz."""
    model = open_oracle(args.oracle)
    f, weights = quadrature(None, model)
    derivatives = Derivatives(model, f)
    m = np.array([q.m for q in derivatives.modes], dtype=float)
    x = sample(args.n, seed=5)
    # small Lambda; q to 1; large Lambda_1 with small Lambda_2 at high q (the known bad corner)
    special = np.array([[1.5, 5, 5, .1, -.1], [2.0, 10, 20, 0, 0], [1.02, 400, 500, .1, .1],
                        [2.8, 4500, 50, 0.1, 0.1], [2.5, 3000, 20, -.3, .2]])
    x = np.concatenate([special, x])
    margin = np.array([0.01, 0, 0, 0.01, 0.01])
    x = np.clip(x, BOX[:, 0] + margin + np.array([0, 4, 4, 0, 0]), BOX[:, 1] - margin)
    step = np.column_stack([np.full(len(x), 4e-3), np.minimum(.02 * (x[:, 1] + 20), x[:, 1] - 1.01),
                            np.minimum(.02 * (x[:, 2] + 20), x[:, 2] - 1.01), *np.full((2, len(x)), 4e-3)])
    step *= args.step
    moved = np.concatenate([x[:, None] + s * step[:, :, None] * np.eye(5) for s in (1, -1)], axis=1)  # (n, 10, 5)
    params = ParameterSet(moved.reshape(-1, 5)).waveform_parameters(model.dataset)
    indices = {mode: model.mode_models[mode].downsampling_indices for mode in model.modes}
    references = {mode: model.mode_models[mode].dataset.amplitude_reference_parameters for mode in model.modes}
    _, amplitudes, phases = model._multimode_mode_residuals(
        params, model.dataset.frequencies, indices, references, n_jobs=args.jobs, keep_invalid=True)
    amp, d_amp, d_phase, d_res = derivatives(x)
    fd_amp, fd_res = np.array(d_amp), np.array(d_res)
    import jax.numpy as jnp
    model_residuals = derivatives.surrogate._residuals(jnp, jnp.asarray(x))
    for i, mode in enumerate(model.modes):
        frozen = derivatives.surrogate._frozen.modes_arrays[i]
        for name, raw, nodes, target in (("a", amplitudes[mode], frozen.amp_spline, fd_amp), ("p", phases[mode], frozen.phase_spline, fd_res)):
            raw = np.asarray(raw, dtype=float).reshape(len(x), 2, 5, -1)
            diff = (raw[:, 0] - raw[:, 1]) / (2 * step[:, :, None])
            grid = CubicSpline(np.asarray(nodes.knots), np.eye(len(nodes.knots)), axis=0)
            if name == "p":
                target[:, :, i] = diff @ grid(f).T
            else:  # times the reference amplitude, in the units of the model's
                reference = np.asarray(frozen.pn_amp_nodes)
                model_amp = (np.asarray(model_residuals[i][0]) * reference) @ grid(f).T
                target[:, :5, i] = (diff * reference) @ grid(f).T * np.nanmedian(amp[:, i] / model_amp)
    for j in range(len(x)):
        if not (np.isfinite(fd_amp[j]).all() and np.isfinite(fd_res[j]).all()):
            print(np.round(x[j], 3), "TEOBResumS failed, or outside of the range of the oracle")
            continue
        a = metrics(f, weights["ET_5"], m, amp[j], d_amp[j], d_phase[j], d_res[j])[0]["residual"]
        b = metrics(f, weights["ET_5"], m, amp[j], fd_amp[j], d_phase[j], fd_res[j])[0]["residual"]
        difference = metrics(f, weights["ET_5"], m, amp[j], d_amp[j] - fd_amp[j], d_phase[j], d_res[j] - fd_res[j])[0]["residual"]
        a, b = (in_coordinates(g[None], x[j : j + 1], "linear")[0] for g in (a, b))
        floors = [floored(g, 1e-4) for g in (a[:1], b[:1])]
        print(np.round(x[j], 3), f"|g_jax - g_teob| / |g_teob| all {np.linalg.norm(a[0] - b[0]) / np.linalg.norm(b[0]):.1e}"
              f"  (2,2) {np.linalg.norm(a[1] - b[1]) / np.linalg.norm(b[1]):.1e}"
              f"  log distance ratio, floored 1e-4: {np.abs(log_ratios(*floors)).max():.3f}"
              # the lengths of the five derivatives, in units of the box: the model's, TEOBResumS's, and of their difference
              f"\n    jax {lengths(a[0])}\n   teob {lengths(b[0])}"
              f"\n   diff {lengths(in_coordinates(difference[None], x[j : j + 1], 'linear')[0][0])}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(required=True)
    c = commands.add_parser("metric")
    c.add_argument("out")
    c.add_argument("--n", type=int, default=2048)
    c.add_argument("--batch", type=int, default=16)
    c.add_argument("--stencils", help="from the waveforms of this directory instead")
    c.add_argument("--masses", type=int, default=N_MASSES, help="to average over")
    c.add_argument("--psd-dir")
    c.set_defaults(run=metric_command)
    c = commands.add_parser("stencils")
    c.add_argument("directory")
    c.add_argument("--n", type=int, default=1024)
    c.add_argument("--shard", type=int, default=32)
    c.add_argument("--jobs", type=int, default=8)
    c.set_defaults(run=stencils_command)
    c = commands.add_parser("report")
    c.add_argument("out")
    c.add_argument("--floors", type=float, nargs="+", default=[1e-6, 1e-4, 1e-2])
    c.set_defaults(run=report_command)
    c = commands.add_parser("check-mismatch")
    c.add_argument("--weights", default="ET_5_M4", help="those of a single mass")
    c.set_defaults(run=check_mismatch_command)
    c = commands.add_parser("check-teob")
    c.add_argument("--n", type=int, default=20)
    c.add_argument("--jobs", type=int, default=8)
    c.add_argument("--step", type=float, default=1.0, help="multiplies the finite-difference steps")
    c.set_defaults(run=check_teob_command)
    for c in commands.choices.values():
        c.add_argument("--oracle", default=ORACLE, help="shipped model the derivatives are of")
        c.add_argument("--lambda-max", type=float, default=LAMBDA_MAX, help="of the box")
    args = parser.parse_args()
    BOX[1:3, 1] = args.lambda_max
    args.run(args)


if __name__ == "__main__":
    main()
