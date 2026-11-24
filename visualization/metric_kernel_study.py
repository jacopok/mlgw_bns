r"""Is the Fisher metric of the waveforms a good distance for the kernels of the
regressors? Stage 1: the metric itself, and how much the PSD matters.

::

    M=visualization/metric_kernel_study.py
    python $M metric OUT.npz --n 2048 --psd-dir DIR   # the metrics at N points (JAX)
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


def quadrature(psd_dir, n_grid=768):
    """Frequencies and ``name -> weights`` of the inner product."""
    f = np.geomspace(5.0, F_MAX, n_grid)
    trapezoid = _trapezoid_weights(f)
    weights = {}
    for name, (file, f_min) in PSDS.items():
        if file is None:
            inverse = np.ones_like(f)
        else:
            path = PSD_PATH / "ET_psd.txt" if file == "ET" else Path(psd_dir or ".") / f"{file}_psd.txt"
            if not path.exists():
                print(f"no {path}: skipping {name}", file=sys.stderr)
                continue
            table = np.loadtxt(path)
            inverse = 1 / np.exp(np.interp(np.log(f), np.log(table[:, 0]), np.log(table[:, 1])))
        weights[name] = 4 * trapezoid * inverse * (f >= f_min)
    return f, weights


def sample(n, seed=0):
    """Half uniform in Lambda, half in log(1 + Lambda)."""
    from scipy.stats import qmc

    u = qmc.Sobol(6, seed=seed).random(n)
    x = BOX[:, 0] + u[:, :5] * (BOX[:, 1] - BOX[:, 0])
    log = np.expm1(np.log1p(BOX[1:3, 0]) + u[:, 1:3] * np.diff(np.log1p(BOX[1:3]), axis=1)[:, 0])
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
    each alone; and the rho^2 of each mode."""
    root = np.sqrt(weights)  # (k,)
    rho2 = np.sum(amp**2 * weights, axis=1)
    flat = lambda re, im: np.concatenate([re * root, im * root], axis=-1)  # noqa: E731
    dh = flat(d_amp, amp * d_phase)                # (6, modes, 2k)
    dr = flat(d_amp[:5], amp * d_res)              # (5, modes, 2k)
    zero = np.zeros_like(amp)
    t = flat(zero, amp * 2 * np.pi * f)            # (modes, 2k)
    phi = flat(zero, amp * m[:, None])
    size = flat(amp, zero)
    out = {kind: [] for kind in KINDS}
    for scope in [slice(None)] + [slice(i, i + 1) for i in range(len(m))]:
        pick = lambda v: v[..., scope, :].reshape(*v.shape[:-2], -1)  # noqa: E731
        wave, mass = pick(dh)[:5], pick(dh)[5]
        shifts = [pick(t), pick(phi)]
        for kind, vectors in (
            ("full", project(wave, np.array(shifts + [pick(size), mass]))),
            ("full_fixed_mass", project(wave, np.array(shifts + [pick(size)]))),
            ("residual", pick(dr)),
            ("residual_shifted", project(pick(dr), np.array(shifts))),
        ):
            out[kind].append(vectors @ vectors.T / rho2.sum())
    return {k: np.array(v) for k, v in out.items()}, rho2


def open_oracle():
    return Model.default_for_testing(ORACLE)


def metric_command(args):
    f, weights = quadrature(args.psd_dir)
    model = open_oracle()
    derivatives = Derivatives(model, f)
    m = np.array([mode.m for mode in derivatives.modes], dtype=float)
    x = sample(args.n)
    results = {f"{p} {k}": [] for p in weights for k in KINDS}
    powers = {p: [] for p in weights}
    start = time.time()
    for i in range(0, len(x), args.batch):
        parts = derivatives(x[i : i + args.batch])
        for row in zip(*parts):
            for p, w in weights.items():
                g, rho2 = metrics(f, w, m, *row)
                powers[p].append(rho2)
                for k in KINDS:
                    results[f"{p} {k}"].append(g[k])
        print(f"{i + args.batch}/{len(x)}: {time.time() - start:.0f} s", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out, parameters=x, modes=np.array([(q.l, q.m) for q in derivatives.modes]),
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
    for floor in args.floors:
        print(f"\n##### eigenvalue floor {floor:g} of the median largest")
        print("\n== spectrum (median over the box of sqrt(eigenvalue / largest), ET_5, all modes) and "
              "variation over the box (std of the log ratio of distances to the log-Euclidean mean; "
              "0 is a constant metric), by coordinates")
        for kind in KINDS:
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
        for kind in KINDS:
            reference = floored(in_coordinates(data[f"ET_5 {kind}"][:, 0], x, "linear"), floor)
            for p in psds:
                ratio = log_ratios(floored(in_coordinates(data[f"{p} {kind}"][:, 0], x, "linear"), floor), reference)
                ratio -= ratio.mean()
                print(f"{kind:>17} {p:>9}: std {ratio.std():.2f}  1-99% [{np.quantile(ratio, .01):+.2f}, {np.quantile(ratio, .99):+.2f}]"
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
    displacements, (2,2), ET from 5 Hz."""
    from scipy.optimize import minimize_scalar

    f, weights = quadrature(None)
    w = weights["ET_5"]
    derivatives = Derivatives(open_oracle(), f)
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


def check_teob_command(args):
    """The ``residual`` metric from central differences of TEOBResumS waveforms,
    against the model's, ET from 5 Hz."""
    f, weights = quadrature(None)
    model = open_oracle()
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
    moved = np.concatenate([x[:, None] + s * step[:, :, None] * np.eye(5) for s in (1, -1)], axis=1)  # (n, 10, 5)
    params = ParameterSet(moved.reshape(-1, 5)).waveform_parameters(model.dataset)
    indices = {mode: model.mode_models[mode].downsampling_indices for mode in model.modes}
    references = {mode: model.mode_models[mode].dataset.amplitude_reference_parameters for mode in model.modes}
    _, amplitudes, phases = model._multimode_mode_residuals(
        params, model.dataset.frequencies, indices, references, n_jobs=args.jobs, keep_invalid=True)
    amp, d_amp, d_phase, d_res = derivatives(x)
    fd_amp, fd_res = np.array(d_amp), np.array(d_res)
    for i, mode in enumerate(model.modes):
        frozen = derivatives.surrogate._frozen.modes_arrays[i]
        for name, raw, nodes, target in (("a", amplitudes[mode], frozen.amp_spline, fd_amp), ("p", phases[mode], frozen.phase_spline, fd_res)):
            raw = np.asarray(raw, dtype=float).reshape(len(x), 2, 5, -1)
            diff = (raw[:, 0] - raw[:, 1]) / (2 * step[:, :, None])
            grid = CubicSpline(np.asarray(nodes.knots), np.eye(len(nodes.knots)), axis=0)
            if name == "p":
                target[:, :, i] = diff @ grid(f).T
            else:  # times the reference amplitude, in the units of the model's
                model_res = np.asarray(derivatives.surrogate._residuals(np, np.asarray(x))[i][0])
                scale = np.nanmedian(amp[:, i] / (model_res @ grid(f).T), axis=0)
                target[:, :5, i] = diff @ grid(f).T * scale
    for j in range(len(x)):
        if not (np.isfinite(fd_amp[j]).all() and np.isfinite(fd_res[j]).all()):
            print(np.round(x[j], 3), "TEOBResumS failed")
            continue
        a = metrics(f, weights["ET_5"], m, amp[j], d_amp[j], d_phase[j], d_res[j])[0]["residual"]
        b = metrics(f, weights["ET_5"], m, amp[j], fd_amp[j], d_phase[j], fd_res[j])[0]["residual"]
        a, b = (in_coordinates(g[None], x[j : j + 1], "linear")[0] for g in (a, b))
        floors = [floored(g, 1e-4) for g in (a[:1], b[:1])]
        print(np.round(x[j], 3), f"|g_jax - g_teob| / |g_teob| all {np.linalg.norm(a[0] - b[0]) / np.linalg.norm(b[0]):.1e}"
              f"  (2,2) {np.linalg.norm(a[1] - b[1]) / np.linalg.norm(b[1]):.1e}"
              f"  log distance ratio, floored 1e-4: {np.abs(log_ratios(*floors)).max():.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(required=True)
    c = commands.add_parser("metric")
    c.add_argument("out")
    c.add_argument("--n", type=int, default=2048)
    c.add_argument("--batch", type=int, default=16)
    c.add_argument("--psd-dir")
    c.set_defaults(run=metric_command)
    c = commands.add_parser("report")
    c.add_argument("out")
    c.add_argument("--floors", type=float, nargs="+", default=[1e-6, 1e-4, 1e-2])
    c.set_defaults(run=report_command)
    c = commands.add_parser("check-mismatch")
    c.set_defaults(run=check_mismatch_command)
    c = commands.add_parser("check-teob")
    c.add_argument("--n", type=int, default=20)
    c.add_argument("--jobs", type=int, default=8)
    c.set_defaults(run=check_teob_command)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
