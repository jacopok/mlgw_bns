"""Does the sub-sample SPA attachment of TEOBResumS improve a (2,2) model?

Generate the same waveforms with two EOBRun_modules, then compare: the PCA
truncation, and kernel ridges on each, tested on held-out waveforms.

    PYTHONPATH=.:<build dir> python visualization/clean_training_data.py generate OUT.npz [N]
    python visualization/clean_training_data.py compare JITTERY.npz CLEAN.npz
"""
import os
import sys
import warnings

import numpy as np
from scipy.interpolate import CubicSpline

from mlgw_bns import Model
from mlgw_bns.dataset_generation import EffectiveParameterGenerator
from mlgw_bns.higher_order_modes import Mode

M22 = Mode(2, 2)
N_TEST = 1024
TIMES_HZ = [100.0, 400.0, 1000.0]


def model():
    warnings.simplefilter("ignore")
    return Model.default_for_testing("default_hom_lambda2000")


def generate(out, n):
    m = model()
    ds = m.mode_models[M22].dataset
    ranges = ds.parameter_ranges
    generator = EffectiveParameterGenerator(dataset=ds, parameter_ranges=ranges, seed=7)
    params = [next(generator) for _ in range(n)]
    parameters, _, phases = m._multimode_mode_residuals(
        params, ds.frequencies,
        downsampling_indices_by_mode={k: v.downsampling_indices for k, v in m.mode_models.items()},
        n_jobs=int(os.environ.get("N_JOBS", "8")), keep_invalid=True)
    knots = ds.frequencies_hz[m.mode_models[M22].downsampling_indices.phase_indices]
    np.savez(out, parameters=parameters, phases=phases[M22], knots=knots)


def times(knots, residual_errors):
    """Errors (us) of the time at the top knot and at TIMES_HZ, and of the
    phase at the top with the time and phase referenced at 100 Hz."""
    spline = CubicSpline(knots, residual_errors, axis=1)
    at = {f: -spline(f, 1) / (2 * np.pi) * 1e6 for f in [*TIMES_HZ, knots[-1]]}
    return at


def report(label, knots, errors):
    at = times(knots, errors)
    t100 = at[100.0]
    print(f"{label:>44}: " + "  ".join(
        f"t({f:.0f}) {np.median(abs(v)):6.3f}/{np.quantile(abs(v), .99):7.3f}" for f, v in at.items())
        + f"   t(top)-t(100) {np.median(abs(at[knots[-1]] - t100)):6.3f} us (median/99%)", flush=True)


def compare(*paths):
    from sklearn.kernel_ridge import KernelRidge

    from mlgw_bns.neural_network import regression_inputs

    data = [np.load(p) for p in paths]
    parameters, knots = data[0]["parameters"], data[0]["knots"]
    ok = np.all([np.isfinite(d["phases"]).all(axis=1) for d in data], axis=0)
    x = regression_inputs(parameters[ok], "log_lambda", np)
    x = (x - x.mean(0)) / x.std(0)
    phases = [d["phases"][ok] for d in data]
    test, train = slice(0, N_TEST), slice(N_TEST, None)
    clean = phases[-1]
    report("jittery - clean (the jitter)", knots, phases[0][test] - clean[test])
    for name, ph in zip(("jittery", "clean"), phases):
        mean = ph[train].mean(0)
        _, _, vt = np.linalg.svd(ph[train] - mean, full_matrices=False)
        for k in (30, 60, 80):
            v = vt[:k]
            report(f"{name}: PCA {k}", knots, (clean[test] - mean) @ v.T @ v + mean - clean[test])
        v = vt[:60]
        for gamma, alpha in ((0.1, 1e-11), (0.1, 1e-9), (0.1, 1e-7)):
            krr = KernelRidge(kernel="rbf", gamma=gamma, alpha=alpha)
            y = (ph[train] - mean) @ v.T
            krr.fit(x[train], y)
            report(f"{name}: PCA 60, KRR alpha {alpha:g}, N {len(y)}", knots,
                   krr.predict(x[test]) @ v + mean - clean[test])


if __name__ == "__main__":
    if sys.argv[1] == "generate":
        generate(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6144)
    else:
        compare(*sys.argv[2:])
