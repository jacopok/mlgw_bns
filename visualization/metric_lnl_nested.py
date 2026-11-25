"""Reweighting efficiency of models to TEOBResumS on an ET teob_injections run:
the likelihood of a model at the nested samples where TEOBResumS' is known
(xg_pe scripts/surrogate_fine_nested.py, whose functions, data and (t, phi)
maximization this uses; run from where it and its modules are), with the
mlgw_bns and bilby_xG first on PYTHONPATH.

    python metric_lnl_nested.py run RUN_DIR FINE_PICKLE PARTS_GLOB OUT.npz --chunk i/K [--npool 24]
    python metric_lnl_nested.py report RUN_DIR PARTS_GLOB OUT_GLOB...

Only the samples of weight above --min-weight of the largest one; the efficiency
is the ratio of the Kish sizes after and before reweighting.
"""
import argparse
import glob
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np

sys.path.insert(0, os.getcwd())


def known(pattern, inner=False):
    parts = [np.load(f) for f in sorted(glob.glob(pattern)) if "partial" not in f]
    keys = [f"{m}/{k}" for m in ("mlgw", "teob") for k in ("lnl", "max", "dt", "dphi")]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ["index", "stored"] + keys}
    if inner:
        m = np.array([int(str(v).split(",")[1]) for v in parts[0]["modes"]])
        for model in ("mlgw", "teob"):
            out[f"{model}/phase"] = np.concatenate(
                [phase_maximum(p[f"{model}/z"].sum(1), p[f"{model}/G"].sum(1), m) for p in parts])
        out["m"] = m
    return out, bool(parts[0]["phase_flip"])


def own_phase(run, target):
    """In place of ``TEOBResumSRun._fit_delta``, which turns TEOBResumS to the
    orbital phase of largest overlap with the surrogate: the turn for which its
    (2,2) phase at the top of the trained band is what the merger convention
    makes it. The surrogate has that phase by construction, whatever its errors
    (its tangent there has no intercept), so nothing is fitted to it."""
    from mlgw_bns.higher_order_modes import Mode, mode_to_k

    top = np.asarray(run.top)
    grid = np.union1d(np.geomspace(top[1] / 2, top[1], 16), top)
    at = int(np.searchsorted(grid, top[1]))

    def phase(delta):
        hp = run.run(grid, [mode_to_k(Mode(2, 2))], delta)[0]
        return np.angle(hp[at] * np.exp(-2j * np.pi * run.merger_time(0.0) * grid[at]))

    # (2,2) alone: plus polarization of the surrogate at the same frequency
    reference = np.angle(own_phase.surrogate(np.array([top[1]]))[0])
    start = phase(0.0)
    sign = np.sign(np.angle(np.exp(1j * (phase(0.2) - start))))  # each mode turns by sign m delta
    delta = sign * np.angle(np.exp(1j * (reference - start))) / 2  # modulo pi: the odd modes choose
    grid_fit = np.geomspace(25.0, 2000.0, 400)
    hp_s, hc_s = target(grid_fit)
    overlaps = [float(np.sum(np.conj(run.polarisations(grid_fit, d)[0]) * hp_s * grid_fit ** (-4 / 3)).real) for d in (delta, delta + np.pi)]
    run.overlap = np.nan
    return delta if overlaps[0] > overlaps[1] else delta + np.pi


def initialize_teob(fine):
    """Workers of the raw TEOBResumS likelihood (``--teob``)."""
    import bilby_xG.teobresums_mlgw as tm
    import surrogate_fine_nested as s

    source = tm.mlgw_bns_individual_modes
    make = tm.TEOBResumSRun.__init__

    def init(run, model, start, f0, minimum_frequency, target):
        P = start  # aligned: the surrogate's (2,2) alone, at the binary of the run

        def surrogate(f):
            out = source(f, P.total_mass, P.mass_ratio, P.chi_1[2], P.chi_2[2], P.lambda_1, P.lambda_2,
                         P.distance_mpc, P.inclination, initialize_teob.angle, mode_array=[(2, 2)])
            return out["2,2"]["plus"]

        own_phase.surrogate = surrogate
        make(run, model, start, f0, minimum_frequency, target)

    tm.TEOBResumSRun.__init__ = init
    tm.TEOBResumSRun._fit_delta = own_phase
    initialize_teob.angle = 0.0
    original = tm.teobresums_mlgw_individual_modes

    def teob(frequency_array, M, q, chi1z, chi2z, LambdaAl2, LambdaBl2, distance, inclination, coalescence_angle, **kwargs):
        initialize_teob.angle = coalescence_angle
        return original(frequency_array, M, q, chi1z, chi2z, LambdaAl2, LambdaBl2, distance, inclination, coalescence_angle, **kwargs)

    teob.evaluates_on_bin_edges = True
    tm.teobresums_mlgw_individual_modes = teob
    if hasattr(s, "marginal"):
        s.marginal = lambda *_: (np.nan, np.nan, np.nan)
    s.MODELS = ["mlgw", "teob"]
    s._init(fine, {})


def initialize(fine):
    import surrogate_fine_nested as s
    s.MODELS = ["mlgw"]
    if hasattr(s, "marginal"):  # the (t, phi) marginal, slow and not needed here
        s.marginal = lambda *_: (np.nan, np.nan, np.nan)
    s._init(fine, {})


def work(task):
    import surrogate_fine_nested as s
    i, theta = task
    e = s.evaluate(s.ST["fine"]["teob" if os.environ.get("LNL_TEOB") else "mlgw"], theta)
    # summed over interferometers: what a turn of the orbital phase needs
    return i, [e[k] for k in ("lnl", "max", "dt", "dphi")], e["z"].sum(0), e["G"].sum(0)


def nested_samples(run_dir):
    with open(os.path.join(run_dir, "BNS_result.json")) as file:
        result = json.load(file)
    post, nested = result["posterior"]["content"], result["nested_samples"]["content"]
    fixed = {k: post[k][0] for k in result.get("fixed_parameter_keys", []) if k in post and np.ndim(post[k][0]) == 0}
    return nested, result["search_parameter_keys"], fixed


def run(a):
    if a.teob:
        os.environ["LNL_TEOB"] = "1"
    table, flip = known(a.parts)
    nested, keys, fixed = nested_samples(a.run_dir)
    weights = np.asarray(nested["weights"], dtype=float)
    index = np.sort(table["index"][weights[table["index"]] > a.min_weight * weights.max()])
    i, k = (int(v) for v in a.chunk.split("/"))
    index = index[i::k][: a.limit]
    with open(a.fine_pickle, "rb") as file:
        fine, _ = pickle.load(file)
    if flip:
        from bilby_xG.likelihood import _clear_waveform_cache
        from xg_pe.phase_convention import FlippedPhaseConversion
        generator = fine.waveform_generator
        generator.parameter_conversion = FlippedPhaseConversion(generator.parameter_conversion)
        _clear_waveform_cache(generator)
    import mlgw_bns
    print(mlgw_bns.__file__, len(index), "samples, flip", flip, flush=True)
    values = {}
    tasks = [(int(j), dict(fixed, **{key: float(nested[key][j]) for key in keys})) for j in index]
    with ProcessPoolExecutor(a.npool, mp_context=multiprocessing.get_context("spawn"), initializer=initialize_teob if a.teob else initialize, initargs=(fine,)) as pool:
        for j, *v in pool.map(work, tasks, chunksize=4):
            values[j] = v
            if len(values) % 500 == 0:
                print(len(values), flush=True)
    array = np.array([values[int(j)][0] for j in index])
    z, G = (np.array([values[int(j)][c] for j in index]) for c in (1, 2))
    np.savez(a.out, index=index, z=z, G=G, **{k: array[:, c] for c, k in enumerate(("lnl", "max", "dt", "dphi"))})


def phase_maximum(z, G, m):
    """ln L maximized over the orbital phase alone, the time fixed (the TEOBResumS
    here are turned, sample by sample, to match the run's own surrogate: as
    evaluated they are not comparable to other models)."""
    grid = np.linspace(-0.03, 0.03, 601)  # rad
    out = np.empty(len(z))
    for start in range(0, len(z), 512):
        rows = slice(start, start + 512)
        e = np.exp(1j * np.outer(grid, m))
        lnl = (z[rows] @ e.T).real - 0.5 * np.einsum("nkl,pk,pl->np", G[rows], e, e.conj(), optimize=True).real
        i = np.clip(lnl.argmax(axis=1), 1, len(grid) - 2)
        a, b, c = (lnl[np.arange(len(i)), i + k] for k in (-1, 0, 1))
        out[rows] = b - (a - c) ** 2 / (8 * (a - 2 * b + c))  # vertex of the parabola
    return out


def efficiency(w, e):
    ok = np.isfinite(e) & (w > 0)
    w, e = w[ok], e[ok]
    r = w * np.exp(e - e.max())
    mean = np.average(e, weights=w)
    return (r.sum() ** 2 / (r**2).sum()) / (w.sum() ** 2 / (w**2).sum()), mean, np.sqrt(np.average((e - mean) ** 2, weights=w))


def report(a):
    table, _ = known(a.parts, inner=True)
    weights = np.asarray(nested_samples(a.run_dir)[0]["weights"], dtype=float)
    position = {int(j): n for n, j in enumerate(table["index"])}
    models = {"run's model, all samples": None}
    models.update({g: [np.load(f) for f in sorted(glob.glob(g))] for g in a.out})
    for name, parts in models.items():
        if parts is None:
            rows = np.arange(len(table["index"]))
            values = {k: table[f"mlgw/{k}"] for k in ("lnl", "phase", "max", "dt", "dphi")}
        else:
            index = np.concatenate([p["index"] for p in parts])
            rows = np.array([position[int(j)] for j in index])
            values = {k: np.concatenate([p[k] for p in parts]) for k in ("lnl", "max", "dt", "dphi")}
            values["phase"] = np.concatenate([phase_maximum(p["z"], p["G"], table["m"]) for p in parts])
        w = weights[table["index"][rows]]
        print(name, f"({len(rows)} samples, weight {w.sum() / weights.sum():.4f})")
        for label, reference in (("", values), *([("   the run's model, same samples", {k: table[f'mlgw/{k}'][rows] for k in values})] if parts else [])):
            for k in ("lnl", "phase", "max"):
                e = table[f"teob/{k}"][rows] - reference[k]
                print(f"  {label} {k}: efficiency %.4f, mean %+.4f, std %.4f" % efficiency(w, e))
            dt = 1e6 * (table["teob/dt"][rows] - reference["dt"])
            mean = np.average(dt, weights=w)
            print(f"  {label} time shift: mean {mean:+.3f} us, std {np.sqrt(np.average((dt - mean) ** 2, weights=w)):.3f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(required=True)
    c = commands.add_parser("run")
    for name in ("run_dir", "fine_pickle", "parts", "out"):
        c.add_argument(name)
    c.add_argument("--chunk", default="0/1")
    c.add_argument("--npool", type=int, default=24)
    c.add_argument("--limit", type=int)
    c.add_argument("--teob", action="store_true", help="TEOBResumS instead, with its own orbital phase")
    c.add_argument("--min-weight", type=float, default=1e-3)
    c.set_defaults(run=run)
    c = commands.add_parser("report")
    c.add_argument("run_dir")
    c.add_argument("parts")
    c.add_argument("out", nargs="+")
    c.set_defaults(run=report)
    args = parser.parse_args()
    args.run(args)
