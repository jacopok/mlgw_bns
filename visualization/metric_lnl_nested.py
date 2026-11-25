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


def known(pattern):
    parts = [np.load(f) for f in sorted(glob.glob(pattern)) if "partial" not in f]
    keys = [f"{m}/{k}" for m in ("mlgw", "teob") for k in ("lnl", "max", "dt", "dphi")]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ["index", "stored"] + keys}
    return out, bool(parts[0]["phase_flip"])


def initialize(fine):
    import surrogate_fine_nested as s
    s.MODELS = ["mlgw"]
    if hasattr(s, "marginal"):  # the (t, phi) marginal, slow and not needed here
        s.marginal = lambda *_: (np.nan, np.nan, np.nan)
    s._init(fine, {})


def work(task):
    import surrogate_fine_nested as s
    i, theta = task
    e = s.evaluate(s.ST["fine"]["mlgw"], theta)
    return i, [e[k] for k in ("lnl", "max", "dt", "dphi")]


def nested_samples(run_dir):
    with open(os.path.join(run_dir, "BNS_result.json")) as file:
        result = json.load(file)
    post, nested = result["posterior"]["content"], result["nested_samples"]["content"]
    fixed = {k: post[k][0] for k in result.get("fixed_parameter_keys", []) if k in post and np.ndim(post[k][0]) == 0}
    return nested, result["search_parameter_keys"], fixed


def run(a):
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
    with ProcessPoolExecutor(a.npool, mp_context=multiprocessing.get_context("spawn"), initializer=initialize, initargs=(fine,)) as pool:
        for j, v in pool.map(work, tasks, chunksize=4):
            values[j] = v
            if len(values) % 500 == 0:
                print(len(values), flush=True)
    array = np.array([values[int(j)] for j in index])
    np.savez(a.out, index=index, **{k: array[:, c] for c, k in enumerate(("lnl", "max", "dt", "dphi"))})


def efficiency(w, e):
    ok = np.isfinite(e) & (w > 0)
    w, e = w[ok], e[ok]
    r = w * np.exp(e - e.max())
    mean = np.average(e, weights=w)
    return (r.sum() ** 2 / (r**2).sum()) / (w.sum() ** 2 / (w**2).sum()), mean, np.sqrt(np.average((e - mean) ** 2, weights=w))


def report(a):
    table, _ = known(a.parts)
    weights = np.asarray(nested_samples(a.run_dir)[0]["weights"], dtype=float)
    position = {int(j): n for n, j in enumerate(table["index"])}
    models = {"run's model, all samples": None}
    models.update({g: [np.load(f) for f in sorted(glob.glob(g))] for g in a.out})
    for name, parts in models.items():
        if parts is None:
            rows = np.arange(len(table["index"]))
            values = {k: table[f"mlgw/{k}"] for k in ("lnl", "max", "dt", "dphi")}
        else:
            index = np.concatenate([p["index"] for p in parts])
            rows = np.array([position[int(j)] for j in index])
            values = {k: np.concatenate([p[k] for p in parts]) for k in ("lnl", "max", "dt", "dphi")}
        w = weights[table["index"][rows]]
        print(name, f"({len(rows)} samples, weight {w.sum() / weights.sum():.4f})")
        for label, reference in (("", values), *([("   the run's model, same samples", {k: table[f'mlgw/{k}'][rows] for k in values})] if parts else [])):
            for k in ("lnl", "max"):
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
    c.add_argument("--min-weight", type=float, default=1e-3)
    c.set_defaults(run=run)
    c = commands.add_parser("report")
    c.add_argument("run_dir")
    c.add_argument("parts")
    c.add_argument("out", nargs="+")
    c.set_defaults(run=report)
    args = parser.parse_args()
    args.run(args)
