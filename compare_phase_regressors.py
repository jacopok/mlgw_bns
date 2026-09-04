"""Head-to-head of regressors for the per-mode reference-phase leftover.

Generates one EOB dataset (cached to ``--cache``) of reference-phase
targets, applies the production analytic-backbone calibration
(``ModePhasesNN._backbone_design`` linear fit), then fits several
regressors on nested subsets and reports the held-out error mode by mode,
both as the wrapped phase error (what a waveform sees) and the raw
unwrapped leftover-residual RMS (so an under-fitting regressor is
distinguishable from a merely-imprecise one -- the wrapped metric
saturates near ``pi/2`` for anything that misses by more than a couple of
radians).

Candidates kept after the first-round screen (poly / gbt dropped -- they
under-fit the O(100 rad) leftover):

* ``rff``    -- RBFSampler(4000, gamma=1) + Ridge         (current production)
* ``nystroem`` -- Nystroem(2500) + RidgeCV                (data-adaptive kernel)
* ``mlp``    -- MLPRegressor((128, 64), tanh, adam + early stopping)
* ``mlp_lbfgs`` -- MLPRegressor((64, 64), tanh, lbfgs)  [small-n reference]

Run with::

    python compare_phase_regressors.py            # 100k @ 5 Hz, overnight
    python compare_phase_regressors.py --train-size 8000 --initial-frequency-hz 20
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

from mlgw_bns.model import DEFAULT_MODES, Model
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.neural_network import ModePhasesNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ALPHAS = np.logspace(-10.0, 0.0, 21)


def wrapped_abs(delta):
    return np.abs(np.angle(np.exp(1j * np.asarray(delta))))


def generate(args):
    model = Model(
        modes=list(DEFAULT_MODES),
        initial_frequency_hz=args.initial_frequency_hz,
        reference_amplitude=True,
    )
    modes = list(model.modes)
    grid_hz, f_ref_natural, f0 = model._reference_grid(args.grid_points, args.fmax_hz)

    def sweep(n, seed, label):
        gen = model.mode_models[Mode(2, 2)].dataset.make_parameter_generator(seed=seed)
        p, _dt, phi = model._reference_sweep_targets(
            gen, n, grid_hz, f_ref_natural, batch_size=args.batch_size,
            progress_label=label,
        )
        return p, phi

    p_tr, phi_tr = sweep(args.train_size, 1, "train")
    p_val, phi_val = sweep(args.val_size, 2, "val")
    return dict(
        p_tr=p_tr, phi_tr=phi_tr, p_val=p_val, phi_val=phi_val,
        f0=np.float64(f0), modes=np.array([(m.l, m.m) for m in modes]),
        initial_frequency_hz=np.float64(args.initial_frequency_hz),
    )


def build(name):
    if name == "rff":
        return make_pipeline(
            MinMaxScaler(),
            RBFSampler(n_components=4000, gamma=1.0, random_state=42),
            Ridge(alpha=1e-9),
        )
    if name == "rff8k":
        return make_pipeline(
            MinMaxScaler(),
            RBFSampler(n_components=8000, gamma=1.0, random_state=42),
            Ridge(alpha=1e-9),
        )
    if name == "nystroem":
        return make_pipeline(
            MinMaxScaler(),
            Nystroem(n_components=2500, gamma=1.0, random_state=42),
            RidgeCV(alphas=ALPHAS),
        )
    if name == "mlp":
        return make_pipeline(
            MinMaxScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 64), activation="tanh", solver="adam",
                alpha=1e-5, batch_size=256, learning_rate_init=1e-3,
                max_iter=800, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=25, random_state=0,
            ),
        )
    if name == "mlp_lbfgs":
        return make_pipeline(
            MinMaxScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 64), activation="tanh", solver="lbfgs",
                alpha=1e-4, max_iter=2000, random_state=0,
            ),
        )
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument(
        "--subsets", type=str, default="4000,8000,16000,32000,64000,100000"
    )
    parser.add_argument("--grid-points", type=int, default=64)
    parser.add_argument("--fmax-hz", type=float, default=512.0)
    parser.add_argument("--initial-frequency-hz", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--regressors", type=str, default="rff,nystroem,mlp")
    parser.add_argument("--modes", type=str, default="2,1;3,3;4,4")
    parser.add_argument("--cache", type=str, default="")
    parser.add_argument("--out", type=str, default="compare_phase_regressors.png")
    args = parser.parse_args()

    cache = args.cache or f"phase_regressor_data_{args.initial_frequency_hz:g}hz.npz"

    try:
        d = dict(np.load(cache))
        if (
            len(d["p_tr"]) < args.train_size
            or len(d["p_val"]) < args.val_size
            or float(d.get("initial_frequency_hz", -1)) != args.initial_frequency_hz
        ):
            raise FileNotFoundError
        logging.info("loaded cached data from %s (%d train)", cache, len(d["p_tr"]))
    except (FileNotFoundError, KeyError):
        logging.info("generating %d train / %d val @ %g Hz -> %s",
                     args.train_size, args.val_size, args.initial_frequency_hz, cache)
        d = generate(args)
        np.savez(cache, **d)
        logging.info("wrote %s", cache)

    subsets = sorted(int(s) for s in args.subsets.split(","))
    subsets = [n for n in subsets if n <= len(d["p_tr"])]
    regressors = args.regressors.split(",")
    want_modes = [tuple(int(x) for x in m.split(",")) for m in args.modes.split(";")]
    mode_rows = {tuple(r): i for i, r in enumerate(d["modes"].tolist())}
    f0 = float(d["f0"])
    p_tr, phi_tr, p_val, phi_val = d["p_tr"], d["phi_tr"], d["p_val"], d["phi_val"]

    results = {}  # (mode, regressor) -> {n: (wrapped_median, wrapped_90, raw_rms)}
    for lm in want_modes:
        j = mode_rows[lm]
        design_tr = ModePhasesNN._backbone_design(p_tr, lm, f0)
        design_val = ModePhasesNN._backbone_design(p_val, lm, f0)
        calib = LinearRegression().fit(design_tr[: max(subsets)], phi_tr[: max(subsets), j])
        left_tr = phi_tr[:, j] - calib.predict(design_tr)
        left_val = phi_val[:, j] - calib.predict(design_val)
        logging.info("mode %s: leftover std %.2f rad (val %.2f)", lm, left_tr.std(), left_val.std())

        for rname in regressors:
            errs = {}
            for n in subsets:
                if rname == "mlp_lbfgs" and n > 20000:
                    continue  # lbfgs full-batch is impractical past ~20k
                est = build(rname)
                est.fit(p_tr[:n], left_tr[:n])
                resid = est.predict(p_val) - left_val
                werr = wrapped_abs(resid)
                errs[n] = (
                    float(np.median(werr)),
                    float(np.percentile(werr, 90)),
                    float(np.sqrt(np.mean(resid**2))),
                )
                logging.info(
                    "  %-9s n=%-6d  wrapped med %.4f / 90th %.4f  | raw RMS %.4f rad",
                    rname, n, *errs[n],
                )
            results[(lm, rname)] = errs

    print("\n=== held-out error: wrapped median (90th) [rad]  ||  raw leftover RMS [rad] ===")
    for lm in want_modes:
        print(f"\nmode {lm}:")
        hdr = "  regressor  | " + " | ".join(f"n={n}".ljust(26) for n in subsets)
        print(hdr)
        print("-" * len(hdr))
        for rname in regressors:
            e = results[(lm, rname)]
            row = f"  {rname:10s} | " + " | ".join(
                (f"{e[n][0]:.3f} ({e[n][1]:.2f}) | {e[n][2]:.3f}" if n in e else "-").ljust(26)
                for n in subsets
            )
            print(row)

    fig, axes = plt.subplots(2, len(want_modes), figsize=(6 * len(want_modes), 9),
                             squeeze=False)
    for k, lm in enumerate(want_modes):
        for row, (idx, ylabel) in enumerate(
            [(0, "median wrapped phase error [rad]"), (2, "raw leftover residual RMS [rad]")]
        ):
            ax = axes[row][k]
            for rname in regressors:
                e = results[(lm, rname)]
                ns = [n for n in subsets if n in e]
                ax.plot(ns, [e[n][idx] for n in ns], "o-", label=rname)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("training set size")
            ax.set_ylabel(ylabel)
            ax.set_title(f"mode {lm}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
    fig.suptitle(
        f"reference-phase leftover: regressor comparison ({args.initial_frequency_hz:g} Hz)"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
