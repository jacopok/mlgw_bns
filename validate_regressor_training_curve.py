"""Training-set-size study of the shared time-shift / reference-phase regressors.

The ``Model`` pre-pass fits two ridge regressors from the intrinsic
parameters :math:`[q, \\Lambda_1, \\Lambda_2, \\chi_1, \\chi_2]`:

* :class:`~mlgw_bns.neural_network.TimeshiftsNN` --- the scalar cross-mode
  time shift :math:`\\Delta t(\\theta)` (seconds);
* :class:`~mlgw_bns.neural_network.ModePhasesNN` --- the per-mode reference
  phase :math:`\\phi_{\\ell m}(f_0)` (radians), one output per mode, fitted
  as ``a * backbone + b + ridge`` on the Taylor-F2 backbone leftover.

This script generates one EOB dataset of ``--train-size`` points plus a
disjoint ``--val-size`` validation set (different generator seed), then
retrains both regressors on nested subsets ``--subsets`` and measures the
error on both the training subset and the held-out set, mode by mode for
the phases and on its own for the time shift.

With ``--sweep`` it first runs a Ridge cross-validation sweep over
``(gamma, n_components, alpha)`` on the full pre-generated training set
(cheap: one RFF transform + one GCV solve per grid point) and uses the
best hyperparameters for the training curve.

Output: ``{--out}_phases.png`` (one boxplot panel per mode, |phase error|
in rad, train vs val boxes side by side) and ``{--out}_timeshifts.png``
(|Δt error| in s), both with a log-scaled error axis, plus a printed
median / 90th-percentile table.

Run with::

    python validate_regressor_training_curve.py \\
        --train-size 8000 --val-size 1000 --subsets 500,1000,2000,4000,8000 --sweep
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression, RidgeCV

from mlgw_bns.model import DEFAULT_MODES, Model
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.neural_network import ModePhasesNN, TimeshiftsNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

#: (gamma, n_components) grid explored by ``--sweep``; alpha is swept
#: within each combination by :class:`~sklearn.linear_model.RidgeCV`'s
#: efficient leave-one-out, so it is cheap to make it dense.
SWEEP_GAMMAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
SWEEP_N_COMPONENTS = (1000, 4000)
SWEEP_ALPHAS = np.logspace(-10.0, 0.0, 21)


def wrapped(delta: np.ndarray) -> np.ndarray:
    """Phase difference wrapped to ``(-pi, pi]`` --- the reference phase only
    matters modulo :math:`2\\pi` for the reconstructed waveform."""
    return np.angle(np.exp(1j * np.asarray(delta)))


def build_model(args) -> Model:
    kwargs = dict(
        modes=list(DEFAULT_MODES),
        initial_frequency_hz=args.initial_frequency_hz,
        reference_amplitude=True,
    )
    if args.gw_prior:
        from mlgw_bns.dataset_generation import GWPriorUniformSpinParameterGenerator

        kwargs["parameter_generator_class"] = GWPriorUniformSpinParameterGenerator
    return Model(**kwargs)


def make_targets(model, grid_hz, f_ref_natural, n_points, seed, batch_size, label):
    """(parameter_array, timeshifts, reference_phases) for a fresh draw."""
    generator = model.mode_models[Mode(2, 2)].dataset.make_parameter_generator(seed=seed)
    return model._reference_sweep_targets(
        generator, n_points, grid_hz, f_ref_natural,
        batch_size=batch_size, progress_label=label,
    )


def phase_backbone_leftover(p, phi, modes, f0_natural):
    """Replicate ``ModePhasesNN.fit``'s analytic-backbone calibration per mode.

    Linear fit of ``phi_lm(f0)`` on ``[M_lm, q, Lambda_1, Lambda_2,
    chi_1, chi_2]`` (see ``ModePhasesNN._backbone_design``); returns
    ``(leftover, fitters)`` so the same calibration can be re-applied.
    """
    leftover = np.empty_like(phi, dtype=float)
    fitters = []
    for j, mode in enumerate(modes):
        design = ModePhasesNN._backbone_design(p, (mode.l, mode.m), f0_natural)
        lr = LinearRegression().fit(design, phi[:, j])
        leftover[:, j] = phi[:, j] - lr.predict(design)
        fitters.append(lr)
    return leftover, fitters


def ridge_cv_sweep(train, modes, f0_natural):
    """Sweep ``(gamma, n_components, alpha)`` by GCV on the full training set.

    Returns ``{"timeshifts": (gamma, n_components, alpha),
    "mode_phases": (gamma, n_components, alpha)}`` --- the best combination
    for the scalar time shift and for the (multi-output) phase leftover.
    """
    p, dt, phi = train
    x = MinMaxScaler().fit_transform(p)
    leftover, _ = phase_backbone_leftover(p, phi, modes, f0_natural)

    def best_for(y, name):
        best = None  # (score, gamma, n_components, alpha)
        for n_components in SWEEP_N_COMPONENTS:
            for gamma in SWEEP_GAMMAS:
                z = RBFSampler(
                    n_components=n_components, gamma=gamma, random_state=42
                ).fit_transform(x)
                rcv = RidgeCV(alphas=SWEEP_ALPHAS, scoring="neg_mean_squared_error")
                rcv.fit(z, y)
                score = float(rcv.best_score_)
                logging.info(
                    "  %-11s gamma=%-6g n=%-5d -> alpha=%.2e  cv negMSE=%.4e",
                    name, gamma, n_components, float(rcv.alpha_), score,
                )
                if best is None or score > best[0]:
                    best = (score, gamma, n_components, float(rcv.alpha_))
        logging.info(
            "  %-11s BEST gamma=%g n=%d alpha=%.3e", name, best[1], best[2], best[3]
        )
        return best[1:]

    logging.info("Ridge CV sweep on %d training points", len(p))
    return {
        "timeshifts": best_for(dt, "timeshifts"),
        "mode_phases": best_for(leftover, "mode_phases"),
    }


def _errors_for_fit(ts, mp, p, dt, phi):
    dt_err = np.abs(ts.predict(p) - dt)
    phase_err = np.abs(wrapped(mp.predict(p) - phi))
    return dt_err, phase_err


def evaluate(subsets, train, val, modes, f0_natural, hyper):
    """Fit both regressors on each nested subset, score on train and val.

    ``hyper`` is ``{"timeshifts": (gamma, n_components, alpha) | None,
    "mode_phases": ...}``; ``None`` keeps the class defaults.

    Returns ``dt_errors``, ``phase_errors`` as
    ``{subset_size: {"train": arr, "val": arr}}``; the phase arrays are
    ``(n, n_modes)``.
    """
    p_tr, dt_tr, phi_tr = train
    p_val, dt_val, phi_val = val
    mode_tuples = [(m.l, m.m) for m in modes]

    def kw(key):
        if hyper.get(key) is None:
            return {}
        gamma, n_components, alpha = hyper[key]
        return dict(gamma=gamma, n_components=n_components, ridge_alpha=alpha)

    dt_errors: dict[int, dict] = {}
    phase_errors: dict[int, dict] = {}

    for n in subsets:
        if n > len(p_tr):
            logging.warning("Skipping subset %d: only %d training points", n, len(p_tr))
            continue

        ts = TimeshiftsNN(
            training_params=p_tr[:n], training_timeshifts=dt_tr[:n], **kw("timeshifts")
        ).fit()
        mp = ModePhasesNN(
            modes=mode_tuples,
            f0_natural=f0_natural,
            training_params=p_tr[:n],
            training_mode_phases=phi_tr[:n],
            **kw("mode_phases"),
        ).fit()

        dt_tr_err, phase_tr_err = _errors_for_fit(ts, mp, p_tr[:n], dt_tr[:n], phi_tr[:n])
        dt_val_err, phase_val_err = _errors_for_fit(ts, mp, p_val, dt_val, phi_val)
        dt_errors[n] = {"train": dt_tr_err, "val": dt_val_err}
        phase_errors[n] = {"train": phase_tr_err, "val": phase_val_err}

        logging.info(
            "subset %5d | Δt median train %.2e / val %.2e s | "
            "phase median train %s / val %s rad",
            n,
            np.median(dt_tr_err), np.median(dt_val_err),
            np.array2string(np.median(phase_tr_err, axis=0), precision=2, separator=","),
            np.array2string(np.median(phase_val_err, axis=0), precision=2, separator=","),
        )

    return dt_errors, phase_errors


def print_table(subsets, dt_errors, phase_errors, modes):
    def stats(a):
        return np.median(a), np.percentile(a, 90)

    for kind in ("train", "val"):
        print(f"\n=== {kind} error: median (90th pct) ===")
        header = "  n_train | " + "Δt [s]".ljust(22) + " | " + " | ".join(
            f"({m.l},{m.m}) [rad]".ljust(22) for m in modes
        )
        print(header)
        print("-" * len(header))
        for n in subsets:
            if n not in dt_errors:
                continue
            m, p = stats(dt_errors[n][kind])
            row = f"  {n:6d} | {m:.3e} ({p:.3e})"
            for j, _ in enumerate(modes):
                m, p = stats(phase_errors[n][kind][:, j])
                row += f" | {m:.3e} ({p:.3e})"
            print(row)


def _paired_boxplot(ax, subsets, series_train, series_val):
    """Two boxes (train, val) per subset position, with a legend."""
    positions = np.arange(len(subsets))
    for data, offset, color, label in (
        (series_train, -0.18, "tab:gray", "train"),
        (series_val, +0.18, "tab:blue", "val"),
    ):
        bp = ax.boxplot(
            data,
            positions=positions + offset,
            widths=0.3,
            showfliers=True,
            patch_artist=True,
            flierprops=dict(marker=".", markersize=3, alpha=0.3),
        )
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.5)
        for median in bp["medians"]:
            median.set(color="black")
        bp["boxes"][0].set_label(label)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(subsets)
    ax.legend()


def plot_phases(subsets, phase_errors, modes, outfile):
    subsets = [n for n in subsets if n in phase_errors]
    ncols = 2
    nrows = int(np.ceil(len(modes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for j, mode in enumerate(modes):
        ax = axes[j // ncols][j % ncols]
        _paired_boxplot(
            ax, subsets,
            [phase_errors[n]["train"][:, j] for n in subsets],
            [phase_errors[n]["val"][:, j] for n in subsets],
        )
        ax.set_xlabel("training set size")
        ax.set_ylabel(r"$|\phi_{\ell m}^{\rm pred} - \phi_{\ell m}^{\rm true}|$ [rad]")
        ax.set_title(f"mode ({mode.l}, {mode.m})")
        ax.grid(True, which="both", axis="y", alpha=0.3)

    for k in range(len(modes), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle("ModePhasesNN error vs training set size (train vs held-out)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")


def plot_timeshifts(subsets, dt_errors, outfile):
    subsets = [n for n in subsets if n in dt_errors]
    fig, ax = plt.subplots(figsize=(8, 5))
    _paired_boxplot(
        ax, subsets,
        [dt_errors[n]["train"] for n in subsets],
        [dt_errors[n]["val"] for n in subsets],
    )
    ax.set_xlabel("training set size")
    ax.set_ylabel(r"$|\Delta t_{\rm pred} - \Delta t_{\rm true}|$ [s]")
    ax.set_title("TimeshiftsNN error vs training set size (train vs held-out)")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument(
        "--subsets", type=str, default="500,1000,2000,4000,8000",
        help="comma-separated nested training set sizes",
    )
    parser.add_argument("--grid-points", type=int, default=64)
    parser.add_argument("--fmax-hz", type=float, default=512.0)
    parser.add_argument("--initial-frequency-hz", type=float, default=20.0)
    parser.add_argument(
        "--batch-size", type=int, default=1000,
        help="EOB sweep batch size (memory ~ batch-size * grid-points)",
    )
    parser.add_argument("--seed-train", type=int, default=1)
    parser.add_argument("--seed-val", type=int, default=2)
    parser.add_argument("--gw-prior", action="store_true", help="use the GW analysis prior")
    parser.add_argument(
        "--sweep", action="store_true",
        help="run a Ridge CV sweep and use the best hyperparameters",
    )
    parser.add_argument("--out", type=str, default="regressor_training_curve")
    args = parser.parse_args()

    subsets = sorted(int(s) for s in args.subsets.split(","))
    if max(subsets) > args.train_size:
        parser.error("largest --subsets entry exceeds --train-size")

    model = build_model(args)
    modes = list(model.modes)
    grid_hz, f_ref_natural, f0_natural = model._reference_grid(
        args.grid_points, args.fmax_hz
    )

    train = make_targets(
        model, grid_hz, f_ref_natural, args.train_size,
        args.seed_train, args.batch_size, "train sweep",
    )
    val = make_targets(
        model, grid_hz, f_ref_natural, args.val_size,
        args.seed_val, args.batch_size, "val sweep",
    )
    logging.info("valid waveforms: %d train, %d val", len(train[0]), len(val[0]))

    hyper = {"timeshifts": None, "mode_phases": None}
    if args.sweep:
        hyper = ridge_cv_sweep(train, modes, f0_natural)
        print("\nBest hyperparameters (gamma, n_components, alpha):")
        for k, v in hyper.items():
            print(f"  {k:11s} {v}")

    dt_errors, phase_errors = evaluate(subsets, train, val, modes, f0_natural, hyper)
    print_table(subsets, dt_errors, phase_errors, modes)
    plot_phases(subsets, phase_errors, modes, f"{args.out}_phases.png")
    plot_timeshifts(subsets, dt_errors, f"{args.out}_timeshifts.png")


if __name__ == "__main__":
    main()
