"""Tune :class:`~mlgw_bns.neural_network.KernelRidgeNetwork`'s
``kernel_gamma`` and ``kernel_alpha`` for every mode in
:data:`~mlgw_bns.model.DEFAULT_MODES`, at a fixed training-set size of
8192 waveforms, and save each mode's best trial as its default.

For each mode this loads (or generates, if missing) a dataset of
``N_WAVEFORMS`` waveforms, runs
:class:`~mlgw_bns.hyperparameter_optimization.HyperparameterOptimization`
for a share of the time budget, and writes the best
``(kernel_gamma, kernel_alpha)`` found so far into
``mlgw_bns/data/kernel_ridge_defaults.json`` via
:meth:`~mlgw_bns.hyperparameter_optimization.HyperparameterOptimization.save_best_as_default`.
Studies are checkpointed per mode (``optimization_dataset_<l><m>_study.pkl``),
so re-running with more hours resumes rather than restarting.

Run with: python optimize_n_hours.py <hours> [-g N]
"""

import argparse
import logging

from mlgw_bns.hyperparameter_optimization import HyperparameterOptimization
from mlgw_bns.mode_model import ModeModel
from mlgw_bns.model import DEFAULT_MODES
from mlgw_bns.neural_network import KernelRidgeNetwork

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)

#: Training waveforms per mode, matching HyperparameterOptimization.n_train_fixed.
N_TRAIN = 2**13

#: A dataset generated with N_TRAIN waveforms leaves nothing for the
#: hyperparameter-optimization validation split (hyper_validation_fraction,
#: 1% by default), so a small margin is generated on top of it.
N_WAVEFORMS = int(N_TRAIN / 0.98)


def dataset_filename(mode) -> str:
    return f"optimization_dataset_{mode.l}{mode.m}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hours", metavar="h", type=float, help="hours per mode")
    parser.add_argument(
        "-g",
        "--generate",
        metavar="gen",
        default=False,
        nargs=1,
        type=int,
        help="regenerate the training dataset with this many waveforms",
    )
    args = parser.parse_args()

    for mode in DEFAULT_MODES:
        filename = dataset_filename(mode)
        m = ModeModel(filename, mode=mode, nn_kind=KernelRidgeNetwork)
        try:
            m.load()
        except FileNotFoundError:
            m.generate(512, 1 << 13, N_WAVEFORMS)
            m.save()

        if args.generate:
            m.generate(None, None, args.generate[0])
            m.save()

        ho = HyperparameterOptimization(m, n_train_fixed=N_TRAIN)

        n_hours_before = ho.total_training_time().total_seconds() / 3600
        logging.info(
            "Mode %s%s: optimized for %.2f hours so far",
            mode.l,
            mode.m,
            n_hours_before,
        )

        ho.optimize_and_save(args.hours)

        n_hours_after = ho.total_training_time().total_seconds() / 3600
        logging.info(
            "Mode %s%s: optimized for %.2f more hours",
            mode.l,
            mode.m,
            n_hours_after - n_hours_before,
        )

        ho.save_best_as_default()
