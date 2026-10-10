r"""The regressed precession angles at scale: training sets of up to millions
of binaries (:mod:`mlgw_bns.precession_dataset`), made, refined and trained
on in SLURM jobs (``slurm/precession/submit.sh`` runs the whole pipeline),
or here::

    S=visualization/precession_scale.py
    # a training set (2^20 binaries in shards of 4096) and an independent validation set
    python $S init data/train --n 1048576 --seed 1 --mass-ratio 1 1.5 --smoothing 1e-2
    python $S init data/validation --n 16384 --seed 2 --mass-ratio 1 1.5 --smoothing 1e-2
    # make their shards: any number of these at once, wherever the directories are
    # seen; each claims shards until none is left (--hours: stop claiming after)
    python $S generate data/validation data/train --jobs 8
    python $S status data/train
    python $S pending data/validation data/train --refine-iterations 2   # what is left
    # refinement k of the envelopes: the priors of each fold, then the refits
    python $S refine-prior data/train --iteration 1 --fold 0   # ... --fold 3
    python $S refine-fit data/train --iteration 1
    # the waveforms of the validation set with the integrated angles, once
    python $S exact data/validation
    # train on the first N binaries (refined twice), resumably, and validate
    python $S train data/train --n 65536 --source refine2 --out runs/mlp_65536.joblib \
        --mlp --mlp-steps 200000 --checkpoint runs/mlp_65536.checkpoint --validate data/validation

Everything is resumable: ``generate`` and ``refine-fit`` skip the shards
already made, ``train`` a regressor already saved, and the perceptron resumes
from its checkpoint. A command stopped before it finished (``--hours``, or
``SIGTERM``/``SIGUSR1``, which SLURM sends before the walltime) exits with
``EXIT_REQUEUE`` (:mod:`mlgw_bns.sharding`), for the job to be requeued.

On a laptop core a binary takes ~25 ms to integrate and fit (~8 core-hours
for 2^20, 65 kB each on disk), a refit ~5 ms; kernel ridge needs ~3 N^2 8
bytes of memory (26 GB for 2^15) and N^3 time (~20 minutes for 2^14).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

from mlgw_bns.jax_mlp import TrainingInterrupted
from mlgw_bns.precession_dataset import DatasetConfig, ShardedDataset
from mlgw_bns.sharding import EXIT_REQUEUE, Stopped, stop_on_signals, write_atomically
from mlgw_bns.precession_regression import AngleGrid, TrainingRanges

sys.path.insert(0, os.path.dirname(__file__))
from precession_regression_study import (  # noqa: E402
    add_mlp_arguments,
    exact_waveforms,
    load,
    mlp_config,
    print_unexplained,
    validate,
)

def seconds(hours):
    return None if hours is None else 3600.0 * hours


def init(args) -> None:
    grid = AngleGrid(**({} if args.smoothing is None else {"smoothing": args.smoothing}))
    config = DatasetConfig(
        n_binaries=args.n, shard_size=args.shard_size, seed=args.seed, grid=grid,
        ranges=TrainingRanges(mass_ratio=tuple(args.mass_ratio)),
    )
    dataset = ShardedDataset.create(args.dataset, config)
    print(dataset.status())


def generate(args) -> int:
    stop_on_signals()
    start = time.time()
    complete = True
    for directory in args.dataset:
        budget = None if args.hours is None else seconds(args.hours) - (time.time() - start)
        complete = ShardedDataset(directory).generate(
            n_jobs=args.jobs, batch=args.batch, max_seconds=budget, stale_seconds=60 * args.stale_minutes,
        ) and complete
    return 0 if complete else EXIT_REQUEUE


def status(args) -> None:
    for directory in args.dataset:
        dataset = ShardedDataset(directory)
        print(dataset.status())
        refinements = os.path.join(directory, "refine")
        for iteration in sorted(os.listdir(refinements)) if os.path.isdir(refinements) else []:
            folds = dataset.refinement(int(iteration))["folds"]
            priors = sum(
                os.path.exists(os.path.join(refinements, iteration, f"fold{f}.joblib")) for f in range(folds)
            )
            print(f"  refinement {iteration}: {priors}/{folds} priors")
            print("  " + dataset.status(f"refine{iteration}").replace("\n", "\n  "))


def pending(args) -> None:
    """The stages of submit.sh not yet done, one a line: ``generate``,
    ``exact`` and ``refine<k>`` for k up to ``--refine-iterations``."""
    validation, train = ShardedDataset(args.validation), ShardedDataset(args.train)
    if any(len(d.done()) < d.n_shards for d in (validation, train)):
        print("generate")
    if not os.path.exists(os.path.join(args.validation, "exact.npz")):
        print("exact")
    for iteration in range(1, args.refine_iterations + 1):
        if len(train.done(f"refine{iteration}")) < train.n_shards:
            print(f"refine{iteration}")


def refine_prior(args) -> None:
    dataset = ShardedDataset(args.dataset)
    dataset.refinement(
        args.iteration, folds=args.folds, prior_size=args.prior_size,
        n_components=args.components, kernel_gamma=args.gamma,
    )
    folds = range(dataset.refinement(args.iteration)["folds"]) if args.fold is None else [args.fold]
    for fold in folds:
        print(dataset.train_prior(args.iteration, fold))


def refine_fit(args) -> int:
    stop_on_signals()
    complete = ShardedDataset(args.dataset).refine(
        args.iteration, n_jobs=args.jobs, max_seconds=seconds(args.hours), stale_seconds=60 * args.stale_minutes,
    )
    return 0 if complete else EXIT_REQUEUE


def train(args) -> int:
    dataset = ShardedDataset(args.dataset)
    if os.path.exists(args.out):
        print(f"{args.out} is there")
    else:
        kwargs = dict(n_components=tuple(args.components), kernel_gamma=args.gamma)
        if args.mlp:
            config = mlp_config(args)
            if args.mlp_steps:
                config = config.with_steps(args.mlp_steps, args.n)
            kwargs.update(mlp=config, mlp_loss=args.mlp_loss, mlp_checkpoint=args.checkpoint)
            logging.info("perceptron: %s", config)
        elif args.n > args.max_kernel_ridge:
            raise SystemExit(
                f"kernel ridge on {args.n} binaries needs ~{3 * args.n**2 * 8 / 1e9:.0f} GB: "
                f"above --max-kernel-ridge {args.max_kernel_ridge}"
            )
        else:
            logging.info("kernel ridge on %i binaries: ~%.1f GB", args.n, 3 * args.n**2 * 8 / 1e9)
        # (while the perceptron trains, it checkpoints on them instead)
        stop_on_signals()
        start = time.time()
        try:
            regressor = dataset.train(args.n, args.source, **kwargs)
        except (TrainingInterrupted, Stopped) as error:
            logging.info("stopped: %s", error)
            return EXIT_REQUEUE
        import joblib

        write_atomically(args.out, lambda file: joblib.dump(regressor, file))
        with open(args.out + ".json", "w") as file:
            json.dump({
                "dataset": os.path.abspath(args.dataset), "n": args.n, "source": args.source,
                "seconds": time.time() - start, "arguments": vars(args) | {"run": None},
            }, file, indent=2, default=str)
        print(f"trained on {args.n} binaries in {time.time() - start:.0f} s: {args.out}")
        print_unexplained(regressor)
    if args.validate:
        stop_on_signals()
        out = args.validate_out or os.path.join(
            os.path.dirname(args.out), "validate_" + os.path.basename(args.out).removesuffix(".joblib") + ".npz"
        )
        if os.path.exists(out):
            print(f"{out} is there")
        else:
            validate(argparse.Namespace(
                regressor=args.out, data=[args.validate], n=args.validate_n, out=out, oracle=None,
                refit_cache=None, exact_cache=os.path.join(args.validate, "exact.npz"),
            ))
    return 0


def exact(args) -> None:
    data = load([args.dataset])
    n = len(data["intrinsic"]) if args.n is None else args.n
    exact_waveforms(data, n, os.path.join(args.dataset, "exact.npz"))
    print(f"{os.path.join(args.dataset, 'exact.npz')}: {n} binaries")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("init", help="make (or check, or grow) a dataset's configuration")
    command.add_argument("dataset")
    command.add_argument("--n", type=int, required=True)
    command.add_argument("--shard-size", type=int, default=DatasetConfig.shard_size)
    command.add_argument("--seed", type=int, required=True, help="different for independent datasets")
    command.add_argument("--mass-ratio", type=float, nargs=2, default=list(TrainingRanges().mass_ratio))
    command.add_argument("--smoothing", type=float, help="see AngleGrid.smoothing")
    command.set_defaults(run=init)

    command = commands.add_parser("generate", help="make the shards not yet made")
    command.add_argument("dataset", nargs="+", help="in order")
    command.add_argument("--jobs", type=int, default=-1, help="worker processes")
    command.add_argument("--batch", type=int, default=32, help="binaries a worker integrates at once")
    command.add_argument("--hours", type=float, help="stop claiming shards after this long")
    command.add_argument("--stale-minutes", type=float, default=10.0,
                         help="a lock not refreshed for this long is taken over")
    command.set_defaults(run=generate)

    command = commands.add_parser("status")
    command.add_argument("dataset", nargs="+")
    command.set_defaults(run=status)

    command = commands.add_parser("pending", help="the stages of submit.sh not yet done")
    command.add_argument("validation")
    command.add_argument("train")
    command.add_argument("--refine-iterations", type=int, default=0)
    command.set_defaults(run=pending)

    command = commands.add_parser("refine-prior", help="train the regressor giving the priors of a fold")
    command.add_argument("dataset")
    command.add_argument("--iteration", type=int, required=True)
    command.add_argument("--fold", type=int, help="default: all, one after the other")
    command.add_argument("--folds", type=int, default=4)
    command.add_argument("--prior-size", type=int, default=16384, help="binaries of the other folds trained on")
    command.add_argument("--components", type=int, nargs=2, default=[64, 64])
    command.add_argument("--gamma", type=float, default=0.02)
    command.set_defaults(run=refine_prior)

    command = commands.add_parser("refine-fit", help="refit the envelopes with the priors")
    command.add_argument("dataset")
    command.add_argument("--iteration", type=int, required=True)
    command.add_argument("--jobs", type=int, default=-1)
    command.add_argument("--hours", type=float)
    command.add_argument("--stale-minutes", type=float, default=10.0)
    command.set_defaults(run=refine_fit)

    command = commands.add_parser("train", help="train on the first N binaries (and validate)")
    command.add_argument("dataset")
    command.add_argument("--n", type=int, required=True)
    command.add_argument("--source", default="base", help="the envelopes: base or refine<k>")
    command.add_argument("--out", required=True)
    command.add_argument("--components", type=int, nargs=2, default=[64, 64])
    command.add_argument("--gamma", type=float, default=0.02)
    command.add_argument("--max-kernel-ridge", type=int, default=32768,
                         help="refuse kernel ridge on more binaries (memory)")
    add_mlp_arguments(command)
    command.add_argument("--mlp-steps", type=int, help="train for about this many steps (sets the epochs)")
    command.add_argument("--checkpoint", help="where the perceptron keeps its state, to resume from")
    command.add_argument("--validate", help="then validate on this dataset")
    command.add_argument("--validate-n", type=int, help="its first this many binaries (default: all)")
    command.add_argument("--validate-out", help="default: validate_<regressor>.npz next to it")
    command.set_defaults(run=train)

    command = commands.add_parser("exact", help="cache the waveforms of a validation set with integrated angles")
    command.add_argument("dataset")
    command.add_argument("--n", type=int)
    command.set_defaults(run=exact)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return args.run(args) or 0
    except Stopped as error:
        logging.info("stopped: %s", error)
        return EXIT_REQUEUE


if __name__ == "__main__":
    sys.exit(main())
