r"""Command-line pieces shared by the training-at-scale scripts
(``modes_scale.py``): the options of a perceptron regressor and of a
hyperparameter study, and the styling of the learning-curve plots.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

from mlgw_bns.jax_mlp import MLPConfig

#: colours (the categorical palette, in its fixed order) and markers of the
#: series of a learning-curve plot
PALETTE = [
    ("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "D"), ("#eda100", "^"), ("#e87ba4", "v"),
    ("#7b5cd6", "P"), ("#3f8f9e", "X"), ("#a8632b", "h"), ("#5a5a5a", "*"), ("#b0b84a", "p"),
]


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
    command.add_argument("--mlp-warmup", type=float, default=default.warmup,
                         help="fraction of the steps of linear warmup of the learning rate")
    command.add_argument("--mlp-clip-norm", type=float, default=default.clip_norm,
                         help="clip the gradients to this global norm")
    command.add_argument("--mlp-loss", default="natural", choices=["natural", "uniform"])


def add_optimize_arguments(command) -> None:
    """The options of the ``optimize`` command of
    ``modes_scale.py``: a worker of a study shared by many (see
    :func:`mlgw_bns.hyperparameter_optimization.run_worker`)."""
    command.add_argument("--study", required=True, help="directory of the study (its journal, checkpoints)")
    command.add_argument("--n", type=int, required=True, help="training on the first N")
    command.add_argument("--kind", choices=["mlp", "krr"], default="mlp", help="perceptron or kernel ridge")
    command.add_argument("--hours", type=float,
                         help="the study's budget: hours of its trials, summed over all its workers "
                              "(default: until stopped, e.g. by SLURM at the walltime)")
    command.add_argument("--workers", type=int, default=1,
                         help="processes running trials at once, each on its share of the CPUs")
    command.add_argument("--mlp-steps", type=int, default=200000, help="steps of every perceptron (fixed)")
    command.add_argument("--mlp-loss", default="natural", choices=["natural", "uniform"])
    command.add_argument("--sampler", choices=["tpe", "gp"], default="tpe",
                         help="TPE, or a Gaussian process (completed trials only: best without pruning; needs torch)")
    command.add_argument("--pruner", choices=["none", "median", "halving"], default="halving",
                         help="of the perceptrons, on their held-out loss")
    command.add_argument("--prune-start", type=float,
                         help="fraction of the steps before the first pruning "
                              "(default: halving 0.1, median 0.2); a study keeps its pruner's settings")
    command.add_argument("--prune-reduction", type=int, default=3,
                         help="successive halving: 1 / this many trials kept at each rung")
    command.add_argument("--fix", nargs="+", default=[], metavar="NAME=VALUE",
                         help="hold these parameters of the search space (e.g. activation=gelu depth=4)")
    command.add_argument("--enqueue-from", help="another study's directory: try its best trials first")
    command.add_argument("--enqueue-top", type=int, default=8)
    command.add_argument("--validate-n", type=int, help="validation on its first this many (default: all)")
    command.add_argument("--stale-minutes", type=float, default=30.0,
                         help="a running trial without heartbeat for this long is retried")


def spent(args, study) -> bool:
    """Whether the budget (``--hours``) of the study of an ``optimize``
    command is spent (said so)."""
    from mlgw_bns.hyperparameter_optimization import trial_hours

    if args.hours is None or trial_hours(study) < args.hours:
        return False
    print(f"{study.study_name}: its {args.hours} hours are spent")
    return True


def in_parallel(args):
    """With ``--workers`` above 1, run that many workers of this
    ``optimize`` command (:func:`~mlgw_bns.hyperparameter_optimization.run_in_parallel`)
    and give their exit status; otherwise ``None``."""
    from mlgw_bns.hyperparameter_optimization import run_in_parallel

    if args.workers <= 1:
        return None
    return run_in_parallel([sys.executable, *sys.argv, "--workers", "1"], args.workers)


def parse_fixed(pairs) -> dict:
    """``["depth=4", "activation=gelu"]`` as ``{"depth": 4, "activation": "gelu"}``."""
    fixed = {}
    for pair in pairs:
        name, _, text = pair.partition("=")
        for kind in (int, float, str):
            try:
                fixed[name] = kind(text)
                break
            except ValueError:
                continue
    return fixed


def open_optimization(args, journal: str, name: str, metric: str):
    """The study of an ``optimize`` command (and, with ``--enqueue-from``,
    the best trials of the other one enqueued in it)."""
    from mlgw_bns.hyperparameter_optimization import (
        check_pruner_settings,
        enqueue_best,
        make_pruner,
        open_study,
        pruner_settings,
        read_study,
    )

    kind = args.pruner if args.kind == "mlp" else "none"
    pruner = make_pruner(kind, args.mlp_steps, args.prune_start, args.prune_reduction)
    study = open_study(
        os.path.join(args.study, journal), name, pruner=pruner, fixed=parse_fixed(args.fix), sampler=args.sampler,
        metric=metric,
    )
    check_pruner_settings(study, pruner_settings(kind, args.prune_start, args.prune_reduction))
    if args.enqueue_from:
        source = read_study(os.path.join(args.enqueue_from, journal))
        print(f"{enqueue_best(study, source, args.enqueue_top)} trials of {args.enqueue_from} enqueued")
    return study


def mlp_trial(trial, args, n: int, checkpoints: str, stop):
    """The perceptron configuration of an optimization trial, its
    checkpoint, and its pruning callback."""
    from mlgw_bns.hyperparameter_optimization import (
        checkpoint_name,
        mlp_command_line,
        mlp_config_from_trial,
        pruning_progress,
    )

    config = mlp_config_from_trial(trial, args.mlp_steps, n)
    trial.set_user_attr("command_line", mlp_command_line(config))
    return config, checkpoint_name(checkpoints, trial), pruning_progress(trial, stop, args.mlp_steps)


def report_studies(args) -> None:
    """The best trials of the studies (journals) in some directories, which
    may be running, or copies of ones running."""
    from mlgw_bns.hyperparameter_optimization import read_study, study_report

    for directory in args.study:
        for journal in sorted(glob.glob(os.path.join(directory, "*.journal"))):
            print(study_report(read_study(journal), top=args.top))


def report_best(args) -> None:
    """The best trial of each study (journal) in a directory, a line
    ``<journal> <trial> <value>`` each, for scripts (``submit.sh best``)."""
    from mlgw_bns.hyperparameter_optimization import read_study

    for journal in sorted(glob.glob(os.path.join(args.study, "*.journal"))):
        trial = read_study(journal).best_trial
        print(os.path.basename(journal).removesuffix(".journal"), trial.number, f"{trial.value:.6g}")


def mlp_config(args) -> MLPConfig:
    return MLPConfig(
        hidden=tuple(args.mlp_hidden), activation=args.mlp_activation,
        learning_rate=args.mlp_rate, final_learning_rate=args.mlp_final_rate, epochs=args.mlp_epochs,
        batch_size=args.mlp_batch, weight_decay=args.mlp_weight_decay, check_every=args.mlp_check_every,
        patience=args.mlp_patience, dtype=args.mlp_dtype, seed=args.mlp_seed,
        warmup=args.mlp_warmup, clip_norm=args.mlp_clip_norm,
    )


def scientific(value, _) -> str:
    """``2e-4`` as :math:`2 \times 10^{-4}`, powers of ten bare."""
    exponent = int(np.floor(np.log10(value) + 1e-9))
    mantissa = value / 10.0**exponent
    if abs(mantissa - 1) < 1e-6:
        return f"$10^{{{exponent}}}$"
    return f"${mantissa:.0f}\\times10^{{{exponent}}}$"


def size_labels(sizes) -> list:
    """The sizes, as powers of two where there are many of them."""
    if len(sizes) <= 6 or any(n & (n - 1) for n in sizes):
        return [str(n) for n in sizes]
    return [f"$2^{{{n.bit_length() - 1}}}$" for n in sizes]


