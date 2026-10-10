r"""The co-precessing modes surrogate (:class:`~mlgw_bns.model.Model`, the
seven modes of ``hom7_big``) at scale: training sets of up to millions of
waveforms (:mod:`mlgw_bns.modes_dataset`), made and trained on in SLURM jobs
(``slurm/modes/submit.sh`` runs the whole pipeline), or here::

    M=visualization/modes_scale.py
    # a training set (2^18 waveforms in shards of 1024) and an independent validation set
    python $M init data/train --n 262144 --seed 1
    python $M init data/validation --n 16384 --seed 2
    # the downsampling of the training set, which the validation set shares
    python $M downsampling data/train --jobs 8
    python $M init data/validation --n 16384 --seed 2 --downsampling-from data/train
    # make their shards: any number of these at once, wherever the directories
    # are seen; each claims shards until none is left (--hours: stop claiming after)
    python $M generate data/validation data/train --jobs 8
    python $M status data/train data/validation
    # the principal components, shared by the trainings
    python $M pca data/train --n 32768
    # train on the first N waveforms (a mode at a time, or the modes of --modes) and validate
    python $M train data/train --n 65536 --pca-size 32768 --out runs/krr_65536/model
    python $M train data/train --n 262144 --pca-size 32768 --out runs/mlp_262144/model \
        --mlp --mlp-steps 200000 --checkpoint runs/mlp_262144/checkpoint
    python $M validate data/validation --model runs/krr_65536/model --out runs/validate_krr_65536.npz
    # how big and fast the models are (on this machine), and a table of all of it
    python $M timing runs/*/model
    python $M summary runs

Everything is resumable: ``generate`` skips the shards already made,
``train`` the modes already trained (a perceptron resumes from its
checkpoint), ``validate`` a validation already made. A command stopped
before it finished (``--hours``, or ``SIGTERM``/``SIGUSR1``, which SLURM sends
before the walltime) exits with ``EXIT_REQUEUE`` (:mod:`mlgw_bns.sharding`),
for the job to be requeued.

A waveform takes ~1 s of TEOBResumS on a core (seven modes, from 5 Hz:
~75 core-hours for 2^18) and ~36 kB on disk; kernel ridge needs ~2 N^2 8
bytes of memory (69 GB at 2^16) and N^3 time, a mode at a time.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import logging
import os
import re
import sys
import time

import numpy as np

from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.jax_mlp import TrainingInterrupted
from mlgw_bns.model import Model
from mlgw_bns.model_validation import stored_waveform_mismatches
from mlgw_bns.mode_model import ParametersWithExtrinsic
from mlgw_bns.modes_dataset import HOM7_MODES, ModesDatasetConfig, ShardedModesDataset, same_indices, stored_size
from mlgw_bns.neural_network import Hyperparameters, JaxMLPNetwork, KernelRidgeNetwork
from mlgw_bns.sharding import EXIT_REQUEUE, Stopped, save_arrays, save_json, stop_on_signals

sys.path.insert(0, os.path.dirname(__file__))
from precession_regression_study import add_mlp_arguments, mlp_config  # noqa: E402

#: Memory of an exact kernel ridge fit on N waveforms, in bytes: the kernel
#: and its eigenvectors (measured: 9.8 GB at 24576).
KRR_BYTES = lambda n: 2.1 * n**2 * 8  # noqa: E731

#: The frequencies, Hz, of the timing of :meth:`Model.predict`.
TIMING_FREQUENCIES = np.geomspace(5.0, 2048.0, 1024)


def seconds(hours):
    return None if hours is None else 3600.0 * hours


def parse_mode(text: str) -> tuple:
    match = re.fullmatch(r"l(\d)_m(\d)", text)
    if not match:
        raise argparse.ArgumentTypeError(f"{text!r}: modes are named like l2_m2")
    return int(match[1]), int(match[2])


def open_model(base: str) -> Model:
    """The model saved as ``base`` (``{base}_l2_m2_nn.pkl`` and so on), its
    modes found from its files."""
    modes = sorted(
        parse_mode(name) for path in glob.glob(f"{base}_l*_m*.yaml")
        for name in re.findall(r"_(l\d_m\d)\.yaml$", path)
    )
    if not modes:
        raise FileNotFoundError(f"no model {base}_l*_m*.yaml")
    model = Model(modes=[Mode(*m) for m in modes], filename=base)
    model.load()
    return model


def init(args) -> None:
    ranges = ModesDatasetConfig(n_binaries=args.n).parameter_ranges
    if args.lambda_max is not None:
        ranges = dataclasses.replace(ranges, lambda1_range=(5.0, args.lambda_max), lambda2_range=(5.0, args.lambda_max))
    config = ModesDatasetConfig(
        n_binaries=args.n, shard_size=args.shard_size, seed=args.seed, modes=tuple(args.modes),
        parameter_ranges=ranges,
    )
    dataset = ShardedModesDataset.create(args.dataset, config)
    if args.downsampling_from:
        dataset.copy_downsampling(ShardedModesDataset(args.downsampling_from))
    print(dataset.status())


def downsampling(args) -> None:
    ShardedModesDataset(args.dataset).train_downsampling(args.size, n_jobs=args.jobs)


def generate(args) -> int:
    stop_on_signals()
    start = time.time()
    complete = True
    for directory in args.dataset:
        budget = None if args.hours is None else seconds(args.hours) - (time.time() - start)
        complete = ShardedModesDataset(directory).generate(
            n_jobs=args.jobs, max_seconds=budget, stale_seconds=60 * args.stale_minutes,
        ) and complete
    return 0 if complete else EXIT_REQUEUE


def status(args) -> None:
    for directory in args.dataset:
        dataset = ShardedModesDataset(directory)
        print(dataset.status())
        if not os.path.exists(dataset.downsampling_path):
            print("  no downsampling yet")
        pca = sorted(glob.glob(os.path.join(directory, "pca", "*.npz")))
        if pca:
            print("  principal components: " + ", ".join(os.path.basename(p)[:-4] for p in pca))


def pca(args) -> None:
    stop_on_signals()
    ShardedModesDataset(args.dataset).principal_components(args.n)


def train(args) -> int:
    dataset = ShardedModesDataset(args.dataset)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.mlp:
        config = mlp_config(args)
        if args.mlp_steps:
            config = config.with_steps(args.mlp_steps, args.n)
        logging.info("perceptron: %s", config)
        nn_kind = JaxMLPNetwork
        hyperparameters = lambda mode, n: Hyperparameters.default_jax_mlp(n, config, args.mlp_loss)  # noqa: E731
    else:
        if args.n > args.max_kernel_ridge:
            raise SystemExit(
                f"kernel ridge on {args.n} waveforms needs ~{KRR_BYTES(args.n) / 1e9:.0f} GB: "
                f"above --max-kernel-ridge {args.max_kernel_ridge}"
            )
        logging.info("kernel ridge on %i waveforms: ~%.1f GB", args.n, KRR_BYTES(args.n) / 1e9)
        nn_kind, hyperparameters = KernelRidgeNetwork, None
    # (while the perceptron trains, it checkpoints on them instead)
    stop_on_signals()
    start = time.time()
    try:
        model = dataset.train(
            args.n, args.out, nn_kind=nn_kind, hyperparameters=hyperparameters,
            pca_size=args.pca_size, modes=args.modes, checkpoint=args.checkpoint,
        )
    except (TrainingInterrupted, Stopped) as error:
        logging.info("stopped: %s", error)
        return EXIT_REQUEUE
    if model is None:
        print(f"{args.out}: the modes asked for are trained, not all of them yet")
        return 0
    save_json(args.out + ".json", {
        "dataset": os.path.abspath(args.dataset), "n": args.n, "pca_size": args.pca_size or args.n,
        "kind": nn_kind.__name__, "seconds": time.time() - start,
        "arguments": {k: v for k, v in vars(args).items() if k != "run"},
    })
    print(f"{args.out}: trained on {args.n} waveforms")
    if args.validate:
        out = args.validate_out or default_validation_name(args.out)
        return validate(argparse.Namespace(
            dataset=args.validate, model=args.out, n=args.validate_n, out=out, jobs=args.jobs,
        ))
    return 0


def default_validation_name(base: str) -> str:
    """``runs/krr_65536/model`` validates into ``runs/validate_krr_65536.npz``."""
    run = os.path.dirname(os.path.abspath(base))
    return os.path.join(os.path.dirname(run), f"validate_{os.path.basename(run)}.npz")


def statistics(values: np.ndarray) -> dict:
    return {
        "median": float(np.median(values)), "90%": float(np.percentile(values, 90)),
        "99%": float(np.percentile(values, 99)), "max": float(np.max(values)),
    }


def validate(args) -> int:
    stop_on_signals()
    if os.path.exists(args.out):
        print(f"{args.out} is there")
        return 0
    dataset = ShardedModesDataset(args.dataset)
    model = open_model(args.model)
    indices = dataset.downsampling()
    for mode in model.modes:
        if not same_indices(model.mode_models[mode].downsampling_indices, indices[mode]):
            raise SystemExit(f"{args.model} has other downsampling indices than {args.dataset}")
    start = time.time()
    parameters, residuals = dataset.load_residuals(args.n)
    result = stored_waveform_mismatches(model, parameters, residuals, n_jobs=args.jobs)
    seconds_taken = time.time() - start
    save_arrays(args.out, mismatches=result["full"], parameters=parameters,
                **{key: value for key, value in result.items() if key != "full"})
    summary = {
        "model": os.path.abspath(args.model), "validation": os.path.abspath(args.dataset),
        "n": len(parameters), "seconds": seconds_taken,
        "full": statistics(result["full"]),
        **{key: statistics(value) for key, value in result.items() if key.startswith("mode ")},
        "size": stored_size(model),
        "timing": timing_of(model),
    }
    save_json(args.out[: -len(".npz")] + ".json", summary)
    print(f"{args.out}: {len(parameters)} waveforms in {seconds_taken:.0f} s; full waveform median "
          f"{summary['full']['median']:.2e}, 90% {summary['full']['90%']:.2e}, max {summary['full']['max']:.2e}")
    return 0


class Oracle:
    """A regressor which gives the true principal components of the
    waveforms it is asked about: validated with it, a model's mismatches are
    those of the truncation of the principal components alone, the floor of
    any regressor."""

    def __init__(self, parameters: np.ndarray, reduced: np.ndarray):
        self.hyper = argparse.Namespace(pc_exponent=0.0)
        self.table = {np.asarray(row, dtype=float).tobytes(): r for row, r in zip(parameters, reduced)}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.stack([self.table[np.asarray(row, dtype=float).tobytes()] for row in x])


def oracle(args) -> int:
    stop_on_signals()
    dataset = ShardedModesDataset(args.dataset)
    model = open_model(args.model)
    parameters, residuals = dataset.load_residuals(args.n)
    for mode in model.modes:
        mode_model = model.mode_models[mode]
        reduced = mode_model.pca_model.reduce_data(residuals[mode].combined, mode_model.pca_data)
        mode_model.nn = Oracle(parameters, reduced)
    result = stored_waveform_mismatches(model, parameters, residuals, n_jobs=args.jobs)
    save_arrays(args.out, mismatches=result["full"], parameters=parameters,
                **{key: value for key, value in result.items() if key != "full"})
    print(f"{args.out}: the truncation of the principal components of {args.model}: full waveform "
          + ", ".join(f"{k} {v:.2e}" for k, v in statistics(result["full"]).items()) + "; per mode median "
          + ", ".join(f"{k[5:]} {np.median(v):.1e}" for k, v in result.items() if k.startswith("mode ")))
    return 0


def timing_of(model: Model, repeats: int = 200) -> dict:
    """Median seconds of :meth:`Model.predict` on :data:`TIMING_FREQUENCIES`
    (one waveform, every mode), and of the regressors alone (one parameter
    row, every mode), on this machine."""
    rng = np.random.default_rng(1)
    generator = model.dataset.make_parameter_generator(seed=5)
    params = [next(generator) for _ in range(repeats)]
    full, regressors = [], []
    for p in params:
        extrinsic = ParametersWithExtrinsic(
            mass_ratio=p.mass_ratio, lambda_1=p.lambda_1, lambda_2=p.lambda_2, chi_1=p.chi_1, chi_2=p.chi_2,
            distance_mpc=100.0, inclination=float(rng.uniform(0, np.pi)), total_mass=2.8,
        )
        row = p.array[None, :]
        start = time.perf_counter()
        for mode in model.modes:
            model.mode_models[mode].nn.predict(row)
        regressors.append(time.perf_counter() - start)
        start = time.perf_counter()
        model.predict(TIMING_FREQUENCIES, extrinsic)
        full.append(time.perf_counter() - start)
    return {
        "predict": float(np.median(full[repeats // 10:])), "regressors": float(np.median(regressors[repeats // 10:])),
        "frequencies": len(TIMING_FREQUENCIES), "machine": os.uname().nodename,
    }


def timing(args) -> None:
    for base in args.model:
        model = open_model(base)
        result = {"size": stored_size(model), "timing": timing_of(model)}
        save_json(base + ".timing.json", result)
        print(f"{base}: {result['size']['total'] / 1e6:.1f} MB, predict {1e3 * result['timing']['predict']:.2f} ms "
              f"(regressors {1e3 * result['timing']['regressors']:.2f} ms)")


def summary(args) -> None:
    """A table of the validations (and timings) of a directory of runs."""
    rows = []
    for name in sorted(glob.glob(os.path.join(args.runs, "validate_*_*.json"))):
        match = re.fullmatch(r"validate_(.+)_(\d+)\.json", os.path.basename(name))
        if not match:
            continue
        with open(name) as file:
            result = json.load(file)
        local = os.path.join(args.runs, f"{match[1]}_{match[2]}", "model.timing.json")
        if os.path.exists(local):  # timed again here, comparable with the others
            with open(local) as file:
                result.update(json.load(file))
        rows.append((match[1], int(match[2]), result))
    rows.sort(key=lambda row: (row[0], row[1]))
    modes = sorted({k[len("mode "):] for _, _, r in rows for k in r if k.startswith("mode ")},
                   key=lambda m: parse_mode(m))
    header = ["series", "N", "full median", "full 90%", "full max"] + [f"{m} median" for m in modes] + [
        "size MB", "predict ms", "regressors ms"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for series, n, r in rows:
        cells = [series, str(n)] + [f"{r['full'][k]:.2e}" for k in ("median", "90%", "max")]
        cells += [f"{r['mode ' + m]['median']:.2e}" if "mode " + m in r else "" for m in modes]
        cells += [f"{r['size']['total'] / 1e6:.1f}", f"{1e3 * r['timing']['predict']:.2f}",
                  f"{1e3 * r['timing']['regressors']:.2f}"]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(args.runs, "summary.md"), "w") as file:
        file.write(table + "\n")
    if rows:
        print(plot_summary(rows, os.path.join(args.runs, "summary.png")))


def plot_summary(rows: list, out: str) -> str:
    """Accuracy, size and evaluation time of each series against the
    training-set size, side by side."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
    from precession_learning_curve import PALETTE, scientific, size_labels

    series = sorted({name for name, _, _ in rows})
    sizes = sorted({n for _, n, _ in rows})
    panels = [
        ("full-waveform mismatch\nmedian; dashed: 90%", lambda r: r["full"]["median"], lambda r: r["full"]["90%"]),
        ("size on disk, MB\n", lambda r: r["size"]["total"] / 1e6, None),
        ("evaluation time, ms\nModel.predict; dashed: regressors", lambda r: 1e3 * r["timing"]["predict"],
         lambda r: 1e3 * r["timing"]["regressors"]),
    ]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True)
    figure.patch.set_facecolor("#fcfcfb")
    for ax, (title, main, secondary) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        for name, (colour, marker) in zip(series, PALETTE):
            points = sorted((n, r) for s_, n, r in rows if s_ == name)
            n = [p[0] for p in points]
            ax.plot(n, [main(r) for _, r in points], color=colour, lw=2, marker=marker, ms=6, label=name,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5)
            if secondary is not None:
                ax.plot(n, [secondary(r) for _, r in points], color=colour, lw=1.2, ls="--")
        ax.set(xscale="log", yscale="log", title=title, xlabel="training waveforms")
        ax.set_xticks(sizes, size_labels(sizes))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(scientific))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(True, which="major", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(out, dpi=150, facecolor=figure.get_facecolor())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("init", help="make (or check, or grow) a dataset's configuration")
    command.add_argument("dataset")
    command.add_argument("--n", type=int, required=True)
    command.add_argument("--shard-size", type=int, default=ModesDatasetConfig.shard_size)
    command.add_argument("--seed", type=int, required=True, help="different for independent datasets")
    command.add_argument("--modes", type=parse_mode, nargs="+", default=list(HOM7_MODES))
    command.add_argument("--lambda-max", type=float, help="of both stars (default: that of hom7_big, 12000)")
    command.add_argument("--downsampling-from", help="share the downsampling of this (training) dataset")
    command.set_defaults(run=init)

    command = commands.add_parser("downsampling", help="train the downsampling indices of every mode")
    command.add_argument("dataset")
    command.add_argument("--size", type=int, default=64, help="waveforms of each mode")
    command.add_argument("--jobs", type=int, default=-1)
    command.set_defaults(run=downsampling)

    command = commands.add_parser("generate", help="make the shards not yet made")
    command.add_argument("dataset", nargs="+", help="in order")
    command.add_argument("--jobs", type=int, default=-1, help="worker processes")
    command.add_argument("--hours", type=float, help="stop claiming shards after this long")
    command.add_argument("--stale-minutes", type=float, default=10.0,
                         help="a lock not refreshed for this long is taken over")
    command.set_defaults(run=generate)

    command = commands.add_parser("status")
    command.add_argument("dataset", nargs="+")
    command.set_defaults(run=status)

    command = commands.add_parser("pca", help="fit the principal components of the first N waveforms")
    command.add_argument("dataset")
    command.add_argument("--n", type=int, required=True)
    command.set_defaults(run=pca)

    command = commands.add_parser("train", help="train on the first N waveforms (and validate)")
    command.add_argument("dataset")
    command.add_argument("--n", type=int, required=True)
    command.add_argument("--out", required=True, help="base filename of the model")
    command.add_argument("--pca-size", type=int, help="principal components of the first this many (default: N)")
    command.add_argument("--modes", type=parse_mode, nargs="+", help="only these (default: all)")
    command.add_argument("--max-kernel-ridge", type=int, default=65536,
                         help="refuse kernel ridge on more waveforms (memory)")
    add_mlp_arguments(command)
    command.add_argument("--mlp-steps", type=int, help="train for about this many steps (sets the epochs)")
    command.add_argument("--checkpoint", help="base filename of the perceptrons' checkpoints")
    command.add_argument("--validate", help="then validate on this dataset")
    command.add_argument("--validate-n", type=int, help="its first this many waveforms (default: all)")
    command.add_argument("--validate-out", help="default: validate_<run>.npz next to the run's directory")
    command.add_argument("--jobs", type=int, default=1, help="processes of the validation")
    command.set_defaults(run=train)

    command = commands.add_parser("validate", help="mismatches of a model on a validation set")
    command.add_argument("dataset")
    command.add_argument("--model", required=True, help="base filename")
    command.add_argument("--n", type=int, help="the first this many waveforms (default: all)")
    command.add_argument("--out", help="default: validate_<run>.npz next to the run's directory")
    command.add_argument("--jobs", type=int, default=1)
    command.set_defaults(run=validate)

    command = commands.add_parser("oracle", help="mismatches of the truncation of a model's principal components")
    command.add_argument("dataset", help="validation set")
    command.add_argument("--model", required=True, help="base filename (its principal components are used)")
    command.add_argument("--n", type=int)
    command.add_argument("--out", required=True)
    command.add_argument("--jobs", type=int, default=1)
    command.set_defaults(run=oracle)

    command = commands.add_parser("timing", help="size and evaluation time of models, on this machine")
    command.add_argument("model", nargs="+", help="base filenames")
    command.set_defaults(run=timing)

    command = commands.add_parser("summary", help="a table of the validations of a directory of runs")
    command.add_argument("runs")
    command.set_defaults(run=summary)

    args = parser.parse_args()
    if getattr(args, "command", None) == "validate" and args.out is None:
        args.out = default_validation_name(args.model)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return args.run(args) or 0
    except Stopped as error:
        logging.info("stopped: %s", error)
        return EXIT_REQUEUE


if __name__ == "__main__":
    sys.exit(main())
