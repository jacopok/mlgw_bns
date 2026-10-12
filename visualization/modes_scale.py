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
    # each mode's perceptron of the best trial of its hyperparameter study, where it
    # does better than kernel ridge on the first 16384 (on the study's objective)
    python $M train data/train --n 262144 --pca-size 32768 --out runs/best/model \
        --from-study studies/mlp_n262144 --mlp-steps 200000 --checkpoint runs/best/checkpoint \
        --krr-reference 16384 --reference-set data/validation
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
import optuna

from mlgw_bns.data_management import Residuals
from mlgw_bns.higher_order_modes import Mode
from mlgw_bns.hyperparameter_optimization import mlp_command_line, study_mlp_config
from mlgw_bns.jax_mlp import TrainingInterrupted
from mlgw_bns.model import Model
from mlgw_bns.model_validation import stored_waveform_mismatches
from mlgw_bns.mode_model import NN_KINDS, ParametersWithExtrinsic
from mlgw_bns.modes_dataset import (
    HOM7_MODES,
    ModesDatasetConfig,
    ShardedModesDataset,
    mode_name,
    same_indices,
    stored_size,
)
from mlgw_bns.neural_network import Hyperparameters, JaxMLPNetwork, KernelRidgeNetwork
from mlgw_bns.sharding import EXIT_REQUEUE, Stopped, save_arrays, save_json, stop_on_signals, write_atomically

sys.path.insert(0, os.path.dirname(__file__))
from precession_regression_study import (  # noqa: E402
    add_mlp_arguments,
    add_optimize_arguments,
    in_parallel,
    mlp_config,
    mlp_trial,
    open_optimization,
    report_best,
    report_studies,
    spent,
)

#: Memory of an exact kernel ridge fit on N waveforms, in bytes: the kernel
#: and its eigenvectors (measured: 9.8 GB at 24576).
KRR_BYTES = lambda n: 2.1 * n**2 * 8  # noqa: E731

#: The frequencies, Hz, of the timing of :meth:`Model.predict`.
TIMING_FREQUENCIES = np.geomspace(5.0, 2048.0, 1024)


def seconds(hours):
    return None if hours is None else 3600.0 * hours


def parse_trial(text: str) -> tuple:
    name, _, number = text.partition("=")
    parse_mode(name)
    return name, int(number)


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
    sizes = None
    if args.from_study:
        if not args.mlp_steps:
            raise SystemExit("--from-study: give the --mlp-steps of the study")
        modes = [Mode(*m) for m in args.modes] if args.modes else dataset.modes
        choices = {mode: study_choice(args, dataset, mode) for mode in modes}
        # those of the other modes, made by other jobs, for the model once all are there
        for mode in dataset.modes:
            if mode not in choices and os.path.exists(choice_name(args.out, mode)):
                with open(choice_name(args.out, mode)) as file:
                    choices[mode] = json.load(file)
        nn_kind = {mode: NN_KINDS[choice["kind"]] for mode, choice in choices.items()}
        sizes = {mode: choice["krr"]["n"] for mode, choice in choices.items() if nn_kind[mode] is KernelRidgeNetwork}

        def hyperparameters(mode, n):
            if nn_kind[mode] is KernelRidgeNetwork:
                return Hyperparameters.default_kernel_ridge(n, mode=mode)
            config, _ = study_mlp_config(study_journal(args, mode), args.mlp_steps, n, choices[mode]["trial"])
            return Hyperparameters.default_jax_mlp(n, config, args.mlp_loss)
    elif args.mlp:
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
            pca_size=args.pca_size, modes=args.modes, sizes=sizes, checkpoint=args.checkpoint,
        )
    except (TrainingInterrupted, Stopped) as error:
        logging.info("stopped: %s", error)
        return EXIT_REQUEUE
    if model is None:
        print(f"{args.out}: the modes asked for are trained, not all of them yet")
        return 0
    save_json(args.out + ".json", {
        "dataset": os.path.abspath(args.dataset), "n": args.n, "pca_size": args.pca_size or args.n,
        "kind": {mode_name(m): k.__name__ for m, k in nn_kind.items()} if sizes is not None else nn_kind.__name__,
        "seconds": time.time() - start,
        "arguments": {k: v for k, v in vars(args).items() if k != "run"},
    })
    print(f"{args.out}: trained on {args.n} waveforms")
    if args.validate:
        out = args.validate_out or default_validation_name(args.out)
        return validate(argparse.Namespace(
            dataset=args.validate, model=args.out, n=args.validate_n, out=out, jobs=args.jobs,
        ))
    return 0


def choice_name(base: str, mode: Mode) -> str:
    return f"{base}_{mode_name(mode)}.choice.json"


def study_journal(args, mode: Mode) -> str:
    return os.path.join(args.from_study, f"{mode_name(mode)}.journal")


def study_choice(args, dataset: ShardedModesDataset, mode: Mode) -> dict:
    """The regressor of ``mode`` in a ``train --from-study``: the perceptron
    of the best trial of the mode's study (``--trial`` another), or, with
    ``--krr-reference K``, kernel ridge on the first K waveforms if it does
    as well on the study's objective (:func:`regression_error` on
    ``--reference-set``); kernel ridge chosen is saved as the mode's
    regressor. Made once, then read from ``{out}_l2_m2.choice.json``: the
    study may have a new best trial when a training is resumed."""
    path = choice_name(args.out, mode)
    if os.path.exists(path):
        with open(path) as file:
            return json.load(file)
    name = mode_name(mode)
    config, trial = study_mlp_config(study_journal(args, mode), args.mlp_steps, args.n, args.trial.get(name))
    choice = {
        "study": os.path.abspath(study_journal(args, mode)), "trial": trial.number, "value": trial.value,
        "command_line": mlp_command_line(config), "kind": JaxMLPNetwork.__name__,
    }
    if args.krr_reference:
        if not args.reference_set:
            raise SystemExit("--krr-reference: give the --reference-set (the study's validation set)")
        model = dataset.model(args.out, nn_kind=KernelRidgeNetwork)
        mode_model = model.mode_models[mode]
        mode_model.pca_data = dataset.principal_components(args.pca_size or args.n)[mode]
        start = time.time()
        parameters, reduced, weights = dataset.reduced(model, mode, args.krr_reference)
        nn = mode_model.fit_nn(mode_model.default_hyperparameters(args.krr_reference), parameters, reduced, weights)
        targets = validation_targets(model, mode, ShardedModesDataset(args.reference_set), args.reference_n)
        choice["krr"] = {"n": args.krr_reference, "value": regression_error(mode_model, nn, *targets)}
        logging.info("%s: kernel ridge on %i, %.3f; trial %i of the study, %.3f (%.0f s)", name,
                     args.krr_reference, choice["krr"]["value"], trial.number, trial.value, time.time() - start)
        if choice["krr"]["value"] <= trial.value:
            choice["kind"] = KernelRidgeNetwork.__name__
            write_atomically(mode_model.filename_nn, lambda file: nn.save(file))  # type: ignore[arg-type]
    save_json(path, choice)
    return choice


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


def reconstructed(mode_model, reduced: np.ndarray) -> Residuals:
    """The residuals of the principal components ``reduced`` of a mode."""
    return Residuals.from_combined_residuals(
        mode_model.pca_model.reconstruct_data(reduced, mode_model.pca_data),
        mode_model.downsampling_indices.numbers_of_points,
    )


def validation_targets(model: Model, mode: Mode, validation: ShardedModesDataset, n: Optional[int]) -> tuple:
    """The parameters of the first ``n`` waveforms of ``validation``, the
    residuals of ``mode`` of their principal components in the ``model``'s
    basis, and their power weights: what :func:`regression_error` compares
    with."""
    mode_model = model.mode_models[mode]
    parameters, reduced, weights = validation.reduced(model, mode, n)
    return parameters, reconstructed(mode_model, reduced), weights


def regression_error(mode_model, nn, parameters: np.ndarray, truth: Residuals, weights) -> float:
    """The objective of the hyperparameter studies: the log10 of the mean
    square difference (:meth:`HyperparameterOptimization.residuals_difference
    <mlgw_bns.hyperparameter_optimization.HyperparameterOptimization.residuals_difference>`,
    power-weighted for the modes trained so) of the residuals ``truth``
    (:func:`validation_targets`) from those of the principal components
    predicted by ``nn``."""
    from mlgw_bns.hyperparameter_optimization import HyperparameterOptimization

    predicted = nn.predict(parameters) / mode_model.pca_data.eigenvalues ** nn.hyper.pc_exponent
    return float(np.log10(HyperparameterOptimization.residuals_difference(
        truth, reconstructed(mode_model, predicted), weights
    )))


def optimize(args) -> int:
    """A worker of the hyperparameter study of the regressor of one mode
    (``<study>/l2_m2.journal``): the error of a trial is the mean square
    difference (:meth:`HyperparameterOptimization.residuals_difference
    <mlgw_bns.hyperparameter_optimization.HyperparameterOptimization.residuals_difference>`,
    power-weighted for the modes trained so) of the residuals of the
    validation set from its predicted principal components and from its
    true ones: the error of the regression alone, without the truncation of
    the principal components, which is the same in every trial."""
    from mlgw_bns.hyperparameter_optimization import StopRequested, run_worker

    if (status := in_parallel(args)) is not None:
        return status
    stop = StopRequested()
    mode = Mode(*args.mode)
    name = mode_name(mode)
    study = open_optimization(
        args, f"{name}.journal", f"{os.path.basename(os.path.abspath(args.study))}_{name}", "log10 regression error"
    )
    if spent(args, study):
        return 0
    dataset, validation = ShardedModesDataset(args.dataset), ShardedModesDataset(args.validation)
    if not same_indices(dataset.downsampling()[mode], validation.downsampling()[mode]):
        raise SystemExit(f"{args.validation} has other downsampling indices than {args.dataset}")
    nn_kind = JaxMLPNetwork if args.kind == "mlp" else KernelRidgeNetwork
    if args.kind == "krr" and args.n > args.max_kernel_ridge:
        raise SystemExit(f"kernel ridge on {args.n} waveforms: above --max-kernel-ridge {args.max_kernel_ridge}")
    model = dataset.model(nn_kind=nn_kind)
    mode_model = model.mode_models[mode]
    mode_model.pca_data = dataset.principal_components(args.pca_size or args.n)[mode]
    start = time.time()
    parameters, reduced, weights = dataset.reduced(model, mode, args.n)
    targets = validation_targets(model, mode, validation, args.validate_n)
    logging.info("%s: %i training and %i validation waveforms read in %.0f s", name, len(parameters),
                 len(targets[0]), time.time() - start)
    checkpoints = os.path.join(args.study, "checkpoints", name)
    os.makedirs(checkpoints, exist_ok=True)

    def objective(trial, stop) -> float:
        checkpoint = progress = None
        if args.kind == "mlp":
            config, checkpoint, progress = mlp_trial(trial, args, args.n, checkpoints, stop)
            hyper = Hyperparameters.default_jax_mlp(args.n, config, args.mlp_loss)
        else:
            hyper = Hyperparameters.from_trial_kernel_ridge(trial, n_train=args.n)
        try:
            nn = mode_model.fit_nn(hyper, parameters, reduced, weights, checkpoint=checkpoint, progress=progress)
        except optuna.TrialPruned:
            if checkpoint is not None and os.path.exists(checkpoint):
                os.remove(checkpoint)
            raise
        if checkpoint is not None and os.path.exists(checkpoint):
            os.remove(checkpoint)
        if args.kind == "mlp":
            trial.set_user_attr("weights", int(sum(w.size + b.size for w, b in nn.network.weights)))
        return regression_error(mode_model, nn, *targets)

    return 0 if run_worker(study, objective, args.hours, stop, stale_minutes=args.stale_minutes) else EXIT_REQUEUE


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
    command.add_argument("--from-study", help="each mode's perceptron of the best trial of its study in this "
                         "directory (l2_m2.journal and so on), instead of the --mlp options")
    command.add_argument("--trial", nargs="+", default=[], metavar="MODE=NUMBER", type=parse_trial,
                         help="other trials than the best, e.g. l2_m1=12")
    command.add_argument("--krr-reference", type=int, metavar="K",
                         help="with --from-study: kernel ridge on the first K waveforms for the modes "
                              "where it does as well as the perceptron, on the study's objective")
    command.add_argument("--reference-set", help="the validation set of the study, to compare on")
    command.add_argument("--reference-n", type=int, help="its first this many (default: all, as the study)")
    command.add_argument("--validate", help="then validate on this dataset")
    command.add_argument("--validate-n", type=int, help="its first this many waveforms (default: all)")
    command.add_argument("--validate-out", help="default: validate_<run>.npz next to the run's directory")
    command.add_argument("--jobs", type=int, default=1, help="processes of the validation")
    command.set_defaults(run=train)

    command = commands.add_parser("optimize", help="a worker of the hyperparameter study of a mode's regressor")
    command.add_argument("dataset", help="training set")
    command.add_argument("--validation", required=True, help="validation set")
    command.add_argument("--mode", type=parse_mode, required=True)
    command.add_argument("--pca-size", type=int, help="principal components of the first this many (default: N)")
    command.add_argument("--max-kernel-ridge", type=int, default=65536)
    add_optimize_arguments(command)
    command.set_defaults(run=optimize)

    command = commands.add_parser("studies", help="the best trials of hyperparameter studies")
    command.add_argument("study", nargs="+", help="directories")
    command.add_argument("--top", type=int, default=10)
    command.set_defaults(run=report_studies)

    command = commands.add_parser("best", help="the best trial of each study in a directory: mode, number, value")
    command.add_argument("study", help="directory")
    command.set_defaults(run=report_best)

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
    if getattr(args, "command", None) == "train":
        args.trial = dict(args.trial)
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
