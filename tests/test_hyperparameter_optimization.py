from datetime import timedelta

from mlgw_bns.hyperparameter_optimization import HyperparameterOptimization


def test_study_creation(generated_mode_model):
    ho = HyperparameterOptimization(generated_mode_model)

    assert ho.training_data_number == 100
    assert ho.study_filename == "test_mode_model_loo_study.pkl"

    fixed = HyperparameterOptimization(generated_mode_model, kernel_alpha_selection="fixed")
    assert fixed.study_filename == "test_mode_model_study.pkl"


def test_optimization_smoketest(generated_mode_model):
    ho = HyperparameterOptimization(generated_mode_model)

    ho.optimize(timeout_min=1 / 60)

    # Kernel ridge trials are solved in closed form, so many complete within
    # the timeout: assert trials actually ran, rather than that they took
    # any particular amount of time.
    assert len(ho.study.trials) > 0
    assert ho.total_training_time() > timedelta(0)


import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pytest  # noqa: E402

from mlgw_bns.data_management import Residuals  # noqa: E402
from mlgw_bns.hyperparameter_optimization import (  # noqa: E402
    StopRequested,
    check_pruner_settings,
    enqueue_best,
    make_pruner,
    mlp_command_line,
    mlp_config_from_trial,
    open_study,
    pruner_settings,
    pruning_progress,
    read_study,
    recover_stale_trials,
    run_in_parallel,
    run_worker,
    trial_hours,
)
from mlgw_bns.sharding import EXIT_REQUEUE, Stopped  # noqa: E402
import signal  # noqa: E402


@pytest.fixture(autouse=True)
def restore_signal_handlers():
    """:class:`StopRequested` handles SIGTERM and SIGUSR1 for the rest of the
    process, as a worker should; not for the tests after these."""
    handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGUSR1)}
    yield
    for number, handler in handlers.items():
        signal.signal(number, handler)


def test_residuals_difference_weights():
    rng = np.random.default_rng(0)
    a = Residuals(rng.normal(size=(4, 3)), rng.normal(size=(4, 5)))
    b = Residuals(rng.normal(size=(4, 3)), rng.normal(size=(4, 5)))
    plain = HyperparameterOptimization.residuals_difference(a, b)
    assert HyperparameterOptimization.residuals_difference(a, b, np.full(4, 7.0)) == pytest.approx(plain)
    only_first = HyperparameterOptimization.residuals_difference(a, b, np.array([1.0, 0, 0, 0]))
    first = Residuals(a.amplitude_residuals[:1], a.phase_residuals[:1])
    assert only_first == pytest.approx(
        4 * HyperparameterOptimization.residuals_difference(first, Residuals(b.amplitude_residuals[:1], b.phase_residuals[:1])) / 4
    )


def quadratic(trial, stop):
    """A toy objective, pruned through intermediate values like a perceptron."""
    x = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    value = (np.log10(x) + 3) ** 2
    for step in (1, 2, 3):
        stop.check()
        trial.report(value + 1 / step, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    time.sleep(0.01)
    return value


def test_workers_share_a_study_in_a_journal(tmp_path):
    journal = str(tmp_path / "study.journal")
    study = open_study(journal, "toy", pruner=make_pruner("median", 3))
    assert run_worker(study, quadratic, hours=0.5 / 3600)
    first = len(study.trials)
    # another worker (another process, as far as the journal goes) adds to it
    other = open_study(journal, "toy", fixed={"learning_rate": 1e-3})
    assert run_worker(other, quadratic, hours=1.0 / 3600)
    assert len(open_study(journal, "toy").trials) > first
    assert trial_hours(other) >= 1.0 / 3600
    assert any(t.params == {"learning_rate": 1e-3} for t in other.trials[first:])


def test_a_study_keeps_its_pruner_settings(tmp_path):
    journal = str(tmp_path / "study.journal")
    # a study run (by halving) before the settings were kept
    study = open_study(journal, "toy", pruner=make_pruner("halving", 3))
    assert run_worker(study, quadratic, hours=0.5 / 3600)
    assert "pruner" not in study.user_attrs
    with pytest.raises(ValueError, match="another study"):
        check_pruner_settings(study, pruner_settings("halving", 0.25, 2))
    check_pruner_settings(study, pruner_settings("halving"))
    fresh = open_study(str(tmp_path / "fresh.journal"), "fresh")
    check_pruner_settings(fresh, pruner_settings("halving", 0.25, 2))
    check_pruner_settings(fresh, pruner_settings("halving", 0.25, 2))
    with pytest.raises(ValueError):
        check_pruner_settings(fresh, pruner_settings("median"))


def test_no_pruning_at_the_end_of_training(tmp_path):
    study = open_study(str(tmp_path / "study.journal"), "toy", pruner=make_pruner("halving", 100, 0.25, 2))
    for loss in (1.0, 0.1):  # the second, better, is the one kept at every rung
        trial = study.ask()
        progress = pruning_progress(trial, StopRequested(), steps=100)
        for step in (25, 50, 99):
            progress(step, loss)
        study.tell(trial, loss)
    trial = study.ask()
    progress = pruning_progress(trial, StopRequested(), steps=100)
    with pytest.raises(optuna.TrialPruned):
        progress(25, 10.0)
    trial = study.ask()
    progress = pruning_progress(trial, StopRequested())
    with pytest.raises(optuna.TrialPruned):
        progress(100, 10.0)
    trial = study.ask()
    progress = pruning_progress(trial, StopRequested(), steps=100)
    progress(100, 10.0)  # its last 5%: not pruned


def test_a_stopped_trial_is_retried_and_a_stale_one_recovered(tmp_path):
    study = open_study(str(tmp_path / "study.journal"), "toy")

    def interrupted(trial, stop):
        stop.signal = 10  # as SIGUSR1 sets it, during the trial
        return quadratic(trial, stop)

    assert not run_worker(study, interrupted, hours=1.0)
    failed, retry = study.trials
    assert failed.state == optuna.trial.TrialState.FAIL
    assert retry.state == optuna.trial.TrialState.WAITING and retry.user_attrs["retry_of"] == failed.number

    # the retry is the next trial; its worker dies (no more heartbeat)
    dead = study.ask()
    assert dead.number == retry.number
    dead.set_user_attr("heartbeat", time.time() - 3600)
    assert recover_stale_trials(study, stale_seconds=600) == 1
    assert recover_stale_trials(study, stale_seconds=600) == 0
    again = study.trials[-1]
    assert again.state == optuna.trial.TrialState.WAITING and again.user_attrs["retry_of"] == dead.number
    # and it is run next, with the parameters of the trial first stopped
    assert run_worker(study, quadratic, hours=0.2 / 3600)
    assert study.trials[again.number].params == failed.params
    assert study.trials[again.number].state == optuna.trial.TrialState.COMPLETE


def test_enqueue_best_and_the_search_space(tmp_path):
    source = open_study(str(tmp_path / "small.journal"), "small")
    source.optimize(lambda t: (mlp_config_from_trial(t, 1000, 4096), t.number)[1], n_trials=6)
    target = open_study(str(tmp_path / "large.journal"), "large", metric="log10 error")
    assert read_study(str(tmp_path / "large.journal")).metric_names == ["log10 error"]
    assert enqueue_best(target, source, 3) == 3
    assert enqueue_best(target, source, 3) == 0  # already there
    best = sorted(source.trials, key=lambda t: t.value)[:3]
    trial = target.ask()
    assert trial.user_attrs["enqueued_from"] == f"small#{best[0].number}"
    assert trial.user_attrs["enqueued"] == best[0].params
    config = mlp_config_from_trial(trial, 1000, 4096)
    assert config.hidden == (best[0].params["width"],) * best[0].params["depth"]
    assert config.batch_size == 2 ** best[0].params["log2_batch"]
    # the seed follows the parameters: the same for the same, others otherwise
    assert config.seed == best[0].user_attrs["seed"]
    assert len({t.user_attrs["seed"] for t in source.trials}) == len(source.trials)
    assert "--mlp-hidden" in mlp_command_line(config)
    with pytest.raises(Stopped):
        stop = StopRequested()
        stop.signal = 15
        stop.check()


def test_parallel_workers_share_the_cpus_and_are_stopped_together(tmp_path):
    # each worker writes its CPUs and threads, then waits for SIGUSR1 and
    # exits as a stopped worker does
    script = (
        "import json, os, signal, sys, time; "
        "signal.signal(signal.SIGUSR1, lambda *_: sys.exit(%i)); "
        "json.dump([sorted(os.sched_getaffinity(0)), int(os.environ['OMP_NUM_THREADS'])], "
        "open(os.path.join(sys.argv[1], str(os.getpid())), 'w')); "
        "time.sleep(60)" % EXIT_REQUEUE
    )
    cpus = sorted(os.sched_getaffinity(0))
    workers = min(2, len(cpus))

    def stop_when_started():
        while len(os.listdir(tmp_path)) < workers:
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGUSR1)

    threading.Thread(target=stop_when_started, daemon=True).start()
    start = time.time()
    assert run_in_parallel([sys.executable, "-c", script, str(tmp_path)], workers) == EXIT_REQUEUE
    assert time.time() - start < 30
    shares = [json.load(open(tmp_path / name)) for name in os.listdir(tmp_path)]
    assert sorted(c for share, _ in shares for c in share) == cpus
    assert all(threads == len(share) for share, threads in shares)
    # their exit status: a failure first, then a stop
    assert run_in_parallel([sys.executable, "-c", "import sys; sys.exit(3)"], 2) == 3
    assert run_in_parallel([sys.executable, "-c", "pass"], 2) == 0


def test_a_copy_of_a_running_study_is_read_as_it_was(tmp_path):
    journal = str(tmp_path / "study.journal")
    study = open_study(journal, "toy")
    run_worker(study, quadratic, hours=0.2 / 3600)
    study.ask()  # running
    copy = tmp_path / "copy"
    copy.mkdir()
    text = open(journal, "rb").read()
    # copied in the middle of a line, with the lock of a worker writing
    (copy / "study.journal").write_bytes(text[: text.rindex(b"\n", 0, len(text) - 1) + 20])
    os.symlink("nowhere", copy / "study.journal.lock")
    read = read_study(str(copy / "study.journal"))
    assert read.study_name == "toy"
    assert 0 < len(read.trials) <= len(study.trials)
    assert read.best_value >= study.best_value
    assert (copy / "study.journal").stat().st_size < len(text)  # not written to
