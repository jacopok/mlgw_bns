r"""An optuna dashboard of the hyperparameter studies running on the cluster
(the ``optimize`` stages of ``slurm/modes`` and ``slurm/precession``), kept
up to date: every ``--every`` minutes their journals are copied here with
rsync, merged into one local journal (all the studies, each named after its
pipeline) and the dashboard restarted on it if anything changed. Reload the
page to see the new trials. ::

    python visualization/studies_dashboard.py --modes cluster:/scratch/me/modes_run \
        --precession cluster:/scratch/me/precession_run
    # then open http://localhost:8080

``--modes`` and ``--precession`` are the ``DATA`` of the pipelines on the
cluster (``host:path``, as rsync takes them); their ``optimize/`` is copied
to ``visualization/{modes,precession}_scale/run/optimize/`` (without the
checkpoints and locks: see docs/usage_guides/cluster.md), where
``modes_scale.py studies`` and ``precession_scale.py studies`` read it too.
``--include`` adds local directories of studies. The journals can be copied
while the studies run (:func:`~mlgw_bns.hyperparameter_optimization.read_study`).

Each study gets a note (where it is from, when copied, its best trial and
the ``train`` options of it) and, for the perceptrons, a plot of the value
of the completed trials against their number of weights. Notes written in
the dashboard are lost at the next merge: write them in the studies
themselves.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import logging
import os
import signal
import subprocess
import sys
import time

import optuna

from mlgw_bns.hyperparameter_optimization import _value, read_study

HERE = os.path.dirname(os.path.abspath(__file__))
MIRRORS = {
    "modes": os.path.join(HERE, "modes_scale", "run", "optimize"),
    "precession": os.path.join(HERE, "precession_scale", "run", "optimize"),
}


#: The dashboard, in a process of its own (restarted on a new journal).
SERVE = (
    "import sys; from optuna.storages import JournalStorage; "
    "from optuna.storages.journal import JournalFileBackend; from optuna_dashboard import run_server; "
    "run_server(JournalStorage(JournalFileBackend(sys.argv[1])), host=sys.argv[2], port=int(sys.argv[3]))"
)


def sync(remote: str, local: str) -> bool:
    """Copy ``remote/optimize/`` to ``local``; whether it worked."""
    os.makedirs(local, exist_ok=True)
    command = ["rsync", "-a", "--exclude", "checkpoints/", "--exclude", "*.lock",
               f"{remote.rstrip('/')}/optimize/", f"{local}/"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logging.warning("%s: %s", " ".join(command), result.stderr.strip())
    return result.returncode == 0


def journals(sources: dict) -> list:
    """(label, journal) of every study in the directories of ``sources``
    (label -> directory of study directories)."""
    found = []
    for label, directory in sources.items():
        for journal in sorted(glob.glob(os.path.join(directory, "*", "*.journal"))):
            found.append((label, journal))
    return found


def state(found: list) -> list:
    """What changes when a journal does."""
    return [(journal, os.stat(journal).st_size, os.stat(journal).st_mtime) for _, journal in found]


def note(study: optuna.Study, label: str, journal: str) -> str:
    """The note of a study in the dashboard: where it is from, and its best
    trial."""
    lines = [f"**{label}**: `{journal}`, copied {datetime.datetime.now():%Y-%m-%d %H:%M}.", ""]
    done = [t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE]
    if done:
        best = min(done, key=_value)
        lines.append(f"Best: trial {best.number}, {best.value:.4g}.")
        if "command_line" in best.user_attrs:
            lines += ["", "Its `train` options:", "", f"```\n{best.user_attrs['command_line']}\n```"]
    return "\n".join(lines)


def size_plot(study: optuna.Study):
    """The value of the completed trials against their number of weights
    (perceptrons only), or ``None``."""
    import plotly.graph_objects as go

    done = [
        t for t in study.get_trials(deepcopy=False)
        if t.state == optuna.trial.TrialState.COMPLETE and "weights" in t.user_attrs
    ]
    if not done:
        return None
    figure = go.Figure(go.Scatter(
        x=[t.user_attrs["weights"] for t in done], y=[t.value for t in done], mode="markers",
        text=[f"trial {t.number}: {t.params}" for t in done],
        marker=dict(color=[t.number for t in done], colorscale="Viridis", showscale=True,
                    colorbar=dict(title="trial")),
    ))
    metric = (study.metric_names or ["value"])[0]
    figure.update_layout(title="Accuracy against size", xaxis=dict(title="weights", type="log"),
                         yaxis=dict(title=metric))
    return figure


def merge(found: list, target: str) -> None:
    """All the studies of ``found`` in a new journal ``target``, named
    ``<label>: <study>``, each with its note and size plot."""
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    from optuna_dashboard import save_note, save_plotly_graph_object

    new = target + ".new"
    if os.path.exists(new):
        os.remove(new)
    storage = JournalStorage(JournalFileBackend(new))
    for label, journal in found:
        source = read_study(journal)
        name = f"{label}: {source.study_name}"
        optuna.copy_study(from_study_name=source.study_name, from_storage=source._storage,
                          to_storage=storage, to_study_name=name)
        study = optuna.load_study(study_name=name, storage=storage)
        save_note(study, note(source, label, os.path.relpath(journal)))
        figure = size_plot(source)
        if figure is not None:
            save_plotly_graph_object(study, figure)
    os.replace(new, target)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", help="host:DATA of the modes pipeline on the cluster")
    parser.add_argument("--precession", help="host:DATA of the precession pipeline on the cluster")
    parser.add_argument("--include", nargs="+", default=[], help="local directories of studies, also shown")
    parser.add_argument("--every", type=float, default=10.0, help="minutes between copies")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--journal", default=os.path.join(HERE, "studies_dashboard", "studies.journal"),
                        help="the merged journal the dashboard shows")
    parser.add_argument("--once", action="store_true", help="copy and merge once, without the dashboard")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    remotes = {label: remote for label, remote in (("modes", args.modes), ("precession", args.precession)) if remote}
    sources = {label: MIRRORS[label] for label in remotes}
    sources.update({os.path.normpath(d): d for d in args.include})
    if not sources:
        parser.error("nothing to show: give --modes, --precession or --include")
    os.makedirs(os.path.dirname(os.path.abspath(args.journal)), exist_ok=True)

    # stopped (Ctrl-C, or a kill), the dashboard stops too
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    log = os.path.join(os.path.dirname(os.path.abspath(args.journal)), "dashboard.log")
    dashboard = None
    last = None
    try:
        while True:
            for label, remote in remotes.items():
                sync(remote, sources[label])
            found = journals(sources)
            if state(found) != last:
                merge(found, args.journal)
                last = state(found)
                logging.info("%i studies merged into %s", len(found), args.journal)
                if args.once:
                    return 0
                if dashboard is not None:
                    dashboard.terminate()
                    dashboard.wait()
                with open(log, "a") as output:  # its requests
                    dashboard = subprocess.Popen(
                        [sys.executable, "-c", SERVE, args.journal, args.host, str(args.port)],
                        stdout=output, stderr=subprocess.STDOUT,
                    )
                logging.info("dashboard: http://%s:%i (its log: %s)", args.host, args.port, log)
            elif args.once:
                return 0
            time.sleep(60 * args.every)
    except KeyboardInterrupt:
        return 0
    finally:
        if dashboard is not None:
            dashboard.terminate()
            dashboard.wait()


if __name__ == "__main__":
    sys.exit(main())
