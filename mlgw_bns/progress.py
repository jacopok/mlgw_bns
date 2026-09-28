"""Progress reporting for the parallel dataset-generation stages.

``tqdm`` bars are great in a terminal but unreadable once stdout/stderr is
redirected to a file (a retrain launched from ``make_default_dataset.py``, a
systemd unit, ...): every refresh writes a carriage-return / cursor-up escape
sequence, so the log fills with thousands of half-drawn bar fragments and the
``logging.info`` lines that matter are buried.

:func:`joblib_progress` wraps a :class:`joblib.Parallel` call and picks the
right reporter for where the output is going:

* interactive terminal (``stderr`` is a TTY) --- the usual live ``tqdm`` bar;
* anything else --- a single throttled ``logging.info`` line every
  ``log_interval`` seconds, plus one final line at completion. No carriage
  returns, no ANSI.

Both hook into :meth:`joblib.Parallel.print_progress`, which joblib calls
after every completed task on the sequential (``n_jobs=1``) path and after
every completed batch on the parallel ones. ``BatchCompletionCallBack`` ---
what ``tqdm_joblib`` patches --- is never invoked on the sequential path, so
a bar hooked there stays at 0% whenever ``n_jobs=1``.
"""
from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
from collections.abc import Callable, Iterator

import joblib  # type: ignore
from tqdm import tqdm  # type: ignore


def _stderr_is_interactive() -> bool:
    try:
        return bool(sys.stderr.isatty())
    except Exception:  # pragma: no cover - defensive, e.g. detached stderr
        return False


@contextlib.contextmanager
def _on_joblib_progress(callback: Callable[[int], None]) -> Iterator[None]:
    """Call ``callback(n_completed_tasks)`` whenever a ``joblib.Parallel``
    call in this ``with`` block reports progress."""

    old_print_progress = joblib.Parallel.print_progress

    def print_progress(self):
        callback(self.n_completed_tasks)
        return old_print_progress(self)

    joblib.Parallel.print_progress = print_progress
    try:
        yield
    finally:
        joblib.Parallel.print_progress = old_print_progress


@contextlib.contextmanager
def _joblib_tqdm_progress(desc: str, total: int) -> Iterator[None]:
    """Drive a ``tqdm`` bar from joblib's completed-task count."""

    # `leave=None`: keep the finished bar only when it is top-level, so bars
    # nested under a caller's own outer loop clear themselves.
    with tqdm(desc=desc, total=total, leave=None) as bar:

        def update(count: int) -> None:
            bar.update(count - bar.n)

        with _on_joblib_progress(update):
            yield


@contextlib.contextmanager
def _joblib_logging_progress(
    desc: str, total: int, log_interval: float
) -> Iterator[None]:
    """Emit throttled ``logging.info`` lines from joblib's completed-task count.

    Logs a one-line progress summary at most once per ``log_interval``
    seconds, and once more when the last task lands.
    """

    state_lock = threading.Lock()
    last_count = 0
    start = time.monotonic()
    last_log = start

    def emit(count: int, now: float) -> None:
        elapsed = now - start
        rate = count / elapsed if elapsed > 0 else 0.0
        percent = 100 * count / total if total else 100.0
        if rate > 0 and count < total:
            eta = f"~{(total - count) / rate:.0f}s remaining"
        else:
            eta = "done" if count >= total else "estimating..."
        logging.info(
            "%s: %i/%i (%.0f%%), %.0fs elapsed, %s, %.1f it/s",
            desc,
            count,
            total,
            percent,
            elapsed,
            eta,
            rate,
        )

    def update(count: int) -> None:
        nonlocal last_count, last_log
        with state_lock:
            # joblib also reports once more when the call finishes, with an
            # unchanged count; don't log that twice.
            if count == last_count:
                return
            last_count = count
            now = time.monotonic()
            if now - last_log >= log_interval or count >= total:
                emit(count, now)
                last_log = now

    logging.info("%s: starting on %i items", desc, total)

    with _on_joblib_progress(update):
        yield


@contextlib.contextmanager
def joblib_progress(
    desc: str, total: int, log_interval: float = 30.0
) -> Iterator[None]:
    """Report progress of the ``joblib.Parallel`` call in this ``with`` block.

    Parameters
    ----------
    desc : str
        Label for the stage, used in the bar / log lines.
    total : int
        Number of items handed to ``joblib.Parallel``.
    log_interval : float
        Minimum seconds between ``logging.info`` progress lines, used only on
        the non-interactive (log-file) path. Defaults to 30.
    """

    if _stderr_is_interactive():
        with _joblib_tqdm_progress(desc, total):
            yield
    else:
        with _joblib_logging_progress(desc, total, log_interval):
            yield
