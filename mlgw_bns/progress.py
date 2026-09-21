"""Progress reporting for the parallel dataset-generation stages.

``tqdm`` bars are great in a terminal but unreadable once stdout/stderr is
redirected to a file (a retrain launched from ``make_default_dataset.py``, a
systemd unit, ...): every refresh writes a carriage-return / cursor-up escape
sequence, so the log fills with thousands of half-drawn bar fragments and the
``logging.info`` lines that matter are buried.

:func:`joblib_progress` wraps a :class:`joblib.Parallel` call and picks the
right reporter for where the output is going:

* interactive terminal (``stderr`` is a TTY) --- the usual live ``tqdm`` bar,
  via ``tqdm_joblib``;
* anything else --- a single throttled ``logging.info`` line every
  ``log_interval`` seconds, plus one final line at completion. No carriage
  returns, no ANSI.
"""
from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
from collections.abc import Iterator

import joblib  # type: ignore
from tqdm import tqdm  # type: ignore
from tqdm_joblib import tqdm_joblib


def _stderr_is_interactive() -> bool:
    try:
        return bool(sys.stderr.isatty())
    except Exception:  # pragma: no cover - defensive, e.g. detached stderr
        return False


@contextlib.contextmanager
def _joblib_logging_progress(
    desc: str, total: int, log_interval: float
) -> Iterator[None]:
    """Patch joblib's batch callback to emit throttled ``logging.info`` lines.

    Mirrors what ``tqdm_joblib`` does to hook into ``joblib.Parallel``, but
    instead of updating a bar it counts completed items and logs a one-line
    progress summary at most once per ``log_interval`` seconds (and once more
    when the last batch lands).
    """

    state_lock = threading.Lock()
    done = 0
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

    logging.info("%s: starting on %i items", desc, total)

    old_callback = joblib.parallel.BatchCompletionCallBack

    class LoggingBatchCompletionCallBack(old_callback):  # type: ignore[valid-type,misc]
        def __call__(self, *args, **kwargs):
            nonlocal done, last_log
            with state_lock:
                done += self.batch_size
                now = time.monotonic()
                if now - last_log >= log_interval or done >= total:
                    emit(done, now)
                    last_log = now
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = LoggingBatchCompletionCallBack
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback


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
        with tqdm_joblib(tqdm(desc=desc, total=total)):
            yield
    else:
        with _joblib_logging_progress(desc, total, log_interval):
            yield
