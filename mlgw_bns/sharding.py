r"""Datasets on disk, in shards made by any number of processes.

The training sets of the surrogates grow to millions of waveforms (or of
binaries' precession angles), more than a process makes in a walltime or
holds in memory. A :class:`ShardedStore` keeps one in a directory::

    config.json            what the items are (a frozen dataclass with
                           n_binaries, shard_size and seed)
    shards/00000.npz       items 0 ... shard_size - 1, and so on
    shards/00000.json      when, where and how fast it was made
    locks/00000.lock       a shard being made: by whom, refreshed while it is

Each shard is a function of the configuration and of its index alone (its
items drawn from the seed sequence ``(seed, index)``), so shards can be made
in any order, by any process on any machine sharing the filesystem: a
process claims one by creating its lock file exclusively (taking over a lock
not refreshed for a while, whose owner died), and writes it atomically. A
shard lost is made again, identically. The first ``N`` items are a uniform
sample for every ``N``, so the training sets of a learning curve are nested.

Subclasses say what a shard holds (:meth:`ShardedStore._make`) and may keep
*derived* shards next to the base ones, under other ``source`` names (such as
the refined envelopes of
:class:`~mlgw_bns.precession_dataset.ShardedDataset`), which replace some
fields (:attr:`ShardedStore.overlay_fields`) of the base shards.

The processes making them run as batch jobs with a walltime: on ``SIGTERM``
or ``SIGUSR1`` (which SLURM sends before the walltime) they abandon the shard
they are making (:func:`stop_on_signals`) and exit with :data:`EXIT_REQUEUE`,
for the job to be run again.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import signal
import socket
import threading
import time
import uuid
from typing import Callable, Dict, Iterator, Optional, Sequence

import numpy as np


#: The exit status of a command stopped before it finished, to be run again.
EXIT_REQUEUE = 75


class Stopped(BaseException):
    """A signal asked the process to stop."""


def stop_on_signals() -> None:
    """Raise :class:`Stopped` on ``SIGTERM`` and ``SIGUSR1``: what is being
    made is abandoned (its lock released), what is made is kept."""

    def handler(signum, _):
        raise Stopped(f"signal {signum}")

    for number in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(number, handler)


def write_atomically(filename: str, write: Callable) -> None:
    """``write(file)`` to a temporary file next to ``filename``, synced, then
    renamed over it: a reader sees all of it or none."""
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    temporary = f"{filename}.tmp.{socket.gethostname()}.{os.getpid()}"
    try:
        with open(temporary, "wb") as file:
            write(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, filename)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def save_arrays(filename: str, **arrays) -> None:
    """``np.savez`` (uncompressed), atomically."""
    write_atomically(filename, lambda file: np.savez(file, **arrays))


def save_json(filename: str, value) -> None:
    write_atomically(filename, lambda file: file.write(json.dumps(value, indent=2).encode()))


def owner() -> str:
    """This process, and its SLURM job if any."""
    job = os.environ.get("SLURM_JOB_ID")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    return f"{socket.gethostname()}:{os.getpid()}" + (f" job {job}" if job else "") + (f"_{task}" if task else "")


def format_ranges(indices: Sequence[int]) -> str:
    """``[0, 1, 2, 5]`` as ``0-2,5``."""
    parts, start = [], None
    for i, index in enumerate(indices):
        if start is None:
            start = index
        if i + 1 == len(indices) or indices[i + 1] != index + 1:
            parts.append(str(start) if start == index else f"{start}-{index}")
            start = None
    return ",".join(parts)


class ShardLock:
    """An exclusive claim on a shard, a lock file refreshed by a thread while
    held; one not refreshed for ``stale_seconds`` was left by a process that
    died, and is taken over. Two processes taking over the same stale lock
    both make the shard, identically."""

    def __init__(self, filename: str, stale_seconds: float):
        self.filename = filename
        self.stale_seconds = stale_seconds
        self.owner = f"{owner()} {uuid.uuid4().hex[:8]}"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def acquire(self) -> bool:
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        try:
            descriptor = os.open(self.filename, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - os.stat(self.filename).st_mtime
                previous = open(self.filename).read().strip()
            except FileNotFoundError:
                return False
            if age < self.stale_seconds:
                return False
            logging.warning("taking over %s, not refreshed for %.0f s (%s)", self.filename, age, previous)
            write_atomically(self.filename, lambda file: file.write(self.owner.encode()))
        else:
            with os.fdopen(descriptor, "w") as file:
                file.write(self.owner)
        self._thread = threading.Thread(target=self._refresh, daemon=True)
        self._thread.start()
        return True

    def _refresh(self) -> None:
        while not self._stop.wait(self.stale_seconds / 4):
            try:
                os.utime(self.filename)
            except FileNotFoundError:
                return

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        try:
            if open(self.filename).read().strip() == self.owner:
                os.remove(self.filename)
        except FileNotFoundError:
            pass


class ShardedStore:
    """A dataset on disk, in shards; see the module docstring. Open one with
    ``Store(directory)``, make one with :meth:`create`.

    Subclasses set :attr:`config_class` (a frozen dataclass with
    ``n_binaries``, ``shard_size`` and ``seed``, ``to_json`` and
    ``from_json``) and implement :meth:`_make`; those with derived shards
    also :meth:`source_directory` and :attr:`overlay_fields`.
    """

    config_class: type
    #: Fields of the base shards which those of the other sources replace.
    overlay_fields: Sequence[str] = ()

    def __init__(self, directory: str):
        self.directory = directory
        with open(os.path.join(directory, "config.json")) as file:
            self.config = self.config_class.from_json(file.read())

    @classmethod
    def create(cls, directory: str, config):
        """Make the dataset ``config`` in ``directory``, or open it if it is
        there: the same, or with fewer items and a full last shard (then it
        is grown)."""
        filename = os.path.join(directory, "config.json")
        if os.path.exists(filename):
            existing = cls(directory).config
            if existing != config:
                if (
                    dataclasses.replace(existing, n_binaries=config.n_binaries) != config
                    or config.n_binaries < existing.n_binaries
                    or existing.n_binaries % existing.shard_size
                ):
                    raise ValueError(f"{directory} holds another dataset:\n{existing.to_json()}")
                logging.info("growing %s from %i to %i items", directory, existing.n_binaries, config.n_binaries)
                save_json(filename, json.loads(config.to_json()))
        else:
            save_json(filename, json.loads(config.to_json()))
        return cls(directory)

    # -- layout --------------------------------------------------------------

    @property
    def n_shards(self) -> int:
        return -(-self.config.n_binaries // self.config.shard_size)

    def shard_size(self, index: int) -> int:
        return min(self.config.shard_size, self.config.n_binaries - index * self.config.shard_size)

    def seed(self, index: int) -> list:
        """The seed (sequence) of the items of shard ``index``."""
        return [self.config.seed, index]

    def source_directory(self, source: str) -> str:
        """Where the shards of ``source`` are (``"base"``: the directory)."""
        if source != "base":
            raise ValueError(f"unknown source {source!r}")
        return self.directory

    def path(self, index: int, source: str = "base") -> str:
        """The shard ``index`` of ``source``."""
        return os.path.join(self.source_directory(source), "shards", f"{index:05d}.npz")

    def done(self, source: str = "base") -> list:
        return [i for i in range(self.n_shards) if os.path.exists(self.path(i, source))]

    # -- making shards -------------------------------------------------------

    def work_through(
        self, source: str, work: Callable[[int], None], max_seconds: Optional[float] = None,
        stale_seconds: float = 600.0, poll_seconds: float = 30.0,
    ) -> bool:
        """Claim and ``work(index)`` the shards of ``source`` not yet there,
        until there are none (waiting for those others are making), or
        ``max_seconds`` have passed: whether all are there."""
        start = time.time()
        locks = os.path.join(self.source_directory(source), "locks")
        while True:
            todo = [i for i in range(self.n_shards) if not os.path.exists(self.path(i, source))]
            if not todo:
                return True
            if max_seconds is not None and time.time() - start > max_seconds:
                logging.info("out of time, %i shards of %s left", len(todo), source)
                return False
            for index in todo:
                lock = ShardLock(os.path.join(locks, f"{index:05d}.lock"), stale_seconds)
                if not lock.acquire():
                    continue
                try:
                    if not os.path.exists(self.path(index, source)):
                        work(index)
                finally:
                    lock.release()
                break
            else:
                logging.info("the %i shards of %s left are being made by others: waiting", len(todo), source)
                time.sleep(poll_seconds)

    def generate(
        self, n_jobs: int = -1, max_seconds: Optional[float] = None, stale_seconds: float = 600.0, **kwargs,
    ) -> bool:
        """Make the shards not yet there (see :meth:`work_through`), each
        with ``n_jobs`` processes; whether all are there."""
        return self.work_through(
            "base", lambda index: self._timed_make(index, n_jobs, **kwargs), max_seconds, stale_seconds
        )

    def _timed_make(self, index: int, n_jobs: int, **kwargs) -> None:
        start = time.time()
        arrays, summary = self._make(index, n_jobs, **kwargs)
        valid = arrays["valid"]
        if not valid.any():
            raise RuntimeError(f"shard {index}: no item came out valid")
        if not valid.all():
            logging.warning("shard %i: %i items not valid: %s", index, np.sum(~valid), np.flatnonzero(~valid))
        save_arrays(self.path(index), **arrays)
        seconds = time.time() - start
        save_json(self.path(index)[: -len(".npz")] + ".json", {
            "binaries": len(valid), "seconds": seconds, "by": owner(), "finished": time.ctime(),
            "invalid": int(np.sum(~valid)), **summary,
        })
        logging.info("shard %i/%i: %i items in %.0f s %s", index, self.n_shards, len(valid), seconds,
                     " ".join(f"{k}={v}" for k, v in summary.items()))

    def _make(self, index: int, n_jobs: int, **kwargs) -> tuple:
        """The arrays of shard ``index`` (``valid`` among them: which items
        came out right) and a summary of it for its ``.json``."""
        raise NotImplementedError

    # -- reading -------------------------------------------------------------

    def chunks(
        self, n: Optional[int] = None, fields: Optional[Sequence[str]] = None, source: str = "base",
        select: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Iterator[Dict[str, np.ndarray]]:
        """The ``fields`` (all, by default; and ``index``, the items'
        positions in the dataset) of its first ``n`` valid items (all, by
        default), a shard at a time; of those whose indices pass ``select``
        (a mask) if given. The :attr:`overlay_fields` are those of
        ``source``."""
        remaining = np.inf if n is None else n
        for shard in range(self.n_shards):
            if remaining <= 0:
                break
            filename = self.path(shard)
            if not os.path.exists(filename):
                raise FileNotFoundError(f"{filename} is not there yet (status: {len(self.done())}/{self.n_shards})")
            with np.load(filename) as base:
                index = shard * self.config.shard_size + np.arange(self.shard_size(shard))
                keep = base["valid"] if select is None else base["valid"] & select(index)
                rows = np.flatnonzero(keep)
                rows = rows[: int(min(remaining, len(rows)))]
                chunk = {"index": index[rows]}
                names = [f for f in base.files if f != "valid"] if fields is None else fields
                overlay = [f for f in names if f in self.overlay_fields and source != "base"]
                for name in names:
                    if name not in overlay:
                        chunk[name] = base[name][rows]
            if overlay:
                with np.load(self.path(shard, source)) as data:
                    for name in overlay:
                        chunk[name] = data[name][rows]
            remaining -= len(rows)
            yield chunk
        if n is not None and remaining > 0:
            raise ValueError(f"{self.directory} has fewer than {n} such items")

    def load(self, n: Optional[int] = None, fields: Optional[Sequence[str]] = None, source: str = "base",
             select=None) -> Dict[str, np.ndarray]:
        """:meth:`chunks`, concatenated."""
        parts = list(self.chunks(n, fields, source, select))
        return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}

    # -- progress ------------------------------------------------------------

    def status(self, source: str = "base") -> str:
        """How far the shards of ``source`` are, as text."""
        done = self.done(source)
        locks = os.path.join(self.source_directory(source), "locks")
        claimed = []
        if os.path.isdir(locks):
            for name in sorted(os.listdir(locks)):
                if name.endswith(".lock"):
                    path = os.path.join(locks, name)
                    try:
                        claimed.append((name[:-5], time.time() - os.stat(path).st_mtime, open(path).read().strip()))
                    except FileNotFoundError:
                        pass
        lines = [f"{self.directory} [{source}]: {len(done)}/{self.n_shards} shards "
                 f"({min(len(done) * self.config.shard_size, self.config.n_binaries)}/{self.config.n_binaries})"]
        times = []
        for i in done:
            meta = self.path(i, source)[: -len(".npz")] + ".json"
            if os.path.exists(meta):
                with open(meta) as file:
                    times.append(json.load(file)["seconds"])
        if times:
            lines.append(f"  {np.mean(times):.0f} s a shard on average")
        for name, age, by in claimed:
            lines.append(f"  shard {name}: {by}, refreshed {age:.0f} s ago")
        missing = [i for i in range(self.n_shards) if i not in set(done)]
        if missing:
            lines.append(f"  missing: {format_ranges(missing)}")
        return "\n".join(lines)
