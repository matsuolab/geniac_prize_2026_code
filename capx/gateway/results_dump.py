"""Operator-side results dump for the gateway container; never exposed over ``EnvProtocol``.

Periodic dumps read scoring state without the lanes' locks, so a snapshot can be
momentarily stale relative to an in-flight ``step``. The final dump, which SIGTERM or
SIGINT starts, is the authoritative one: it first ends the job, so nothing
the submission does after the signal is scored, then waits for the calls in progress,
such as a step the end of the job cut short, to record their scores. It never waits
for a periodic dump, and no periodic dump is published after it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import threading
from datetime import datetime, timezone
from types import FrameType
from typing import Protocol

from capx.gateway.scoring import EpisodeScore, build_leaderboard
from capx.protocol.taskpack import Visibility

_logger = logging.getLogger(__name__)

FINAL_DUMP_WAIT_S = 20.0
"""How long the final dump waits for calls in progress: inside the eval queue's 30 s
between SIGTERM and SIGKILL."""


class Scored(Protocol):
    """What the dump reads from a :class:`~capx.gateway.Gateway`."""

    def aggregate(self) -> dict[str, dict[str, float]]: ...
    def scores(self) -> list[EpisodeScore]: ...
    def timings(self) -> dict[str, object]: ...
    def wait_idle(self, timeout_s: float) -> bool: ...
    def end_job(self) -> None: ...
    def finished_at(self) -> float | None: ...


def _snapshot(gateway: Scored, visibility: Visibility) -> dict[str, object]:
    return {
        # timezone.utc, not datetime.UTC: dev venvs still run Python 3.10.
        "updated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "leaderboard": build_leaderboard(gateway.aggregate(), visibility),
        "scores": [
            {"task_id": s.task_id, "reward": s.reward, "success": s.success}
            for s in gateway.scores()
        ],
        "timings": gateway.timings(),
        # The eval queue reads it to tell a finish() from a stop that came just after.
        "finished_at": gateway.finished_at(),
    }


def _write_snapshot(gateway: Scored, visibility: Visibility, path: str) -> str | None:
    """Write a snapshot next to ``path`` under a temporary name of the calling
    thread's own, so two threads never write the same file. Returns that name, or
    None if it failed (logged)."""
    tmp_path = f"{path}.{threading.get_ident()}.tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(_snapshot(gateway, visibility), f, indent=2)
        return tmp_path
    except Exception:
        _logger.exception("results dump to %r failed", path)
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        return None


def _publish(tmp_path: str, path: str) -> None:
    try:
        os.replace(tmp_path, path)
    except OSError:
        _logger.exception("results dump to %r failed", path)
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def dump_once(gateway: Scored, visibility: Visibility, path: str) -> None:
    """Write one results snapshot to ``path`` atomically.

    Never raises: a failure is logged and swallowed so it can never kill the dump
    thread or a signal handler.
    """
    tmp_path = _write_snapshot(gateway, visibility, path)
    if tmp_path is not None:
        _publish(tmp_path, path)


class ResultsDump:
    """Dumps ``gateway``'s results to ``path`` every ``interval_s`` seconds on a daemon
    thread (first dump immediate) until :meth:`final` writes the last one."""

    def __init__(
        self, gateway: Scored, visibility: Visibility, path: str, interval_s: float = 60.0
    ) -> None:
        self._gateway = gateway
        self._visibility = visibility
        self._path = path
        self._interval_s = interval_s
        self._stop = threading.Event()
        # Held only to rename a written snapshot into place, never while one is built.
        self._publishing = threading.Lock()
        self._finalized = False
        self._thread = threading.Thread(target=self._periodic, name="results-dump", daemon=True)

    def start(self) -> ResultsDump:
        self._thread.start()
        return self

    def final(self) -> None:
        """End the job, then write the authoritative dump once the calls in progress
        have recorded their scores. It does not wait for a periodic dump in progress,
        which is then dropped instead of published."""
        self._gateway.end_job()
        self._stop.set()
        if not self._gateway.wait_idle(FINAL_DUMP_WAIT_S):
            _logger.warning("final dump: calls still running after %.0fs", FINAL_DUMP_WAIT_S)
        self._dump(final=True)

    def _periodic(self) -> None:
        while not self._stop.is_set():
            self._dump(final=False)
            self._stop.wait(self._interval_s)

    def _dump(self, *, final: bool) -> None:
        tmp_path = _write_snapshot(self._gateway, self._visibility, self._path)
        if tmp_path is None:
            return
        with self._publishing:
            if self._finalized:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                return
            self._finalized = final
            _publish(tmp_path, self._path)


def install_results_dump(
    gateway: Scored,
    visibility: Visibility,
    path: str,
    interval_s: float = 60.0,
) -> ResultsDump:
    """Start a :class:`ResultsDump` and chain SIGTERM/SIGINT so the final dump happens
    before the previously installed handler runs.

    The handler only starts the final dump, on a thread of its own: ending the job and
    dumping take locks the interrupted main thread may hold (``Gateway.acquire()`` holds
    the referee's), and waiting for one there would wait forever. The thread sends the
    signal again once it is done, for the handler to pass on."""
    results = ResultsDump(gateway, visibility, path, interval_s).start()
    dumped = threading.Event()
    final: threading.Thread | None = None

    def _dump_then_resend(signum: int) -> None:
        try:
            results.final()
        finally:
            dumped.set()
            os.kill(os.getpid(), signum)

    def _chain(sig: int) -> None:
        previous = signal.getsignal(sig)

        def _handler(signum: int, frame: FrameType | None) -> None:
            nonlocal final
            if not dumped.is_set():
                if final is None:
                    final = threading.Thread(
                        target=_dump_then_resend, args=(signum,), name="final-dump"
                    )
                    final.start()
                return
            if callable(previous):
                previous(signum, frame)
            elif previous == signal.SIG_DFL or previous is None:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        signal.signal(sig, _handler)

    for sig in (signal.SIGTERM, signal.SIGINT):
        _chain(sig)

    return results


__all__ = ["ResultsDump", "Scored", "dump_once", "install_results_dump"]
