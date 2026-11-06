"""Where an episode's wall clock goes, turn by turn.

``capx.envs.trial`` only ever reported one number per episode, which is enough to
size a round and useless for deciding what to make faster. This records, for every
turn of the multi-turn loop, how long each phase took and how many model calls it
cost -- written to ``<output_dir>/turn_timings/trial_XX.json`` after every turn, so
an episode cut short still leaves the turns it finished.

The timer is thread-local because a lane is a thread (``runner.run_submission``
drives ``concurrency`` trials from one ``ThreadPoolExecutor``), so the module-level
helpers act on whichever trial is running on the calling thread and are no-ops
where none is. The one place that needs care is the parallel ensemble: its nine
candidates run on their own pool threads, so ``capx.llm.client`` binds the lane's
timer into them explicitly.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_active = threading.local()


class TurnTimer:
    """Accumulates phase durations and model calls for one trial."""

    def __init__(self, output_dir: str, trial: int, task: str | None, config: dict[str, Any]) -> None:
        self._dir = os.path.join(output_dir, "turn_timings")
        self._path = os.path.join(self._dir, f"trial_{trial:02d}.json")
        self._lock = threading.Lock()
        self._record: dict[str, Any] = {
            "trial": trial,
            "task": task,
            "config": config,
            "started_at": time.time(),
            "turns": [],
        }
        self._turn_index = 0
        self._turn_start = time.perf_counter()
        self._phases: dict[str, float] = {}
        self._calls: list[dict[str, Any]] = []
        self._spans: dict[str, dict[str, float]] = {}
        self._phase_name: str | None = None

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        previous = self._phase_name
        self._phase_name = name
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with self._lock:
                self._phases[name] = self._phases.get(name, 0.0) + elapsed
            self._phase_name = previous

    def record_call(
        self,
        *,
        model: str,
        elapsed_s: float,
        retries: int,
        usage: dict[str, Any] | None,
    ) -> None:
        """Record one completed model call under the phase that issued it."""
        usage = usage or {}
        with self._lock:
            self._calls.append({
                "phase": self._phase_name,
                "model": model,
                "elapsed_s": round(elapsed_s, 3),
                "retries": retries,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            })

    def record_span(self, kind: str, elapsed_s: float) -> None:
        """Record time inside a phase: a gateway round trip, a sidecar call.

        Phases are wall clock and must not nest, so these are counted separately
        and keyed by the phase that issued them -- what is left of ``execute``
        after its spans is the agent's own Python.
        """
        key = f"{self._phase_name or '?'}/{kind}"
        with self._lock:
            span = self._spans.setdefault(key, {"calls": 0, "s": 0.0})
            span["calls"] += 1
            span["s"] += elapsed_s

    def end_turn(self, **fields: Any) -> None:
        """Close the current turn, write the file, and start the next one."""
        now = time.perf_counter()
        with self._lock:
            turn = {
                "turn": self._turn_index,
                "wall_s": round(now - self._turn_start, 3),
                "phases": {name: round(value, 3) for name, value in self._phases.items()},
                "spans": {
                    key: {"calls": int(span["calls"]), "s": round(span["s"], 3)}
                    for key, span in self._spans.items()
                },
                "calls": self._calls,
                **fields,
            }
            self._record["turns"].append(turn)
            self._turn_index += 1
            self._turn_start = now
            self._phases = {}
            self._calls = []
            self._spans = {}
        self._write()

    def finish(self, **fields: Any) -> None:
        with self._lock:
            self._record.update(fields)
            self._record["finished_at"] = time.time()
        self._write()

    def _write(self) -> None:
        os.makedirs(self._dir, exist_ok=True)
        # Rewritten every turn, so a reader can catch it half-written; rename in.
        tmp = f"{self._path}.tmp"
        with self._lock:
            payload = json.dumps(self._record)
        with open(tmp, "w") as handle:
            handle.write(payload)
        os.replace(tmp, self._path)


def start_trial(output_dir: str, trial: int, task: str | None, config: dict[str, Any]) -> None:
    """Begin timing a trial on this thread, replacing any unfinished one."""
    _active.timer = TurnTimer(output_dir, trial, task, config)


def active() -> TurnTimer | None:
    return getattr(_active, "timer", None)


@contextmanager
def bind(timer: TurnTimer | None) -> Iterator[None]:
    """Make ``timer`` the active one on this thread (for worker threads)."""
    previous = active()
    _active.timer = timer
    try:
        yield
    finally:
        _active.timer = previous


@contextmanager
def phase(name: str) -> Iterator[None]:
    timer = active()
    if timer is None:
        yield
        return
    with timer.phase(name):
        yield


def record_call(*, model: str, elapsed_s: float, retries: int, usage: dict[str, Any] | None) -> None:
    timer = active()
    if timer is not None:
        timer.record_call(model=model, elapsed_s=elapsed_s, retries=retries, usage=usage)


def record_span(kind: str, elapsed_s: float) -> None:
    timer = active()
    if timer is not None:
        timer.record_span(kind, elapsed_s)


@contextmanager
def span(kind: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        record_span(kind, time.perf_counter() - start)


def end_turn(**fields: Any) -> None:
    timer = active()
    if timer is not None:
        timer.end_turn(**fields)


def finish_trial(**fields: Any) -> None:
    timer = active()
    if timer is not None:
        timer.finish(**fields)
    _active.timer = None
