"""Tests for the operator-side results dump (``capx.gateway.results_dump``) and
its opt-in wiring in ``docker/serve_environment.py``.

All GPU-free: a stub stands in for the real ``Gateway`` (``results_dump.Scored``).
"""

from __future__ import annotations

import importlib
import json
import multiprocessing
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from capx.gateway.results_dump import ResultsDump, dump_once, install_results_dump
from capx.gateway.scoring import EpisodeScore
from capx.protocol.taskpack import Visibility


class StubGateway:
    """Duck-typed stand-in for ``Gateway``: aggregate() / scores() / timings()."""

    def __init__(self) -> None:
        self._scores = [EpisodeScore("t", reward=1.0, success=True)]

    def aggregate(self) -> dict[str, dict[str, float]]:
        return {
            "t": {"episodes": 1, "trials": 1, "successes": 1, "success_rate": 1.0, "mean_reward": 1.0}
        }

    def scores(self) -> list[EpisodeScore]:
        return list(self._scores)

    def timings(self) -> dict[str, object]:
        return {"episodes": [], "by_task": {}}

    def wait_idle(self, timeout_s: float) -> bool:
        return True

    def end_job(self) -> None:
        pass

    def finished_at(self) -> float | None:
        return None


class LateGateway(StubGateway):
    """A step in flight records its score while the final dump waits for it."""

    def wait_idle(self, timeout_s: float) -> bool:
        self._scores.append(EpisodeScore("late", reward=1.0, success=True))
        return True


class RaisingGateway:
    """aggregate()/scores() always raise — the dump must swallow, not crash."""

    def aggregate(self) -> dict[str, dict[str, float]]:
        raise RuntimeError("boom")

    def scores(self) -> list[EpisodeScore]:
        raise RuntimeError("boom")


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.01) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError("condition not met before timeout")


# --- dump_once ---------------------------------------------------------------


def test_dump_once_writes_valid_schema(tmp_path: Path) -> None:
    path = str(tmp_path / "results.json")
    dump_once(StubGateway(), Visibility.PUBLIC, path)

    document = json.loads(Path(path).read_text())
    assert set(document) == {"updated_at", "leaderboard", "scores", "timings", "finished_at"}
    assert document["leaderboard"]["overall_success_rate"] == 1.0
    assert document["leaderboard"]["visibility"] == "public"
    assert document["scores"] == [{"task_id": "t", "reward": 1.0, "success": True}]


def test_the_dump_says_when_the_finish_was_taken(tmp_path: Path) -> None:
    """The eval queue decides from it whether a finish() came before a stop."""

    class Finished(StubGateway):
        def finished_at(self) -> float | None:
            return 1_700_000_000.5

    path = tmp_path / "results.json"
    dump_once(Finished(), Visibility.PUBLIC, str(path))
    assert json.loads(path.read_text())["finished_at"] == 1_700_000_000.5


def test_dump_once_is_atomic_no_tmp_file_left(tmp_path: Path) -> None:
    path = str(tmp_path / "results.json")
    dump_once(StubGateway(), Visibility.PUBLIC, path)

    assert [p.name for p in tmp_path.iterdir()] == ["results.json"]


def test_dump_once_swallows_exceptions(tmp_path: Path) -> None:
    path = str(tmp_path / "results.json")
    # Must not raise even though aggregate()/scores() blow up.
    dump_once(RaisingGateway(), Visibility.PUBLIC, path)
    assert list(tmp_path.iterdir()) == []


class BigGateway(StubGateway):
    """Enough rows that one dump takes long enough to overlap another."""

    def __init__(self, rows: int) -> None:
        super().__init__()
        self._scores = [EpisodeScore("t", reward=1.0, success=True)] * rows


def test_dumps_from_two_threads_never_share_a_temporary_file(tmp_path: Path) -> None:
    path = str(tmp_path / "results.json")
    threads = [
        threading.Thread(target=dump_once, args=(BigGateway(rows), Visibility.PUBLIC, path))
        for rows in (100_000, 10, 50_000, 10)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(json.loads(Path(path).read_text())["scores"]) in {100_000, 50_000, 10}
    assert [p.name for p in tmp_path.iterdir()] == ["results.json"]


def test_dump_once_private_pack_hides_reward_and_episode_count(tmp_path: Path) -> None:
    path = str(tmp_path / "results.json")
    dump_once(StubGateway(), Visibility.PRIVATE, path)

    document = json.loads(Path(path).read_text())
    assert document["leaderboard"]["tasks"] == {"t": {"success_rate": 1.0}}
    # Per-episode scores are gateway-internal regardless of pack visibility —
    # never redacted here (they never cross EnvProtocol either way).
    assert document["scores"] == [{"task_id": "t", "reward": 1.0, "success": True}]


# --- install_results_dump: interval loop -------------------------------------


def test_install_results_dump_writes_immediately_and_on_interval(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    gateway = StubGateway()
    install_results_dump(gateway, Visibility.PUBLIC, str(path), interval_s=0.05)

    _wait_for(path.exists)
    first_mtime = path.stat().st_mtime_ns
    first_document = json.loads(path.read_text())
    assert first_document["scores"][0]["success"] is True

    # Mutate the underlying score so the next interval dump is observably new.
    gateway._scores.append(EpisodeScore("t", reward=0.0, success=False))
    _wait_for(lambda: path.stat().st_mtime_ns != first_mtime, timeout=5.0)

    updated = json.loads(path.read_text())
    assert len(updated["scores"]) == 2


class BlockingGateway(StubGateway):
    """The first snapshot (the periodic thread's) stalls until released; a step then
    records a late score, as one cut short by the end of the job would."""

    def __init__(self) -> None:
        super().__init__()
        self.snapshotting = threading.Event()
        self.release = threading.Event()
        self.snapshots = 0

    def timings(self) -> dict[str, object]:
        self.snapshots += 1
        if self.snapshots == 1:
            self.snapshotting.set()
            self.release.wait(5)
        return super().timings()

    def wait_idle(self, timeout_s: float) -> bool:
        self._scores.append(EpisodeScore("late", reward=1.0, success=True))
        return True


def test_a_blocked_periodic_dump_neither_delays_nor_overwrites_the_final_one(
    tmp_path: Path,
) -> None:
    """The final dump has to land inside the runner's 30 s before SIGKILL, so it must
    not wait for a slow periodic dump; that one, finishing later with an older
    snapshot, must not replace it."""
    path = tmp_path / "results.json"
    gateway = BlockingGateway()
    results = ResultsDump(gateway, Visibility.PUBLIC, str(path), interval_s=0.01).start()
    assert gateway.snapshotting.wait(5)

    final = threading.Thread(target=results.final)
    final.start()
    final.join(5)
    assert not final.is_alive()
    written = path.read_text()
    assert "late" in {s["task_id"] for s in json.loads(written)["scores"]}

    gateway.release.set()
    results._thread.join(5)
    assert not results._thread.is_alive()
    assert path.read_text() == written
    assert [p.name for p in tmp_path.iterdir()] == ["results.json"]


def test_the_final_dump_ends_the_job_before_it_waits_for_calls_in_progress(
    tmp_path: Path,
) -> None:
    """The eval queue stops scoring a submission by sending the gateway SIGTERM: a call
    in progress may finish its control cycle, but nothing the submission does after
    the signal is scored."""
    order: list[str] = []

    class Recording(StubGateway):
        def end_job(self) -> None:
            order.append("end job")

        def wait_idle(self, timeout_s: float) -> bool:
            order.append("wait")
            return True

    ResultsDump(Recording(), Visibility.PUBLIC, str(tmp_path / "results.json"), 1000.0).final()
    assert order == ["end job", "wait"]


# --- install_results_dump: signal chaining -----------------------------------


def _run_until_killed(path: str) -> None:
    """Child-process entry point: install the dump with a long interval so
    only the immediate startup dump and the SIGTERM-triggered final dump
    happen, then block until the parent's SIGTERM arrives."""
    install_results_dump(LateGateway(), Visibility.PUBLIC, path, interval_s=1000.0)
    while True:  # as the server's loop: a handled signal does not end it
        signal.pause()


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM/os.kill semantics are POSIX-only")
def test_sigterm_performs_a_final_dump_before_exit(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    ctx = multiprocessing.get_context("fork")
    proc = ctx.Process(target=_run_until_killed, args=(str(path),))
    proc.start()
    try:
        _wait_for(path.exists)
        first_document = json.loads(path.read_text())

        proc.terminate()  # SIGTERM
        proc.join(timeout=5.0)
        assert not proc.is_alive()
        # Our handler dumps, restores SIG_DFL, and re-sends SIGTERM to itself,
        # so the process still actually dies of SIGTERM (not silently survives).
        assert proc.exitcode == -signal.SIGTERM

        final_document = json.loads(path.read_text())
        # A second dump landed (after the first) as part of the SIGTERM handler,
        # not just the immediate startup dump.
        assert final_document["updated_at"] > first_document["updated_at"]
        # It waited for the calls in progress, so the score they recorded is in it.
        assert "late" not in {s["task_id"] for s in first_document["scores"]}
        assert "late" in {s["task_id"] for s in final_document["scores"]}
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join()


class LockedGateway(StubGateway):
    """Its end_job() takes a lock that is not reentrant, as the referee's is."""

    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.Lock()

    def end_job(self) -> None:
        with self.lock:
            self._scores.append(EpisodeScore("ended", reward=0.0, success=False))


def _killed_while_the_lock_is_held(path: str) -> None:
    gateway = LockedGateway()
    install_results_dump(gateway, Visibility.PUBLIC, path, interval_s=1000.0)
    with gateway.lock:  # as Gateway.acquire() holds the referee's lock
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.5)
    while True:
        signal.pause()


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM/os.kill semantics are POSIX-only")
def test_sigterm_while_the_main_thread_holds_the_referees_lock_still_dumps(
    tmp_path: Path,
) -> None:
    """The handler runs on the main thread, inside whatever it interrupted: ending the
    job there would wait for a lock the same thread holds, forever."""
    path = tmp_path / "results.json"
    proc = multiprocessing.get_context("fork").Process(
        target=_killed_while_the_lock_is_held, args=(str(path),)
    )
    proc.start()
    try:
        proc.join(timeout=10.0)
        assert not proc.is_alive()
        assert proc.exitcode == -signal.SIGTERM
        assert "ended" in {s["task_id"] for s in json.loads(path.read_text())["scores"]}
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join()


# --- serve_environment.py wiring ---------------------------------------------


def _write_pack(tmp_path: Path, *, visibility: str = "public") -> str:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "pack.yaml").write_text(f"pack_id: p\nvisibility: {visibility}\n"
        "limits: {trial_time_s: 60, n_trials: 1, trial_steplimit: 100}\ntasks: []\n")
    return str(pack_dir)


def test_maybe_install_results_dump_noop_when_env_var_unset(tmp_path: Path, monkeypatch) -> None:
    serve_environment = importlib.import_module("docker.serve_environment")
    monkeypatch.delenv("CAPX_RESULTS_PATH", raising=False)
    taskpack = _write_pack(tmp_path)

    # A gateway stub with no aggregate()/scores() at all: if this were ever
    # called it would AttributeError, proving the no-op path never touches it.
    serve_environment._maybe_install_results_dump(object(), taskpack)


def test_maybe_install_results_dump_wires_dump_when_env_var_set(
    tmp_path: Path, monkeypatch
) -> None:
    serve_environment = importlib.import_module("docker.serve_environment")
    path = tmp_path / "results.json"
    monkeypatch.setenv("CAPX_RESULTS_PATH", str(path))
    monkeypatch.setenv("CAPX_RESULTS_INTERVAL_S", "1000")
    taskpack = _write_pack(tmp_path, visibility="private")

    serve_environment._maybe_install_results_dump(StubGateway(), taskpack)

    _wait_for(path.exists)
    document = json.loads(path.read_text())
    assert document["leaderboard"]["visibility"] == "private"
