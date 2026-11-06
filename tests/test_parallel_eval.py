"""``run_parallel_with_setup`` accounts for every trial, whatever its workers do.

The local runner divides its success rate by the number of results it gets back, so
a trial that silently disappears (a worker that failed setup, raised, or died) skews
the rate, and a collector waiting for results that never come hangs the run.

Workers are real spawned processes, so the callables below live at module level.
"""

from __future__ import annotations

import multiprocessing
import os
import struct
import subprocess
import sys
from pathlib import Path

from capx.utils.parallel_eval import _read_result, run_parallel_with_setup


def _setup_ok() -> str:
    return "state"


def _setup_fails() -> str:
    raise RuntimeError("env construction failed")


def _setup_fails_in_one_worker() -> str:
    # The first worker to get here claims the marker and fails; the others set up.
    marker = os.environ["CAPX_TEST_SETUP_MARKER"]
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        return "state"
    os.close(fd)
    raise RuntimeError("this worker could not build its env")


def _trial_ok(state: str, trial: int) -> tuple[str, int]:
    return ("ok", trial)


def _trial_raises_on_3(state: str, trial: int) -> tuple[str, int]:
    if trial == 3:
        raise ValueError("trial 3 broke")
    return ("ok", trial)


def _trial_dies_on_2(state: str, trial: int) -> tuple[str, int]:
    if trial == 2:
        os._exit(1)  # a segfault or OOM kill: no exception, no message
    return ("ok", trial)


def _failed(trial: int, reason: str) -> tuple[str, int]:
    return ("failed", trial)


def _run(setup_fn, trial_fn, trial_ids=(1, 2, 3, 4, 5, 6), num_workers=2):
    return run_parallel_with_setup(
        list(trial_ids),
        num_workers=num_workers,
        setup_fn=setup_fn,
        trial_fn=trial_fn,
        on_failure=_failed,
    )


def test_every_trial_returns_its_result() -> None:
    assert sorted(_run(_setup_ok, _trial_ok)) == [("ok", t) for t in range(1, 7)]


def test_all_workers_failing_setup_returns_every_trial_as_failed() -> None:
    assert sorted(_run(_setup_fails, _trial_ok)) == [("failed", t) for t in range(1, 7)]


def test_one_worker_failing_setup_loses_no_trial(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CAPX_TEST_SETUP_MARKER", str(tmp_path / "claimed"))
    results = _run(_setup_fails_in_one_worker, _trial_ok, num_workers=3)
    assert sorted(results) == [("ok", t) for t in range(1, 7)]


def test_a_trial_that_raises_stays_in_the_results_as_failed() -> None:
    results = _run(_setup_ok, _trial_raises_on_3)
    assert sorted(results) == sorted([("ok", t) for t in (1, 2, 4, 5, 6)] + [("failed", 3)])


def test_a_worker_that_dies_mid_trial_does_not_hang_the_run() -> None:
    results = _run(_setup_ok, _trial_dies_on_2)
    assert ("failed", 2) in results
    assert len(results) == 6
    assert {trial for _, trial in results} == set(range(1, 7))


def test_sequential_mode_records_a_raising_trial_as_failed() -> None:
    results = _run(_setup_ok, _trial_raises_on_3, num_workers=1)
    assert results == [("ok", 1), ("ok", 2), ("failed", 3), ("ok", 4), ("ok", 5), ("ok", 6)]


def test_sequential_mode_records_every_trial_when_setup_fails() -> None:
    assert _run(_setup_fails, _trial_ok, num_workers=1) == [("failed", t) for t in range(1, 7)]


def test_a_message_cut_off_mid_send_ends_that_worker() -> None:
    reader, writer = multiprocessing.Pipe(duplex=False)
    # The frame header promises 100 bytes; the worker died after sending 10.
    os.write(writer.fileno(), struct.pack("!i", 100) + b"x" * 10)
    writer.close()
    assert _read_result(reader) is None


def test_tasks_left_behind_by_failed_workers_do_not_hold_up_exit() -> None:
    """Enough queued tasks to fill the task pipe, and no worker to read them."""
    script = (
        "from test_parallel_eval import _failed, _setup_fails, _trial_ok\n"
        "from capx.utils.parallel_eval import run_parallel_with_setup\n"
        "if __name__ == '__main__':\n"
        "    results = run_parallel_with_setup(list(range(20000)), num_workers=2,\n"
        "        setup_fn=_setup_fails, trial_fn=_trial_ok, on_failure=_failed)\n"
        "    print('results', len(results))\n"
    )
    tests_dir = Path(__file__).parent
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tests_dir), *sys.path])}
    done = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "results 20000" in done.stdout
