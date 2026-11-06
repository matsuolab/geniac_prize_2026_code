from __future__ import annotations

import math
import multiprocessing
import traceback
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.connection import Connection, wait
from typing import Any, TypeVar

BatchResult = TypeVar("BatchResult")


def chunk_into_batches(items: Sequence[int], batch_size: int) -> list[list[int]]:
    """Split ``items`` into ``batch_size`` chunks."""

    if batch_size <= 0:
        return [list(items)]
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


def run_parallel_batches(
    trial_ids: Sequence[int],
    *,
    num_workers: int,
    batch_fn: Callable[[Sequence[int]], Iterable[BatchResult]],
    batch_size: int | None = None,
    mp_start_method: str = "spawn",
) -> list[BatchResult]:
    """Execute ``batch_fn`` across ``trial_ids`` with optional multiprocessing.

    Args:
        trial_ids: Ordered trial identifiers to evaluate.
        num_workers: Number of worker processes. Values <= 1 run sequentially.
        batch_fn: Callable that consumes a batch of trial ids and returns an iterable
            of results. Must be picklable when ``num_workers > 1``.
        batch_size: Optional explicit batch size. Defaults to an even split across
            ``num_workers``.
        mp_start_method: Multiprocessing start method (defaults to ``"spawn"`` for
            compatibility with CUDA + Mujoco setups).

    Returns:
        List of aggregated ``batch_fn`` results (concatenated).
    """

    if not trial_ids:
        return []

    if num_workers <= 1 or len(trial_ids) == 1:
        return list(batch_fn(trial_ids))

    if batch_size is None:
        batch_size = max(1, math.ceil(len(trial_ids) / num_workers))

    batches = chunk_into_batches(trial_ids, batch_size)
    ctx = multiprocessing.get_context(mp_start_method)

    results: list[BatchResult] = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = [executor.submit(batch_fn, batch) for batch in batches]
        # Collect all futures, even if some fail
        for future in as_completed(futures):
            # try:
            results.extend(future.result())
            # except Exception as e:
            #     print(f"Worker failed with exception: {e}")
            #     # Continue processing other workers instead of crashing
            #     continue

    return results


def _worker_loop_with_setup(
    worker_id: int,
    task_queue: multiprocessing.Queue,
    results: Connection,
    setup_fn: Callable[[], Any],
    trial_fn: Callable[[Any, int], BatchResult],
) -> None:
    """Worker loop that initializes state once, then pulls trials from queue.

    Args:
        worker_id: ID of this worker (for logging).
        task_queue: Queue of trial IDs to process.
        results: This worker's own pipe to the collector. A shared queue would not
            do: a worker that dies while its queue thread holds the shared write lock
            blocks every other worker's results for good.
        setup_fn: Function called once to create worker state (e.g., environment).
        trial_fn: Function that takes (state, trial_id) and returns a result.
    """
    # Initialize worker state once
    try:
        state = setup_fn()
    except Exception:
        # Report setup failure and exit
        results.send(("setup_error", worker_id, traceback.format_exc()))
        return

    while True:
        try:
            # Non-blocking get with timeout to allow clean shutdown
            trial_id = task_queue.get(timeout=1.0)
        except Exception:
            # Queue is empty or closed
            break

        if trial_id is None:
            # Sentinel value signals worker to exit
            break

        try:
            result = trial_fn(state, trial_id)
            results.send(("success", trial_id, result))
        except Exception:
            results.send(("error", trial_id, traceback.format_exc()))


def _read_result(pipe: Connection) -> tuple[str, int, Any] | None:
    """A worker's next message, or None once its pipe is closed.

    A worker that dies partway through sending leaves a truncated message, which
    ``recv`` reports as ``OSError`` rather than ``EOFError``; both end that worker.
    """
    try:
        return pipe.recv()
    except (EOFError, OSError):
        return None


def run_parallel_with_setup(
    trial_ids: Sequence[int],
    *,
    num_workers: int,
    setup_fn: Callable[[], Any],
    trial_fn: Callable[[Any, int], BatchResult],
    on_failure: Callable[[int, str], BatchResult],
    mp_start_method: str = "spawn",
) -> list[BatchResult]:
    """Execute trials dynamically with per-worker setup (e.g., environment creation).

    Each worker calls ``setup_fn`` once to create its state (e.g., a simulation
    environment), then loops pulling trials from a shared queue and calling
    ``trial_fn(state, trial_id)`` for each. This provides:
    - Efficient environment reuse (no re-creation per trial)
    - Dynamic load balancing (workers grab trials as they become available)

    Every trial id yields exactly one result, so a summary over them keeps the right
    denominator: ``trial_fn``'s, or ``on_failure(trial_id, reason)`` when the trial
    raised, the worker running it died, or no worker could be set up to run it.

    Args:
        trial_ids: Ordered trial identifiers to evaluate.
        num_workers: Number of worker processes. Values <= 1 run sequentially.
        setup_fn: Callable that creates per-worker state (called once per worker).
            Must be picklable.
        trial_fn: Callable that takes (state, trial_id) and returns a result.
            Must be picklable.
        on_failure: Builds the result recorded for a trial that produced none. Runs
            in the calling process.
        mp_start_method: Multiprocessing start method (defaults to ``"spawn"`` for
            compatibility with CUDA + Mujoco setups).

    Returns:
        List of results (one per trial, order may differ from input).
    """
    if not trial_ids:
        return []

    if num_workers <= 1:
        try:
            state = setup_fn()
        except Exception:
            reason = traceback.format_exc()
            print(f"Setup failed: {reason}")
            return [on_failure(tid, reason) for tid in trial_ids]
        results: list[BatchResult] = []
        for tid in trial_ids:
            try:
                results.append(trial_fn(state, tid))
            except Exception:
                results.append(on_failure(tid, traceback.format_exc()))
        return results

    ctx = multiprocessing.get_context(mp_start_method)
    task_queue: multiprocessing.Queue = ctx.Queue()

    # Populate task queue
    for trial_id in trial_ids:
        task_queue.put(trial_id)

    # Add sentinel values to signal workers to exit
    for _ in range(num_workers):
        task_queue.put(None)

    # Start worker processes
    workers = []
    pipes: list[Connection] = []
    for worker_id in range(num_workers):
        reader, writer = ctx.Pipe(duplex=False)
        p = ctx.Process(
            target=_worker_loop_with_setup,
            args=(worker_id, task_queue, writer, setup_fn, trial_fn),
        )
        p.start()
        # Only the worker holds the write end now, so its pipe reads EOF once it exits.
        writer.close()
        workers.append(p)
        pipes.append(reader)

    results = []
    outstanding = Counter(trial_ids)
    failed = 0
    setup_errors = 0

    def record(status: str, identifier: int, data: Any) -> None:
        nonlocal failed, setup_errors
        if status == "setup_error":
            setup_errors += 1
            print(f"Worker {identifier} failed during setup: {data}")
            return
        if outstanding[identifier] <= 0:
            return
        outstanding[identifier] -= 1
        if status == "success":
            results.append(data)
            return
        failed += 1
        print(f"Trial {identifier} failed with error: {data}")
        results.append(on_failure(identifier, data))

    try:
        while outstanding.total() > 0 and pipes:
            for pipe in wait(pipes):
                assert isinstance(pipe, Connection)
                message = _read_result(pipe)
                if message is None:
                    pipes.remove(pipe)
                else:
                    record(*message)
    finally:
        # Always clean up workers, including on KeyboardInterrupt
        for p in workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=3.0)
                if p.is_alive():
                    p.kill()  # SIGKILL for workers stuck in C extension code
        # Tasks no worker took are abandoned. Without this, the queue's feeder thread
        # stays blocked on a full pipe and holds up interpreter exit.
        task_queue.close()
        task_queue.cancel_join_thread()

    if outstanding.total() > 0:
        exit_codes = [p.exitcode for p in workers]
        reason = f"no result: the workers exited without reporting one (exit codes {exit_codes})"
        for trial_id, count in outstanding.items():
            for _ in range(count):
                failed += 1
                print(f"Trial {trial_id} failed: {reason}")
                results.append(on_failure(trial_id, reason))

    if setup_errors:
        print(f"WARNING: {setup_errors} worker(s) failed during setup")
    if failed:
        print(f"WARNING: {failed} trial(s) failed")

    return results


__all__ = [
    "chunk_into_batches",
    "run_parallel_batches",
    "run_parallel_with_setup",
]
