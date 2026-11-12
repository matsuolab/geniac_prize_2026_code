"""The GPU-side half: claim a job, evaluate it, publish the result, repeat.

On SIGTERM or SIGINT the worker stops claiming. A job it is evaluating is stopped
and put back on the queue without a result: Slurm signals every process of the job,
the submission included, so how that run ended says nothing about the submission.
While a job runs, the worker touches its claim every ``HEARTBEAT_S``. A claim left
by a worker that was killed outright stays in ``queue/claimed`` until it is
``STALE_CLAIM_S`` old; then the next worker of the same slot, or an operator's
``capx evalq requeue-stale``, puts it back. The worker looks for those between jobs
and with every heartbeat, so a long job of its own does not hold one up.

The GPU of a remote simulator serves one evaluation at a time. A worker whose subsets
may use one claims a job only once it holds that GPU's lease, and keeps it until the job
is published or put back. While another worker, of this queue or another, holds it, or
its host does not answer, the worker claims nothing.

A claimed job whose declared weights are not on disk yet is held, not failed: the
watcher fetches a gated repo once the organizer account is granted access, and the job
keeps its place in the queue meanwhile.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
from types import FrameType

from capx.evalqueue.config import QueueConfig
from capx.evalqueue.deadline import close_round, past_deadline
from capx.evalqueue.models import ErrorCode, QueueJob, RunResult
from capx.evalqueue.queue import ClaimedJob, FileQueue
from capx.evalqueue.runner import EvalRunner, Interrupted, LeaseLost
from capx.evalqueue.scratch import prune_runs
from capx.evalqueue.store import EvalStore, run_id_for, utcnow
from capx.evalqueue.weights import describe_mount, prune_weights

_logger = logging.getLogger(__name__)

HEARTBEAT_S = 300.0

STALE_CLAIM_S = 3 * HEARTBEAT_S
"""A claim untouched this long belongs to no running worker."""

MAX_ATTEMPTS = 3
"""Runs a job may start that end with their worker gone. Past that it fails as
``internal_error`` rather than taking down one worker after another."""

WAIT_WARNING_S = 600.0
"""How often a worker that waits for a remote GPU's lease or for weights says so again."""


def default_worker_id() -> str:
    """Stable across restarts of the same Slurm job, so claims can be recovered."""
    job_id = os.environ.get("SLURM_JOB_ID")
    return f"slurm{job_id}" if job_id else socket.gethostname().split(".")[0]


class EvalWorker:
    def __init__(
        self,
        config: QueueConfig,
        store: EvalStore,
        queue: FileQueue,
        runner: EvalRunner,
        worker_id: str | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._queue = queue
        self._runner = runner
        self.worker_id = worker_id or default_worker_id()
        self._stopping = False
        self._waiting_for_gpu = False
        self._next_wait_warning = 0.0

    def install_signal_handlers(self) -> None:
        def _handler(signum: int, _frame: FrameType | None) -> None:
            _logger.warning(
                "signal %s: stopping; a job in hand goes back on the queue",
                signal.Signals(signum).name,
            )
            self.stop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, _handler)

    def stop(self) -> None:
        self._stopping = True
        self._runner.interrupt()

    def run_forever(self, max_jobs: int | None = None) -> int:
        """Claim and evaluate until stopped. Returns how many jobs were run."""
        _logger.info("worker %s ready (root=%s)", self.worker_id, self._store.root)
        completed = 0
        while not self._stopping and (max_jobs is None or completed < max_jobs):
            self._recover()
            if past_deadline(self._config):
                close_round(self._store, self._queue)
                time.sleep(self._config.watcher.poll_interval_s)
                continue
            claimed = self._claim()
            if claimed is None:
                time.sleep(self._config.watcher.poll_interval_s)
                continue
            try:
                if self._evaluate(claimed) is not None:
                    completed += 1
            finally:
                self._runner.release_remote_gpus()
        _logger.info("worker %s stopping after %d job(s)", self.worker_id, completed)
        return completed

    def _evaluate(self, claimed: ClaimedJob) -> RunResult | None:
        """Run a claimed job and publish its result. None if the worker was stopped
        meanwhile: the job is back on the queue, and nothing is published."""
        job = claimed.job
        run_id = run_id_for(job.job_id)
        if job.attempts > MAX_ATTEMPTS:
            gave_up = RunResult.failed(
                run_id,
                job.job_id,
                job.team_id,
                job.submission_id,
                f"gave up after {MAX_ATTEMPTS} runs that ended with their worker gone",
                ErrorCode.INTERNAL_ERROR,
                submitted_at=job.enqueued_at,
            )
            _logger.error("%s: %s", job.job_id, gave_up.error)
            return self._publish(claimed, gave_up)

        _logger.info("running %s as %s (attempt %d)", job.job_id, run_id, job.attempts)
        done = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat, args=(claimed.path, done), daemon=True, name="heartbeat"
        )
        heartbeat.start()
        result: RunResult | None = None
        lost: LeaseLost | None = None
        try:
            # With the heartbeat going: deleting a large tree can outlast a stale claim.
            self._prune_runs(job.team_id)
            self._wait_for_weights(job)
            result = self._runner.run(job, run_id)
        except LeaseLost as exc:
            lost = exc
        except Interrupted:
            pass
        except Exception as exc:  # a runner crash is the submission's result, not ours
            _logger.exception("run %s crashed", run_id)
            result = RunResult.failed(
                run_id,
                job.job_id,
                job.team_id,
                job.submission_id,
                f"{type(exc).__name__}: {exc}",
                ErrorCode.INTERNAL_ERROR,
                submitted_at=job.enqueued_at,
                started_at=utcnow(),
                logs_dir=str(self._store.run_dir(run_id)),
            )
        finally:
            done.set()
            heartbeat.join()

        if lost is not None and not self._stopping:
            # Kept as an attempt: a lease that keeps getting lost must not loop forever.
            self._queue.requeue(claimed)
            _logger.error("%s during %s; %s is back on the queue", lost, run_id, job.job_id)
            return None
        if result is None or self._stopping:
            # Slurm's SIGTERM also ends the lease's ssh: that is this stop, not a loss.
            self._queue.release(claimed)
            _logger.warning("stopped during %s; %s is back on the queue", run_id, job.job_id)
            return None
        return self._publish(claimed, result)

    def _claim(self) -> ClaimedJob | None:
        """A job to run, with the lease on every remote GPU it may use. None, holding
        no lease, when nothing waits or a GPU is not free."""
        if not self._take_remote_gpus():
            return None
        claimed = self._queue.claim(self.worker_id)
        if claimed is None:
            self._runner.release_remote_gpus()
        return claimed

    def _take_remote_gpus(self) -> bool:
        if not self._config.remote_simulators():
            return True
        if not self._queue.pending_jobs():
            return False
        busy = self._runner.take_remote_gpus()
        if busy is not None and time.monotonic() >= self._next_wait_warning:
            _logger.warning("not claiming jobs until the remote GPU is free: %s", busy)
            self._next_wait_warning = time.monotonic() + WAIT_WARNING_S
        elif busy is None and self._waiting_for_gpu:
            _logger.info("the remote GPU is free again")
            self._next_wait_warning = 0.0
        self._waiting_for_gpu = busy is not None
        return busy is None

    def _wait_for_weights(self, job: QueueJob) -> None:
        """Raises :class:`Interrupted` if the worker is stopped meanwhile."""
        next_warning = 0.0
        while missing := [
            describe_mount(mount)
            for mount in job.weights
            if not self._store.weights_dir(mount).is_dir()
        ]:
            if self._stopping:
                raise Interrupted()
            if time.monotonic() >= next_warning:
                _logger.warning(
                    "%s waits for weights %s; the watcher fetches them once the organizer "
                    "account has access",
                    job.job_id,
                    ", ".join(missing),
                )
                next_warning = time.monotonic() + WAIT_WARNING_S
            time.sleep(self._config.watcher.poll_interval_s)

    def _prune_runs(self, team_id: str) -> None:
        try:
            removed = prune_runs(self._store, team_id)
        except Exception:  # never worth the job: the next run of the team tries again
            _logger.exception("removing %s's earlier run directories failed", team_id)
            return
        if removed:
            _logger.info("removed the directories of earlier runs: %s", removed)

    def _recover(self) -> None:
        recovered = self._queue.recover_claims(self.worker_id, STALE_CLAIM_S)
        if recovered:
            _logger.warning("requeued job(s) a dead worker of this slot left: %s", recovered)

    def _heartbeat(self, path: os.PathLike[str], done: threading.Event) -> None:
        while True:
            try:
                os.utime(path)
            except OSError:
                _logger.warning("could not touch claim %s", path)
            try:
                self._recover()
            except OSError:
                _logger.exception("looking for stale claims failed")
            if done.wait(HEARTBEAT_S):
                return

    def _publish(self, claimed: ClaimedJob, result: RunResult) -> RunResult:
        self._store.publish_result(result)
        self._queue.drop_image(claimed, result)
        self._queue.finish(claimed, result.status)
        _logger.info("published %s: %s", result.run_id, result.status.value)
        try:
            prune_weights(self._store, self._queue)
        except OSError:
            _logger.exception("pruning weights after %s failed", result.run_id)
        return result


__all__ = ["HEARTBEAT_S", "MAX_ATTEMPTS", "STALE_CLAIM_S", "EvalWorker", "default_worker_id"]
