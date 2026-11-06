"""A durable queue built from directory renames.

State is the directory a job's record sits in (``pending`` -> ``claimed`` ->
``done``/``failed``). Claiming is ``rename(pending/x, claimed/x.<worker>.<stamp>)``:
the rename is atomic and the loser of a race gets ``ENOENT``, so exactly one worker
takes a job without any lock, lease, or broker. The stamp (``<epoch>-<random>``)
makes every claim's name its own, so nothing aimed at an old claim can hit a newer
claim of the same job, and says when it was claimed from the moment it exists: the
renamed record still has the pending file's mtime until it is rewritten.

Ordering is FIFO by enqueue time, except that a team already holding a claimed job
is skipped — one submission at a time per team, so a team that queues five in a row
cannot block everyone else behind them.

A record left in ``claimed`` whose heartbeat stopped means a worker died holding it:
the worker that takes over its slot recovers it (:meth:`FileQueue.recover_claims`),
and an operator can requeue any (:meth:`FileQueue.requeue_stale`). A recovery first
renames the claim to a name of its own, the same exclusive step as a claim, so two
recoverers never both requeue it. One whose result was published already is moved on
instead of being run again.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import secrets
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from capx.evalqueue.models import ErrorCode, QueueJob, RunResult, RunSpec, RunStatus
from capx.evalqueue.store import EvalStore, utcnow, write_json_atomic

_logger = logging.getLogger(__name__)

_STAMP = re.compile(r"r?(\d+)-[0-9a-f]{8}")
"""A time stamped into a claim's name: when it was claimed, and ``r...`` when a
recovery took it (:meth:`FileQueue._take`)."""

OUR_SIDE = {ErrorCode.INTERNAL_ERROR, ErrorCode.NO_SCORE}
"""Failures that are ours, not the submission's: its image is kept to run it again,
and its run directory for an operator to look at."""


def failed_on_our_side(result: RunResult) -> bool:
    codes = {result.error_code, *(subset.error_code for subset in result.subsets.values())}
    return bool(codes & OUR_SIDE)


def _stamp() -> str:
    return f"{int(time.time())}-{secrets.token_hex(4)}"


def _claim_name(name: str) -> tuple[str, list[int]]:
    """``<job_id>.<worker_id>`` and the times stamped into a claim's file name. A
    claim from before the stamps (``<job_id>.<worker_id>.json``) has none."""
    parts = name.removesuffix(".json").split(".")
    stamps: list[int] = []
    while len(parts) > 2 and (stamp := _STAMP.fullmatch(parts[-1])):
        stamps.append(int(stamp[1]))
        parts.pop()
    return ".".join(parts), stamps


@dataclass(frozen=True)
class ClaimedJob:
    job: QueueJob
    path: Path
    """Where the record now lives, under ``queue/claimed``."""


def _load(path: Path) -> QueueJob | None:
    try:
        return QueueJob.model_validate(json.loads(path.read_text()))
    except (OSError, ValueError) as exc:
        _logger.error("unreadable queue record %s: %s", path, exc)
        return None


class FileQueue:
    def __init__(self, store: EvalStore) -> None:
        self._store = store

    def enqueue(self, job: QueueJob) -> Path:
        path = self._store.pending / f"{job.job_id}.json"
        write_json_atomic(path, job.model_dump(mode="json"))
        return path

    def is_queued(self, job_id: str) -> bool:
        """True while a job is waiting or running — the resubmission guard."""
        if (self._store.pending / f"{job_id}.json").exists():
            return True
        return any(self._store.claimed.glob(f"{job_id}.*.json"))

    def has_record(self, job_id: str) -> bool:
        """True while a job is waiting or running, and after it ended, however. Read in
        the order a job moves, so one moving meanwhile is not missed, except back from
        claimed to pending: a requeue."""
        return self.is_queued(job_id) or any(
            (state / f"{job_id}.json").exists() for state in (self._store.done, self._store.failed)
        )

    def running_count(self, team_id: str) -> int:
        return sum(job.team_id == team_id for job in self.claimed_jobs())

    def team_pending(self, team_id: str) -> list[QueueJob]:
        """A team's waiting jobs, oldest first."""
        return [job for job in self.pending_jobs() if job.team_id == team_id]

    def withdraw(self, job_id: str) -> bool:
        """Take a job off the queue before any worker has it.

        The same atomic filesystem step as :meth:`claim`, so exactly one of a worker
        and a withdrawal wins: False means a worker claimed it first and it is running.
        """
        try:
            (self._store.pending / f"{job_id}.json").unlink()
        except FileNotFoundError:
            return False
        return True

    def pending_jobs(self) -> list[QueueJob]:
        jobs = [job for job in map(_load, sorted(self._store.pending.glob("*.json"))) if job]
        return sorted(jobs, key=lambda job: job.enqueued_at)

    def claimed_jobs(self) -> list[QueueJob]:
        return [job for job in map(_load, sorted(self._store.claimed.glob("*.json"))) if job]

    def finished_jobs(self) -> list[QueueJob]:
        ended = [*self._store.done.glob("*.json"), *self._store.failed.glob("*.json")]
        return [job for job in map(_load, sorted(ended)) if job]

    def claim(self, worker_id: str) -> ClaimedJob | None:
        """Take the oldest eligible job, or None when there is nothing to do."""
        busy_teams = {job.team_id for job in self.claimed_jobs()}
        for job in self.pending_jobs():
            if job.team_id in busy_teams:
                continue
            source = self._store.pending / f"{job.job_id}.json"
            target = self._store.claimed / f"{job.job_id}.{worker_id}.{_stamp()}.json"
            try:
                os.rename(source, target)
            except OSError:
                continue  # another worker got there first, or it was withdrawn
            if self._team_claimed_elsewhere(job.team_id, target):
                # Two workers that looked at once can each take one of a team's jobs.
                # Whoever then sees the other's claim backs off, so at most one stays.
                with contextlib.suppress(FileNotFoundError):
                    os.rename(target, source)
                busy_teams.add(job.team_id)
                continue
            claimed = job.model_copy(update={"attempts": job.attempts + 1, "claimed_at": utcnow()})
            write_json_atomic(target, claimed.model_dump(mode="json"))
            return ClaimedJob(job=claimed, path=target)
        return None

    def _team_claimed_elsewhere(self, team_id: str, mine: Path) -> bool:
        others = (path for path in self._store.claimed.glob("*.json") if path != mine)
        return any(job.team_id == team_id for job in map(_load, others) if job)

    def drop_image(self, claimed: ClaimedJob, result: RunResult) -> None:
        """Delete what the watcher fetched for a job (``incoming/<team>/<key>/``) once
        its result is out: images go up to 80 GB, and the Hub keeps them. Kept when
        our side failed, so it can be run again by hand, and while another queued job
        uses it. Runs before :meth:`finish`, so a worker that dies in between leaves the
        claim, and the recovery that finishes it drops the image again."""
        if failed_on_our_side(result):
            return
        team_dir = self._store.incoming / claimed.job.team_id
        image = Path(claimed.job.image_path)
        if not image.is_relative_to(team_dir) or len(image.relative_to(team_dir).parts) < 2:
            return  # queued by hand from somewhere else
        fetched = team_dir / image.relative_to(team_dir).parts[0]
        other_claims = (path for path in self._store.claimed.glob("*.json") if path != claimed.path)
        queued = [*self.pending_jobs(), *(job for job in map(_load, other_claims) if job)]
        if not any(Path(job.image_path).is_relative_to(fetched) for job in queued):
            shutil.rmtree(fetched, ignore_errors=True)

    def finish(self, claimed: ClaimedJob, status: RunStatus) -> Path:
        destination = self._store.done if status is RunStatus.COMPLETED else self._store.failed
        target = destination / f"{claimed.job.job_id}.json"
        write_json_atomic(target, claimed.job.model_dump(mode="json"))
        claimed.path.unlink(missing_ok=True)
        return target

    def requeue(self, claimed: ClaimedJob) -> Path:
        target = self.enqueue(claimed.job.model_copy(update={"claimed_at": None}))
        claimed.path.unlink(missing_ok=True)
        return target

    def release(self, claimed: ClaimedJob) -> Path:
        """Put a job back as it was before it was claimed, for a worker that stops
        before it ends: that run does not count against ``attempts``."""
        job = claimed.job
        return self.requeue(
            ClaimedJob(job=job.model_copy(update={"attempts": job.attempts - 1}), path=claimed.path)
        )

    def recover_claims(self, worker_id: str, older_than_s: float) -> list[str]:
        """Requeue records this worker id left behind, once untouched for
        ``older_than_s``. A fresh one belongs to a worker that is still running, such
        as a second job Slurm started for the same slot."""
        cutoff = time.time() - older_than_s
        return self._recover(self._untouched(cutoff, worker_id), cutoff)

    def requeue_stale(self, older_than_s: float, *, dry_run: bool = False) -> list[str]:
        """Requeue claims whose worker stopped touching them ``older_than_s`` ago."""
        cutoff = time.time() - older_than_s
        return self._recover(self._untouched(cutoff), cutoff, dry_run=dry_run)

    def _untouched(self, cutoff: float, worker_id: str | None = None) -> list[Path]:
        """Claims (of ``worker_id``, if given) neither touched nor claimed or taken
        since ``cutoff``."""
        found: list[Path] = []
        for path in sorted(self._store.claimed.glob("*.json")):
            owner, stamps = _claim_name(path.name)
            if worker_id is not None and not owner.endswith(f".{worker_id}"):
                continue
            try:
                touched = path.stat().st_mtime
            except FileNotFoundError:
                continue  # finished since the glob
            if max([touched, *stamps]) < cutoff:
                found.append(path)
        return found

    def _take(self, path: Path, cutoff: float) -> Path | None:
        """Rename a stale claim to a name of its own, so that only one recoverer
        handles it. The name carries when it was taken, which keeps others off it
        while this one requeues it, and lets them recover it if this one dies first.
        None if it is gone, or was touched after all."""
        taken = path.with_name(f"{path.stem}.r{_stamp()}.json")
        try:
            os.rename(path, taken)
            if taken.stat().st_mtime < cutoff:
                return taken
            os.rename(taken, path)  # its worker touched it since: it is alive
        except FileNotFoundError:
            pass  # its run ended, or another recoverer took it
        return None

    def _recover(self, paths: Iterable[Path], cutoff: float, *, dry_run: bool = False) -> list[str]:
        requeued: list[str] = []
        for path in paths:
            job = _load(path)
            if job is None:
                continue
            taken = path if dry_run else self._take(path, cutoff)
            if taken is None:
                continue
            claimed = ClaimedJob(job=job, path=taken)
            published = self._published_result(job)
            if published is not None:
                # The worker published the result and died before moving the record.
                _logger.warning(
                    "%s already has a %s result; finishing it", job.job_id, published.status.value
                )
                if not dry_run:
                    self.drop_image(claimed, published)
                    self.finish(claimed, published.status)
                continue
            if not dry_run:
                self.requeue(claimed)
            requeued.append(job.job_id)
        return requeued

    def _published_result(self, job: QueueJob) -> RunResult | None:
        """The result the run of this claim published, if it did."""
        if job.claimed_at is None:
            return None
        for path in self._store.results.glob(f"{job.job_id}__*.json"):
            try:
                result = RunResult.model_validate_json(path.read_text())
            except (OSError, ValueError):
                continue
            if (
                result.job_id == job.job_id
                and result.finished_at is not None
                and result.finished_at >= job.claimed_at
            ):
                return result
        return None


def new_job(
    job_id: str,
    spec: RunSpec,
    image: Path,
    *,
    hub_repo: str | None = None,
    hub_revision: str | None = None,
) -> QueueJob:
    return QueueJob(
        job_id=job_id,
        team_id=spec.team_id,
        submission_id=spec.submission_id,
        image_path=str(image),
        entrypoint=spec.entrypoint,
        workdir=spec.workdir,
        env=spec.env,
        weights=spec.weights,
        hub_repo=hub_repo,
        hub_revision=hub_revision,
        enqueued_at=utcnow(),
    )


__all__ = ["OUR_SIDE", "ClaimedJob", "FileQueue", "failed_on_our_side", "new_job"]
