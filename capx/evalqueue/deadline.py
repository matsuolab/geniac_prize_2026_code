"""The round's deadline (``QueueConfig.deadline``, PRELIM_RULEBOOK.md §6.4).

From the deadline on, nothing new is queued, waiting submissions are cancelled and a
running one is stopped without a result.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from capx.evalqueue.config import QueueConfig
from capx.evalqueue.models import RunResult, RunStatus
from capx.evalqueue.queue import FileQueue
from capx.evalqueue.store import EvalStore, utcnow

_logger = logging.getLogger(__name__)


def past_deadline(config: QueueConfig, now: datetime | None = None) -> bool:
    return config.deadline is not None and (now or utcnow()) >= config.deadline


def publish_cancelled(
    store: EvalStore, job_id: str, team_id: str, submission_id: str, submitted_at: datetime
) -> None:
    store.publish_result(_cancellation(job_id, team_id, submission_id, submitted_at))


def _cancellation(
    job_id: str, team_id: str, submission_id: str, submitted_at: datetime
) -> RunResult:
    # One run id per job: the watcher and every worker cancel what waits, and any of
    # them may stop halfway and leave it to another, so it may be published again.
    return RunResult(
        run_id=f"{job_id}__deadline",
        job_id=job_id,
        team_id=team_id,
        submission_id=submission_id,
        status=RunStatus.CANCELLED,
        submitted_at=submitted_at,
        finished_at=utcnow(),
        error="the round's deadline passed before it ran",
    )


def close_round(store: EvalStore, queue: FileQueue) -> list[str]:
    """Cancel every submission still waiting: queued, or deferred behind the team's
    running one. Each gets a ``cancelled`` result. Returns their job ids."""
    cancelled: list[str] = []
    for job in queue.pending_jobs():
        if not queue.withdraw(job.job_id):
            continue  # a worker claimed it first; the runner stops it
        publish_cancelled(store, job.job_id, job.team_id, job.submission_id, job.enqueued_at)
        cancelled.append(job.job_id)
    for marker in sorted(store.deferred.glob("*.json")):
        try:
            note = json.loads(marker.read_text())
            result = _cancellation(
                note["job_id"],
                note["team_id"],
                note["submission_id"],
                datetime.fromisoformat(note["since"]),
            )
        except FileNotFoundError:
            continue  # another process finished it
        except OSError as exc:
            _logger.error("deadline: cannot read deferral marker %s: %s", marker, exc)
            continue
        except (ValueError, KeyError, TypeError):
            _logger.warning("deadline: dropping unreadable deferral marker %s", marker)
            marker.unlink(missing_ok=True)
            continue
        try:
            store.publish_result(result)
        except OSError as exc:
            # Kept: this process or another publishes it on its next try.
            _logger.error("deadline: cannot publish the cancellation of %s: %s", result.job_id, exc)
            continue
        cancelled.append(result.job_id)
        marker.unlink(missing_ok=True)
    if cancelled:
        _logger.info("deadline: cancelled %d waiting submission(s): %s", len(cancelled), cancelled)
    return cancelled


__all__ = ["close_round", "past_deadline", "publish_cancelled"]
