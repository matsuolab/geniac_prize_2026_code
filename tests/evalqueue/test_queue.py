"""The queue's guarantees: one claimer per job, per-team fairness, nothing dropped."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest

from capx.evalqueue import queue as queue_module
from capx.evalqueue.models import QueueJob, RunResult, RunStatus
from capx.evalqueue.queue import FileQueue
from capx.evalqueue.store import EvalStore, utcnow


@pytest.fixture
def queue(tmp_path) -> FileQueue:
    store = EvalStore(tmp_path)
    store.ensure_layout()
    return FileQueue(store)


def _later(monkeypatch, seconds: float = 7200.0) -> float:
    """Move the clock the queue reads ahead, so that what was just claimed is stale.
    Returns the new now."""
    now = time.time() + seconds
    monkeypatch.setattr(queue_module.time, "time", lambda: now)
    return now


def make_job(job_id: str, team_id: str = "t", *, age_s: float = 0.0) -> QueueJob:
    return QueueJob(
        job_id=job_id,
        team_id=team_id,
        submission_id=job_id,
        image_path=f"/incoming/{job_id}/agent.sif",
        enqueued_at=utcnow() - timedelta(seconds=age_s),
    )


def test_claim_is_fifo_by_enqueue_time(queue) -> None:
    queue.enqueue(make_job("newer", "a", age_s=1))
    queue.enqueue(make_job("older", "b", age_s=100))
    assert queue.claim("w1").job.job_id == "older"


def test_exactly_one_worker_wins_a_contested_job(queue) -> None:
    queue.enqueue(make_job("only"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(queue.claim, [f"w{i}" for i in range(8)]))
    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    assert winners[0].job.job_id == "only"


def test_a_team_holds_at_most_one_running_job(queue) -> None:
    """Otherwise one team queueing five in a row starves everyone behind it."""
    queue.enqueue(make_job("a1", "alpha", age_s=100))
    queue.enqueue(make_job("a2", "alpha", age_s=99))
    queue.enqueue(make_job("b1", "beta", age_s=50))

    assert queue.claim("w1").job.job_id == "a1"
    assert queue.claim("w2").job.job_id == "b1"  # a2 skipped: alpha is busy
    assert queue.claim("w3") is None


def test_claim_records_the_attempt(queue) -> None:
    queue.enqueue(make_job("j"))
    claimed = queue.claim("w1").job
    assert claimed.attempts == 1
    assert claimed.claimed_at is not None
    assert queue.claimed_jobs()[0].claimed_at == claimed.claimed_at


def test_requeue_clears_the_claim_time(queue) -> None:
    queue.enqueue(make_job("j"))
    queue.requeue(queue.claim("w1"))
    assert queue.pending_jobs()[0].claimed_at is None


def test_finish_moves_the_record_out_of_claimed(queue, tmp_path) -> None:
    queue.enqueue(make_job("j"))
    claimed = queue.claim("w1")
    queue.finish(claimed, RunStatus.COMPLETED)
    assert not claimed.path.exists()
    assert (tmp_path / "queue" / "done" / "j.json").exists()
    assert queue.claim("w1") is None


def test_failed_jobs_land_in_failed(queue, tmp_path) -> None:
    queue.enqueue(make_job("j"))
    queue.finish(queue.claim("w1"), RunStatus.FAILED)
    assert (tmp_path / "queue" / "failed" / "j.json").exists()


def test_recover_claims_puts_a_dead_workers_job_back(queue, monkeypatch) -> None:
    queue.enqueue(make_job("j"))
    queue.claim("w1")  # worker then dies without finishing

    assert queue.claim("w2") is None  # still claimed, so nobody else takes it
    assert queue.recover_claims("w1", 3600) == []  # its heartbeat may still be going
    _later(monkeypatch)
    assert queue.recover_claims("w1", 3600) == ["j"]
    assert queue.claim("w2").job.job_id == "j"


def _publish(queue, job_id: str, status: RunStatus, finished_at) -> None:
    queue._store.publish_result(
        RunResult(
            run_id=f"{job_id}__x",
            job_id=job_id,
            team_id="t",
            submission_id=job_id,
            status=status,
            finished_at=finished_at,
        )
    )


def test_a_claim_whose_result_is_published_is_finished_not_run_again(
    queue, tmp_path, monkeypatch
) -> None:
    """The worker published, then died before moving the record. Running it again
    would give the team a second scored run."""
    queue.enqueue(make_job("j"))
    claimed = queue.claim("w1")
    _publish(queue, "j", RunStatus.COMPLETED, claimed.job.claimed_at + timedelta(hours=1))
    _later(monkeypatch)

    assert queue.requeue_stale(3600) == []
    assert not claimed.path.exists() and queue.pending_jobs() == []
    assert (tmp_path / "queue" / "done" / "j.json").exists()


def test_a_result_from_an_earlier_run_does_not_finish_a_claim(queue, monkeypatch) -> None:
    queue.enqueue(make_job("j"))
    claimed = queue.claim("w1")
    _publish(queue, "j", RunStatus.FAILED, claimed.job.claimed_at - timedelta(hours=1))
    _publish(queue, "j__other", RunStatus.COMPLETED, claimed.job.claimed_at + timedelta(hours=1))
    _later(monkeypatch)

    assert queue.recover_claims("w1", 3600) == ["j"]


def test_requeue_stale_skips_a_claim_that_finishes_meanwhile(queue, monkeypatch) -> None:
    queue.enqueue(make_job("j"))
    claimed = queue.claim("w1")
    glob = type(claimed.path).glob

    def glob_then_finish(self, pattern):
        found = list(glob(self, pattern))
        claimed.path.unlink(missing_ok=True)
        return iter(found)

    monkeypatch.setattr(type(claimed.path), "glob", glob_then_finish)
    assert queue.requeue_stale(0) == []


def test_release_puts_a_job_back_as_it_was(queue) -> None:
    queue.enqueue(make_job("j"))
    queue.release(queue.claim("w1"))
    (job,) = queue.pending_jobs()
    assert job.attempts == 0 and job.claimed_at is None


def test_two_workers_never_both_hold_a_teams_jobs(queue, monkeypatch) -> None:
    """The busy-team look and the rename are separate steps, so a second worker can
    take a team's second job while the first takes its first."""
    queue.enqueue(make_job("a1", "alpha", age_s=100))
    queue.enqueue(make_job("a2", "alpha", age_s=99))
    claimed_jobs = FileQueue.claimed_jobs
    first_look = []

    def look_before_the_other_claims(self):
        jobs = claimed_jobs(self)
        if not first_look:
            first_look.append(1)
            assert queue.claim("w1").job.job_id == "a1"  # lands between look and rename
        return jobs

    monkeypatch.setattr(FileQueue, "claimed_jobs", look_before_the_other_claims)
    assert queue.claim("w2") is None
    assert [job.job_id for job in queue.pending_jobs()] == ["a2"]
    assert queue.running_count("alpha") == 1


def test_is_queued_covers_both_pending_and_claimed(queue) -> None:
    queue.enqueue(make_job("j"))
    assert queue.is_queued("j")
    claimed = queue.claim("w1")
    assert queue.is_queued("j")
    queue.finish(claimed, RunStatus.COMPLETED)
    assert not queue.is_queued("j")


def test_an_unreadable_record_does_not_break_the_queue(queue, tmp_path) -> None:
    queue.enqueue(make_job("good", "a"))
    (tmp_path / "queue" / "pending" / "corrupt.json").write_text("{not json")
    assert [job.job_id for job in queue.pending_jobs()] == ["good"]


def test_running_count_and_a_teams_waiting_jobs_are_per_team(queue) -> None:
    queue.enqueue(make_job("a1", team_id="a"))
    queue.enqueue(make_job("a2", team_id="a", age_s=-1))
    queue.enqueue(make_job("b1", team_id="b"))
    assert [job.job_id for job in queue.team_pending("a")] == ["a1", "a2"]  # oldest first
    assert queue.running_count("a") == 0
    claimed = queue.claim("w0")
    assert claimed is not None and claimed.job.team_id == "a"
    assert queue.running_count("a") == 1
    assert [job.job_id for job in queue.team_pending("a")] == ["a2"]
    assert queue.running_count("nobody") == 0


def test_a_withdrawal_and_a_claim_cannot_both_win(queue) -> None:
    queue.enqueue(make_job("a1", team_id="a"))
    assert queue.withdraw("a1") is True
    assert queue.claim("w0") is None  # it is gone

    queue.enqueue(make_job("a2", team_id="a"))
    assert queue.claim("w0") is not None
    assert queue.withdraw("a2") is False  # a worker has it; it keeps running
    assert queue.running_count("a") == 1


def test_requeue_stale_takes_back_only_claims_whose_heartbeat_stopped(queue, monkeypatch) -> None:
    queue.enqueue(make_job("stale", "a"))
    queue.enqueue(make_job("live", "b"))
    stale, live = queue.claim("w1"), queue.claim("w2")
    now = _later(monkeypatch)
    os.utime(live.path, (now, now))  # its heartbeat goes on

    assert queue.requeue_stale(3600, dry_run=True) == ["stale"]
    assert stale.path.exists()
    assert queue.requeue_stale(3600) == ["stale"]
    assert [job.job_id for job in queue.pending_jobs()] == ["stale"]
    assert live.path.exists()


def test_a_record_with_a_removed_field_is_still_read(queue, tmp_path) -> None:
    """Records written before a field was dropped must still be claimed and recovered."""
    import json

    from capx.evalqueue.queue import _load

    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "job_id": "alpha__s1",
                "team_id": "alpha",
                "submission_id": "s1",
                "image_path": "/x/agent.sif",
                "enqueued_at": "2026-10-01T00:00:00+00:00",
                "submission_dir": "/x",
                "image_gb": 9.5,
            }
        )
    )
    job = _load(path)
    assert job is not None and job.job_id == "alpha__s1"


def test_concurrent_recoveries_requeue_a_stale_claim_once(queue, monkeypatch) -> None:
    queue.enqueue(make_job("j"))
    queue.claim("w1")
    _later(monkeypatch)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: queue.recover_claims("w1", 3600), range(8)))

    assert sorted(results) == [[]] * 7 + [["j"]]
    assert [job.job_id for job in queue.pending_jobs()] == ["j"]
    assert queue.claimed_jobs() == []


def test_a_late_recovery_leaves_the_jobs_new_claim_alone(queue, monkeypatch) -> None:
    """Two workers of one slot find the same stale claim. One requeues it and claims the
    job again; the other, acting on what it found, must not touch that new claim."""
    queue.enqueue(make_job("j"))
    queue.claim("w1")
    cutoff = _later(monkeypatch) - 3600
    found_by_the_second = queue._untouched(cutoff, "w1")

    assert queue.recover_claims("w1", 3600) == ["j"]
    fresh = queue.claim("w1")
    assert queue._recover(found_by_the_second, cutoff) == []

    assert fresh.path.exists()
    assert queue.pending_jobs() == []


def test_a_claim_touched_after_it_was_found_stale_is_left_alone(queue, monkeypatch) -> None:
    queue.enqueue(make_job("j"))
    claimed = queue.claim("w1")
    now = _later(monkeypatch)
    found = queue._untouched(now - 3600)
    os.utime(claimed.path, (now, now))  # its worker's heartbeat came back
    cutoff = now - 3600

    assert queue._recover(found, cutoff) == []
    assert claimed.path.exists() and queue.pending_jobs() == []


def test_a_claim_a_dead_recovery_took_is_recovered_once_that_is_stale(queue, monkeypatch) -> None:
    queue.enqueue(make_job("j"))
    claimed = queue.claim("w1")
    now = _later(monkeypatch)
    assert queue._take(claimed.path, now - 3600) is not None  # then that recoverer died

    assert queue.recover_claims("w1", 3600) == []  # it may still be requeueing it
    _later(monkeypatch, 2 * 7200)
    assert queue.recover_claims("w1", 3600) == ["j"]
    assert queue.claimed_jobs() == []


def test_a_claim_is_not_stale_before_its_record_is_rewritten(queue, monkeypatch) -> None:
    """The rename keeps the pending file's mtime, which is old for a job that waited.
    Until the claim's record is written, only its name says it is fresh."""
    queue.enqueue(make_job("j"))
    os.utime(queue._store.pending / "j.json", (0, 0))  # it waited a long time
    write = queue_module.write_json_atomic
    looks: list[list[str]] = []

    def another_worker_of_the_slot_looks_first(path, document):
        if path.parent == queue._store.claimed and not looks:
            looks.append(queue.recover_claims("w1", 3600))
        write(path, document)

    monkeypatch.setattr(queue_module, "write_json_atomic", another_worker_of_the_slot_looks_first)
    claimed = queue.claim("w1")

    assert looks == [[]]
    assert claimed.path.exists() and queue.pending_jobs() == []


def test_a_claim_named_before_the_stamps_is_still_recovered(queue) -> None:
    """Claims left by the version that named them ``<job>.<worker>.json``."""
    job = make_job("j").model_copy(update={"attempts": 1, "claimed_at": utcnow()})
    for name in ("j.w1.json", "k.w10.json"):
        path = queue._store.claimed / name
        path.write_text(job.model_copy(update={"job_id": name.split(".")[0]}).model_dump_json())
        os.utime(path, (0, 0))

    assert queue.recover_claims("w1", 3600) == ["j"]
    assert queue.recover_claims("w10", 3600) == ["k"]
    assert sorted(job.job_id for job in queue.pending_jobs()) == ["j", "k"]
    assert queue.claimed_jobs() == []


def test_recovering_a_claim_whose_result_is_out_drops_its_image(queue, monkeypatch) -> None:
    """The worker published and died before it dropped the image and moved the record."""
    image = queue._store.incoming / "t" / "abc123" / "agent.sif"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"sif")
    queue.enqueue(make_job("j").model_copy(update={"image_path": str(image)}))
    claimed = queue.claim("w1")
    _publish(queue, "j", RunStatus.COMPLETED, claimed.job.claimed_at + timedelta(hours=1))
    _later(monkeypatch)

    assert queue.recover_claims("w1", 3600) == []
    assert not image.parent.exists()
    assert (queue._store.done / "j.json").exists()
