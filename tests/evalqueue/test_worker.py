"""The worker's bookkeeping: every claim ends as a published result and a moved record."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from capx.evalqueue import runner as runner_module
from capx.evalqueue import weights
from capx.evalqueue import worker as worker_module
from capx.evalqueue.config import QueueConfig
from capx.evalqueue.models import ErrorCode, QueueJob, RunResult, RunStatus, WeightsMount
from capx.evalqueue.queue import FileQueue
from capx.evalqueue.runner import EvalRunner, Interrupted, LeaseLost
from capx.evalqueue.store import EvalStore, utcnow
from capx.evalqueue.worker import EvalWorker


class StubRunner:
    def __init__(
        self,
        status: RunStatus = RunStatus.COMPLETED,
        *,
        crash: bool = False,
        error_code: ErrorCode | None = None,
    ) -> None:
        self._status = status
        self._crash = crash
        self._error_code = error_code
        self.runs: list[str] = []
        self.interrupted = False
        self.during_run = lambda: None

    def interrupt(self) -> None:
        self.interrupted = True

    def release_remote_gpus(self) -> None:
        pass

    def run(self, job: QueueJob, run_id: str) -> RunResult:
        self.runs.append(run_id)
        self.during_run()
        if self.interrupted:
            raise Interrupted()
        if self._crash:
            raise RuntimeError("simulator exploded")
        return RunResult(
            run_id=run_id,
            job_id=job.job_id,
            team_id=job.team_id,
            submission_id=job.submission_id,
            status=self._status,
            error_code=self._error_code,
        )


def build(tmp_path, runner, *, image_path=None, weights=(), **settings):
    config = QueueConfig(root=str(tmp_path), master_seed=1, **settings)
    config.watcher.poll_interval_s = 0.01
    store = EvalStore(tmp_path)
    store.ensure_layout()
    queue = FileQueue(store)
    queue.enqueue(
        QueueJob(
            job_id="alpha__s1",
            team_id="alpha",
            submission_id="s1",
            image_path=str(image_path or tmp_path / "agent.sif"),
            weights=list(weights),
            enqueued_at=utcnow(),
        )
    )
    return store, queue, EvalWorker(config, store, queue, runner, worker_id="w1")


def _fetched(store: EvalStore, key: str = "abc123") -> Path:
    """An image where the watcher puts what it fetched."""
    image = store.incoming / "alpha" / key / "agent.sif"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"sif")
    (image.parent / "submission.yaml").write_text("team_id: alpha\n")
    return image


def test_a_completed_run_is_published_and_the_record_moves_to_done(tmp_path) -> None:
    store, queue, worker = build(tmp_path, StubRunner())
    assert worker.run_forever(max_jobs=1) == 1

    published = list(store.results.glob("alpha__s1__*.json"))
    assert len(published) == 1
    assert json.loads(published[0].read_text())["status"] == "completed"
    assert (store.done / "alpha__s1.json").exists()
    assert not queue.is_queued("alpha__s1")


def test_a_failed_run_still_publishes_and_lands_in_failed(tmp_path) -> None:
    store, _, worker = build(tmp_path, StubRunner(RunStatus.FAILED))
    worker.run_forever(max_jobs=1)

    assert json.loads(next(store.results.glob("*.json")).read_text())["status"] == "failed"
    assert (store.failed / "alpha__s1.json").exists()


def test_a_crashing_runner_becomes_the_submissions_result_not_a_dead_worker(tmp_path) -> None:
    store, _, worker = build(tmp_path, StubRunner(crash=True))
    assert worker.run_forever(max_jobs=1) == 1

    document = json.loads(next(store.results.glob("*.json")).read_text())
    assert document["status"] == "failed"
    assert "simulator exploded" in document["error"]


def test_a_job_left_claimed_by_a_dead_worker_is_retried_on_restart(tmp_path, monkeypatch) -> None:
    store, queue, worker = build(tmp_path, StubRunner())
    queue.claim("w1")  # the previous incarnation of this worker died holding it
    later = time.time() + worker_module.STALE_CLAIM_S + 1
    monkeypatch.setattr(time, "time", lambda: later)

    assert worker.run_forever(max_jobs=1) == 1
    assert (store.done / "alpha__s1.json").exists()


def test_a_live_claim_of_the_same_slot_is_left_alone(tmp_path, monkeypatch) -> None:
    """Slurm can run two jobs for one slot (a failed squeue). The second must not take
    back the job the first is still running, or the job runs twice."""
    runner = StubRunner()
    store, queue, worker = build(tmp_path, runner)
    claim = queue.claim("w1")  # the other job of this slot is running it

    looks = []

    def one_look(_seconds: float) -> None:
        looks.append(1)
        worker._stopping = len(looks) >= 2

    monkeypatch.setattr(time, "sleep", one_look)
    assert worker.run_forever() == 0
    assert runner.runs == []
    assert claim.path.exists() and queue.pending_jobs() == []


def test_a_stopped_worker_puts_the_job_back_and_publishes_nothing(tmp_path) -> None:
    """Slurm's SIGTERM reaches the submission too, so the run says nothing about it."""
    runner = StubRunner()
    store, queue, worker = build(tmp_path, runner, image_path=_fetched(EvalStore(tmp_path)))
    runner.during_run = worker.stop

    assert worker.run_forever() == 0
    assert runner.interrupted
    assert list(store.results.glob("*.json")) == []
    assert not any(store.claimed.glob("*.json"))
    (job,) = queue.pending_jobs()
    assert job.job_id == "alpha__s1" and job.attempts == 0 and job.claimed_at is None
    assert (store.incoming / "alpha" / "abc123" / "agent.sif").exists()


def test_a_job_whose_runs_keep_losing_their_worker_fails_as_ours(tmp_path) -> None:
    runner = StubRunner()
    store, queue, worker = build(tmp_path, runner)
    for _ in range(worker_module.MAX_ATTEMPTS):
        queue.requeue(queue.claim("w0"))  # each run ended with its worker killed

    assert worker.run_forever(max_jobs=1) == 1
    assert runner.runs == []
    document = json.loads(next(store.results.glob("*.json")).read_text())
    assert document["status"] == "failed" and document["error_code"] == "internal_error"
    assert (store.failed / "alpha__s1.json").exists()


class LeaseLosingRunner(StubRunner):
    def run(self, job: QueueJob, run_id: str) -> RunResult:
        self.runs.append(run_id)
        assert len(self.runs) <= worker_module.MAX_ATTEMPTS, "the lost runs were not counted"
        self.during_run()
        raise LeaseLost("lost the lease on simbox GPU 0 (ssh exit 255)")


def test_a_run_that_loses_its_lease_goes_back_and_counts_as_an_attempt(tmp_path) -> None:
    """Another evaluation may have used the GPU meanwhile, so nothing of the run is
    published. Counting it keeps a lease that is lost every time from looping forever."""
    runner = LeaseLosingRunner()
    store, queue, worker = build(tmp_path, runner)

    assert worker.run_forever(max_jobs=1) == 1

    assert len(runner.runs) == worker_module.MAX_ATTEMPTS
    (published,) = store.results.glob("*.json")
    document = json.loads(published.read_text())
    assert document["error_code"] == "internal_error" and "gave up" in document["error"]


def test_a_lease_that_ends_with_the_worker_s_own_stop_is_not_counted(tmp_path) -> None:
    """Slurm's SIGTERM reaches the lease's ssh too, so a stop loses the lease as well."""
    runner = LeaseLosingRunner()
    store, queue, worker = build(tmp_path, runner)
    runner.during_run = worker.stop

    assert worker.run_forever() == 0

    (job,) = queue.pending_jobs()
    assert job.attempts == 0
    assert list(store.results.glob("*.json")) == []


def test_stopping_prevents_further_claims(tmp_path) -> None:
    runner = StubRunner()
    _, _, worker = build(tmp_path, runner)
    worker._stopping = True
    assert worker.run_forever() == 0
    assert runner.runs == []


REMOTE = {
    "simulators": {"behavior": {"remote": {"host": "simbox"}, "max_lanes": 8}},
    "subsets": [{"name": "s", "taskpack": "taskpacks/original", "simulators": ["behavior"]}],
    "slurm": {"workers": 1},
}


class RemoteGpuRunner(StubRunner):
    """Finds the remote GPU's lease taken for its first ``busy_for`` tries."""

    def __init__(self, busy_for: int) -> None:
        super().__init__()
        self.tries = 0
        self.held = False
        self.held_during_run: bool | None = None
        self._busy_for = busy_for

    def take_remote_gpus(self) -> str | None:
        self.tries += 1
        if self.tries <= self._busy_for:
            return "simbox GPU 0 is in use by another evaluation"
        self.held = True
        return None

    def release_remote_gpus(self) -> None:
        self.held = False

    def run(self, job: QueueJob, run_id: str) -> RunResult:
        self.held_during_run = self.held
        return super().run(job, run_id)


@pytest.mark.parametrize(("wait_warning_s", "warnings"), [(600.0, 1), (0.0, 3)])
def test_the_worker_claims_nothing_until_it_holds_the_remote_gpu(
    tmp_path, monkeypatch, caplog, wait_warning_s, warnings
) -> None:
    """It holds the lease through the job and lets go once the result is out. The
    warning repeats every WAIT_WARNING_S, not every look."""
    caplog.set_level(logging.INFO, logger=worker_module.__name__)
    monkeypatch.setattr(worker_module, "WAIT_WARNING_S", wait_warning_s)
    runner = RemoteGpuRunner(busy_for=3)
    store, queue, worker = build(tmp_path, runner, **REMOTE)
    waiting: list[int] = []
    monkeypatch.setattr(time, "sleep", lambda _s: waiting.append(len(queue.pending_jobs())))

    assert worker.run_forever(max_jobs=1) == 1

    assert waiting == [1, 1, 1]
    assert runner.tries == 4 and runner.held_during_run and not runner.held
    assert (store.done / "alpha__s1.json").exists()
    messages = [record.getMessage() for record in caplog.records]
    waits = [m for m in messages if m.startswith("not claiming jobs until the remote GPU is free")]
    assert len(waits) == warnings and "simbox GPU 0 is in use" in waits[0]
    assert "the remote GPU is free again" in messages


def test_an_idle_worker_does_not_take_the_lease(tmp_path, monkeypatch) -> None:
    runner = RemoteGpuRunner(busy_for=0)
    _, queue, worker = build(tmp_path, runner, **REMOTE)
    queue.withdraw("alpha__s1")
    monkeypatch.setattr(time, "sleep", lambda _s: worker.stop())

    assert worker.run_forever() == 0
    assert runner.tries == 0


def _worker_sleeps(monkeypatch, sleep) -> None:
    """Replace the worker's own sleep only: subprocess sleeps too, while it waits on a
    lease's ssh."""
    monkeypatch.setattr(worker_module, "time", SimpleNamespace(sleep=sleep, monotonic=time.monotonic))


@pytest.fixture
def leased_queue(tmp_path, monkeypatch):
    """Queues whose remote GPU leases are real lock files under ``tmp_path/locks``,
    taken by a local shell instead of over ssh. Their runners' runs are stubbed."""
    monkeypatch.setattr(runner_module, "_ssh_argv", lambda host, command: ["sh", "-c", command])
    runners: list[EvalRunner] = []

    def make(name: str, *, gpu: str = "0", jobs: int = 1):
        remote = {"host": "simbox", "socket_dir": str(tmp_path / "locks"), "gpu": gpu}
        config = QueueConfig(
            root=str(tmp_path / name),
            master_seed=1,
            **{**REMOTE, "simulators": {"behavior": {"remote": remote, "max_lanes": 8}}},
        )
        config.watcher.poll_interval_s = 0.01
        store = EvalStore(config.root)
        store.ensure_layout()
        queue = FileQueue(store)
        for team in ("alpha", "beta")[:jobs]:
            queue.enqueue(
                QueueJob(
                    job_id=f"{team}__s1",
                    team_id=team,
                    submission_id="s1",
                    image_path=str(tmp_path / "agent.sif"),
                    enqueued_at=utcnow(),
                )
            )

        def worker(worker_id: str, during_run=lambda: None) -> EvalWorker:
            runner = EvalRunner(config, store)
            runners.append(runner)

            def run(job: QueueJob, run_id: str) -> RunResult:
                during_run()
                return StubRunner().run(job, run_id)

            runner.run = run
            return EvalWorker(config, store, queue, runner, worker_id=worker_id)

        return queue, worker

    yield make
    for runner in runners:
        runner.release_remote_gpus()


def _no_wait(_seconds: float) -> None:
    raise AssertionError("waited for a GPU that is free")


@pytest.mark.parametrize("same_queue", [True, False], ids=["same-queue", "other-queue"])
def test_a_second_worker_waits_while_the_first_holds_the_remote_gpu(
    leased_queue, monkeypatch, same_queue
) -> None:
    """A duplicate Slurm job of the same slot, or the other round's queue. It claims
    nothing, so it cannot clear the first one's live containers, until the first has
    published its result."""
    queue, workers = leased_queue("q", jobs=2)
    second_queue, second_workers = (queue, workers) if same_queue else leased_queue("other")
    second = second_workers("w2")
    _worker_sleeps(monkeypatch, lambda _s: second.stop())
    claimed_meanwhile: list[int] = []
    first = workers("w1", during_run=lambda: claimed_meanwhile.append(second.run_forever()))

    assert first.run_forever(max_jobs=1) == 1

    assert claimed_meanwhile == [0]
    assert len(second_queue.pending_jobs()) == 1
    _worker_sleeps(monkeypatch, _no_wait)
    assert second_workers("w3").run_forever(max_jobs=1) == 1


def test_queues_on_two_gpus_of_one_host_do_not_wait_for_each_other(
    leased_queue, monkeypatch
) -> None:
    _worker_sleeps(monkeypatch, _no_wait)
    _, workers = leased_queue("q", gpu="0")
    _, other_workers = leased_queue("other", gpu="1")
    ran_meanwhile: list[int] = []
    first = workers(
        "w1", during_run=lambda: ran_meanwhile.append(other_workers("w2").run_forever(max_jobs=1))
    )

    assert first.run_forever(max_jobs=1) == 1
    assert ran_meanwhile == [1]


def test_a_worker_waiting_for_the_lease_still_stops_when_told(
    leased_queue, tmp_path, monkeypatch
) -> None:
    """What the SIGTERM handler does: the job it never claimed stays as it was."""
    queue, workers = leased_queue("q")
    holder = runner_module.RemoteGpuLease(("simbox", str(tmp_path / "locks"), "0"), lambda why: None)
    assert holder.take() is None
    worker = workers("w1")
    _worker_sleeps(monkeypatch, lambda _s: worker.stop())
    try:
        assert worker.run_forever() == 0
    finally:
        holder.release()
    (job,) = queue.pending_jobs()
    assert job.attempts == 0


def test_weights_no_queued_job_needs_are_dropped_once_a_job_finishes(tmp_path) -> None:
    store, _, worker = build(tmp_path, StubRunner())
    leftover = store.weights_dir(WeightsMount(repo="org/m", revision="a" * 40, path="/models/m"))
    leftover.mkdir(parents=True)
    past = leftover.stat().st_mtime - weights.GRACE_S - 60
    os.utime(leftover, (past, past))

    worker.run_forever(max_jobs=1)
    assert not leftover.exists()


GATED = WeightsMount(repo="org/gated", revision="a" * 40, path="/models/g", gated=True)


def test_a_job_waits_for_weights_still_being_granted_then_runs(tmp_path, monkeypatch) -> None:
    """It keeps its claim, so the jobs behind it wait too: none overtakes it."""
    runner = StubRunner()
    store, queue, worker = build(tmp_path, runner, weights=[GATED])
    queue.enqueue(
        QueueJob(
            job_id="beta__s1",
            team_id="beta",
            submission_id="s1",
            image_path=str(tmp_path / "agent.sif"),
            enqueued_at=utcnow(),
        )
    )
    waits: list[int] = []

    def watcher_fetches_on_the_third_look(_seconds: float) -> None:
        waits.append(1)
        assert runner.runs == [] and [job.job_id for job in queue.pending_jobs()] == ["beta__s1"]
        if len(waits) == 3:
            store.weights_dir(GATED).mkdir(parents=True)

    monkeypatch.setattr(time, "sleep", watcher_fetches_on_the_third_look)
    assert worker.run_forever(max_jobs=1) == 1

    assert len(waits) == 3
    assert runner.runs and runner.runs[0].startswith("alpha__s1")
    assert (store.done / "alpha__s1.json").exists()


def test_a_worker_stopped_while_waiting_for_weights_puts_the_job_back(
    tmp_path, monkeypatch
) -> None:
    runner = StubRunner()
    store, queue, worker = build(tmp_path, runner, weights=[GATED])
    monkeypatch.setattr(time, "sleep", lambda _seconds: worker.stop())

    assert worker.run_forever() == 0

    assert runner.runs == []
    assert list(store.results.glob("*.json")) == []
    (job,) = queue.pending_jobs()
    assert job.job_id == "alpha__s1" and job.attempts == 0


def test_a_running_job_keeps_its_claim_fresh(tmp_path, monkeypatch) -> None:
    """requeue-stale goes by when a claim was last touched, so a long run must keep
    touching it or it would be taken from a worker that is still going."""
    from capx.evalqueue import worker as worker_module

    monkeypatch.setattr(worker_module, "HEARTBEAT_S", 0.01)
    store = EvalStore(tmp_path)
    ages: list[float] = []

    class SlowRunner(StubRunner):
        def run(self, job: QueueJob, run_id: str) -> RunResult:
            (claim,) = store.claimed.glob("*.json")
            os.utime(claim, (0, 0))  # as if claimed long ago
            time.sleep(0.2)
            ages.append(time.time() - claim.stat().st_mtime)
            return super().run(job, run_id)

    _, _, worker = build(tmp_path, SlowRunner())
    worker.run_forever(max_jobs=1)
    assert ages and ages[0] < 60


def test_from_the_deadline_the_worker_claims_nothing_and_cancels_what_waits(tmp_path, monkeypatch) -> None:
    runner = StubRunner()
    store, queue, worker = build(tmp_path, runner)
    worker._config.deadline = utcnow() - timedelta(seconds=1)

    def stop_after_one_look(_seconds: float) -> None:
        worker._stopping = True

    monkeypatch.setattr(time, "sleep", stop_after_one_look)
    assert worker.run_forever() == 0
    assert runner.runs == []
    assert queue.pending_jobs() == []
    assert json.loads(next(store.results.glob("*.json")).read_text())["status"] == "cancelled"


def test_the_fetched_image_is_deleted_once_the_result_is_out(tmp_path) -> None:
    store = EvalStore(tmp_path)
    store.ensure_layout()
    image = _fetched(store)
    _, _, worker = build(tmp_path, StubRunner(RunStatus.FAILED), image_path=image)

    worker.run_forever(max_jobs=1)
    assert not image.parent.exists()
    assert (store.incoming / "alpha").is_dir()


@pytest.mark.parametrize("code", [ErrorCode.INTERNAL_ERROR, ErrorCode.NO_SCORE])
def test_the_image_is_kept_when_our_side_failed(tmp_path, code) -> None:
    store = EvalStore(tmp_path)
    store.ensure_layout()
    image = _fetched(store)
    _, _, worker = build(tmp_path, StubRunner(RunStatus.FAILED, error_code=code), image_path=image)

    worker.run_forever(max_jobs=1)
    assert image.exists()


def test_an_image_another_queued_job_uses_is_kept(tmp_path) -> None:
    store = EvalStore(tmp_path)
    store.ensure_layout()
    image = _fetched(store)
    _, queue, worker = build(tmp_path, StubRunner(), image_path=image)
    queue.enqueue(
        QueueJob(
            job_id="alpha__by-hand",
            team_id="alpha",
            submission_id="by-hand",
            image_path=str(image),
            enqueued_at=utcnow(),
        )
    )

    worker.run_forever(max_jobs=1)
    assert image.exists()


def test_an_image_queued_by_hand_from_elsewhere_is_left_alone(tmp_path) -> None:
    image = tmp_path / "mine" / "agent.sif"
    image.parent.mkdir()
    image.write_bytes(b"sif")
    _, _, worker = build(tmp_path, StubRunner(), image_path=image)

    worker.run_forever(max_jobs=1)
    assert image.exists()


def test_a_teams_next_run_removes_the_directories_of_its_earlier_ones(tmp_path) -> None:
    """Only the team's runs whose results are out: another team's run, and one with no
    result yet, are left."""
    runner = StubRunner()
    store, _, worker = build(tmp_path, runner)
    for run_id, team_id in (("alpha__s0__1", "alpha"), ("beta__s0__1", "beta")):
        (store.run_dir(run_id) / "public").mkdir(parents=True)
        store.publish_result(
            RunResult(run_id=run_id, team_id=team_id, submission_id="s0", status=RunStatus.COMPLETED)
        )
    (store.run_dir("alpha__s9__1") / "public").mkdir(parents=True)
    seen: list[set[str]] = []
    runner.during_run = lambda: seen.append({path.name for path in store.runs.iterdir()})

    worker.run_forever(max_jobs=1)

    assert seen == [{"beta__s0__1", "alpha__s9__1"}]
    assert (store.results / "alpha__s0__1.json").exists()


def test_removing_earlier_runs_happens_under_the_heartbeat_and_cannot_fail_the_job(
    tmp_path, monkeypatch
) -> None:
    """Deleting a large tree can outlast a stale claim, so the claim is kept fresh
    meanwhile; and whatever goes wrong, the job runs and its result is published."""
    heartbeat_running: list[bool] = []

    def prune_runs(store, team_id):
        heartbeat_running.append(
            any(thread.name == "heartbeat" and thread.is_alive() for thread in threading.enumerate())
        )
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(worker_module, "prune_runs", prune_runs)
    runner = StubRunner()
    store, _, worker = build(tmp_path, runner)

    assert worker.run_forever(max_jobs=1) == 1

    assert heartbeat_running == [True]
    assert len(runner.runs) == 1 and (store.done / "alpha__s1.json").exists()


def test_a_long_job_does_not_hold_up_a_dead_predecessors_claim(tmp_path, monkeypatch) -> None:
    """The successor started before its predecessor's claim went stale, then claimed a
    long job. The orphan is put back while that job runs, not hours later."""
    monkeypatch.setattr(worker_module, "HEARTBEAT_S", 0.01)
    store = EvalStore(tmp_path)
    store.ensure_layout()
    orphan_job = QueueJob(
        job_id="beta__s1", team_id="beta", submission_id="s1", image_path="/x/agent.sif",
        enqueued_at=utcnow(), claimed_at=utcnow(), attempts=1,
    )
    orphan = store.claimed / "beta__s1.w1.1000000000-0badc0de.json"
    orphan.write_text(orphan_job.model_dump_json())  # touched just before its worker died
    put_back: list[bool] = []

    class LongRunner(StubRunner):
        def run(self, job: QueueJob, run_id: str) -> RunResult:
            os.utime(orphan, (0, 0))  # now it is stale
            deadline = time.monotonic() + 5
            while not queue.pending_jobs() and time.monotonic() < deadline:
                threading.Event().wait(0.01)
            put_back.append([j.job_id for j in queue.pending_jobs()] == ["beta__s1"])
            return super().run(job, run_id)

    _, queue, worker = build(tmp_path, LongRunner())
    assert worker.run_forever(max_jobs=1) == 1
    assert put_back == [True]
