"""Declared weights: what a manifest may ask for, what gets fetched, and what gets dropped."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from capx.evalqueue import hubsource, weights as weights_module
from capx.evalqueue.config import HubConfig, QueueConfig
from capx.evalqueue.hubsource import HubWeights
from capx.evalqueue.models import QueueJob, Rejected, RunSpec, WeightsMount
from capx.evalqueue.queue import FileQueue
from capx.evalqueue.store import EvalStore, utcnow
from capx.evalqueue import watcher as watcher_module
from capx.evalqueue.watcher import SubmissionWatcher
from capx.evalqueue.weights import RepoInfo, WeightsCache, prune_weights

from .test_watcher import FakeFetcher, FakeSource, only_result

SHA_A = "a" * 40
SHA_B = "b" * 40


def mount(repo: str = "org/model", revision: str = SHA_A, path: str = "/models/m", **extra):
    return WeightsMount(repo=repo, revision=revision, path=path, **extra)


@pytest.fixture
def store(tmp_path) -> EvalStore:
    store = EvalStore(tmp_path)
    store.ensure_layout()
    return store


def age(path: Path) -> None:
    past = time.time() - weights_module.GRACE_S - 60
    os.utime(path, (past, past))


def job_with(job_id: str, *mounts: WeightsMount) -> QueueJob:
    return QueueJob(
        job_id=job_id,
        team_id=job_id.split("__")[0],
        submission_id="s1",
        image_path="/x/agent.sif",
        weights=list(mounts),
        enqueued_at=utcnow(),
    )


# --- what a manifest may declare ---------------------------------------------


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"revision": "main"}, "full 40-character commit sha"),
        ({"path": "/opt/m"}, "must be /models/"),
        ({"path": "/models"}, "must be /models/"),
        ({"path": "/models/m/"}, "must be /models/"),
        ({"path": "/models/m:rw"}, "must be /models/"),
        ({"path": "/models/../usr"}, "must be /models/"),
        ({"repo": "model"}, "<owner>/<name>"),
        ({"repo": "org/a--b"}, "<owner>/<name>"),
    ],
)
def test_a_weights_entry_that_cannot_be_mounted_safely_is_refused(fields, message) -> None:
    with pytest.raises(ValueError, match=message):
        mount(**fields)


def test_overlapping_weights_paths_are_refused() -> None:
    with pytest.raises(ValueError, match="overlap"):
        RunSpec(
            team_id="a",
            submission_id="s",
            weights=[mount(path="/models/m"), mount(repo="org/other", path="/models/m/sub")],
        )


def test_a_sibling_path_sharing_a_prefix_is_not_an_overlap() -> None:
    RunSpec(
        team_id="a",
        submission_id="s",
        weights=[mount(path="/models/m"), mount(repo="org/other", path="/models/m-2")],
    )


# --- check and fetch ----------------------------------------------------------


def test_a_gated_repo_must_be_declared_gated(store) -> None:
    cache = WeightsCache(store, FakeFetcher(gated={"org/model"}), max_gb=0)
    with pytest.raises(Rejected, match="declare `gated: true`"):
        cache.check("alpha", [mount()])
    cache.check("alpha", [mount(gated=True)])


def test_check_returns_the_gated_repos_that_cannot_be_downloaded_yet(store) -> None:
    fetcher = FakeFetcher(gated={"org/model", "org/open"}, ungranted={"org/model"})
    cache = WeightsCache(store, fetcher, max_gb=0)
    waiting = mount(gated=True)
    granted = mount(repo="org/open", path="/models/o", gated=True)
    assert cache.check("alpha", [waiting, granted]) == [waiting]


def test_fetch_missing_fetches_only_what_can_be_downloaded(store) -> None:
    fetcher = FakeFetcher(ungranted={"org/model"})
    cache = WeightsCache(store, fetcher, max_gb=0)
    waiting, ready = mount(gated=True), mount(repo="org/open", path="/models/o")

    assert cache.fetch_missing([waiting, ready]) == [waiting]
    assert fetcher.downloaded == ["org/open"]

    fetcher.ungranted.clear()
    assert cache.fetch_missing([waiting, ready]) == []
    assert fetcher.downloaded == ["org/open", "org/model"]


def test_declared_weights_over_the_cap_are_refused(store) -> None:
    cache = WeightsCache(store, FakeFetcher(sizes={"org/model": 2 * 1024**3}), max_gb=1)
    with pytest.raises(Rejected, match="over the 1 GiB limit"):
        cache.check("alpha", [mount()])


def test_a_team_may_declare_its_own_repos_but_not_another_teams(store) -> None:
    fetcher = FakeFetcher()
    cache = WeightsCache(
        store, fetcher, max_gb=0, team_orgs=lambda: {"Alpha-Org": "alpha", "beta-org": "beta"}
    )
    cache.check("alpha", [mount(repo="alpha-org/finetuned"), mount(repo="Qwen/x", path="/models/q")])
    with pytest.raises(Rejected, match="belongs to team 'beta'"):
        cache.check("alpha", [mount(repo="beta-org/finetuned")])
    assert "beta-org/finetuned" not in fetcher.described


def test_a_private_repo_is_allowed_only_from_the_team_s_own_org(store) -> None:
    """§6.3: weights from outside the team's org must be public. The organizer's token
    can read more than the registered orgs."""
    fetcher = FakeFetcher(private={"alpha-org/finetuned", "gamma-org/finetuned"})
    cache = WeightsCache(store, fetcher, max_gb=0, team_orgs=lambda: {"Alpha-Org": "alpha"})
    cache.check("alpha", [mount(repo="alpha-org/finetuned"), mount(repo="Qwen/x", path="/models/q")])
    # A team whose registration is not merged yet: its org is no one's as far as we know.
    with pytest.raises(Rejected, match="is private and not in your team's org"):
        cache.check("alpha", [mount(repo="gamma-org/finetuned")])
    with pytest.raises(Rejected, match="is private and not in your team's org"):
        cache.check("gamma", [mount(repo="gamma-org/finetuned")])


def test_a_fetched_repo_is_reused_not_downloaded_again(store) -> None:
    fetcher = FakeFetcher()
    cache = WeightsCache(store, fetcher, max_gb=0)
    cache.fetch([mount()])
    cache.fetch([mount(path="/models/elsewhere")])

    assert fetcher.downloaded == ["org/model"]
    copy = store.weights_dir(mount())
    assert copy == store.weights / "org--model" / SHA_A
    assert (copy / "config.json").is_file()
    assert not list(copy.parent.glob(".*partial"))


def test_a_fetch_stages_under_its_own_name(store) -> None:
    """Another process fetching the same commit has its own staging directory, which
    this fetch must neither delete nor publish."""
    theirs = store.weights_dir(mount()).with_name(f".{SHA_A}.partial.otherhost.1")
    theirs.mkdir(parents=True)
    (theirs / "half-written").write_text("")
    WeightsCache(store, FakeFetcher(), max_gb=0).fetch([mount()])
    assert (theirs / "half-written").exists()
    assert (store.weights_dir(mount()) / "config.json").is_file()
    assert not (store.weights_dir(mount()) / "half-written").exists()


def test_every_mount_stays_fresh_while_a_later_one_downloads(store, monkeypatch) -> None:
    """The job that names them is enqueued only after the last download, which can take
    longer than the grace period."""
    monkeypatch.setattr(weights_module, "_REFRESH_S", 0.01)
    queue = FileQueue(store)
    first, second = mount(repo="org/small"), mount(repo="org/large", path="/models/l")
    pruned: list[Path] = []

    class SlowFetcher(FakeFetcher):
        def download(self, each: WeightsMount, target: Path) -> None:
            super().download(each, target)
            if each == second:
                age(store.weights_dir(first))  # as if the transfer took an hour
                time.sleep(0.2)
                pruned.extend(prune_weights(store, queue))

    WeightsCache(store, SlowFetcher(), max_gb=0).fetch([first, second])
    assert pruned == []
    assert store.weights_dir(first).is_dir()


# --- pruning ------------------------------------------------------------------


def test_prune_keeps_what_a_queued_or_running_job_declares(store) -> None:
    queue = FileQueue(store)
    cache = WeightsCache(store, FakeFetcher(), max_gb=0)
    pending, running, finished = (
        mount(revision=SHA_A),
        mount(revision=SHA_B),
        mount(repo="org/old"),
    )
    cache.fetch([pending, running, finished])
    for each in (pending, running, finished):
        age(store.weights_dir(each))
    queue.enqueue(job_with("alpha__s1", running))
    assert queue.claim("w1") is not None
    queue.enqueue(job_with("beta__s1", pending))

    removed = prune_weights(store, queue)

    assert removed == [store.weights_dir(finished)]
    assert store.weights_dir(pending).is_dir()
    assert store.weights_dir(running).is_dir()
    assert not (store.weights / "org--old").exists()


def test_prune_spares_a_copy_fetched_moments_ago(store) -> None:
    """Fetching and enqueueing happen in another process from pruning; the job that
    declares a fresh copy may not be in the queue yet."""
    WeightsCache(store, FakeFetcher(), max_gb=0).fetch([mount()])
    assert prune_weights(store, FileQueue(store)) == []
    assert store.weights_dir(mount()).is_dir()


def test_prune_removes_nothing_while_the_queue_will_not_hold_still(store, monkeypatch) -> None:
    queue = FileQueue(store)
    WeightsCache(store, FakeFetcher(), max_gb=0).fetch([mount()])
    age(store.weights_dir(mount()))
    reads = iter(range(1000))
    monkeypatch.setattr(queue, "pending_jobs", lambda: [job_with(f"alpha__{next(reads)}")])

    assert prune_weights(store, queue) == []
    assert store.weights_dir(mount()).is_dir()


def test_prune_drops_staging_left_by_a_fetch_that_died(store) -> None:
    parent = store.weights / "org--model"
    dead, live = parent / f".{SHA_A}.partial.host.1", parent / f".{SHA_B}.partial.host.2"
    for each in (dead, live):
        each.mkdir(parents=True)
    age(dead)
    assert prune_weights(store, FileQueue(store)) == [dead]
    assert live.is_dir()


def test_prune_leaves_anything_it_did_not_lay_out(store) -> None:
    legacy = store.weights / "hub" / "models--facebook--sam3"
    legacy.mkdir(parents=True)
    age(legacy)
    assert prune_weights(store, FileQueue(store)) == []
    assert legacy.is_dir()


# --- the watcher --------------------------------------------------------------


@dataclass
class RecordingNotifier:
    sent: list[str] = field(default_factory=list)

    def notify(self, text: str) -> None:
        self.sent.append(text)


@pytest.fixture
def notifier() -> RecordingNotifier:
    return RecordingNotifier()


@pytest.fixture
def watched(store, notifier):
    config = QueueConfig(root=str(store.root), master_seed=1)
    queue = FileQueue(store)
    source = FakeSource(store)
    fetcher = FakeFetcher()
    watcher = SubmissionWatcher(
        config, store, queue, source, WeightsCache(store, fetcher, max_gb=1), notifier=notifier
    )
    return store, queue, source, fetcher, watcher


def test_declared_weights_are_fetched_and_carried_by_the_job(watched) -> None:
    store, queue, source, fetcher, watcher = watched
    source.offer(weights=[mount()])
    assert watcher.poll() == ["alpha__s1"]

    assert fetcher.downloaded == ["org/model"]
    assert queue.pending_jobs()[0].weights == [mount()]
    assert store.weights_dir(mount()).is_dir()


def test_weights_refused_from_metadata_cost_no_image_transfer(watched) -> None:
    store, _, source, fetcher, watcher = watched
    fetcher.gated.add("org/model")
    source.offer(weights=[mount()])
    assert watcher.poll() == []

    assert "declare `gated: true`" in only_result(store)["error"]
    assert source.fetched == []
    assert fetcher.downloaded == []


def test_a_gated_repo_already_granted_is_fetched_without_a_notice(watched, notifier) -> None:
    store, _, source, fetcher, watcher = watched
    fetcher.gated.add("org/model")
    source.offer(weights=[mount(gated=True)])
    assert watcher.poll() == ["alpha__s1"]
    assert store.weights_dir(mount()).is_dir()
    assert notifier.sent == []


def test_a_gated_repo_without_access_is_queued_in_order_and_announced(watched, notifier) -> None:
    """Waiting for a person to accept the terms must not cost the team its place."""
    store, queue, source, fetcher, watcher = watched
    fetcher.gated.add("org/model")
    fetcher.ungranted.add("org/model")
    source.offer(weights=[mount(gated=True)])
    source.offer(team_id="beta", key="s1")

    assert watcher.poll() == ["alpha__s1", "beta__s1"]

    assert [job.job_id for job in queue.pending_jobs()] == ["alpha__s1", "beta__s1"]
    assert not store.weights_dir(mount()).exists()
    assert source.fetched  # the image is fetched as usual
    (notice,) = notifier.sent
    assert "alpha__s1" in notice and "org/model" in notice
    assert list(store.results.glob("*.json")) == []


def test_weights_granted_later_are_fetched_on_a_later_poll(watched, notifier) -> None:
    store, _, source, fetcher, watcher = watched
    fetcher.gated.add("org/model")
    fetcher.ungranted.add("org/model")
    source.offer(weights=[mount(gated=True)])
    watcher.poll()
    watcher.poll()
    assert not store.weights_dir(mount()).exists()

    fetcher.ungranted.clear()
    watcher.poll()

    assert store.weights_dir(mount()).is_dir()
    assert len(notifier.sent) == 1  # waiting in line is not worth a second notice


def test_a_running_job_still_waiting_for_weights_stops_the_queue_loudly(
    watched, notifier, monkeypatch
) -> None:
    store, queue, source, fetcher, watcher = watched
    fetcher.gated.add("org/model")
    fetcher.ungranted.add("org/model")
    source.offer(weights=[mount(gated=True)])
    watcher.poll()
    assert queue.claim("w1") is not None
    clock = [1000.0]
    monkeypatch.setattr(watcher_module.time, "monotonic", lambda: clock[0])

    watcher.poll()
    watcher.poll()
    stopped = [text for text in notifier.sent if "queue is stopped" in text]
    assert len(stopped) == 1 and "alpha__s1" in stopped[0]

    clock[0] += watcher_module.STALLED_REMINDER_S
    watcher.poll()
    assert sum("queue is stopped" in text for text in notifier.sent) == 2

    fetcher.ungranted.clear()
    watcher.poll()
    assert store.weights_dir(mount()).is_dir()
    clock[0] += watcher_module.STALLED_REMINDER_S
    watcher.poll()
    assert sum("queue is stopped" in text for text in notifier.sent) == 2


# --- the Hub fetcher ----------------------------------------------------------


@dataclass
class _Sibling:
    size: int


@dataclass
class _ModelInfo:
    gated: object
    private: bool | None
    siblings: list[_Sibling]


@dataclass
class FakeHubApi:
    gated: object = False
    private: bool | None = False
    granted: bool = True

    def model_info(self, repo_id, *, revision=None, files_metadata=False):
        return _ModelInfo(gated=self.gated, private=self.private, siblings=[_Sibling(3), _Sibling(4)])

    def auth_check(self, repo_id):
        if not self.granted:
            raise hubsource.GatedRepoError("no access", response=_Response())


@dataclass
class _Response:
    status_code: int = 403
    headers: dict = field(default_factory=dict)
    request: object = None


def test_the_hub_fetcher_reports_size_gating_and_visibility() -> None:
    info = HubWeights(HubConfig(), api=FakeHubApi(gated="manual")).describe(mount(gated=True))
    assert info == RepoInfo(size_bytes=7, gated=True, private=False)
    assert HubWeights(HubConfig(), api=FakeHubApi(private=True)).describe(mount()).private
    # Not knowing is not taken for public.
    assert HubWeights(HubConfig(), api=FakeHubApi(private=None)).describe(mount()).private


def test_a_gated_repo_the_organizer_cannot_read_is_reported_not_refused() -> None:
    fetcher = HubWeights(HubConfig(), api=FakeHubApi(gated="manual", granted=False))
    assert not fetcher.describe(mount(gated=True)).accessible
    assert HubWeights(HubConfig(), api=FakeHubApi(gated="manual")).describe(mount()).accessible
