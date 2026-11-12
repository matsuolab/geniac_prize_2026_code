"""The rules the watcher applies to whatever a source offers it.

Kept separate from ``test_hubsource.py``, which is about what HuggingFace has to
look like for a commit to become a candidate at all. These are the rules that would
survive a different source: the ceilings, the dedup, and the hash of
what actually landed.
"""

from __future__ import annotations

import errno
import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from capx.evalqueue import deadline
from capx.evalqueue.config import HubTeam, QueueConfig
from capx.evalqueue.models import (
    ErrorCode,
    ImageRef,
    Rejected,
    RunResult,
    RunSpec,
    RunStatus,
    SubmissionManifest,
    WeightsMount,
)
from capx.evalqueue.queue import FileQueue, new_job
from capx.evalqueue.hubsource import (
    MANIFEST_NAME,
    RANKING_MANIFEST_NAME,
    Candidate,
    RepoUnreadableError,
)
from capx.evalqueue.store import EvalStore, utcnow
from capx.evalqueue.watcher import SubmissionWatcher
from capx.evalqueue.weights import RepoInfo, WeightsCache

IMAGE = b"sif"


class FakeSource:
    """Offers whatever it is handed, and materializes it into a temp directory.

    ``vouched`` mirrors the one thing a real source varies on: the Hub knows an
    image's hash from its metadata, so the watcher can refuse a mismatch before the
    transfer. A source that does not know leaves it None.
    """

    poll_interval_s = 1.0

    def __init__(self, store: EvalStore) -> None:
        self._store = store
        self.offers: list[Candidate] = []
        self.bytes_by_key: dict[str, bytes] = {}
        self.fetched: list[str] = []
        self.on_fetch = None  # a test's hook for something that happens mid-transfer
        self.on_candidates = None  # and one for something that happens mid-inspection
        self.on_head_at = None  # and one for a team's main read at the deadline
        self.registered: list[HubTeam] = []
        self.at_deadline: dict[str, Candidate | RepoUnreadableError] = {}
        """What each team's ``main`` holds at the deadline: the ranking candidate in its
        commit, or an error for a repo that cannot be read then. Unset: no commit."""
        self.commits: dict[tuple[str, str], Candidate] = {}
        """Ranking candidates by team and commit, for ``ranking_at``."""
        self.unreadable: set[str] = set()
        """Commits ``ranking_at`` cannot read."""

    def offer(
        self,
        *,
        team_id: str = "alpha",
        key: str = "s1",
        image_bytes: bytes = IMAGE,
        vouched: bool = False,
        rejection: Rejected | None = None,
        revision: str | None = None,
        ranking: bool = False,
        **manifest_fields,
    ) -> Candidate:
        manifest_fields.setdefault(
            "image", ImageRef(repo=f"{team_id}-org/sub", revision="c" * 40, path="agent.sif")
        )
        manifest = SubmissionManifest(team_id=team_id, submission_id=key, **manifest_fields)
        candidate = Candidate(
            team_id=team_id,
            key=key,
            origin=f"fake://{team_id}/{key}",
            revision=revision,
            manifest=manifest,
            image_gb=len(image_bytes) / 1024**3,
            image_sha256=hashlib.sha256(image_bytes).hexdigest() if vouched else None,
            rejection=rejection,
            hub_repo=f"{team_id}-org/sub",
            hub_revision=(revision or key).ljust(40, "0"),
            manifest_name=RANKING_MANIFEST_NAME if ranking else MANIFEST_NAME,
        )
        self.bytes_by_key[f"{team_id}/{key}"] = image_bytes
        self.offers.append(candidate)
        if ranking:
            self.commits[team_id, candidate.hub_revision] = candidate
        return candidate

    def teams(self) -> list[HubTeam]:
        return list(self.registered)

    def known_teams(self) -> list[HubTeam]:
        return list(self.registered)

    def candidates(self, note_head=None) -> list[Candidate]:
        for offered in self.offers:  # like the Hub's: heads first, inspection after
            if note_head and offered.hub_repo and offered.hub_revision:
                note_head(HubTeam(team_id=offered.team_id, repo=offered.hub_repo), offered.hub_revision)
        if self.on_candidates:
            self.on_candidates()
        return list(self.offers)

    def head_at(self, team: HubTeam, deadline: datetime) -> str | None:
        if self.on_head_at:
            self.on_head_at(team)
        found = self.at_deadline.get(team.team_id)
        if isinstance(found, RepoUnreadableError):
            raise found
        if found is None:
            return None
        self.commits[team.team_id, found.hub_revision] = found
        return found.hub_revision

    def ranking_at(self, team: HubTeam, revision: str) -> Candidate:
        if revision in self.unreadable:
            raise RepoUnreadableError(f"{team.repo}@{revision[:12]}: 503")
        return self.commits.get((team.team_id, revision)) or Candidate(
            team_id=team.team_id,
            key="none",
            origin="fake",
            hub_repo=team.repo,
            hub_revision=revision,
            manifest_name=RANKING_MANIFEST_NAME,
        )

    def fetch(self, candidate: Candidate) -> Path:
        self.fetched.append(candidate.key)
        if self.on_fetch:
            self.on_fetch()
        assert candidate.manifest is not None
        directory = self._store.incoming / candidate.team_id / candidate.key
        directory.mkdir(parents=True, exist_ok=True)
        image = directory / candidate.manifest.image.path
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(self.bytes_by_key[f"{candidate.team_id}/{candidate.key}"])
        return image

    def preflight(self) -> list[str]:
        return []


@dataclass
class FakeFetcher:
    sizes: dict[str, int] = field(default_factory=dict)
    gated: set[str] = field(default_factory=set)
    private: set[str] = field(default_factory=set)
    ungranted: set[str] = field(default_factory=set)
    """Gated repos the organizer account has not been given access to."""
    described: list[str] = field(default_factory=list)
    downloaded: list[str] = field(default_factory=list)

    def describe(self, mount: WeightsMount) -> RepoInfo:
        self.described.append(mount.repo)
        return RepoInfo(
            size_bytes=self.sizes.get(mount.repo, 1),
            gated=mount.repo in self.gated,
            private=mount.repo in self.private,
            accessible=mount.repo not in self.ungranted,
        )

    def download(self, mount: WeightsMount, target: Path) -> None:
        if mount.repo in self.ungranted:
            raise Rejected(ErrorCode.WEIGHTS_GATED, f"no access to {mount.repo}")
        self.downloaded.append(mount.repo)
        (target / "config.json").write_text("{}")


@pytest.fixture
def setup(tmp_path):
    config = QueueConfig(root=str(tmp_path), master_seed=1)
    store = EvalStore(tmp_path)
    store.ensure_layout()
    queue = FileQueue(store)
    source = FakeSource(store)
    weights = WeightsCache(store, FakeFetcher(), max_gb=0)
    return config, store, queue, source, SubmissionWatcher(config, store, queue, source, weights)


def only_result(store: EvalStore) -> dict:
    results = list(store.results.glob("*.json"))
    assert len(results) == 1, [p.name for p in results]
    return json.loads(results[0].read_text())


def test_an_offered_submission_is_enqueued(setup) -> None:
    _, _, queue, source, watcher = setup
    source.offer()
    assert watcher.poll() == ["alpha__s1"]
    assert [job.job_id for job in queue.pending_jobs()] == ["alpha__s1"]


def test_the_same_submission_is_only_queued_once(setup) -> None:
    _, _, _, source, watcher = setup
    source.offer()
    assert watcher.poll() == ["alpha__s1"]
    assert watcher.poll() == []
    assert source.fetched == ["s1"]  # and it is not re-fetched either


def test_a_submission_already_queued_from_another_revision_is_not_queued_again(setup) -> None:
    _, store, queue, source, watcher = setup
    source.offer(revision="r1")
    assert watcher.poll() == ["alpha__s1"]

    source.offers.clear()
    source.offer(revision="r2")
    assert watcher.poll() == []
    assert len(queue.pending_jobs()) == 1
    assert list(store.results.glob("*.json")) == []


def test_a_rejection_is_remembered_per_revision_not_per_submission(setup) -> None:
    _, store, _, source, watcher = setup
    source.offer(revision="r1", image_sha256="0" * 64, vouched=True)
    assert watcher.poll() == []
    assert watcher.poll() == []
    assert len(list(store.results.glob("*.json"))) == 1

    source.offers.clear()
    source.offer(revision="r2")
    assert watcher.poll() == ["alpha__s1"]


def test_a_source_that_already_knows_it_cannot_run_is_believed(setup) -> None:
    _, store, _, source, watcher = setup
    source.offer(
        rejection=Rejected(ErrorCode.IMAGE_NOT_FOUND, "image 'agent.sif' is not in the revision")
    )
    assert watcher.poll() == []
    document = only_result(store)
    assert "is not in the revision" in document["error"]
    assert document["error_code"] == "image_not_found"
    assert source.fetched == []


def test_an_oversized_image_is_rejected_before_it_is_fetched(setup) -> None:
    _, store, queue, source, watcher = setup
    # The ceiling is in GB; make the file look bigger than it by shrinking the limit.
    watcher._eval_config.submission.max_image_gb = 0.0
    source.offer(image_bytes=b"x" * 1024)

    assert watcher.poll() == []
    assert queue.pending_jobs() == []
    document = only_result(store)
    assert document["status"] == "rejected"
    assert document["submitted_at"] is not None
    assert document["team_id"] == "alpha"
    assert "over the" in document["error"]
    assert document["error_code"] == "image_too_large"
    assert source.fetched == []


def test_a_hash_the_source_vouched_for_is_checked_before_fetching(setup) -> None:
    _, store, _, source, watcher = setup
    source.offer(image_sha256="0" * 64, vouched=True)
    assert watcher.poll() == []
    assert "sha256 mismatch" in only_result(store)["error"]
    assert only_result(store)["error_code"] == "image_hash_mismatch"
    assert source.fetched == []


def test_a_hash_nobody_vouched_for_is_checked_on_what_landed(setup) -> None:
    """The metadata answers "is the right image described"; only the file answers
    "did the right bytes arrive"."""
    _, store, _, source, watcher = setup
    source.offer(image_sha256="0" * 64, vouched=False)
    assert watcher.poll() == []
    assert "sha256 mismatch" in only_result(store)["error"]
    assert source.fetched == ["s1"]  # it had to be fetched to find out
    assert not (store.incoming / "alpha" / "s1").exists()  # and is not kept


def test_a_matching_sha256_passes(setup) -> None:
    _, _, _, source, watcher = setup
    source.offer(image_sha256=hashlib.sha256(IMAGE).hexdigest())
    assert watcher.poll() == ["alpha__s1"]


def test_a_manifest_id_that_could_escape_the_tree_is_refused() -> None:
    with pytest.raises(ValueError, match="team_id"):
        SubmissionManifest(team_id="../etc", submission_id="s1")
    with pytest.raises(ValueError, match="must be a relative path"):
        ImageRef(repo="a/b", revision="c" * 40, path="/etc/passwd")


def test_a_fetch_that_fails_is_the_team_s_rejection_not_a_crash(setup) -> None:
    _, store, _, source, watcher = setup
    source.offer()

    def explode(candidate):
        raise Rejected(ErrorCode.REPO_UNREADABLE, "cannot download 'agent.sif': gone")

    source.fetch = explode
    assert watcher.poll() == []
    assert "cannot download" in only_result(store)["error"]


def results_by_status(store: EvalStore, status: str) -> list[dict]:
    documents = (json.loads(path.read_text()) for path in store.results.glob("*.json"))
    return [doc for doc in documents if doc["status"] == status]


def test_a_new_submission_replaces_the_team_s_waiting_one(setup) -> None:
    _, store, queue, source, watcher = setup
    source.offer(key="s1")
    assert watcher.poll() == ["alpha__s1"]
    old_dir = store.incoming / "alpha" / "s1"
    assert old_dir.exists()

    source.offers.clear()
    source.offer(key="s2")
    assert watcher.poll() == ["alpha__s2"]
    assert [job.job_id for job in queue.pending_jobs()] == ["alpha__s2"]

    (cancelled,) = results_by_status(store, "cancelled")
    assert cancelled["job_id"] == "alpha__s1"
    assert "alpha__s2" in cancelled["error"]
    # The withdrawn job's queue record is gone, so the result has to remember when it
    # was submitted for the leaderboard to show it.
    assert cancelled["submitted_at"] is not None
    assert not old_dir.exists()  # the image of a job that will never run is not kept
    assert (store.incoming / "alpha" / "s2").exists()


def test_the_replaced_submission_can_be_pushed_again(setup) -> None:
    _, _, queue, source, watcher = setup
    source.offer(key="s1")
    watcher.poll()
    source.offers.clear()
    source.offer(key="s2")
    watcher.poll()

    source.offers.clear()
    source.offer(key="s1")  # the team rolls back
    assert watcher.poll() == ["alpha__s1"]
    assert [job.job_id for job in queue.pending_jobs()] == ["alpha__s1"]


def test_repeated_pushes_while_waiting_leave_only_the_latest(setup) -> None:
    _, store, queue, source, watcher = setup
    for key in ("s1", "s2", "s3"):
        source.offers.clear()
        source.offer(key=key)
        assert watcher.poll() == [f"alpha__{key}"]
    assert [job.job_id for job in queue.pending_jobs()] == ["alpha__s3"]
    assert {doc["job_id"] for doc in results_by_status(store, "cancelled")} == {
        "alpha__s1",
        "alpha__s2",
    }


def test_a_submission_that_is_rejected_cancels_nothing(setup) -> None:
    _, store, queue, source, watcher = setup
    source.offer(key="s1")
    watcher.poll()

    source.offers.clear()
    source.offer(key="s2", image_sha256="0" * 64, vouched=True)  # declared hash is wrong
    assert watcher.poll() == []
    assert [job.job_id for job in queue.pending_jobs()] == ["alpha__s1"]
    assert results_by_status(store, "cancelled") == []
    assert len(results_by_status(store, "rejected")) == 1


def test_a_team_with_a_submission_running_has_its_next_one_deferred(setup) -> None:
    _, store, queue, source, watcher = setup
    source.offer(key="s1")
    watcher.poll()
    claimed = queue.claim("w0")
    assert claimed is not None

    source.offers.clear()
    source.offer(key="s2")
    assert watcher.poll() == []
    assert source.fetched == ["s1"]  # deferred before any transfer
    assert list(store.results.glob("*.json")) == []  # neither rejected nor cancelled
    assert queue.running_count("alpha") == 1  # and the running job is left alone
    # ...but it is on record, so the leaderboard can tell the team it is waiting.
    note = json.loads((store.deferred / "alpha.json").read_text())
    assert note["job_id"] == "alpha__s2"
    assert note["submission_id"] == "s2"


def test_a_deferred_submission_is_queued_once_the_running_one_finishes(setup) -> None:
    _, store, queue, source, watcher = setup
    source.offer(key="s1")
    watcher.poll()
    claimed = queue.claim("w0")

    source.offers.clear()
    source.offer(key="s2")
    assert watcher.poll() == []
    assert watcher.poll() == []  # still waiting, and still not marked as seen

    queue.finish(claimed, RunStatus.COMPLETED)
    assert watcher.poll() == ["alpha__s2"]
    assert not (store.deferred / "alpha.json").exists()


def test_going_back_to_the_running_submission_drops_the_deferred_one(setup) -> None:
    _, store, queue, source, watcher = setup
    source.offer(key="s1")
    watcher.poll()
    queue.claim("w0")
    source.offers.clear()
    source.offer(key="s2", revision="r2")
    watcher.poll()
    assert (store.deferred / "alpha.json").exists()

    source.offers.clear()
    source.offer(key="s1", revision="r3")  # the team reverts its manifest
    assert watcher.poll() == []
    assert not (store.deferred / "alpha.json").exists()


def test_pushes_made_while_a_job_runs_collapse_to_the_latest(setup) -> None:
    _, _, queue, source, watcher = setup
    source.offer(key="s1")
    watcher.poll()
    claimed = queue.claim("w0")

    for key in ("s2", "s3"):  # the repo's head moves; the watcher only ever sees the head
        source.offers.clear()
        source.offer(key=key)
        assert watcher.poll() == []

    queue.finish(claimed, RunStatus.COMPLETED)
    assert watcher.poll() == ["alpha__s3"]
    assert "s2" not in source.fetched


def test_an_old_job_started_during_the_fetch_is_not_cancelled(setup) -> None:
    """The worker can claim the old job after the watcher looked and before it withdrew."""
    _, store, queue, source, watcher = setup
    source.offer(key="s1")
    watcher.poll()

    claimed: list = []
    source.on_fetch = lambda: claimed.append(queue.claim("w0"))
    source.offers.clear()
    source.offer(key="s2")
    assert watcher.poll() == []  # s2 waits behind the job that just started
    assert claimed and claimed[0] is not None
    assert queue.running_count("alpha") == 1
    assert queue.pending_jobs() == []
    assert results_by_status(store, "cancelled") == []
    assert (store.incoming / "alpha" / "s2").exists()  # the head: kept to be taken in

    source.on_fetch = None
    queue.finish(claimed[0], RunStatus.COMPLETED)
    assert watcher.poll() == ["alpha__s2"]


def test_a_submission_deferred_after_its_fetch_leaves_nothing_when_the_team_pushes_on(
    setup,
) -> None:
    _, store, queue, source, watcher = setup
    source.offer(key="s1")
    watcher.poll()
    claimed: list = []
    source.on_fetch = lambda: claimed.append(queue.claim("w0"))
    source.offers.clear()
    source.offer(key="s2")
    watcher.poll()
    source.on_fetch = None

    source.offers.clear()
    source.offer(key="s3")
    assert watcher.poll() == []  # deferred behind s1, before any transfer
    queue.finish(claimed[0], RunStatus.COMPLETED)
    assert watcher.poll() == ["alpha__s3"]
    assert "s2" not in {path.name for path in (store.incoming / "alpha").iterdir()}


def test_another_team_s_waiting_job_is_never_cancelled(setup) -> None:
    _, store, queue, source, watcher = setup
    source.offer(team_id="alpha", key="s1")
    source.offer(team_id="beta", key="s1")
    assert sorted(watcher.poll()) == ["alpha__s1", "beta__s1"]

    source.offers.clear()
    source.offer(team_id="alpha", key="s2")
    assert watcher.poll() == ["alpha__s2"]
    assert sorted(job.job_id for job in queue.pending_jobs()) == ["alpha__s2", "beta__s1"]
    assert [doc["team_id"] for doc in results_by_status(store, "cancelled")] == ["alpha"]


def test_with_a_higher_cap_only_the_oldest_waiting_job_is_replaced(setup) -> None:
    config, _, queue, source, watcher = setup
    config.watcher.max_active_per_team = 2
    for key in ("s1", "s2"):
        source.offers.clear()
        source.offer(key=key)
        assert watcher.poll() == [f"alpha__{key}"]
    source.offers.clear()
    source.offer(key="s3")
    assert watcher.poll() == ["alpha__s3"]
    assert [job.job_id for job in queue.pending_jobs()] == ["alpha__s2", "alpha__s3"]


def test_with_the_cap_off_nothing_is_cancelled_or_deferred(setup) -> None:
    config, store, queue, source, watcher = setup
    config.watcher.max_active_per_team = 0
    for key in ("s1", "s2", "s3"):
        source.offers.clear()
        source.offer(key=key)
        assert watcher.poll() == [f"alpha__{key}"]
    assert len(queue.pending_jobs()) == 3
    assert results_by_status(store, "cancelled") == []


def test_from_the_deadline_nothing_is_queued_and_what_waits_is_cancelled(setup) -> None:
    config, store, queue, source, watcher = setup
    source.offer(key="s1")
    watcher.poll()
    deferred = {"job_id": "beta__s9", "team_id": "beta", "submission_id": "s9",
                "since": utcnow().isoformat()}
    (store.deferred / "beta.json").write_text(json.dumps(deferred))
    config.deadline = utcnow() - timedelta(seconds=1)
    source.offer(key="s2")

    assert watcher.poll() == []
    assert queue.pending_jobs() == []
    results = {json.loads(p.read_text())["job_id"]: json.loads(p.read_text())["status"]
               for p in store.results.glob("*.json")}
    # Both the queued one and the one deferred behind a running job.
    assert results == {"alpha__s1": "cancelled", "beta__s9": "cancelled"}
    assert source.fetched == ["s1"]
    assert not (store.deferred / "beta.json").exists()


def test_a_submission_fetched_as_the_deadline_passes_is_cancelled_not_queued(setup) -> None:
    config, store, queue, source, watcher = setup
    source.offer(key="s1")
    source.on_fetch = lambda: setattr(config, "deadline", utcnow() - timedelta(seconds=1))
    assert watcher.poll() == []
    assert queue.pending_jobs() == []
    assert only_result(store)["status"] == "cancelled"


# --- one submission against the others ---------------------------------------


def test_a_submission_that_fails_unexpectedly_stops_no_other(setup, caplog) -> None:
    _, store, _, source, watcher = setup
    source.offer(team_id="alpha", key="s1")
    source.offer(team_id="beta", key="s1")
    real_fetch = source.fetch

    def fetch(candidate):
        if candidate.team_id == "alpha":
            raise RuntimeError("a bug of ours")
        return real_fetch(candidate)

    source.fetch = fetch
    assert watcher.poll() == ["beta__s1"]
    assert "a bug of ours" in caplog.text
    assert list(store.results.glob("*.json")) == []  # not the team's rejection

    source.fetch = real_fetch
    assert watcher.poll() == ["alpha__s1"]


def test_a_hub_hiccup_is_tried_again_before_it_is_published(setup) -> None:
    _, store, _, source, watcher = setup
    source.offer(rejection=Rejected(ErrorCode.REPO_UNREADABLE, "503", transient=True))
    assert watcher.poll() == []
    assert watcher.poll() == []
    assert list(store.results.glob("*.json")) == []

    rewind_retries(store, timedelta(hours=2))
    watcher.poll()
    assert only_result(store)["error_code"] == "repo_unreadable"
    assert list((store.state / "retrying").iterdir()) == []


def rewind_retries(store: EvalStore, by: timedelta) -> None:
    """As if the Hub had failed that much earlier."""
    for path in (store.state / "retrying").glob("*.json"):
        since = datetime.fromisoformat(json.loads(path.read_text())["since"]) - by
        path.write_text(json.dumps({"since": since.isoformat()}))


def test_a_rejection_that_cannot_be_published_does_not_start_the_hour_over(
    setup, monkeypatch
) -> None:
    _, store, _, source, watcher = setup
    source.offer(rejection=Rejected(ErrorCode.REPO_UNREADABLE, "503", transient=True))
    watcher.poll()
    rewind_retries(store, timedelta(hours=2))
    real_publish = store.publish_result

    def full_disk(result):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(store, "publish_result", full_disk)
    watcher.poll()
    assert len(list((store.state / "retrying").iterdir())) == 1

    monkeypatch.setattr(store, "publish_result", real_publish)
    watcher.poll()
    assert only_result(store)["error_code"] == "repo_unreadable"
    assert list((store.state / "retrying").iterdir()) == []


def test_the_hour_of_retries_survives_a_watcher_restart(setup) -> None:
    config, store, queue, source, watcher = setup
    source.offer(rejection=Rejected(ErrorCode.REPO_UNREADABLE, "503", transient=True))
    watcher.poll()
    rewind_retries(store, timedelta(minutes=59))

    restarted = SubmissionWatcher(config, store, queue, source, WeightsCache(store, FakeFetcher(), max_gb=0))
    restarted.poll()
    assert list(store.results.glob("*.json")) == []
    rewind_retries(store, timedelta(minutes=2))
    SubmissionWatcher(config, store, queue, source, WeightsCache(store, FakeFetcher(), max_gb=0)).poll()
    assert only_result(store)["error_code"] == "repo_unreadable"


def test_a_hiccup_that_clears_leaves_nothing_to_retry(setup) -> None:
    _, store, _, source, watcher = setup
    offered = source.offer(rejection=Rejected(ErrorCode.REPO_UNREADABLE, "503", transient=True))
    watcher.poll()
    assert len(list((store.state / "retrying").iterdir())) == 1
    source.offers = [replace(offered, rejection=None)]
    assert watcher.poll() == ["alpha__s1"]
    assert list((store.state / "retrying").iterdir()) == []


WEIGHTS = WeightsMount(repo="org/model", revision="b" * 40, path="/models/m")


@pytest.fixture
def weights_down(setup):
    """The image arrives; the weights repo answers 503 until ``fetcher.download`` is put
    back."""
    config, store, queue, source, _ = setup
    fetcher = FakeFetcher()
    real_download = fetcher.download

    def unavailable(mount, target):
        raise Rejected(ErrorCode.REPO_UNREADABLE, "503", transient=True)

    fetcher.download = unavailable
    watcher = SubmissionWatcher(config, store, queue, source, WeightsCache(store, fetcher, max_gb=0))
    return store, queue, source, watcher, lambda: setattr(fetcher, "download", real_download)


def test_a_submission_pushed_past_while_it_is_retried_leaves_nothing_behind(weights_down) -> None:
    store, _, source, watcher, hub_is_back = weights_down
    source.offer(key="s1", revision="r1", weights=[WEIGHTS])
    assert watcher.poll() == []
    assert (store.incoming / "alpha" / "s1" / "agent.sif").is_file()

    source.offers.clear()
    source.offer(key="s2", revision="r2", weights=[WEIGHTS])
    assert watcher.poll() == []
    assert not (store.incoming / "alpha" / "s1").exists()

    hub_is_back()
    assert watcher.poll() == ["alpha__s2"]
    assert list(store.results.glob("*.json")) == []  # s1 was neither refused nor cancelled


def test_a_commit_unreadable_at_first_leaves_nothing_once_pushed_past(weights_down) -> None:
    """The manifest could not be read at first, so the attempt had the commit's key;
    the next poll read it, fetched the image, and the weights failed."""
    store, _, source, watcher, _ = weights_down
    unread = source.offer(
        key="r1", revision="r1", rejection=Rejected(ErrorCode.REPO_UNREADABLE, "503", transient=True)
    )
    watcher.poll()
    rewind_retries(store, timedelta(minutes=10))
    since = json.loads((store.state / "retrying" / "alpha__r1.json").read_text())["since"]

    source.offers.remove(unread)
    source.offer(key="s1", revision="r1", weights=[WEIGHTS])
    watcher.poll()
    record = json.loads((store.state / "retrying" / "alpha__r1.json").read_text())
    assert record == {"since": since}  # the hour goes on

    source.offers.clear()
    source.offer(key="s2", revision="r2", weights=[WEIGHTS])
    watcher.poll()
    assert not (store.incoming / "alpha" / "s1").exists()


def test_a_submission_our_disk_failed_is_dropped_when_the_team_pushes_on(setup, caplog) -> None:
    config, store, queue, source, _ = setup
    fetcher = FakeFetcher()
    real_download = fetcher.download

    def full_disk(mount, target):
        raise OSError(errno.ENOSPC, "No space left on device")

    fetcher.download = full_disk
    watcher = SubmissionWatcher(config, store, queue, source, WeightsCache(store, fetcher, max_gb=0))
    source.offer(key="s1", revision="r1", weights=[WEIGHTS])
    assert watcher.poll() == []
    assert "No space left" in caplog.text
    assert (store.incoming / "alpha" / "s1" / "agent.sif").is_file()  # kept for the next try

    fetcher.download = real_download
    source.offers.clear()
    source.offer(key="s2", revision="r2", weights=[WEIGHTS])
    assert watcher.poll() == ["alpha__s2"]
    assert not (store.incoming / "alpha" / "s1").exists()


def test_an_attempt_left_at_the_deadline_leaves_nothing_behind(weights_down) -> None:
    store, _, source, watcher, _ = weights_down
    source.offer(key="s1", revision="r1", weights=[WEIGHTS])
    watcher.poll()
    assert (store.incoming / "alpha" / "s1").exists()

    watcher._config.deadline = utcnow() - timedelta(seconds=1)
    watcher.poll()
    assert not (store.incoming / "alpha" / "s1").exists()


def test_a_stale_attempt_s_image_stays_while_a_job_uses_it(weights_down) -> None:
    store, queue, source, watcher, _ = weights_down
    source.offer(key="s1", revision="r1", weights=[WEIGHTS])
    watcher.poll()
    _queue_by_hand(store, queue, store.incoming / "alpha" / "s1" / "agent.sif")

    source.offers.clear()
    source.offer(key="s2", revision="r2", weights=[WEIGHTS])
    watcher.poll()
    assert (store.incoming / "alpha" / "s1" / "agent.sif").is_file()


def test_the_same_manifest_in_a_new_commit_keeps_what_was_fetched(weights_down) -> None:
    store, _, source, watcher, _ = weights_down
    source.offer(key="s1", revision="r1", weights=[WEIGHTS])
    watcher.poll()
    kept = store.incoming / "alpha" / "s1" / "kept"
    kept.write_text("fetched by the first attempt")

    source.offers.clear()
    source.offer(key="s1", revision="r2", weights=[WEIGHTS])  # a README edit
    watcher.poll()
    assert kept.is_file()


@pytest.mark.parametrize("moved_on", ["pushed past", "deadline"])
def test_a_retried_submission_deferred_in_a_new_commit_goes_once_the_team_moves_on(
    weights_down, moved_on
) -> None:
    """Fetched, the weights failed, an earlier job started, and the same manifest came
    again in a new commit: deferred before anything else is done with it."""
    store, queue, source, watcher, _ = weights_down
    source.offer(key="s0", revision="r0")
    assert watcher.poll() == ["alpha__s0"]
    source.offers.clear()
    source.offer(key="s1", revision="r1", weights=[WEIGHTS])
    assert watcher.poll() == []
    running = queue.claim("w0")
    source.offers.clear()
    source.offer(key="s1", revision="r2", weights=[WEIGHTS])
    assert watcher.poll() == []
    assert (store.incoming / "alpha" / "s1").exists()  # still the head

    if moved_on == "deadline":
        watcher._config.deadline = utcnow() - timedelta(seconds=1)
    else:
        source.offers.clear()
        source.offer(key="s2", revision="r3")
    watcher.poll()
    assert not (store.incoming / "alpha" / "s1").exists()
    assert Path(running.job.image_path).is_file()


def test_an_image_kept_after_our_failure_stays_though_its_job_was_never_marked(
    setup, monkeypatch
) -> None:
    """The watcher stopped between queuing a job and recording it, and the job then
    failed on our side, so the worker kept its image for an operator to run again."""
    _, store, queue, source, watcher = setup
    source.offer(key="s1")
    def stopped(*_args, **_kwargs):
        raise RuntimeError("the watcher stopped here")

    monkeypatch.setattr(watcher, "_mark_seen", stopped)
    watcher.poll()
    monkeypatch.undo()
    assert not (store.seen / "alpha__s1.json").exists()

    claimed = queue.claim("w0")
    result = RunResult(run_id="alpha__s1__1", job_id="alpha__s1", team_id="alpha", submission_id="s1",
                       status=RunStatus.FAILED, error_code=ErrorCode.INTERNAL_ERROR)
    store.publish_result(result)
    queue.drop_image(claimed, result)
    queue.finish(claimed, RunStatus.FAILED)
    image = Path(claimed.job.image_path)
    assert image.is_file()

    source.offers.clear()
    source.offer(key="s2")
    assert watcher.poll() == ["alpha__s2"]
    assert image.is_file()


# --- jobs queued by hand -----------------------------------------------------


def _queue_by_hand(store: EvalStore, queue: FileQueue, image: Path, submission_id: str = "manual") -> None:
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(IMAGE)
    queue.enqueue(new_job(f"alpha__{submission_id}", RunSpec(team_id="alpha", submission_id=submission_id), image))


def test_a_job_queued_by_hand_is_neither_counted_nor_replaced(setup) -> None:
    """`evalq submit` bypasses the per-team cap (docs/operations.md §1)."""
    _, store, queue, source, watcher = setup
    _queue_by_hand(store, queue, store.root / "elsewhere" / "agent.sif")
    source.offer(key="s1")
    assert watcher.poll() == ["alpha__s1"]
    assert sorted(job.job_id for job in queue.pending_jobs()) == ["alpha__manual", "alpha__s1"]
    assert results_by_status(store, "cancelled") == []

    queue.withdraw("alpha__s1")
    assert queue.claim("w0").job.job_id == "alpha__manual"
    source.offers.clear()
    source.offer(key="s2")
    assert watcher.poll() == ["alpha__s2"]  # not deferred behind the job queued by hand


def test_a_replaced_job_s_image_stays_while_a_job_queued_by_hand_uses_it(setup) -> None:
    """The operations guide re-runs an image by submitting it from incoming/."""
    _, store, queue, source, watcher = setup
    source.offer(key="s1")
    watcher.poll()
    _queue_by_hand(store, queue, store.incoming / "alpha" / "s1" / "agent.sif")

    source.offers.clear()
    source.offer(key="s2")
    assert watcher.poll() == ["alpha__s2"]
    assert [doc["job_id"] for doc in results_by_status(store, "cancelled")] == ["alpha__s1"]
    assert (store.incoming / "alpha" / "s1" / "agent.sif").is_file()


def test_a_replaced_job_s_image_stays_while_a_job_queued_by_hand_is_requeued(
    setup, monkeypatch
) -> None:
    """A worker restarting moves its claim back to pending while the watcher looks: read
    pending, then claimed, and the job is in neither."""
    _, store, queue, source, watcher = setup
    _queue_by_hand(store, queue, store.incoming / "alpha" / "s1" / "agent.sif")
    running = queue.claim("w0")
    assert running.job.job_id == "alpha__manual"
    source.offer(key="s1")
    assert watcher.poll() == ["alpha__s1"]

    real_withdraw, real_claimed_jobs = queue.withdraw, queue.claimed_jobs
    withdrawn: list[str] = []

    def withdraw(job_id: str) -> bool:
        withdrawn.append(job_id)
        return real_withdraw(job_id)

    def claimed_jobs() -> list:
        if withdrawn and running.path.exists():
            queue.requeue(running)
        return real_claimed_jobs()

    monkeypatch.setattr(queue, "withdraw", withdraw)
    monkeypatch.setattr(queue, "claimed_jobs", claimed_jobs)
    source.offers.clear()
    source.offer(key="s2")
    assert watcher.poll() == ["alpha__s2"]
    assert withdrawn == ["alpha__s1"]
    assert (store.incoming / "alpha" / "s1" / "agent.sif").is_file()


def test_a_replaced_job_s_image_stays_while_a_job_queued_by_hand_is_requeued_and_reclaimed(
    setup, monkeypatch
) -> None:
    """Both moves within one read of the queue: back to pending after pending was read,
    and claimed again after claimed was read."""
    _, store, queue, source, watcher = setup
    _queue_by_hand(store, queue, store.incoming / "alpha" / "s1" / "agent.sif")
    running = queue.claim("w0")
    source.offer(key="s1")
    assert watcher.poll() == ["alpha__s1"]

    real_withdraw, real_claimed_jobs = queue.withdraw, queue.claimed_jobs
    withdrawn: list[str] = []
    moved: list = []

    def withdraw(job_id: str) -> bool:
        withdrawn.append(job_id)
        return real_withdraw(job_id)

    def claimed_jobs() -> list:
        if not withdrawn or moved:
            return real_claimed_jobs()
        moved.append(queue.requeue(running))
        jobs = real_claimed_jobs()
        moved.append(queue.claim("w0"))
        return jobs

    monkeypatch.setattr(queue, "withdraw", withdraw)
    monkeypatch.setattr(queue, "claimed_jobs", claimed_jobs)
    source.offers.clear()
    source.offer(key="s2")
    assert watcher.poll() == ["alpha__s2"]
    assert moved[1].job.job_id == "alpha__manual"
    assert (store.incoming / "alpha" / "s1" / "agent.sif").is_file()


def test_a_replaced_job_s_whole_fetch_directory_goes_with_a_nested_image(setup) -> None:
    _, store, _, source, watcher = setup
    nested = ImageRef(repo="alpha-org/sub", revision="c" * 40, path="images/agent.sif")
    source.offer(key="s1", image=nested)
    watcher.poll()
    assert (store.incoming / "alpha" / "s1" / "images" / "agent.sif").is_file()

    source.offers.clear()
    source.offer(key="s2", image=nested)
    assert watcher.poll() == ["alpha__s2"]
    assert not (store.incoming / "alpha" / "s1").exists()


# --- what a manifest's env may say -------------------------------------------


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"K": "${CAPX_HF_TOKEN}"}, "may not contain"),
        ({"K": "`id`"}, "may not contain"),
        ({"K": "a\\b"}, "may not contain"),
        ({"K": "x\nCAPX_ENV_SOCKET=/evil.sock"}, "may not contain"),
        ({"K": "x\0"}, "may not contain"),
        ({"CAPX_ENV_SOCKET": "/evil.sock"}, "set by the organizers"),
        ({"HF_HUB_OFFLINE": "0"}, "set by the organizers"),
        ({"HOME": "/tmp"}, "set by the organizers"),
        ({"XDG_CACHE_HOME": "/tmp"}, "set by the organizers"),
        ({"SINGULARITYENV_X": "1"}, "set by the organizers"),
        ({"APPTAINER_BIND": "/"}, "set by the organizers"),
        ({"printenv\n": "1"}, "letters, digits and _"),
        ({"HOME\n": "/tmp"}, "letters, digits and _"),
        ({"CAPX_ENV_SOCKET\n": "/evil.sock"}, "letters, digits and _"),
        ({"1X": "1"}, "letters, digits and _"),
        ({"A-B": "1"}, "letters, digits and _"),
        ({"A=B": "1"}, "letters, digits and _"),
    ],
)
def test_an_env_entry_that_could_reach_beyond_the_team_s_container_is_refused(env, message) -> None:
    with pytest.raises(ValueError, match=message):
        RunSpec(team_id="alpha", submission_id="s1", env=env)


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: SubmissionManifest(team_id="alpha\n", submission_id="s1"), "team_id"),
        (lambda: SubmissionManifest(team_id="alpha", submission_id="s1\n"), "submission_id"),
        (lambda: ImageRef(repo="org/name\n", revision="c" * 40, path="a.sif"), "<owner>/<name>"),
        (lambda: ImageRef(repo="org/name", revision="c" * 40 + "\n", path="a.sif"), "40-character"),
        (lambda: WeightsMount(repo="org/m", revision="c" * 40, path="/models/m\n"), "must be /models/"),
    ],
)
def test_a_trailing_newline_does_not_pass_for_a_safe_name(make, message) -> None:
    """A regex's `$` also matches before a final newline; these names reach paths, the
    board and --bind specs."""
    with pytest.raises(ValueError, match=message):
        make()


def test_an_ordinary_env_is_accepted() -> None:
    env = {
        "CAPX_MODEL_PATH": "/models/qwen3.6-27b",
        "CAPX_SERVE_EXTRA_ARGS": "--attention-backend TRITON_ATTN --max-num-seqs 64",
        "_PRIVATE": "it's \"quoted\", #1",
    }
    assert RunSpec(team_id="alpha", submission_id="s1", env=env).env == env


# --- the deadline ------------------------------------------------------------


def _defer(store: EvalStore, team_id: str = "beta", key: str = "s9") -> Path:
    marker = store.deferred / f"{team_id}.json"
    marker.write_text(json.dumps({"job_id": f"{team_id}__{key}", "team_id": team_id,
                                  "submission_id": key, "since": utcnow().isoformat()}))
    return marker


def test_a_deferred_submission_has_one_cancellation_however_many_close_the_round(setup) -> None:
    """The watcher and every worker close the round, together or after one that stopped
    between publishing and removing the marker."""
    _, store, queue, _, _ = setup
    note = {"job_id": "beta__s9", "team_id": "beta", "submission_id": "s9",
            "since": utcnow().isoformat()}
    marker = store.deferred / "beta.json"
    marker.write_text(json.dumps(note))
    assert deadline.close_round(store, queue) == ["beta__s9"]

    marker.write_text(json.dumps(note))  # as if that closer died before the unlink
    assert deadline.close_round(store, queue) == ["beta__s9"]
    assert deadline.close_round(store, queue) == []
    (cancelled,) = results_by_status(store, "cancelled")
    assert cancelled["job_id"] == "beta__s9"
    assert list(store.deferred.iterdir()) == []


def test_a_cancellation_that_cannot_be_published_keeps_its_marker(setup, monkeypatch) -> None:
    _, store, queue, _, _ = setup
    marker = _defer(store)
    real_publish = store.publish_result

    def full_disk(result):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(store, "publish_result", full_disk)
    assert deadline.close_round(store, queue) == []
    assert marker.exists()

    monkeypatch.setattr(store, "publish_result", real_publish)
    assert deadline.close_round(store, queue) == ["beta__s9"]
    assert not marker.exists()


def test_an_unreadable_deferral_marker_is_dropped(setup) -> None:
    _, store, queue, _, _ = setup
    (store.deferred / "beta.json").write_text('{"job_id": "beta__s9"}')
    assert deadline.close_round(store, queue) == []
    assert list(store.deferred.iterdir()) == []
    assert list(store.results.glob("*.json")) == []


def test_a_deferred_submission_fetched_across_the_deadline_is_cancelled_once(setup) -> None:
    """The fetch that starts once the team's running job is done, and runs into the
    deadline, and the round's closing, both cancel it."""
    config, store, queue, source, watcher = setup
    source.offer(key="s1")
    watcher.poll()
    running = queue.claim("w0")
    source.offers.clear()
    source.offer(key="s2")
    watcher.poll()
    assert (store.deferred / "alpha.json").exists()

    queue.finish(running, RunStatus.COMPLETED)
    source.on_fetch = lambda: setattr(config, "deadline", utcnow() - timedelta(seconds=1))
    assert watcher.poll() == []
    watcher.poll()
    deadline.close_round(store, queue)
    assert [doc["job_id"] for doc in results_by_status(store, "cancelled")] == ["alpha__s2"]
    assert not (store.incoming / "alpha" / "s2").exists()


def test_after_the_deadline_the_images_of_what_was_cancelled_go(setup) -> None:
    config, store, queue, source, watcher = setup
    source.offer(team_id="alpha", key="s1")
    source.offer(team_id="beta", key="s1")
    watcher.poll()
    running = queue.claim("w0")
    waiting = "beta" if running.job.team_id == "alpha" else "alpha"

    config.deadline = utcnow() - timedelta(seconds=1)
    watcher.poll()
    assert [doc["job_id"] for doc in results_by_status(store, "cancelled")] == [f"{waiting}__s1"]
    assert not (store.incoming / waiting / "s1").exists()
    assert Path(running.job.image_path).is_file()  # the runner stops it; its result drops it
