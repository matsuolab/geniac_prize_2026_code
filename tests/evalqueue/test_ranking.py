"""Ranking submissions: checked before the deadline, recorded at it, queued after it."""

from __future__ import annotations

import errno
import json
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import pytest
import yaml

from capx.evalqueue import hubsource
from capx.evalqueue import watcher as watcher_module
from capx.evalqueue.config import HubConfig, HubTeam, QueueConfig
from capx.evalqueue.hubsource import RANKING_MANIFEST_NAME, Candidate, HubSource, RepoUnreadableError
from capx.evalqueue.leaderboard import build_snapshot
from capx.evalqueue.models import ErrorCode, Rejected, RunResult, RunStatus, WeightsMount
from capx.evalqueue.queue import FileQueue
from capx.evalqueue.ranking import RankingBook, RankingStatus, queue_ranking_runs
from capx.evalqueue.store import EvalStore, utcnow
from capx.evalqueue.teams import save_registered_teams
from capx.evalqueue.watcher import RETRY_S, SubmissionWatcher
from capx.evalqueue.weights import WeightsCache

from .test_hubsource import IMAGE, FakeHub, manifest_bytes
from .test_watcher import FakeFetcher, FakeSource

ALPHA = HubTeam(team_id="alpha", repo="alpha-org/sub")
WEIGHTS = WeightsMount(repo="org/model", revision="b" * 40, path="/models/m")


@pytest.fixture
def setup(tmp_path):
    config = QueueConfig(root=str(tmp_path), master_seed=1)
    store = EvalStore(tmp_path)
    store.ensure_layout()
    queue = FileQueue(store)
    source = FakeSource(store)
    source.registered = [ALPHA]
    fetcher = FakeFetcher()
    watcher = SubmissionWatcher(
        config, store, queue, source, WeightsCache(store, fetcher, max_gb=0)
    )
    return config, store, queue, source, fetcher, watcher


def test_a_ranking_submission_is_checked_but_neither_fetched_nor_queued(setup) -> None:
    _, store, queue, source, fetcher, watcher = setup
    source.offer(key="r1", ranking=True, weights=[WEIGHTS])
    assert watcher.poll() == []
    entry = RankingBook(store).get("alpha")
    assert entry is not None and entry.status is RankingStatus.ACCEPTED and not entry.final
    assert source.fetched == [] and fetcher.downloaded == [] and queue.pending_jobs() == []
    assert fetcher.described == ["org/model"]


def test_a_ranking_submission_is_checked_once_per_commit(setup) -> None:
    _, _, _, source, fetcher, watcher = setup
    source.offer(key="r1", ranking=True, weights=[WEIGHTS])
    watcher.poll()
    watcher.poll()
    assert fetcher.described == ["org/model"]


def test_a_refused_ranking_submission_says_why(setup) -> None:
    _, store, _, source, _, watcher = setup
    source.offer(key="r1", ranking=True, image_bytes=b"x" * 10, image_sha256="0" * 64, vouched=True)
    watcher.poll()
    entry = RankingBook(store).get("alpha")
    assert entry is not None and entry.status is RankingStatus.REJECTED
    assert entry.error_code is ErrorCode.IMAGE_HASH_MISMATCH


def test_a_hub_hiccup_is_not_recorded_as_the_commit_s_outcome(setup) -> None:
    _, store, _, source, fetcher, watcher = setup
    candidate = source.offer(key="r1", ranking=True, weights=[WEIGHTS])
    real_describe = fetcher.describe

    def describe(mount):
        raise Rejected(ErrorCode.REPO_UNREADABLE, "503", transient=True)

    fetcher.describe = describe
    watcher.poll()
    assert RankingBook(store).get("alpha") is None

    fetcher.describe = real_describe
    watcher.poll()
    entry = RankingBook(store).get("alpha")
    assert entry.status is RankingStatus.ACCEPTED and entry.hub_revision == candidate.hub_revision


def test_taking_the_file_away_takes_the_entry_away(setup) -> None:
    _, store, _, source, _, watcher = setup
    source.offer(key="r1", ranking=True)
    watcher.poll()
    source.offers = [
        Candidate(team_id="alpha", key="r2", origin="fake", hub_repo=ALPHA.repo,
                  hub_revision="2" * 40, manifest_name=RANKING_MANIFEST_NAME)
    ]
    watcher.poll()
    assert RankingBook(store).get("alpha") is None


def _ranking_candidate(source: FakeSource, key: str, team_id: str = "alpha") -> Candidate:
    candidate = source.offer(team_id=team_id, key=key, ranking=True)
    source.offers.remove(candidate)
    return candidate


def test_at_the_deadline_each_team_s_submission_is_recorded_for_good(setup) -> None:
    config, store, _, source, _, watcher = setup
    beta = HubTeam(team_id="beta", repo="beta-org/sub")
    source.registered = [ALPHA, beta]
    source.offer(key="early", ranking=True)
    watcher.poll()
    config.deadline = utcnow() - timedelta(seconds=1)
    source.at_deadline = {"alpha": _ranking_candidate(source, "final")}  # beta: nothing

    watcher.poll()
    book = RankingBook(store)
    alpha, missing = book.get("alpha"), book.get("beta")
    assert alpha is not None and alpha.final and alpha.key == "final"
    assert missing is not None and missing.final and missing.status is RankingStatus.MISSING
    assert book.is_final

    source.at_deadline = {"alpha": _ranking_candidate(source, "later")}
    watcher.poll()
    assert RankingBook(store).get("alpha").key == "final"  # recorded once


def test_a_submission_that_cannot_be_read_is_tried_again_at_the_recorded_commit(setup) -> None:
    """Only the content is read again. A commit pushed after the deadline, however it is
    dated, is never what gets recorded."""
    config, store, _, source, _, watcher = setup
    config.deadline = utcnow() - timedelta(seconds=1)
    final = _ranking_candidate(source, "final")
    source.at_deadline = {"alpha": final}
    source.unreadable = {final.hub_revision}
    watcher.poll()
    assert not RankingBook(store).is_final and RankingBook(store).get("alpha") is None

    source.at_deadline = {"alpha": _ranking_candidate(source, "later")}
    source.unreadable = set()
    watcher.poll()
    assert RankingBook(store).is_final and RankingBook(store).get("alpha").key == "final"


def test_a_main_that_cannot_be_read_at_the_deadline_falls_back_to_the_head_seen_before(setup) -> None:
    config, store, _, source, _, watcher = setup
    source.offer(key="early", ranking=True)
    watcher.poll()
    config.deadline = utcnow() - timedelta(seconds=1)
    source.at_deadline = {"alpha": RepoUnreadableError("503")}

    watcher.poll()
    (head,) = RankingBook(store).heads_at_deadline()
    assert head.revision == "early".ljust(40, "0") and head.unreadable == "503"
    entry = RankingBook(store).get("alpha")
    assert entry is not None and entry.final and entry.key == "early"


def test_a_head_is_kept_as_soon_as_it_is_read_though_the_poll_runs_into_the_deadline(
    setup,
) -> None:
    config, store, _, source, _, watcher = setup
    source.offer(key="early", ranking=True)
    watcher.poll()
    source.offers.clear()
    source.offer(key="newer", ranking=True)
    # Inspecting the rest of the poll is slow, and the deadline passes meanwhile.
    source.on_candidates = lambda: setattr(config, "deadline", utcnow() - timedelta(seconds=1))
    watcher.poll()
    source.on_candidates = None
    source.at_deadline = {"alpha": RepoUnreadableError("503")}

    watcher.poll()
    assert RankingBook(store).get("alpha").key == "newer"


def test_a_failed_write_of_the_heads_keeps_what_was_read_at_the_deadline(
    setup, monkeypatch
) -> None:
    """Reading the Hub again on the next try would take what was pushed since."""
    config, store, _, source, _, watcher = setup
    config.deadline = utcnow() - timedelta(seconds=1)
    source.at_deadline = {"alpha": _ranking_candidate(source, "final")}
    book = watcher._ranking
    real_record = book.record_heads_at_deadline

    def full_disk(heads):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(book, "record_heads_at_deadline", full_disk)
    with pytest.raises(OSError):
        watcher.poll()
    source.at_deadline = {"alpha": _ranking_candidate(source, "pushed-late")}
    monkeypatch.setattr(book, "record_heads_at_deadline", real_record)

    watcher.poll()
    assert [head.revision for head in book.heads_at_deadline()] == ["final".ljust(40, "0")]
    assert book.get("alpha").key == "final"


def test_the_head_seen_before_the_deadline_survives_a_restart(setup) -> None:
    config, store, queue, source, fetcher, watcher = setup
    source.offer(key="early", ranking=True)
    watcher.poll()
    restarted = SubmissionWatcher(config, store, queue, source, WeightsCache(store, fetcher, max_gb=0))
    config.deadline = utcnow() - timedelta(seconds=1)
    source.at_deadline = {"alpha": RepoUnreadableError("503")}

    restarted.poll()
    assert RankingBook(store).get("alpha").key == "early"


def test_a_team_never_seen_and_unreadable_at_the_deadline_is_refused_at_once(setup) -> None:
    """Asking again cannot help: the commit is fixed at the deadline, and there is none."""
    config, store, _, source, _, watcher = setup
    config.deadline = utcnow() - timedelta(seconds=1)
    source.at_deadline = {"alpha": RepoUnreadableError("503")}
    watcher.poll()
    book = RankingBook(store)
    assert book.is_final
    entry = book.get("alpha")
    assert entry.status is RankingStatus.REJECTED and entry.error_code is ErrorCode.REPO_UNREADABLE


def _push_once_the_heads_are_recorded(book: RankingBook, source: FakeSource) -> None:
    """Stands for a poll still busy when the deadline passes, while the team pushes."""
    for _ in range(100):
        if book.heads_at_deadline() is not None:
            break
        time.sleep(0.05)
    source.at_deadline = {"alpha": _ranking_candidate(source, "pushed-late")}


@pytest.mark.parametrize("busy_in", ["on_fetch", "on_candidates"])
def test_the_heads_are_recorded_at_the_deadline_while_a_poll_is_still_busy(setup, busy_in) -> None:
    """A long transfer, or a slow inspection of new heads, that runs past the deadline
    must not leave the heads open to whatever teams push meanwhile."""
    config, store, _, source, _, watcher = setup
    config.deadline = utcnow() + timedelta(seconds=0.3)
    source.at_deadline = {"alpha": _ranking_candidate(source, "final")}
    book = RankingBook(store)
    source.offer(key="s1")
    setattr(source, busy_in, lambda: _push_once_the_heads_are_recorded(book, source))

    recorder = threading.Thread(target=watcher._record_heads_when_due)
    recorder.start()
    watcher.poll()
    recorder.join(timeout=5)

    assert [head.revision for head in book.heads_at_deadline()] == ["final".ljust(40, "0")]
    setattr(source, busy_in, None)
    watcher.poll()
    assert book.get("alpha").key == "final"


def test_a_main_that_does_not_answer_in_time_falls_back_to_the_head_seen_before(
    setup, monkeypatch
) -> None:
    """The Hub client has no timeout: one hung read must not hold the others, or the
    record, past the deadline."""
    config, store, _, source, _, watcher = setup
    monkeypatch.setattr(watcher_module, "HEAD_READ_TIMEOUT_S", 0.2)
    source.registered = [ALPHA, HubTeam(team_id="beta", repo="beta-org/sub")]
    source.offer(key="early", ranking=True)
    watcher.poll()
    config.deadline = utcnow() - timedelta(seconds=1)
    source.at_deadline = {
        "alpha": _ranking_candidate(source, "late"),
        "beta": _ranking_candidate(source, "b", team_id="beta"),
    }
    hung = threading.Event()
    source.on_head_at = lambda team: hung.wait(10) if team.team_id == "alpha" else None

    started = time.monotonic()
    watcher.poll()
    hung.set()
    assert time.monotonic() - started < 5
    alpha, beta = RankingBook(store).heads_at_deadline()
    assert alpha.revision == "early".ljust(40, "0") and "no answer within" in alpha.unreadable
    assert beta.revision == "b".ljust(40, "0") and beta.unreadable is None


def test_one_team_s_check_failing_does_not_hold_back_the_others(setup, monkeypatch) -> None:
    config, store, _, source, _, watcher = setup
    source.registered = [ALPHA, HubTeam(team_id="beta", repo="beta-org/sub")]
    source.at_deadline = {
        "alpha": _ranking_candidate(source, "a"),
        "beta": _ranking_candidate(source, "b", team_id="beta"),
    }
    config.deadline = utcnow() - timedelta(seconds=1)
    real = source.ranking_at

    def ranking_at(team, revision):
        if team.team_id == "alpha":
            raise RecursionError("a bug of ours")
        return real(team, revision)

    monkeypatch.setattr(source, "ranking_at", ranking_at)
    watcher.poll()
    book = RankingBook(store)
    assert not book.is_final and book.get("alpha") is None
    assert book.get("beta").final


def test_a_reference_that_never_resolves_is_recorded_as_refused_after_the_retries(setup) -> None:
    """A manifest naming an image commit that does not exist reads as unreadable on the
    Hub forever; it must not hold every other team's ranking evaluation back."""
    config, store, _, source, _, watcher = setup
    beta = HubTeam(team_id="beta", repo="beta-org/sub")
    source.registered = [ALPHA, beta]
    bad = replace(
        _ranking_candidate(source, "bad"),
        rejection=Rejected(ErrorCode.REPO_UNREADABLE, "no such commit"),
    )
    source.at_deadline = {"alpha": bad, "beta": RepoUnreadableError("gone")}

    config.deadline = utcnow() - timedelta(minutes=10)
    watcher.poll()
    assert not RankingBook(store).is_final  # still within the retries

    config.deadline = utcnow() - timedelta(seconds=RETRY_S + 60)
    watcher.poll()
    book = RankingBook(store)
    assert book.is_final
    for team_id in ("alpha", "beta"):
        entry = book.get(team_id)
        assert entry is not None and entry.final and entry.status is RankingStatus.REJECTED
        assert entry.error_code is ErrorCode.REPO_UNREADABLE


def test_the_watcher_wakes_at_the_deadline(setup) -> None:
    config, _, _, source, _, watcher = setup
    source.poll_interval_s = 60.0
    config.deadline = utcnow() + timedelta(seconds=5)
    assert 0 < watcher._until_next_poll_s() <= 5
    config.deadline = utcnow() - timedelta(seconds=5)
    assert watcher._until_next_poll_s() == 60.0


def test_the_board_shows_only_how_the_check_came_out(setup) -> None:
    config, store, _, source, _, watcher = setup
    save_registered_teams(store, [ALPHA])
    source.offer(key="r1", ranking=True, rejection=Rejected(ErrorCode.FOREIGN_REPO, "/secret/path"))
    watcher.poll()
    team = build_snapshot(config, store)["queue"]["teams"][0]
    assert team["ranking"]["status"] == "rejected"
    assert team["ranking"]["error_code"] == "foreign_repo"
    assert "/secret/path" not in json.dumps(team)


# --- queueing the ranking runs ------------------------------------------------


def _final_book(tmp_path, source: FakeSource, setup_watcher) -> EvalStore:
    config, store, _, _, _, watcher = setup_watcher
    save_registered_teams(store, source.registered)
    config.deadline = utcnow() - timedelta(seconds=1)
    watcher.poll()
    return store


def test_after_the_deadline_each_accepted_team_is_queued_once(setup, tmp_path) -> None:
    _, _, _, source, _, _ = setup
    source.registered = [ALPHA, HubTeam(team_id="beta", repo="beta-org/sub"),
                         HubTeam(team_id="gamma", repo="gamma-org/sub")]
    rejected = _ranking_candidate(source, "bad")
    source.at_deadline = {
        "alpha": _ranking_candidate(source, "good"),
        "beta": replace(rejected, team_id="beta", rejection=Rejected(ErrorCode.IMAGE_TOO_LARGE, "big")),
    }
    interim = _final_book(tmp_path, source, setup)

    ranking = EvalStore(tmp_path / "ranking")
    ranking.ensure_layout()
    queue = FileQueue(ranking)
    run = lambda **kw: queue_ranking_runs(  # noqa: E731
        RankingBook(interim), interim, ranking, queue, source,
        WeightsCache(ranking, FakeFetcher(), max_gb=0), **kw,
    )
    outcome = run()
    assert outcome["alpha"] == "queued as alpha__good"
    assert outcome["beta"].startswith("rejected")
    assert outcome["gamma"] == "no ranking submission"
    assert [job.job_id for job in queue.pending_jobs()] == ["alpha__good"]
    assert json.loads(next(ranking.results.glob("beta__*.json")).read_text())["status"] == "rejected"

    # Never twice at once, even when asked to run again.
    assert run(again=True)["alpha"] == "already waiting or running"
    claimed = queue.claim("w1")
    assert claimed is not None
    assert run(again=True)["alpha"] == "already waiting or running"
    ranking.publish_result(
        RunResult(run_id="alpha__good__1", job_id="alpha__good", team_id="alpha",
                  submission_id="good", status=RunStatus.FAILED)
    )
    queue.finish(claimed, RunStatus.FAILED)
    assert run()["alpha"] == "already run"
    assert run(team_ids={"alpha"}, again=True)["alpha"] == "queued as alpha__good"


def test_nothing_is_queued_before_the_submissions_are_recorded(tmp_path) -> None:
    store = EvalStore(tmp_path)
    store.ensure_layout()
    with pytest.raises(SystemExit, match="not been recorded"):
        queue_ranking_runs(RankingBook(store), store, store, FileQueue(store), FakeSource(store),
                           WeightsCache(store, FakeFetcher(), max_gb=0))


# --- from the Hub -------------------------------------------------------------


@dataclass
class _Commit:
    commit_id: str
    created_at: datetime


def _hub(tmp_path, monkeypatch, files: dict[str, bytes]) -> tuple[FakeHub, HubSource]:
    fake = FakeHub(files=files)
    monkeypatch.setattr(hubsource, "hf_hub_download", fake.download)
    store = EvalStore(tmp_path)
    store.ensure_layout()
    team = HubTeam(team_id="team-alpha", repo="team-alpha-org/capx-submission")
    return fake, HubSource(HubConfig(), store, lambda: [team], api=fake)


def test_the_hub_offers_the_ranking_file_beside_the_interim_one(tmp_path, monkeypatch) -> None:
    fake, source = _hub(tmp_path, monkeypatch, {RANKING_MANIFEST_NAME: manifest_bytes(), "agent.sif": IMAGE})
    interim, ranking = source.candidates()
    assert interim.rejection is not None  # no submission.yaml
    assert ranking.manifest_name == RANKING_MANIFEST_NAME and ranking.manifest is not None
    assert "agent.sif" not in fake.downloads


def test_a_repo_without_a_ranking_file_offers_an_absent_one(tmp_path, monkeypatch) -> None:
    _, source = _hub(tmp_path, monkeypatch, {"submission.yaml": manifest_bytes(), "agent.sif": IMAGE})
    assert source.candidates()[1].absent


def test_the_deadline_s_commit_is_the_last_one_made_by_it(tmp_path, monkeypatch) -> None:
    fake, source = _hub(tmp_path, monkeypatch, {})
    deadline = utcnow()
    fake.list_repo_commits = lambda repo, **_: [
        _Commit("c" * 40, deadline + timedelta(seconds=3)),  # pushed while we looked
        _Commit("b" * 40, deadline - timedelta(minutes=1)),
        _Commit("a" * 40, deadline - timedelta(hours=1)),
    ]
    assert source.head_at(source.teams()[0], deadline) == "b" * 40


def test_a_ranking_manifest_yaml_cannot_construct_is_refused_and_the_round_is_final(
    tmp_path, monkeypatch
) -> None:
    """PyYAML raises ValueError, not YAMLError, for a date like 2026-13-01; that must not
    leave the recording retrying for good, and every other team without its run."""
    fake, source = _hub(tmp_path, monkeypatch, {RANKING_MANIFEST_NAME: b"team_id: team-alpha\nnotes: 2026-13-01\n"})
    config = QueueConfig(root=str(tmp_path), master_seed=1, deadline=utcnow() - timedelta(seconds=1))
    fake.list_repo_commits = lambda repo, **_: [_Commit("b" * 40, config.deadline - timedelta(minutes=1))]
    store = EvalStore(tmp_path)
    watcher = SubmissionWatcher(config, store, FileQueue(store), source, WeightsCache(store, FakeFetcher(), max_gb=0))

    watcher.poll()
    book = RankingBook(store)
    assert book.is_final
    entry = book.get("team-alpha")
    assert entry.status is RankingStatus.REJECTED and entry.error_code is ErrorCode.INVALID_MANIFEST


def test_a_repo_with_no_commit_by_the_deadline_has_no_head(tmp_path, monkeypatch) -> None:
    fake, source = _hub(tmp_path, monkeypatch, {})
    deadline = utcnow()
    fake.list_repo_commits = lambda repo, **_: [_Commit("c" * 40, deadline + timedelta(seconds=1))]
    assert source.head_at(source.teams()[0], deadline) is None


def test_the_ranking_submission_is_read_at_the_commit_it_is_given(tmp_path, monkeypatch) -> None:
    fake, source = _hub(tmp_path, monkeypatch, {RANKING_MANIFEST_NAME: manifest_bytes(), "agent.sif": IMAGE})
    team = source.teams()[0]
    candidate = source.ranking_at(team, "b" * 40)
    assert candidate.manifest is not None
    assert (team.repo, "b" * 40) in fake.reads


def test_a_ranking_submission_is_named_by_its_manifest_s_values(tmp_path, monkeypatch) -> None:
    """The interim rule (PRELIM_RULEBOOK.md §6.4): the run's job id changes with any value
    of the file, a weights revision included, and not with how it is written."""
    fake, source = _hub(tmp_path, monkeypatch, {RANKING_MANIFEST_NAME: manifest_bytes(), "agent.sif": IMAGE})
    team = source.teams()[0]
    weights = {"repo": "team-alpha-org/policy", "revision": "e" * 40, "path": "/models/policy"}

    def key_of(manifest: bytes) -> str:
        fake.files[RANKING_MANIFEST_NAME] = manifest
        return source.ranking_at(team, "b" * 40).key

    first = key_of(manifest_bytes(weights=[weights]))
    reordered = dict(reversed(list(yaml.safe_load(manifest_bytes(weights=[weights])).items())))
    assert key_of(b"# reformatted\n" + yaml.safe_dump(reordered, indent=4, sort_keys=False).encode()) == first
    assert key_of(manifest_bytes(weights=[{**weights, "revision": "f" * 40}])) != first
    assert key_of(manifest_bytes(weights=[weights], submission_id="again")) != first


def test_a_commit_that_cannot_be_read_is_reported_as_such(tmp_path, monkeypatch) -> None:
    fake, source = _hub(tmp_path, monkeypatch, {})
    fake.missing = True
    with pytest.raises(RepoUnreadableError, match="cannot read"):
        source.ranking_at(source.teams()[0], "b" * 40)


def test_the_heads_are_recorded_while_a_poll_is_reading_the_registry(tmp_path, monkeypatch) -> None:
    """Reading the registry is a git fetch. One a poll has in progress at the deadline
    must not hold the recording back: it uses the teams as last read."""
    fake = FakeHub(files={})
    monkeypatch.setattr(hubsource, "hf_hub_download", fake.download)
    store = EvalStore(tmp_path)
    store.ensure_layout()
    team = HubTeam(team_id="team-alpha", repo="team-alpha-org/capx-submission")
    reading, release = threading.Event(), threading.Event()
    reads: list[int] = []

    def registry() -> list[HubTeam]:
        reads.append(1)
        if len(reads) > 1:
            reading.set()
            release.wait(10)
        return [team]

    source = HubSource(HubConfig(), store, registry, api=fake)
    source.teams()
    config = QueueConfig(root=str(tmp_path), master_seed=1, deadline=utcnow() + timedelta(seconds=0.3))
    fake.list_repo_commits = lambda repo, **_: [_Commit("b" * 40, config.deadline - timedelta(minutes=1))]
    watcher = SubmissionWatcher(config, store, FileQueue(store), source, WeightsCache(store, FakeFetcher(), max_gb=0))

    poll = threading.Thread(target=watcher.poll)
    poll.start()
    assert reading.wait(5)
    recorder = threading.Thread(target=watcher._record_heads_when_due)
    recorder.start()
    recorder.join(5)
    recorded = RankingBook(store).heads_at_deadline()
    release.set()
    poll.join(5)
    assert recorded is not None and [head.revision for head in recorded] == ["b" * 40]
