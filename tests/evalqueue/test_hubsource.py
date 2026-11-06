"""Intake from HuggingFace: what a commit has to look like to become a job.

The Hub is faked. What is worth pinning is not that ``huggingface_hub`` works but
that the source spends the Hub's expensive operations only when it has to: one
``repo_info`` per team per poll, the manifest once per new sha, and the image only
after the watcher has accepted the submission.
"""

from __future__ import annotations

import errno
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import pytest
import yaml

from capx.evalqueue import hubsource
from capx.evalqueue.config import HubTeam, QueueConfig
from capx.evalqueue.hubsource import (
    MAX_MANIFEST_BYTES,
    Candidate,
    HubSource,
    load_manifest,
    manifest_key,
)
from capx.evalqueue.models import Rejected, RunStatus
from capx.evalqueue.queue import FileQueue
from capx.evalqueue.store import EvalStore, utcnow
from capx.evalqueue.watcher import RETRY_S, SubmissionWatcher
from capx.evalqueue.weights import WeightsCache

from .test_watcher import FakeFetcher, rewind_retries

IMAGE = b"pretend this is nine gigabytes of squashfs"
IMAGE_SHA = hashlib.sha256(IMAGE).hexdigest()
IMAGE_REVISION = "9f8e7d6c5b4a39281706f5e4d3c2b1a098765432"
"""The earlier commit the manifest pins its image to."""


@dataclass
class _Response:
    """The little of a response that ``HfHubHTTPError`` reads on the way up."""

    status_code: int = 404
    headers: dict = field(default_factory=dict)
    request: object = None


def not_found(message: str) -> hubsource.HfHubHTTPError:
    return hubsource.HfHubHTTPError(message, response=_Response())


def hub_error(status: int) -> hubsource.HfHubHTTPError:
    return hubsource.HfHubHTTPError(f"{status} from the Hub", response=_Response(status))


@dataclass
class _Lfs:
    sha256: str


@dataclass
class _Sibling:
    rfilename: str
    size: int
    lfs: _Lfs | None = None


@dataclass
class _Info:
    sha: str
    siblings: list[_Sibling]


@dataclass
class FakeHub:
    """One repo's worth of Hub, plus counters for what it was asked to do."""

    files: dict[str, bytes] = field(default_factory=dict)
    sha: str = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
    orgs: list[dict] = field(default_factory=lambda: [{"name": "team-alpha-org"}])
    auth: dict = field(default_factory=dict)
    missing: bool = False
    failing: dict[str, Exception] = field(default_factory=dict)
    """What a read of a repo or a download of a file raises instead."""
    repo_info_calls: int = 0
    reads: list[tuple[str, str | None]] = field(default_factory=list)
    downloads: list[str] = field(default_factory=list)

    def repo_info(self, repo_id, *, repo_type=None, revision=None, files_metadata=False):
        self.repo_info_calls += 1
        self.reads.append((repo_id, revision))
        if self.missing:
            raise not_found(f"404 for {repo_id}")
        if repo_id in self.failing:
            raise self.failing[repo_id]
        siblings = [
            _Sibling(name, len(payload), _Lfs(hashlib.sha256(payload).hexdigest()))
            for name, payload in self.files.items()
        ]
        return _Info(sha=self.sha, siblings=siblings)

    def whoami(self):
        return {"name": "organizer", "orgs": self.orgs, "auth": self.auth}

    def download(self, repo_id, filename, *, repo_type=None, revision=None, token=None,
                 local_dir=None):
        self.downloads.append(filename)
        if filename in self.failing:
            raise self.failing[filename]
        if filename not in self.files:
            raise not_found(f"no {filename} in {repo_id}")
        target = Path(local_dir or "/tmp") / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])
        return str(target)


def manifest_bytes(**overrides) -> bytes:
    document = {
        "team_id": "team-alpha",
        "submission_id": "week-3",
        "image": {
            "repo": "team-alpha-org/capx-submission",
            "revision": IMAGE_REVISION,
            "path": "agent.sif",
        },
        **overrides,
    }
    return yaml.safe_dump(document).encode()


def job_of(manifest: bytes) -> str:
    return f"team-alpha__{manifest_key(load_manifest(manifest))}"


JOB = job_of(manifest_bytes())
WEIGHTS = {"repo": "team-alpha-org/policy", "revision": "e" * 40, "path": "/models/policy"}
NEXT_SHA = "0f1e2d3c4b5a69788796a5b4c3d2e1f001234567"


@pytest.fixture
def hub(tmp_path, monkeypatch):
    fake = FakeHub(files={"submission.yaml": manifest_bytes(), "agent.sif": IMAGE})
    monkeypatch.setattr(hubsource, "hf_hub_download", fake.download)

    config = QueueConfig(root=str(tmp_path), master_seed=1)
    store = EvalStore(tmp_path)
    store.ensure_layout()
    queue = FileQueue(store)
    teams = [HubTeam(team_id="team-alpha", repo="team-alpha-org/capx-submission")]
    source = HubSource(config.hub, store, lambda: teams, api=fake)
    watcher = SubmissionWatcher(
        config, store, queue, source, WeightsCache(store, FakeFetcher(), max_gb=0)
    )
    return fake, config, store, queue, source, watcher


def only_result(store: EvalStore) -> dict:
    results = list(store.results.glob("*.json"))
    assert len(results) == 1, [p.name for p in results]
    return json.loads(results[0].read_text())


def test_a_new_manifest_becomes_a_job_named_after_its_values(hub) -> None:
    fake, _, store, queue, _, watcher = hub
    assert watcher.poll() == [JOB]

    job = queue.pending_jobs()[0]
    assert job.hub_repo == "team-alpha-org/capx-submission"
    assert job.hub_revision == fake.sha
    assert job.submission_id == "week-3"
    assert Path(job.image_path).read_bytes() == IMAGE
    assert Path(job.image_path).parent == store.incoming / "team-alpha" / JOB.split("__")[1]


def test_an_unchanged_head_costs_one_call_and_no_transfer(hub) -> None:
    fake, _, _, _, _, watcher = hub
    watcher.poll()
    before = len(fake.downloads)

    for _ in range(3):
        assert watcher.poll() == []
    # One per poll, plus one for the image's commit when the head was new.
    assert fake.repo_info_calls == 5
    assert len(fake.downloads) == before


def test_a_commit_that_leaves_the_manifest_alone_runs_nothing(hub) -> None:
    """A README edit, or a new image pushed but not named, is a new commit but not a
    new submission."""
    fake, _, store, queue, _, watcher = hub
    assert watcher.poll() == [JOB]
    queue.finish(queue.claim("w0"), RunStatus.COMPLETED)
    downloads = len(fake.downloads)

    fake.sha = NEXT_SHA
    fake.files["README.md"] = b"now with a README"
    fake.files["agent.sif"] = b"a rebuilt image"
    assert watcher.poll() == []
    assert list(store.results.glob("*.json")) == []
    assert len(fake.downloads) == downloads + 1  # the new manifest, not the image


def test_a_manifest_written_differently_is_the_same_submission(hub) -> None:
    fake, _, store, queue, _, watcher = hub
    assert watcher.poll() == [JOB]
    queue.finish(queue.claim("w0"), RunStatus.COMPLETED)

    fake.sha = NEXT_SHA
    fake.files["submission.yaml"] = (
        b"# week 3, with comments, other indentation and the keys reordered\n"
        b"image:\n"
        b"    path: agent.sif   # the same file\n"
        b"    revision: " + IMAGE_REVISION.encode() + b"\n"
        b"    repo: team-alpha-org/capx-submission\n"
        b"submission_id:   week-3\n"
        b"workdir: /workspace/capx  # the default, written out\n"
        b"notes: ''\n"
        b"team_id: team-alpha\n"
    )
    assert watcher.poll() == []
    assert list(store.results.glob("*.json")) == []


def test_a_new_weights_revision_is_a_new_submission(hub) -> None:
    """The normal way to iterate: fine-tune, push the weights, name the new commit."""
    fake, _, _, queue, _, watcher = hub
    fake.files["submission.yaml"] = manifest_bytes(weights=[WEIGHTS])
    first = job_of(fake.files["submission.yaml"])
    assert watcher.poll() == [first]
    queue.finish(queue.claim("w0"), RunStatus.COMPLETED)

    fake.sha = NEXT_SHA
    fake.files["submission.yaml"] = manifest_bytes(weights=[{**WEIGHTS, "revision": "f" * 40}])
    second = job_of(fake.files["submission.yaml"])
    assert second != first
    assert watcher.poll() == [second]
    assert queue.pending_jobs()[0].weights[0].revision == "f" * 40


@pytest.mark.parametrize(
    "change",
    [
        {"submission_id": "week-3-again"},
        {"notes": "the same image and weights, run again"},
        {"env": {"CAPX_TEMPERATURE": "0.2"}},
        {"entrypoint": ["python", "-m", "agent"]},
        {"workdir": "/workspace/agent"},
        {"image": {"repo": "team-alpha-org/capx-submission", "revision": "1" * 40, "path": "agent.sif"}},
    ],
    ids=["submission_id", "notes", "env", "entrypoint", "workdir", "image"],
)
def test_any_changed_value_is_a_new_submission(hub, change) -> None:
    fake, _, _, queue, _, watcher = hub
    assert watcher.poll() == [JOB]
    queue.finish(queue.claim("w0"), RunStatus.COMPLETED)  # the per-team cap would defer it otherwise

    fake.sha = NEXT_SHA
    fake.files["submission.yaml"] = manifest_bytes(**change)
    changed = job_of(fake.files["submission.yaml"])
    assert changed != JOB
    assert watcher.poll() == [changed]


def test_a_manifest_changed_back_is_not_run_again(hub) -> None:
    fake, _, store, queue, _, watcher = hub
    assert watcher.poll() == [JOB]
    queue.finish(queue.claim("w0"), RunStatus.COMPLETED)
    fake.sha = NEXT_SHA
    fake.files["submission.yaml"] = manifest_bytes(submission_id="week-4")
    assert watcher.poll() == [job_of(fake.files["submission.yaml"])]
    queue.finish(queue.claim("w0"), RunStatus.COMPLETED)

    fake.sha = "1" * 40
    fake.files["submission.yaml"] = manifest_bytes()
    assert watcher.poll() == []
    assert queue.pending_jobs() == [] and list(store.results.glob("*.json")) == []


def test_an_image_refused_for_its_manifest_runs_once_the_manifest_is_fixed(hub) -> None:
    """The team should not have to rebuild 10 GB to correct a number in a YAML file."""
    fake, _, store, _, _, watcher = hub
    fake.files["submission.yaml"] = manifest_bytes(image_sha256="0" * 64)
    assert watcher.poll() == []
    assert "sha256 mismatch" in only_result(store)["error"]

    fake.sha = NEXT_SHA
    fake.files["submission.yaml"] = manifest_bytes(image_sha256=IMAGE_SHA)
    assert watcher.poll() == [job_of(fake.files["submission.yaml"])]


def test_a_refused_manifest_is_looked_at_again_when_pushed_again(hub) -> None:
    """What refused it can lie outside the manifest: here, a repo the organizer account
    could not read until the team gave it access."""
    fake, _, store, _, _, watcher = hub
    fake.files["submission.yaml"] = manifest_bytes(
        image={"repo": "team-alpha-org/images", "revision": IMAGE_REVISION, "path": "agent.sif"}
    )
    fake.failing["team-alpha-org/images"] = hub_error(404)
    assert watcher.poll() == []
    assert only_result(store)["error_code"] == "repo_unreadable"

    del fake.failing["team-alpha-org/images"]
    assert watcher.poll() == []  # the same commit is not looked at again
    fake.sha = NEXT_SHA
    assert watcher.poll() == [job_of(fake.files["submission.yaml"])]


@pytest.mark.parametrize(
    "manifest",
    [
        None,
        b"team_id: [unterminated",
        manifest_bytes(submission_id="../up"),
        manifest_bytes(
            image={"repo": "team-alpha-org/capx-submission", "revision": IMAGE_REVISION, "path": "gone.sif"}
        ),
    ],
    ids=["missing", "unparseable", "invalid", "image-missing"],
)
def test_a_rejected_commit_is_reported_once_not_every_poll(hub, manifest) -> None:
    fake, _, store, _, _, watcher = hub
    if manifest is None:
        del fake.files["submission.yaml"]
    else:
        fake.files["submission.yaml"] = manifest
    for _ in range(3):
        watcher.poll()
    assert len(list(store.results.glob("*.json"))) == 1


def test_a_manifest_claiming_another_team_is_refused(hub) -> None:
    fake, _, store, _, _, watcher = hub
    fake.files["submission.yaml"] = manifest_bytes(team_id="team-beta")

    assert watcher.poll() == []
    document = only_result(store)
    assert document["team_id"] == "team-alpha"  # the org decides, not the manifest
    assert "registered to 'team-alpha'" in document["error"]
    assert "agent.sif" not in fake.downloads


def test_a_repo_with_no_manifest_is_refused_without_asking_for_one(hub) -> None:
    """The commonest state of a registered repo before a team has submitted. The
    tree from ``repo_info`` already answers it, so no request is spent on a 404."""
    fake, _, store, _, _, watcher = hub
    del fake.files["submission.yaml"]

    assert watcher.poll() == []
    assert only_result(store)["error"] == (
        "submission.yaml is not in team-alpha-org/capx-submission@a1b2c3d4e5f6"
    )
    assert fake.downloads == []


def test_an_image_missing_from_the_revision_is_refused(hub) -> None:
    fake, _, store, _, _, watcher = hub
    del fake.files["agent.sif"]

    assert watcher.poll() == []
    assert (
        f"is not in team-alpha-org/capx-submission@{IMAGE_REVISION[:12]}"
        in only_result(store)["error"]
    )


def test_the_image_is_read_at_the_commit_the_manifest_pins(hub) -> None:
    fake, *_, watcher = hub
    assert watcher.poll() == [JOB]
    assert ("team-alpha-org/capx-submission", IMAGE_REVISION) in fake.reads


def test_an_image_from_another_repo_of_the_team_s_org_is_accepted(hub) -> None:
    fake, *_, watcher = hub
    fake.files["submission.yaml"] = manifest_bytes(
        image={"repo": "Team-Alpha-Org/images", "revision": IMAGE_REVISION, "path": "agent.sif"}
    )
    assert watcher.poll() == [job_of(fake.files["submission.yaml"])]
    assert ("Team-Alpha-Org/images", IMAGE_REVISION) in fake.reads


def test_an_image_from_another_org_is_refused_before_it_is_read(hub) -> None:
    fake, _, store, _, _, watcher = hub
    fake.files["submission.yaml"] = manifest_bytes(
        image={"repo": "team-beta-org/capx-submission", "revision": IMAGE_REVISION, "path": "agent.sif"}
    )
    assert watcher.poll() == []
    assert "the image must come from your own org" in only_result(store)["error"]
    assert all(repo != "team-beta-org/capx-submission" for repo, _ in fake.reads)
    assert "agent.sif" not in fake.downloads


def test_a_declared_hash_is_checked_against_the_hub_before_any_transfer(hub) -> None:
    fake, _, store, _, _, watcher = hub
    fake.files["submission.yaml"] = manifest_bytes(image_sha256="b" * 64)

    assert watcher.poll() == []
    assert "sha256 mismatch" in only_result(store)["error"]
    assert "agent.sif" not in fake.downloads


def test_a_matching_hash_passes(hub) -> None:
    fake, _, _, _, _, watcher = hub
    fake.files["submission.yaml"] = manifest_bytes(image_sha256=IMAGE_SHA)
    assert watcher.poll() == [job_of(fake.files["submission.yaml"])]


def test_an_oversized_image_is_refused_before_any_transfer(hub) -> None:
    fake, _, store, _, _, watcher = hub
    watcher._eval_config.submission.max_image_gb = 0.0

    assert watcher.poll() == []
    assert "over the" in only_result(store)["error"]
    assert "agent.sif" not in fake.downloads


def test_an_unreadable_repo_is_logged_not_published_as_a_rejection(hub, caplog) -> None:
    """There is nothing in a submission for a team to fix when the repo cannot be
    read at all — the invite has not been accepted, or the repo does not exist."""
    fake, _, store, _, _, watcher = hub
    fake.missing = True

    assert watcher.poll() == []
    assert list(store.results.glob("*.json")) == []
    assert "cannot read" in caplog.text


def test_a_repo_that_cannot_be_read_is_reported_once_not_every_poll(hub, caplog) -> None:
    fake, _, _, _, _, watcher = hub
    fake.missing = True
    for _ in range(5):
        watcher.poll()
    assert caplog.text.count("cannot read") == 1


def test_an_unparseable_manifest_is_refused_with_the_team_named(hub) -> None:
    """Unlike the directory source, which has no id to file the rejection under: the
    org already told us whose submission this is."""
    fake, _, store, _, _, watcher = hub
    fake.files["submission.yaml"] = b"team_id: [unterminated"

    assert watcher.poll() == []
    document = only_result(store)
    assert document["team_id"] == "team-alpha"
    assert document["status"] == "rejected"


def test_preflight_warns_about_an_expiring_token(hub) -> None:
    fake, _, _, _, source, _ = hub
    fake.auth = {"type": "oauth", "expiresAt": "2026-09-05T08:49:16.000Z"}
    assert any("expires 2026-09-05" in w for w in source.preflight())


def test_preflight_names_orgs_on_both_sides_of_the_registry(hub) -> None:
    fake, _, _, _, source, _ = hub
    fake.orgs = [{"name": "team-gamma-org"}]

    warnings = source.preflight()
    assert any("team-gamma-org" in w and "no team is registered" in w for w in warnings)
    assert any("team-alpha-org" in w and "not a member" in w for w in warnings)


def test_preflight_is_quiet_when_the_registry_matches(hub) -> None:
    _, _, _, _, source, _ = hub
    assert source.preflight() == []


def test_a_registry_entry_that_could_escape_the_tree_is_refused() -> None:
    """``team_id`` is an organizer's typo away from being a path component. Same
    rule as the manifest's, applied at the other end."""
    with pytest.raises(ValueError, match="team_id"):
        HubTeam(team_id="../etc", repo="org/repo")
    with pytest.raises(ValueError, match="must be a HuggingFace '<owner>/<name>'"):
        HubTeam(team_id="team-alpha", repo="no-slash-here")


def test_fetch_refuses_a_candidate_it_never_inspected(hub) -> None:
    _, _, _, _, source, _ = hub
    with pytest.raises(Rejected, match="never inspected"):
        source.fetch(Candidate(team_id="team-alpha", key="deadbeef", origin="x"))


# --- one team's manifest against everyone's intake --------------------------


class TwoTeamHub:
    """Team bravo's repo holds whatever a test puts in it; team alpha's is fine."""

    def __init__(self, bravo_files: dict[str, bytes]) -> None:
        self.alpha = FakeHub(files={"submission.yaml": manifest_bytes(), "agent.sif": IMAGE})
        self.bravo = FakeHub(files=bravo_files, sha="b" * 40)

    def _pick(self, repo_id: str) -> FakeHub:
        return self.bravo if repo_id.startswith("bravo-org/") else self.alpha

    def repo_info(self, repo_id, **kw):
        return self._pick(repo_id).repo_info(repo_id, **kw)

    def download(self, repo_id, filename, **kw):
        return self._pick(repo_id).download(repo_id, filename, **kw)


@pytest.fixture
def two_teams(tmp_path, monkeypatch):
    def make(bravo_files: dict[str, bytes]):
        hub = TwoTeamHub(bravo_files)
        monkeypatch.setattr(hubsource, "hf_hub_download", hub.download)
        config = QueueConfig(root=str(tmp_path), master_seed=1)
        store = EvalStore(tmp_path)
        store.ensure_layout()
        queue = FileQueue(store)
        teams = [
            HubTeam(team_id="bravo", repo="bravo-org/capx-submission"),
            HubTeam(team_id="team-alpha", repo="team-alpha-org/capx-submission"),
        ]
        source = HubSource(config.hub, store, lambda: teams, api=hub)
        watcher = SubmissionWatcher(
            config, store, queue, source, WeightsCache(store, FakeFetcher(), max_gb=0)
        )
        return hub, store, watcher

    return make


def bravo_rejection(store: EvalStore) -> dict:
    (document,) = [
        doc
        for doc in (json.loads(p.read_text()) for p in store.results.glob("*.json"))
        if doc["team_id"] == "bravo"
    ]
    return document


def nested_aliases(levels: int) -> bytes:
    """A few hundred bytes that stand for 10**levels strings."""
    lines = ["a: &a [x,x,x,x,x,x,x,x,x,x]"]
    for level in range(1, levels):
        previous, current = chr(ord("a") + level - 1), chr(ord("a") + level)
        lines.append(f"{current}: &{current} [" + ",".join([f"*{previous}"] * 10) + "]")
    return ("\n".join(lines) + f"\nteam_id: bravo\nnotes: *{chr(ord('a') + levels - 1)}\n").encode()


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (b"team_id: bravo\nnotes: \xff\xfe\n", "is not UTF-8"),
        (b"notes: " + b"[" * 5000 + b"]" * 5000 + b"\n", "nested deeper than"),
        (nested_aliases(9), "anchors and aliases are not allowed"),
        (b"team_id: bravo\nnotes: 2026-13-01\n", "month must be in 1..12"),
        (b"team_id: bravo\nnotes: !!int 0x\n", "invalid literal for int()"),
    ],
    ids=["not-utf8", "deep", "aliases", "bad-date", "bad-int"],
)
def test_a_hostile_manifest_is_refused_without_stopping_another_team(
    two_teams, manifest, message
) -> None:
    _, store, watcher = two_teams({"submission.yaml": manifest})
    started = time.monotonic()
    assert watcher.poll() == [JOB]
    assert time.monotonic() - started < 5
    rejection = bravo_rejection(store)
    assert rejection["error_code"] == "invalid_manifest"
    assert message in rejection["error"]


def test_an_oversized_manifest_is_refused_unread(two_teams) -> None:
    hub, store, watcher = two_teams({"submission.yaml": b"#" * (MAX_MANIFEST_BYTES + 1)})
    assert watcher.poll() == [JOB]
    assert "KiB limit" in bravo_rejection(store)["error"]
    assert hub.bravo.downloads == []


def test_a_refusal_does_not_repeat_what_the_team_wrote(two_teams) -> None:
    hub, store, watcher = two_teams({"submission.yaml": manifest_bytes(team_id="bravo", notes=["y" * 1000])})
    watcher.poll()
    error = bravo_rejection(store)["error"]
    assert "notes: Input should be a valid string" in error
    assert "yyyy" not in error


def test_a_team_whose_inspection_fails_is_skipped_until_it_works(two_teams, caplog) -> None:
    """Whatever goes wrong with one team's commit, every other team's is taken in."""
    hub, store, watcher = two_teams({"submission.yaml": manifest_bytes(team_id="bravo")})
    hub.bravo.failing["submission.yaml"] = RuntimeError("a bug of ours")
    assert watcher.poll() == [JOB]
    assert "a bug of ours" in caplog.text
    assert list(store.results.glob("bravo__*.json")) == []

    del hub.bravo.failing["submission.yaml"]
    watcher.poll()
    assert "image must come from your own org" in bravo_rejection(store)["error"]


def test_a_head_is_heard_of_before_its_commit_is_inspected(two_teams) -> None:
    hub, _, watcher = two_teams({"submission.yaml": manifest_bytes(team_id="bravo")})
    hub.bravo.failing["submission.yaml"] = RuntimeError("a bug of ours")
    heard: list[tuple[str, str]] = []
    watcher._source.candidates(lambda team, sha: heard.append((team.team_id, sha)))
    assert ("bravo", "b" * 40) in heard


# --- when the Hub, or our disk, is at fault ---------------------------------


@pytest.mark.parametrize("status", [429, 500, 503])
def test_a_hub_hiccup_is_asked_again_not_published(hub, status) -> None:
    fake, _, store, queue, _, watcher = hub
    fake.failing["submission.yaml"] = hub_error(status)
    assert watcher.poll() == []
    assert list(store.results.glob("*.json")) == []

    del fake.failing["submission.yaml"]
    assert watcher.poll() == [JOB]


def test_a_hub_hiccup_during_the_transfer_is_asked_again(hub) -> None:
    fake, _, store, _, _, watcher = hub
    fake.failing["agent.sif"] = hub_error(502)
    assert watcher.poll() == []
    assert list(store.results.glob("*.json")) == []

    del fake.failing["agent.sif"]
    assert watcher.poll() == [JOB]


def test_a_hub_hiccup_that_lasts_is_published_after_the_retries(hub) -> None:
    fake, _, store, _, _, watcher = hub
    fake.failing["submission.yaml"] = hub_error(503)
    watcher.poll()
    rewind_retries(store, timedelta(seconds=RETRY_S + 1))

    watcher.poll()
    document = only_result(store)
    assert document["error_code"] == "repo_unreadable"


def test_an_image_commit_the_hub_says_is_not_there_is_refused_at_once(hub) -> None:
    fake, _, store, _, _, watcher = hub
    fake.files["submission.yaml"] = manifest_bytes(
        image={"repo": "team-alpha-org/images", "revision": IMAGE_REVISION, "path": "agent.sif"}
    )
    fake.failing["team-alpha-org/images"] = hub_error(404)
    assert watcher.poll() == []
    assert only_result(store)["error_code"] == "repo_unreadable"


def test_our_full_disk_is_not_the_team_s_rejection(hub, caplog) -> None:
    fake, _, store, _, _, watcher = hub
    fake.failing["agent.sif"] = OSError(errno.ENOSPC, "No space left on device")
    assert watcher.poll() == []
    assert list(store.results.glob("*.json")) == []
    assert "No space left" in caplog.text

    del fake.failing["agent.sif"]
    assert watcher.poll() == [JOB]
