"""Submissions that arrive as commits on HuggingFace.

Each team owns an org, pushes ``submission.yaml`` to one registered repo in it, and
invites the organizer account with ``read``. The manifest names the image by repo,
commit and path — any repo of the team's own org. The watcher polls the head of
``main`` of each registered repo (capx/evalqueue/teams.py); a sha it has not seen is
a new submission.

**What the manifest says is the identity.** ``job_id`` is ``<team_id>__<key>``, where
``key`` hashes the manifest's values (:func:`manifest_key`). A commit that leaves them
alone — a README edit, an image or weights pushed but not named — runs nothing; any
changed value — a weights revision, ``submission_id``, ``notes`` — is a new submission.
The image and the weights are named by commit sha, so the values pin everything that
runs. The commit a job came from is still recorded, as ``hub_revision``.

**``team_id`` is not self-declared.** The org that holds the repo decides, and org
membership is authenticated by HuggingFace. A manifest claiming a different team is
refused rather than quietly overridden.

``candidates`` runs every poll for every team, so it does only cheap work: one
``repo_info`` per team, plus the manifest and the image's commit when the head is new.
``fetch`` moves gigabytes and runs once, after the watcher has accepted the submission.

**A manifest is untrusted input, and one team's must not stop the rest.** It is
refused unread past ``MAX_MANIFEST_BYTES``, must be UTF-8, and is parsed without YAML
anchors and aliases, which let a few hundred bytes stand for gigabytes.

**A failed Hub request is the team's fault only when the Hub says so** (a 4xx). No
response, a 429 or a 5xx makes a ``transient`` rejection the watcher asks again about,
and an ``OSError`` of our own disk is not a rejection at all.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol

import yaml
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    HTTPError,
    LocalEntryNotFoundError,
)
from pydantic import BaseModel, ValidationError
from yaml.composer import ComposerError

from capx.evalqueue.config import HubConfig, HubTeam
from capx.evalqueue.models import (
    GB,
    ErrorCode,
    Rejected,
    SubmissionManifest,
    WeightsMount,
    describe_errors,
    hf_owner,
)
from capx.evalqueue.store import EvalStore
from capx.evalqueue.weights import RepoInfo, describe_mount

_logger = logging.getLogger(__name__)

MANIFEST_NAME = "submission.yaml"
RANKING_MANIFEST_NAME = "submission_ranking.yaml"
WATCHED_REVISION = "main"
MAX_MANIFEST_BYTES = 64 * 1024
_MAX_MANIFEST_DEPTH = 16
_KEY_LENGTH = 12

_HUB_ERRORS = (HTTPError, LocalEntryNotFoundError)
"""What ``huggingface_hub`` raises when the Hub did not answer as asked. ``HTTPError`` is
the base error of the HTTP client it uses (httpx's before 2.0, httpx2's from 2.0), so
it covers connection failures as well as ``HfHubHTTPError``. Any other ``OSError`` is
our own disk's."""


@dataclass(frozen=True)
class Candidate:
    """One submission on offer, described without moving its image.

    ``key`` is the identity the queue dedups on: ``job_id`` is ``<team_id>__<key>``. It
    is :func:`manifest_key`, or the commit's sha while there is no valid manifest.
    """

    team_id: str
    key: str
    origin: str
    """Human-readable provenance, for logs."""
    revision: str | None = None
    """The commit this offer came from. A rejection is remembered under it rather
    than under ``key``: what refused it may lie outside the manifest (an access grant,
    an invite), so pushing the same manifest again is looked at afresh. None means the
    same as ``key``."""
    manifest: SubmissionManifest | None = None
    image_gb: float = 0.0
    image_sha256: str | None = None
    """The image's LFS object id, which lets a hash mismatch be refused before the
    transfer."""
    rejection: Rejected | None = None
    """Set when the source already knows this cannot run."""
    hub_repo: str | None = None
    hub_revision: str | None = None
    manifest_name: str = MANIFEST_NAME
    """Which submission this is: an interim one, or ``RANKING_MANIFEST_NAME``. A ranking
    candidate with neither ``manifest`` nor ``rejection`` means the file is not there."""

    @property
    def absent(self) -> bool:
        return self.manifest is None and self.rejection is None


class RepoUnreadableError(RuntimeError):
    """A registered repo could not be read, so nothing can be said about it yet."""


class SubmissionSource(Protocol):
    @property
    def poll_interval_s(self) -> float: ...

    def teams(self) -> list[HubTeam]: ...

    def known_teams(self) -> list[HubTeam]:
        """The teams as last read, without reading the registry again (unless it never
        was)."""
        ...

    def candidates(
        self, note_head: Callable[[HubTeam, str], None] | None = None
    ) -> list[Candidate]:
        """Each team's interim and ranking submissions at the head of its repo. A team
        whose repo cannot be read or inspected is left out until it can. ``note_head``
        hears each team's head as soon as it is read, before it is inspected."""
        ...

    def head_at(self, team: HubTeam, deadline: datetime) -> str | None:
        """The commit the team's ``main`` stood at at ``deadline``: the newest one in its
        history made by then; None if there is none. Raises :class:`RepoUnreadableError`."""
        ...

    def ranking_at(self, team: HubTeam, revision: str) -> Candidate:
        """The team's ranking submission in commit ``revision`` of its repo. Raises
        :class:`RepoUnreadableError`."""
        ...

    def fetch(self, candidate: Candidate) -> Path:
        """Put the image on local disk and return its path. Raises :class:`Rejected`."""
        ...

    def preflight(self) -> list[str]: ...


class _ManifestLoader(yaml.SafeLoader):
    """YAML's safe subset without anchors and aliases, nested no deeper than a manifest
    needs: past that the composer recurses until Python gives up."""

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._depth = 0

    def compose_node(self, parent: yaml.Node | None, index: object) -> yaml.Node | None:
        event = self.peek_event()
        if isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None) is not None:
            raise ComposerError(None, None, "anchors and aliases are not allowed", event.start_mark)
        if self._depth >= _MAX_MANIFEST_DEPTH:
            raise ComposerError(
                None, None, f"nested deeper than {_MAX_MANIFEST_DEPTH} levels", event.start_mark
            )
        self._depth += 1
        try:
            return super().compose_node(parent, index)
        finally:
            self._depth -= 1


def load_manifest(data: bytes, what: str = MANIFEST_NAME) -> SubmissionManifest:
    """Parse a manifest as a team pushed it. Raises :class:`Rejected`."""
    _check_manifest_size(len(data), what)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Rejected(
            ErrorCode.INVALID_MANIFEST, f"{what} is not UTF-8 (byte {exc.start}: {exc.reason})"
        ) from exc
    # Not only YAMLError: the constructors raise ValueError for `2026-13-01` or `!!int 0x`.
    # Whatever parsing the team's file raises is the file's fault.
    try:
        document = yaml.load(text, Loader=_ManifestLoader)
    except Exception as exc:
        raise Rejected(ErrorCode.INVALID_MANIFEST, f"{what} is not valid YAML: {exc}") from exc
    try:
        return SubmissionManifest.model_validate(document)
    except ValidationError as exc:
        raise Rejected(
            ErrorCode.INVALID_MANIFEST, f"{what} is invalid: {describe_errors(exc)}"
        ) from exc


def _check_manifest_size(size: int, what: str) -> None:
    if size > MAX_MANIFEST_BYTES:
        raise Rejected(
            ErrorCode.INVALID_MANIFEST,
            f"{what} is {size} bytes, over the {MAX_MANIFEST_BYTES // 1024} KiB limit",
        )


def manifest_key(manifest: SubmissionManifest) -> str:
    """The submission's identity: a hash of the manifest's values, so comments, layout
    and key order do not count. A value left at its default is left out, so writing it
    out is no change, and neither is a field added to the schema later."""
    values = manifest.model_dump(mode="json", exclude_defaults=True)
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:_KEY_LENGTH]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _File:
    size: int
    lfs_sha256: str | None


@dataclass(frozen=True)
class _Tree:
    sha: str
    files: dict[str, _File]


class TeamStatus(BaseModel):
    team_id: str
    repo: str
    revision: str | None = None
    files: list[str] | None = None
    error: str | None = None


def _origin(repo: str, sha: str) -> str:
    return f"{repo}@{sha[:_KEY_LENGTH]}"


def incoming_dir(store: EvalStore, team_id: str, key: str) -> Path:
    """Where a submission is fetched to. Keyed by the submission, so a repo that moves on
    does not overwrite the bytes an already-published result was produced from."""
    return store.incoming / team_id / key


def _brief(exc: Exception) -> str:
    """The first line of a Hub error, without the request id and URL."""
    return str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__


def _unreadable(what: str, exc: Exception) -> Rejected:
    """A Hub request that failed. With no response, a 408, a 429 or a 5xx the Hub is at
    fault and the request is worth making again; any other status is the reference's."""
    status = exc.response.status_code if isinstance(exc, HfHubHTTPError) else None
    transient = status is None or status in (408, 429) or status >= 500
    return Rejected(ErrorCode.REPO_UNREADABLE, f"{what}: {_brief(exc)}", transient=transient)


class HubSource:
    def __init__(
        self,
        config: HubConfig,
        store: EvalStore,
        teams: Callable[[], list[HubTeam]],
        api: HfApi | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._teams = teams
        self._token = os.environ.get(config.token_env) or None
        self._api = api or HfApi(token=self._token)
        self._seen_heads: dict[str, tuple[str, list[Candidate]]] = {}
        self._teams_lock = threading.Lock()
        """The watcher records the heads at the deadline beside a poll, and reading the
        registry (a git fetch and a snapshot write) is not safe twice at once."""
        self._known_teams: list[HubTeam] | None = None
        self._last_error: dict[str, str] = {}

    @property
    def poll_interval_s(self) -> float:
        return self._config.poll_interval_s

    def teams(self) -> list[HubTeam]:
        with self._teams_lock:
            self._known_teams = self._teams()
            return self._known_teams

    def known_teams(self) -> list[HubTeam]:
        return self._known_teams if self._known_teams is not None else self.teams()

    def candidates(
        self, note_head: Callable[[HubTeam, str], None] | None = None
    ) -> list[Candidate]:
        found: list[Candidate] = []
        for team in self.teams():
            # An unreadable repo is an operator's problem, not a bad submission: it is
            # logged rather than published as a rejection.
            try:
                head = self._tree(team.repo, WATCHED_REVISION)
            except Rejected as exc:
                self._report(team.repo, str(exc))
                continue
            self._last_error.pop(team.repo, None)
            if note_head:
                note_head(team, head.sha)
            cached = self._seen_heads.get(team.repo)
            if cached is not None and cached[0] == head.sha:
                found.extend(cached[1])
                continue
            try:
                inspected = [
                    self._inspect(team, head, MANIFEST_NAME),
                    self._inspect(team, head, RANKING_MANIFEST_NAME),
                ]
            except Exception:  # one team's repo must never stop every other team's intake
                _logger.exception(
                    "%s: cannot inspect %s; trying again next poll",
                    team.repo,
                    head.sha[:_KEY_LENGTH],
                )
                continue
            if not any(each.rejection and each.rejection.transient for each in inspected):
                self._seen_heads[team.repo] = (head.sha, inspected)
            found.extend(inspected)
        return found

    def head_at(self, team: HubTeam, deadline: datetime) -> str | None:
        try:
            commits = self._api.list_repo_commits(
                team.repo, repo_type=self._config.repo_type, revision=WATCHED_REVISION
            )
        except HTTPError as exc:
            raise RepoUnreadableError(f"{team.repo}: {_brief(exc)}") from exc
        # Newest first. A push in the moments between the deadline and this call is
        # passed over by its commit time.
        made = next((commit for commit in commits if commit.created_at <= deadline), None)
        return made.commit_id if made else None

    def ranking_at(self, team: HubTeam, revision: str) -> Candidate:
        try:
            tree = self._tree(team.repo, revision)
        except Rejected as exc:
            raise RepoUnreadableError(str(exc)) from exc
        return self._inspect(team, tree, RANKING_MANIFEST_NAME)

    def _tree(self, repo: str, revision: str) -> _Tree:
        """The files of ``repo`` at ``revision``. Raises :class:`Rejected`."""
        try:
            info = self._api.repo_info(
                repo,
                repo_type=self._config.repo_type,
                revision=revision,
                files_metadata=True,
            )
        except HTTPError as exc:
            raise _unreadable(f"cannot read {_origin(repo, revision)}", exc) from exc
        if not info.sha:
            raise Rejected(ErrorCode.REPO_UNREADABLE, f"{repo} has no revision {revision!r}")
        files = {
            sibling.rfilename: _File(sibling.size or 0, sibling.lfs.sha256 if sibling.lfs else None)
            for sibling in (info.siblings or [])
        }
        return _Tree(info.sha, files)

    def _inspect(self, team: HubTeam, head: _Tree, name: str) -> Candidate:
        sha = head.sha
        origin = _origin(team.repo, sha)
        base = Candidate(
            team_id=team.team_id,
            key=sha[:_KEY_LENGTH],
            origin=origin,
            revision=sha[:_KEY_LENGTH],
            hub_repo=team.repo,
            hub_revision=sha,
            manifest_name=name,
        )
        manifest_file = head.files.get(name)
        if manifest_file is None:
            if name == RANKING_MANIFEST_NAME:
                return base  # not submitted (yet), which is not a mistake
            return replace(
                base,
                rejection=Rejected(ErrorCode.INVALID_MANIFEST, f"{name} is not in {origin}"),
            )
        try:
            _check_manifest_size(manifest_file.size, name)
            manifest = load_manifest(self._read_manifest(team, sha, name), name)
        except Rejected as exc:
            return replace(base, rejection=exc)
        base = replace(base, key=manifest_key(manifest))

        if manifest.team_id != team.team_id:
            return replace(
                base,
                rejection=Rejected(
                    ErrorCode.INVALID_MANIFEST,
                    f"{name} says team_id {manifest.team_id!r}, but {team.repo} is "
                    f"registered to {team.team_id!r}; set team_id: {team.team_id}",
                ),
            )
        base = replace(base, manifest=manifest)
        image = manifest.image
        # The organizer account can read every team's org, so without this one team
        # could submit another's image as its own.
        if hf_owner(image.repo) != hf_owner(team.repo):
            return replace(
                base,
                rejection=Rejected(
                    ErrorCode.FOREIGN_REPO,
                    f"image repo {image.repo} is not in {team.repo.split('/', 1)[0]}; "
                    "the image must come from your own org",
                ),
            )
        image_origin = _origin(image.repo, image.revision)
        try:
            image_tree = self._tree(image.repo, image.revision)
        except Rejected as exc:
            return replace(base, rejection=exc)
        image_file = image_tree.files.get(image.path)
        if image_file is None:
            return replace(
                base,
                rejection=Rejected(
                    ErrorCode.IMAGE_NOT_FOUND, f"image {image.path!r} is not in {image_origin}"
                ),
            )
        return replace(
            base,
            image_gb=image_file.size / GB,
            image_sha256=image_file.lfs_sha256,
        )

    def _read_manifest(self, team: HubTeam, sha: str, name: str) -> bytes:
        try:
            path = hf_hub_download(
                team.repo,
                name,
                repo_type=self._config.repo_type,
                revision=sha,
                token=self._token,
            )
        except _HUB_ERRORS as exc:
            raise _unreadable(f"cannot read {name} from {_origin(team.repo, sha)}", exc) from exc
        return Path(path).read_bytes()

    def fetch(self, candidate: Candidate) -> Path:
        if candidate.manifest is None or candidate.hub_repo is None:
            raise Rejected(ErrorCode.INTERNAL_ERROR, f"{candidate.origin} was never inspected")
        target = incoming_dir(self._store, candidate.team_id, candidate.key)
        target.mkdir(parents=True, exist_ok=True)
        image = candidate.manifest.image
        for repo, revision, name in (
            (candidate.hub_repo, candidate.hub_revision, candidate.manifest_name),
            (image.repo, image.revision, image.path),
        ):
            try:
                hf_hub_download(
                    repo,
                    name,
                    repo_type=self._config.repo_type,
                    revision=revision,
                    token=self._token,
                    # Not the blob cache: the runner bind-mounts this path into a container.
                    local_dir=str(target),
                )
            except _HUB_ERRORS as exc:
                raise _unreadable(
                    f"cannot download {name!r} from {_origin(repo, revision or '')}", exc
                ) from exc
        return target / image.path

    def preflight(self) -> list[str]:
        """Things that will not stop a poll but will ruin a round.

        An expiring token (``hf auth login``) makes every repo unreadable at once when
        it lapses, which looks exactly like nobody submitting.
        """
        warnings: list[str] = []
        try:
            me = self._api.whoami()
        except HTTPError as exc:
            return [f"cannot authenticate to HuggingFace: {_brief(exc)}"]

        expires = (me.get("auth") or {}).get("expiresAt")
        if expires:
            warnings.append(
                f"the token is an expiring one (expires {expires}) — issue a "
                f"non-expiring fine-grained token and set {self._config.token_env}"
            )

        registered = {team.repo.split("/", 1)[0] for team in self.teams()}
        invited = {org["name"] for org in (me.get("orgs") or []) if org.get("name")}
        for org in sorted(invited - registered):
            warnings.append(f"invited to org {org!r} but no team is registered for it")
        for org in sorted(registered - invited):
            warnings.append(f"team registered under org {org!r} but the account is not a member")
        return warnings

    def describe(self) -> list[TeamStatus]:
        rows: list[TeamStatus] = []
        for team in self.teams():
            row = TeamStatus(team_id=team.team_id, repo=team.repo)
            try:
                head = self._tree(team.repo, WATCHED_REVISION)
            except Rejected as exc:
                row.error = str(exc)
            else:
                row.revision, row.files = head.sha, sorted(head.files)
            rows.append(row)
        return rows

    def _report(self, repo: str, problem: str) -> None:
        """Log a repo's problem once, not on every poll."""
        if self._last_error.get(repo) != problem:
            self._last_error[repo] = problem
            _logger.warning("%s", problem)


class HubWeights:
    """Fetches the model repos a submission declares (capx/evalqueue/weights.py)."""

    def __init__(self, config: HubConfig, api: HfApi | None = None) -> None:
        self._token = os.environ.get(config.token_env) or None
        self._api = api or HfApi(token=self._token)

    def describe(self, mount: WeightsMount) -> RepoInfo:
        try:
            info = self._api.model_info(mount.repo, revision=mount.revision, files_metadata=True)
        except HTTPError as exc:
            raise _unreadable(f"cannot read weights {describe_mount(mount)}", exc) from exc
        gated = bool(info.gated)
        accessible = True
        if gated:
            try:
                self._api.auth_check(mount.repo)
            except GatedRepoError:
                accessible = False
            except HTTPError as exc:
                raise _unreadable(
                    f"cannot check access to weights {describe_mount(mount)}", exc
                ) from exc
        return RepoInfo(
            size_bytes=sum(sibling.size or 0 for sibling in info.siblings or []),
            gated=gated,
            private=info.private is not False,
            accessible=accessible,
        )

    def download(self, mount: WeightsMount, target: Path) -> None:
        try:
            snapshot_download(
                mount.repo,
                revision=mount.revision,
                local_dir=str(target),
                token=self._token,
                max_workers=8,
            )
        except GatedRepoError as exc:
            raise _no_access(mount) from exc
        except _HUB_ERRORS as exc:
            raise _unreadable(f"cannot download weights {describe_mount(mount)}", exc) from exc


def _no_access(mount: WeightsMount) -> Rejected:
    return Rejected(
        ErrorCode.WEIGHTS_GATED,
        f"the organizer account has not been granted access to gated repo {mount.repo}",
    )


__all__ = [
    "MANIFEST_NAME",
    "MAX_MANIFEST_BYTES",
    "RANKING_MANIFEST_NAME",
    "RepoUnreadableError",
    "Candidate",
    "HubSource",
    "HubWeights",
    "SubmissionSource",
    "TeamStatus",
    "incoming_dir",
    "load_manifest",
    "manifest_key",
    "sha256_of",
]
