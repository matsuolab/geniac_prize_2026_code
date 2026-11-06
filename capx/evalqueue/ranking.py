"""Ranking submissions: ``submission_ranking.yaml`` (PRELIM_RULEBOOK.md §6.5).

Before the deadline the watcher only checks a team's ranking submission, the same way
it checks an interim one but without moving any bytes, and keeps the outcome as the
team's :class:`RankingEntry`. At the deadline it records the commit each team's
``main`` stood at (:class:`TeamHead`), then, once and for good, the entry for that
commit (``final``). After it, an organizer queues one run per accepted final entry on
the ranking round's own queue (:func:`queue_ranking_runs`).
"""

from __future__ import annotations

import enum
import json
import logging
from datetime import datetime

from pydantic import BaseModel, ConfigDict, TypeAdapter

from capx.evalqueue.hubsource import RANKING_MANIFEST_NAME, Candidate, SubmissionSource, sha256_of
from capx.evalqueue.limits import check_sha256
from capx.evalqueue.models import ErrorCode, Rejected, RunResult, RunStatus, SubmissionManifest
from capx.evalqueue.queue import FileQueue, new_job
from capx.evalqueue.store import EvalStore, job_id_for, run_id_for, utcnow, write_json_atomic
from capx.evalqueue.teams import registered_teams, save_registered_teams
from capx.evalqueue.weights import WeightsCache

_logger = logging.getLogger(__name__)

_FINAL_MARKER = "FINAL"


class TeamHead(BaseModel):
    """The commit a team's ``main`` stood at."""

    model_config = ConfigDict(extra="forbid")

    team_id: str
    hub_repo: str
    revision: str | None
    """None when the repo had no commit (by the deadline)."""
    seen_at: datetime
    unreadable: str | None = None
    """Why ``main`` could not be read at the deadline. ``revision`` is then the last head
    the watcher saw before it, or None if it never saw one."""


_HEADS = TypeAdapter(list[TeamHead])


class RankingStatus(str, enum.Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    MISSING = "missing"
    """No ranking submission in the commit at the deadline. Only ever final."""


class RankingEntry(BaseModel):
    """A team's ranking submission as last checked."""

    model_config = ConfigDict(extra="forbid")

    team_id: str
    status: RankingStatus
    hub_repo: str
    hub_revision: str | None
    """The commit of the team's submission repo the entry was read from; None when the
    repo had no commit by the deadline."""
    key: str
    """The submission's identity, as for an interim job (``<team_id>__<key>``)."""
    manifest: SubmissionManifest | None = None
    error: str | None = None
    error_code: ErrorCode | None = None
    checked_at: datetime
    final: bool = False
    """Recorded at the deadline: this is what the ranking evaluation runs."""

    @classmethod
    def from_check(
        cls, candidate: Candidate, rejection: Rejected | None, *, final: bool = False
    ) -> RankingEntry:
        """The entry for a checked candidate; ``rejection`` is what the check refused it
        for, if anything. An absent one is ``missing``."""
        assert candidate.hub_repo is not None
        status = RankingStatus.REJECTED if rejection else RankingStatus.ACCEPTED
        return cls(
            team_id=candidate.team_id,
            status=RankingStatus.MISSING if candidate.absent else status,
            hub_repo=candidate.hub_repo,
            hub_revision=candidate.hub_revision,
            key=candidate.key,
            manifest=candidate.manifest,
            error=str(rejection) if rejection else None,
            error_code=rejection.code if rejection else None,
            checked_at=utcnow(),
            final=final,
        )

    def candidate(self) -> Candidate:
        assert self.hub_revision is not None
        return Candidate(
            team_id=self.team_id,
            key=self.key,
            origin=f"{self.hub_repo}@{self.hub_revision[:12]}",
            manifest=self.manifest,
            hub_repo=self.hub_repo,
            hub_revision=self.hub_revision,
            manifest_name=RANKING_MANIFEST_NAME,
        )


class RankingBook:
    """The ranking entries of one round, under ``state/ranking/`` of its store."""

    def __init__(self, store: EvalStore) -> None:
        self._dir = store.state / "ranking"
        self._heads_seen = self._dir / "heads" / "seen.json"
        self._heads_at_deadline = self._dir / "heads" / "deadline.json"

    def heads_seen(self) -> dict[str, TeamHead]:
        """The last head of each team's ``main`` the watcher saw before the deadline."""
        try:
            heads = _HEADS.validate_json(self._heads_seen.read_bytes())
        except (OSError, ValueError):
            return {}
        return {head.team_id: head for head in heads}

    def note_heads_seen(self, heads: list[TeamHead]) -> None:
        write_json_atomic(self._heads_seen, _HEADS.dump_python(heads, mode="json"))

    def heads_at_deadline(self) -> list[TeamHead] | None:
        """Where each team's ``main`` stood at the deadline; None until it is recorded."""
        try:
            return _HEADS.validate_json(self._heads_at_deadline.read_bytes())
        except FileNotFoundError:
            return None

    def record_heads_at_deadline(self, heads: list[TeamHead]) -> None:
        write_json_atomic(self._heads_at_deadline, _HEADS.dump_python(heads, mode="json"))

    def get(self, team_id: str) -> RankingEntry | None:
        try:
            return RankingEntry.model_validate(
                json.loads((self._dir / f"{team_id}.json").read_text())
            )
        except (OSError, ValueError):
            return None

    def put(self, entry: RankingEntry) -> None:
        write_json_atomic(self._dir / f"{entry.team_id}.json", entry.model_dump(mode="json"))

    def remove(self, team_id: str) -> None:
        (self._dir / f"{team_id}.json").unlink(missing_ok=True)

    def entries(self) -> list[RankingEntry]:
        found = [self.get(path.stem) for path in sorted(self._dir.glob("*.json"))]
        return [entry for entry in found if entry is not None]

    @property
    def is_final(self) -> bool:
        return (self._dir / _FINAL_MARKER).exists()

    def finalize(self) -> None:
        """Mark the round's entries final, once every team has its final entry. An
        entry left from before the deadline for a team no longer registered goes."""
        for entry in self.entries():
            if not entry.final:
                self.remove(entry.team_id)
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / _FINAL_MARKER).write_text(utcnow().isoformat())


def queue_ranking_runs(
    entries: RankingBook,
    teams_from: EvalStore,
    store: EvalStore,
    queue: FileQueue,
    source: SubmissionSource,
    weights: WeightsCache,
    *,
    team_ids: set[str] | None = None,
    again: bool = False,
) -> dict[str, str]:
    """Queue one run per accepted final entry on the ranking round's queue (``store``).

    A team is run once: one that is waiting or running here is skipped, and so is one
    that already has a result, unless ``again`` (a run that our side broke is run
    again, §6.5). A rejected entry gets a
    ``rejected`` result instead, so the ranking board says why the team has no rank.
    Returns what happened to each team.
    """
    if not entries.is_final:
        raise SystemExit("the ranking submissions have not been recorded at the deadline yet")
    save_registered_teams(store, registered_teams(teams_from))
    outcome: dict[str, str] = {}
    for entry in entries.entries():
        if team_ids is not None and entry.team_id not in team_ids:
            continue
        job_id = job_id_for(entry.team_id, entry.key)
        if queue.is_queued(job_id):
            outcome[entry.team_id] = "already waiting or running"
            continue
        if not again and any(store.results.glob(f"{job_id}__*.json")):
            outcome[entry.team_id] = "already run"
            continue
        if entry.status is RankingStatus.MISSING:
            outcome[entry.team_id] = "no ranking submission"
            continue
        if entry.status is RankingStatus.REJECTED or entry.manifest is None:
            store.publish_result(
                RunResult.failed(
                    run_id_for(job_id),
                    job_id,
                    entry.team_id,
                    entry.manifest.submission_id if entry.manifest else entry.key,
                    entry.error or "rejected",
                    entry.error_code or ErrorCode.INVALID_MANIFEST,
                    status=RunStatus.REJECTED,
                    submitted_at=entry.checked_at,
                )
            )
            outcome[entry.team_id] = f"rejected ({entry.error_code})"
            continue
        manifest = entry.manifest
        try:
            image = source.fetch(entry.candidate())
            if manifest.image_sha256:
                check_sha256(manifest.image_sha256, sha256_of(image))
            weights.fetch(manifest.weights)
        except (Rejected, OSError) as exc:
            # Checked before the deadline, so this is the Hub or the disk, not the team.
            _logger.error("could not fetch %s's ranking submission: %s", entry.team_id, exc)
            outcome[entry.team_id] = f"fetch failed, try again: {exc}"
            continue
        queue.enqueue(
            new_job(
                job_id, manifest, image, hub_repo=entry.hub_repo, hub_revision=entry.hub_revision
            )
        )
        outcome[entry.team_id] = f"queued as {job_id}"
    return outcome


__all__ = [
    "RANKING_MANIFEST_NAME",
    "RankingBook",
    "RankingEntry",
    "RankingStatus",
    "TeamHead",
    "queue_ranking_runs",
]
