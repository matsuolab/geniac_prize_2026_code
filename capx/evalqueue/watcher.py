"""Turn a submission a source is offering into a queued job.

The checks run in cost order. Everything answerable from metadata runs before
:meth:`SubmissionSource.fetch`, so an oversized image, or one whose hash is not what
the manifest claims, is refused without moving gigabytes. Only whether the bytes
that landed are the bytes that were described is left until afterwards.

Rejections are published like any other outcome, as a ``RunResult`` with status
``rejected``, so a team finds out why in the same place it finds its scores. One the
Hub caused (``Rejected.transient``) is first tried again on later polls, and anything
else that goes wrong with one submission is logged and tried again, without holding up
the others.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from capx.config import EvalConfig, load_eval_config
from capx.evalqueue.config import HubTeam, QueueConfig
from capx.evalqueue.deadline import close_round, past_deadline, publish_cancelled
from capx.evalqueue.hubsource import (
    RANKING_MANIFEST_NAME,
    Candidate,
    RepoUnreadableError,
    SubmissionSource,
    sha256_of,
)
from capx.evalqueue.limits import check_image_size, check_sha256
from capx.evalqueue.models import (
    ErrorCode,
    QueueJob,
    Rejected,
    RunResult,
    RunSpec,
    RunStatus,
    SubmissionManifest,
)
from capx.evalqueue.queue import FileQueue, new_job
from capx.evalqueue.ranking import RankingBook, RankingEntry, TeamHead
from capx.evalqueue.store import EvalStore, job_id_for, run_id_for, utcnow, write_json_atomic
from capx.evalqueue.weights import WeightsCache, active_jobs, describe_mount

_logger = logging.getLogger(__name__)

HEAD_READ_TIMEOUT_S = 30.0
"""How long the recording at the deadline waits for each team's ``main``. A read that
takes longer is left behind, and the head the watcher saw before the deadline stands in."""

RETRY_S = 3600.0
"""How long a submission the Hub cannot serve is tried again, a poll at a time, before
it is refused: an interim one from the first failure, a ranking one from the deadline."""


class SubmissionWatcher:
    def __init__(
        self,
        config: QueueConfig,
        store: EvalStore,
        queue: FileQueue,
        source: SubmissionSource,
        weights: WeightsCache,
        eval_config: EvalConfig | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._queue = queue
        self._source = source
        self._weights = weights
        self._eval_config = eval_config or load_eval_config()
        self._ranking = RankingBook(store)
        self._heads_lock = threading.Lock()
        """Between noting a head a poll read and recording the heads at the deadline,
        which may run beside a poll."""
        self._heads_seen = self._ranking.heads_seen()
        self._heads_read_at_deadline: list[TeamHead] | None = None
        """What the recording read, until it is written: written again on a failed
        write, never read again from the Hub."""

    def run_forever(self) -> None:
        for warning in self._source.preflight():
            _logger.warning("%s", warning)
        _logger.info("watching %s", type(self._source).__name__)
        if self._config.deadline is not None and self._ranking.heads_at_deadline() is None:
            threading.Thread(
                target=self._record_heads_when_due, name="deadline", daemon=True
            ).start()
        while True:
            try:
                self.poll()
            except Exception:  # a bad submission must never stop the watcher
                _logger.exception("watcher poll failed")
            time.sleep(self._until_next_poll_s())

    def _until_next_poll_s(self) -> float:
        """The poll interval, cut short to wake at the deadline, when the ranking
        submissions are recorded."""
        interval = self._source.poll_interval_s
        deadline = self._config.deadline
        if deadline is None or past_deadline(self._config):
            return interval
        return min(interval, max(0.0, (deadline - utcnow()).total_seconds()))

    def poll(self) -> list[str]:
        """One scan. Returns the job ids enqueued this pass. From the deadline on,
        nothing is queued, what still waits is cancelled, and each team's ranking
        submission is recorded as it stood at the deadline."""
        if past_deadline(self._config):
            close_round(self._store, self._queue)
            self._sweep_every_team()
            self._record_heads_at_deadline()
            if not self._ranking.is_final:
                self._record_final_ranking()
            return []
        candidates = self._source.candidates(self._note_head)
        enqueued: list[str] = []
        for candidate in candidates:
            if past_deadline(self._config):
                break  # a long poll ran into the deadline; the next one closes the round
            try:
                if candidate.manifest_name == RANKING_MANIFEST_NAME:
                    self._check_ranking(candidate)
                elif job_id := self._consider(candidate):
                    enqueued.append(job_id)
            except Exception:  # one submission must never stop the others
                _logger.exception("taking in %s failed; trying again next poll", candidate.origin)
        return enqueued

    def _consider(self, candidate: Candidate) -> str | None:
        job_id = job_id_for(candidate.team_id, candidate.key)
        attempt_id = job_id_for(candidate.team_id, candidate.revision or candidate.key)
        self._sweep(candidate.team_id, keep=candidate.key)
        if self._was_enqueued(job_id) or self._queue.is_queued(job_id):
            # The team went back to a submission already taken in, so one deferred
            # since will never run.
            self._clear_deferred(candidate.team_id)
            return None
        if self._is_seen(attempt_id):
            return None
        if self._team_is_running_at_cap(candidate.team_id):
            # A running job cannot be replaced. Not a rejection and not marked seen:
            # the next poll looks again, and by then the repo's head may be newer.
            _logger.info(
                "deferring %s: team %r has a submission running", job_id, candidate.team_id
            )
            self._mark_deferred(candidate.team_id, job_id, self._submission_id(candidate))
            return None

        try:
            manifest = self._accept(candidate)
            image = self._source.fetch(candidate)
            self._verify(image, manifest)
            self._weights.fetch(manifest.weights)
        except Rejected as exc:
            if exc.transient and self._still_trying(attempt_id):
                _logger.warning("%s: %s; trying again next poll", job_id, exc)
                return None
            self._clear_deferred(candidate.team_id)
            _logger.warning("rejecting %s: %s", job_id, exc)
            self._publish_rejection(candidate, exc)
            self._mark_seen(attempt_id, candidate.team_id, enqueued=False)
            self._sweep(candidate.team_id)
            # Only now: had the rejection not been recorded, the next try would start
            # the hour over.
            self._done_trying(attempt_id)
            return None

        if past_deadline(self._config):
            # Fetched while the round closed: it waited, and waiting ones are cancelled.
            publish_cancelled(
                self._store, job_id, candidate.team_id, manifest.submission_id, utcnow()
            )
            self._clear_deferred(candidate.team_id)
            self._done_trying(attempt_id)
            return None
        if not self._make_room(candidate.team_id, job_id):
            _logger.info(
                "deferring %s: a worker started the team's earlier submission meanwhile", job_id
            )
            self._mark_deferred(candidate.team_id, job_id, manifest.submission_id)
            return None
        self._clear_deferred(candidate.team_id)
        self._queue.enqueue(
            new_job(
                job_id,
                manifest,
                image,
                hub_repo=candidate.hub_repo,
                hub_revision=candidate.hub_revision,
            )
        )
        self._mark_seen(job_id, candidate.team_id, enqueued=True)
        self._done_trying(attempt_id)
        _logger.info("enqueued %s from %s", job_id, candidate.origin)
        return job_id

    def _check_ranking(self, candidate: Candidate) -> None:
        """Check a ranking submission without fetching it, once per commit."""
        current = self._ranking.get(candidate.team_id)
        if current is not None and current.hub_revision == candidate.hub_revision:
            return
        if candidate.absent:
            self._ranking.remove(candidate.team_id)
            return
        refusal = self._refusal(candidate)
        if refusal is not None and refusal.transient:
            return  # the Hub's trouble, not the commit's: checked again next poll
        entry = RankingEntry.from_check(candidate, refusal)
        self._ranking.put(entry)
        _logger.info("ranking submission of %s: %s", candidate.team_id, entry.status.value)

    def _note_head(self, team: HubTeam, revision: str) -> None:
        """Keep the head a team's ``main`` was last seen at before the deadline, for when
        it cannot be read at the deadline. Called as soon as the head is read: inspecting
        the rest of the poll may run into the deadline."""
        with self._heads_lock:
            if past_deadline(self._config):
                return
            known = self._heads_seen.get(team.team_id)
            if known and (known.hub_repo, known.revision) == (team.repo, revision):
                return
            self._heads_seen[team.team_id] = TeamHead(
                team_id=team.team_id, hub_repo=team.repo, revision=revision, seen_at=utcnow()
            )
            self._ranking.note_heads_seen(list(self._heads_seen.values()))

    def _record_heads_when_due(self) -> None:
        """Beside the polls, so that a transfer in progress cannot hold the recording
        back while teams keep pushing."""
        deadline = self._config.deadline
        assert deadline is not None
        while (left := (deadline - utcnow()).total_seconds()) > 0:
            time.sleep(min(left, 60.0))
        try:
            self._record_heads_at_deadline()
        except Exception:
            _logger.exception("recording the heads at the deadline failed; the next poll will")

    def _record_heads_at_deadline(self) -> None:
        """Record, once, the commit each team's ``main`` stood at at the deadline
        (PRELIM_RULEBOOK.md §6.5). Its ranking submission is read at that commit from
        then on, however long the checks take or are retried."""
        with self._heads_lock:
            if self._ranking.heads_at_deadline() is not None:
                return
            deadline = self._config.deadline
            assert deadline is not None
            if self._heads_read_at_deadline is None:
                # The teams as last read: reading the registry again is a git fetch, and
                # one a poll has in progress would hold this back.
                self._heads_read_at_deadline = self._heads_at(self._source.known_teams(), deadline)
            heads = self._heads_read_at_deadline
            self._ranking.record_heads_at_deadline(heads)
        _logger.info(
            "recorded each team's main at the deadline: %s",
            {head.team_id: head.revision for head in heads},
        )

    def _heads_at(self, teams: list[HubTeam], deadline: datetime) -> list[TeamHead]:
        """Each team's ``main`` at the deadline, all read at once, none waited on longer
        than ``HEAD_READ_TIMEOUT_S``: the Hub client sets no timeout of its own."""
        answers: dict[str, str | None | Exception] = {}

        def read(team: HubTeam) -> None:
            try:
                answers[team.team_id] = self._source.head_at(team, deadline)
            except Exception as exc:  # one team's repo must never stop the others'
                answers[team.team_id] = exc

        readers = [
            threading.Thread(target=read, args=(team,), name=f"head-{team.team_id}", daemon=True)
            for team in teams
        ]
        for reader in readers:
            reader.start()
        until = time.monotonic() + HEAD_READ_TIMEOUT_S
        for reader in readers:
            reader.join(max(0.0, until - time.monotonic()))
        answered = dict(answers)

        heads: list[TeamHead] = []
        for team in teams:
            if team.team_id not in answered:
                problem = f"no answer within {HEAD_READ_TIMEOUT_S:.0f} s"
                heads.append(self._head_seen_before(team, problem))
            elif isinstance(answer := answered[team.team_id], Exception):
                heads.append(self._head_seen_before(team, str(answer) or type(answer).__name__))
            else:
                head = TeamHead(
                    team_id=team.team_id, hub_repo=team.repo, revision=answer, seen_at=utcnow()
                )
                heads.append(head)
        return heads

    def _head_seen_before(self, team: HubTeam, problem: str) -> TeamHead:
        seen = self._heads_seen.get(team.team_id)
        if seen is None or seen.hub_repo != team.repo:
            _logger.error("%s: main cannot be read at the deadline: %s", team.team_id, problem)
            return TeamHead(
                team_id=team.team_id,
                hub_repo=team.repo,
                revision=None,
                seen_at=utcnow(),
                unreadable=problem,
            )
        _logger.warning(
            "%s: main cannot be read at the deadline (%s); using the head seen at %s",
            team.team_id,
            problem,
            seen.seen_at.isoformat(),
        )
        return seen.model_copy(update={"unreadable": problem})

    def _record_final_ranking(self) -> None:
        """Record each team's ranking submission in the commit recorded for it at the
        deadline.

        A team whose image or weights cannot be read there is tried again on the next
        poll for ``RETRY_S`` after the deadline, so a Hub outage does not count against
        it. After that it is recorded as refused: the reference may never resolve (a
        commit that does not exist), and the round's entries are final only once every
        team has one.
        """
        deadline = self._config.deadline
        heads = self._ranking.heads_at_deadline()
        assert deadline is not None and heads is not None
        retrying = utcnow() - deadline < timedelta(seconds=RETRY_S)
        waiting = False
        for head in heads:
            current = self._ranking.get(head.team_id)
            if current is not None and current.final:
                continue
            try:
                candidate = self._ranking_candidate(head)
                refusal = None if candidate.absent else self._refusal(candidate)
            except Exception:  # one team's must never hold back the others'
                _logger.exception("%s's ranking submission: trying again", head.team_id)
                waiting = True
                continue
            if (
                refusal is not None
                and refusal.code is ErrorCode.REPO_UNREADABLE
                and head.revision is not None
            ):
                if retrying:
                    _logger.warning(
                        "%s's ranking submission: %s; trying again", head.team_id, refusal
                    )
                    waiting = True
                    continue
                _logger.error(
                    "%s's ranking submission is recorded as refused: %s", head.team_id, refusal
                )
            self._ranking.put(RankingEntry.from_check(candidate, refusal, final=True))
        if not waiting:
            self._ranking.finalize()
            _logger.info("recorded the ranking submissions at the deadline")

    def _ranking_candidate(self, head: TeamHead) -> Candidate:
        nothing = Candidate(
            team_id=head.team_id,
            key="none",
            origin=head.hub_repo,
            hub_repo=head.hub_repo,
            hub_revision=head.revision,
            manifest_name=RANKING_MANIFEST_NAME,
        )
        if head.revision is None:
            if head.unreadable is None:
                return nothing
            return replace(nothing, rejection=Rejected(ErrorCode.REPO_UNREADABLE, head.unreadable))
        try:
            return self._source.ranking_at(
                HubTeam(team_id=head.team_id, repo=head.hub_repo), head.revision
            )
        except RepoUnreadableError as exc:
            return replace(nothing, rejection=Rejected(ErrorCode.REPO_UNREADABLE, str(exc)))

    def _refusal(self, candidate: Candidate) -> Rejected | None:
        try:
            self._accept(candidate)
        except Rejected as exc:
            return exc
        return None

    def _accept(self, candidate: Candidate) -> SubmissionManifest:
        """Everything that can be decided without the image's bytes."""
        if candidate.rejection:
            raise candidate.rejection
        if candidate.manifest is None:
            raise Rejected(ErrorCode.INVALID_MANIFEST, "no manifest")

        manifest = candidate.manifest
        check_image_size(candidate.image_gb, self._eval_config)
        if manifest.image_sha256 and candidate.image_sha256:
            check_sha256(manifest.image_sha256, candidate.image_sha256)
        for mount in manifest.weights:
            if mount.gated:
                # Only a person can request access for the organizer account.
                _logger.warning(
                    "%s declares gated weights %s; the organizer account needs access",
                    candidate.origin,
                    describe_mount(mount),
                )
        self._weights.check(candidate.team_id, manifest.weights)
        return manifest

    def _verify(self, image: Path, manifest: SubmissionManifest) -> None:
        """Whether the right bytes arrived: ``hf_hub_download`` checks only the length."""
        if not image.is_file():
            raise Rejected(
                ErrorCode.IMAGE_NOT_FOUND,
                f"image {manifest.image.path!r} not found in the submission directory",
            )
        if manifest.image_sha256:
            check_sha256(manifest.image_sha256, sha256_of(image))

    def _team_is_running_at_cap(self, team_id: str) -> bool:
        cap = self._config.watcher.max_active_per_team
        return cap > 0 and self._running_count(team_id) >= cap

    def _running_count(self, team_id: str) -> int:
        return sum(job.team_id == team_id and _from_hub(job) for job in self._queue.claimed_jobs())

    def _make_room(self, team_id: str, new_job_id: str) -> bool:
        """Cancel the team's oldest waiting jobs until the new one fits under the cap.

        Runs only once the new submission has passed every check and its bytes are on
        disk, so one that would be rejected never cancels a good one. False means a
        worker claimed an old job between our look and the withdrawal: that one is now
        running and holds the slot, so the new one waits for it instead.
        """
        cap = self._config.watcher.max_active_per_team
        if cap <= 0:
            return True
        room = cap - self._running_count(team_id)
        if room <= 0:
            return False
        waiting = [job for job in self._queue.team_pending(team_id) if _from_hub(job)]
        for old in waiting[: max(0, len(waiting) + 1 - room)]:
            if not self._queue.withdraw(old.job_id):
                return False
            self._supersede(old, new_job_id)
        return True

    def _supersede(self, old: QueueJob, new_job_id: str) -> None:
        _logger.info("cancelling %s: superseded by %s", old.job_id, new_job_id)
        self._store.publish_result(
            RunResult(
                run_id=run_id_for(old.job_id),
                job_id=old.job_id,
                team_id=old.team_id,
                submission_id=old.submission_id,
                status=RunStatus.CANCELLED,
                submitted_at=old.enqueued_at,
                finished_at=utcnow(),
                error=f"superseded by {new_job_id} before it started",
            )
        )
        # It never ran, so it should not be refused as a duplicate if the team
        # pushes that manifest again.
        (self._store.seen / f"{old.job_id}.json").unlink(missing_ok=True)
        self._sweep(old.team_id, keep=new_job_id.removeprefix(f"{old.team_id}__"))

    def _sweep(self, team_id: str, keep: str | None = None) -> None:
        """Delete what was fetched for the team that no queue record names or points
        into, in any state, apart from ``keep``: its head, while it is being taken in.

        So whatever stopped a submission between the transfer and the queue — a
        refusal, a newer push, a deferral, the Hub, our disk, the watcher itself — its
        image goes once the team has moved on. A record keeps it: the worker deletes a
        job's image once its result is out, or keeps it after a failure of ours so that
        it can be run again (`evalq submit`), and the record stays in done/ or failed/.
        Only the watcher writes incoming/, so nothing is being fetched into it."""
        try:
            fetched = list((self._store.incoming / team_id).iterdir())
        except FileNotFoundError:
            return
        unnamed = [
            path
            for path in fetched
            if path.name != keep and not self._queue.has_record(job_id_for(team_id, path.name))
        ]
        if not unnamed:
            return
        # A job queued by hand has an id of its own, wherever its image is.
        active = active_jobs(self._queue)
        if active is None:
            return  # the queue would not hold still; the next poll looks again
        images = [Path(job.image_path) for job in [*active, *self._queue.finished_jobs()]]
        for path in unnamed:
            if not any(image.is_relative_to(path) for image in images):
                _logger.info("deleting %s: no job uses it", path)
                shutil.rmtree(path, ignore_errors=True)

    def _sweep_every_team(self) -> None:
        """From the deadline on nothing is taken in, so no head is kept."""
        if self._store.incoming.is_dir():
            for team_dir in self._store.incoming.iterdir():
                self._sweep(team_dir.name)

    @staticmethod
    def _submission_id(candidate: Candidate) -> str:
        return candidate.manifest.submission_id if candidate.manifest else candidate.key[:8]

    def _mark_deferred(self, team_id: str, job_id: str, submission_id: str) -> None:
        """Leave a note that a submission is waiting for the team's running one, which
        nothing else records: it was neither fetched nor enqueued."""
        path = self._store.deferred / f"{team_id}.json"
        try:
            if json.loads(path.read_text()).get("job_id") == job_id:
                return  # keep `since` from the first poll that deferred it
        except (OSError, ValueError):
            pass
        write_json_atomic(
            path,
            {
                "job_id": job_id,
                "team_id": team_id,
                "submission_id": submission_id,
                "since": utcnow().isoformat(),
            },
        )

    def _clear_deferred(self, team_id: str) -> None:
        (self._store.deferred / f"{team_id}.json").unlink(missing_ok=True)

    def _still_trying(self, attempt_id: str) -> bool:
        """Whether a submission the Hub could not serve is within ``RETRY_S`` of the
        first time it could not. That time is kept in the store, so that restarting
        the watcher, or `watch --once` again, does not start the hour over."""
        path = self._store.state / "retrying" / f"{attempt_id}.json"
        try:
            since = datetime.fromisoformat(json.loads(path.read_text())["since"])
        except (OSError, ValueError, KeyError, TypeError):
            since = utcnow()
            write_json_atomic(path, {"since": since.isoformat()})
        return utcnow() - since < timedelta(seconds=RETRY_S)

    def _done_trying(self, attempt_id: str) -> None:
        (self._store.state / "retrying" / f"{attempt_id}.json").unlink(missing_ok=True)

    def _is_seen(self, job_id: str) -> bool:
        return (self._store.seen / f"{job_id}.json").exists()

    def _was_enqueued(self, job_id: str) -> bool:
        """A rejection does not count: one refused for something outside the manifest
        (an access grant, an invite) is looked at again when the same manifest is
        pushed again."""
        try:
            record = json.loads((self._store.seen / f"{job_id}.json").read_text())
        except (OSError, ValueError):
            return False
        return bool(record.get("enqueued"))

    def _mark_seen(self, job_id: str, team_id: str, *, enqueued: bool) -> None:
        write_json_atomic(
            self._store.seen / f"{job_id}.json",
            {
                "job_id": job_id,
                "team_id": team_id,
                "enqueued": enqueued,
                "at": utcnow().isoformat(),
            },
        )

    def _publish_rejection(self, candidate: Candidate, rejection: Rejected) -> None:
        job_id = job_id_for(candidate.team_id, candidate.key)
        submission_id = candidate.manifest.submission_id if candidate.manifest else candidate.key
        self._store.publish_result(
            RunResult.failed(
                run_id_for(job_id),
                job_id,
                candidate.team_id,
                submission_id,
                str(rejection),
                rejection.code,
                status=RunStatus.REJECTED,
                submitted_at=utcnow(),
            )
        )


def _from_hub(job: QueueJob) -> bool:
    """Queued by the watcher, so under the per-team cap; ``evalq submit`` names no repo."""
    return job.hub_repo is not None


__all__ = ["SubmissionWatcher"]
