"""Publish the leaderboard and every team's submissions as one small JSON document.

The document is the whole interface to the static page (web/leaderboard/index_*.html)
that shows it. It holds only what teams are meant to see — one success rate per
team, ranked by its best run, and a single status per submission — and is built
from the published ``RunResult`` files and the queue directories, so it needs no
state of its own beyond the hash of the last document it pushed.

Deliberately left out: per-task and per-subset scores, task counts, budgets, images
and repos, run directories, error text. Those can say things about the hidden pack
or the cluster.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from capx.evalqueue.config import QueueConfig, SubsetConfig
from capx.evalqueue.models import ErrorCode, RunResult, RunStatus, SubsetResult
from capx.evalqueue.queue import FileQueue
from capx.evalqueue.ranking import RankingBook, RankingEntry
from capx.evalqueue.store import EvalStore, utcnow, write_json_atomic
from capx.evalqueue.teams import registered_teams
from capx.protocol.taskpack import load_pack

_logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _load_results(store: EvalStore, teams: set[str]) -> dict[str, list[RunResult]]:
    """Every readable result of a registered team, oldest first. Runs that were queued
    by hand for other teams are not on the board."""
    by_team: dict[str, list[RunResult]] = {team: [] for team in teams}
    for path in store.results.glob("*.json"):
        try:
            result = RunResult.model_validate(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
        if result.team_id in by_team:
            by_team[result.team_id].append(result)
    for results in by_team.values():
        results.sort(key=lambda r: r.finished_at or datetime.min.replace(tzinfo=UTC))
    return by_team


@dataclass(frozen=True)
class Weighting:
    """How the score weighs each simulator's success rate (PRELIM_RULEBOOK.md §4.5): the
    subsets' packs' ``score_weights``, and per subset each task's simulator and trials."""

    weights: dict[str, float]
    tasks: dict[str, dict[str, tuple[str, int]]]


_weighting_cache: dict[tuple[tuple[str, str], ...], Weighting | None] = {}


def score_weighting(config: QueueConfig) -> Weighting | None:
    """The subsets' weighting, or None when their packs weigh every trial the same. The
    packs must agree, since one score cannot weigh a simulator two ways."""
    key = tuple((subset.name, config.resolve(subset.taskpack)) for subset in config.subsets)
    if key not in _weighting_cache:
        packs = {name: load_pack(path)[0] for name, path in key}
        weights = [pack.score_weights for pack in packs.values()]
        if any(w != weights[0] for w in weights):
            raise ValueError("the subsets' task packs set different score_weights")
        _weighting_cache[key] = (
            Weighting(
                weights=dict(weights[0]),
                tasks={
                    name: {t.task_id: (t.simulator, t.n_trials) for t in pack.tasks}
                    for name, pack in packs.items()
                },
            )
            if weights and weights[0]
            else None
        )
    return _weighting_cache[key]


def score_result(
    result: RunResult, expected_subsets: list[str], weighting: Weighting | None = None
) -> tuple[float | None, int]:
    """(score, subsets scored) for one run (PRELIM_RULEBOOK.md §4.5): each simulator's
    successful trials over its trials, averaged with ``weighting``'s weights over the
    simulators the run has trials on. Without a weighting, successful trials over every
    trial of every subset.

    A subset that ran past its time budget, or whose submission exited early or was
    stopped for writing too much, is scored on what the gateway recorded (§5): the
    trials that ran, and a failure for every trial it never reached
    (``Gateway.aggregate``). One that failed on our side is not, since it is run again,
    and neither is a subset that never reported, so ``score`` is None unless every
    expected subset has numbers. Counting only the survivors would reward a crash on
    the hard one.
    """
    successes: dict[str, float] = {}
    trials: dict[str, float] = {}
    done = 0
    for name in expected_subsets:
        subset = result.subsets.get(name)
        board = subset.leaderboard if subset and _counts(subset) else None
        counts = _subset_counts(board, weighting.tasks.get(name, {}) if weighting else None)
        if counts is None:
            continue
        for simulator, (subset_successes, subset_trials) in counts.items():
            successes[simulator] = successes.get(simulator, 0) + subset_successes
            trials[simulator] = trials.get(simulator, 0) + subset_trials
        done += 1
    if done != len(expected_subsets) or not expected_subsets:
        return None, done
    if weighting is None:
        total = sum(trials.values())
        return (sum(successes.values()) / total if total else None), done
    weights = {sim: weighting.weights[sim] for sim, n in trials.items() if n}
    if not weights:
        return None, done
    weighted = sum(w * successes[sim] / trials[sim] for sim, w in weights.items())
    return weighted / sum(weights.values()), done


def _subset_counts(
    board: dict[str, Any] | None, tasks: dict[str, tuple[str, int]] | None
) -> dict[str, tuple[float, float]] | None:
    """A subset's (successes, trials) per simulator, from the gateway's leaderboard; all
    under one key when ``tasks`` (task_id -> simulator, trials) is None. None when the
    board lacks them: written before the totals existed, or for tasks the pack no
    longer has."""
    if not board:
        return None
    if tasks is None:
        subset_successes, subset_trials = board.get("successes"), board.get("trials")
        if not isinstance(subset_successes, int) or not isinstance(subset_trials, int):
            return None
        return {"": (subset_successes, subset_trials)}
    per_task = board.get("tasks")
    if not isinstance(per_task, dict):
        return None
    counts: dict[str, tuple[float, float]] = {}
    for task_id, metrics in per_task.items():
        rate = metrics.get("success_rate") if isinstance(metrics, dict) else None
        if task_id not in tasks or not isinstance(rate, int | float):
            return None
        simulator, n = tasks[task_id]
        prior_successes, prior_trials = counts.get(simulator, (0.0, 0.0))
        counts[simulator] = (prior_successes + rate * n, prior_trials + n)
    return counts


_SCORED_FAILURES = {ErrorCode.SUBMISSION_FAILED, ErrorCode.SUBMISSION_STORAGE_EXCEEDED}
"""Failures that end a submission the way an exit does (PRELIM_RULEBOOK.md A.1)."""


def _counts(subset: SubsetResult) -> bool:
    if subset.status in (RunStatus.COMPLETED, RunStatus.TIMEOUT):
        return True
    return subset.status is RunStatus.FAILED and subset.error_code in _SCORED_FAILURES


_budget_cache: dict[tuple[str, tuple[str, ...]], float] = {}


def _budget_s(config: QueueConfig, subset: SubsetConfig) -> float:
    """The subset's job budget, as the runner gives it: the pack's, over the subset's tasks."""
    key = (subset.taskpack, tuple(subset.simulators))
    if key not in _budget_cache:
        manifest, _ = load_pack(config.resolve(subset.taskpack))
        _budget_cache[key] = manifest.job_timeout_s(subset.simulators or None)
    return _budget_cache[key]


def _time_taken_s(result: RunResult, config: QueueConfig) -> float | None:
    """How long the submission spent inside the benchmarks, on the clock the referee
    enforces the time limit with: per subset, from the submission's first contact with
    the gateway to its exit, capped at the subset's job budget, so the total never
    exceeds the published limit. Loading the model is left out; the time between
    episodes counts, as it does against the limit. This is the tiebreaker.
    """
    spent = []
    for subset in config.subsets:
        used = result.subsets.get(subset.name)
        if used and used.budget_used_s is not None:
            spent.append(min(used.budget_used_s, _budget_s(config, subset)))
    return round(sum(spent), 1) if spent else None


def _wall_s(result: RunResult) -> float | None:
    """The whole evaluation on the clock: from the worker starting the run to its result,
    container and model start-up included. Shown beside Time Taken, never ranked on."""
    if result.started_at and result.finished_at:
        return round((result.finished_at - result.started_at).total_seconds(), 1)
    return None


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if value else None


def run_over_time(result: RunResult) -> bool:
    """Whether any subset of this run ran out of time."""
    return any(s.status == "timeout" or s.over_time for s in result.subsets.values())


def _team_entry(
    team_id: str,
    results: list[RunResult],
    config: QueueConfig,
    expected: list[str],
    weighting: Weighting | None,
) -> tuple[dict[str, Any], str | None]:
    """The team's leaderboard entry, and the run_id it is ranked by (None if unranked)."""
    scored = [
        (r, *score_result(r, expected, weighting))
        for r in results
        if r.status in (RunStatus.COMPLETED, RunStatus.TIMEOUT, RunStatus.FAILED)
    ]
    ranked = [(r, score, done) for r, score, done in scored if score is not None]
    entry: dict[str, Any] = {"team_id": team_id, "score": None, "rank": None}
    if ranked:
        # The same order teams are ranked in, then who got there first, so
        # resubmitting an equal result does not replace the entry.
        chosen: tuple[RunResult, float | None, int] = min(
            ranked,
            key=lambda row: (
                *_standing(row[1], _time_taken_s(row[0], config)),
                _iso(row[0].finished_at) or "",
            ),
        )
    elif scored:
        chosen = scored[-1]
    else:
        return entry, None
    result, score, done = chosen
    entry.update(
        score=score,
        submission_id=result.submission_id,
        submitted_at=_iso(result.submitted_at),
        finished_at=_iso(result.finished_at),
        time_taken_s=_time_taken_s(result, config),
        over_time=run_over_time(result),
        subsets_completed=done,
        subsets_expected=len(expected),
    )
    return entry, result.run_id if score is not None else None


def _standing(score: float, time_taken_s: float | None) -> tuple[float, float]:
    """What a rank is decided on, best first (PRELIM_RULEBOOK.md §4.5): the higher
    score, then the shorter time taken. Teams equal on both share a rank. The score is
    rounded so float noise does not split a tie."""
    return (-round(score, 9), time_taken_s if time_taken_s is not None else float("inf"))


def _status_of(result: RunResult) -> str:
    """One word for how a finished submission ended. A run that ran out of time is
    still scored, but that is the more useful thing to say than "done"."""
    if result.status is RunStatus.TIMEOUT:
        return "over-time"
    if result.status is RunStatus.COMPLETED:
        return "over-time" if run_over_time(result) else "done"
    return result.status.value


def _team_activity(
    team_id: str,
    results: list[RunResult],
    running: dict[str, Any],
    pending: list[Any],
    store: EvalStore,
    ranked_run: str | None,
    expected: list[str],
    weighting: Weighting | None,
) -> list[dict[str, Any]]:
    """Every submission of a team, the most recently submitted first, each with exactly
    one status, and its score when it has one. The run the team is ranked by carries
    ``ranked: true``.

    A submission that is running, waiting or held back is one entry; a finished one
    is one entry; so a team's whole history reads top to bottom. The page shows the
    first ``activity_per_team`` and folds the rest behind "Show more". The raw
    ``error`` of a result never appears (it can name repos, hashes and paths, and
    this document is meant to be public); its ``error_code`` does, so a team sees why a
    submission was rejected or failed.
    """
    floor = datetime.min.replace(tzinfo=UTC)
    rows: list[tuple[datetime, dict[str, Any]]] = []

    if team_id in running:
        job = running[team_id]
        rows.append(
            (
                job.enqueued_at,
                {"status": "running", "submission_id": job.submission_id,
                 "submitted_at": _iso(job.enqueued_at), "started_at": _iso(job.claimed_at)},
            )
        )
        marker = store.deferred / f"{team_id}.json"
        # A marker without a running job is stale (the running job finished and the
        # next poll has not taken the deferred one in yet), so it only counts while the
        # job it waits behind runs.
        try:
            note = json.loads(marker.read_text())
            since = datetime.fromisoformat(note["since"])
            rows.append(
                (
                    since,
                    {"status": "deferred", "submission_id": note.get("submission_id", ""),
                     "submitted_at": _iso(since)},
                )
            )
        except (OSError, ValueError, KeyError):
            pass
    for position, job in enumerate(pending, start=1):
        if job.team_id == team_id:
            rows.append(
                (
                    job.enqueued_at,
                    {"status": "waiting", "position": position,
                     "submission_id": job.submission_id, "submitted_at": _iso(job.enqueued_at)},
                )
            )
    for result in results:
        finished = result.status in (RunStatus.COMPLETED, RunStatus.TIMEOUT, RunStatus.FAILED)
        rows.append(
            (
                result.submitted_at or result.finished_at or floor,
                {
                    "status": _status_of(result),
                    "submission_id": result.submission_id,
                    "wall_s": _wall_s(result) if finished else None,
                    "score": score_result(result, expected, weighting)[0] if finished else None,
                    "submitted_at": _iso(result.submitted_at),
                    "finished_at": _iso(result.finished_at),
                    "ranked": result.run_id == ranked_run,
                    "error_code": result.error_code.value if result.error_code else None,
                },
            )
        )
    rows.sort(key=lambda row: row[0], reverse=True)
    return [entry for _, entry in rows]


def _ranking_view(entry: RankingEntry | None) -> dict[str, Any] | None:
    """What the board says of a team's ranking submission: only how its check came out
    (PRELIM_RULEBOOK.md §6.5). ``final`` once it is the one recorded at the deadline."""
    if entry is None:
        return None
    return {
        "status": entry.status.value,
        "submission_id": entry.manifest.submission_id if entry.manifest else None,
        "checked_at": _iso(entry.checked_at),
        "error_code": entry.error_code.value if entry.error_code else None,
        "final": entry.final,
    }


def build_snapshot(config: QueueConfig, store: EvalStore, now: datetime | None = None) -> dict:
    now = now or utcnow()
    team_ids = [team.team_id for team in registered_teams(store)]
    expected = [subset.name for subset in config.subsets]
    weighting = score_weighting(config)
    queue = FileQueue(store)
    results = _load_results(store, set(team_ids))

    running = {job.team_id: job for job in queue.claimed_jobs()}
    pending = [job for job in queue.pending_jobs() if job.team_id in set(team_ids)]

    entries = {t: _team_entry(t, results[t], config, expected, weighting) for t in team_ids}
    board = [entry for entry, _ in entries.values()]
    # Teams that share a rank are listed by who finished first.
    ranked = sorted(
        (e for e in board if e["score"] is not None),
        key=lambda e: (*_standing(e["score"], e["time_taken_s"]), e["finished_at"] or ""),
    )
    previous: tuple[float, float] | None = None
    rank = 0
    for index, entry in enumerate(ranked, start=1):
        standing = _standing(entry["score"], entry["time_taken_s"])
        if standing != previous:
            rank, previous = index, standing
        entry["rank"] = rank
    board = ranked + sorted(
        (e for e in board if e["score"] is None), key=lambda e: e["team_id"]
    )

    ranking = RankingBook(store)
    teams = [
        {
            "team_id": team_id,
            "activity": _team_activity(
                team_id,
                results[team_id],
                running,
                pending,
                store,
                entries[team_id][1],
                expected,
                weighting,
            ),
            "ranking": _ranking_view(ranking.get(team_id)),
        }
        for team_id in team_ids
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _iso(now),
        "heartbeat_s": config.leaderboard.heartbeat_s,
        "scoring": {
            "subsets": [config.leaderboard.subset_labels.get(name, name) for name in expected],
            # Only the total: each benchmark's own budget stays private.
            "time_limit_s": sum(_budget_s(config, s) for s in config.subsets),
        },
        "leaderboard": board,
        "queue": {"teams": teams, "shown": config.leaderboard.activity_per_team},
    }


def _fingerprint(snapshot: dict) -> str:
    stable = {k: v for k, v in snapshot.items() if k != "generated_at"}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


Uploader = Callable[[bytes, str], None]
"""``(document bytes, commit message) -> None``; swapped out in tests."""


def hub_uploader(config: QueueConfig) -> Uploader:
    from huggingface_hub import HfApi  # imported here for the same reason as HubSource

    leaderboard = config.leaderboard
    api = HfApi(token=os.environ.get(leaderboard.token_env) or None)

    def upload(document: bytes, message: str) -> None:
        api.upload_file(
            path_or_fileobj=document,
            path_in_repo=leaderboard.path,
            repo_id=leaderboard.repo,
            repo_type=leaderboard.repo_type,
            commit_message=message,
        )

    return upload


def publish_once(
    config: QueueConfig,
    store: EvalStore,
    uploader: Uploader | None = None,
    *,
    force: bool = False,
) -> bool:
    """Build the snapshot and push it if it differs from the last one pushed, or if
    the last push is ``heartbeat_s`` old.

    Returns whether it pushed. ``generated_at`` is left out of the comparison, or
    every call would be a commit. The marker's mtime is when the last push happened,
    so the heartbeat survives a restart.
    """
    snapshot = build_snapshot(config, store)
    fingerprint = _fingerprint(snapshot)
    marker = store.state / "leaderboard.sha256"
    unchanged = marker.exists() and marker.read_text().strip() == fingerprint
    if not force and unchanged and time.time() - marker.stat().st_mtime < config.leaderboard.heartbeat_s:
        return False
    upload = uploader or hub_uploader(config)
    document = json.dumps(snapshot, indent=2, sort_keys=True).encode()
    upload(document, f"leaderboard {snapshot['generated_at']}")
    write_json_atomic(store.state / "leaderboard.json", snapshot)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(fingerprint)
    return True


def run_forever(config: QueueConfig, store: EvalStore) -> None:
    if not config.leaderboard.repo:
        raise SystemExit("leaderboard.repo is not set in the queue config")
    while True:
        try:
            if publish_once(config, store):
                _logger.info("published leaderboard to %s", config.leaderboard.repo)
        except Exception:  # a Hub hiccup must not end the loop
            _logger.exception("leaderboard publish failed")
        time.sleep(config.leaderboard.interval_s)


__all__ = ["Weighting", "build_snapshot", "publish_once", "run_forever", "score_result", "score_weighting"]
