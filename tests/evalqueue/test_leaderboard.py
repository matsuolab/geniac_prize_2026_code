"""The published snapshot: who ranks where, and what a team sees of the queue."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from capx.evalqueue.config import HubTeam, LeaderboardConfig, QueueConfig, SubsetConfig
from capx.evalqueue.leaderboard import (
    build_snapshot,
    publish_once,
    score_result,
    score_weighting,
)
from capx.evalqueue.models import ErrorCode, QueueJob, RunResult, RunStatus, SubsetResult
from capx.evalqueue.queue import FileQueue
from capx.evalqueue.store import EvalStore, utcnow
from capx.evalqueue.teams import save_registered_teams

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _pack(path, trial_time_s=1800.0):
    """A one-task pack, so its job budget is ``trial_time_s``."""
    path.mkdir(exist_ok=True)
    task = {"task_id": "t", "embodiment": "franka", "simulator": "robosuite", "instruction": "go",
            "camera_names": ["c"], "env_target": "fake"}
    limits = {"trial_time_s": trial_time_s, "n_trials": 1, "trial_steplimit": 100}
    (path / "pack.yaml").write_text(yaml.safe_dump(
        {"pack_id": path.name, "version": "1", "visibility": "public", "limits": limits, "tasks": [task]}
    ))
    return str(path)


def _config(tmp_path, **board) -> QueueConfig:
    return QueueConfig(
        root=str(tmp_path),
        master_seed=1,
        subsets=[
            SubsetConfig(name="rs", taskpack=_pack(tmp_path / "p")),
            SubsetConfig(name="lib", taskpack=_pack(tmp_path / "q")),
        ],
        leaderboard=LeaderboardConfig(repo="o/lb", **board),
    )


def _subset(tasks: dict[str, float], status="completed", trials=2) -> SubsetResult:
    """``tasks`` maps each task to its success rate over ``trials`` trials."""
    successes = {k: round(v * trials) for k, v in tasks.items()}
    board = {
        "overall_success_rate": sum(successes.values()) / (trials * len(tasks)),
        "successes": sum(successes.values()),
        "trials": trials * len(tasks),
        "tasks": {k: {"success_rate": v} for k, v in tasks.items()},
    }
    return SubsetResult(status=status, leaderboard=board if status == "completed" else None)


def _result(store, team, n, rs, lib, *, status=RunStatus.COMPLETED, hours=0, minutes=(10, 20),
            submitted=None):
    """``minutes`` is each subset's time on the job clock; ``submitted`` is minutes after T0."""
    if rs.budget_used_s is None:
        rs.budget_used_s = minutes[0] * 60
    if lib is not None and lib.budget_used_s is None:
        lib.budget_used_s = minutes[1] * 60
    subsets = {"rs": rs}
    if lib is not None:
        subsets["lib"] = lib
    result = RunResult(
        run_id=f"{team}__{n}",
        job_id=f"{team}__{n}",
        team_id=team,
        submission_id=f"sub-{n}",
        status=status,
        submitted_at=T0 + timedelta(minutes=submitted) if submitted is not None else None,
        finished_at=T0 + timedelta(hours=hours),
        subsets=subsets,
        error="/home/secret/path blew up",
    )
    store.publish_result(result)
    return result


def _job(team, n, enqueued=0):
    return QueueJob(
        job_id=f"{team}__{n}",
        team_id=team,
        submission_id=f"sub-{n}",
        image_path="/x/agent.sif",
        hub_repo=f"{team}-org/sub",
        enqueued_at=T0 + timedelta(minutes=enqueued),
    )


def _store(tmp_path):
    store = EvalStore(tmp_path)
    store.ensure_layout()
    save_registered_teams(
        store, [HubTeam(team_id=t, repo=f"{t}-org/sub") for t in ("alpha", "beta", "gamma")]
    )
    return store


def test_the_score_is_successes_over_every_trial(tmp_path):
    store = _store(tmp_path)
    # 3 of 4 trials in one subset and 0 of 4 in the other. The mean of the per-task
    # rates would be (1.0 + 0.5 + 0.0) / 3.
    rs, lib = _subset({"a": 1.0, "b": 0.5}), _subset({"d": 0.0}, trials=4)
    result = _result(store, "alpha", 1, rs, lib)
    assert score_result(result, ["rs", "lib"])[0] == 3 / 8


def _weighted_pack(path, weights, extends=None):
    """Task ``a`` on robosuite with 4 trials, ``b`` on libero with 1."""
    path.mkdir(exist_ok=True)
    limits = {"trial_time_s": 60.0, "n_trials": 1, "trial_steplimit": 100}
    pack = {"pack_id": path.name, "visibility": "private", "limits": limits, "score_weights": weights}
    if extends:
        pack.update(extends=extends, task_ids=["a", "b"])
    else:
        task = {"embodiment": "franka", "instruction": "go", "camera_names": ["c"], "env_target": "fake"}
        pack["tasks"] = [
            {**task, "task_id": "a", "simulator": "robosuite", "n_trials": 4},
            {**task, "task_id": "b", "simulator": "libero"},
        ]
    (path / "pack.yaml").write_text(yaml.safe_dump(pack))
    return str(path)


def _weighted_config(tmp_path, *packs, simulators=()):
    return QueueConfig(
        root=str(tmp_path),
        master_seed=1,
        subsets=[
            SubsetConfig(name=f"s{i}", taskpack=pack, simulators=list(simulators))
            for i, pack in enumerate(packs)
        ],
        leaderboard=LeaderboardConfig(repo="o/lb"),
    )


def _weighted_result(rates):
    board = {"tasks": {task: {"success_rate": rate} for task, rate in rates.items()}}
    return RunResult(
        run_id="r", job_id="r", team_id="alpha", submission_id="s", status=RunStatus.COMPLETED,
        subsets={"s0": SubsetResult(status="completed", leaderboard=board)},
    )


def test_the_score_weighs_each_simulators_success_rate(tmp_path):
    pack = _weighted_pack(tmp_path / "w", {"robosuite": 0.25, "libero": 0.75})
    weighting = score_weighting(_weighted_config(tmp_path, pack))
    score, done = score_result(_weighted_result({"a": 0.75, "b": 0.0}), ["s0"], weighting)
    # robosuite 3/4 and libero 0/1; every trial alike would give 3/5.
    assert score == pytest.approx(0.25 * 0.75) and done == 1


def test_the_weights_are_shared_by_the_simulators_the_run_has(tmp_path):
    pack = _weighted_pack(tmp_path / "w", {"robosuite": 0.25, "libero": 0.75})
    weighting = score_weighting(_weighted_config(tmp_path, pack, simulators=["robosuite"]))
    assert score_result(_weighted_result({"a": 0.5}), ["s0"], weighting)[0] == pytest.approx(0.5)


def test_an_extending_pack_weighs_with_its_own_weights(tmp_path):
    _weighted_pack(tmp_path / "base", {"robosuite": 0.25, "libero": 0.75})
    pack = _weighted_pack(tmp_path / "sub", {"robosuite": 0.5, "libero": 0.5}, extends="base")
    weighting = score_weighting(_weighted_config(tmp_path, pack))
    assert weighting is not None and weighting.weights == {"robosuite": 0.5, "libero": 0.5}


def test_subsets_whose_packs_weigh_differently_are_refused(tmp_path):
    first = _weighted_pack(tmp_path / "w1", {"robosuite": 0.25, "libero": 0.75})
    second = _weighted_pack(tmp_path / "w2", {"robosuite": 0.5, "libero": 0.5})
    with pytest.raises(ValueError, match="different score_weights"):
        score_weighting(_weighted_config(tmp_path, first, second))


def test_packs_without_weights_weigh_every_trial_the_same(tmp_path):
    assert score_weighting(_config(tmp_path)) is None


def test_a_result_without_trial_totals_is_unscored(tmp_path):
    store = _store(tmp_path)
    old = SubsetResult(status="completed", leaderboard={"tasks": {"a": {"success_rate": 1.0}}})
    result = _result(store, "alpha", 1, _subset({"a": 1.0}), old)
    assert score_result(result, ["rs", "lib"]) == (None, 1)


def test_a_timed_out_subset_is_scored_on_what_it_recorded(tmp_path):
    store = _store(tmp_path)
    cut = _subset({"d": 0.0, "e": 1.0}, status="completed")
    cut.status = "timeout"
    _result(store, "alpha", 1, _subset({"a": 1.0}), cut, status=RunStatus.TIMEOUT)
    entry = build_snapshot(_config(tmp_path), store)["leaderboard"][0]
    assert entry["rank"] == 1 and entry["over_time"] is True
    assert entry["score"] == (1.0 + 0.0 + 1.0) / 3


def test_a_full_run_is_not_marked_over_time(tmp_path):
    store = _store(tmp_path)
    _result(store, "alpha", 1, _subset({"a": 1.0}), _subset({"b": 1.0}))
    assert build_snapshot(_config(tmp_path), store)["leaderboard"][0]["over_time"] is False


def test_a_timeout_without_numbers_is_still_unranked(tmp_path):
    store = _store(tmp_path)
    cut = _subset({"d": 0.0}, status="failed")
    cut.status = "timeout"  # the budget ran out before the gateway reported anything
    _result(store, "alpha", 1, _subset({"a": 1.0}), cut, status=RunStatus.TIMEOUT)
    assert build_snapshot(_config(tmp_path), store)["leaderboard"][0]["rank"] is None


def test_a_missing_or_failed_subset_means_no_score(tmp_path):
    store = _store(tmp_path)
    missing = _result(store, "alpha", 1, _subset({"a": 1.0}), None)
    failed = _result(store, "alpha", 2, _subset({"a": 1.0}), _subset({"d": 0.0}, status="failed"))
    assert score_result(missing, ["rs", "lib"])[0] is None
    assert score_result(failed, ["rs", "lib"])[0] is None
    assert score_result(failed, ["rs", "lib"])[1] == 1


@pytest.mark.parametrize("code", [ErrorCode.SUBMISSION_FAILED, ErrorCode.SUBMISSION_STORAGE_EXCEEDED])
def test_a_submission_that_exited_early_is_scored_on_what_it_recorded(tmp_path, code):
    """Stopping one that wrote too much ends it the same way (rulebook §5)."""
    store = _store(tmp_path)
    crashed = _subset({"b": 0.5})
    crashed.status, crashed.error_code = RunStatus.FAILED, code
    result = _result(store, "alpha", 1, _subset({"a": 1.0}), crashed, status=RunStatus.FAILED)
    assert score_result(result, ["rs", "lib"]) == (3 / 4, 2)


def test_a_subset_that_failed_on_our_side_is_not_scored(tmp_path):
    store = _store(tmp_path)
    broken = _subset({"b": 0.5})
    broken.status, broken.error_code = RunStatus.FAILED, ErrorCode.INTERNAL_ERROR
    result = _result(store, "alpha", 1, _subset({"a": 1.0}), broken, status=RunStatus.FAILED)
    assert score_result(result, ["rs", "lib"]) == (None, 1)


def test_ranks_by_score_and_lists_unscored_teams_last(tmp_path):
    store = _store(tmp_path)
    _result(store, "alpha", 1, _subset({"a": 0.5}), _subset({"b": 0.5}))
    _result(store, "beta", 1, _subset({"a": 1.0}), _subset({"b": 1.0}))
    board = build_snapshot(_config(tmp_path), store)["leaderboard"]
    assert [(e["team_id"], e["rank"]) for e in board] == [
        ("beta", 1), ("alpha", 2), ("gamma", None)
    ]
    assert board[0]["score"] == 1.0 and board[2]["score"] is None


def test_a_team_is_ranked_by_its_best_run_not_its_latest(tmp_path):
    store = _store(tmp_path)
    _result(store, "alpha", 1, _subset({"a": 1.0}), _subset({"b": 1.0}), hours=1)
    _result(store, "alpha", 2, _subset({"a": 0.0}), _subset({"b": 0.0}), hours=2)
    best = build_snapshot(_config(tmp_path), store)["leaderboard"][0]
    assert (best["submission_id"], best["score"]) == ("sub-1", 1.0)


def test_a_teams_best_tie_goes_to_the_shorter_then_the_earlier_run(tmp_path):
    store = _store(tmp_path)
    _result(store, "alpha", 1, _subset({"a": 1.0}), _subset({"b": 0.0}), hours=1, minutes=(30, 30))
    _result(store, "alpha", 2, _subset({"a": 1.0}), _subset({"b": 0.0}), hours=2, minutes=(5, 5))
    _result(store, "alpha", 3, _subset({"a": 1.0}), _subset({"b": 0.0}), hours=3, minutes=(5, 5))
    assert build_snapshot(_config(tmp_path), store)["leaderboard"][0]["submission_id"] == "sub-2"


def test_the_ranked_submission_is_marked_in_the_status_table(tmp_path):
    store, activity = _activity(tmp_path)
    _result(store, "alpha", 1, _subset({"a": 1.0}), _subset({"b": 1.0}), hours=1)
    _result(store, "alpha", 2, _subset({"a": 0.0}), _subset({"b": 0.0}), hours=2)
    _result(store, "beta", 1, _subset({"a": 1.0}), _subset({"b": 0.0}), hours=1)
    rows = activity()["alpha"]
    assert [(a["submission_id"], a["ranked"]) for a in rows] == [("sub-2", False), ("sub-1", True)]
    store.publish_result(RunResult(run_id="gamma__1", job_id="gamma__1", team_id="gamma", submission_id="bad",
                  status=RunStatus.REJECTED, finished_at=T0),
    )
    assert activity()["gamma"][0]["ranked"] is False  # nothing scored, nothing ranked


def test_results_of_teams_that_are_not_registered_are_ignored(tmp_path):
    store = _store(tmp_path)
    _result(store, "stranger", 1, _subset({"a": 1.0}), _subset({"b": 1.0}))
    board = build_snapshot(_config(tmp_path), store)["leaderboard"]
    assert {e["team_id"] for e in board} == {"alpha", "beta", "gamma"}


def _activity(tmp_path, **board):
    store = _store(tmp_path)
    return store, (lambda: {
        t["team_id"]: t["activity"]
        for t in build_snapshot(_config(tmp_path, **board), store)["queue"]["teams"]
    })


def test_each_team_lists_its_running_deferred_and_waiting_submissions(tmp_path):
    store, activity = _activity(tmp_path)
    queue = FileQueue(store)
    queue.enqueue(_job("alpha", 1))
    assert queue.claim("w1")
    queue.enqueue(_job("beta", 1, enqueued=1))
    queue.enqueue(_job("gamma", 1, enqueued=2))
    (store.deferred / "alpha.json").write_text(
        json.dumps(
            {
                "job_id": "alpha__2",
                "submission_id": "newer",
                "since": (utcnow() + timedelta(hours=1)).isoformat(),  # pushed after the claim
            }
        )
    )
    rows = activity()
    assert [(a["status"], a["submission_id"]) for a in rows["alpha"]] == [
        ("deferred", "newer"), ("running", "sub-1")
    ]
    assert rows["alpha"][1]["started_at"] is not None  # the queue section times the run from it
    assert [(a["status"], a["position"]) for a in rows["beta"]] == [("waiting", 1)]
    assert [(a["status"], a["position"]) for a in rows["gamma"]] == [("waiting", 2)]
    assert all(set(a) >= {"status", "submission_id"} for rs in rows.values() for a in rs)


def test_progress_is_a_floored_percent_of_every_trial_and_never_a_count(tmp_path):
    # Two one-trial packs: two trials in all.
    store, activity = _activity(tmp_path)
    queue = FileQueue(store)
    queue.enqueue(_job("alpha", 1))
    assert queue.claim("w1")
    assert activity()["alpha"][0]["progress"] is None  # no run dir yet
    run = store.runs / "alpha__1__20261001T000000Z" / "rs"
    run.mkdir(parents=True)
    assert activity()["alpha"][0]["progress"] == 0  # started, no gateway dump yet
    (run / "gateway-results.json").write_text(json.dumps({"scores": [0.0]}))
    assert activity()["alpha"][0]["progress"] == 50

    rs, lib = _subset({"t": 1.0}, trials=1), _subset({"t": 0.0}, status="failed", trials=1)
    rs.trials_run, lib.trials_run = 1, 0
    _result(store, "beta", 1, rs, lib, status=RunStatus.FAILED)
    _result(store, "gamma", 1, _subset({"t": 1.0}, trials=1), _subset({"t": 1.0}, trials=1))
    rows = activity()
    assert rows["beta"][0]["progress"] == 50
    assert rows["gamma"][0]["progress"] is None  # a result from before the count was kept

    snapshot = json.dumps(build_snapshot(_config(tmp_path), store))
    assert "trials_run" not in snapshot and "scores" not in snapshot


def test_a_stale_deferral_marker_is_not_shown(tmp_path):
    store, activity = _activity(tmp_path)
    (store.deferred / "alpha.json").write_text(
        json.dumps({"job_id": "x", "submission_id": "s", "since": "2026-10-01T00:00:00+00:00"})
    )
    assert activity()["alpha"] == []


def test_cancelled_and_rejected_are_listed_with_one_status_each(tmp_path):
    store, activity = _activity(tmp_path)
    for n, status, hours in ((1, RunStatus.CANCELLED, 1), (2, RunStatus.REJECTED, 2)):
        store.publish_result(
            RunResult(
                run_id=f"alpha__{n}", job_id=f"alpha__{n}", team_id="alpha", submission_id=f"s{n}",
                status=status, finished_at=T0 + timedelta(hours=hours),
                error="alpha-org/private-repo@deadbeef: bad sha256 /home/secret",
            ),
        )
    rows = activity()["alpha"]
    assert [(a["status"], a["submission_id"]) for a in rows] == [
        ("rejected", "s2"), ("cancelled", "s1")
    ]
    assert "deadbeef" not in json.dumps(rows) and "/home/secret" not in json.dumps(rows)
    store.publish_result(
        RunResult(
            run_id="alpha__3", job_id="alpha__3", team_id="alpha", submission_id="s3",
            status=RunStatus.CANCELLED, submitted_at=T0 + timedelta(minutes=5),
            finished_at=T0 + timedelta(hours=3),
        ),
    )
    by_id = {a["submission_id"]: a for a in activity()["alpha"]}
    assert by_id["s3"]["submitted_at"] == "2026-10-01T00:05:00Z"
    assert activity()["gamma"] == []


def test_rejected_and_failed_submissions_show_their_error_code_not_their_error(tmp_path):
    store, activity = _activity(tmp_path)
    for n, status, code in ((1, RunStatus.REJECTED, ErrorCode.IMAGE_TOO_LARGE),
                            (2, RunStatus.FAILED, ErrorCode.SUBMISSION_FAILED)):
        store.publish_result(RunResult.failed(
            f"alpha__{n}", f"alpha__{n}", "alpha", f"s{n}", "/home/secret/agent.sif: exit 3", code,
            status=status, submitted_at=T0 + timedelta(hours=n),
        ))
    rows = {a["submission_id"]: a for a in activity()["alpha"]}
    assert (rows["s1"]["status"], rows["s1"]["error_code"]) == ("rejected", "image_too_large")
    assert (rows["s2"]["status"], rows["s2"]["error_code"]) == ("failed", "submission_failed")
    assert all("error" not in a for a in rows.values())
    assert "/home/secret" not in json.dumps(rows)
    # Codes are published as they are, so none may carry a path.
    assert all("/" not in code.value for code in ErrorCode)


def test_the_snapshot_leaks_no_paths_images_or_task_scores(tmp_path):
    store = _store(tmp_path)
    _result(store, "alpha", 1, _subset({"secret_task": 1.0}), _subset({"b": 1.0}))
    queue = FileQueue(store)
    queue.enqueue(_job("beta", 1))
    text = json.dumps(build_snapshot(_config(tmp_path), store))
    for forbidden in ("secret_task", "/home/secret", "agent.sif", "beta-org/sub", "/x"):
        assert forbidden not in text
    # Not even how many tasks there are: the page shows a score, not the pack.
    assert all("tasks" not in e for e in json.loads(text)["leaderboard"])


def test_publish_only_pushes_when_something_changed(tmp_path):
    store = _store(tmp_path)
    config = _config(tmp_path)
    pushed: list[bytes] = []
    upload = lambda doc, message: pushed.append(doc)  # noqa: E731

    assert publish_once(config, store, upload) is True
    assert publish_once(config, store, upload) is False
    _result(store, "alpha", 1, _subset({"a": 1.0}), _subset({"b": 1.0}))
    assert publish_once(config, store, upload) is True
    assert publish_once(config, store, upload, force=True) is True
    assert len(pushed) == 3
    assert json.loads(pushed[-1])["leaderboard"][0]["team_id"] == "alpha"
    assert utcnow() > T0


def test_an_unchanged_board_is_pushed_again_once_the_heartbeat_is_due(tmp_path):
    store = _store(tmp_path)
    config = _config(tmp_path, heartbeat_s=600.0)
    pushed: list[bytes] = []
    upload = lambda doc, message: pushed.append(doc)  # noqa: E731
    assert publish_once(config, store, upload) is True
    assert publish_once(config, store, upload) is False
    marker = store.state / "leaderboard.sha256"
    old = marker.stat().st_mtime - 601
    os.utime(marker, (old, old))
    assert publish_once(config, store, upload) is True
    assert json.loads(pushed[-1])["heartbeat_s"] == 600.0


def test_the_entry_carries_when_it_was_submitted(tmp_path):
    store = _store(tmp_path)
    _result(store, "alpha", 1, _subset({"a": 1.0}), _subset({"b": 1.0}), hours=5, submitted=30)
    entry = build_snapshot(_config(tmp_path), store)["leaderboard"][0]
    assert entry["submitted_at"] == "2026-10-01T00:30:00Z"
    assert entry["finished_at"] == "2026-10-01T05:00:00Z"


def test_equal_scores_are_ordered_by_time_taken(tmp_path):
    store = _store(tmp_path)
    # alpha finished first on the clock, but spent longer on the runs.
    _result(store, "alpha", 1, _subset({"a": 1.0}), _subset({"b": 0.0}), hours=1, minutes=(30, 60))
    _result(store, "beta", 1, _subset({"a": 1.0}), _subset({"b": 0.0}), hours=2, minutes=(5, 10))
    board = build_snapshot(_config(tmp_path), store)["leaderboard"]
    assert [e["team_id"] for e in board[:2]] == ["beta", "alpha"]
    # alpha's 60-minute subset counts only up to its 30-minute budget (the pack's).
    assert board[0]["time_taken_s"] == 900.0 and board[1]["time_taken_s"] == 1800.0 + 1800.0


def test_time_taken_is_the_budget_clock_capped_at_each_budget(tmp_path):
    store = _store(tmp_path)
    config = _config(tmp_path)
    _pack(tmp_path / "p", 600.0)
    _pack(tmp_path / "q", 600.0)
    rs, lib = _subset({"a": 1.0}), _subset({"b": 1.0})
    rs.budget_used_s, lib.budget_used_s = 450.0, 900.0  # lib ran past its budget
    _result(store, "alpha", 1, rs, lib, minutes=(40, 40))
    entry = build_snapshot(config, store)["leaderboard"][0]
    assert entry["time_taken_s"] == 450.0 + 600.0  # never above the published 20 min


def test_equal_score_and_time_share_a_rank(tmp_path):
    store = _store(tmp_path)
    # alpha and beta are equal on both; beta's run finished first, so it is listed first.
    _result(store, "alpha", 1, _subset({"a": 1.0}), _subset({"b": 0.0}), hours=2, minutes=(5, 5))
    _result(store, "beta", 1, _subset({"a": 1.0}), _subset({"b": 0.0}), hours=1, minutes=(5, 5))
    _result(store, "gamma", 1, _subset({"a": 0.0}), _subset({"b": 0.0}))
    board = build_snapshot(_config(tmp_path), store)["leaderboard"]
    assert [(e["team_id"], e["rank"]) for e in board] == [("beta", 1), ("alpha", 1), ("gamma", 3)]


def test_runs_that_use_the_whole_budget_tie_on_time(tmp_path):
    store = _store(tmp_path)
    _result(store, "alpha", 1, _subset({"a": 1.0}), _subset({"b": 0.0}), minutes=(40, 50))
    _result(store, "beta", 1, _subset({"a": 1.0}), _subset({"b": 0.0}), minutes=(90, 31))
    board = build_snapshot(_config(tmp_path), store)["leaderboard"]
    assert [e["rank"] for e in board[:2]] == [1, 1]


def test_a_better_score_beats_a_faster_run(tmp_path):
    store = _store(tmp_path)
    _result(store, "alpha", 1, _subset({"a": 1.0}), _subset({"b": 1.0}), minutes=(60, 60))
    _result(store, "beta", 1, _subset({"a": 1.0}), _subset({"b": 0.0}), minutes=(1, 1))
    board = build_snapshot(_config(tmp_path), store)["leaderboard"]
    assert board[0]["team_id"] == "alpha"


def test_a_finished_submission_carries_its_times_and_a_single_status(tmp_path):
    store, activity = _activity(tmp_path)
    _result(store, "alpha", 1, _subset({"a": 1.0}), _subset({"b": 1.0}), hours=5, submitted=30)
    [row] = activity()["alpha"]
    assert row["status"] == "done"
    assert row["submitted_at"] == "2026-10-01T00:30:00Z"
    assert row["finished_at"] == "2026-10-01T05:00:00Z"
    assert row["wall_s"] is None  # this result has no started_at


def test_a_finished_submission_carries_its_wall_time(tmp_path):
    store, activity = _activity(tmp_path)
    result = _result(store, "alpha", 1, _subset({"a": 1.0}), _subset({"b": 1.0}), hours=1)
    store.publish_result(result.model_copy(update={"started_at": T0}))
    assert activity()["alpha"][0]["wall_s"] == 3600.0


def test_the_most_recently_submitted_is_on_top_whatever_its_state(tmp_path):
    store, activity = _activity(tmp_path)
    queue = FileQueue(store)
    queue.enqueue(_job("alpha", 1, enqueued=10))
    assert queue.claim("w1")  # claimed *later* than the deferred push below was made
    (store.deferred / "alpha.json").write_text(
        json.dumps({"job_id": "alpha__2", "submission_id": "newer", "since": "2026-10-01T00:20:00+00:00"})
    )
    _result(store, "alpha", 0, _subset({"a": 1.0}), _subset({"b": 1.0}), hours=0)
    assert [a["status"] for a in activity()["alpha"]] == ["deferred", "running", "done"]


def test_a_team_carries_every_submission_and_the_page_is_told_how_many_to_show(tmp_path):
    store, activity = _activity(tmp_path)
    for n in range(1, 6):
        _result(store, "alpha", n, _subset({"a": 1.0}), _subset({"b": 1.0}), hours=n)
    rows = activity()["alpha"]
    assert [a["submission_id"] for a in rows] == ["sub-5", "sub-4", "sub-3", "sub-2", "sub-1"]
    snapshot = build_snapshot(_config(tmp_path, activity_per_team=2), store)
    assert snapshot["queue"]["shown"] == 2


def test_a_run_that_ran_out_of_time_shows_over_time_not_done(tmp_path):
    store, activity = _activity(tmp_path)
    cut = _subset({"d": 0.0}, status="completed")
    cut.status = "timeout"
    _result(store, "alpha", 1, _subset({"a": 1.0}), cut, status=RunStatus.TIMEOUT)
    assert [a["status"] for a in activity()["alpha"]] == ["over-time"]


def test_subsets_are_shown_under_their_labels(tmp_path):
    store = _store(tmp_path)
    scoring = build_snapshot(_config(tmp_path, subset_labels={"rs": "robosuite-smoke"}), store)["scoring"]
    assert scoring["subsets"] == ["robosuite-smoke", "lib"]


def test_only_the_total_time_limit_is_published(tmp_path):
    store = _store(tmp_path)
    config = _config(tmp_path)
    _pack(tmp_path / "p", 600.0)
    scoring = build_snapshot(config, store)["scoring"]
    assert scoring["time_limit_s"] == 600.0 + 1800.0
    assert "job_timeout_s" not in json.dumps(build_snapshot(config, store))


def test_history_rows_carry_their_score_when_there_is_one(tmp_path):
    store, activity = _activity(tmp_path)
    _result(store, "alpha", 1, _subset({"a": 1.0}), _subset({"b": 0.0}), hours=1)
    _result(store, "alpha", 2, _subset({"a": 1.0}), _subset({"b": 0.0}, status="failed"), hours=2)
    store.publish_result(RunResult(run_id="alpha__3", job_id="alpha__3", team_id="alpha", submission_id="bad",
                                   status=RunStatus.REJECTED, finished_at=T0 + timedelta(hours=3)))
    assert [(a["submission_id"], a["score"]) for a in activity()["alpha"]] == [
        ("bad", None), ("sub-2", None), ("sub-1", 0.5)
    ]


def test_a_submission_that_never_reached_the_gateway_is_failed_and_unranked(tmp_path):
    """Rulebook §4.3: such a job is not evaluated. It did not run out of time either:
    its clock never started."""
    store, activity = _activity(tmp_path)
    never = SubsetResult(status=RunStatus.FAILED, error_code=ErrorCode.SUBMISSION_STARTUP_TIMEOUT)
    _result(store, "alpha", 1, _subset({"a": 1.0}), never, status=RunStatus.FAILED)
    entry = next(e for e in build_snapshot(_config(tmp_path), store)["leaderboard"] if e["team_id"] == "alpha")
    assert entry["rank"] is None and entry["score"] is None and entry["over_time"] is False
    assert [a["status"] for a in activity()["alpha"]] == ["failed"]
