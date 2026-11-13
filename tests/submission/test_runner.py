"""Unit tests for ``capx.submission.runner``, sim-free.

Importability note: ``capx.submission.runner`` imports ``capx.envs.trial``,
which (via ``capx.envs.tasks.base``) forces ``capx/integrations/__init__.py``
to run, which unconditionally imports ``capx.integrations.franka.control`` ->
``capx.integrations.motion.pyroki_snippets`` -> ``jax`` — a dependency this
sim-free test venv intentionally excludes (see
``tests/submission/test_code_env.py``'s module docstring for the identical
issue). ``_stub_capx_integrations_package`` below applies the
exact same "pre-seed the parent package in ``sys.modules``" workaround, so
``capx.envs.tasks.base``'s ``from capx.integrations.base_api import ...`` (a
submodule import with no jax dependency of its own) resolves without ever
running the real package ``__init__.py``. When the real package already
imports cleanly (e.g. jax installed), the stub is skipped and the genuine
package is used unmodified.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _stub_capx_integrations_package() -> None:
    if "capx.integrations" in sys.modules:
        return
    try:
        import capx.integrations  # noqa: F401

        return
    except ImportError:
        pass

    import capx

    pkg = types.ModuleType("capx.integrations")
    pkg.__path__ = [str(Path(capx.__file__).parent / "integrations")]
    sys.modules["capx.integrations"] = pkg


_stub_capx_integrations_package()

import capx.submission.runner as runner_module  # noqa: E402
from capx.integrations.base_api import ApiBase, register_api  # noqa: E402
from capx.protocol import Embodiment, ErrorCode, GatewayError, ManifestEntry  # noqa: E402
from capx.protocol.taskpack import Simulator  # noqa: E402
from capx.submission.interfaces import R1ProKnobs, SubmissionConfig, TaskRunConfig  # noqa: E402
from capx.utils.launch_utils import TrialSummary  # noqa: E402
from tests.submission.fakes import FakeEnvClient  # noqa: E402

# ---------------------------------------------------------------------------
# A one-function stub API, registered under a test-unique name so
# `build_code_env` (called for real by `run_submission`) can resolve
# `TaskRunConfig.apis` without depending on the real Franka/R1Pro APIs.
# ---------------------------------------------------------------------------


class _StubApi(ApiBase):
    def functions(self) -> dict[str, Any]:
        return {}


_STUB_API_NAME = "StubApi__test_runner"
register_api(_STUB_API_NAME, lambda env: _StubApi(env))


def _write_task_config(tmp_path: Path, name: str, **overrides: Any) -> str:
    """Write a minimal TaskRunConfig YAML and return its path."""
    import yaml

    data: dict[str, Any] = {
        "apis": [_STUB_API_NAME],
        "prompt": "{task_instruction}",
        "api_servers": [],  # no real perception/IK servers spawned in tests
        "record_video": True,
    }
    data.update(overrides)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return str(path)


def _submission_config(**overrides: Any) -> SubmissionConfig:
    base: dict[str, Any] = {
        "model": "test-model",
        "server_url": "http+unix://model.sock/v1/chat/completions",
        "trials_per_task": 1,
        "output_dir": "outputs/test",
        "task_map": [],
    }
    base.update(overrides)
    return SubmissionConfig(**base)


# ---------------------------------------------------------------------------
# (a) task-map resolution: exact match + the fallback branches
# ---------------------------------------------------------------------------


def _entry(task_id: str, embodiment: str, camera_names: list[str]) -> ManifestEntry:
    return ManifestEntry(
        task_id=task_id,
        embodiment=Embodiment(embodiment),
        instruction="do the thing",
        camera_names=camera_names,
        action_spec_ref="PRELIM_RULEBOOK.md#action-franka",
        n_trials=2,
        trial_steplimit=100,
    )


def test_resolve_exact_task_id_match_wins() -> None:
    cfg = _submission_config(task_map=[{"task_id": "stack_cubes", "config": "custom/path.yaml"}])
    entry = _entry("stack_cubes", "franka", ["robot0_robotview"])
    assert runner_module.resolve_task_config_path(entry, cfg) == "custom/path.yaml"


def test_resolve_fallback_r1pro() -> None:
    cfg = _submission_config()
    entry = _entry("some_new_r1pro_task", "r1pro", ["ego", "left_wrist", "right_wrist"])
    assert runner_module.resolve_task_config_path(entry, cfg) == runner_module._DEFAULT_R1PRO_CONFIG


def test_resolve_fallback_libero_via_agentview_camera() -> None:
    cfg = _submission_config()
    entry = _entry("some_new_libero_task", "franka", ["agentview", "robot0_eye_in_hand"])
    assert runner_module.resolve_task_config_path(entry, cfg) == runner_module._DEFAULT_LIBERO_CONFIG


def test_resolve_fallback_robosuite_default() -> None:
    cfg = _submission_config()
    entry = _entry("some_new_robosuite_task", "franka", ["robot0_robotview"])
    assert runner_module.resolve_task_config_path(entry, cfg) == runner_module._DEFAULT_ROBOSUITE_CONFIG


@pytest.mark.parametrize(
    ("embodiment", "cameras", "simulator"),
    [
        ("r1pro", ["ego", "left_wrist", "right_wrist"], Simulator.BEHAVIOR),
        ("franka_bimanual", ["agentview"], Simulator.ROBOSUITE),
        ("franka", ["agentview", "robot0_eye_in_hand"], Simulator.LIBERO),
        ("franka", ["robot0_robotview"], Simulator.ROBOSUITE),
    ],
)
def test_the_simulator_is_told_from_embodiment_and_cameras(embodiment, cameras, simulator) -> None:
    assert runner_module.simulator_of(_entry("t", embodiment, cameras)) is simulator


# ---------------------------------------------------------------------------
# (a2) _build_shim: TaskRunConfig's shim settings reach the shim
# ---------------------------------------------------------------------------


def test_build_shim_forwards_step_chunk_for_franka() -> None:
    task_cfg = TaskRunConfig(apis=[_STUB_API_NAME], prompt="x", step_chunk=7)
    entry = _entry("stack_cubes", "franka", ["robot0_robotview"])
    shim = runner_module._build_shim(object(), entry, task_cfg)
    assert shim._step_chunk == 7


def test_build_shim_forwards_step_chunk_and_knobs_for_r1pro() -> None:
    task_cfg = TaskRunConfig(apis=[_STUB_API_NAME], prompt="x", step_chunk=4, r1pro=R1ProKnobs(settle_cycles=3))
    entry = _entry("picking_up_trash", "r1pro", ["ego", "left_wrist", "right_wrist"])
    shim = runner_module._build_shim(object(), entry, task_cfg)
    assert shim._step_chunk == 4
    assert shim._knobs.settle_cycles == 3


def test_a_mounted_model_is_used_in_place_of_its_hub_name(tmp_path: Path) -> None:
    """The two-arm config named the agent0 image's /models paths, which a native run
    does not have; it names the hub models now, with the mounts as local copies."""
    mounted = tmp_path / "owlv2"
    mounted.mkdir()
    task_cfg = TaskRunConfig(
        apis=[_STUB_API_NAME],
        prompt="x",
        api_servers=[
            {"_target_": "owlvit.main", "model_name": "google/owlv2", "port": 8117},
            {"_target_": "sam2.main", "model_name": "facebook/sam2", "port": 8113},
            {"_target_": "graspnet.main", "port": 8115},
        ],
        local_models={"google/owlv2": str(mounted), "facebook/sam2": str(tmp_path / "absent")},
    )
    specs = runner_module._server_specs(task_cfg)
    assert [spec.get("model_name") for spec in specs] == [str(mounted), "facebook/sam2", None]


# ---------------------------------------------------------------------------
# (b) _build_config_dict: every key capx.envs.trial._run_single_trial reads
# ---------------------------------------------------------------------------

_REQUIRED_CONFIG_KEYS = {
    "output_dir",
    "record_video",
    "use_visual_feedback",
    "use_img_differencing",
    "use_parallel_ensemble",
    "use_oracle_code",
}
_OPTIONAL_CONFIG_KEYS = {
    "use_video_differencing",
    "use_wrist_camera",
    "use_multimodel",
    "save_multiturn_prompts",
    "task_name",
}


def test_config_dict_has_every_key_trial_py_reads() -> None:
    cfg = _submission_config(use_img_differencing=True, use_video_differencing=True, use_wrist_camera=True)
    task_cfg = TaskRunConfig(apis=[_STUB_API_NAME], prompt="x", record_video=True)
    config_dict = runner_module._build_config_dict(cfg, task_cfg, "some_task", "/tmp/out/some_task")

    assert set(config_dict) == _REQUIRED_CONFIG_KEYS | _OPTIONAL_CONFIG_KEYS
    assert config_dict["output_dir"] == "/tmp/out/some_task"
    assert config_dict["record_video"] is True
    assert config_dict["use_img_differencing"] is True
    assert config_dict["use_video_differencing"] is True
    assert config_dict["use_wrist_camera"] is True
    assert config_dict["use_oracle_code"] is False
    assert config_dict["use_parallel_ensemble"] is False
    assert config_dict["task_name"] == "some_task"


# ---------------------------------------------------------------------------
# (c) _attempt_trial: what became of one trial
# ---------------------------------------------------------------------------


def _summary(trial: int) -> TrialSummary:
    return TrialSummary(trial=trial, success=True, reward=0.0, terminated=False, truncated=False, sandbox_rc=0, log="")


class _ScriptedRunner:
    """Stand-in for ``capx.envs.trial._run_single_trial``: pops one scripted
    action (a value to return or an exception to raise) per call."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.call_count = 0

    def __call__(self, env, trial, launch_args, config_dict, multi_turn_prompt):
        self.call_count += 1
        action = self._script.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action


class _ShimState:
    def __init__(self, *, voided: bool = False, disconnected: bool = False) -> None:
        self.voided = voided
        self.disconnected = disconnected


def _run_attempt(
    monkeypatch: pytest.MonkeyPatch, script: list[Any], shim: _ShimState | None = None
) -> tuple[Any, Any]:
    monkeypatch.setattr(runner_module, "_run_single_trial", _ScriptedRunner(script))
    return runner_module._attempt_trial(object(), shim or _ShimState(), 0, object(), {}, None)


def test_attempt_trial_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    outcome, summary = _run_attempt(monkeypatch, [_summary(0)])
    assert outcome is runner_module._Outcome.DONE
    assert summary is not None and summary.trial == 0


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (ErrorCode.TRIALS_EXHAUSTED, "TASK_EXHAUSTED"),
        (ErrorCode.JOB_TIMEOUT, "RUN_OVER"),
        (ErrorCode.JOB_FINISHED, "RUN_OVER"),
        (ErrorCode.ENV_CRASH, "VOIDED"),
    ],
)
def test_attempt_trial_maps_reset_verdicts(
    monkeypatch: pytest.MonkeyPatch, code: ErrorCode, expected: str
) -> None:
    outcome, summary = _run_attempt(monkeypatch, [GatewayError(code, "verdict")])
    assert outcome is runner_module._Outcome[expected]
    assert summary is None


def test_attempt_trial_other_gateway_error_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(GatewayError):
        _run_attempt(monkeypatch, [GatewayError(ErrorCode.CLIENT_TIMEOUT, "slow")])


def test_attempt_trial_a_dead_connection_at_reset_is_a_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    outcome, _ = _run_attempt(monkeypatch, [ConnectionError("socket closed mid-frame")])
    assert outcome is runner_module._Outcome.DISCONNECTED


def test_attempt_trial_a_disk_error_is_not_a_lost_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(OSError):
        _run_attempt(monkeypatch, [OSError(28, "No space left on device")])


def test_attempt_trial_reads_a_mid_episode_crash_off_the_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent's code sees a crash mid-episode as a RuntimeError it may catch, and
    the trial then returns normally; the shim is what remembers it was voided."""
    outcome, summary = _run_attempt(monkeypatch, [_summary(0)], _ShimState(voided=True))
    assert outcome is runner_module._Outcome.VOIDED
    assert summary is not None


def test_attempt_trial_an_error_after_a_crash_is_still_a_void(monkeypatch: pytest.MonkeyPatch) -> None:
    outcome, _ = _run_attempt(monkeypatch, [RuntimeError("gateway error [client_timeout]")], _ShimState(voided=True))
    assert outcome is runner_module._Outcome.VOIDED


def test_attempt_trial_a_lost_connection_mid_episode_wins_over_a_void(monkeypatch: pytest.MonkeyPatch) -> None:
    outcome, _ = _run_attempt(monkeypatch, [_summary(0)], _ShimState(voided=True, disconnected=True))
    assert outcome is runner_module._Outcome.DISCONNECTED


def test_attempt_trial_other_errors_reraise(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(TypeError):
        _run_attempt(monkeypatch, [TypeError("bad reply")])


# ---------------------------------------------------------------------------
# (d) run_submission: full manifest loop, server dedup, leaderboard writing
# ---------------------------------------------------------------------------


def _patch_no_op_servers(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace _start_api_servers/_stop_api_servers with call-recording stubs."""
    started: list[str] = []

    def fake_start(api_servers, wait_timeout: float = 120.0):
        started.append("started")
        return []

    monkeypatch.setattr(runner_module, "_start_api_servers", fake_start)
    monkeypatch.setattr(runner_module, "_stop_api_servers", lambda procs: None)
    return started


def test_run_submission_remote_mode_runs_every_task_and_trial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = FakeEnvClient()
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: [client])
    started = _patch_no_op_servers(monkeypatch)

    config_path = _write_task_config(tmp_path, "robosuite.yaml")
    seen_trials: list[tuple[str, int]] = []

    def fake_run(env, trial, launch_args, config_dict, multi_turn_prompt):
        seen_trials.append((config_dict["task_name"], trial))
        return _summary(trial)

    monkeypatch.setattr(runner_module, "_run_single_trial", fake_run)

    cfg = _submission_config(
        trials_per_task=2,
        output_dir=str(tmp_path / "out"),
        task_map=[
            {"task_id": tid, "config": config_path}
            for tid in (
                "stack_cubes",
                "put_alphabet_soup_and_tomato_sauce_in_basket",
                "two_arm_lift",
                "picking_up_trash",
            )
        ],
    )
    runner_module.run_submission(cfg)

    assert seen_trials == [
        ("stack_cubes", 0), ("stack_cubes", 1),
        ("put_alphabet_soup_and_tomato_sauce_in_basket", 0), ("put_alphabet_soup_and_tomato_sauce_in_basket", 1),
        ("two_arm_lift", 0), ("two_arm_lift", 1),
        ("picking_up_trash", 0), ("picking_up_trash", 1),
    ]
    # Every task maps to the SAME config path -> servers started once.
    assert started == ["started"]
    assert not (tmp_path / "out" / "leaderboard.json").exists()
    assert client.finished


def test_run_submission_starts_servers_once_per_distinct_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = FakeEnvClient()
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: [client])
    started = _patch_no_op_servers(monkeypatch)
    monkeypatch.setattr(runner_module, "_run_single_trial", lambda *a, **k: _summary(0))

    config_a = _write_task_config(tmp_path, "a.yaml")
    config_b = _write_task_config(tmp_path, "b.yaml")
    cfg = _submission_config(
        trials_per_task=1,
        output_dir=str(tmp_path / "out"),
        task_map=[
            {"task_id": "stack_cubes", "config": config_a},
            {"task_id": "put_alphabet_soup_and_tomato_sauce_in_basket", "config": config_b},
            {"task_id": "two_arm_lift", "config": config_b},
            {"task_id": "picking_up_trash", "config": config_a},
        ],
    )
    runner_module.run_submission(cfg)
    assert started == ["started", "started"]  # config_a, config_b -- not a third time for picking_up_trash


def test_run_submission_spreads_the_worklist_over_lanes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Concurrency is the only knob that shortens a round: a trial is ~99.8% this
    side's own generation and perception, so lanes wait on the model server rather
    than on each other."""
    clients = [FakeEnvClient(), FakeEnvClient()]
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: clients)
    _patch_no_op_servers(monkeypatch)

    seen: list[tuple[str, int]] = []
    threads: set[str] = set()
    lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=10.0)

    def fake_run(env, trial, launch_args, config_dict, multi_turn_prompt):
        barrier.wait()  # deadlocks unless two trials really are in flight at once
        with lock:
            seen.append((config_dict["task_name"], trial))
            threads.add(threading.current_thread().name)
        return _summary(trial)

    monkeypatch.setattr(runner_module, "_run_single_trial", fake_run)

    config_path = _write_task_config(tmp_path, "robosuite.yaml")
    tasks = (
        "stack_cubes",
        "put_alphabet_soup_and_tomato_sauce_in_basket",
        "two_arm_lift",
        "picking_up_trash",
    )
    cfg = _submission_config(
        trials_per_task=2,
        concurrency=2,
        output_dir=str(tmp_path / "out"),
        task_map=[{"task_id": tid, "config": config_path} for tid in tasks],
    )
    runner_module.run_submission(cfg)

    assert sorted(seen) == sorted((tid, trial) for tid in tasks for trial in range(2))
    assert len(threads) == 2


def test_run_submission_keeps_behavior_within_its_lane_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With BEHAVIOR capped at one lane, the second lane runs the other simulators'
    trials meanwhile instead of waiting for a BEHAVIOR backend."""
    clients = [FakeEnvClient(), FakeEnvClient()]
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: clients)
    _patch_no_op_servers(monkeypatch)

    lock = threading.Lock()
    running = {"behavior": 0, "most": 0}
    order: list[str] = []

    def fake_run(env, trial, launch_args, config_dict, multi_turn_prompt):
        behavior = config_dict["task_name"] == "picking_up_trash"
        with lock:
            order.append(config_dict["task_name"])
            running["behavior"] += behavior
            running["most"] = max(running["most"], running["behavior"])
        time.sleep(0.05)
        with lock:
            running["behavior"] -= behavior
        return _summary(trial)

    monkeypatch.setattr(runner_module, "_run_single_trial", fake_run)
    config_path = _write_task_config(tmp_path, "robosuite.yaml")
    tasks = ("stack_cubes", "put_alphabet_soup_and_tomato_sauce_in_basket", "two_arm_lift", "picking_up_trash")
    cfg = _submission_config(
        trials_per_task=3,
        concurrency=2,
        simulator_lanes={"behavior": 1},
        output_dir=str(tmp_path / "out"),
        task_map=[{"task_id": tid, "config": config_path} for tid in tasks],
    )
    runner_module.run_submission(cfg)

    assert sorted(order) == sorted(tid for tid in tasks for _ in range(3))
    assert running["most"] == 1
    assert "picking_up_trash" in order[:2]  # the capped simulator's trials go out first


def test_a_busy_lane_does_not_strand_the_work_behind_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Lanes pull from one shared worklist instead of owning a fixed slice.

    Striding decided the split before anything ran, so whichever lane drew the
    long trials kept the round alive long after the others had gone idle. Here a
    lane that is still busy simply stops taking work and the free one drains the
    rest -- which a per-lane slice cannot do, however idle it is.
    """
    clients = [FakeEnvClient(), FakeEnvClient()]
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: clients)
    _patch_no_op_servers(monkeypatch)

    tasks = ("stack_cubes", "put_alphabet_soup_and_tomato_sauce_in_basket")
    total = len(tasks) * 2  # two trials each: a stride would deal 2 to each lane
    rest_drained = threading.Event()
    lock = threading.Lock()
    by_thread: dict[str, int] = {}

    def fake_run(env, trial, launch_args, config_dict, multi_turn_prompt):
        with lock:
            name = threading.current_thread().name
            by_thread[name] = by_thread.get(name, 0) + 1
            taken = sum(by_thread.values())
        if taken == 1:
            assert rest_drained.wait(timeout=10.0), "the free lane never drained the worklist"
        elif taken == total:
            rest_drained.set()
        return _summary(trial)

    monkeypatch.setattr(runner_module, "_run_single_trial", fake_run)

    config_path = _write_task_config(tmp_path, "robosuite.yaml")
    cfg = _submission_config(
        trials_per_task=2,
        concurrency=2,
        output_dir=str(tmp_path / "out"),
        task_filter=list(tasks),
        task_map=[{"task_id": tid, "config": config_path} for tid in tasks],
    )
    runner_module.run_submission(cfg)

    assert sorted(by_thread.values()) == [1, total - 1]


def test_a_lane_that_runs_out_of_work_disconnects_while_others_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A lane holds its simulator backend until it disconnects, and backends are
    capped per simulator; a finished lane that stayed connected could leave a busy
    one waiting on a backend until the job was killed."""
    clients = [FakeEnvClient(), FakeEnvClient()]
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: clients)
    _patch_no_op_servers(monkeypatch)
    lock = threading.Lock()
    started: list[int] = []

    def fake_run(env, trial, launch_args, config_dict, multi_turn_prompt):
        with lock:
            started.append(trial)
            busy = len(started) == 1
        if busy:
            # Still running when the other lane has nothing left: it should have
            # let go of its connection by the time this trial ends.
            deadline = time.monotonic() + 10.0
            while not any(getattr(c, "closed", False) for c in clients):
                assert time.monotonic() < deadline, "the idle lane stayed connected"
                time.sleep(0.01)
        return _summary(trial)

    monkeypatch.setattr(runner_module, "_run_single_trial", fake_run)
    config_path = _write_task_config(tmp_path, "robosuite.yaml")
    cfg = _submission_config(
        trials_per_task=2,
        concurrency=2,
        output_dir=str(tmp_path / "out"),
        task_filter=["stack_cubes"],
        task_map=[{"task_id": "stack_cubes", "config": config_path}],
    )
    runner_module.run_submission(cfg)
    assert all(getattr(client, "closed", False) for client in clients)


def test_a_job_timeout_in_one_lane_stops_the_others(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """job_timeout is a verdict about the run, not about the lane that heard it."""
    clients = [FakeEnvClient(), FakeEnvClient()]
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: clients)
    _patch_no_op_servers(monkeypatch)

    attempts: list[int] = []
    lock = threading.Lock()

    def fake_run(env, trial, launch_args, config_dict, multi_turn_prompt):
        with lock:
            attempts.append(trial)
        raise GatewayError(ErrorCode.JOB_TIMEOUT, "budget hit")

    monkeypatch.setattr(runner_module, "_run_single_trial", fake_run)

    config_path = _write_task_config(tmp_path, "robosuite.yaml")
    cfg = _submission_config(
        trials_per_task=20,
        concurrency=2,
        output_dir=str(tmp_path / "out"),
        task_filter=["stack_cubes"],
        task_map=[{"task_id": "stack_cubes", "config": config_path}],
    )
    runner_module.run_submission(cfg)

    # 40 trials were queued; both lanes stop as soon as either hears the verdict.
    assert len(attempts) < 10


def test_trials_exhausted_in_one_lane_skips_that_task_in_the_others(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clients = [FakeEnvClient(), FakeEnvClient()]
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: clients)
    _patch_no_op_servers(monkeypatch)

    calls: list[str] = []
    lock = threading.Lock()

    def fake_run(env, trial, launch_args, config_dict, multi_turn_prompt):
        task_id = config_dict["task_name"]
        with lock:
            calls.append(task_id)
        if task_id == "stack_cubes":
            raise GatewayError(ErrorCode.TRIALS_EXHAUSTED, "no trials left")
        return _summary(trial)

    monkeypatch.setattr(runner_module, "_run_single_trial", fake_run)

    config_path = _write_task_config(tmp_path, "cfg.yaml")
    cfg = _submission_config(
        trials_per_task=10,
        concurrency=2,
        output_dir=str(tmp_path / "out"),
        task_filter=["stack_cubes", "put_alphabet_soup_and_tomato_sauce_in_basket"],
        task_map=[
            {"task_id": tid, "config": config_path}
            for tid in ("stack_cubes", "put_alphabet_soup_and_tomato_sauce_in_basket")
        ],
    )
    runner_module.run_submission(cfg)

    assert calls.count("stack_cubes") <= 2  # one per lane at worst, then skipped
    assert calls.count("put_alphabet_soup_and_tomato_sauce_in_basket") == 10


def test_run_submission_task_filter_restricts_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = FakeEnvClient()
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: [client])
    _patch_no_op_servers(monkeypatch)
    seen_tasks: list[str] = []
    monkeypatch.setattr(
        runner_module, "_run_single_trial",
        lambda env, trial, launch_args, config_dict, mtp: (seen_tasks.append(config_dict["task_name"]), _summary(trial))[1],
    )

    config_path = _write_task_config(tmp_path, "robosuite.yaml")
    cfg = _submission_config(
        trials_per_task=1,
        output_dir=str(tmp_path / "out"),
        task_filter=["stack_cubes"],
        task_map=[{"task_id": "stack_cubes", "config": config_path}],
    )
    runner_module.run_submission(cfg)
    assert seen_tasks == ["stack_cubes"]


def test_run_submission_trials_exhausted_moves_to_next_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = FakeEnvClient()
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: [client])
    _patch_no_op_servers(monkeypatch)

    calls: list[str] = []

    def fake_run(env, trial, launch_args, config_dict, multi_turn_prompt):
        task_id = config_dict["task_name"]
        calls.append(task_id)
        if task_id == "stack_cubes":
            raise GatewayError(ErrorCode.TRIALS_EXHAUSTED, "no trials left")
        return _summary(trial)

    monkeypatch.setattr(runner_module, "_run_single_trial", fake_run)

    config_path = _write_task_config(tmp_path, "cfg.yaml")
    cfg = _submission_config(
        trials_per_task=3,
        output_dir=str(tmp_path / "out"),
        task_filter=["stack_cubes", "put_alphabet_soup_and_tomato_sauce_in_basket"],
        task_map=[
            {"task_id": "stack_cubes", "config": config_path},
            {"task_id": "put_alphabet_soup_and_tomato_sauce_in_basket", "config": config_path},
        ],
    )
    runner_module.run_submission(cfg)
    # stack_cubes: 1 attempt (exhausted immediately, no retry loop for this code);
    # the other task still gets its full 3 trials.
    assert calls == [
        "stack_cubes",
        "put_alphabet_soup_and_tomato_sauce_in_basket",
        "put_alphabet_soup_and_tomato_sauce_in_basket",
        "put_alphabet_soup_and_tomato_sauce_in_basket",
    ]


def test_run_submission_job_timeout_stops_whole_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = FakeEnvClient()
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: [client])
    _patch_no_op_servers(monkeypatch)

    calls: list[str] = []

    def fake_run(env, trial, launch_args, config_dict, multi_turn_prompt):
        calls.append(config_dict["task_name"])
        raise GatewayError(ErrorCode.JOB_TIMEOUT, "budget exhausted")

    monkeypatch.setattr(runner_module, "_run_single_trial", fake_run)

    config_path = _write_task_config(tmp_path, "cfg.yaml")
    cfg = _submission_config(
        trials_per_task=3,
        output_dir=str(tmp_path / "out"),
        task_map=[
            {"task_id": "stack_cubes", "config": config_path},
            {"task_id": "put_alphabet_soup_and_tomato_sauce_in_basket", "config": config_path},
        ],
    )
    runner_module.run_submission(cfg)
    # First task's first trial hits job_timeout -> the whole run stops; the
    # second task in the manifest is never attempted.
    assert calls == ["stack_cubes"]


def test_one_bad_trial_does_not_end_the_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A model reply the code extractor cannot parse used to raise out of the lane
    and take the whole submission with it — over a 137-task pack that is 136 tasks
    lost to one bad reply."""
    client = FakeEnvClient()
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: [client])
    _patch_no_op_servers(monkeypatch)

    seen: list[int] = []

    def fake_run(env, trial, launch_args, config_dict, multi_turn_prompt):
        seen.append(trial)
        if trial == 0:
            raise TypeError("object of type 'NoneType' has no len()")
        return _summary(trial)

    monkeypatch.setattr(runner_module, "_run_single_trial", fake_run)

    config_path = _write_task_config(tmp_path, "robosuite.yaml")
    cfg = _submission_config(
        trials_per_task=3,
        output_dir=str(tmp_path / "out"),
        task_filter=["stack_cubes"],
        task_map=[{"task_id": "stack_cubes", "config": config_path}],
    )
    runner_module.run_submission(cfg)

    assert seen == [0, 1, 2]  # the crash cost its own trial and nothing else


class _RefusedClient(FakeEnvClient):
    """A connection past the gateway's lane cap: the first call is answered with the
    reason, and the gateway then hangs up."""

    def __init__(self) -> None:
        super().__init__()
        self._answered = False

    def _refuse(self) -> None:
        if self._answered:
            raise ConnectionError("socket closed mid-frame")
        self._answered = True
        raise RuntimeError("gateway allows 1 concurrent lane(s); all 1 are in use")

    def manifest(self) -> list[dict[str, Any]]:
        self._refuse()
        return []

    def reset(self, task_id: str) -> Any:
        self._refuse()


def _one_task_config(tmp_path: Path, trials: int, **overrides: Any) -> SubmissionConfig:
    config_path = _write_task_config(tmp_path, "robosuite.yaml")
    return _submission_config(
        trials_per_task=trials,
        output_dir=str(tmp_path / "out"),
        task_filter=["stack_cubes"],
        task_map=[{"task_id": "stack_cubes", "config": config_path}],
        **overrides,
    )


def test_lanes_the_gateway_refuses_do_not_eat_the_worklist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A refused lane used to fail every item it pulled in microseconds, draining the
    shared worklist while the healthy lane was still on its first trial; finish()
    then fixed the score with most trials never run."""
    clients = [FakeEnvClient(), _RefusedClient(), _RefusedClient()]
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: clients)
    _patch_no_op_servers(monkeypatch)
    seen: list[int] = []
    lock = threading.Lock()

    def fake_run(env, trial, launch_args, config_dict, multi_turn_prompt):
        env.reset()
        with lock:
            seen.append(trial)
        return _summary(trial)

    monkeypatch.setattr(runner_module, "_run_single_trial", fake_run)
    runner_module.run_submission(_one_task_config(tmp_path, 8, concurrency=3))

    assert sorted(seen) == list(range(8))
    assert clients[0].finished


def test_a_lane_that_loses_its_connection_hands_its_trial_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clients = [FakeEnvClient(), FakeEnvClient()]
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: clients)
    _patch_no_op_servers(monkeypatch)
    dead_lane_tried = threading.Event()
    seen: list[int] = []
    lock = threading.Lock()

    def fake_run(env, trial, launch_args, config_dict, multi_turn_prompt):
        if env.low_level_env._client is clients[1]:
            dead_lane_tried.set()
            raise ConnectionError("socket closed mid-frame")
        # Hold the healthy lane until the other one has pulled an item.
        assert dead_lane_tried.wait(timeout=10.0)
        with lock:
            seen.append(trial)
        return _summary(trial)

    monkeypatch.setattr(runner_module, "_run_single_trial", fake_run)
    runner_module.run_submission(_one_task_config(tmp_path, 4, concurrency=2))

    assert sorted(seen) == [0, 1, 2, 3]
    assert clients[0].finished


def test_a_trial_handed_back_after_the_others_went_idle_still_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A healthy lane used to exit as soon as the list was empty; when the last busy
    lane then lost its connection, the trial it handed back had nobody to run it."""
    clients = [FakeEnvClient(), FakeEnvClient()]
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: clients)
    _patch_no_op_servers(monkeypatch)
    dead_lane_started = threading.Event()
    ran_on_healthy_lane: list[int] = []

    def fake_run(env, trial, launch_args, config_dict, multi_turn_prompt):
        if env.low_level_env._client is clients[1]:
            dead_lane_started.set()
            # The healthy lane has run out of work and let go of its backend.
            deadline = time.monotonic() + 10.0
            while not getattr(clients[0], "closed", False):
                assert time.monotonic() < deadline, "the healthy lane never went idle"
                time.sleep(0.01)
            raise ConnectionError("socket closed mid-frame")
        assert dead_lane_started.wait(timeout=10.0)
        ran_on_healthy_lane.append(trial)
        return _summary(trial)

    monkeypatch.setattr(runner_module, "_run_single_trial", fake_run)
    runner_module.run_submission(_one_task_config(tmp_path, 2, concurrency=2))

    assert sorted(ran_on_healthy_lane) == [0, 1]
    assert clients[0].finished


def test_the_job_is_not_finished_when_every_lane_is_lost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeEnvClient()
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: [client])
    _patch_no_op_servers(monkeypatch)

    def fake_run(env, trial, launch_args, config_dict, multi_turn_prompt):
        raise ConnectionError("socket closed mid-frame")

    monkeypatch.setattr(runner_module, "_run_single_trial", fake_run)
    with pytest.raises(RuntimeError, match="3 trial"):
        runner_module.run_submission(_one_task_config(tmp_path, 3))
    assert not client.finished


def _crash_mid_episode(client: FakeEnvClient, crashes: dict[int, int], limit: int):
    """A ``_run_single_trial`` whose motion hits an env_crash that the agent's code
    catches, for the first ``limit`` runs of each trial."""

    def fake_run(env, trial, launch_args, config_dict, multi_turn_prompt):
        env.reset()
        crashes[trial] = crashes.get(trial, 0) + 1
        if crashes[trial] <= limit:
            client.fail_at(client._call_count, GatewayError(ErrorCode.ENV_CRASH, "sim died"))
        try:
            env.low_level_env._step_once()
        except RuntimeError:
            pass
        return _summary(trial)

    return fake_run


def test_a_trial_voided_mid_episode_is_run_again(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The gateway refunds a voided trial; a worklist of exactly n_trials items used
    to move on, so every mid-episode crash cost a trial of score."""
    client = FakeEnvClient()
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: [client])
    _patch_no_op_servers(monkeypatch)
    runs: dict[int, int] = {}
    monkeypatch.setattr(runner_module, "_run_single_trial", _crash_mid_episode(client, runs, limit=1))

    runner_module.run_submission(_one_task_config(tmp_path, 2))

    assert runs == {0: 2, 1: 2}
    assert client.finished


def test_a_trial_that_keeps_crashing_is_given_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = FakeEnvClient()
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: [client])
    _patch_no_op_servers(monkeypatch)
    runs: dict[int, int] = {}
    monkeypatch.setattr(runner_module, "_run_single_trial", _crash_mid_episode(client, runs, limit=99))

    runner_module.run_submission(_one_task_config(tmp_path, 1))

    assert runs == {0: 1 + runner_module._MAX_RETRIES}
    assert client.finished


class _SharedBudget:
    """One gateway's trial budget for one task, seen from several lanes."""

    def __init__(self, trials: int) -> None:
        self.left = trials
        self.resets = 0
        self.lock = threading.Lock()


class _BudgetedClient(FakeEnvClient):
    def __init__(self, budget: _SharedBudget) -> None:
        super().__init__()
        self.budget = budget

    def reset(self, task_id: str) -> Any:
        with self.budget.lock:
            if self.budget.left == 0:
                raise GatewayError(ErrorCode.TRIALS_EXHAUSTED, "no trials left")
            self.budget.left -= 1
            self.budget.resets += 1
        return super().reset(task_id)

    def void(self) -> None:
        """The gateway voids the running episode and refunds its trial."""
        with self.budget.lock:
            self.budget.left += 1
        self.fail_at(self._call_count, GatewayError(ErrorCode.ENV_CRASH, "sim died"))


def test_a_trial_refunded_after_another_lane_saw_the_budget_spent_still_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One trial left, two queued: while the first runs, the other lane is told the
    budget is spent and marks the task exhausted. The first is then voided and
    refunded, and its retry used to be skipped as part of an exhausted task."""
    budget = _SharedBudget(trials=1)
    clients = [_BudgetedClient(budget), _BudgetedClient(budget)]
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: clients)
    _patch_no_op_servers(monkeypatch)
    voided = threading.Event()

    def fake_run(env, trial, launch_args, config_dict, multi_turn_prompt):
        env.reset()
        if not voided.is_set():
            voided.set()
            mine = env.low_level_env._client
            other = clients[1] if mine is clients[0] else clients[0]
            # Wait for the other lane to hear "exhausted" and go idle.
            deadline = time.monotonic() + 10.0
            while not getattr(other, "closed", False):
                assert time.monotonic() < deadline, "the other lane never went idle"
                time.sleep(0.01)
            mine.void()
            with contextlib.suppress(RuntimeError):
                env.low_level_env._step_once()
        return _summary(trial)

    monkeypatch.setattr(runner_module, "_run_single_trial", fake_run)
    runner_module.run_submission(_one_task_config(tmp_path, 2, concurrency=2))

    assert budget.resets == 2  # the voided trial and its retry
    assert budget.left == 0


def test_job_finished_stops_every_lane(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clients = [FakeEnvClient(), FakeEnvClient()]
    monkeypatch.setattr(runner_module, "build_env_clients", lambda cfg, lanes: clients)
    _patch_no_op_servers(monkeypatch)
    attempts: list[int] = []
    lock = threading.Lock()

    def fake_run(env, trial, launch_args, config_dict, multi_turn_prompt):
        with lock:
            attempts.append(trial)
        raise GatewayError(ErrorCode.JOB_FINISHED, "the job was finished")

    monkeypatch.setattr(runner_module, "_run_single_trial", fake_run)
    runner_module.run_submission(_one_task_config(tmp_path, 20, concurrency=2))
    assert len(attempts) <= 2


# --- the trial loop stops at the end of the episode ---------------------------


class _MovesApi(ApiBase):
    def functions(self) -> dict[str, Any]:
        return {"move": self.move, "careful_move": self.careful_move}

    def move(self) -> None:
        self._env.move_to_joints_blocking(np.full(7, 0.1), max_steps=20)

    def careful_move(self) -> None:
        try:
            self.move()
        except RuntimeError as exc:
            print(f"move failed: {exc}")


_MOVES_API_NAME = "MovesApi__test_runner"
register_api(_MOVES_API_NAME, lambda env: _MovesApi(env))


@pytest.mark.parametrize("code", ["move()", "careful_move()", "move()\nmove()"])
def test_the_trial_loop_stops_once_the_gateway_ends_the_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, code: str
) -> None:
    """It used to look only for the shim's error in stderr: a block that finished
    cleanly, or caught the error, sent the model up to ten more turns against an
    episode that could no longer move."""
    import capx.envs.trial as trial_module
    from capx.submission.code_env import build_code_env

    client = FakeEnvClient()
    client.set_flags_after(1, truncated=True)
    task_cfg = TaskRunConfig(apis=[_MOVES_API_NAME], prompt="{task_instruction}", record_video=False)
    entry = ManifestEntry.model_validate(client.manifest()[0])
    env = build_code_env(runner_module._build_shim(client, entry, task_cfg), task_cfg)

    turns: list[int] = []
    monkeypatch.setattr(trial_module, "_query_initial_code", lambda args, config, obs: (code, None, None, None))
    monkeypatch.setattr(
        trial_module,
        "_handle_multi_turn_step",
        lambda *a, **k: turns.append(1) or ("regenerate", "move()", None, None, None, None),
    )
    cfg = _submission_config(use_visual_feedback=False, output_dir=str(tmp_path))
    summary = trial_module._run_single_trial(
        env,
        0,
        runner_module._build_launch_args(cfg, "unused"),
        runner_module._build_config_dict(cfg, task_cfg, entry.task_id, str(tmp_path)),
        task_cfg.multi_turn_prompt,
    )

    assert turns == []
    assert summary.truncated
    assert summary.sandbox_rc == 0


def test_no_decision_is_asked_for_after_the_last_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The decision after the eleventh block was paid for (ten generations with the
    ensemble), and a regenerated block then never ran."""
    import capx.envs.trial as trial_module
    from capx.submission.code_env import build_code_env

    client = FakeEnvClient()
    task_cfg = TaskRunConfig(apis=[_MOVES_API_NAME], prompt="{task_instruction}", record_video=False)
    entry = ManifestEntry.model_validate(client.manifest()[0])
    env = build_code_env(runner_module._build_shim(client, entry, task_cfg), task_cfg)
    turns: list[int] = []
    monkeypatch.setattr(trial_module, "_query_initial_code", lambda args, config, obs: ("x = 1", None, None, None))
    monkeypatch.setattr(
        trial_module,
        "_handle_multi_turn_step",
        lambda *a, **k: turns.append(1) or ("regenerate", "x = 1", None, None, None, None),
    )
    cfg = _submission_config(use_visual_feedback=False, output_dir=str(tmp_path))
    summary = trial_module._run_single_trial(
        env,
        0,
        runner_module._build_launch_args(cfg, "unused"),
        runner_module._build_config_dict(cfg, task_cfg, entry.task_id, str(tmp_path)),
        task_cfg.multi_turn_prompt,
    )

    assert summary.num_code_blocks == trial_module.MULTITURN_LIMIT + 1
    assert len(turns) == trial_module.MULTITURN_LIMIT


class _FailsApi(ApiBase):
    def functions(self) -> dict[str, Any]:
        return {"fail": self.fail}

    def fail(self, message: str) -> None:
        raise ValueError(message)


_FAILS_API_NAME = "FailsApi__test_runner"
register_api(_FAILS_API_NAME, lambda env: _FailsApi(env))


@pytest.mark.parametrize("last_block_ends_the_episode", [False, True])
def test_the_summary_keeps_every_blocks_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, last_block_ends_the_episode: bool
) -> None:
    """stderr was collected only when the model was asked for a decision, so the
    error of the block that ended the trial was missing from the summary."""
    import capx.envs.trial as trial_module
    from capx.submission.code_env import build_code_env

    client = FakeEnvClient()
    if last_block_ends_the_episode:
        client.set_flags_after(1, truncated=True)
    task_cfg = TaskRunConfig(
        apis=[_MOVES_API_NAME, _FAILS_API_NAME], prompt="{task_instruction}", record_video=False
    )
    entry = ManifestEntry.model_validate(client.manifest()[0])
    env = build_code_env(runner_module._build_shim(client, entry, task_cfg), task_cfg)
    # The messages are put together at run time so the code itself, which the summary
    # also carries, does not contain them.
    second = "fail('second' + ' block failed')"
    if last_block_ends_the_episode:
        second = "careful_move()\n" + second
    monkeypatch.setattr(
        trial_module,
        "_query_initial_code",
        lambda args, config, obs: ("fail('first' + ' block failed')", None, None, None),
    )
    replies = iter([("regenerate", second, None, None, None, None), ("finish", "", None, None, None, None)])
    monkeypatch.setattr(trial_module, "_handle_multi_turn_step", lambda *a, **k: next(replies))
    monkeypatch.setattr(trial_module, "MULTITURN_LIMIT", 1)
    cfg = _submission_config(use_visual_feedback=False, output_dir=str(tmp_path))
    summary = trial_module._run_single_trial(
        env,
        0,
        runner_module._build_launch_args(cfg, "unused"),
        runner_module._build_config_dict(cfg, task_cfg, entry.task_id, str(tmp_path)),
        task_cfg.multi_turn_prompt,
    )

    assert "first block failed" in summary.log
    assert "second block failed" in summary.log


def test_the_libero_goal_goes_into_a_prompt_that_contains_braces() -> None:
    """``str.format`` on the whole prompt raised on the first literal brace in the
    API docs (``{"rgb": ...}``)."""
    import capx.envs.trial as trial_module
    from types import SimpleNamespace

    env = SimpleNamespace(low_level_env=SimpleNamespace(handle=SimpleNamespace(task_language="open the drawer")))
    text = 'Goal: {libero_environment_goal}\nobs = {"rgb": image}'
    obs = {"full_prompt": [{"role": "user", "content": [{"type": "text", "text": text}]}]}

    trial_module._patch_libero_goal(env, obs)

    assert obs["full_prompt"][-1]["content"][0]["text"] == 'Goal: open the drawer\nobs = {"rgb": image}'


# --- what the organizer tells the submission ---------------------------------


def _overrides(**env: str):
    """`capx.submission.run`'s env-over-YAML layering, with a clean environment."""
    import capx.submission.run as run_module

    cfg = _submission_config(trials_per_task=3, concurrency=1)
    args = run_module.SubmissionArgs(config_path="unused")
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("CAPX_MAX_LANES", raising=False)
        for key, value in env.items():
            mp.setenv(key, value)
        return run_module._apply_overrides(cfg, args)


def test_the_manifest_decides_the_trial_count_when_the_config_does_not() -> None:
    """A round of 100 trials per task ran 3 and published as if it had finished:
    the reference runner read only its own YAML. It now runs what the pack gives."""
    cfg = _submission_config(trials_per_task=None)
    work = runner_module._worklist([_entry("t", "franka", ["robot0_robotview"])], cfg)
    assert [w.trial for w in work] == [0, 1]


def test_an_explicit_flag_still_beats_the_config() -> None:
    import capx.submission.run as run_module

    cfg = _submission_config(trials_per_task=3)
    args = run_module.SubmissionArgs(config_path="unused", trials_per_task=7)
    assert run_module._apply_overrides(cfg, args).trials_per_task == 7


@pytest.mark.parametrize("trials", [0, -1])
def test_a_flag_is_validated_like_the_yaml(trials: int) -> None:
    """``model_copy(update=...)`` skipped validation: ``--trials-per-task -1``
    queued nothing and the runner finished the job at once."""
    import capx.submission.run as run_module
    from pydantic import ValidationError

    args = run_module.SubmissionArgs(config_path="unused", trials_per_task=trials)
    with pytest.raises(ValidationError):
        run_module._apply_overrides(_submission_config(), args)


def test_an_unset_environment_leaves_the_config_alone() -> None:
    got = _overrides()
    assert got.trials_per_task == 3
    assert got.concurrency == 1


def test_the_gateway_caps_the_lanes_the_config_asks_for() -> None:
    assert _overrides(CAPX_MAX_LANES="1").concurrency == 1


def test_a_lane_cap_above_the_config_leaves_it_alone() -> None:
    assert _overrides(CAPX_MAX_LANES="32").concurrency == 1
