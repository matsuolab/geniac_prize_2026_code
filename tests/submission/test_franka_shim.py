"""Unit tests for ``FrankaGatewayShim``, sim-free.

Exercises the shim purely against ``tests.submission.fakes.FakeEnvClient`` —
no simulator, no network. Where ``FakeFrankaModel``'s single-synthetic-camera
limitation makes an end-to-end RPC scenario awkward (pose_mat synthesis, the
depth-omitted-when-None case), the test constructs a ``FullObs``/``CameraObs``
by hand and calls ``_remap`` directly — still exercising the exact production
code path, just without a round trip through the fake wire.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from capx.protocol import (
    CameraObs,
    Embodiment,
    ErrorCode,
    FullObs,
    GatewayError,
    InvalidActionError,
    ManifestEntry,
    ProprioObs,
)
from capx.config import load_eval_config
from capx.protocol.actions import MAX_REPEAT, check_franka_bounds
from capx.submission.interfaces import EPISODE_OVER_MESSAGE
from capx.submission.shims import FrankaGatewayShim
from capx.submission.shims.franka import clamp_to_joint_limits
from tests.submission.fakes import FakeEnvClient

TASK_ID = "stack_cubes"
CAMERA_NAMES = ["robot0_robotview"]


def _make_shim(**kwargs) -> tuple[FakeEnvClient, FrankaGatewayShim]:
    client = FakeEnvClient()
    shim = FrankaGatewayShim(client, TASK_ID, camera_names=CAMERA_NAMES, **kwargs)
    return client, shim


def _wrist_entry_client() -> FakeEnvClient:
    """A single-task manifest whose only camera is the LIBERO wrist name."""
    entry = ManifestEntry(
        task_id="wrist_task",
        embodiment=Embodiment.FRANKA,
        instruction="wrist-only task",
        camera_names=["robot0_eye_in_hand"],
        action_spec_ref="PRELIM_RULEBOOK.md#action-franka",
        n_trials=1,
        trial_steplimit=100,
    )
    return FakeEnvClient(manifest=[entry.model_dump(mode="json")])


# ---------------------------------------------------------------------------
# reset() / obs remap
# ---------------------------------------------------------------------------


def test_reset_holds_a_joint_resting_past_a_stop_at_the_stop():
    """``_step_once`` sends ``_current_joints`` before any move, and the gateway's
    bounds check has no tolerance for a reading a hair past a stop."""
    upper = np.asarray(load_eval_config().embodiments.franka.joint_upper, dtype=np.float64)
    _, shim = _make_shim()
    joints = np.zeros(8, dtype=np.float32)
    joints[3] = upper[3] + 1e-4
    shim._on_reset(FullObs(
        proprio=ProprioObs(joint_pos=joints, cartesian_pos=np.zeros(8, dtype=np.float32), sim_time=0.0),
        cameras={},
    ))

    assert shim._current_joints[3] == pytest.approx(upper[3])
    assert shim.home_joint_position[3] == pytest.approx(float(joints[3]))


def test_reset_snapshots_joints_and_returns_remapped_obs():
    client, shim = _make_shim()
    obs, info = shim.reset()

    assert shim._episode_id == "ep-0"
    assert shim.last_instruction == client.manifest_entries[0]["instruction"]
    assert info == {"task_prompt": shim.last_instruction}
    # The fake starts at all zeros, a hair outside joint 4's upper limit: the hold
    # target is clamped to it, the home position is what was measured.
    np.testing.assert_array_equal(shim._current_joints, clamp_to_joint_limits(np.zeros(7))[0])
    np.testing.assert_array_equal(shim.home_joint_position, np.zeros(7))
    assert shim._current_joints.dtype == np.float64
    assert shim._gripper_fraction == 1.0
    assert not shim._episode_over
    assert shim._sim_step_count == 0

    assert set(obs.keys()) == {"robot0_robotview", "robot_joint_pos", "robot_cartesian_pos", "sim_time"}
    assert obs["robot_joint_pos"].dtype == np.float64
    assert obs["robot_cartesian_pos"].dtype == np.float64
    assert "cube_poses" not in obs and "task" not in obs


def test_reset_time_gateway_error_propagates_unchanged():
    client, shim = _make_shim()
    client.fail_at(0, GatewayError(ErrorCode.TRIALS_EXHAUSTED, "no trials left"))

    with pytest.raises(GatewayError) as exc_info:
        shim.reset()
    assert exc_info.value.code is ErrorCode.TRIALS_EXHAUSTED


def test_obs_depth_is_hw1_and_supports_both_legacy_access_patterns():
    client, shim = _make_shim()
    obs, _ = shim.reset()

    depth = obs["robot0_robotview"]["images"]["depth"]
    assert depth.shape == (32, 32, 1)
    assert depth.dtype == np.float32
    # control.py: depth[:, :, 0]
    assert depth[:, :, 0].shape == (32, 32)
    # libero.py / libero_reduced.py / control_reduced.py: .squeeze(-1)
    assert depth.squeeze(-1).shape == (32, 32)


def test_obs_depth_key_omitted_when_wire_depth_is_none():
    client, shim = _make_shim()
    shim.reset()
    proprio = ProprioObs(
        joint_pos=np.zeros(8, dtype=np.float32),
        cartesian_pos=np.zeros(8, dtype=np.float32),
        sim_time=1.0,
    )
    cam = CameraObs(
        name="robot0_robotview",
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=None,
        intrinsics=np.eye(3, dtype=np.float32),
        pose=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        pose_mat=np.eye(4, dtype=np.float32),
    )
    remapped = shim._remap(FullObs(proprio=proprio, cameras={"robot0_robotview": cam}))
    assert "depth" not in remapped["robot0_robotview"]["images"]


def test_pose_mat_synthesized_when_wire_pose_mat_is_none():
    client, shim = _make_shim()
    shim.reset()
    proprio = ProprioObs(
        joint_pos=np.zeros(8, dtype=np.float32),
        cartesian_pos=np.zeros(8, dtype=np.float32),
        sim_time=1.0,
    )
    # translation (1, 2, 3), identity rotation (wxyz = 1,0,0,0)
    cam = CameraObs(
        name="robot0_robotview",
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=None,
        intrinsics=np.eye(3, dtype=np.float32),
        pose=np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        pose_mat=None,
    )
    remapped = shim._remap(FullObs(proprio=proprio, cameras={"robot0_robotview": cam}))
    expected = np.array(
        [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=np.float64
    )
    np.testing.assert_allclose(remapped["robot0_robotview"]["pose_mat"], expected, atol=1e-9)


def test_pose_mat_passed_through_when_present_on_wire():
    client, shim = _make_shim()
    obs, _ = shim.reset()
    # FakeEnvClient's synthetic camera always ships an explicit pose_mat.
    np.testing.assert_array_equal(obs["robot0_robotview"]["pose_mat"], np.eye(4, dtype=np.float32))


# ---------------------------------------------------------------------------
# move_to_joints_blocking: convergence, RPC count, chunk clamping
# ---------------------------------------------------------------------------


def test_move_to_joints_blocking_converges_and_early_exits():
    client, shim = _make_shim(step_chunk=10)
    shim.reset()
    target = np.array([0.1, -0.2, 0.05, -1.0, 0.0, 0.3, 0.0])

    shim.move_to_joints_blocking(target, tolerance=0.02, max_steps=100)

    # Convergence in 2 chunks of 10 (20 raw cycles at JOINT_STEP=0.05/cycle);
    # early-exit means well under ceil(100/10) = 10 RPCs.
    assert len(client.actions) < 10
    np.testing.assert_allclose(shim._latest_proprio().joint_pos[:7], target, atol=1e-9)


def test_move_to_joints_blocking_clamps_chunk_to_max_repeat():
    client, shim = _make_shim(step_chunk=300)
    shim.reset()
    # The first RPC's repeat is bounded by MAX_REPEAT (200), not by step_chunk
    # (300) or by the full remaining budget (250). The target is the far end of
    # joint 5's published range: a target outside the range would be saturated
    # before it ever reached an action.
    target = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 3.7525, 0.0])

    shim.move_to_joints_blocking(target, tolerance=0.02, max_steps=250)

    assert client.actions[0].repeat == MAX_REPEAT
    assert all(a.repeat <= MAX_REPEAT for a in client.actions)
    np.testing.assert_allclose(shim._latest_proprio().joint_pos[:7], target, atol=1e-9)


# ---------------------------------------------------------------------------
# joint limits
# ---------------------------------------------------------------------------


def test_a_target_past_the_joint_stop_is_saturated_not_sent():
    """The IK behind the Cartesian API returns targets a hair outside the limits,
    and the referee's bounds check ends the episode over it — 22 of 76 trials in
    one measured round, on actions the harness itself produced. A position
    controller saturates at the stop instead."""
    client, shim = _make_shim()
    shim.reset()
    limits = load_eval_config().embodiments.franka
    target = np.zeros(7)
    target[3] = limits.joint_upper[3] + 0.001  # q4's upper limit is negative

    shim.move_to_joints_blocking(target, tolerance=0.02, max_steps=20)

    assert client.actions, "the move still happens; it is the target that is clipped"
    for action in client.actions:
        assert action.joints[3] == pytest.approx(limits.joint_upper[3])
        check_franka_bounds(action, limits.joint_lower, limits.joint_upper)


def test_a_target_inside_the_limits_is_untouched():
    client, shim = _make_shim()
    shim.reset()
    target = np.array([0.1, -0.2, 0.05, -1.0, 0.0, 0.3, 0.0])

    shim.move_to_joints_non_blocking(target)

    np.testing.assert_array_equal(shim._current_joints, target)


def test_move_to_joints_non_blocking_sets_target_without_rpc():
    client, shim = _make_shim()
    shim.reset()
    # q4 is negative over its whole published range, so 0.4 (what this used to
    # ask for) is not a pose the arm has.
    target = np.array([0.1, 0.2, 0.3, -0.4, 0.5, 0.6, 0.7])

    shim.move_to_joints_non_blocking(target)

    assert len(client.actions) == 0
    np.testing.assert_array_equal(shim._current_joints, target)


# ---------------------------------------------------------------------------
# _step_once / _set_gripper
# ---------------------------------------------------------------------------


def test_step_once_holds_current_joints_and_gripper_passthrough():
    client, shim = _make_shim()
    shim.reset()
    held = np.array([0.5, -0.5, 0.1, -1.5, 0.2, 1.0, 0.3])
    shim._current_joints = held
    shim._set_gripper(0.37)

    shim._step_once()

    assert len(client.actions) == 1
    action = client.actions[0]
    np.testing.assert_array_equal(action.joints, held)
    assert action.gripper == pytest.approx(0.37)
    assert action.repeat == 1


def test_set_gripper_clips_to_unit_interval():
    client, shim = _make_shim()
    shim.reset()

    shim._set_gripper(1.5)
    assert shim._gripper_fraction == 1.0

    shim._set_gripper(-0.5)
    assert shim._gripper_fraction == 0.0


# ---------------------------------------------------------------------------
# episode lifecycle: magic string, GatewayError conversion
# ---------------------------------------------------------------------------


def test_step_after_termination_raises_magic_string():
    client, shim = _make_shim()
    shim.reset()
    client.set_flags_after(1, truncated=True)

    shim._step_once()  # 1st raw cycle: flips truncated -> episode_over
    assert shim._episode_over
    assert shim._sim_step_count == shim.max_steps

    with pytest.raises(RuntimeError, match=EPISODE_OVER_MESSAGE) as exc_info:
        shim._step_once()
    assert str(exc_info.value) == EPISODE_OVER_MESSAGE
    # The second call never reached the fake wire.
    assert len(client.actions) == 1


def test_gateway_error_mid_step_becomes_runtime_error_and_ends_episode():
    client, shim = _make_shim()
    shim.reset()
    client.fail_at(1, GatewayError(ErrorCode.ENV_CRASH, "sim exploded"))

    with pytest.raises(RuntimeError) as exc_info:
        shim._step_once()

    assert "gateway error [env_crash]: sim exploded" == str(exc_info.value)
    assert not isinstance(exc_info.value, GatewayError)
    assert shim._episode_over
    assert shim._sim_step_count == shim.max_steps

    # A further step raises the terminated-episode magic string, not a
    # second RPC error.
    with pytest.raises(RuntimeError, match=EPISODE_OVER_MESSAGE):
        shim._step_once()


def test_an_invalid_action_raises_but_the_trial_goes_on():
    """The gateway refuses an out-of-bounds action and keeps the episode alive, and
    so does the shim: the agent's code sees the error and may try another action."""
    client, shim = _make_shim()
    shim.reset()
    client.fail_at(1, GatewayError(ErrorCode.INVALID_ACTION, "joints outside limits"))

    with pytest.raises(RuntimeError, match="invalid_action"):
        shim._step_once()
    assert not shim._episode_over

    shim._step_once()  # the next action reaches the gateway
    assert len(client.actions) >= 1


def test_a_client_side_invalid_action_does_not_end_the_trial():
    client, shim = _make_shim()
    shim.reset()
    real_step = client.step

    def _boom_once(episode_id, action):
        client.step = real_step  # type: ignore[method-assign]
        raise InvalidActionError("franka.joints outside published joint limits")

    client.step = _boom_once  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="invalid action"):
        shim._step_once()
    assert not shim._episode_over
    shim._step_once()

def test_compute_reward_always_zero():
    client, shim = _make_shim()
    shim.reset()
    assert shim.compute_reward() == 0.0


def test_enable_video_capture_signature_has_wrist_camera_kwarg():
    sig = inspect.signature(FrankaGatewayShim.enable_video_capture)
    assert "wrist_camera" in sig.parameters


def test_video_buffer_wrist_camera_flag_gates_wrist_buffer():
    client = _wrist_entry_client()
    shim = FrankaGatewayShim(client, "wrist_task", camera_names=["robot0_eye_in_hand"])
    shim.reset()

    # No main camera in this manifest (only the wrist-named one) -> main
    # camera resolution is None, so nothing lands in the main buffer even
    # while capturing; the wrist flag controls the wrist buffer only.
    shim.enable_video_capture(True, wrist_camera=False)
    shim._step_once()
    assert shim.get_video_frame_count() == 0
    assert len(shim.get_wrist_video_frames()) == 0

    shim.enable_video_capture(True, wrist_camera=True)
    for _ in range(3):
        shim._step_once()
    assert shim.get_video_frame_count() == 0
    assert len(shim.get_wrist_video_frames()) == 3


def test_enable_video_capture_clear_resets_buffers():
    client, shim = _make_shim(step_chunk=1)
    shim.reset()
    shim.enable_video_capture(True)
    shim._step_once()
    shim._step_once()
    assert shim.get_video_frame_count() == 2

    shim.enable_video_capture(True, clear=True)
    shim._step_once()
    assert shim.get_video_frame_count() == 1


def test_get_video_frames_clear_flag():
    client, shim = _make_shim(step_chunk=1)
    shim.reset()
    shim.enable_video_capture(True)
    shim._step_once()
    shim._step_once()
    assert shim.get_video_frame_count() == 2

    frames = shim.get_video_frames(clear=True)
    assert len(frames) == 2
    assert shim.get_video_frame_count() == 0


def test_get_video_frames_range():
    client, shim = _make_shim(step_chunk=1)
    shim.reset()
    shim.enable_video_capture(True)
    for _ in range(5):
        shim._step_once()
    assert len(shim.get_video_frames_range(1, 3)) == 2


# ---------------------------------------------------------------------------
# get_observation() / render()
# ---------------------------------------------------------------------------


def test_get_observation_is_fresh_rpc_and_remapped():
    client, shim = _make_shim()
    shim.reset()

    obs = shim.get_observation()

    assert set(obs.keys()) == {"robot0_robotview", "robot_joint_pos", "robot_cartesian_pos", "sim_time"}


def test_render_returns_main_camera_rgb():
    client, shim = _make_shim()
    shim.reset()

    frame = shim.render()

    assert frame is not None
    assert frame.shape == (32, 32, 3)
    assert frame.dtype == np.uint8


def test_render_wrist_none_when_no_wrist_camera_in_manifest():
    client, shim = _make_shim()  # only robot0_robotview: no wrist camera
    shim.reset()
    assert shim.render_wrist() is None


def test_render_wrist_returns_frame_when_wrist_camera_present():
    client = _wrist_entry_client()
    shim = FrankaGatewayShim(client, "wrist_task", camera_names=["robot0_eye_in_hand"])
    shim.reset()

    assert shim.render() is None  # no non-wrist camera in this manifest
    frame = shim.render_wrist()
    assert frame is not None
    assert frame.shape == (32, 32, 3)


def _count_get_observation_calls(client: FakeEnvClient) -> list[int]:
    """Wrap ``client.get_observation`` with a call counter, returned as a
    mutable single-element list (so the closure can be read after patching)."""
    calls: list[int] = []
    original = client.get_observation

    def _counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    client.get_observation = _counting  # type: ignore[method-assign]
    return calls


def test_render_calls_reuse_cached_obs_when_no_step_in_between():
    """Two render() calls with no _remote_step() in between must cost exactly
    1 get_observation RPC, not 2."""
    client, shim = _make_shim()
    shim.reset()
    shim._step_once()  # invalidate the reset-time cache
    calls = _count_get_observation_calls(client)

    shim.render()
    shim.render()

    assert len(calls) == 1


def test_render_after_step_triggers_a_new_rpc():
    client, shim = _make_shim()
    shim.reset()
    shim._step_once()  # invalidate the reset-time cache
    calls = _count_get_observation_calls(client)

    shim.render()
    assert len(calls) == 1

    shim._step_once()  # a raw cycle ran -> the cached full obs is now stale again
    shim.render()
    assert len(calls) == 2


def test_get_observation_also_reuses_cache_when_not_stale():
    client, shim = _make_shim()
    shim.reset()
    shim._step_once()
    calls = _count_get_observation_calls(client)

    shim.get_observation()
    shim.get_observation()

    assert len(calls) == 1


# ---------------------------------------------------------------------------
# no sim imports anywhere under capx/submission/shims/
# ---------------------------------------------------------------------------


def test_no_sim_imports_in_shim_modules():
    import ast
    import pathlib

    shims_dir = pathlib.Path(__file__).resolve().parents[2] / "capx" / "submission" / "shims"
    banned = ("capx.envs.simulators", "robosuite", "libero", "omnigibson")
    for path in shims_dir.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                assert not any(name.startswith(b) for b in banned), f"{path}: banned import {name!r}"


def test_the_fake_client_runs_a_two_arm_task_on_two_arms():
    """It used to build the 28-dim R1Pro model for ``franka_bimanual`` entries."""
    from capx.submission.shims import BimanualFrankaGatewayShim

    client = FakeEnvClient()
    shim = BimanualFrankaGatewayShim(client, "two_arm_lift", camera_names=["agentview"])
    shim.reset()
    target = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])

    shim.move_to_joints_blocking_arm1(target, max_steps=60)

    measured = shim._measured_joints()
    np.testing.assert_allclose(measured[1], target, atol=1e-6)
    np.testing.assert_allclose(measured[0], clamp_to_joint_limits(np.zeros(7))[0], atol=1e-6)
