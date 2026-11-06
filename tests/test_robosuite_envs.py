"""The Robosuite envs, built for real but without a renderer.

Privileged envs with rendering off need only MuJoCo, so these run on a CPU. They skip
where Robosuite or viser is not installed (a bare checkout); the env images have both.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("robosuite")
pytest.importorskip("viser")

import capx.envs.simulators  # noqa: E402,F401  — registers the low-level envs
from capx.envs.base import get_env  # noqa: E402
from capx.envs.simulators.robosuite_spill_wipe import FrankaRobosuiteSpillWipeLowLevel  # noqa: E402

SINGLE_ARM = [
    "franka_robosuite_cubes_low_level",
    "franka_robosuite_cube_lift_low_level",
    "franka_robosuite_cubes_restack_low_level",
    "franka_robosuite_spill_wipe_low_level",
    "franka_robosuite_nut_assembly_low_level",
]
TWO_ARM = ["two_arm_lift_robosuite", "two_arm_handover_robosuite"]

# Shorter than nut assembly's 100 settling cycles, so a reset that spent them on the
# episode could not even finish.
STEP_LIMIT = 40


def _build(name: str) -> Any:
    return get_env.__wrapped__(name, privileged=True, enable_render=False, max_steps=STEP_LIMIT)


@pytest.fixture(params=SINGLE_ARM)
def single_arm(request):
    env = _build(request.param)
    env.reset(seed=1)
    yield env
    env.close()


@pytest.fixture(params=TWO_ARM)
def two_arm(request):
    env = _build(request.param)
    env.reset(seed=1)
    yield env
    env.close()


def test_reset_leaves_the_whole_step_limit_to_the_episode(single_arm) -> None:
    assert single_arm.robosuite_env.timestep == 0
    assert single_arm._sim_step_count == 0
    for _ in range(STEP_LIMIT):
        single_arm._step_once()
    assert single_arm._sim_step_count == STEP_LIMIT
    with pytest.raises(ValueError, match="terminated episode"):
        single_arm._step_once()


def test_step_counts_restart_on_every_reset(single_arm) -> None:
    for _ in range(STEP_LIMIT):
        *_, truncated, _ = single_arm.step(None)
    assert truncated
    single_arm.reset(seed=2)
    *_, truncated, _ = single_arm.step(None)
    assert not truncated
    assert single_arm._step_count == 1


def test_proprio_and_full_observations_are_eight_wide(single_arm) -> None:
    from capx.gateway.robosuite_adapter import RobosuiteLowLevelEnv

    proprio = RobosuiteLowLevelEnv(single_arm).get_proprio()
    full = single_arm.get_observation()
    for obs in (proprio, full):
        assert obs["robot_joint_pos"].shape == (8,)
        assert obs["robot_cartesian_pos"].shape == (8,)


def test_cartesian_pose_follows_the_arm(single_arm) -> None:
    before = single_arm.get_observation()
    target = before["robot0_joint_pos"] + np.array([0.3, 0.2, 0.0, -0.2, 0.0, 0.2, 0.0])
    single_arm.move_to_joints_blocking(target, max_steps=10)
    after = single_arm.get_observation()
    moved = np.linalg.norm(after["robot_cartesian_pos"][:3] - before["robot_cartesian_pos"][:3])
    assert moved > 0.01


def test_non_blocking_move_keeps_its_target(single_arm) -> None:
    target = single_arm._joint_positions() + 0.1
    single_arm.move_to_joints_non_blocking(target)
    np.testing.assert_allclose(single_arm._current_joints, target)
    assert single_arm._sim_step_count == 1


def test_close_frees_the_sim() -> None:
    env = _build("franka_robosuite_cubes_low_level")
    env.close()
    assert env.robosuite_env.sim is None


def test_spill_wipe_gripper_has_no_fingers_and_reads_open() -> None:
    """Wipe's gripper has no joints, so robot0_gripper_qpos is empty."""

    class _Data:
        xquat = np.array([[1.0, 0.0, 0.0, 0.0]] * 2)
        xpos = np.zeros((2, 3))

    class _Sim:
        data = _Data()

    class _Robosuite:
        sim = _Sim()

    env = object.__new__(FrankaRobosuiteSpillWipeLowLevel)
    env.robosuite_env = _Robosuite()
    env.gripper_link_idx = 1
    env.base_link_wxyz_xyz = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    env.gripper_metric_length = 0.04
    obs: dict[str, Any] = {"robot0_joint_pos": np.zeros(7), "robot0_gripper_qpos": np.array([])}
    env._compute_gripper_obs(obs)
    assert obs["robot_joint_pos"].shape == (8,)
    assert obs["robot_cartesian_pos"].shape == (8,)
    assert obs["robot_joint_pos"][-1] == obs["robot_cartesian_pos"][-1] == 1.0


def test_two_arm_reset_reopens_both_grippers(two_arm) -> None:
    two_arm._set_gripper(0.0)
    two_arm._set_gripper_arm1(0.0)
    two_arm.reset(seed=2)
    assert (two_arm._gripper_fraction_0, two_arm._gripper_fraction_1) == (1.0, 1.0)


def test_two_arm_observation_without_cameras(two_arm) -> None:
    obs = two_arm.get_observation()
    assert "robot0_robotview" not in obs
    assert obs["robot0_cartesian_pos"].shape == (8,)


def test_two_arm_control_cycle_skips_camera_renders(two_arm, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    step = two_arm.robosuite_env.step

    def recording_step(action, **kwargs):
        calls.append(kwargs)
        return step(action, **kwargs)

    monkeypatch.setattr(two_arm.robosuite_env, "step", recording_step)
    joints = two_arm.robosuite_env._get_observations()
    two_arm.move_to_joints_blocking_both(
        joints["robot0_joint_pos"], joints["robot1_joint_pos"], tolerance=0.0, max_steps=1
    )
    two_arm._step_once()
    assert calls == [{"skip_render_images": True}] * 2
    assert two_arm._sim_step_count == 2
