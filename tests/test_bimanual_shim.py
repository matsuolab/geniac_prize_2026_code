"""The submission side of a two-arm task.

The bimanual control APIs are written against a low-level env with per-arm methods
(``move_to_joints_blocking_arm1``, ``_set_gripper_arm1``, ``base_link_wxyz_xyz_1``).
The shim has to present that surface over an action that moves both arms at once,
which means every one-arm move is really a two-arm command where the other arm is
told to stay put. Getting that wrong looks like the idle arm drifting or snapping to
an old target, so the tests pin what the idle arm is commanded to do.

Runs against the in-process gateway, so no simulator is involved.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_bimanual import CAMERA, FakeBimanualFactory, bimanual_tasks, valid_joints
from test_http_transport import VALID_JOINTS

from capx.gateway import EnforcementMode, Gateway
from capx.protocol import EnvClient, InProcessTransport
from capx.submission.shims import BimanualFrankaGatewayShim


def shim_for() -> tuple[BimanualFrankaGatewayShim, FakeBimanualFactory]:
    factory = FakeBimanualFactory()
    gateway = Gateway(bimanual_tasks(), factory, EnforcementMode.DEV)
    client = EnvClient(InProcessTransport(gateway.acquire()))
    shim = BimanualFrankaGatewayShim(client, "t", camera_names=[CAMERA], step_chunk=4)
    shim.reset()
    return shim, factory


def resting() -> np.ndarray:
    """Where the fake env starts, i.e. what an arm told to hold is commanded."""
    return np.asarray(VALID_JOINTS, dtype=np.float64)


def commands(factory: FakeBimanualFactory):
    return factory.built[0].commands


def test_reset_exposes_both_base_frames() -> None:
    """A single-arm task never needs these; with two arms the APIs cannot relate
    the arms without them."""
    shim, _ = shim_for()
    np.testing.assert_allclose(shim.base_link_wxyz_xyz_0, np.zeros(7))
    np.testing.assert_allclose(shim.base_link_wxyz_xyz_1, np.ones(7))
    assert shim.home_joint_position.shape == (2, 7)


def test_moving_one_arm_tells_the_other_to_stay() -> None:
    shim, factory = shim_for()
    target = valid_joints()[0]

    shim.move_to_joints_blocking(target, max_steps=4)

    joints, _ = commands(factory)[0]
    np.testing.assert_allclose(joints[0], target, atol=1e-9)
    np.testing.assert_allclose(joints[1], resting(), atol=1e-6)  # measured, held


def test_moving_arm1_leaves_arm0_where_it_is() -> None:
    shim, factory = shim_for()
    target = valid_joints()[0]

    shim.move_to_joints_blocking_arm1(target, max_steps=4)

    joints, _ = commands(factory)[0]
    np.testing.assert_allclose(joints[0], resting(), atol=1e-6)  # measured, held
    np.testing.assert_allclose(joints[1], target, atol=1e-9)


def test_moving_both_sends_both_targets() -> None:
    shim, factory = shim_for()
    first, second = valid_joints()[0], valid_joints()[0] * 0.5

    shim.move_to_joints_blocking_both(first, second, max_steps=4)

    joints, _ = commands(factory)[0]
    np.testing.assert_allclose(joints[0], first, atol=1e-9)
    np.testing.assert_allclose(joints[1], second, atol=1e-9)


def test_a_one_arm_move_converges_on_that_arm_alone() -> None:
    """Checking both arms would let a one-arm move return as soon as the arm that
    was told to hold has settled — which it already has."""
    shim, factory = shim_for()

    shim.move_to_joints_blocking(valid_joints()[0], tolerance=1e-6, max_steps=8)

    # The fake env snaps to the command, so one chunk converges the moving arm.
    assert len(commands(factory)) == shim._step_chunk


def test_the_grippers_are_commanded_independently() -> None:
    shim, factory = shim_for()
    shim._set_gripper(0.25)
    shim._set_gripper_arm1(0.75)

    shim._step_once()

    _, grippers = commands(factory)[-1]
    np.testing.assert_allclose(grippers, [0.25, 0.75])


def test_step_once_holds_both_arms_where_they_are() -> None:
    shim, factory = shim_for()
    shim.move_to_joints_blocking_both(valid_joints()[0], valid_joints()[0], max_steps=4)
    before = commands(factory)[-1][0]

    shim._step_once()

    joints, _ = commands(factory)[-1]
    np.testing.assert_allclose(joints, before, atol=1e-9)


def test_the_obs_carries_both_arms_under_the_names_the_apis_read() -> None:
    shim, _ = shim_for()
    obs = shim.get_observation()
    assert obs["robot0_cartesian_pos"].shape == (8,)
    assert obs["robot1_cartesian_pos"].shape == (8,)
    assert obs[CAMERA]["images"]["rgb"].shape == (4, 4, 3)
    assert obs[CAMERA]["images"]["depth"].shape[-1] == 1  # the APIs index depth[:, :, 0]


def test_a_single_arm_task_is_rejected_loudly() -> None:
    """A mismatched pack entry should say so, not index into an empty list."""
    from test_http_transport import FakeFactory, make_tasks

    gateway = Gateway(make_tasks(), FakeFactory(), EnforcementMode.DEV)
    shim = BimanualFrankaGatewayShim(
        EnvClient(InProcessTransport(gateway.acquire())), "t", camera_names=["robot0_robotview"]
    )
    with pytest.raises(RuntimeError, match="franka_bimanual"):
        shim.reset()


def test_an_arm_resting_past_a_stop_is_held_at_the_stop() -> None:
    """MuJoCo's limits are soft and proprio crosses the wire as float32, so an arm
    at a stop can read a hair past it; the gateway's bounds check has no tolerance,
    and echoing that reading back as the hold refused every step of the episode."""
    from capx.config import load_eval_config

    upper = np.asarray(load_eval_config().embodiments.franka.joint_upper, dtype=np.float64)
    factory = FakeBimanualFactory()
    gateway = Gateway(bimanual_tasks(), factory, EnforcementMode.OFFICIAL)
    shim = BimanualFrankaGatewayShim(
        EnvClient(InProcessTransport(gateway.acquire())), "t", camera_names=[CAMERA]
    )
    shim.reset()
    env = factory.built[0]
    real_apply = env.apply_raw_action

    def settle_past_stop(action):
        out = real_apply(action)
        joints = np.array(env._joints, dtype=np.float64)
        joints[1][3] = upper[3] + 1e-4
        env._joints = joints
        return out

    env.apply_raw_action = settle_past_stop
    shim._step_once()

    shim._step_once()
    shim.move_to_joints_blocking(valid_joints()[0], max_steps=4)

    held, _ = commands(factory)[-1]
    assert held[1][3] == pytest.approx(upper[3])


def test_the_main_camera_is_also_under_the_name_the_reduced_apis_read() -> None:
    """The reduced Franka APIs index ``obs["robot0_robotview"]``; the local two-arm
    sims alias their render camera to it, so the gateway path has to as well."""
    shim, _ = shim_for()
    obs = shim.get_observation()
    assert obs["robot0_robotview"] is obs[CAMERA]
