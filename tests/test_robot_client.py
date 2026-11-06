"""Phase 3: the RobotClient primitive library drives the robot over EnvClient.step."""

from __future__ import annotations

import numpy as np

from capx.agent_client import RobotClient
from capx.gateway import EnforcementMode, Gateway
from capx.protocol import EnvClient, InProcessTransport
from test_gateway import FakeFactory, make_tasks

TARGET = np.array([0.1, -0.4, 0.1, -1.8, 0.1, 1.6, 0.1])  # within Panda limits


def robot() -> RobotClient:
    gateway = Gateway(make_tasks(), FakeFactory(), EnforcementMode.DEV)
    return RobotClient.start(EnvClient(InProcessTransport(gateway.acquire())), "t")


def test_move_to_joints_reaches_target() -> None:
    reached = robot().move_to_joints(TARGET, max_steps=5)
    np.testing.assert_allclose(reached, TARGET, atol=1e-5)


def test_gripper_open_close_tracks_command() -> None:
    r = robot()
    r.close_gripper(cycles=2)
    assert r.get_joints().shape == (7,)  # servo state stays valid after gripper hold
    # the fake env writes the commanded gripper into proprio's last joint element
    assert r.observe().proprio.joint_pos[-1] == 0.0
    r.open_gripper(cycles=2)
    assert r.observe().proprio.joint_pos[-1] == 1.0


def test_goto_pose_uses_injected_ik() -> None:
    r = robot()
    reached = r.goto_pose(np.zeros(7), ik=lambda _pose: TARGET, max_steps=5)
    np.testing.assert_allclose(reached, TARGET, atol=1e-5)
