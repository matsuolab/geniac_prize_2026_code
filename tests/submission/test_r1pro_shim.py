"""Unit tests for ``R1ProGatewayShim``/``R1ProKinematics``, sim-free.

Everything except the IK/FK tests (guarded by ``pytest.importorskip("pyroki")``)
is pure numpy/torch and runs against ``tests.submission.fakes.FakeR1ProModel``
via ``FakeEnvClient`` — no simulator, no network, no pyroki/JAX.

Env-surface coverage and the traced ``_move_hand`` quaternion convention
are documented in ``capx/submission/shims/r1pro.py``'s
module docstring; this file exercises both.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest
import torch

from capx.gateway.r1pro_adapter import _CAMERA_SENSOR_SUFFIX as _ADAPTER_CAMERA_SENSOR_SUFFIX
from capx.protocol import CameraObs, FullObs, ProprioObs
from capx.protocol.actions import MAX_REPEAT
from capx.submission.interfaces import R1ProKnobs
from capx.submission.shims import R1ProGatewayShim
from capx.submission.shims import r1pro as r1pro_module
from capx.submission.shims.r1pro import _CAMERA_SENSOR_SUFFIX as _SHIM_CAMERA_SENSOR_SUFFIX
from capx.submission.shims.r1pro_kinematics import R1ProKinematics
from tests.submission.fakes import FakeEnvClient

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

TASK_ID = "picking_up_trash"
CAMERA_NAMES = ["ego", "left_wrist", "right_wrist"]


def _make_shim(**kwargs) -> tuple[FakeEnvClient, R1ProGatewayShim]:
    client = FakeEnvClient()
    shim = R1ProGatewayShim(client, TASK_ID, camera_names=CAMERA_NAMES, **kwargs)
    return client, shim


def _set_q28(shim: R1ProGatewayShim, q28: np.ndarray) -> None:
    """Force the shim's cached proprio to a specific 28-dim joint vector
    (test-only hook into ``GatewayShimBase``'s cached-proprio attribute)."""
    shim._latest_proprio_obs = ProprioObs(
        joint_pos=np.asarray(q28, dtype=np.float32),
        cartesian_pos=np.zeros(14, dtype=np.float32),
        sim_time=0.0,
    )


# ---------------------------------------------------------------------------
# R1ProKinematics: q28_to_action25 (pure numpy)
# ---------------------------------------------------------------------------


def test_q28_to_action25_layout_at_zero_yaw():
    kinematics = R1ProKinematics()
    q = np.arange(28, dtype=np.float64) * 0.01
    action = kinematics.q28_to_action25(q, base_pose_now=(0.0, 0.0, 0.0))

    np.testing.assert_allclose(action[0:2], q[0:2], atol=1e-12)  # base delta == q[0:2] at (0,0,yaw=0)
    assert action[2] == pytest.approx(q[5])
    np.testing.assert_allclose(action[3:7], q[6:10])  # trunk
    np.testing.assert_allclose(action[7:14], q[10:24:2])  # left arm (interleaved)
    np.testing.assert_allclose(action[14:16], q[24:26])  # left fingers
    np.testing.assert_allclose(action[16:23], q[11:24:2])  # right arm (interleaved)
    np.testing.assert_allclose(action[23:25], q[26:28])  # right fingers


def test_q28_to_action25_base_local_delta_at_yaw_pi_over_2():
    kinematics = R1ProKinematics()
    q = np.zeros(28, dtype=np.float64)
    q[0:2] = [1.0, 0.0]  # target 1m along world +x
    q[5] = np.pi / 2  # target yaw == current yaw (no rotation error)

    # Base is currently at the origin, facing +pi/2 (world +y).
    action = kinematics.q28_to_action25(q, base_pose_now=(0.0, 0.0, np.pi / 2))

    # World delta (1, 0) rotated into the base-local frame at yaw=pi/2:
    # local = R(-yaw) @ world = R(-pi/2) @ (1, 0) = (0, -1).
    np.testing.assert_allclose(action[0:2], [0.0, -1.0], atol=1e-9)
    assert action[2] == pytest.approx(0.0, abs=1e-9)


def test_q28_to_action25_wraps_yaw_error():
    kinematics = R1ProKinematics()
    q = np.zeros(28, dtype=np.float64)
    q[5] = -np.pi + 0.1  # target yaw just past -pi

    action = kinematics.q28_to_action25(q, base_pose_now=(0.0, 0.0, np.pi - 0.1))
    # Raw difference would be (-pi+0.1) - (pi-0.1) = -2*pi + 0.2, wrapped to 0.2.
    assert action[2] == pytest.approx(0.2, abs=1e-9)


def test_joint_limits_reads_urdf_active_limit_ignoring_comments():
    """The vendored URDF comments out a stricter torso_joint1 limit above the
    active ``<limit lower="-1" upper="1" .../>`` tag — this pins the parser
    to the ACTIVE (uncommented) element, not a naive string/regex scan."""
    kinematics = R1ProKinematics()
    lower, upper = kinematics.joint_limits("torso_joint1")
    assert lower == pytest.approx(-1.0)
    assert upper == pytest.approx(1.0)

    lower, upper = kinematics.joint_limits("left_gripper_finger_joint1")
    assert lower == pytest.approx(0.0)
    assert upper == pytest.approx(0.05)


def test_joint_limits_unknown_joint_raises_keyerror():
    kinematics = R1ProKinematics()
    with pytest.raises(KeyError):
        kinematics.joint_limits("not_a_real_joint")


def test_urdf_path_resolved_via_pathlib_not_cwd():
    assert R1ProKinematics.URDF_PATH.is_absolute()
    assert R1ProKinematics.URDF_PATH.name == "r1pro_ik.urdf"
    assert R1ProKinematics.URDF_PATH.is_file()


# ---------------------------------------------------------------------------
# PYROKI_JOINT_MAPPING / CONTROLLER_JOINT_MAPPING parity against the legacy
# API's own verbatim source (capx/integrations/r1pro/control.py:175-176),
# parsed with ast -- no import, so this never needs jax/pyroki installed.
# ---------------------------------------------------------------------------


def _extract_self_attr_list_literal(source: str, attr_name: str) -> list[int]:
    """AST-extract the RHS of a ``self.<attr_name> = [...]`` assignment."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == attr_name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"no assignment to self.{attr_name} found in source")


def _control_py_source() -> str:
    return (REPO_ROOT / "capx" / "integrations" / "r1pro" / "control.py").read_text()


def test_pyroki_joint_mapping_matches_control_py_source():
    legacy = _extract_self_attr_list_literal(_control_py_source(), "pyroki_joint_mapping")
    assert legacy == R1ProKinematics.PYROKI_JOINT_MAPPING


def test_controller_joint_mapping_matches_control_py_source():
    legacy = _extract_self_attr_list_literal(_control_py_source(), "controller_joint_mapping")
    assert legacy == R1ProKinematics.CONTROLLER_JOINT_MAPPING


# ---------------------------------------------------------------------------
# _CAMERA_SENSOR_SUFFIX: direct import from capx.gateway.r1pro_adapter, not a
# duplicated dict (reuse fix)
# ---------------------------------------------------------------------------


def test_camera_sensor_suffix_is_the_same_object_as_the_adapters():
    assert _SHIM_CAMERA_SENSOR_SUFFIX is _ADAPTER_CAMERA_SENSOR_SUFFIX


# ---------------------------------------------------------------------------
# reset() / obs remap
# ---------------------------------------------------------------------------


def test_reset_returns_robot_gateway_nested_dict():
    client, shim = _make_shim()
    obs, info = shim.reset()

    assert set(obs.keys()) == {"robot_gateway"}
    robot_dict = obs["robot_gateway"]
    assert set(robot_dict.keys()) == {
        "robot_gateway:zed_link:Camera:0",
        "robot_gateway:left_realsense_link:Camera:0",
        "robot_gateway:right_realsense_link:Camera:0",
    }
    ego = robot_dict["robot_gateway:zed_link:Camera:0"]
    assert isinstance(ego["rgb"], torch.Tensor)
    assert ego["rgb"].shape == (32, 32, 3)
    assert isinstance(ego["depth_linear"], torch.Tensor)
    assert ego["depth_linear"].shape == (32, 32)  # squeezed from (32,32,1)
    assert info == {"task_prompt": shim.last_instruction}


def test_reset_populates_robot_sensors():
    client, shim = _make_shim()
    shim.reset()

    assert set(shim.robot.sensors.keys()) == {
        "robot_gateway:zed_link:Camera:0",
        "robot_gateway:left_realsense_link:Camera:0",
        "robot_gateway:right_realsense_link:Camera:0",
    }
    ego_sensor = shim.robot.sensors["robot_gateway:zed_link:Camera:0"]
    assert ego_sensor.intrinsic_matrix.shape == (3, 3)


def test_env_property_raises_on_any_attribute_access():
    client, shim = _make_shim()
    shim.reset()
    with pytest.raises(RuntimeError, match=r"not available on the gateway client: \.external_sensors"):
        _ = shim.env.external_sensors


def test_torso_joint_limits_matches_urdf():
    client, shim = _make_shim()
    expected = R1ProKinematics().joint_limits("torso_joint1")
    assert shim.torso_joint_limits[0].item() == pytest.approx(expected[0])
    assert shim.torso_joint_limits[1].item() == pytest.approx(expected[1])


# ---------------------------------------------------------------------------
# R1ProRobotView / R1ProSensorView pose composition
# ---------------------------------------------------------------------------


def test_sensor_view_world_pose_composition_against_hand_built_case():
    client, shim = _make_shim()
    shim.reset()

    # Base at world (1, 0, 0), yaw = pi/2 (pure Z-axis rotation; roll=pitch=0).
    q28 = np.zeros(28, dtype=np.float64)
    q28[0:3] = [1.0, 0.0, 0.0]
    q28[3:6] = [0.0, 0.0, np.pi / 2]
    _set_q28(shim, q28)

    # Camera 1m ahead in the base's local +x, identity orientation.
    cam = CameraObs(
        name="ego",
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=None,
        intrinsics=np.eye(3, dtype=np.float32),
        pose=np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # [xyz, qwxyz]
        pose_mat=None,
    )
    shim._remap(FullObs(proprio=ProprioObs(joint_pos=q28.astype(np.float32), cartesian_pos=np.zeros(14, np.float32), sim_time=0.0), cameras={"ego": cam}))

    world_pos, world_quat_xyzw = shim.robot.sensors["robot_gateway:zed_link:Camera:0"].get_position_orientation()

    # R_z(pi/2) @ (1, 0, 0) = (0, 1, 0); world = base_pos + that = (1, 1, 0).
    np.testing.assert_allclose(world_pos.numpy(), [1.0, 1.0, 0.0], atol=1e-5)
    # Composed rotation == base rotation alone (camera-local rotation is identity):
    # a pure z rotation by pi/2, xyzw = (0, 0, sin(pi/4), cos(pi/4)).
    expected_quat = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    np.testing.assert_allclose(world_quat_xyzw.numpy(), expected_quat, atol=1e-5)


def test_robot_view_get_position_orientation_identity_at_origin():
    client, shim = _make_shim()
    shim.reset()  # FakeR1ProModel starts at q28 == 0

    pos, quat_xyzw = shim.robot.get_position_orientation()
    np.testing.assert_allclose(pos.numpy(), [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(quat_xyzw.numpy(), [0.0, 0.0, 0.0, 1.0], atol=1e-9)


# ---------------------------------------------------------------------------
# R1ProRobotView.get_relative_eef_pose (correctness fix: control.py's
# get_robot_relative_eef_pose calls self._env.robot.get_relative_eef_pose,
# which the shim previously lacked entirely). IK/FK is stubbed via
# monkeypatch (no pyroki/JAX needed) -- the IK/FK values themselves are
# already covered by test_solve_ik_and_fk_eef_roundtrip below.
# ---------------------------------------------------------------------------


def test_get_relative_eef_pose_returns_base_frame_fk_pose(monkeypatch):
    client, shim = _make_shim()
    shim.reset()

    def _stub_fk_eef(q28, arm):
        pos = np.array([0.1, 0.2, 0.3])
        quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # identity
        return pos, quat_wxyz

    monkeypatch.setattr(shim._kinematics, "fk_eef", _stub_fk_eef)

    pos, quat_xyzw = shim.robot.get_relative_eef_pose(arm="left")
    assert isinstance(pos, torch.Tensor)
    assert isinstance(quat_xyzw, torch.Tensor)
    np.testing.assert_allclose(pos.numpy(), [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(quat_xyzw.numpy(), [0.0, 0.0, 0.0, 1.0], atol=1e-6)  # wxyz[1,0,0,0] -> xyzw


def test_get_relative_eef_pose_string_arm_matches_int_arm(monkeypatch):
    """control.py:899-909 maps its own int arm to the string 'left'/'right'
    before calling robot.get_relative_eef_pose -- both spellings must resolve
    to the same arm."""
    client, shim = _make_shim()
    shim.reset()

    def _stub_fk_eef(q28, arm):
        pos = np.array([0.1, 0.2, 0.3]) if arm == 0 else np.array([0.4, 0.5, 0.6])
        quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        return pos, quat_wxyz

    monkeypatch.setattr(shim._kinematics, "fk_eef", _stub_fk_eef)

    pos_left_str, quat_left_str = shim.robot.get_relative_eef_pose(arm="left")
    pos_left_int, quat_left_int = shim.robot.get_relative_eef_pose(arm=0)
    np.testing.assert_allclose(pos_left_str.numpy(), pos_left_int.numpy())
    np.testing.assert_allclose(quat_left_str.numpy(), quat_left_int.numpy())

    pos_right_str, quat_right_str = shim.robot.get_relative_eef_pose(arm="right")
    pos_right_int, quat_right_int = shim.robot.get_relative_eef_pose(arm=1)
    np.testing.assert_allclose(pos_right_str.numpy(), pos_right_int.numpy())
    np.testing.assert_allclose(quat_right_str.numpy(), quat_right_int.numpy())
    # And the two arms actually differ (stub is arm-dependent) -- sanity check
    # this test isn't vacuously true from both branches returning the same value.
    assert not np.allclose(pos_left_str.numpy(), pos_right_str.numpy())


def test_get_relative_eef_pose_equals_world_pose_at_identity_base(monkeypatch):
    """At an identity base pose (origin, zero yaw -- FakeR1ProModel's initial
    state), BASE frame == WORLD frame, so the "relative" (base-frame) pose
    must equal get_robot_eef_pose's WORLD-frame pose -- pinning that
    get_relative_eef_pose is the base-frame FK pose, not
    get_robot_eef_pose's own world-composed value reused verbatim."""
    client, shim = _make_shim()
    shim.reset()

    def _stub_fk_eef(q28, arm):
        return np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(shim._kinematics, "fk_eef", _stub_fk_eef)

    rel_pos, rel_quat = shim.robot.get_relative_eef_pose(arm=0)
    world_pos, world_quat = shim.get_robot_eef_pose(arm=0)
    np.testing.assert_allclose(rel_pos.numpy(), world_pos.numpy(), atol=1e-6)
    np.testing.assert_allclose(rel_quat.numpy(), world_quat.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# JointView / R1ProControllerView
# ---------------------------------------------------------------------------


def test_controller_joint_view_get_state_reads_live_proprio():
    client, shim = _make_shim()
    shim.reset()
    q28 = np.zeros(28, dtype=np.float64)
    q28[6] = 0.42  # torso_joint1
    _set_q28(shim, q28)

    position, velocity, effort = shim.controller.robot.joints["torso_joint1"].get_state()
    assert position[0].item() == pytest.approx(0.42)
    assert velocity[0].item() == pytest.approx(0.0)
    assert effort[0].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _navigate_to_pose: base servo convergence on the fake base
# ---------------------------------------------------------------------------


def test_navigate_to_pose_converges_and_respects_per_command_clips():
    client, shim = _make_shim(step_chunk=5)
    shim.reset()

    ok = shim._navigate_to_pose((0.3, 0.2, 0.5))
    assert ok is True

    q = shim._q28_now()
    assert q[0] == pytest.approx(0.3, abs=0.05)
    assert q[1] == pytest.approx(0.2, abs=0.05)
    assert q[5] == pytest.approx(0.5, abs=0.1)

    # Every recorded action's base slots are within the per-command clip
    # (NAV_MAX_LIN=0.05, NAV_MAX_ANG=0.1).
    for action in client.actions:
        assert abs(action.action[0]) <= 0.05 + 1e-9
        assert abs(action.action[1]) <= 0.05 + 1e-9
        assert abs(action.action[2]) <= 0.1 + 1e-9
        assert action.repeat == 5  # NAV_REPEAT default


def test_navigate_to_pose_gives_up_after_max_iters(monkeypatch):
    client, shim = _make_shim()
    monkeypatch.setattr(r1pro_module, "NAV_MAX_ITERS", 2)  # too small a budget to converge
    shim.reset()

    ok = shim._navigate_to_pose((10.0, 10.0, 0.0))
    assert ok is False
    assert len(client.actions) == 2


# ---------------------------------------------------------------------------
# _open_close_gripper: convergence + closed-cmd bookkeeping
# ---------------------------------------------------------------------------


def test_open_close_gripper_converges_and_tracks_closed_cmd():
    client, shim = _make_shim()
    shim.reset()

    shim._open_close_gripper(arm=0, max_steps=100, open=False)
    q = shim._q28_now()
    assert q[24] == pytest.approx(0.0, abs=0.005)
    assert q[25] == pytest.approx(0.0, abs=0.005)
    assert shim._gripper_closed_cmd[0] is True

    shim._open_close_gripper(arm=0, max_steps=100, open=True)
    q = shim._q28_now()
    assert q[24] == pytest.approx(0.05, abs=0.005)
    assert q[25] == pytest.approx(0.05, abs=0.005)
    assert shim._gripper_closed_cmd[0] is False


def test_open_close_gripper_right_arm_uses_right_finger_indices():
    client, shim = _make_shim()
    shim.reset()

    shim._open_close_gripper(arm=1, max_steps=100, open=False)
    q = shim._q28_now()
    assert q[26] == pytest.approx(0.0, abs=0.005)
    assert q[27] == pytest.approx(0.0, abs=0.005)
    # Left fingers untouched.
    assert q[24] == pytest.approx(0.0, abs=1e-6)
    assert shim._gripper_closed_cmd[1] is True
    assert shim._gripper_closed_cmd[0] is False


# ---------------------------------------------------------------------------
# check_object_in_hand: 3-case in-hand heuristic
# ---------------------------------------------------------------------------


def test_check_object_in_hand_obstructed_close_returns_true():
    client = FakeEnvClient()
    client.set_min_gap(TASK_ID, 0.02)  # object obstructs full closure
    shim = R1ProGatewayShim(client, TASK_ID, camera_names=CAMERA_NAMES)
    shim.reset()

    shim._open_close_gripper(arm=0, open=False)
    assert shim.check_object_in_hand(arm=0) is True


def test_check_object_in_hand_free_close_returns_false():
    client = FakeEnvClient()  # no obstruction: fingers close fully
    shim = R1ProGatewayShim(client, TASK_ID, camera_names=CAMERA_NAMES)
    shim.reset()

    shim._open_close_gripper(arm=0, open=False)
    assert shim.check_object_in_hand(arm=0) is False


def test_check_object_in_hand_after_open_returns_false():
    client = FakeEnvClient()
    client.set_min_gap(TASK_ID, 0.02)
    shim = R1ProGatewayShim(client, TASK_ID, camera_names=CAMERA_NAMES)
    shim.reset()

    shim._open_close_gripper(arm=0, open=False)
    shim._open_close_gripper(arm=0, open=True)
    assert shim.check_object_in_hand(arm=0) is False


# ---------------------------------------------------------------------------
# _move_to_joint_positions / _execute_motion_plan / update_torso / _lift_arm
# ---------------------------------------------------------------------------


def test_move_to_joint_positions_converges_and_holds_base():
    client, shim = _make_shim(step_chunk=5)
    shim.reset()

    target = shim._q28_now()
    target[6] = -0.5  # torso_joint1
    target[10] = 0.3  # left_arm_joint1

    ok = shim._move_to_joint_positions(target, max_steps=100, settle_steps=1000)
    assert ok is True
    q = shim._q28_now()
    assert q[6] == pytest.approx(-0.5, abs=0.005)
    assert q[10] == pytest.approx(0.3, abs=0.005)
    # Base held at (0, 0, 0) throughout.
    for action in client.actions:
        np.testing.assert_allclose(action.action[0:3], [0.0, 0.0, 0.0], atol=1e-9)


def test_move_to_joint_positions_returns_false_when_budget_exhausted():
    client, shim = _make_shim(step_chunk=1)
    shim.reset()

    target = shim._q28_now()
    target[10] = 5.0  # unreachable in a handful of 0.05-rad steps

    ok = shim._move_to_joint_positions(target, max_steps=3, settle_steps=1000)
    assert ok is False


def test_execute_motion_plan_drives_through_waypoints():
    # A fine interpolation (like `execute_motion_plan`'s own `np.linspace(...,
    # num_steps=20)`) so each waypoint's step is well within the fake model's
    # per-cycle JOINT_STEP budget (`step_chunk` cycles per waypoint RPC).
    client, shim = _make_shim(step_chunk=10)
    shim.reset()

    start = shim._q28_now()
    end = start.copy()
    end[10] = 0.5
    traj = np.linspace(start, end, 50)

    shim._execute_motion_plan(traj)
    assert len(client.actions) == 50
    q = shim._q28_now()
    assert q[10] == pytest.approx(0.5, abs=0.05)


def test_execute_motion_plan_clamps_repeat_to_max_repeat():
    """A ``step_chunk`` above ``MAX_REPEAT`` must not be sent verbatim as
    ``repeat`` (correctness fix: the sibling servo loops all clamp with
    ``min(self._step_chunk, ..., MAX_REPEAT)``; ``_execute_motion_plan`` was
    the one holdout sending ``repeat=self._step_chunk`` unclamped)."""
    client, shim = _make_shim(step_chunk=MAX_REPEAT + 50)
    shim.reset()

    traj = shim._q28_now()[None, :]  # a single waypoint is enough
    shim._execute_motion_plan(traj)

    assert len(client.actions) == 1
    assert client.actions[0].repeat == MAX_REPEAT


def test_update_torso_clamps_to_urdf_limits():
    client, shim = _make_shim(step_chunk=5)
    shim.reset()
    lower, upper = R1ProKinematics().joint_limits("torso_joint2")

    shim.update_torso("torso_joint2", upper + 10.0, max_steps=100)  # way out of range

    q = shim._q28_now()
    assert q[7] == pytest.approx(upper, abs=0.01)


def test_lift_arm_applies_the_legacy_delta_to_either_arm():
    client, shim = _make_shim(step_chunk=5)
    shim.reset()

    shim._lift_arm(arm=0)
    q = shim._q28_now()
    assert q[10] == pytest.approx(-0.4, abs=0.01)

    client2, shim2 = _make_shim(step_chunk=5)
    shim2.reset()
    shim2._lift_arm(arm=1)
    q2 = shim2._q28_now()
    assert q2[11] == pytest.approx(-0.4, abs=0.01)


def test_settle_robot_holds_current_pose():
    client, shim = _make_shim(step_chunk=10)
    shim.reset()
    before = shim._q28_now().copy()

    shim._settle_robot()

    np.testing.assert_allclose(shim._q28_now(), before, atol=1e-6)
    assert sum(a.repeat for a in client.actions) == 50  # SETTLE_CYCLES


def test_settle_robot_cycles_overridable_via_knobs():
    client, shim = _make_shim(step_chunk=10, knobs=R1ProKnobs(settle_cycles=10))
    shim.reset()

    shim._settle_robot()

    assert sum(a.repeat for a in client.actions) == 10


def test_move_to_joint_positions_clamps_commanded_delta_per_rpc():
    """Physics-NaN guard: with max_joint_delta_per_rpc set, no single action
    may command a torso/arm target further than the clamp from live proprio
    (an unclamped far IK solution used to be commanded in one cycle)."""
    client, shim = _make_shim(step_chunk=1, knobs=R1ProKnobs(max_joint_delta_per_rpc=0.3))
    shim.reset()

    target = shim._q28_now()
    target[10] += 2.0  # left_arm_joint1: far jump, as measured live

    shim._move_to_joint_positions(target, max_steps=3, settle_steps=99)

    q0 = 0.0
    for action in client.actions:
        commanded = float(action.action[7])  # left arm joint1 slot
        assert abs(commanded - q0) <= 0.3 + 1e-9
        q0 = commanded


def test_the_per_rpc_clamp_compares_against_the_measured_state_not_the_command():
    """``_q28_now`` used to hand back the cached proprio itself when it already was
    float64, so building a command edited the measured state and the clamp saw no
    difference to limit."""
    client, shim = _make_shim(step_chunk=1, knobs=R1ProKnobs(max_joint_delta_per_rpc=0.3))
    shim.reset()
    shim._latest_proprio_obs = ProprioObs(
        joint_pos=np.zeros(28, dtype=np.float64),
        cartesian_pos=np.zeros(14, dtype=np.float32),
        sim_time=0.0,
    )

    command = shim._q28_now()
    command[10] = 2.0
    shim._step_toward(command, 1)

    assert float(client.actions[0].action[7]) == pytest.approx(0.3)


def test_move_to_joint_positions_unclamped_when_knob_disabled():
    client, shim = _make_shim(step_chunk=1, knobs=R1ProKnobs(max_joint_delta_per_rpc=0))
    shim.reset()

    target = shim._q28_now()
    target[10] += 2.0

    shim._move_to_joint_positions(target, max_steps=1, settle_steps=99)

    assert float(client.actions[0].action[7]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _move_hand: IK mocked via monkeypatch (no pyroki needed)
# ---------------------------------------------------------------------------


def test_move_hand_target_pose_is_xyzw_world_and_ik_gets_base_frame_target(monkeypatch):
    client, shim = _make_shim(step_chunk=5)
    shim.reset()  # base at identity: world frame == base frame here

    recorded_calls = []

    def _stub_solve_ik(target_pos_base, target_wxyz_base, arm, seed_q28):
        recorded_calls.append((np.array(target_pos_base), np.array(target_wxyz_base), arm))
        q28 = np.array(seed_q28, dtype=np.float64).copy()
        q28[10] += 0.2  # left_arm_joint1 (the interleaved q28 layout)
        return q28

    monkeypatch.setattr(shim._kinematics, "solve_ik", _stub_solve_ik)
    # The IK-residual guard FKs the (stubbed) solution; report it exactly on
    # target so this test keeps pinning ONLY the quat-convention handoff.
    monkeypatch.setattr(
        shim._kinematics,
        "fk_eef",
        lambda q28, arm: (np.array([0.5, 0.0, 0.3]), np.array([1.0, 0.0, 0.0, 0.0])),
    )

    target_pos_world = np.array([0.5, 0.0, 0.3])
    target_quat_xyzw_world = np.array([0.0, 0.0, 0.0, 1.0])  # identity
    ok = shim._move_hand((target_pos_world, target_quat_xyzw_world), arm=0)

    assert ok is True
    assert len(recorded_calls) == 1
    called_pos, called_wxyz, called_arm = recorded_calls[0]
    # Base at identity, so world == base frame here.
    np.testing.assert_allclose(called_pos, target_pos_world, atol=1e-9)
    np.testing.assert_allclose(called_wxyz, [1.0, 0.0, 0.0, 0.0], atol=1e-9)
    assert called_arm == 0

    q = shim._q28_now()
    assert q[10] == pytest.approx(0.2, abs=0.05)


def test_move_hand_returns_false_when_ik_target_unreachable_in_time(monkeypatch):
    client, shim = _make_shim(step_chunk=1)
    shim.reset()

    def _stub_solve_ik(target_pos_base, target_wxyz_base, arm, seed_q28):
        q28 = np.array(seed_q28, dtype=np.float64).copy()
        q28[10] += 5.0  # far beyond what 9 waypoints * a few cycles can reach
        return q28

    monkeypatch.setattr(shim._kinematics, "solve_ik", _stub_solve_ik)
    # The IK-residual guard FKs the solution; put it on target so the motion runs
    # and runs out of waypoints, without pyroki.
    monkeypatch.setattr(
        shim._kinematics, "fk_eef", lambda q28, arm: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    )

    ok = shim._move_hand((np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])), arm=0)
    assert ok is False
    assert client.actions


# ---------------------------------------------------------------------------
# _sample_grasp_pose
# ---------------------------------------------------------------------------


class _FakeObb:
    def __init__(self, center, extent):
        self.center = np.asarray(center, dtype=np.float64)
        self.extent = np.asarray(extent, dtype=np.float64)


def test_sample_grasp_pose_top_down_from_obb():
    client, shim = _make_shim()
    shim.reset()
    obb = _FakeObb(center=[1.0, 2.0, 0.5], extent=[0.1, 0.1, 0.2])

    pregrasp_pose, grasp_pose = shim._sample_grasp_pose("can_of_soda", obb)

    grasp_pos, grasp_quat = grasp_pose
    pregrasp_pos, pregrasp_quat = pregrasp_pose
    assert isinstance(grasp_pos, torch.Tensor)
    assert isinstance(grasp_quat, torch.Tensor)
    # grasp_pos == obb.center + [0, 0, extent_z / 2] == (1.0, 2.0, 0.6)
    np.testing.assert_allclose(grasp_pos.numpy(), [1.0, 2.0, 0.6], atol=1e-5)
    # Top-down: pregrasp is offset from grasp along +z (approach_dir == -z with identity quat).
    assert pregrasp_pos[2].item() > grasp_pos[2].item()
    np.testing.assert_allclose(pregrasp_pos[:2].numpy(), grasp_pos[:2].numpy(), atol=1e-5)


# ---------------------------------------------------------------------------
# video surface (dict keyed ego/left_wrist/right_wrist)
# ---------------------------------------------------------------------------


def test_get_video_frames_returns_dict_with_only_populated_cameras():
    client, shim = _make_shim(step_chunk=1)
    shim.reset()
    shim.enable_video_capture(True, wrist_camera=False)

    shim._settle_robot()  # a few raw cycles -> several captured RPCs

    frames = shim.get_video_frames()
    assert set(frames.keys()) == {"ego"}  # wrist not captured (wrist_camera=False)
    assert all(isinstance(f, np.ndarray) for f in frames["ego"])


def test_get_video_frames_wrist_camera_flag_adds_both_wrist_buffers():
    client, shim = _make_shim(step_chunk=1)
    shim.reset()
    shim.enable_video_capture(True, wrist_camera=True)

    shim._settle_robot()

    frames = shim.get_video_frames()
    assert set(frames.keys()) == {"ego", "left_wrist", "right_wrist"}


def test_get_video_frame_count_and_clear():
    client, shim = _make_shim(step_chunk=1)
    shim.reset()
    shim.enable_video_capture(True)
    shim._settle_robot()

    count_before = shim.get_video_frame_count()
    assert count_before > 0

    shim.get_video_frames(clear=True)
    assert shim.get_video_frame_count() == 0


# ---------------------------------------------------------------------------
# no sim imports (acceptance: no omnigibson import anywhere in the 2 modules)
# ---------------------------------------------------------------------------


def test_no_omnigibson_import_in_r1pro_shim_modules():
    """No ``import omnigibson`` / ``from omnigibson ...`` statement (module
    docstrings/comments MAY mention the word, e.g. to document its absence)."""
    import pathlib
    import re

    shims_dir = pathlib.Path(__file__).resolve().parents[2] / "capx" / "submission" / "shims"
    pattern = re.compile(r"^\s*(import|from)\s+omnigibson\b", re.MULTILINE)
    for name in ("r1pro.py", "r1pro_kinematics.py"):
        source = (shims_dir / name).read_text()
        assert not pattern.search(source), f"{name} must never import omnigibson"


def test_no_sim_imports_ast_scan_r1pro_shim_modules():
    import ast
    import pathlib

    shims_dir = pathlib.Path(__file__).resolve().parents[2] / "capx" / "submission" / "shims"
    banned = ("capx.envs.simulators", "robosuite", "libero", "omnigibson")
    for name in ("r1pro.py", "r1pro_kinematics.py"):
        path = shims_dir / name
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            for module_name in names:
                assert not any(module_name.startswith(b) for b in banned), f"{path}: banned import {module_name!r}"


# ---------------------------------------------------------------------------
# IK/FK (require pyroki/JAX -- skipped in the sim-free test venv)
# ---------------------------------------------------------------------------


def test_solve_ik_and_fk_eef_roundtrip():
    pytest.importorskip("pyroki")
    kinematics = R1ProKinematics()
    seed_q28 = np.zeros(28, dtype=np.float64)

    target_pos = np.array([0.3, 0.2, 0.9])
    target_wxyz = np.array([0.0, 0.0, 1.0, 0.0])  # eef-frame "rest" orientation (RotY180)
    q28 = kinematics.solve_ik(target_pos, target_wxyz, arm=0, seed_q28=seed_q28)
    assert q28.shape == (28,)

    fk_pos, fk_wxyz = kinematics.fk_eef(q28, arm=0)
    assert fk_pos.shape == (3,)
    assert fk_wxyz.shape == (4,)
    np.testing.assert_allclose(fk_pos, target_pos, atol=0.05)


def test_solve_ik_reaches_floor_level_target_via_relaxed_torso_weights():
    """A top-down target at floor height (the picking_up_trash soda cans sit
    at z~0.07 in base frame) needs a deep torso bend. With the legacy strict
    torso rest weights alone pyroki leaves a ~0.19 m residual (measured live,
    probe_03); the relaxed-weight retry must bring the solution within a few
    cm so floor grasps are reachable at all."""
    pytest.importorskip("pyroki")
    kinematics = R1ProKinematics()
    seed_q28 = np.zeros(28, dtype=np.float64)

    target_pos = np.array([0.37, 0.0, 0.10])   # base frame, from probe_03's live grasp
    target_wxyz = np.array([0.0, 1.0, 0.0, 0.0])  # top-down (eef frame)
    q28 = kinematics.solve_ik(target_pos, target_wxyz, arm=0, seed_q28=seed_q28)

    fk_pos, _ = kinematics.fk_eef(q28, arm=0)
    assert float(np.linalg.norm(fk_pos - target_pos)) < 0.03


def test_fk_eef_matches_live_sim_relative_eef_pose_at_rest():
    """Pin the eef tool transform against the LIVE sim's own numbers
    (scripts/behavior_ground_truth_dump.py, 2026-07-17): at the reset pose
    (all torso/arm joints ~0) OmniGibson's `robot.get_relative_eef_pose`
    reports pos [-0.0796, 0.2520, 0.6566], quat_xyzw ~[0, 1, 0, 0] for the
    left arm — i.e. the URDF gripper-link FK composed with (RotY180,
    [0, 0, -0.06]). The raw gripper-link FK sits at z 0.7166 (0.06 higher)."""
    pytest.importorskip("pyroki")
    kinematics = R1ProKinematics()
    q28_rest = np.zeros(28, dtype=np.float64)

    fk_pos, fk_wxyz = kinematics.fk_eef(q28_rest, arm=0)
    np.testing.assert_allclose(fk_pos, [-0.0796, 0.2520, 0.6566], atol=2e-3)
    # wxyz RotY180 up to sign
    assert abs(fk_wxyz[2]) == pytest.approx(1.0, abs=1e-3)
    assert abs(fk_wxyz[0]) < 1e-3 and abs(fk_wxyz[1]) < 1e-3 and abs(fk_wxyz[3]) < 1e-3

    fk_pos_r, _ = kinematics.fk_eef(q28_rest, arm=1)
    np.testing.assert_allclose(fk_pos_r, [-0.0796, -0.2520, 0.6566], atol=2e-3)
