"""``R1ProGatewayShim``: the mobile bimanual R1Pro (BEHAVIOR) low-level env.

Every motion primitive sends the 25-dim ``R1ProRawAction`` through
``GatewayShimBase._remote_step``; the ``robot``/``controller``/``env`` views below
present the surface ``R1ProControlApi`` reads on its ``self._env``. Poses handed to
and from ``R1ProControlApi`` are ``(position, quaternion_xyzw)`` in the world frame.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, NoReturn

import numpy as np
import torch
import viser.transforms as vtf
from scipy.spatial.transform import Rotation as SciRotation

from capx.gateway.r1pro_adapter import _CAMERA_SENSOR_SUFFIX
from capx.protocol import EnvClient, FullObs
from capx.protocol.actions import MAX_REPEAT, R1ProRawAction
from capx.submission.interfaces import STEP_CHUNK_DEFAULT, R1ProKnobs
from capx.submission.shims.base import GatewayShimBase
from capx.submission.shims.r1pro_kinematics import R1ProKinematics, _wrap_to_pi

ROBOT_KEY = "robot_gateway"

# q28 index of each named joint: torso 6-9, arms interleaved (left even slots
# 10..22, right odd slots 11..23), fingers 24-27.
_TORSO_Q28_INDEX = {"torso_joint1": 6, "torso_joint2": 7, "torso_joint3": 8, "torso_joint4": 9}
_LEFT_ARM_Q28_INDEX = {f"left_arm_joint{i + 1}": 10 + 2 * i for i in range(7)}
_RIGHT_ARM_Q28_INDEX = {f"right_arm_joint{i + 1}": 11 + 2 * i for i in range(7)}
_GRIPPER_Q28_INDEX = {
    "left_gripper_finger_joint1": 24,
    "left_gripper_finger_joint2": 25,
    "right_gripper_finger_joint1": 26,
    "right_gripper_finger_joint2": 27,
}
_JOINT_Q28_INDEX: dict[str, int] = {
    **_TORSO_Q28_INDEX,
    **_LEFT_ARM_Q28_INDEX,
    **_RIGHT_ARM_Q28_INDEX,
    **_GRIPPER_Q28_INDEX,
}
_LEFT_ARM_Q28_SLICE = np.arange(10, 24, 2)
_RIGHT_ARM_Q28_SLICE = np.arange(11, 24, 2)

NAV_REPEAT = 5
NAV_MAX_ITERS = 400
NAV_LIN_TOL = 0.05
NAV_ANG_TOL = 0.1
MOVE_HAND_WAYPOINTS = 10
MOVE_HAND_REPEAT = 3
MOVE_HAND_TOLERANCE = 0.05
LIFT_ARM_DELTA = 0.4
SERVO_TOL = 0.005
# Per-finger `robot.eef_to_fingertip_lengths` of the R1Pro model, which the client
# cannot read from the sim.
_AVG_FINGER_OFFSET = 0.0188


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _as_np28(value: Any) -> np.ndarray:
    return _to_numpy(value).reshape(28)


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _se3(pos: np.ndarray, quat_xyzw: np.ndarray) -> vtf.SE3:
    return vtf.SE3.from_rotation_and_translation(
        rotation=vtf.SO3(wxyz=_xyzw_to_wxyz(quat_xyzw)), translation=pos
    )


def _pose_base_to_world(
    base_pos: np.ndarray, base_quat_xyzw: np.ndarray, local_pos: np.ndarray, local_quat_xyzw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    world_se3 = _se3(base_pos, base_quat_xyzw) @ _se3(local_pos, local_quat_xyzw)
    return world_se3.translation(), _wxyz_to_xyzw(world_se3.rotation().wxyz)


def _pose_world_to_base(
    base_pos: np.ndarray, base_quat_xyzw: np.ndarray, world_pos: np.ndarray, world_quat_xyzw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    local_se3 = _se3(base_pos, base_quat_xyzw).inverse() @ _se3(world_pos, world_quat_xyzw)
    return local_se3.translation(), _wxyz_to_xyzw(local_se3.rotation().wxyz)


def _torso_and_arm_q28_index(arm: int) -> np.ndarray:
    arm_slice = _LEFT_ARM_Q28_SLICE if arm == 0 else _RIGHT_ARM_Q28_SLICE
    return np.concatenate([np.arange(6, 10), arm_slice])


class _Unavailable:
    """Stands in for the raw sim env and its external camera, which the client cannot reach."""

    def __getattr__(self, name: str) -> NoReturn:
        raise RuntimeError("external camera / raw sim env is not available on the gateway client: ." + name)


_ENV_UNAVAILABLE = _Unavailable()


class R1ProSensorView:
    """One camera as an OmniGibson sensor: the wire's base-frame camera pose composed
    with the live base pose gives the world pose ``get_camera_pose`` needs."""

    def __init__(self, intrinsics: np.ndarray, cam_pose_base: np.ndarray, robot_view: R1ProRobotView) -> None:
        self.intrinsic_matrix = torch.tensor(np.asarray(intrinsics, dtype=np.float32), dtype=torch.float32)
        pose = np.asarray(cam_pose_base, dtype=np.float64)
        self._cam_pos_base = pose[0:3]
        self._cam_quat_xyzw_base = _wxyz_to_xyzw(pose[3:7])
        self._robot_view = robot_view

    def get_position_orientation(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_pos, base_quat_xyzw = self._robot_view.get_position_orientation()
        world_pos, world_quat_xyzw = _pose_base_to_world(
            _to_numpy(base_pos), _to_numpy(base_quat_xyzw), self._cam_pos_base, self._cam_quat_xyzw_base
        )
        return torch.tensor(world_pos, dtype=torch.float32), torch.tensor(world_quat_xyzw, dtype=torch.float32)


class R1ProRobotView:
    """The ``robot`` object: base pose, joint vector, arm names and camera sensors.

    ``sensors`` is rebuilt by every ``R1ProGatewayShim._remap``.
    """

    arm_names = ["left", "right"]

    def __init__(self, q28_getter: Callable[[], np.ndarray], shim: R1ProGatewayShim) -> None:
        self._q28_getter = q28_getter
        self._shim = shim
        self.sensors: dict[str, R1ProSensorView] = {}

    def get_position_orientation(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World base pose from ``joint_pos[0:6]``, read as translation plus intrinsic XYZ euler."""
        q = self._q28_getter()
        pos = torch.tensor(q[0:3], dtype=torch.float32)
        quat_xyzw = SciRotation.from_euler("XYZ", q[3:6]).as_quat()
        return pos, torch.tensor(quat_xyzw, dtype=torch.float32)

    def get_joint_positions(self) -> torch.Tensor:
        return torch.tensor(self._q28_getter(), dtype=torch.float32)

    def get_relative_eef_pose(self, arm: str | int = "left") -> tuple[torch.Tensor, torch.Tensor]:
        """``(pos, quat_xyzw)`` in the base frame. ``R1ProControlApi`` passes the arm name."""
        arm_index = arm if isinstance(arm, int) else self.arm_names.index(arm)
        return self._shim._relative_eef_pose(arm_index)


class JointView:
    """An OmniGibson ``JointPrim``; ``R1ProControlApi`` only reads ``get_state()[0][0]``."""

    def __init__(self, name: str, q28_getter: Callable[[], np.ndarray]) -> None:
        self._name = name
        self._q28_getter = q28_getter

    def get_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        position = torch.tensor([float(self._q28_getter()[_JOINT_Q28_INDEX[self._name]])], dtype=torch.float32)
        zero = torch.zeros(1, dtype=torch.float32)
        return position, zero, zero


class R1ProControllerView:
    """The ``controller`` object; only ``.robot.joints[name]`` is read."""

    def __init__(self, q28_getter: Callable[[], np.ndarray]) -> None:
        self.robot = SimpleNamespace(joints={name: JointView(name, q28_getter) for name in _JOINT_Q28_INDEX})


class R1ProGatewayShim(GatewayShimBase):
    """R1Pro shim. Every primitive but ``_navigate_to_pose`` holds the base still."""

    def __init__(
        self,
        client: EnvClient,
        task_id: str,
        *,
        camera_names: list[str],
        step_chunk: int = STEP_CHUNK_DEFAULT,
        knobs: R1ProKnobs | None = None,
    ) -> None:
        super().__init__(client, task_id, camera_names=camera_names, step_chunk=step_chunk)
        self._knobs = knobs or R1ProKnobs()

        self._kinematics = R1ProKinematics()
        self._gripper_closed_cmd: dict[int, bool] = {0: False, 1: False}
        self._camera_frames: dict[str, list[np.ndarray]] = {name: [] for name in _CAMERA_SENSOR_SUFFIX}

        self.robot = R1ProRobotView(self._q28_now, self)
        self.controller = R1ProControllerView(self._q28_now)
        lower, upper = self._kinematics.joint_limits("torso_joint1")
        self.torso_joint_limits = [torch.tensor(lower), torch.tensor(upper)]

    @property
    def env(self) -> _Unavailable:
        return _ENV_UNAVAILABLE

    def _on_reset(self, obs: FullObs) -> None:
        self._gripper_closed_cmd = {0: False, 1: False}
        self._camera_frames = {name: [] for name in _CAMERA_SENSOR_SUFFIX}

    def _main_camera_name(self) -> str | None:
        return "ego" if "ego" in self._camera_names else None

    def _wrist_camera_name(self) -> str | None:
        return "left_wrist" if "left_wrist" in self._camera_names else None

    def _remap(self, obs: FullObs) -> dict[str, Any]:
        """The OmniGibson-shaped obs dict under ``ROBOT_KEY``; also refreshes ``robot.sensors``."""
        sensors: dict[str, R1ProSensorView] = {}
        robot_dict: dict[str, Any] = {}
        for wire_name, suffix in _CAMERA_SENSOR_SUFFIX.items():
            cam = obs.cameras.get(wire_name)
            if cam is None:
                continue
            sensor_key = f"{ROBOT_KEY}:{suffix}"
            images: dict[str, torch.Tensor] = {"rgb": torch.from_numpy(np.asarray(cam.rgb, dtype=np.uint8))}
            if cam.depth is not None:
                depth = np.asarray(cam.depth, dtype=np.float32)
                if depth.ndim == 3:
                    depth = depth[..., 0]
                images["depth_linear"] = torch.from_numpy(depth)
            robot_dict[sensor_key] = images
            sensors[sensor_key] = R1ProSensorView(
                intrinsics=np.asarray(cam.intrinsics, dtype=np.float32),
                cam_pose_base=np.asarray(cam.pose, dtype=np.float64),
                robot_view=self.robot,
            )
        self.robot.sensors = sensors
        return {ROBOT_KEY: robot_dict}

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True, wrist_camera: bool = False) -> None:
        super().enable_video_capture(enabled, clear=clear, wrist_camera=wrist_camera)
        if clear:
            self._camera_frames = {name: [] for name in _CAMERA_SENSOR_SUFFIX}

    def _append_video_frames(self, obs: FullObs) -> None:
        ego = obs.cameras.get("ego")
        if ego is not None:
            self._camera_frames.setdefault("ego", []).append(ego.rgb.copy())
        if self._capture_wrist:
            for name in ("left_wrist", "right_wrist"):
                cam = obs.cameras.get(name)
                if cam is not None:
                    self._camera_frames.setdefault(name, []).append(cam.rgb.copy())

    def get_video_frames(self, *, clear: bool = False) -> dict[str, list[np.ndarray]]:
        """Frames per wire camera name: R1Pro has three cameras and no single main one."""
        frames = {name: list(buf) for name, buf in self._camera_frames.items() if buf}
        if clear:
            self._camera_frames = {name: [] for name in _CAMERA_SENSOR_SUFFIX}
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._camera_frames.get("ego", []))

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._camera_frames.get("ego", [])[start:end]]

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = list(self._camera_frames.get("left_wrist", []))
        if clear:
            self._camera_frames["left_wrist"] = []
        return frames

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._camera_frames.get("left_wrist", [])[start:end]]

    def _q28_now(self) -> np.ndarray:
        """A copy: the motion code edits it into a command, and the cached obs must keep
        the measured state ``_clamp_towards`` compares against."""
        return np.array(self._latest_proprio().joint_pos, dtype=np.float64)

    def _base_pose_now(self) -> tuple[float, float, float]:
        q = self._q28_now()
        return float(q[0]), float(q[1]), float(q[5])

    def get_joint_positions(self) -> torch.Tensor:
        return torch.tensor(self._q28_now(), dtype=torch.float32)

    def _fk_eef_base(self, arm: int) -> tuple[np.ndarray, np.ndarray]:
        pos_base, quat_wxyz_base = self._kinematics.fk_eef(self._q28_now(), arm)
        return pos_base, _wxyz_to_xyzw(quat_wxyz_base)

    def get_robot_eef_pose(self, arm: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """World ``(pos, quat_xyzw)`` from client-side FK."""
        pos_base, quat_xyzw_base = self._fk_eef_base(arm)
        base_pos, base_quat_xyzw = self.robot.get_position_orientation()
        world_pos, world_quat_xyzw = _pose_base_to_world(
            _to_numpy(base_pos), _to_numpy(base_quat_xyzw), pos_base, quat_xyzw_base
        )
        return torch.tensor(world_pos, dtype=torch.float32), torch.tensor(world_quat_xyzw, dtype=torch.float32)

    def _relative_eef_pose(self, arm: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos_base, quat_xyzw_base = self._fk_eef_base(arm)
        return torch.tensor(pos_base, dtype=torch.float32), torch.tensor(quat_xyzw_base, dtype=torch.float32)

    def check_object_in_hand(self, arm: int = 0) -> bool:
        """After a settle, a closed gripper whose fingers stay apart is holding something."""
        self._settle_robot()
        q = self._q28_now()
        i0, i1 = (24, 25) if arm == 0 else (26, 27)
        gap = float(q[i0] + q[i1])
        return bool(self._gripper_closed_cmd.get(arm, False) and gap > self._knobs.in_hand_gap_threshold)

    def _sample_grasp_pose(
        self, obj_name: str, object_obb: Any
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Top-down grasp at the top of the box, with the pregrasp straight above it.

        Returns ``(pregrasp_pose, grasp_pose)``, each ``(position, quaternion_xyzw)``.
        """
        del obj_name
        center = torch.tensor(np.asarray(object_obb.center), dtype=torch.float32)
        extent = torch.tensor(np.asarray(object_obb.extent), dtype=torch.float32)
        grasp_pos = center + torch.tensor([0.0, 0.0, float(extent[2]) / 2.0])
        pregrasp_pos = grasp_pos + torch.tensor([0.0, 0.0, _AVG_FINGER_OFFSET + 0.25])
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        return (pregrasp_pos, quat), (grasp_pos, quat)

    def _clamp_towards(self, q28_cmd: np.ndarray) -> np.ndarray:
        """Limit how far the torso/arm slots lead the live proprio; ``<= 0`` disables."""
        limit = self._knobs.max_joint_delta_per_rpc
        if limit <= 0:
            return q28_cmd
        q_now = self._q28_now()
        clamped = q28_cmd.copy()
        delta = np.clip(q28_cmd[6:24] - q_now[6:24], -limit, limit)
        clamped[6:24] = q_now[6:24] + delta
        return clamped

    def _step_toward(self, q28_cmd: np.ndarray, repeat: int) -> None:
        """Hold the base and command ``q28_cmd`` (clamped) for ``repeat`` cycles."""
        action = self._kinematics.q28_to_action25(self._clamp_towards(q28_cmd), self._base_pose_now())
        self._remote_step(R1ProRawAction(action=action, repeat=repeat))

    def _settle_robot(self) -> None:
        """Hold still for ``settle_cycles``; joint velocity is not on the wire to check instead."""
        action = self._kinematics.q28_to_action25(self._q28_now(), self._base_pose_now())
        for chunk in self._chunks(self._knobs.settle_cycles):
            self._remote_step(R1ProRawAction(action=action, repeat=chunk))

    def _move_to_joint_positions(self, target_joint_positions: Any, max_steps: int = 20, settle_steps: int = 10) -> bool:
        """Servo torso, arms and fingers to a q28 target; the base holds where it is."""
        target = _as_np28(target_joint_positions)
        settle_period = max(1, int(settle_steps))
        cycles_since_settle = 0
        for chunk in self._chunks(int(max_steps)):
            q28_cmd = self._q28_now()
            q28_cmd[6:28] = target[6:28]
            self._step_toward(q28_cmd, chunk)
            cycles_since_settle += chunk
            if cycles_since_settle >= settle_period:
                self._settle_robot()
                cycles_since_settle = 0
            if np.allclose(self._q28_now()[6:28], target[6:28], atol=SERVO_TOL):
                return True
        return bool(np.allclose(self._q28_now()[6:28], target[6:28], atol=SERVO_TOL))

    def _execute_motion_plan(self, q_traj: Any) -> None:
        """Play a joint trajectory, one q28 row per RPC."""
        traj = _to_numpy(q_traj)
        if traj.ndim == 1:
            traj = traj[None, :]
        for row in traj:
            q28_cmd = self._q28_now()
            q28_cmd[6:28] = row[6:28]
            self._step_toward(q28_cmd, min(self._step_chunk, MAX_REPEAT))

    def _open_close_gripper(self, arm: int = 0, max_steps: int = 100, open: bool = True) -> None:
        """Drive both fingers to their URDF upper (open) or lower (closed) limit."""
        name0 = "left_gripper_finger_joint1" if arm == 0 else "right_gripper_finger_joint1"
        name1 = "left_gripper_finger_joint2" if arm == 0 else "right_gripper_finger_joint2"
        i0, i1 = _JOINT_Q28_INDEX[name0], _JOINT_Q28_INDEX[name1]
        lower0, upper0 = self._kinematics.joint_limits(name0)
        lower1, upper1 = self._kinematics.joint_limits(name1)
        target0 = upper0 if open else lower0
        target1 = upper1 if open else lower1

        for chunk in self._chunks(int(max_steps)):
            q28_cmd = self._q28_now()
            q28_cmd[i0] = target0
            q28_cmd[i1] = target1
            action = self._kinematics.q28_to_action25(q28_cmd, self._base_pose_now())
            self._remote_step(R1ProRawAction(action=action, repeat=chunk))
            q = self._q28_now()
            if abs(q[i0] - target0) < SERVO_TOL and abs(q[i1] - target1) < SERVO_TOL:
                break

        self._gripper_closed_cmd[arm] = not open

    def _lift_arm(self, arm: int = 0) -> None:
        """Raise the arm by ``LIFT_ARM_DELTA`` on its first joint, without a mid-servo settle."""
        target = self._q28_now()
        target[10 if arm == 0 else 11] -= LIFT_ARM_DELTA
        self._move_to_joint_positions(target, max_steps=20, settle_steps=21)

    def update_torso(self, torso_joint_name: str, angle: Any, max_steps: int = 50) -> None:
        idx = _TORSO_Q28_INDEX[torso_joint_name]
        lower, upper = self._kinematics.joint_limits(torso_joint_name)
        angle_val = float(angle.item()) if isinstance(angle, torch.Tensor) else float(angle)
        target = self._q28_now()
        target[idx] = float(np.clip(angle_val, lower, upper))
        self._move_to_joint_positions(target, max_steps=max_steps, settle_steps=max_steps + 1)

    def _navigate_to_pose(self, pose_2d: Any) -> bool:
        """Straight-line base servo with no obstacle avoidance; other joints hold."""
        goal_x, goal_y, goal_yaw = (float(v) for v in pose_2d)
        # The base command is a local delta re-applied every cycle it is held, so a
        # command held for `nav_repeat` cycles is scaled down to keep each RPC's
        # displacement within the clip.
        per_cycle_scale = 1.0 / NAV_REPEAT
        max_lin, max_ang = self._knobs.nav_max_lin, self._knobs.nav_max_ang
        for _ in range(NAV_MAX_ITERS):
            if self._nav_converged(goal_x, goal_y, goal_yaw):
                return True
            q = self._q28_now()
            x, y, yaw = float(q[0]), float(q[1]), float(q[5])
            dx_world, dy_world = goal_x - x, goal_y - y
            dyaw = _wrap_to_pi(goal_yaw - yaw)
            cos_neg_yaw, sin_neg_yaw = math.cos(-yaw), math.sin(-yaw)
            dx_local = cos_neg_yaw * dx_world - sin_neg_yaw * dy_world
            dy_local = sin_neg_yaw * dx_world + cos_neg_yaw * dy_world
            dx_cmd = float(np.clip(dx_local, -max_lin, max_lin)) * per_cycle_scale
            dy_cmd = float(np.clip(dy_local, -max_lin, max_lin)) * per_cycle_scale
            dyaw_cmd = float(np.clip(dyaw, -max_ang, max_ang)) * per_cycle_scale

            action = self._kinematics.q28_to_action25(q, (x, y, yaw))
            action[0], action[1], action[2] = dx_cmd, dy_cmd, dyaw_cmd
            self._remote_step(R1ProRawAction(action=action, repeat=NAV_REPEAT))
        return self._nav_converged(goal_x, goal_y, goal_yaw)

    def _nav_converged(self, goal_x: float, goal_y: float, goal_yaw: float) -> bool:
        q = self._q28_now()
        lin_err = math.hypot(goal_x - float(q[0]), goal_y - float(q[1]))
        ang_err = abs(_wrap_to_pi(goal_yaw - float(q[5])))
        return lin_err < NAV_LIN_TOL and ang_err < NAV_ANG_TOL

    def _move_hand(
        self,
        target_pose: tuple[Any, Any],
        arm: int = 0,
        ignore_all_obstacles: bool = True,
        skip_obstacle_update: bool = True,
    ) -> bool:
        """IK to a world ``(position, quaternion_xyzw)`` target, then interpolate there in
        joint space. Obstacles are always ignored; the two flags exist for the signature.
        """
        target_pos_world = _to_numpy(target_pose[0]).reshape(3)
        target_quat_xyzw_world = _to_numpy(target_pose[1]).reshape(4)

        base_pos, base_quat_xyzw = self.robot.get_position_orientation()
        target_pos_base, target_quat_xyzw_base = _pose_world_to_base(
            _to_numpy(base_pos), _to_numpy(base_quat_xyzw), target_pos_world, target_quat_xyzw_world
        )
        target_wxyz_base = _xyzw_to_wxyz(target_quat_xyzw_base)

        seed_q28 = self._q28_now()
        q28_target = self._kinematics.solve_ik(target_pos_base, target_wxyz_base, arm, seed_q28)

        abort_at = self._knobs.ik_residual_abort
        if abort_at > 0:
            fk_pos_base, _ = self._kinematics.fk_eef(q28_target, arm)
            residual = float(np.linalg.norm(fk_pos_base - np.asarray(target_pos_base, dtype=np.float64)))
            if residual > abort_at:
                print(
                    f"_move_hand: IK cannot reach target (residual {residual:.3f} m > "
                    f"{abort_at:.3f} m); not executing"
                )
                return False

        # Enough waypoints that no step leads the proprio by more than the per-RPC clamp.
        n_waypoints = MOVE_HAND_WAYPOINTS
        limit = self._knobs.max_joint_delta_per_rpc
        if limit > 0:
            max_dq = float(np.max(np.abs(q28_target[6:24] - seed_q28[6:24])))
            n_waypoints = max(n_waypoints, int(math.ceil(max_dq / limit)) + 1)

        for q28_wp in np.linspace(seed_q28, q28_target, n_waypoints)[1:]:
            q28_cmd = self._q28_now()
            q28_cmd[6:24] = q28_wp[6:24]
            self._step_toward(q28_cmd, MOVE_HAND_REPEAT)

        solved_index = _torso_and_arm_q28_index(arm)
        current = self._q28_now()
        return bool(np.allclose(current[solved_index], q28_target[solved_index], atol=MOVE_HAND_TOLERANCE))


__all__ = [
    "R1ProGatewayShim",
    "R1ProRobotView",
    "R1ProSensorView",
    "R1ProControllerView",
    "JointView",
]
