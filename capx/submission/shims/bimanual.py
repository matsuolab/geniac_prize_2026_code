"""``BimanualFrankaGatewayShim``: Robosuite's two-arm Franka envs (TwoArmLift, TwoArmHandover).

One step moves both arms, so moving one arm means holding the other: the shim
commands the idle arm's latest measured joints, as the sim's own
``move_to_joints_blocking`` does, clamped to the published limits: an arm resting at
a stop can read a hair past it, and the gateway refuses any target outside them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from capx.protocol import FullObs
from capx.protocol.actions import N_ARMS_BIMANUAL, BimanualFrankaRawAction
from capx.submission.shims.base import GatewayShimBase, remap_cameras
from capx.submission.shims.franka import clamp_to_joint_limits

_MAIN_CAMERA_NAME = "agentview"
# The camera key the reduced Franka APIs read, which the local two-arm sims alias to
# their one render camera.
_API_CAMERA_ALIAS = "robot0_robotview"


class BimanualFrankaGatewayShim(GatewayShimBase):
    """Two-arm Franka shim; ``base_link_wxyz_xyz_0``/``_1`` are each arm's base in the
    world frame, which the bimanual APIs need to relate the two arms."""

    home_joint_position: np.ndarray | None = None
    base_link_wxyz_xyz_0: np.ndarray | None = None
    base_link_wxyz_xyz_1: np.ndarray | None = None

    def _on_reset(self, obs: FullObs) -> None:
        joints = self._measured_joints(obs)
        self._current_joints = joints.copy()
        self.home_joint_position = joints.copy()
        self._gripper_fractions = np.ones(N_ARMS_BIMANUAL, dtype=np.float64)
        self._absorb_base_poses(obs)

    def _main_camera_name(self) -> str | None:
        if _MAIN_CAMERA_NAME in self._camera_names:
            return _MAIN_CAMERA_NAME
        return self._camera_names[0] if self._camera_names else None

    def _wrist_camera_name(self) -> str | None:
        return None

    def _remap(self, obs: FullObs) -> dict[str, Any]:
        self._absorb_base_poses(obs)
        remapped: dict[str, Any] = {
            "robot_joint_pos": np.asarray(obs.proprio.joint_pos, dtype=np.float64),
            "robot_cartesian_pos": np.asarray(obs.proprio.cartesian_pos, dtype=np.float64),
            "sim_time": obs.proprio.sim_time,
        }
        for index, arm in enumerate(obs.proprio.arms):
            remapped[f"robot{index}_joint_pos"] = np.asarray(arm.joint_pos, dtype=np.float64)
            remapped[f"robot{index}_cartesian_pos"] = np.asarray(
                arm.cartesian_pos, dtype=np.float64
            )
        cameras = remap_cameras(obs)
        remapped.update(cameras)
        main = self._main_camera_name()
        if main in cameras and _API_CAMERA_ALIAS not in cameras:
            remapped[_API_CAMERA_ALIAS] = cameras[main]
        return remapped

    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 100
    ) -> None:
        self._servo({0: joints}, tolerance=tolerance, max_steps=max_steps)

    def move_to_joints_blocking_arm1(
        self, joints: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 100
    ) -> None:
        self._servo({1: joints}, tolerance=tolerance, max_steps=max_steps)

    def move_to_joints_blocking_both(
        self,
        joints0: np.ndarray,
        joints1: np.ndarray,
        *,
        tolerance: float = 0.02,
        max_steps: int = 100,
    ) -> None:
        self._servo({0: joints0, 1: joints1}, tolerance=tolerance, max_steps=max_steps)

    def _set_gripper(self, fraction: float) -> None:
        self._gripper_fractions[0] = float(np.clip(fraction, 0.0, 1.0))

    def _set_gripper_arm1(self, fraction: float) -> None:
        self._gripper_fractions[1] = float(np.clip(fraction, 0.0, 1.0))

    def _step_once(self) -> None:
        """One raw control cycle holding both arms where they are, applying the gripper targets."""
        self._remote_step(self._action(self._measured_joints(), repeat=1))

    def _servo(
        self, targets: dict[int, np.ndarray], *, tolerance: float, max_steps: int
    ) -> None:
        """Drive the named arms to their targets while the others hold.

        Convergence is checked on the moving arms only: the holding arm is already
        where it is told to be.
        """
        commanded = self._measured_joints()
        moving = sorted(targets)
        for arm, joints in targets.items():
            commanded[arm], overshoot = clamp_to_joint_limits(joints)
            if overshoot > 0.0:
                print(
                    f"arm {arm} joint target was outside the published limits by "
                    f"{overshoot:.4f} rad; clamped to the limit"
                )
        self._current_joints = commanded.copy()

        for chunk in self._chunks(max_steps):
            self._remote_step(self._action(commanded, repeat=chunk))
            measured = self._measured_joints()
            if all(
                np.linalg.norm(measured[arm] - commanded[arm]) < tolerance for arm in moving
            ):
                return

    def _action(self, joints: np.ndarray, *, repeat: int) -> BimanualFrankaRawAction:
        clamped = np.stack([clamp_to_joint_limits(arm)[0] for arm in joints])
        return BimanualFrankaRawAction(
            joints=clamped, grippers=self._gripper_fractions.copy(), repeat=repeat
        )

    def _measured_joints(self, obs: FullObs | None = None) -> np.ndarray:
        arms = (obs.proprio if obs is not None else self._latest_proprio()).arms
        if len(arms) != N_ARMS_BIMANUAL:
            raise RuntimeError(
                f"expected {N_ARMS_BIMANUAL} arms in proprio, got {len(arms)}; "
                "is this task's embodiment franka_bimanual?"
            )
        return np.stack([np.asarray(arm.joint_pos[:7], dtype=np.float64) for arm in arms])

    def _absorb_base_poses(self, obs: FullObs) -> None:
        arms = obs.proprio.arms
        if len(arms) == N_ARMS_BIMANUAL:
            self.base_link_wxyz_xyz_0 = np.asarray(arms[0].base_pose, dtype=np.float64)
            self.base_link_wxyz_xyz_1 = np.asarray(arms[1].base_pose, dtype=np.float64)


__all__ = ["BimanualFrankaGatewayShim"]
