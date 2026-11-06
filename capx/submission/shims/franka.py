"""``FrankaGatewayShim``: the single-arm Franka (Robosuite and LIBERO) low-level env."""

from __future__ import annotations

from typing import Any

import numpy as np

from capx.config import load_eval_config
from capx.protocol import FullObs
from capx.protocol.actions import FrankaRawAction
from capx.submission.shims.base import GatewayShimBase, remap_cameras

# LIBERO's wrist camera; Robosuite tasks have none.
_WRIST_CAMERA_NAME = "robot0_eye_in_hand"

_FRANKA_LIMITS = load_eval_config().embodiments.franka


def clamp_to_joint_limits(joints: np.ndarray) -> tuple[np.ndarray, float]:
    """Saturate a joint target at the published limits, as a position controller would.

    The agent's Cartesian APIs get joint targets from an IK solver that treats the
    limits as a cost and returns solutions a hair outside them, which the gateway
    would refuse. The limits come from the same `capx/config/defaults.yaml` the
    gateway checks against.

    Returns the clamped target and how far it had to move.
    """
    target = np.asarray(joints, dtype=np.float64).reshape(7)
    clamped = np.clip(target, _FRANKA_LIMITS.joint_lower, _FRANKA_LIMITS.joint_upper)
    return clamped, float(np.abs(clamped - target).max())


class FrankaGatewayShim(GatewayShimBase):
    """Single-arm Franka shim.

    ``home_joint_position`` is the joints at ``reset()``, read by LIBERO's
    ``goto_home_joint_position``. ``_gripper_fraction`` runs from 1.0 open to 0.0
    closed, the same convention as ``FrankaRawAction.gripper``.
    """

    home_joint_position: np.ndarray | None = None

    def _on_reset(self, obs: FullObs) -> None:
        joints = np.array(obs.proprio.joint_pos[:7], dtype=np.float64)
        # `_step_once` holds this before any move, and a joint resting at a stop can
        # read a hair past it, which the gateway would refuse.
        self._current_joints, _ = clamp_to_joint_limits(joints)
        self.home_joint_position = joints.copy()
        self._gripper_fraction = 1.0

    def _main_camera_name(self) -> str | None:
        for name in self._camera_names:
            if name != _WRIST_CAMERA_NAME and not name.endswith("wrist"):
                return name
        return None

    def _wrist_camera_name(self) -> str | None:
        for name in self._camera_names:
            if name == _WRIST_CAMERA_NAME or name.endswith("wrist"):
                return name
        return None

    def _remap(self, obs: FullObs) -> dict[str, Any]:
        remapped: dict[str, Any] = {
            "robot_joint_pos": np.asarray(obs.proprio.joint_pos, dtype=np.float64),
            "robot_cartesian_pos": np.asarray(obs.proprio.cartesian_pos, dtype=np.float64),
            "sim_time": obs.proprio.sim_time,
        }
        remapped.update(remap_cameras(obs))
        return remapped

    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 100
    ) -> None:
        """Hold the target for up to ``max_steps`` cycles, checking convergence after
        each RPC of ``step_chunk`` cycles."""
        target, overshoot = clamp_to_joint_limits(joints)
        if overshoot > 0.0:
            # Goes to the console the multiturn prompt shows the model, so a target
            # it cannot reach reads as a fact about the arm rather than as silence.
            print(
                f"joint target was outside the published limits by {overshoot:.4f} rad; "
                "clamped to the limit"
            )
        self._current_joints = target
        for chunk in self._chunks(max_steps):
            self._remote_step(
                FrankaRawAction(joints=target, gripper=self._gripper_fraction, repeat=chunk)
            )
            current = np.asarray(self._latest_proprio().joint_pos[:7], dtype=np.float64)
            if np.linalg.norm(current - target) < tolerance:
                return

    def move_to_joints_non_blocking(self, joints: np.ndarray) -> None:
        """Set the target, applied on the next ``_step_once``."""
        self._current_joints, _ = clamp_to_joint_limits(joints)

    def _set_gripper(self, fraction: float) -> None:
        """State-only gripper command; takes effect on the next ``_step_once``."""
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))

    def _step_once(self) -> None:
        """One raw control cycle holding ``_current_joints`` and ``_gripper_fraction``."""
        self._remote_step(
            FrankaRawAction(joints=self._current_joints, gripper=self._gripper_fraction, repeat=1)
        )


__all__ = ["FrankaGatewayShim"]
