"""Embodiment-specific raw actions — the ``step`` payload.

One ``step`` is one simulator control cycle; the servo loop, IK and every other
primitive live on the submission side, so the gateway only validates shape and
bounds. Validation lives here so the client (fail fast) and the gateway (refuse
with ``invalid_action``) share one source of truth.

- ``repeat``: hold this command for N control cycles; only the final obs returns.
- ``include_cameras``: return the full obs tier from this ``step`` instead of the
  proprio tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class InvalidActionError(ValueError):
    """Raised when an action violates its embodiment schema (-> ``invalid_action``)."""


@dataclass
class RawAction:
    """Base raw action. ``embodiment`` selects the concrete schema on decode."""

    embodiment: str
    repeat: int = 1
    include_cameras: bool = False

    def validate(self) -> None:
        repeat = _integral_repeat(self.repeat)
        if repeat < 1 or repeat > MAX_REPEAT:
            raise InvalidActionError(f"repeat must be in [1, {MAX_REPEAT}], got {repeat}")

    def to_wire(self) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def from_wire(msg: dict[str, Any]) -> RawAction:
        raise NotImplementedError


MAX_REPEAT = 200

_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _integral_repeat(value: object) -> int:
    if isinstance(value, int | np.integer) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float | np.floating) and np.isfinite(value) and value == int(value):
        return int(value)
    raise InvalidActionError(f"repeat must be an integer, got {value!r}")


@dataclass
class FrankaRawAction(RawAction):
    """Single-arm Franka (Robosuite / LIBERO).

    Maps 1:1 onto ``RobosuiteBaseEnv`` control state: ``joints`` -> the joint
    position target, ``gripper`` -> open(1.0)/closed(0.0) fraction. One
    ``step`` == one ``_step_once`` (``capx/envs/simulators/robosuite_base.py``).
    """

    embodiment: str = "franka"
    joints: np.ndarray | None = None  # (7,) target joint positions (rad)
    gripper: float = 1.0  # 1.0 open .. 0.0 closed

    def validate(self) -> None:
        super().validate()
        if self.joints is None:
            raise InvalidActionError("franka.joints is required")
        j = np.asarray(self.joints, dtype=np.float64)
        if j.shape != (7,):
            raise InvalidActionError(f"franka.joints must be shape (7,), got {j.shape}")
        if not np.all(np.isfinite(j)):
            raise InvalidActionError("franka.joints contains non-finite values")
        if not (0.0 <= float(self.gripper) <= 1.0):
            raise InvalidActionError(f"franka.gripper must be in [0, 1], got {self.gripper}")

    def to_wire(self) -> dict[str, Any]:
        # float64 on the wire: a float32 round-trip can nudge a joint validated
        # exactly at a limit to just outside it, turning a client-valid action
        # into a server-side invalid_action.
        return {
            "embodiment": self.embodiment,
            "joints": np.asarray(self.joints, dtype=np.float64),
            "gripper": float(self.gripper),
            "repeat": self.repeat,
            "include_cameras": self.include_cameras,
        }

    @staticmethod
    def from_wire(msg: dict[str, Any]) -> FrankaRawAction:
        return FrankaRawAction(
            joints=np.asarray(msg["joints"], dtype=np.float64),
            gripper=float(msg["gripper"]),
            repeat=_integral_repeat(msg.get("repeat", 1)),
            include_cameras=bool(msg.get("include_cameras", False)),
        )


# Arms are indexed, not named: Robosuite's two-arm envs call them robot0 and
# robot1, and which one a task treats as "left" is the task's business.
N_ARMS_BIMANUAL = 2


@dataclass
class BimanualFrankaRawAction(RawAction):
    """Two-arm Franka (Robosuite ``TwoArmLift`` / ``TwoArmHandover``).

    One ``step`` is one control cycle for BOTH arms, which is what the sim's own
    action is: ``[joints0, gripper0, joints1, gripper1]`` in a single
    ``robosuite_env.step``. Sending the arms separately would be a different
    physics — one arm holding still for a cycle while the other moves — so the
    action carries both or neither.

    ``joints`` is (2, 7) and ``grippers`` is (2,), indexed by arm.
    """

    embodiment: str = "franka_bimanual"
    joints: np.ndarray | None = None  # (2, 7) target joint positions (rad)
    grippers: np.ndarray | None = None  # (2,) 1.0 open .. 0.0 closed

    def validate(self) -> None:
        super().validate()
        if self.joints is None:
            raise InvalidActionError("franka_bimanual.joints is required")
        j = np.asarray(self.joints, dtype=np.float64)
        if j.shape != (N_ARMS_BIMANUAL, 7):
            raise InvalidActionError(
                f"franka_bimanual.joints must be shape ({N_ARMS_BIMANUAL}, 7), got {j.shape}"
            )
        if not np.all(np.isfinite(j)):
            raise InvalidActionError("franka_bimanual.joints contains non-finite values")
        if self.grippers is None:
            raise InvalidActionError("franka_bimanual.grippers is required")
        g = np.asarray(self.grippers, dtype=np.float64)
        if g.shape != (N_ARMS_BIMANUAL,):
            raise InvalidActionError(
                f"franka_bimanual.grippers must be shape ({N_ARMS_BIMANUAL},), got {g.shape}"
            )
        if not np.all(np.isfinite(g)) or np.any(g < 0.0) or np.any(g > 1.0):
            raise InvalidActionError("franka_bimanual.grippers must all be in [0, 1]")

    def to_wire(self) -> dict[str, Any]:
        # float64 for the same reason the single-arm action uses it: a float32
        # round-trip can nudge a joint validated exactly at a limit past it.
        return {
            "embodiment": self.embodiment,
            "joints": np.asarray(self.joints, dtype=np.float64),
            "grippers": np.asarray(self.grippers, dtype=np.float64),
            "repeat": self.repeat,
            "include_cameras": self.include_cameras,
        }

    @staticmethod
    def from_wire(msg: dict[str, Any]) -> BimanualFrankaRawAction:
        return BimanualFrankaRawAction(
            joints=np.asarray(msg["joints"], dtype=np.float64),
            grippers=np.asarray(msg["grippers"], dtype=np.float64),
            repeat=_integral_repeat(msg.get("repeat", 1)),
            include_cameras=bool(msg.get("include_cameras", False)),
        )


@dataclass
class R1ProRawAction(RawAction):
    """R1Pro mobile bimanual (BEHAVIOR / Isaac Sim via OmniGibson).

    The raw action is the OmniGibson controller vector for the robot; one
    ``step`` == one ``env.step(action)``. Its dimension is checked by the gateway
    against the configured ``embodiments.r1pro.action_dim``.
    """

    embodiment: str = "r1pro"
    action: np.ndarray | None = None  # (action_dim,) OmniGibson control vector

    def validate(self) -> None:
        super().validate()
        if self.action is None:
            raise InvalidActionError("r1pro.action is required")
        a = np.asarray(self.action, dtype=np.float64)
        if a.ndim != 1:
            raise InvalidActionError(f"r1pro.action must be 1-D, got shape {a.shape}")
        if not np.all(np.isfinite(a)):
            raise InvalidActionError("r1pro.action contains non-finite values")
        # The backend link carries the action as float32.
        if np.any(np.abs(a) > _FLOAT32_MAX):
            raise InvalidActionError("r1pro.action has values outside the float32 range")

    def to_wire(self) -> dict[str, Any]:
        return {
            "embodiment": self.embodiment,
            "action": np.asarray(self.action, dtype=np.float32),
            "repeat": self.repeat,
            "include_cameras": self.include_cameras,
        }

    @staticmethod
    def from_wire(msg: dict[str, Any]) -> R1ProRawAction:
        return R1ProRawAction(
            action=np.asarray(msg["action"], dtype=np.float64),
            repeat=_integral_repeat(msg.get("repeat", 1)),
            include_cameras=bool(msg.get("include_cameras", False)),
        )


_ACTION_REGISTRY: dict[str, type[RawAction]] = {
    "franka": FrankaRawAction,
    "franka_bimanual": BimanualFrankaRawAction,
    "r1pro": R1ProRawAction,
}


def _describe_violation(
    joints: np.ndarray, joint_lower: np.ndarray, joint_upper: np.ndarray
) -> str | None:
    """Which joint is out, which way, and by how much — or None if all are in.

    A miss by a micro-radian and a miss by a radian call for opposite fixes.
    """
    below = joint_lower - joints
    above = joints - joint_upper
    worst = np.maximum(below, above)
    index = int(np.argmax(worst))
    if worst[index] <= 0.0:
        return None
    edge = "below" if below[index] > above[index] else "above"
    limit = joint_lower[index] if edge == "below" else joint_upper[index]
    return (
        f"joint {index} is {worst[index]:.6f} rad {edge} its limit "
        f"({joints[index]:.6f} vs {limit:.6f})"
    )


def check_franka_bounds(
    action: FrankaRawAction,
    joint_lower: list[float],
    joint_upper: list[float],
    gripper_min: float = 0.0,
    gripper_max: float = 1.0,
) -> None:
    """Reject joints/gripper outside the published limits (-> ``invalid_action``)."""
    j = np.asarray(action.joints, dtype=np.float64)
    violation = _describe_violation(j, np.asarray(joint_lower), np.asarray(joint_upper))
    if violation is not None:
        raise InvalidActionError(f"franka.joints outside published limits: {violation}")
    if not (gripper_min <= float(action.gripper) <= gripper_max):
        raise InvalidActionError("franka.gripper outside range")


def check_bimanual_franka_bounds(
    action: BimanualFrankaRawAction,
    joint_lower: list[float],
    joint_upper: list[float],
    gripper_min: float = 0.0,
    gripper_max: float = 1.0,
) -> None:
    """Reject either arm's joints/gripper outside the published (Panda) limits."""
    joints = np.asarray(action.joints, dtype=np.float64)
    grippers = np.asarray(action.grippers, dtype=np.float64)
    lower, upper = np.asarray(joint_lower), np.asarray(joint_upper)
    for arm in range(joints.shape[0]):
        violation = _describe_violation(joints[arm], lower, upper)
        if violation is not None:
            raise InvalidActionError(
                f"franka_bimanual.joints[{arm}] outside published limits: {violation}"
            )
        if not (gripper_min <= float(grippers[arm]) <= gripper_max):
            raise InvalidActionError(f"franka_bimanual.grippers[{arm}] outside range")


def encode_action(action: RawAction) -> dict[str, Any]:
    """Validate and serialize an action for the wire."""
    action.validate()
    return action.to_wire()


def decode_action(msg: dict[str, Any]) -> RawAction:
    """Deserialize and validate an action received by the gateway.

    Any decode failure of the untrusted payload is an ``InvalidActionError``, never
    a bare KeyError/TypeError escaping the coded-error channel.
    """
    embodiment = msg.get("embodiment")
    cls = _ACTION_REGISTRY.get(embodiment) if isinstance(embodiment, str) else None
    if cls is None:
        raise InvalidActionError(f"unknown embodiment: {embodiment!r}")
    try:
        action = cls.from_wire(msg)
        action.validate()
    except InvalidActionError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidActionError(f"malformed action payload: {type(exc).__name__}: {exc}") from exc
    return action


__all__ = [
    "RawAction",
    "FrankaRawAction",
    "BimanualFrankaRawAction",
    "R1ProRawAction",
    "InvalidActionError",
    "N_ARMS_BIMANUAL",
    "check_franka_bounds",
    "check_bimanual_franka_bounds",
    "encode_action",
    "decode_action",
]
