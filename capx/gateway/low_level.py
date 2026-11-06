"""The simulator surface the gateway drives, and the adapter base the real sims share."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from capx.protocol.actions import FrankaRawAction, RawAction

if TYPE_CHECKING:
    from capx.envs.base import BaseEnv
    from capx.envs.simulators.libero import FrankaLiberoEnv
    from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv


@dataclass
class StepOutcome:
    """Result of one raw-action cycle. ``success`` is whether the task is achieved
    right after it; the gateway keeps it, like reward, to itself."""

    terminated: bool = False
    truncated: bool = False
    success: bool = False


@runtime_checkable
class LowLevelEnv(Protocol):
    """``apply_raw_action`` runs one control cycle and checks success after it, so a
    success reached partway through a ``repeat`` is not missed. ``get_proprio`` is
    the cheap self-state path (no camera render) polled every step;
    ``get_observation`` returns the full raw dict, privileged fields included, which
    ``ObsFilter`` then sanitizes. ``embodiment`` must match the task's."""

    embodiment: str

    @property
    def sim_time(self) -> float: ...

    def reset(self, *, seed: int) -> None: ...
    def apply_raw_action(self, action: RawAction) -> StepOutcome: ...
    def get_proprio(self) -> dict[str, Any]: ...
    def get_observation(self) -> dict[str, Any]: ...
    def compute_reward(self) -> float: ...
    def close(self) -> None: ...


class SimAdapter:
    """A ``LowLevelEnv`` over a registered ``BaseEnv``: the calls every simulator shares."""

    embodiment: str

    def __init__(self, base: BaseEnv) -> None:
        self._base = base

    def reset(self, *, seed: int) -> None:
        self._base.reset(seed=seed)

    def get_observation(self) -> dict[str, Any]:
        return self._base.get_observation()

    def compute_reward(self) -> float:
        return float(self._base.compute_reward())

    def close(self) -> None:
        self._base.close()


class FrankaSimAdapter(SimAdapter):
    """One control cycle of a single-arm Franka env: set the joint target and the
    gripper, then ``_step_once`` (the path ``move_to_joints_*`` use)."""

    embodiment = "franka"
    _base: RobosuiteBaseEnv | FrankaLiberoEnv

    def apply_raw_action(self, action: RawAction) -> StepOutcome:
        assert isinstance(action, FrankaRawAction)
        self._base._current_joints = np.asarray(action.joints, dtype=np.float64).reshape(7)
        self._base._set_gripper(float(action.gripper))
        self._base._step_once()
        return StepOutcome(success=bool(self._base.task_completed()))


__all__ = ["FrankaSimAdapter", "LowLevelEnv", "SimAdapter", "StepOutcome"]
