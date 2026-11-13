"""Robosuite's ``RobosuiteBaseEnv`` (single-arm Franka) as a ``LowLevelEnv``.

Needs a GPU (Robosuite/MuJoCo); imported lazily by the factory so the rest of the
gateway stays importable without the sim stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from capx.gateway.env_build import build_low_level_env
from capx.gateway.low_level import FrankaSimAdapter, LowLevelEnv, render_mujoco
from capx.protocol.taskpack import Embodiment, TaskSpec

if TYPE_CHECKING:
    from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv

_PROPRIO_KEYS = ("robot_joint_pos", "robot_cartesian_pos")


class RobosuiteLowLevelEnv(FrankaSimAdapter):
    _base: RobosuiteBaseEnv

    @property
    def sim_time(self) -> float:
        return float(self._base.robosuite_env.sim.data.time)

    def _render_frame(self) -> np.ndarray:
        return render_mujoco(self._base.robosuite_env.sim, self._base.save_camera_name)

    def get_proprio(self) -> dict[str, Any]:
        obs = self._base.robosuite_env._get_observations()
        self._base._compute_gripper_obs(obs)
        # The rest of obs is every enabled observable, cached camera images
        # included, which would otherwise travel the backend link on every step.
        return {key: obs[key] for key in _PROPRIO_KEYS}


class RobosuiteEnvFactory:
    """Builds real Robosuite envs, one- or two-armed as the task's embodiment says.

    Not parallel-safe: several MuJoCo sims and render contexts driven at once from
    one process race on each other. ``RemoteEnvFactory`` runs one backend process
    per lane instead.
    """

    supports_parallel = False

    def __init__(self, *, enable_render: bool = True) -> None:
        import capx.envs.simulators  # noqa: F401  — registers the low-level envs

        self._enable_render = enable_render

    def build(self, spec: TaskSpec) -> LowLevelEnv:
        base = build_low_level_env(spec, enable_render=self._enable_render)
        # The adapter is picked from the task, so it is the env underneath that has
        # to match: the two-arm envs are the ones with this two-arm primitive.
        two_armed = hasattr(base, "move_to_joints_blocking_both")
        if two_armed != (spec.embodiment is Embodiment.FRANKA_BIMANUAL):
            base.close()
            raise ValueError(
                f"task {spec.task_id!r} is {spec.embodiment.value}, but its env "
                f"{spec.env_target!r} is {'two' if two_armed else 'one'}-armed"
            )
        if spec.embodiment is Embodiment.FRANKA_BIMANUAL:
            from capx.gateway.robosuite_bimanual_adapter import RobosuiteBimanualLowLevelEnv

            return RobosuiteBimanualLowLevelEnv(base)
        return RobosuiteLowLevelEnv(base)


__all__ = ["RobosuiteEnvFactory", "RobosuiteLowLevelEnv"]
