"""LIBERO's ``FrankaLiberoEnv`` as a ``LowLevelEnv``.

LIBERO has no cheap-obs helper like ``RobosuiteBaseEnv._compute_gripper_obs``, so
``get_proprio`` rebuilds the ``robot_joint_pos`` / ``robot_cartesian_pos`` fields
``get_observation`` returns from the cached ``_current_obs`` and link poses, without
the per-camera work.

Needs LIBERO's vendored stack (mutually exclusive with Robosuite in one venv);
imported lazily by the factory so the rest of the gateway stays importable without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import viser.transforms as vtf

from capx.gateway.env_build import build_low_level_env
from capx.gateway.low_level import FrankaSimAdapter
from capx.protocol.taskpack import TaskSpec

if TYPE_CHECKING:
    from capx.envs.simulators.libero import FrankaLiberoEnv

# The wrist-to-flange offset FrankaLiberoEnv.get_observation() bakes into
# robot_cartesian_pos, so get_proprio matches it exactly.
_GRIPPER_TCP_ROTATION = vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2.0)
_GRIPPER_TCP_TRANSLATION = np.array([0, 0, -0.107])


class LiberoLowLevelEnv(FrankaSimAdapter):
    _base: FrankaLiberoEnv

    @property
    def sim_time(self) -> float:
        return float(self._base.handle.env.sim.data.time)

    def get_proprio(self) -> dict[str, Any]:
        base = self._base
        gripper_robot_base = (
            vtf.SE3(wxyz_xyz=base.base_link_wxyz_xyz).inverse()
            @ vtf.SE3(wxyz_xyz=base.gripper_link_wxyz_xyz)
            @ vtf.SE3.from_rotation_and_translation(
                rotation=_GRIPPER_TCP_ROTATION, translation=_GRIPPER_TCP_TRANSLATION
            )
        )
        gripper_norm = base._current_obs["robot0_gripper_qpos"][0] / base.gripper_metric_length
        return {
            "robot_joint_pos": np.concatenate(
                [base._current_obs["robot0_joint_pos"], [gripper_norm]]
            ),
            "robot_cartesian_pos": np.concatenate(
                [
                    gripper_robot_base.translation(),
                    gripper_robot_base.rotation().wxyz,
                    [gripper_norm],
                ]
            ),
        }


class LiberoEnvFactory:
    """Builds real LIBERO envs. Not parallel-safe, like ``RobosuiteEnvFactory``."""

    supports_parallel = False

    def __init__(self, *, enable_render: bool = True) -> None:
        import capx.envs.simulators  # noqa: F401  — registers the low-level envs

        self._enable_render = enable_render

    def build(self, spec: TaskSpec) -> LiberoLowLevelEnv:
        return LiberoLowLevelEnv(build_low_level_env(spec, enable_render=self._enable_render))


__all__ = ["LiberoLowLevelEnv", "LiberoEnvFactory"]
