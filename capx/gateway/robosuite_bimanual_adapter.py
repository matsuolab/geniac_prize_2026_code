"""Robosuite's two-arm envs (``TwoArmLift``, ``TwoArmHandover``) as a ``LowLevelEnv``.

Those envs do not subclass ``RobosuiteBaseEnv``: they have no ``_current_joints``
target and their ``_step_once`` holds the joints rather than driving them. One
iteration of ``move_to_joints_blocking_both`` is one control cycle toward a target,
so ``max_steps=1`` with ``tolerance=0.0`` (a norm is never below zero) is exactly
the primitive the gateway needs. Both arms move in the one ``robosuite_env.step``;
stepping them separately would be different physics.

Needs a GPU (Robosuite/MuJoCo); imported lazily by the factory so the rest of the
gateway stays importable without the sim stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from capx.gateway.low_level import SimAdapter, StepOutcome, render_mujoco
from capx.protocol.actions import BimanualFrankaRawAction, RawAction

if TYPE_CHECKING:
    from capx.envs.simulators.robosuite_handover import RobosuiteHandoverEnv
    from capx.envs.simulators.robosuite_two_arm_lift import RobosuiteTwoArmLiftEnv

# Every arm of a Robosuite two-arm env is a Panda, and their observation keys are
# the same up to the index.
_ARM_KEYS = (("robot0_joint_pos", "robot0_cartesian_pos"), ("robot1_joint_pos", "robot1_cartesian_pos"))


class RobosuiteBimanualLowLevelEnv(SimAdapter):
    embodiment = "franka_bimanual"
    _base: RobosuiteTwoArmLiftEnv | RobosuiteHandoverEnv

    @property
    def sim_time(self) -> float:
        return float(self._base.robosuite_env.sim.data.time)

    def _render_frame(self) -> np.ndarray:
        # agentview shows both arms; only the handover env names its own camera.
        camera = getattr(self._base, "save_camera_name", "agentview")
        return render_mujoco(self._base.robosuite_env.sim, camera)

    def apply_raw_action(self, action: RawAction) -> StepOutcome:
        assert isinstance(action, BimanualFrankaRawAction)
        joints = np.asarray(action.joints, dtype=np.float64)
        grippers = np.asarray(action.grippers, dtype=np.float64)
        self._base._set_gripper(float(grippers[0]))
        self._base._set_gripper_arm1(float(grippers[1]))
        self._base.move_to_joints_blocking_both(
            joints[0], joints[1], tolerance=0.0, max_steps=1
        )
        if self._recording is not None:
            self._capture_if_due(self.sim_time)
        return StepOutcome(success=bool(self._base.task_completed()))

    def get_proprio(self) -> dict[str, Any]:
        """The cheap tier: per-arm state without rendering a camera.

        The two-arm envs compute the end-effector poses inside ``get_observation``,
        which re-renders; ``_compute_arm_cartesian_obs`` is that computation on its
        own, exactly as ``RobosuiteBaseEnv._compute_gripper_obs`` is for one arm.
        """
        raw = self._base.robosuite_env._get_observations()
        self._base._compute_arm_cartesian_obs(raw)
        return self._arms(raw)

    def get_observation(self) -> dict[str, Any]:
        obs = self._base.get_observation()
        obs.update(self._arms(obs))
        return obs

    def _arms(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Per-arm self-state in the shape ``ObsFilter`` allowlists.

        ``robot_joint_pos`` / ``robot_cartesian_pos`` also mirror arm 0, so the
        single-arm proprio shape stays valid for anything that reads it generically.
        """
        base_poses = (self._base.base_link_wxyz_xyz_0, self._base.base_link_wxyz_xyz_1)
        arms = []
        for (joint_key, cartesian_key), base_pose in zip(_ARM_KEYS, base_poses, strict=True):
            cartesian = np.asarray(raw[cartesian_key], dtype=np.float64)
            arms.append(
                {
                    # (8,) = 7 joints + gripper fraction, matching the single-arm
                    # convention. The gripper fraction is the last cartesian element.
                    "robot_joint_pos": np.concatenate(
                        [np.asarray(raw[joint_key], dtype=np.float64), cartesian[-1:]]
                    ),
                    "robot_cartesian_pos": cartesian,
                    "base_pose": np.asarray(base_pose, dtype=np.float64),
                }
            )
        return {
            "arms": arms,
            "robot_joint_pos": arms[0]["robot_joint_pos"],
            "robot_cartesian_pos": arms[0]["robot_cartesian_pos"],
        }


__all__ = ["RobosuiteBimanualLowLevelEnv"]
