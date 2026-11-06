"""``RobotClient`` — the forkable control-primitive library.

``robosuite_base``'s servo / gripper primitives rebuilt on ``EnvClient``; the
gateway only ever sees raw joint targets. IK is team-specific, so ``goto_pose``
takes a solver callable.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from capx.protocol import EnvClient, FrankaRawAction, FullObs, ProprioObs, ResetResult

IKSolver = Callable[[np.ndarray], np.ndarray]  # cartesian [xyz, qwxyz] -> (7,) joints


class RobotClient:
    def __init__(self, env: EnvClient, episode_id: str, initial_obs: FullObs) -> None:
        self._env = env
        self._episode_id = episode_id
        self._gripper = 1.0  # last commanded gripper (1 open .. 0 closed)
        self._joints = np.zeros(7)
        self._cartesian = np.zeros(8)
        self._track(initial_obs)

    @classmethod
    def start(cls, env: EnvClient, task_id: str) -> RobotClient:
        reset: ResetResult = env.reset(task_id)
        return cls(env, reset.episode_id, reset.obs)

    # -- state ---------------------------------------------------------------

    def get_joints(self) -> np.ndarray:
        return self._joints.copy()

    def get_cartesian(self) -> np.ndarray:
        return self._cartesian.copy()

    def observe(self) -> FullObs:
        obs = self._env.get_observation(self._episode_id)
        self._track(obs)
        return obs

    # -- primitives ----------------------------------------------------------

    def move_to_joints(
        self, target: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 100, repeat: int = 1
    ) -> np.ndarray:
        """Servo to a joint target, polling proprio until within tolerance."""
        goal = np.asarray(target, dtype=np.float64).reshape(7)
        for _ in range(max_steps):
            self._command(goal, self._gripper, repeat=repeat)
            if np.linalg.norm(self._joints - goal) < tolerance:
                break
        return self.get_joints()

    def open_gripper(self, *, cycles: int = 10) -> None:
        self._hold_gripper(1.0, cycles)

    def close_gripper(self, *, cycles: int = 10) -> None:
        self._hold_gripper(0.0, cycles)

    def goto_pose(
        self, pose: np.ndarray, ik: IKSolver, *, tolerance: float = 0.02, max_steps: int = 100
    ) -> np.ndarray:
        """IK the cartesian target to joints (team-supplied solver), then servo there."""
        joints = ik(np.asarray(pose, dtype=np.float64))
        return self.move_to_joints(joints, tolerance=tolerance, max_steps=max_steps)

    # -- internals -----------------------------------------------------------

    def _hold_gripper(self, value: float, cycles: int) -> None:
        self._gripper = value
        self._command(self._joints, value, repeat=cycles)

    def _command(self, joints: np.ndarray, gripper: float, *, repeat: int) -> None:
        result = self._env.step(
            self._episode_id, FrankaRawAction(joints=joints, gripper=gripper, repeat=repeat)
        )
        self._track(result.obs)

    def _track(self, obs: FullObs | ProprioObs) -> None:
        proprio = obs.proprio if isinstance(obs, FullObs) else obs
        self._joints = np.asarray(proprio.joint_pos[:7], dtype=np.float64)
        self._cartesian = np.asarray(proprio.cartesian_pos, dtype=np.float64)


__all__ = ["RobotClient", "IKSolver"]
