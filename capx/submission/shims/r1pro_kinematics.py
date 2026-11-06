"""R1Pro kinematics: pyroki IK/FK plus the ``q28`` <-> ``action25`` mapping.

pyroki/JAX are imported lazily, so the action mapping and the URDF joint limits
work without them.
"""

from __future__ import annotations

import math
import pathlib
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

_CAPX_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _wrap_to_pi(angle: float) -> float:
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


# Tool transform X from OmniGibson's ``{arm}_eef_link`` (the frame of the wire's
# ``cartesian_pos`` and every grasp pose) to the IK URDF's ``{arm}_gripper_link``:
# ``P_gripper_link = P_eef ∘ X``. X is an involution, so the same composition
# converts in both directions.
_EEF_TOOL_TRANSLATION = np.array([0.0, 0.0, -0.06])
_EEF_TOOL_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])  # RotY(180deg), [w,x,y,z]


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_mul_wxyz(q1: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """Hamilton product ``q1 * q0`` for [w, x, y, z] quaternions."""
    w1, x1, y1, z1 = (float(v) for v in q1)
    w0, x0, y0, z0 = (float(v) for v in q0)
    return np.array(
        [
            w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
            w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
            w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
            w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
        ]
    )


def _compose_eef_tool(pos: np.ndarray, quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot = _quat_wxyz_to_mat(quat_wxyz)
    return pos + rot @ _EEF_TOOL_TRANSLATION, _quat_mul_wxyz(quat_wxyz, _EEF_TOOL_QUAT_WXYZ)


class R1ProKinematics:
    """R1Pro IK/FK and the 25-dim controller-action mapping.

    The joint mappings convert between pyroki's and the controller's joint order
    and must match ``R1ProControlApi``'s own.
    """

    URDF_PATH = _CAPX_ROOT / "third_party" / "b1k" / "assets" / "r1pro_ik.urdf"

    PYROKI_JOINT_MAPPING: list[int] = [0, 1, 2, 3, 4, 11, 5, 12, 6, 13, 7, 14, 8, 15, 9, 16, 10, 17]
    CONTROLLER_JOINT_MAPPING: list[int] = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 5, 7, 9, 11, 13, 15, 17]

    # IK prefers moving the arm over the torso when both can reach the target.
    _TORSO_REST_WEIGHT = 20.0
    _TORSO_JOINT_NAMES = ("torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4")
    # When the strict solve misses by more than this, retry with uniform rest
    # weights: floor-level targets need a deep torso bend the torso weight refuses.
    _IK_RELAX_RESIDUAL = 0.05

    def __init__(self) -> None:
        self._robot: Any = None
        self._rest_cost_weights: Any = None
        self._rest_cost_weights_relaxed: Any = None
        self._urdf_limits: dict[str, tuple[float, float]] | None = None

    @property
    def robot(self) -> Any:
        if self._robot is None:
            import jax.numpy as jnp
            import pyroki as pk  # type: ignore
            import yourdfpy  # type: ignore

            urdf = yourdfpy.URDF.load(str(self.URDF_PATH))
            self._robot = pk.Robot.from_urdf(urdf)

            weights = jnp.array([1.0] * self._robot.joints.num_actuated_joints)
            self._rest_cost_weights_relaxed = weights
            for name in self._TORSO_JOINT_NAMES:
                idx = self._robot.joints.actuated_names.index(name)
                weights = weights.at[idx].set(self._TORSO_REST_WEIGHT)
            self._rest_cost_weights = weights
        return self._robot

    def solve_ik(
        self,
        target_pos_base: np.ndarray,
        target_wxyz_base: np.ndarray,
        arm: int,
        seed_q28: np.ndarray,
    ) -> np.ndarray:
        """IK for one arm's ``eef_link`` pose in the base frame (orientation wxyz).

        Returns ``seed_q28`` with only the torso and that arm's joints replaced.
        """
        from capx.integrations.motion import pyroki_snippets as pks

        target_link_name = "left_gripper_link" if arm == 0 else "right_gripper_link"
        seed_q28 = np.asarray(seed_q28, dtype=np.float64).reshape(28)
        seed_controller_q18 = seed_q28[6:24]
        seed_pyroki_q18 = seed_controller_q18[self.CONTROLLER_JOINT_MAPPING]

        gl_pos, gl_wxyz = _compose_eef_tool(
            np.asarray(target_pos_base, dtype=np.float64).reshape(3),
            np.asarray(target_wxyz_base, dtype=np.float64).reshape(4),
        )
        robot = self.robot
        gl_link_index = robot.links.names.index(target_link_name)

        def _pos_residual(pyroki_q18: np.ndarray) -> float:
            fk = np.asarray(robot.forward_kinematics(np.asarray(pyroki_q18)))
            return float(np.linalg.norm(np.asarray(fk[gl_link_index][4:]) - gl_pos))

        solved_pyroki_q18 = np.asarray(
            pks.solve_ik_rest(
                robot=robot,
                target_link_name=target_link_name,
                target_position=gl_pos,
                target_wxyz=gl_wxyz,
                rest_cost_weights=self._rest_cost_weights,
                initial_q=seed_pyroki_q18,
            )
        )
        if _pos_residual(solved_pyroki_q18) > self._IK_RELAX_RESIDUAL:
            relaxed = np.asarray(
                pks.solve_ik_rest(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_position=gl_pos,
                    target_wxyz=gl_wxyz,
                    rest_cost_weights=self._rest_cost_weights_relaxed,
                    initial_q=seed_pyroki_q18,
                )
            )
            if _pos_residual(relaxed) < _pos_residual(solved_pyroki_q18):
                solved_pyroki_q18 = relaxed
        solved_controller_q18 = np.asarray(solved_pyroki_q18)[self.PYROKI_JOINT_MAPPING]

        if arm == 0:
            joint_index = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16]
        else:
            joint_index = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17]

        q28 = seed_q28.copy()
        controller_index = np.asarray(joint_index, dtype=np.int64) + 6
        q28[controller_index] = solved_controller_q18[joint_index]
        return q28

    def fk_eef(self, q28: np.ndarray, arm: int) -> tuple[np.ndarray, np.ndarray]:
        """``(pos, quat_wxyz)`` of one arm's ``eef_link`` in the base frame."""
        robot = self.robot
        controller_q18 = np.asarray(q28, dtype=np.float64).reshape(28)[6:24]
        pyroki_q18 = controller_q18[self.CONTROLLER_JOINT_MAPPING]

        fk = np.asarray(robot.forward_kinematics(np.asarray(pyroki_q18)))
        link_name = "left_gripper_link" if arm == 0 else "right_gripper_link"
        link_index = robot.links.names.index(link_name)
        pose = fk[link_index]
        eef_pos, eef_wxyz = _compose_eef_tool(np.asarray(pose[4:]), np.asarray(pose[:4]))
        return eef_pos, eef_wxyz

    def q28_to_action25(self, q28: np.ndarray, base_pose_now: tuple[float, float, float]) -> np.ndarray:
        """A target ``q28`` as the 25-dim controller action.

        ``q28`` holds the base virtual joints ``[0:6]`` (translation, euler), torso
        ``[6:10]``, interleaved arms ``[10:24]`` (left even, right odd slots) and
        fingers ``[24:28]``. The action's base slots ``[0:3]`` are a local
        ``[dx, dy, dyaw]`` from ``base_pose_now = (x, y, yaw)``; the rest are absolute
        targets: trunk ``[3:7]``, left arm ``[7:14]``, left fingers ``[14:16]``, right
        arm ``[16:23]``, right fingers ``[23:25]``.
        """
        q = np.asarray(q28, dtype=np.float64).reshape(28)
        x, y, yaw = (float(v) for v in base_pose_now)

        cos_neg_yaw, sin_neg_yaw = math.cos(-yaw), math.sin(-yaw)
        rot_neg_yaw = np.array([[cos_neg_yaw, -sin_neg_yaw], [sin_neg_yaw, cos_neg_yaw]])
        local_delta = rot_neg_yaw @ (q[0:2] - np.array([x, y]))

        action = np.zeros(25, dtype=np.float64)
        action[0:2] = local_delta
        action[2] = _wrap_to_pi(float(q[5]) - yaw)
        action[3:7] = q[6:10]
        action[7:14] = q[10:24:2]
        action[14:16] = q[24:26]
        action[16:23] = q[11:24:2]
        action[23:25] = q[26:28]
        return action

    def joint_limits(self, name: str) -> tuple[float, float]:
        """``(lower, upper)`` of a URDF joint, read without pyroki."""
        limits = self._load_urdf_limits()
        if name not in limits:
            raise KeyError(f"no <limit> found for joint {name!r} in {self.URDF_PATH}")
        return limits[name]

    def _load_urdf_limits(self) -> dict[str, tuple[float, float]]:
        if self._urdf_limits is None:
            root = ET.parse(str(self.URDF_PATH)).getroot()
            limits: dict[str, tuple[float, float]] = {}
            for joint in root.findall("joint"):
                name = joint.get("name")
                limit = joint.find("limit")
                if name is None or limit is None:
                    continue
                lower = limit.get("lower")
                upper = limit.get("upper")
                if lower is None or upper is None:
                    continue
                limits[name] = (float(lower), float(upper))
            self._urdf_limits = limits
        return self._urdf_limits


__all__ = ["R1ProKinematics"]
