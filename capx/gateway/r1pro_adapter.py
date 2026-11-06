"""BEHAVIOR/OmniGibson's ``R1ProBehaviourLowLevel`` as a ``LowLevelEnv``.

``apply_raw_action`` is one ``env.step(action)``, as the vendored oracle scripts drive
the sim. ``get_proprio`` / ``get_observation`` remap OmniGibson's native obs (keyed by
a runtime-generated robot prim name, e.g. ``robot_wjqjxx``) onto the ``ObsFilter``
contract: stable camera aliases, RGB without alpha, and end-effector / camera poses
in the robot base frame. The base itself is not hidden: the first six entries of
``robot_joint_pos`` are the base's virtual joints, i.e. its pose in the world frame,
as PRELIM_RULEBOOK.md A.3 publishes.

Needs Isaac Sim + OmniGibson; imported lazily by the factory (and inside the
methods that touch the sim) so the rest of the gateway stays importable without them.
"""

from __future__ import annotations

import contextlib
import signal
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from capx.gateway.low_level import SimAdapter, StepOutcome
from capx.protocol.actions import R1ProRawAction, RawAction
from capx.protocol.taskpack import TaskSpec

if TYPE_CHECKING:
    from capx.envs.base import BaseEnv
    from capx.envs.simulators.r1pro_b1k import R1ProBehaviourLowLevel

# Stable aliases exposed to the wire for the robot-mounted cameras, independent of
# OmniGibson's per-instance robot prim name baked into its raw sensor keys (e.g.
# "robot_wjqjxx:zed_link:Camera:0" — the "robot_wjqjxx" part is instance-specific).
_CAMERA_SENSOR_SUFFIX = {
    "ego": "zed_link:Camera:0",
    "left_wrist": "left_realsense_link:Camera:0",
    "right_wrist": "right_realsense_link:Camera:0",
}


_T = TypeVar("_T")

# Kit binds its state to the thread that launched it, while EnvBackendServer serves
# each gateway connection on a fresh thread: a second connection driving the cached
# sim from its own thread hangs before the first step. Every sim call is funnelled
# through this one thread instead.
_SIM_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="isaac-sim")


def _on_sim_thread(fn: Callable[[], _T]) -> _T:
    return _SIM_THREAD.submit(fn).result()


@contextlib.contextmanager
def _tolerate_signal_handlers_off_main_thread():
    """Let ``og.launch`` run outside the main thread.

    OmniGibson registers a SIGINT handler — at the end of ``og.launch`` in 3.7, at
    import in 3.9 — and ``signal.signal`` raises ``ValueError`` off the main thread,
    which is exactly where the backend builds envs (the sim thread). The handler
    only wires Ctrl-C to a graceful shutdown for interactive runs; a backend process
    is torn down by its container runtime instead, so skipping the registration is safe.
    """
    if threading.current_thread() is threading.main_thread():
        yield
        return
    original = signal.signal

    def _tolerant(signalnum: int, handler: Any) -> Any:
        try:
            return original(signalnum, handler)
        except ValueError:  # "signal only works in main thread"
            return None

    signal.signal = _tolerant
    try:
        yield
    finally:
        signal.signal = original


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pose_wire(pos: Any, quat_xyzw: Any) -> np.ndarray:
    """(pos, quat_xyzw) -> the wire's (7,) [x, y, z, qw, qx, qy, qz]."""
    pos = _to_numpy(pos).astype(np.float32)
    quat_xyzw = _to_numpy(quat_xyzw).astype(np.float32)
    return np.concatenate([pos, quat_xyzw[[3, 0, 1, 2]]])


class R1ProLowLevelEnv(SimAdapter):
    embodiment = "r1pro"
    _base: R1ProBehaviourLowLevel

    @property
    def sim_time(self) -> float:
        import omnigibson as og

        return _on_sim_thread(lambda: float(og.sim.current_time))

    def reset(self, *, seed: int) -> None:
        # The seed selects a presampled BEHAVIOR task instance, not a sim RNG seed.
        _on_sim_thread(lambda: self._base.reset(options={"trial": seed}))

    def apply_raw_action(self, action: RawAction) -> StepOutcome:
        assert isinstance(action, R1ProRawAction)
        import torch

        command = torch.as_tensor(np.asarray(action.action, dtype=np.float32))

        def cycle() -> bool:
            self._base.step(command)
            return bool(self._base.task_completed())

        return StepOutcome(success=_on_sim_thread(cycle))

    def get_proprio(self) -> dict[str, Any]:
        return _on_sim_thread(self._proprio_fields)

    def get_observation(self) -> dict[str, Any]:
        def observe() -> dict[str, Any]:
            raw = self._proprio_fields()
            raw.update(self._camera_fields())
            return raw

        return _on_sim_thread(observe)

    def compute_reward(self) -> float:
        return _on_sim_thread(lambda: float(self._base.compute_reward()))

    def close(self) -> None:
        _on_sim_thread(self._base.close)

    def _proprio_fields(self) -> dict[str, Any]:
        robot = self._base.robot
        joint_pos = _to_numpy(self._base.get_joint_positions()).astype(np.float32)
        cartesian = np.concatenate(
            [_pose_wire(*robot.get_relative_eef_pose(arm=arm)) for arm in robot.arm_names]
        ).astype(np.float32)
        return {"robot_joint_pos": joint_pos, "robot_cartesian_pos": cartesian}

    def _camera_fields(self) -> dict[str, Any]:
        from omnigibson.utils import transform_utils

        obs = self._base.get_observation()
        robot_key = next(key for key in obs if "robot" in key)
        robot_obs = obs[robot_key]
        robot = self._base.robot
        base_pos, base_quat = robot.get_position_orientation()

        cameras: dict[str, Any] = {}
        for alias, suffix in _CAMERA_SENSOR_SUFFIX.items():
            sensor_key = f"{robot_key}:{suffix}"
            sensor_obs = robot_obs.get(sensor_key)
            sensor = robot.sensors.get(sensor_key)
            if sensor_obs is None or sensor is None:
                continue
            images: dict[str, np.ndarray] = {
                "rgb": _to_numpy(sensor_obs["rgb"])[..., :3].astype(np.uint8)
            }
            if "depth_linear" in sensor_obs:
                images["depth"] = _to_numpy(sensor_obs["depth_linear"]).astype(np.float32)
            cam_pos, cam_quat = sensor.get_position_orientation()
            rel_pos, rel_quat = transform_utils.relative_pose_transform(
                cam_pos, cam_quat, base_pos, base_quat
            )
            cameras[alias] = {
                "images": images,
                "intrinsics": _to_numpy(sensor.intrinsic_matrix).astype(np.float32),
                "pose": _pose_wire(rel_pos, rel_quat),
            }
        return cameras


def _hashable(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


class R1ProEnvFactory:
    """Builds real BEHAVIOR/OmniGibson envs.

    Not parallel-safe: ``get_env`` is ``@lru_cache``'d (one shared env per process)
    and Isaac Sim does not support concurrent stages in one process.
    """

    supports_parallel = False

    def __init__(self, *, enable_render: bool = True) -> None:
        self._enable_render = enable_render

    def build(self, spec: TaskSpec) -> R1ProLowLevelEnv:
        unhashable = [key for key, value in spec.env_kwargs.items() if not _hashable(value)]
        if unhashable:
            # get_env is lru_cache'd here (one env per process), so its kwargs are its key.
            raise ValueError(
                f"task {spec.task_id!r}: BEHAVIOR env_kwargs must be hashable: {unhashable}"
            )

        def build_on_sim_thread() -> BaseEnv:
            # Imported here, not in __init__, so the backend is listening long before
            # the ~15 s omnigibson import finishes.
            with _tolerate_signal_handlers_off_main_thread():
                import capx.envs.simulators  # noqa: F401 — registers the low-level envs
                from capx.envs.base import get_env

                return get_env(
                    spec.env_target,
                    privileged=spec.privileged,
                    enable_render=self._enable_render,
                    motion_planner=False,
                    **spec.env_kwargs,
                )

        return R1ProLowLevelEnv(_on_sim_thread(build_on_sim_thread))


__all__ = ["R1ProLowLevelEnv", "R1ProEnvFactory"]
