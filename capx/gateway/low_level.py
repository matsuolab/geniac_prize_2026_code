"""The simulator surface the gateway drives, and the adapter base the real sims share."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
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


@runtime_checkable
class Recordable(Protocol):
    """An env that can film its episode for the organizers (``capx.gateway.recording``).

    ``stop_recording`` returns the JPEG frames captured since ``start_recording``, one
    every ``1 / fps`` seconds of sim time."""

    def start_recording(self, fps: float) -> None: ...
    def stop_recording(self) -> list[bytes]: ...


RECORDING_SIZE = 256
"""Width and height of a recorded frame, in pixels."""

MAX_RECORDED_FRAMES = 6000
"""Frames one recording keeps at most, which bounds a backend's memory."""

_logger = logging.getLogger(__name__)


@dataclass
class _Recording:
    interval_s: float
    next_at: float = float("-inf")
    frames: list[bytes] = field(default_factory=list)


def encode_jpeg(frame: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame[..., :3]).astype(np.uint8)).save(
        buffer, format="JPEG", quality=80
    )
    return buffer.getvalue()


def render_mujoco(sim: Any, camera_name: str) -> np.ndarray:
    """A frame from a MuJoCo camera of robosuite's or LIBERO's sim, right side up."""
    frame = sim.render(
        camera_name=camera_name, width=RECORDING_SIZE, height=RECORDING_SIZE, depth=False
    )
    return np.asarray(frame)[::-1]


class SimAdapter:
    """A ``LowLevelEnv`` over a registered ``BaseEnv``: the calls every simulator shares."""

    embodiment: str
    _recording: _Recording | None = None

    def __init__(self, base: BaseEnv) -> None:
        self._base = base

    @property
    def sim_time(self) -> float:
        raise NotImplementedError

    def start_recording(self, fps: float) -> None:
        self._recording = _Recording(interval_s=1.0 / fps)

    def stop_recording(self) -> list[bytes]:
        recording, self._recording = self._recording, None
        return recording.frames if recording is not None else []

    def _render_frame(self) -> np.ndarray:
        """One RGB frame of the scene camera, about ``RECORDING_SIZE`` square."""
        raise NotImplementedError

    def _capture_if_due(self, now: float) -> None:
        """Called after every control cycle. A failure to film ends the recording, never
        the episode."""
        recording = self._recording
        if recording is None or now < recording.next_at:
            return
        if len(recording.frames) >= MAX_RECORDED_FRAMES:
            return
        recording.next_at = now + recording.interval_s
        try:
            recording.frames.append(encode_jpeg(self._render_frame()))
        except Exception:
            _logger.exception("recording stopped: capturing a frame failed")
            self._recording = None

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
        if self._recording is not None:
            self._capture_if_due(self.sim_time)
        return StepOutcome(success=bool(self._base.task_completed()))


__all__ = [
    "RECORDING_SIZE",
    "FrankaSimAdapter",
    "LowLevelEnv",
    "Recordable",
    "SimAdapter",
    "StepOutcome",
    "encode_jpeg",
    "render_mujoco",
]
