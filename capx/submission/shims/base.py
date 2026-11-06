"""``GatewayShimBase``: the episode, RPC and video plumbing every embodiment shares.

A shim is a plain object, not a ``gymnasium.Env``: ``CodeExecutionEnvBase`` and the
``capx/integrations`` control APIs only duck-type it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, TypeVar

import numpy as np

from capx.protocol import (
    EnvClient,
    ErrorCode,
    FullObs,
    GatewayError,
    InvalidActionError,
    ProprioObs,
    RawAction,
    StepResult,
)
from capx.protocol.actions import MAX_REPEAT
from capx.submission.interfaces import EPISODE_OVER_MESSAGE, STEP_CHUNK_DEFAULT
from capx.utils import turn_timing

_T = TypeVar("_T")


def remap_cameras(obs: FullObs) -> dict[str, dict[str, Any]]:
    """Each wire camera as the ``{"images", "intrinsics", "pose", "pose_mat"}`` dict
    the Franka control APIs index, with depth forced to ``(H, W, 1)``."""
    remapped: dict[str, dict[str, Any]] = {}
    for name, cam in obs.cameras.items():
        images: dict[str, Any] = {"rgb": cam.rgb}
        if cam.depth is not None:
            depth = np.asarray(cam.depth, dtype=np.float32)
            if depth.ndim == 2:
                depth = depth[:, :, np.newaxis]
            images["depth"] = depth

        pose_mat = cam.pose_mat
        if pose_mat is None:
            # The adapters always send pose_mat; importing viser only here keeps it
            # off the import path of every run.
            import viser.transforms as vtf

            pose = np.asarray(cam.pose, dtype=np.float64)
            pose_mat = vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3(wxyz=pose[3:]), translation=pose[:3]
            ).as_matrix()

        remapped[name] = {
            "images": images,
            "intrinsics": cam.intrinsics,
            "pose": cam.pose,
            "pose_mat": pose_mat,
        }
    return remapped


class GatewayShimBase:
    """Common episode/RPC/video plumbing for a gateway-backed low-level env.

    Subclasses implement ``_remap``, ``_main_camera_name`` and ``_wrist_camera_name``,
    and may override ``_on_reset``.
    """

    # `CodeExecutionEnvBase.step` reports `truncated` once `_sim_step_count >= max_steps`;
    # the gateway owns the step budget, so this only signals that the episode ended.
    max_steps: int = 1

    def __init__(
        self,
        client: EnvClient,
        task_id: str,
        *,
        camera_names: list[str],
        step_chunk: int = STEP_CHUNK_DEFAULT,
    ) -> None:
        self._client = client
        self._task_id = task_id
        self._camera_names = list(camera_names)
        self._step_chunk = step_chunk

        self._episode_id: str | None = None
        self._last_instruction: str | None = None
        self._episode_over = False
        self._voided = False
        self._disconnected = False
        self._latest_full_obs: FullObs | None = None
        self._latest_proprio_obs: ProprioObs | None = None
        self._full_obs_fresh = False

        self._capturing = False
        self._capture_wrist = False
        self._main_frames: list[np.ndarray] = []
        self._wrist_frames: list[np.ndarray] = []

    @property
    def last_instruction(self) -> str | None:
        return self._last_instruction

    @property
    def episode_over(self) -> bool:
        """The gateway ended this episode, or this shim can no longer reach it."""
        return self._episode_over

    @property
    def voided(self) -> bool:
        """The gateway voided this episode (``env_crash``): the trial was refunded."""
        return self._voided

    @property
    def disconnected(self) -> bool:
        """The connection to the gateway failed; this lane cannot run another trial."""
        return self._disconnected

    @property
    def _sim_step_count(self) -> int:
        return int(self._episode_over)

    def _remap(self, obs: FullObs) -> dict[str, Any]:
        raise NotImplementedError("subclasses must implement _remap")

    def _main_camera_name(self) -> str | None:
        raise NotImplementedError("subclasses must implement _main_camera_name")

    def _wrist_camera_name(self) -> str | None:
        raise NotImplementedError("subclasses must implement _wrist_camera_name")

    def _on_reset(self, obs: FullObs) -> None:
        return None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Begin a new episode. ``seed``/``options`` are ignored: the gateway seeds.

        A ``GatewayError`` from the reset propagates unchanged for the runner to handle.
        """
        self._episode_id = None
        self._episode_over = False
        self._voided = False
        self._disconnected = False
        result = self._client.reset(self._task_id)
        self._episode_id = result.episode_id
        self._last_instruction = result.instruction
        self._latest_full_obs = result.obs
        self._latest_proprio_obs = result.obs.proprio
        self._full_obs_fresh = True
        self._capturing = False
        self._capture_wrist = False
        self._main_frames = []
        self._wrist_frames = []
        self._on_reset(result.obs)
        return self._remap(result.obs), {"task_prompt": self._last_instruction}

    def get_observation(self) -> dict[str, Any]:
        return self._remap(self._fetch_full_obs())

    def compute_reward(self) -> float:
        """Always 0.0: reward never crosses the gateway boundary."""
        return 0.0

    def render(self, mode: str = "rgb_array") -> np.ndarray | None:
        return self._camera_rgb(self._fetch_full_obs(), self._main_camera_name())

    def render_wrist(self) -> np.ndarray | None:
        return self._camera_rgb(self._fetch_full_obs(), self._wrist_camera_name())

    def enable_video_capture(
        self, enabled: bool = True, *, clear: bool = True, wrist_camera: bool = False
    ) -> None:
        """Buffer one frame per ``_remote_step`` call while enabled."""
        self._capturing = enabled
        self._capture_wrist = wrist_camera
        if clear:
            self._main_frames = []
            self._wrist_frames = []

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = list(self._main_frames)
        if clear:
            self._main_frames = []
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._main_frames)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._main_frames[start:end]]

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = list(self._wrist_frames)
        if clear:
            self._wrist_frames = []
        return frames

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._wrist_frames[start:end]]

    @staticmethod
    def _camera_rgb(obs: FullObs, name: str | None) -> np.ndarray | None:
        cam = obs.cameras.get(name) if name is not None else None
        return None if cam is None else cam.rgb

    def _fetch_full_obs(self) -> FullObs:
        """The cached full obs while no cycle has run since it was stored, else a fresh one."""
        if self._episode_id is None:
            raise RuntimeError("shim used before reset()")
        if self._latest_full_obs is not None and self._full_obs_fresh:
            return self._latest_full_obs
        episode_id = self._episode_id
        with turn_timing.span("gateway.observation"):
            obs = self._gateway_call(lambda: self._client.get_observation(episode_id))
        self._latest_full_obs = obs
        self._latest_proprio_obs = obs.proprio
        self._full_obs_fresh = True
        return obs

    def _latest_proprio(self) -> ProprioObs:
        if self._latest_proprio_obs is None:
            raise RuntimeError("no proprio observation available before reset()")
        return self._latest_proprio_obs

    def _chunks(self, cycles: int) -> Iterator[int]:
        """Split a servo budget of ``cycles`` raw cycles into per-RPC ``repeat`` counts."""
        while cycles > 0:
            chunk = min(self._step_chunk, cycles, MAX_REPEAT)
            yield chunk
            cycles -= chunk

    def _remote_step(self, action: RawAction, *, want_full: bool = False) -> StepResult:
        """Send one action; every motion primitive goes through here.

        Stepping an ended episode raises ``EPISODE_OVER_MESSAGE``, which ``trial.py``
        treats as truncation.
        """
        if self._episode_over:
            raise RuntimeError(EPISODE_OVER_MESSAGE)
        if self._episode_id is None:
            raise RuntimeError("shim used before reset()")

        capture_this_call = self._capturing
        action.include_cameras = bool(action.include_cameras or want_full or capture_this_call)

        episode_id = self._episode_id
        try:
            with turn_timing.span("gateway.step"):
                result = self._gateway_call(lambda: self._client.step(episode_id, action))
        except InvalidActionError as exc:
            raise RuntimeError(f"invalid action: {exc}") from exc

        if result.terminated or result.truncated:
            self._episode_over = True

        if isinstance(result.obs, FullObs):
            self._latest_full_obs = result.obs
            self._latest_proprio_obs = result.obs.proprio
            self._full_obs_fresh = True
            if capture_this_call:
                self._append_video_frames(result.obs)
        else:
            self._latest_proprio_obs = result.obs
            self._full_obs_fresh = False

        return result

    def _gateway_call(self, call: Callable[[], _T]) -> _T:
        """Make one mid-episode RPC, re-raising its failure as a plain ``RuntimeError``.

        The agent's code sees an ordinary failure; the runner reads what happened from
        ``voided`` and ``disconnected`` instead, since the agent's code may catch it.
        Anything but an ``invalid_action`` ends the episode.
        """
        try:
            return call()
        except GatewayError as exc:
            if exc.code is not ErrorCode.INVALID_ACTION:
                self._episode_over = True
            if exc.code is ErrorCode.ENV_CRASH:
                self._voided = True
            raise RuntimeError(f"gateway error [{exc.code.value}]: {exc.detail}") from exc
        except ConnectionError as exc:
            self._episode_over = True
            self._disconnected = True
            raise RuntimeError(f"lost the gateway connection: {exc}") from exc

    def _append_video_frames(self, obs: FullObs) -> None:
        main_name = self._main_camera_name()
        if main_name is not None and main_name in obs.cameras:
            self._main_frames.append(obs.cameras[main_name].rgb.copy())
        if self._capture_wrist:
            wrist_name = self._wrist_camera_name()
            if wrist_name is not None and wrist_name in obs.cameras:
                self._wrist_frames.append(obs.cameras[wrist_name].rgb.copy())


__all__ = ["GatewayShimBase", "remap_cameras"]
