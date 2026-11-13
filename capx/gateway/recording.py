"""Videos of a few episodes per run, for the organizers to check that a submission runs.

Which episodes are filmed is drawn from the OS's randomness when the gateway starts,
and never crosses the wire. Frames come from the env backend (``Recordable``); this
side writes them out as MP4 on a thread of its own, so encoding does not hold a lane.
A failure to record is logged and never touches the episode or its score.
"""

from __future__ import annotations

import io
import logging
import os
import random
import threading
from collections.abc import Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

from capx.protocol.taskpack import TaskSpec

_logger = logging.getLogger(__name__)

RECORDING_FPS = 10.0
"""Frames per second of sim time, which is also the playback rate."""

DIR_MODE = 0o700


def choose_episodes(
    tasks: Mapping[str, TaskSpec], per_simulator: int, rng: random.Random
) -> frozenset[tuple[str, int]]:
    """Up to ``per_simulator`` ``(task_id, trial_index)`` pairs for each simulator,
    drawn from every trial of its tasks."""
    by_simulator: dict[str, list[tuple[str, int]]] = {}
    for task_id, spec in sorted(tasks.items()):
        trials = by_simulator.setdefault(spec.simulator, [])
        trials.extend((task_id, index) for index in range(spec.n_trials))
    chosen: set[tuple[str, int]] = set()
    for trials in by_simulator.values():
        chosen.update(rng.sample(trials, min(per_simulator, len(trials))))
    return frozenset(chosen)


class Recorder:
    """Writes ``<out_dir>/<simulator>/<task_id>__trial<k>__<success|fail>.mp4``."""

    def __init__(
        self, out_dir: Path, chosen: Iterable[tuple[str, int]], *, fps: float = RECORDING_FPS
    ) -> None:
        self._out_dir = out_dir
        self._chosen = frozenset(chosen)
        self.fps = fps
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recording")
        self._pending: list[Future[None]] = []
        self._closed = False
        self._lock = threading.Lock()

    def wants(self, task_id: str, trial_index: int) -> bool:
        return (task_id, trial_index) in self._chosen

    def save(self, spec: TaskSpec, trial_index: int, success: bool, frames: list[bytes]) -> None:
        if not frames:
            _logger.warning("no frames recorded for %s trial %d", spec.task_id, trial_index)
            return
        outcome = "success" if success else "fail"
        path = self._out_dir / spec.simulator / f"{spec.task_id}__trial{trial_index}__{outcome}.mp4"
        with self._lock:
            if self._closed:
                _logger.warning("dropped the recording %s: the gateway is shutting down", path)
                return
            self._pending.append(self._writer.submit(self._write, path, frames))

    def close(self, timeout_s: float | None = None) -> bool:
        """Wait for the videos queued so far. False if some were still being written."""
        with self._lock:
            self._closed = True
            pending, self._pending = self._pending, []
            self._writer.shutdown(wait=False)
        return not wait(pending, timeout=timeout_s).not_done

    def _write(self, path: Path, frames: list[bytes]) -> None:
        try:
            path.parent.mkdir(mode=DIR_MODE, parents=True, exist_ok=True)
            partial = path.with_name(f".{path.name}.partial")
            write_mp4(partial, frames, self.fps)
            os.replace(partial, path)
        except Exception:
            _logger.exception("writing the recording %s failed", path)


def write_mp4(path: Path, frames: list[bytes], fps: float) -> None:
    """Encode JPEG ``frames`` as H.264. Frames are cropped to the first one's size,
    rounded down to even, which yuv420p needs."""
    import imageio_ffmpeg
    from PIL import Image

    first = np.asarray(Image.open(io.BytesIO(frames[0])).convert("RGB"))
    height, width = first.shape[0] // 2 * 2, first.shape[1] // 2 * 2
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        output_params=["-f", "mp4"],
        macro_block_size=1,
    )
    writer.send(None)
    try:
        for jpeg in frames:
            frame = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
            writer.send(np.ascontiguousarray(frame[:height, :width]))
    finally:
        writer.close()


__all__ = ["RECORDING_FPS", "Recorder", "choose_episodes", "write_mp4"]
