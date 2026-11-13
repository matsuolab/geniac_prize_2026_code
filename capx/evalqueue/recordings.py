"""Where a run's videos go (``capx.gateway.recording``), and when they are deleted.

They live under ``<root>/recordings``, outside ``runs/``: a run directory is deleted
when the team's next run starts, and its ``submission-out`` is the submission's own.
Only the gateway container mounts a run's recordings directory.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from capx.evalqueue.store import EvalStore

_logger = logging.getLogger(__name__)

DAY_S = 86400.0


def recordings_dir(store: EvalStore, run_id: str, subset: str) -> Path:
    path = store.recordings / run_id / subset
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _last_written(path: Path) -> float:
    newest = path.stat().st_mtime
    for current, _dirs, files in os.walk(path):
        for name in [*files, *_dirs]:
            try:
                newest = max(newest, (Path(current) / name).stat().st_mtime)
            except OSError:
                continue
    return newest


def prune_recordings(
    store: EvalStore, keep_days: float, *, now: Callable[[], float] = time.time
) -> list[str]:
    """Delete the recordings of runs last written more than ``keep_days`` ago, and
    return their run ids."""
    if not store.recordings.is_dir():
        return []
    cutoff = now() - keep_days * DAY_S
    removed: list[str] = []
    for run_dir in sorted(store.recordings.iterdir()):
        try:
            if _last_written(run_dir) >= cutoff:
                continue
        except OSError:
            continue
        shutil.rmtree(run_dir, ignore_errors=True)
        if not run_dir.exists():
            removed.append(run_dir.name)
    return removed


__all__ = ["DAY_S", "prune_recordings", "recordings_dir"]
