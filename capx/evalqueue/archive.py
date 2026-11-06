"""A compact copy of everything the queue knows, for keeping.

The public leaderboard shows a team's score, time and status. Everything else lives
only in the shared tree: each run's full ``RunResult`` (per-task scores, errors),
the queue records (which repo and commit an image came from), the watcher's
bookkeeping, and the logs and ``gateway-results.json`` of each team's latest run and
of the runs that failed on our side (the rest are deleted, capx/evalqueue/scratch.py).
None of it is large until the scratch a submission writes is counted, so this
leaves that out (the submission's HOME caches and outputs, per-process tmp) and the
images, which the Hub still has.
"""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

from capx.evalqueue.store import EvalStore, utcnow

_SKIP_DIRS = {"submission-home", "submission-out"}


def _skip(name: str) -> bool:
    return name in _SKIP_DIRS or name.startswith("tmp-")


def build_archive(store: EvalStore, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"backlog-{utcnow().strftime('%Y%m%dT%H%M%SZ')}.tar.gz"
    with tarfile.open(target, "w:gz") as archive:
        for top in (store.results, store.root / "queue", store.state):
            if top.exists():
                archive.add(top, arcname=str(top.relative_to(store.root)))
        for current, dirs, files in os.walk(store.runs, onerror=lambda _: None):
            dirs[:] = [d for d in dirs if not _skip(d)]
            for name in files:
                path = Path(current) / name
                try:
                    archive.add(path, arcname=str(path.relative_to(store.root)))
                except OSError:
                    continue  # unreadable scratch is not worth failing the archive for
    return target


__all__ = ["build_archive"]
