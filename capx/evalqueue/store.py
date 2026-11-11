"""The shared directory — the only channel between the queue's processes.

The watcher runs on the login node and the worker on a GPU node; they never talk
to each other directly, only through this tree on the cluster-wide filesystem:

    <root>/
      incoming/<team_id>/<key>/              what the watcher pulled off the Hub
      queue/{pending,claimed,done,failed}/   one JSON per job; state == directory
      runs/<run_id>/                         per-run logs and scratch, until the team's next run
      results/<run_id>.json                  published RunResult
      state/seen/                            watcher bookkeeping (already enqueued)
      state/deferred/<team_id>.json          a submission waiting behind the team's running one
      weights/<owner>--<name>/<commit>/      model repos submissions declared, mounted read-only

``rename`` within the tree is atomic and fails with ``ENOENT`` for the loser of a
race, so nothing here needs a database or a lock server. Documents are written to a
temp file beside the target and renamed over it, so a reader never sees half of one.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from capx.evalqueue.models import RunResult, WeightsMount


def utcnow() -> datetime:
    return datetime.now(UTC)


def run_id_for(job_id: str) -> str:
    return f"{job_id}__{utcnow().strftime('%Y%m%dT%H%M%SZ')}"


def job_id_for(team_id: str, key: str) -> str:
    """Stable per-submission id, and the dedup key a resubmission is caught by."""
    return f"{team_id}__{key}"


class EvalStore:
    def __init__(self, root: str | Path) -> None:
        # Resolved, not just expanded: the conventional root goes through a symlink
        # (~/shared-storage -> /home/<group>), and a container that binds $HOME sees
        # the link but not its target, so every path under it would vanish inside.
        self.root = Path(root).expanduser().resolve()
        self.incoming = self.root / "incoming"
        self.pending = self.root / "queue" / "pending"
        self.claimed = self.root / "queue" / "claimed"
        self.done = self.root / "queue" / "done"
        self.failed = self.root / "queue" / "failed"
        self.runs = self.root / "runs"
        self.results = self.root / "results"
        self.state = self.root / "state"
        self.seen = self.state / "seen"
        self.deferred = self.state / "deferred"
        self.weights = self.root / "weights"

    def ensure_layout(self) -> None:
        for path in (
            self.incoming,
            self.pending,
            self.claimed,
            self.done,
            self.failed,
            self.runs,
            self.results,
            self.seen,
            self.deferred,
            self.weights,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def weights_dir(self, mount: WeightsMount) -> Path:
        return self.weights / mount.repo.replace("/", "--") / mount.revision

    def run_dir(self, run_id: str) -> Path:
        return self.runs / run_id

    def publish_result(self, result: RunResult) -> Path:
        path = self.results / f"{result.run_id}.json"
        write_json_atomic(path, result.model_dump(mode="json"))
        return path


def trials_finished(results_path: Path) -> int | None:
    """Trials that ended on their own terms (ran to their step limit, or the submission
    reset or closed them), from a gateway's results dump. Trials the job's end cut off
    do not count; voided ones are not in the dump. None without a readable dump, or
    one from a gateway that did not record how its episodes ended."""
    try:
        episodes = json.loads(results_path.read_text())["timings"]["episodes"]
        return sum(e["ended"] in ("steplimit", "reset") for e in episodes)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_json_atomic(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Beside the target: rename is only atomic within one filesystem.
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=True))
    os.replace(tmp, path)


__all__ = [
    "EvalStore",
    "job_id_for",
    "run_id_for",
    "trials_finished",
    "utcnow",
    "write_json_atomic",
]
