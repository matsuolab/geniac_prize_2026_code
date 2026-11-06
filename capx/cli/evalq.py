"""``capx evalq`` — operate the submission evaluation queue.

    capx evalq submit --manifest submission.yaml --image agent.sif   # by hand
    capx evalq watch                 # login node: HuggingFace -> queue/
    capx evalq worker                # GPU node: queue/ -> results/
    capx evalq supervise             # login node: keep a worker alive in Slurm
    capx evalq status
    capx evalq hub-status            # what the organizer account can see on the Hub
    capx evalq requeue-stale
    capx evalq publish               # push leaderboard.json (scores + queue state) to the Hub
    capx evalq archive               # tar results, queue records and run logs for keeping
    capx evalq queue-ranking --interim-root ROOT   # after the deadline: queue the ranking runs

Subcommand dispatch mirrors ``capx/cli/main.py``: one dataclass of flags per
subcommand, parsed by tyro.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
import yaml
from pydantic import ValidationError

from capx.config import load_eval_config
from capx.evalqueue import slurm
from capx.evalqueue.config import QueueConfig, load_queue_config
from capx.evalqueue.hubsource import HubSource, HubWeights, sha256_of
from capx.evalqueue.limits import check_image_size, check_sha256
from capx.evalqueue.models import GB, Rejected, RunResult, RunSpec
from capx.evalqueue.queue import FileQueue, new_job
from capx.evalqueue.ranking import RankingBook, queue_ranking_runs
from capx.evalqueue.runner import EvalRunner
from capx.evalqueue.store import EvalStore, job_id_for
from capx.evalqueue.teams import TeamRegistry, registered_teams, team_orgs
from capx.evalqueue.watcher import SubmissionWatcher
from capx.evalqueue.weights import WeightsCache
from capx.evalqueue.worker import EvalWorker

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class _Common:
    config: str | None = None
    """Queue config YAML, over the cluster defaults in capx/evalqueue/config.py."""
    verbose: bool = False


@dataclass
class WatchArgs(_Common):
    once: bool = False
    """Run a single scan and exit, instead of polling forever."""


@dataclass
class HubStatusArgs(_Common):
    pass


@dataclass
class WorkerArgs(_Common):
    max_jobs: int | None = None
    """Stop after this many jobs (useful for a one-shot verification run)."""
    worker_id: str | None = None


@dataclass
class SuperviseArgs(_Common):
    interval_s: float = 60.0
    once: bool = False


@dataclass
class StatusArgs(_Common):
    results: int = 5
    """How many recent result documents to summarize."""


@dataclass
class RequeueStaleArgs(_Common):
    older_than_h: float = 1.0
    """Requeue claimed jobs untouched for this long — a worker that died hard. A
    running worker touches its claim every few minutes (``worker.HEARTBEAT_S``)."""
    dry_run: bool = False


@dataclass(kw_only=True)
class QueueRankingArgs(_Common):
    """After the deadline: queue the ranking runs on the ranking round's queue (--config)."""

    interim_root: str
    """The interim round's queue root, where the watcher recorded the ranking submissions."""
    team: list[str] | None = None
    """Only these teams. Default: every team."""
    again: bool = False
    """Queue a team that already ran. Only for a run our side broke (rulebook §6.5)."""


@dataclass
class PublishArgs(_Common):
    once: bool = False
    """Publish a single time and exit, instead of every `leaderboard.interval_s`."""
    dry_run: bool = False
    """Print the snapshot instead of pushing it."""
    force: bool = False
    """Push even if nothing changed since the last push."""


@dataclass(kw_only=True)
class SubmitArgs(_Common):
    """Queue an image by hand. Teams submit by pushing a commit; this is for us."""

    manifest: str
    """A submission.yaml, as a team would push it. Its `image` entry is ignored."""
    image: str
    """Path to the .sif. Queued where it is, not copied."""
    team_id: str = ""
    """Overrides the manifest's."""
    submission_id: str = ""
    """Overrides the manifest's."""


def _setup(args: _Common) -> tuple[QueueConfig, EvalStore, FileQueue]:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    config = load_queue_config(args.config)
    store = EvalStore(config.root)
    store.ensure_layout()
    return config, store, FileQueue(store)


def _weights_cache(config: QueueConfig, store: EvalStore) -> WeightsCache:
    return WeightsCache(
        store,
        HubWeights(config.hub),
        config.watcher.max_weights_gb,
        team_orgs=lambda: team_orgs(registered_teams(store)),
    )


def _hub_source(config: QueueConfig, store: EvalStore) -> HubSource:
    if not config.teams.repo:
        raise SystemExit("teams.repo is not set in the queue config; no team can register")
    return HubSource(config.hub, store, TeamRegistry(config.teams, store).refresh)


def watch(args: WatchArgs) -> None:
    config, store, queue = _setup(args)
    source = _hub_source(config, store)
    watcher = SubmissionWatcher(config, store, queue, source, _weights_cache(config, store))
    if args.once:
        for warning in source.preflight():
            print(f"warning: {warning}", file=sys.stderr)
        enqueued = watcher.poll()
        print(json.dumps({"enqueued": enqueued}, indent=2))
        return
    watcher.run_forever()


def hub_status(args: HubStatusArgs) -> None:
    config, store, _ = _setup(args)
    source = _hub_source(config, store)
    teams = [team.model_dump(exclude_none=True) for team in source.describe()]
    print(json.dumps({"warnings": source.preflight(), "teams": teams}, indent=2))


def worker(args: WorkerArgs) -> None:
    config, store, queue = _setup(args)
    runner = EvalRunner(config, store)
    evaluator = EvalWorker(config, store, queue, runner, worker_id=args.worker_id)
    evaluator.install_signal_handlers()
    evaluator.run_forever(max_jobs=args.max_jobs)


def supervise(args: SuperviseArgs) -> None:
    config, _, _ = _setup(args)
    slurm.supervise(
        config,
        _REPO_ROOT,
        config_path=args.config,
        interval_s=args.interval_s,
        once=args.once,
    )


def status(args: StatusArgs) -> None:
    _, store, queue = _setup(args)
    stamped = [(_mtime(path), path) for path in store.results.glob("*.json")]
    recent = sorted(((mtime, path) for mtime, path in stamped if mtime is not None), reverse=True)
    document = {
        "root": str(store.root),
        "pending": [job.job_id for job in queue.pending_jobs()],
        "claimed": [job.job_id for job in queue.claimed_jobs()],
        "done": len(list(store.done.glob("*.json"))),
        "failed": len(list(store.failed.glob("*.json"))),
        "recent_results": [_summary(path) for _, path in recent[: args.results]],
    }
    print(json.dumps(document, indent=2))


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _summary(path: Path) -> dict[str, object]:
    try:
        result = RunResult.model_validate_json(path.read_text())
    except (OSError, ValueError) as exc:
        return {"run_id": path.stem, "unreadable": f"{type(exc).__name__}: {exc}"}
    return {"run_id": path.stem} | result.model_dump(
        mode="json", include={"team_id", "status", "error"}
    )


def requeue_stale(args: RequeueStaleArgs) -> None:
    _, _, queue = _setup(args)
    requeued = queue.requeue_stale(args.older_than_h * 3600, dry_run=args.dry_run)
    print(json.dumps({"requeued": requeued, "dry_run": args.dry_run}, indent=2))


def submit(args: SubmitArgs) -> None:
    """Queue an image the operator already has, without going through the Hub.

    Not how a team submits — that is a commit, and the watcher finds it. This is for
    running one by hand: checking a rebuilt image before committing hours of GPU to
    it, and driving a measurement over a pack.

    The image is queued where it is, not copied: the worker reads the same shared
    filesystem. Moving or rewriting it while the job is queued breaks the run.
    """
    config, store, queue = _setup(args)
    manifest, image = _manifest_and_image(args)
    if not image.is_file():
        raise SystemExit(f"image not found: {image}")

    job_id = job_id_for(manifest.team_id, manifest.submission_id)
    if queue.is_queued(job_id):
        raise SystemExit(f"{job_id} is already queued")
    try:
        check_image_size(image.stat().st_size / GB, load_eval_config())
        if manifest.image_sha256:
            check_sha256(manifest.image_sha256, sha256_of(image))
        if manifest.weights:
            weights = _weights_cache(config, store)
            weights.check(manifest.team_id, manifest.weights)
            weights.fetch(manifest.weights)
    except Rejected as exc:
        raise SystemExit(str(exc)) from exc

    queue.enqueue(new_job(job_id, manifest, image))
    print(json.dumps({"job_id": job_id, "image_path": str(image)}, indent=2))


def _manifest_and_image(args: SubmitArgs) -> tuple[RunSpec, Path]:
    image = Path(args.image).expanduser().resolve()
    path = Path(args.manifest).expanduser().resolve()
    overrides = {
        key: value
        for key, value in (("team_id", args.team_id), ("submission_id", args.submission_id))
        if value
    }
    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(str(exc)) from exc
    if not isinstance(document, dict):
        raise SystemExit(f"{path} is not a mapping")
    document.pop("image", None)
    try:
        return RunSpec.model_validate({**document, **overrides}), image
    except ValidationError as exc:
        raise SystemExit(f"{path} is invalid: {exc}") from exc


@dataclass
class ArchiveArgs(_Common):
    out: str = ""
    """Directory for the tarball. Defaults to <root>/archive."""


def archive(args: ArchiveArgs) -> None:
    from capx.evalqueue.archive import build_archive

    _, store, _ = _setup(args)
    print(build_archive(store, Path(args.out).expanduser() if args.out else store.root / "archive"))


def queue_ranking(args: QueueRankingArgs) -> None:
    config, store, queue = _setup(args)
    interim = EvalStore(args.interim_root)
    outcome = queue_ranking_runs(
        RankingBook(interim),
        interim,
        store,
        queue,
        HubSource(config.hub, store, lambda: registered_teams(interim)),
        _weights_cache(config, store),
        team_ids=set(args.team) if args.team else None,
        again=args.again,
    )
    print(json.dumps(outcome, indent=2, sort_keys=True))


def publish(args: PublishArgs) -> None:
    from capx.evalqueue import leaderboard

    config, store, _ = _setup(args)
    if args.dry_run:
        print(json.dumps(leaderboard.build_snapshot(config, store), indent=2, sort_keys=True))
    elif args.once:
        if not config.leaderboard.repo:
            raise SystemExit("leaderboard.repo is not set in the queue config")
        pushed = leaderboard.publish_once(config, store, force=args.force)
        print("published" if pushed else "unchanged")
    else:
        leaderboard.run_forever(config, store)


_COMMANDS = {
    "submit": (SubmitArgs, submit),
    "watch": (WatchArgs, watch),
    "worker": (WorkerArgs, worker),
    "supervise": (SuperviseArgs, supervise),
    "status": (StatusArgs, status),
    "hub-status": (HubStatusArgs, hub_status),
    "requeue-stale": (RequeueStaleArgs, requeue_stale),
    "publish": (PublishArgs, publish),
    "queue-ranking": (QueueRankingArgs, queue_ranking),
    "archive": (ArchiveArgs, archive),
}


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in _COMMANDS:
        raise SystemExit(f"usage: capx evalq {{{','.join(_COMMANDS)}}} [options]")
    args_type, handler = _COMMANDS[argv[0]]
    handler(tyro.cli(args_type, args=argv[1:], prog=f"capx evalq {argv[0]}"))


if __name__ == "__main__":
    main(sys.argv[1:])
