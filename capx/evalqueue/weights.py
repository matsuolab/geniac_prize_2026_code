"""Model weights a submission declares, from acceptance to cleanup.

A submission runs with no network, so it cannot fetch a model when it starts. It
names what it needs instead — repo, commit, and where to mount it
(``SubmissionManifest.weights``) — and the queue does the fetching:

    watcher   check()  not another team's repo, private only in the team's own org,
                       exists at that commit, gated as declared, under the cap
              fetch()  materialize into weights/<owner>--<name>/<commit>/
              fetch_missing()  on every poll, whatever a queued job still lacks: a gated
                       repo the organizer account has no access to yet is queued without
                       its copy, which arrives once a person accepts the repo's terms
    worker             wait until every copy is in place
    runner             bind each one read-only at its declared path
    worker    prune_weights()  after each job, drop what no queued job still needs

The copies are real files, not the HuggingFace cache layout. A snapshot directory
in ``HF_HOME`` is relative symlinks into a sibling ``blobs/``, and bind-mounting
the snapshot on its own leaves every one of them dangling.

They are keyed by repo and commit, so submissions that use the same model share
one copy, and a copy stays as long as any pending or running job declares it.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import socket
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from capx.evalqueue.models import (
    GB,
    ErrorCode,
    QueueJob,
    Rejected,
    WeightsMount,
    hf_owner,
    is_commit_sha,
)
from capx.evalqueue.queue import FileQueue
from capx.evalqueue.store import EvalStore

_logger = logging.getLogger(__name__)

GRACE_S = 600.0
"""How long a copy is kept after it was fetched or reused, whether or not a job
names it yet: fetching and enqueueing happen in another process from pruning."""

_REFRESH_S = GRACE_S / 10
"""How often a fetch touches the copies it has, while it waits on the rest."""

_STAGING = ".partial."

_QUEUE_SCANS = 10
"""How many times active_jobs reads the queue before it gives up on it holding still."""


@dataclass(frozen=True)
class RepoInfo:
    size_bytes: int
    gated: bool
    private: bool
    accessible: bool = True
    """False for a gated repo whose terms the organizer account has not accepted."""


class WeightsFetcher(Protocol):
    def describe(self, mount: WeightsMount) -> RepoInfo:
        """Metadata only. Raises :class:`Rejected` if the repo cannot be had."""

    def download(self, mount: WeightsMount, target: Path) -> None:
        """Put the repo's files under ``target``. Raises :class:`Rejected`."""


def describe_mount(mount: WeightsMount) -> str:
    return f"{mount.repo}@{mount.revision[:12]}"


class WeightsCache:
    def __init__(
        self,
        store: EvalStore,
        fetcher: WeightsFetcher,
        max_gb: float,
        team_orgs: Callable[[], Mapping[str, str]] | None = None,
    ) -> None:
        self._store = store
        self._fetcher = fetcher
        self._max_gb = max_gb
        self._team_orgs = team_orgs
        """HuggingFace org -> the registered team it belongs to, read at each check
        because teams register while the queue runs."""

    def check(self, team_id: str, mounts: list[WeightsMount]) -> list[WeightsMount]:
        """Everything that can be refused from metadata, before any transfer. Returns
        the gated mounts the organizer account cannot download yet."""
        if not mounts:
            return []
        registered = self._team_orgs() if self._team_orgs else {}
        orgs = {org.casefold(): team for org, team in registered.items()}
        for mount in mounts:
            # The organizer account can read every team's org, so without this one
            # team could have another's fine-tuned model fetched and mounted for it.
            owner_team = orgs.get(hf_owner(mount.repo))
            if owner_team is not None and owner_team != team_id:
                raise Rejected(
                    ErrorCode.FOREIGN_REPO,
                    f"{mount.repo} belongs to team {owner_team!r}; a submission may only "
                    "declare weights from its own team's org or from outside every team",
                )
        total_bytes = 0
        awaiting: list[WeightsMount] = []
        for mount in mounts:
            info = self._fetcher.describe(mount)
            # The organizer's token reads more than the registered teams' orgs: teams
            # not registered yet, and whatever else the account was ever given.
            if info.private and orgs.get(hf_owner(mount.repo)) != team_id:
                raise Rejected(
                    ErrorCode.FOREIGN_REPO,
                    f"{mount.repo} is private and not in your team's org; weights from "
                    "outside it must be in a public repo",
                )
            if info.gated and not mount.gated:
                raise Rejected(
                    ErrorCode.WEIGHTS_GATED,
                    f"{mount.repo} is gated on HuggingFace; declare `gated: true` for it so "
                    "the organizers know to request access",
                )
            if not info.accessible:
                awaiting.append(mount)
            total_bytes += info.size_bytes
        if self._max_gb > 0 and total_bytes / GB > self._max_gb:
            raise Rejected(
                ErrorCode.WEIGHTS_TOO_LARGE,
                f"declared weights total {total_bytes / GB:.1f} GiB, over the "
                f"{self._max_gb:.0f} GiB limit",
            )
        return awaiting

    def fetch_missing(self, mounts: list[WeightsMount]) -> list[WeightsMount]:
        """Fetch each mount that has no copy and can be downloaded now. Returns the ones
        still without a copy. A mount that fails is logged and left for the next call."""
        still_missing: list[WeightsMount] = []
        for mount in mounts:
            if self._store.weights_dir(mount).is_dir():
                continue
            try:
                if self._fetcher.describe(mount).accessible:
                    self.fetch([mount])
                    continue
            except Rejected as exc:
                _logger.warning("cannot fetch weights %s yet: %s", describe_mount(mount), exc)
            still_missing.append(mount)
        return still_missing

    def fetch(self, mounts: list[WeightsMount]) -> None:
        """Put every mount in place. All of them stay fresh for prune_weights until this
        returns, however long the last one takes, so the caller can enqueue the job that
        names them."""
        staging = {
            target: target.with_name(
                f".{target.name}{_STAGING}{socket.gethostname()}.{os.getpid()}"
            )
            for target in map(self._store.weights_dir, mounts)
        }
        with _kept_fresh([*staging, *staging.values()]):
            for mount in mounts:
                target = self._store.weights_dir(mount)
                if _touch(target):
                    continue
                partial = staging[target]
                shutil.rmtree(partial, ignore_errors=True)
                partial.mkdir(parents=True)
                _logger.info("fetching weights %s", describe_mount(mount))
                self._fetcher.download(mount, partial)
                try:
                    os.rename(partial, target)
                except OSError:
                    if not target.is_dir():
                        raise
                    # Someone else finished the same commit first; theirs is as good.
                    shutil.rmtree(partial, ignore_errors=True)


def prune_weights(store: EvalStore, queue: FileQueue) -> list[Path]:
    """Remove every copy no pending or running job declares. Returns what went."""
    jobs = active_jobs(queue)
    if jobs is None:
        _logger.info("the queue kept moving; weights are pruned after the next job")
        return []
    needed = {store.weights_dir(mount) for job in jobs for mount in job.weights}
    cutoff = time.time() - GRACE_S
    removed: list[Path] = []
    if not store.weights.is_dir():
        return removed
    for repo_dir in store.weights.iterdir():
        # Only what fetch() lays out. Anything else under weights/ is not ours to delete.
        if "--" not in repo_dir.name or not repo_dir.is_dir():
            continue
        for copy in repo_dir.iterdir():
            # A staging directory still being filled is kept fresh by its fetch.
            if not (is_commit_sha(copy.name) or _is_staging(copy.name)) or copy in needed:
                continue
            try:
                if copy.stat().st_mtime > cutoff:
                    continue
                # Renamed out of the way first, so nothing ever sees a half-deleted copy
                # under its real name, and a second pruner loses the race cleanly.
                trash = store.weights / f".trash.{repo_dir.name}.{copy.name}.{os.getpid()}"
                os.rename(copy, trash)
            except FileNotFoundError:
                continue
            shutil.rmtree(trash, ignore_errors=True)
            removed.append(copy)
            _logger.info("removed weights %s/%s", repo_dir.name, copy.name)
        with contextlib.suppress(OSError):
            repo_dir.rmdir()
    return removed


def active_jobs(queue: FileQueue) -> list[QueueJob] | None:
    """Every waiting or running job, or None when the queue would not hold still.

    Reading pending and claimed is not one step, and a claim moves a job pending ->
    claimed while a requeue moves it back, so a job can be in neither read. The reads
    are repeated until two in a row find the same jobs. Known limit: a job requeued,
    claimed again and requeued again within those few reads is still missed by both.
    """
    seen: set[str] | None = None
    for _ in range(_QUEUE_SCANS):
        jobs = [*queue.pending_jobs(), *queue.claimed_jobs()]
        ids = {job.job_id for job in jobs}
        if ids == seen:
            return jobs
        seen = ids
    return None


def _is_staging(name: str) -> bool:
    return name.startswith(".") and is_commit_sha(name[1:41]) and name[41:].startswith(_STAGING)


@contextlib.contextmanager
def _kept_fresh(paths: list[Path]) -> Iterator[None]:
    """Touch whichever of ``paths`` exist every ``_REFRESH_S`` meanwhile, and once more
    on the way out."""
    stop = threading.Event()

    def refresh() -> None:
        while not stop.wait(_REFRESH_S):
            for path in paths:
                _touch(path)

    thread = threading.Thread(target=refresh, name="weights-refresh", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        for path in paths:
            _touch(path)


def _touch(path: Path) -> bool:
    try:
        os.utime(path)
    except FileNotFoundError:
        return False
    return True


__all__ = [
    "GRACE_S",
    "RepoInfo",
    "WeightsCache",
    "WeightsFetcher",
    "active_jobs",
    "describe_mount",
    "prune_weights",
]
