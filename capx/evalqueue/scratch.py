"""What a submission writes to the queue's shared root, and how long it is kept.

A submission's ``/capx/out`` and HOME are directories of its run
(``runs/<run_id>/<subset>/submission-{out,home}``, capx/evalqueue/runner.py), on the
filesystem every other team's evaluation and the intake write to. While it runs,
:class:`StorageLimit` measures both, and the runner stops it once they hold more than
``limits.submission_storage_*`` allow. A run's directory is kept until the same team's
next run starts (:func:`prune_runs`).
"""

from __future__ import annotations

import errno
import logging
import os
import stat
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from capx.evalqueue.models import GB, RunResult
from capx.evalqueue.queue import failed_on_our_side
from capx.evalqueue.store import EvalStore, utcnow

_logger = logging.getLogger(__name__)

CHECK_INTERVAL_S = 30.0

FIND_CONTAINER_S = 60.0
"""How long after the submission starts its container's /proc must have been found:
past that, what it holds open cannot be measured, which counts as over the limit."""

MAX_DEPTH = 256
"""How many directories deep a measurement goes before it counts the tree as over the
limit. It holds one directory open per level, and no real tree comes near this."""

_OPEN_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
"""Each directory is opened by its name in the one above, never by path, and never
through a symlink: one swapped in for a directory after it was listed is not followed."""


class UnmeasurableError(Exception):
    """Part of the tree cannot be measured: a directory the submission made unreadable,
    or nested past :data:`MAX_DEPTH`. That counts as over the limit, since it would
    otherwise hide what is in it."""


@dataclass
class DiskUse:
    bytes: int = 0
    """Allocated, so a sparse file counts what it takes rather than its length, and a
    file with several names counts once."""
    files: int = 0
    """Files and directories, every name of a hard-linked file included."""


class _Tally:
    def __init__(self, max_bytes: int, max_files: int) -> None:
        self.use = DiskUse()
        self._max_bytes = max_bytes
        self._max_files = max_files
        self._seen: set[tuple[int, int]] = set()

    @property
    def over(self) -> bool:
        return self.use.bytes > self._max_bytes or self.use.files > self._max_files

    def add_blocks(self, info: os.stat_result) -> bool:
        """Count a file's blocks, unless they were counted under another name. Whether
        they were not."""
        if (info.st_dev, info.st_ino) in self._seen:
            return False
        self._seen.add((info.st_dev, info.st_ino))
        self.use.bytes += info.st_blocks * 512
        return True

    def walk(self, root: str) -> None:
        """Count the tree under ``root``, holding open one directory per level. The
        root's own blocks count; the root is not one of the files."""
        fds: list[int] = []
        names: list[str] = []
        todo = [[root]]
        try:
            while todo and not self.over:
                if not todo[-1]:
                    todo.pop()
                    if fds:
                        os.close(fds.pop())
                        names.pop()
                    continue
                name = todo[-1].pop()
                if len(fds) > MAX_DEPTH:
                    raise UnmeasurableError(
                        f"{os.path.join(*names)} is over {MAX_DEPTH} levels deep"
                    )
                try:
                    fds.append(os.open(name, _OPEN_DIR, dir_fd=fds[-1] if fds else None))
                except (FileNotFoundError, NotADirectoryError):
                    continue
                except OSError as exc:
                    if exc.errno != errno.ELOOP:
                        raise
                    continue  # a symlink took its place since it was listed
                if not names:
                    self.add_blocks(os.fstat(fds[-1]))
                names.append(name)
                todo.append(self._list(fds[-1]))
        except PermissionError as exc:
            raise UnmeasurableError(f"cannot read all of {os.path.join(root, *names[1:])}") from exc
        finally:
            for fd in fds:
                os.close(fd)

    def _list(self, fd: int) -> list[str]:
        """Count what the directory holds, and return its subdirectories."""
        subdirs: list[str] = []
        with os.scandir(fd) as entries:
            for entry in entries:
                try:
                    info = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                self.add_blocks(info)
                self.use.files += 1
                if self.over:
                    break
                if stat.S_ISDIR(info.st_mode):
                    subdirs.append(entry.name)
        return subdirs


def disk_use(
    roots: Iterable[Path],
    max_bytes: int,
    max_files: int,
    *,
    procfs: int | None = None,
    seen_as: Iterable[str] = (),
) -> DiskUse:
    """What the trees under ``roots`` hold. Counting stops once either total passes its
    maximum, so a tree of millions of files takes no longer to look at than one at the
    limit. Symlinks are counted, never followed.

    With ``procfs``, a descriptor of the /proc of the submission's container
    (:func:`container_procfs`), it also counts the files deleted from the trees that a
    process in there still holds open: their blocks stay allocated until they are
    closed. ``seen_as`` are the trees' paths inside the container.

    Raises :class:`UnmeasurableError`, and the ``OSError`` of anything else it cannot read
    unless that went away meanwhile.
    """
    paths = [os.fspath(root) for root in roots]
    tally = _Tally(max_bytes, max_files)
    for path in paths:
        tally.walk(path)
        if tally.over:
            return tally.use
    if procfs is not None:
        prefixes = tuple(f"{path.rstrip('/')}/" for path in (*paths, *seen_as))
        for info in _held_open(procfs, prefixes):
            if tally.add_blocks(info):
                tally.use.files += 1
            if tally.over:
                break
    return tally.use


def container_procfs(tree: Path, inside: str) -> str | None:
    """The /proc of the container that has ``tree`` mounted at ``inside``, reached through
    one of its processes: ``/proc/<pid>/root/proc``. None while no process of it can be
    seen.

    Not the host's /proc: there, ``hidepid`` hides the processes that are not dumpable,
    which are Singularity's init and any a submission makes so. The container mounts a
    /proc of its own that lists every process in its PID namespace (``--containall``).
    Reaching it needs a process that is dumpable when it is opened; one held open stays
    readable after that (:class:`StorageLimit`).
    """
    want = os.stat(tree)
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            seen = os.stat(f"/proc/{name}/root{inside}")
        except OSError:
            continue
        if (seen.st_dev, seen.st_ino) == (want.st_dev, want.st_ino):
            return f"/proc/{name}/root/proc"
    return None


def _held_open(procfs: int, prefixes: tuple[str, ...]) -> Iterator[os.stat_result]:
    """Regular files deleted from under ``prefixes`` that a process in the /proc
    ``procfs`` holds open. A process of this user whose descriptors cannot be read (it
    made itself not dumpable, before or during the look) is :class:`UnmeasurableError`,
    and so is a /proc that can no longer be read. A process or descriptor that goes away
    meanwhile is skipped, and so is a process of another user."""
    try:
        processes = [name for name in os.listdir(procfs) if name.isdigit()]
    except OSError as exc:
        raise UnmeasurableError("lost the /proc of the submission's container") from exc
    for process in processes:
        if process == "1":
            continue  # Singularity's own init, which is not dumpable
        try:
            if _real_uid(procfs, process) != os.getuid():
                continue
            descriptors = os.open(f"{process}/fd", os.O_RDONLY | os.O_DIRECTORY, dir_fd=procfs)
        except _GONE:
            continue
        except PermissionError as exc:
            raise UnmeasurableError(f"cannot see what process {process} holds open") from exc
        try:
            yield from _deleted_files(descriptors, prefixes)
        except _GONE:
            continue
        except PermissionError as exc:
            raise UnmeasurableError(f"cannot see what process {process} holds open") from exc
        finally:
            os.close(descriptors)


_GONE = (FileNotFoundError, ProcessLookupError)
"""What reading a process's /proc entries raises once it has ended."""


def _deleted_files(descriptors: int, prefixes: tuple[str, ...]) -> Iterator[os.stat_result]:
    for descriptor in os.listdir(descriptors):
        try:
            target = os.readlink(descriptor, dir_fd=descriptors)
            if not (target.endswith(" (deleted)") and target.startswith(prefixes)):
                continue
            info = os.stat(descriptor, dir_fd=descriptors)
        except _GONE:
            continue  # closed since the listing
        if stat.S_ISREG(info.st_mode) and info.st_nlink == 0:
            yield info


def _real_uid(procfs: int, process: str) -> int:
    with open(os.open(f"{process}/status", os.O_RDONLY, dir_fd=procfs)) as handle:
        for line in handle:
            if line.startswith("Uid:"):
                return int(line.split()[1])
    raise FileNotFoundError(f"no Uid in {process}/status")


class StorageLimit:
    """Measures the submission's trees about every :data:`CHECK_INTERVAL_S` while
    :meth:`poll` is called, about once a second from the submission's start. Each
    measurement runs on a thread of its own: a large tree on the shared filesystem takes
    a while to walk, and the runner's other checks cannot wait.

    Until it has found the /proc of the submission's container, each poll looks for it,
    and then keeps it open: a process that makes itself not dumpable can no longer be
    reached from the host, but stays listed there. Not found within
    :data:`FIND_CONTAINER_S`, it counts as over the limit. :meth:`close` lets it go.
    """

    def __init__(
        self,
        trees: dict[Path, str],
        max_gb: float,
        max_files: int,
        *,
        on_exceeded: Callable[[], None],
    ) -> None:
        """``trees``: each directory the submission writes to, and its path inside the
        container. ``on_exceeded`` runs as soon as the trees are found over the limit,
        on whichever thread found it."""
        self._trees = trees
        self._max_gb = max_gb
        self._max_files = max_files
        self._on_exceeded = on_exceeded
        self._started = time.monotonic()
        self._due = self._started + CHECK_INTERVAL_S
        self._walk: threading.Thread | None = None
        self._procfs: int | None = None
        self._lock = threading.Lock()
        """Guards ``_procfs`` against :meth:`close` while a measurement copies it."""
        self.exceeded: tuple[str, datetime] | None = None
        """Why the trees are over the limit, and when that was found."""

    def poll(self) -> None:
        """Look for the container's /proc until found, and start a measurement if one
        is due."""
        if self.exceeded is not None:
            return
        if self._procfs is None:
            self._find_container()
        idle = self._walk is None or not self._walk.is_alive()
        if self.exceeded is None and idle and time.monotonic() >= self._due:
            self._due = time.monotonic() + CHECK_INTERVAL_S
            self._walk = threading.Thread(target=self._measure, daemon=True, name="storage")
            self._walk.start()

    def close(self) -> None:
        with self._lock:
            if self._procfs is not None:
                os.close(self._procfs)
                self._procfs = None

    def _find_container(self) -> None:
        path = container_procfs(*next(iter(self._trees.items())))
        if path is not None:
            try:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            except OSError:
                pass  # it ended or stopped being dumpable since it was found
            else:
                with self._lock:
                    self._procfs = fd
                return
        if time.monotonic() - self._started > FIND_CONTAINER_S:
            self._exceed(
                "could not measure what the submission wrote: no process of its container "
                f"could be seen within {FIND_CONTAINER_S:.0f}s"
            )

    def _measure(self) -> None:
        max_bytes = round(self._max_gb * GB)
        with self._lock:
            procfs = None if self._procfs is None else os.dup(self._procfs)
        try:
            use = disk_use(
                self._trees, max_bytes, self._max_files, procfs=procfs, seen_as=self._trees.values()
            )
        except UnmeasurableError as exc:
            self._exceed(f"could not measure what the submission wrote: {exc}")
            return
        except OSError as exc:
            _logger.warning("could not measure what the submission wrote: %s", exc)
            return
        finally:
            if procfs is not None:
                os.close(procfs)
        if use.bytes > max_bytes or use.files > self._max_files:
            self._exceed(
                f"submission wrote more than {self._max_gb:g} GiB or {self._max_files} files "
                f"to /capx/out and HOME (stopped counting at {use.bytes / GB:.1f} GiB, "
                f"{use.files} files)"
            )

    def _exceed(self, reason: str) -> None:
        self.exceeded = (reason, utcnow())
        self._on_exceeded()


def prune_runs(store: EvalStore, team_id: str) -> list[str]:
    """Delete the directories of the team's runs whose results are out, and return their
    run ids. Called as the team's next run starts, so each team keeps the directory of
    its latest run. Results and queue records stay. Left alone: a directory with no
    result (its run is still going, or its worker was stopped before it published), and
    one whose run failed on our side, for an operator to look at and delete."""
    removed: list[str] = []
    for run_dir in sorted(store.runs.iterdir()):
        try:
            result = RunResult.model_validate_json(
                (store.results / f"{run_dir.name}.json").read_text()
            )
        except (OSError, ValueError):
            continue
        if result.team_id != team_id or failed_on_our_side(result):
            continue
        try:
            _remove_tree(run_dir)
        except OSError:
            _logger.exception("could not remove %s", run_dir)
            continue
        removed.append(run_dir.name)
    return removed


def _remove_tree(path: Path) -> None:
    """Delete ``path`` and everything in it, however deep, including what the submission
    made read-only or unreadable. One directory is open at a time, and the walk climbs
    back through ``..``, so only for a tree nothing changes anymore."""
    fd = os.open(path, _OPEN_DIR)
    try:
        todo = [_clear(fd)]
        names: list[str] = []
        while todo:
            if todo[-1]:
                name = todo[-1].pop()
                os.chmod(name, 0o700, dir_fd=fd)
                child = os.open(name, _OPEN_DIR, dir_fd=fd)
                os.close(fd)
                fd = child
                names.append(name)
                todo.append(_clear(fd))
                continue
            todo.pop()
            if names:
                parent = os.open("..", _OPEN_DIR, dir_fd=fd)
                os.close(fd)
                fd = parent
                os.rmdir(names.pop(), dir_fd=fd)
    finally:
        os.close(fd)
    os.rmdir(path)


def _clear(fd: int) -> list[str]:
    """Delete everything in the directory but its subdirectories, and return those."""
    with os.scandir(fd) as entries:
        listed = [(entry.name, entry.is_dir(follow_symlinks=False)) for entry in entries]
    for name, is_dir in listed:
        if not is_dir:
            os.unlink(name, dir_fd=fd)
    return [name for name, is_dir in listed if is_dir]


__all__ = [
    "CHECK_INTERVAL_S",
    "FIND_CONTAINER_S",
    "MAX_DEPTH",
    "DiskUse",
    "StorageLimit",
    "UnmeasurableError",
    "container_procfs",
    "disk_use",
    "prune_runs",
]
