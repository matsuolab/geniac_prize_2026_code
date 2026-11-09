"""What a submission may write to the shared root, and how long a run's directory stays."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from capx.evalqueue import scratch
from capx.evalqueue.models import ErrorCode, RunResult, RunStatus, SubsetResult
from capx.evalqueue.scratch import StorageLimit, UnmeasurableError, disk_use, prune_runs
from capx.evalqueue.store import EvalStore

KIB = 1024
NO_LIMIT = 2**62

needs_permissions = pytest.mark.skipif(os.geteuid() == 0, reason="root reads any directory")


def _blocks(path) -> int:
    return os.lstat(path).st_blocks * 512


def test_both_trees_are_counted_together_in_allocated_bytes_and_entries(tmp_path) -> None:
    out, home = tmp_path / "out", tmp_path / "home"
    (out / "logs").mkdir(parents=True)
    (out / "logs" / "a.log").write_bytes(b"x" * 10 * KIB)
    (home / ".cache").mkdir(parents=True)
    (home / ".cache" / "b.bin").write_bytes(b"y" * 3 * KIB)

    use = disk_use([out, home], NO_LIMIT, NO_LIMIT)

    entries = [out / "logs", out / "logs" / "a.log", home / ".cache", home / ".cache" / "b.bin"]
    assert use.files == len(entries)
    assert use.bytes == sum(_blocks(path) for path in [out, home, *entries])


def test_the_roots_own_blocks_count_but_the_roots_are_not_files(tmp_path) -> None:
    """A directory takes blocks for its entries, the two the submission was given too."""
    for index in range(2000):
        (tmp_path / f"a-rather-long-file-name-{index:05}").touch()

    use = disk_use([tmp_path], NO_LIMIT, NO_LIMIT)

    assert use == scratch.DiskUse(bytes=_blocks(tmp_path), files=2000)
    assert use.bytes > 0


def test_a_sparse_file_counts_what_it_takes_not_its_length(tmp_path) -> None:
    with (tmp_path / "sparse").open("wb") as handle:
        handle.truncate(2**40)
    assert disk_use([tmp_path], NO_LIMIT, NO_LIMIT).bytes < 2**20


def test_a_file_with_several_names_takes_its_bytes_once_and_counts_every_name(tmp_path) -> None:
    (tmp_path / "a").write_bytes(b"x" * 64 * KIB)
    for name in ("b", "c"):
        os.link(tmp_path / "a", tmp_path / name)

    use = disk_use([tmp_path], NO_LIMIT, NO_LIMIT)

    assert use.files == 3
    assert use.bytes == _blocks(tmp_path) + _blocks(tmp_path / "a")


def test_counting_stops_as_soon_as_a_limit_is_passed(tmp_path) -> None:
    """A tree with millions of files costs no more to look at than one at the limit."""
    for index in range(50):
        (tmp_path / f"f{index}").write_bytes(b"x" * KIB)

    assert disk_use([tmp_path], NO_LIMIT, 5).files == 6
    by_bytes = disk_use([tmp_path], 4 * KIB, NO_LIMIT)
    assert by_bytes.files < 50 and by_bytes.bytes > 4 * KIB


def test_symlinks_are_counted_but_never_followed(tmp_path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "big").write_bytes(b"x" * 64 * KIB)
    out = tmp_path / "out"
    out.mkdir()
    (out / "to-dir").symlink_to(elsewhere)
    (out / "to-file").symlink_to(elsewhere / "big")
    (out / "loop").symlink_to(out / "loop")

    use = disk_use([out], NO_LIMIT, NO_LIMIT)

    assert use.files == 3
    assert use.bytes < 64 * KIB


def test_a_directory_swapped_for_a_symlink_after_it_was_listed_is_not_followed(
    tmp_path, monkeypatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "big").write_bytes(b"x" * 64 * KIB)
    out = tmp_path / "out"
    (out / "sub").mkdir(parents=True)
    open_ = os.open

    def swap_then_open(path, flags, mode=0o777, *, dir_fd=None):
        if path == "sub" and not (out / "sub").is_symlink():
            (out / "sub").rename(tmp_path / "moved")
            (out / "sub").symlink_to(elsewhere)
        return open_(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(scratch.os, "open", swap_then_open)
    use = disk_use([out], NO_LIMIT, NO_LIMIT)

    assert (out / "sub").is_symlink()
    assert use.bytes < 64 * KIB


def test_a_missing_tree_holds_nothing(tmp_path) -> None:
    assert disk_use([tmp_path / "gone"], NO_LIMIT, NO_LIMIT) == scratch.DiskUse(0, 0)


@needs_permissions
def test_a_directory_that_cannot_be_read_cannot_be_measured(tmp_path) -> None:
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / "big").write_bytes(b"x" * KIB)
    hidden.chmod(0)
    try:
        with pytest.raises(UnmeasurableError, match="cannot read"):
            disk_use([tmp_path], NO_LIMIT, NO_LIMIT)
    finally:
        hidden.chmod(0o700)


def _nest(top, depth: int, *, locked: bool = False) -> None:
    """``depth`` directories, each in the one before, below ``top``, and a file at the
    bottom, with a directory no one may read beside it if ``locked``. Made one level at
    a time, so the depth is not bounded by PATH_MAX."""
    fd = os.open(top, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for _ in range(depth):
            os.mkdir("d", dir_fd=fd)
            child = os.open("d", os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = child
        os.close(os.open("bottom", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=fd))
        if locked:
            os.mkdir("locked", dir_fd=fd)
            os.chmod("locked", 0, dir_fd=fd)
    finally:
        os.close(fd)


def test_a_tree_deeper_than_the_walk_goes_cannot_be_measured(tmp_path) -> None:
    deep, shallow = tmp_path / "deep", tmp_path / "shallow"
    deep.mkdir()
    shallow.mkdir()
    _nest(deep, scratch.MAX_DEPTH + 1)
    (shallow / "d").mkdir()

    with pytest.raises(UnmeasurableError, match="levels deep"):
        disk_use([deep], NO_LIMIT, NO_LIMIT)
    assert disk_use([shallow], NO_LIMIT, NO_LIMIT).files == 1


HOLDER = """
import ctypes, os, sys
handle = open(os.path.join(sys.argv[1], "gone.bin"), "wb")
os.unlink(handle.name)
handle.write(b"x" * 256 * 1024)
handle.flush()
os.fsync(handle.fileno())
if sys.argv[2] == "hide":
    assert ctypes.CDLL(None).prctl(4, 0, 0, 0, 0) == 0  # PR_SET_DUMPABLE
print("ready", flush=True)
sys.stdin.read()
"""


@contextlib.contextmanager
def _holding_a_deleted_file(out, *, dumpable: bool = True):
    """A process that keeps a file it deleted from ``out`` open; yields its pid."""
    command = [sys.executable, "-c", HOLDER, str(out), "show" if dumpable else "hide"]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as holder:
        assert holder.stdout is not None and holder.stdout.readline() == b"ready\n"
        yield holder.pid
        holder.stdin.close()


@contextlib.contextmanager
def _opened(directory):
    """A descriptor of ``directory``, as the runner holds the container's /proc."""
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield fd
    finally:
        os.close(fd)


def _procfs(tmp_path, *pids: int):
    """A stand-in for the container's /proc that lists only these processes."""
    procfs = tmp_path / "proc"
    procfs.mkdir()
    for pid in pids:
        (procfs / str(pid)).symlink_to(f"/proc/{pid}")
    return _opened(procfs)


@contextlib.contextmanager
def _fake_process(procfs, name: str, *, uid: int):
    """A process of ``uid`` in a stand-in /proc whose descriptors cannot be read, as a
    process that is not dumpable shows in the container's /proc."""
    process = procfs / name
    (process / "fd").mkdir(parents=True)
    (process / "status").write_text(f"Name:\tx\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
    (process / "stat").write_text(f"{name} (x) S 1 {name} {name} 0 -1\n")  # alive
    (process / "fd").chmod(0)
    try:
        yield
    finally:
        (process / "fd").chmod(0o700)


def test_the_submissions_container_is_found_by_what_is_mounted_where(tmp_path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    with _holding_a_deleted_file(out):
        found = scratch.container_procfs(out, str(out))
    assert found is not None and found.startswith("/proc/") and found.endswith("/root/proc")
    assert scratch.container_procfs(out, "/capx/out") is None


def test_a_deleted_file_a_process_of_the_submission_keeps_open_still_counts(tmp_path) -> None:
    """Its blocks stay allocated until it is closed, and no walk can see it."""
    out = tmp_path / "out"
    out.mkdir()
    with _holding_a_deleted_file(out) as pid, _procfs(tmp_path, pid) as procfs:
        use = disk_use([out], NO_LIMIT, NO_LIMIT, procfs=procfs)
    assert use.files == 1 and use.bytes >= _blocks(out) + 256 * KIB


def test_a_non_dumpable_process_holding_a_deleted_file_is_over_the_limit(tmp_path) -> None:
    """It made its descriptors unreadable to us. This needs a /proc that shows the
    process: the container's does, while a host /proc mounted with hidepid hides it."""
    out = tmp_path / "out"
    out.mkdir()
    with _holding_a_deleted_file(out, dumpable=False) as pid:
        if not os.path.exists(f"/proc/{pid}"):
            pytest.skip("this host's /proc hides processes that are not dumpable (hidepid)")
        with _procfs(tmp_path, pid) as procfs, pytest.raises(UnmeasurableError, match="holds open"):
            disk_use([out], NO_LIMIT, NO_LIMIT, procfs=procfs)


@needs_permissions
def test_a_process_whose_descriptors_cannot_be_read_is_over_the_limit(tmp_path) -> None:
    """What the container's /proc shows for a process that made itself not dumpable."""
    procfs = tmp_path / "proc"
    with _fake_process(procfs, "16", uid=os.getuid()), _opened(procfs) as fd, pytest.raises(
        UnmeasurableError, match="process 16"
    ):
        disk_use([tmp_path / "out"], NO_LIMIT, NO_LIMIT, procfs=fd)


@needs_permissions
def test_a_zombie_whose_descriptors_cannot_be_read_is_skipped(tmp_path) -> None:
    """A process that has exited shows its /proc entries as root's: it holds nothing."""
    procfs = tmp_path / "proc"
    with _fake_process(procfs, "16", uid=os.getuid()):
        (procfs / "16" / "stat").write_text("16 (x y) Z 1 16 16 0 -1\n")
        with _opened(procfs) as fd:
            disk_use([tmp_path / "out"], NO_LIMIT, NO_LIMIT, procfs=fd)


@needs_permissions
def test_a_live_process_that_stays_unreadable_is_still_over_the_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(scratch, "EXIT_GRACE_S", 0.1)
    procfs = tmp_path / "proc"
    with _fake_process(procfs, "16", uid=os.getuid()):
        (procfs / "16" / "stat").write_text("16 (x) S 1 16 16 0 -1\n")
        with _opened(procfs) as fd, pytest.raises(UnmeasurableError, match="process 16"):
            disk_use([tmp_path / "out"], NO_LIMIT, NO_LIMIT, procfs=fd)


def test_a_real_zombie_does_not_stop_the_measurement(tmp_path) -> None:
    """The kernel makes an exited, unreaped child's fd directory root's while its status
    still names this user: the race that stopped real submissions."""
    out = tmp_path / "out"
    out.mkdir()
    child = os.fork()
    if child == 0:
        os._exit(0)
    try:
        deadline = time.monotonic() + 5
        while Path(f"/proc/{child}/stat").read_text().rpartition(")")[2].split()[0] != "Z":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        with _procfs(tmp_path, child) as procfs:
            disk_use([out], NO_LIMIT, NO_LIMIT, procfs=procfs)
    finally:
        os.waitpid(child, 0)


def _process_with_a_descriptor(procfs, name: str):
    """A process of this user in a stand-in /proc, holding one descriptor."""
    process = procfs / name
    (process / "fd").mkdir(parents=True)
    (process / "status").write_text(f"Name:\tx\nUid:\t{os.getuid()}\n")
    (process / "stat").write_text(f"{name} (x) S 1 {name} {name} 0 -1\n")  # alive
    (process / "fd" / "3").symlink_to("/capx/out/gone.bin (deleted)")
    return process / "fd"


@needs_permissions
def test_descriptors_that_turn_unreadable_after_the_directory_was_opened_are_over_the_limit(
    tmp_path,
) -> None:
    """The process stopped being dumpable between the opening of its fd directory and
    the look at an entry: each entry is checked again, and refuses."""
    fd_dir = _process_with_a_descriptor(tmp_path / "proc", "16")
    fd_dir.chmod(0o400)  # listed, but no entry can be looked up
    try:
        with _opened(tmp_path / "proc") as procfs, pytest.raises(UnmeasurableError, match="16"):
            disk_use([tmp_path / "out"], NO_LIMIT, NO_LIMIT, procfs=procfs, seen_as=["/capx/out"])
    finally:
        fd_dir.chmod(0o700)


def test_an_fd_directory_that_cannot_be_listed_after_it_was_opened_is_over_the_limit(
    tmp_path, monkeypatch
) -> None:
    _process_with_a_descriptor(tmp_path / "proc", "16")
    listdir = os.listdir
    with _opened(tmp_path / "proc") as procfs:

        def refuse_fd_listing(path):
            if isinstance(path, int) and path != procfs:
                raise PermissionError(13, "Permission denied")
            return listdir(path)

        monkeypatch.setattr(scratch.os, "listdir", refuse_fd_listing)
        with pytest.raises(UnmeasurableError, match="16"):
            disk_use([tmp_path / "out"], NO_LIMIT, NO_LIMIT, procfs=procfs, seen_as=["/capx/out"])


def test_a_descriptor_closed_since_the_listing_is_skipped(tmp_path, monkeypatch) -> None:
    _process_with_a_descriptor(tmp_path / "proc", "16")
    readlink = os.readlink

    def closed(path, *, dir_fd=None):
        if dir_fd is not None:
            raise FileNotFoundError(2, "No such file or directory")
        return readlink(path)

    monkeypatch.setattr(scratch.os, "readlink", closed)
    with _opened(tmp_path / "proc") as procfs:
        use = disk_use([tmp_path / "out"], NO_LIMIT, NO_LIMIT, procfs=procfs, seen_as=["/capx/out"])
    assert use == scratch.DiskUse(0, 0)


@needs_permissions
def test_singularitys_init_another_users_and_an_ended_process_are_left_alone(tmp_path) -> None:
    procfs = tmp_path / "proc"
    with _fake_process(procfs, "1", uid=os.getuid()), _fake_process(procfs, "7", uid=0):
        (procfs / "9").mkdir()  # gone between the listing and the look
        with _opened(procfs) as fd:
            use = disk_use([tmp_path / "out"], NO_LIMIT, NO_LIMIT, procfs=fd)
    assert use == scratch.DiskUse(0, 0)


def _limit(tree, max_gb: float, max_files: int) -> tuple[StorageLimit, list[bool]]:
    stops: list[bool] = []
    limit = StorageLimit(
        {tree: "/capx/out"}, max_gb, max_files, on_exceeded=lambda: stops.append(True)
    )
    return limit, stops


def _measure(limit: StorageLimit):
    """Starts a measurement, waits for it to end, and returns what it found."""
    limit.poll()
    deadline = time.monotonic() + 5
    while limit._walk is not None and limit._walk.is_alive():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return limit.exceeded


def test_the_limit_is_checked_only_when_due(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(scratch, "CHECK_INTERVAL_S", 3600.0)
    (tmp_path / "a").write_bytes(b"x")
    limit, _ = _limit(tmp_path, 1.0, 0)
    limit.poll()
    assert limit._walk is None and limit.exceeded is None


def test_a_tree_within_the_limits_is_left_running(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(scratch, "CHECK_INTERVAL_S", 0.0)
    (tmp_path / "a").write_bytes(b"x" * KIB)
    limit, stops = _limit(tmp_path, 1.0, 10)
    assert _measure(limit) is None and stops == []


@pytest.mark.parametrize(("max_gb", "max_files"), [(1.0, 3), (KIB / 2**30, 100)], ids=["files", "bytes"])
def test_a_tree_over_a_limit_is_stopped_at_once_and_reported(
    tmp_path, monkeypatch, max_gb, max_files
) -> None:
    monkeypatch.setattr(scratch, "CHECK_INTERVAL_S", 0.0)
    for index in range(5):
        (tmp_path / f"f{index}").write_bytes(b"x" * 8 * KIB)
    limit, stops = _limit(tmp_path, max_gb, max_files)

    exceeded = _measure(limit)

    assert exceeded is not None
    reason, at = exceeded
    assert "/capx/out and HOME" in reason and at.tzinfo is not None
    assert stops == [True]
    limit.poll()
    assert not limit._walk.is_alive() and stops == [True]  # no measurement after that


class _Seen:
    """Stands in for container_procfs: the container's /proc is ``procfs`` from the
    ``after``-th look on, and None before."""

    def __init__(self, procfs, after: int = 1) -> None:
        self.procfs, self.after, self.looks = procfs, after, 0

    def __call__(self, tree, inside) -> str | None:
        self.looks += 1
        return str(self.procfs) if self.looks >= self.after else None


@needs_permissions
def test_the_containers_proc_is_looked_for_until_found_then_held(tmp_path, monkeypatch) -> None:
    """Once every process of the submission has made itself not dumpable, none of them
    can be reached from the host. The /proc opened before that still lists them."""
    monkeypatch.setattr(scratch, "CHECK_INTERVAL_S", 3600.0)
    procfs = tmp_path / "proc"
    procfs.mkdir()
    seen = _Seen(procfs, after=3)
    monkeypatch.setattr(scratch, "container_procfs", seen)
    out = tmp_path / "out"
    out.mkdir()
    limit, stops = _limit(out, 20.0, 100_000)

    for _ in range(5):
        limit.poll()
    assert seen.looks == 3 and limit._procfs is not None

    hidden = tmp_path / "hidden"
    procfs.rename(hidden)  # no longer reachable by the path it was found at
    monkeypatch.setattr(scratch, "CHECK_INTERVAL_S", 0.0)
    limit._due = 0.0
    with _fake_process(hidden, "16", uid=os.getuid()):
        exceeded = _measure(limit)
    limit.close()

    assert exceeded is not None and "process 16" in exceeded[0] and stops == [True]
    assert limit._procfs is None


def test_a_container_whose_proc_is_never_seen_counts_as_over_the_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(scratch, "CHECK_INTERVAL_S", 3600.0)
    monkeypatch.setattr(scratch, "container_procfs", _Seen(None, after=10**9))
    monkeypatch.setattr(scratch, "FIND_CONTAINER_S", 0.0)
    limit, stops = _limit(tmp_path, 20.0, 100_000)

    limit.poll()

    assert limit.exceeded is not None and "could be seen" in limit.exceeded[0]
    assert stops == [True]


def test_a_container_found_but_never_opened_counts_as_over_the_limit_in_time(
    tmp_path, monkeypatch
) -> None:
    """Each time it is found, its process ends or stops being dumpable before its /proc
    is opened: that is no better than not finding it."""
    monkeypatch.setattr(scratch, "CHECK_INTERVAL_S", 3600.0)
    seen = _Seen(tmp_path / "never-there")
    monkeypatch.setattr(scratch, "container_procfs", seen)
    limit, stops = _limit(tmp_path, 20.0, 100_000)

    for _ in range(3):
        limit.poll()
    assert seen.looks == 3 and limit._procfs is None and limit.exceeded is None

    monkeypatch.setattr(scratch, "FIND_CONTAINER_S", 0.0)
    limit.poll()
    assert limit.exceeded is not None and "could be seen" in limit.exceeded[0]
    assert stops == [True]


def test_a_container_found_in_time_is_not_held_against_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(scratch, "CHECK_INTERVAL_S", 3600.0)
    procfs = tmp_path / "proc"
    procfs.mkdir()
    monkeypatch.setattr(scratch, "container_procfs", _Seen(procfs))
    limit, stops = _limit(tmp_path, 20.0, 100_000)
    limit.poll()
    monkeypatch.setattr(scratch, "FIND_CONTAINER_S", 0.0)
    limit.poll()
    assert limit._procfs is not None
    limit.close()
    assert limit.exceeded is None and stops == [] and limit._procfs is None


@needs_permissions
def test_a_directory_hidden_from_the_count_counts_as_over_the_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(scratch, "CHECK_INTERVAL_S", 0.0)
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    hidden.chmod(0)
    limit, stops = _limit(tmp_path, 20.0, 100_000)
    try:
        exceeded = _measure(limit)
    finally:
        hidden.chmod(0o700)
    assert exceeded is not None and "could not measure" in exceeded[0] and stops == [True]


def _run(
    store: EvalStore,
    run_id: str,
    team_id: str,
    *,
    published: bool = True,
    status: RunStatus = RunStatus.COMPLETED,
    **result,
):
    run_dir = store.run_dir(run_id)
    (run_dir / "public" / "submission-out").mkdir(parents=True)
    (run_dir / "public" / "submission.log").write_text("log\n")
    if published:
        store.publish_result(
            RunResult(run_id=run_id, team_id=team_id, submission_id="s", status=status, **result)
        )
    return run_dir


def test_a_teams_published_runs_are_removed_and_nothing_else(tmp_path) -> None:
    store = EvalStore(tmp_path)
    store.ensure_layout()
    first, second = _run(store, "r1", "alpha"), _run(store, "r2", "alpha")
    other_team = _run(store, "r3", "beta")
    unpublished = _run(store, "r4", "alpha", published=False)

    assert prune_runs(store, "alpha") == ["r1", "r2"]

    assert not first.exists() and not second.exists()
    assert other_team.exists() and unpublished.exists()
    assert {path.stem for path in store.results.glob("*.json")} == {"r1", "r2", "r3"}


def test_a_run_that_failed_on_our_side_is_kept_for_an_operator(tmp_path) -> None:
    """Like its image (FileQueue.drop_image): it is looked at, and run again by hand.
    A failure that is the submission's own goes."""
    store = EvalStore(tmp_path)
    store.ensure_layout()
    failed = RunStatus.FAILED
    broken_run = _run(store, "r1", "alpha", status=failed, error_code=ErrorCode.INTERNAL_ERROR)
    no_score = SubsetResult(status=failed, error_code=ErrorCode.NO_SCORE)
    broken_subset = _run(store, "r2", "alpha", status=failed, subsets={"public": no_score})
    crashed = _run(store, "r3", "alpha", status=failed, error_code=ErrorCode.SUBMISSION_FAILED)

    assert prune_runs(store, "alpha") == ["r3"]

    assert broken_run.exists() and broken_subset.exists() and not crashed.exists()


@needs_permissions
def test_what_the_submission_made_read_only_or_unreadable_is_removed_too(tmp_path) -> None:
    store = EvalStore(tmp_path)
    store.ensure_layout()
    run_dir = _run(store, "r1", "alpha")
    out = run_dir / "public" / "submission-out"
    (out / "modules" / "pkg").mkdir(parents=True)
    (out / "modules" / "pkg" / "go.mod").write_text("module x\n")
    (out / "modules" / "pkg").chmod(0o555)
    (out / "modules").chmod(0o555)
    (out / "hidden").mkdir()
    (out / "hidden" / "data").write_bytes(b"x")
    (out / "hidden").chmod(0)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o750)
    (out / "link").symlink_to(outside)

    assert prune_runs(store, "alpha") == ["r1"]

    assert not run_dir.exists()
    assert outside.exists() and outside.stat().st_mode & 0o777 == 0o750


def test_a_tree_deeper_than_python_recurses_is_removed(tmp_path) -> None:
    """shutil.rmtree recurses once per level and raises RecursionError on such a tree,
    which a submission can build well inside both limits."""
    store = EvalStore(tmp_path)
    store.ensure_layout()
    run_dir = _run(store, "r1", "alpha")
    _nest(run_dir / "public" / "submission-out", sys.getrecursionlimit() + 500, locked=True)

    assert prune_runs(store, "alpha") == ["r1"]

    assert not run_dir.exists()
