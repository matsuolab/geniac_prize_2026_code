"""Run one submission against one task pack, with the submission cut off the network.

    organizer side (host network)        submission side (no network at all)
    env-<sim> backend  <--unix socket-->  gateway  <--unix socket-->  agent.sif
                                                                      (--net --network=none)

The submission can reach only the gateway socket bind-mounted into it. Backend
sockets sit in a sibling directory that is never mounted into the submission,
because that link carries raw observations and reward:

    <socket root>/gw/gateway.sock        bind-mounted into the submission
    <socket root>/be/<sim>-<lane>.sock   the gateway's only

A backend starts when the gateway first asks for it (``be/<sim>-<lane>.sock.want``),
so ``max_lanes`` is a cap rather than a count of processes. A ``remote`` simulator
runs in Docker on another host, and ssh forwards its socket to the same path.
"""

from __future__ import annotations

import json
import logging
import os
import re
import select
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import IO

from capx.evalqueue.config import QueueConfig, RemoteBackendConfig, SubsetConfig
from capx.evalqueue.deadline import past_deadline
from capx.evalqueue.models import ErrorCode, QueueJob, RunResult, RunStatus, SubsetResult
from capx.evalqueue.scratch import StorageLimit
from capx.evalqueue.store import EvalStore, utcnow
from capx.protocol.framing import check_unix_path
from capx.protocol.taskpack import TaskPackManifest, load_pack, pack_dirs

_logger = logging.getLogger(__name__)

IMAGE_WORKDIR = "/workspace/capx"
"""Where the organizer images keep the checkout (singularity/*.def)."""

CONTAINER_HOME = "/capx/home"
"""HOME inside the submission container. --containall's own HOME is a small tmpfs,
and libraries that cache without being asked (Triton) run out of space on it."""

CONTAINER_SOCKET_DIR = "/capx/sock"
"""Where the socket directory is mounted inside the organizer containers. Not under
/tmp: each organizer container gets a private /tmp, because simulator stacks write
fixed paths there (LIBERO's /tmp/robosuite.log) that another job may already own."""

CLOCK_STARTED = "clock-started"
"""The gateway writes the epoch time of the submission's first contact here."""

JOB_FINISHED = "job-finished"
"""The gateway writes the epoch time of the submission's ``finish()`` here."""

SSH_OPTIONS = [
    "-o", "BatchMode=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=4",
]  # fmt: skip

HOST_ENV = ("PATH", "HOME", "LANG")
"""The only variables of the worker's environment that the commands it starts
(singularity, ssh, rsync) get. Singularity expands the variables in an env file with
its own environment and takes settings such as SINGULARITY_BIND from it, and the
worker's environment can hold tokens."""

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

SUBMISSION_LOG_BYTES = 64 * 2**20
"""``submission.log`` keeps the first and the last half of this much of the
submission's output, so a submission cannot fill the shared filesystem with it."""

GPU_LABEL = "capx.gpu"
"""Docker label on every remote backend container: the GPU it runs on. Other people
run their own containers on the remote host, and those carry no such label."""

LEASE_BUSY = 75
"""``flock``'s exit code when another worker holds the lease."""

LEASE_FENCED = 76
"""A remote command's exit code when another worker has taken the GPU's lease since."""

LEASE_OP_WAIT_S = 60
"""How long a remote command waits for the GPU's op lock (:meth:`RemoteGpuLease.guard`)."""

LEASE_HELD = "capx-lease-held"
"""What the lease's remote side prints, with the lease's generation, once it holds the
lock. Distinct, because the remote login's shell startup files can print first,
without a newline."""

_LEASE_HELD = re.compile(re.escape(LEASE_HELD.encode()) + rb" ([0-9]+)\n")


class SubsetFailed(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        at: datetime | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.at = at or utcnow()
        """When it went wrong, which is before the teardown that follows."""

    @property
    def status(self) -> RunStatus:
        return RunStatus.TIMEOUT if self.code is ErrorCode.SUBMISSION_TIMEOUT else RunStatus.FAILED


class DeadlinePassed(SubsetFailed):
    """The round closed while the submission ran: the run is cancelled, not scored."""

    def __init__(self) -> None:
        super().__init__("stopped at the round's deadline")

    @property
    def status(self) -> RunStatus:
        return RunStatus.CANCELLED


class Interrupted(SubsetFailed):
    """The worker was told to stop (:meth:`EvalRunner.interrupt`). The run is neither
    scored nor published: it runs again from the start."""

    def __init__(self, message: str = "the worker was told to stop") -> None:
        super().__init__(message)


class LeaseLost(Interrupted):
    """This worker no longer holds a remote GPU's lease, so another evaluation may have
    started there and removed this run's containers. Neither scored nor published, like
    an interruption, but it counts as one of the job's attempts."""


@dataclass
class _Proc:
    name: str
    popen: subprocess.Popen[bytes]
    log: Path
    on_stop: Callable[[], None] | None = None
    """Runs after the process is gone, e.g. to remove what it left on a remote host."""
    copier: threading.Thread | None = None
    """Copies the process's output to ``log`` when the log is capped."""


@dataclass(frozen=True)
class _Layout:
    """Where one subset's pack, sockets and logs are."""

    taskpack: Path
    pack_dirs: list[Path]
    socket_root: Path
    subset_dir: Path

    def spawn_organizer(
        self,
        name: str,
        image: Path,
        script: str,
        env: dict[str, str],
        *,
        nv: bool = False,
        binds: tuple[str, ...] = (),
    ) -> _Proc:
        """Start ``docker/<script>`` in an organizer image, with the pack mounted over
        whatever copy the image bakes in."""
        env_file = _write_env_file(self.subset_dir / f"{name}.env", env)
        private_tmp = self.subset_dir / f"tmp-{name}"
        private_tmp.mkdir(parents=True, exist_ok=True)
        command = [
            "singularity", "exec", *(["--nv"] if nv else []), "--cleanenv",
            # Without --pwd a relative path resolves against the host's checkout.
            "--pwd", IMAGE_WORKDIR,
            "--bind", f"{self.socket_root}:{CONTAINER_SOCKET_DIR}",
            "--bind", f"{private_tmp}:/tmp",
            *(arg for bind in binds for arg in ("--bind", bind)),
            *(
                arg
                for directory in self.pack_dirs
                for arg in ("--bind", f"{directory}:{IMAGE_WORKDIR}/taskpacks/{directory.name}:ro")
            ),
            "--env-file", str(env_file),
            str(image),
            "python", f"{IMAGE_WORKDIR}/docker/{script}",
        ]  # fmt: skip
        return _spawn(name, command, self.subset_dir / f"{name}.log")


def _simulators_for(manifest: TaskPackManifest, subset: SubsetConfig) -> list[str]:
    wanted = set(subset.simulators) if subset.simulators else None
    return sorted(
        {task.simulator for task in manifest.tasks if wanted is None or task.simulator in wanted}
    )


def _wait_until(
    ready: Callable[[], bool],
    proc: _Proc,
    timeout_s: float,
    interval_s: float,
    what: str,
    cancelled: Callable[[], bool] = lambda: False,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cancelled():
            raise SubsetFailed(f"stopped waiting for {what}: the subset is ending")
        if proc.popen.poll() is not None:
            raise SubsetFailed(
                f"{proc.name} exited with code {proc.popen.returncode} before its "
                f"socket appeared (see {proc.log})"
            )
        if ready():
            return
        time.sleep(interval_s)
    raise SubsetFailed(f"{proc.name} did not create {what} within {timeout_s:.0f}s")


def _stop(proc: _Proc, grace_s: float = 30.0) -> None:
    """SIGTERM, then SIGKILL. The gateway writes its results dump on SIGTERM."""
    try:
        if proc.popen.poll() is not None:
            return
        proc.popen.send_signal(signal.SIGTERM)
        try:
            proc.popen.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            _logger.warning("%s ignored SIGTERM; killing", proc.name)
            proc.popen.kill()
            proc.popen.wait(timeout=grace_s)
    finally:
        if proc.copier is not None:
            proc.copier.join(grace_s)
        if proc.on_stop is not None:
            proc.on_stop()


def _host_env() -> dict[str, str]:
    return {key: os.environ[key] for key in HOST_ENV if key in os.environ}


def _ssh_argv(host: str, remote_command: str) -> list[str]:
    return ["ssh", *SSH_OPTIONS, host, remote_command]


def _ssh(host: str, remote_command: str, timeout_s: float = 60.0) -> int:
    """Run one command on ``host``; its exit code (255 when ssh itself failed)."""
    try:
        done = subprocess.run(
            _ssh_argv(host, remote_command),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=_host_env(),
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 255
    return done.returncode


def _copy_packs_to_remote(host: str, directories: list[Path], remote_dir: str) -> None:
    """Copy the pack's directories to ``<remote_dir>/taskpacks``: a remote image
    carries no hidden pack, and cannot bind the local one."""
    remote_packs = f"{remote_dir}/taskpacks"
    if _ssh(host, f"mkdir -p -m 700 {shlex.quote(remote_packs)}") != 0:
        raise SubsetFailed(f"could not create {host}:{remote_packs}")
    for directory in directories:
        done = subprocess.run(
            [
                "rsync", "-a", "--delete", "--exclude", "__pycache__",
                "-e", shlex.join(["ssh", *SSH_OPTIONS]),
                f"{directory}/", f"{host}:{remote_packs}/{directory.name}/",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=_host_env(),
            timeout=600,
            check=False,
        )  # fmt: skip
        if done.returncode != 0:
            raise SubsetFailed(
                f"could not copy {directory} to {host}: {done.stderr.decode(errors='replace')}"
            )


def _remove_leftovers(lease: RemoteGpuLease) -> None:
    """Remove every capx container and socket directory on the lease's GPU. Only under
    the lease, when they are all a killed worker's: they keep their GPU memory, so this
    run's backends would not fit. Unlabelled containers are left."""
    status = _ssh(
        lease.host,
        lease.guard(
            f"ids=$(docker ps -aq --filter label={GPU_LABEL}={lease.gpu})"
            ' && { [ -z "$ids" ] || docker rm -f $ids >/dev/null; }'
            f" && rm -rf {shlex.quote(lease.socket_dir)}/capx-*-gpu-{lease.gpu}"
        ),
        timeout_s=300.0,
    )
    if status == LEASE_FENCED:
        raise lease.fenced()
    if status != 0:
        raise SubsetFailed(
            f"could not remove the leftover containers on {lease.host} (exit {status})"
        )


GpuKey = tuple[str, str, str]
"""A remote GPU: host, socket_dir (where its lock file is) and device."""


def _gpu_key(remote: RemoteBackendConfig) -> GpuKey:
    return remote.host, remote.socket_dir, remote.gpu


def _gpu_file(socket_dir: str, gpu: str, kind: str) -> str:
    """One of the GPU's files on the host, quoted for its shell: ``lock`` (the lease),
    ``gen`` (its generation) and ``op`` (held while the generation moves, or while a
    command acts on the GPU's containers)."""
    return shlex.quote(f"{socket_dir}/capx-gpu-{gpu}.{kind}")


def _under_op_lock(socket_dir: str, gpu: str, command: str) -> str:
    """``command`` in a subshell that holds the GPU's op lock while it runs. It waits up
    to :data:`LEASE_OP_WAIT_S` for the lock, then exits with :data:`LEASE_BUSY`. Tries
    ``flock -n`` in a loop: ``flock -w`` needs a timer, which some sandboxes refuse."""
    op = _gpu_file(socket_dir, gpu, "op")
    return (
        f"(i=0; while :; do flock -n -E {LEASE_BUSY} {op} sh -c {shlex.quote(command)};"
        f" s=$?; [ $s -ne {LEASE_BUSY} ] && exit $s; i=$((i + 1));"
        f" [ $i -ge {LEASE_OP_WAIT_S * 10} ] && exit {LEASE_BUSY}; sleep 0.1; done)"
    )


def _lease_command(socket_dir: str, gpu: str) -> str:
    """Holds the GPU's lock file until its stdin closes, which ssh passes on when the
    worker exits, however it exits. Once it has the lock, it counts the GPU's lease
    generation up and prints :data:`LEASE_HELD` with it."""
    generation = _gpu_file(socket_dir, gpu, "gen")
    count_up = (
        f"n=$(( $(cat {generation} 2>/dev/null || echo 0) + 1 ))"
        f" && echo $n > {generation}.tmp && mv {generation}.tmp {generation} && echo $n"
    )
    held = (
        f"n=$( {_under_op_lock(socket_dir, gpu, count_up)} )"
        f" && echo {LEASE_HELD} $n && exec cat >/dev/null"
    )
    lock = _gpu_file(socket_dir, gpu, "lock")
    return (
        f"mkdir -p {shlex.quote(socket_dir)}"
        f" && exec flock -n -E {LEASE_BUSY} {lock} sh -c {shlex.quote(held)}"
    )


class RemoteGpuLease:
    """One worker's hold on one remote GPU: an evaluation runs only while it holds the
    GPU's lease, so at most one does, whichever queue it belongs to."""

    def __init__(self, key: GpuKey, on_lost: Callable[[str], None]) -> None:
        self.host, self.socket_dir, self.gpu = key
        self._on_lost = on_lost
        """Told why, if the lease ends before :meth:`release`."""
        self.generation: int | None = None
        """Counted up on the host by each lease taken on the GPU: only the newest one's
        :meth:`fence` lets a command through."""
        self._proc: subprocess.Popen[bytes] | None = None
        self._ended = threading.Event()
        self._released = False

    def take(self, timeout_s: float = LEASE_OP_WAIT_S + 30.0) -> str | None:
        """None once held; else why not."""
        proc = subprocess.Popen(
            _ssh_argv(self.host, _lease_command(self.socket_dir, self.gpu)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            env=_host_env(),
            start_new_session=True,
        )
        assert proc.stdout is not None
        stdout = proc.stdout.fileno()
        deadline = time.monotonic() + timeout_s
        printed = b""
        while (left := deadline - time.monotonic()) > 0:
            if not select.select([stdout], [], [], left)[0]:
                continue
            chunk = os.read(stdout, 4096)
            if not chunk:
                break
            printed += chunk
            if held := _LEASE_HELD.search(printed):
                self.generation = int(held[1])
                self._proc = proc
                threading.Thread(target=self._watch, daemon=True, name="gpu-lease").start()
                return None
        else:
            proc.kill()
            _close(proc)
            return f"{self.host} did not answer within {timeout_s:.0f}s for GPU {self.gpu}'s lease"
        _close(proc)
        if proc.returncode == LEASE_BUSY:
            return f"{self.host} GPU {self.gpu} is in use by another evaluation"
        return f"could not take the lease on {self.host} GPU {self.gpu} (exit {proc.returncode})"

    def held(self) -> bool:
        return self._proc is not None and not self._ended.is_set() and not self._released

    def guard(self, command: str) -> str:
        """``command``, for one that removes or creates the GPU's containers or socket
        directories: it runs only while this lease is still the newest on the host,
        however long it took to get there, and exits with :data:`LEASE_FENCED`
        otherwise. It holds the GPU's op lock while it runs, so no newer lease can be
        taken meanwhile."""
        return _under_op_lock(self.socket_dir, self.gpu, f"{self._fence()}; {command}")

    def confirm(self) -> str | None:
        """None if the host still has this lease as the GPU's newest; else why not.
        Asked before a result counts: in a network partition the host can drop the
        lock, and another worker take it, while this worker's ssh still waits out its
        keepalive."""
        status = _ssh(self.host, self._fence(), timeout_s=30.0)
        if status == 0:
            return None
        if status == LEASE_FENCED:
            return str(self.fenced())
        return f"could not confirm the lease on {self.host} GPU {self.gpu} (exit {status})"

    def _fence(self) -> str:
        generation = _gpu_file(self.socket_dir, self.gpu, "gen")
        return f'[ "$(cat {generation} 2>/dev/null)" = "{self.generation}" ] || exit {LEASE_FENCED}'

    def fenced(self) -> LeaseLost:
        """For a command the host refused: another worker has taken the GPU since."""
        why = f"{self.host} GPU {self.gpu} has been taken over since lease {self.generation}"
        self._lose(why)
        return LeaseLost(why)

    def release(self) -> None:
        self._released = True
        if self._proc is not None:
            _close(self._proc)

    def _watch(self) -> None:
        assert self._proc is not None
        code = self._proc.wait()
        self._lose(f"lost the lease on {self.host} GPU {self.gpu} (ssh exit {code})")

    def _lose(self, why: str) -> None:
        if self._ended.is_set() or self._released:
            return
        self._ended.set()
        self._on_lost(why)


def _close(proc: subprocess.Popen[bytes]) -> None:
    """End a lease's ssh: closing its stdin lets the remote side exit and drop the lock."""
    for stream in (proc.stdin, proc.stdout):
        if stream is not None:
            stream.close()
    try:
        proc.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _remove_remote_backend(host: str, container: str, remote_sock: str) -> None:
    # The container outlives the local ssh. Only this lane's socket goes: other lanes
    # share the directory.
    status = _ssh(
        host,
        f"docker rm -f {shlex.quote(container)} >/dev/null 2>&1; rm -f {shlex.quote(remote_sock)}",
    )
    if status != 0:
        _logger.warning("could not remove %s on %s (exit %d)", container, host, status)


class EvalRunner:
    def __init__(self, config: QueueConfig, store: EvalStore) -> None:
        self._config = config
        self._store = store
        self._leases: dict[GpuKey, RemoteGpuLease] = {}
        self._lease_lost: str | None = None
        """Why a lease of the job in hand ended before it was released."""
        self._budget_used_s: float | None = None
        """The current subset's time on the referee's clock, set by _run_subset."""
        self._over_time = False
        """Whether the current subset's job budget ran out, set by _run_subset."""
        self._interrupted = False

    def interrupt(self) -> None:
        """Make the run in hand raise :class:`Interrupted` at its next check, about once
        a second, and every later run at once. Safe to call from a signal handler."""
        self._interrupted = True

    def take_remote_gpus(self) -> str | None:
        """Take the lease on every remote GPU this queue's subsets may use, before a
        job is claimed. None once all are held; else why not, and none is held.
        :meth:`run` refuses to start without them."""
        self._lease_lost = None
        for key in self._remote_gpus():
            lease = RemoteGpuLease(key, self._lose_lease)
            if (busy := lease.take()) is not None:
                self.release_remote_gpus()
                return busy
            self._leases[key] = lease
        return None

    def release_remote_gpus(self) -> None:
        for lease in self._leases.values():
            lease.release()
        self._leases.clear()

    def _remote_gpus(self) -> list[GpuKey]:
        remotes = (
            self._config.simulators[name].remote for name in self._config.remote_simulators()
        )
        return sorted({_gpu_key(remote) for remote in remotes if remote is not None})

    def _lose_lease(self, why: str) -> None:
        """From the lease's watcher thread: the run stops at its next check, about once
        a second, and no backend starts meanwhile."""
        self._lease_lost = why
        _logger.error("%s: stopping the run, since another evaluation can start there", why)

    def run(self, job: QueueJob, run_id: str) -> RunResult:
        """Raises :class:`Interrupted` when told to stop, and :class:`LeaseLost` when a
        remote GPU's lease ends before the run does."""
        if unleased := [key for key in self._remote_gpus() if key not in self._leases]:
            raise RuntimeError(f"no lease on the remote GPUs {unleased}: take_remote_gpus() first")
        run_dir = self._store.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        started_at = utcnow()
        missing = [
            mount.repo for mount in job.weights if not self._store.weights_dir(mount).is_dir()
        ]
        if missing:
            return RunResult.failed(
                run_id,
                job.job_id,
                job.team_id,
                job.submission_id,
                f"declared weights are not in {self._store.weights}: {', '.join(missing)}",
                ErrorCode.INTERNAL_ERROR,
                submitted_at=job.enqueued_at,
                started_at=started_at,
                logs_dir=str(run_dir),
            )
        subsets: dict[str, SubsetResult] = {}
        for subset in self._config.subsets:
            outcome = self._run_subset_guarded(job, subset, run_dir)
            if outcome.status is RunStatus.CANCELLED:
                # The round is over: nothing of this run is scored or shown.
                return RunResult(
                    run_id=run_id,
                    job_id=job.job_id,
                    team_id=job.team_id,
                    submission_id=job.submission_id,
                    status=RunStatus.CANCELLED,
                    submitted_at=job.enqueued_at,
                    started_at=started_at,
                    finished_at=utcnow(),
                    error=outcome.error,
                    logs_dir=str(run_dir),
                )
            subsets[subset.name] = outcome
        if self._lease_lost is not None:
            raise LeaseLost(self._lease_lost)  # in the last subset, after its last check
        for lease in self._leases.values():
            why = lease.confirm()
            if self._interrupted:
                raise Interrupted()  # Slurm's SIGTERM ends the lease's ssh too
            if why is not None:
                raise LeaseLost(why)
        if self._lease_lost is not None:
            raise LeaseLost(self._lease_lost)  # while the host was being asked
        failed = [name for name, result in subsets.items() if result.status != RunStatus.COMPLETED]
        status = RunStatus.COMPLETED
        if failed:
            timed_out = all(subsets[name].status == RunStatus.TIMEOUT for name in failed)
            status = RunStatus.TIMEOUT if timed_out else RunStatus.FAILED
        return RunResult(
            run_id=run_id,
            job_id=job.job_id,
            team_id=job.team_id,
            submission_id=job.submission_id,
            status=status,
            submitted_at=job.enqueued_at,
            started_at=started_at,
            finished_at=utcnow(),
            subsets=subsets,
            error=f"subsets did not complete: {', '.join(failed)}" if failed else None,
            error_code=subsets[failed[0]].error_code if failed else None,
            logs_dir=str(run_dir),
        )

    def _run_subset_guarded(
        self, job: QueueJob, subset: SubsetConfig, run_dir: Path
    ) -> SubsetResult:
        started_at = utcnow()
        self._budget_used_s = None
        self._over_time = False
        leaderboard: dict[str, object] | None = None
        failure: SubsetFailed | None = None
        error: str | None = None
        try:
            self._check_stop()
            leaderboard, failure = self._run_subset(job, subset, run_dir)
            if failure is not None:
                _logger.error("subset %s cut short for %s: %s", subset.name, job.job_id, failure)
                error = f"scored what completed before stopping: {failure}"
        except Interrupted:
            raise
        except SubsetFailed as exc:
            _logger.error("subset %s failed for %s: %s", subset.name, job.job_id, exc)
            failure, error = exc, str(exc)
        except Exception as exc:  # a runner bug must not take the worker down
            _logger.exception("subset %s crashed for %s", subset.name, job.job_id)
            failure = SubsetFailed(f"{type(exc).__name__}: {exc}")
            error = str(failure)
        if (
            failure is not None
            and not isinstance(failure, DeadlinePassed)
            and past_deadline(self._config, failure.at)
        ):
            # It was still going at the deadline, however it ended: no result (§6.4). One
            # that stopped before it keeps its scores, however long the teardown took.
            failure = DeadlinePassed()
            error = str(failure)
        finished_at = utcnow()
        return SubsetResult(
            status=failure.status if failure else RunStatus.COMPLETED,
            leaderboard=leaderboard,
            error=error,
            error_code=failure.code if failure else None,
            budget_used_s=self._budget_used_s,
            over_time=bool(failure and failure.status is RunStatus.TIMEOUT) or self._over_time,
            started_at=started_at,
            finished_at=finished_at,
            duration_s=(finished_at - started_at).total_seconds(),
        )

    def _run_subset(
        self, job: QueueJob, subset: SubsetConfig, run_dir: Path
    ) -> tuple[dict[str, object], SubsetFailed | None]:
        """Returns the gateway's leaderboard, and why the submission stopped early if
        it did. A submission cut short keeps what the gateway scored; only a subset
        that produced no leaderboard at all raises."""
        config = self._config
        subset_dir = run_dir / subset.name
        subset_dir.mkdir(parents=True, exist_ok=True)
        results_path = subset_dir / "gateway-results.json"
        clock_path = subset_dir / CLOCK_STARTED
        finished_path = subset_dir / JOB_FINISHED
        clock_path.unlink(missing_ok=True)
        finished_path.unlink(missing_ok=True)

        taskpack = config.resolve(subset.taskpack)
        manifest, _ = load_pack(taskpack)
        simulators = _simulators_for(manifest, subset)
        if not simulators:
            raise SubsetFailed(f"subset {subset.name!r} selects no tasks from {taskpack}")
        job_timeout_s = manifest.job_timeout_s(simulators)
        lanes = self._lane_caps(simulators)

        socket_root = Path(tempfile.mkdtemp(prefix="capx-", dir=config.socket_dir))
        gateway_dir = socket_root / "gw"
        gateway_dir.mkdir()
        (socket_root / "be").mkdir()
        gateway_sock = gateway_dir / "gateway.sock"
        check_unix_path(str(gateway_sock))
        layout = _Layout(taskpack, pack_dirs(taskpack), socket_root, subset_dir)
        backends = _Backends(config, layout, simulators, self._leases)
        try:
            backends.prepare_remote_hosts()
            backends.copy_remote_packs()
            self._check_stop()
            gateway = self._start_gateway(layout, simulators, lanes, run_dir)
            cut_short: SubsetFailed | None = None
            try:
                startup_s = config.limits.backend_startup_s
                _wait_until(
                    lambda: (
                        gateway_sock.exists()
                        or past_deadline(config)
                        or self._interrupted
                        or self._lease_lost is not None
                    ),
                    gateway,
                    startup_s,
                    0.5,
                    str(gateway_sock),
                )
                self._check_stop()
                try:
                    stopped_at = self._run_submission(
                        job,
                        gateway_dir,
                        subset_dir,
                        job_timeout_s,
                        max_lanes=max(lanes.values()),
                        on_tick=backends.poll,
                        end_job=partial(gateway.popen.send_signal, signal.SIGTERM),
                    )
                except Interrupted:
                    raise
                except SubsetFailed as exc:
                    if exc.code is ErrorCode.SUBMISSION_STARTUP_TIMEOUT:
                        raise  # never reached the gateway, so nothing is scored (§4.3)
                    cut_short, stopped_at = exc, exc.at.timestamp()
                self._budget_used_s, self._over_time = _job_clock(
                    clock_path, stopped_at, job_timeout_s
                )
            finally:
                # Before reading: the gateway writes the authoritative dump on SIGTERM.
                _stop(gateway)
            if cut_short is not None and cut_short.code is ErrorCode.SUBMISSION_STORAGE_EXCEEDED:
                finished = _accepted_finish(results_path, finished_path)
                if finished is not None and finished <= cut_short.at.timestamp():
                    # The finish() came first, though its job-finished file may have come
                    # too late for the runner to see it then.
                    cut_short = None
                    self._budget_used_s, self._over_time = _job_clock(
                        clock_path, finished, job_timeout_s
                    )
            try:
                return _read_leaderboard(results_path), cut_short
            except SubsetFailed:
                if cut_short is not None:
                    raise cut_short from None
                raise
        finally:
            backends.close()
            shutil.rmtree(socket_root, ignore_errors=True)

    def _check_stop(self) -> None:
        if self._interrupted:
            raise Interrupted()
        if self._lease_lost is not None:
            raise LeaseLost(self._lease_lost)
        if past_deadline(self._config):
            raise DeadlinePassed()

    def _lane_caps(self, simulators: list[str]) -> dict[str, int]:
        """Each simulator's lane cap, checking up front that it can be run at all."""
        caps: dict[str, int] = {}
        for simulator in simulators:
            sim = self._config.simulators.get(simulator)
            if sim is None:
                raise SubsetFailed(f"no backend configured for simulator {simulator!r}")
            if sim.image is not None and not self._config.resolve(sim.image).exists():
                raise SubsetFailed(f"env-backend image missing: {self._config.resolve(sim.image)}")
            caps[simulator] = sim.max_lanes
        return caps

    def _start_gateway(
        self, layout: _Layout, simulators: list[str], lanes: dict[str, int], run_dir: Path
    ) -> _Proc:
        config = self._config
        image = config.resolve(config.gateway_image)
        if not image.exists():
            raise SubsetFailed(f"gateway image missing: {image}")
        # One backend process per lane: the factory inside one caches one simulator.
        routes = [
            f"{simulator}=unix:{CONTAINER_SOCKET_DIR}/be/{simulator}-{lane}.sock"
            for simulator in simulators
            for lane in range(lanes[simulator])
        ]
        env = {
            "CAPX_TASKPACK": f"taskpacks/{layout.taskpack.name}",
            "CAPX_MODE": config.mode,
            "CAPX_SIMULATORS": ",".join(simulators),
            "CAPX_UNIX": f"{CONTAINER_SOCKET_DIR}/gw/gateway.sock",
            "CAPX_BACKENDS": ",".join(routes),
            "CAPX_RESULTS_PATH": str(layout.subset_dir / "gateway-results.json"),
            "CAPX_CLOCK_STARTED_PATH": str(layout.subset_dir / CLOCK_STARTED),
            "CAPX_JOB_FINISHED_PATH": str(layout.subset_dir / JOB_FINISHED),
            "CAPX_MASTER_SEED": str(config.master_seed),
            "CAPX_BACKENDS_ON_DEMAND": "1",
            "CAPX_BACKEND_START_TIMEOUT_S": str(config.limits.backend_startup_s),
        }
        return layout.spawn_organizer(
            "gateway", image, "serve_environment.py", env, binds=(f"{run_dir}:{run_dir}",)
        )

    def _run_submission(
        self,
        job: QueueJob,
        gateway_dir: Path,
        subset_dir: Path,
        job_timeout_s: float,
        *,
        max_lanes: int,
        on_tick: Callable[[], None],
        end_job: Callable[[], None],
    ) -> float:
        """Start the team's image with no network and wait for it to exit, call
        ``finish()``, run out of its job budget, or write more than it may. ``on_tick``
        runs about once a second meanwhile; ``end_job`` makes the gateway end the job and
        write its final dump. Returns when the job clock stopped (epoch seconds)."""
        image_path = Path(job.image_path)
        if not image_path.exists():
            raise SubsetFailed(f"submission image missing: {image_path}")
        weight_binds = [
            arg
            for mount in job.weights
            for arg in ("--bind", f"{self._store.weights_dir(mount)}:{mount.path}:ro")
        ]
        outputs = subset_dir / "submission-out"
        outputs.mkdir(parents=True, exist_ok=True)
        home = subset_dir / "submission-home"
        (home / ".cache").mkdir(parents=True, exist_ok=True)
        env_file = _write_env_file(
            subset_dir / "submission.env",
            {
                # The organizer's variables win over the team's.
                **job.env,
                "CAPX_ENV_SOCKET": "/capx/run/gateway.sock",
                "CAPX_OUTPUT_DIR": "/capx/out",
                "XDG_CACHE_HOME": f"{CONTAINER_HOME}/.cache",
                "CAPX_MAX_LANES": str(max_lanes),
                "HF_HUB_OFFLINE": "1",
            },
        )
        command = [
            # No entrypoint means the image's own runscript.
            "singularity", "exec" if job.entrypoint else "run",
            # The isolation: no network stack, nothing of the host filesystem beyond
            # the explicit binds, and a clean environment.
            "--net", "--network=none", "--containall", "--nv",
            "--pwd", job.workdir,
            # Read-only: connecting needs no write, and the directory is the node's /tmp.
            "--bind", f"{gateway_dir}:/capx/run:ro",
            *weight_binds,
            "--bind", f"{outputs}:/capx/out",
            # --home, not HOME=: Singularity refuses to let --env set HOME.
            "--home", f"{home}:{CONTAINER_HOME}",
            "--env-file", str(env_file),
            str(image_path),
            *job.entrypoint,
        ]  # fmt: skip

        proc = _spawn(
            "submission", command, subset_dir / "submission.log", log_cap=SUBMISSION_LOG_BYTES
        )

        def stop_scoring() -> None:
            # Nothing it does once it is over the limit counts: the gateway ends the job
            # at once, and the submission gets no grace period to go on meanwhile.
            end_job()
            proc.popen.kill()

        limits = self._config.limits
        storage = StorageLimit(
            {outputs: "/capx/out", home: CONTAINER_HOME},
            limits.submission_storage_gb,
            limits.submission_storage_files,
            on_exceeded=stop_scoring,
        )

        def over_the_limit() -> SubsetFailed | None:
            """Ended like an exit (§5). Whether a finish() came first is for
            :meth:`_run_subset` to say, once the gateway's final dump is in."""
            if storage.exceeded is None:
                return None
            reason, at = storage.exceeded
            return SubsetFailed(reason, code=ErrorCode.SUBMISSION_STORAGE_EXCEEDED, at=at)

        def tick() -> None:
            self._check_stop()
            if (failure := over_the_limit()) is not None:
                raise failure
            storage.poll()
            on_tick()

        try:
            finished, stopped_at = _wait_for_submission(
                proc,
                subset_dir / CLOCK_STARTED,
                subset_dir / JOB_FINISHED,
                self._config.limits.submission_startup_s,
                job_timeout_s,
                tick,
            )
        finally:
            storage.close()
            _stop(proc)
        if (failure := over_the_limit()) is not None:
            raise failure
        ended = datetime.fromtimestamp(stopped_at, UTC)
        # Ended between two ticks: still cut off if that was after the deadline.
        if past_deadline(self._config, ended):
            raise DeadlinePassed()
        # After finish() the container is ours to stop, so how it exits says nothing.
        if not finished and proc.popen.returncode != 0:
            raise SubsetFailed(
                f"submission exited with code {proc.popen.returncode} (see {proc.log})",
                code=ErrorCode.SUBMISSION_FAILED,
                at=ended,
            )
        return stopped_at


def _timestamp(path: Path) -> float | None:
    """The epoch time the gateway wrote to ``path``, if it did."""
    try:
        return float(path.read_text())
    except (OSError, ValueError):
        return None


def _job_clock(
    clock_path: Path, stopped_at: float, job_timeout_s: float
) -> tuple[float | None, bool]:
    """Time on the job clock, capped at the budget, and whether the budget ran out.
    None and False if the clock never started."""
    started = _timestamp(clock_path)
    if started is None:
        return None, False
    spent = stopped_at - started
    return round(min(spent, job_timeout_s), 1), spent >= job_timeout_s


def _wait_for_submission(
    proc: _Proc,
    clock_path: Path,
    finished_path: Path,
    startup_s: float,
    job_timeout_s: float,
    on_tick: Callable[[], None],
) -> tuple[bool, float]:
    """Wait for the submission to exit; stop it when it calls ``finish()`` or its job
    budget runs out. Returns whether it called ``finish()``, and when its job clock
    stopped: at the ``finish()``, else when it was seen to have exited.

    The budget starts at its first contact with the gateway, and the gateway records
    that in ``clock_path`` and a ``finish()`` in ``finished_path``. Stopping it here,
    rather than waiting on the referee, is what ends a submission that stops calling:
    the referee only sees requests.
    """
    startup_deadline = time.monotonic() + startup_s
    connect_by = time.time() + startup_s  # the same, on the clock the gateway stamps with
    deadline: float | None = None
    while proc.popen.poll() is None:
        # Before the tick: a finish() that came first is not cut off by what the tick finds.
        finished = _timestamp(finished_path)
        if finished is not None:
            _stop(proc)
            if (_timestamp(clock_path) or 0) > connect_by:
                raise _never_connected(startup_s)  # its first contact came too late
            return True, finished
        on_tick()
        started = _timestamp(clock_path) if deadline is None else None
        if started is not None and started > connect_by:
            # Reached the gateway, but after the startup limit, between two looks.
            failure = _never_connected(startup_s)
            _stop(proc)
            raise failure
        if started is not None:
            deadline = time.monotonic() + job_timeout_s - (time.time() - started)
        # Each failure is made before the stop, which can take a while, so it carries
        # the time the limit was reached.
        if deadline is None and time.monotonic() > startup_deadline:
            failure = _never_connected(startup_s)
            _stop(proc)
            raise failure
        if deadline is not None and time.monotonic() > deadline:
            failure = SubsetFailed(
                f"submission used its {job_timeout_s:.0f}s job budget",
                code=ErrorCode.SUBMISSION_TIMEOUT,
            )
            _stop(proc)
            raise failure
        time.sleep(1.0)
    # Exited between two ticks, which can be after the startup limit with no contact,
    # or after a contact that came too late.
    started = _timestamp(clock_path)
    if started is None:
        if time.monotonic() > startup_deadline:
            raise _never_connected(startup_s)
    elif started > connect_by:
        raise _never_connected(startup_s)
    finished = _timestamp(finished_path)
    return finished is not None, finished if finished is not None else utcnow().timestamp()


def _never_connected(startup_s: float) -> SubsetFailed:
    return SubsetFailed(
        f"submission did not reach the gateway within {startup_s:.0f}s",
        code=ErrorCode.SUBMISSION_STARTUP_TIMEOUT,
    )


class _Backends:
    """Starts a subset's simulator backends as the gateway asks for them, and stops
    them all at the end.

    Starting happens off the polling thread: a remote backend spends minutes in ssh,
    and the job deadline has to keep being checked meanwhile. A backend that exits,
    or never comes up, is cleared away (processes stopped, socket removed), so the
    gateway asks for it again on the next episode instead of connecting to a dead
    socket for the rest of the run.
    """

    def __init__(
        self,
        config: QueueConfig,
        layout: _Layout,
        simulators: list[str],
        leases: dict[GpuKey, RemoteGpuLease],
    ) -> None:
        self._config = config
        self._layout = layout
        self._leases = leases
        self._backend_dir = layout.socket_root / "be"
        self._remotes: dict[str, tuple[RemoteBackendConfig, str]] = {}
        for simulator in simulators:
            remote = config.simulators[simulator].remote
            if remote is not None:
                remote_dir = f"{remote.socket_dir}/{layout.socket_root.name}-gpu-{remote.gpu}"
                self._remotes[simulator] = (remote, remote_dir)
        self._procs: list[_Proc] = []
        self._requested: set[str] = set()
        self._running: dict[str, list[_Proc]] = {}
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._generation: dict[str, int] = {}
        """Per backend name, which start is the latest: an older one that fails late
        cleans up only its own processes, not its replacement's."""
        # One thread per lane that may be asked for: each start can wait out the
        # startup limit, and the gateway's own wait begins as soon as it asks.
        lanes = sum(config.simulators[simulator].max_lanes for simulator in simulators)
        self._pool = ThreadPoolExecutor(max_workers=lanes, thread_name_prefix="backend")

    def _remote_dirs(self) -> set[tuple[str, str]]:
        return {(remote.host, remote_dir) for remote, remote_dir in self._remotes.values()}

    def prepare_remote_hosts(self) -> None:
        """Before the job clock can start, so neither a socket path the kernel would
        refuse nor a killed worker's containers cost the team its time."""
        for simulator in self._remotes:
            last_lane = self._config.simulators[simulator].max_lanes - 1
            try:
                check_unix_path(self._remote_sock(simulator, last_lane))
            except ValueError as exc:
                raise SubsetFailed(f"{simulator} backends: {exc}") from None
        for key in sorted({_gpu_key(remote) for remote, _ in self._remotes.values()}):
            _remove_leftovers(self._check_lease(key))

    def _check_lease(self, key: GpuKey) -> RemoteGpuLease:
        """Another evaluation may be using the GPU of a lease this worker lost: its
        containers there are not leftovers, and backends started now would share it."""
        lease = self._leases.get(key)
        if lease is None or not lease.held():
            host, _, gpu = key
            raise LeaseLost(f"this worker does not hold the lease on {host} GPU {gpu}")
        return lease

    def copy_remote_packs(self) -> None:
        """Once, before any backend can ask for it: lanes starting together would
        otherwise rsync --delete into the same directory at once."""
        for host, remote_dir in self._remote_dirs():
            _copy_packs_to_remote(host, self._layout.pack_dirs, remote_dir)

    def poll(self) -> None:
        with self._lock:
            dead = {
                name: (self._generation[name], self._running.pop(name))
                for name, procs in list(self._running.items())
                if any(proc.popen.poll() is not None for proc in procs)
            }
        for want in sorted(self._backend_dir.glob("*.sock.want")):
            name = want.name.removesuffix(".sock.want")
            with self._lock:
                if name in self._requested:
                    continue
                self._requested.add(name)
                generation = self._generation[name] = self._generation.get(name, 0) + 1
            simulator, _, lane = name.rpartition("-")
            self._pool.submit(self._start, simulator, int(lane), generation)
        for name, (generation, procs) in dead.items():
            _logger.warning("the %s backend exited; it starts again when next needed", name)
            self._pool.submit(self._retire, name, procs, generation)

    def close(self) -> None:
        """Stop every backend, then wait for the threads that start them: a start
        stops waiting once the subset is ending, or once its process is gone."""
        self._closing.set()
        self._stop_all()
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._stop_all()  # what a start spawned before it saw the subset end
        for host, remote_dir in self._remote_dirs():
            if _ssh(host, f"rm -rf {shlex.quote(remote_dir)}") != 0:
                _logger.warning("could not remove %s on %s", remote_dir, host)

    def _stop_all(self) -> None:
        with self._lock:
            procs = list(reversed(self._procs))
        for proc in procs:
            _stop(proc)

    def _add(self, name: str, proc: _Proc, mine: list[_Proc]) -> None:
        mine.append(proc)
        with self._lock:
            self._procs.append(proc)
            self._running.setdefault(name, []).append(proc)

    def _retire(self, name: str, procs: list[_Proc], generation: int) -> None:
        for proc in procs:
            _stop(proc)
        with self._lock:
            for proc in procs:
                if proc in self._procs:
                    self._procs.remove(proc)
            if self._generation.get(name) != generation:
                return  # a newer start owns the name, its socket and its request
            sock = self._backend_dir / f"{name}.sock"
            sock.unlink(missing_ok=True)
            sock.with_name(sock.name + ".want").unlink(missing_ok=True)
            self._requested.discard(name)

    def _start(self, simulator: str, lane: int, generation: int) -> None:
        name = f"{simulator}-{lane}"
        mine: list[_Proc] = []
        try:
            if simulator in self._remotes:
                self._start_remote(simulator, lane, mine)
            else:
                backend = self._start_local(simulator, lane)
                self._add(name, backend, mine)
                sock = self._backend_dir / f"{name}.sock"
                startup_s = self._config.limits.backend_startup_s
                _wait_until(sock.exists, backend, startup_s, 0.5, str(sock), self._closing.is_set)
        except Exception:
            # The gateway's wait for this socket times out and voids that episode;
            # clearing the name lets the next one ask for a fresh start.
            if not self._closing.is_set():
                _logger.exception("could not start the %s backend", name)
            with self._lock:
                others = [proc for proc in self._running.pop(name, []) if proc not in mine]
                if others:
                    self._running[name] = others
            self._retire(name, mine, generation)

    def _start_local(self, simulator: str, lane: int) -> _Proc:
        image = self._config.simulators[simulator].image
        assert image is not None
        env = {
            "CAPX_SIMULATOR": simulator,
            "CAPX_BACKEND_UNIX": f"{CONTAINER_SOCKET_DIR}/be/{simulator}-{lane}.sock",
            "CAPX_TASKPACK": f"taskpacks/{self._layout.taskpack.name}",
        }
        return self._layout.spawn_organizer(
            f"env-{simulator}-{lane}",
            self._config.resolve(image),
            "serve_env_backend.py",
            env,
            nv=True,
        )

    def _remote_sock(self, simulator: str, lane: int) -> str:
        return f"{self._remotes[simulator][1]}/{simulator}-{lane}.sock"

    def _start_remote(self, simulator: str, lane: int, mine: list[_Proc]) -> None:
        """Start the backend in Docker on the remote host and forward its socket."""
        remote, remote_dir = self._remotes[simulator]
        lease = self._check_lease(_gpu_key(remote))
        name = f"{simulator}-{lane}"
        remote_sock = self._remote_sock(simulator, lane)
        container = f"{self._layout.socket_root.name}-{name}"
        docker = [
            "docker", "create", "--rm", "--name", container,
            "--label", f"{GPU_LABEL}={remote.gpu}",
            "--gpus", f"device={remote.gpu}",
            "--shm-size", "16g",
            "--network", "none",
            "-e", f"CAPX_SIMULATOR={simulator}",
            "-e", f"CAPX_BACKEND_UNIX={CONTAINER_SOCKET_DIR}/{name}.sock",
            "-v", f"{remote_dir}:{CONTAINER_SOCKET_DIR}",
            "-e", f"CAPX_TASKPACK=taskpacks/{self._layout.taskpack.name}",
        ]  # fmt: skip
        for directory in self._layout.pack_dirs:
            inside = f"{IMAGE_WORKDIR}/taskpacks/{directory.name}"
            docker += ["-v", f"{remote_dir}/taskpacks/{directory.name}:{inside}:ro"]
        for volume in remote.volumes:
            docker += ["-v", volume]
        create = " ".join(
            [
                "mkdir", "-p", "-m", "700", shlex.quote(remote_dir), "&&",
                *(shlex.quote(arg) for arg in docker),
                # Expanded by the remote shell: whoever ssh logs in as owns the socket.
                '-e "CAPX_BACKEND_UNIX_OWNER=$(id -u):$(id -g)"',
                shlex.quote(remote.image), ">/dev/null",
            ]
        )  # fmt: skip
        # Created under the lease, started outside it: a newer lease's cleanup removes
        # a container this one created before it, and refuses one created after.
        remote_command = f"{lease.guard(create)} && exec docker start -a {shlex.quote(container)}"
        log_dir = self._layout.subset_dir
        backend = _spawn(
            f"env-{name}", _ssh_argv(remote.host, remote_command), log_dir / f"env-{name}.log"
        )
        backend.on_stop = partial(_remove_remote_backend, remote.host, container, remote_sock)
        self._add(name, backend, mine)

        # Not the forward: ssh creates its local end at once and dials the remote one
        # per connection. And owned by us: the container binds as root, then hands the
        # 0600 socket over (CAPX_BACKEND_UNIX_OWNER).
        quoted = shlex.quote(remote_sock)
        startup_s = self._config.limits.backend_startup_s
        try:
            _wait_until(
                lambda: _ssh(remote.host, f"test -S {quoted} -a -O {quoted}", timeout_s=30.0) == 0,
                backend,
                startup_s,
                2.0,
                f"{remote.host}:{remote_sock}",
                self._closing.is_set,
            )
        except SubsetFailed:
            if backend.popen.returncode == LEASE_FENCED:
                raise lease.fenced() from None
            raise
        local_sock = self._backend_dir / f"{name}.sock"
        forward = _spawn(
            f"forward-{name}",
            [
                "ssh", *SSH_OPTIONS, "-N",
                "-o", "ExitOnForwardFailure=yes",
                "-L", f"{local_sock}:{remote_sock}",
                remote.host,
            ],
            log_dir / f"forward-{name}.log",
        )  # fmt: skip
        self._add(name, forward, mine)
        _wait_until(
            local_sock.exists, forward, startup_s, 0.5, str(local_sock), self._closing.is_set
        )


def _write_env_file(path: Path, env: dict[str, str]) -> Path:
    """Container environment as a file: Singularity's --env splits its value on
    commas, which silently truncated CAPX_BACKENDS to its first route. Singularity
    reads the file as shell, so values are quoted: unquoted, a space keeps the
    container from starting and a ``$`` expands on the host. A name is written as it
    is, so one that is not a plain variable name is refused here too, whatever let
    it through."""
    unsafe = sorted(key for key in env if not _ENV_NAME.fullmatch(key))
    if unsafe:
        raise SubsetFailed(f"refusing to write environment variable names {unsafe!r}")
    path.write_text("".join(f"{key}={shlex.quote(value)}\n" for key, value in sorted(env.items())))
    return path


def _spawn(name: str, command: list[str], log: Path, *, log_cap: int | None = None) -> _Proc:
    """Start ``command`` with its output in ``log``, which keeps the first and last
    ``log_cap / 2`` bytes when ``log_cap`` is set."""
    log.parent.mkdir(parents=True, exist_ok=True)
    _logger.info("starting %s: %s", name, " ".join(command))
    # Closed right after the spawn: the child holds its own dup.
    with log.open("wb") as handle:
        handle.write(f"$ {' '.join(command)}\n\n".encode())
        handle.flush()
        popen = subprocess.Popen(
            command,
            stdout=handle if log_cap is None else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=_host_env(),
            # Its own process group, so a stray SIGINT to the worker does not race
            # the orderly teardown that writes the results dump.
            start_new_session=True,
        )
    proc = _Proc(name=name, popen=popen, log=log)
    if log_cap is not None:
        assert popen.stdout is not None
        proc.copier = threading.Thread(
            target=_copy_capped, args=(popen.stdout, log, log_cap), daemon=True, name=f"log-{name}"
        )
        proc.copier.start()
    return proc


def _copy_capped(source: IO[bytes], log: Path, cap: int) -> None:
    """Append ``source`` to ``log`` until it closes, keeping its first and last
    ``cap / 2`` bytes. What goes past the first half is held here until the end."""
    half = cap // 2
    written = left_out = held = 0
    tail: deque[bytes] = deque()
    with source, log.open("ab") as out:
        while chunk := os.read(source.fileno(), 1 << 16):
            if written < half:
                head, chunk = chunk[: half - written], chunk[half - written :]
                out.write(head)
                out.flush()
                written += len(head)
            if not chunk:
                continue
            tail.append(chunk)
            held += len(chunk)
            while held - len(tail[0]) >= half:
                dropped = tail.popleft()
                held -= len(dropped)
                left_out += len(dropped)
        if left_out:
            out.write(f"\n[capx: {left_out} bytes of output left out here]\n".encode())
        out.writelines(tail)


def _accepted_finish(results_path: Path, finished_path: Path) -> float | None:
    """When the gateway took the submission's ``finish()`` (epoch seconds), from its
    final dump; from the job-finished file it wrote then, for a gateway whose dump does
    not say. None if it took none."""
    try:
        document = json.loads(results_path.read_text())
    except (OSError, ValueError):
        document = None
    if isinstance(document, dict) and "finished_at" in document:
        finished = document["finished_at"]
        return float(finished) if isinstance(finished, int | float) else None
    return _timestamp(finished_path)


def _read_leaderboard(results_path: Path) -> dict[str, object]:
    if not results_path.exists():
        raise SubsetFailed(f"gateway wrote no results to {results_path}", code=ErrorCode.NO_SCORE)
    try:
        document = json.loads(results_path.read_text())
    except ValueError as exc:
        raise SubsetFailed(f"gateway results at {results_path} are not valid JSON") from exc
    leaderboard = document.get("leaderboard")
    if not isinstance(leaderboard, dict):
        raise SubsetFailed(
            f"gateway results at {results_path} carry no leaderboard", code=ErrorCode.NO_SCORE
        )
    return leaderboard


__all__ = ["DeadlinePassed", "EvalRunner", "Interrupted", "LeaseLost", "SubsetFailed"]
