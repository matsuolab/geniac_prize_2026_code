"""Keep ``slurm.workers`` eval workers queued or running in Slurm.

Slurm only runs the workers; job selection and per-team fairness belong to
:mod:`capx.evalqueue.queue`. Liveness is checked with ``squeue``, the one query an
ordinary user may run on this cluster.
"""

from __future__ import annotations

import grp
import logging
import os
import subprocess
import time
from pathlib import Path

from capx.evalqueue.config import QueueConfig

_logger = logging.getLogger(__name__)


WORKER_ENV = ("PATH", "HOME", "LANG", "CAPX_EVALQ_PYTHON", "CAPX_SINGULARITY_MODULE")
"""All of the supervisor's environment a worker job gets: sbatch runs with only these,
and ``--export=ALL``. The supervisor runs where the operator keeps the intake and
leaderboard tokens, and a worker needs neither. Naming the variables in ``--export``
instead would make Slurm load the login environment on the node as well (as with
``--get-user-env``), and login files can export tokens."""


class SlurmError(RuntimeError):
    pass


def default_partition() -> str:
    """This cluster names a group's partition ``part-<primary group>``."""
    return f"part-{grp.getgrgid(os.getgid()).gr_name}"


def slot_name(job_name: str, slot: int) -> str:
    """Name of worker slot ``slot``, which is also its worker id: a replacement
    carries the same id, so it recovers the claims the old job left behind."""
    return f"{job_name}-{slot}"


def submit_worker(
    config: QueueConfig,
    repo_root: Path,
    config_path: str | None = None,
    slot: int = 0,
    *,
    recordings: bool = True,
) -> str:
    """sbatch the worker job for one slot and return its job id."""
    slurm = config.slurm
    script = repo_root / "scripts" / "evalq" / "worker.sbatch"
    if not script.is_file():
        raise SlurmError(f"worker script missing: {script}")
    name = slot_name(slurm.job_name, slot)
    command = [
        "sbatch",
        "--parsable",
        f"--partition={slurm.partition or default_partition()}",
        f"--gres={slurm.gres}",
        f"--cpus-per-task={slurm.cpus_per_task}",
        f"--mem={slurm.mem}",
        f"--job-name={name}",
        "--export=ALL",
    ]
    if slurm.time_limit:
        command.append(f"--time={slurm.time_limit}")
    command.append(str(script))
    command.extend(["--worker-id", name])
    if config_path:
        command.extend(["--config", config_path])
    if not recordings:
        command.append("--no-recordings")

    env = {key: os.environ[key] for key in WORKER_ENV if key in os.environ}
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=repo_root, env=env, check=False
    )
    if result.returncode != 0:
        raise SlurmError(f"sbatch failed: {result.stderr.strip() or result.stdout.strip()}")
    job_id = result.stdout.strip().split(";")[0]
    _logger.info("submitted worker job %s", job_id)
    return job_id


def find_worker_jobs(job_name: str) -> list[str]:
    """Ids of this user's queued/running jobs with ``job_name``. Raises SlurmError when
    squeue fails: its empty output then does not mean there are none."""
    result = subprocess.run(
        ["squeue", "-h", "--me", "-n", job_name, "-o", "%i"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SlurmError(f"squeue failed: {result.stderr.strip() or result.stdout.strip()}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def supervise(
    config: QueueConfig,
    repo_root: Path,
    *,
    config_path: str | None = None,
    interval_s: float = 60.0,
    once: bool = False,
    recordings: bool = True,
) -> None:
    """Keep ``slurm.workers`` worker jobs queued or running."""
    while True:
        for slot in range(config.slurm.workers):
            try:
                if find_worker_jobs(slot_name(config.slurm.job_name, slot)):
                    continue
                submit_worker(config, repo_root, config_path, slot, recordings=recordings)
            except SlurmError:
                _logger.exception("worker slot %d: trying again next tick", slot)
        if once:
            return
        time.sleep(interval_s)


__all__ = [
    "WORKER_ENV",
    "SlurmError",
    "default_partition",
    "find_worker_jobs",
    "submit_worker",
    "supervise",
]
