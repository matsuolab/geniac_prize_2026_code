"""The supervisor: how many workers exist, and whether a replacement can pick up
the claims the one it replaces left behind."""

from __future__ import annotations

import pytest

from capx.evalqueue import slurm
from capx.evalqueue.config import QueueConfig


class FakeSbatch:
    """Records submissions and reports the jobs it has 'started'."""

    def __init__(self, repo_root) -> None:
        self.submitted: list[list[str]] = []
        self.envs: list[dict[str, str] | None] = []
        self.running: set[str] = set()
        self.squeue_fails = False
        self._repo_root = repo_root

    def submit(self, command, capture_output, text, check, cwd=None, env=None):  # noqa: ANN001
        self.submitted.append(command)
        self.envs.append(env)
        name = next(
            token.split("=", 1)[1] for token in command if token.startswith("--job-name=")
        )
        self.running.add(name)
        return _Completed(stdout=f"{1000 + len(self.submitted)}\n")

    def squeue(self, command, capture_output, text, check):  # noqa: ANN001
        if self.squeue_fails:
            return _Completed(stdout="", returncode=1)
        name = command[command.index("-n") + 1]
        return _Completed(stdout="12345\n" if name in self.running else "")


class _Completed:
    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "scripts" / "evalq" / "worker.sbatch"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/bash\n")
    sbatch = FakeSbatch(tmp_path)

    def run(command, **kwargs):
        return sbatch.submit(command, **kwargs) if command[0] == "sbatch" else sbatch.squeue(
            command, **kwargs
        )

    monkeypatch.setattr(slurm.subprocess, "run", run)
    monkeypatch.setattr(slurm, "default_partition", lambda: "part-test")
    return sbatch, tmp_path


def test_it_starts_one_worker_per_slot(fake) -> None:
    sbatch, repo = fake
    config = QueueConfig(master_seed=1)
    config.slurm.workers = 4

    slurm.supervise(config, repo, once=True)

    assert len(sbatch.submitted) == 4
    names = sorted(sbatch.running)
    assert names == [f"capx-evalq-worker-{i}" for i in range(4)]


def test_it_only_replaces_the_slots_that_are_gone(fake) -> None:
    sbatch, repo = fake
    config = QueueConfig(master_seed=1)
    config.slurm.workers = 3
    slurm.supervise(config, repo, once=True)
    sbatch.submitted.clear()

    sbatch.running.remove("capx-evalq-worker-1")  # that worker died
    slurm.supervise(config, repo, once=True)

    assert len(sbatch.submitted) == 1
    assert "--job-name=capx-evalq-worker-1" in sbatch.submitted[0]


def test_a_replacement_carries_the_slot_id_so_it_can_recover_claims(fake) -> None:
    """A worker only requeues claims left under its own id, so the id has to be the
    slot's rather than the Slurm job's — otherwise a crashed worker's job sits in
    `claimed` until an operator notices."""
    sbatch, repo = fake
    config = QueueConfig(master_seed=1)
    config.slurm.workers = 2

    slurm.supervise(config, repo, once=True)

    for command in sbatch.submitted:
        name = next(t.split("=", 1)[1] for t in command if t.startswith("--job-name="))
        assert command[command.index("--worker-id") + 1] == name


def test_the_config_path_is_passed_through(fake) -> None:
    sbatch, repo = fake
    config = QueueConfig(master_seed=1)

    slurm.supervise(config, repo, config_path="/etc/capx/queue.yaml", once=True)

    command = sbatch.submitted[0]
    assert command[command.index("--config") + 1] == "/etc/capx/queue.yaml"


def test_a_failed_squeue_starts_no_worker(fake) -> None:
    """An empty answer from a failed squeue is not an empty queue: a second job for a
    live slot would take back the job that slot is running."""
    sbatch, repo = fake
    config = QueueConfig(master_seed=1)
    sbatch.squeue_fails = True

    slurm.supervise(config, repo, once=True)

    assert sbatch.submitted == []


def test_a_worker_job_gets_none_of_the_supervisors_secrets(fake, monkeypatch) -> None:
    """sbatch itself runs with only WORKER_ENV and exports all of that. Naming the
    variables in --export would also load the login environment on the node."""
    sbatch, repo = fake
    monkeypatch.setenv("CAPX_HF_TOKEN", "hf_organizer")
    monkeypatch.setenv("CAPX_EVALQ_PYTHON", "/venv/bin/python")
    slurm.supervise(QueueConfig(master_seed=1), repo, once=True)

    assert "--export=ALL" in sbatch.submitted[0]
    env = sbatch.envs[0]
    assert env is not None and set(env) <= set(slurm.WORKER_ENV)
    assert env["CAPX_EVALQ_PYTHON"] == "/venv/bin/python"
    assert not any("TOKEN" in name for name in slurm.WORKER_ENV)
