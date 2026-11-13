"""Deployment config for the eval queue: where this cluster keeps things and how
long it waits on processes to start.

The competition's rules are elsewhere: a round's job budget, trial count and step
limit are its task pack's ``limits``, and action bounds are
:class:`capx.config.EvalConfig`. The defaults below are this cluster's; a
``--config`` file sets only what differs from them.
"""

from __future__ import annotations

import posixpath
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from capx.evalqueue.models import validate_id, validate_repo

_REPO_ROOT = Path(__file__).resolve().parents[2]


class SubsetConfig(BaseModel):
    """One task pack a submission is scored on."""

    model_config = ConfigDict(extra="forbid")

    name: str
    taskpack: str
    simulators: list[str] = Field(default_factory=list)
    """Restrict the pack to these simulators; empty runs every task in it."""


class WatcherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    poll_interval_s: float = 15.0
    """Paces the worker's claim loop. The watcher runs on `hub.poll_interval_s`."""
    max_active_per_team: int = 1
    """How many of a team's submissions may be waiting or running at once; 0 disables
    the cap. A new submission replaces the team's oldest *waiting* one: that job is
    cancelled (a `cancelled` result says by what) and the new one takes its place. A
    *running* job is never cancelled, so while the team is at the cap with running
    jobs the new submission just waits: it is neither fetched nor rejected, and the
    next poll looks again. `evalq submit` bypasses the cap."""
    max_weights_gb: float = 300.0
    """Cap on the total size of the model repos one submission declares, in GiB, as a
    guard on shared storage. 0 disables it."""


class HubTeam(BaseModel):
    """One team's registration: its submission repo on HuggingFace, whose ``main`` the
    watcher follows. A team writes this itself (capx/evalqueue/teams.py)."""

    model_config = ConfigDict(extra="forbid")

    team_id: str
    repo: str
    """``<org>/<name>``. The org is what identifies the team."""

    @field_validator("team_id")
    @classmethod
    def _team_id_is_safe(cls, value: str) -> str:
        return validate_id(value, "team_id")

    @field_validator("repo")
    @classmethod
    def _repo_is_owner_and_name(cls, value: str) -> str:
        return validate_repo(value, "submission")


class HubConfig(BaseModel):
    """Intake from HuggingFace — see PRELIM_RULEBOOK.md §6."""

    model_config = ConfigDict(extra="forbid")

    repo_type: Literal["model", "dataset"] = "model"
    token_env: str = "CAPX_HF_TOKEN"
    """Environment variable holding the organizer's token. Falls back to the
    ambient HuggingFace credentials."""
    poll_interval_s: float = 60.0


class TeamsConfig(BaseModel):
    """Where the team registry is: one file per team in a git repository, added by the
    team's pull request (capx/evalqueue/teams.py)."""

    model_config = ConfigDict(extra="forbid")

    repo: str = ""
    """A git URL or a local repository. Unset, no team is registered."""
    ref: str = "main"
    path: str = "teams/prelim"
    """The directory of ``<team_id>.yaml`` files in ``repo``."""


class LeaderboardConfig(BaseModel):
    """Where ``evalq publish`` writes the public leaderboard and queue snapshot.

    Unset ``repo`` leaves publishing off. The snapshot is one small JSON file next to
    the static page (web/leaderboard/index_jp.html, index_en.html) in a HuggingFace static Space.
    """

    model_config = ConfigDict(extra="forbid")

    repo: str = ""
    """``<owner>/<name>`` of the repo that holds ``leaderboard.json``."""
    repo_type: Literal["model", "dataset", "space"] = "space"
    path: str = "leaderboard.json"
    token_env: str = "CAPX_HF_LEADERBOARD_TOKEN"
    """Environment variable holding a token that can write to ``repo`` only. Falls
    back to the ambient HuggingFace login. Distinct from the intake token, which
    must stay read-only."""
    interval_s: float = Field(default=60.0, ge=60.0)
    """At most one commit per interval. The Hub refuses a repo's 129th commit in an
    hour, and a busy queue changes the snapshot on nearly every 30 s look."""
    heartbeat_s: float = 1800.0
    """Push at least this often even when nothing changed, so the page can tell a quiet
    queue from a publisher that stopped. Every push is a commit to ``repo``."""
    subset_labels: dict[str, str] = Field(default_factory=dict)
    """Subset name -> the name the page shows, so a subset can be renamed for readers
    without renaming it in results, run directories and queue records."""
    activity_per_team: int = 3
    """How many of a team's latest submissions the submission history lists before
    "Show more". The snapshot carries every submission."""


class QueueLimits(BaseModel):
    """How long the queue waits on processes to start, and how much a submission may
    write (PRELIM_RULEBOOK.md §5)."""

    model_config = ConfigDict(extra="forbid")

    submission_startup_s: float = Field(default=21600.0, gt=0)
    """From starting a submission to its first contact with the gateway, where its
    job budget starts."""
    backend_startup_s: float = Field(default=600.0, gt=0)
    """For a simulator backend's or the gateway's socket to appear."""
    submission_storage_gb: float = Field(default=20.0, gt=0)
    """What the submission may write to ``/capx/out`` and HOME together, in GiB. Both
    are on the shared root, so a submission over it is stopped (capx/evalqueue/scratch.py)."""
    submission_storage_files: int = Field(default=100_000, gt=0)
    """Files and directories it may create there."""


class RemoteBackendConfig(BaseModel):
    """A simulator whose backend runs in Docker on another machine, reached over ssh.

    BEHAVIOR needs Isaac Sim, which does not run on the B300 nodes. The runner
    forwards the remote socket into the run's backend directory, so the gateway sees
    a local backend and the submission still cannot see it.
    """

    model_config = ConfigDict(extra="forbid")

    host: str
    """ssh destination, as the worker node's ssh config knows it."""
    image: str = "capx-env-behavior:local"
    socket_dir: str = "/tmp"
    """Parent of the per-run socket directory on the remote host, absolute. Same
    107-byte AF_UNIX limit as ``QueueConfig.socket_dir``."""
    gpu: str = Field(default="0", pattern=r"^(0|[1-9][0-9]*)$")
    """One device by its index, passed as ``--gpus device=<gpu>``. Evaluations take
    turns on it: a worker holds a lock file in ``socket_dir`` named for the index while
    it evaluates a job, so every queue on the host names its GPUs by index and shares
    ``socket_dir``."""
    volumes: list[str] = Field(default_factory=list)
    """Extra ``docker run -v`` specs, e.g. the asset volume at ``/data``."""

    @field_validator("socket_dir")
    @classmethod
    def _one_spelling_per_socket_dir(cls, value: str) -> str:
        # The GPU's lock file is found by it: "/tmp" and "/tmp/" would be two leases,
        # and a worker holding one would wait on the other for ever.
        if not value.startswith("/"):
            raise ValueError("socket_dir must be an absolute path")
        if ".." in value.split("/"):
            # The host resolves it through symlinks, which normpath cannot see.
            raise ValueError("socket_dir must not contain '..'")
        return "/" + posixpath.normpath(value).lstrip("/")


class SimulatorConfig(BaseModel):
    """How one simulator's backends run, and how many may run at once."""

    model_config = ConfigDict(extra="forbid")

    image: str | None = None
    """Env-backend SIF, run on the worker node."""
    remote: RemoteBackendConfig | None = None
    """Or a backend on another machine, reached over ssh."""
    max_lanes: int = Field(ge=1)
    """Episodes of this simulator one submission may run at once, each in its own
    backend process started when first needed."""

    @model_validator(mode="after")
    def _one_way_to_run(self) -> SimulatorConfig:
        if (self.image is None) == (self.remote is None):
            raise ValueError("a simulator needs exactly one of image or remote")
        return self


class RecordingsConfig(BaseModel):
    """Videos of a few episodes of each run, for the organizers only (docs/operations.md)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    """``capx evalq worker --no-recordings`` turns it off for that worker."""
    per_simulator: int = Field(default=10, ge=1)
    """Episodes filmed per simulator in each run, chosen at random."""
    keep_days: float = Field(default=7.0, gt=0)
    """A run's videos are deleted this long after the run."""


class SlurmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    partition: str = ""
    """Empty resolves to ``part-<primary group>``, this cluster's convention."""
    gres: str = "gpu:1"
    cpus_per_task: int = 32
    mem: str = "512G"
    time_limit: str = ""
    """Empty submits without --time (unlimited where the partition allows it)."""
    job_name: str = "capx-evalq-worker"
    workers: int = Field(default=2, ge=1)
    """How many worker jobs to keep alive, i.e. how many submissions are evaluated
    at once, one GPU each. Must be 1 when a subset may use a ``remote`` simulator:
    its GPU serves one evaluation at a time, so a second worker would only wait."""


class QueueConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: str = "~/shared-storage/submission"
    socket_dir: str = "/tmp"
    """Parent of the per-run socket directory. Must be node-local and short: an
    AF_UNIX path is capped at 107 bytes."""
    mode: Literal["dev", "official"] = "official"
    master_seed: int = Field(default=0, ge=0)
    """Every episode's seed is derived from it (capx/gateway/referee.py), and the rest of
    the derivation is public, so in ``official`` mode it must be set: a fresh secret per
    round (docs/operations.md §3). The default is for ``dev`` only."""
    deadline: datetime | None = None
    """When the round closes (capx/evalqueue/deadline.py). Unset, it never does."""
    limits: QueueLimits = Field(default_factory=QueueLimits)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)
    hub: HubConfig = Field(default_factory=HubConfig)
    teams: TeamsConfig = Field(default_factory=TeamsConfig)
    leaderboard: LeaderboardConfig = Field(default_factory=LeaderboardConfig)
    recordings: RecordingsConfig = Field(default_factory=RecordingsConfig)
    subsets: list[SubsetConfig] = Field(
        min_length=1,
        default_factory=lambda: [
            SubsetConfig(
                name="original", taskpack="taskpacks/original", simulators=["robosuite", "libero"]
            )
        ],
    )
    """Every submission is scored on each of these packs, in this order. The default
    leaves out BEHAVIOR, which needs a ``remote`` simulator (docs/operations.md)."""
    gateway_image: str = "singularity/images/capx-gateway.sif"
    simulators: dict[str, SimulatorConfig] = Field(
        default_factory=lambda: {
            "robosuite": SimulatorConfig(
                image="singularity/images/capx-env-robosuite.sif", max_lanes=32
            ),
            "libero": SimulatorConfig(image="singularity/images/capx-env-libero.sif", max_lanes=32),
        }
    )
    """simulator id -> how its backends run. The lane caps are fixed for a round and
    published with it, because the job budget is wall clock."""
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)

    @field_validator("deadline")
    @classmethod
    def _deadline_has_a_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("deadline needs a timezone, e.g. 2026-12-13T12:00:00+09:00")
        return value

    @field_validator("subsets")
    @classmethod
    def _subset_names_are_unique(cls, value: list[SubsetConfig]) -> list[SubsetConfig]:
        # Results and run directories are keyed by name.
        names = [subset.name for subset in value]
        if duplicated := sorted({name for name in names if names.count(name) > 1}):
            raise ValueError(f"subset names must be unique: {', '.join(duplicated)}")
        return value

    @model_validator(mode="after")
    def _official_mode_has_its_own_seed(self) -> QueueConfig:
        if self.mode == "official" and "master_seed" not in self.model_fields_set:
            raise ValueError(
                "master_seed is required in official mode: set it to a fresh secret, e.g. "
                "python3 -c 'import secrets; print(secrets.randbits(63))'"
            )
        return self

    @model_validator(mode="after")
    def _one_evaluation_on_a_remote_gpu(self) -> QueueConfig:
        if (used := self.remote_simulators()) and self.slurm.workers != 1:
            raise ValueError(
                f"slurm.workers must be 1 when a subset uses a remote simulator "
                f"({', '.join(sorted(used))}): its GPU is not shared between evaluations "
                "(docs/operations.md §6)"
            )
        return self

    def remote_simulators(self) -> set[str]:
        """The ``remote`` simulators a subset may use. A subset without a simulator
        list runs whatever its pack holds, so it may use any of them."""
        remote = {name for name, sim in self.simulators.items() if sim.remote is not None}
        return {name for subset in self.subsets for name in subset.simulators or remote} & remote

    def resolve(self, path: str) -> Path:
        """Resolve a config path: absolute and ``~`` as written, else repo-relative."""
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else (_REPO_ROOT / candidate)


def load_queue_config(path: str | None = None) -> QueueConfig:
    """The defaults above, with ``path``'s settings over them."""
    if path is None:
        return QueueConfig()
    return QueueConfig.model_validate(yaml.safe_load(Path(path).read_text()) or {})


__all__ = [
    "HubConfig",
    "HubTeam",
    "LeaderboardConfig",
    "QueueConfig",
    "QueueLimits",
    "RecordingsConfig",
    "RemoteBackendConfig",
    "SimulatorConfig",
    "SlurmConfig",
    "SubsetConfig",
    "TeamsConfig",
    "WatcherConfig",
    "load_queue_config",
]
