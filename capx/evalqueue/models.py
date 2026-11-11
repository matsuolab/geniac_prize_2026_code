"""Schemas for the documents the eval queue passes between its processes.

- :class:`SubmissionManifest` — written by a team. Untrusted: the ids end up in
  filesystem paths and on the board, so they are charset-restricted.
- :class:`QueueJob` — written by the watcher (or ``submit``), read by the worker.
- :class:`RunResult` — written by the worker, or by the watcher on rejection; what
  the board is built from.
"""

from __future__ import annotations

import enum
import re
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def validate_id(value: str, what: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(
            f"{what} {value!r} must be 1-64 chars of [A-Za-z0-9._-] and start "
            "alphanumeric (it is used as a directory name)"
        )
    return value


_HF_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

GB = 1024**3
"""The unit of the image and weights limits: a GiB, as PRELIM_RULEBOOK.md §6.3 states them."""

WEIGHTS_MOUNT_ROOT = PurePosixPath("/models")
"""Every declared weights path sits under this, so a mount cannot cover the image's
own tree (/usr, the checkout) or the organizer's mounts under /capx."""


def is_commit_sha(value: str) -> bool:
    return bool(_COMMIT_SHA.fullmatch(value))


def hf_owner(repo: str) -> str:
    """The user or org half of ``<owner>/<name>``, compared case-insensitively as the
    Hub does."""
    return repo.split("/", 1)[0].casefold()


def validate_repo(value: str, what: str) -> str:
    owner, _, name = value.partition("/")
    if (
        not (_HF_NAME.fullmatch(owner) and _HF_NAME.fullmatch(name))
        or "--" in value
        or ".." in value
    ):
        raise ValueError(f"{what} repo {value!r} must be a HuggingFace '<owner>/<name>'")
    return value


def describe_errors(exc: ValidationError) -> str:
    """A validation error without the input it was given, which can be anything a team
    pushed."""
    return "; ".join(
        f"{'.'.join(map(str, error['loc'])) or 'file'}: {error['msg']}"
        for error in exc.errors(include_url=False, include_input=False)
    )


def _check_revision(value: str, what: str) -> str:
    if not is_commit_sha(value):
        raise ValueError(
            f"{what} revision {value!r} must be a full 40-character commit sha, not a branch or tag"
        )
    return value


class ImageRef(BaseModel):
    """The submission image: a file in one of the team's own repos, pinned to a commit.

    A manifest cannot name the commit it is itself part of, so an image in the
    submission repo is pushed first and named by that commit.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str
    revision: str
    path: str

    @field_validator("repo")
    @classmethod
    def _repo_is_owner_and_name(cls, value: str) -> str:
        return validate_repo(value, "image")

    @field_validator("revision")
    @classmethod
    def _revision_is_a_commit(cls, value: str) -> str:
        return _check_revision(value, "image")

    @field_validator("path")
    @classmethod
    def _path_stays_in_the_repo(cls, value: str) -> str:
        if not value or value.startswith("/") or ".." in value.split("/"):
            raise ValueError(f"image path {value!r} must be a relative path inside the repo")
        return value


class WeightsMount(BaseModel):
    """One HuggingFace model repo the submission needs, mounted read-only at ``path``
    (capx/evalqueue/weights.py)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str
    revision: str
    """Full commit sha, so the bytes cannot change between acceptance and the run."""
    path: str
    gated: bool = False
    """The repo is gated on HuggingFace. A gated repo that is not declared is refused:
    the declaration is how an organizer learns there is an access request to make."""

    @field_validator("repo")
    @classmethod
    def _repo_is_owner_and_name(cls, value: str) -> str:
        return validate_repo(value, "weights")

    @field_validator("revision")
    @classmethod
    def _revision_is_a_commit(cls, value: str) -> str:
        return _check_revision(value, "weights")

    @field_validator("path")
    @classmethod
    def _path_is_under_the_mount_root(cls, value: str) -> str:
        path = PurePosixPath(value)
        under_root = path.is_relative_to(WEIGHTS_MOUNT_ROOT)
        names = path.relative_to(WEIGHTS_MOUNT_ROOT).parts if under_root else ()
        # Compared as a string too: PurePosixPath quietly drops a trailing slash and
        # collapses "//", and the value goes verbatim into a --bind spec, where ':'
        # and ',' are separators.
        if not names or str(path) != value or not all(_HF_NAME.fullmatch(name) for name in names):
            raise ValueError(
                f"weights path {value!r} must be {WEIGHTS_MOUNT_ROOT}/<name>[/<name>...], "
                "each name [A-Za-z0-9._-]"
            )
        return value


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

ORGANIZER_ENV = frozenset(
    {
        "CAPX_ENV_SOCKET",
        "CAPX_OUTPUT_DIR",
        "CAPX_MAX_LANES",
        "HF_HUB_OFFLINE",
        "HOME",
        "XDG_CACHE_HOME",
    }
)
"""What the runner sets in the submission's container (capx/evalqueue/runner.py)."""

_RUNTIME_ENV_PREFIXES = ("SINGULARITY", "APPTAINER")

_UNSAFE_IN_ENV = frozenset("$`\\\n\r\0")
"""Singularity reads the env file as shell on the host, with the worker's environment:
``$`` would expand the organizer's variables into the container."""


class RunSpec(BaseModel):
    """What a submission asks of its run, apart from where its image comes from
    (``evalq submit`` takes the image from the operator's disk instead)."""

    # A manifest can be anything a team pushed; an error that repeats it can be huge.
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    team_id: str
    submission_id: str
    image_sha256: str | None = None
    entrypoint: list[str] = Field(default_factory=list)
    """Command to run inside the image. Empty runs the image's own runscript."""
    workdir: str = "/workspace/capx"
    """Working directory inside the image; the default is the baseline's checkout."""
    weights: list[WeightsMount] = Field(default_factory=list)
    """Model repos to mount into the container."""
    env: dict[str, str] = Field(default_factory=dict)
    """Extra environment for the container. It may not set the organizer's variables."""
    notes: str = ""

    @field_validator("team_id")
    @classmethod
    def _team_id_is_safe(cls, value: str) -> str:
        return validate_id(value, "team_id")

    @field_validator("submission_id")
    @classmethod
    def _submission_id_is_safe(cls, value: str) -> str:
        return validate_id(value, "submission_id")

    @field_validator("env")
    @classmethod
    def _env_is_the_team_s_own_and_plain(cls, value: dict[str, str]) -> dict[str, str]:
        for name, setting in value.items():
            if not _ENV_NAME.fullmatch(name):
                raise ValueError(
                    f"env name {name!r} must be letters, digits and _, not starting with a digit"
                )
            if name in ORGANIZER_ENV or name.startswith(_RUNTIME_ENV_PREFIXES):
                raise ValueError(f"env {name} is set by the organizers or the container runtime")
            if _UNSAFE_IN_ENV.intersection(setting):
                raise ValueError(
                    f"env {name} may not contain $, a backtick, a backslash or a line break"
                )
        return value

    @model_validator(mode="after")
    def _weights_paths_do_not_overlap(self) -> RunSpec:
        paths = sorted(PurePosixPath(mount.path) for mount in self.weights)
        for outer, inner in pairwise(paths):
            if inner.is_relative_to(outer):
                raise ValueError(f"weights paths {str(outer)!r} and {str(inner)!r} overlap")
        return self


class SubmissionManifest(RunSpec):
    """``submission.yaml``, pushed by a team to its registered submission repo."""

    image: ImageRef


class QueueJob(BaseModel):
    """One unit of work: evaluate one submission over every configured subset."""

    # Ignore, not forbid: records outlive code changes, and a pending or claimed
    # record with a field since removed must still be run or recovered.
    model_config = ConfigDict(extra="ignore")

    schema_version: Literal[1] = 1
    job_id: str
    team_id: str
    submission_id: str
    image_path: str
    entrypoint: list[str] = Field(default_factory=list)
    workdir: str = "/workspace/capx"
    env: dict[str, str] = Field(default_factory=dict)
    weights: list[WeightsMount] = Field(default_factory=list)
    hub_repo: str | None = None
    hub_revision: str | None = None
    """Where the image came from, so a result can be traced back to its commit."""
    enqueued_at: datetime
    claimed_at: datetime | None = None
    """When a worker last took it, i.e. when its current run started."""
    attempts: int = 0
    """Incremented each time a worker claims it."""


class RunStatus(str, enum.Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    """Never ran: the watcher refused the submission (bad manifest, oversized, ...)."""
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    """Never ran: a newer submission from the same team replaced it while it waited."""


class ErrorCode(str, enum.Enum):
    """Why a submission did not score, in terms safe to show everyone.

    The ``error`` strings beside these carry paths, hosts and exception text, so
    they stay on the cluster; this is the only part of a failure that leaves it.
    """

    INVALID_MANIFEST = "invalid_manifest"
    IMAGE_NOT_FOUND = "image_not_found"
    IMAGE_TOO_LARGE = "image_too_large"
    IMAGE_HASH_MISMATCH = "image_hash_mismatch"
    FOREIGN_REPO = "foreign_repo"
    """The image or weights come from an org that is not the team's own."""
    REPO_UNREADABLE = "repo_unreadable"
    WEIGHTS_GATED = "weights_gated"
    WEIGHTS_TOO_LARGE = "weights_too_large"
    SUBMISSION_FAILED = "submission_failed"
    SUBMISSION_TIMEOUT = "submission_timeout"
    SUBMISSION_STARTUP_TIMEOUT = "submission_startup_timeout"
    """Never reached the gateway within ``limits.submission_startup_s``: not scored."""
    SUBMISSION_STORAGE_EXCEEDED = "submission_storage_exceeded"
    """Wrote more than ``limits.submission_storage_*`` allow: stopped, and scored like a
    submission that exited at that moment."""
    NO_SCORE = "no_score"
    INTERNAL_ERROR = "internal_error"
    """Our side failed: an image, a backend, the runner itself."""


class Rejected(ValueError):
    """The submission will not be queued, and the team is told why."""

    def __init__(self, code: ErrorCode, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.transient = transient
        """The Hub failed, not the submission: worth asking again before telling the team."""


class SubsetResult(BaseModel):
    """Outcome of one subset (one task pack) for one submission."""

    model_config = ConfigDict(extra="forbid")

    status: RunStatus
    leaderboard: dict[str, object] | None = None
    """Verbatim ``build_leaderboard`` output; None when the subset did not finish."""
    error: str | None = None
    error_code: ErrorCode | None = None
    budget_used_s: float | None = None
    """Time on the referee's job clock: from the submission's first contact with the
    gateway to its exit, capped at the subset's job budget. Loading the model is not
    on it; the time between episodes is. None when the clock never started."""
    over_time: bool = False
    """The job budget ran out before the submission called ``finish()`` or exited, as
    the gateway's clock files tell. One that handles it and exits 0 still reads
    "completed", scored on what was recorded."""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_s: float | None = None
    trials_run: int | None = None
    """Trials the gateway started, from its results dump. The leaderboard turns it into
    a share of the pack's trials and never publishes the count. None when unknown."""


class RunResult(BaseModel):
    """The published result document."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    job_id: str | None = None
    team_id: str
    submission_id: str
    status: RunStatus
    submitted_at: datetime | None = None
    """When the watcher queued it (or, for a rejection, when it looked). The queue record
    that held this is gone once a job is withdrawn, so a result carries its own."""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    subsets: dict[str, SubsetResult] = Field(default_factory=dict)
    error: str | None = None
    error_code: ErrorCode | None = None
    logs_dir: str | None = None

    @classmethod
    def failed(
        cls,
        run_id: str,
        job_id: str,
        team_id: str,
        submission_id: str,
        error: str,
        error_code: ErrorCode,
        *,
        status: RunStatus = RunStatus.FAILED,
        submitted_at: datetime | None = None,
        started_at: datetime | None = None,
        logs_dir: str | None = None,
    ) -> RunResult:
        return cls(
            run_id=run_id,
            job_id=job_id,
            team_id=team_id,
            submission_id=submission_id,
            status=status,
            submitted_at=submitted_at,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            error=error,
            error_code=error_code,
            logs_dir=logs_dir,
        )


__all__ = [
    "ImageRef",
    "ORGANIZER_ENV",
    "RunSpec",
    "SubmissionManifest",
    "describe_errors",
    "hf_owner",
    "WEIGHTS_MOUNT_ROOT",
    "WeightsMount",
    "is_commit_sha",
    "validate_id",
    "validate_repo",
    "ErrorCode",
    "GB",
    "Rejected",
    "QueueJob",
    "RunStatus",
    "SubsetResult",
    "RunResult",
]
