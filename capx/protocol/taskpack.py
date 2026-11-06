"""Task-pack schema: a directory with a ``pack.yaml`` the gateway loads at startup.

The same gateway serves a public pack (shipped in the repo) and a private one
(organizer-only); only the pack and the enforcement mode differ. A ``TaskSpec``
keeps everything about building and grading the task gateway-side; only
:class:`ManifestEntry` (goal + I/O spec) is ever sent to the submission.
"""

from __future__ import annotations

import enum
import importlib
import sys
from collections import Counter
from collections.abc import Collection
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


class Embodiment(str, enum.Enum):
    FRANKA = "franka"
    FRANKA_BIMANUAL = "franka_bimanual"
    R1PRO = "r1pro"


class Simulator(str, enum.Enum):
    """The simulators this repo builds envs for; each env backend runs one."""

    ROBOSUITE = "robosuite"
    LIBERO = "libero"
    BEHAVIOR = "behavior"


# Which env backend a task is routed to. Usually a `Simulator` value; a deployment
# may run further backends under ids of its own, from an image that sets
# CAPX_SIMULATOR to the id and CAPX_ENV_FACTORY to the `Simulator` that builds its envs.
SimulatorId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]


class Visibility(str, enum.Enum):
    """Leaderboard exposure for a pack."""

    PUBLIC = "public"  # full per-episode data published
    PRIVATE = "private"  # success rate + wall-clock only


class SeedPolicy(BaseModel):
    """How the gateway draws episode seeds. The seed itself is never disclosed."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["fixed", "random"] = "random"
    seeds: list[int] | None = None  # required (non-empty) when mode == "fixed"

    @model_validator(mode="after")
    def _fixed_needs_seeds(self) -> SeedPolicy:
        if self.mode == "fixed" and not self.seeds:
            raise ValueError("seed_policy mode='fixed' requires a non-empty 'seeds' list")
        return self


class PackLimits(BaseModel):
    """A round's rules, set once per pack. README.md publishes them."""

    model_config = ConfigDict(extra="forbid")

    trial_time_s: float = Field(default=300.0, gt=0)
    """What one trial adds to the job budget: the whole run gets this times the number of
    trials it holds (``TaskPackManifest.job_timeout_s``). No single trial is limited to it."""
    n_trials: int = Field(ge=1)
    """Trials per task, unless a task sets its own."""
    trial_steplimit: int = Field(ge=1)
    """Control cycles per trial, unless a task sets its own. Without one, a policy
    that moves at random long enough could stumble into success."""


class ManifestEntry(BaseModel):
    """The only per-task info sent to the submission: the goal and the I/O contract."""

    task_id: str
    embodiment: Embodiment
    instruction: str
    camera_names: list[str]
    action_spec_ref: str = Field(
        description="Pointer into PRELIM_RULEBOOK.md for this embodiment's raw action schema."
    )
    n_trials: int
    trial_steplimit: int


class TaskSpec(BaseModel):
    """A full task definition. Only ``to_manifest_entry`` reaches the submission."""

    # A misspelt or unsupported key would otherwise be dropped without a word, and
    # the task would run as something other than what the pack says.
    model_config = ConfigDict(extra="forbid")

    # ---- public contract (surfaced via ManifestEntry) ----
    task_id: str
    embodiment: Embodiment
    simulator: SimulatorId
    instruction: str
    camera_names: list[str] = Field(default_factory=lambda: ["robot0_robotview"])

    # ---- how the gateway builds the low-level env ----
    # Name registered via `register_env` (e.g. "franka_robosuite_cubes_low_level"),
    # by capx itself or by one of the pack's `register_modules`.
    env_target: str
    # Extra constructor kwargs for `env_target`, so one registered env can serve
    # several conditions (e.g. a variation preset). Never sent to the submission.
    env_kwargs: dict[str, Any] = Field(default_factory=dict)
    privileged: bool = False  # MUST be False in official packs (no GT-pose APIs)

    # ---- referee parameters (gateway-enforced) ----
    # A pack.yaml task may leave these out; the pack's `limits` fill them in.
    n_trials: int = Field(ge=1)
    trial_steplimit: int = Field(ge=1)
    seed_policy: SeedPolicy = Field(default_factory=SeedPolicy)

    @field_validator("env_kwargs")
    @classmethod
    def _horizon_is_the_steplimit(cls, value: dict[str, Any]) -> dict[str, Any]:
        # The gateway sets the env's horizon from trial_steplimit, so a second value
        # here would be a limit nobody publishes.
        if "max_steps" in value:
            raise ValueError("set trial_steplimit, not env_kwargs.max_steps")
        return value

    @model_validator(mode="after")
    def _behavior_needs_fixed_seeds(self) -> TaskSpec:
        # BEHAVIOR interprets the seed as a presampled task-instance INDEX
        # (r1pro_adapter: reset(options={"trial": seed})), so the referee's random
        # 31-bit seeds would be out of range and crash every reset.
        if self.simulator == Simulator.BEHAVIOR and self.seed_policy.mode != "fixed":
            raise ValueError(
                f"task {self.task_id!r}: BEHAVIOR tasks need seed_policy mode='fixed' "
                "(seeds are presampled instance indices, not RNG seeds)"
            )
        return self

    def to_manifest_entry(self) -> ManifestEntry:
        """Project to the client-safe view. This is the privacy boundary."""
        return ManifestEntry(
            task_id=self.task_id,
            embodiment=self.embodiment,
            instruction=self.instruction,
            camera_names=self.camera_names,
            action_spec_ref=f"PRELIM_RULEBOOK.md#action-{self.embodiment.value}",
            n_trials=self.n_trials,
            trial_steplimit=self.trial_steplimit,
        )


class TaskPackManifest(BaseModel):
    """A pack's ``pack.yaml`` — discovered by the gateway at startup."""

    model_config = ConfigDict(extra="forbid")

    pack_id: str
    version: str = "0.1"
    visibility: Visibility = Visibility.PUBLIC
    limits: PackLimits
    """Every pack sets its own, including one that extends another: a subset of a
    round's tasks needs its own job budget, not the whole round's."""
    score_weights: dict[SimulatorId, Annotated[float, Field(gt=0)]] = Field(default_factory=dict)
    """Per simulator, how much its success rate weighs in the score (PRELIM_RULEBOOK.md
    §4.5). Empty, every trial weighs the same. Not published, and like ``limits`` not
    taken from a pack this one extends."""
    # Per simulator, modules shipped with the pack (in its directory or one of its
    # `code_dirs`) whose `register_env` calls add the pack's own low-level envs. Imported by `import_pack_modules` in
    # the process that builds that simulator's envs (its env backend, or the
    # in-process factory), never by the gateway, whose image carries no simulator.
    # Keyed by simulator because each backend image has only its own sim stack.
    # Built-in envs register via `import capx.envs.simulators`, which the env
    # factory already does.
    register_modules: dict[SimulatorId, list[str]] = Field(default_factory=dict)
    # Sibling directories (names under the same parent as the pack's own)
    # that hold env code several packs import. They go on `sys.path` after the
    # pack's own directory.
    code_dirs: list[str] = Field(default_factory=list)
    tasks: list[TaskSpec] = Field(default_factory=list)
    # A pack may instead select `task_ids` from a sibling pack (its directory name),
    # reusing that pack's tasks and env code as they are, e.g. one round's tasks as
    # a subset of a later round's. `load_pack` resolves it.
    extends: str | None = None
    task_ids: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _fill_task_limits(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        limits, tasks = data.get("limits"), data.get("tasks")
        if not isinstance(limits, dict) or not isinstance(tasks, list):
            return data
        defaults = {key: limits[key] for key in ("n_trials", "trial_steplimit") if key in limits}
        filled = [{**defaults, **task} if isinstance(task, dict) else task for task in tasks]
        return {**data, "tasks": filled}

    @model_validator(mode="after")
    def _tasks_or_extends(self) -> TaskPackManifest:
        if self.extends is None:
            if self.task_ids is not None:
                raise ValueError(f"pack {self.pack_id!r}: task_ids needs extends")
        elif self.tasks or self.register_modules or self.code_dirs or not self.task_ids:
            raise ValueError(
                f"pack {self.pack_id!r} extends {self.extends!r}: give task_ids only, "
                "not tasks, register_modules or code_dirs"
            )
        return self

    @model_validator(mode="after")
    def _every_simulator_is_weighed(self) -> TaskPackManifest:
        if self.score_weights:
            unweighed = sorted({t.simulator for t in self.tasks} - self.score_weights.keys())
            if unweighed:
                raise ValueError(f"pack {self.pack_id!r}: score_weights has no {unweighed}")
        return self

    @model_validator(mode="after")
    def _task_ids_are_unique(self) -> TaskPackManifest:
        counts = Counter(t.task_id for t in self.tasks)
        duplicates = [task_id for task_id, n in counts.items() if n > 1]
        if duplicates:
            raise ValueError(f"pack {self.pack_id!r} repeats task ids: {sorted(duplicates)}")
        return self

    @model_validator(mode="after")
    def _private_pack_stays_unprivileged(self) -> TaskPackManifest:
        # Makes TaskSpec.privileged's "MUST be False in official packs" real: a
        # hidden-eval pack exposing GT-pose APIs would defeat its own secrecy.
        if self.visibility is Visibility.PRIVATE:
            leaky = [t.task_id for t in self.tasks if t.privileged]
            if leaky:
                raise ValueError(f"private pack {self.pack_id!r} has privileged tasks: {leaky}")
        return self

    def job_timeout_s(self, simulators: Collection[str] | None = None) -> float:
        """Wall clock for the whole run: ``trial_time_s`` for each trial of each task,
        counting only tasks on ``simulators`` when given."""
        trials = sum(
            task.n_trials
            for task in self.tasks
            if simulators is None or task.simulator in simulators
        )
        return self.limits.trial_time_s * trials


def _read_raw(root: Path) -> dict[str, Any]:
    data = yaml.safe_load((root / "pack.yaml").read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{root / 'pack.yaml'} is not a mapping")
    return data


def _read_manifest(root: Path) -> TaskPackManifest:
    return TaskPackManifest.model_validate(_read_raw(root))


def _raw_tasks(root: Path) -> list[dict[str, Any]]:
    """The pack's tasks as written, before any pack's limits fill them in.

    An extending pack resolves its tasks from these rather than from the base
    pack's validated ones, so it is its own limits that fill the gaps.
    """
    data = _read_raw(root)
    manifest = TaskPackManifest.model_validate(data)
    if manifest.extends is None:
        return list(data.get("tasks") or [])
    assert manifest.task_ids is not None
    base_dir = root.parent / manifest.extends
    by_id = {task["task_id"]: task for task in _raw_tasks(base_dir)}
    missing = [task_id for task_id in manifest.task_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"pack {manifest.pack_id!r}: {missing} not in {base_dir}")
    return [by_id[task_id] for task_id in manifest.task_ids]


def pack_dirs(taskpack_dir: str | Path) -> list[Path]:
    """Every directory the pack needs at run time, so what a container has to mount.

    The pack's own, its ``code_dirs``, then the same for each pack it extends.
    """
    root = Path(taskpack_dir).resolve()
    manifest = _read_manifest(root)
    dirs = [root, *(root.parent / name for name in manifest.code_dirs)]
    if manifest.extends is not None:
        dirs += pack_dirs(root.parent / manifest.extends)
    return list(dict.fromkeys(dirs))


def load_pack(taskpack_dir: str | Path) -> tuple[TaskPackManifest, list[Path]]:
    """Read a pack, resolving ``extends``; returns it and the directories its code is in.

    A pack that extends another gets that pack's tasks in its own ``task_ids``
    order and its ``register_modules``, and keeps that pack's visibility unless it
    sets its own. Its own ``limits`` and ``score_weights`` apply; a task's explicit
    values stay.
    """
    root = Path(taskpack_dir).resolve()
    manifest = _read_manifest(root)
    if manifest.extends is None:
        return manifest, [root, *(root.parent / name for name in manifest.code_dirs)]
    base, code_paths = load_pack(root.parent / manifest.extends)
    visibility = (
        manifest.visibility if "visibility" in manifest.model_fields_set else base.visibility
    )
    resolved = TaskPackManifest.model_validate(
        {
            "pack_id": manifest.pack_id,
            "version": manifest.version,
            "visibility": visibility,
            "limits": manifest.limits.model_dump(),
            "score_weights": manifest.score_weights,
            "register_modules": base.register_modules,
            "tasks": _raw_tasks(root),
        }
    )
    return resolved, code_paths


def load_pack_visibility(taskpack_dir: str | Path) -> Visibility:
    """A task pack's leaderboard visibility, ``extends`` resolved."""
    return load_pack(taskpack_dir)[0].visibility


def import_pack_modules(taskpack_dir: str | Path, simulator: str) -> None:
    """Import a pack's ``register_modules`` for ``simulator`` from the pack's code.

    The pack's directory and its ``code_dirs`` go on ``sys.path``, so a pack can
    carry its env code with it, and the env images stay the same for every pack.
    """
    manifest, code_paths = load_pack(taskpack_dir)
    modules = manifest.register_modules.get(simulator, [])
    if not modules:
        return
    for path in reversed(code_paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    for module in modules:
        importlib.import_module(module)


def load_task_packs(paths: list[str]) -> dict[str, TaskSpec]:
    """Discover packs and return task_id -> TaskSpec.

    Only the specs are read; the pack's env code is imported separately, by
    ``import_pack_modules``, where the envs are actually built.

    Each ``paths`` entry is a directory containing ``pack.yaml``.
    """
    tasks: dict[str, TaskSpec] = {}
    for path in paths:
        manifest, _ = load_pack(path)
        for spec in manifest.tasks:
            if spec.task_id in tasks:
                raise ValueError(f"duplicate task_id across packs: {spec.task_id!r}")
            tasks[spec.task_id] = spec
    return tasks


__all__ = [
    "Embodiment",
    "Simulator",
    "SimulatorId",
    "Visibility",
    "SeedPolicy",
    "ManifestEntry",
    "PackLimits",
    "TaskSpec",
    "TaskPackManifest",
    "import_pack_modules",
    "load_pack",
    "pack_dirs",
    "load_task_packs",
    "load_pack_visibility",
]
