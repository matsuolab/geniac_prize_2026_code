"""Constants and config schemas shared by the submission-side gateway client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from capx.protocol.taskpack import Simulator
from capx.utils.unix_http import ServerUrl

# Robosuite's error for stepping past its horizon. The shims raise the same text once
# the gateway has ended the episode, and `capx/envs/trial.py` reads it in stderr as
# the end of the episode either way.
EPISODE_OVER_MESSAGE = "executing action in terminated episode"

STEP_CHUNK_DEFAULT = 10

# The runner config and the task configs it falls back to ship here.
GATEWAY_CONFIG_DIR = Path(__file__).resolve().parents[2] / "env_configs" / "gateway"

DEFAULT_MULTI_TURN_PROMPT = """\
The following code blocks were just executed:
```python
{executed_code}
```
The current console stdout is:
```
{console_stdout}
```
The current console stderr is:
```
{console_stderr}
```
If there were no errors during compilation or execution, the console stderr will be empty. If there were errors, consider regenerating the code.
Consider querying the environment state through perception-based tool calls and prints to see if the task is completed.
Please respond with EXACTLY ONE of the following:
- The word 'REGENERATE' followed immediately by new Python code in a fenced code block (```python...```) if you want to modify the code.
- The word 'FINISH' if the task appears to be complete
"""


class R1ProKnobs(BaseModel):
    """Servo and safety settings of ``R1ProGatewayShim``."""

    model_config = ConfigDict(extra="forbid")

    in_hand_gap_threshold: float = 0.008
    settle_cycles: int = 50
    nav_max_lin: float = 0.05
    nav_max_ang: float = 0.1
    # How far a commanded torso/arm target may lead the live proprio per RPC. An
    # unreachable IK solution can jump >2 rad, and commanding that in one go swings
    # the arm hard enough to blow up the physics. <= 0 disables the clamp.
    max_joint_delta_per_rpc: float = 0.5
    # pyroki's solver has no failure signal: it returns the cost minimizer even for a
    # target metres out of reach. `_move_hand` refuses to move when the solution's FK
    # lands further than this from the target. <= 0 disables the check.
    ik_residual_abort: float = 0.08


class TaskRunConfig(BaseModel):
    """One task family's config (``env_configs/gateway/*.yaml``).

    ``multi_turn_prompt`` must keep the ``{executed_code}``/``{console_stdout}``/
    ``{console_stderr}`` placeholders ``trial.py`` formats. ``api_servers`` are
    hydra ``_target_`` specs for the perception/IK servers the runner starts.
    """

    model_config = ConfigDict(extra="forbid")

    apis: list[str]
    prompt: str
    multi_turn_prompt: str | None = DEFAULT_MULTI_TURN_PROMPT
    api_servers: list[dict[str, Any]] = Field(default_factory=list)
    # Local copies of the hub models an api_servers entry names as ``model_name``,
    # used in its place when the directory exists (where a submission mounts its
    # weights); elsewhere the hub name loads as usual.
    local_models: dict[str, str] = Field(default_factory=dict)
    record_video: bool = True
    step_chunk: int = STEP_CHUNK_DEFAULT
    r1pro: R1ProKnobs = Field(default_factory=R1ProKnobs)


class TaskMapEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    config: str


class SubmissionConfig(BaseModel):
    """Runner-level config (``env_configs/gateway/runner.yaml``).

    ``trials_per_task`` unset runs every trial the manifest gives each task.
    ``concurrency`` is how many trials run at once, each on its own gateway
    connection; the gateway caps it (``CAPX_MAX_LANES``). ``simulator_lanes`` is the
    gateway's lane cap per simulator (PRELIM_RULEBOOK.md §4.4), which the runner keeps
    to so no lane waits for a backend (capx/submission/worklist.py). ``task_map`` names a task
    family config per ``task_id``, ahead of the embodiment/camera fallback.
    ``use_parallel_ensemble`` generates nine candidates per turn and synthesises one
    answer, ten generations where a single call costs one.
    """

    model_config = ConfigDict(extra="forbid")

    gateway_host: str = "gateway"
    gateway_port: int = 8000
    model: str
    server_url: ServerUrl
    temperature: float = 0.0
    max_tokens: int = 4096
    trials_per_task: int | None = Field(default=None, ge=1)
    concurrency: int = Field(default=1, ge=1)
    simulator_lanes: dict[Simulator, PositiveInt] = Field(default_factory=dict)
    output_dir: str = "outputs"
    task_filter: list[str] | None = None
    task_map: list[TaskMapEntry] = Field(default_factory=list)
    use_visual_feedback: bool = True
    use_img_differencing: bool = False
    use_video_differencing: bool = False
    use_wrist_camera: bool = False
    use_parallel_ensemble: bool = False


def load_submission_config(path: str) -> SubmissionConfig:
    """A relative ``task_map`` config is resolved against this file's directory, not
    against wherever the runner happens to be started."""
    config_path = Path(path).resolve()
    cfg = SubmissionConfig.model_validate(yaml.safe_load(config_path.read_text()))
    for entry in cfg.task_map:
        entry.config = str(config_path.parent / entry.config)
    return cfg


def load_task_run_config(path: str) -> TaskRunConfig:
    return TaskRunConfig.model_validate(yaml.safe_load(Path(path).read_text()))


__all__ = [
    "DEFAULT_MULTI_TURN_PROMPT",
    "EPISODE_OVER_MESSAGE",
    "GATEWAY_CONFIG_DIR",
    "STEP_CHUNK_DEFAULT",
    "R1ProKnobs",
    "TaskMapEntry",
    "TaskRunConfig",
    "SubmissionConfig",
    "load_submission_config",
    "load_task_run_config",
]
