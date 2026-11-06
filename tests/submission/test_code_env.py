"""Unit tests for :class:`capx.submission.code_env.GatewayCodeExecEnv`.

No simulator, no network: ``StubShim`` below is a minimal, self-contained
low-level env defined ONLY in this file (never imported from the shim
packages, per the work-package split) that plays the exact duck-typed role
``CodeExecutionEnvBase``/``GatewayCodeExecEnv`` need: ``reset``,
``get_observation``, ``compute_reward``, ``last_instruction``,
``_sim_step_count``, ``max_steps``.

Importability note: ``capx.envs.tasks.base`` (which ``GatewayCodeExecEnv``
subclasses) resolves ``from capx.integrations.base_api import ApiBase,
get_api`` — a *submodule* import, which forces Python to first run
``capx/integrations/__init__.py``. That module unconditionally imports
``capx.integrations.franka.control``, which imports
``capx.integrations.motion.pyroki_snippets`` (a local jax-based IK solver),
which needs the ``jax`` package. This repo's sim-free test venv intentionally
excludes jax/pyroki (see the harness's own comment and
``tests/submission/test_integrations_importable.py::
test_capx_envs_tasks_base_importable_or_skips``, which already skips on
exactly this). ``capx.integrations.base_api`` itself has no such dependency,
so ``_stub_capx_integrations_package`` below installs a minimal stand-in
package object in ``sys.modules`` (the same "pre-seed the parent package"
technique ``test_integrations_importable.py`` uses to isolate
``capx.integrations.r1pro.control`` from its jax-needing siblings) — this
never touches any file under ``capx/integrations/``, it only affects what
THIS test process resolves ``import capx.integrations`` to. When the real
package (or a venv with jax installed) is already importable, the stub is not
installed and the genuine package is used unmodified.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any


def _stub_capx_integrations_package() -> None:
    if "capx.integrations" in sys.modules:
        return
    try:
        import capx.integrations  # noqa: F401  (use the real package if it imports cleanly)

        return
    except ImportError:
        pass

    import capx

    pkg = types.ModuleType("capx.integrations")
    pkg.__path__ = [str(Path(capx.__file__).parent / "integrations")]
    sys.modules["capx.integrations"] = pkg


_stub_capx_integrations_package()

from capx.envs.tasks.base import CodeExecEnvConfig  # noqa: E402
from capx.integrations.base_api import ApiBase, register_api  # noqa: E402
from capx.submission.code_env import GatewayCodeExecEnv, build_code_env  # noqa: E402
from capx.submission.interfaces import TaskRunConfig  # noqa: E402

# ---------------------------------------------------------------------------
# StubShim: a minimal, file-local low-level env (NOT the design's
# GatewayShimBase/FrankaGatewayShim/R1ProGatewayShim -- those belong to other
# work packages and must not be imported here).
# ---------------------------------------------------------------------------


class StubShim:
    """Bare-minimum stand-in for a gateway shim's ``CodeExecutionEnvBase`` surface."""

    max_steps: int = 5

    def __init__(self, *, instruction: str = "stub instruction") -> None:
        self._instruction = instruction
        self.last_instruction: str | None = None
        self._sim_step_count = 0
        self.reset_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.reset_count += 1
        self._sim_step_count = 0
        self.last_instruction = self._instruction
        return {"stub": True}, {}

    def get_observation(self) -> dict[str, Any]:
        return {"stub": True, "sim_step_count": self._sim_step_count}

    def compute_reward(self) -> float:
        return 0.0

    # Deliberately no `task_completed` attribute: CodeExecutionEnvBase.step's
    # `hasattr` guard must fall back to `task_completed=None`.


class _StubToolApi(ApiBase):
    """A one-function fake API, registered under a test-unique name."""

    def functions(self) -> dict[str, Any]:
        return {"my_tool": self.my_tool}

    def my_tool(self) -> str:
        """Return a fixed marker string, printing it so tests can check stdout capture."""
        print("my_tool called")
        return "tool-result"


_STUB_API_NAME = "StubToolApi__test_code_env"
register_api(_STUB_API_NAME, lambda env: _StubToolApi(env))

_PROMPT_TEMPLATE = "Task instruction: {task_instruction}"


def _make_env(*, instruction: str = "pick up the cube") -> tuple[GatewayCodeExecEnv, StubShim]:
    shim = StubShim(instruction=instruction)
    cfg = CodeExecEnvConfig(low_level=shim, apis=[_STUB_API_NAME], prompt=_PROMPT_TEMPLATE)
    return GatewayCodeExecEnv(cfg), shim


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_build_code_env_helper() -> None:
    shim = StubShim()
    task_cfg = TaskRunConfig(apis=[_STUB_API_NAME], prompt=_PROMPT_TEMPLATE)
    env = build_code_env(shim, task_cfg)
    assert isinstance(env, GatewayCodeExecEnv)
    assert env.low_level_env is shim


# ---------------------------------------------------------------------------
# (a) exec-namespace bindings
# ---------------------------------------------------------------------------


def test_reset_then_step_runs_bound_api_function() -> None:
    env, _shim = _make_env()
    env.reset()
    obs, reward, terminated, truncated, info = env.step("RESULT = my_tool()")
    assert info["sandbox_rc"] == 0
    assert "my_tool called" in info["stdout"]


# ---------------------------------------------------------------------------
# (b) instruction injection: a second reset re-injects from the pristine
# template with no double-substitution leakage.
# ---------------------------------------------------------------------------


def test_instruction_injected_into_prompt() -> None:
    env, _shim = _make_env(instruction="pick up the cube")
    obs, _info = env.reset()

    text = env._full_prompt[-1]["content"][0]["text"]
    assert "{task_instruction}" not in text
    assert text.startswith("Task instruction: pick up the cube")
    # obs["full_prompt"] must be the exact same list object CodeExecutionEnvBase
    # hands out -- the injected text is visible through it too.
    assert obs["full_prompt"] is env._full_prompt
    assert obs["full_prompt"][-1]["content"][0]["text"] == text


def test_second_reset_with_different_instruction_does_not_accumulate() -> None:
    shim = StubShim(instruction="pick up the cube")
    cfg = CodeExecEnvConfig(low_level=shim, apis=[_STUB_API_NAME], prompt=_PROMPT_TEMPLATE)
    env = GatewayCodeExecEnv(cfg)

    env.reset()
    first_text = env._full_prompt[-1]["content"][0]["text"]
    assert "pick up the cube" in first_text

    shim._instruction = "put the mug in the sink"
    obs, _info = env.reset()
    second_text = env._full_prompt[-1]["content"][0]["text"]

    assert "put the mug in the sink" in second_text
    # No leftover from the first substitution and no duplicated instruction text.
    assert "pick up the cube" not in second_text
    assert second_text.count("put the mug in the sink") == 1
    assert "{task_instruction}" not in second_text
    assert obs["full_prompt"] is env._full_prompt


# ---------------------------------------------------------------------------
# (d) truncation via the `_sim_step_count` >= `max_steps` trick
# ---------------------------------------------------------------------------


def test_truncated_tracks_shim_sim_step_count_vs_max_steps() -> None:
    env, shim = _make_env()
    env.reset()

    shim._sim_step_count = 0
    _obs, _reward, _terminated, truncated, _info = env.step("pass")
    assert truncated is False

    shim._sim_step_count = shim.max_steps
    _obs, _reward, _terminated, truncated, _info = env.step("pass")
    assert truncated is True


# ---------------------------------------------------------------------------
# (e) task_completed is None when the shim defines no task_completed
# ---------------------------------------------------------------------------


def test_task_completed_is_none_when_shim_has_no_task_completed() -> None:
    env, _shim = _make_env()
    env.reset()
    assert not hasattr(env.low_level_env, "task_completed")
    _obs, _reward, _terminated, _truncated, info = env.step("pass")
    assert info["task_completed"] is None
