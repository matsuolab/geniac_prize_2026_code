"""The local runner's trial deadline, and what generated code may and may not swallow.

The deadline arrives as an exception raised from SIGALRM wherever the trial happens
to be, which is usually inside generated code or an API helper. Neither may turn it
into an ordinary error, and a trial that ran past it is a timeout even if something
caught it anyway.

Needs the task-env stack (gymnasium, omegaconf, the LLM client); skips without it.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest


def _stub_capx_integrations_package() -> None:
    # As in tests/submission/test_code_env.py: the executor only needs base_api, not
    # the perception stack capx/integrations/__init__.py imports.
    if "capx.integrations" in sys.modules:
        return
    try:
        import capx.integrations  # noqa: F401

        return
    except ImportError:
        pass
    import capx

    pkg = types.ModuleType("capx.integrations")
    pkg.__path__ = [str(Path(capx.__file__).parent / "integrations")]
    sys.modules["capx.integrations"] = pkg


_stub_capx_integrations_package()
runner = pytest.importorskip("capx.envs.runner")
from capx.envs.tasks.base import CodeExecutionEnvBase  # noqa: E402


class _LowLevel:
    def get_observation(self) -> dict[str, Any]:
        return {}


def _code_env() -> CodeExecutionEnvBase:
    env = object.__new__(CodeExecutionEnvBase)
    env.low_level_env = _LowLevel()
    env._apis = {}
    env._full_prompt = []
    env._init_exec_globals()
    return env


@pytest.mark.parametrize(
    "code",
    [
        "raise ValueError('bad')",
        "exit(3)",
        "raise TimeoutError('io')",
        "raise GeneratorExit('failed')",
        "class Stop(BaseException): pass\nraise Stop()",
    ],
)
def test_the_codes_own_failures_go_back_to_the_model(code: str) -> None:
    result = _code_env()._exec_user_code(code)
    assert result["ok"] is False
    assert "Traceback" in result["stderr"]


@pytest.mark.parametrize(
    "exc", [KeyboardInterrupt, runner.TrialTimeout], ids=["interrupt", "deadline"]
)
def test_interrupts_and_the_deadline_get_past_the_code(exc: type[BaseException]) -> None:
    env = _code_env()
    env._exec_globals["Raised"] = exc
    with pytest.raises(exc):
        env._exec_user_code("raise Raised('stop')")


def _fake_trial(swallow: type[BaseException]):
    def run(env, trial, args, config, multi_turn_prompt, partial_artifacts=None):
        try:
            time.sleep(5)
        except swallow:
            pass
        return "finished"

    return run


@pytest.fixture
def timeout_summary(monkeypatch):
    built: list[BaseException] = []

    def build(trial, timeout_seconds, partial_artifacts, config, exc):
        built.append(exc)
        return "timed out"

    monkeypatch.setattr(runner, "_build_timeout_summary", build)
    return built


def _run(raise_on_timeout: bool = False) -> Any:
    return runner._run_single_trial_with_timeout(
        env=None,
        trial=1,
        args=None,
        config={},
        multi_turn_prompt=None,
        timeout_s=1,
        raise_on_timeout=raise_on_timeout,
    )


def test_code_that_catches_exception_cannot_stop_the_deadline(monkeypatch, timeout_summary) -> None:
    monkeypatch.setattr(runner, "_run_single_trial", _fake_trial(Exception))
    assert _run() == "timed out"
    assert isinstance(timeout_summary[0], runner.TrialTimeout)


def test_a_trial_that_swallowed_the_deadline_still_timed_out(monkeypatch, timeout_summary) -> None:
    monkeypatch.setattr(runner, "_run_single_trial", _fake_trial(BaseException))
    assert _run() == "timed out"


def test_a_swallowed_deadline_is_retried_like_any_timeout(monkeypatch, timeout_summary) -> None:
    monkeypatch.setattr(runner, "_run_single_trial", _fake_trial(BaseException))
    with pytest.raises(TimeoutError):
        _run(raise_on_timeout=True)


def test_a_deadline_swallowed_once_is_raised_again(monkeypatch, timeout_summary) -> None:
    monkeypatch.setattr(runner, "DEADLINE_REPEAT_SECONDS", 0.2)
    ran_on: list[bool] = []

    def run(env, trial, args, config, multi_turn_prompt, partial_artifacts=None):
        try:
            time.sleep(5)
        except BaseException:
            pass
        # Keeps working after swallowing the deadline; only a repeat can stop it.
        until = time.monotonic() + 10
        while time.monotonic() < until:
            time.sleep(0.01)
        ran_on.append(True)
        return "finished"

    monkeypatch.setattr(runner, "_run_single_trial", run)
    started = time.monotonic()
    assert _run() == "timed out"
    assert time.monotonic() - started < 4
    assert ran_on == []


def test_one_worker_records_a_raising_trial_and_runs_the_rest(monkeypatch, tmp_path) -> None:
    from capx.utils.launch_utils import TrialSummary

    def run_trial(env, trial, args, config, multi_turn_prompt):
        if trial == 2:
            raise RuntimeError("the model server went away")
        return TrialSummary(
            trial=trial, success=True, reward=1.0, terminated=True, truncated=False,
            sandbox_rc=0, log="ok",
        )

    reported: list[list[TrialSummary]] = []
    monkeypatch.setattr(runner, "_worker_setup", lambda *, env_factory: ("env", None))
    monkeypatch.setattr(runner, "_run_trial_with_retries", run_trial)
    monkeypatch.setattr(
        runner, "_print_and_save_summary", lambda summaries, *rest: reported.append(summaries)
    )
    config = {
        "total_trials": 3,
        "num_workers": 1,
        "record_video": False,
        "output_dir": str(tmp_path / "run"),
        "resume_idx": None,
    }
    args = types.SimpleNamespace(use_oracle_code=False, model="model")
    runner._run_headless_trials(args, {"cfg": {}}, config, start_time=0.0)

    [summaries] = reported
    assert [s.trial for s in summaries] == [1, 2, 3]
    assert [s.success for s in summaries] == [True, False, True]
    assert "the model server went away" in summaries[1].log


class _FakeProc:
    def __init__(self, *, ignores_sigterm: bool = False) -> None:
        self.ignores_sigterm = ignores_sigterm
        self.alive = True
        self.killed = False

    def terminate(self) -> None:
        if not self.ignores_sigterm:
            self.alive = False

    def kill(self) -> None:
        self.killed = True
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        pass

    def is_alive(self) -> bool:
        return self.alive


def test_servers_started_before_a_failed_one_are_stopped(monkeypatch) -> None:
    started: list[_FakeProc] = []

    def start(cfg: dict[str, Any]) -> _FakeProc:
        started.append(_FakeProc())
        return started[-1]

    monkeypatch.setattr(runner, "run_server_proc", start)
    with pytest.raises(ValueError):
        runner._start_api_servers([{"name": "first"}, {"name": "second", "port": "not-a-port"}])
    assert len(started) == 1
    assert started[0].alive is False


def test_stopping_kills_a_server_that_ignores_sigterm() -> None:
    stubborn, polite = _FakeProc(ignores_sigterm=True), _FakeProc()
    runner._stop_api_servers([stubborn, polite], grace_s=0.0)
    assert stubborn.killed and not stubborn.alive
    assert not polite.killed and not polite.alive
