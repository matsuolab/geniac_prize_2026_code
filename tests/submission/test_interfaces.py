"""``capx.submission.interfaces``: the shared constants and the config schemas."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from capx.submission.interfaces import (
    EPISODE_OVER_MESSAGE,
    GATEWAY_CONFIG_DIR,
    SubmissionConfig,
    TaskRunConfig,
    load_submission_config,
    load_task_run_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_episode_over_message_is_robosuites_own():
    """trial.py reads this text in stderr as the end of the episode, for a local
    Robosuite env stepped past its horizon as much as for the shims."""
    assert EPISODE_OVER_MESSAGE == "executing action in terminated episode"


def test_the_shipped_configs_are_where_the_runner_looks():
    assert GATEWAY_CONFIG_DIR == REPO_ROOT / "env_configs" / "gateway"


def test_shipped_gateway_configs_load():
    for path in sorted((REPO_ROOT / "env_configs" / "gateway").glob("*.yaml")):
        if path.name == "runner.yaml":
            load_submission_config(str(path))
        else:
            load_task_run_config(str(path))


def test_r1pro_knobs_reject_unknown_keys():
    with pytest.raises(ValidationError):
        TaskRunConfig(apis=["SomeApi"], prompt="p", r1pro={"nav_speed": 1.0})


@pytest.mark.parametrize(
    "model",
    [
        lambda: SubmissionConfig(model="m", server_url="u", concurency=16),
        lambda: TaskRunConfig(apis=["SomeApi"], prompt="p", record_vidoe=False),
        lambda: SubmissionConfig(model="m", server_url="u", task_map=[{"task_id": "t", "cfg": "x"}]),
    ],
)
def test_a_misspelt_key_is_an_error_not_a_default(model):
    """A typo in a team's fork (``concurency: 16``) used to run with the default of
    one lane, found out only after a scored run."""
    with pytest.raises(ValidationError):
        model()


def test_task_map_configs_are_relative_to_the_runner_config(tmp_path, monkeypatch):
    (tmp_path / "runner.yaml").write_text(
        yaml.safe_dump(
            {
                "model": "m",
                "server_url": "u",
                "task_map": [
                    {"task_id": "a", "config": "a.yaml"},
                    {"task_id": "b", "config": "/abs/b.yaml"},
                ],
            }
        )
    )
    monkeypatch.chdir(REPO_ROOT)
    cfg = load_submission_config(str(tmp_path / "runner.yaml"))
    assert [entry.config for entry in cfg.task_map] == [str(tmp_path / "a.yaml"), "/abs/b.yaml"]


def test_the_shipped_task_map_resolves_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_submission_config(str(GATEWAY_CONFIG_DIR / "runner.yaml"))
    for entry in cfg.task_map:
        load_task_run_config(entry.config)
