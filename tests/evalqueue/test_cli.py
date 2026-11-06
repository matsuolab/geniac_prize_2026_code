"""``capx evalq`` subcommands an operator runs while the queue is busy."""

from __future__ import annotations

import json

from capx.cli import evalq
from capx.evalqueue.models import RunResult, RunStatus
from capx.evalqueue.store import EvalStore


def test_status_reports_an_unreadable_result_instead_of_crashing(tmp_path, capsys) -> None:
    config = tmp_path / "queue.yaml"
    config.write_text(f"root: {tmp_path / 'root'}\nmaster_seed: 1\n")
    store = EvalStore(tmp_path / "root")
    store.ensure_layout()
    store.publish_result(
        RunResult(run_id="good", team_id="alpha", submission_id="s1", status=RunStatus.COMPLETED)
    )
    (store.results / "torn.json").write_text("{not json")

    evalq.status(evalq.StatusArgs(config=str(config)))

    recent = {entry["run_id"]: entry for entry in json.loads(capsys.readouterr().out)["recent_results"]}
    assert recent["good"]["status"] == "completed"
    assert "unreadable" in recent["torn"]
