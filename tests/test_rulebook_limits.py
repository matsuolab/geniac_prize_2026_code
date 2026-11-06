"""PRELIM_RULEBOOK.md publishes the public pack's limits; the pack is where they take effect."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from capx.protocol.taskpack import load_pack

ROOT = Path(__file__).resolve().parents[1]
ROW = re.compile(
    r"^\| `(?P<pack>[^`]+)` \| (?P<tasks>\d+) \| (?P<trials>\d+) \| (?P<job>\d+) "
    r"\| (?P<steps>\d+) \|$"
)


def _rows() -> list[dict[str, str]]:
    rulebook = (ROOT / "PRELIM_RULEBOOK.md").read_text()
    return [match.groupdict() for line in rulebook.splitlines() if (match := ROW.match(line))]


def test_the_rulebook_lists_the_public_pack() -> None:
    assert "taskpacks/original" in {row["pack"] for row in _rows()}


@pytest.mark.parametrize("row", _rows(), ids=lambda row: row["pack"])
def test_the_rulebook_matches_the_pack(row: dict[str, str]) -> None:
    pack = ROOT / row["pack"]
    if not (pack / "pack.yaml").exists():
        pytest.skip(f"{row['pack']} is not checked out")
    manifest, _ = load_pack(pack)
    assert len(manifest.tasks) == int(row["tasks"])
    assert sum(task.n_trials for task in manifest.tasks) == int(row["trials"])
    assert manifest.job_timeout_s() == float(row["job"])
    assert manifest.limits.trial_steplimit == int(row["steps"])
