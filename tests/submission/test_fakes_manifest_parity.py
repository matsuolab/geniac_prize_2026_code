"""Drift guard: ``tests/submission/fakes.py``'s ``_DEFAULT_ENTRIES`` must track
``taskpacks/original/pack.yaml`` (its module docstring's explicit claim), so
shim/runner unit tests keep exercising the real embodiment/camera_names
combinations instead of a silently stale fixture.

What "track" can mean changed when the pack went from 3 tasks to 137. The fixture
is a sample: requiring one entry per pack task would mean 130 near-identical
LIBERO fixtures and a rewrite of every test that names one. What the shim tests
actually depend on is the *shapes* -- which embodiment, which cameras -- so that
is what is pinned here, plus real field equality for the entries whose task_id the
pack does have.

Loads ``pack.yaml`` with plain ``yaml.safe_load`` -- no sim imports, no
``capx.protocol.taskpack`` pydantic validation needed for a field-equality check.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from tests.submission.fakes import _DEFAULT_ENTRIES

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_YAML = REPO_ROOT / "taskpacks" / "original" / "pack.yaml"


def _load_pack_tasks() -> dict[str, dict]:
    data = yaml.safe_load(PACK_YAML.read_text())
    return {task["task_id"]: task for task in data["tasks"]}


def test_every_shape_in_the_pack_is_represented():
    """A pack that grows an embodiment or a camera set must not leave the fakes
    covering only the old ones — that is how the two-arm shims went untested."""
    pack_tasks = _load_pack_tasks()
    in_pack = {(t["embodiment"], tuple(t["camera_names"])) for t in pack_tasks.values()}
    in_fixture = {
        (entry.embodiment.value, tuple(entry.camera_names)) for entry in _DEFAULT_ENTRIES
    }
    assert in_pack <= in_fixture, f"no fixture entry for: {sorted(in_pack - in_fixture)}"


def test_entries_the_pack_still_has_match_it_field_for_field():
    pack_tasks = _load_pack_tasks()
    checked = 0
    for entry in _DEFAULT_ENTRIES:
        task = pack_tasks.get(entry.task_id)
        if task is None:
            continue
        checked += 1
        assert entry.embodiment.value == task["embodiment"], entry.task_id
        assert entry.instruction == task["instruction"], entry.task_id
        assert entry.camera_names == task["camera_names"], entry.task_id
    assert checked, "no fixture entry names a real pack task; the guard checks nothing"
