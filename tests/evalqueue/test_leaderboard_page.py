"""The leaderboard pages show an error code by a fixed index (web/leaderboard/index_*.html),
and a finished run either its error code or its trial share beside the status."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from capx.evalqueue.models import ErrorCode

PAGES = sorted((Path(__file__).resolve().parents[2] / "web" / "leaderboard").glob("index_*.html"))


def _table(page: Path, name: str) -> dict[str, str]:
    block = re.search(rf"const {name} = \{{(.*?)\n\}};", page.read_text(), re.S)
    assert block, f"{page.name} has no {name}"
    return dict(re.findall(r'^\s+(\w+): "([^"]*)",$', block.group(1), re.M))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_error_code_has_its_own_index_and_message(page: Path) -> None:
    index = _table(page, "CODE_INDEX")
    codes = {code.value for code in ErrorCode}
    assert set(index) == codes
    assert len(set(index.values())) == len(index)
    assert all(re.fullmatch(r"E\d{2}", value) for value in index.values())
    assert set(_table(page, "ERROR_CODES")) == codes


def test_the_pages_agree_on_the_indices() -> None:
    assert len(PAGES) == 2
    assert _table(PAGES[0], "CODE_INDEX") == _table(PAGES[1], "CODE_INDEX")


def test_a_row_shows_the_error_code_when_it_failed_and_the_share_otherwise() -> None:
    for page in PAGES:
        body = re.search(r"function statusTag\(a\) \{\n(.*?)\n\}", page.read_text(), re.S)
        assert body, f"{page.name} has no statusTag"
        lines = body.group(1).splitlines()
        assert lines[0].strip() == (
            'if (["failed", "rejected"].includes(a.status)) '
            "return a.error_code ? codeTag(a.error_code) : null;"
        )
        assert '"pill code"' in lines[-1]  # the share looks like a code
