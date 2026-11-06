"""Reading code and the multi-turn decision out of a model reply."""

from __future__ import annotations

import pytest

from capx.utils.launch_utils import _extract_code, _parse_multi_turn_decision


@pytest.mark.parametrize(
    ("reply", "code"),
    [
        ("x = 1\nprint(x)", "x = 1\nprint(x)"),
        ("```python\nx = 1\n```", "x = 1"),
        ("```py\nx = 1\n```", "x = 1"),
        ("```\nx = 1\n```", "x = 1"),
        # Two blocks used to keep the prose and the fences between them: a SyntaxError.
        ("```python\nx = 1\n```\nThen:\n```python\ny = 2\n```", "x = 1"),
        ("Plan first.\n```python\nx = 1\n```\nDone.", "x = 1"),
        # A reply cut off at the token limit keeps what it has.
        ("```python\nx = 1\ny =", "x = 1\ny ="),
        ("```python\nx = 1```", "x = 1"),
        ("Here it is: ```python\nx = 1\n```", "x = 1"),
        # Backticks inside the code neither open nor close a fence.
        ('```python\nprint("```")\nx = 1\n```', 'print("```")\nx = 1'),
        ("print('```')\nx = 1", "print('```')\nx = 1"),
        (None, ""),
    ],
)
def test_extract_code(reply: str | None, code: str) -> None:
    assert _extract_code(reply) == [code]


@pytest.mark.parametrize(
    ("reply", "decision", "code"),
    [
        ("REGENERATE\n```python\nx = 1\n```", "regenerate", "x = 1"),
        ("**REGENERATE**\n```python\nx = 1\n```", "regenerate", "x = 1"),
        ("_REGENERATE_\n```python\nx = 1\n```", "regenerate", "x = 1"),
        ("__REGENERATE__ x = 1", "regenerate", "x = 1"),
        ("REGENERATED code below", "finish", None),
        ("The cube is not stacked yet.\nREGENERATE\n```python\nx = 1\n```", "regenerate", "x = 1"),
        ("FINISH", "finish", None),
        # "REGENERATE" anywhere in the text used to decide.
        ("FINISH. No need to REGENERATE.", "finish", None),
        ("The task is done; I will not REGENERATE.\nFINISH", "finish", None),
        ("The task looks complete.", "finish", None),
        (None, "finish", None),
    ],
)
def test_parse_multi_turn_decision(reply: str | None, decision: str, code: str | None) -> None:
    got_decision, rest = _parse_multi_turn_decision(reply)
    assert got_decision == decision
    if code is not None:
        assert _extract_code(rest) == [code]
