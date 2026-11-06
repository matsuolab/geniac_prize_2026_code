"""Submission size-limit gate (PRELIM_RULEBOOK.md §6.3): the image ceiling."""

from __future__ import annotations

import pytest

from capx.config import load_eval_config
from capx.evalqueue.limits import check_image_size
from capx.evalqueue.models import ErrorCode, Rejected


def test_within_limits_passes() -> None:
    config = load_eval_config()
    check_image_size(config.submission.max_image_gb - 1.0, config)


def test_over_image_limit_raises() -> None:
    config = load_eval_config()
    with pytest.raises(Rejected, match="image") as caught:
        check_image_size(config.submission.max_image_gb + 1.0, config)
    assert caught.value.code is ErrorCode.IMAGE_TOO_LARGE
