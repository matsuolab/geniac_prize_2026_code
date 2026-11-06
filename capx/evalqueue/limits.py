"""The official-mode ceiling on a submission image.

The model needs none: a submission is given one GPU, so "fits one B300" is enforced
by the allocation itself.
"""

from __future__ import annotations

from capx.config import EvalConfig
from capx.evalqueue.models import ErrorCode, Rejected


def check_image_size(image_gb: float, config: EvalConfig) -> None:
    limit = config.submission.max_image_gb
    if image_gb > limit:
        raise Rejected(
            ErrorCode.IMAGE_TOO_LARGE,
            f"submission image is {image_gb:.1f} GiB, over the {limit:.1f} GiB limit",
        )


def check_sha256(declared: str, actual: str) -> None:
    if declared.lower() != actual.lower():
        raise Rejected(
            ErrorCode.IMAGE_HASH_MISMATCH,
            f"image sha256 mismatch: manifest says {declared}, the file is {actual}",
        )


__all__ = ["check_image_size", "check_sha256"]
