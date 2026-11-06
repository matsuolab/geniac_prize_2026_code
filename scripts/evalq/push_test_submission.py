"""Push a test submission to one of the queue test repos.

    push_test_submission.py N VARIANT SUBMISSION_ID [--from SIF] [--image-only] [--notes TEXT]
                            [--entrypoint "CMD ..."]

Uploads ``agent-VARIANT.sif`` to ``<owner>/<prefix>-N`` (unless the repo has it), then
pushes a ``submission.yaml`` for team ``<team-prefix>-N`` pointing at that commit.
``--entrypoint`` replaces the image's command (shell-style), e.g. ``"sh -c 'exit 3'"``
for a submission that fails on purpose.

The queue recognises an image by its hash, so a rerun needs bytes it has not seen.
``--from SIF`` makes a variant by appending a few bytes to a copy of SIF, uploads it,
and deletes the copy; without it the file must already be in ``$CAPX_QT_IMAGES``.

Settings come from the environment: CAPX_QT_OWNER (required: the HuggingFace user or org),
CAPX_QT_REPO_PREFIX (capx-submission-test), CAPX_QT_TEAM_PREFIX (test-team),
CAPX_QT_IMAGES (~/capx-queue/qt-images).
"""

from __future__ import annotations

import io
import os
import shlex
import shutil
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi

OWNER = os.environ.get("CAPX_QT_OWNER", "")
if not OWNER:
    sys.exit("set CAPX_QT_OWNER to the HuggingFace user or org that holds the test repos")
REPO_PREFIX = os.environ.get("CAPX_QT_REPO_PREFIX", "capx-submission-test")
TEAM_PREFIX = os.environ.get("CAPX_QT_TEAM_PREFIX", "test-team")
# What the reference agent image declares (shared with the smoke and stress tests).
WEIGHTS = [
    {"repo": "Qwen/Qwen3.6-27B", "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
     "path": "/models/qwen3.6-27b"},
    {"repo": "facebook/sam3", "revision": "3c879f39826c281e95690f02c7821c4de09afae7",
     "path": "/models/sam3", "gated": True},
]
ENV = {"CAPX_MODEL_PATH": "/models/qwen3.6-27b", "CAPX_SAM3_CHECKPOINT": "/models/sam3/sam3.pt"}


def _option(name: str, default: str = "") -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> None:
    n, variant, submission_id = sys.argv[1:4]
    repo = f"{OWNER}/{REPO_PREFIX}-{n}"
    team = f"{TEAM_PREFIX}-{n}"
    images = Path(os.environ.get("CAPX_QT_IMAGES", "~/capx-queue/qt-images")).expanduser()
    filename = f"agent-{variant}.sif"
    local = images / filename

    api = HfApi()
    if filename not in {s.rfilename for s in api.repo_info(repo).siblings}:
        made = False
        source = _option("--from")
        if source:
            images.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, local)
            with local.open("ab") as handle:
                handle.write(f"\n# queue-test variant {variant}\n".encode() * 8)
            made = True
        try:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=filename, repo_id=repo,
                            commit_message=f"image {variant}")
        finally:
            if made:
                local.unlink(missing_ok=True)

    revision = api.list_repo_commits(repo)[0].commit_id
    if "--image-only" in sys.argv:
        print("image", revision)
        return
    manifest = {
        "team_id": team,
        "submission_id": submission_id,
        "image": {"repo": repo, "revision": revision, "path": filename},
        "notes": _option("--notes", f"queue test {submission_id}"),
        "weights": WEIGHTS,
        "env": ENV,
    }
    if "--entrypoint" in sys.argv:
        manifest["entrypoint"] = shlex.split(_option("--entrypoint"))
    api.upload_file(path_or_fileobj=io.BytesIO(yaml.safe_dump(manifest, sort_keys=False).encode()),
                    path_in_repo="submission.yaml", repo_id=repo, commit_message=submission_id)
    print("pushed", team, variant, submission_id, "image@", revision[:8])


if __name__ == "__main__":
    main()
