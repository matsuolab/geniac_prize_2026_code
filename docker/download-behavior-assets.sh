#!/usr/bin/env bash
# Fetch the BEHAVIOR asset set at RUNTIME into OmniGibson's data path, so the
# env-behavior image stays asset-free. In the compose setup OMNIGIBSON_DATA_PATH
# points at the `behavior-assets` named volume, so this is a one-time download
# that survives image rebuilds:
#
#   docker compose --profile assets run --rm fetch-behavior-assets
#
# Every step is skipped when its target already exists, so a re-run only fills
# what is missing. OmniGibson 3.9.2 reads the 2026 challenge metadata at import;
# the 2025 instances are the ones CaP-X's BEHAVIOR tasks were authored against.
set -euo pipefail

if [[ "${1:-}" != "--accept-dataset-tos" ]]; then
    echo "usage: $0 --accept-dataset-tos" >&2
    echo "(downloads the BEHAVIOR-1K assets and accepts the BEHAVIOR dataset license)" >&2
    exit 1
fi

export OMNI_KIT_ACCEPT_EULA=YES

python - <<'PY'
from omnigibson.macros import gm
from omnigibson.utils import asset_utils

print(f"[assets] target data path: {gm.DATA_PATH}", flush=True)
asset_utils.download_omnigibson_robot_assets()
asset_utils.download_behavior_1k_assets(accept_license=True)
asset_utils.download_2026_challenge_task_instances()
asset_utils.download_2025_challenge_task_instances()
print(f"[assets] done -> {gm.DATA_PATH}", flush=True)
PY
