#!/usr/bin/env bash
# One-time Robosuite extras install. Run INSIDE a cap-x container:
#   bash /workspace/docker/setup-robosuite.sh
#
# Creates the cap-x BASE venv at /workspace/cap-x/.venv in the mounted checkout if
# it is not there yet, and syncs cap-x with the `robosuite` extra (Robosuite 1.5)
# into it.
#
# NOTE on mutual exclusion:
#   pyproject.toml declares `robosuite` and `libero` as conflicting extras
#   (they pin different robosuite forks -- 1.5 vs 1.4). LIBERO already
#   lives in its own separate venv at /workspace/cap-x/.venv-libero, so
#   adding robosuite to the BASE venv here does NOT disturb LIBERO setup.
set -euo pipefail

CAPX_DIR="${CAPX_DIR:-/workspace/cap-x}"
cd "${CAPX_DIR}"

git config --global --add safe.directory '*'

if [[ ! -d .venv ]]; then
    echo "[setup-robosuite] Creating the base venv .venv (Python 3.10)"
    uv venv -p 3.10 .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup-robosuite] Syncing cap-x with the robosuite extra into the base venv"
uv sync --active --extra robosuite

echo "[setup-robosuite] Running setup_macros.py so robosuite 1.5 has a private macros file"
python /workspace/cap-x/capx/third_party/robosuite/robosuite/scripts/setup_macros.py || true

echo
echo "[setup-robosuite] Done."
echo "Activate with:  source ${CAPX_DIR}/.venv/bin/activate"
echo "Smoke run:      bash /workspace/docker/run-robosuite-smoke.sh"
