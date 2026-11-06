#!/usr/bin/env bash
# One-time LIBERO venv bootstrap. Run INSIDE a cap-x container:
#   docker run --rm -it --gpus all -v ...  <image> \
#       bash /workspace/cap-x/../docker/setup-libero.sh
# or, from inside an already-running container:
#   bash /workspace/setup-libero.sh
#
# Creates /workspace/cap-x/.venv-libero (Python 3.12) with libero + contactgraspnet
# extras, and writes ~/.libero/config.yaml so headless LIBERO tasks can find assets.
set -euo pipefail

CAPX_DIR="${CAPX_DIR:-/workspace/cap-x}"
cd "${CAPX_DIR}"

# When cap-x is bind-mounted from the host, the working tree is owned by the
# host user but we run as root inside the container. setuptools-scm in
# nvidia-curobo / verl runs `git describe`, which git refuses unless the dir
# is whitelisted as safe.
echo "[setup-libero] Whitelisting bind-mounted repos for git"
git config --global --add safe.directory '*'

echo "[setup-libero] Ensuring Python 3.12 is available via uv..."
uv python install 3.12

if [[ ! -d .venv-libero ]]; then
    echo "[setup-libero] Creating .venv-libero (Python 3.12)"
    uv venv .venv-libero --python 3.12
else
    echo "[setup-libero] .venv-libero already exists — reusing"
fi

echo "[setup-libero] Installing libero + contactgraspnet extras"
# shellcheck disable=SC1091
source .venv-libero/bin/activate
uv sync --active --extra libero --extra contactgraspnet

echo "[setup-libero] Writing ~/.libero/config.yaml"
mkdir -p "${HOME}/.libero"
cat > "${HOME}/.libero/config.yaml" <<EOF
assets: ${CAPX_DIR}/capx/third_party/LIBERO-PRO/libero/libero/assets
bddl_files: ${CAPX_DIR}/capx/third_party/LIBERO-PRO/libero/libero/bddl_files
benchmark_root: ${CAPX_DIR}/capx/third_party/LIBERO-PRO/libero/libero
datasets: ${CAPX_DIR}/capx/third_party/LIBERO-PRO/libero/libero/../datasets
init_states: ${CAPX_DIR}/capx/third_party/LIBERO-PRO/libero/libero/init_files
EOF

echo
echo "[setup-libero] Done."
echo "Activate with:  source ${CAPX_DIR}/.venv-libero/bin/activate"
