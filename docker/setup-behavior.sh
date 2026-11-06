#!/usr/bin/env bash
# One-time BEHAVIOR (Isaac Sim + OmniGibson) bootstrap.
# Run INSIDE the cap-x container:
#   bash /workspace/docker/setup-behavior.sh
#
# Disk / time budget (first run):
#   ~70-90 GB, 1-2 hours total
#     - 25 Isaac Sim wheels from pypi.nvidia.com (~15 GB)
#     - OmniGibson robot assets + BEHAVIOR-1K scenes + 2025 challenge tasks (~55 GB)
#     - curobo / pyroki / SAM3 (re)install
#
# All downloads go to /workspace/cap-x (bind-mounted host dir), so state
# persists across container restarts and a second run is fast.
set -euo pipefail

CAPX_DIR="${CAPX_DIR:-/workspace/cap-x}"
B1K_DIR="${CAPX_DIR}/capx/third_party/b1k"
SAM3_DIR="${CAPX_DIR}/capx/third_party/sam3"
SKIP_DATASET="${SKIP_DATASET:-0}"       # set 1 to skip the ~55 GB asset download (install only)

# Submodule git dirs are owned by host user; allow git introspection as root.
git config --global --add safe.directory '*'

# Note: cap-x README suggests removing /usr/share/vulkan/icd.d/nvidia_icd.json
# on multi-GPU HOSTS that have DUPLICATE ICDs installed. Inside a single-GPU
# container this file is the ONLY ICD and removing it breaks Vulkan entirely.
# We intentionally leave it in place here.

cd "${B1K_DIR}"

echo "[behavior] preparing Python 3.10 venv at ${B1K_DIR}/.venv"
uv python install 3.10
if [[ ! -d .venv ]]; then
    uv venv .venv --python 3.10
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# uv_install.sh pins Isaac Sim to cp310-only wheels; fail loud if the venv
# somehow ended up on another Python.
python --version | grep -q "Python 3.10" || {
    echo "ERROR: venv is not Python 3.10 (got $(python --version 2>&1))" >&2
    exit 1
}

# Defensive: uv_install.sh aborts if Isaac Sim env leaks in from the shell.
unset EXP_PATH CARB_APP_PATH ISAAC_PATH

INSTALL_ARGS=()
if [[ "${SKIP_DATASET}" != "1" ]]; then
    INSTALL_ARGS+=( --dataset --accept-dataset-tos )
else
    echo "[behavior] SKIP_DATASET=1 -> installing wheels only; assets must be fetched separately"
fi

# OmniGibson imports cv2 at module init but uv_install.sh does not declare it
# as a dep (it's a transitive OmniGibson requirement that's normally pulled by
# pip's resolver). Install the headless variant explicitly so the post-install
# verify step (`import omnigibson`) succeeds.
echo "[behavior] pre-installing opencv-python-headless (required by omnigibson)"
uv pip install opencv-python-headless

# During `docker build` there is no GPU/driver, so uv_install.sh's driver check
# (nvidia-smi) would see "no driver" and downgrade torch to 2.6.0+cu124 —
# mismatching the cu130 torch curobo was just built against. Shim nvidia-smi to
# report an R580 driver; the image targets R570+ hosts (Dockerfile.env-behavior).
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[behavior] nvidia-smi not found (image build?) -> shimming driver 580 for uv_install.sh"
    NVSMI_SHIM_DIR="$(mktemp -d)"
    printf '#!/bin/sh\necho 580.65.06\n' > "${NVSMI_SHIM_DIR}/nvidia-smi"
    chmod +x "${NVSMI_SHIM_DIR}/nvidia-smi"
    export PATH="${NVSMI_SHIM_DIR}:${PATH}"
fi

echo "[behavior] running uv_install.sh ${INSTALL_ARGS[*]:-}"
bash uv_install.sh "${INSTALL_ARGS[@]}"

# b1k's uv_install.sh contains a CaP-X-specific SAM3 install block, but its
# CAPX_ROOT calculation is one directory too shallow. Install SAM3 here using
# the absolute path from the cap-x checkout so BEHAVIOR does not depend on that
# broken conditional path check.
if [[ ! -d "${SAM3_DIR}" ]]; then
    echo "ERROR: SAM3 submodule missing at ${SAM3_DIR}" >&2
    echo "       Initialize submodules with: git submodule update --init --recursive capx/third_party/sam3" >&2
    exit 1
fi

echo "[behavior] installing SAM3 perception server from ${SAM3_DIR}"
# SAM3 currently imports pkg_resources at runtime. Setuptools >=82 no longer
# ships pkg_resources, so pin a compatible version explicitly.
uv pip install "setuptools<82"
uv pip install "${SAM3_DIR}" --no-deps
uv pip install \
    iopath einops timm "ftfy==6.1.1" decord pycocotools \
    regex typing_extensions huggingface_hub

python -c "import pkg_resources; from sam3.model.sam3_image_processor import Sam3Processor; print('[behavior] SAM3 import OK')"

# b1k's uv_install.sh only pulls OmniGibson's direct deps. cap-x's launch.py
# expects a handful of lightweight packages (fastapi, tyro, omegaconf, ...)
# that are normally provided by the cap-x base `uv sync`. Install them here
# so BEHAVIOR evals can run without falling back to PYTHONPATH hacks alone.
echo "[behavior] installing cap-x runtime deps (fastapi, tyro, omegaconf, ...)"
uv pip install \
    fastapi uvicorn openai tyro "pydantic>=2.7" \
    omegaconf pyyaml rich tqdm h5py einops decord pycocotools \
    msgpack "ray==2.48.0" mediapy matplotlib "imageio[ffmpeg]" \
    cloudpickle requests

# Post-install: OmniGibson's curobo install is from a Stanford fork, but the
# JIT (runtime-compiled) CUDA extensions expect header files that live under
# the cap-x submodule path. Copy them into site-packages so `curobolib` can
# find them at import time.
CUROBO_HEADERS_SRC="${CAPX_DIR}/capx/third_party/curobo/src/curobo/curobolib/cpp"
CUROBO_HEADERS_DST="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/curobo/curobolib/cpp"
if compgen -G "${CUROBO_HEADERS_SRC}/*.h" > /dev/null && [[ -d "${CUROBO_HEADERS_DST}" ]]; then
    echo "[behavior] copying curobo JIT headers: ${CUROBO_HEADERS_SRC} -> ${CUROBO_HEADERS_DST}"
    cp -v "${CUROBO_HEADERS_SRC}"/*.h "${CUROBO_HEADERS_DST}/"
else
    echo "[behavior] WARNING: curobo header source or destination missing; skipping copy" >&2
    echo "           src=${CUROBO_HEADERS_SRC}  dst=${CUROBO_HEADERS_DST}" >&2
fi

echo
echo "[behavior] done."
echo "Activate with:  source ${B1K_DIR}/.venv/bin/activate"
echo "Smoke run:      bash /workspace/docker/run-behavior-smoke.sh"
