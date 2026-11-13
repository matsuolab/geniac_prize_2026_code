#!/usr/bin/env bash
# Launch an interactive shell in the cap-x image with GPU access.
#
# This script lives at cap-x/docker/run.sh, so REPO_ROOT here resolves to the
# host's cap-x checkout. That entire directory is bind-mounted at
# /workspace/cap-x inside the container so:
#   - newly added files (e.g. capx/serving/openai_server.py, smoke configs)
#     are immediately visible inside the container,
#   - .venv-libero created by docker/setup-libero.sh persists on the host,
#   - outputs/ end up directly in the host tree.
#
# OPENAI_API_KEY from the host shell is forwarded into the container.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# After moving docker/ under cap-x/, REPO_ROOT IS the cap-x checkout itself
# (it used to be the parent geniac/ dir).
CAPX_HOST="${CAPX_HOST:-$(cd "${HERE}/.." && pwd)}"
USER=$(whoami)
GPU_ARCH="${GPU_ARCH:-blackwell}"
IMAGE="${IMAGE:-${USER}_cap_x_${GPU_ARCH}}"

# Pin a specific GPU with: GPUS='device=7' bash docker/run.sh
# Host is shared with other users, so by default we auto-pick a single GPU
# that currently has no running compute processes, rather than hardcoding
# an index or exposing all GPUs. Quoting matters for `device=...` form.
select_free_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[run.sh] warning: nvidia-smi not found on host; defaulting to GPU 0" >&2
        echo 0
        return
    fi
    local busy_uuids idx uuid
    busy_uuids="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u)"
    while IFS=',' read -r idx uuid; do
        idx="${idx// /}"; uuid="${uuid// /}"
        if ! grep -qF "${uuid}" <<< "${busy_uuids}"; then
            echo "${idx}"
            return
        fi
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)
    # Every GPU has at least one running process -- fall back to the one
    # with the least memory used rather than refusing to start.
    echo "[run.sh] warning: every GPU has a running process; picking the least-loaded one" >&2
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | sort -t',' -k2 -n | head -n1 | cut -d',' -f1 | tr -d ' '
}

if [[ -n "${GPUS:-}" ]]; then
    echo "[run.sh] using pinned GPUS=${GPUS}"
else
    GPUS="device=$(select_free_gpu)"
    echo "[run.sh] auto-selected free GPU: ${GPUS}"
fi

# The only published port is the web UI's (8200, behind its token). The LLM proxy,
# vLLM and the perception/IK servers listen on Unix sockets inside the container.
# Host is shared with other users, so the host-side port is randomized by default
# to avoid collisions. Pin it with: WEBUI_PORT=8200 bash docker/run.sh
WEBUI_PORT="${WEBUI_PORT:-$(shuf -i 20000-59999 -n1)}"
echo "[run.sh] host port ${WEBUI_PORT} -> container 8200 (web UI)"
PORTS=(-p "${WEBUI_PORT}:8200")

MOUNTS=(
    -v "${CAPX_HOST}:/workspace/cap-x"
    -v "${HERE}:/workspace/docker"
)

# A dataset directory shared between users on one host (e.g. the b1k datasets the
# repo symlinks to), mounted at the same path so those symlinks resolve inside.
SHARED_DATASET_DIR="${CAPX_SHARED_DATASET_DIR:-}"
if [[ -n "${SHARED_DATASET_DIR}" && -d "${SHARED_DATASET_DIR}" ]]; then
    MOUNTS+=( -v "${SHARED_DATASET_DIR}:${SHARED_DATASET_DIR}:ro" )
fi

# HuggingFace cache; point it at a shared directory so weights are downloaded once.
HF_CACHE_DIR="${CAPX_HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
MOUNTS+=( -v "${HF_CACHE_DIR}:/root/.cache/huggingface" )

mkdir -p "${CAPX_HOST}/outputs"

ENVS=(
    -e OMNI_KIT_ACCEPT_EULA=YES
    -e OMNIGIBSON_HEADLESS=1
    # NVIDIA container runtime exposes only compute+utility by default; Isaac
    # Sim's Vulkan renderer needs the `graphics` capability too. Request all
    # so Vulkan/EGL/display/video are available.
    -e NVIDIA_DRIVER_CAPABILITIES=all
    -e CAPX_WEB_UI_HOST=0.0.0.0  # so the published port 8200 reaches the web UI (it needs its token)
    -e "CAPX_WEB_UI_PUBLIC_URL=http://127.0.0.1:${WEBUI_PORT}"  # the token URL must name the host port
)
# OPENAI_API_KEY resolution order: env var > docker/api_keys.yaml.
# api_keys.yaml stores the secret directly (gitignored), format: `OPENAI_API_KEY: sk-...`.
API_KEYS_FILE="${API_KEYS_FILE:-${HERE}/api_keys.yaml}"
if [[ -z "${OPENAI_API_KEY:-}" && -f "${API_KEYS_FILE}" ]]; then
    OPENAI_API_KEY="$(awk -F': *' '/^OPENAI_API_KEY:/ {print $2; exit}' "${API_KEYS_FILE}" | tr -d '[:space:]"'"'"'')"
    if [[ -n "${OPENAI_API_KEY:-}" ]]; then
        echo "[run.sh] loaded OPENAI_API_KEY from ${API_KEYS_FILE}"
    fi
fi
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    ENVS+=( -e "OPENAI_API_KEY=${OPENAI_API_KEY}" )
else
    echo "[run.sh] warning: OPENAI_API_KEY not set on host and not found in ${API_KEYS_FILE}; either export it, write it to docker/api_keys.yaml, or place the key in cap-x/.openaikey before running evaluations." >&2
fi

# HuggingFace token for gated models (SAM3 etc.). Read from cap-x/hf_token
# (first non-blank line) so secrets stay out of shell history. The token is
# exported as both HF_TOKEN and HUGGING_FACE_HUB_TOKEN — different downstream
# libraries look for one or the other.
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${CAPX_HOST}/hf_token}"
if [[ -f "${HF_TOKEN_FILE}" ]]; then
    HF_TOKEN_VALUE="$(grep -v '^[[:space:]]*\(#\|$\)' "${HF_TOKEN_FILE}" | head -n1 | tr -d '[:space:]')"
    if [[ -n "${HF_TOKEN_VALUE}" ]]; then
        ENVS+=(
            -e "HF_TOKEN=${HF_TOKEN_VALUE}"
            -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN_VALUE}"
        )
        echo "[run.sh] loaded HF token from ${HF_TOKEN_FILE}"
    fi
elif [[ -n "${HF_TOKEN:-}" ]]; then
    ENVS+=(
        -e "HF_TOKEN=${HF_TOKEN}"
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}"
    )
fi

exec docker run --rm -it \
    --gpus "${GPUS}" \
    --shm-size=16g \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "${MOUNTS[@]}" \
    "${PORTS[@]}" \
    "${ENVS[@]}" \
    "${IMAGE}" \
    /bin/bash -lc 'echo "=== GPU visibility check ==="; \
        echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"; \
        nvidia-smi -L 2>&1 || echo "nvidia-smi not available"; \
        echo "==========================="; \
        exec bash'
