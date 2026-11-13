#!/usr/bin/env bash
# Persistent Molmo VLM server for cap-x perception.
#
# capx/integrations/vision/molmo.py expects a vLLM-served OpenAI-compatible
# endpoint on the Unix socket ${CAPX_SOCKET_DIR:-~/.capx/sockets}/molmo.sock
# (CAPX_MOLMO_URL=http+unix://molmo.sock/v1) with model name
# "allenai/Molmo2-8B" (the molmo client sends `model:` matching that string,
# so vLLM's --served-model-name must match).
#
# Run in a SEPARATE shell inside the cap-x container, BEFORE starting the eval:
#   bash /workspace/docker/run-molmo-server.sh           # foreground, ctrl-C to stop
#   bash /workspace/docker/run-molmo-server.sh &         # background
#
# First invocation:
#   - creates /workspace/cap-x/.venv-vllm (Python 3.12)
#   - uv pip install vllm  (~5 min, pulls torch+cuda wheels)
#   - downloads Molmo2-8B weights (~16 GB) to ~/.cache/huggingface  (HF token must be set)
#
# Subsequent invocations: reuse the venv and cached weights, cold start ~30-60 s.
#
# Memory: with default GPU_UTIL=0.35 on a 96 GB card, vLLM caps at ~33 GB,
# leaving room for SAM3 + GraspNet + LIBERO sim workers on the same GPU.
set -euo pipefail

CAPX_DIR="${CAPX_DIR:-/workspace/cap-x}"
MOLMO_MODEL="${MOLMO_MODEL:-allenai/Molmo2-8B}"
GPU_UTIL="${GPU_UTIL:-0.35}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

cd "${CAPX_DIR}"

if [[ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "WARN: HF_TOKEN not set. Molmo weights download will fail if the repo is gated." >&2
fi

if [[ ! -d .venv-vllm ]]; then
    echo "[molmo] bootstrapping .venv-vllm (Python 3.12, vllm)"
    uv python install 3.12
    uv venv .venv-vllm --python 3.12
    # shellcheck disable=SC1091
    source .venv-vllm/bin/activate
    # --isolated: ignore cap-x's pyproject.toml override-dependencies
    # (transformers<5.0 etc.) which otherwise resolve to ancient versions.
    uv pip install --isolated --upgrade pip
    uv pip install --isolated "vllm" "huggingface_hub[hf_transfer]" "safetensors" "transformers>=4.45"
else
    # shellcheck disable=SC1091
    source .venv-vllm/bin/activate
fi

SOCKET_DIR="${CAPX_SOCKET_DIR:-${HOME}/.capx/sockets}"
mkdir -p -m 700 "${SOCKET_DIR}"
if [[ ! -O "${SOCKET_DIR}" ]] || (( 0$(stat -c %a "${SOCKET_DIR}") & 077 )); then
    echo "ERROR: ${SOCKET_DIR} must be owned by you with mode 700 (chmod 700 it)" >&2
    exit 1
fi
SOCKET="${SOCKET_DIR}/molmo.sock"

mkdir -p outputs/_molmo_logs
LOG="outputs/_molmo_logs/molmo_vllm.log"

echo "[molmo] starting vLLM"
echo "        model=${MOLMO_MODEL}"
echo "        socket=${SOCKET}"
echo "        gpu_memory_utilization=${GPU_UTIL}"
echo "        max_model_len=${MAX_MODEL_LEN}"
echo "        log -> ${LOG}"
echo "        client expects served-model-name=${MOLMO_MODEL}"

# trust-remote-code: Molmo ships custom modeling code on the HF Hub.
# served-model-name fixed to MOLMO_MODEL so molmo.py's `model:` field matches.
# HF_HUB_ENABLE_HF_TRANSFER=1 speeds up the first weight pull.
# vLLM binds the socket itself and fails on a file a dead server left behind.
[[ -S "${SOCKET}" ]] && rm -f "${SOCKET}"
umask 077
HF_HUB_ENABLE_HF_TRANSFER=1 exec python -m vllm.entrypoints.openai.api_server \
    --model "${MOLMO_MODEL}" \
    --served-model-name "${MOLMO_MODEL}" \
    --uds "${SOCKET}" \
    --dtype bfloat16 \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code 2>&1 | tee "${LOG}"
