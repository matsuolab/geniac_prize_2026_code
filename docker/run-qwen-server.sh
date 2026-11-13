#!/usr/bin/env bash
# Persistent Qwen3.6-27B server for cap-x code generation.
#
# Serves an OpenAI-compatible endpoint on the Unix socket
# ${CAPX_SOCKET_DIR:-~/.capx/sockets}/model.sock, in a directory only you can enter.
# capx/llm/client.py sends any model name not in its hardcoded lists as a plain
# OpenAI chat-completions payload, so pass to launch.py:
#   --model Qwen/Qwen3.6-27B --server-url http+unix://model.sock/v1/chat/completions
# (NOTE: the /v1 prefix is required for vLLM; /chat/completions alone is the
#  OpenAI-proxy path used by run-robosuite-smoke.sh and will 404 here.)
#
# Run in a SEPARATE shell inside the cap-x container, BEFORE starting the eval:
#   bash /workspace/docker/run-qwen-server.sh            # foreground, ctrl-C to stop
#   bash /workspace/docker/run-qwen-server.sh &          # background
#
# First invocation:
#   - creates ${CAPX_DIR}/.venv-serve (Python 3.12)
#   - installs latest vLLM + transformers>=5 (Qwen3.6 arch `qwen3_5` needs both;
#     vLLM <=0.15 does not know the arch, transformers<5 is rejected by vLLM >=0.24)
#   - aligns torch to TORCH_BACKEND (default cu130: recent vLLM wheels are built
#     against CUDA 13 -> host driver must be R580+; plain `uv pip install vllm`
#     may resolve torch +cu128 and then fail with `libcudart.so.13: not found`)
#   - downloads Qwen3.6-27B weights (~52 GB) to ~/.cache/huggingface
#
# Memory: fp8 on-the-fly quantization (requires compute capability >= 8.9, i.e.
# Ada/Hopper/Blackwell) fits the 27B in ~28 GB weights + fp8 KV cache; with
# GPU_UTIL=0.72 on a 48 GB card the server uses ~34 GB and tolerates neighbors.
# MAX_MODEL_LEN >= 16384 is required for multiturn configs (2nd-turn prompts
# carry executed code + console output; 12288 caused 400 Bad Request).
#
# Thinking mode is ON by default; --reasoning-parser qwen3 splits the thinking
# into `reasoning_content` so `message.content` stays clean final code.
set -euo pipefail

CAPX_DIR="${CAPX_DIR:-/workspace/cap-x}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.6-27B}"
GPU_UTIL="${GPU_UTIL:-0.72}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
TORCH_BACKEND="${TORCH_BACKEND:-cu130}"
# Which GPU to serve on (index within the GPUs visible to this container).
SERVE_GPU="${SERVE_GPU:-0}"

cd "${CAPX_DIR}"

if [[ ! -d .venv-serve ]]; then
    echo "[qwen] bootstrapping .venv-serve (Python 3.12, vllm + transformers>=5)"
    uv python install 3.12
    uv venv .venv-serve --python 3.12
    # shellcheck disable=SC1091
    source .venv-serve/bin/activate
    # --isolated: ignore cap-x's pyproject.toml override-dependencies
    # (transformers<5.0) which would silently keep transformers 4.x installed.
    #
    # vllm and torch must be installed together: vLLM's precompiled
    # flash-attention extensions are ABI-tied to whatever torch it resolves.
    uv pip install --isolated --torch-backend="${TORCH_BACKEND}" --upgrade \
        vllm "transformers>=5" "huggingface_hub[hf_transfer]"
else
    # shellcheck disable=SC1091
    source .venv-serve/bin/activate
fi

SOCKET_DIR="${CAPX_SOCKET_DIR:-${HOME}/.capx/sockets}"
mkdir -p -m 700 "${SOCKET_DIR}"
if [[ ! -O "${SOCKET_DIR}" ]] || (( 0$(stat -c %a "${SOCKET_DIR}") & 077 )); then
    echo "ERROR: ${SOCKET_DIR} must be owned by you with mode 700 (chmod 700 it)" >&2
    exit 1
fi
SOCKET="${SOCKET_DIR}/model.sock"

mkdir -p outputs/_qwen_logs
LOG="outputs/_qwen_logs/qwen_vllm.log"

echo "[qwen] starting vLLM"
echo "       model=${QWEN_MODEL}  gpu=${SERVE_GPU}"
echo "       gpu_memory_utilization=${GPU_UTIL}  max_model_len=${MAX_MODEL_LEN}"
echo "       log -> ${LOG}"
echo "       socket: ${SOCKET}"

# vLLM binds the socket itself and fails on a file a dead server left behind.
# Only that file is removed: curl exit 7 means nothing accepted the connection.
if [[ -S "${SOCKET}" ]]; then
    rc=0
    curl -s -o /dev/null --max-time 5 --unix-socket "${SOCKET}" http://localhost/ || rc=$?
    if (( rc != 7 )); then
        echo "ERROR: a server is already listening on ${SOCKET}" >&2
        exit 1
    fi
    rm -f "${SOCKET}"
fi
umask 077
HF_HUB_ENABLE_HF_TRANSFER=1 CUDA_VISIBLE_DEVICES="${SERVE_GPU}" \
exec python -m vllm.entrypoints.openai.api_server \
    --model "${QWEN_MODEL}" \
    --served-model-name "${QWEN_MODEL}" \
    --uds "${SOCKET}" \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enforce-eager \
    --reasoning-parser qwen3 2>&1 | tee "${LOG}"
