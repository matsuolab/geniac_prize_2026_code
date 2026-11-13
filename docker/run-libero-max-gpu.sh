#!/usr/bin/env bash
# Max-GPU smoke: same LIBERO task but with the FULL perception pipeline.
#
#   - non-privileged FrankaLiberoApi  -> calls SAM3 + ContactGraspNet
#   - parallel workers (default 4)    -> concurrent perception requests
#   - 4 trials so the run finishes in a few minutes
#
# Requires:
#   - cap-x/hf_token (already wired by docker/run.sh)  for SAM3 weights download
#   - OPENAI_API_KEY                                    for the LLM
#   - first run downloads ~5 GB of SAM3 weights into ~/.cache/huggingface
#
# Watch the GPU in another shell on the host:
#   watch -n 1 nvidia-smi
# or inside the container:
#   nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 1
#
# Approximate steady-state VRAM (single GPU):
#   SAM3 ~10 GB + ContactGraspNet ~4 GB + PyRoKi ~1 GB
#   + LIBERO sim ~2 GB per worker
#   total ~20-25 GB at NUM_WORKERS=4
set -euo pipefail

CAPX_DIR="${CAPX_DIR:-/workspace/cap-x}"
MODEL="${MODEL:-gpt-5}"
CONFIG="${CONFIG:-env_configs/libero/franka_libero_spatial_0.yaml}"
SOCKET="${CAPX_SOCKET_DIR:-${HOME}/.capx/sockets}/model.sock"
NUM_WORKERS="${NUM_WORKERS:-4}"
TOTAL_TRIALS="${TOTAL_TRIALS:-4}"

if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ ! -f "${CAPX_DIR}/.openaikey" ]]; then
    echo "ERROR: need OPENAI_API_KEY env var or ${CAPX_DIR}/.openaikey file" >&2
    exit 1
fi

if [[ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "WARN: HF_TOKEN not in env. SAM3 is a gated repo and will fail to download." >&2
    echo "      Re-enter the container via docker/run.sh after putting the token in cap-x/hf_token." >&2
fi

cd "${CAPX_DIR}"

if [[ ! -d .venv-libero ]]; then
    echo "ERROR: .venv-libero missing — run docker/setup-libero.sh first" >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv-libero/bin/activate

LOG_DIR="${CAPX_DIR}/outputs/_max_gpu_logs"
mkdir -p "${LOG_DIR}"
PROXY_LOG="${LOG_DIR}/openai_proxy.log"

echo "[max-gpu] Starting OpenAI proxy on ${SOCKET} (log: ${PROXY_LOG})"
python capx/serving/openai_server.py --socket "${SOCKET}" > "${PROXY_LOG}" 2>&1 &
PROXY_PID=$!
trap 'echo "[max-gpu] Stopping proxy (pid ${PROXY_PID})"; kill "${PROXY_PID}" 2>/dev/null || true' EXIT

for i in {1..30}; do
    if curl -fsS --unix-socket "${SOCKET}" "http://localhost/health" >/dev/null 2>&1; then
        echo "[max-gpu] proxy ready"
        break
    fi
    if ! kill -0 "${PROXY_PID}" 2>/dev/null; then
        echo "[max-gpu] proxy died; log:"; tail -40 "${PROXY_LOG}"; exit 1
    fi
    sleep 1
done

echo "[max-gpu] Launching: model=${MODEL} workers=${NUM_WORKERS} trials=${TOTAL_TRIALS}"
echo "[max-gpu] First run will download SAM3 weights (~5 GB) on demand."
python capx/envs/launch.py \
    --config-path "${CONFIG}" \
    --model "${MODEL}" \
    --server-url "http+unix://model.sock/chat/completions" \
    --max-tokens "${MAX_TOKENS:-32768}" \
    --total-trials "${TOTAL_TRIALS}" \
    --num-workers "${NUM_WORKERS}"

echo "[max-gpu] Done. Outputs under ${CAPX_DIR}/outputs/franka_libero_spatial_0/"
