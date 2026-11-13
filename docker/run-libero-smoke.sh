#!/usr/bin/env bash
# LIBERO smoke test. Run INSIDE a cap-x container after setup-libero.sh has succeeded.
#
#   export OPENAI_API_KEY=sk-...          # on the host
#   bash docker/run.sh                    # enter the container (forwards OPENAI_API_KEY)
#   # inside container:
#   bash /workspace/setup-libero.sh       # one-time
#   bash /workspace/run-libero-smoke.sh
#
# Does:
#   1. Starts the OpenAI proxy on the Unix socket model.sock in the background.
#   2. Runs 2 trials of libero_spatial task 0 (privileged API -> no SAM3/GraspNet).
#   3. Writes outputs/ videos and a result JSON, then tears the proxy down.
set -euo pipefail

CAPX_DIR="${CAPX_DIR:-/workspace/cap-x}"
MODEL="${MODEL:-gpt-5}"
CONFIG="${CONFIG:-env_configs/libero/franka_libero_spatial_0_smoke.yaml}"
export CAPX_SOCKET_DIR="$(realpath -m "${CAPX_SOCKET_DIR:-${HOME}/.capx/sockets}")"
SOCKET="${CAPX_SOCKET_DIR}/model.sock"

if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ ! -f "${CAPX_DIR}/.openaikey" ]]; then
    echo "ERROR: need OPENAI_API_KEY env var or ${CAPX_DIR}/.openaikey file" >&2
    exit 1
fi

cd "${CAPX_DIR}"

if [[ ! -d .venv-libero ]]; then
    echo "ERROR: .venv-libero missing — run docker/setup-libero.sh first" >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv-libero/bin/activate

LOG_DIR="${CAPX_DIR}/outputs/_smoke_logs"
mkdir -p "${LOG_DIR}"
PROXY_LOG="${LOG_DIR}/openai_proxy.log"

echo "[smoke] Starting OpenAI proxy on ${SOCKET} (log: ${PROXY_LOG})"
python capx/serving/openai_server.py --socket "${SOCKET}" > "${PROXY_LOG}" 2>&1 &
PROXY_PID=$!
trap 'echo "[smoke] Stopping proxy (pid ${PROXY_PID})"; kill "${PROXY_PID}" 2>/dev/null || true' EXIT

# Wait for /health
for i in {1..30}; do
    if curl -fsS --unix-socket "${SOCKET}" "http://localhost/health" >/dev/null 2>&1; then
        echo "[smoke] proxy ready"
        break
    fi
    if ! kill -0 "${PROXY_PID}" 2>/dev/null; then
        echo "[smoke] proxy died; log:"; tail -40 "${PROXY_LOG}"; exit 1
    fi
    sleep 1
done

echo "[smoke] Launching eval: model=${MODEL} config=${CONFIG}"
python capx/envs/launch.py \
    --config-path "${CONFIG}" \
    --model "${MODEL}" \
    --server-url "http+unix://model.sock/chat/completions" \
    --max-tokens "${MAX_TOKENS:-8192}" \
    --total-trials 2 \
    --num-workers 1

echo "[smoke] Done. Outputs under ${CAPX_DIR}/outputs/franka_libero_spatial_0_smoke/"
