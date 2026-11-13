#!/usr/bin/env bash
# BEHAVIOR smoke test. Run INSIDE the cap-x container after setup-behavior.sh.
#
#   bash /workspace/docker/run-behavior-smoke.sh
#
# Default: oracle config (use_oracle_code=true). No LLM calls — we just check
# that Isaac Sim + OmniGibson + SAM3 + GraspNet + the R1Pro controller all
# wire up and a reference solution runs. Switch to the non-oracle config with
#
#   MODE=llm bash /workspace/docker/run-behavior-smoke.sh
#
# to go through gpt-5 + OpenAI proxy.
set -euo pipefail

CAPX_DIR="${CAPX_DIR:-/workspace/cap-x}"
B1K_DIR="${CAPX_DIR}/capx/third_party/b1k"
MODE="${MODE:-oracle}"
MODEL="${MODEL:-gpt-5}"
SOCKET="${CAPX_SOCKET_DIR:-${HOME}/.capx/sockets}/model.sock"
TOTAL_TRIALS="${TOTAL_TRIALS:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"

case "${MODE}" in
    oracle) CONFIG="env_configs/r1pro/r1pro_pick_up_radio_oracle.yaml" ;;
    llm)    CONFIG="env_configs/r1pro/r1pro_pick_up_radio.yaml" ;;
    *)      echo "MODE must be 'oracle' or 'llm'" >&2; exit 1 ;;
esac

cd "${CAPX_DIR}"

if [[ ! -d "${B1K_DIR}/.venv" ]]; then
    echo "ERROR: ${B1K_DIR}/.venv missing — run docker/setup-behavior.sh first" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${B1K_DIR}/.venv/bin/activate"

# BEHAVIOR-specific env. Isaac Sim reads OMNIGIBSON_GPU_ID (literal EGL index),
# NOT CUDA_VISIBLE_DEVICES. Inside the container with --gpus '"device=N"'
# there is only one GPU visible as index 0.
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1
export OMNIGIBSON_GPU_ID=0

# b1k's uv_install.sh only installs OmniGibson, not the cap-x package itself.
# Expose cap-x sources via PYTHONPATH so `import capx` works in this venv.
export PYTHONPATH="${CAPX_DIR}:${PYTHONPATH:-}"

LOG_DIR="${CAPX_DIR}/outputs/_behavior_logs"
mkdir -p "${LOG_DIR}"

PROXY_PID=""
if [[ "${MODE}" == "llm" ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ ! -f "${CAPX_DIR}/.openaikey" ]]; then
        echo "ERROR: MODE=llm needs OPENAI_API_KEY env var or ${CAPX_DIR}/.openaikey" >&2
        exit 1
    fi
    # The b1k venv does not ship fastapi/uvicorn/openai, so run the proxy
    # with whichever cap-x venv has those installed. Try the base venv first
    # (setup-robosuite.sh) and fall back to the libero venv (which also pulls the
    # cap-x base deps).
    PROXY_PY=""
    for candidate in "${CAPX_DIR}/.venv/bin/python" "${CAPX_DIR}/.venv-libero/bin/python"; do
        if [[ -x "${candidate}" ]]; then
            PROXY_PY="${candidate}"
            break
        fi
    done
    if [[ -z "${PROXY_PY}" ]]; then
        echo "ERROR: no cap-x venv with fastapi/uvicorn found." >&2
        echo "       Expected one of:" >&2
        echo "         ${CAPX_DIR}/.venv/bin/python          (base sync)" >&2
        echo "         ${CAPX_DIR}/.venv-libero/bin/python   (libero venv)" >&2
        exit 1
    fi
    PROXY_LOG="${LOG_DIR}/openai_proxy.log"
    echo "[behavior] starting OpenAI proxy on ${SOCKET} via ${PROXY_PY} (log: ${PROXY_LOG})"
    "${PROXY_PY}" capx/serving/openai_server.py --socket "${SOCKET}" > "${PROXY_LOG}" 2>&1 &
    PROXY_PID=$!
    trap '[[ -n "${PROXY_PID}" ]] && kill "${PROXY_PID}" 2>/dev/null || true' EXIT

    for _ in {1..30}; do
        if curl -fsS --unix-socket "${SOCKET}" "http://localhost/health" >/dev/null 2>&1; then
            echo "[behavior] proxy ready"
            break
        fi
        if ! kill -0 "${PROXY_PID}" 2>/dev/null; then
            echo "[behavior] proxy died; log:"; tail -40 "${PROXY_LOG}"; exit 1
        fi
        sleep 1
    done
fi

echo "[behavior] MODE=${MODE} CONFIG=${CONFIG} trials=${TOTAL_TRIALS} workers=${NUM_WORKERS}"
if [[ "${MODE}" == "llm" ]]; then
    python capx/envs/launch.py \
        --config-path "${CONFIG}" \
        --model "${MODEL}" \
        --server-url "http+unix://model.sock/chat/completions" \
        --max-tokens "${MAX_TOKENS:-32768}" \
        --total-trials "${TOTAL_TRIALS}" \
        --num-workers "${NUM_WORKERS}"
else
    # oracle ignores --model / --server-url; still pass sane defaults so tyro is happy.
    python capx/envs/launch.py \
        --config-path "${CONFIG}" \
        --total-trials "${TOTAL_TRIALS}" \
        --num-workers "${NUM_WORKERS}"
fi

echo "[behavior] done. Outputs under ${CAPX_DIR}/outputs/pick_up_radio_oracle/"
