#!/usr/bin/env bash
# Robosuite eval/rollout with a locally served Qwen3.6-27B. Run INSIDE a cap-x
# container after setup-robosuite.sh:
#
#   bash docker/run.sh                                   # enter the container
#   # inside container (separate shells):
#   bash /workspace/docker/run-qwen-server.sh &          # 1. LLM server (model.sock)
#   bash /workspace/docker/run-molmo-server.sh &         # 2. only for S3 (*_reduced_api*) configs:
#                                                        #    point_prompt_molmo needs molmo.sock
#   bash /workspace/docker/run-robosuite-qwen.sh         # 3. rollout
#
# Perception servers (sam3.sock, graspnet.sock, pyroki.sock in $CAPX_SOCKET_DIR) are
# auto-started by launch.py from the YAML's api_servers list (live sockets are reused).
#
# GPU pinning notes:
#   - CUDA_VISIBLE_DEVICES pins CUDA work (perception servers, sim physics).
#   - MuJoCo's EGL renderer IGNORES CUDA_VISIBLE_DEVICES and enumerates devices
#     itself -> pin it with MUJOCO_EGL_DEVICE_ID (EGL device order differs from
#     the CUDA order; probe with nvidia-smi if rendering must stay on one GPU).
set -euo pipefail

CAPX_DIR="${CAPX_DIR:-/workspace/cap-x}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
CONFIG="${CONFIG:-env_configs/cube_lifting/franka_robosuite_cube_lifting_reduced_api.yaml}"
SOCKET="${CAPX_SOCKET_DIR:-${HOME}/.capx/sockets}/model.sock"
TRIALS="${TRIALS:-10}"
WORKERS="${WORKERS:-3}"
TEMP="${TEMP:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/qwen36_rollout}"
EGL_DEVICE="${EGL_DEVICE:-0}"

cd "${CAPX_DIR}"

if [[ ! -d .venv ]]; then
    echo "ERROR: ${CAPX_DIR}/.venv (base venv) missing -- run docker/setup-robosuite.sh first" >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# vLLM endpoint needs the /v1 prefix (unlike the OpenAI proxy used by the smoke script).
SERVER_URL="http+unix://model.sock/v1/chat/completions"
if ! curl -fsS --unix-socket "${SOCKET}" "http://localhost/v1/models" >/dev/null 2>&1; then
    echo "ERROR: no vLLM server on ${SOCKET} -- start docker/run-qwen-server.sh first" >&2
    exit 1
fi

echo "[qwen-rollout] model=${MODEL} config=${CONFIG} trials=${TRIALS} workers=${WORKERS}"
# NOTE: launch.py inserts the sanitized model name into the output path:
#   ${OUTPUT_DIR%/*}/Qwen_Qwen3.6-27B/$(basename ${OUTPUT_DIR})/trial_XX_..._taskcompleted_{0,1}
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE}" \
python capx/envs/launch.py \
    --config-path "${CONFIG}" \
    --model "${MODEL}" \
    --server-url "${SERVER_URL}" \
    --total-trials "${TRIALS}" \
    --num-workers "${WORKERS}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMP}" \
    --record-video False \
    --use-img-differencing False \
    --output-dir "${OUTPUT_DIR}"

echo "[qwen-rollout] Done. Success label = dir name suffix taskcompleted_1."
