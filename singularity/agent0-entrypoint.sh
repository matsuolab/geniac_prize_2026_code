#!/usr/bin/env bash
# Entrypoint of the CaP-Agent0 submission image (singularity/capx-submission-agent0.def).
#
# Starts the model server on loopback, waits for it, then runs the agent against
# whatever gateway the organizer wired up (CAPX_ENV_SOCKET / CAPX_ENV_HOST). The
# perception and IK sidecars are NOT started here — capx.submission.runner launches
# the ones each task config asks for (`api_servers`) and stops them afterwards.
#
# The eval queue gives this container a loopback-only network namespace, which is
# exactly enough for these local servers and nothing else.
#
# Env:
#   CAPX_MODEL         name vLLM serves the model under, which the agent then asks
#                      for whatever the runner config says (default Qwen/Qwen3.6-27B)
#   CAPX_MODEL_PATH    where its weights are mounted (submission.yaml `weights`);
#                      defaults to CAPX_MODEL, which only works with network access
#   CAPX_SAM3_CHECKPOINT  mounted sam3.pt, read by the SAM3 server the runner starts
#   CAPX_RUNNER_CONFIG runner YAML, absolute or relative to /workspace/capx
#                      (default env_configs/gateway/runner.yaml)
#   CAPX_SERVE_PORT    vLLM port (default 8110); the agent is pointed at it
#   CAPX_GPU_UTIL / CAPX_MAX_MODEL_LEN / CAPX_MAX_NUM_SEQS
#   CAPX_SERVE_TIMEOUT_S  how long to wait for vLLM to come up (default 1800)
#   CAPX_SERVE_EXTRA_ARGS extra vLLM server flags (default: --attention-backend TRITON_ATTN)
#   CAPX_ENFORCE_EAGER 1 runs vLLM eagerly instead of compiling (default 0)
#   CAPX_MAX_LANES     the most lanes the gateway accepts (set by the organizer)
#   CAPX_MOLMO_URL / CAPX_MOLMO_MODEL  fallback object localizer endpoint
#                      (defaults to this container's own model server)
#   CAPX_OUTPUT_DIR    writable directory for the agent's artifacts (bound in)
set -euo pipefail
cd /workspace/capx

MODEL="${CAPX_MODEL:-Qwen/Qwen3.6-27B}"
MODEL_PATH="${CAPX_MODEL_PATH:-${MODEL}}"
RUNNER_CONFIG="${CAPX_RUNNER_CONFIG:-env_configs/gateway/runner.yaml}"
[[ "${RUNNER_CONFIG}" = /* ]] || RUNNER_CONFIG="/workspace/capx/${RUNNER_CONFIG}"
PORT="${CAPX_SERVE_PORT:-8110}"
GPU_UTIL="${CAPX_GPU_UTIL:-0.72}"
# The runner opens `--concurrency` or the runner config's `concurrency` lanes, capped
# at CAPX_MAX_LANES; the server's batch is sized for that many trials at once.
CONCURRENCY="$(python - "${RUNNER_CONFIG}" "${CAPX_MAX_LANES:-1}" "$@" <<'PY'
import argparse, sys, yaml
config, max_lanes, *flags = sys.argv[1:]
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--concurrency", type=int)
lanes = parser.parse_known_args(flags)[0].concurrency
if lanes is None:
    lanes = int(yaml.safe_load(open(config)).get("concurrency", 1))
print(min(lanes, int(max_lanes)))
PY
)"
# A multiturn prompt carries the executed code, the console output and the images,
# and grows every turn; once it overflows, vLLM answers 400 and the trial dies.
MAX_MODEL_LEN="${CAPX_MAX_MODEL_LEN:-65536}"
# Ten generations per turn per lane: the reference config runs the parallel
# ensemble (nine candidates plus a synthesis).
MAX_NUM_SEQS="${CAPX_MAX_NUM_SEQS:-$(( CONCURRENCY * 10 > 32 ? CONCURRENCY * 10 : 32 ))}"
SERVE_TIMEOUT_S="${CAPX_SERVE_TIMEOUT_S:-1800}"

# Default under CAPX_OUTPUT_DIR, which the queue binds out of the container:
# a vLLM log in the private /tmp would be unreadable exactly when it matters.
LOGS="${CAPX_AGENT_LOG_DIR:-${CAPX_OUTPUT_DIR:-/tmp/capx-agent0}/logs}"
mkdir -p "${LOGS}"

# Kernel/compile caches, pinned under the writable HOME the queue binds in. On
# their defaults they land in the container's session tmpfs, where Triton's JIT
# runs out of space mid-compile and takes the engine down with it.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${HOME}/.triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${HOME}/.cache/torchinductor}"
mkdir -p "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}"

# With no network, flashinfer cannot fetch prebuilt sm_103a kernels and falls back
# to nvcc, which this runtime-flavoured base image does not carry — so both the
# sampler and the attention backend have to be steered away from it.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# The attention backend is NOT settable by environment in vLLM 0.26: the selector
# reads --attention-backend and otherwise picks FLASHINFER on its own. Extra server
# flags are exposed so a fork can retune the server without rebuilding the image.
SERVE_EXTRA_ARGS="${CAPX_SERVE_EXTRA_ARGS:---attention-backend TRITON_ATTN}"
# Compiling and capturing cudagraphs works offline and decodes several times faster
# than eager mode, for under a minute of extra startup.
ENFORCE_EAGER="${CAPX_ENFORCE_EAGER:-0}"
[ "${ENFORCE_EAGER}" = "0" ] && EAGER_FLAG="" || EAGER_FLAG="--enforce-eager"

# The fallback object localizer (capx/integrations/vision/molmo.py) runs when
# SAM3's language path finds nothing. Its default Molmo server does not exist in
# this container, so point it at the model already served here, which takes images.
export CAPX_MOLMO_URL="${CAPX_MOLMO_URL:-http://127.0.0.1:${PORT}/v1}"
export CAPX_MOLMO_MODEL="${CAPX_MOLMO_MODEL:-${MODEL}}"

echo "[agent0] serving ${MODEL} from ${MODEL_PATH} on 127.0.0.1:${PORT}"
echo "[agent0] fallback localizer -> ${CAPX_MOLMO_URL} (${CAPX_MOLMO_MODEL})"
# fp8 on the fly keeps a 27B inside a single card's budget alongside the sim's
# rendering context; it needs compute capability >= 8.9 (Ada/Hopper/Blackwell).
/opt/venv-serve/bin/python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    ${EAGER_FLAG} \
    --reasoning-parser qwen3 \
    ${SERVE_EXTRA_ARGS} > "${LOGS}/vllm.log" 2>&1 &
SERVE_PID=$!
stop_server() {
    kill "${SERVE_PID}" 2>/dev/null || true
    wait "${SERVE_PID}" 2>/dev/null || true
}
trap stop_server EXIT

deadline=$(( SECONDS + SERVE_TIMEOUT_S ))
while :; do
    # A server that accepts the connection and never answers must not hold the
    # loop past the deadline.
    remaining=$(( deadline - SECONDS ))
    probe_s=$(( remaining < 10 ? remaining : 10 ))
    (( probe_s > 0 )) && curl -sf --connect-timeout "${probe_s}" --max-time "${probe_s}" \
        "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1 && break
    if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
        echo "[agent0] vLLM died during startup; last log lines:" >&2
        tail -40 "${LOGS}/vllm.log" >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "[agent0] vLLM did not become healthy within ${SERVE_TIMEOUT_S}s" >&2
        tail -40 "${LOGS}/vllm.log" >&2
        exit 1
    fi
    sleep 5
done
echo "[agent0] vLLM ready after ${SECONDS}s"

# Run from a writable directory. The agent executes LLM-written code, and that
# code saves files by relative path (depth_image.jpg and friends) — from inside
# the read-only image those are OSError. Imports still resolve through PYTHONPATH.
OUT="${CAPX_OUTPUT_DIR:-/tmp/capx-agent0/outputs}"
mkdir -p "${OUT}"
cd "${OUT}"
# Not exec'd: this shell stays to stop vLLM when the runner exits. The model and URL
# are the ones served above; flags passed to the container come after and win.
python -m capx.submission.run \
    --config-path "${RUNNER_CONFIG}" \
    --model "${MODEL}" \
    --server-url "http://127.0.0.1:${PORT}/v1/chat/completions" \
    --output-dir "${OUT}" "$@" &
RUNNER_PID=$!
trap 'kill -TERM "${RUNNER_PID}" 2>/dev/null || true' TERM INT
status=0
wait "${RUNNER_PID}" || status=$?
# A trapped signal ends the first wait early; let the runner finish shutting down.
if kill -0 "${RUNNER_PID}" 2>/dev/null; then
    status=0
    wait "${RUNNER_PID}" || status=$?
fi
exit "${status}"
