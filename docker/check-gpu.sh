#!/usr/bin/env bash
# Verify which host GPU(s) a cap-x container actually has access to.
#
# Usage:
#   bash docker/check-gpu.sh                  # list all host GPUs with busy/free status
#   bash docker/check-gpu.sh <container_id>   # also resolve which host GPU the container sees
#
# docker/run.sh auto-picks a free GPU (no running compute processes) unless
# GPUS is pinned explicitly, so there's no fixed "expected" index to check
# against -- this script cross-references the container's GPU by UUID
# instead of assuming one.
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found on host" >&2
    exit 1
fi

echo "=== host GPUs ==="
BUSY_UUIDS="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u)"
while IFS=',' read -r idx uuid mem util; do
    idx="${idx// /}"; uuid="${uuid// /}"; mem="${mem# }"; util="${util# }"
    status="free"
    grep -qF "${uuid}" <<< "${BUSY_UUIDS}" && status="busy"
    echo "  GPU ${idx}: ${uuid}  mem_used=${mem}  util=${util}  [${status}]"
done < <(nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)

if [[ "$#" -eq 0 ]]; then
    echo
    echo "(no container_id provided -- pass one to resolve which host GPU it's pinned to)"
    exit 0
fi

CONTAINER="$1"

echo
echo "=== container: ${CONTAINER} ==="
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found on host" >&2
    exit 2
fi
docker exec "${CONTAINER}" nvidia-smi -L | sed 's/^/  /'

CONTAINER_UUIDS=$(docker exec "${CONTAINER}" nvidia-smi -L | grep -oE 'GPU-[A-Za-z0-9-]+')
COUNT=$(echo "${CONTAINER_UUIDS}" | grep -c . || true)

echo
echo "=== verdict ==="
if [[ "${COUNT}" -ne 1 ]]; then
    echo "  container sees ${COUNT} GPU(s) -- docker/run.sh pins exactly 1" >&2
    exit 3
fi

HOST_MATCH=$(nvidia-smi -L | grep -F "${CONTAINER_UUIDS}" || true)
if [[ -n "${HOST_MATCH}" ]]; then
    echo "  OK: container is pinned to host ${HOST_MATCH}"
else
    echo "  MISMATCH: container UUID ${CONTAINER_UUIDS} not found among host GPUs" >&2
    exit 4
fi
