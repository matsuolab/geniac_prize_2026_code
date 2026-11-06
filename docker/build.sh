#!/usr/bin/env bash
# Build the cap-x image.
# Uses BuildKit for the uv cache mount.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU_ARCH selects the compute capability baked into every CUDA extension the
# docker/setup-*.sh scripts compile in the container.
#   blackwell (RTX PRO 6000 Blackwell)  -> 12.0  (default)
#   ada       (RTX 6000 Ada, L40, L40S) -> 8.9
#   hopper    (H100/H200)               -> 9.0
#   ampere    (A100/A40/A6000)          -> 8.0
# Override: GPU_ARCH=ada bash build.sh   (image name is also derived from it)
GPU_ARCH="${GPU_ARCH:-blackwell}"

case "${GPU_ARCH}" in
    blackwell) ARCH_LIST="12.0" ;;
    ada)       ARCH_LIST="8.9"  ;;
    hopper)    ARCH_LIST="9.0"  ;;
    ampere)    ARCH_LIST="8.0"  ;;
    *)
        if [[ "${GPU_ARCH}" =~ ^[0-9]+\.[0-9]+(\+PTX)?(;[0-9]+\.[0-9]+(\+PTX)?)*$ ]]; then
            ARCH_LIST="${GPU_ARCH}"
        else
            echo "GPU_ARCH must be one of: blackwell, ada, hopper, ampere, or an explicit arch list (e.g. 8.9;9.0)" >&2
            exit 1
        fi
        ;;
esac

# Image name defaults to appending the arch tag so multiple variants coexist; an
# explicit arch list ("8.9;9.0+PTX") has characters a tag cannot hold.
USER=$(whoami)
IMAGE="${IMAGE:-${USER}_cap_x_$(tr ';.+' '___' <<< "${GPU_ARCH}" | tr '[:upper:]' '[:lower:]')}"

echo "[build] GPU_ARCH=${GPU_ARCH}  TORCH_CUDA_ARCH_LIST=${ARCH_LIST}  IMAGE=${IMAGE}"

DOCKER_BUILDKIT=0 docker build \
    -f "${HERE}/Dockerfile" \
    --build-arg TORCH_CUDA_ARCH_LIST="${ARCH_LIST}" \
    -t "${IMAGE}" \
    "${HERE}"

echo
echo "Built ${IMAGE}"
echo "Run with:  IMAGE=${IMAGE} ${HERE}/run.sh"
