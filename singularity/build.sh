#!/usr/bin/env bash
# Build CapX Singularity images. Run INSIDE a Slurm allocation — never on the
# login node (the cluster's own guidance: image builds blow past the login node's
# temp quota).
#
#   sbatch -p part-$(id -gn) singularity/build.sbatch gateway env-robosuite
#   # or, inside salloc:
#   bash singularity/build.sh gateway env-robosuite
#
# A target is a name (singularity/capx-<name>.def -> capx-<name>.sif) or the path of
# a def file elsewhere in the repo (<dir>/capx-<name>.def -> capx-<name>.sif), for
# images an organizer keeps outside singularity/.
#
# Images land in $CAPX_IMAGE_DIR (default <repo>/singularity/images).
#
# Why a staging copy: Singularity's %files section resolves sources relative to
# the build CWD and has no ignore-file mechanism, so the files git tracks (submodules
# included) are copied to a clean tree first. Nothing else in the checkout reaches
# an image: not a gitignored key file, a venv, downloaded assets or a built .sif.
# Node-local scratch is used for the build's temp/cache: the sandbox holds an
# unpacked copy of every layer, which is many GB and would otherwise hammer Lustre.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_DIR="${CAPX_IMAGE_DIR:-${REPO}/singularity/images}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run inside a Slurm allocation (sbatch singularity/build.sbatch ... or salloc)." >&2
    exit 1
fi

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    # submission-agent0, not submission: the plain one runs HoldPolicy, which stands
    # still and scores 0 by construction. It is the skeleton teams fork, not something
    # an organizer bringing the queue up wants to evaluate. Build it by name.
    TARGETS=(gateway env-robosuite env-libero submission-agent0)
fi

if ! command -v singularity >/dev/null; then
    # shellcheck disable=SC1091
    source /opt/share/etc/lmod/lmod/init/bash 2>/dev/null || true
    module load "${CAPX_SINGULARITY_MODULE:-singularitypro/4.1}"
fi

# Pick the roomiest node-local scratch; Lustre is the fallback of last resort.
pick_scratch() {
    local best="" best_avail=0
    for candidate in "/scratch/${USER}" "/var/tmp/${USER}" "/tmp/${USER}"; do
        local parent; parent="$(dirname "${candidate}")"
        [[ -d "${parent}" && -w "${parent}" ]] || continue
        mkdir -p "${candidate}" 2>/dev/null || continue
        local avail; avail="$(df -k --output=avail "${candidate}" | tail -1 | tr -d ' ')"
        if (( avail > best_avail )); then best="${candidate}"; best_avail="${avail}"; fi
    done
    [[ -n "${best}" ]] || { echo "ERROR: no writable node-local scratch found" >&2; exit 1; }
    echo "${best}"
}

SCRATCH="$(pick_scratch)"
BUILD_ROOT="${SCRATCH}/capx-sif-build.${SLURM_JOB_ID}"
STAGE="${BUILD_ROOT}/stage"
export SINGULARITY_TMPDIR="${BUILD_ROOT}/tmp"
export SINGULARITY_CACHEDIR="${SCRATCH}/capx-sif-cache"
mkdir -p "${STAGE}" "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}" "${IMAGE_DIR}"
trap 'rm -rf "${BUILD_ROOT}"' EXIT

echo "[build] repo=${REPO}"
echo "[build] scratch=${SCRATCH} ($(df -h --output=avail "${SCRATCH}" | tail -1 | tr -d ' ') free)"
echo "[build] images=${IMAGE_DIR}"
echo "[build] targets=${TARGETS[*]}"

# The defs copy whole directories, and the submission image is handed to teams:
# a key file kept in the checkout (docker/api_keys.yaml, hf_token) must not ride along.
echo "[build] staging tracked files -> ${STAGE}"
git -C "${REPO}" ls-files -z --recurse-submodules \
    | rsync -a --from0 --files-from=- "${REPO}/" "${STAGE}/"

for target in "${TARGETS[@]}"; do
    if [[ "${target}" == *.def ]]; then
        def="${REPO}/${target}"
        name="$(basename "${target}" .def)"
    else
        def="${REPO}/singularity/capx-${target}.def"
        name="capx-${target}"
    fi
    [[ -f "${def}" ]] || { echo "ERROR: no def file for target '${target}' (${def})" >&2; exit 1; }
    out="${IMAGE_DIR}/${name}.sif"
    echo
    echo "[build] === ${target} -> ${out} ==="
    cp "${def}" "${STAGE}/build.def"
    ( cd "${STAGE}" && singularity build --fakeroot --force "${out}" build.def )
    ls -lh "${out}"
done

echo
echo "[build] done: ${TARGETS[*]}"
