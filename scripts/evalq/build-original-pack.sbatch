#!/bin/bash
# Regenerate taskpacks/original/pack.yaml from the installed benchmarks.
#
# LIBERO's benchmark index is only importable inside the env-libero image, so the
# generator runs there against the working tree.
#
#   sbatch -p part-$(id -gn) scripts/evalq/build-original-pack.sbatch [--n-trials 3]
#SBATCH --job-name=capx-build-pack
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x.%j.log

set -euo pipefail

IMAGE="${CAPX_LIBERO_IMAGE:-${SLURM_SUBMIT_DIR}/singularity/images/capx-env-libero.sif}"

if ! command -v singularity >/dev/null; then
    # The system profile sets up `module` whatever environment the job got, without
    # anyone's login files. It does not run under -e or -u.
    set +eu
    # shellcheck disable=SC1091
    source /etc/profile >/dev/null 2>&1
    set -eu
    module load "${CAPX_SINGULARITY_MODULE:-singularitypro/4.1}"
fi

singularity exec \
    --bind "${SLURM_SUBMIT_DIR}:/repo" \
    --env PYTHONPATH=/repo \
    --pwd /repo \
    "${IMAGE}" \
    python /repo/scripts/evalq/build_original_pack.py "$@"
