#!/bin/bash
# Run the parts of the test suite whose imports only exist inside a built image.
#
# tests/submission and tests/integrations pull torch, robosuite and the perception
# stack, so they cannot run from a plain checkout on this cluster. The submission
# image already carries all of it: bind the working tree over it and let the image
# supply the interpreter and the dependencies.
#
#   sbatch -p part-$(id -gn) scripts/evalq/test-in-image.sbatch tests/submission
#SBATCH --job-name=capx-test-in-image
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x.%j.log

set -euo pipefail

IMAGE="${CAPX_TEST_IMAGE:-${SLURM_SUBMIT_DIR}/singularity/images/capx-submission-agent0.sif}"
TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(tests/submission)
fi

if ! command -v singularity >/dev/null; then
    # The system profile sets up `module` whatever environment the job got, without
    # anyone's login files. It does not run under -e or -u.
    set +eu
    # shellcheck disable=SC1091
    source /etc/profile >/dev/null 2>&1
    set -eu
    module load "${CAPX_SINGULARITY_MODULE:-singularitypro/4.1}"
fi

# A submission image has no business carrying test tooling, so pytest is installed
# beside it into node-local scratch and put on the path for this job only.
PYTEST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/capx-pytest.XXXXXX")"
trap 'rm -rf "${PYTEST_DIR}"' EXIT
singularity exec --bind "${PYTEST_DIR}:/pytest" "${IMAGE}" \
    python -m pip install --no-cache-dir --quiet --target /pytest pytest

# The working tree wins over the copy baked into the image, which is the point:
# this checks the code as it stands, not as it was when the image was built.
singularity exec --nv \
    --bind "${SLURM_SUBMIT_DIR}:/repo" \
    --bind "${PYTEST_DIR}:/pytest" \
    --env PYTHONPATH=/repo:/pytest \
    --pwd /repo \
    "${IMAGE}" \
    python -m pytest "${TARGETS[@]}" -q -p no:cacheprovider
