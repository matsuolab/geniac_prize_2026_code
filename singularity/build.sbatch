#!/bin/bash
# Build CapX Singularity images on a compute node.
#
#   sbatch -p part-$(id -gn) singularity/build.sbatch gateway env-robosuite
#
# No GPU is requested: the builds are pip/apt work, and holding the eval GPU for
# them would starve the eval worker. #SBATCH lines cannot expand $(...), so the
# partition is passed on the command line.
#SBATCH --job-name=capx-sif-build
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.log

set -euo pipefail
# sbatch copies the submitted script into the job spool dir, so ${BASH_SOURCE[0]}
# points at the copy — resolve the repo from the submit directory instead.
bash "${SLURM_SUBMIT_DIR}/singularity/build.sh" "$@"
