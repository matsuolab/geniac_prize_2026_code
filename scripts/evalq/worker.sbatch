#!/bin/bash
# CapX eval worker — the long-lived GPU job that drains the submission queue.
#
# Normally submitted by `capx evalq supervise` (which resubmits it whenever it
# ends), but it also works standalone; every argument is passed to the worker:
#
#   env -i PATH="$PATH" HOME="$HOME" LANG="$LANG" CAPX_EVALQ_PYTHON="$CAPX_EVALQ_PYTHON" \
#       CAPX_SINGULARITY_MODULE="${CAPX_SINGULARITY_MODULE:-}" \
#     sbatch -p part-$(id -gn) --gres=gpu:1 --export=ALL scripts/evalq/worker.sbatch [--max-jobs 1]
#
# sbatch gets only those variables (capx.evalqueue.slurm.WORKER_ENV): the worker
# needs no token, and the shell it is submitted from may hold the intake or
# leaderboard one. Not --export=<names>: that also loads the login environment.
#
# CAPX_EVALQ_PYTHON must point at an interpreter with pydantic/pyyaml/tyro (the
# worker itself is light — everything heavy lives inside the Singularity images).
#
# Resource flags come from the queue config's `slurm:` section via
# capx.evalqueue.slurm, so there are deliberately no #SBATCH resource lines here —
# two places to change one number is one too many.
# %x is the job name, which the supervisor sets per worker slot — without it
# every slot writes to a name that does not say which slot it was.
#SBATCH --output=logs/%x.%j.log
#SBATCH --open-mode=append

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

if ! command -v singularity >/dev/null; then
    # The system profile sets up `module` whatever environment the job got, without
    # anyone's login files. It does not run under -e or -u.
    set +eu
    # shellcheck disable=SC1091
    source /etc/profile >/dev/null 2>&1
    set -eu
    module load "${CAPX_SINGULARITY_MODULE:-singularitypro/4.1}"
fi

echo "[worker] node=$(hostname) job=${SLURM_JOB_ID} gpus=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[worker] started at $(TZ=Asia/Tokyo date +%Y/%m/%d\ %H:%M:%S)"

PYTHON="${CAPX_EVALQ_PYTHON:-python3}"

"${PYTHON}" -m capx.cli.main evalq worker "$@"
