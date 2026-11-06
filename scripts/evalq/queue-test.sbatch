#!/bin/bash
# Queue test rig: one Slurm job running the HF watcher and the worker side by side, from
# the checkout it was submitted from. The leaderboard publisher is its own, CPU-only job
# (leaderboard-publisher.sbatch) so it can be restarted without interrupting a run.
#
# For tests only (leaderboard_smoke_test.py): the watcher holds the intake token, so a
# real round runs it apart from the workers, as docs/operations.md §5 describes.
#
#   CAPX_EVALQ_PYTHON=... CAPX_QT_CONFIG=... sbatch -p part-$(id -gn) scripts/evalq/queue-test.sbatch
#
# CAPX_EVALQ_PYTHON  interpreter with the queue's dependencies (the venv, not the system python3)
# CAPX_QT_CONFIG     queue config for the watcher and worker
#SBATCH --job-name=auto-eval-pipeline-watcher-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --open-mode=append
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
if ! command -v singularity >/dev/null; then
    # The system profile sets up `module` whatever environment the job got, without
    # anyone's login files. It does not run under -e or -u.
    set +u
    # shellcheck disable=SC1091
    source /etc/profile >/dev/null 2>&1
    set -u
    module load "${CAPX_SINGULARITY_MODULE:-singularitypro/4.1}"
fi
PY="${CAPX_EVALQ_PYTHON:?set CAPX_EVALQ_PYTHON}"
CFG="${CAPX_QT_CONFIG:?set CAPX_QT_CONFIG}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
echo "[queue-test] node=$(hostname) job=${SLURM_JOB_ID} tree=$PWD start=$(date)"
# Not a pipe: $! would be sed's pid, and the trap would leave the watcher running.
$PY -m capx.cli.main evalq watch --config "$CFG" > >(sed -u 's/^/[watch] /') 2>&1 &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT
$PY -m capx.cli.main evalq worker --config "$CFG" 2>&1 | sed -u 's/^/[worker] /'
