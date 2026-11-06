#!/bin/bash
# The leaderboard publisher on its own: it only reads the shared queue directory and
# writes to the Hub, so it needs a CPU, not the GPU the worker holds. Restarting it
# (after a change to the page's data) leaves the watcher and worker alone.
#
#   CAPX_EVALQ_PYTHON=... CAPX_QT_CONFIG=... sbatch -p part-$(id -gn) \
#       scripts/evalq/leaderboard-publisher.sbatch
#SBATCH --job-name=auto-eval-leaderboard-publisher
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --open-mode=append
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
PY="${CAPX_EVALQ_PYTHON:?set CAPX_EVALQ_PYTHON}"
CFG="${CAPX_QT_CONFIG:?set CAPX_QT_CONFIG}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
echo "[publisher] node=$(hostname) job=${SLURM_JOB_ID} tree=$PWD start=$(date)"
exec $PY -m capx.cli.main evalq publish --config "$CFG"
