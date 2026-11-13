#!/bin/bash
# Start all API servers and launch LIBERO-PRO evaluation
# Usage: bash scripts/start_servers_and_eval.sh [--workers N] [--suites SUITE1,SUITE2,...]
set -e

cd "$(git rev-parse --show-toplevel)"
source scripts/sockets.sh
mkdir -p logs

NUM_WORKERS=${1:-4}

echo "=== Cleaning up existing batch processes ==="
pkill -f run_libero_batch 2>/dev/null || true
sleep 2

echo "=== Checking API servers ==="
all_up=true
for sock in sam3.sock graspnet.sock pyroki.sock; do
    if sock_up "$sock"; then
        echo "  $sock: UP"
    else
        echo "  $sock: DOWN - starting..."
        all_up=false
    fi
done
if sock_ok model.sock /health; then
    echo "  model.sock (NV LLM): UP"
else
    echo "  model.sock (NV LLM): DOWN - starting..."
    all_up=false
fi

if [ "$all_up" = false ]; then
    echo ""
    echo "=== Starting missing servers ==="
    source .venv/bin/activate

    # SAM3 server (GPU)
    if ! sock_up sam3.sock; then
        echo "Starting SAM3 on sam3.sock..."
        nohup python -m capx.serving.launch_sam3_server --device cuda --socket sam3.sock > logs/sam3.log 2>&1 &
    fi

    # GraspNet server (GPU)
    if ! sock_up graspnet.sock; then
        echo "Starting GraspNet on graspnet.sock..."
        nohup python -m capx.serving.launch_contact_graspnet_server --socket graspnet.sock > logs/graspnet.log 2>&1 &
    fi

    # PyRoKi server (CPU)
    if ! sock_up pyroki.sock; then
        echo "Starting PyRoKi on pyroki.sock..."
        nohup python -m capx.serving.launch_pyroki_server --socket pyroki.sock --robot panda_description --target-link panda_hand > logs/pyroki.log 2>&1 &
    fi

    # NV inference server
    if ! sock_ok model.sock /health; then
        echo "Starting NV inference on model.sock..."
        nohup python -m capx.serving.openrouter_server --key-file .openrouterkey > logs/nv_server.log 2>&1 &
    fi

    echo "Waiting 30s for servers..."
    sleep 30

    # Re-check
    echo "=== Server Health Re-check ==="
    for sock in sam3.sock graspnet.sock pyroki.sock; do
        echo "  $sock: $(sock_up "$sock" && echo "UP" || echo "DOWN")"
    done
    echo "  model.sock: $(sock_ok model.sock /health && echo "UP" || echo "DOWN")"
fi

echo ""
echo "=== Launching LIBERO-PRO Evaluation ==="
echo "6 suites × 10 tasks × 50 trials = 3000 total trials"
echo "Workers: $NUM_WORKERS"
echo "Ensemble: Gemini-3-Pro only (3 temps) — fast mode"
echo "Output: ./outputs/libero_batch_run/"
echo ""

source .venv-libero/bin/activate
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
nohup python -m capx.envs.scripts.run_libero_batch \
    --args.num-workers $NUM_WORKERS \
    --args.output-dir ./outputs/libero_batch_run \
    > logs/libero_batch.log 2>&1 &
BATCH_PID=$!
echo "Batch PID: $BATCH_PID"
echo ""
echo "Monitor with:"
echo "  tail -f logs/libero_batch.log"
echo "  watch -n 30 'find outputs/libero_batch_run -name \"trial_*\" -type d | wc -l'"
echo "  watch -n 30 'find outputs/libero_batch_run -name \"*taskcompleted_1*\" -type d | wc -l'"
