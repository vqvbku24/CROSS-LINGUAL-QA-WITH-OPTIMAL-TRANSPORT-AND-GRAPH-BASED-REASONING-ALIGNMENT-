#!/bin/bash
# run_seed_inner.sh — Runs Stage 2 training inside the container with a specific seed
# Usage: bash run_seed_inner.sh <SEED>

set -e

SEED=$1

if [ -z "$SEED" ]; then
    echo "Error: Seed must be provided as the first argument."
    exit 1
fi

BASE=$(pwd)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p logs checkpoints

LOG_FILE="$BASE/logs/exp_stage2_seed_${SEED}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== [SEED-$SEED] Running on $(hostname) at $(date) ==="
nvidia-smi

CKPT="$BASE/checkpoints/stage1_squad_best.pt"
echo "[SEED-$SEED] Using stage1 checkpoint: $CKPT"

echo "=== [SEED-$SEED] Starting Stage 2 ==="

# We use torchrun with 4 GPUs as in run_exp_margin.sh
# Port randomization to avoid collisions if multiple seeds somehow run on the same node
PORT=$((29500 + $SEED))

torchrun --nproc_per_node=4 --master_port=$PORT train_stage2.py \
    --seed "$SEED" \
    --stage1_ckpt "$CKPT" \
    --max_epochs 8 \
    --batch_size 32 \
    --stage2_head_lr 1e-4 \
    --lambda_ot 0.5 \
    --lambda_span 1.0 \
    --lambda_margin 0.5 \
    --lambda_qa 0.3 \
    --lambda_reg 50.0 \
    --output_dir "$BASE/checkpoint_stage2_seed_${SEED}"

echo "=== [SEED-$SEED] Finished at $(date) ==="
