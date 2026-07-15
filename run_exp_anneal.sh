#!/bin/bash
# run_exp_anneal.sh — Thí nghiệm 2: Stage 2 + Anneal Margin
# Chạy bên TRONG container (enroot start ... bash -lc "bash run_exp_anneal.sh")

set -e

BASE=$(pwd)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p logs checkpoints

LOG_FILE="$BASE/logs/exp_anneal_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== [EXP-ANNEAL] Running on $(hostname) at $(date) ==="
nvidia-smi

CKPT="$BASE/checkpoints/stage1_squad_best.pt"
echo "[EXP-ANNEAL] Using stage1 checkpoint: $CKPT"

echo "=== [EXP-ANNEAL] Starting Stage 2 (Anneal Margin) ==="

# DDP: torchrun launches 4 processes (1 per GPU)
# batch_size=32 per GPU → effective batch = 32 × 4 = 128
torchrun --nproc_per_node=4 --master_port=29501 train_stage2.py \
    --stage1_ckpt "$BASE/checkpoints/stage1_squad_best.pt" \
    --max_epochs 8 \
    --batch_size 32 \
    --stage2_head_lr 1e-4 \
    --lambda_ot 0.5 \
    --lambda_span 1.0 \
    --anneal_margin \
    --output_dir "$BASE/checkpoint_stage2_anneal_margin"

echo "=== [EXP-ANNEAL] Finished at $(date) ==="


