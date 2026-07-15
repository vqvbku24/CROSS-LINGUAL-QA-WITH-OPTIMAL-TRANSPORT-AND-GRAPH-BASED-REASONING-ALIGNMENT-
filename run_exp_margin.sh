#!/bin/bash
# run_exp_margin.sh — Thí nghiệm 1: Stage 2 + Margin Loss
# Chạy bên TRONG container (enroot start ... bash -lc "bash run_exp_margin.sh")

set -e

BASE=$(pwd)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p logs checkpoints

LOG_FILE="$BASE/logs/exp_margin_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== [EXP-MARGIN] Running on $(hostname) at $(date) ==="
nvidia-smi

CKPT="$BASE/checkpoints/stage1_squad_best.pt"
echo "[EXP-MARGIN] Using stage1 checkpoint: $CKPT"

echo "=== [EXP-MARGIN] Starting Stage 2 (Margin) ==="

# DDP: torchrun launches 4 processes (1 per GPU)
# batch_size=32 per GPU → effective batch = 32 × 4 = 128
torchrun --nproc_per_node=4 --master_port=29500 train_stage2.py \
    --stage1_ckpt "$BASE/checkpoints/stage1_squad_best.pt" \
    --max_epochs 8 \
    --batch_size 32 \
    --stage2_head_lr 1e-4 \
    --lambda_ot 0.5 \
    --lambda_span 1.0 \
    --lambda_margin 0.5 \
    --lambda_qa 0.3 \
    --lambda_reg 50.0 \
    --output_dir "$BASE/checkpoint_stage2"

echo "=== [EXP-MARGIN] Finished at $(date) ==="


