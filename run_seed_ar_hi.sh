#!/usr/bin/env bash
# Script to run M4 (Static Margin) for Arabic and Hindi
# 3 seeds each.

set -e

STAGE1_CKPT="checkpoints/stage1_squad_best.pt"
SEEDS=(42 43 44)

mkdir -p checkpoint_stage2_ar
mkdir -p checkpoint_stage2_hi

echo "=========================================="
echo " Running (Static Margin) - 3 seeds for Arabic and Hindi"
echo "=========================================="

# ── Arabic ───────────────────────────────────────────
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "▶️ [Seed ${SEED}] Training (Static Margin) for Arabic ..."
    torchrun --nproc_per_node=2 --master_port=$((29700 + SEED)) arabic/train_stage2_ar.py \
        --stage1_ckpt "${STAGE1_CKPT}" \
        --output_dir "checkpoint_stage2_ar/m4_static_seed${SEED}" \
        --lambda_ot 0.5 \
        --lambda_span 1.0 \
        --lambda_margin 1.0 \
        --lambda_reg 50.0 \
        --lambda_kd 0.0 \
        --batch_size 32 \
        --seed ${SEED}
done

# ── Hindi ───────────────────────────────────────────
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "▶️ [Seed ${SEED}] Training (Static Margin) for Hindi ..."
    torchrun --nproc_per_node=2 --master_port=$((29800 + SEED)) hindi/train_stage2_hi.py \
        --stage1_ckpt "${STAGE1_CKPT}" \
        --output_dir "checkpoint_stage2_hi/m4_static_seed${SEED}" \
        --lambda_ot 0.5 \
        --lambda_span 1.0 \
        --lambda_margin 1.0 \
        --lambda_reg 50.0 \
        --lambda_kd 0.0 \
        --batch_size 32 \
        --seed ${SEED}
done

echo "=========================================="
echo " All static margin runs for Arabic and Hindi completed successfully!"
echo "=========================================="
