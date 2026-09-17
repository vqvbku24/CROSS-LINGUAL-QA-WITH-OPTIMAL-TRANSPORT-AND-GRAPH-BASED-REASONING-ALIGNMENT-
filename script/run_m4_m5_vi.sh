#!/usr/bin/env bash
# Script to run M4 (Static Margin) and M5 (Anneal Margin) for Vietnamese
# 3 seeds each.

set -e

STAGE1_CKPT="checkpoints/stage1_squad_best.pt"
SEEDS=(42 43 44)

mkdir -p checkpoint_stage2_vi

echo "=========================================="
echo " Running M4 (Static Margin) and M5 (Anneal Margin) - 3 seeds each"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
    # ── M4: Static Margin ───────────────────────────────────────────
    echo ""
    echo "▶️ [Seed ${SEED}] Training M4 (Static Margin) ..."
    torchrun --nproc_per_node=2 --master_port=$((29500 + SEED)) train_stage2.py \
        --stage1_ckpt "${STAGE1_CKPT}" \
        --output_dir "checkpoint_stage2_vi/m4_static_seed${SEED}" \
        --lambda_ot 0.5 \
        --lambda_span 1.0 \
        --lambda_margin 1.0 \
        --lambda_reg 50.0 \
        --lambda_kd 0.0 \
        --batch_size 32 \
        --seed ${SEED}

    # ── M5: Anneal Margin ───────────────────────────────────────────
    echo ""
    echo "▶️ [Seed ${SEED}] Training M5 (Anneal Margin) ..."
    torchrun --nproc_per_node=2 --master_port=$((29600 + SEED)) train_stage2.py \
        --stage1_ckpt "${STAGE1_CKPT}" \
        --output_dir "checkpoint_stage2_vi/m5_anneal_seed${SEED}" \
        --lambda_ot 0.5 \
        --lambda_span 1.0 \
        --lambda_margin 1.0 \
        --lambda_reg 50.0 \
        --lambda_kd 0.0 \
        --batch_size 32 \
        --anneal_margin \
        --seed ${SEED}
done

echo "=========================================="
echo " All M4 and M5 runs for Vietnamese completed successfully!"
echo "=========================================="
