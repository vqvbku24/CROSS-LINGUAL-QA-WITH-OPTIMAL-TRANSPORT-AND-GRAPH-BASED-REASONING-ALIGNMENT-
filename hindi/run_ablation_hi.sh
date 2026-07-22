#!/usr/bin/env bash
# hindi/run_ablation_hi.sh
# Minimal ablation for Hindi: M2 and M5 only (per spec.md).
#
# Purpose: confirm "OT alone can hurt; coordination fixes it"
# Reuses checkpoints from run_hi.sh (no re-training needed if already run).
#
# Usage:
#   bash hindi/run_ablation_hi.sh <stage1_ckpt> [--force_retrain]
#
# Flags:
#   --force_retrain   Re-train even if checkpoint exists

set -e

BASE=$(pwd)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p logs results_hi checkpoint_stage2_hi

LOG_FILE="$BASE/logs/run_ablation_hi_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== [RUN-ABLATION-HI] Running on $(hostname) at $(date) ==="
nvidia-smi

STAGE1_CKPT="${1:-checkpoints/stage1_squad_best.pt}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MLQA_HI="${ROOT_DIR}/dataset/MLQA/test-context-hi-question-hi.json"
FORCE_RETRAIN=0

shift || true
for arg in "$@"; do
    if [ "$arg" = "--force_retrain" ]; then
        FORCE_RETRAIN=1
    fi
done

mkdir -p "${ROOT_DIR}/results_hi"

echo "=========================================="
echo " Hindi Ablation: M2 (OT only) vs M5 (Ours)"
echo " Stage1 ckpt: ${STAGE1_CKPT}"
echo "=========================================="

# ── M2: OT only ─────────────────────────────────────────────────
M2_CKPT="${ROOT_DIR}/checkpoint_stage2_hi/m2_static_ot/stage2_hi_best.pt"

if [ ! -f "${M2_CKPT}" ] || [ "${FORCE_RETRAIN}" = "1" ]; then
    echo ""
    echo "[1/2] Training M2 (OT only) ..."
    torchrun --nproc_per_node=4 --master_port=29502 "${ROOT_DIR}/hindi/train_stage2_hi.py" \
        --stage1_ckpt "${STAGE1_CKPT}" \
        --output_dir "${ROOT_DIR}/checkpoint_stage2_hi/m2_static_ot" \
        --lambda_ot 0.5 \
        --lambda_span 0.0 \
        --lambda_margin 0.0 \
        --lambda_reg 50.0 \
        --lambda_kd 0.0 \
        --batch_size 32
else
    echo ""
    echo "[1/2] M2 checkpoint found — skipping training (use --force_retrain to override)"
fi

python "${ROOT_DIR}/hindi/generate_preds_hi.py" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --ckpt "${M2_CKPT}" \
    --eval_file "${MLQA_HI}" \
    --output_pred_file "${ROOT_DIR}/results_hi/ablation_m2_ot_mlqa_hi_preds.json"

# ── M5: Ours ────────────────────────────────────────────────────
M5_CKPT="${ROOT_DIR}/checkpoint_stage2_hi/m5_ours/stage2_hi_best.pt"

if [ ! -f "${M5_CKPT}" ] || [ "${FORCE_RETRAIN}" = "1" ]; then
    echo ""
    echo "[2/2] Training M5 (Ours, full coordinated) ..."
    torchrun --nproc_per_node=4 --master_port=29502 "${ROOT_DIR}/hindi/train_stage2_hi.py" \
        --stage1_ckpt "${STAGE1_CKPT}" \
        --output_dir "${ROOT_DIR}/checkpoint_stage2_hi/m5_ours" \
        --lambda_ot 0.5 \
        --lambda_span 1.0 \
        --lambda_margin 1.0 \
        --lambda_reg 50.0 \
        --lambda_kd 0.0 \
        --anneal_margin \
        --batch_size 32
else
    echo ""
    echo "[2/2] M5 checkpoint found — skipping training (use --force_retrain to override)"
fi

python "${ROOT_DIR}/hindi/generate_preds_hi.py" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --ckpt "${M5_CKPT}" \
    --eval_file "${MLQA_HI}" \
    --output_pred_file "${ROOT_DIR}/results_hi/ablation_m5_ours_mlqa_hi_preds.json"

echo ""
echo "=========================================="
echo " Ablation done. Score with:"
echo "   python mlqa_evaluation_v1.py results_hi/ablation_m2_ot_mlqa_hi_preds.json ${MLQA_HI} hi"
echo "   python mlqa_evaluation_v1.py results_hi/ablation_m5_ours_mlqa_hi_preds.json ${MLQA_HI} hi"
echo "=========================================="
