#!/usr/bin/env bash
# arabic/run_ablation_ar.sh
# Minimal ablation for Arabic: M2 and M5 only (per spec.md).
#
# Purpose: confirm "OT alone can hurt; coordination fixes it"
# Reuses checkpoints from run_ar.sh (no re-training needed if already run).
#
# Usage:
#   bash arabic/run_ablation_ar.sh <stage1_ckpt> [--force_retrain]
#
# Flags:
#   --force_retrain   Re-train even if checkpoint exists

set -e

STAGE1_CKPT="${1:-checkpoints/stage1_squad_best.pt}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MLQA_AR="${ROOT_DIR}/dataset/MLQA/test-context-ar-question-ar.json"
FORCE_RETRAIN=0

shift
for arg in "$@"; do
    if [ "$arg" = "--force_retrain" ]; then
        FORCE_RETRAIN=1
    fi
done

mkdir -p "${ROOT_DIR}/results_ar"

echo "=========================================="
echo " Arabic Ablation: M2 (OT only) vs M5 (Ours)"
echo " Stage1 ckpt: ${STAGE1_CKPT}"
echo "=========================================="

# ── M2: OT only ─────────────────────────────────────────────────
M2_CKPT="${ROOT_DIR}/checkpoint_stage2_ar/m2_static_ot/stage2_ar_best.pt"

if [ ! -f "${M2_CKPT}" ] || [ "${FORCE_RETRAIN}" = "1" ]; then
    echo ""
    echo "[1/2] Training M2 (OT only) ..."
    python "${ROOT_DIR}/arabic/train_stage2_ar.py" \
        --stage1_ckpt "${STAGE1_CKPT}" \
        --output_dir "${ROOT_DIR}/checkpoint_stage2_ar/m2_static_ot" \
        --lambda_ot 0.5 \
        --lambda_span 0.0 \
        --lambda_margin 0.0 \
        --lambda_reg 50.0 \
        --lambda_kd 0.0
else
    echo ""
    echo "[1/2] M2 checkpoint found — skipping training (use --force_retrain to override)"
fi

python "${ROOT_DIR}/arabic/generate_preds_ar.py" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --ckpt "${M2_CKPT}" \
    --eval_file "${MLQA_AR}" \
    --output_pred_file "${ROOT_DIR}/results_ar/ablation_m2_ot_mlqa_ar_preds.json"

# ── M5: Ours ────────────────────────────────────────────────────
M5_CKPT="${ROOT_DIR}/checkpoint_stage2_ar/m5_ours/stage2_ar_best.pt"

if [ ! -f "${M5_CKPT}" ] || [ "${FORCE_RETRAIN}" = "1" ]; then
    echo ""
    echo "[2/2] Training M5 (Ours, full coordinated) ..."
    python "${ROOT_DIR}/arabic/train_stage2_ar.py" \
        --stage1_ckpt "${STAGE1_CKPT}" \
        --output_dir "${ROOT_DIR}/checkpoint_stage2_ar/m5_ours" \
        --lambda_ot 0.5 \
        --lambda_span 1.0 \
        --lambda_margin 1.0 \
        --lambda_reg 50.0 \
        --lambda_kd 0.0 \
        --anneal_margin
else
    echo ""
    echo "[2/2] M5 checkpoint found — skipping training (use --force_retrain to override)"
fi

python "${ROOT_DIR}/arabic/generate_preds_ar.py" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --ckpt "${M5_CKPT}" \
    --eval_file "${MLQA_AR}" \
    --output_pred_file "${ROOT_DIR}/results_ar/ablation_m5_ours_mlqa_ar_preds.json"

echo ""
echo "=========================================="
echo " Ablation done. Score with:"
echo "   python mlqa_evaluation_v1.py results_ar/ablation_m2_ot_mlqa_ar_preds.json ${MLQA_AR} ar"
echo "   python mlqa_evaluation_v1.py results_ar/ablation_m5_ours_mlqa_ar_preds.json ${MLQA_AR} ar"
echo "=========================================="
