#!/usr/bin/env bash
# arabic/run_ar.sh
# Run all 4 configs for Table 2 (Arabic branch).
#
# Configs:
#   Zero-shot     — stage1 only, no training, just generate preds
#   Vanilla KD    — M1: lambda_kd=1.0, lambda_ot=0, lambda_span=0, lambda_margin=0
#   Static OT     — M2: lambda_ot>0, lambda_span=0, lambda_margin=0, lambda_kd=0
#   Ours          — M5: full coordinated + anneal_margin
#
# Usage:
#   bash arabic/run_ar.sh <stage1_ckpt> [extra_args...]
#
# Example:
#   bash arabic/run_ar.sh checkpoints/stage1_squad_best.pt
#   bash arabic/run_ar.sh checkpoints/stage1_squad_best.pt --batch_size 16

set -e

STAGE1_CKPT="${1:-checkpoints/stage1_squad_best.pt}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MLQA_AR="${ROOT_DIR}/dataset/MLQA/test-context-ar-question-ar.json"
XQUAD_AR_LOCAL="${ROOT_DIR}/dataset/xquad.ar.json"
SHIFT; EXTRA_ARGS="$@"

echo "=========================================="
echo " Arabic Table 2 — All 4 configs"
echo " Stage1 ckpt: ${STAGE1_CKPT}"
echo "=========================================="

# ── 0. Zero-shot (no Stage 2 training) ──────────────────────────
echo ""
echo "[1/4] Zero-shot (Stage 1 only)"
python "${ROOT_DIR}/arabic/generate_preds_ar.py" \
    --ckpt "${STAGE1_CKPT}" \
    --eval_file "${MLQA_AR}" \
    --output_pred_file "${ROOT_DIR}/results_ar/zeroshot_mlqa_ar_preds.json" \
    --zero_shot
echo "  → predictions saved to results_ar/zeroshot_mlqa_ar_preds.json"

if [ -f "${XQUAD_AR_LOCAL}" ]; then
    python "${ROOT_DIR}/arabic/generate_preds_ar.py" \
        --ckpt "${STAGE1_CKPT}" \
        --eval_file "${XQUAD_AR_LOCAL}" \
        --output_pred_file "${ROOT_DIR}/results_ar/zeroshot_xquad_ar_preds.json" \
        --zero_shot
    echo "  → predictions saved to results_ar/zeroshot_xquad_ar_preds.json"
fi

# ── 1. Vanilla KD (M1) ──────────────────────────────────────────
echo ""
echo "[2/4] Vanilla KD — M1 (lambda_kd=1.0, no OT/span/margin)"
python "${ROOT_DIR}/arabic/train_stage2_ar.py" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --output_dir "${ROOT_DIR}/checkpoint_stage2_ar/m1_vanilla_kd" \
    --lambda_ot 0.0 \
    --lambda_span 0.0 \
    --lambda_margin 0.0 \
    --lambda_reg 50.0 \
    --lambda_kd 1.0 \
    --kd_temperature 2.0 \
    ${EXTRA_ARGS}

python "${ROOT_DIR}/arabic/generate_preds_ar.py" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --ckpt "${ROOT_DIR}/checkpoint_stage2_ar/m1_vanilla_kd/stage2_ar_best.pt" \
    --eval_file "${MLQA_AR}" \
    --output_pred_file "${ROOT_DIR}/results_ar/m1_kd_mlqa_ar_preds.json"
echo "  → M1 predictions saved"

# ── 2. Static OT / M2 (ablation: OT only) ───────────────────────
echo ""
echo "[3/4] Static OT — M2 (OT only, no span/margin/kd)"
python "${ROOT_DIR}/arabic/train_stage2_ar.py" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --output_dir "${ROOT_DIR}/checkpoint_stage2_ar/m2_static_ot" \
    --lambda_ot 0.5 \
    --lambda_span 0.0 \
    --lambda_margin 0.0 \
    --lambda_reg 50.0 \
    --lambda_kd 0.0 \
    ${EXTRA_ARGS}

python "${ROOT_DIR}/arabic/generate_preds_ar.py" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --ckpt "${ROOT_DIR}/checkpoint_stage2_ar/m2_static_ot/stage2_ar_best.pt" \
    --eval_file "${MLQA_AR}" \
    --output_pred_file "${ROOT_DIR}/results_ar/m2_ot_mlqa_ar_preds.json"
echo "  → M2 predictions saved"

# ── 3. Ours / M5 (full coordinated + dynamic margin) ────────────
echo ""
echo "[4/4] Ours — M5 (full coordinated, dynamic margin)"
python "${ROOT_DIR}/arabic/train_stage2_ar.py" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --output_dir "${ROOT_DIR}/checkpoint_stage2_ar/m5_ours" \
    --lambda_ot 0.5 \
    --lambda_span 1.0 \
    --lambda_margin 1.0 \
    --lambda_reg 50.0 \
    --lambda_kd 0.0 \
    --anneal_margin \
    ${EXTRA_ARGS}

python "${ROOT_DIR}/arabic/generate_preds_ar.py" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --ckpt "${ROOT_DIR}/checkpoint_stage2_ar/m5_ours/stage2_ar_best.pt" \
    --eval_file "${MLQA_AR}" \
    --output_pred_file "${ROOT_DIR}/results_ar/m5_ours_mlqa_ar_preds.json"
echo "  → M5 predictions saved"

echo ""
echo "=========================================="
echo " All configs done. Predictions in results_ar/"
echo " Use mlqa_evaluation_v1.py to compute official scores:"
echo "   python mlqa_evaluation_v1.py <preds.json> ${MLQA_AR} ar"
echo "=========================================="
