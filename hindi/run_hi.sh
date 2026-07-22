#!/usr/bin/env bash
# hindi/run_hi.sh
# Run all 4 configs for Table 2 (Hindi branch).
#
# Configs:
#   Zero-shot     — stage1 only, no training, just generate preds
#   Vanilla KD    — M1: lambda_kd=1.0, lambda_ot=0, lambda_span=0, lambda_margin=0
#   Static OT     — M2: lambda_ot>0, lambda_span=0, lambda_margin=0, lambda_kd=0
#   Ours          — M5: full coordinated + anneal_margin
#
# Usage:
#   bash hindi/run_hi.sh <stage1_ckpt> [extra_args...]
#
# Example:
#   bash hindi/run_hi.sh checkpoints/stage1_squad_best.pt
#   bash hindi/run_hi.sh checkpoints/stage1_squad_best.pt --batch_size 16

set -e

BASE=$(pwd)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p logs results_hi checkpoint_stage2_hi

LOG_FILE="$BASE/logs/run_hi_$(date +%Y%m%d_%H%M%S).log"
export TORCHELASTIC_ERROR_FILE="$BASE/logs/torchelastic_error_hi.json"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== [RUN-HI] Running on $(hostname) at $(date) ==="
nvidia-smi

STAGE1_CKPT="${1:-checkpoints/stage1_squad_best.pt}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MLQA_HI="${ROOT_DIR}/dataset/MLQA/test-context-hi-question-hi.json"
XQUAD_HI_LOCAL="${ROOT_DIR}/dataset/xquad.hi.json"
shift || true
EXTRA_ARGS="$@"

echo "=========================================="
echo " Hindi Table 2 — All 4 configs"
echo " Stage1 ckpt: ${STAGE1_CKPT}"
echo "=========================================="

# ── 0. Zero-shot (no Stage 2 training) ──────────────────────────
echo ""
echo "[1/4] Zero-shot (Stage 1 only)"
python "${ROOT_DIR}/hindi/generate_preds_hi.py" \
    --ckpt "${STAGE1_CKPT}" \
    --eval_file "${MLQA_HI}" \
    --output_pred_file "${ROOT_DIR}/results_hi/zeroshot_mlqa_hi_preds.json" \
    --zero_shot
echo "  → predictions saved to results_hi/zeroshot_mlqa_hi_preds.json"

if [ -f "${XQUAD_HI_LOCAL}" ]; then
    python "${ROOT_DIR}/hindi/generate_preds_hi.py" \
        --ckpt "${STAGE1_CKPT}" \
        --eval_file "${XQUAD_HI_LOCAL}" \
        --output_pred_file "${ROOT_DIR}/results_hi/zeroshot_xquad_hi_preds.json" \
        --zero_shot
    echo "  → predictions saved to results_hi/zeroshot_xquad_hi_preds.json"
fi

# ── 1. Vanilla KD (M1) ──────────────────────────────────────────
echo ""
echo "[2/4] Vanilla KD — M1 (lambda_kd=1.0, no OT/span/margin)"
torchrun --nproc_per_node=4 --master_port=29502 "${ROOT_DIR}/hindi/train_stage2_hi.py" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --output_dir "${ROOT_DIR}/checkpoint_stage2_hi/table2_m1_vanilla_kd" \
    --lambda_ot 0.0 \
    --lambda_span 0.0 \
    --lambda_margin 0.0 \
    --lambda_reg 50.0 \
    --lambda_kd 1.0 \
    --kd_temperature 2.0 \
    --batch_size 32 \
    ${EXTRA_ARGS}

python "${ROOT_DIR}/hindi/generate_preds_hi.py" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --ckpt "${ROOT_DIR}/checkpoint_stage2_hi/table2_m1_vanilla_kd/stage2_hi_best.pt" \
    --eval_file "${MLQA_HI}" \
    --output_pred_file "${ROOT_DIR}/results_hi/m1_kd_mlqa_hi_preds.json"
echo "  → M1 predictions saved"

# ── 2. Static OT / M2 (ablation: OT only) ───────────────────────
echo ""
echo "[3/4] Static OT — M2 (OT only, no span/margin/kd)"
torchrun --nproc_per_node=4 --master_port=29502 "${ROOT_DIR}/hindi/train_stage2_hi.py" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --output_dir "${ROOT_DIR}/checkpoint_stage2_hi/table2_m2_static_ot" \
    --lambda_ot 0.5 \
    --lambda_span 0.0 \
    --lambda_margin 0.0 \
    --lambda_reg 50.0 \
    --lambda_kd 0.0 \
    --batch_size 32 \
    ${EXTRA_ARGS}

python "${ROOT_DIR}/hindi/generate_preds_hi.py" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --ckpt "${ROOT_DIR}/checkpoint_stage2_hi/table2_m2_static_ot/stage2_hi_best.pt" \
    --eval_file "${MLQA_HI}" \
    --output_pred_file "${ROOT_DIR}/results_hi/m2_ot_mlqa_hi_preds.json"
echo "  → M2 predictions saved"

# ── 3. Ours / M5 (full coordinated + dynamic margin) ────────────
echo ""
echo "[4/4] Ours — M5 (full coordinated, dynamic margin)"
torchrun --nproc_per_node=4 --master_port=29502 "${ROOT_DIR}/hindi/train_stage2_hi.py" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --output_dir "${ROOT_DIR}/checkpoint_stage2_hi/table2_m5_ours" \
    --lambda_ot 0.5 \
    --lambda_span 1.0 \
    --lambda_margin 1.0 \
    --lambda_reg 50.0 \
    --lambda_kd 0.0 \
    --anneal_margin \
    --batch_size 32 \
    ${EXTRA_ARGS}

python "${ROOT_DIR}/hindi/generate_preds_hi.py" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --ckpt "${ROOT_DIR}/checkpoint_stage2_hi/table2_m5_ours/stage2_hi_best.pt" \
    --eval_file "${MLQA_HI}" \
    --output_pred_file "${ROOT_DIR}/results_hi/m5_ours_mlqa_hi_preds.json"
echo "  → M5 predictions saved"

echo ""
echo "=========================================="
echo " All configs done. Predictions in results_hi/"
echo " Use mlqa_evaluation_v1.py to compute official scores:"
echo "   python mlqa_evaluation_v1.py <preds.json> ${MLQA_HI} hi"
echo "=========================================="
