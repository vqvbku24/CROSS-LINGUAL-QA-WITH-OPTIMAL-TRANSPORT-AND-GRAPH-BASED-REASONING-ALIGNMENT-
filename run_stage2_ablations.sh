#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# run_stage2_ablations.sh — Chạy trực tiếp các ablation studies cho Stage 2
# ═══════════════════════════════════════════════════════════════

BASE="$(pwd)"
STAGE1_CKPT="checkpoint/best.pt"
HF_REPO="vinhvo1205/Sinkhorn_2_stages"
COMMON_ARGS="--stage1_ckpt $STAGE1_CKPT --batch_size 32 --max_epochs 10 --hf_repo_id $HF_REPO"


export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Kích hoạt môi trường ảo nếu có
if [ -f "$BASE/venv/bin/activate" ]; then
    source "$BASE/venv/bin/activate"
fi

echo "═══════════════════════════════════════════════════"
echo "  Bắt đầu chạy Stage 2 Ablation Studies"
echo "═══════════════════════════════════════════════════"

# ── 1. Full Stage 2 Curriculum (OT -> OT+Cons -> OT+Cons+Span) ──
echo "=== Run 1: FULL STAGE 2 CURRICULUM ==="
python3 -u "$BASE/train_stage2.py" $COMMON_ARGS \
    --output_dir "$BASE/checkpoints/run_full" 2>&1 | tee "$BASE/run_stage2_full.log"

# ── 2. Pure OT Backbone Alignment (Freeze QA Head, No Span/Cons in early epochs) ──
# Lưu ý: Khi freeze_qa_head là True, QA head không học gì, chỉ học LoRA backbone
echo "=== Run 2: PURE OT ALIGNMENT (FREEZE QA HEAD) ==="
python3 -u "$BASE/train_stage2.py" $COMMON_ARGS \
    --freeze_qa_head \
    --output_dir "$BASE/checkpoints/run_ot_only" 2>&1 | tee "$BASE/run_stage2_ot_only.log"

# ── 3. No Consistency Loss (λ_cons = 0.0) ──
echo "=== Run 3: NO CONSISTENCY LOSS ==="
python3 -u "$BASE/train_stage2.py" $COMMON_ARGS \
    --lambda_cons 0.0 \
    --output_dir "$BASE/checkpoints/run_no_cons" 2>&1 | tee "$BASE/run_stage2_no_cons.log"

# ── 4. No Span Loss (λ_span = 0.0) ──
echo "=== Run 4: NO SPAN LOSS ==="
python3 -u "$BASE/train_stage2.py" $COMMON_ARGS \
    --lambda_span 0.0 \
    --output_dir "$BASE/checkpoints/run_no_span" 2>&1 | tee "$BASE/run_stage2_no_span.log"

echo "═══════════════════════════════════════════════════"
echo "  Tất cả các lượt chạy Stage 2 hoàn thành!"
echo "═══════════════════════════════════════════════════"
