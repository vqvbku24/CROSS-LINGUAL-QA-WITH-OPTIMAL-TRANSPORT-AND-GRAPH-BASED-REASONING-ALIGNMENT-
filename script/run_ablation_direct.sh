#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# run_ablation_direct.sh — Chạy trực tiếp ablation jobs trên máy chủ (Không SLURM)
#
# Cách dùng:
#   chmod +x run_ablation_direct.sh
#   ./run_ablation_direct.sh 
#
# Lưu ý: Các job dưới đây đang được set chạy NỐI TIẾP (tuần tự) để tránh 
# hết bộ nhớ GPU nếu máy bạn chỉ có 1 GPU. Nếu máy có nhiều GPU, bạn 
# có thể thêm `CUDA_VISIBLE_DEVICES=0`, `CUDA_VISIBLE_DEVICES=1`, v.v... 
# và thêm `&` ở cuối mỗi lệnh python để chạy song song.
# ═══════════════════════════════════════════════════════════════

# Lấy thư mục hiện tại làm BASE
BASE="$(pwd)"
COMMON_ARGS="--epochs 10 --batch_size 32 --K 128 --root_dir $BASE"
HF_REPO="vinhvo1205/CrossLingual-OT-QA"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$BASE/.cache/huggingface"

# Đọc token nếu có
if [ -f "$BASE/.hf_token" ]; then
    export HF_TOKEN=$(cat "$BASE/.hf_token")
fi

# Kích hoạt môi trường ảo
if [ -f "$BASE/venv/bin/activate" ]; then
    source "$BASE/venv/bin/activate"
fi

echo "═══════════════════════════════════════════════════"
echo "  Bắt đầu chạy Ablation Study Trực Tiếp"
echo "═══════════════════════════════════════════════════"

# ── Run 1: Full Model (Main Proposed) ─────────────────────────
# echo "=== Run 1: FULL MODEL ==="
# python3 -u "$BASE/main.py" --mode train $COMMON_ARGS \
#     --output_dir "$BASE/checkpoints" \
#     --hf_repo_id $HF_REPO 2>&1 | tee "$BASE/run1_full.log"

# ── Run 2: No Consistency (λ_cons = 0) ────────────────────────
echo "=== Run 2: NO CONSISTENCY ==="
python3 -u "$BASE/main.py" --mode train $COMMON_ARGS \
    --lambda_cons 0.0 \
    --output_dir "$BASE/checkpoints_no_cons" \
    --hf_repo_id $HF_REPO 2>&1 | tee "$BASE/run2_no_cons.log"

# ── Run 3: No Span Projection (λ_span = 0) ───────────────────
echo "=== Run 3: NO SPAN PROJECTION ==="
python3 -u "$BASE/main.py" --mode train $COMMON_ARGS \
    --lambda_span 0.0 --lambda_cons 0.0 \
    --output_dir "$BASE/checkpoints_no_span" \
    --hf_repo_id $HF_REPO 2>&1 | tee "$BASE/run3_no_span.log"

# ── Run 4: Baseline XLM-R (No OT, No GAT) ────────────────────
echo "=== Run 4: BASELINE XLM-R ==="
python3 -u "$BASE/train_baseline.py" --mode train \
    --epochs 3 --batch_size 32 \
    --output_dir "$BASE/checkpoints_baseline" \
    --hf_repo_id $HF_REPO \
    --root_dir "$BASE" 2>&1 | tee "$BASE/run4_baseline.log"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Tất cả 4 runs đã hoàn thành!"
echo "═══════════════════════════════════════════════════"
