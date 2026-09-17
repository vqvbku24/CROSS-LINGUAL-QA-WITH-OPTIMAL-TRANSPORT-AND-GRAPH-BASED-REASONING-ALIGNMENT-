#!/usr/bin/env bash
# ==============================================================================
# run_m3.sh
#
# FULL END-TO-END PIPELINE FOR TIER 1: M3 ABLATION (ARABIC & HINDI)
# 1. Tự động kiểm tra & tải Stage 1 checkpoint và train_hindi.json từ Hugging Face
# 2. Tự động nhận diện GPU & chạy DDP an toàn bằng torchrun
# 3. Huấn luyện M3 Arabic (tự động upload checkpoint mỗi epoch lên Hugging Face)
# 4. Huấn luyện M3 Hindi (tự động upload checkpoint mỗi epoch lên Hugging Face)
# 5. Đánh giá toàn diện trên XQuAD & MLQA (chọn Best khách quan + Epoch chẩn đoán)
# 6. Tổng hợp báo cáo và upload lên Hugging Face Hub (results/)
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Load token từ .env hoặc .hf_token
if [ -f "$DIR/.env" ]; then
    set -a
    source "$DIR/.env" 2>/dev/null || true
    set +a
fi
if [ -f "$DIR/.hf_token" ]; then
    export HF_TOKEN=$(cat "$DIR/.hf_token" | tr -d '\r\n')
fi

# Cấu hình mặc định
BATCH_SIZE=32
MAX_EPOCHS=6
SEED=42
HF_REPO="${HF_REPO_ID:-vinhvo1205/Sinkhorn_2_stages}"
EVAL_ONLY=0
SKIP_MLQA=0
FORCE_TRAIN=0

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --hf_repo_id)
            HF_REPO="$2"
            shift 2
            ;;
        --eval_only)
            EVAL_ONLY=1
            shift
            ;;
        --skip_mlqa)
            SKIP_MLQA=1
            shift
            ;;
        --force_train)
            FORCE_TRAIN=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            shift
            ;;
    esac
done

echo "======================================================================"
echo "  PIPELINE M3 ABLATION (ARABIC & HINDI) WITH TORCHRUN DDP"
echo "  Batch Size:   $BATCH_SIZE"
echo "  Max Epochs:   $MAX_EPOCHS"
echo "  Seed:         $SEED"
echo "  HF Repo:      $HF_REPO"
echo "  Eval Only:    $EVAL_ONLY"
echo "  Date:         $(date)"
echo "======================================================================"

# ───────────────────────────────────────────────────────────────────
# BƯỚC 1: Tải prerequisites từ Hugging Face
# ───────────────────────────────────────────────────────────────────
echo ""
echo "[BƯỚC 1/4] Kiểm tra & tải các file tiền đề từ Hugging Face Hub..."
python download_prerequisites.py

STAGE1_CKPT="checkpoints/stage1_squad_best.pt"
if [ ! -f "$STAGE1_CKPT" ]; then
    echo "Không tìm thấy Stage 1 checkpoint tại $STAGE1_CKPT"
    exit 1
fi

# ───────────────────────────────────────────────────────────────────
# BƯỚC 2: Nhận diện GPU & Khởi tạo torchrun runner
# ───────────────────────────────────────────────────────────────────
echo ""
echo "[BƯỚC 2/4] Nhận diện cấu hình phần cứng GPU cho torchrun DDP..."
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo 1)

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "  Không phát hiện GPU, chạy trên CPU (nproc=1)..."
    NUM_GPUS=1
else
    echo "  Phát hiện $NUM_GPUS GPU khả dụng -> Sử dụng torchrun với --nproc_per_node=$NUM_GPUS"
fi

# ───────────────────────────────────────────────────────────────────
# BƯỚC 3: Huấn luyện M3 Arabic & Hindi
# ───────────────────────────────────────────────────────────────────
AR_OUT_DIR="checkpoint_stage2_ar/m3_ot_span"
HI_OUT_DIR="checkpoint_stage2_hi/m3_ot_span"
AR_BEST_CKPT="$AR_OUT_DIR/stage2_ar_best.pt"
HI_BEST_CKPT="$HI_OUT_DIR/stage2_hi_best.pt"

if [ "$EVAL_ONLY" -eq 0 ]; then
    echo ""
    echo "[BƯỚC 3/4] Huấn luyện M3 (OT=0.5, Span=1.0, Margin=0.0)..."

    # --- 3a. Arabic M3 ---
    if [ "$FORCE_TRAIN" -eq 0 ] && [ -f "$AR_BEST_CKPT" ]; then
        echo "  Đã tìm thấy checkpoint M3 Arabic: $AR_BEST_CKPT (Bỏ qua training)"
    else
        echo ""
        echo "  [3a] Bắt đầu huấn luyện M3 Arabic (torchrun port 29700)..."
        torchrun --nproc_per_node="$NUM_GPUS" --master_port=29700 \
            arabic/train_stage2_ar.py \
            --stage1_ckpt "$STAGE1_CKPT" \
            --output_dir "$AR_OUT_DIR" \
            --lambda_ot 0.5 \
            --lambda_span 1.0 \
            --lambda_margin 0.0 \
            --lambda_reg 50.0 \
            --lambda_kd 0.0 \
            --batch_size "$BATCH_SIZE" \
            --max_epochs "$MAX_EPOCHS" \
            --seed "$SEED" \
            --hf_repo_id "$HF_REPO"
        echo "  Hoàn tất huấn luyện M3 Arabic!"
    fi

    # --- 3b. Hindi M3 ---
    if [ "$FORCE_TRAIN" -eq 0 ] && [ -f "$HI_BEST_CKPT" ]; then
        echo "  Đã tìm thấy checkpoint M3 Hindi: $HI_BEST_CKPT (Bỏ qua training)"
    else
        echo ""
        echo "  [3b] Bắt đầu huấn luyện M3 Hindi (torchrun port 29800)..."
        torchrun --nproc_per_node="$NUM_GPUS" --master_port=29800 \
            hindi/train_stage2_hi.py \
            --stage1_ckpt "$STAGE1_CKPT" \
            --output_dir "$HI_OUT_DIR" \
            --lambda_ot 0.5 \
            --lambda_span 1.0 \
            --lambda_margin 0.0 \
            --lambda_reg 50.0 \
            --lambda_kd 0.0 \
            --batch_size "$BATCH_SIZE" \
            --max_epochs "$MAX_EPOCHS" \
            --seed "$SEED" \
            --hf_repo_id "$HF_REPO"
        echo "  Hoàn tất huấn luyện M3 Hindi!"
    fi
else
    echo ""
    echo "[BƯỚC 3/4] Chế độ --eval_only được bật: Bỏ qua bước huấn luyện."
fi

# ───────────────────────────────────────────────────────────────────
# BƯỚC 4: Đánh giá XQuAD & MLQA, Tổng Hợp Báo Cáo & Upload Lên HF
# ───────────────────────────────────────────────────────────────────
echo ""
echo "[BƯỚC 4/4] Đánh giá toàn diện trên XQuAD & MLQA, tổng hợp và upload..."

EVAL_CMD="python eval_and_aggregate_m3.py --stage1_ckpt $STAGE1_CKPT --hf_repo_id $HF_REPO"
if [ "$SKIP_MLQA" -eq 1 ]; then
    EVAL_CMD="$EVAL_CMD --skip_mlqa"
fi

$EVAL_CMD

echo ""
echo "======================================================================"
echo "  PIPELINE HOÀN TẤT THÀNH CÔNG!"
echo "  File kết quả Markdown: results_m3_rebuttal.md"
echo "  File kết quả JSON:     results_m3_rebuttal.json"
echo "  Đã tự động upload lên Hugging Face: ${HF_REPO} (thư mục results/)"
echo "======================================================================"
