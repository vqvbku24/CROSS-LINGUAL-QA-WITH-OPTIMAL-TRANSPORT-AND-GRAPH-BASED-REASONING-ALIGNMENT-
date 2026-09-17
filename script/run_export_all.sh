#!/bin/bash
# Script để chạy export dữ liệu cho 3 ngôn ngữ (VI, AR, HI) tuần tự trên 1 GPU (T4).
# Tự động ghi đè output vào các thư mục paper_tools/export_multi_*.
#
# LƯU Ý: Bạn hãy sửa lại ĐƯỜNG DẪN CHECKPOINT ở dưới cho đúng với checkpoint
# của từng ngôn ngữ mà bạn đã train xong nhé!

# 1. Đường dẫn tới checkpoint gốc (Stage 1) - chung cho cả 3
STAGE1_CKPT="checkpoints/stage1_squad_best.pt"

# 2. Đường dẫn tới checkpoint Stage 2 (LoRA) của từng ngôn ngữ
# (Sửa lại tên file nếu của bạn khác)
STAGE2_VI_CKPT="checkpoints/stage2_vi_best.pt"
STAGE2_AR_CKPT="checkpoints/stage2_ar_best.pt"
STAGE2_HI_CKPT="checkpoints/stage2_hi_best.pt"

# Số mẫu cần export (ví dụ 50 mẫu đầu tiên để vẽ Panel a)
SAMPLE_INDICES="0-49"
NUM_GPUS=1

echo "============================================================"
echo "BẮT ĐẦU EXPORT DỮ LIỆU CHO 3 NGÔN NGỮ (VI, AR, HI)"
echo "============================================================"

# --- 1. Tiếng Việt (VI) ---
echo -e "\n[1/3] Đang xử lý tiếng Việt (VI)..."
if [ -f "$STAGE2_VI_CKPT" ]; then
    python paper_tools/export_many_samples.py \
        --checkpoint "$STAGE2_VI_CKPT" \
        --stage1_checkpoint "$STAGE1_CKPT" \
        --dataset "dataset/xquad.vi.json" \
        --output_root "paper_tools/export_multi_vi" \
        --sample_indices "$SAMPLE_INDICES" \
        --stage 2 \
        --num_gpus "$NUM_GPUS"
else
    echo "  [Bỏ qua] Không tìm thấy checkpoint: $STAGE2_VI_CKPT"
fi

# --- 2. Tiếng Ả Rập (AR) ---
echo -e "\n[2/3] Đang xử lý tiếng Ả Rập (AR)..."
if [ -f "$STAGE2_AR_CKPT" ]; then
    python paper_tools/export_many_samples.py \
        --checkpoint "$STAGE2_AR_CKPT" \
        --stage1_checkpoint "$STAGE1_CKPT" \
        --dataset "dataset/xquad.ar.json" \
        --output_root "paper_tools/export_multi_ar" \
        --sample_indices "$SAMPLE_INDICES" \
        --stage 2 \
        --num_gpus "$NUM_GPUS"
else
    echo "  [Bỏ qua] Không tìm thấy checkpoint: $STAGE2_AR_CKPT"
fi

# --- 3. Tiếng Hindi (HI) ---
echo -e "\n[3/3] Đang xử lý tiếng Hindi (HI)..."
if [ -f "$STAGE2_HI_CKPT" ]; then
    python paper_tools/export_many_samples.py \
        --checkpoint "$STAGE2_HI_CKPT" \
        --stage1_checkpoint "$STAGE1_CKPT" \
        --dataset "dataset/xquad.hi.json" \
        --output_root "paper_tools/export_multi_hi" \
        --sample_indices "$SAMPLE_INDICES" \
        --stage 2 \
        --num_gpus "$NUM_GPUS"
else
    echo "  [Bỏ qua] Không tìm thấy checkpoint: $STAGE2_HI_CKPT"
fi

echo -e "\n============================================================"
echo "Hoàn tất! Bây giờ bạn có thể chạy lại script aggregate_alignment_stats.py"
echo "để ra kết quả Panel (a) chính xác."
