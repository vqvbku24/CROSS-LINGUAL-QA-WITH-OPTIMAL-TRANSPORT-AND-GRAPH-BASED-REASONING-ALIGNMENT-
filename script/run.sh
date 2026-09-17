#!/bin/bash
#SBATCH --job-name=ot_contrastive
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out

set -e

echo "Running on $(hostname)"
echo "Working dir: $PWD"

# Đảm bảo working directory đúng với nơi submit job
cd $SLURM_SUBMIT_DIR
BASE=$PWD

# Load module và activate conda environment một cách an toàn
module load python/anaconda3
eval "$(conda shell.bash hook)"
conda activate qa

# Kiểm tra môi trường
python --version
which python

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Nếu có file lưu token HuggingFace
if [ -f "$BASE/.hf_token" ]; then
    export HF_TOKEN=$(cat $BASE/.hf_token)
fi

# Tạo thư mục logs nếu chưa có
mkdir -p logs

echo "=== Job started at $(date) ==="
echo "GPU info:"
nvidia-smi || true

echo "Starting Stage 2 Training"

# Chọn một trong các cấu hình loss dưới đây bằng cách uncomment/comment các trọng số:
#
# 1) L = L_QA + L_OT + L_Span + L_Margin + L_Reg  (OT + Margin + Span) -> mặc định
# 2) L = L_QA + L_Span + L_Margin + L_Reg         (Chỉ Margin, xem Margin có thay thế OT được không): đặt --lambda_ot 0.0
# 3) L = L_QA + L_OT + L_Span + L_Reg             (OT cũ không có Margin): đặt --lambda_margin 0.0

python3 -u $BASE/train_stage2.py \
    --stage1_ckpt $BASE/checkpoint/best.pt \
    --max_epochs 10 \
    --batch_size 32 \
    --lambda_ot 0.5 \
    --lambda_span 1.0 \
    --lambda_margin 0.5 \
    --lambda_qa 0.3 \
    --lambda_reg 50.0 \
    --hf_repo_id "vinhvo1205/CrossLingual-OT-QA" \
    --output_dir $BASE/checkpoint_stage2

python3 -u $BASE/train_stage2.py \
    --stage1_ckpt $BASE/checkpoint/best.pt \
    --max_epochs 10 \
    --batch_size 32 \
    --lambda_ot 0.5 \
    --lambda_span 1.0 \
    --anneal_margin \
    --output_dir $BASE/checkpoint_stage2_anneal_margin

echo "=== Training finished at $(date) ==="
nvidia-smi