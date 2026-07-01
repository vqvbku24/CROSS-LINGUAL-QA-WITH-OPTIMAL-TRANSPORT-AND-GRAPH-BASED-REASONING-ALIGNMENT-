#!/bin/bash
#SBATCH --job-name=train_ot_qa
#SBATCH -p kisski-h100
#SBATCH --mem=148G              
#SBATCH -c 8
#SBATCH -G H100:1
#SBATCH --gpus-per-task=1       # Thêm cái này cho rõ ràng
#SBATCH --gpu-bind=closest      # Tối ưu băng GPU
#SBATCH -t 48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --constraint=inet

BASE=/projects/extern/kisski/kisski-imm/dir.project/CROSS-LINGUAL-QA-WITH-OPTIMAL-TRANSPORT-AND-GRAPH-BASED-REASONING-ALIGNMENT-

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$BASE/.cache/huggingface
export HF_TOKEN=$(cat $BASE/.hf_token)

# Optional: Log GPU ngay khi job bắt đầu
echo "=== Job started at $(date) ==="
nvidia-smi

source $BASE/venv/bin/activate

echo "Starting Stage 2 training..."

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