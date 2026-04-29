#!/bin/bash
#SBATCH --job-name=train_ot_qa
#SBATCH -p kisski-h100
#SBATCH --mem=148G              
#SBATCH -c 16
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

echo "Starting training with batch_size=32, K=96..."

python3 -u $BASE/main.py \
    --mode train \
    --epochs 10 \
    --batch_size 32 \
    --K 96 \
    --hf_repo_id "vinhvo1205/CrossLingual-OT-QA" \
    --root_dir $BASE

echo "=== Training finished at $(date) ==="
nvidia-smi