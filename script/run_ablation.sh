#!/bin/bash

set -e

# Project directory
BASE=$(pwd)

echo "Running on $(hostname)"
echo "Project: $BASE"

python --version
which python

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Hugging Face token (optional)
if [ -f "$BASE/.hf_token" ]; then
    export HF_TOKEN=$(cat "$BASE/.hf_token")
fi

mkdir -p logs checkpoints

echo "========== GPU =========="
nvidia-smi

# ── Download stage1 checkpoint nếu chưa có ──────────────────────────
CKPT="$BASE/checkpoints/stage1_squad_best.pt"
if [ ! -f "$CKPT" ]; then
    echo "========== Downloading stage1 checkpoint from HF Hub =========="
    python -c "
import os, shutil
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='vinhvo1205/Sinkhorn_2_stages',
    filename='checkpoints/stage1_squad_best.pt',
    token=os.environ.get('HF_TOKEN')
)
os.makedirs('checkpoints', exist_ok=True)
shutil.copy(path, 'checkpoints/stage1_squad_best.pt')
print('Download OK ->', 'checkpoints/stage1_squad_best.pt')
"
else
    echo "Stage1 checkpoint found: $CKPT"
fi

echo "========== Stage 2 (Margin) =========="

python train_stage2.py \
    --stage1_ckpt "$BASE/checkpoints/stage1_squad_best.pt" \
    --max_epochs 8 \
    --batch_size 32 \
    --lambda_ot 0.5 \
    --lambda_span 1.0 \
    --lambda_margin 0.5 \
    --lambda_qa 0.3 \
    --lambda_reg 50.0 \
    --hf_repo_id "vinhvo1205/final_test" \
    --output_dir "$BASE/checkpoint_stage2"

echo "========== Stage 2 (Anneal Margin) =========="

python train_stage2.py \
    --stage1_ckpt "$BASE/checkpoints/stage1_squad_best.pt" \
    --max_epochs 8 \
    --batch_size 32 \
    --lambda_ot 0.5 \
    --lambda_span 1.0 \
    --anneal_margin \
    --hf_repo_id "vinhvo1205/final_test" \
    --output_dir "$BASE/checkpoint_stage2_anneal_margin"

echo "========== Finished at $(date) =========="
nvidia-smi