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

mkdir -p logs

echo "========== GPU =========="
nvidia-smi

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

echo "========== Finished =========="
nvidia-smi