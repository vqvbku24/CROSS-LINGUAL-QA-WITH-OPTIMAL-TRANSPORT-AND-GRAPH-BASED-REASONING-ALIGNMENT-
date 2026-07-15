#!/bin/bash
set -e

export ENROOT_CACHE_PATH=/lustre/user129002/.cache/enroot
export ENROOT_DATA_PATH=/lustre/user129002/.local/share/enroot
export HF_HOME=/lustre/user129002/.cache/huggingface
export HF_TOKEN=$(cat /lustre/user129002/.hf_secrets/token 2>/dev/null)

CODE_DIR=/lustre/user129002/Research_infra/OT-vinh/code
HF_CACHE=/lustre/user129002/.cache/huggingface

mkdir -p ${CODE_DIR}/logs

echo "=== Submitting 2 parallel ablation jobs (4 GPU each) at $(date) ==="

srun -p 129-partition --gres=gpu:4 --cpus-per-task=16 --mem=64G --job-name=exp_margin \
  enroot start \
    --mount ${CODE_DIR}:/workspace/code \
    --mount ${HF_CACHE}:/root/.cache/huggingface \
    --env HF_TOKEN \
    comer-pytorch-vinh \
    bash -lc "cd /workspace/code && bash run_exp_margin.sh" &
JOB1_PID=$!

srun -p 129-partition --gres=gpu:4 --cpus-per-task=16 --mem=64G --job-name=exp_anneal \
  enroot start \
    --mount ${CODE_DIR}:/workspace/code \
    --mount ${HF_CACHE}:/root/.cache/huggingface \
    --env HF_TOKEN \
    comer-pytorch-vinh \
    bash -lc "cd /workspace/code && bash run_exp_anneal.sh" &
JOB2_PID=$!

echo "Both jobs submitted. Monitor: squeue -u user129002"

wait $JOB1_PID && echo "[JOB-1] EXP-MARGIN DONE" || echo "[JOB-1] EXP-MARGIN FAILED"
wait $JOB2_PID && echo "[JOB-2] EXP-ANNEAL DONE" || echo "[JOB-2] EXP-ANNEAL FAILED"

echo "=== All jobs finished at $(date) ==="