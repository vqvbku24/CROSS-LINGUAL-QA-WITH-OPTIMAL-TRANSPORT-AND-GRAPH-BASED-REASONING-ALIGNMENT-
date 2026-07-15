#!/bin/bash
#SBATCH --job-name=ot_stage2_seeds
#SBATCH --partition=129-partition
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out

set -e

# Setup Enroot and HuggingFace environment paths (as in run_parallel.sh)
export ENROOT_CACHE_PATH=/lustre/user129002/.cache/enroot
export ENROOT_DATA_PATH=/lustre/user129002/.local/share/enroot
export HF_HOME=/lustre/user129002/.cache/huggingface
export HF_TOKEN=$(cat /lustre/user129002/.hf_secrets/token 2>/dev/null)

CODE_DIR=/lustre/user129002/Research_infra/OT-vinh/code
HF_CACHE=/lustre/user129002/.cache/huggingface

# Define array of seeds
SEEDS=(42 43 44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Job Array ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Selected Seed: $SEED"
echo "Submitting enroot container for seed $SEED..."

# Start enroot container and execute inner bash script
enroot start \
    --mount ${CODE_DIR}:/workspace/code \
    --mount ${HF_CACHE}:/root/.cache/huggingface \
    --env HF_TOKEN \
    comer-pytorch-vinh \
    bash -lc "cd /workspace/code && bash run_seed_inner.sh $SEED"
