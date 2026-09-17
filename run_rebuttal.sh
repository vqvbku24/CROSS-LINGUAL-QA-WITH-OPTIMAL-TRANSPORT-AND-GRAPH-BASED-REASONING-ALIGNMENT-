#!/usr/bin/env bash
# ==============================================================================
# run_rebuttal.sh
#
# Master Shell Entrypoint for ACL 2025 Rebuttal Experiments
# Works seamlessly on Linux servers and Google Colab environments.
#
# Quick Usage:
#   bash run_rebuttal.sh --tier1       # Run Tier 1 (M3-AR, M3-HI, XQuAD Table)
#   bash run_rebuttal.sh --tier2       # Run Tier 2 (mmBERT pipeline)
#   bash run_rebuttal.sh --all         # Run both Tier 1 & Tier 2
#   bash run_rebuttal.sh --eval_only   # Print formatted Rebuttal tables
#   bash run_rebuttal.sh --help        # Show all options
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Load HF token if present
if [ -f "$DIR/.hf_token" ]; then
    export HF_TOKEN=$(cat "$DIR/.hf_token" | tr -d '\r\n')
fi

# Print environment summary
echo "======================================================================"
echo " ACL 2025 REBUTTAL MASTER RUNNER (BASH WRAPPER)"
echo " Working Directory: $DIR"
echo " Date: $(date)"
echo " Hostname: $(hostname 2>/dev/null || echo 'colab/local')"
if command -v nvidia-smi &> /dev/null; then
    echo " GPU Detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi
echo "======================================================================"

python run_rebuttal.py "$@"
