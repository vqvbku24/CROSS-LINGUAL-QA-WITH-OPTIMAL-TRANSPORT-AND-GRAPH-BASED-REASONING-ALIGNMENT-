#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# run_ablation.sh — Submit tất cả ablation jobs lên SLURM
#
# Cách dùng:
#   chmod +x run_ablation.sh
#   ./run_ablation.sh          # Submit 4 jobs (sẽ chạy song song nếu đủ GPU)
#
# Kết quả:
#   Run 1: Full Model        → checkpoints/         (Main Proposed)
#   Run 2: No Consistency    → checkpoints_no_cons/ 
#   Run 3: No Span Proj      → checkpoints_no_span/
#   Run 4: Baseline XLM-R    → checkpoints_baseline/
# ═══════════════════════════════════════════════════════════════

BASE=/projects/extern/kisski/kisski-imm/dir.project/CROSS-LINGUAL-QA-WITH-OPTIMAL-TRANSPORT-AND-GRAPH-BASED-REASONING-ALIGNMENT-
COMMON_ARGS="--epochs 10 --batch_size 32 --K 96 --root_dir $BASE"
HF_REPO="vinhvo1205/CrossLingual-OT-QA"

echo "═══════════════════════════════════════════════════"
echo "  Submitting Ablation Study Jobs"
echo "═══════════════════════════════════════════════════"

# ── Run 1: Full Model (Main Proposed) ─────────────────────────
JOB1=$(sbatch --parsable \
    --job-name=OT_full \
    -p kisski-h100 --mem=148G -c 8 -G H100:1 \
    --gpus-per-task=1 --gpu-bind=closest \
    -t 48:00:00 \
    --output=$BASE/slurm-full-%j.out \
    --error=$BASE/slurm-full-%j.err \
    --constraint=inet \
    --wrap="
        export PYTHONUNBUFFERED=1
        export TOKENIZERS_PARALLELISM=false
        export HF_HOME=$BASE/.cache/huggingface
        export HF_TOKEN=\$(cat $BASE/.hf_token)
        source $BASE/venv/bin/activate
        echo '=== Run 1: FULL MODEL ==='
        python3 -u $BASE/main.py --mode train $COMMON_ARGS \
            --output_dir $BASE/checkpoints \
            --hf_repo_id $HF_REPO
    ")
echo "✅ Run 1 (Full Model)       → Job $JOB1"

# ── Run 2: No Consistency (λ_cons = 0) ────────────────────────
JOB2=$(sbatch --parsable \
    --job-name=OT_no_cons \
    -p kisski-h100 --mem=148G -c 8 -G H100:1 \
    --gpus-per-task=1 --gpu-bind=closest \
    -t 48:00:00 \
    --output=$BASE/slurm-no_cons-%j.out \
    --error=$BASE/slurm-no_cons-%j.err \
    --constraint=inet \
    --wrap="
        export PYTHONUNBUFFERED=1
        export TOKENIZERS_PARALLELISM=false
        export HF_HOME=$BASE/.cache/huggingface
        export HF_TOKEN=\$(cat $BASE/.hf_token)
        source $BASE/venv/bin/activate
        echo '=== Run 2: NO CONSISTENCY ==='
        python3 -u $BASE/main.py --mode train $COMMON_ARGS \
            --lambda_cons 0.0 \
            --output_dir $BASE/checkpoints_no_cons \
            --hf_repo_id $HF_REPO
    ")
echo "✅ Run 2 (No Consistency)   → Job $JOB2"

# ── Run 3: No Span Projection (λ_span = 0) ───────────────────
JOB3=$(sbatch --parsable \
    --job-name=OT_no_span \
    -p kisski-h100 --mem=148G -c 8 -G H100:1 \
    --gpus-per-task=1 --gpu-bind=closest \
    -t 48:00:00 \
    --output=$BASE/slurm-no_span-%j.out \
    --error=$BASE/slurm-no_span-%j.err \
    --constraint=inet \
    --wrap="
        export PYTHONUNBUFFERED=1
        export TOKENIZERS_PARALLELISM=false
        export HF_HOME=$BASE/.cache/huggingface
        export HF_TOKEN=\$(cat $BASE/.hf_token)
        source $BASE/venv/bin/activate
        echo '=== Run 3: NO SPAN PROJECTION ==='
        python3 -u $BASE/main.py --mode train $COMMON_ARGS \
            --lambda_span 0.0 --lambda_cons 0.0 \
            --output_dir $BASE/checkpoints_no_span \
            --hf_repo_id $HF_REPO
    ")
echo "✅ Run 3 (No Span+Cons)     → Job $JOB3"

# ── Run 4: Baseline XLM-R (No OT, No GAT) ────────────────────
JOB4=$(sbatch --parsable \
    --job-name=baseline_xlmr \
    -p kisski-h100 --mem=148G -c 8 -G H100:1 \
    --gpus-per-task=1 --gpu-bind=closest \
    -t 24:00:00 \
    --output=$BASE/slurm-baseline-%j.out \
    --error=$BASE/slurm-baseline-%j.err \
    --constraint=inet \
    --wrap="
        export PYTHONUNBUFFERED=1
        export TOKENIZERS_PARALLELISM=false
        export HF_HOME=$BASE/.cache/huggingface
        export HF_TOKEN=\$(cat $BASE/.hf_token)
        source $BASE/venv/bin/activate
        echo '=== Run 4: BASELINE XLM-R ==='
        python3 -u $BASE/train_baseline.py --mode train \
            --epochs 3 --batch_size 32 \
            --output_dir $BASE/checkpoints_baseline \
            --hf_repo_id $HF_REPO \
            --root_dir $BASE
    ")
echo "✅ Run 4 (Baseline XLM-R)   → Job $JOB4"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Tất cả 4 jobs đã submit!"
echo "  Theo dõi: squeue -u \$USER"
echo "  Log: tail -f $BASE/slurm-*.out"
echo "═══════════════════════════════════════════════════"
