#!/usr/bin/env bash

# Usage: bash vietnamese/run_eval_all_vi.sh <ckpt_dir_1> <ckpt_dir_2> ...
# Example: bash vietnamese/run_eval_all_vi.sh checkpoint_stage2_vi/m4_static_seed42 checkpoint_stage2_vi/m5_anneal_seed42

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <checkpoint_dir_1> [checkpoint_dir_2] ..."
    echo "Example: $0 checkpoint_stage2_vi/m4_static_seed42"
    exit 1
fi

STAGE1_CKPT="checkpoints/stage1_squad_best.pt"
EPOCHS=("001" "002" "003" "best")

# 4 GPUs mapped to 0,1,2,3 inside container
GPUS=(0 1 2 3)

# Datasets
XQUAD_VI="dataset/xquad.vi.json"
XQUAD_EN="dataset/xquad.en.json"
SQUAD2="dataset/Squad2.0/dev-v2.0.json"
MLQA_VI="dataset/MLQA/test-context-vi-question-vi.json"
MLQA_EN="dataset/MLQA/test-context-en-question-en.json"

for CKPT_DIR in "$@"; do
    echo "========================================================="
    echo "Evaluating Checkpoint Directory: $CKPT_DIR"
    echo "========================================================="

    PRED_DIR="${CKPT_DIR}/preds"
    mkdir -p "$PRED_DIR"

    for i in "${!EPOCHS[@]}"; do
        EP="${EPOCHS[$i]}"
        GPU_ID="${GPUS[$i]}"

        if [ "$EP" = "best" ]; then
            CKPT_FILE="${CKPT_DIR}/stage2_best.pt"
        else
            CKPT_FILE="${CKPT_DIR}/stage2_epoch_${EP}.pt"
        fi

        if [ ! -f "$CKPT_FILE" ]; then
            echo "[-] Checkpoint not found: $CKPT_FILE (Skipping...)"
            continue
        fi

        echo "[+] Launching evaluation for Epoch $EP on GPU $GPU_ID in background..."

        (
            export CUDA_VISIBLE_DEVICES=$GPU_ID
            LOG_FILE="${CKPT_DIR}/eval_epoch_${EP}.log"

            {
                echo "---------------------------------------------------------"
                echo " Epoch: $EP | Checkpoint: $CKPT_FILE"
                echo " GPU: $GPU_ID"
                echo "---------------------------------------------------------"

                # 1. XQuAD Vietnamese
                echo "[1] Evaluating XQuAD Vietnamese..."
                python phase4-evaluation/quick_eval.py \
                    --stage1_ckpt "$STAGE1_CKPT" \
                    --ckpt "$CKPT_FILE" \
                    --eval_file "$XQUAD_VI" \
                    --n_samples 0

                # 2. XQuAD English
                echo "[2] Evaluating XQuAD English..."
                python phase4-evaluation/quick_eval.py \
                    --stage1_ckpt "$STAGE1_CKPT" \
                    --ckpt "$CKPT_FILE" \
                    --eval_file "$XQUAD_EN" \
                    --n_samples 0

                # 3. SQuAD 2.0
                echo "[3] Evaluating SQuAD 2.0..."
                if [ -f "$SQUAD2" ]; then
                    python phase4-evaluation/quick_eval.py \
                        --stage1_ckpt "$STAGE1_CKPT" \
                        --ckpt "$CKPT_FILE" \
                        --eval_file "$SQUAD2" \
                        --n_samples 0
                else
                    echo "SQuAD 2.0 file not found at $SQUAD2, skipping."
                fi

                # 4. MLQA Vietnamese
                echo "[4] Evaluating MLQA Vietnamese..."
                PRED_VI="${PRED_DIR}/preds_vi_epoch_${EP}.json"
                python generate_mlqa_preds.py \
                    --stage1_ckpt "$STAGE1_CKPT" \
                    --ckpt "$CKPT_FILE" \
                    --eval_file "$MLQA_VI" \
                    --output_pred_file "$PRED_VI"

                python mlqa_evaluation_v1.py "$MLQA_VI" "$PRED_VI" vi

                # 5. MLQA English
                echo "[5] Evaluating MLQA English..."
                PRED_EN="${PRED_DIR}/preds_en_epoch_${EP}.json"
                python generate_mlqa_preds.py \
                    --stage1_ckpt "$STAGE1_CKPT" \
                    --ckpt "$CKPT_FILE" \
                    --eval_file "$MLQA_EN" \
                    --output_pred_file "$PRED_EN"

                python mlqa_evaluation_v1.py "$MLQA_EN" "$PRED_EN" en

                echo "Epoch $EP DONE."
            } > "$LOG_FILE" 2>&1

        ) &  # Run in background

    done

    echo "Waiting for all 4 epochs (001, 002, 003, best) to finish evaluation..."
    wait
    echo "Done with $CKPT_DIR! Results in: ${CKPT_DIR}/eval_epoch_*.log"
    echo ""
done

echo "========================================================="
echo "ALL DONE!"
echo "========================================================="
