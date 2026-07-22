#!/usr/bin/env bash

# Usage: bash arabic/run_eval_all_ar.sh <ckpt_dir_1> <ckpt_dir_2> ...
# Example: bash arabic/run_eval_all_ar.sh checkpoint_stage2_ar/m1_vanilla_kd checkpoint_stage2_ar/m2_static_ot

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <checkpoint_dir_1> [checkpoint_dir_2] ..."
    echo "Example: $0 checkpoint_stage2_ar/m1_vanilla_kd"
    exit 1
fi

STAGE1_CKPT="checkpoints/stage1_squad_best.pt"
EPOCHS=("001" "002" "003" "best")

# Do bạn đang truyền 4 GPUs (VD: CUDA_VISIBLE_DEVICES=4,5,6,7 từ host), 
# thì bên trong container chúng sẽ được map lại thành 0, 1, 2, 3.
# Ta sẽ chạy song song 4 epoch, mỗi epoch chiếm 1 GPU để tối ưu thời gian.
GPUS=(0 1 2 3)

# Datasets
XQUAD_AR="dataset/xquad.ar.json"
XQUAD_EN="dataset/xquad.en.json"
SQUAD2="dataset/Squad2.0/dev-v2.0.json"
MLQA_AR="dataset/MLQA/test-context-ar-question-ar.json"
MLQA_EN="dataset/MLQA/test-context-en-question-en.json"

for CKPT_DIR in "$@"; do
    echo "========================================================="
    echo "Evaluating Checkpoint Directory: $CKPT_DIR"
    echo "========================================================="
    
    # Thư mục lưu file predictions cho MLQA
    PRED_DIR="${CKPT_DIR}/preds"
    mkdir -p "$PRED_DIR"
    
    for i in "${!EPOCHS[@]}"; do
        EP="${EPOCHS[$i]}"
        GPU_ID="${GPUS[$i]}"
        
        if [ "$EP" = "best" ]; then
            CKPT_FILE="${CKPT_DIR}/stage2_ar_best.pt"
        else
            CKPT_FILE="${CKPT_DIR}/stage2_ar_epoch_${EP}.pt"
        fi
        
        if [ ! -f "$CKPT_FILE" ]; then
            echo "[-] Checkpoint not found: $CKPT_FILE (Skipping...)"
            continue
        fi
        
        echo "[+] Launching evaluation for Epoch $EP on GPU $GPU_ID in background..."
        
        (
            # Gán GPU riêng cho tiến trình này
            export CUDA_VISIBLE_DEVICES=$GPU_ID
            
            # Lưu log ra file để không bị in đè lên nhau trên terminal
            LOG_FILE="${CKPT_DIR}/eval_epoch_${EP}.log"
            
            {
                echo "---------------------------------------------------------"
                echo " Epoch: $EP | Checkpoint: $CKPT_FILE"
                echo " GPU: $GPU_ID"
                echo "---------------------------------------------------------"
                
                # 1. XQuAD Arabic
                echo "[1] Evaluating XQuAD Arabic..."
                python arabic/phase4_evaluation/quick_eval_ar.py \
                    --stage1_ckpt "$STAGE1_CKPT" \
                    --ckpt "$CKPT_FILE" \
                    --eval_file "$XQUAD_AR" \
                    --n_samples 0
                    
                # 2. XQuAD English
                echo "[2] Evaluating XQuAD English..."
                python arabic/phase4_evaluation/quick_eval_ar.py \
                    --stage1_ckpt "$STAGE1_CKPT" \
                    --ckpt "$CKPT_FILE" \
                    --eval_file "$XQUAD_EN" \
                    --n_samples 0
                    
                # 3. SQuAD 2.0
                echo "[3] Evaluating SQuAD 2.0..."
                if [ -f "$SQUAD2" ]; then
                    python arabic/phase4_evaluation/quick_eval_ar.py \
                        --stage1_ckpt "$STAGE1_CKPT" \
                        --ckpt "$CKPT_FILE" \
                        --eval_file "$SQUAD2" \
                        --n_samples 0
                else
                    echo "SQuAD 2.0 file not found at $SQUAD2, skipping."
                fi
                    
                # 4. MLQA Arabic
                echo "[4] Evaluating MLQA Arabic..."
                PRED_AR="${PRED_DIR}/preds_ar_epoch_${EP}.json"
                python arabic/generate_preds_ar.py \
                    --stage1_ckpt "$STAGE1_CKPT" \
                    --ckpt "$CKPT_FILE" \
                    --eval_file "$MLQA_AR" \
                    --output_pred_file "$PRED_AR"
                    
                python mlqa_evaluation_v1.py "$MLQA_AR" "$PRED_AR" ar
                
                # 5. MLQA English
                echo "[5] Evaluating MLQA English..."
                PRED_EN="${PRED_DIR}/preds_en_epoch_${EP}.json"
                python arabic/generate_preds_ar.py \
                    --stage1_ckpt "$STAGE1_CKPT" \
                    --ckpt "$CKPT_FILE" \
                    --eval_file "$MLQA_EN" \
                    --output_pred_file "$PRED_EN"
                    
                python mlqa_evaluation_v1.py "$MLQA_EN" "$PRED_EN" en
                
                echo "Epoch $EP DONE."
            } > "$LOG_FILE" 2>&1
            
        ) &  # Chạy process ngầm (background)
        
    done
    
    echo "Wait: Đang chờ cả 4 epochs (001, 002, 003, best) đánh giá xong..."
    wait # Lệnh wait sẽ đợi cho đến khi toàn bộ background jobs của thư mục hiện tại hoàn thành
    echo "Hoàn thành thư mục $CKPT_DIR! Xem kết quả chi tiết tại: ${CKPT_DIR}/eval_epoch_*.log"
    echo ""
done

echo "========================================================="
echo "ALL DONE!"
echo "========================================================="
