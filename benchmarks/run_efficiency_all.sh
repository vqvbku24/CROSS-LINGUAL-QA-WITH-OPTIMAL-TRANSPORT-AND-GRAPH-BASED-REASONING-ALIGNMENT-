#!/usr/bin/env bash
# Script to run all 3 efficiency benchmark variants in parallel

echo "========================================================="
echo "Running efficiency benchmarks for 3 variants in PARALLEL"
echo "========================================================="

VARIANTS=("full_ft" "vanilla_kd" "coordinated")
GPUS=(0 1 2) # Gán mỗi variant cho 1 card GPU riêng biệt để không nhiễu kết quả

for i in "${!VARIANTS[@]}"; do
    VARIANT="${VARIANTS[$i]}"
    GPU_ID="${GPUS[$i]}"
    
    echo "[+] Launching $VARIANT variant on GPU $GPU_ID in background..."
    
    (
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        LOG_FILE="benchmarks/efficiency_${VARIANT}.log"
        
        {
            echo "---------------------------------------------------------"
            echo " Variant: $VARIANT"
            echo " GPU: $GPU_ID"
            echo "---------------------------------------------------------"
            python benchmarks/measure_efficiency.py --variant "$VARIANT"
            echo "$VARIANT DONE."
        } > "$LOG_FILE" 2>&1
        
    ) &
done

echo "Wait: Đang chờ cả 3 variants đo đạc xong..."
wait
echo "Hoàn thành! Kết quả tổng hợp đã được lưu trong: results/efficiency_benchmark.json"
echo "Log chi tiết của từng variant nằm ở: benchmarks/efficiency_*.log"
