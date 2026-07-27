#!/usr/bin/env bash
# run_hypothesis_tests.sh
#
# Kiểm định 4 giả thuyết về sự suy giảm XQuAD-hi (Table 9 / Appendix I.1).
# Thứ tự chạy: H1 → H4 (rẻ, không cần GPU) → H2 → H3 (cần load checkpoint).
#
# RÀNG BUỘC: đây là investigation-only — không thay đổi code training,
# không train lại, không tạo checkpoint mới.
#
# Usage:
#   bash run_hypothesis_tests.sh [--gpu GPU_ID] [--subsample N] [--skip-h2] [--skip-h3]
#
# Options:
#   --gpu      GPU index for H2 and H3 (default: 0)
#   --subsample  Number of examples to use in H2 forward passes (default: 200)
#   --skip-h2  Skip H2 (transport entropy) — useful if checkpoints are not available
#   --skip-h3  Skip H3 (representation drift) — useful if checkpoints are not available
#
# Outputs (all in analysis/):
#   fragmentation_ratio_by_language.csv        ← H1
#   dataset_distribution_by_language.csv       ← H4
#   transport_entropy_by_epoch_language.csv    ← H2
#   transport_entropy_chart.png                ← H2 (line chart)
#   hindi_representation_drift.csv             ← H3
#   ../HINDI_ANOMALY_FINDINGS.md               ← final report (compiled from all above)

set -euo pipefail

# ── parse args ───────────────────────────────────────────────────
GPU_ID=0
SUBSAMPLE=200
SKIP_H2=0
SKIP_H3=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)      GPU_ID="$2"; shift 2 ;;
        --subsample) SUBSAMPLE="$2"; shift 2 ;;
        --skip-h2)  SKIP_H2=1; shift ;;
        --skip-h3)  SKIP_H3=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── resolve paths ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
ANALYSIS_DIR="$SCRIPT_DIR"

# Ensure we run from project root so relative imports work
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/hypothesis_tests_${TIMESTAMP}.log"

# Tee all output to master log
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "========================================================"
echo " Hindi Anomaly Hypothesis Tests"
echo " Started: $(date)"
echo " GPU: ${GPU_ID}  |  Subsample (H2): ${SUBSAMPLE}"
echo " Root: ${ROOT_DIR}"
echo "========================================================"
echo ""

# ── check Python environment ─────────────────────────────────────
echo "[PREFLIGHT] Checking Python environment..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
python -c "import scipy" 2>/dev/null \
    && echo "  scipy: OK" \
    || echo "  WARNING: scipy not found — H3 statistical tests will fail. Install: pip install scipy"
python -c "import matplotlib" 2>/dev/null \
    && echo "  matplotlib: OK" \
    || echo "  WARNING: matplotlib not found — H2 chart will be skipped."
echo ""

# ── checkpoint discovery ──────────────────────────────────────────
echo "[PREFLIGHT] Discovering checkpoints..."
STAGE1_CKPT="${ROOT_DIR}/checkpoints/stage1_squad_best.pt"
if [ ! -f "$STAGE1_CKPT" ]; then
    echo "  ERROR: Stage 1 checkpoint not found: $STAGE1_CKPT"
    echo "  Cannot proceed with H2 and H3 (H1 and H4 do not require it)."
    SKIP_H2=1
    SKIP_H3=1
else
    echo "  Stage 1 checkpoint: OK ($STAGE1_CKPT)"
fi

# Check at least one HI/VI checkpoint via python resolver
HI_FOUND=$(python -c "from analysis.transport_entropy import resolve_ckpt_path; print('OK' if any(resolve_ckpt_path('HI', s, 1) for s in [42,43,44]) else 'MISSING')")
VI_FOUND=$(python -c "from analysis.transport_entropy import resolve_ckpt_path; print('OK' if any(resolve_ckpt_path('VI', s, 1) for s in [42,43,44]) else 'MISSING')")

if [ "$HI_FOUND" = "OK" ]; then
    echo "  HI stage-2 checkpoints: FOUND"
else
    echo "  WARNING: No HI stage-2 checkpoints found — H2/H3 will report CKPT_NOT_FOUND for HI"
fi

if [ "$VI_FOUND" = "OK" ]; then
    echo "  VI stage-2 checkpoints: FOUND"
else
    echo "  WARNING: No VI stage-2 checkpoints found — H2/H3 will report CKPT_NOT_FOUND for VI"
fi
echo ""

# ── dataset check ────────────────────────────────────────────────
echo "[PREFLIGHT] Checking datasets..."
for LANG_FILE in \
    "dataset/xquad.vi.json" \
    "dataset/xquad.ar.json" \
    "dataset/xquad.hi.json"; do
    FULL="${ROOT_DIR}/${LANG_FILE}"
    if [ -f "$FULL" ]; then
        echo "  $LANG_FILE: OK"
    else
        echo "  $LANG_FILE: NOT FOUND — H1/H4/H2/H3 will skip this language"
    fi
done
echo ""

# ════════════════════════════════════════════════════════════════
# H1: Tokenization Fragmentation Ratio  (~5 min, no GPU needed)
# ════════════════════════════════════════════════════════════════
echo "========================================================"
echo " [H1] Tokenization Fragmentation Ratio"
echo " Script: analysis/tokenization_fragmentation.py"
echo "========================================================"
H1_LOG="${LOG_DIR}/h1_fragmentation_${TIMESTAMP}.log"
if python "${ANALYSIS_DIR}/tokenization_fragmentation.py" \
       2>&1 | tee "$H1_LOG"; then
    echo "[H1] DONE — output: analysis/fragmentation_ratio_by_language.csv"
    echo "     log:    $H1_LOG"
else
    echo "[H1] FAILED — see log: $H1_LOG"
fi
echo ""

# ════════════════════════════════════════════════════════════════
# H4: Data-Distributional Check  (~5 min, no GPU needed)
# ════════════════════════════════════════════════════════════════
echo "========================================================"
echo " [H4] Data-Distributional Check"
echo " Script: analysis/dataset_distribution_check.py"
echo "========================================================"
H4_LOG="${LOG_DIR}/h4_distribution_${TIMESTAMP}.log"
if python "${ANALYSIS_DIR}/dataset_distribution_check.py" \
       2>&1 | tee "$H4_LOG"; then
    echo "[H4] DONE — output: analysis/dataset_distribution_by_language.csv"
    echo "     log:    $H4_LOG"
else
    echo "[H4] FAILED — see log: $H4_LOG"
fi
echo ""

# ════════════════════════════════════════════════════════════════
# H2: Transport Plan Entropy  (needs GPU + checkpoints)
# ════════════════════════════════════════════════════════════════
if [ "$SKIP_H2" -eq 1 ]; then
    echo "[H2] SKIPPED (--skip-h2 or missing Stage 1 checkpoint)"
else
    echo "========================================================"
    echo " [H2] Sinkhorn Transport Plan Entropy"
    echo " Script: analysis/transport_entropy.py"
    echo " GPU: ${GPU_ID} | Subsample: ${SUBSAMPLE} examples (seed=0, fixed)"
    echo "========================================================"
    H2_LOG="${LOG_DIR}/h2_entropy_${TIMESTAMP}.log"
    if CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${ANALYSIS_DIR}/transport_entropy.py" \
           --gpu 0 \
           --subsample "${SUBSAMPLE}" \
           2>&1 | tee "$H2_LOG"; then
        echo "[H2] DONE — output: analysis/transport_entropy_by_epoch_language.csv"
        echo "     chart:  analysis/transport_entropy_chart.png"
        echo "     log:    $H2_LOG"
    else
        echo "[H2] FAILED — see log: $H2_LOG"
    fi
fi
echo ""

# ════════════════════════════════════════════════════════════════
# H3: Representation Drift  (needs GPU + checkpoints)
# ════════════════════════════════════════════════════════════════
if [ "$SKIP_H3" -eq 1 ]; then
    echo "[H3] SKIPPED (--skip-h3 or missing Stage 1 checkpoint)"
else
    echo "========================================================"
    echo " [H3] Representation Drift (Appendix F diagnostics)"
    echo " Script: analysis/hindi_representation_drift.py"
    echo " GPU: ${GPU_ID} | n_pairs=50 (seed=42, fixed)"
    echo "========================================================"
    H3_LOG="${LOG_DIR}/h3_drift_${TIMESTAMP}.log"
    if CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${ANALYSIS_DIR}/hindi_representation_drift.py" \
           --gpu 0 \
           2>&1 | tee "$H3_LOG"; then
        echo "[H3] DONE — output: analysis/hindi_representation_drift.csv"
        echo "     log:    $H3_LOG"
    else
        echo "[H3] FAILED — see log: $H3_LOG"
    fi
fi
echo ""

# ════════════════════════════════════════════════════════════════
# Compile findings → HINDI_ANOMALY_FINDINGS.md
# ════════════════════════════════════════════════════════════════
echo "========================================================"
echo " Compiling all results → HINDI_ANOMALY_FINDINGS.md"
echo "========================================================"
COMPILE_LOG="${LOG_DIR}/compile_findings_${TIMESTAMP}.log"
if python "${ANALYSIS_DIR}/compile_findings.py" \
       2>&1 | tee "$COMPILE_LOG"; then
    echo "[compile] DONE — report: HINDI_ANOMALY_FINDINGS.md"
else
    echo "[compile] FAILED — see log: $COMPILE_LOG"
fi
echo ""

# ── summary ──────────────────────────────────────────────────────
echo "========================================================"
echo " All steps complete. Summary of outputs:"
echo ""
echo "  analysis/fragmentation_ratio_by_language.csv      (H1)"
echo "  analysis/dataset_distribution_by_language.csv     (H4)"
if [ "$SKIP_H2" -eq 0 ]; then
    echo "  analysis/transport_entropy_by_epoch_language.csv  (H2)"
    echo "  analysis/transport_entropy_chart.png              (H2)"
fi
if [ "$SKIP_H3" -eq 0 ]; then
    echo "  analysis/hindi_representation_drift.csv           (H3)"
fi
echo "  HINDI_ANOMALY_FINDINGS.md                         (final report)"
echo ""
echo "  Master log: $MASTER_LOG"
echo ""
echo " Finished: $(date)"
echo "========================================================"
