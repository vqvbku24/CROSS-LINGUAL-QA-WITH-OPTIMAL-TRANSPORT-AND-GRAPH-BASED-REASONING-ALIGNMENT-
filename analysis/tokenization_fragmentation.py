"""
analysis/tokenization_fragmentation.py
H1: Tokenization Fragmentation Ratio

Measures subword/word ratio for each language using the same XLM-R tokenizer
used in training (xlm-roberta-base, as configured in train_stage2.py /
train_stage2_hi.py / train_stage2_ar.py).

Datasets used (validation split of each branch):
  - Vietnamese : dataset/xquad.vi.json       (XQuAD-vi, context field)
  - Arabic     : dataset/xquad.ar.json        (XQuAD-ar)
  - Hindi      : dataset/xquad.hi.json        (XQuAD-hi / IndicSQuAD proxy)

Output: analysis/fragmentation_ratio_by_language.csv
"""

import os
import sys
import json
import csv
import statistics

# ── resolve project root ─────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT)

from transformers import AutoTokenizer

# ── tokenizer (same as training) ────────────────────────────────
MODEL_NAME = "xlm-roberta-base"
print(f"[H1] Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def fragmentation_ratio(text: str) -> float:
    """subwords / words (whitespace-split proxy)."""
    words = text.split()
    subwords = tokenizer.tokenize(text)
    return len(subwords) / max(len(words), 1)


def load_contexts_from_xquad(path: str) -> list:
    """Extract all context strings from a SQuAD-format JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    contexts = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            contexts.append(para["context"])
    return contexts


# ── dataset paths ────────────────────────────────────────────────
DATASETS = {
    "VI": os.path.join(_ROOT, "dataset", "xquad.vi.json"),
    "AR": os.path.join(_ROOT, "dataset", "xquad.ar.json"),
    "HI": os.path.join(_ROOT, "dataset", "xquad.hi.json"),
}

results = []

for lang, path in DATASETS.items():
    if not os.path.exists(path):
        print(f"[H1] WARNING: dataset not found for {lang}: {path}")
        results.append({
            "language": lang,
            "mean_ratio": "N/A",
            "median_ratio": "N/A",
            "std_ratio": "N/A",
            "n_samples": 0,
            "status": "FILE_NOT_FOUND",
        })
        continue

    print(f"[H1] Processing {lang} from {path}...")
    contexts = load_contexts_from_xquad(path)
    ratios = [fragmentation_ratio(ctx) for ctx in contexts]

    mean_r   = statistics.mean(ratios)
    median_r = statistics.median(ratios)
    std_r    = statistics.stdev(ratios) if len(ratios) > 1 else 0.0

    print(f"  {lang}: n={len(ratios)}, mean={mean_r:.4f}, median={median_r:.4f}, std={std_r:.4f}")
    results.append({
        "language": lang,
        "mean_ratio": round(mean_r, 4),
        "median_ratio": round(median_r, 4),
        "std_ratio": round(std_r, 4),
        "n_samples": len(ratios),
        "status": "OK",
    })

# ── write CSV ────────────────────────────────────────────────────
out_csv = os.path.join(_SCRIPT_DIR, "fragmentation_ratio_by_language.csv")
fieldnames = ["language", "mean_ratio", "median_ratio", "std_ratio", "n_samples", "status"]
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\n[H1] Results written to: {out_csv}")

# ── interpret ────────────────────────────────────────────────────
ok_results = [r for r in results if r["status"] == "OK"]
if len(ok_results) >= 2:
    hi_row = next((r for r in ok_results if r["language"] == "HI"), None)
    others = [r for r in ok_results if r["language"] != "HI"]
    if hi_row and others:
        hi_mean = hi_row["mean_ratio"]
        other_means = [r["mean_ratio"] for r in others]
        max_other = max(other_means)
        pct_higher = (hi_mean - max_other) / max_other * 100
        if pct_higher > 20:
            verdict = f"SUPPORTS H1: HI ratio ({hi_mean:.4f}) is {pct_higher:.1f}% higher than best non-HI ({max_other:.4f})"
        else:
            verdict = f"DOES NOT SUPPORT H1: HI ratio ({hi_mean:.4f}) only {pct_higher:.1f}% above best non-HI ({max_other:.4f}) — threshold is >20%"
        print(f"\n[H1 VERDICT] {verdict}")
