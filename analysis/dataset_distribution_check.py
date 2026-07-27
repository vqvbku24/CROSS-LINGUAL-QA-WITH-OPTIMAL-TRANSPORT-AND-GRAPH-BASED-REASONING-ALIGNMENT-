"""
analysis/dataset_distribution_check.py
H4: Data-Distributional Check

Computes, for each language (VI, AR, HI), on the validation split:
  - Answer length (words & tokens): mean / median / std
  - % unanswerable questions (is_impossible / empty answers)
  - Context length (tokens): mean / median / std
  - Answer position in context (% char position): head / middle / tail

Output: analysis/dataset_distribution_by_language.csv
"""

import os
import sys
import json
import csv
import statistics

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT)

from transformers import AutoTokenizer

MODEL_NAME = "xlm-roberta-base"
print(f"[H4] Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_squad_examples(path: str) -> list:
    """Load all QA examples from a SQuAD-format JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    examples = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                answers = qa.get("answers", [])
                is_impossible = qa.get("is_impossible", False) or (len(answers) == 0)
                examples.append({
                    "context": ctx,
                    "question": qa.get("question", ""),
                    "answers": answers,
                    "is_impossible": is_impossible,
                })
    return examples


def analyze_language(lang: str, path: str) -> dict:
    if not os.path.exists(path):
        print(f"[H4] WARNING: dataset not found for {lang}: {path}")
        return {"language": lang, "status": "FILE_NOT_FOUND",
                "n_total": 0, "n_unanswerable": 0, "pct_unanswerable": "N/A",
                "ans_len_words_mean": "N/A", "ans_len_words_median": "N/A", "ans_len_words_std": "N/A",
                "ans_len_tokens_mean": "N/A", "ans_len_tokens_median": "N/A", "ans_len_tokens_std": "N/A",
                "ctx_len_tokens_mean": "N/A", "ctx_len_tokens_median": "N/A", "ctx_len_tokens_std": "N/A",
                "ans_pos_pct_mean": "N/A", "ans_pos_pct_median": "N/A", "ans_pos_pct_std": "N/A",
                "ans_pos_head_pct": "N/A", "ans_pos_mid_pct": "N/A", "ans_pos_tail_pct": "N/A"}

    print(f"[H4] Processing {lang} from {path}...")
    examples = load_squad_examples(path)
    n_total = len(examples)
    n_unanswerable = sum(1 for ex in examples if ex["is_impossible"])

    ans_len_words = []
    ans_len_tokens = []
    ctx_len_tokens = []
    ans_pos_pct = []
    pos_buckets = {"head": 0, "mid": 0, "tail": 0}  # thirds of context

    for ex in examples:
        ctx = ex["context"]
        ctx_toks = tokenizer.tokenize(ctx)
        ctx_len_tokens.append(len(ctx_toks))

        if not ex["is_impossible"] and ex["answers"]:
            ans_text = ex["answers"][0]["text"]
            ans_start = ex["answers"][0].get("answer_start", -1)

            # word length
            ans_len_words.append(len(ans_text.split()))
            # token length
            ans_len_tokens.append(len(tokenizer.tokenize(ans_text)))
            # position
            if ans_start >= 0 and len(ctx) > 0:
                pos_frac = ans_start / len(ctx)
                ans_pos_pct.append(pos_frac * 100)
                if pos_frac < 1/3:
                    pos_buckets["head"] += 1
                elif pos_frac < 2/3:
                    pos_buckets["mid"] += 1
                else:
                    pos_buckets["tail"] += 1

    n_answerable = len(ans_len_words)

    def safe_stats(lst):
        if len(lst) < 2:
            m = lst[0] if lst else "N/A"
            return m, m, 0.0
        return (round(statistics.mean(lst), 4),
                round(statistics.median(lst), 4),
                round(statistics.stdev(lst), 4))

    alw_mean, alw_med, alw_std = safe_stats(ans_len_words) if ans_len_words else ("N/A", "N/A", "N/A")
    alt_mean, alt_med, alt_std = safe_stats(ans_len_tokens) if ans_len_tokens else ("N/A", "N/A", "N/A")
    cl_mean,  cl_med,  cl_std  = safe_stats(ctx_len_tokens)
    ap_mean,  ap_med,  ap_std  = safe_stats(ans_pos_pct) if ans_pos_pct else ("N/A", "N/A", "N/A")

    n_pos = sum(pos_buckets.values())
    head_pct = round(pos_buckets["head"] / n_pos * 100, 1) if n_pos else "N/A"
    mid_pct  = round(pos_buckets["mid"]  / n_pos * 100, 1) if n_pos else "N/A"
    tail_pct = round(pos_buckets["tail"] / n_pos * 100, 1) if n_pos else "N/A"

    print(f"  {lang}: n={n_total}, unanswerable={n_unanswerable} ({n_unanswerable/n_total*100:.1f}%)")
    print(f"       ans_words: mean={alw_mean}, median={alw_med}, std={alw_std}")
    print(f"       ans_toks:  mean={alt_mean}, median={alt_med}, std={alt_std}")
    print(f"       ctx_toks:  mean={cl_mean}, median={cl_med}, std={cl_std}")
    print(f"       ans_pos%%:  mean={ap_mean}, head={head_pct}%%, mid={mid_pct}%%, tail={tail_pct}%%")

    return {
        "language": lang,
        "status": "OK",
        "n_total": n_total,
        "n_unanswerable": n_unanswerable,
        "pct_unanswerable": round(n_unanswerable / n_total * 100, 2) if n_total else "N/A",
        "ans_len_words_mean": alw_mean, "ans_len_words_median": alw_med, "ans_len_words_std": alw_std,
        "ans_len_tokens_mean": alt_mean, "ans_len_tokens_median": alt_med, "ans_len_tokens_std": alt_std,
        "ctx_len_tokens_mean": cl_mean, "ctx_len_tokens_median": cl_med, "ctx_len_tokens_std": cl_std,
        "ans_pos_pct_mean": ap_mean, "ans_pos_pct_median": ap_med, "ans_pos_pct_std": ap_std,
        "ans_pos_head_pct": head_pct, "ans_pos_mid_pct": mid_pct, "ans_pos_tail_pct": tail_pct,
    }


DATASETS = {
    "VI": os.path.join(_ROOT, "dataset", "xquad.vi.json"),
    "AR": os.path.join(_ROOT, "dataset", "xquad.ar.json"),
    "HI": os.path.join(_ROOT, "dataset", "xquad.hi.json"),
}

results = [analyze_language(lang, path) for lang, path in DATASETS.items()]

out_csv = os.path.join(_SCRIPT_DIR, "dataset_distribution_by_language.csv")
fieldnames = [
    "language", "status", "n_total", "n_unanswerable", "pct_unanswerable",
    "ans_len_words_mean", "ans_len_words_median", "ans_len_words_std",
    "ans_len_tokens_mean", "ans_len_tokens_median", "ans_len_tokens_std",
    "ctx_len_tokens_mean", "ctx_len_tokens_median", "ctx_len_tokens_std",
    "ans_pos_pct_mean", "ans_pos_pct_median", "ans_pos_pct_std",
    "ans_pos_head_pct", "ans_pos_mid_pct", "ans_pos_tail_pct",
]
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\n[H4] Results written to: {out_csv}")

# ── interpret ────────────────────────────────────────────────────
# NOTE: verdict uses TOKEN-based metrics only (ans_len_tokens_mean, ctx_len_tokens_mean).
# ans_len_words_mean is excluded — whitespace tokenization differs across scripts
# for languages like Hindi (compound morphology) vs Vietnamese (monosyllabic), so
# word-count differences would reflect the splitting heuristic, not true complexity.
ok = {r["language"]: r for r in results if r["status"] == "OK"}
if "HI" in ok and len(ok) >= 2:
    hi = ok["HI"]
    others = {k: v for k, v in ok.items() if k != "HI"}
    notes = []
    for lang, row in others.items():
        for metric in ["ans_len_tokens_mean", "ctx_len_tokens_mean"]:
            try:
                hi_val = float(hi[metric])
                ot_val = float(row[metric])
                diff_pct = abs(hi_val - ot_val) / max(ot_val, 1e-9) * 100
                if diff_pct > 25:
                    notes.append(f"  {metric}: HI={hi_val:.2f} vs {lang}={ot_val:.2f} ({diff_pct:.1f}% diff)")
            except (TypeError, ValueError):
                pass
    if notes:
        print("\n[H4 VERDICT] POSSIBLE SUPPORT — notable token-based distributional differences (>25%):")
        for n in notes:
            print(n)
    else:
        print("\n[H4 VERDICT] DOES NOT SUPPORT H4: token-based distributions broadly comparable across languages.")
        print("  (Note: ans_len_words_mean excluded from verdict — word-split artifacts differ per language)")
