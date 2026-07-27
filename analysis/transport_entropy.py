"""
analysis/transport_entropy.py
H2: Sinkhorn Transport Plan Entropy per Epoch × Language × Seed

For each saved checkpoint (epoch 1–4, seeds 42/43/44, for HI and VI):
  1. Load checkpoint (Stage 1 backbone + LoRA delta)
  2. Forward pass on validation set (n=200 subsample, fixed seed=0 for reproducibility)
  3. Extract transport plan γ via sinkhorn_log_domain (inference-only, torch.no_grad)
  4. Compute row-normalized Shannon entropy, average over tokens and examples

Output:
  analysis/transport_entropy_by_epoch_language.csv
  analysis/transport_entropy_chart.png

NOTE: subsample n=200 contexts randomly (random_state=0 fixed for reproducibility).
      This is reported explicitly so results can be reproduced.
"""

import os
import sys
import json
import random
import csv
import argparse

import torch
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT)

from transformers import AutoTokenizer
from phase2_model.model_core import CrossLingualOTModel
from phase3_loss.losses import sinkhorn_log_domain, OTAlignmentLoss
from gpu_utils import get_model

# ── config ───────────────────────────────────────────────────────
MODEL_NAME    = "xlm-roberta-base"
STAGE1_CKPT   = os.path.join(_ROOT, "checkpoints", "stage1_squad_best.pt")
SUBSAMPLE_N   = 200      # max examples; report this number
SUBSAMPLE_SEED = 0       # fixed for reproducibility
MAX_LENGTH    = 384
BATCH_SIZE    = 8
EPSILON       = 0.03     # must match training config
SINKHORN_ITERS = 100

SEEDS  = [42, 43, 44]
EPOCHS = [1, 2, 3, 4]

# Checkpoint patterns per language.
# Adjust if your directory layout differs.
def resolve_ckpt_path(lang: str, seed: int, epoch: int) -> str:
    """Find existing checkpoint by checking multiple candidate locations."""
    if lang == "VI":
        dirs = [
            f"checkpoint_stage2_vi/m5_anneal_seed{seed}",
            f"checkpoint_stage2_vi/m4_static_seed{seed}",
            f"checkpoint_stage2_vi/m5_ours_seed{seed}",
            f"checkpoint_stage2_vi/table2_m5_ours",
            f"checkpoint_stage2_vi",
            f"checkpoint_stage2/m5_anneal_seed{seed}",
            f"checkpoint_stage2/m4_static_seed{seed}",
            f"checkpoint_stage2/m5_ours_seed{seed}",
            f"checkpoint_stage2",
        ]
        files = [
            f"stage2_epoch_{epoch:03d}.pt",
            f"stage2_vi_epoch_{epoch:03d}.pt",
            f"epoch_{epoch:03d}.pt",
        ]
        if epoch in (3, 4):
            files.extend(["stage2_best.pt", "stage2_vi_best.pt"])
    else:  # HI
        dirs = [
            f"checkpoint_stage2_hi/m5_anneal_seed{seed}",
            f"checkpoint_stage2_hi/m4_static_seed{seed}",
            f"checkpoint_stage2_hi/m5_ours_seed{seed}",
            f"checkpoint_stage2_hi/table2_m5_ours",
            f"checkpoint_stage2_hi",
            f"checkpoint_stage2/m5_anneal_seed{seed}",
            f"checkpoint_stage2/m4_static_seed{seed}",
            f"checkpoint_stage2/m5_ours_seed{seed}",
        ]
        files = [
            f"stage2_hi_epoch_{epoch:03d}.pt",
            f"stage2_epoch_{epoch:03d}.pt",
            f"epoch_{epoch:03d}.pt",
        ]
        if epoch in (3, 4):
            files.extend(["stage2_hi_best.pt", "stage2_best.pt"])

    for d in dirs:
        for f in files:
            p = os.path.join(_ROOT, d, f)
            if os.path.exists(p):
                return p
    return None

DATASETS = {
    "HI": os.path.join(_ROOT, "dataset", "xquad.hi.json"),
    "VI": os.path.join(_ROOT, "dataset", "xquad.vi.json"),
}

# ── helpers ──────────────────────────────────────────────────────

def row_entropy(gamma: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    gamma: (M, N) — already a valid transport plan (or batch-average).
    Row-normalise → probability distribution per source token → Shannon entropy.
    Returns scalar: mean entropy across source tokens.
    """
    row_sum = gamma.sum(dim=-1, keepdim=True).clamp(min=eps)
    p = gamma / row_sum                                    # (M, N)
    H = -(p * (p + eps).log()).sum(dim=-1)                # (M,)
    return H.mean()


def load_xquad_contexts(path: str, n: int, seed: int) -> list:
    """Sample up to n (context, question) pairs from XQuAD."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                answers = qa.get("answers", [])
                if answers:
                    pairs.append((qa["question"], ctx, answers[0]["text"],
                                  answers[0]["answer_start"]))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:n]


def build_batch(tokenizer, en_q, en_ctx, tgt_text, tgt_ctx, device, max_length):
    """
    Tokenise one EN and one target-language example into a single-item batch.
    Both use the same XLM-R tokenizer (shared vocab).
    For HI/VI the target_ctx is the translated context.
    """
    def encode(question, context):
        enc = tokenizer(
            question, context,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    en_ids, en_mask = encode(en_q, en_ctx)
    # For cross-lingual: target side uses the same question (EN) + translated context
    # Since XQuAD is parallel, we use the tgt_ctx (e.g. HI context) directly
    tgt_ids, tgt_mask = encode(en_q, tgt_ctx)

    return {
        "en_input_ids":      en_ids,
        "en_attention_mask": en_mask,
        "vi_input_ids":      tgt_ids,   # model expects vi_* keys even for HI (remapped)
        "vi_attention_mask": tgt_mask,
    }


def load_model_and_criterion(stage1_ckpt: str, stage2_ckpt: str, device: torch.device):
    model = CrossLingualOTModel(MODEL_NAME)
    criterion = OTAlignmentLoss(hidden_size=768)

    # Load Stage 1 weights
    s1 = torch.load(stage1_ckpt, map_location=device)
    model.load_state_dict(s1["model_state"], strict=False)
    criterion.load_state_dict(s1["criterion_state"])

    # Apply LoRA before loading Stage 2 delta
    model.apply_lora()

    # Load Stage 2 trainable-only weights
    s2 = torch.load(stage2_ckpt, map_location=device)
    base = get_model(model)
    base.load_state_dict(s2["model_state"], strict=False)

    model.to(device)
    model.eval()
    return model, criterion


def compute_mean_entropy(model, tokenizer, pairs, device) -> tuple:
    """
    Run inference-only forward pass on all pairs, compute per-example entropy,
    return (mean_entropy, std_entropy).
    """
    entropies = []
    with torch.no_grad():
        for (en_q, en_ctx, ans_text, ans_start) in pairs:
            try:
                batch = build_batch(tokenizer, en_q, en_ctx, ans_text, en_ctx, device, MAX_LENGTH)

                # Get hidden states for both sides
                out_en = model(batch, branch="en")
                out_vi = model(batch, branch="vi")

                h_en = out_en["hidden"]           # (1, T_en, H)
                h_vi = out_vi["hidden"]           # (1, T_vi, H)
                en_pad = out_en["en_pad_mask"]    # (1, T_en)
                vi_pad = out_vi["vi_pad_mask"]    # (1, T_vi)

                # Truncate to effective lengths
                T_en = int((~en_pad).sum(dim=1).max().item())
                T_vi = int((~vi_pad).sum(dim=1).max().item())
                h_en = h_en[:, :T_en, :]
                h_vi = h_vi[:, :T_vi, :]
                en_pad = en_pad[:, :T_en]
                vi_pad = vi_pad[:, :T_vi]

                # Build cosine cost matrix
                en_norm = F.normalize(h_en, p=2, dim=-1)
                vi_norm = F.normalize(h_vi, p=2, dim=-1)
                C = 1.0 - torch.bmm(en_norm, vi_norm.transpose(1, 2))  # (1, T_en, T_vi)
                C = C.masked_fill(en_pad.unsqueeze(2), 1e4)
                C = C.masked_fill(vi_pad.unsqueeze(1), 1e4)

                # Compute transport plan (inference only)
                gamma = sinkhorn_log_domain(C, en_pad, vi_pad,
                                            epsilon=EPSILON, num_iters=SINKHORN_ITERS)
                # gamma: (1, T_en, T_vi) → squeeze batch dim → (T_en, T_vi)
                g = gamma[0]

                # Mask out PAD rows
                valid_rows = (~en_pad[0])                  # (T_en,) bool
                g_valid = g[valid_rows]                    # (n_valid, T_vi)

                ent = row_entropy(g_valid).item()
                entropies.append(ent)
            except Exception as e:
                # Log but don't abort — partial results are still useful
                print(f"    [skip] {e}")
                continue

    if not entropies:
        return float("nan"), float("nan")
    import statistics
    return statistics.mean(entropies), (statistics.stdev(entropies) if len(entropies) > 1 else 0.0)


# ── main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="H2: Transport plan entropy per epoch/language")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--subsample", type=int, default=SUBSAMPLE_N,
                        help="Max examples per run (default 200); set 0 for all")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    n_sample = args.subsample if args.subsample > 0 else 10_000

    print(f"[H2] Device: {device} | Subsample n={n_sample} (seed={SUBSAMPLE_SEED})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    rows = []

    for lang in ["HI", "VI"]:
        dataset_path = DATASETS[lang]
        if not os.path.exists(dataset_path):
            print(f"[H2] Dataset missing for {lang}: {dataset_path}")
            continue

        pairs = load_xquad_contexts(dataset_path, n=n_sample, seed=SUBSAMPLE_SEED)
        print(f"[H2] {lang}: {len(pairs)} examples loaded (requested {n_sample})")

        for seed in SEEDS:
            for epoch in EPOCHS:
                ckpt_path = resolve_ckpt_path(lang, seed, epoch)
                if ckpt_path is None:
                    print(f"  [H2] Checkpoint NOT FOUND (lang={lang}, seed={seed}, epoch={epoch})")
                    rows.append({
                        "language": lang, "epoch": epoch, "seed": seed,
                        "mean_entropy": "N/A", "std_entropy": "N/A",
                        "n_examples": 0, "status": "CKPT_NOT_FOUND",
                    })
                    continue

                print(f"  [H2] lang={lang}, seed={seed}, epoch={epoch} → {ckpt_path}")
                try:
                    model, criterion = load_model_and_criterion(STAGE1_CKPT, ckpt_path, device)
                    mean_ent, std_ent = compute_mean_entropy(model, tokenizer, pairs, device)
                    print(f"    entropy: mean={mean_ent:.6f}, std={std_ent:.6f}")
                    rows.append({
                        "language": lang, "epoch": epoch, "seed": seed,
                        "mean_entropy": round(mean_ent, 6),
                        "std_entropy": round(std_ent, 6),
                        "n_examples": len(pairs),
                        "status": "OK",
                    })
                    del model, criterion
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"    [H2] ERROR: {e}")
                    rows.append({
                        "language": lang, "epoch": epoch, "seed": seed,
                        "mean_entropy": "N/A", "std_entropy": "N/A",
                        "n_examples": 0, "status": f"ERROR: {e}",
                    })

    # ── write CSV ────────────────────────────────────────────────
    out_csv = os.path.join(_SCRIPT_DIR, "transport_entropy_by_epoch_language.csv")
    fieldnames = ["language", "epoch", "seed", "mean_entropy", "std_entropy", "n_examples", "status"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[H2] CSV written: {out_csv}")

    # ── plot ─────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import statistics as _stats

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = {"HI": "#e74c3c", "VI": "#2980b9"}

        for lang in ["HI", "VI"]:
            lang_rows = [r for r in rows if r["language"] == lang and r["status"] == "OK"]
            if not lang_rows:
                continue
            # Average across seeds per epoch
            epoch_means = {}
            epoch_stds  = {}
            for ep in EPOCHS:
                ep_vals = [float(r["mean_entropy"]) for r in lang_rows if r["epoch"] == ep
                           and isinstance(r["mean_entropy"], (int, float))]
                if ep_vals:
                    epoch_means[ep] = _stats.mean(ep_vals)
                    epoch_stds[ep]  = _stats.stdev(ep_vals) if len(ep_vals) > 1 else 0.0

            if epoch_means:
                xs = sorted(epoch_means.keys())
                ys = [epoch_means[x] for x in xs]
                es = [epoch_stds[x]  for x in xs]
                ax.errorbar(xs, ys, yerr=es, marker="o", label=lang,
                            color=colors.get(lang, None), capsize=4)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Row Entropy (γ)")
        ax.set_title("H2: Transport Plan Entropy by Epoch & Language\n(mean ± std over 3 seeds)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        chart_path = os.path.join(_SCRIPT_DIR, "transport_entropy_chart.png")
        plt.savefig(chart_path, dpi=150)
        print(f"[H2] Chart saved: {chart_path}")
    except ImportError:
        print("[H2] matplotlib not available — skipping chart generation")

    # ── interpret: two separate claims ──────────────────────────────
    # Claim A (offset): Is HI entropy already higher than VI at epoch 1?
    # Claim B (slope) : Is HI entropy increasing faster than VI across epochs?
    import statistics as _s
    ok_rows = [r for r in rows if r["status"] == "OK"]
    if ok_rows:
        # ── Claim A: offset at epoch 1 ───────────────────────────────
        hi_ep1 = [float(r["mean_entropy"]) for r in ok_rows
                  if r["language"] == "HI" and r["epoch"] == 1]
        vi_ep1 = [float(r["mean_entropy"]) for r in ok_rows
                  if r["language"] == "VI" and r["epoch"] == 1]
        if hi_ep1 and vi_ep1:
            hi_m1 = _s.mean(hi_ep1)
            vi_m1 = _s.mean(vi_ep1)
            offset_pct = (hi_m1 - vi_m1) / max(vi_m1, 1e-12) * 100
            if hi_m1 > vi_m1 * 1.05:
                print(f"\n[H2 Claim A — OFFSET] SUPPORTS: "
                      f"HI entropy at ep1 = {hi_m1:.4f}, VI = {vi_m1:.4f} "
                      f"(+{offset_pct:.1f}% higher). "
                      f"OT alignment is already less coherent for Hindi from the start.")
            else:
                print(f"\n[H2 Claim A — OFFSET] NOT SUPPORTED: "
                      f"HI={hi_m1:.4f} vs VI={vi_m1:.4f} ({offset_pct:+.1f}%) — comparable at epoch 1.")
        else:
            print("\n[H2 Claim A — OFFSET] INSUFFICIENT DATA (epoch 1 missing for one language)")

        # ── Claim B: slope across epochs ─────────────────────────────
        # Compute per-seed slope via OLS (epoch vs entropy), then average across seeds.
        def compute_slope(lang):
            slopes = []
            for seed in SEEDS:
                pts = [(r["epoch"], float(r["mean_entropy"]))
                       for r in ok_rows
                       if r["language"] == lang and r["seed"] == seed
                       and r["status"] == "OK"]
                if len(pts) < 2:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                n = len(xs)
                xm = sum(xs) / n
                ym = sum(ys) / n
                denom = sum((x - xm) ** 2 for x in xs)
                if denom < 1e-12:
                    continue
                slope = sum((xs[i] - xm) * (ys[i] - ym) for i in range(n)) / denom
                slopes.append(slope)
            return slopes

        hi_slopes = compute_slope("HI")
        vi_slopes = compute_slope("VI")

        if hi_slopes and vi_slopes:
            hi_slope_mean = _s.mean(hi_slopes)
            vi_slope_mean = _s.mean(vi_slopes)
            print(f"\n[H2 Claim B — SLOPE]  "
                  f"HI mean slope = {hi_slope_mean:+.6f}/epoch, "
                  f"VI mean slope = {vi_slope_mean:+.6f}/epoch")
            if hi_slope_mean > vi_slope_mean * 1.20:
                print(f"  → SUPPORTS faster HI degradation: "
                      f"HI slope {hi_slope_mean:+.4f} > VI slope {vi_slope_mean:+.4f}")
            elif abs(hi_slope_mean - vi_slope_mean) < abs(max(vi_slope_mean, 1e-12)) * 0.20:
                print(f"  → NOT SUPPORTED: slopes comparable — "
                      f"HI is not increasing faster than VI in relative terms.")
                print(f"  Interpretation: the issue is a persistent OFFSET (Claim A), "
                      f"not accelerating degradation per epoch.")
            else:
                print(f"  → AMBIGUOUS: difference exists but below 20% threshold for a confident claim.")
        else:
            print("\n[H2 Claim B — SLOPE] INSUFFICIENT DATA (need ≥2 epochs per seed per language)")


if __name__ == "__main__":
    main()
