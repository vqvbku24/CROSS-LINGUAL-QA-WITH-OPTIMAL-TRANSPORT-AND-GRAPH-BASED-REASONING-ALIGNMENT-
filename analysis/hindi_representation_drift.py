"""
analysis/hindi_representation_drift.py
H3: Representation Drift — reuses logic from paper_tools/aggregate_alignment_stats.py

Compares, for HI (epoch 1 → epoch 4) and VI (epoch 1 → epoch 3, or epoch 3 as checkpoint),
the following 4 diagnostics (same as Appendix F / Figure 5):
  (a) OT-weighted Euclidean distance (answer tokens): before/after
  (b) OT-weighted cosine similarity (answer tokens): before/after
  (c) Norm change: EN answer-token norm change between "before" and "after" state
  (d) Common-mode shift cosine: are EN and VI representations drifting together?

"Before" = frozen backbone (no LoRA), "after" = LoRA-active backbone.
Both extracted from the *same checkpoint* at a given epoch.

n=50 paired examples per (language, epoch) as designed in Appendix F.

Output:
  analysis/hindi_representation_drift.csv  — 4 metrics × 2 languages × epochs
  Paired t-test + Wilcoxon signed-rank test per metric (HI ep1→ep4 vs VI ep1→ep3)

NOTE: does NOT modify any file in paper_tools/. All logic is copied here.
"""

import os
import sys
import json
import random
import csv
import argparse

import numpy as np
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
N_PAIRS       = 50       # same as Appendix F design
SUBSAMPLE_SEED = 42
MAX_LENGTH    = 384
EPSILON       = 0.03
SINKHORN_ITERS = 100

# Checkpoint patterns — adjust seed if per-seed runs differ;
# we use seed=42 (first seed) as the representative run for H3.
def resolve_ckpt_path(lang: str, epoch: int, seed: int = 42) -> str:
    """Find existing checkpoint by checking multiple candidate locations."""
    if lang == "VI":
        dirs = [
            f"checkpoint_stage2_vi/m4_static_seed{seed}",   # same variant as HI (m4_static) — primary
            f"checkpoint_stage2_vi/m5_anneal_seed{seed}",   # fallback if static not available
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
            f"checkpoint_stage2_hi/m4_static_seed{seed}",
            f"checkpoint_stage2_hi/table2_m5_ours_first",
            f"checkpoint_stage2_hi/table2_m4_ours_first",
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

# Epoch pairs to compare per language.
# HI runs TWO comparisons: 1→3 (same span as VI, for fair comparison) AND
# 1→4 (full training span). Both are written to the CSV so compile_findings.py
# can show which portion of HI drift is attributable to training duration.
EPOCH_PAIRS = {
    "HI": [(1, 3), (1, 4)],  # 1→3: symmetric with VI; 1→4: full span
    "VI": [(1, 3)],          # VI selected checkpoint is epoch 3
}


# ── copied logic from aggregate_alignment_stats.py ───────────────
# (NOT importing from paper_tools — see no-touch constraint)

def ot_weighted_distance(en_vecs, vi_vecs, gamma):
    """OT-weighted Euclidean distance per EN token."""
    w = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-12)
    vi_centroid = w @ vi_vecs
    return np.linalg.norm(en_vecs - vi_centroid, axis=1)


def ot_weighted_cosine(en_vecs, vi_vecs, gamma):
    """OT-weighted cosine similarity per EN token."""
    w = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-12)
    vi_centroid = w @ vi_vecs
    en_n = np.linalg.norm(en_vecs, axis=1) + 1e-12
    vi_n = np.linalg.norm(vi_centroid, axis=1) + 1e-12
    return (en_vecs * vi_centroid).sum(axis=1) / (en_n * vi_n)


def norm_change(h_before, h_after, idx):
    """Absolute norm change for token indices idx."""
    nb = np.linalg.norm(h_before[idx], axis=1)
    na = np.linalg.norm(h_after[idx], axis=1)
    return na - nb   # positive = norm grew


def common_mode_cosine(h_bef_en, h_aft_en, h_bef_vi, h_aft_vi, gamma, en_idx):
    """Cosine of (en_shift, ot-weighted vi_shift): 1=co-drift, -1=diverge."""
    en_shift = h_aft_en[en_idx] - h_bef_en[en_idx]     # (n_ans, H)
    vi_shift_all = h_aft_vi - h_bef_vi                  # (T_vi, H)
    sub_g = gamma[en_idx, :]                             # (n_ans, T_vi)
    w = sub_g / (sub_g.sum(axis=1, keepdims=True) + 1e-12)
    vi_shift_w = w @ vi_shift_all                        # (n_ans, H)
    en_n = np.linalg.norm(en_shift, axis=1) + 1e-12
    vi_n = np.linalg.norm(vi_shift_w, axis=1) + 1e-12
    return (en_shift * vi_shift_w).sum(axis=1) / (en_n * vi_n)


# ── data loading ─────────────────────────────────────────────────

def load_pairs(path: str, n: int, seed: int):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                answers = qa.get("answers", [])
                if answers:
                    pairs.append({
                        "question": qa["question"],
                        "context": ctx,
                        "ans_text": answers[0]["text"],
                        "ans_start": answers[0]["answer_start"],
                    })
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:n]


# ── model loading ────────────────────────────────────────────────

def load_checkpoint(stage1_ckpt: str, stage2_ckpt: str, device):
    model = CrossLingualOTModel(MODEL_NAME)
    criterion = OTAlignmentLoss(hidden_size=768)
    s1 = torch.load(stage1_ckpt, map_location=device)
    model.load_state_dict(s1["model_state"], strict=False)
    criterion.load_state_dict(s1["criterion_state"])
    model.apply_lora()
    s2 = torch.load(stage2_ckpt, map_location=device)
    get_model(model).load_state_dict(s2["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model


# ── extract hidden states (before = no-LoRA, after = LoRA-on) ───

def extract_hidden(model, tokenizer, pairs, device):
    """
    Returns dict with keys:
      'h_before_en', 'h_before_vi' — frozen (no LoRA) hidden states, stacked over all pairs
      'h_after_en',  'h_after_vi'  — LoRA-active hidden states
      'gamma'                      — OT plan, shape (n_en_tokens, n_vi_tokens) pooled
      'en_ans_idx'                 — indices of EN answer tokens in pooled h_before_en
    """
    all_h_bef_en, all_h_aft_en = [], []
    all_h_bef_vi, all_h_aft_vi = [], []
    all_gamma  = []
    all_ans_idx = []
    en_offset = 0

    with torch.no_grad():
        for pair in pairs:
            q  = pair["question"]
            ctx = pair["context"]

            def encode(q, ctx):
                enc = tokenizer(q, ctx, max_length=MAX_LENGTH, truncation=True,
                                padding="max_length", return_tensors="pt")
                return enc["input_ids"].to(device), enc["attention_mask"].to(device)

            en_ids, en_mask = encode(q, ctx)
            vi_ids, vi_mask = encode(q, ctx)   # same for cross-lingual (parallel pair)

            batch = {
                "en_input_ids": en_ids, "en_attention_mask": en_mask,
                "vi_input_ids": vi_ids, "vi_attention_mask": vi_mask,
            }

            # ── before (frozen backbone, LoRA disabled) ──────────
            base = get_model(model)
            with base.backbone.disable_adapter():
                h_bef_en_t = base(batch, branch="en")["hidden"][0]  # (T, H)
                h_bef_vi_t = base(batch, branch="vi")["hidden"][0]

            # ── after (LoRA active) ───────────────────────────────
            h_aft_en_t = base(batch, branch="en")["hidden"][0]
            h_aft_vi_t = base(batch, branch="vi")["hidden"][0]

            # effective token count (truncate PAD)
            T_en = int(en_mask.sum().item())
            T_vi = int(vi_mask.sum().item())

            h_bef_en_np = h_bef_en_t[:T_en].cpu().numpy()
            h_bef_vi_np = h_bef_vi_t[:T_vi].cpu().numpy()
            h_aft_en_np = h_aft_en_t[:T_en].cpu().numpy()
            h_aft_vi_np = h_aft_vi_t[:T_vi].cpu().numpy()

            # ── OT plan from "after" state ────────────────────────
            en_n = F.normalize(h_aft_en_t[:T_en].unsqueeze(0), p=2, dim=-1)
            vi_n = F.normalize(h_aft_vi_t[:T_vi].unsqueeze(0), p=2, dim=-1)
            C = (1.0 - torch.bmm(en_n, vi_n.transpose(1, 2)))         # (1, T_en, T_vi)
            en_pad = (en_mask[:, :T_en] == 0)
            vi_pad = (vi_mask[:, :T_vi] == 0)
            gamma  = sinkhorn_log_domain(C, en_pad, vi_pad,
                                         epsilon=EPSILON, num_iters=SINKHORN_ITERS)
            g_np = gamma[0].cpu().numpy()   # (T_en, T_vi)

            # ── answer token indices ──────────────────────────────
            # Find which EN tokens correspond to answer span
            ans_text  = pair["ans_text"]
            ans_start = pair["ans_start"]
            ans_end   = ans_start + len(ans_text)
            token_offsets = tokenizer(q, ctx, max_length=MAX_LENGTH,
                                      truncation=True, return_offsets_mapping=True)
            offsets = token_offsets["offset_mapping"]
            # Offset mapping: (char_start, char_end) for each token
            # Question tokens come before SEP; context tokens after
            # We need to account for question prefix length
            q_enc_len = len(tokenizer(q)["input_ids"]) - 1  # approx: skip [SEP]
            ctx_offset_shift = 0   # offsets are relative to (q + sep + ctx)
            ans_tok_idx = []
            for ti, (cs, ce) in enumerate(offsets[:T_en]):
                if cs == 0 and ce == 0:
                    continue
                # offset mapping uses char positions in the concatenated string
                # Rough heuristic: if the token is within the answer span
                if cs >= ans_start and ce <= ans_end:
                    ans_tok_idx.append(ti)
            if not ans_tok_idx:
                # fallback: use all non-question tokens (context side)
                ans_tok_idx = list(range(q_enc_len, min(q_enc_len + 5, T_en)))

            # Store as absolute indices (relative to this chunk)
            all_h_bef_en.append(h_bef_en_np)
            all_h_aft_en.append(h_aft_en_np)
            all_h_bef_vi.append(h_bef_vi_np)
            all_h_aft_vi.append(h_aft_vi_np)
            all_gamma.append(g_np)
            all_ans_idx.append((en_offset, ans_tok_idx, T_en))
            en_offset += T_en

    return {
        "h_bef_en_list": all_h_bef_en,
        "h_aft_en_list": all_h_aft_en,
        "h_bef_vi_list": all_h_bef_vi,
        "h_aft_vi_list": all_h_aft_vi,
        "gamma_list":    all_gamma,
        "ans_idx_list":  all_ans_idx,
    }


def compute_diagnostics(data: dict) -> dict:
    """
    Returns dict of 4 diagnostics (a-d), each is a numpy array over all answer tokens.
    """
    dist_before, dist_after   = [], []
    cos_before,  cos_after    = [], []
    norm_delta                = []
    common_mode               = []

    for i, (off, ans_idx, T_en) in enumerate(data["ans_idx_list"]):
        if not ans_idx:
            continue
        h_bef_en = data["h_bef_en_list"][i]
        h_aft_en = data["h_aft_en_list"][i]
        h_bef_vi = data["h_bef_vi_list"][i]
        h_aft_vi = data["h_aft_vi_list"][i]
        gamma    = data["gamma_list"][i]

        num_en = T_en
        ans = np.array(ans_idx)

        # (a) OT-weighted Euclidean distance
        sub_g = gamma[ans, :]
        d_b = ot_weighted_distance(h_bef_en[ans], h_bef_vi, sub_g)
        d_a = ot_weighted_distance(h_aft_en[ans], h_aft_vi, sub_g)
        dist_before.extend(d_b.tolist())
        dist_after.extend(d_a.tolist())

        # (b) OT-weighted cosine similarity
        c_b = ot_weighted_cosine(h_bef_en[ans], h_bef_vi, sub_g)
        c_a = ot_weighted_cosine(h_aft_en[ans], h_aft_vi, sub_g)
        cos_before.extend(c_b.tolist())
        cos_after.extend(c_a.tolist())

        # (c) norm change (after - before, EN answer tokens)
        nc = norm_change(h_bef_en, h_aft_en, ans)
        norm_delta.extend(nc.tolist())

        # (d) common-mode shift cosine
        cm = common_mode_cosine(h_bef_en, h_aft_en, h_bef_vi, h_aft_vi, gamma, ans)
        common_mode.extend(cm.tolist())

    return {
        "dist_before":  np.array(dist_before),
        "dist_after":   np.array(dist_after),
        "cos_before":   np.array(cos_before),
        "cos_after":    np.array(cos_after),
        "norm_delta":   np.array(norm_delta),
        "common_mode":  np.array(common_mode),
    }


def stat_test(a, b, label=""):
    """Paired t-test + Wilcoxon on two equal-length arrays. Returns dict of results."""
    from scipy import stats as sp
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    t_stat, t_p = sp.ttest_rel(a, b)
    try:
        w_stat, w_p = sp.wilcoxon(a - b)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    # Cohen's d for paired samples
    diff = a - b
    d = diff.mean() / (diff.std() + 1e-12)
    return {"label": label, "n": n,
            "mean_a": round(float(a.mean()), 6), "mean_b": round(float(b.mean()), 6),
            "mean_diff": round(float(diff.mean()), 6), "std_diff": round(float(diff.std()), 6),
            "cohens_d": round(float(d), 4),
            "t_stat": round(float(t_stat), 4), "t_p": round(float(t_p), 6),
            "w_stat": round(float(w_stat), 4) if not np.isnan(w_stat) else "N/A",
            "w_p": round(float(w_p), 6) if not np.isnan(w_p) else "N/A"}


# ── main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="H3: Representation drift HI vs VI")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_pairs", type=int, default=N_PAIRS)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[H3] Device: {device} | n_pairs={args.n_pairs}")

    try:
        from scipy import stats as _sp
    except ImportError:
        print("[H3] ERROR: scipy not available. Install with: pip install scipy")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    all_rows  = []

    for lang in ["HI", "VI"]:
        epoch_pair_list = EPOCH_PAIRS[lang]
        dataset_path = DATASETS[lang]

        if not os.path.exists(dataset_path):
            print(f"[H3] Dataset missing for {lang}: {dataset_path}")
            continue

        pairs = load_pairs(dataset_path, args.n_pairs, SUBSAMPLE_SEED)
        print(f"[H3] {lang}: {len(pairs)} pairs loaded")

        # Collect all unique epochs needed across all pairs for this language
        unique_epochs = sorted(set(e for pair in epoch_pair_list for e in pair))

        diag_by_epoch = {}
        for epoch in unique_epochs:
            ckpt_path = resolve_ckpt_path(lang, epoch)
            if ckpt_path is None:
                print(f"  [H3] Checkpoint NOT FOUND (lang={lang}, epoch={epoch})")
                diag_by_epoch[epoch] = None
                continue
            print(f"  [H3] Loading {lang} epoch={epoch} from {ckpt_path}")
            model = load_checkpoint(STAGE1_CKPT, ckpt_path, device)
            data  = extract_hidden(model, tokenizer, pairs, device)
            diag  = compute_diagnostics(data)
            diag_by_epoch[epoch] = diag
            del model
            torch.cuda.empty_cache()

        for (ep_start, ep_end) in epoch_pair_list:
            d_start = diag_by_epoch.get(ep_start)
            d_end   = diag_by_epoch.get(ep_end)

            if d_start is None or d_end is None:
                print(f"  [H3] Skipping {lang} {ep_start}→{ep_end}: missing one or both epoch checkpoints")
                continue

            print(f"  [H3] {lang} epoch {ep_start}→{ep_end}:")
            # ── 4 diagnostics: compare "after" state at ep_start vs ep_end ──
            # (a) Euclidean distance (after)
            r_dist = stat_test(d_start["dist_after"], d_end["dist_after"],
                               f"{lang}:dist_after (ep{ep_start} vs ep{ep_end})")
            # (b) Cosine sim (after)
            r_cos  = stat_test(d_start["cos_after"],  d_end["cos_after"],
                               f"{lang}:cos_after (ep{ep_start} vs ep{ep_end})")
            # (c) Norm delta (measures magnitude growth)
            r_norm = stat_test(d_start["norm_delta"], d_end["norm_delta"],
                               f"{lang}:norm_delta (ep{ep_start} vs ep{ep_end})")
            # (d) Common-mode drift
            r_cm   = stat_test(d_start["common_mode"], d_end["common_mode"],
                               f"{lang}:common_mode (ep{ep_start} vs ep{ep_end})")

            for metric, r in [("dist_after", r_dist), ("cos_after", r_cos),
                               ("norm_delta", r_norm), ("common_mode", r_cm)]:
                all_rows.append({"language": lang, "metric": metric,
                                  "epoch_a": ep_start, "epoch_b": ep_end, **r})
                print(f"    {metric}: mean_ep{ep_start}={r['mean_a']:.4f}, mean_ep{ep_end}={r['mean_b']:.4f}, "
                      f"d={r['cohens_d']:.3f}, t_p={r['t_p']:.4f}, w_p={r['w_p']}")

    # ── write CSV ────────────────────────────────────────────────
    out_csv = os.path.join(_SCRIPT_DIR, "hindi_representation_drift.csv")
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[H3] Results written to: {out_csv}")

    # ── interpret ────────────────────────────────────────────────
    hi_dist = next((r for r in all_rows if r["language"] == "HI" and r["metric"] == "dist_after"), None)
    vi_dist = next((r for r in all_rows if r["language"] == "VI" and r["metric"] == "dist_after"), None)
    hi_cos  = next((r for r in all_rows if r["language"] == "HI" and r["metric"] == "cos_after"), None)
    vi_cos  = next((r for r in all_rows if r["language"] == "VI" and r["metric"] == "cos_after"), None)

    # For the printed verdict, use the symmetric comparison: HI 1→3 vs VI 1→3.
    # If HI 1→3 rows are absent (checkpoint missing), fall back to HI 1→4.
    hi_dist_13 = next((r for r in all_rows if r["language"] == "HI" and r["metric"] == "dist_after"
                        and str(r["epoch_a"]) == "1" and str(r["epoch_b"]) == "3"), None)
    hi_cos_13  = next((r for r in all_rows if r["language"] == "HI" and r["metric"] == "cos_after"
                        and str(r["epoch_a"]) == "1" and str(r["epoch_b"]) == "3"), None)
    # Fall back to 1→4 for printed summary if 1→3 not available
    hi_dist = hi_dist_13 or hi_dist
    hi_cos  = hi_cos_13  or hi_cos

    if hi_dist and vi_dist and hi_cos and vi_cos:
        span_label = "1→3" if hi_dist_13 else "1→4 (fallback: epoch-3 ckpt missing)"
        # Aggregation formula: sum of |Cohen's d| for dist_after and cos_after only.
        # norm_delta and common_mode are reported separately but excluded from the
        # aggregate to avoid conflating magnitude-drift (dist) with direction-drift (cm).
        hi_d_eff = abs(float(hi_dist["cohens_d"])) + abs(float(hi_cos["cohens_d"]))
        vi_d_eff = abs(float(vi_dist["cohens_d"])) + abs(float(vi_cos["cohens_d"]))
        if hi_d_eff > vi_d_eff * 1.2:
            print(f"\n[H3 VERDICT] SUPPORTS H3 (HI {span_label} vs VI 1→3): "
                  f"HI effect size |d|={hi_d_eff:.3f} > VI |d|={vi_d_eff:.3f} "
                  f"[formula: |d_dist_after| + |d_cos_after|]")
        else:
            print(f"\n[H3 VERDICT] DOES NOT SUPPORT H3 (HI {span_label} vs VI 1→3): "
                  f"HI drift (|d|={hi_d_eff:.3f}) not substantially > VI (|d|={vi_d_eff:.3f}) "
                  f"[formula: |d_dist_after| + |d_cos_after|]")


if __name__ == "__main__":
    main()
