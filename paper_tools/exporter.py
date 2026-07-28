"""
exporter.py — Extract intermediate representations for paper figures.

Produces files consumed by the four visualize_*.py scripts:
    visualize_ot.py           → gamma.npy, english_tokens.txt, vietnamese_tokens.txt
    visualize_representation.py → hidden_before.npy, hidden_after.npy, gamma_fig5.npy,
                                   langs.npy, labels.npy
    visualize_layer.py        → layer_weights.npy  (margin_history.csv: from training logs)
    visualize_ablation.py     → ablation.csv       (must be created manually from eval results)

Usage:
    python paper_tools/exporter.py \
        --checkpoint checkpoints/stage1_squad_best.pt \
        --dataset    dataset/xquad.vi.json \
        --output_dir paper_tools/export \
        --sample_index 0 \
        --alpha 0.5

For a Stage 2 (LoRA) checkpoint, you must also point at the Stage 1 base
backbone it was fine-tuned from — otherwise the backbone silently loads with
random weights and every downstream number is meaningless:
    python paper_tools/exporter.py \
        --checkpoint checkpoint_stage2/stage2_epoch_003.pt \
        --stage1_checkpoint checkpoints/stage1_squad_best.pt \
        --dataset    dataset/xquad.vi.json \
        --output_dir paper_tools/export \
        --sample_index 0 \
        --alpha 0.5
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Project root on sys.path so phase2_model, phase3_loss, gpu_utils are importable
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from phase2_model.model_core import CrossLingualOTModel
from phase3_loss.losses import OTAlignmentLoss, sinkhorn_masked, _extract_question_embeddings
from gpu_utils import get_model

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract intermediate representations for paper figures"
    )
    parser.add_argument("--checkpoint",    type=str, required=True,
                        help="Path to checkpoint (Stage 1 or Stage 2)")
    parser.add_argument("--dataset",       type=str, required=True,
                        help="Path to XQuAD validation dataset")
    parser.add_argument("--output_dir",    type=str, required=True,
                        help="Directory to save extracted .npy / .txt / .json files")
    parser.add_argument("--sample_index",  type=int, default=0,
                        help="Which sample to use (0-indexed)")
    parser.add_argument("--alpha",         type=float, default=0.5,
                        help="EMA factor for gamma mix: alpha*teacher + (1-alpha)*student")
    parser.add_argument("--stage",         type=str, default="auto",
                        choices=["1", "2", "auto"],
                        help="Checkpoint stage. 'auto' = has LoRA if 'config' key present")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Explicit path to the Stage 1 (full backbone) checkpoint. "
                             "Overrides config['stage1_ckpt'] from the Stage 2 checkpoint, "
                             "which is often a stale/relative path from a different machine. "
                             "Required (one way or another) whenever the checkpoint is Stage 2 — "
                             "without it the backbone silently falls back to random init.")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _save_npy(path, tensor_or_array):
    arr = tensor_or_array
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    np.save(path, arr)


def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def _is_stage2_checkpoint(ckpt: dict) -> bool:
    """Heuristic: Stage 2 checkpoint contains 'config' key and LoRA param names."""
    if "config" not in ckpt:
        return False
    model_state = ckpt.get("model_state", {})
    return any("lora" in k for k in model_state.keys())


def load_sample(dataset_path, tokenizer, sample_index):
    """
    Load a single (EN, VI) pair from XQuAD using the project's xquad_loader.
    Falls back to a small random dummy batch if the loader fails.
    """
    try:
        from data.xquad_loader import create_xquad_dataloaders
        loader, _, val_pairs = create_xquad_dataloaders(
            root_dir=str(Path(__file__).parent.parent),
            tokenizer=tokenizer,
            batch_size=1,
            max_length=384,
        )
        for i, batch in enumerate(loader):
            if i == sample_index:
                pair_info = val_pairs[sample_index] if sample_index < len(val_pairs) else {}
                return batch, pair_info
        raise ValueError(f"sample_index {sample_index} out of range ({len(loader)} batches)")
    except Exception as e:
        print(f"[warn] xquad_loader failed ({e}). Using dummy batch.")
        batch = {
            "en_input_ids":       torch.randint(0, 1000, (1, 64)),
            "en_attention_mask":  torch.ones((1, 64), dtype=torch.long),
            "vi_input_ids":       torch.randint(0, 1000, (1, 72)),
            "vi_attention_mask":  torch.ones((1, 72), dtype=torch.long),
            "en_question_end":    torch.tensor([12]),
            "vi_question_end":    torch.tensor([14]),
            "en_start_position":  torch.tensor([25]),
            "en_end_position":    torch.tensor([28]),
            "en_is_answerable":   torch.tensor([1]),
        }
        pair_info = {"question": "Dummy?", "answer": "Dummy"}
        return batch, pair_info


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load checkpoint ───────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config", {})

    model_name    = config.get("model_name", "xlm-roberta-base")
    epsilon       = config.get("epsilon",       0.03)
    sinkhorn_iters = config.get("sinkhorn_iters", 100)

    # Determine stage
    if args.stage == "auto":
        is_stage2 = _is_stage2_checkpoint(ckpt)
    else:
        is_stage2 = (args.stage == "2")
    print(f"Stage: {'2 (LoRA)' if is_stage2 else '1 (full backbone)'}")

    # ── 2. Build model + criterion ───────────────────────────
    # CrossLingualOTModel constructor: only model_name and compute_cost_matrix
    model     = CrossLingualOTModel(model_name=model_name, compute_cost_matrix=False)
    criterion = OTAlignmentLoss(hidden_size=model.hidden_size)

    if is_stage2:
        # Stage 2: load Stage 1 weights first (full backbone), THEN apply LoRA,
        # THEN load Stage 2 delta (LoRA + layer_weights only).
        # Prefer an explicit CLI override — config['stage1_ckpt'] is often a path
        # baked in at training time on a different machine, so it silently fails
        # to resolve here and previously just fell back to a random backbone.
        stage1_path = args.stage1_checkpoint or config.get("stage1_ckpt", "")
        if stage1_path and not os.path.isabs(stage1_path):
            stage1_path = os.path.join(str(Path(__file__).parent.parent), stage1_path)

        if stage1_path and os.path.exists(stage1_path):
            print(f"Loading Stage 1 base weights from {stage1_path}")
            s1 = torch.load(stage1_path, map_location=device)
            model.load_state_dict(s1["model_state"], strict=True)
            criterion.load_state_dict(s1["criterion_state"])
        else:
            # This used to be a soft [warn] that let the run continue on a
            # randomly-initialized backbone — every downstream number (gamma,
            # hidden states, alignment metrics) would then be meaningless.
            # Fail loudly instead.
            raise FileNotFoundError(
                f"Stage 2 checkpoint requires a Stage 1 base backbone, but none was "
                f"found. config['stage1_ckpt']={config.get('stage1_ckpt', '<missing>')!r} "
                f"did not resolve to an existing file, and --stage1_checkpoint was not "
                f"passed. Pass it explicitly, e.g.:\n"
                f"  --stage1_checkpoint checkpoints/stage1_squad_best.pt"
            )

        print("Applying LoRA...")
        model.apply_lora()
        model.load_state_dict(ckpt["model_state"], strict=False)   # LoRA delta + layer_weights
        criterion.load_state_dict(ckpt["criterion_state"])
    else:
        # Stage 1: full state dict — no LoRA
        model.load_state_dict(ckpt["model_state"], strict=True)
        criterion.load_state_dict(ckpt["criterion_state"])

    model.to(device).eval()
    criterion.to(device).eval()

    # ── 3. Tokenizer + sample ────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print(f"Loading sample {args.sample_index} from {args.dataset}")
    batch, pair_info = load_sample(args.dataset, tokenizer, args.sample_index)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # ── 4. Inference ─────────────────────────────────────────
    print("Running inference...")
    with torch.no_grad():

        # ── 4a. Frozen EN (no LoRA) — teacher anchor ────────
        if is_stage2:
            with get_model(model).backbone.disable_adapter():
                en_frz_out = model(batch, branch="en")
                h_en_frz   = en_frz_out["hidden"]           # (1, T_en, H)
                en_mask    = ~en_frz_out["en_pad_mask"]     # (1, T_en) True=real

                vi_frz_out = model(batch, branch="vi")
                h_vi_frz   = vi_frz_out["hidden"]           # (1, T_vi, H)
                vi_mask_frz = ~vi_frz_out["vi_pad_mask"]    # (1, T_vi)

                en_q_emb, en_q_mask = _extract_question_embeddings(
                    h_en_frz, batch["en_question_end"]
                )
                teacher_start_logits, teacher_end_logits, _ = criterion.qa_head(
                    h_en_frz, en_q_emb, en_q_mask
                )
        else:
            # Stage 1: no LoRA wrapper — "frozen" is just the EN branch
            en_frz_out = model(batch, branch="en")
            h_en_frz   = en_frz_out["hidden"]
            en_mask    = ~en_frz_out["en_pad_mask"]
            h_vi_frz   = None
            vi_mask_frz = None

            en_q_emb, en_q_mask = _extract_question_embeddings(
                h_en_frz, batch["en_question_end"]
            )
            teacher_start_logits, teacher_end_logits, _ = criterion.qa_head(
                h_en_frz, en_q_emb, en_q_mask
            )

        # ── 4b. EN LoRA (or Stage-1 pass) ───────────────────
        en_lora_out = model(batch, branch="en")
        h_en_lora   = en_lora_out["hidden"]                  # (1, T_en, H) — has LoRA grad path

        en_q_emb_l, en_q_mask_l = _extract_question_embeddings(
            h_en_lora, batch["en_question_end"]
        )
        en_lora_start_logits, en_lora_end_logits, _ = criterion.qa_head(
            h_en_lora, en_q_emb_l, en_q_mask_l
        )

        # ── 4c. VI LoRA ──────────────────────────────────────
        vi_out  = model(batch, branch="vi")
        h_vi    = vi_out["hidden"]                           # (1, T_vi, H)
        vi_mask = ~vi_out["vi_pad_mask"]                     # (1, T_vi) True=real

        vi_q_emb, vi_q_mask = _extract_question_embeddings(
            h_vi, batch["vi_question_end"]
        )
        vi_start_logits, vi_end_logits, _ = criterion.qa_head(
            h_vi, vi_q_emb, vi_q_mask
        )

        # ── 4d. Sinkhorn OT ──────────────────────────────────
        if is_stage2 and h_vi_frz is not None:
            # Teacher: frozen EN vs frozen VI
            gamma_teacher, _ = sinkhorn_masked(
                h_en_frz, h_vi_frz, en_mask, vi_mask_frz,
                epsilon=epsilon, n_iters=sinkhorn_iters,
            )
            # Student: frozen EN vs trainable VI
            gamma_student, _ = sinkhorn_masked(
                h_en_frz, h_vi, en_mask, vi_mask,
                epsilon=epsilon, n_iters=sinkhorn_iters,
            )
            alpha = args.alpha
            g_teacher = gamma_teacher[0]   # Tensor[n_en, n_vi]
            g_student = gamma_student[0]   # Tensor[n_en, n_vi]
            g_mix     = alpha * g_teacher + (1.0 - alpha) * g_student
        else:
            # Stage 1: no VI teacher — compute single gamma between EN and VI
            gamma_list, _ = sinkhorn_masked(
                h_en_frz, h_vi, en_mask, vi_mask,
                epsilon=epsilon, n_iters=sinkhorn_iters,
            )
            g_teacher = gamma_list[0]
            g_student = gamma_list[0]
            g_mix     = gamma_list[0]

        # ── 4e. Per-layer hidden states (direct backbone call) ─
        # get_model() handles both DDP and non-DDP wrappers
        backbone = get_model(model).backbone
        out_en_layers = backbone(
            batch["en_input_ids"], batch["en_attention_mask"]
        )
        out_vi_layers = backbone(
            batch["vi_input_ids"], batch["vi_attention_mask"]
        )

    # ── 5. Sequence lengths (valid tokens only) ──────────────
    en_seq_len = int(en_mask[0].sum().item())
    vi_seq_len = int(vi_mask[0].sum().item())
    print(f"Sequence lengths — EN: {en_seq_len}, VI: {vi_seq_len}")

    # ── 6. Save gamma files ──────────────────────────────────
    print("Saving gamma files...")
    _save_npy(output_dir / "gamma.npy",         g_mix[:en_seq_len, :vi_seq_len])
    _save_npy(output_dir / "gamma_teacher.npy", g_teacher[:en_seq_len, :vi_seq_len])
    _save_npy(output_dir / "gamma_student.npy", g_student[:en_seq_len, :vi_seq_len])
    _save_npy(output_dir / "gamma_fig5.npy",    g_mix[:en_seq_len, :vi_seq_len])

    # ── 7. Save hidden states ─────────────────────────────────
    print("Saving hidden states...")

    # For visualize_representation.py:
    #   hidden_before.npy = [EN_frz | VI_frz] stacked — "before alignment"
    #   hidden_after.npy  = [EN_lora | VI_lora] stacked — "after alignment"
    h_en_before = h_en_frz[0, :en_seq_len].cpu().numpy()    # (T_en, H)
    h_vi_before = (h_vi_frz[0, :vi_seq_len].cpu().numpy()
                   if h_vi_frz is not None
                   else h_vi[0, :vi_seq_len].cpu().numpy())  # (T_vi, H)
    h_en_after  = h_en_lora[0, :en_seq_len].cpu().numpy()   # (T_en, H)
    h_vi_after  = h_vi[0, :vi_seq_len].cpu().numpy()        # (T_vi, H)

    hidden_before = np.vstack([h_en_before, h_vi_before])   # (T_en+T_vi, H)
    hidden_after  = np.vstack([h_en_after,  h_vi_after])    # (T_en+T_vi, H)

    np.save(output_dir / "hidden_before.npy", hidden_before)
    np.save(output_dir / "hidden_after.npy",  hidden_after)

    # Extra: individual branches (for ad-hoc analysis)
    _save_npy(output_dir / "hidden_en_frz.npy",  h_en_frz[0, :en_seq_len])
    _save_npy(output_dir / "hidden_en_lora.npy", h_en_lora[0, :en_seq_len])
    _save_npy(output_dir / "hidden_vi.npy",      h_vi[0, :vi_seq_len])

    # Per-layer hidden states (layers 6, 7, 8, 9)
    target_layers = [6, 7, 8, 9]
    for idx in target_layers:
        _save_npy(
            output_dir / f"hidden_en_layer{idx}.npy",
            out_en_layers.hidden_states[idx][0, :en_seq_len],
        )
        _save_npy(
            output_dir / f"hidden_vi_layer{idx}.npy",
            out_vi_layers.hidden_states[idx][0, :vi_seq_len],
        )

    # ── 8. langs / labels metadata for visualize_representation.py ──
    langs  = ["EN"] * en_seq_len + ["VI"] * vi_seq_len
    labels = ["normal"] * (en_seq_len + vi_seq_len)

    # Mark answer tokens if we have span positions
    en_start = int(batch.get("en_start_positions", torch.tensor([0]))[0].item())
    en_end   = int(batch.get("en_end_positions",   torch.tensor([0]))[0].item())
    if en_start < en_seq_len and en_end < en_seq_len and en_start <= en_end:
        for i in range(en_start, en_end + 1):
            labels[i] = "answer"

    np.save(output_dir / "langs.npy",  np.array(langs))
    np.save(output_dir / "labels.npy", np.array(labels))

    # ── 9. Layer weights (for visualize_layer.py) ────────────
    print("Saving layer weights...")
    lw = get_model(model).layer_weights.detach().cpu()
    _save_npy(output_dir / "layer_weights.npy", lw)
    print(f"  layer_weights (raw):    {lw.tolist()}")
    print(f"  layer_weights (softmax): {F.softmax(lw, dim=0).tolist()}")

    # ── 10. Logits ────────────────────────────────────────────
    print("Saving logits...")
    _save_npy(output_dir / "teacher_start_logits.npy", teacher_start_logits[0, :en_seq_len])
    _save_npy(output_dir / "teacher_end_logits.npy",   teacher_end_logits[0, :en_seq_len])
    _save_npy(output_dir / "en_lora_start_logits.npy", en_lora_start_logits[0, :en_seq_len])
    _save_npy(output_dir / "en_lora_end_logits.npy",   en_lora_end_logits[0, :en_seq_len])
    _save_npy(output_dir / "vi_start_logits.npy",      vi_start_logits[0, :vi_seq_len])
    _save_npy(output_dir / "vi_end_logits.npy",        vi_end_logits[0, :vi_seq_len])

    # ── 11. Tokens ────────────────────────────────────────────
    print("Saving tokens...")
    en_ids     = batch["en_input_ids"][0, :en_seq_len].tolist()
    vi_ids     = batch["vi_input_ids"][0, :vi_seq_len].tolist()
    en_tokens  = tokenizer.convert_ids_to_tokens(en_ids)
    vi_tokens  = tokenizer.convert_ids_to_tokens(vi_ids)

    (output_dir / "english_tokens.txt").write_text(
        "\n".join(en_tokens), encoding="utf-8"
    )
    (output_dir / "vietnamese_tokens.txt").write_text(
        "\n".join(vi_tokens), encoding="utf-8"
    )

    # ── 12. Margin info ───────────────────────────────────────
    margin_info = {
        "alpha":             args.alpha,
        "lambda_margin":     config.get("lambda_margin", 1.0),
        "en_top1_logit":     float(teacher_start_logits[0, :en_seq_len].max()),
        "vi_top1_logit":     float(vi_start_logits[0, :vi_seq_len].max()),
        "en_top1_end_logit": float(teacher_end_logits[0, :en_seq_len].max()),
        "vi_top1_end_logit": float(vi_end_logits[0, :vi_seq_len].max()),
    }
    _save_json(output_dir / "margin.json", margin_info)

    # ── 13. Metadata ──────────────────────────────────────────
    metadata = {
        "checkpoint":        args.checkpoint,
        "stage":             "2" if is_stage2 else "1",
        "sample_index":      args.sample_index,
        "dataset":           args.dataset,
        "question":          pair_info.get("question", ""),
        "answer":            pair_info.get("answer", ""),
        "en_seq_len":        en_seq_len,
        "vi_seq_len":        vi_seq_len,
        "alpha":             args.alpha,
        "epsilon":           epsilon,
        "sinkhorn_iters":    sinkhorn_iters,
    }
    _save_json(output_dir / "metadata.json", metadata)

    # ── 14. UMAP projections (optional) ──────────────────────
    if HAS_UMAP:
        print("UMAP found — saving projections...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        try:
            umap_en = reducer.fit_transform(h_en_before)
            umap_vi = reducer.fit_transform(h_vi_before)
            np.save(output_dir / "hidden_en_umap.npy", umap_en)
            np.save(output_dir / "hidden_vi_umap.npy", umap_vi)
        except Exception as e:
            print(f"[warn] UMAP failed: {e}")
    else:
        print("UMAP not installed — skipping projections.")

    # ── 15. Summary ───────────────────────────────────────────
    files = sorted(output_dir.iterdir())
    print(f"\nDone. {len(files)} files saved to {output_dir}:")
    for f in files:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()