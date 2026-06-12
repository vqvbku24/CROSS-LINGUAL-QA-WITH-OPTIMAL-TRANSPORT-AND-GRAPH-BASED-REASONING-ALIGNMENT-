# train_stage2.py
"""
Stage 2 Training Loop: Teacher-Student Sinkhorn Alignment.

Aligns VI embedding space to EN using XQuAD parallel data.
No VI ground-truth labels are used. QA head is adapted to VI via
pseudo-labels derived from the Sinkhorn transport plan γ.

Key invariants:
  - EN backbone: always frozen (no_grad) throughout Stage 2.
  - ViQuAD: never used for training — evaluation only.
  - XQuAD VI val split (15%): never used for training — early stopping only.
  - Stage 1 checkpoint: loaded read-only; never overwritten.

Usage:
  python train_stage2.py --stage1_ckpt checkpoint/best.pt
  python train_stage2.py --stage1_ckpt checkpoint/best.pt --batch_size 8 --max_epochs 5
"""

import os
import sys
import math
import argparse
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Default Config
# ──────────────────────────────────────────────────────────────

STAGE2_CONFIG = {
    "stage1_ckpt"     : "checkpoint/best.pt",
    "model_name"      : "xlm-roberta-base",

    # Loss weights
    "lambda_ot"       : 1.0,
    "lambda_span"     : 1.0,
    "lambda_cons"     : 0.5,

    # OT hyperparameters
    "epsilon"         : 0.1,        # Sinkhorn regularization
    "sinkhorn_iters"  : 50,

    # Optimizer
    "stage2_head_lr"  : 5e-5,       # QA head + layer_weights
    "weight_decay"    : 0.01,
    "warmup_ratio"    : 0.06,

    # Training
    "batch_size"      : 16,
    "max_epochs"      : 10,
    "max_grad_norm"   : 1.0,
    "max_length"      : 384,

    # Early stopping
    "patience"        : 3,
    "min_delta_em"    : 0.5,        # minimum EM improvement to reset patience
    "en_em_safety"    : 5.0,        # hard stop if EN EM drops more than this

    # Curriculum (in steps, computed relative to steps_per_epoch)
    # CONS_DELAY  = steps_per_epoch // 2   (L_cons starts at epoch 0.5)
    # CONS_WARMUP = steps_per_epoch        (ramps over 1 full epoch)

    # Logging
    "log_every"       : 50,
    "save_every"      : 1,          # save checkpoint every N epochs

    # Paths
    "root_dir"        : os.path.dirname(os.path.abspath(__file__)),
    "output_dir"      : os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint_stage2"),
}


# ──────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────

def load_stage1_checkpoint(ckpt_path: str, model, criterion, device: torch.device):
    """
    Load Stage 1 checkpoint into model and criterion.
    Checkpoint is read-only — Stage 1 files are never overwritten.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {ckpt_path}")

    log.info(f"Loading Stage 1 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    criterion.load_state_dict(ckpt["criterion_state"])
    log.info("  Stage 1 weights loaded (model + criterion/QA head)")

    en_em_baseline = ckpt.get("em", None)
    if en_em_baseline is not None:
        log.info(f"  Stage 1 EN EM (from checkpoint): {en_em_baseline:.2f}%")

    return en_em_baseline


def freeze_en_backbone(model):
    """Freeze EN backbone: disable gradients and set eval mode."""
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    model.backbone.eval()
    log.info("EN backbone frozen (requires_grad=False, eval mode)")


def save_stage2_checkpoint(path: str, epoch: int, global_step: int,
                            model, criterion, optimizer, scheduler,
                            config: dict, vi_em: float):
    torch.save({
        "epoch"           : epoch,
        "global_step"     : global_step,
        "model_state"     : model.state_dict(),
        "criterion_state" : criterion.state_dict(),
        "optimizer_state" : optimizer.state_dict(),
        "scheduler_state" : scheduler.state_dict() if scheduler else None,
        "config"          : config,
        "vi_em"           : vi_em,
    }, path)
    log.info(f"  Checkpoint saved: {path}")


# ──────────────────────────────────────────────────────────────
# Baseline EN EM (computed once before Stage 2 starts)
# ──────────────────────────────────────────────────────────────

def compute_en_em_baseline(model, criterion, tokenizer, config: dict, device: torch.device) -> float:
    """Evaluate EN EM on 200 SQuAD dev samples using Stage 1 checkpoint weights."""
    import importlib.util
    eval_file = os.path.join(config["root_dir"], "phase4-evaluation", "quick_eval.py")
    spec = importlib.util.spec_from_file_location("quick_eval", eval_file)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    dev_file = os.path.join(config["root_dir"], "dataset", "Squad2.0", "dev-v2.0.json")
    if not os.path.exists(dev_file):
        log.warning(f"SQuAD dev not found at {dev_file} — EN EM safety check disabled")
        return float("inf")  # disable safety check

    em = mod.quick_em(model, criterion, tokenizer, dev_file, n_samples=200, device=device)
    log.info(f"Stage 1 EN EM baseline (200 samples): {em:.2f}%")
    return em


# ──────────────────────────────────────────────────────────────
# Two-forward-pass batch step
# ──────────────────────────────────────────────────────────────

def stage2_step(
    batch: dict,
    model,
    criterion,
    stage2_loss,
    epsilon: float,
    n_iters: int,
    global_step: int,
    device: torch.device,
) -> dict:
    """
    One Stage 2 training step with two forward passes:
      1. EN branch (no_grad) → h_en, p_en_start, p_en_end
      2. VI branch (with grad) → h_vi, vi_logits_*

    Then: sinkhorn_masked → compute_span_loss + compute_cons_loss → Stage2Loss.

    Returns:
        dict with all loss tensors and debug info
    """
    from phase3_loss.losses import (
        sinkhorn_masked, compute_span_loss, compute_cons_loss, gamma_entropy,
        _extract_question_embeddings,
    )

    # ── 1. EN branch — no gradient ──────────────────────────────
    with torch.no_grad():
        en_out = model(batch, branch="en")
        h_en      = en_out["hidden"]       # (B, T_en, H)
        en_mask   = ~en_out["en_pad_mask"] # True = real token

        # QA head on EN to get soft pseudo-label distributions
        en_q_emb, en_q_mask = _extract_question_embeddings(
            h_en, batch["en_question_end"]
        )
        en_start_logits, en_end_logits, _ = criterion.qa_head(
            h_en, en_q_emb, en_q_mask
        )
        p_en_start = F.softmax(en_start_logits, dim=-1)  # (B, T_en)
        p_en_end   = F.softmax(en_end_logits,   dim=-1)  # (B, T_en)

    # ── 2. VI branch — with gradient ────────────────────────────
    vi_out = model(batch, branch="vi")
    h_vi      = vi_out["hidden"]           # (B, T_vi, H)
    vi_mask   = ~vi_out["vi_pad_mask"]     # True = real token

    vi_q_emb, vi_q_mask = _extract_question_embeddings(
        h_vi, batch["vi_question_end"]
    )
    vi_start_logits, vi_end_logits, _ = criterion.qa_head(
        h_vi, vi_q_emb, vi_q_mask
    )

    # ── 3. Sinkhorn OT ──────────────────────────────────────────
    gamma_list, L_ot = sinkhorn_masked(
        h_en, h_vi, en_mask, vi_mask,
        epsilon=epsilon, n_iters=n_iters,
    )

    # ── 4. Span loss (KL pseudo-label) ──────────────────────────
    L_span = compute_span_loss(
        gamma_list, p_en_start, p_en_end,
        vi_start_logits, vi_end_logits,
        en_mask, vi_mask,
    )

    # ── 5. Consistency loss (feature MSE) ───────────────────────
    L_cons = compute_cons_loss(gamma_list, h_en, h_vi, en_mask, vi_mask)

    # ── 6. Combine losses with curriculum ───────────────────────
    losses = stage2_loss(L_ot, L_span, L_cons, global_step)

    # ── 7. Debug metrics ────────────────────────────────────────
    with torch.no_grad():
        g_entropy = gamma_entropy(gamma_list)
        if g_entropy > 6.0:
            log.warning(f"  [Gamma] High entropy={g_entropy:.2f} — transport plan approaching uniform")
        elif g_entropy < 0.5:
            log.warning(f"  [Gamma] Low entropy={g_entropy:.2f} — transport plan may be collapsed")

    losses["gamma_entropy"] = g_entropy
    return losses


# ──────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────

def run_stage2(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info("=" * 60)
    log.info("STAGE 2: Teacher-Student Sinkhorn Alignment")
    log.info("=" * 60)

    os.makedirs(config["output_dir"], exist_ok=True)

    # ── Load model and criterion ─────────────────────────────────
    from phase2_model.model_core import CrossLingualOTModel
    from phase3_loss.losses import OTAlignmentLoss, Stage2Loss

    model = CrossLingualOTModel(model_name=config["model_name"]).to(device)
    criterion = OTAlignmentLoss(
        hidden_size=model.backbone.hidden_size,
    ).to(device)

    # ── Load Stage 1 checkpoint (read-only) ──────────────────────
    ckpt_path = config["stage1_ckpt"]
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(config["root_dir"], ckpt_path)

    en_em_baseline = load_stage1_checkpoint(ckpt_path, model, criterion, device)

    # ── Freeze EN backbone immediately ──────────────────────────
    freeze_en_backbone(model)

    # ── Tokenizer ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)

    # ── Compute EN EM baseline (if not in checkpoint) ───────────
    if en_em_baseline is None:
        en_em_baseline = compute_en_em_baseline(model, criterion, tokenizer, config, device)

    # ── XQuAD dataloaders ────────────────────────────────────────
    from data.xquad_loader import create_xquad_dataloaders
    train_loader, val_loader, val_pairs = create_xquad_dataloaders(
        root_dir=config["root_dir"],
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["max_length"],
    )
    log.info(f"XQuAD: {len(train_loader)} train batches | {len(val_pairs)} val pairs")

    # ── Optimizer — differential learning rates ──────────────────
    # backbone: lr=0.0 (frozen)
    # layer_weights + QA head: stage2_head_lr
    optimizer = AdamW([
        {"params": list(model.backbone.parameters()),  "lr": 0.0},  # frozen
        {"params": [model.layer_weights],              "lr": config["stage2_head_lr"]},
        {"params": list(criterion.parameters()),       "lr": config["stage2_head_lr"]},
    ], weight_decay=config["weight_decay"])

    # ── Scheduler ────────────────────────────────────────────────
    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * config["max_epochs"]
    warmup_steps    = int(total_steps * config["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    log.info(f"Scheduler: linear warmup {warmup_steps}/{total_steps} steps")

    # ── Curriculum delays ────────────────────────────────────────
    CONS_DELAY  = steps_per_epoch // 2   # L_cons starts after 50% of epoch 1
    CONS_WARMUP = steps_per_epoch        # ramps over 1 full epoch

    stage2_loss = Stage2Loss(
        lambda_ot   = config["lambda_ot"],
        lambda_span = config["lambda_span"],
        lambda_cons = config["lambda_cons"],
        cons_delay  = CONS_DELAY,
        cons_warmup = CONS_WARMUP,
    ).to(device)

    log.info(
        f"Curriculum: CONS_DELAY={CONS_DELAY} steps | CONS_WARMUP={CONS_WARMUP} steps"
    )

    # ── TensorBoard ─────────────────────────────────────────────
    tb_dir = os.path.join(config["output_dir"], "tensorboard_stage2")
    writer = SummaryWriter(log_dir=tb_dir)
    log.info(f"TensorBoard: {tb_dir}")

    # ── Early stopping state ─────────────────────────────────────
    best_vi_em     = 0.0
    patience_count = 0
    global_step    = 0

    # ── Training epochs ─────────────────────────────────────────
    for epoch in range(1, config["max_epochs"] + 1):

        # EN backbone must stay in eval mode even after model.train()
        model.train()
        model.backbone.eval()   # ← spec requirement: called every epoch
        criterion.train()

        log.info(f"{'━'*60}")
        log.info(f"Epoch {epoch}/{config['max_epochs']}")

        epoch_losses = {"total": 0.0, "ot": 0.0, "span": 0.0, "cons": 0.0}
        step_count   = 0

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad()

            losses = stage2_step(
                batch, model, criterion, stage2_loss,
                epsilon=config["epsilon"],
                n_iters=config["sinkhorn_iters"],
                global_step=global_step,
                device=device,
            )

            losses["total"].backward()

            # Clip gradients (backbone grads are zero — clipping them is harmless)
            trainable_params = [model.layer_weights] + list(criterion.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, config["max_grad_norm"])

            optimizer.step()
            scheduler.step()
            global_step += 1
            step_count  += 1

            for k in ("total", "ot", "span", "cons"):
                v = losses.get(k)
                if isinstance(v, torch.Tensor):
                    epoch_losses[k] += v.item()
                elif isinstance(v, float):
                    epoch_losses[k] += v

            # ── Per-step TensorBoard logging ─────────────────────
            if global_step % config["log_every"] == 0:
                w_cons = losses.get("cons_weight", torch.tensor(0.0))
                w_cons_val = w_cons.item() if isinstance(w_cons, torch.Tensor) else w_cons
                g_ent  = losses.get("gamma_entropy", 0.0)

                log.info(
                    f"  Step {global_step} | "
                    f"total={losses['total'].item():.4f} | "
                    f"ot={losses['ot'].item():.4f} | "
                    f"span={losses['span'].item():.4f} | "
                    f"cons={losses['cons'].item():.4f} | "
                    f"w_cons={w_cons_val:.3f} | "
                    f"γ_H={g_ent:.2f}"
                )

                writer.add_scalar("Loss/Stage2_Total", losses["total"].item(), global_step)
                writer.add_scalar("Loss/OT",           losses["ot"].item(),    global_step)
                writer.add_scalar("Loss/Span",         losses["span"].item(),  global_step)
                writer.add_scalar("Loss/Cons",         losses["cons"].item(),  global_step)
                writer.add_scalar("Lambda/Cons_Weight", w_cons_val,            global_step)
                writer.add_scalar("Debug/Gamma_Entropy", g_ent,                global_step)
                writer.add_scalar("Learning_Rate/Head",
                                  optimizer.param_groups[1]["lr"],             global_step)

        # ── End of epoch summary ─────────────────────────────────
        avg = {k: v / max(step_count, 1) for k, v in epoch_losses.items()}
        log.info(
            f"Epoch {epoch} avg | total={avg['total']:.4f} | "
            f"ot={avg['ot']:.4f} | span={avg['span']:.4f} | cons={avg['cons']:.4f}"
        )

        # ── Evaluation ───────────────────────────────────────────
        import importlib.util
        eval_file = os.path.join(config["root_dir"], "phase4-evaluation", "quick_eval.py")
        spec = importlib.util.spec_from_file_location("quick_eval", eval_file)
        quick_eval_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(quick_eval_mod)

        # VI EM on XQuAD val
        vi_em = quick_eval_mod.quick_em_xquad_vi(
            model, criterion, tokenizer, val_pairs, device,
            max_length=config["max_length"],
        )
        log.info(f"Epoch {epoch} XQuAD VI EM: {vi_em:.2f}%")
        writer.add_scalar("Eval/XQuAD_VI_EM", vi_em, epoch)

        # EN EM regression check (200 SQuAD samples)
        dev_file = os.path.join(config["root_dir"], "dataset", "Squad2.0", "dev-v2.0.json")
        if os.path.exists(dev_file):
            en_em = quick_eval_mod.quick_em(
                model, criterion, tokenizer, dev_file, n_samples=200, device=device,
            )
            log.info(f"Epoch {epoch} SQuAD EN EM (200): {en_em:.2f}% (baseline={en_em_baseline:.2f}%)")
            writer.add_scalar("Eval/SQuAD_EN_EM_Quick", en_em, epoch)

            drop = en_em_baseline - en_em
            if drop > config["en_em_safety"]:
                log.warning(
                    f"EN EM dropped {drop:.1f} pts (>{config['en_em_safety']}) — hard stop!"
                )
                break

        # ── Checkpoint saving ─────────────────────────────────────
        if epoch % config["save_every"] == 0:
            ckpt_out = os.path.join(config["output_dir"], f"stage2_epoch_{epoch:03d}.pt")
            save_stage2_checkpoint(
                ckpt_out, epoch, global_step,
                model, criterion, optimizer, scheduler,
                config, vi_em,
            )

        # ── Early stopping ────────────────────────────────────────
        if vi_em > best_vi_em + config["min_delta_em"]:
            best_vi_em     = vi_em
            patience_count = 0
            best_path = os.path.join(config["output_dir"], "stage2_best.pt")
            save_stage2_checkpoint(
                best_path, epoch, global_step,
                model, criterion, optimizer, scheduler,
                config, vi_em,
            )
            log.info(f"  ★ New best VI EM={vi_em:.2f}% — saved {best_path}")
        else:
            patience_count += 1
            log.info(
                f"  No improvement. Patience {patience_count}/{config['patience']}"
            )
            if patience_count >= config["patience"]:
                log.info(
                    f"Early stopping at epoch {epoch} — best VI EM={best_vi_em:.2f}%"
                )
                break

    writer.close()
    log.info("=" * 60)
    log.info(f"Stage 2 complete. Best VI EM: {best_vi_em:.2f}%")
    log.info(f"Best checkpoint: {os.path.join(config['output_dir'], 'stage2_best.pt')}")
    log.info("=" * 60)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Stage 2: Teacher-Student Sinkhorn Alignment")
    parser.add_argument("--stage1_ckpt",    default=STAGE2_CONFIG["stage1_ckpt"],
                        help="Path to Stage 1 checkpoint (default: checkpoint/best.pt)")
    parser.add_argument("--model_name",     default=STAGE2_CONFIG["model_name"])
    parser.add_argument("--batch_size",     type=int,   default=STAGE2_CONFIG["batch_size"])
    parser.add_argument("--max_epochs",     type=int,   default=STAGE2_CONFIG["max_epochs"])
    parser.add_argument("--stage2_head_lr", type=float, default=STAGE2_CONFIG["stage2_head_lr"])
    parser.add_argument("--lambda_ot",      type=float, default=STAGE2_CONFIG["lambda_ot"])
    parser.add_argument("--lambda_span",    type=float, default=STAGE2_CONFIG["lambda_span"])
    parser.add_argument("--lambda_cons",    type=float, default=STAGE2_CONFIG["lambda_cons"])
    parser.add_argument("--epsilon",        type=float, default=STAGE2_CONFIG["epsilon"])
    parser.add_argument("--sinkhorn_iters", type=int,   default=STAGE2_CONFIG["sinkhorn_iters"])
    parser.add_argument("--patience",       type=int,   default=STAGE2_CONFIG["patience"])
    parser.add_argument("--max_length",     type=int,   default=STAGE2_CONFIG["max_length"])
    parser.add_argument("--output_dir",     default=STAGE2_CONFIG["output_dir"])
    parser.add_argument("--log_every",      type=int,   default=STAGE2_CONFIG["log_every"])
    args = parser.parse_args()

    config = {**STAGE2_CONFIG, **vars(args)}
    return config


if __name__ == "__main__":
    config = parse_args()

    log.info("Stage 2 config:")
    for k, v in config.items():
        if k != "root_dir":
            log.info(f"  {k:20s}: {v}")

    run_stage2(config)
