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
    "lambda_ot"       : 10.0,
    "lambda_reg"      : 50.0,   # EN consistency regularisation (paper: 50, start lower for encoder)
    "lambda_span"     : 0.0,    # Disabled — gamma too uniform for reliable pseudo-labels
    "lambda_qa"       : 1.0,    # Supervised EN QA loss weight

    # OT hyperparameters
    "epsilon"         : 0.1,   # Restored to paper default (0.05 hurts XSQuAD per ablation)
    "epsilon_end"     : 0.03,
    "sinkhorn_iters"  : 150,

    # Optimizer
    "stage2_head_lr"  : 5e-5,       # QA head + layer_weights
    "weight_decay"    : 0.01,
    "warmup_ratio"    : 0.06,

    # Training
    "batch_size"      : 32,
    "max_epochs"      : 10,
    "max_grad_norm"   : 1,
    "max_length"      : 384,

    # Early stopping
    "patience"        : 3,
    "min_delta_em"    : 0.5,        # minimum EM improvement to reset patience
    "en_em_safety"    : 25.0,       # hard stop if EN EM drops more than this

    # Curriculum (in steps, computed relative to steps_per_epoch)

    # Logging
    "log_every"       : 50,
    "save_every"      : 1,          # save checkpoint every N epochs

    # Paths
    "root_dir"        : os.path.dirname(os.path.abspath(__file__)),
    "output_dir"      : os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint_stage2"),
    "hf_repo_id"      : "",
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

    # Khôi phục keys map 1-1, bỏ logic peft replace vì model chưa bọc LoRA
    model.load_state_dict(ckpt["model_state"], strict=False)
    criterion.load_state_dict(ckpt["criterion_state"])
    log.info("  Stage 1 weights loaded (model base + criterion/QA head)")

    en_em_baseline = ckpt.get("em", None)
    if en_em_baseline is not None:
        log.info(f"  Stage 1 EN EM (from checkpoint): {en_em_baseline:.2f}%")

    return en_em_baseline




def freeze_qa_head(criterion):
    """Freeze QA Head: disable gradients entirely."""
    for p in criterion.qa_head.parameters():
        p.requires_grad_(False)
    log.info("QA head frozen (requires_grad=False)")


def unfreeze_qa_head(criterion):
    """Unfreeze QA Head (for future use)."""
    for p in criterion.qa_head.parameters():
        p.requires_grad_(True)
    log.info("QA head unfrozen")


def save_stage2_checkpoint(path: str, epoch: int, global_step: int,
                            model, criterion, optimizer, scheduler,
                            config: dict, vi_em: float, best_vi_em: float,
                            patience_count: int):
    # Save only trainable parameters to save space (LoRA + layer_weights)
    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
    trainable_state_dict = {k: v for k, v in model.state_dict().items() if k in trainable_keys}

    torch.save({
        "epoch"           : epoch,
        "global_step"     : global_step,
        "model_state"     : trainable_state_dict,
        "criterion_state" : criterion.state_dict(),
        "optimizer_state" : optimizer.state_dict(),
        "scheduler_state" : scheduler.state_dict() if scheduler else None,
        "config"          : config,
        "vi_em"           : vi_em,
        "best_vi_em"      : best_vi_em,
        "patience_count"  : patience_count,
        "rng_state_cpu"   : torch.get_rng_state(),
        "rng_state_cuda"  : torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
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
    epoch: int,
    device: torch.device,
) -> dict:
    """
    One Stage 2 training step with THREE forward passes (per paper):
      1. EN branch — LoRA OFF, no_grad  → h_en_frz  (frozen anchor)
      2. EN branch — LoRA ON,  with_grad → h_en_lora (for L_Reg)
      3. VI branch — LoRA ON,  with_grad → h_vi

    Loss = λ_ot * L_ot + λ_reg * L_reg
    L_reg prevents LoRA shared weights from drifting EN representations.

    Returns:
        dict with all loss tensors and debug info
    """
    from phase3_loss.losses import (
        sinkhorn_masked, compute_span_loss,
        gamma_entropy, _extract_question_embeddings,
        qa_loss,
    )

    # ── 1. EN branch — LoRA OFF, no gradient (frozen anchor) ────
    with torch.no_grad():
        with model.backbone.disable_adapter():
            en_frz_out = model(batch, branch="en")
            h_en_frz   = en_frz_out["hidden"]        # (B, T_en, H) — detached anchor
            en_mask    = ~en_frz_out["en_pad_mask"]  # (B, T_en) True = real token

            # QA head on frozen EN → pseudo-label logits (for L_span if ever used)
            en_q_emb, en_q_mask = _extract_question_embeddings(
                h_en_frz, batch["en_question_end"]
            )
            en_start_logits, en_end_logits, _ = criterion.qa_head(
                h_en_frz, en_q_emb, en_q_mask
            )

    # ── 2. EN branch — LoRA ON, with gradient (for L_Reg, L_qa, L_has_ans) ───
    # NOTE: LoRA is ON here — gradient flows through LoRA weights via L_Reg, L_qa, L_has_ans
    en_lora_out = model(batch, branch="en")
    h_en_lora   = en_lora_out["hidden"]   # (B, T_en, H) — has gradient

    en_q_emb_lora, en_q_mask_lora = _extract_question_embeddings(
        h_en_lora, batch["en_question_end"]
    )
    en_lora_start_logits, en_lora_end_logits, en_lora_has_ans = criterion.qa_head(
        h_en_lora, en_q_emb_lora, en_q_mask_lora
    )

    # ── 3. VI branch — LoRA ON, with gradient ───────────────────
    vi_out    = model(batch, branch="vi")
    h_vi      = vi_out["hidden"]           # (B, T_vi, H)
    vi_mask   = ~vi_out["vi_pad_mask"]     # (B, T_vi) True = real token

    vi_q_emb, vi_q_mask = _extract_question_embeddings(
        h_vi, batch["vi_question_end"]
    )
    vi_start_logits, vi_end_logits, _ = criterion.qa_head(
        h_vi, vi_q_emb, vi_q_mask
    )

    # ── 4. Sinkhorn OT (uses frozen h_en_frz as anchor) ─────────
    gamma_list, L_ot = sinkhorn_masked(
        h_en_frz, h_vi, en_mask, vi_mask,
        epsilon=epsilon, n_iters=n_iters,
    )

    # ── 5. EN Consistency Regularisation (KEY NEW LOSS) ─────────
    # Calculate cosine similarity along the hidden dimension (dim=-1)
    cos_sim = torch.nn.functional.cosine_similarity(h_en_lora, h_en_frz, dim=-1) # (B, T_en)
    
    # Mask out PAD tokens so they don't dilute the regularisation
    valid_tokens_count = en_mask.float().sum().clamp(min=1.0)
    L_reg = 1.0 - (cos_sim * en_mask.float()).sum() / valid_tokens_count

    # ── 5b. Supervised EN QA & HasAnswer Loss ───────────────────
    en_seq_len = h_en_lora.size(1)
    en_start = batch["en_start_position"].clamp(max=en_seq_len - 1)
    en_end   = batch["en_end_position"].clamp(max=en_seq_len - 1)

    answerable_mask = batch["en_is_answerable"].bool().to(device)
    if answerable_mask.any():
        L_qa, _, _ = qa_loss(
            en_lora_start_logits[answerable_mask],
            en_lora_end_logits[answerable_mask],
            en_start[answerable_mask],
            en_end[answerable_mask],
        )
    else:
        L_qa = torch.tensor(0.0, device=device)

    has_answer_label = batch["en_is_answerable"].float().to(device)
    L_has_ans = F.binary_cross_entropy_with_logits(
        en_lora_has_ans,
        has_answer_label,
    )

    # ── 6. Span loss (disabled by default, kept for ablation) ────
    L_span = compute_span_loss(
        gamma_list, en_start_logits, en_end_logits,
        vi_start_logits, vi_end_logits,
        en_mask, vi_mask,
    )

    # ── 7. Combine losses ────────────────────────────────────────
    losses = stage2_loss(L_ot, L_reg, L_span, L_qa, L_has_ans, epoch)

    # ── 8. Debug metrics ─────────────────────────────────────────
    with torch.no_grad():
        g_entropy = gamma_entropy(gamma_list)
        import math
        avg_n_en = en_mask.sum(dim=1).float().mean().item()
        avg_n_vi = vi_mask.sum(dim=1).float().mean().item()
        h_max    = math.log(max(avg_n_en * avg_n_vi, 1.0))
        h_ratio  = g_entropy / h_max if h_max > 0 else 0

        if h_ratio > 0.90:
            log.warning(
                f"  [Gamma] entropy ratio={h_ratio:.2f} "
                f"(H={g_entropy:.2f}/H_max={h_max:.2f}) — near uniform"
            )
        elif h_ratio < 0.30:
            log.warning(f"  [Gamma] entropy ratio={h_ratio:.2f} — may be collapsed")
        else:
            log.info(f"  [Gamma] entropy ratio={h_ratio:.2f} H={g_entropy:.2f} — healthy")

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
        hidden_size=model.hidden_size,
    ).to(device)

    # ── 1. Load Stage 1 checkpoint (trước khi bọc LoRA, keys khớp 1-1) ───────────────
    ckpt_path = config["stage1_ckpt"]
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(config["root_dir"], ckpt_path)

    en_em_baseline = load_stage1_checkpoint(ckpt_path, model, criterion, device)

    # ── 2. Apply LoRA ───────────────────────────────────────────────
    log.info("Applying LoRA to backbone...")
    model.apply_lora()
    model.to(device) # Ensure new LoRA layers are on the target device

    # Disable dropout to prevent representation drift from random masks
    log.info("Disabling dropout in backbone, adapters & criterion to prevent L_reg variance...")
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    for m in criterion.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    # ── Verify QA Head Freeze State ─────────────────────────────
    if config.get("freeze_qa_head", False):
        freeze_qa_head(criterion)

    # ── Tokenizer ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)

    # ── Compute EN EM baseline (if not in checkpoint) ───────────
    if en_em_baseline is None:
        en_em_baseline = compute_en_em_baseline(model, criterion, tokenizer, config, device)

    # ── XQuAD dataloaders for Evaluation ────────────────────────
    from data.xquad_loader import create_xquad_dataloaders
    _, _, val_pairs = create_xquad_dataloaders(
        root_dir=config["root_dir"],
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["max_length"],
    )

    # ── Squad Parallel dataloaders ──────────────────────────────
    from squad_parallel_loader import create_squad_parallel_dataloaders
    train_loader, _ = create_squad_parallel_dataloaders(
        tokenizer=tokenizer,
        en_path=os.path.join(config["root_dir"], "dataset", "Squad2.0", "train-v2.0.json"),
        vi_path=os.path.join(config["root_dir"], "dataset", "AIForge_vietnamese-squad", "train-00000-of-00001.parquet"),
        batch_size=config["batch_size"],
        max_length=config["max_length"],
    )
    log.info(f"Train (SQuAD Parallel): {len(train_loader)} batches | XQuAD Val: {len(val_pairs)} pairs")

    # ── Optimizer — differential learning rates ──────────────────
    # backbone: frozen except for LoRA parameters
    # layer_weights + QA head: stage2_head_lr (QA head frozen if freeze_qa_head is True)
    trainable_backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    if config.get("freeze_qa_head", False):
        optimizer = AdamW([
            {"params": trainable_backbone,           "lr": config["stage2_head_lr"], "weight_decay": config["weight_decay"]},
            {"params": [model.layer_weights],        "lr": config["stage2_head_lr"], "weight_decay": 0.0},
        ])
        log.info("Optimizer: LoRA (with decay) + layer_weights (no decay). QA head frozen.")
    else:
        optimizer = AdamW([
            {"params": trainable_backbone,           "lr": config["stage2_head_lr"], "weight_decay": config["weight_decay"]},
            {"params": [model.layer_weights],        "lr": config["stage2_head_lr"], "weight_decay": 0.0},
            {"params": list(criterion.parameters()), "lr": config["stage2_head_lr"], "weight_decay": 0.0},
        ])
        log.info("Optimizer: LoRA (with decay) + layer_weights (no decay) + QA head (no decay).")

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

    stage2_loss = Stage2Loss(
        lambda_ot   = config["lambda_ot"],
        lambda_reg  = config["lambda_reg"],
        lambda_span = config["lambda_span"],
        lambda_qa   = config["lambda_qa"],
    ).to(device)

    log.info(
        f"Stage 2 Loss config: lambda_ot={config['lambda_ot']}, lambda_reg={config['lambda_reg']}, lambda_qa={config['lambda_qa']}"
    )

    # ── TensorBoard ─────────────────────────────────────────────
    tb_dir = os.path.join(config["output_dir"], "tensorboard_stage2")
    writer = SummaryWriter(log_dir=tb_dir)
    log.info(f"TensorBoard: {tb_dir}")

    # ── Verification Check ──────────────────────────────────────
    if config.get("freeze_qa_head", False):
        assert not any(p.requires_grad for p in criterion.qa_head.parameters()), \
            "BUG: QA head still has requires_grad=True"
    else:
        assert any(p.requires_grad for p in criterion.qa_head.parameters()), \
            "BUG: QA head does not have requires_grad=True"

    assert any(p.requires_grad for p in model.backbone.parameters()), \
        "BUG: Backbone has no trainable parameters (LoRA missing)"

    assert model.layer_weights.requires_grad, \
        "BUG: layer_weights is frozen"

    log.info("Ablation setup verified successfully.")

    # ── Initial state ───────────────────────────────────────────
    start_epoch    = 1
    best_vi_em     = 0.0
    patience_count = 0
    global_step    = 0

    # ── Resume Logic ─────────────────────────────────────────────
    if config.get("resume_from"):
        resume_path = config["resume_from"]
        if os.path.exists(resume_path):
            log.info(f"Resuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            
            # Load states
            model.load_state_dict(ckpt["model_state"], strict=False)
            criterion.load_state_dict(ckpt["criterion_state"])
            
            # Re-apply QA head freeze if enabled in config
            if config.get("freeze_qa_head", False):
                freeze_qa_head(criterion)
                
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if ckpt.get("scheduler_state") and scheduler:
                scheduler.load_state_dict(ckpt["scheduler_state"])
                
            # Restore iteration state
            start_epoch    = ckpt["epoch"] + 1
            global_step    = ckpt["global_step"]
            best_vi_em     = ckpt.get("best_vi_em", 0.0)
            patience_count = ckpt.get("patience_count", 0)
            
            # Restore RNG state
            if "rng_state_cpu" in ckpt:
                torch.set_rng_state(ckpt["rng_state_cpu"])
            if "rng_state_cuda" in ckpt and torch.cuda.is_available() and ckpt["rng_state_cuda"] is not None:
                torch.cuda.set_rng_state(ckpt["rng_state_cuda"])
                
            log.info(f"  Resumed at Epoch {start_epoch}, Global Step {global_step}, Best VI EM: {best_vi_em:.2f}%")
        else:
            log.warning(f"Resume checkpoint not found: {resume_path}. Starting from scratch.")

    # ── Training epochs ─────────────────────────────────────────
    for epoch in range(start_epoch, config["max_epochs"] + 1):

        # LoRA adapter handles freeze/unfreeze automatically during forward
        model.train()
        criterion.train()

        log.info(f"{'━'*60}")
        log.info(f"Epoch {epoch}/{config['max_epochs']}")

        epoch_losses = {
            "total": 0.0, "ot": 0.0, "reg": 0.0, "span": 0.0, "qa": 0.0, "has_ans": 0.0,
            "raw_ot_loss": 0.0, "raw_reg_loss": 0.0, "raw_qa_loss": 0.0,
            "weighted_ot": 0.0, "weighted_reg": 0.0, "weighted_qa": 0.0
        }
        step_count   = 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad()

            steps_per_epoch = len(train_loader)
            if epoch == 1:
                # Linearly decay from 0.1 to 0.03 over the first epoch
                decay_ratio = min(1.0, (step + 1) / steps_per_epoch)
                current_eps = 0.1 - decay_ratio * (0.1 - 0.03)
            else:
                # Keep it sharp at 0.03 for all subsequent epochs
                current_eps = 0.03

            losses = stage2_step(
                batch, model, criterion, stage2_loss,
                epsilon=current_eps,
                n_iters=config["sinkhorn_iters"],
                epoch=epoch,
                device=device,
            )

            losses["total"].backward()

            # Clip gradients (exclude non-trainable parameters to be clean)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            trainable_params += [p for p in criterion.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, config["max_grad_norm"])

            optimizer.step()
            scheduler.step()
            global_step += 1
            step_count  += 1

            for k in ("total", "ot", "reg", "span", "qa", "has_ans", "raw_ot_loss", "raw_reg_loss", "raw_qa_loss", "weighted_ot", "weighted_reg", "weighted_qa"):
                v = losses.get(k)
                if isinstance(v, torch.Tensor):
                    epoch_losses[k] += v.item()
                elif isinstance(v, float):
                    epoch_losses[k] += v

            # ── Per-step TensorBoard logging ─────────────────────
            if global_step % config["log_every"] == 0:
                g_ent  = losses.get("gamma_entropy", 0.0)
                span_w = losses.get("span_weight", torch.tensor(0.0)).item()

                log.info(
                    f"  Step {global_step} | "
                    f"total={losses['total'].item():.4f} | "
                    f"raw_ot={losses['raw_ot_loss'].item():.4f} | "
                    f"raw_reg={losses['raw_reg_loss'].item():.4f} | "
                    f"raw_qa={losses['raw_qa_loss'].item():.4f} | "
                    f"has_ans={losses['has_ans'].item():.4f} | "
                    f"w_ot={losses['weighted_ot'].item():.4f} | "
                    f"w_reg={losses['weighted_reg'].item():.4f} | "
                    f"w_qa={losses['weighted_qa'].item():.4f} | "
                    f"eps={current_eps:.4f} | "
                    f"γ_H={g_ent:.2f}"
                )

                writer.add_scalar("Loss/Stage2_Total",  losses["total"].item(), global_step)
                writer.add_scalar("Loss/OT",            losses["ot"].item(),    global_step)
                writer.add_scalar("Loss/Reg",           losses["reg"].item(),   global_step)
                writer.add_scalar("Loss/Span",          losses["span"].item(),  global_step)
                writer.add_scalar("Loss/QA",            losses["qa"].item(),    global_step)
                writer.add_scalar("Loss/HasAns",        losses["has_ans"].item(), global_step)
                
                # Raw and Weighted losses
                writer.add_scalar("Loss/Raw_OT",        losses["raw_ot_loss"].item(), global_step)
                writer.add_scalar("Loss/Raw_Reg",       losses["raw_reg_loss"].item(), global_step)
                writer.add_scalar("Loss/Raw_QA",        losses["raw_qa_loss"].item(), global_step)
                writer.add_scalar("Loss/Weighted_OT",   losses["weighted_ot"].item(), global_step)
                writer.add_scalar("Loss/Weighted_Reg",  losses["weighted_reg"].item(), global_step)
                writer.add_scalar("Loss/Weighted_QA",   losses["weighted_qa"].item(), global_step)
                
                writer.add_scalar("Lambda/Span_Weight", span_w,                 global_step)
                writer.add_scalar("Debug/Gamma_Entropy", g_ent,                 global_step)
                writer.add_scalar("Learning_Rate/Head",
                                  optimizer.param_groups[1]["lr"],             global_step)
                writer.add_scalar("Hyperparameters/Epsilon", current_eps, global_step)

        # ── End of epoch summary ─────────────────────────────────
        avg = {k: v / max(step_count, 1) for k, v in epoch_losses.items()}
        log.info(
            f"Epoch {epoch} avg | total={avg['total']:.4f} | "
            f"raw_ot={avg['raw_ot_loss']:.4f} | raw_reg={avg['raw_reg_loss']:.4f} | raw_qa={avg['raw_qa_loss']:.4f} | "
            f"w_ot={avg['weighted_ot']:.4f} | w_reg={avg['weighted_reg']:.4f} | w_qa={avg['weighted_qa']:.4f}"
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
            if epoch >= 4 and drop > config["en_em_safety"]:
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
                config, vi_em, best_vi_em, patience_count,
            )

            # Upload to Hugging Face
            if config.get("hf_repo_id"):
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=os.environ.get("HF_TOKEN"))
                    output_basename = os.path.basename(os.path.normpath(config["output_dir"])) or "checkpoint_stage2"
                    log.info(f"   Uploading epoch checkpoint to Hugging Face ({config['hf_repo_id']})...")
                    api.upload_file(
                        path_or_fileobj=ckpt_out,
                        path_in_repo=f"{output_basename}/stage2_epoch_{epoch:03d}.pt",
                        repo_id=config["hf_repo_id"],
                        repo_type="model"
                    )
                    if writer is not None:
                        api.upload_folder(
                            folder_path=tb_dir,
                            path_in_repo=f"logs/{output_basename}_tensorboard",
                            repo_id=config["hf_repo_id"],
                            repo_type="model"
                        )
                    log.info("   ✅ Epoch checkpoint & TensorBoard logs uploaded successfully!")
                except Exception as e:
                    log.error(f"   Upload epoch checkpoint error (local file still safe): {e}")


        # ── Early stopping ────────────────────────────────────────
        if epoch >= 4:
            if vi_em > best_vi_em + config["min_delta_em"]:
                best_vi_em     = vi_em
                patience_count = 0
                best_path = os.path.join(config["output_dir"], "stage2_best.pt")
                save_stage2_checkpoint(
                    best_path, epoch, global_step,
                    model, criterion, optimizer, scheduler,
                    config, vi_em, best_vi_em, patience_count,
                )
                log.info(f"  ★ New best VI EM={vi_em:.2f}% — saved {best_path}")

                # Upload best checkpoint to Hugging Face
                if config.get("hf_repo_id"):
                    try:
                        from huggingface_hub import HfApi
                        api = HfApi(token=os.environ.get("HF_TOKEN"))
                        output_basename = os.path.basename(os.path.normpath(config["output_dir"])) or "checkpoint_stage2"
                        log.info(f"   Uploading best checkpoint to Hugging Face...")
                        api.upload_file(
                            path_or_fileobj=best_path,
                            path_in_repo=f"{output_basename}/stage2_best.pt",
                            repo_id=config["hf_repo_id"],
                            repo_type="model"
                        )
                        log.info("   ✅ Best checkpoint uploaded successfully!")
                    except Exception as e:
                        log.error(f"   Upload best checkpoint error: {e}")

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
        else:
            log.info(f"  Epoch {epoch} < 4. Early stopping monitoring is suspended.")

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
    parser.add_argument("--lambda_reg",     type=float, default=STAGE2_CONFIG["lambda_reg"])
    parser.add_argument("--lambda_span",    type=float, default=STAGE2_CONFIG["lambda_span"])
    parser.add_argument("--lambda_qa",      type=float, default=STAGE2_CONFIG["lambda_qa"])
    parser.add_argument("--epsilon",        type=float, default=STAGE2_CONFIG["epsilon"])
    parser.add_argument("--epsilon_end",    type=float, default=STAGE2_CONFIG["epsilon_end"])
    parser.add_argument("--sinkhorn_iters", type=int,   default=STAGE2_CONFIG["sinkhorn_iters"])
    parser.add_argument("--patience",       type=int,   default=STAGE2_CONFIG["patience"])
    parser.add_argument("--max_length",     type=int,   default=STAGE2_CONFIG["max_length"])
    parser.add_argument("--output_dir",     default=STAGE2_CONFIG["output_dir"])
    parser.add_argument("--log_every",      type=int,   default=STAGE2_CONFIG["log_every"])
    parser.add_argument("--freeze_qa_head", action="store_true", default=False,
                        help="Freeze QA head (ablation: pure OT backbone alignment)")
    parser.add_argument("--resume_from",    type=str,   default=None,
                        help="Path to Stage 2 checkpoint to resume training from")
    parser.add_argument("--en_em_safety", type=float, default=STAGE2_CONFIG["en_em_safety"],
                        help="Hard stop threshold for EN EM drop")
    parser.add_argument("--hf_repo_id",    type=str,   default=STAGE2_CONFIG["hf_repo_id"],
                        help="HuggingFace repo ID for auto backup of checkpoints/logs")
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
