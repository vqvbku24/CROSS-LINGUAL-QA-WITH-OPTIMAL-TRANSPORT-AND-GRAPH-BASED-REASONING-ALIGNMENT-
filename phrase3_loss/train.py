# train.py
"""
Training loop cho Cross-Lingual QA with OT & Graph Alignment.

Gồm hai chế độ:
  1. --mode overfit  : "Overfit on a single batch" — sanity check Phase 2/3.
                       Verify gradient flow + loss giảm đều → kiến trúc hợp lệ.
  2. --mode train    : Full training loop với gradient accumulation + scheduler.
"""

import os
import sys
import math
import argparse
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Config mặc định
# ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Model
    "model_name"    : "xlm-roberta-base",
    "K"             : 160,
    "gat_hidden"    : 512,
    "gat_out"       : 256,
    "gat_layers"    : 2,
    "fgw_alpha"     : 0.5,
    "fgw_epsilon"   : 0.01,
    "use_partial"   : True,
    "partial_m"     : 0.85,

    # Loss weights
    "lambda_fgw"    : 0.1,
    "lambda_span"   : 0.5,
    "lambda_cons"   : 0.3,
    "cons_temp"     : 2.0,
    "max_span_len"  : 30,

    # Training
    "batch_size"        : 4,    # nhỏ vì FGW tốn memory
    "grad_accum_steps"  : 8,    # effective batch = 32
    "lr"                : 2e-5,
    "weight_decay"      : 0.01,
    "warmup_ratio"      : 0.06,  # 6% tổng steps
    "max_epochs"        : 10,
    "max_grad_norm"     : 1.0,
    "pairing_strategy"  : "topic",

    # Overfit test
    "overfit_steps"     : 150,
    "overfit_lr"        : 5e-4,  # LR cao hơn để hội tụ nhanh

    # I/O
    "root_dir"      : "/content/drive/MyDrive/CROSS-LINGUAL-QA-WITH-OPTIMAL-TRANSPORT-AND-GRAPH-BASED-REASONING-ALIGNMENT-",
    "output_dir"    : "/content/drive/MyDrive/CROSS-LINGUAL-QA-WITH-OPTIMAL-TRANSPORT-AND-GRAPH-BASED-REASONING-ALIGNMENT-/checkpoints",
    "save_every"    : 1,     # save mỗi N epoch
    "log_every"     : 10,    # log mỗi N steps
}


# ──────────────────────────────────────────────────────────────
# Patch: model_core cần trả về D_en, D_vi, M
# ──────────────────────────────────────────────────────────────

def _patch_model_outputs(model, batch: dict, raw_outputs: dict) -> dict:
    """
    Nếu model_core.forward() chưa trả về D_en, D_vi, M
    (cần update model_core để trả về các trường này),
    hàm này sẽ tính tạm từ node embeddings.

    TODO: Update model_core.forward() để trả về D_en, D_vi, M trực tiếp.
    """
    outputs = dict(raw_outputs)

    if "D_en" not in outputs or "D_vi" not in outputs or "M" not in outputs:
        en_emb = outputs["en_node_emb"]  # (B, K, H)
        vi_emb = outputs["vi_node_emb"]  # (B, K, H)
        B = en_emb.size(0)

        D_en_list, D_vi_list, M_list = [], [], []
        for b in range(B):
            D_en_list.append(torch.cdist(en_emb[b], en_emb[b], p=2))
            D_vi_list.append(torch.cdist(vi_emb[b], vi_emb[b], p=2))
            en_norm = nn.functional.normalize(en_emb[b], dim=-1)
            vi_norm = nn.functional.normalize(vi_emb[b], dim=-1)
            M_list.append(1.0 - en_norm @ vi_norm.T)

        outputs["D_en"] = torch.stack(D_en_list)  # (B, K, K)
        outputs["D_vi"] = torch.stack(D_vi_list)
        outputs["M"]    = torch.stack(M_list)

    return outputs


# ──────────────────────────────────────────────────────────────
# Setup DataLoader
# ──────────────────────────────────────────────────────────────

def setup_dataloader(config: dict) -> DataLoader:
    """
    Khởi tạo CrossLingualQADataset + DataLoader.
    Dùng data_setup.get_setup_objects() để load dataset local.
    """
    from phrase1_dataloader.data_setup import get_setup_objects
    from phrase1_dataloader.cross_lingual_dataset import create_dataloader

    log.info("Loading datasets và tokenizer...")
    teacher_ds, student_ds, tokenizer = get_setup_objects(
        root_dir=config["root_dir"],
    )

    train_loader = create_dataloader(
        teacher_ds=teacher_ds,
        student_ds=student_ds,
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        shuffle=True,
        pairing_strategy=config["pairing_strategy"],
        num_workers=2,
        pin_memory=True,
    )
    log.info(f"DataLoader sẵn sàng: {len(train_loader)} batches/epoch")
    return train_loader


# ──────────────────────────────────────────────────────────────
# Setup Model + Criterion
# ──────────────────────────────────────────────────────────────

def setup_model_and_criterion(config: dict, device: torch.device):
    """Khởi tạo CrossLingualOTModel và OTAlignmentLoss."""
    from phrase2_model.model_core import CrossLingualOTModel
    from losses import OTAlignmentLoss

    model = CrossLingualOTModel(
        model_name  = config["model_name"],
        K           = config["K"],
        gat_hidden  = config["gat_hidden"],
        gat_out     = config["gat_out"],
        gat_layers  = config["gat_layers"],
        fgw_alpha   = config["fgw_alpha"],
        fgw_epsilon = config["fgw_epsilon"],
        use_partial = config["use_partial"],
        partial_m   = config["partial_m"],
    ).to(device)

    criterion = OTAlignmentLoss(
        qa_hidden_size = config["gat_out"],
        K              = config["K"],
        lambda_fgw     = config["lambda_fgw"],
        lambda_span    = config["lambda_span"],
        lambda_cons    = config["lambda_cons"],
        fgw_alpha      = config["fgw_alpha"],
        consistency_temperature = config["cons_temp"],
        max_span_len   = config["max_span_len"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"Model: {total_params:.1f}M params | Device: {device}")

    return model, criterion


# ──────────────────────────────────────────────────────────────
# Mode 1: Overfit on a single batch (Sanity Check)
# ──────────────────────────────────────────────────────────────

def run_overfit(config: dict, device: torch.device):
    """
    Sanity check: overfit trên đúng 1 batch trong N steps.

    Nếu loss giảm mượt về gần 0 → gradient flow OK, kiến trúc hợp lệ.
    Nếu loss đứng yên / nhảy loạn → cần debug model hoặc loss.

    Tham khảo: "Overfit on a single batch" — idea.docx Phase 2.
    """
    log.info("=" * 60)
    log.info("MODE: OVERFIT ON A SINGLE BATCH (Sanity Check)")
    log.info("=" * 60)

    train_loader = setup_dataloader(config)
    model, criterion = setup_model_and_criterion(config, device)

    # Lấy đúng 1 batch, fix nó
    fixed_batch = next(iter(train_loader))
    fixed_batch = {k: v.to(device) for k, v in fixed_batch.items()}
    log.info(f"Fixed batch shapes: { {k: tuple(v.shape) for k, v in fixed_batch.items()} }")

    optimizer = AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=config["overfit_lr"],
        weight_decay=0.0,  # không regularize khi overfit
    )

    model.train()
    criterion.train()

    log.info(f"Bắt đầu overfit {config['overfit_steps']} steps...")
    prev_loss = float("inf")
    stagnant_count = 0

    for step in range(1, config["overfit_steps"] + 1):
        optimizer.zero_grad()

        raw_outputs = model(fixed_batch)
        outputs     = _patch_model_outputs(model, fixed_batch, raw_outputs)
        losses      = criterion(outputs, fixed_batch)

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(criterion.parameters()),
            config["max_grad_norm"]
        )
        optimizer.step()

        total = losses["total"].item()

        if step % 10 == 0 or step == 1:
            log.info(
                f"Step {step:>4d}/{config['overfit_steps']} | "
                f"total={total:.4f} | "
                f"qa={losses['qa'].item():.4f} | "
                f"fgw={losses['fgw'].item():.4f} | "
                f"span={losses['span_proj'].item():.4f} | "
                f"cons={losses['cons'].item():.4f}"
            )

        # Early stop nếu loss không giảm sau 30 steps liên tiếp
        if total >= prev_loss - 1e-5:
            stagnant_count += 1
            if stagnant_count >= 30:
                log.warning("⚠️  Loss không giảm sau 30 steps liên tiếp — kiểm tra lại kiến trúc!")
                break
        else:
            stagnant_count = 0
        prev_loss = total

    final_loss = losses["total"].item()
    log.info("=" * 60)
    if final_loss < 0.1:
        log.info(f"✅ OVERFIT PASSED! Final loss = {final_loss:.6f} (< 0.1)")
        log.info("   Gradient flow thông suốt, kiến trúc FGW hợp lệ.")
    elif final_loss < 1.0:
        log.info(f"✅ OVERFIT OK. Final loss = {final_loss:.6f}")
        log.info("   Có thể cần thêm steps hoặc tăng overfit_lr.")
    else:
        log.warning(f"⚠️  OVERFIT CHƯA ĐẠT. Final loss = {final_loss:.6f}")
        log.warning("   Gợi ý: tăng overfit_lr, giảm K, hoặc debug gradient.")
    log.info("=" * 60)


# ──────────────────────────────────────────────────────────────
# Mode 2: Full Training Loop
# ──────────────────────────────────────────────────────────────

def run_training(config: dict, device: torch.device):
    """
    Vòng lặp training đầy đủ với:
      - Gradient accumulation
      - Linear warmup + linear decay scheduler
      - Checkpoint save theo epoch
      - Logging từng step
    """
    log.info("=" * 60)
    log.info("MODE: FULL TRAINING")
    log.info("=" * 60)

    os.makedirs(config["output_dir"], exist_ok=True)

    train_loader = setup_dataloader(config)
    model, criterion = setup_model_and_criterion(config, device)

    # Optimizer — tách params backbone (nhỏ hơn LR) vs head (LR đầy đủ)
    backbone_params = list(model.backbone.parameters())
    other_params    = (
        list(model.gat.parameters())
        + list(criterion.parameters())
    )
    optimizer = AdamW([
        {"params": backbone_params, "lr": config["lr"] * 0.1},
        {"params": other_params,    "lr": config["lr"]},
    ], weight_decay=config["weight_decay"])

    # Scheduler
    steps_per_epoch = math.ceil(len(train_loader) / config["grad_accum_steps"])
    total_steps     = steps_per_epoch * config["max_epochs"]
    warmup_steps    = int(total_steps * config["warmup_ratio"])

    try:
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        log.info(f"Scheduler: linear warmup {warmup_steps} steps / {total_steps} total")
    except ImportError:
        scheduler = None
        log.warning("transformers không tìm thấy — chạy không có scheduler")

    # ── Training Loop ─────────────────────────────────────────
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        criterion.train()

        epoch_loss  = 0.0
        accum_count = 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward
            try:
                raw_outputs = model(batch)
            except RuntimeError as e:
                log.error(f"[Epoch {epoch} Step {step}] Forward error: {e}")
                continue

            outputs = _patch_model_outputs(model, batch, raw_outputs)
            losses  = criterion(outputs, batch)

            # Scale loss cho gradient accumulation
            loss = losses["total"] / config["grad_accum_steps"]
            loss.backward()

            epoch_loss  += losses["total"].item()
            accum_count += 1

            # Optimizer step sau N accumulation steps
            if (step + 1) % config["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(criterion.parameters()),
                    config["max_grad_norm"]
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config["log_every"] == 0:
                    log.info(
                        f"Epoch {epoch} | GlobalStep {global_step} | "
                        f"total={losses['total'].item():.4f} | "
                        f"qa={losses['qa'].item():.4f} | "
                        f"fgw={losses['fgw'].item():.4f} | "
                        f"span={losses['span_proj'].item():.4f} | "
                        f"cons={losses['cons'].item():.4f}"
                    )

        # Epoch summary
        avg_loss = epoch_loss / max(accum_count, 1)
        log.info(f"━━ Epoch {epoch}/{config['max_epochs']} done | avg_loss={avg_loss:.4f}")

        # Save checkpoint
        if epoch % config["save_every"] == 0:
            ckpt_path = os.path.join(config["output_dir"], f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch"           : epoch,
                "global_step"     : global_step,
                "model_state"     : model.state_dict(),
                "criterion_state" : criterion.state_dict(),
                "optimizer_state" : optimizer.state_dict(),
                "scheduler_state" : scheduler.state_dict() if scheduler else None,
                "config"          : config,
                "avg_loss"        : avg_loss,
            }, ckpt_path)
            log.info(f"   Checkpoint saved: {ckpt_path}")

    log.info("✅ Training hoàn thành!")


# ──────────────────────────────────────────────────────────────
# Resume từ checkpoint
# ──────────────────────────────────────────────────────────────

def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load checkpoint và restore states."""
    log.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    criterion.load_state_dict(ckpt["criterion_state"])

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    log.info(f"  Resumed from epoch={ckpt['epoch']}, global_step={ckpt['global_step']}, "
             f"avg_loss={ckpt['avg_loss']:.4f}")
    return ckpt


# ──────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Cross-Lingual OT QA Model")
    parser.add_argument("--mode",      choices=["overfit", "train"], default="overfit",
                        help="'overfit' để sanity check, 'train' để full training")
    parser.add_argument("--root_dir",  type=str, default=DEFAULT_CONFIG["root_dir"])
    parser.add_argument("--output_dir",type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--epochs",    type=int, default=DEFAULT_CONFIG["max_epochs"])
    parser.add_argument("--batch_size",type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",        type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--overfit_steps", type=int, default=DEFAULT_CONFIG["overfit_steps"])
    parser.add_argument("--K",         type=int, default=DEFAULT_CONFIG["K"])
    parser.add_argument("--use_full",  action="store_true",
                        help="dùng fgw_bapg thay vì partial_fgw")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config.update({
        "root_dir"    : args.root_dir,
        "output_dir"  : args.output_dir,
        "max_epochs"  : args.epochs,
        "batch_size"  : args.batch_size,
        "lr"          : args.lr,
        "overfit_steps": args.overfit_steps,
        "K"           : args.K,
        "use_partial" : not args.use_full,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    if args.mode == "overfit":
        run_overfit(config, device)
    else:
        run_training(config, device)


if __name__ == "__main__":
    main()