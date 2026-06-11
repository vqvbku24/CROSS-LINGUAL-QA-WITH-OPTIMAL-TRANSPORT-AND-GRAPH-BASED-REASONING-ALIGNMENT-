# train.py
"""
Training loop for Cross-Lingual QA with Sinkhorn OT Alignment.

Two modes:
  1. --mode overfit  : Overfit on a single batch — sanity check.
  2. --mode train    : Full training loop with gradient accumulation + scheduler.

Architecture (Zero-Shot + Global OT):
  - No graph, no GAT, no subsampling, no FGW.
  - Sinkhorn OT on full XLM-R hidden states.
  - L_total = L_qa + L_has_ans + λ_ot * L_ot
  - No L_span or L_cons (Topological Attractor problem).
"""

import os
import sys

# Add project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load variables from .env file
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(env_path)
except ImportError:
    pass

import math
import argparse
import logging

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Import Hugging Face API for checkpoint upload
try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Distributed Setup / Cleanup
# ──────────────────────────────────────────────────────────────

def is_distributed() -> bool:
    """Check if we're running in distributed mode (via torchrun)."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed():
    """
    Initialize distributed process group if LOCAL_RANK env var is set.
    Called from main() before any training.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return  # Not running with torchrun → single-GPU mode

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    log.info(f"[Rank {dist.get_rank()}/{dist.get_world_size()}] "
             f"Distributed initialized on GPU {local_rank}")


def cleanup_distributed():
    """Clean up distributed process group."""
    if is_distributed():
        dist.destroy_process_group()


# ──────────────────────────────────────────────────────────────
# Default Config
# ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "model_name"        : "xlm-roberta-base",

    # OT hyperparameters
    "sinkhorn_epsilon"  : 0.05,
    "sinkhorn_iters"    : 300,

    # Loss weights (Zero-Shot + Global OT: only λ_ot)
    "lambda_ot"         : 0.1,

    # Training hyperparameters
    "batch_size"        : 32,
    "grad_accum_steps"  : 1,
    "lr"                : 1e-5,
    "head_lr"           : 8e-5,
    "weight_decay"      : 0.01,
    "warmup_ratio"      : 0.08,
    "max_epochs"        : 15,
    "max_grad_norm"     : 1.0,
    "pairing_strategy"  : "topic",

    # Overfit mode
    "overfit_steps"     : 400,
    "overfit_lr"        : 3e-4,

    # Paths
    "root_dir"          : os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output_dir"        : os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints"),
    "hf_repo_id"        : "",
    "save_every"        : 1,
    "log_every"         : 10,
}


# ──────────────────────────────────────────────────────────────
# Setup DataLoader
# ──────────────────────────────────────────────────────────────

def setup_dataloader(config: dict, for_training: bool = True):
    from phase1_dataloader.data_setup import get_setup_objects
    from phase1_dataloader.cross_lingual_dataset import create_dataloader

    if is_main_process():
        log.info("Loading datasets and tokenizer...")
    teacher_ds, student_ds, tokenizer = get_setup_objects(
        root_dir=config["root_dir"],
    )

    # Use DistributedSampler when running with torchrun
    sampler = None
    shuffle = True
    if for_training and is_distributed():
        from phase1_dataloader.cross_lingual_dataset import CrossLingualQADataset
        # create_dataloader builds the dataset internally, so we create it
        # and pass DistributedSampler via the loader afterwards.
        # For now, keep create_dataloader as-is but disable shuffle
        # (DistributedSampler handles shuffling)
        shuffle = False

    train_loader = create_dataloader(
        teacher_ds=teacher_ds,
        student_ds=student_ds,
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        shuffle=shuffle and not is_distributed(),
        pairing_strategy=config["pairing_strategy"],
    )

    # Wrap with DistributedSampler for DDP
    if for_training and is_distributed():
        dataset = train_loader.dataset
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
        )
        train_loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            sampler=sampler,
            collate_fn=train_loader.collate_fn,
            num_workers=getattr(train_loader, 'num_workers', 0),
            pin_memory=True,
        )

    if is_main_process():
        log.info(f"DataLoader ready: {len(train_loader)} batches/epoch"
                 f"{' (distributed)' if is_distributed() else ''}")
    return train_loader, tokenizer, sampler


# ──────────────────────────────────────────────────────────────
# Setup Model + Criterion
# ──────────────────────────────────────────────────────────────

def setup_model_and_criterion(config: dict, device: torch.device):
    from phase2_model.model_core import CrossLingualOTModel
    from phase3_loss.losses import OTAlignmentLoss

    model = CrossLingualOTModel(
        model_name=config["model_name"],
    ).to(device)

    criterion = OTAlignmentLoss(
        hidden_size         = model.backbone.hidden_size,
        lambda_ot           = config["lambda_ot"],
        sinkhorn_epsilon    = config["sinkhorn_epsilon"],
        sinkhorn_iters      = config["sinkhorn_iters"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    head_params  = sum(p.numel() for p in criterion.parameters()) / 1e6
    log.info(f"Model: {total_params:.1f}M params | QA Head: {head_params:.2f}M params | Device: {device}")

    return model, criterion


# ──────────────────────────────────────────────────────────────
# Mode 1: Overfit on a single batch (Sanity Check)
# ──────────────────────────────────────────────────────────────

def run_overfit(config: dict, device: torch.device):
    log.info("=" * 60)
    log.info("MODE: OVERFIT ON A SINGLE BATCH (Sanity Check)")
    log.info("=" * 60)

    train_loader, tokenizer, _ = setup_dataloader(config, for_training=False)
    model, criterion = setup_model_and_criterion(config, device)

    fixed_batch = next(iter(train_loader))
    fixed_batch = {k: v.to(device, non_blocking=True) for k, v in fixed_batch.items()}
    log.info(f"Fixed batch shapes: { {k: tuple(v.shape) for k, v in fixed_batch.items()} }")

    # ── Strategy: Freeze backbone, train QA head only ──────────
    log.info("Freezing backbone. Training QA head only (overfit sanity check).")
    for p in model.parameters():
        p.requires_grad_(False)

    qa_params = [p for p in criterion.parameters() if p.requires_grad]
    log.info(f"Trainable: QA head {sum(p.numel() for p in qa_params)} params")

    # QA head LR = 10× overfit_lr (3e-4 × 10 = 3e-3)
    # ×100 was too aggressive (0.03) → oscillation; ×10 is fast yet stable
    opt_qa = AdamW(qa_params, lr=config["overfit_lr"] * 10, weight_decay=0.0)

    model.eval()
    criterion.train()

    # Disable OT in overfit: just verify QA head works
    orig_lambda_ot = criterion.lambda_ot
    criterion.lambda_ot = 0.0
    log.info("Overfit mode: lambda_ot=0 (verifying qa head only)")

    prev_loss      = float("inf")
    stagnant_count = 0

    for step in range(1, config["overfit_steps"] + 1):
        opt_qa.zero_grad()

        outputs = model(fixed_batch)
        losses  = criterion(outputs, fixed_batch, global_step=step, spe=config["overfit_steps"])

        losses["total"].backward()
        gn_qa = torch.nn.utils.clip_grad_norm_(qa_params, max_norm=10.0).item()
        opt_qa.step()

        total = losses["total"].item()

        if step % 10 == 0 or step == 1:
            log.info(
                f"Step {step:>4d}/{config['overfit_steps']} | "
                f"total={total:.4f} | "
                f"qa={losses['qa'].item():.4f} | "
                f"ot={losses['ot'].item():.4f} | "
                f"gn_qa={gn_qa:.3f}"
            )

        if total >= prev_loss - 1e-5:
            stagnant_count += 1
            if stagnant_count >= 30:
                log.warning("Loss not decreasing after 30 consecutive steps")
                break
        else:
            stagnant_count = 0
        prev_loss = total

    final_qa = losses["qa"].item()
    log.info("=" * 60)
    if final_qa < 1.0:
        log.info(f"OVERFIT PASSED! qa={final_qa:.4f} < 1.0")
    else:
        log.warning(f"OVERFIT NOT CONVERGED. qa={final_qa:.4f}")
    log.info("=" * 60)

    # Restore
    criterion.lambda_ot = orig_lambda_ot
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()


# ──────────────────────────────────────────────────────────────
# Mode 1b: Overfit FULL — unfreeze backbone + head
# ──────────────────────────────────────────────────────────────

def run_overfit_full(config: dict, device: torch.device):
    log.info("=" * 60)
    log.info("MODE: OVERFIT FULL (Backbone + QA head — all unfrozen)")
    log.info("=" * 60)

    train_loader, tokenizer, _ = setup_dataloader(config, for_training=False)
    model, criterion = setup_model_and_criterion(config, device)

    fixed_batch = next(iter(train_loader))
    fixed_batch = {k: v.to(device, non_blocking=True) for k, v in fixed_batch.items()}
    log.info(f"Fixed batch shapes: { {k: tuple(v.shape) for k, v in fixed_batch.items()} }")

    for p in model.parameters():
        p.requires_grad_(True)

    backbone_params   = list(model.backbone.parameters())
    layer_w_params    = [model.layer_weights]          # nn.Parameter — learns which layer (6-9) is best
    head_params       = list(criterion.parameters())
    all_params        = backbone_params + layer_w_params + head_params

    total_trainable = sum(p.numel() for p in all_params if p.requires_grad)
    log.info(f"Trainable: ALL params — {total_trainable/1e6:.2f}M")
    log.info(f"LR: backbone=1e-5 | layer_weights=1e-4 | head=1e-4 | weight_decay=0.01")
    log.info(f"lambda_ot={config['lambda_ot']}")

    optimizer = AdamW([
        {"params": backbone_params,  "lr": 1e-5},
        {"params": layer_w_params,   "lr": 1e-4},   # layer mixing weights (6,7,8,9)
        {"params": head_params,      "lr": 1e-4},
    ], weight_decay=0.01)

    model.train()
    criterion.train()

    prev_loss      = float("inf")
    stagnant_count = 0

    for step in range(1, config["overfit_steps"] + 1):

        # ── Curriculum: OT warmup ──────────────────────────────
        # OT: Phase1 step 1-50 = 0, Phase2 step 51-100 = 0→max
        if step <= 50:
            criterion.lambda_ot = 0.0
        elif step <= 100:
            criterion.lambda_ot = config["lambda_ot"] * (step - 50) / 50.0
        else:
            criterion.lambda_ot = config["lambda_ot"]

        if step % 10 == 0 and 51 <= step <= 110:
            log.info(
                f"   [Annealing step {step}] "
                f"OT={criterion.lambda_ot:.4f}"
            )

        optimizer.zero_grad()

        outputs = model(fixed_batch)
        losses  = criterion(outputs, fixed_batch, global_step=step, spe=config["overfit_steps"])

        losses["total"].backward()

        gn_bb   = torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=0.15).item()
        gn_lw   = torch.nn.utils.clip_grad_norm_(layer_w_params,  max_norm=1.0).item()
        gn_head = torch.nn.utils.clip_grad_norm_(head_params,     max_norm=1.5).item()

        optimizer.step()

        total = losses["total"].item()

        if step % 10 == 0 or step == 1:
            log.info(
                f"Step {step:>4d}/{config['overfit_steps']} | "
                f"total={total:.4f} | "
                f"qa={losses['qa'].item():.4f} | "
                f"ot={losses['ot'].item():.4f} | "
                f"gn_bb={gn_bb:.3f} gn_head={gn_head:.3f}"
            )

        if total >= prev_loss - 1e-5:
            stagnant_count += 1
            if stagnant_count >= 30:
                log.warning("Loss not decreasing after 30 consecutive steps")
                break
        else:
            stagnant_count = 0
        prev_loss = total

    final_total = losses["total"].item()
    log.info("=" * 60)
    log.info(
        f"Final: total={final_total:.4f} | qa={losses['qa'].item():.4f} | "
        f"ot={losses['ot'].item():.4f}"
    )
    if final_total < 3.0:
        log.info("OVERFIT_FULL: Loss decreasing steadily — joint training OK!")
    else:
        log.warning("OVERFIT_FULL: Loss may not have converged sufficiently.")
    log.info("=" * 60)


# ──────────────────────────────────────────────────────────────
# Mode 2: Full Training Loop
# ──────────────────────────────────────────────────────────────

def run_training(config: dict, device: torch.device):
    log.info("=" * 60)
    log.info("MODE: FULL TRAINING (Zero-Shot + Global OT)")
    log.info("=" * 60)

    os.makedirs(config["output_dir"], exist_ok=True)

    train_loader, tokenizer, sampler = setup_dataloader(config, for_training=True)
    model, criterion = setup_model_and_criterion(config, device)

    # Wrap with DDP if distributed
    if is_distributed():
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)
        criterion = DDP(criterion, device_ids=[device.index], find_unused_parameters=True)
        log.info(f"[Rank {get_rank()}] Model & Criterion wrapped with DDP")

    # Access underlying module for param groups (DDP wraps .module)
    _model = model.module if is_distributed() else model
    _criterion = criterion.module if is_distributed() else criterion

    backbone_params = list(_model.backbone.parameters())
    layer_w_params  = [_model.layer_weights]           # nn.Parameter — learns which layer (6-9) is best
    head_params     = list(_criterion.parameters())

    optimizer = AdamW([
        {"params": backbone_params, "lr": config.get("lr", 1e-5)},
        {"params": layer_w_params,  "lr": config.get("head_lr", 1e-4)},  # layer mixing weights
        {"params": head_params,     "lr": config.get("head_lr", 1e-4)},
    ], weight_decay=config["weight_decay"])

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
        log.warning("transformers scheduler not found — running without scheduler")

    start_epoch = 1
    global_step = 0
    best_em = 0.0
    optimizer.zero_grad()

    # TensorBoard (only on main process)
    writer = None
    if is_main_process():
        tb_log_dir = os.path.join(config["output_dir"], "tensorboard_logs")
        writer = SummaryWriter(log_dir=tb_log_dir)
        log.info(f"TensorBoard enabled. Logs at: {tb_log_dir}")

    if config.get("resume_from"):
        if os.path.exists(config["resume_from"]):
            log.info(f"Loading checkpoint from {config['resume_from']}...")
            checkpoint = torch.load(config["resume_from"], map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            criterion.load_state_dict(checkpoint["criterion_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if scheduler is not None and checkpoint.get("scheduler_state") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            global_step = checkpoint.get("global_step", 0)
            log.info(f"Resumed from epoch {checkpoint.get('epoch')}, global step {global_step}")
        else:
            log.warning(f"Checkpoint not found at {config['resume_from']}, starting from scratch.")

    # Access criterion's lambda attrs (through DDP wrapper if needed)
    _criterion = criterion.module if is_distributed() else criterion

    # OT warmup schedule: first half of epoch 1 = 0, then linear ramp over 1 epoch
    _SPE = steps_per_epoch
    _OT_DELAY,   _OT_WARMUP   = _SPE // 2,  _SPE

    if is_main_process():
        log.info(
            f"OT Warmup (steps): delay={_OT_DELAY}, ramp={_OT_DELAY}→{_OT_DELAY+_OT_WARMUP}"
        )

    for epoch in range(start_epoch, config["max_epochs"] + 1):
        # Set epoch for DistributedSampler (ensures proper shuffling)
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        criterion.train()

        if is_main_process():
            log.info(f"Epoch {epoch}/{config['max_epochs']}: Training...")

        epoch_losses = {"total": 0.0, "qa": 0.0, "has_ans": 0.0, "ot": 0.0}
        accum_count = 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Forward
            try:
                outputs = model(batch)
            except RuntimeError as e:
                log.error(f"[Epoch {epoch} Step {step}] Forward error: {e}")
                continue

            losses = criterion(outputs, batch, global_step=global_step, spe=_SPE)

            loss = losses["total"] / config["grad_accum_steps"]
            loss.backward()

            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
                else:
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            accum_count += 1

            if (step + 1) % config["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(backbone_params, config["max_grad_norm"] * 0.15)
                torch.nn.utils.clip_grad_norm_(layer_w_params,  config["max_grad_norm"] * 1.0)
                torch.nn.utils.clip_grad_norm_(head_params,     config["max_grad_norm"] * 1.5)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1  # increment first

                # ── OT Warmup (only curriculum remaining) ──
                if global_step <= _OT_DELAY:
                    _criterion.lambda_ot = 0.0
                elif global_step <= _OT_DELAY + _OT_WARMUP:
                    _criterion.lambda_ot = config["lambda_ot"] * (global_step - _OT_DELAY) / _OT_WARMUP
                else:
                    _criterion.lambda_ot = config["lambda_ot"]

                if global_step % config["log_every"] == 0 and is_main_process():
                    log.info(
                        f"Epoch {epoch} | GlobalStep {global_step} | "
                        f"total={losses['total'].item():.4f} | "
                        f"qa={losses['qa'].item():.4f} | "
                        f"has_ans={losses.get('has_ans', torch.tensor(0)).item():.4f} | "
                        f"ot={losses['ot'].item():.4f} | "
                        f"λ_ot={_criterion.lambda_ot:.3f}"
                    )

                    # --- TENSORBOARD LOGGING ---
                    if writer is not None:
                        writer.add_scalar("Loss/Total",               losses['total'].item(),    global_step)
                        writer.add_scalar("Loss/QA",                  losses['qa'].item(),       global_step)
                        writer.add_scalar("Loss/QA_Start",            losses['qa_start'].item(), global_step)
                        writer.add_scalar("Loss/QA_End",              losses['qa_end'].item(),   global_step)
                        writer.add_scalar("Loss/HasAnswer (BCE)",     losses.get('has_ans', torch.tensor(0)).item(), global_step)
                        writer.add_scalar("Loss/OT (Transport)",      losses['ot'].item(),       global_step)

                        # Entropy metrics
                        if 'dbg/entropy_start' in losses:
                            writer.add_scalar("Metrics/VI_StartLogit_Entropy", losses['dbg/entropy_start'], global_step)
                            writer.add_scalar("Metrics/VI_EndLogit_Entropy",   losses['dbg/entropy_end'],   global_step)

                        # VI has_ans probability (zero-shot monitoring)
                        if 'dbg/vi_has_ans_prob' in losses:
                            writer.add_scalar("Metrics/VI_HasAns_Prob", losses['dbg/vi_has_ans_prob'], global_step)

                        writer.add_scalar("Lambda/OT", _criterion.lambda_ot, global_step)

                        writer.add_scalar("Learning_Rate/Backbone", optimizer.param_groups[0]['lr'], global_step)
                        writer.add_scalar("Learning_Rate/Head",     optimizer.param_groups[1]['lr'], global_step)

                        # CLS collapse detector
                        _cls_s = losses.get('dbg/cls_start', torch.tensor(0)).item()
                        _max_s = losses.get('dbg/max_start', torch.tensor(1)).item()
                        _cls_e = losses.get('dbg/cls_end',   torch.tensor(0)).item()
                        _max_e = losses.get('dbg/max_end',   torch.tensor(1)).item()
                        writer.add_scalar("Debug/CLS_Start_Logit",      _cls_s, global_step)
                        writer.add_scalar("Debug/Max_Start_Logit",      _max_s, global_step)
                        writer.add_scalar("Debug/CLS_End_Logit",        _cls_e, global_step)
                        writer.add_scalar("Debug/Max_End_Logit",        _max_e, global_step)
                        _denom_s = abs(_max_s) + 1e-8
                        _denom_e = abs(_max_e) + 1e-8
                        writer.add_scalar("Debug/Collapse_Ratio_Start", abs(_cls_s) / _denom_s, global_step)
                        writer.add_scalar("Debug/Collapse_Ratio_End",   abs(_cls_e) / _denom_e, global_step)
                        _has_acc = losses.get('dbg/has_ans_acc', torch.tensor(float('nan'))).item()
                        if not torch.isnan(torch.tensor(_has_acc)):
                            writer.add_scalar("Debug/HasAnswer_Accuracy", _has_acc, global_step)

        if accum_count > 0:
            avg_losses = {k: v / accum_count for k, v in epoch_losses.items()}
            if is_main_process():
                log.info(f"━━ Epoch {epoch}/{config['max_epochs']} done | avg_loss={avg_losses.get('total', 0):.4f}")
                log.info(f"Loss breakdown (Epoch {epoch}):")
                log.info(f"  total_loss : {avg_losses.get('total', 0):.4f}")
                log.info(f"  qa_loss    : {avg_losses.get('qa', 0):.4f} (Span Extraction)")
                log.info(f"  has_ans    : {avg_losses.get('has_ans', 0):.4f} (Answerable BCE)")
                log.info(f"  ot_loss    : {avg_losses.get('ot', 0):.4f} (Transport Cost)")
                log.info(f"  Current λ  : OT={_criterion.lambda_ot:.3f}")
                with torch.no_grad():
                    lw = torch.softmax(_model.layer_weights, dim=0)
                    log.info(
                        f"  Layer weights: "
                        f"L6={lw[0].item():.4f} L7={lw[1].item():.4f} "
                        f"L8={lw[2].item():.4f} L9={lw[3].item():.4f}"
                    )
                    if writer is not None:
                        for i, name in enumerate(["L6", "L7", "L8", "L9"]):
                            writer.add_scalar(f"LayerWeights/{name}", lw[i].item(), epoch)
                log.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            avg_loss = avg_losses.get('total', 0)
        else:
            avg_loss = 0.0

        # Quick Eval (every 2 epochs, main process only)
        if epoch % 2 == 0 and is_main_process():
            import importlib.util
            eval_file = os.path.join(config["root_dir"], "phase4-evaluation", "quick_eval.py")
            spec = importlib.util.spec_from_file_location("quick_eval", eval_file)
            quick_eval_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(quick_eval_mod)
            quick_em = quick_eval_mod.quick_em
            dev_file = os.path.join(config["root_dir"], "dataset", "Squad2.0", "dev-v2.0.json")
            # Use unwrapped model for eval
            _eval_model = model.module if is_distributed() else model
            _eval_criterion = criterion.module if is_distributed() else criterion
            if os.path.exists(dev_file):
                log.info(f"Running Quick Eval on dev set (200 samples)...")
                try:
                    em = quick_em(_eval_model, _eval_criterion, tokenizer, dev_file, n_samples=200, device=device)
                    if writer is not None:
                        writer.add_scalar("Eval/QuickEM", em, epoch)
                    log.info(f"Epoch {epoch} Quick EM (200 samples): {em:.2f}%")
                    if em > best_em:
                        best_em = em
                        best_ckpt_path = os.path.join(config["output_dir"], "best.pt")
                        torch.save({
                            "epoch": epoch,
                            "model_state": _eval_model.state_dict(),
                            "criterion_state": _eval_criterion.state_dict(),
                            "em": em,
                        }, best_ckpt_path)
                        log.info(f"   Saved best checkpoint: {best_ckpt_path}")
                except Exception as e:
                    log.error(f"Quick Eval error: {e}")
            else:
                log.warning(f"Dev file not found at {dev_file}")

        # Save checkpoint (main process only)
        if epoch % config["save_every"] == 0 and is_main_process():
            _save_model = model.module if is_distributed() else model
            _save_criterion = criterion.module if is_distributed() else criterion
            ckpt_path = os.path.join(config["output_dir"], f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch"           : epoch,
                "global_step"     : global_step,
                "model_state"     : _save_model.state_dict(),
                "criterion_state" : _save_criterion.state_dict(),
                "optimizer_state" : optimizer.state_dict(),
                "scheduler_state" : scheduler.state_dict() if scheduler else None,
                "config"          : config,
                "avg_loss"        : avg_loss,
            }, ckpt_path)
            log.info(f"   Checkpoint saved: {ckpt_path}")

            # Upload to Hugging Face
            if config.get("hf_repo_id") and HfApi is not None:
                api = HfApi(token=os.environ.get("HF_TOKEN"))
                output_basename = os.path.basename(os.path.normpath(config["output_dir"])) or "checkpoints"
                try:
                    log.info(f"   Uploading to Hugging Face ({config['hf_repo_id']})...")
                    api.upload_file(
                        path_or_fileobj=ckpt_path,
                        path_in_repo=f"{output_basename}/epoch_{epoch:03d}.pt",
                        repo_id=config["hf_repo_id"],
                        repo_type="model"
                    )
                    if writer is not None:
                        api.upload_folder(
                            folder_path=tb_log_dir,
                            path_in_repo=f"logs/{output_basename}_tensorboard",
                            repo_id=config["hf_repo_id"],
                            repo_type="model"
                        )
                    log.info(f"   Checkpoint & TensorBoard logs uploaded successfully!")
                except Exception as e:
                    log.error(f"   Upload error (local file still safe): {e}")
            elif config.get("hf_repo_id") and HfApi is None:
                log.warning("   huggingface_hub not installed! Run: pip install huggingface_hub")

        # Synchronize all processes before next epoch
        if is_distributed():
            dist.barrier()

    if is_main_process():
        log.info("Training complete!")
    if writer is not None:
        writer.close()


# ──────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Cross-Lingual OT QA Model")
    parser.add_argument("--mode",      choices=["overfit", "overfit_full", "train"], default="overfit",
                        help="'overfit': freeze backbone | 'overfit_full': unfreeze all | 'train': full training")
    parser.add_argument("--root_dir",  type=str, default=DEFAULT_CONFIG["root_dir"])
    parser.add_argument("--output_dir",type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--hf_repo_id",type=str, default=DEFAULT_CONFIG["hf_repo_id"],
                        help="HuggingFace repo ID for auto backup")
    parser.add_argument("--epochs",    type=int, default=DEFAULT_CONFIG["max_epochs"])
    parser.add_argument("--batch_size",type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",        type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--overfit_steps", type=int, default=DEFAULT_CONFIG["overfit_steps"])
    parser.add_argument("--resume_from", type=str, default="",
                        help="Path to checkpoint to resume training")

    # OT hyperparameters
    parser.add_argument("--sinkhorn_epsilon", type=float, default=DEFAULT_CONFIG["sinkhorn_epsilon"],
                        help="Sinkhorn entropic regularization")
    parser.add_argument("--sinkhorn_iters",   type=int,   default=DEFAULT_CONFIG["sinkhorn_iters"],
                        help="Number of Sinkhorn iterations")

    # Loss weights (only OT remains as tunable)
    parser.add_argument("--lambda_ot",   type=float, default=DEFAULT_CONFIG["lambda_ot"],
                        help="Weight for OT transport cost loss. Set=0 to disable.")

    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config.update({
        "root_dir"          : args.root_dir,
        "output_dir"        : args.output_dir,
        "hf_repo_id"        : args.hf_repo_id,
        "max_epochs"        : args.epochs,
        "batch_size"        : args.batch_size,
        "lr"                : args.lr,
        "head_lr"           : DEFAULT_CONFIG["head_lr"],
        "overfit_steps"     : args.overfit_steps,
        "resume_from"       : args.resume_from,
        "sinkhorn_epsilon"  : args.sinkhorn_epsilon,
        "sinkhorn_iters"    : args.sinkhorn_iters,
        "lambda_ot"         : args.lambda_ot,
    })

    # Log ablation config
    if config["lambda_ot"] == 0.0:
        log.info("⚗️  ABLATION MODE: No OT (pure zero-shot baseline)")
    else:
        log.info(f"🔬 Zero-Shot + Global OT: λ_ot={config['lambda_ot']}")

    # Initialize distributed if running with torchrun
    setup_distributed()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.backends.cudnn.benchmark = True
        if is_main_process():
            log.info(f"✅ CUDA benchmark enabled | Device: {device}"
                     f"{f' | World size: {get_world_size()}' if is_distributed() else ''}")
    else:
        device = torch.device("cpu")
        if is_main_process():
            log.info(f"Device: {device}")

    try:
        if args.mode == "overfit":
            run_overfit(config, device)
        elif args.mode == "overfit_full":
            run_overfit_full(config, device)
        else:
            run_training(config, device)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()