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

# Add project root directory to sys.path to enable importing modules like phrase1_dataloader and phrase2_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import argparse
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

# Import Hugging Face API để upload checkpoint
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
# Config mặc định
# ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Model
    "model_name"    : "xlm-roberta-base",
    "K"             : 64,        # 160→64: FGW nhanh hơn ~15× (O(K³))
    "gat_hidden"    : 512,
    "gat_out"       : 256,
    "gat_layers"    : 2,
    "fgw_alpha"     : 0.5,
    "fgw_epsilon"   : 0.1,   # 0.01 → 0.1: làm mềm Sinkhorn, tránh exp(±∞)
    "use_partial"   : False,     # dùng BAPG thay network simplex → GPU-friendly
    "partial_m"     : 0.85,

    # Loss weights
    "lambda_fgw"    : 0.1,
    "lambda_span"   : 0.5,
    "lambda_cons"   : 0.3,
    "cons_temp"     : 2.0,
    "max_span_len"  : 30,

    # Training
    "batch_size"        : 4,    # nhỏ vì FGW tốn memory
    "grad_accum_steps"  : 4,    # effective batch=16, log nhiều hơn để debug
    "lr"                : 1e-5, # lr cho backbone
    "head_lr"           : 1e-4, # lr cho GAT và QA head
    "weight_decay"      : 0.01,
    "warmup_ratio"      : 0.06,  # 6% tổng steps
    "max_epochs"        : 10,
    "max_grad_norm"     : 1.0,
    "pairing_strategy"  : "topic",

    # Overfit test
    "overfit_steps"     : 200,
    "overfit_lr"        : 5e-4,  # LR cao hơn để hội tụ nhanh

    # I/O
    "root_dir"      : os.path.dirname(os.path.dirname(os.path.abspath(__file__))), # Absolute path to project root
    "output_dir"    : os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints"), # Absolute path to checkpoints
    "hf_repo_id"    : "",    # Để trống, truyền vào qua Argparse khi chạy
    "save_every"    : 1,     # save mỗi N epoch
    "log_every"     : 10,    # log mỗi N steps
}


# ──────────────────────────────────────────────────────────────
# Patch: model_core cần trả về D_en, D_vi, M
# (Chỉ fallback nếu model_core chưa tính — hiện tại model_core
#  đã trả về đủ cả D_en, D_vi, M, keep_idx_en trong dict.)
# ──────────────────────────────────────────────────────────────

def _patch_model_outputs(model, batch: dict, raw_outputs: dict) -> dict:
    outputs = dict(raw_outputs)

    if "D_en" not in outputs or "D_vi" not in outputs or "M" not in outputs:
        en_emb = outputs["en_node_emb"]  # (B, K, H)
        vi_emb = outputs["vi_node_emb"]  # (B, K, H)
        B = en_emb.size(0)

        # ── L2-normalize trước khi tính cdist ────────────────────────────
        # Không chuẩn hóa → cdist có thể = 50~200 → exp(-C/ε) → ±∞
        #   → gradient FGW vọt lên 55k.
        # Sau khi normalize: ||v||=1 → cdist ∈ [0, 2] → exp(-C/ε) ∈ [e^{-20}, 1]
        #   → Sinkhorn ổn định hoàn toàn, gradient thuần hóa.
        # ─────────────────────────────────────────────────────────────────
        en_emb_norm = nn.functional.normalize(en_emb, p=2, dim=-1)
        vi_emb_norm = nn.functional.normalize(vi_emb, p=2, dim=-1)

        D_en_list, D_vi_list, M_list = [], [], []
        for b in range(B):
            # D_en / D_vi: DETACH — đây là "cost geometry" của graph (cấu trúc),
            # không cần backprop qua. Nếu để grad, C1 @ g @ C2.T trong fgw_alignment_loss
            # tạo luồng gradient O(K³) ngược về backbone → gn_bb vọt lên 14k+.
            # Chỉ M (cross-language cosine distance) giữ grad → kéo EN↔VI lại gần.
            D_en_list.append(torch.cdist(en_emb_norm[b], en_emb_norm[b], p=2).detach())
            D_vi_list.append(torch.cdist(vi_emb_norm[b], vi_emb_norm[b], p=2).detach())
            # Cosine distance dùng normalized vector: 1 - cos(u,v) ∈ [0, 2]
            M_list.append(1.0 - en_emb_norm[b] @ vi_emb_norm[b].T)

        outputs["D_en"] = torch.stack(D_en_list)  # (B, K, K), max=2.0
        outputs["D_vi"] = torch.stack(D_vi_list)
        outputs["M"]    = torch.stack(M_list)

    return outputs


# ──────────────────────────────────────────────────────────────
# Setup DataLoader
# ──────────────────────────────────────────────────────────────

def setup_dataloader(config: dict) -> DataLoader:
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
    from phrase2_model.model_core import CrossLingualOTModel
    from phrase3_loss.losses import OTAlignmentLoss

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
    log.info("=" * 60)
    log.info("MODE: OVERFIT ON A SINGLE BATCH (Sanity Check)")
    log.info("=" * 60)

    train_loader = setup_dataloader(config)
    model, criterion = setup_model_and_criterion(config, device)

    fixed_batch = next(iter(train_loader))
    fixed_batch = {k: v.to(device) for k, v in fixed_batch.items()}
    log.info(f"Fixed batch shapes: { {k: tuple(v.shape) for k, v in fixed_batch.items()} }")

    # ── OVERFIT STRATEGY ─────────────────────────────────────────────────
    # GAT gradient explosion (21K-1.6M) do O(K²) từ cdist(x,x) FGW gradient.
    # BatchNorm1d giúp embedding diversity nhưng làm explosion tệ hơn (500K-1.6M).
    # Joint clipping kill QA head update (effective LR ~1e-7/step).
    #
    # → Freeze cả backbone + GAT. Chỉ train QA head (0.5K params).
    # Mục đích: "given fixed XLM-R+GAT embeddings, QA head có học đúng span?"
    # Đây là sanity check hợp lệ nhất: verify label mapping + loss + backward.
    # ─────────────────────────────────────────────────────────────────────
    log.info("Freezing backbone + GAT. Training QA head only (overfit sanity check).")
    for p in model.parameters():
        p.requires_grad_(False)

    qa_params = [p for p in criterion.parameters() if p.requires_grad]
    log.info(f"Trainable: QA head {sum(p.numel() for p in qa_params)} params | LR={config['overfit_lr']*100:.0e}")

    opt_qa = AdamW(qa_params, lr=config["overfit_lr"] * 100, weight_decay=0.0)

    model.eval()   # frozen → eval mode
    criterion.train()

    # Tắt cons + fgw trong overfit: đây không phải là training thật.
    # cons bị thổi phồng khi backbone+GAT frozen (EN confident, VI stuck)
    # và fgw cố định (GAT frozen). Chỉ cần verify qa + span giảm.
    orig_lambda_cons = criterion.lambda_cons
    orig_lambda_fgw  = getattr(criterion, "lambda_fgw", 0.1)
    criterion.lambda_cons = 0.0
    criterion.lambda_fgw  = 0.0
    log.info("Overfit mode: lambda_cons=0, lambda_fgw=0 (verifying qa + span only)")

    log.info(
        f"Bắt đầu overfit {config['overfit_steps']} steps | "
        f"LR_QA={config['overfit_lr']*100:.0e}..."
    )

    prev_loss      = float("inf")
    stagnant_count = 0

    for step in range(1, config["overfit_steps"] + 1):
        opt_qa.zero_grad()

        raw_outputs = model(fixed_batch)
        outputs     = _patch_model_outputs(model, fixed_batch, raw_outputs)
        losses      = criterion(outputs, fixed_batch)

        losses["total"].backward()
        gn_qa = torch.nn.utils.clip_grad_norm_(qa_params, max_norm=10.0).item()
        opt_qa.step()

        total = losses["total"].item()

        if step % 10 == 0 or step == 1:
            log.info(
                f"Step {step:>4d}/{config['overfit_steps']} | "
                f"total={total:.4f} | "
                f"qa={losses['qa'].item():.4f} | "
                f"fgw={losses['fgw'].item():.4f} | "
                f"span={losses['span_proj'].item():.4f} | "
                f"cons={losses['cons'].item():.4f} | "
                f"gn_qa={gn_qa:.3f}"
            )

        if total >= prev_loss - 1e-5:
            stagnant_count += 1
            if stagnant_count >= 30:
                log.warning("Loss không giảm sau 30 steps liên tiếp — kiểm tra lại kiến trúc")
                break
        else:
            stagnant_count = 0
        prev_loss = total

    final_qa   = losses["qa"].item()
    final_span = losses["span_proj"].item()
    final_sum  = final_qa + final_span
    log.info("=" * 60)
    if final_sum < 2.0:
        log.info(f"OVERFIT PASSED! qa={final_qa:.4f} span={final_span:.4f} (sum={final_sum:.4f} < 2.0)")
    else:
        log.warning(f"OVERFIT CHƯA ĐẠT. qa={final_qa:.4f} span={final_span:.4f} (sum={final_sum:.4f})")
    log.info("=" * 60)

    # Restore lambdas và unfreeze
    criterion.lambda_cons = orig_lambda_cons
    criterion.lambda_fgw  = orig_lambda_fgw
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()


# ──────────────────────────────────────────────────────────────
# Mode 1b: Overfit FULL — unfreeze backbone + GAT + head
# Bật lambda_fgw và lambda_cons để kiểm tra joint stability.
# ──────────────────────────────────────────────────────────────

def run_overfit_full(config: dict, device: torch.device):
    log.info("=" * 60)
    log.info("MODE: OVERFIT FULL (Backbone + GAT + QA head — all unfrozen)")
    log.info("=" * 60)

    train_loader = setup_dataloader(config)
    model, criterion = setup_model_and_criterion(config, device)

    fixed_batch = next(iter(train_loader))
    fixed_batch = {k: v.to(device) for k, v in fixed_batch.items()}
    log.info(f"Fixed batch shapes: { {k: tuple(v.shape) for k, v in fixed_batch.items()} }")

    # ── STRATEGY ─────────────────────────────────────────────────────────
    # Toàn bộ mạng được unfreeze (backbone + GAT + QA head).
    # lambda_fgw và lambda_cons BẬT theo config (0.1 / 0.3).
    #
    # LR (Liều 1 — XLM-R không bao giờ dùng LR > 5e-5):
    #   backbone : 1e-5  (cực thấp, tránh catastrophic forgetting)
    #   GAT+head : 1e-4  (layer mới khởi tạo, hội tụ nhanh hơn)
    #
    # Gradient clipping (Liều 2 — chặn cú tát 5000+ ở step 1):
    #   1. Clip riêng backbone (max_norm=0.5) và gat+head (max_norm=1.0)
    #      → log được gn_bb / gn_other để quan sát explosion.
    #   2. Clip joint toàn bộ (max_norm=1.0) ngay trước optimizer.step()
    #      → đảm bảo không gradient nào vượt ngưỡng dù ở group nào.
    # ─────────────────────────────────────────────────────────────────────

    # Đảm bảo toàn bộ params trainable
    for p in model.parameters():
        p.requires_grad_(True)

    backbone_params = list(model.backbone.parameters())
    other_params    = list(model.gat.parameters()) + list(criterion.parameters())
    all_params      = backbone_params + other_params

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_trainable += sum(p.numel() for p in criterion.parameters() if p.requires_grad)
    log.info(f"Trainable: ALL params — {total_trainable/1e6:.2f}M")
    log.info("LR: backbone=1e-5 | gat+head=1e-4 | weight_decay=0.01")
    log.info(
        f"lambda_fgw={config['lambda_fgw']} | "
        f"lambda_cons={config['lambda_cons']} | "
        f"lambda_span={config['lambda_span']}"
    )

    optimizer = AdamW([
        {"params": backbone_params, "lr": 1e-5},   # Liều 1: XLM-R luôn ≤ 5e-5
        {"params": other_params,    "lr": 1e-4},   # Liều 1: GAT + head mới init
    ], weight_decay=0.01)

    model.train()
    criterion.train()

    log.info(
        f"Bắt đầu overfit_full {config['overfit_steps']} steps..."
    )

    prev_loss      = float("inf")
    stagnant_count = 0

    for step in range(1, config["overfit_steps"] + 1):

        # ── Curriculum Learning — Dual Annealing (Liều 3 v2) ────────────
        # FGW  : Phase1 step 1–50 = 0, Phase2 step 51–100 = 0→0.1 (linear)
        #        Phase3 step 101+       = 0.1 (max)
        # Cons : Phase1 step 1–50 = 0, Phase2 step 51–150 = 0→0.1 (linear, chậm hơn)
        #        Phase3 step 151+       = 0.1 (cap thấp hơn config để tránh KL diverge)
        # → Tách lịch annealing: FGW hội tụ nhanh, Cons "bơm" từ từ hơn.
        # ─────────────────────────────────────────────────────────────────

        # FGW schedule (51 → 100)
        if step <= 50:
            criterion.lambda_fgw = 0.0
        elif step <= 100:
            criterion.lambda_fgw = config["lambda_fgw"] * (step - 50) / 50.0
        else:
            criterion.lambda_fgw = config["lambda_fgw"]

        # Cons schedule (51 → 150, cap tại 0.1 thay vì 0.3)
        _cons_max = 0.1   # cap thấp hơn config["lambda_cons"]=0.3
        if step <= 50:
            criterion.lambda_cons = 0.0
        elif step <= 150:
            criterion.lambda_cons = _cons_max * (step - 50) / 100.0
        else:
            criterion.lambda_cons = _cons_max

        if step % 10 == 0 and 51 <= step <= 160:
            log.info(
                f"   [Annealing step {step}] "
                f"FGW={criterion.lambda_fgw:.4f}  "
                f"CONS={criterion.lambda_cons:.4f}"
            )

        optimizer.zero_grad()

        raw_outputs = model(fixed_batch)
        outputs     = _patch_model_outputs(model, fixed_batch, raw_outputs)
        losses      = criterion(outputs, fixed_batch)

        losses["total"].backward()

        # Liều 2a: clip riêng từng group để quan sát explosion
        gn_bb = torch.nn.utils.clip_grad_norm_(
            backbone_params, max_norm=0.5
        ).item()
        gn_other = torch.nn.utils.clip_grad_norm_(
            other_params, max_norm=1.0
        ).item()
        # Liều 2b: clip joint toàn bộ — safety net cuối cùng
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

        optimizer.step()

        total = losses["total"].item()

        if step % 10 == 0 or step == 1:
            log.info(
                f"Step {step:>4d}/{config['overfit_steps']} | "
                f"total={total:.4f} | "
                f"qa={losses['qa'].item():.4f} | "
                f"fgw={losses['fgw'].item():.4f} | "
                f"span={losses['span_proj'].item():.4f} | "
                f"cons={losses['cons'].item():.4f} | "
                f"gn_bb={gn_bb:.3f} gn_other={gn_other:.3f}"
            )

        if total >= prev_loss - 1e-5:
            stagnant_count += 1
            if stagnant_count >= 30:
                log.warning(
                    "Loss không giảm sau 30 steps liên tiếp — "
                    "gradient explosion hay learning rate quá lớn"
                )
                break
        else:
            stagnant_count = 0
        prev_loss = total

    final_total = losses["total"].item()
    final_qa    = losses["qa"].item()
    final_span  = losses["span_proj"].item()
    final_fgw   = losses["fgw"].item()
    final_cons  = losses["cons"].item()
    log.info("=" * 60)
    log.info(
        f"Final: total={final_total:.4f} | qa={final_qa:.4f} | "
        f"span={final_span:.4f} | fgw={final_fgw:.4f} | cons={final_cons:.4f}"
    )
    if final_total < 3.0:
        log.info("OVERFIT_FULL: Loss giảm ổn định — kiến trúc joint training OK!")
    else:
        log.warning(
            "OVERFIT_FULL chưa đủ giảm. Kiểm tra gradient explosion "
            "(gn_bb / gn_other) và cân nhắc giảm lambda_fgw hoặc overfit_lr."
        )
    log.info("=" * 60)


# ──────────────────────────────────────────────────────────────
# Mode 2: Full Training Loop
# ──────────────────────────────────────────────────────────────

def run_training(config: dict, device: torch.device):
    log.info("=" * 60)
    log.info("MODE: FULL TRAINING")
    log.info("=" * 60)

    os.makedirs(config["output_dir"], exist_ok=True)

    train_loader = setup_dataloader(config)
    model, criterion = setup_model_and_criterion(config, device)

    backbone_params = list(model.backbone.parameters())
    other_params    = list(model.gat.parameters()) + list(criterion.parameters())
    all_params      = backbone_params + other_params
    
    # Sử dụng LR phân tách: 1e-5 cho backbone, 1e-4 cho GAT/QA head (như overfit_full)
    optimizer = AdamW([
        {"params": backbone_params, "lr": config.get("lr", 1e-5)}, 
        {"params": other_params,    "lr": config.get("head_lr", 1e-4)},
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
        log.warning("transformers không tìm thấy — chạy không có scheduler")

    start_epoch = 1
    global_step = 0
    optimizer.zero_grad()

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

    # Scale curriculum delay theo dataset size (tránh hardcode)
    _SPE = steps_per_epoch
    _FGW_DELAY,  _FGW_WARMUP  = _SPE // 2,  _SPE
    _SPAN_DELAY, _SPAN_WARMUP = _SPE,        _SPE
    _CONS_DELAY, _CONS_WARMUP = _SPE * 2,   _SPE // 2
    _CONS_MAX = 0.1
    log.info(
        f"Curriculum delays (steps): "
        f"FGW={_FGW_DELAY}→{_FGW_DELAY+_FGW_WARMUP} | "
        f"Span={_SPAN_DELAY}→{_SPAN_DELAY+_SPAN_WARMUP} | "
        f"Cons={_CONS_DELAY}→{_CONS_DELAY+_CONS_WARMUP}"
    )

    for epoch in range(start_epoch, config["max_epochs"] + 1):
        model.train()
        criterion.train()

        # (Sẽ dùng step-based curriculum bên trong loop thay vì bật/tắt cứng theo Epoch)
        if epoch == 1:
            log.info(f"Epoch {epoch}: Bắt đầu Dual Annealing theo step...")
        else:
            log.info(f"Epoch {epoch}: Duy trì max lambdas (trừ khi warmup chưa xong).")

        epoch_loss  = 0.0
        accum_count = 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # ── Curriculum Learning — Dual Annealing (cho Full Train) ──────
            # Phase 1 (step 1–200)   : chỉ L_qa — backbone + GAT ổn định trước
            # Phase 2 (step 201–500) : bật FGW dần (γ bắt đầu có ý nghĩa)
            # Phase 3 (step 501+)    : bật Span (cần γ tốt để project đúng)
            # Phase 4 (step 1001+)   : bật Cons (cần cả 2 nhánh đã học tạm)
            # ───────────────────────────────────────────────────────────────
            current_step = global_step + 1

            if current_step <= _FGW_DELAY:
                criterion.lambda_fgw = 0.0
            elif current_step <= _FGW_DELAY + _FGW_WARMUP:
                criterion.lambda_fgw = config["lambda_fgw"] * (current_step - _FGW_DELAY) / _FGW_WARMUP
            else:
                criterion.lambda_fgw = config["lambda_fgw"]

            if current_step <= _SPAN_DELAY:
                criterion.lambda_span = 0.0
            elif current_step <= _SPAN_DELAY + _SPAN_WARMUP:
                criterion.lambda_span = config["lambda_span"] * (current_step - _SPAN_DELAY) / _SPAN_WARMUP
            else:
                criterion.lambda_span = config["lambda_span"]

            if current_step <= _CONS_DELAY:
                criterion.lambda_cons = 0.0
            elif current_step <= _CONS_DELAY + _CONS_WARMUP:
                criterion.lambda_cons = _CONS_MAX * (current_step - _CONS_DELAY) / _CONS_WARMUP
            else:
                criterion.lambda_cons = _CONS_MAX

            # Forward
            try:
                raw_outputs = model(batch)
            except RuntimeError as e:
                log.error(f"[Epoch {epoch} Step {step}] Forward error: {e}")
                continue

            outputs = _patch_model_outputs(model, batch, raw_outputs)
            losses  = criterion(outputs, batch)

            loss = losses["total"] / config["grad_accum_steps"]
            loss.backward()

            epoch_loss  += losses["total"].item()
            accum_count += 1

            if (step + 1) % config["grad_accum_steps"] == 0:
                # Clip riêng từng group
                torch.nn.utils.clip_grad_norm_(
                    backbone_params,
                    config["max_grad_norm"] * 0.5,   # backbone: 0.5
                )
                torch.nn.utils.clip_grad_norm_(
                    other_params,
                    config["max_grad_norm"],           # GAT + QA head: 1.0
                )
                # Liều safety net cuối cùng (học từ overfit)
                torch.nn.utils.clip_grad_norm_(all_params, config["max_grad_norm"])
                
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
                        f"cons={losses['cons'].item():.4f} | "
                        f"λ=({criterion.lambda_fgw:.3f},{criterion.lambda_span:.3f},{criterion.lambda_cons:.3f})"
                    )

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
            log.info(f"   Checkpoint saved local: {ckpt_path}")

            # ========================================================
            # TỰ ĐỘNG ĐẨY LÊN HUGGING FACE NẾU CÓ CẤU HÌNH REPO_ID
            # ========================================================
            if config.get("hf_repo_id") and HfApi is not None:
                api = HfApi()
                try:
                    log.info(f"   Đang đẩy file {ckpt_path} lên Hugging Face ({config['hf_repo_id']})...")
                    api.upload_file(
                        path_or_fileobj=ckpt_path,
                        path_in_repo=f"checkpoints/epoch_{epoch:03d}.pt",
                        repo_id=config["hf_repo_id"],
                        repo_type="model"
                    )
                    log.info(f" Đã backup an toàn lên mây!")
                except Exception as e:
                    log.error(f" Lỗi upload lên HF (file local vẫn còn): {e}")
            elif config.get("hf_repo_id") and HfApi is None:
                log.warning(" Bạn chưa cài huggingface_hub! Chạy: pip install huggingface_hub để auto upload.")
            # ========================================================

    log.info("Training hoàn thành!")


# ──────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Cross-Lingual OT QA Model")
    parser.add_argument("--mode",      choices=["overfit", "overfit_full", "train"], default="overfit",
                        help="'overfit': freeze bb+GAT (QA head only) | 'overfit_full': unfreeze all + fgw+cons | 'train': full training")
    parser.add_argument("--root_dir",  type=str, default=DEFAULT_CONFIG["root_dir"])
    parser.add_argument("--output_dir",type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--hf_repo_id",type=str, default=DEFAULT_CONFIG["hf_repo_id"],
                        help="Tên repo HuggingFace (VD: username/my-model) để auto backup")
    parser.add_argument("--epochs",    type=int, default=DEFAULT_CONFIG["max_epochs"])
    parser.add_argument("--batch_size",type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",        type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--overfit_steps", type=int, default=DEFAULT_CONFIG["overfit_steps"])
    parser.add_argument("--K",         type=int, default=DEFAULT_CONFIG["K"])
    parser.add_argument("--use_full",  action="store_true",
                        help="dùng fgw_bapg thay vì partial_fgw")
    parser.add_argument("--resume_from", type=str, default="",
                        help="Path to checkpoint (e.g. ./checkpoints/epoch_001.pt) to resume training")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config.update({
        "root_dir"    : args.root_dir,
        "output_dir"  : args.output_dir,
        "hf_repo_id"  : args.hf_repo_id,
        "max_epochs"  : args.epochs,
        "batch_size"  : args.batch_size,
        "lr"          : args.lr,
        "head_lr"     : DEFAULT_CONFIG["head_lr"],
        "overfit_steps": args.overfit_steps,
        "K"           : args.K,
        "use_partial" : not args.use_full,
        "resume_from" : args.resume_from,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    if args.mode == "overfit":
        run_overfit(config, device)
    elif args.mode == "overfit_full":
        run_overfit_full(config, device)
    else:
        run_training(config, device)


if __name__ == "__main__":
    main()