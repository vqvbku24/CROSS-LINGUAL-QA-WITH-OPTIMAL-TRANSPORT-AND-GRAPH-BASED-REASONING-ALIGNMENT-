# arabic/train_stage2_ar.py
"""
Stage 2 Training Loop — Arabic Branch.

Identical in structure to ../train_stage2.py (VI) but adapted for Arabic:
  - Uses ZIZOUArabic_Squad parallel data (EN-AR)
  - Uses XQuAD-ar for val evaluation
  - Uses 'ar_*' batch keys throughout
  - Has lambda_kd / kd_temperature natively (M1 Vanilla KD is built-in)
  - Output dir: checkpoint_stage2_ar/
  - TensorBoard: tensorboard_stage2_ar

Invariants (same as VI branch):
  - EN backbone: always frozen (no_grad) throughout Stage 2
  - AR ground-truth labels: NEVER used (strict zero-shot)
  - Stage 1 checkpoint: loaded read-only; never overwritten

Usage:
    # M5: Ours (full coordinated, dynamic margin)
    python arabic/train_stage2_ar.py --stage1_ckpt checkpoints/stage1_squad_best.pt --anneal_margin

    # M2: OT only (ablation)
    python arabic/train_stage2_ar.py --stage1_ckpt checkpoints/stage1_squad_best.pt \\
        --lambda_span 0.0 --lambda_margin 0.0 --lambda_kd 0.0

    # M1: Vanilla KD
    python arabic/train_stage2_ar.py --stage1_ckpt checkpoints/stage1_squad_best.pt \\
        --lambda_ot 0.0 --lambda_span 0.0 --lambda_margin 0.0 --lambda_kd 1.0
"""

import os
import sys
import math
import argparse
import logging

# Add parent project root to path (so we can import phase2_model, phase3_loss, etc.)
_ARABIC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_ARABIC_DIR)
sys.path.insert(0, _ROOT_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT_DIR, '.env'))
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from gpu_utils import auto_select_free_gpus, get_model, setup_ddp, cleanup_ddp, is_main_process, get_local_rank, set_seed
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Default Config (AR-specific)
# ──────────────────────────────────────────────────────────────

STAGE2_AR_CONFIG = {
    'stage1_ckpt'    : 'checkpoints/stage1_squad_best.pt',
    'model_name'     : 'xlm-roberta-base',

    # Loss weights
    'lambda_ot'      : 0.5,
    'lambda_reg'     : 50.0,
    'lambda_span'    : 1.0,
    'lambda_margin'  : 1.0,
    'anneal_margin'  : False,
    'lambda_qa'      : 1.0,
    # Vanilla KD — native to AR branch (M1 uses lambda_kd=1.0)
    'lambda_kd'      : 0.0,
    'kd_temperature' : 2.0,

    # OT hyperparameters
    'epsilon'        : 0.03,
    'epsilon_end'    : 0.03,
    'sinkhorn_iters' : 100,

    # Optimizer
    'stage2_head_lr' : 5e-5,
    'weight_decay'   : 0.01,
    'warmup_ratio'   : 0.06,

    # Training
    'batch_size'     : 32,
    'max_epochs'     : 6,
    'max_grad_norm'  : 1,
    'max_length'     : 384,

    # Early stopping
    'patience'       : 2,
    'min_delta_em'   : 0.5,
    'en_em_safety'   : 25.0,

    # Logging
    'log_every'      : 50,
    'save_every'     : 1,

    # Paths (AR-specific output dir)
    'root_dir'       : _ROOT_DIR,
    'output_dir'     : os.path.join(_ROOT_DIR, 'checkpoint_stage2_ar'),
    'hf_repo_id'     : '',
    'seed'           : 42,
}


# ──────────────────────────────────────────────────────────────
# Checkpoint helpers (same logic as VI, but saves to _ar dir)
# ──────────────────────────────────────────────────────────────

def load_stage1_checkpoint(ckpt_path, model, criterion, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Stage 1 checkpoint not found: {ckpt_path}')
    log.info(f'Loading Stage 1 checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)
    get_model(model).load_state_dict(ckpt['model_state'], strict=False)
    criterion.load_state_dict(ckpt['criterion_state'])
    log.info('  Stage 1 weights loaded')
    en_em_baseline = ckpt.get('em', None)
    if en_em_baseline is not None:
        log.info(f'  Stage 1 EN EM (from checkpoint): {en_em_baseline:.2f}%')
    return en_em_baseline


def save_stage2_checkpoint(path, epoch, global_step, model, criterion, optimizer, scheduler, config, ar_em, best_ar_em, patience_count):
    base_model = get_model(model)
    trainable_keys = {n for n, p in base_model.named_parameters() if p.requires_grad}
    trainable_state_dict = {k: v for k, v in base_model.state_dict().items() if k in trainable_keys}
    torch.save({
        'epoch'          : epoch,
        'global_step'    : global_step,
        'model_state'    : trainable_state_dict,
        'criterion_state': criterion.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler else None,
        'config'         : config,
        'ar_em'          : ar_em,
        'best_ar_em'     : best_ar_em,
        'patience_count' : patience_count,
        'rng_state_cpu'  : torch.get_rng_state(),
        'rng_state_cuda' : torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }, path)
    log.info(f'  Checkpoint saved: {path}')


def compute_en_em_baseline(model, criterion, tokenizer, config, device):
    import importlib.util
    eval_file = os.path.join(config['root_dir'], 'phase4-evaluation', 'quick_eval.py')
    spec = importlib.util.spec_from_file_location('quick_eval', eval_file)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dev_file = os.path.join(config['root_dir'], 'dataset', 'Squad2.0', 'dev-v2.0.json')
    if not os.path.exists(dev_file):
        log.warning(f'SQuAD dev not found at {dev_file} — EN EM safety check disabled')
        return float('inf')
    em, f1 = mod.quick_em_f1(model, criterion, tokenizer, dev_file, n_samples=200, device=device)
    log.info(f'Stage 1 EN EM baseline (200 samples): {em:.2f}%, F1: {f1:.2f}%')
    return em


# ──────────────────────────────────────────────────────────────
# AR training step (mirrors stage2_step but with ar_* batch keys)
# ──────────────────────────────────────────────────────────────

def stage2_step_ar(batch, model, criterion, stage2_loss, epsilon, alpha, n_iters, epoch, device, lambda_kd=0.0, kd_temperature=2.0):
    """
    One Stage 2 AR training step.

    Three forward passes:
      1. EN branch — LoRA OFF, no_grad  → h_en_frz (frozen anchor)
      2. EN branch — LoRA ON, with_grad → h_en_lora (for L_Reg, L_qa)
      3. AR branch — LoRA ON, with_grad → h_ar

    Batch keys expected: en_input_ids, en_attention_mask, en_start_position,
                         en_end_position, en_is_answerable, en_question_end,
                         ar_input_ids, ar_attention_mask, ar_question_end
    """
    from phase3_loss.losses import (
        sinkhorn_masked, compute_span_loss, compute_pure_margin_loss,
        gamma_entropy, _extract_question_embeddings,
        qa_loss, compute_reg_loss,
    )

    # Remap ar_* keys to vi_* for model forward (model expects vi_*)
    # We pass a remapped batch so we don't touch model internals
    batch_remapped = {
        **{k: v for k, v in batch.items() if not k.startswith('ar_')},
        'vi_input_ids':      batch['ar_input_ids'],
        'vi_attention_mask': batch['ar_attention_mask'],
        'vi_question_end':   batch['ar_question_end'],
    }

    # ── 1. EN frozen pass ────────────────────────────────────
    with torch.no_grad():
        with get_model(model).backbone.disable_adapter():
            en_frz_out = model(batch_remapped, branch='en')
            h_en_frz   = en_frz_out['hidden']
            en_mask    = ~en_frz_out['en_pad_mask']

            ar_frz_out = model(batch_remapped, branch='vi')
            h_ar_frz   = ar_frz_out['hidden']

            en_q_emb, en_q_mask = _extract_question_embeddings(h_en_frz, batch_remapped['en_question_end'])
            en_start_logits, en_end_logits, _ = criterion.qa_head(h_en_frz, en_q_emb, en_q_mask)

    # ── 2. EN LoRA pass ──────────────────────────────────────
    en_lora_out = model(batch_remapped, branch='en')
    h_en_lora   = en_lora_out['hidden']

    en_q_emb_lora, en_q_mask_lora = _extract_question_embeddings(h_en_lora, batch_remapped['en_question_end'])
    en_lora_start_logits, en_lora_end_logits, en_lora_has_ans = criterion.qa_head(h_en_lora, en_q_emb_lora, en_q_mask_lora)

    # ── 3. AR branch ─────────────────────────────────────────
    ar_out  = model(batch_remapped, branch='vi')
    h_ar    = ar_out['hidden']
    ar_mask = ~ar_out['vi_pad_mask']

    ar_q_emb, ar_q_mask = _extract_question_embeddings(h_ar, batch_remapped['vi_question_end'])
    ar_start_logits, ar_end_logits, _ = criterion.qa_head(h_ar, ar_q_emb, ar_q_mask)

    # ── 4. OT (EMA teacher-student) ──────────────────────────
    gamma_teacher, _ = sinkhorn_masked(h_en_frz, h_ar_frz, en_mask, ar_mask, epsilon=epsilon, n_iters=n_iters)
    gamma_student, _ = sinkhorn_masked(h_en_frz, h_ar,     en_mask, ar_mask, epsilon=epsilon, n_iters=n_iters)

    L_ot_list  = []
    gamma_list = []
    for i in range(len(gamma_teacher)):
        g_t   = gamma_teacher[i]
        g_s   = gamma_student[i]
        g_mix = alpha * g_t + (1.0 - alpha) * g_s
        gamma_list.append(g_mix)

        h_en_b = h_en_frz[i][en_mask[i]]
        h_ar_b = h_ar[i][ar_mask[i]]
        if h_en_b.size(0) == 0 or h_ar_b.size(0) == 0:
            L_ot_list.append(torch.tensor(0.0, device=device, requires_grad=True))
            continue

        h_en_n = F.normalize(h_en_b, dim=-1)
        h_ar_n = F.normalize(h_ar_b, dim=-1)
        C_student = 1.0 - h_en_n @ h_ar_n.T
        L_ot_list.append((g_mix.detach() * C_student).sum())

    L_ot = torch.stack(L_ot_list).mean() if L_ot_list else torch.tensor(0.0, device=device, requires_grad=True)

    # ── 5. Regularisation (EN consistency) ───────────────────
    L_reg = compute_reg_loss(h_en_lora, h_en_frz, en_mask)

    # ── 5b. Supervised EN QA ─────────────────────────────────
    en_seq_len      = h_en_lora.size(1)
    en_start        = batch['en_start_position'].clamp(max=en_seq_len - 1)
    en_end          = batch['en_end_position'].clamp(max=en_seq_len - 1)
    answerable_mask = batch['en_is_answerable'].bool().to(device)

    if answerable_mask.any():
        L_qa, _, _ = qa_loss(
            en_lora_start_logits[answerable_mask],
            en_lora_end_logits[answerable_mask],
            en_start[answerable_mask],
            en_end[answerable_mask],
        )
    else:
        L_qa = torch.tensor(0.0, device=device)

    L_has_ans = torch.tensor(0.0, device=device)

    # ── 6. Span loss ─────────────────────────────────────────
    L_span = compute_span_loss(
        gamma_list, en_start_logits, en_end_logits,
        ar_start_logits, ar_end_logits,
        en_mask, ar_mask,
    )

    # ── 6b. Margin loss ──────────────────────────────────────
    L_margin = compute_pure_margin_loss(
        en_start, en_end,
        en_start_logits, en_end_logits,
        ar_start_logits, ar_end_logits,
        en_mask, ar_mask,
        answerable_mask,
        device,
    )

    # ── 6c. Vanilla KD (M1 mode) ────────────────────────────
    L_kd = torch.tensor(0.0, device=device)
    kd_valid_ratio = 1.0
    if lambda_kd > 0.0:
        from losses.vanilla_kd_loss import naive_index_to_index_kd_loss
        en_valid_len = en_mask.sum(dim=1)
        ar_valid_len = ar_mask.sum(dim=1)
        L_kd, kd_valid_mask = naive_index_to_index_kd_loss(
            student_start_logits=ar_start_logits,
            student_end_logits=ar_end_logits,
            teacher_start_logits=en_start_logits,
            teacher_end_logits=en_end_logits,
            student_valid_len=ar_valid_len,
            teacher_valid_len=en_valid_len,
            student_gold_start=en_start,
            student_gold_end=en_end,
            temperature=kd_temperature,
        )
        kd_valid_ratio = kd_valid_mask.float().mean().item()

    # ── 7. Combine losses ────────────────────────────────────
    losses = stage2_loss(L_ot, L_reg, L_span, L_margin, L_qa, L_has_ans, epoch)
    losses['kd'] = L_kd
    losses['kd_valid_ratio'] = kd_valid_ratio
    if lambda_kd > 0.0:
        losses['total'] = losses['total'] + lambda_kd * L_kd

    # ── 8. Debug metrics ─────────────────────────────────────
    with torch.no_grad():
        g_entropy = gamma_entropy(gamma_list)
        avg_n_en  = en_mask.sum(dim=1).float().mean().item()
        avg_n_ar  = ar_mask.sum(dim=1).float().mean().item()
        h_max     = math.log(max(avg_n_en * avg_n_ar, 1.0))
        h_ratio   = g_entropy / h_max if h_max > 0 else 0

        if h_ratio > 0.90:
            log.warning(f'  [Gamma] entropy ratio={h_ratio:.2f} (H={g_entropy:.2f}/H_max={h_max:.2f}) — near uniform')
        elif h_ratio < 0.30:
            log.warning(f'  [Gamma] entropy ratio={h_ratio:.2f} — may be collapsed')

    losses['gamma_entropy'] = g_entropy
    return losses


# ──────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────

def run_stage2_ar(config: dict):
    local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')

    if is_main_process():
        log.info(f'Device: {device} (DDP world_size={world_size})')
        log.info('=' * 60)
        log.info('STAGE 2 AR: Teacher-Student Sinkhorn Alignment (Arabic)')
        log.info('=' * 60)

    os.makedirs(config['output_dir'], exist_ok=True)

    # ── Model ────────────────────────────────────────────────
    from phase2_model.model_core import CrossLingualOTModel
    from phase3_loss.losses import OTAlignmentLoss, Stage2Loss

    model     = CrossLingualOTModel(model_name=config['model_name']).to(device)
    criterion = OTAlignmentLoss(hidden_size=get_model(model).hidden_size).to(device)

    ckpt_path = config['stage1_ckpt']
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(config['root_dir'], ckpt_path)
    en_em_baseline = load_stage1_checkpoint(ckpt_path, model, criterion, device)

    if is_main_process():
        log.info('Applying LoRA to backbone...')
    model.apply_lora()
    model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Disable dropout (same as VI branch)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    for m in criterion.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    if config.get('freeze_qa_head', False):
        for p in criterion.qa_head.parameters():
            p.requires_grad_(False)
        log.info('QA head frozen')

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], use_fast=True)

    if en_em_baseline is None:
        en_em_baseline = compute_en_em_baseline(model, criterion, tokenizer, config, device)

    # ── XQuAD-ar val pairs (rank 0 only, local file only) ───
    # Do NOT call HuggingFace inside DDP — causes network race conditions.
    val_pairs = []
    if is_main_process():
        from arabic.data.xquad_loader_ar import load_xquad_ar_pairs
        val_pairs = load_xquad_ar_pairs(config['root_dir'])
        log.info(f'XQuAD-ar val pairs loaded: {len(val_pairs)}')

    # ── AR parallel train dataloader (all ranks) ────────────
    from arabic.squad_parallel_loader_ar import create_squad_parallel_dataloaders_ar
    train_loader, _, train_sampler = create_squad_parallel_dataloaders_ar(
        tokenizer=tokenizer,
        en_path=os.path.join(config['root_dir'], 'dataset', 'Squad2.0', 'train-v2.0.json'),
        ar_path=os.path.join(config['root_dir'], 'dataset', 'ZIZOUArabic_Squad', 'train.json'),
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        distributed=True,
    )
    if is_main_process():
        log.info(f'Train (SQuAD Parallel AR): {len(train_loader)} batches/GPU | XQuAD-ar Val: {len(val_pairs)} pairs')

    # ── Optimizer ────────────────────────────────────────────
    trainable_backbone = [p for p in get_model(model).backbone.parameters() if p.requires_grad]
    if config.get('freeze_qa_head', False):
        optimizer = AdamW([
            {'params': trainable_backbone,               'lr': config['stage2_head_lr'], 'weight_decay': config['weight_decay']},
            {'params': [get_model(model).layer_weights], 'lr': config['stage2_head_lr'], 'weight_decay': 0.0},
        ])
    else:
        optimizer = AdamW([
            {'params': trainable_backbone,               'lr': config['stage2_head_lr'], 'weight_decay': config['weight_decay']},
            {'params': [get_model(model).layer_weights], 'lr': config['stage2_head_lr'], 'weight_decay': 0.0},
            {'params': list(criterion.parameters()),     'lr': config['stage2_head_lr'], 'weight_decay': 0.0},
        ])

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * config['max_epochs']
    warmup_steps    = int(total_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    stage2_loss = Stage2Loss(
        lambda_ot     = config['lambda_ot'],
        lambda_reg    = config['lambda_reg'],
        lambda_span   = config['lambda_span'],
        lambda_margin = config['lambda_margin'],
        lambda_qa     = config['lambda_qa'],
    ).to(device)

    # ── TensorBoard ──────────────────────────────────────────
    writer = None
    tb_dir = os.path.join(config['output_dir'], 'tensorboard_stage2_ar')
    if is_main_process():
        writer = SummaryWriter(log_dir=tb_dir)
        log.info(f'TensorBoard: {tb_dir}')

    # ── Initial state ────────────────────────────────────────
    start_epoch           = 1
    best_ar_em            = 0.0
    patience_count        = 0
    global_step           = 0
    margin_schedule_by_epoch = {1: 1.0, 2: 0.7, 3: 0.5, 4: 0.3, 5: 0.3}

    # ── Resume ───────────────────────────────────────────────
    if config.get('resume_from'):
        resume_path = config['resume_from']
        if os.path.exists(resume_path):
            log.info(f'Resuming from: {resume_path}')
            ckpt = torch.load(resume_path, map_location=device)
            get_model(model).load_state_dict(ckpt['model_state'], strict=False)
            criterion.load_state_dict(ckpt['criterion_state'])
            if config.get('freeze_qa_head', False):
                for p in criterion.qa_head.parameters():
                    p.requires_grad_(False)
            optimizer.load_state_dict(ckpt['optimizer_state'])
            if ckpt.get('scheduler_state') and scheduler:
                scheduler.load_state_dict(ckpt['scheduler_state'])
            start_epoch    = ckpt['epoch'] + 1
            global_step    = ckpt['global_step']
            best_ar_em     = ckpt.get('best_ar_em', 0.0)
            patience_count = ckpt.get('patience_count', 0)
            log.info(f'  Resumed at Epoch {start_epoch}, Step {global_step}, Best AR EM: {best_ar_em:.2f}%')

    # ── Training epochs ──────────────────────────────────────
    for epoch in range(start_epoch, config['max_epochs'] + 1):

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        criterion.train()

        if is_main_process():
            log.info(f"{'━'*60}")
            log.info(f'Epoch {epoch}/{config["max_epochs"]}')

        epoch_losses = {
            'total': 0.0, 'ot': 0.0, 'reg': 0.0, 'span': 0.0, 'margin': 0.0, 'qa': 0.0, 'has_ans': 0.0,
            'raw_ot_loss': 0.0, 'raw_reg_loss': 0.0, 'raw_qa_loss': 0.0, 'raw_span_loss': 0.0, 'raw_margin_loss': 0.0,
            'weighted_ot': 0.0, 'weighted_reg': 0.0, 'weighted_qa': 0.0, 'weighted_span': 0.0, 'weighted_margin': 0.0,
            'kd': 0.0,
        }
        step_count = 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad()

            current_eps   = 0.03
            current_alpha = 1.0

            if config.get('anneal_margin'):
                current_margin = margin_schedule_by_epoch.get(epoch, 0.3)
                stage2_loss.lambda_margin = current_margin
            else:
                current_margin = config['lambda_margin']

            losses = stage2_step_ar(
                batch, model, criterion, stage2_loss,
                epsilon=current_eps,
                alpha=current_alpha,
                n_iters=config['sinkhorn_iters'],
                epoch=epoch,
                device=device,
                lambda_kd=config.get('lambda_kd', 0.0),
                kd_temperature=config.get('kd_temperature', 2.0),
            )

            losses['total'].backward()

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            trainable_params += [p for p in criterion.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, config['max_grad_norm'])

            optimizer.step()
            scheduler.step()
            global_step += 1
            step_count  += 1

            track_keys = ['total', 'ot', 'reg', 'span', 'margin', 'qa', 'has_ans', 'kd',
                          'raw_ot_loss', 'raw_reg_loss', 'raw_qa_loss', 'raw_span_loss', 'raw_margin_loss',
                          'weighted_ot', 'weighted_reg', 'weighted_qa', 'weighted_span', 'weighted_margin']
            for k in track_keys:
                v = losses.get(k)
                if isinstance(v, torch.Tensor):
                    epoch_losses[k] += v.item()
                elif isinstance(v, (float, int)):
                    epoch_losses[k] += float(v)

            # ── Per-step logging ─────────────────────────────
            if global_step % config['log_every'] == 0 and is_main_process():
                kd_ratio = losses.get('kd_valid_ratio', 1.0)
                log.info(
                    f'  Step {global_step} | '
                    f'total={losses["total"].item():.4f} | '
                    f'raw_ot={losses["raw_ot_loss"].item():.4f} | '
                    f'raw_reg={losses["raw_reg_loss"].item():.4f} | '
                    f'raw_span={losses.get("raw_span_loss", torch.tensor(0.0)).item():.4f} | '
                    f'raw_margin={losses.get("raw_margin_loss", torch.tensor(0.0)).item():.4f} | '
                    f'raw_qa={losses["raw_qa_loss"].item():.4f} | '
                    f'kd={losses["kd"].item():.4f} (valid_ratio={kd_ratio:.2f}) | '
                    f'eps={current_eps:.4f} | λ_margin={current_margin:.4f}'
                )
                if writer:
                    writer.add_scalar('Loss/Stage2_Total_AR',  losses['total'].item(),  global_step)
                    writer.add_scalar('Loss/OT_AR',            losses['ot'].item(),      global_step)
                    writer.add_scalar('Loss/Reg_AR',           losses['reg'].item(),     global_step)
                    writer.add_scalar('Loss/Span_AR',          losses['span'].item(),    global_step)
                    writer.add_scalar('Loss/Margin_AR',        losses['margin'].item(),  global_step)
                    writer.add_scalar('Loss/QA_AR',            losses['qa'].item(),      global_step)
                    writer.add_scalar('Loss/KD_AR',            losses['kd'].item(),      global_step)
                    writer.add_scalar('Hyperparameters/Lambda_Margin_AR', current_margin, global_step)

        # ── End of epoch eval ────────────────────────────────
        ar_em        = 0.0
        should_break = False

        if is_main_process():
            import importlib.util

            # AR eval on XQuAD-ar val
            ar_eval_file = os.path.join(_ARABIC_DIR, 'phase4_evaluation', 'quick_eval_ar.py')
            ar_spec = importlib.util.spec_from_file_location('quick_eval_ar', ar_eval_file)
            ar_eval_mod = importlib.util.module_from_spec(ar_spec)
            ar_spec.loader.exec_module(ar_eval_mod)

            ar_em, ar_f1 = ar_eval_mod.quick_em_f1_xquad_ar(
                model, criterion, tokenizer, val_pairs, device,
                max_length=config['max_length'],
            )
            log.info(f'Epoch {epoch} XQuAD-ar EM: {ar_em:.2f}%, F1: {ar_f1:.2f}%')
            if writer:
                writer.add_scalar('Eval/XQuAD_AR_EM', ar_em, epoch)
                writer.add_scalar('Eval/XQuAD_AR_F1', ar_f1, epoch)

            # Margin annealing (Fixed epoch-based schedule)
            if config.get('anneal_margin'):
                next_margin = margin_schedule_by_epoch.get(epoch + 1, 0.3)
                current_margin = margin_schedule_by_epoch.get(epoch, 0.3)
                if next_margin != current_margin:
                    log.info(f'  [Margin] Schedule update: Epoch {epoch+1} → margin={next_margin:.1f}')

            # EN EM safety check
            dev_file = os.path.join(config['root_dir'], 'dataset', 'Squad2.0', 'dev-v2.0.json')
            if os.path.exists(dev_file):
                eval_file = os.path.join(config['root_dir'], 'phase4-evaluation', 'quick_eval.py')
                en_spec = importlib.util.spec_from_file_location('quick_eval', eval_file)
                en_eval_mod = importlib.util.module_from_spec(en_spec)
                en_spec.loader.exec_module(en_eval_mod)
                en_em, en_f1 = en_eval_mod.quick_em_f1(model, criterion, tokenizer, dev_file, n_samples=200, device=device)
                log.info(f'Epoch {epoch} SQuAD EN EM (200): {en_em:.2f}% (baseline={en_em_baseline:.2f}%), F1: {en_f1:.2f}%')
                if writer:
                    writer.add_scalar('Eval/SQuAD_EN_EM_Quick_AR', en_em, epoch)
                drop = en_em_baseline - en_em
                if epoch >= 4 and drop > config['en_em_safety']:
                    log.warning(f'EN EM dropped {drop:.1f} pts (>{config["en_em_safety"]}) — hard stop!')
                    should_break = True

            # Checkpoint
            if epoch % config['save_every'] == 0:
                ckpt_out = os.path.join(config['output_dir'], f'stage2_ar_epoch_{epoch:03d}.pt')
                save_stage2_checkpoint(ckpt_out, epoch, global_step, model, criterion, optimizer, scheduler, config, ar_em, best_ar_em, patience_count)

            # Early stopping
            if epoch >= 4:
                if ar_em > best_ar_em + config['min_delta_em']:
                    best_ar_em     = ar_em
                    patience_count = 0
                    best_path = os.path.join(config['output_dir'], 'stage2_ar_best.pt')
                    save_stage2_checkpoint(best_path, epoch, global_step, model, criterion, optimizer, scheduler, config, ar_em, best_ar_em, patience_count)
                    log.info(f'  ★ New best AR EM={ar_em:.2f}% — saved {best_path}')
                else:
                    patience_count += 1
                    log.info(f'  No improvement. Patience {patience_count}/{config["patience"]}')
                    if patience_count >= config['patience']:
                        log.info(f'Early stopping at epoch {epoch} — best AR EM={best_ar_em:.2f}%')
                        should_break = True
            else:
                log.info(f'  Epoch {epoch} < 4. Early stopping monitoring suspended.')

        # Broadcast break signal (all ranks must participate)
        if world_size > 1:
            signal_tensor = torch.tensor([1 if should_break else 0], dtype=torch.long, device=device)
            torch.distributed.broadcast(signal_tensor, src=0)
            should_break = signal_tensor[0].item() == 1

        if should_break:
            break

    if is_main_process() and writer is not None:
        writer.close()
        log.info('=' * 60)
        log.info(f'Stage 2 AR complete. Best AR EM: {best_ar_em:.2f}%')
        log.info(f'Best checkpoint: {os.path.join(config["output_dir"], "stage2_ar_best.pt")}')
        log.info('=' * 60)

    cleanup_ddp()


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description='Stage 2 AR: Teacher-Student Sinkhorn Alignment (Arabic)')
    parser.add_argument('--stage1_ckpt',    default=STAGE2_AR_CONFIG['stage1_ckpt'])
    parser.add_argument('--model_name',     default=STAGE2_AR_CONFIG['model_name'])
    parser.add_argument('--batch_size',     type=int,   default=STAGE2_AR_CONFIG['batch_size'])
    parser.add_argument('--max_epochs',     type=int,   default=STAGE2_AR_CONFIG['max_epochs'])
    parser.add_argument('--stage2_head_lr', type=float, default=STAGE2_AR_CONFIG['stage2_head_lr'])
    parser.add_argument('--lambda_ot',      type=float, default=STAGE2_AR_CONFIG['lambda_ot'])
    parser.add_argument('--lambda_reg',     type=float, default=STAGE2_AR_CONFIG['lambda_reg'])
    parser.add_argument('--lambda_span',    type=float, default=STAGE2_AR_CONFIG['lambda_span'])
    parser.add_argument('--lambda_margin',  type=float, default=STAGE2_AR_CONFIG['lambda_margin'])
    parser.add_argument('--anneal_margin',  action='store_true', default=STAGE2_AR_CONFIG.get('anneal_margin', False))
    parser.add_argument('--lambda_qa',      type=float, default=STAGE2_AR_CONFIG['lambda_qa'])
    parser.add_argument('--lambda_kd',      type=float, default=STAGE2_AR_CONFIG['lambda_kd'],
                        help='Vanilla KD weight (0=disabled, 1.0 for M1)')
    parser.add_argument('--kd_temperature', type=float, default=STAGE2_AR_CONFIG['kd_temperature'])
    parser.add_argument('--epsilon',        type=float, default=STAGE2_AR_CONFIG['epsilon'])
    parser.add_argument('--sinkhorn_iters', type=int,   default=STAGE2_AR_CONFIG['sinkhorn_iters'])
    parser.add_argument('--patience',       type=int,   default=STAGE2_AR_CONFIG['patience'])
    parser.add_argument('--max_length',     type=int,   default=STAGE2_AR_CONFIG['max_length'])
    parser.add_argument('--output_dir',     default=STAGE2_AR_CONFIG['output_dir'])
    parser.add_argument('--log_every',      type=int,   default=STAGE2_AR_CONFIG['log_every'])
    parser.add_argument('--freeze_qa_head', action='store_true', default=False)
    parser.add_argument('--resume_from',    type=str,   default=None)
    parser.add_argument('--en_em_safety',   type=float, default=STAGE2_AR_CONFIG['en_em_safety'])
    parser.add_argument('--hf_repo_id',     type=str,   default=STAGE2_AR_CONFIG['hf_repo_id'])
    parser.add_argument('--seed',           type=int,   default=STAGE2_AR_CONFIG['seed'])
    args = parser.parse_args()
    config = {**STAGE2_AR_CONFIG, **vars(args)}
    return config


if __name__ == '__main__':
    config = parse_args()
    set_seed(config['seed'])

    if is_main_process():
        log.info('Stage 2 AR config:')
        for k, v in config.items():
            if k != 'root_dir':
                log.info(f'  {k:20s}: {v}')

    run_stage2_ar(config)
