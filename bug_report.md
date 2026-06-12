# STAGE2_SPEC.md
# Cross-Lingual QA — Stage 2: Teacher-Student Sinkhorn Alignment
# Target: coding agent (Claude Opus / automated)
# Status: DRAFT — awaiting human sign-off before execution

---

## 0. Context & Goals

Stage 1 produced a strong EN QA model (XLM-R backbone + QA Head fine-tuned on SQuAD).
Stage 2 aligns the VI embedding space to EN using XQuAD parallel data, WITHOUT any VI
ground-truth labels. The QA Head is simultaneously adapted to VI syntax via pseudo-labels
derived from the Sinkhorn transport plan γ.

**Key invariants that must never be violated:**
- EN backbone: always frozen (no_grad) throughout Stage 2
- ViQuAD: never used for training — evaluation only
- XQuAD VI val split (15%): never used for training — early stopping only
- Stage 1 checkpoint: loaded read-only; never overwritten

---

## 1. Priority Table

| ID     | Priority | File(s)                        | Description                                      |
|--------|----------|--------------------------------|--------------------------------------------------|
| S2-01  | P0       | `data/xquad_loader.py` (new)   | XQuAD dataloader with train/val split            |
| S2-02  | P0       | `phase3_loss/losses.py`        | Add `sinkhorn_masked` + `L_span` (KL) + `L_cons`|
| S2-03  | P0       | `train_stage2.py` (new)        | Stage 2 training loop, differential LR           |
| S2-04  | P1       | `phase4-evaluation/quick_eval.py` | Extend to support XQuAD VI val eval          |
| S2-05  | P1       | `train_stage2.py`              | γ entropy monitor + early stopping logic         |
| S2-06  | P2       | `train_stage2.py`              | TensorBoard logging for all Stage 2 metrics      |

---

## 2. S2-01 — XQuAD Dataloader (`data/xquad_loader.py`)

### 2.1 Dataset split

```
XQuAD VI: ~1190 parallel (EN, VI) pairs
  train: first 1010 pairs  (85%) — used for Stage 2 training
  val:   last  180 pairs   (15%) — used ONLY for early stopping eval
```

Split must be deterministic (fixed seed or index-based, not random per run).

### 2.2 Batch format

Each batch must contain BOTH EN and VI fields so one forward pass can
handle both branches:

```python
batch = {
    # English (Teacher)
    "en_input_ids":      Tensor[B, L_en],
    "en_attention_mask": Tensor[B, L_en],
    "en_start_positions": Tensor[B],      # ground-truth from XQuAD EN
    "en_end_positions":   Tensor[B],

    # Vietnamese (Student)
    "vi_input_ids":      Tensor[B, L_vi],
    "vi_attention_mask": Tensor[B, L_vi],
    # NO vi_start/end_positions — these come from pseudo-labels
}
```

### 2.3 Constraints

- Max token length: 384 for both EN and VI (consistent with Stage 1)
- Padding: right-pad to batch max length (not global max)
- `L_en` and `L_vi` are independent — they will differ within a batch; this is expected
- Shuffle train split; val split is always in fixed order

---

## 3. S2-02 — Loss Functions (`phase3_loss/losses.py`)

### 3.1 `sinkhorn_masked(h_en, h_vi, en_mask, vi_mask, epsilon, n_iters)`

**Purpose:** Compute Sinkhorn transport plan γ on non-padding tokens only.

**Algorithm:**
```python
def sinkhorn_masked(h_en, h_vi, en_mask, vi_mask, epsilon=0.1, n_iters=50):
    """
    Args:
        h_en:     Tensor[B, L_en, D]
        h_vi:     Tensor[B, L_vi, D]
        en_mask:  BoolTensor[B, L_en]  — True for real tokens
        vi_mask:  BoolTensor[B, L_vi]  — True for real tokens
        epsilon:  float — entropic regularization
        n_iters:  int   — Sinkhorn iterations
    Returns:
        gamma_list: List[B] of Tensor[n_en_i, n_vi_i] — one per sample
        swd_loss:   Tensor scalar — mean OT cost across batch (for L_ot)
    """
    gamma_list = []
    costs = []
    for b in range(B):
        h_en_b = h_en[b][en_mask[b]]   # [n_en, D]  — real tokens only
        h_vi_b = h_vi[b][vi_mask[b]]   # [n_vi, D]

        # Cost matrix: cosine distance
        h_en_n = F.normalize(h_en_b, dim=-1)
        h_vi_n = F.normalize(h_vi_b, dim=-1)
        C = 1.0 - h_en_n @ h_vi_n.T    # [n_en, n_vi], values in [0, 2]

        # Uniform marginals
        n_en, n_vi = C.shape
        mu = torch.full((n_en,), 1.0/n_en, device=C.device)
        nu = torch.full((n_vi,), 1.0/n_vi, device=C.device)

        # Log-domain Sinkhorn (numerically stable)
        log_K = -C / epsilon
        log_u = torch.zeros(n_en, device=C.device)
        log_v = torch.zeros(n_vi, device=C.device)
        for _ in range(n_iters):
            log_u = torch.log(mu) - torch.logsumexp(log_K + log_v[None,:], dim=1)
            log_v = torch.log(nu) - torch.logsumexp(log_K + log_u[:,None], dim=0)

        gamma_b = torch.exp(log_K + log_u[:,None] + log_v[None,:])  # [n_en, n_vi]
        gamma_list.append(gamma_b)
        costs.append((gamma_b * C).sum())

    swd_loss = torch.stack(costs).mean()
    return gamma_list, swd_loss
```

**Critical:** `gamma_b` must not have `.detach()` here — detach happens selectively
in `L_cons` only (see 3.3). `L_ot` needs gradient through cost.

### 3.2 `compute_span_loss(gamma_list, p_en_start, p_en_end, vi_logits_start, vi_logits_end, en_mask, vi_mask)`

**Purpose:** Compute $L_{\text{span}}$ = KL divergence between pseudo-label and VI prediction.

**Algorithm:**
```python
def compute_span_loss(gamma_list, p_en_start, p_en_end,
                      vi_logits_start, vi_logits_end,
                      en_mask, vi_mask):
    """
    p_en_start: Tensor[B, L_en] — softmax output from frozen EN QA head
    p_en_end:   Tensor[B, L_en]
    vi_logits_*: Tensor[B, L_vi] — raw logits from trainable VI QA head
    Returns: scalar L_span
    """
    kl_losses = []
    for b in range(B):
        n_en = en_mask[b].sum().item()
        n_vi = vi_mask[b].sum().item()
        gamma_b = gamma_list[b]                          # [n_en, n_vi]

        # Project EN probabilities to VI space
        p_en_start_b = p_en_start[b][en_mask[b]]        # [n_en]
        p_en_end_b   = p_en_end[b][en_mask[b]]

        pseudo_start = gamma_b.T @ p_en_start_b         # [n_vi]
        pseudo_end   = gamma_b.T @ p_en_end_b

        # Normalize pseudo-labels (gamma rows sum to 1, but projection
        # may introduce small numerical errors)
        pseudo_start = pseudo_start / (pseudo_start.sum() + 1e-8)
        pseudo_end   = pseudo_end   / (pseudo_end.sum()   + 1e-8)

        # VI predictions on real tokens only
        vi_log_p_start = F.log_softmax(vi_logits_start[b][vi_mask[b]], dim=-1)
        vi_log_p_end   = F.log_softmax(vi_logits_end[b][vi_mask[b]],   dim=-1)

        # KL(pseudo || vi_pred) — pseudo is the "true" distribution
        kl_s = F.kl_div(vi_log_p_start, pseudo_start.detach(), reduction='sum')
        kl_e = F.kl_div(vi_log_p_end,   pseudo_end.detach(),   reduction='sum')
        kl_losses.append((kl_s + kl_e) / 2.0)

    return torch.stack(kl_losses).mean()
```

**Note:** `pseudo_start.detach()` — pseudo-labels are targets, not parameters.
Gradient flows only through `vi_log_p_start`.

### 3.3 `compute_cons_loss(gamma_list, h_en, h_vi, en_mask, vi_mask)`

**Purpose:** Feature-space consistency — prevent VI hidden states from drifting
away from EN space. Uses γ as soft alignment target.

**Algorithm:**
```python
def compute_cons_loss(gamma_list, h_en, h_vi, en_mask, vi_mask):
    """
    Returns: scalar L_cons
    """
    mse_losses = []
    for b in range(B):
        gamma_b  = gamma_list[b]                      # [n_en, n_vi]
        h_en_b   = h_en[b][en_mask[b]]               # [n_en, D]
        h_vi_b   = h_vi[b][vi_mask[b]]               # [n_vi, D]

        # Target: weighted EN features projected to VI positions
        # MUST detach — EN backbone frozen; gradient through h_en is dropped anyway,
        # but explicit detach makes intent clear and prevents silent no-ops
        target = (gamma_b.T @ h_en_b).detach()       # [n_vi, D]

        mse_losses.append(F.mse_loss(h_vi_b, target))

    return torch.stack(mse_losses).mean()
```

**Critical:** `.detach()` on target is mandatory. Without it, gradient flows into
frozen EN backbone and is silently dropped — loss value appears correct but
gradient computation is wasted.

### 3.4 Total Stage 2 loss

```python
# In Stage 2 training loop (not inside losses.py — kept in train_stage2.py
# for visibility of curriculum schedule)

gamma_list, L_ot   = sinkhorn_masked(h_en, h_vi, en_mask, vi_mask, epsilon, n_iters)
L_span             = compute_span_loss(gamma_list, p_en_start, p_en_end,
                                       vi_logits_start, vi_logits_end,
                                       en_mask, vi_mask)
L_cons             = compute_cons_loss(gamma_list, h_en, h_vi, en_mask, vi_mask)

w_cons = min(1.0, max(0.0, (global_step - cons_delay) / cons_warmup))

L_total = (lambda_ot   * L_ot
         + lambda_span * L_span
         + lambda_cons * w_cons * L_cons)
```

---

## 4. S2-03 — Training Loop (`train_stage2.py`)

### 4.1 Checkpoint loading

```python
# Load Stage 1 checkpoint
ckpt = torch.load(config["stage1_ckpt"], map_location=device)
model.load_state_dict(ckpt["model_state"])
criterion.load_state_dict(ckpt["criterion_state"])

# Freeze EN backbone immediately after loading
for p in model.backbone.parameters():
    p.requires_grad_(False)
model.backbone.eval()   # also set eval mode — disables dropout in EN branch
```

`model.backbone.eval()` must be called at the start of every epoch to ensure
dropout in EN branch stays disabled even if `model.train()` is called globally.

### 4.2 Two forward passes per batch

```python
# EN branch — no gradient
with torch.no_grad():
    en_outputs   = model(en_batch)          # uses en_input_ids, en_attention_mask
    h_en         = en_outputs["hidden"]     # [B, L_en, D]
    p_en_start   = F.softmax(en_outputs["start_logits"], dim=-1)
    p_en_end     = F.softmax(en_outputs["end_logits"],   dim=-1)

# VI branch — with gradient
vi_outputs       = model(vi_batch)          # uses vi_input_ids, vi_attention_mask
h_vi             = vi_outputs["hidden"]     # [B, L_vi, D]
vi_logits_start  = vi_outputs["start_logits"]
vi_logits_end    = vi_outputs["end_logits"]
```

`model_core.py` must support separate batches for EN and VI. If current
`forward()` expects a single combined batch, agent must add a `branch` argument:
```python
def forward(self, batch, branch="both"):
    # branch="en" → only process en_input_ids/en_attention_mask
    # branch="vi" → only process vi_input_ids/vi_attention_mask
    # branch="both" → Stage 1 behavior (unchanged)
```

### 4.3 Differential learning rates

```python
optimizer = AdamW([
    {"params": list(model.backbone.parameters()),   # frozen — no grad, but keep in group
     "lr": 0.0},                                    # effectively disabled
    {"params": [model.layer_weights],
     "lr": config["head_lr"]},
    {"params": list(criterion.parameters()),        # QA Head
     "lr": config.get("stage2_head_lr", 5e-5)},
], weight_decay=config["weight_decay"])
```

Agent decision authority: if backbone params in optimizer group with lr=0.0
causes issues, alternatively remove backbone params entirely from optimizer.
Either approach is acceptable.

### 4.4 Curriculum schedule

```python
# Delay periods (in global steps)
OT_DELAY    = steps_per_epoch // 4    # L_ot starts immediately (no delay)
CONS_DELAY  = steps_per_epoch // 2    # L_cons delayed by 50% of epoch 1
CONS_WARMUP = steps_per_epoch         # L_cons ramps over 1 full epoch

# w_cons at each step:
w_cons = max(0.0, min(1.0, (global_step - CONS_DELAY) / CONS_WARMUP))
```

### 4.5 Default hyperparameters

```python
STAGE2_CONFIG = {
    "lambda_ot"       : 1.0,
    "lambda_span"     : 1.0,
    "lambda_cons"     : 0.5,
    "stage2_head_lr"  : 5e-5,
    "backbone_lr"     : 1e-5,   # for future unfreeze experiments
    "epsilon"         : 0.1,    # Sinkhorn regularization
    "sinkhorn_iters"  : 50,
    "batch_size"      : 16,     # smaller than Stage 1 — two forward passes per step
    "max_epochs"      : 10,
    "patience"        : 3,      # early stopping patience (epochs)
    "min_delta_em"    : 0.5,    # minimum EM improvement to reset patience
    "en_em_safety"    : 5.0,    # hard stop if EN EM drops more than this
}
```

---

## 5. S2-04/05 — Evaluation & Early Stopping

### 5.1 Validation loop (end of every epoch)

```python
def eval_xquad_vi(model, criterion, tokenizer, val_pairs, device, n_samples=180):
    """
    val_pairs: list of {"question": str, "context_vi": str, "answers": [...]}
    Returns: EM score (float, 0-100)
    """
    model.eval()
    model.backbone.eval()
    # Standard span extraction on VI input
    # Compare predicted span string to ground-truth answers
    # Use exact_match_score from SQuAD evaluation script
    ...
```

### 5.2 EN regression check (end of every epoch)

```python
def eval_squad_en_quick(model, criterion, tokenizer, squad_dev, device, n_samples=200):
    """Reuse existing quick_eval.py infrastructure."""
    ...
```

### 5.3 γ entropy monitor (every 50 steps, main process only)

```python
def gamma_entropy(gamma_list):
    """
    Returns mean entropy of transport plans across batch.
    Healthy range: 2.0 – 4.0 (depends on sequence length).
    Alert if: entropy > 6.0 (approaching uniform) or < 0.5 (collapsed).
    """
    entropies = []
    for gamma_b in gamma_list:
        H = -(gamma_b * (gamma_b + 1e-10).log()).sum()
        entropies.append(H.item())
    return sum(entropies) / len(entropies)
```

Log to TensorBoard as `"Debug/Gamma_Entropy"`.

### 5.4 Early stopping logic

```python
best_vi_em     = 0.0
patience_count = 0

for epoch in range(1, config["max_epochs"] + 1):
    run_stage2_epoch(...)

    vi_em = eval_xquad_vi(...)
    en_em = eval_squad_en_quick(...)

    # Safety check
    if (stage1_en_em - en_em) > config["en_em_safety"]:
        log.warning(f"EN EM dropped {stage1_en_em - en_em:.1f} pts — hard stop")
        break

    # Primary stopping criterion
    if vi_em > best_vi_em + config["min_delta_em"]:
        best_vi_em     = vi_em
        patience_count = 0
        save_best_checkpoint()
    else:
        patience_count += 1
        if patience_count >= config["patience"]:
            log.info(f"Early stopping at epoch {epoch} — VI EM={vi_em:.2f}")
            break
```

`stage1_en_em` must be computed once before Stage 2 starts (eval on 200 SQuAD
EN samples using Stage 1 checkpoint) and stored as baseline.

---

## 6. S2-06 — TensorBoard Logging

Log the following scalars every `log_every` steps:

```
Loss/Stage2_Total
Loss/OT
Loss/Span
Loss/Cons
Lambda/Cons_Weight        ← w_cons curriculum value
Debug/Gamma_Entropy
Learning_Rate/Head
```

Log the following scalars every epoch:

```
Eval/XQuAD_VI_EM
Eval/SQuAD_EN_EM_Quick
```

---

## 7. No-Touch Zones

The following must NOT be modified by the agent:

| File / Component                        | Reason                                      |
|-----------------------------------------|---------------------------------------------|
| `phase1_dataloader/`                    | Stage 1 dataloader — unchanged              |
| `phase4-evaluation/quick_eval.py` logic | Extend only — do not refactor existing code |
| `train.py` (Stage 1 loop)               | No changes to Stage 1 training              |
| Stage 1 checkpoint files (`*.pt`)       | Read-only                                   |
| `phase2_model/model_core.py` backbone   | Additive change only (`branch` arg) — do not alter existing `forward()` behavior for `branch="both"` |
| ViQuAD dataset files                    | Must not appear in any training DataLoader  |

---

## 8. Lightweight Verification Checklist (no large compute)

Agent must run these checks before declaring implementation complete:

```python
# CHECK 1: EN backbone truly frozen
model.backbone.eval()
p_before = next(model.backbone.parameters()).data.clone()
# run one Stage 2 step
p_after  = next(model.backbone.parameters()).data
assert torch.allclose(p_before, p_after), "EN backbone modified — FAIL"

# CHECK 2: gamma rows sum to ~1 (valid transport plan)
for gamma_b in gamma_list:
    row_sums = gamma_b.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3), \
        f"gamma row sums off: {row_sums}"

# CHECK 3: pseudo-label sums to ~1
assert abs(pseudo_start.sum().item() - 1.0) < 1e-3, "pseudo_start not normalized"

# CHECK 4: L_cons target has no gradient
target = (gamma_b.T @ h_en_b).detach()
assert not target.requires_grad, "L_cons target must be detached"

# CHECK 5: L_span KL is non-negative
assert L_span.item() >= 0, f"KL divergence negative: {L_span.item()}"

# CHECK 6: val split never in train loader
train_ids = set(train_dataset.indices)
val_ids   = set(val_dataset.indices)
assert len(train_ids & val_ids) == 0, "Data leakage: val in train"
```

---

## 9. Open Decisions (agent has authority)

- Whether to subclass `OTAlignmentLoss` or create a new `Stage2Loss` class in `losses.py`
- Exact collate_fn implementation for variable-length EN/VI sequences in same batch
- Whether `gamma_list` is returned as List[Tensor] or padded into a single Tensor[B, n_en, n_vi] with masking
- Implementation of `branch` arg in `model_core.py` (single forward with masking, or truly separate paths)
- Log format for gamma entropy (per-sample histogram vs scalar mean)

---

*End of STAGE2_SPEC.md*
*Version: 0.1 — 2026-06*