# BUGFIX SPEC — VI Span Collapse (Agent Implementation Plan)

> **For coding agent.** Apply fixes in priority order.
> Fix 2 and Fix 6 first (no retraining needed), verify eval numbers change,
> then apply Fix 1 + Fix 4 + run overfit check before full retrain.

---

## Priority Table

| Order | Fix | File(s) | Risk | Retraining |
|-------|-----|---------|------|------------|
| **P0** | Fix 2: layer mixing in eval | `quick_eval.py`, `inference_to_json.py` | None | No |
| **P1** | Fix 6: max answer length at inference | `quick_eval.py`, `inference_to_json.py` | None | No |
| **P2** | Fix 1: confidence-gated span projection | `losses.py` | Low | Yes — overfit check |
| **P3** | Fix 4: consistency PAD masking | `losses.py` | Low | Yes — overfit check |
| **P4** | Fix 3: layer_weights logging | `train.py` | None | No |
| **P5** | Fix 5: soft span projection | `losses.py` | Medium | Yes — configurable only |

---

## FIX 2 — P0: quick_eval + inference use `last_hidden_state` (confirmed bug)

**Files:** `quick_eval.py` AND `inference_to_json.py`  
**Impact:** Every VI prediction generated so far used layer 12 output — QA head was trained on layer 6-9 mix. This alone explains poor eval numbers.

### Find in BOTH files:
```python
hidden = model.backbone(input_ids, attn_mask).last_hidden_state  # (1, L, H)
```

### Replace with:
```python
out = model.backbone(input_ids, attn_mask)
target_layers = [6, 7, 8, 9]
stacked = torch.stack([out.hidden_states[i] for i in target_layers], dim=0)  # (4, 1, L, H)
weights = torch.softmax(model.layer_weights, dim=0).view(4, 1, 1, 1)
hidden = (stacked * weights).sum(dim=0)   # (1, L, H)
```

> **Verify:** `model.backbone` must have `output_hidden_states=True`.
> Check `model_core.py` — if `AutoModel.from_pretrained(..., output_hidden_states=True)` is set, this is already correct.
> If `out.hidden_states` is None → root cause found: backbone init needs `output_hidden_states=True`.

---

## FIX 6 — P1: No max answer length constraint at inference

**Files:** `quick_eval.py` AND `inference_to_json.py`  
**Impact:** Fixes EN "end too far right" (69% predictions > gold×2) and VI single-token issue simultaneously.

### Find span decoding in BOTH files (pattern):
```python
start_idx = start_logits[0].argmax().item()
end_idx   = end_logits[0].argmax().item()
```

### Replace with:
```python
MAX_ANSWER_LEN = 30   # SQuAD/ViQuAD answers rarely exceed 30 tokens

start_idx = start_logits[0].argmax().item()

# Mask end positions: only allow [start_idx, start_idx + MAX_ANSWER_LEN]
end_logits_masked = end_logits[0].clone()
end_logits_masked[:start_idx] = float('-inf')
end_logits_masked[start_idx + MAX_ANSWER_LEN:] = float('-inf')
end_idx = end_logits_masked.argmax().item()
```

---

## FIX 1 — P2: Confidence-gated span projection

**File:** `losses.py`  
**Function:** `span_projection_loss()`  
**Impact:** Prevents noisy γ pseudo-labels from poisoning VI head.

### Step 1 — Add `confidence_threshold` parameter:
```python
# Find function signature:
def span_projection_loss(
    vi_start_logits: torch.Tensor,
    vi_end_logits: torch.Tensor,
    gamma: torch.Tensor,
    en_start: torch.Tensor,
    en_end: torch.Tensor,
) -> torch.Tensor:

# Replace with:
def span_projection_loss(
    vi_start_logits: torch.Tensor,
    vi_end_logits: torch.Tensor,
    gamma: torch.Tensor,
    en_start: torch.Tensor,
    en_end: torch.Tensor,
    confidence_threshold: float = 0.05,
) -> torch.Tensor:
```

### Step 2 — Add confidence gate INSIDE the `torch.no_grad()` block:

```python
# Find the block where hat_s_vi and hat_e_vi are computed (inside no_grad):
with torch.no_grad():
    B_ans = gamma.size(0)
    batch_idx = torch.arange(B_ans, device=gamma.device)

    start_mass = gamma[batch_idx, en_start, :]   # (B_ans, T_vi)
    end_mass   = gamma[batch_idx, en_end, :]     # (B_ans, T_vi)

    hat_s_vi = start_mass.argmax(dim=-1)         # (B_ans,)
    hat_e_vi = end_mass.argmax(dim=-1)           # (B_ans,)

    # enforce end >= start
    hat_e_vi = torch.max(hat_e_vi, hat_s_vi)

    # ADD AFTER hat_e_vi computation:
    start_confidence = start_mass[batch_idx, hat_s_vi]   # (B_ans,)
    end_confidence   = end_mass[batch_idx, hat_e_vi]     # (B_ans,)
    confident_mask   = (start_confidence > confidence_threshold) & \
                       (end_confidence > confidence_threshold)

# OUTSIDE no_grad block — use confident_mask:
if not confident_mask.any():
    return torch.tensor(0.0, device=vi_start_logits.device, requires_grad=False)

loss_start = F.cross_entropy(
    vi_start_logits[confident_mask], hat_s_vi[confident_mask]
)
loss_end = F.cross_entropy(
    vi_end_logits[confident_mask], hat_e_vi[confident_mask]
)
return (loss_start + loss_end) / 2.0
```

> **Scope note:** `confident_mask` is computed inside `no_grad` but used outside — this is correct.
> `hat_s_vi` and `hat_e_vi` are plain tensors (indices), safe to use outside `no_grad`.

---

## FIX 4 — P3: Consistency loss includes PAD positions

**File:** `losses.py`  
**Function:** `consistency_loss()`  
**Impact:** Prevents probability from leaking into PAD positions, sharpens VI distribution.

### Step 1 — Add `vi_pad_mask` parameter:
```python
# Find function signature:
def consistency_loss(
    en_start_logits: torch.Tensor,
    en_end_logits: torch.Tensor,
    vi_start_logits: torch.Tensor,
    vi_end_logits: torch.Tensor,
    gamma: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:

# Replace with:
def consistency_loss(
    en_start_logits: torch.Tensor,
    en_end_logits: torch.Tensor,
    vi_start_logits: torch.Tensor,
    vi_end_logits: torch.Tensor,
    gamma: torch.Tensor,
    temperature: float = 2.0,
    vi_pad_mask: torch.Tensor | None = None,   # (B, T_vi) True = PAD
) -> torch.Tensor:
```

### Step 2 — Mask PAD positions after computing soft targets:
```python
# Find where vi_target_start and vi_target_end are computed, then ADD:

if vi_pad_mask is not None:
    # Zero out PAD positions in soft targets
    vi_target_start = vi_target_start.masked_fill(vi_pad_mask, 0.0)
    vi_target_end   = vi_target_end.masked_fill(vi_pad_mask, 0.0)

    # Re-normalize so targets sum to 1 over valid positions
    vi_target_start = vi_target_start / vi_target_start.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    vi_target_end   = vi_target_end   / vi_target_end.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Mask logits so softmax assigns ~0 to PAD positions
    vi_start_logits = vi_start_logits.masked_fill(vi_pad_mask, -1e9)
    vi_end_logits   = vi_end_logits.masked_fill(vi_pad_mask, -1e9)
```

### Step 3 — Update caller in `OTAlignmentLoss.forward()`:
```python
# Find:
l_cons = consistency_loss(
    en_start_logits=...,
    en_end_logits=...,
    vi_start_logits=...,
    vi_end_logits=...,
    gamma=gamma[answerable_mask],
    temperature=self.temperature,
)

# Add vi_pad_mask argument:
l_cons = consistency_loss(
    en_start_logits=...,
    en_end_logits=...,
    vi_start_logits=...,
    vi_end_logits=...,
    gamma=gamma[answerable_mask],
    temperature=self.temperature,
    vi_pad_mask=vi_pad_mask[answerable_mask],   # ADD THIS
)
```

---

## FIX 3 — P4: Log layer_weights per epoch (diagnostic, no behavior change)

**File:** `train.py`  
**Location:** After epoch loss summary log (find the block that logs final epoch metrics).

```python
# ADD after epoch summary:
if is_main_process():
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
```

---

## FIX 5 — P5: Soft span projection (configurable, do not change default)

**File:** `losses.py`  
**Function:** `span_projection_loss()`  
**Note:** Add as option only — default stays `soft=False` to preserve current behavior.

```python
def span_projection_loss(
    vi_start_logits, vi_end_logits, gamma, en_start, en_end,
    confidence_threshold=0.05,
    soft=False,           # ADD: False = hard argmax (default), True = soft target
) -> torch.Tensor:

    if soft:
        with torch.no_grad():
            B_ans = gamma.size(0)
            batch_idx = torch.arange(B_ans, device=gamma.device)
            start_target = gamma[batch_idx, en_start, :]   # (B_ans, T_vi)
            end_target   = gamma[batch_idx, en_end, :]     # (B_ans, T_vi)
            start_target = start_target / start_target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            end_target   = end_target   / end_target.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        loss_start = -(start_target * F.log_softmax(vi_start_logits, dim=-1)).sum(dim=-1).mean()
        loss_end   = -(end_target   * F.log_softmax(vi_end_logits,   dim=-1)).sum(dim=-1).mean()
        return (loss_start + loss_end) / 2.0

    # ... existing hard argmax code with confidence gate (Fix 1) ...
```

---

## Verification Checklist

### After Fix 2 + Fix 6 (no retraining):
- [ ] Re-run inference on same ViQuAD dev samples
- [ ] Check: are predictions still all "Paris"?
- [ ] Check: EN EM should improve (end constraint fixes over-extension)
- [ ] If VI still collapses → confirms problem is in training, not eval pipeline

### After Fix 1 + Fix 4 (requires overfit check):
- [ ] Run `--mode overfit_full` for 400 steps
- [ ] Confirm: `span` loss still decreases (confidence gate not too aggressive)
- [ ] Confirm: `cons` loss still decreases (PAD masking not breaking gradients)
- [ ] If both pass → safe to resume full training from epoch 10 checkpoint

---

## No-touch zones

- Sinkhorn solver — do not change `epsilon` or `num_iters`
- `lambda_ot`, `lambda_span`, `lambda_cons` values
- Curriculum annealing schedule
- QA head architecture (`start_proj`, `end_proj`, cross-attention)
- DDP / distributed logic