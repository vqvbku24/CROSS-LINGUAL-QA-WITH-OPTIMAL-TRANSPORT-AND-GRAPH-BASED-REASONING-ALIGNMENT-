# BUGFIX SPEC — Cross-Lingual OT QA Model

> **For coding agent.** Fix 3 bugs in priority order. Each section includes: file, location, problem, exact fix.

---

## BUG 1 — CRITICAL: Index out-of-bounds after dynamic truncation

**File:** `losses.py`  
**Function:** `OTAlignmentLoss.forward()`  
**Trigger:** Any sample where `en_start_position` or `en_end_position >= T_en` (truncated length)

### Problem

`model_core.py` truncates hidden states to `T_en = max(en_attention_mask.sum())`, which is often 200–350 tokens. But `en_start_position` and `en_end_position` in the batch are token indices in the original 512-token space. If any sample has `en_start >= T_en`, the following lines crash with `IndexError`:

```python
# losses.py — span_projection_loss()
start_mass = gamma[batch_idx, en_start, :]   # CRASH if en_start >= T_en

# losses.py — OTAlignmentLoss.forward() — qa_loss()
en_start_logits[answerable_mask]             # shape (B_ans, T_en)
en_start[answerable_mask]                    # values may be >= T_en → CE crash
```

### Fix

In `OTAlignmentLoss.forward()`, after reading `en_start` and `en_end` from batch and before any loss computation, add:

```python
en_start = batch["en_start_position"]
en_end   = batch["en_end_position"]

# Clamp to truncated sequence length — prevents IndexError
en_seq_len = en_hidden.size(1)   # T_en after dynamic truncation
en_start = en_start.clamp(max=en_seq_len - 1)
en_end   = en_end.clamp(max=en_seq_len - 1)
```

> **Note:** `en_end` must also be clamped, not just `en_start`.

---

## BUG 2 — CRASH ON INIT: Missing `hidden_size` attribute on model

**File:** `model_core.py`  
**Class:** `CrossLingualOTModel.__init__()`  
**Trigger:** `train.py` calls `model.backbone.hidden_size` — attribute does not exist

### Problem

`train.py` reads:
```python
criterion = OTAlignmentLoss(
    hidden_size = model.backbone.hidden_size,  # AttributeError
    ...
)
```

`CrossLingualOTModel` uses `AutoModel` directly (not `SharedBackbone`), and `AutoModel` does not expose a `hidden_size` attribute — only `model.backbone.config.hidden_size` exists.

### Fix

In `CrossLingualOTModel.__init__()`, add one line after `self.backbone = AutoModel.from_pretrained(...)`:

```python
self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True)
self.hidden_size = self.backbone.config.hidden_size   # ADD THIS LINE — 768 (base) / 1024 (large)
self.layer_weights = nn.Parameter(torch.ones(4))
```

---

## BUG 3 — STALE FILE: `backbone.py` contradicts `model_core.py`

**File:** `backbone.py`  
**Risk:** Silent wrong behavior if any code imports `SharedBackbone` instead of using `model_core.py` directly

### Problem

`backbone.py` defines `SharedBackbone` with `output_hidden_states=False`:
```python
self.encoder = AutoModel.from_pretrained(
    model_name,
    output_hidden_states=False,   # ← WRONG for layer mixing
)
```

`model_core.py` does NOT use `SharedBackbone` — it creates its own `AutoModel` inline with `output_hidden_states=True`. The two files are out of sync. If any import accidentally uses `SharedBackbone`, `out.hidden_states` will be `None` → crash.

### Fix (choose one)

**Option A — Delete the file (recommended):**
```bash
rm backbone.py   # or modules/backbone.py depending on project structure
```

**Option B — Sync it (if backbone.py is used elsewhere):**
```python
# backbone.py — change output_hidden_states to True
self.encoder = AutoModel.from_pretrained(
    model_name,
    output_hidden_states=True,    # CHANGED
)
```

---

## Summary Table

| # | Priority | File | Type | Impact |
|---|----------|------|------|--------|
| 1 | CRITICAL | `losses.py` — `OTAlignmentLoss.forward()` | Runtime crash | Crashes during training on any batch where answer span is near end of truncated sequence |
| 2 | HIGH | `model_core.py` — `__init__()` | Crash on init | `AttributeError` on every `run_training()` call |
| 3 | LOW | `backbone.py` | Stale file | Silent wrong behavior if imported; safe to delete |

---

## No-touch zones

The following are **intentional design choices** — do not change:

- `gamma.detach()` in `ot_transport_loss()` — gradient flows through `C` only, not `gamma`
- `gamma.detach()` in `consistency_loss()` — EN is teacher, VI is student; EN logits also detached
- `layer_weights` optimizer group with `lr=1e-4` separate from backbone `lr=1e-5` — required per design spec
- Curriculum annealing logic in `run_overfit_full()` and `run_training()` — intentional, not a bug
