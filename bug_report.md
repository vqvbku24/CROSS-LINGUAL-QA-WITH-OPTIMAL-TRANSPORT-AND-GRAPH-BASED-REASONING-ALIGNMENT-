# BUGFIX SPEC — train.py (pre-full-train fixes)

> **For coding agent.** 3 fixes required before running `--mode train`.
> Fix in priority order. Do not touch anything outside the specified locations.

---

## Summary Table

| # | Priority | File | Function | Type | Impact |
|---|----------|------|----------|------|--------|
| 1 | **BUG** | `train.py` | `run_overfit_full()` | Wrong cap value | `L_cons` capped at 0.1 instead of 0.15 — never reaches design target |
| 2 | **RISK** | `train.py` | `run_overfit_full()` + `run_training()` | Missing grad clip | `layer_weights` gradient uncontrolled during spikes |
| 3 | **MINOR** | `train.py` | `run_training()` | Logic offset | `global_step` off-by-one causes curriculum to shift by 1 step |

---

## FIX 1 — BUG: `_cons_max` capped too low in `run_overfit_full`

**File:** `train.py`  
**Function:** `run_overfit_full()`  
**Root cause:** `min(0.1, config["lambda_cons"])` evaluates to `0.1` when `lambda_cons=0.15`, silently capping consistency loss below its designed maximum. This is why `cons` plateaued at ~0.45 instead of continuing to decrease.

### Find this line (inside the step loop):
```python
_cons_max = min(0.1, config["lambda_cons"])
```

### Replace with:
```python
_cons_max = config["lambda_cons"]
```

> **Note:** This line is recalculated every step inside the loop — move it outside the loop after the fix (just before `for step in range(...):`) for efficiency, but correctness-only fix is the one-line change above.

---

## FIX 2 — RISK: `layer_weights` gradient never clipped

**File:** `train.py`  
**Functions:** `run_overfit_full()` AND `run_training()`  
**Root cause:** Gradient clipping covers `backbone_params` and `head_params` but skips `layer_w_params`. During spikes (e.g. step 380: `gn_bb=15.2`), `layer_weights` can receive large uncontrolled gradients.

### In `run_overfit_full()` — find:
```python
gn_bb = torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=0.15).item()
gn_head = torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.5).item()
```

### Replace with:
```python
gn_bb   = torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=0.15).item()
gn_lw   = torch.nn.utils.clip_grad_norm_(layer_w_params,  max_norm=1.0).item()
gn_head = torch.nn.utils.clip_grad_norm_(head_params,     max_norm=1.5).item()
```

### In `run_training()` — find:
```python
torch.nn.utils.clip_grad_norm_(backbone_params, config["max_grad_norm"] * 0.15)
torch.nn.utils.clip_grad_norm_(head_params,     config["max_grad_norm"] * 1.5)
```

### Replace with:
```python
torch.nn.utils.clip_grad_norm_(backbone_params, config["max_grad_norm"] * 0.15)
torch.nn.utils.clip_grad_norm_(layer_w_params,  config["max_grad_norm"] * 1.0)
torch.nn.utils.clip_grad_norm_(head_params,     config["max_grad_norm"] * 1.5)
```

---

## FIX 3 — MINOR: `global_step` off-by-one in `run_training`

**File:** `train.py`  
**Function:** `run_training()`  
**Root cause:** `global_step` is incremented *after* `optimizer.step()`, so curriculum uses `current_step = global_step + 1` as a workaround. This makes the code harder to reason about and is easy to break if curriculum logic is touched later.

### Find this block (inside `if (step + 1) % config["grad_accum_steps"] == 0:`):
```python
optimizer.step()
if scheduler is not None:
    scheduler.step()
optimizer.zero_grad()
global_step += 1
```

### Replace with:
```python
optimizer.step()
if scheduler is not None:
    scheduler.step()
optimizer.zero_grad()
global_step += 1  # increment first
```

### Then find ALL occurrences of `current_step = global_step + 1` in `run_training` and the curriculum block that uses it:
```python
current_step = global_step + 1

if current_step <= _OT_DELAY:
    _criterion.lambda_ot = 0.0
elif current_step <= _OT_DELAY + _OT_WARMUP:
    _criterion.lambda_ot = config["lambda_ot"] * (current_step - _OT_DELAY) / _OT_WARMUP
else:
    _criterion.lambda_ot = config["lambda_ot"]

if current_step <= _SPAN_DELAY:
    _criterion.lambda_span = 0.0
elif current_step <= _SPAN_DELAY + _SPAN_WARMUP:
    _criterion.lambda_span = config["lambda_span"] * (current_step - _SPAN_DELAY) / _SPAN_WARMUP
else:
    _criterion.lambda_span = config["lambda_span"]

if current_step <= _CONS_DELAY:
    _criterion.lambda_cons = 0.0
elif current_step <= _CONS_DELAY + _CONS_WARMUP:
    _criterion.lambda_cons = _CONS_MAX * (current_step - _CONS_DELAY) / _CONS_WARMUP
else:
    _criterion.lambda_cons = _CONS_MAX
```

### Replace with (remove `current_step`, use `global_step` directly):
```python
if global_step <= _OT_DELAY:
    _criterion.lambda_ot = 0.0
elif global_step <= _OT_DELAY + _OT_WARMUP:
    _criterion.lambda_ot = config["lambda_ot"] * (global_step - _OT_DELAY) / _OT_WARMUP
else:
    _criterion.lambda_ot = config["lambda_ot"]

if global_step <= _SPAN_DELAY:
    _criterion.lambda_span = 0.0
elif global_step <= _SPAN_DELAY + _SPAN_WARMUP:
    _criterion.lambda_span = config["lambda_span"] * (global_step - _SPAN_DELAY) / _SPAN_WARMUP
else:
    _criterion.lambda_span = config["lambda_span"]

if global_step <= _CONS_DELAY:
    _criterion.lambda_cons = 0.0
elif global_step <= _CONS_DELAY + _CONS_WARMUP:
    _criterion.lambda_cons = _CONS_MAX * (global_step - _CONS_DELAY) / _CONS_WARMUP
else:
    _criterion.lambda_cons = _CONS_MAX
```

> **Note:** The curriculum block must be moved to run **after** `global_step += 1` (i.e. inside the `if (step + 1) % grad_accum_steps == 0:` block, after the increment). Currently it runs before — that's the source of the offset.

---

## No-touch zones

- All optimizer param groups and their `lr` values — intentional design
- Curriculum delay values (`_OT_DELAY`, `_SPAN_DELAY`, `_CONS_DELAY`) — do not change
- `lambda_ot`, `lambda_span`, `lambda_cons` default values in `DEFAULT_CONFIG` — do not change
- All DDP / distributed logic — do not touch
- `run_overfit()` (frozen backbone mode) — unaffected by these fixes, do not modify

---