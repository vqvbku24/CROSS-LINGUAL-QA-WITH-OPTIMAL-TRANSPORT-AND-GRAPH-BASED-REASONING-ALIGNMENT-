# BUGFIX SPEC — span=0.0000 Every Step (Confidence Gate Too Aggressive)

> **For coding agent.** Single fix needed immediately — stop training first.
> `span_projection_loss` returns 0 for every batch because confidence_threshold=0.05
> is higher than actual γ max mass values. VI branch receives zero gradient from span loss.

---

## Observed Symptom

```
span=0.0000  (step 2230)
span=0.0000  (step 2240)
span=0.0000  (step 2250)
```

All three losses running, but `span` is completely dead.

---

## Root Cause

`confidence_threshold=0.05` in `span_projection_loss` means:
> "Only use pseudo-label if γ assigns > 5% of mass to a single VI token"

For a sequence of ~250 tokens with uniform-ish γ, expected max mass ≈ `1/250 = 0.004`.
Even a moderately peaked γ rarely exceeds 0.05 at epoch 10.

**Result:** `confident_mask` is all-False every batch → returns `torch.tensor(0.0)` → no gradient.

---

## Fix — Lower threshold to match actual γ distribution

**File:** `losses.py`  
**Function:** `span_projection_loss()`

### Option A — Change default threshold (recommended):
```python
# Find:
def span_projection_loss(
    ...
    confidence_threshold: float = 0.05,
    ...
)

# Replace with:
def span_projection_loss(
    ...
    confidence_threshold: float = 0.0,   # disable gate — use all samples
    ...
)
```

### Option B — Pass threshold via OTAlignmentLoss config (if configurable):
```python
# In OTAlignmentLoss.__init__() find:
self.span_confidence_threshold = span_confidence_threshold  # currently 0.05

# Change default in OTAlignmentLoss.__init__() signature:
def __init__(self, ..., span_confidence_threshold: float = 0.0, ...):
```

> **Which option to use:** If `OTAlignmentLoss` exposes `span_confidence_threshold`
> as a constructor parameter, use Option B (cleaner). Otherwise use Option A.

---

## Why threshold=0.0 is safe

With `threshold=0.0`, all samples pass the gate — equivalent to the original behavior
before Fix 1 was added, but with the soft/hard option still available.

The confidence gate was added to filter noisy γ pseudo-labels. However:
- γ at epoch 10 has low max mass because sequences are long (~250 tokens)
- A threshold calibrated to sequence length would be `1 / T_vi` (≈ 0.004 for T=250)
- `0.05` is ~12x too high for typical VI sequences

If you want to keep some filtering, use `threshold=0.005` (just above uniform random):
```python
confidence_threshold: float = 0.005
```

---

## Verification

After fix, restart training from epoch 10 checkpoint.
At step 2230+, `span` should be non-zero:
```
span=X.XXXX   ← any non-zero value confirms fix worked
```

Expected range at epoch 11: `span` ≈ 0.05–0.5 (similar to epoch 10 before Fix 1 was added)

---

## No-touch zones

- `soft` parameter — leave as-is
- `lambda_span` value — do not change
- Sinkhorn parameters — do not change
- All other loss functions — unaffected