# IDEA SPEC — Cross-Lingual Extractive QA via Sinkhorn Optimal Transport

> **For coding agent.** This document describes the full architecture and training objective.
> Read this before touching any model/loss/training code.

---

## 1. Problem Statement

**Goal:** Transfer span extraction ability from English (supervised) to Vietnamese (zero-shot) using token-level alignment via Optimal Transport.

**Key constraint:** Vietnamese data has NO answer labels — only question + context. All supervision comes from the English side.

**Inputs per training sample:**
- `EN`: question + context + answer span `(start_idx, end_idx)` — supervised
- `VI`: question + context only — no labels

---

## 2. Architecture Overview

```
EN input ──┐
           ├──► Shared XLM-RoBERTa ──► Layer Mix (6,7,8,9) ──► H_en [B, T_en, 768]
VI input ──┘                                                  ──► H_vi [B, T_vi, 768]
                                                                        │
                                              Cosine Distance Cost Matrix C [B, T_en, T_vi]
                                                                        │
                                                         Sinkhorn OT (50 iters, log-domain)
                                                                        │
                                                              γ [B, T_en, T_vi]
                                              ┌─────────────────────────┼──────────────────────┐
                                         Span Projection           Consistency             OT Cost
                                         (hard pseudo-label)       (soft KL)              (L_ot)
                                              │                         │
                                         L_span_proj               L_cons
                                                                        
                                    QA Head (shared EN+VI)
                                         │
                                    L_qa (EN only, supervised)
```

---

## 3. Shared Encoder — Learnable Layer Weighting

**Model:** `xlm-roberta-base` (or large). One encoder shared for both EN and VI.

**Layer selection:** Layers 6, 7, 8, 9 — capture syntactic/semantic structure (justified by Clark et al. 2019).

**Implementation:**
```python
self.layer_weights = nn.Parameter(torch.ones(4))   # learns best layer combination

weights = torch.softmax(self.layer_weights, dim=0).view(4, 1, 1, 1)
stacked = torch.stack([hidden_states[i] for i in [6,7,8,9]], dim=0)  # (4, B, L, H)
H = (stacked * weights).sum(dim=0)                 # (B, L, H)
```

**Optimizer:** `layer_weights` must be in its own param group with `lr=1e-4` (separate from backbone `lr=1e-5`).

---

## 4. Dynamic Sequence Truncation

Instead of always using the full 512-token sequence, truncate to the actual max non-PAD length in each batch:

```python
T_en = int(en_attention_mask.sum(dim=1).max())   # e.g. 280 instead of 512
T_vi = int(vi_attention_mask.sum(dim=1).max())

H_en = H_en[:, :T_en, :]    # (B, T_en, H)
H_vi = H_vi[:, :T_vi, :]    # (B, T_vi, H)
```

**Why:** Reduces Sinkhorn cost matrix from `O(512²)` → `O(T_en × T_vi)`, saving 50-75% memory/compute on typical QA sequences (200-350 tokens).

> **CRITICAL:** After truncation, `en_start_position` and `en_end_position` from the original 512-space must be clamped to `T_en - 1` before use in any loss or indexing.

---

## 5. Cost Matrix & PAD Masking

```python
C[b, i, j] = 1 - cosine_similarity(H_en[b, i], H_vi[b, j])   # shape: (B, T_en, T_vi)
```

PAD positions are masked to `1e4` so Sinkhorn assigns zero transport mass to them:
```python
C = C.masked_fill(en_pad_mask.unsqueeze(2), 1e4)   # mask PAD rows
C = C.masked_fill(vi_pad_mask.unsqueeze(1), 1e4)   # mask PAD cols
```

---

## 6. Sinkhorn OT Solver

**Algorithm:** Log-domain Sinkhorn-Knopp (pure PyTorch, batched), ~50 iterations.

**Marginals:** Non-uniform — `mu[i] = 1/n_valid` for valid tokens, `0` for PAD. This ensures zero mass flows through padding.

**Output:** Transport plan `γ` of shape `[B, T_en, T_vi]`.

**Key property:** `γ[b, i, j]` = how much probability mass flows from EN token `i` to VI token `j`.

```python
# Conceptually:
gamma = sinkhorn_log_domain(C, en_pad_mask, vi_pad_mask, epsilon=0.05, num_iters=50)
```

---

## 7. Knowledge Transfer via γ

### 7.1 Span Projection — Hard Pseudo-label

Maps EN answer span → VI pseudo-labels using argmax over transport mass:

```python
hat_s_vi = argmax_j gamma[b, en_start[b], j]    # where does EN start token flow to?
hat_e_vi = argmax_j gamma[b, en_end[b], j]      # where does EN end token flow to?
# Constraint: hat_e_vi >= hat_s_vi (enforced by masking positions before hat_s_vi)
```

Loss:
```python
L_span_proj = CE(vi_start_logits, hat_s_vi) + CE(vi_end_logits, hat_e_vi)
```

Applied only to **answerable** samples. `gamma` is used in `no_grad()` context (pseudo-labels are fixed targets).

### 7.2 Transport-Guided Consistency — Soft Distribution Transfer

Maps EN QA probability distribution → VI token space via `γᵀ`:

```python
P_target_vi = γᵀ @ softmax(EN_logits.detach() / T)   # (B, T_vi)
L_cons = T² × KL(softmax(VI_logits / T) || P_target_vi)   # T=2.0
```

**Gradient stops:** `EN_logits.detach()` + `gamma.detach()` — EN branch is pure teacher.

---

## 8. Loss Functions

```
L_total = L_qa
        + 0.5        × L_has_answer      (BCE on EN branch — answerable detection)
        + λ_ot       × L_ot              (transport cost regularizer)
        + λ_span     × L_span_proj       (pseudo-label VI span)
        + λ_cons     × L_consistency     (soft KL divergence)
```

| Loss | Formula | Gradient target | Applied to |
|------|---------|----------------|------------|
| `L_qa` | CE(EN start/end logits, true labels) | backbone + QA head | answerable EN only |
| `L_has_answer` | BCE(has_ans_logit, is_answerable) | QA head | all EN |
| `L_ot` | `<γ.detach(), C>` | C only (backbone) | all samples |
| `L_span_proj` | CE(VI logits, projected pseudo-labels) | VI QA head | answerable EN only |
| `L_cons` | KL(VI dist \|\| γᵀ @ EN dist) | VI backbone + head | answerable EN only |

**Default λ values:** `λ_ot=0.1`, `λ_span=0.3`, `λ_cons=0.15`

---

## 9. QA Head

Shared for EN and VI. Architecture:
- **Cross-attention:** context tokens attend to question tokens (question = `[CLS ... SEP)`)
- **Residual + LayerNorm**
- **Linear projections:** `start_proj`, `end_proj` → span logits over all T positions
- **has_answer_head:** MLP on CLS embedding → binary answerable logit (EN only)

```python
# Input shapes (after dynamic truncation):
context_hidden:  (B, T_en or T_vi, 768)
question_hidden: (B, max_q_len, 768)

# Output:
start_logits:     (B, T)
end_logits:       (B, T)
has_answer_logit: (B,)
```

---

## 10. Training Details

### Optimizer Groups
```python
AdamW([
    {"params": backbone_params,  "lr": 1e-5},
    {"params": layer_w_params,   "lr": 1e-4},   # layer_weights MUST be separate
    {"params": head_params,      "lr": 1e-4},
], weight_decay=0.01)
```

### Curriculum Annealing (full training)
Loss components are introduced gradually to prevent early collapse:
- **OT:** starts at step `SPE/2`, linearly ramps over `SPE` steps
- **Span:** starts at step `SPE`, linearly ramps over `SPE` steps  
- **Consistency:** starts at step `2×SPE`, linearly ramps over `SPE/2` steps

*(SPE = steps per epoch)*

### Gradient Clipping
```python
clip_grad_norm_(backbone_params, max_norm=0.15)   # tight — backbone is pretrained
clip_grad_norm_(head_params,     max_norm=1.5)    # looser — head trains from scratch
```

### Multi-GPU
DDP via `torchrun`. `DistributedSampler` with `set_epoch(epoch)` for proper shuffling.

---

## 11. Evaluation Datasets

| Dataset | Type | Size | Domain | Role |
|---------|------|------|--------|------|
| ViQuAD 1.0 | Extractive span | ~23,000 | Wikipedia | Main baseline — SQuAD-style |
| ViQuAD 2.0 | Extractive + unanswerable | ~35,000 | Wikipedia | Tests unanswerable handling |
| ViNewsQA | Extractive span | ~22,057 | Health/Medical | Cross-domain transfer test |

---

## 12. Novelty vs Baselines

| Feature | XLM-R Zero-shot | OPTICAL (WSDM'23) | This work |
|---------|----------------|-------------------|-----------|
| OT alignment | ✗ | Token embedding | Token embedding + Dynamic Truncation |
| Span projection via γ | ✗ | ✗ | ✓ |
| Transport consistency loss | ✗ | ✗ | ✓ |
| Task | — | Cross-lingual IR | Zero-shot Extractive QA |

**Key difference from OPTICAL:** OPTICAL uses OT only for representation alignment (IR task). This work extends OT to structured prediction — γ directly generates pseudo-labels for span extraction.

---

## 13. File Map

| File | Role |
|------|------|
| `model_core.py` | `CrossLingualOTModel` — backbone + layer mixing + cost matrix |
| `losses.py` | `OTAlignmentLoss` — Sinkhorn solver + QA head + all loss functions |
| `train.py` | Training loop — optimizer groups, curriculum annealing, DDP, checkpointing |
| `backbone.py` | **STALE — do not use.** Superseded by `model_core.py` |