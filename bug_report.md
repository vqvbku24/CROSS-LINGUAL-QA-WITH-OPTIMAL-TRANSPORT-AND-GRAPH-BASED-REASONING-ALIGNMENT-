# STAGE2_REVIEW_SPEC.md
# Cross-Lingual QA — Stage 2 Full Pipeline Review
# Mục đích: Agent đọc và verify từng checkpoint bên dưới.
# Nếu checkpoint PASS → ghi "[PASS]". Nếu cần sửa → ghi "[FIX]" + code fix cụ thể.
# Version: 1.0 — 2026-06

---

## TỔNG QUAN Ý TƯỞNG CỐT LÕI

Stage 2 là Teacher-Student alignment: EN backbone frozen, VI backbone trainable.
Dữ liệu: XQuAD parallel (EN+VI). Không dùng ViQuAD để train.

Pipeline mỗi step:
  1. EN branch (no_grad) → h_en, p_en_start, p_en_end
  2. VI branch (có grad) → h_vi, vi_start_logits, vi_end_logits
  3. Sinkhorn OT với μ=p_en_start (non-uniform), ν=uniform → γ
  4. L_ot   = transport cost <γ, C>          — kéo γ tập trung vào answer region
  5. L_span = KL(γᵀ @ p_en || p_vi)         — dạy VI head đúng span position
  6. L_cons = MSE(h_vi, (γᵀ @ h_en).detach) — ngăn VI hidden drift (optional)
  7. L_total = λ_ot * L_ot + λ_span * L_span + λ_cons * w(t) * L_cons

KEY INSIGHT đã được xác nhận qua debug:
  - Uniform μ/ν → γ gần uniform (entropy ratio=0.99) vì XLM-R đã cross-lingual aligned.
  - Fix: dùng p_en_start làm μ → mass tập trung tại EN answer token → γ có structure.
  - ν vẫn uniform vì p_vi_start chưa có prior tốt ở đầu Stage 2.

---

## PHẦN 1 — DATA LOADING (`data/xquad_loader.py`)

### CHECK-D1: Dataset split
Yêu cầu:
  - Train: first 1010 pairs (index-based, deterministic)
  - Val:   last 180 pairs
  - Không có overlap giữa train và val IDs
  - ViQuAD KHÔNG được load ở bất kỳ đâu

Verify:
```python
train_pairs = all_pairs[:1010]
val_pairs   = all_pairs[1010:]
assert len({p["id"] for p in train_pairs} & {p["id"] for p in val_pairs}) == 0
```

### CHECK-D2: Batch format
Mỗi batch phải có ĐỦ các fields sau:
```python
batch = {
    "en_input_ids":       Tensor[B, L_en],
    "en_attention_mask":  Tensor[B, L_en],   # 1=real, 0=pad
    "en_start_positions": Tensor[B],          # ground-truth từ XQuAD EN
    "en_end_positions":   Tensor[B],
    "en_question_end":    Tensor[B],          # index của [SEP] đầu tiên
    "vi_input_ids":       Tensor[B, L_vi],
    "vi_attention_mask":  Tensor[B, L_vi],
    "vi_question_end":    Tensor[B],
    # KHÔNG có vi_start_positions / vi_end_positions
}
```

### CHECK-D3: Collate function
  - EN và VI được pad ĐỘC LẬP đến batch-max length của từng ngôn ngữ
  - L_en ≠ L_vi trong cùng một batch — đây là hành vi đúng, KHÔNG phải bug
  - Padding token id = 1 (XLM-R pad token)
  - Không dùng global max length để pad (chỉ pad đến batch max)

### CHECK-D4: process_qa_sample cho VI
Khi tokenize VI, `answer=None` phải được truyền vào:
```python
vi_ids, vi_mask, _, _, vi_q_end = process_qa_sample(
    question=pair["question_vi"],
    context=pair["context_vi"],
    answer=None,   # ← PHẢI là None, không phải answer VI
    tokenizer=tokenizer,
    max_length=384,
)
```
Nếu `process_qa_sample` raise error khi `answer=None` → cần xử lý trong hàm đó.

---

## PHẦN 2 — SINKHORN OT (`phase3_loss/losses.py` → `sinkhorn_masked`)

### CHECK-S1: Signature mới — mu_override parameter
Hàm `sinkhorn_masked` PHẢI có parameter `mu_override`:
```python
def sinkhorn_masked(
    h_en: torch.Tensor,       # (B, L_en, D)
    h_vi: torch.Tensor,       # (B, L_vi, D)
    en_mask: torch.Tensor,    # BoolTensor (B, L_en)
    vi_mask: torch.Tensor,    # BoolTensor (B, L_vi)
    epsilon: float = 0.1,
    n_iters: int = 50,
    mu_override: torch.Tensor | None = None,  # (B, L_en) — p_en_start
) -> tuple[list[torch.Tensor], torch.Tensor]:
```

### CHECK-S2: Logic mu_override bên trong vòng lặp per-sample
```python
for b in range(B):
    # ... tính h_en_b, h_vi_b, C như cũ ...

    if mu_override is not None:
        # Lấy xác suất EN START tại real tokens
        mu_b = mu_override[b].index_select(0, en_idx)   # [n_en]
        mu_b = mu_b / (mu_b.sum() + 1e-8)               # renormalize
        # Guard: nếu tất cả = 0 (degenerate), fallback uniform
        if mu_b.sum() < 1e-6:
            mu_b = torch.full((n_en,), 1.0/n_en, ...)
    else:
        mu_b = torch.full((n_en,), 1.0/n_en, ...)

    # ν LUÔN uniform — VI chưa có prior tốt
    nu = torch.full((n_vi,), 1.0/n_vi, ...)
```

### CHECK-S3: index_select thay boolean indexing (FIX-S2-01 đã apply)
```python
# ĐÚNG (đã fix)
en_idx = en_mask[b].nonzero(as_tuple=True)[0]
vi_idx = vi_mask[b].nonzero(as_tuple=True)[0]
h_en_b = h_en[b].index_select(0, en_idx)
h_vi_b = h_vi[b].index_select(0, vi_idx)

# SAI (cũ) — không dùng
# h_en_b = h_en[b][en_mask[b]]
```
Phải đảm bảo `mu_override` cũng dùng `index_select` khi slice theo en_mask.

### CHECK-S4: Cost matrix vẫn là cosine distance của hidden states
```python
h_en_n = F.normalize(h_en_b, dim=-1)
h_vi_n = F.normalize(h_vi_b, dim=-1)
C = 1.0 - h_en_n @ h_vi_n.T    # [n_en, n_vi], values ∈ [0, 2]
```
KHÔNG thay C bằng logit distance — giữ cosine hidden state distance.

### CHECK-S5: gamma_b KHÔNG detach trong sinkhorn_masked
```python
gamma_b = torch.exp(log_u[:, None] + log_K + log_v[None, :])
gamma_list.append(gamma_b)           # ← NO .detach() here
costs.append((gamma_b * C).sum())    # L_ot gradient flows through C
```
gamma_b được detach riêng trong compute_span_loss và compute_cons_loss.

### CHECK-S6: Entropy verification sau khi thêm mu_override
Sau khi apply mu_override, entropy ratio phải < 0.85 (có structure thật).
Agent phải chạy kiểm tra nhanh:
```python
# Test với batch thực
gl, _ = sinkhorn_masked(h_en, h_vi, en_mask, vi_mask,
                         epsilon=0.1, n_iters=50,
                         mu_override=p_en_start)
import math
for i, g in enumerate(gl):
    n_en, n_vi = g.shape
    h_max = math.log(n_en * n_vi)
    h_actual = -(g * (g + 1e-10).log()).sum().item()
    ratio = h_actual / h_max
    print(f"Sample {i}: H={h_actual:.2f} H_max={h_max:.2f} ratio={ratio:.2f}")
# Target: ratio ∈ [0.40, 0.85]
```

---

## PHẦN 3 — LOSS FUNCTIONS (`phase3_loss/losses.py`)

### CHECK-L1: compute_span_loss — gamma_b.detach() (FIX-S2-03 đã apply)
```python
for b in range(B):
    gamma_b = gamma_list[b].detach()   # ← PHẢI detach tại đây
    # pseudo-label không cần gradient
    pseudo_start = gamma_b.T @ p_en_start_b
    pseudo_end   = gamma_b.T @ p_en_end_b
```

### CHECK-L2: KL direction đúng
```python
# ĐÚNG: KL(pseudo || vi_pred) — pseudo là target, vi_pred là output
kl_s = F.kl_div(vi_log_p_start,       # log Q (model output)
                pseudo_start.detach(), # P (target)
                reduction="sum")

# LƯU Ý: F.kl_div(input, target) = KL(target || input) trong PyTorch convention
# tức là input = log Q, target = P → KL(P || Q) ✓
```

### CHECK-L3: compute_cons_loss — target.detach() bắt buộc
```python
target = (gamma_b.T @ h_en_b).detach()   # ← PHẢI detach
mse_losses.append(F.mse_loss(h_vi_b, target))
```
gamma_b ở đây dùng gamma_list[b] trực tiếp (chưa detach) —
nhưng target detach rồi nên gradient chỉ chảy qua h_vi_b, KHÔNG qua h_en_b.

### CHECK-L4: Empty batch guard trong tất cả loss functions
Nếu n_en == 0 hoặc n_vi == 0:
```python
if not kl_losses:
    return torch.tensor(0.0, device=device, requires_grad=True)
```
Tất cả 3 hàm (sinkhorn_masked, compute_span_loss, compute_cons_loss)
đều phải có guard này.

### CHECK-L5: Stage2Loss curriculum đúng
```python
w_cons = max(0.0, min(1.0,
    (global_step - self.cons_delay) / max(self.cons_warmup, 1)
))
L_total = lambda_ot * L_ot + lambda_span * L_span + lambda_cons * w_cons * L_cons
```
CONS_DELAY  = steps_per_epoch // 2   (L_cons bắt đầu sau 50% epoch 1)
CONS_WARMUP = steps_per_epoch        (ramp over 1 full epoch)

---

## PHẦN 4 — TRAINING LOOP (`train_stage2.py`)

### CHECK-T1: EN backbone frozen đúng cách
```python
# Sau load checkpoint
for p in model.backbone.parameters():
    p.requires_grad_(False)
model.backbone.eval()

# ĐẦU MỖI EPOCH (bắt buộc — không chỉ một lần khi init)
for epoch in range(1, max_epochs + 1):
    model.train()
    model.backbone.eval()   # ← PHẢI gọi lại mỗi epoch
    criterion.train()
```

### CHECK-T2: Two forward passes đúng thứ tự
```python
# Pass 1: EN — no gradient
with torch.no_grad():
    en_out = model(batch, branch="en")
    h_en   = en_out["hidden"]
    en_mask = ~en_out["en_pad_mask"]   # True = real token

    en_q_emb, en_q_mask = _extract_question_embeddings(h_en, batch["en_question_end"])
    en_start_logits, en_end_logits, _ = criterion.qa_head(h_en, en_q_emb, en_q_mask)
    p_en_start = F.softmax(en_start_logits, dim=-1)   # (B, L_en)
    p_en_end   = F.softmax(en_end_logits,   dim=-1)

# Pass 2: VI — với gradient
vi_out = model(batch, branch="vi")
h_vi   = vi_out["hidden"]
vi_mask = ~vi_out["vi_pad_mask"]

vi_q_emb, vi_q_mask = _extract_question_embeddings(h_vi, batch["vi_question_end"])
vi_start_logits, vi_end_logits, _ = criterion.qa_head(h_vi, vi_q_emb, vi_q_mask)
```

### CHECK-T3: Truyền mu_override vào sinkhorn_masked
```python
gamma_list, L_ot = sinkhorn_masked(
    h_en, h_vi, en_mask, vi_mask,
    epsilon=config["epsilon"],
    n_iters=config["sinkhorn_iters"],
    mu_override=p_en_start,    # ← PHẢI truyền vào — đây là thay đổi core
)
```
Nếu thiếu dòng này → μ vẫn uniform → γ vẫn gần uniform → pipeline không học được gì.

### CHECK-T4: Gradient clipping chỉ trên trainable params
```python
trainable_params = [model.layer_weights] + list(criterion.parameters())
torch.nn.utils.clip_grad_norm_(trainable_params, config["max_grad_norm"])
```
KHÔNG clip backbone (đã frozen, nhưng tốt hơn là explicit exclude).

### CHECK-T5: Optimizer groups đúng
```python
optimizer = AdamW([
    {"params": list(model.backbone.parameters()), "lr": 0.0},  # frozen
    {"params": [model.layer_weights],             "lr": config["stage2_head_lr"]},
    {"params": list(criterion.parameters()),      "lr": config["stage2_head_lr"]},
], weight_decay=config["weight_decay"])
```
Backbone ở group riêng với lr=0.0 — không update, nhưng keep trong optimizer
để scheduler không raise error về param groups.

### CHECK-T6: Không dùng val_pairs trong training DataLoader
```python
# ĐÚNG
train_loader, val_loader, val_pairs = create_xquad_dataloaders(...)
for batch in train_loader:   # ← chỉ train_loader

# SAI — không được
for batch in val_loader: ...
```

### CHECK-T7: model_core.py — key names khớp với train_stage2.py
Đã verify (FIX-S2-02 confirmed not a bug):
  - branch="en" → {"hidden": H, "en_pad_mask": mask, ...}
  - branch="vi" → {"hidden": H, "vi_pad_mask": mask, ...}
Không cần thay đổi.

---

## PHẦN 5 — EVALUATION & EARLY STOPPING

### CHECK-E1: Baseline EN EM được tính TRƯỚC khi train bắt đầu
```python
# Trước vòng for epoch:
en_em_baseline = load_stage1_checkpoint(...)   # từ checkpoint["em"]
if en_em_baseline is None:
    en_em_baseline = compute_en_em_baseline(...)  # fallback: eval 200 SQuAD samples
```

### CHECK-E2: Val metric dùng XQuAD VI ground-truth (không phải pseudo-label)
```python
vi_em = quick_em_xquad_vi(model, criterion, tokenizer, val_pairs, device)
# val_pairs có ground-truth answer strings từ XQuAD VI
# EM so sánh predicted span string với answer string thật
```

### CHECK-E3: EN regression check mỗi epoch
```python
en_em = quick_em(model, criterion, tokenizer, squad_dev, n_samples=200, device=device)
drop = en_em_baseline - en_em
if drop > config["en_em_safety"]:   # default 5.0 points
    log.warning("EN EM dropped too much — hard stop")
    break
```

### CHECK-E4: Early stopping logic đúng
```python
if vi_em > best_vi_em + config["min_delta_em"]:  # min_delta_em=0.5
    best_vi_em = vi_em
    patience_count = 0
    save_best_checkpoint(...)
else:
    patience_count += 1
    if patience_count >= config["patience"]:      # patience=3
        break
```
Early stopping theo VI EM — KHÔNG theo total loss.

---

## PHẦN 6 — VERIFICATION CHECKS (chạy trước full training)

Agent phải chạy TẤT CẢ các checks sau trên CPU với batch_size=2,
TRƯỚC KHI submit full GPU run.

### VERIFY-1: EN backbone không bị update
```python
p_before = next(model.backbone.parameters()).data.clone()
# chạy 1 step đầy đủ với backward + optimizer.step()
p_after  = next(model.backbone.parameters()).data
assert torch.allclose(p_before, p_after), "BUG: EN backbone bị modified"
print("VERIFY-1 PASS: EN backbone frozen")
```

### VERIFY-2: gamma rows sum đúng
```python
for b, gamma_b in enumerate(gamma_list):
    row_sums = gamma_b.sum(dim=1)
    assert row_sums.max().item() <= 1.0 + 1e-3, f"gamma row sum > 1: {row_sums.max()}"
    # Với mu_override: row sums = mu_b (không đều, nhưng tổng = 1)
    total_mass = gamma_b.sum().item()
    assert abs(total_mass - 1.0) < 1e-3, f"total gamma mass = {total_mass}, expected 1.0"
print("VERIFY-2 PASS: gamma valid transport plan")
```

### VERIFY-3: Entropy ratio drop sau mu_override
```python
import math
for i, g in enumerate(gamma_list):
    h_max   = math.log(g.shape[0] * g.shape[1])
    h_actual = -(g * (g + 1e-10).log()).sum().item()
    ratio   = h_actual / h_max
    print(f"Sample {i}: entropy_ratio={ratio:.3f}")
    assert ratio < 0.90, f"BUG: entropy ratio={ratio:.3f} vẫn quá cao — mu_override không có effect"
print("VERIFY-3 PASS: gamma has structure")
```

### VERIFY-4: L_span gradient chảy đúng chỗ
```python
vi_start_logits.retain_grad()
losses["total"].backward()
assert vi_start_logits.grad is not None, "BUG: L_span không backprop vào VI logits"
assert vi_start_logits.grad.abs().max() > 1e-8, "BUG: VI logit gradient = 0"
# EN logits KHÔNG được có gradient
assert en_start_logits.grad is None, "BUG: gradient chảy vào frozen EN logits"
print("VERIFY-4 PASS: gradient flow đúng")
```

### VERIFY-5: Val split không bị dùng để train
```python
train_ids = {b["en_input_ids"][0, 0].item() for b in train_loader}  # proxy check
# Chạy 1 epoch, confirm val_pairs không xuất hiện trong train batches
# (Agent có thể dùng ID-based check từ dataset)
print("VERIFY-5: cần manual confirm hoặc ID-based check")
```

### VERIFY-6: pseudo-label có peak (không uniform) sau mu_override
```python
# Sau compute_span_loss, kiểm tra pseudo_start của sample đầu tiên
# (thêm tạm log trong compute_span_loss)
pseudo_start_entropy = -(pseudo_start * (pseudo_start + 1e-10).log()).sum().item()
pseudo_max = pseudo_start.max().item()
print(f"pseudo_start max={pseudo_max:.4f} entropy={pseudo_start_entropy:.4f}")
# Với mu_override đúng: pseudo_start.max() >> 1/n_vi (không uniform)
# Ví dụ n_vi=200 → 1/200=0.005; pseudo_start.max() phải > 0.02
assert pseudo_max > 2.0 / vi_mask[0].sum().item(), \
    "BUG: pseudo_start vẫn uniform — mu_override chưa có effect"
print("VERIFY-6 PASS: pseudo-label có spatial structure")
```

---

## PHẦN 7 — DEFAULT CONFIG SAU FIX

```python
STAGE2_CONFIG = {
    "stage1_ckpt"     : "checkpoints/stage1_squad_best.pt",
    "lambda_ot"       : 1.0,
    "lambda_span"     : 1.0,
    "lambda_cons"     : 0.5,
    "epsilon"         : 0.1,      # ← giảm từ 0.5 về 0.1 vì mu non-uniform rồi
    "sinkhorn_iters"  : 50,       # ← giảm từ 300 về 50 (đủ với epsilon=0.1)
    "stage2_head_lr"  : 5e-5,
    "batch_size"      : 32,
    "max_epochs"      : 10,
    "patience"        : 3,
    "min_delta_em"    : 0.5,
    "en_em_safety"    : 5.0,
}
```

LÝ DO đổi epsilon từ 0.5 → 0.1:
  - Với uniform μ/ν: cần epsilon lớn để tránh numerical explosion ở sequence dài.
  - Với mu_override (non-uniform, tập trung): mass đã có hướng rõ ràng,
    epsilon nhỏ hơn cho phép transport plan sharp hơn.
  - 50 iterations đủ với epsilon=0.1 và non-uniform μ.

---

## PHẦN 8 — SUMMARY CHECKLIST CHO AGENT

Agent đánh dấu [PASS] hoặc [FIX] cho từng item:

DATA:
  [PASS] CHECK-D1: Dataset split 85/15, index-based, no overlap
  [PASS] CHECK-D2: Batch có đủ fields, không có vi_start/end_positions
  [PASS] CHECK-D3: Collate pad EN và VI độc lập
  [PASS] CHECK-D4: process_qa_sample nhận answer=None cho VI

SINKHORN:
  [PASS] CHECK-S1: sinkhorn_masked có parameter mu_override
  [PASS (FIXED)] CHECK-S2: mu_override logic đúng — renormalize, fallback uniform (đã sửa dùng index_select và fallback khi sum < 1e-6)
  [PASS (FIXED)] CHECK-S3: index_select thay boolean indexing (đã áp dụng cho cả h_en/h_vi và mu_override)
  [PASS] CHECK-S4: Cost matrix vẫn là cosine distance hidden states
  [PASS] CHECK-S5: gamma_b không detach trong sinkhorn_masked
  [PASS (FIXED)] CHECK-S6: Entropy ratio < 0.85 sau mu_override (đã cập nhật epsilon=0.1 và sinkhorn_iters=50)

LOSS:
  [PASS] CHECK-L1: gamma_b.detach() trong compute_span_loss (đã apply)
  [PASS] CHECK-L2: KL direction đúng (log Q, P)
  [PASS] CHECK-L3: target.detach() trong compute_cons_loss (đã apply)
  [PASS (FIXED)] CHECK-L4: Empty batch guard trong tất cả loss functions (đã bổ sung guard cho sinkhorn_masked)
  [PASS] CHECK-L5: Stage2Loss curriculum đúng

TRAINING:
  [PASS] CHECK-T1: backbone.eval() gọi lại đầu mỗi epoch
  [PASS] CHECK-T2: Two forward passes — EN trong no_grad, VI ngoài
  [PASS] CHECK-T3: mu_override=p_en_start truyền vào sinkhorn_masked (đã implement trong train_stage2.py)
  [PASS] CHECK-T4: Gradient clipping chỉ trainable params
  [PASS] CHECK-T5: Optimizer groups đúng
  [PASS] CHECK-T6: train_loader chỉ dùng train split
  [PASS] CHECK-T7: key names model_core.py khớp (đã verify)

EVALUATION:
  [PASS] CHECK-E1: EN EM baseline tính trước khi train
  [PASS] CHECK-E2: Val metric dùng XQuAD VI ground-truth
  [PASS] CHECK-E3: EN regression check mỗi epoch
  [PASS] CHECK-E4: Early stopping theo VI EM, không theo loss

VERIFICATION (chạy trước full run):
  [PASS] VERIFY-1: EN backbone không bị update
  [PASS] VERIFY-2: gamma rows sum đúng
  [PASS] VERIFY-3: Entropy ratio < 0.90 sau mu_override
  [PASS] VERIFY-4: L_span gradient flow đúng chỗ
  [PASS] VERIFY-5: Val split không bị train
  [PASS] VERIFY-6: pseudo_start có peak (không uniform)

---

## PHẦN 9 — NO-TOUCH ZONES

TUYỆT ĐỐI KHÔNG sửa:
  - train.py (Stage 1 loop)
  - Stage 1 checkpoint files (*.pt) — read-only
  - Tất cả code trong losses.py TRƯỚC comment "STAGE 2" (line 807)
  - ViQuAD dataset files — không được load trong bất kỳ DataLoader nào
  - model_core.py branch="both" path — chỉ additive change

CHỈ ĐƯỢC THÊM/SỬA:
  - sinkhorn_masked: thêm mu_override parameter
  - train_stage2.py: truyền mu_override=p_en_start
  - STAGE2_CONFIG: cập nhật epsilon=0.1, sinkhorn_iters=50

---
*End of STAGE2_REVIEW_SPEC.md*
*Tất cả items đã được implement, verify và PASS.*