# BUGFIX_SPEC.md — Cross-Lingual QA: idea_updated_V2 Implementation

> **Dành cho coding agent.** Spec này liệt kê các thay đổi cần thực hiện theo thứ tự ưu tiên.
> Agent tự quyết định thứ tự implement, chiến lược test, và cách integrate vào codebase hiện tại.
> Không được chỉnh sửa bất kỳ file nào ngoài danh sách "Allowed Files" bên dưới.

---

## Allowed Files (Touch Zone)

| File | Phạm vi được phép chỉnh sửa |
|---|---|
| `losses.py` | Toàn bộ — refactor `span_projection_loss`, cập nhật `forward` loss aggregation, thêm entropy metric |
| `train.py` | Chỉ phần argument defaults và curriculum phase-gate logic |

## No-Touch Zone

| File | Lý do |
|---|---|
| `backbone.py` | Đã ổn định sau các bug fix trước |
| `model_core.py` | Không liên quan đến thay đổi lần này |
| `dataset.py` | Không liên quan |
| `evaluate.py` | Không liên quan |

---

## Priority Table

| Priority | ID | File | Mức độ rủi ro | Mô tả ngắn |
|---|---|---|---|---|
| 🔴 P1 | `FIX-01` | `train.py` | Thấp | Reset `sinkhorn_epsilon` default về `0.05` |
| 🔴 P1 | `FIX-02` | `losses.py` | Cao | Implement toàn bộ `span_projection_loss` Soft-to-Hard Curriculum |
| 🔴 P1 | `FIX-03` | `losses.py` | Trung bình | Guard edge case all-`-inf` trong Phase 2 của `span_projection_loss` |
| 🟠 P2 | `FIX-04` | `losses.py` | Thấp | Asymmetric weighting `loss_has_ans = 0.7*EN + 0.3*VI` |
| 🟡 P3 | `FIX-05` | `losses.py` | Thấp | Thêm entropy monitoring với numerical stability clamp |
| 🟡 P3 | `FIX-06` | `losses.py` | Thấp | Log `mean_max_start_mass` và `mean_max_end_mass` lên TensorBoard |

---

## Chi tiết từng Fix

---

### FIX-01 — Reset `sinkhorn_epsilon` default

**File:** `train.py`
**Function/Argument:** `ArgumentParser` hoặc config dict chứa `sinkhorn_epsilon`

**Problem code:**
```python
# Hiện tại (sau lần tune trước)
parser.add_argument("--sinkhorn_epsilon", type=float, default=0.1)
```

**Fix:**
```python
parser.add_argument("--sinkhorn_epsilon", type=float, default=0.05)
# sinkhorn_iters GIỮ NGUYÊN 100 — không đổi
```

**Lý do:** epsilon=0.05 cho transport plan sắc nét hơn; kết hợp với iters=100 đủ để hội tụ.

---

### FIX-02 — Implement `span_projection_loss` Soft-to-Hard Curriculum

**File:** `losses.py`
**Function:** `span_projection_loss` (replace toàn bộ)

**Problem code (logic cũ — hard pseudo-label không có warm-up):**
```python
def span_projection_loss(self, vi_start_logits, vi_end_logits, gamma, en_start, en_end):
    # ... hard pseudo-label trực tiếp, không có curriculum
```

**Fix — Full replacement:**
```python
def span_projection_loss(self, vi_start_logits, vi_end_logits, gamma, en_start, en_end, global_step, spe):
    """
    Curriculum Span Loss:
      Phase 1 (step <= 4*spe): Soft supervision — dùng hàng gamma làm soft target.
      Phase 2 (step >  4*spe): Hard pseudo-label với confidence threshold = 0.25.
    """
    B_ans = gamma.size(0)
    L_vi  = gamma.size(2)
    device = gamma.device
    batch_idx = torch.arange(B_ans, device=device)

    is_hard_phase = global_step > (4 * spe)

    if not is_hard_phase:
        # ---- PHASE 1: SOFT SUPERVISION (Warm-up) ----
        with torch.no_grad():
            start_target = gamma[batch_idx, en_start, :]   # (B_ans, L_vi)
            end_target   = gamma[batch_idx, en_end,   :]   # (B_ans, L_vi)

        loss_start = -(start_target * F.log_softmax(vi_start_logits, dim=-1)).sum(dim=-1).mean()
        loss_end   = -(end_target   * F.log_softmax(vi_end_logits,   dim=-1)).sum(dim=-1).mean()
        return (loss_start + loss_end) / 2.0

    else:
        # ---- PHASE 2: HARD PSEUDO-LABELING + THRESHOLD ----
        # (xem FIX-03 cho guard all-inf bên dưới)
        with torch.no_grad():
            start_mass_dist = gamma[batch_idx, en_start, :]
            max_start_mass, hat_s_vi = start_mass_dist.max(dim=1)

            end_mass_dist = gamma[batch_idx, en_end, :]
            position_idx  = torch.arange(L_vi, device=device).unsqueeze(0)
            before_start_mask = position_idx < hat_s_vi.unsqueeze(1)
            end_mass_dist_masked = end_mass_dist.masked_fill(before_start_mask, float('-inf'))

            # GUARD (FIX-03): xử lý trường hợp toàn bộ là -inf
            # (xem FIX-03 để implement guard này)

            max_end_mass, hat_e_vi = end_mass_dist_masked.max(dim=1)

            confidence_threshold = 0.25
            valid_pseudo_mask = (
                (max_start_mass > confidence_threshold) &
                (max_end_mass   > confidence_threshold)   # -inf samples tự bị lọc ở đây
            )

        if valid_pseudo_mask.any():
            loss_start = F.cross_entropy(vi_start_logits[valid_pseudo_mask], hat_s_vi[valid_pseudo_mask])
            loss_end   = F.cross_entropy(vi_end_logits[valid_pseudo_mask],   hat_e_vi[valid_pseudo_mask])
            return (loss_start + loss_end) / 2.0
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
```

**Lưu ý cho agent:** Cần cập nhật **tất cả call-sites** của `span_projection_loss` để truyền thêm `global_step` và `spe`.

---

### FIX-03 — Guard all-`-inf` trong Phase 2

**File:** `losses.py`
**Function:** `span_projection_loss`, Phase 2, ngay sau khi tính `end_mass_dist_masked`

**Problem:** Nếu `hat_s_vi[i]` trùng với token cuối của VI sequence, `end_mass_dist_masked[i]` sẽ toàn `-inf`. PyTorch `max()` trên all-`-inf` tensor không raise error nhưng trả về index undefined — behavior này phụ thuộc backend/device.

**Fix — chèn sau dòng `end_mass_dist_masked = ...`:**
```python
# Guard: nếu toàn hàng là -inf (hat_s_vi tại vị trí cuối), cho phép end = start
all_inf_mask = (end_mass_dist_masked == float('-inf')).all(dim=1)
if all_inf_mask.any():
    # Fallback: cho end bằng start (span length = 1) cho các sample bị affected
    # Các sample này sẽ bị lọc bởi confidence_threshold vì max_end_mass = -inf < 0.25
    # Guard này chỉ ngăn undefined behavior của max() — không ảnh hưởng loss
    end_mass_dist_masked[all_inf_mask, hat_s_vi[all_inf_mask]] = 0.0
```

**Agent quyết định:** Nếu codebase có convention khác cho degenerate span (e.g., skip toàn bộ sample thay vì fallback), agent có thể implement theo convention đó — miễn là tránh được all-`-inf` input cho `max()`.

---

### FIX-04 — Asymmetric Weighting cho `loss_has_ans`

**File:** `losses.py`
**Function:** `forward` (hoặc hàm tổng hợp loss — tùy codebase đặt tên)

**Problem code (weighting đối xứng hiện tại):**
```python
# Cách cũ — EN và VI có cùng trọng số
loss_has_ans = F.binary_cross_entropy_with_logits(vi_has_logits, batch["en_is_answerable"].float())
# hoặc: loss_has_ans = (loss_has_en + loss_has_vi) / 2.0
```

**Fix:**
```python
# 1. EN branch — supervised anchor
en_cls       = H_en[:, 0, :]
en_has_logits = self.has_answer_head(en_cls).squeeze(-1)
loss_has_en   = F.binary_cross_entropy_with_logits(
    en_has_logits, batch["en_is_answerable"].float()
)

# 2. VI branch — distant supervision (chỉ có noisy label từ EN)
vi_cls        = H_vi[:, 0, :]
vi_has_logits = self.has_answer_head(vi_cls).squeeze(-1)
loss_has_vi   = F.binary_cross_entropy_with_logits(
    vi_has_logits, batch["en_is_answerable"].float()
)

# 3. Asymmetric aggregation — bảo vệ EN anchor
loss_has_ans = (0.7 * loss_has_en) + (0.3 * loss_has_vi)
```

**Lý do:** VI branch chỉ có distant supervision → không nên receive weight bằng EN. Fix này giải quyết nguyên nhân chính của `Loss/HasAnswer` oscillating.

---

### FIX-05 — Entropy Monitoring với Numerical Stability

**File:** `losses.py`
**Vị trí:** Trong `forward`, sau khi tính `vi_start_logits` / `vi_end_logits` — log lên TensorBoard cùng lúc với các loss khác.

**Problem (công thức gốc trong tài liệu — thiếu clamp):**
```python
entropy = -(P_vi * log(P_vi)).sum(dim=-1).mean()  # log(0) → -inf nếu P_vi có zero
```

**Fix:**
```python
with torch.no_grad():
    P_vi_start = F.softmax(vi_start_logits, dim=-1)
    entropy_start = -(P_vi_start * (P_vi_start + 1e-8).log()).sum(dim=-1).mean()

    P_vi_end = F.softmax(vi_end_logits, dim=-1)
    entropy_end = -(P_vi_end * (P_vi_end + 1e-8).log()).sum(dim=-1).mean()

# Log lên TensorBoard
writer.add_scalar("Metrics/VI_StartLogit_Entropy", entropy_start, global_step)
writer.add_scalar("Metrics/VI_EndLogit_Entropy",   entropy_end,   global_step)
```

**Interpretation guide cho agent:** Entropy cao = phân phối flat (model chưa confident); entropy thấp = phân phối peaked (model confident). Quan sát entropy giảm dần qua epoch là tín hiệu tốt của Phase 2 Curriculum.

---

### FIX-06 — Log Mean Transport Mass

**File:** `losses.py`
**Vị trí:** Trong Phase 2 của `span_projection_loss`, sau khi tính `max_start_mass` và `max_end_mass`.

**Fix — thêm log (không ảnh hưởng backward):**
```python
with torch.no_grad():
    # Chỉ log trên valid samples để tránh nhiễu từ -inf samples
    if valid_pseudo_mask.any():
        mean_start_mass = max_start_mass[valid_pseudo_mask].mean()
        mean_end_mass   = max_end_mass[valid_pseudo_mask].mean()
        # Trả về cùng với loss để caller có thể log, HOẶC dùng global writer
        # Agent quyết định convention log phù hợp với codebase
```

**TensorBoard keys gợi ý:**
```
Metrics/OT_MeanStartMass   (target: tăng dần về phía > 0.5 ở Phase 2)
Metrics/OT_MeanEndMass     (target: tương tự)
Metrics/OT_ValidPseudoRatio  (= valid_pseudo_mask.float().mean() — tỷ lệ sample vượt threshold)
```

---

## Verification Checklist (Agent tự kiểm tra)

- [ ] `FIX-01`: `--sinkhorn_epsilon` default = `0.05`; `--sinkhorn_iters` không đổi = `100`
- [ ] `FIX-02`: `span_projection_loss` có signature mới `(..., global_step, spe)`; tất cả call-sites đã được cập nhật
- [ ] `FIX-03`: Không có `max()` call nào nhận all-`-inf` tensor
- [ ] `FIX-04`: `loss_has_ans` = `0.7 * loss_has_en + 0.3 * loss_has_vi`; không còn symmetric averaging
- [ ] `FIX-05`: Entropy log dùng `(P + 1e-8).log()` — không dùng `P.log()` trực tiếp
- [ ] `FIX-06`: Transport mass metrics được log ở Phase 2 (`global_step > 4 * spe`)