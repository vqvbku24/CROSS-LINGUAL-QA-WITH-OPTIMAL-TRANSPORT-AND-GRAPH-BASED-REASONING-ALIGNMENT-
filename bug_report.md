# PHASE_SYNC_SPEC.md — Đồng bộ Phase 2 với `span_start_epoch`

> **Dành cho coding agent.** Fix conflict giữa lambda curriculum (train.py) và phase curriculum
> (losses.py). Hiện tại Phase 2 bắt đầu ở epoch 4 (hardcode) trong khi lambda_span = 0 đến epoch 7
> → valid_pseudo_mask được tính khi γ chưa chín → span_loss = 0 mãi sau khi lambda bật lên.
> Agent tự quyết định cách refactor. Không được chỉnh sửa file ngoài Allowed Files.

---

## Allowed Files (Touch Zone)

| File | Phạm vi được phép chỉnh sửa |
|---|---|
| `losses.py` | Chỉ `span_projection_loss` signature + `is_hard_phase` condition + call-site trong `OTAlignmentLoss.forward` |
| `train.py` | Chỉ call-site `criterion(...)` — truyền thêm `span_start_epoch` vào kwargs nếu cần |

## No-Touch Zone

| File | Lý do |
|---|---|
| `backbone.py` | Stable |
| `model_core.py` | Stable |
| `train.py` — lambda curriculum | Lớp curriculum này đúng, không đổi |
| `losses.py` — `consistency_loss` | Không liên quan |
| `losses.py` — `OTAlignmentLoss.__init__` | Không cần thêm attribute mới nếu agent dùng cách forward-pass param |

---

## Priority Table

| Priority | ID | File | Rủi ro | Mô tả |
|---|---|---|---|---|
| 🔴 P1 | `SYNC-01` | `losses.py` | Trung bình | Thay `4 * spe` bằng `span_start_epoch * spe` trong `is_hard_phase` |
| 🔴 P1 | `losses.py` | Trung bình | Cập nhật signature `span_projection_loss` nhận `span_start_epoch` |
| 🔴 P1 | `SYNC-02` | `losses.py` | Thấp | Cập nhật call-site trong `OTAlignmentLoss.forward` truyền `span_start_epoch` |
| 🟠 P2 | `SYNC-03` | `train.py` | Thấp | Truyền `span_start_epoch` vào `criterion()` call nếu cần |

---

## Chi tiết

---

### SYNC-01 — Fix `is_hard_phase` trong `span_projection_loss`

**File:** `losses.py`
**Function:** `span_projection_loss`

**Problem:**
```python
def span_projection_loss(
    vi_start_logits, vi_end_logits, gamma,
    en_start, en_end,
    global_step: int,
    spe: int,
) -> tuple:
    ...
    is_hard_phase = global_step > (4 * spe)   # ← hardcode epoch 4, không đồng bộ với span_start_epoch
```

**Fix — thêm param `span_start_epoch`, xóa hardcode `4`:**
```python
def span_projection_loss(
    vi_start_logits, vi_end_logits, gamma,
    en_start, en_end,
    global_step: int,
    spe: int,
    span_start_epoch: int = 5,   # ← default khớp với DEFAULT_CONFIG
) -> tuple:
    ...
    # Phase 2 bắt đầu đúng lúc lambda_span warmup bắt đầu
    # → γ đã có (span_start_epoch) epochs để học trước khi hard threshold được áp dụng
    is_hard_phase = global_step > (span_start_epoch * spe)
```

**Lý do:** Phase 2 (hard pseudo-label với threshold) chỉ có ý nghĩa khi γ đã đủ sắc nét.
γ sắc nét được nhờ OT + Cons chạy từ epoch 1 đến `span_start_epoch`.
Bật Phase 2 sớm hơn (epoch 4 hardcode) khi γ còn flat → `valid_pseudo_mask` toàn False →
khi `lambda_span` bật lên ở epoch 7 thì Phase 2 đã "thất bại" từ trước, loss vẫn = 0.

---

### SYNC-02 — Cập nhật call-site trong `OTAlignmentLoss.forward`

**File:** `losses.py`
**Function:** `OTAlignmentLoss.forward`

**Problem (call-site hiện tại):**
```python
l_span, mean_s_mass, mean_e_mass, valid_ratio = span_projection_loss(
    vi_start_logits[answerable_mask],
    vi_end_logits[answerable_mask],
    gamma[answerable_mask],
    en_start[answerable_mask],
    en_end[answerable_mask],
    global_step=global_step,
    spe=spe,
    # ← span_start_epoch không được truyền vào
)
```

**Fix:**
```python
l_span, mean_s_mass, mean_e_mass, valid_ratio = span_projection_loss(
    vi_start_logits[answerable_mask],
    vi_end_logits[answerable_mask],
    gamma[answerable_mask],
    en_start[answerable_mask],
    en_end[answerable_mask],
    global_step=global_step,
    spe=spe,
    span_start_epoch=self.span_start_epoch,   # ← từ OTAlignmentLoss attribute
)
```

**Agent quyết định cách lưu `span_start_epoch` vào OTAlignmentLoss:**

Option A (đơn giản nhất): Thêm `span_start_epoch: int = 5` vào `OTAlignmentLoss.__init__` và `self.span_start_epoch = span_start_epoch`.

Option B: Truyền trực tiếp qua `OTAlignmentLoss.forward(... span_start_epoch: int = 5)` và forward xuống call-site — không cần lưu vào `self`.

Option C: Đọc từ `config` dict nếu `forward()` đã nhận config. Không preferred vì tăng coupling.

**Agent chọn option phù hợp nhất với convention hiện tại của codebase.**

---

### SYNC-03 — Truyền `span_start_epoch` từ `train.py` (nếu cần theo Option B)

**File:** `train.py`
**Function:** `run_training`, dòng gọi `criterion(...)`

**Chỉ cần thay đổi nếu agent chọn Option B:**

```python
# TRƯỚC:
losses = criterion(outputs, batch, global_step=global_step, spe=_SPE)

# SAU (Option B):
losses = criterion(
    outputs, batch,
    global_step=global_step,
    spe=_SPE,
    span_start_epoch=config.get("span_start_epoch", 5),
)
```

**Nếu agent chọn Option A** (lưu vào `__init__`), cần truyền `span_start_epoch` khi khởi tạo trong `setup_model_and_criterion`:
```python
criterion = OTAlignmentLoss(
    ...
    span_start_epoch = config.get("span_start_epoch", 5),
)
```

---

## Validation — Agent tự kiểm tra

- [ ] `span_projection_loss` không còn hardcode `4` — thay bằng `span_start_epoch`
- [ ] `is_hard_phase` với `span_start_epoch=7, spe=100, global_step=650` → `False` (epoch 6.5, chưa đến)
- [ ] `is_hard_phase` với `span_start_epoch=7, spe=100, global_step=701` → `True` (epoch 7.01)
- [ ] Chạy 1 forward pass với `global_step = span_start_epoch * spe + 1`: `mean_start_mass` và `mean_end_mass` được gán (có thể vẫn 0 nếu γ flat — nhưng không còn là "Phase 1 never executed Phase 2")
- [ ] `DEFAULT_CONFIG["span_start_epoch"]` và default của `span_start_epoch` param nhất quán (đều = 5 hoặc đều được override bởi CLI arg)
- [ ] Không có regression ở `run_overfit` và `run_overfit_full` (không trong scope nhưng kiểm tra không bị ảnh hưởng)

---

## Ghi chú

```
Timeline sau fix (span_start_epoch=7, cons_start_epoch=4):

  Epoch 1-4:  Phase 1 (soft), lambda_span=0, lambda_cons=0
              → L_span = 0 * soft_loss = 0 (đúng — chưa bật)
              → γ học qua L_ot

  Epoch 4-7:  Phase 1 (soft), lambda_cons warmup, lambda_span=0
              → L_span = 0 (đúng — chưa bật)
              → γ sắc nét dần qua OT + Cons
              → "gò đất" VI distribution được đắp lên

  Epoch 7+:   Phase 2 (hard), lambda_span warmup → 0.3
              → γ đã có 7 epochs để học → mass vượt threshold
              → valid_pseudo_mask có True samples
              → span_loss > 0 ✓
              → Metrics/OT_MeanStartMass > 0 ✓
```