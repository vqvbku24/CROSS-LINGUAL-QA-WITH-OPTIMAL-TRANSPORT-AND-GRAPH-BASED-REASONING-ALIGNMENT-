# CURRICULUM_REORDER_SPEC.md — 3-Stage Rocket Curriculum

> **Dành cho coding agent.** Thay đổi thứ tự curriculum từ QA→Span→Cons thành QA→Cons→Span,
> đồng thời đưa các mốc epoch ra thành CLI arguments để dễ ablation.
> Agent tự quyết định cách integrate, đặt tên biến, và refactor nếu cần.
> Không được chỉnh sửa bất kỳ file nào ngoài danh sách Allowed Files.
> **Scope:** Chỉ `run_training`. Các hàm `run_overfit` và `run_overfit_full` KHÔNG nằm trong scope lần này.

---

## Allowed Files (Touch Zone)

| File | Phạm vi được phép chỉnh sửa |
|---|---|
| `train.py` | `DEFAULT_CONFIG`, `ArgumentParser`, `run_training` only |

## No-Touch Zone

| File | Lý do |
|---|---|
| `losses.py` | Không liên quan đến thay đổi lần này |
| `backbone.py` | Stable |
| `model_core.py` | Stable |

---

## Priority Table

| Priority | ID | Mức độ rủi ro | Mô tả ngắn |
|---|---|---|---|
| 🔴 P1 | `CUR-01` | Trung bình | Đổi thứ tự delay: Cons trước Span trong `run_training` |
| 🔴 P1 | `CUR-02` | Thấp | Đưa `cons_start_epoch` và `span_start_epoch` ra CLI args |
| 🟠 P2 | `CUR-03` | Thấp | Đưa `cons_start_epoch` và `span_start_epoch` vào `DEFAULT_CONFIG` |
| 🟡 P3 | `CUR-04` | Thấp | Cập nhật log message curriculum để phản ánh thứ tự mới |

---

## Chi tiết từng Change

---

### CUR-01 — Đổi thứ tự delay Cons → trước Span

**File:** `train.py`
**Function:** `run_training`

**Problem (thứ tự hiện tại — Span và Cons bật cùng lúc sau epoch 4):**
```python
_SPAN_DELAY, _SPAN_WARMUP = _SPE * 4,   _SPE * 2
_CONS_DELAY, _CONS_WARMUP = _SPE * 4,   _SPE * 2
```

**Fix — Cons trước Span, lệch nhau 2 epoch:**
```python
# Tầng 2: Bật Cons sau Epoch 3 (step = 3 * _SPE)
_CONS_DELAY  = int(_SPE * config.get("cons_start_epoch",  3))
_CONS_WARMUP = int(_SPE * config.get("cons_warmup_epochs", 2))

# Tầng 3: Bật Span sau Epoch 5 (step = 5 * _SPE)
_SPAN_DELAY  = int(_SPE * config.get("span_start_epoch",  5))
_SPAN_WARMUP = int(_SPE * config.get("span_warmup_epochs", 2))
```

**Lý do thứ tự:** `L_cons` không có confidence threshold → học được ngay khi γ còn noisy.
`L_span` Phase 2 có `confidence_threshold = 0.25` → cần γ đủ sắc nét mới có valid samples.
Cho Cons chạy trước 2 epoch = cho OT + Cons đủ thời gian làm sắc nét γ trước khi Span enforce hard threshold.

**Điều kiện bất biến phải giữ:** `_SPAN_DELAY > _CONS_DELAY` luôn luôn đúng.
Agent nên thêm assertion hoặc log warning nếu user truyền tham số vi phạm điều kiện này.

---

### CUR-02 — CLI Arguments mới

**File:** `train.py`
**Function:** `main` → `ArgumentParser`

**Fix — thêm 4 args sau nhóm `--lambda_*` args:**
```python
# Curriculum epoch milestones
parser.add_argument(
    "--cons_start_epoch", type=int, default=3,
    help="Epoch at which L_cons starts warming up. Default: 3."
)
parser.add_argument(
    "--cons_warmup_epochs", type=int, default=2,
    help="Number of epochs for L_cons linear warmup. Default: 2."
)
parser.add_argument(
    "--span_start_epoch", type=int, default=5,
    help="Epoch at which L_span starts warming up. Must be > cons_start_epoch. Default: 5."
)
parser.add_argument(
    "--span_warmup_epochs", type=int, default=2,
    help="Number of epochs for L_span linear warmup. Default: 2."
)
```

**Cập nhật `config` dict trong `main`:**
```python
config.update({
    # ... existing keys ...
    "cons_start_epoch"   : args.cons_start_epoch,
    "cons_warmup_epochs" : args.cons_warmup_epochs,
    "span_start_epoch"   : args.span_start_epoch,
    "span_warmup_epochs" : args.span_warmup_epochs,
})
```

---

### CUR-03 — Cập nhật DEFAULT_CONFIG

**File:** `train.py`
**Section:** `DEFAULT_CONFIG` dict

**Fix — thêm 4 keys:**
```python
DEFAULT_CONFIG = {
    # ... existing keys ...

    # Curriculum epoch milestones (3-Stage Rocket)
    # Stage 1: QA + OT only (Epochs 1 → cons_start_epoch)
    # Stage 2: + Cons warmup (cons_start_epoch → span_start_epoch)
    # Stage 3: + Span warmup (span_start_epoch → span_start_epoch + span_warmup_epochs)
    "cons_start_epoch"   : 3,
    "cons_warmup_epochs" : 2,
    "span_start_epoch"   : 5,
    "span_warmup_epochs" : 2,
}
```

---

### CUR-04 — Cập nhật log message

**File:** `train.py`
**Function:** `run_training`, đoạn log curriculum delays

**Problem (log hiện tại không phản ánh thứ tự mới):**
```python
log.info(
    f"Curriculum delays (steps): "
    f"OT={_OT_DELAY}→{_OT_DELAY+_OT_WARMUP} | "
    f"Span={_SPAN_DELAY}→{_SPAN_DELAY+_SPAN_WARMUP} | "
    f"Cons={_CONS_DELAY}→{_CONS_DELAY+_CONS_WARMUP}"
)
```

**Fix:**
```python
log.info(
    f"3-Stage Curriculum (steps): "
    f"[Stage1] OT: {_OT_DELAY}→{_OT_DELAY+_OT_WARMUP} | "
    f"[Stage2] Cons: {_CONS_DELAY}→{_CONS_DELAY+_CONS_WARMUP} | "
    f"[Stage3] Span: {_SPAN_DELAY}→{_SPAN_DELAY+_SPAN_WARMUP}"
)
log.info(
    f"Epoch milestones: "
    f"Cons starts ep.{config['cons_start_epoch']} | "
    f"Span starts ep.{config['span_start_epoch']}"
)
```

---

## Validation — Agent tự kiểm tra

- [ ] `_SPAN_DELAY > _CONS_DELAY` — assertion hoặc log warning nếu user truyền tham số vi phạm
- [ ] `python train.py --mode train --cons_start_epoch 3 --span_start_epoch 5` — chạy không error
- [ ] `python train.py --mode train --cons_start_epoch 2 --span_start_epoch 4` — ablation variant chạy được
- [ ] Log output hiển thị đúng: `[Stage2] Cons: X→Y` trước `[Stage3] Span: A→B`
- [ ] `python train.py --help` — 4 args mới xuất hiện trong help text
- [ ] Các args cũ (`--lambda_ot`, `--lambda_span`, etc.) không bị ảnh hưởng
- [ ] `run_overfit` và `run_overfit_full` **không bị chỉnh sửa**

---

## Ablation Variants Gợi ý (cho paper)

| Run | `--cons_start_epoch` | `--span_start_epoch` | Mục đích |
|---|---|---|---|
| **Proposed** | 3 | 5 | Chiến lược đề xuất |
| Aggressive | 2 | 4 | Bật sớm hơn 1 epoch |
| Conservative | 4 | 7 | Cho OT nhiều thời gian hơn |
| Simultaneous | 4 | 4 | Baseline — bật cùng lúc (config cũ) |
| No-Cons | 99 | 5 | Ablation: bỏ Cons, chỉ Span |

Chạy 5 variants này = đủ dữ liệu cho ablation table trong paper.

---

## Ghi chú Kiến trúc

```
3-Stage Rocket Timeline (default):
  Epoch 1-3:  QA + OT only
              → XLM-R học EN span extraction
              → OT kéo EN↔VI embeddings gần nhau
              → γ bắt đầu có cấu trúc

  Epoch 3-5:  + Cons warmup (linear 0 → lambda_cons)
              → VI logits học hình dáng phân phối EN qua γ
              → γ tiếp tục sắc nét nhờ OT
              → "Gò đất" VI distribution nhô lên gần threshold

  Epoch 5-7:  + Span warmup (linear 0 → lambda_span)
              → confidence_threshold = 0.25 được vượt bởi các samples tốt
              → span_loss != 0.0 → logits VI vót nhọn thành Start/End boundaries

  Epoch 7+:   Full model — tất cả loss components ở max weight
```