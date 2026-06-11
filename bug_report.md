# DIAGNOSTIC_SPEC.md — Span Loss = 0 Root Cause Investigation

> **Dành cho coding agent.** Đây là spec CHẨN ĐOÁN, không phải fix spec.
> Mục tiêu: Tìm ra chính xác tại sao `Loss/Span (Vietnamese)` = 0 sau khi `lambda_span` đã bật
> (cons_start_epoch=4, span_start_epoch=6, confidence_threshold=0.07).
> Agent KHÔNG được sửa logic nghiệp vụ, KHÔNG được thay đổi hyperparameter.
> Agent chỉ được thêm temporary diagnostic code, chạy, đọc kết quả, rồi báo cáo.

---

## Allowed Files (Touch Zone — diagnostic only)

| File | Phạm vi được phép |
|---|---|
| `losses.py` | Thêm `print`/`log` statements tạm thời để đo giá trị nội bộ |
| `train.py` | Thêm `print`/`log` statements tạm thời nếu cần trace curriculum state |

## No-Touch Zone (tuyệt đối)

| File/Thành phần | Lý do |
|---|---|
| Bất kỳ logic tính loss nào | Không được thay đổi kết quả tính toán |
| Hyperparameters (`threshold`, `epsilon`, `lambda_*`) | Đây là input của chẩn đoán, không phải output |
| `backbone.py`, `model_core.py` | Không liên quan |
| `optimizer`, `scheduler` | Không liên quan |

---

## Bối cảnh vấn đề

**Triệu chứng quan sát được:**
- `Loss/Span (Vietnamese)` = 0 từ khi `lambda_span` bắt đầu bật (epoch 6 trở đi)
- `Metrics/OT_MeanStartMass` = 0 suốt toàn bộ run
- `Metrics/OT_MeanEndMass` = 0 suốt toàn bộ run
- `Metrics/OT_ValidPseudoRatio` = 0 suốt toàn bộ run
- `Loss/Span` ở Phase 1 (soft) tăng dần rồi drop về 0 khi Phase 2 kick in

**Config đang chạy:**
```
cons_start_epoch  = 4
span_start_epoch  = 6
confidence_threshold = 0.07   ← đã hạ từ 0.25 xuống 0.07
sinkhorn_epsilon  = 0.05
sinkhorn_iters    = 100
```

**Câu hỏi cần trả lời:**
1. `is_hard_phase` có thực sự = `True` sau epoch 6 không?
2. `valid_pseudo_mask` có sample nào = `True` không? Tỷ lệ bao nhiêu?
3. Giá trị thực tế của `max_start_mass` và `max_end_mass` là bao nhiêu so với threshold 0.07?
4. γ (gamma) có cấu trúc hay vẫn flat (uniform)?
5. `span_projection_loss` có được gọi không, hay bị short-circuit trước đó?

---

## Nhiệm vụ Agent

### Task 1 — Đọc và map toàn bộ data flow của `span_projection_loss`

Scan theo thứ tự:

1. **`train.py`**: Tìm dòng gọi `criterion(...)` hoặc `criterion.forward(...)`.
   - `global_step` và `spe` được truyền vào như thế nào?
   - `span_start_epoch` có được truyền vào không, hay dùng default?

2. **`losses.py` — `OTAlignmentLoss.forward`**: Tìm chỗ gọi `span_projection_loss(...)`.
   - `answerable_mask` được tính như thế nào? Có sample nào pass mask không?
   - Nếu `answerable_mask.sum() == 0` → hàm không bao giờ được gọi → đây là root cause.

3. **`losses.py` — `span_projection_loss`**: Đọc toàn bộ hàm.
   - `is_hard_phase` được tính bằng công thức nào (`4 * spe`? `span_start_epoch * spe`?).
   - `confidence_threshold` được lấy từ đâu — hardcode hay param?
   - Metrics `mean_start_mass`, `mean_end_mass`, `valid_ratio` được gán ở đâu (trong hay ngoài `if valid_pseudo_mask.any()`)?

4. **`losses.py` — `sinkhorn_log_domain`**: Kiểm tra output γ.
   - γ được normalize như thế nào? Row-sum = 1 hay column-sum = 1?
   - Với sequence length ~256-512 tokens và epsilon=0.05, giá trị max mỗi row của γ kỳ vọng là bao nhiêu?

---

### Task 2 — Thêm Diagnostic Prints tạm thời

Thêm các print sau vào `span_projection_loss`, **bao bọc bằng `with torch.no_grad()`**, chỉ in khi `global_step % 50 == 0` để tránh spam:

```python
# ── DIAGNOSTIC BLOCK (tạm thời — xóa sau khi chẩn đoán xong) ──
if global_step % 50 == 0:
    print(f"\n[DIAG step={global_step}] is_hard_phase={is_hard_phase}")
    print(f"[DIAG] gamma shape: {gamma.shape}")
    print(f"[DIAG] gamma row-sum (mean): {gamma.sum(dim=2).mean():.4f}")   # nên = 1.0 nếu row-normalized
    print(f"[DIAG] gamma max per cell (mean): {gamma.max(dim=2).values.mean():.6f}")
    
    # Chỉ in thêm nếu đang ở Phase 2
    if is_hard_phase:
        start_mass = gamma[torch.arange(gamma.size(0)), en_start, :]
        max_start  = start_mass.max(dim=1).values
        print(f"[DIAG Phase2] max_start_mass — mean={max_start.mean():.6f}, min={max_start.min():.6f}, max={max_start.max():.6f}")
        print(f"[DIAG Phase2] threshold={confidence_threshold}")
        print(f"[DIAG Phase2] samples above threshold: {(max_start > confidence_threshold).sum().item()} / {gamma.size(0)}")
        print(f"[DIAG Phase2] valid_pseudo_mask: {valid_pseudo_mask.sum().item()} / {valid_pseudo_mask.size(0)}")
# ── END DIAGNOSTIC BLOCK ──
```

**Lưu ý vị trí đặt print:**
- Print `is_hard_phase` và `gamma` ngay đầu hàm, trước `if not is_hard_phase`.
- Print Phase 2 metrics ngay sau khi tính `valid_pseudo_mask`, trước `if valid_pseudo_mask.any()`.

---

### Task 3 — Thêm Diagnostic tại `OTAlignmentLoss.forward`

Ngay trước call-site `span_projection_loss(...)`, thêm:

```python
# ── DIAGNOSTIC BLOCK ──
if global_step % 50 == 0:
    print(f"[DIAG forward] answerable_mask sum: {answerable_mask.sum().item()} / {answerable_mask.size(0)}")
    print(f"[DIAG forward] lambda_span={self.lambda_span:.4f}")
    print(f"[DIAG forward] global_step={global_step}, spe={spe}")
# ── END DIAGNOSTIC BLOCK ──
```

---
### Task 4 — Báo cáo

Sau khi có output, agent tổng hợp kết quả theo template:

```
## Diagnostic Report

### 1. Data flow
- `span_projection_loss` có được gọi không? [Yes/No]
- `answerable_mask.sum()` trung bình: [giá trị]
- `is_hard_phase` trở thành True từ step nào? [step]
- Công thức `is_hard_phase` đang dùng: [4 * spe / span_start_epoch * spe / khác]
- `confidence_threshold` đang là: [giá trị — hardcode hay param?]

### 2. Gamma analysis
- γ row-sum mean: [giá trị — nên = 1.0]
- γ max per cell mean: [giá trị]
- Với threshold=0.07: [X%] samples vượt ngưỡng

### 3. Root cause (agent kết luận)
- [ ] `answerable_mask` rỗng → hàm không được gọi
- [ ] `is_hard_phase` không bao giờ True (spe=0 hoặc công thức sai)
- [ ] γ quá flat → max mass << threshold kể cả 0.07
- [ ] `confidence_threshold` hardcode sai giá trị (không dùng 0.07)
- [ ] Metrics chỉ log trong `if valid_pseudo_mask.any()` → không bao giờ log
- [ ] Khác: [mô tả]

### 4. Recommended fix
[Agent mô tả fix ngắn gọn, không implement — chờ approval]
```

---

## Cleanup sau chẩn đoán

Sau khi báo cáo xong, agent **xóa toàn bộ diagnostic print blocks** (tìm bằng comment `# ── DIAGNOSTIC BLOCK ──`). Code phải trở về trạng thái ban đầu, chỉ khác ở báo cáo.