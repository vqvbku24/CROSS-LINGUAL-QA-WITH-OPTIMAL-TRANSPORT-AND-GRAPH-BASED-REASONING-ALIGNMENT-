# CODE_AUDIT_SPEC.md
## Mục tiêu
Kiểm toán toàn bộ codebase của hệ thống Cross-lingual QA (EN→VI) để xác nhận rằng 4 kỹ thuật đã được validated khớp hoàn toàn với implementation hiện tại. Agent **chỉ đọc và báo cáo sai lệch**, không tự sửa trừ khi được chỉ định rõ ràng.

---

## Phạm vi file cần kiểm tra
| File | Vai trò |
|---|---|
| `model_core.py` | Kiến trúc, LoRA wrapping, context manager |
| `losses.py` | `L_ot`, `L_cons`, `L_span` |
| `train.py` | Optimizer groups, curriculum schedule, early stopping |
| `quick_eval.py` | Evaluation logic |
| `inference_to_json.py` | Inference pipeline |

---

## Checklist kiểm toán (theo từng kỹ thuật)

---

### [AUDIT-1] LoRA Adapter & Context Manager (Phân thân mô hình)

**Lý do quan trọng:** Đây là điều kiện sống còn. Nếu LoRA không được wrap đúng, gradient của nhánh EN (Teacher) sẽ rò sang backbone và làm chết nhánh VI (Student).

**Yêu cầu agent kiểm tra trong `model_core.py`:**

1. **LoRA config tồn tại:** Xác nhận `LoraConfig` được áp dụng lên backbone XLM-R. Ghi lại `r`, `lora_alpha`, `target_modules`.

2. **Context manager Teacher:** Xác nhận có hàm `with model.backbone.disable_adapter():` (hoặc tương đương) bao quanh forward pass của nhánh EN (Teacher). Nếu dùng cơ chế khác để freeze gradient Teacher, ghi rõ tên hàm/cơ chế đó.

3. **Student không bị freeze:** Xác nhận nhánh VI (Student) **không** nằm trong bất kỳ `torch.no_grad()` hay `requires_grad_(False)` nào trong forward pass huấn luyện.

4. **Báo lỗi nếu:** Backbone bị gọi `requires_grad_(False)` ở cấp module thay vì dùng LoRA/context manager.

---

### [AUDIT-2] Load Checkpoint an toàn (Tránh lỗi key mismatch)

**Lý do quan trọng:** Load checkpoint Stage 1 sai thứ tự sẽ khiến tất cả tri thức SQuAD bị mất, EM rớt về ~7%.

**Yêu cầu agent kiểm tra trong `train_stage2.py`:**

1. **Thứ tự load đúng:** Xác nhận quy trình theo đúng thứ tự sau:
   - Bước A: Load Stage 1 weights → **Base XLM-R model** (chưa có LoRA wrapper)
   - Bước B: **Sau đó** mới áp dụng `LoraConfig` lên model vừa load

2. **Báo lỗi nếu:** LoRA được wrap trước, rồi mới gọi `load_state_dict` với checkpoint Stage 1 (thứ tự ngược gây key mismatch).

3. **Kiểm tra `strict` flag:** Ghi lại giá trị của `strict=` trong lần gọi `load_state_dict`. Nếu `strict=False`, ghi chú đây là rủi ro tiềm ẩn có thể che giấu key mismatch thầm lặng.

---

### [AUDIT-3] Optimizer Groups — Chống bào mòn QA Head

**Lý do quan trọng:** Khi `L_span = 0` (giai đoạn đầu curriculum), AdamW vẫn trừ `weight_decay` mỗi step, làm QA Head "quên" cách trích xuất. EM sụp về 2.22%.

**Yêu cầu agent kiểm tra trong `train_stage2.py`:**

1. **Tách parameter groups:** Xác nhận optimizer được khởi tạo với **ít nhất 2 groups riêng biệt**:
   - Group A: LoRA parameters → có `weight_decay > 0`
   - Group B: QA Head parameters + `layer_weights` → `weight_decay = 0.0` **(bắt buộc)**

2. **Ghi lại cấu hình:** Copy đoạn code tạo optimizer groups vào báo cáo.

3. **Báo lỗi nếu:**
   - Chỉ có 1 parameter group duy nhất cho toàn bộ model
   - QA Head nằm cùng group với LoRA và nhận `weight_decay > 0`
   - `layer_weights` không được tách ra riêng hoặc nhận `weight_decay > 0`

---

### [AUDIT-4] Curriculum Learning Schedule

**Lý do quan trọng:** Ma trận γ nhiễu ở epoch đầu. Bật `L_span` quá sớm làm QA Head học nhãn giả bị nhòe.

**Schedule đã được validated (từ CURRICULUM_REORDER_SPEC.md):**
```
Epoch 1–3   : chỉ L_ot  (OT alignment)
Epoch 4–6   : L_ot + L_cons  (cons_start_epoch = 4)
Epoch 7–20  : L_ot + L_cons + L_span  (span_start_epoch = 7)
```
*Tổng: 20 epochs.*

**Yêu cầu agent kiểm tra trong `train_stage2.py`:**

1. **Giá trị default của CLI args:** Xác nhận:
   - `--cons_start_epoch` default = **4**
   - `--span_start_epoch` default = **7**
   - `--num_train_epochs` default = **20**
   (So sánh với giá trị thực tế trong `argparse` hoặc config)

2. **Logic gate của từng loss:** Xác nhận điều kiện kích hoạt trong training loop:
   ```python
   # Cấu trúc mong đợi (pseudo-code):
   if epoch >= cons_start_epoch:
       loss += lambda_cons * L_cons
   if epoch >= span_start_epoch:
       loss += lambda_span * L_span
   ```
   Ghi lại code thực tế và so sánh.

3. **Early stopping bị chặn:** Xác nhận Early Stopping **không thể kích hoạt** trong 3 epoch đầu (hoặc trước `cons_start_epoch`). Ghi lại cơ chế chặn (e.g., `warmup_epochs`, điều kiện `if epoch < warmup`).

4. **Báo lỗi nếu:**
   - Default `cons_start_epoch` ≠ 4 hoặc `span_start_epoch` ≠ 7
   - `L_span` có thể được kích hoạt trước `L_cons`
   - Không có cơ chế chặn Early Stopping ở các epoch warmup

---

### [AUDIT-5] Tính đúng đắn của Loss Functions

**Yêu cầu agent kiểm tra trong `losses.py`:**

#### 5a. Sinkhorn OT (`L_ot`)
1. Xác nhận `sinkhorn_epsilon = 0.1` (đã đổi từ 0.05).
2. Xác nhận `sinkhorn_iters = 100` (đã đổi từ 50).
3. Xác nhận ma trận cost được chuẩn hóa trước khi đưa vào Sinkhorn (tránh numerical overflow).
4. Xác nhận output γ được `detach()` hoặc không lan gradient ngược vào backbone qua path không mong muốn (ngoài những gì đã thiết kế).

#### 5b. Span Projection (`L_span`)
1. Xác nhận Teacher logits được tính bằng `torch.softmax` (hoặc `F.softmax`) với **temperature scaling** trước khi tạo soft-label.
2. Xác nhận phép nhân γ với soft-label được thực hiện đúng chiều (matmul dimension phải nhất quán: VI tokens × EN tokens).
3. Xác nhận `L_span` chỉ được tính khi `current_epoch >= span_start_epoch` — kiểm tra xem có guard condition trong `losses.py` hay guard chỉ nằm trong `train.py`.
4. **Kiểm tra span_loss = 0.0 bug:** Xác nhận **không** có nhánh code nào trả về `0.0` sớm (early return) trước khi thực sự tính loss, trừ khi là intentional guard theo epoch.

#### 5c. Consistency Loss (`L_cons`)
1. Xác nhận `lambda_cons = 0.1` (đã đổi từ 0.2).
2. Xác nhận `cons_temp = 4.0` (đã đổi từ 2.0).
3. Xác nhận loss dùng MSE (không phải KL divergence hay cosine) để neo token VI vào anchor EN.
4. Xác nhận gradient **không** chảy ngược vào Teacher features (Teacher features phải được `detach()`).

---

### [AUDIT-6] Layer Weighting

**Yêu cầu agent kiểm tra trong `model_core.py` và `train_stage2.py`:**

1. **Layers được dùng:** Xác nhận OT alignment được tính trên hidden states từ **layers 6–9** của XLM-R.
2. **`layer_weights` là learnable:** Xác nhận `layer_weights` là `nn.Parameter` (không phải tensor thường hoặc hardcoded constants).
3. **`layer_weights` trong optimizer:** Cross-check với AUDIT-3 — `layer_weights` phải nằm trong Group B (weight_decay = 0.0).

---

### [AUDIT-7] Evaluation & Inference Consistency

**Yêu cầu agent kiểm tra trong `quick_eval.py` và `inference_to_json.py`:**

1. **Mode eval:** Xác nhận `model.eval()` và `torch.no_grad()` được gọi đúng chỗ.
2. **LoRA active khi inference:** Xác nhận khi inference trên VI, LoRA adapter **được bật** (không nằm trong `disable_adapter()` context).
3. **Postprocessing:** Xác nhận logic lấy start/end token từ logits nhất quán giữa `quick_eval.py` và `inference_to_json.py` (không có sai lệch off-by-one hay softmax/argmax khác nhau).

---

## Output format yêu cầu từ Agent

Agent phải xuất báo cáo theo cấu trúc sau:

```
## AUDIT REPORT

### [AUDIT-1] LoRA & Context Manager
STATUS: ✅ PASS / ⚠️ WARNING / ❌ FAIL
Findings: ...
Code snippet (nếu có vấn đề): ...

### [AUDIT-2] Checkpoint Loading
...

### [AUDIT-3] Optimizer Groups
...

### [AUDIT-4] Curriculum Schedule
...

### [AUDIT-5] Loss Functions
  [5a] L_ot: ...
  [5b] L_span: ...
  [5c] L_cons: ...

### [AUDIT-6] Layer Weighting
...

### [AUDIT-7] Eval & Inference
...

---
## SUMMARY TABLE
| Audit Item | Status | Ghi chú ngắn |
|---|---|---|
| AUDIT-1 LoRA | ✅/⚠️/❌ | |
| AUDIT-2 Checkpoint | | |
| AUDIT-3 Optimizer | | |
| AUDIT-4 Curriculum | | |
| AUDIT-5a L_ot | | |
| AUDIT-5b L_span | | |
| AUDIT-5c L_cons | | |
| AUDIT-6 Layers | | |
| AUDIT-7 Eval | | |

## ACTION ITEMS (nếu có)
Liệt kê từng vấn đề theo format:
[FILE:FUNCTION:LINE] Mô tả vấn đề → Sửa thành gì
```

---

## No-touch zones
Agent **chỉ đọc, không được sửa** bất kỳ file nào trong lần chạy này. Mục tiêu duy nhất là tạo báo cáo kiểm toán. Nếu phát hiện bug, ghi vào ACTION ITEMS để tạo BUGFIX_SPEC riêng trong bước tiếp theo.