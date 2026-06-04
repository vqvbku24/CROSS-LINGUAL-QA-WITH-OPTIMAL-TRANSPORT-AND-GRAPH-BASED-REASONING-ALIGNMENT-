# Bug Report — Sinkhorn OT Refactor

## Bug 1 🔴 `losses.py` — Gradient rò rỉ từ `gamma` vào Span Projection và Consistency Loss

**Vấn đề:**
`gamma` được tính từ Sinkhorn qua `log_K = -C / epsilon`. Vì `C` có gradient từ backbone,
`gamma` cũng mang gradient ngầm qua `C`. Khi `gamma` được truyền vào `span_projection_loss`
và `consistency_loss`, gradient **vô tình chảy ngược về backbone** — ngoài ý muốn.
Chỉ `L_ot` mới được phép train backbone qua `C`; span và consistency chỉ nên train VI branch.

**Yêu cầu agent:** Phân tích dataflow gradient trong `OTAlignmentLoss.forward()` và đề xuất
cách isolate gradient hợp lý nhất (ví dụ: `gamma.detach()`, hay tách Sinkhorn ra khỏi
computation graph, hay cách khác). Áp dụng fix sao cho `L_ot` vẫn train được backbone
thông qua `C`, còn `L_span_proj` và `L_cons` thì không.

---

## Bug 2 🔴 `train.py` — Learning Rate × 100 quá cao trong `run_overfit`

**Vấn đề:**
```python
opt_qa = AdamW(qa_params, lr=config["overfit_lr"] * 100, weight_decay=0.0)
# overfit_lr = 3e-4 → lr thực tế = 0.03  ← quá cao, sẽ explode hoặc oscillate
```
Loss sẽ không giảm monotonically trong overfit sanity check.

**Yêu cầu agent:** Xác định lr hợp lý cho QA head trong overfit mode (mục tiêu: loss
giảm nhanh nhưng ổn định), sửa lại dòng này với lý giải rõ ràng.

---

## Bug 3 🟡 `losses.py` — Off-by-one: [SEP] token bị include vào question embeddings

**Vấn đề:**
```python
max_q_len = question_end.max().item() + 1
q_mask = positions > question_end.unsqueeze(1)
# position == question_end → mask=False → [SEP] token bị include vào cross-attention
```
Comment nói "exclusive end" nhưng implementation lại **inclusive** — không nhất quán.

**Yêu cầu agent:** Xem xét kiến trúc QA Head cross-attention và quyết định xem [SEP]
token có nên tham gia vào question embeddings không. Sửa code và comment cho nhất quán,
với lý giải rõ ràng về lựa chọn.

---

## Bug 4 🟡 `train.py` — Multi-GPU không được handle

**Vấn đề:**
Code chỉ có `torch.cuda.set_device(0)` — training chỉ chạy trên GPU 0 dù cluster có nhiều GPU.
Sinkhorn trên `[B, 512, 512]` sẽ không tận dụng được các GPU còn lại.

**Yêu cầu agent:** Đề xuất và implement chiến lược multi-GPU phù hợp nhất với kiến trúc
hiện tại (lưu ý: `OTAlignmentLoss` chứa Sinkhorn stateful logic — cần cân nhắc khi wrap).
Có thể chọn `DataParallel`, `DistributedDataParallel` với `torchrun`, hoặc strategy khác
— miễn là có lý giải rõ ràng về trade-off.

---

## Thứ tự fix

| Priority | Bug | File |
|---|---|---|
| 1 | Gamma gradient leak | `losses.py` |
| 2 | LR × 100 | `train.py` |
| 3 | [SEP] off-by-one | `losses.py` |
| 4 | Multi-GPU | `train.py` |

Fix Bug 1 và Bug 2 trước khi chạy bất kỳ experiment nào.
Bug 3 và Bug 4 có thể fix song song hoặc sau.
