### BUGFIX_SPEC.md: Khắc phục lỗi rò rỉ Question và crash Sinkhorn

#### 1. Bug 1.1: Rò rỉ Logits vào Question Tokens

* **File ảnh hưởng:** `quick_eval.py`
* **Hàm cần sửa:** `quick_em` và `quick_em_xquad_vi`
* **Vấn đề:** Model hiện tại chỉ mask các padding tokens (`attn_mask == 0`), dẫn đến việc QA head có thể trích xuất nhầm các token nằm trong câu hỏi thay vì context (vi phạm nguyên tắc extractive QA).
* **Giải pháp:** Bổ sung `question_mask` dựa trên `q_end_val` để ép logits của vùng câu hỏi về $-\infty$.
* **Chi tiết triển khai:**
Chèn đoạn code sau ngay bên dưới logic `padding_mask`:
```python
# Mask out padding tokens
padding_mask = (attn_mask[0] == 0)
start_logits[0].masked_fill_(padding_mask, float('-inf'))
end_logits[0].masked_fill_(padding_mask, float('-inf'))

# [NEW] Mask out question tokens (từ index 0 đến q_end_val)
question_mask = torch.arange(start_logits.size(1), device=device) <= q_end_val
start_logits[0].masked_fill_(question_mask, float('-inf'))
end_logits[0].masked_fill_(question_mask, float('-inf'))

```



#### 2. Bug 1.2: Rủi ro $\log(0)$ sinh ra $-\infty$ trong Sinkhorn Log-domain

* **File ảnh hưởng:** `losses.py`
* **Hàm cần sửa:** `sinkhorn_masked`
* **Vấn đề:** Trong trường hợp `mu_override` được kích hoạt và chứa các giá trị tiệm cận hoặc bằng 0, phép toán `torch.log(mu)` sẽ trả về $-\infty$. Điều này gây ra lỗi `NaN` ở các bước tính toán `logsumexp` tiếp theo, làm sụp đổ toàn bộ gradient của batch.
* **Giải pháp:** Thêm hàm `.clamp(min=1e-8)` vào `mu` (tương tự như đã làm với `nu` nếu cần thiết) để đảm bảo an toàn số học.
* **Chi tiết triển khai:**
Sửa đổi dòng tính `log_u` (khoảng dòng 761):
```python
# [OLD]
# log_u = torch.log(mu) - torch.logsumexp(log_K + log_v[None, :], dim=1)

# [NEW] Clamp mu để tránh log(0) -> -inf
log_u = torch.log(mu.clamp(min=1e-8)) - torch.logsumexp(log_K + log_v[None, :], dim=1)

# Giữ nguyên log_v hoặc kẹp thêm nu cho đồng bộ tính an toàn
log_v = torch.log(nu.clamp(min=1e-8)) - torch.logsumexp(log_K + log_u[:, None], dim=0)