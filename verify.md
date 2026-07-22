# Spec: Verification Tasks + Figure Regeneration (v2)

**Cách dùng:** Đưa file này cho coding agent CÙNG VỚI file `acl_latex_v3.tex` (paper hiện
tại) và toàn bộ codebase/log training thật. Agent cần đọc cả hai, đối chiếu chéo, rồi thực
hiện đúng theo thứ tự dưới đây. **Không tự đoán số liệu hoặc suy diễn khi thiếu thông tin
— nếu code/log không xác nhận được điều gì, dừng lại và báo cáo, không tự điền giá trị.**

---

## PHẦN 1 — Verification Tasks (bắt buộc làm trước, ảnh hưởng trực tiếp đến Phần 2)

### V1. [CHẶN Figure 2] Xác minh công thức Dynamic Margin thật trong code

Paper hiện viết công thức:
```
L_margin = ReLU(m_src.detach() - m_tgt)
```
Công thức này **không chứa biến schedule nào**, dù prose mô tả margin threshold được anneal
theo epoch. Đồng thời `λ_margin` đang bị dùng cho 2 đại lượng khác nhau: (a) trọng số loss
cố định `λ_margin = 1.0` (Table 6), và (b) giá trị schedule biến thiên theo epoch (Figure 2).

**Việc cần làm:**
1. Tìm đúng đoạn code implement `L_margin` trong training loop thật.
2. Xác định chính xác: biến schedule (gọi là `m(ε)` để tách khỏi `λ_margin`) tham gia vào
   công thức ở đâu — nhân vào `m_src`? Cộng/trừ như một threshold riêng? Hay điều chỉnh
   `ReLU(...)` theo cách khác?
3. Viết lại công thức đúng với ký hiệu `m(ε)` tách biệt, ví dụ (chỉ là ví dụ khả dĩ, PHẢI
   xác nhận với code thật trước khi dùng):
   ```
   L_margin = ReLU(m(ε) · m_src.detach() - m_tgt)
   ```
4. Xác định lại **motivation text có nhất quán với chiều hướng thật của m(ε) không**:
   - Nếu `m(ε)` lớn (1.0) → tương ứng ràng buộc CHẶT, `m(ε)` nhỏ (0.3) → ràng buộc LỎNG:
     thì lịch trình `m(ε)`: 1.0 (3 epoch đầu) → 0.7 → 0.5 → 0.3 (đã xác nhận ở V2) nghĩa là
     **CHẶT ở đầu, LỎNG ở cuối** — điều này NGƯỢC với prose hiện tại ("relaxing margin
     early... enforcing strict boundary discrimination" ở cuối). Nếu đúng vậy, phải viết
     lại motivation trong Section 3.4, không phải đổi số.
   - Nếu ngược lại (giá trị `m(ε)` lớn = ràng buộc LỎNG), thì prose hiện tại đúng, không
     cần sửa motivation.
   - **Agent phải xác định đúng chiều này dựa trên code thật, không suy đoán.**

### V3. Kiểm tra data overlap / contamination
Kiểm tra overlap giữa Vietnamese-SQuAD (dùng train Stage 2) và XQuAD-vi/MLQA-vi (dùng eval)
theo: ID, question text, context text, nguồn SQuAD gốc. Báo cáo số lượng overlap tìm được
(nếu có), không giả định là sạch nếu chưa kiểm tra thật

XQuAD-vi, MLQA-vi, XQuAD-en, MLQA-en. Checkpoint selection for all models is based strictly on early stopping evaluated on the XQuAD-vi validation split; we do not use the evaluation benchmarks for model tuning. Hãy sửa trên file latex để hợp lí phần này.

### V5. Xác nhận data usage chi tiết cho Appendix B.1
Xác nhận: dùng split Vietnamese-SQuAD train cho Stage 2; cách ghép
cặp EN-VI; xác nhận rõ gold English span có được dùng để tạo pseudo-label qua γ-projection
hay không (hiện paper đã có công thức này ở Section 3.2 nhưng Appendix B.1 mô tả data
pipeline chưa nói rõ điểm này). Bạn giúp mình note lại và chỉnh sửa file latex cho hợp lí luôn,

---

## PHẦN 2 — Figure Regeneration (thực hiện sau khi V1, V2 đã xác nhận xong)

### F1. Figure 1 (Overview)
**Bỏ qua — user tự điều chỉnh.**

### F2. Figure 2 (Dynamic Margin Schedule)
- Vẽ lại step function theo đúng lịch trình đã xác nhận ở V2: epoch 1 giữ là 0, epoch 2–3 giữ 1.0, epoch 4
  = 0.7, epoch 5 = 0.5, epoch 6–8 giữ 0.3. Trục x: `Epoch`, chạy 1 đến 8 (không phải 1-6
  như bản cũ).
- Trục y: đổi label từ `λ_margin` (gây symbol collision, xem V1) sang `m(ε)` hoặc
  "Margin Threshold" — không dùng lại ký hiệu `λ_margin` cho trục này.
- Caption cần cập nhật khớp con số mới: "annealed in discrete steps (1.0 → 0.7 → 0.5 →
  0.3) across 8 training epochs" thay vì "1.0 to 0.3... across the training epochs" mơ hồ
  như cũ.
- **Chỉ vẽ sau khi V1 xác nhận xong** — nếu V1 phát hiện motivation cần viết lại
  (chiều ngược), phải sửa luôn câu chữ Section 3.4 cùng lúc với hình, không tách rời.

### F3. Figures 3 & 4 (Radar M0–M5 → Grouped Bar Chart)
- Đổi từ radar chart sang **grouped bar chart**, 3 nhóm cột trên trục x: SQuAD-EN
  (Source), XQuAD-VI (Target), MLQA-VI (Target); mỗi nhóm có 6 cột con (M0–M5), phân biệt
  bằng màu/pattern.
- Figure 3: giá trị F1. Figure 4: giá trị EM. Giữ đúng số liệu hiện có trong Table 4
  (không đổi số, chỉ đổi cách vẽ).
- Font trục và legend đủ lớn để đọc được khi thu nhỏ vào cột đơn của layout 2-column ACL
  (tối thiểu 12-14pt sau khi scale).
- Sắp xếp thứ tự cột con M0→M5 nhất quán giữa Figure 3 và Figure 4, cùng bảng màu.
- Cân nhắc thêm đường kẻ ngang (horizontal reference line) tại giá trị M0 (baseline) trên
  mỗi nhóm để dễ so sánh trực quan "cao hơn/thấp hơn baseline bao nhiêu" — không bắt buộc
  nhưng nên có nếu không tốn nhiều công.

### F4. Figures 5 & 6 (XQuAD-vi/MLQA-vi F1+EM → gộp 1 figure 2×2)
- Gộp 4 subplot hiện tại (Figure 5: F1+EM trên XQuAD-vi; Figure 6: F1+EM trên MLQA-vi)
  thành **1 figure duy nhất, layout 2×2 grid**: hàng 1 = XQuAD-vi (F1 | EM), hàng 2 =
  MLQA-vi (F1 | EM). Hoặc cột theo dataset, hàng theo metric — chọn layout nào rõ ràng
  hơn khi test thử, miễn nhất quán với F5 (Appendix, cùng layout).
- Trục x mọi subplot: `Epoch`, chạy 1 đến 8 (theo V2).
- Giữ nguyên 2 đường M4 (Static Margin) và M5 (Dynamic Curriculum Margin) như hiện tại.
- Một caption chung duy nhất cho cả 4 panel, không lặp lại 2 caption riêng như cũ.
- Cập nhật `\label{}` mới (ví dụ `fig:margin_dynamics_vi_combined`) và sửa mọi
  `\ref{}` trong Section 4.5 trỏ tới label cũ (`fig:margin_dynamics_xquad`,
  `fig:margin_dynamics_mlqa`) thành label mới.

### F5. Figures 9 & 10 (Appendix I, held-out EN — gộp tương tự F4)
- Áp dụng đúng layout 2×2 grid như F4 (XQuAD-en/MLQA-en × F1/EM) — giữ nhất quán trực quan
  giữa 2 cặp figure (main text vs appendix).
- Trục x: 1 đến 8 (theo V2).
- Cập nhật `\label{}` mới, sửa `\ref{}` trong Appendix I text trỏ tới label cũ
  (`fig:margin_dynamics_xquad_en`, `fig:margin_dynamics_mlqa_en`).
- Lựa chọn được đề cập trong yêu cầu: có thể gộp CHUNG cả 4 dataset (XQuAD-vi, MLQA-vi,
  XQuAD-en, MLQA-en) vào 1 figure 4×2 lớn duy nhất thay vì 2 figure 2×2 riêng — nếu làm
  vậy, đặt ở main text (Section 4.5) một bản rút gọn (chỉ 2 dataset VI) và giữ bản đầy đủ
  4 dataset ở Appendix, tránh trùng lặp hoàn toàn nội dung giữa main text và appendix.
  **Quyết định layout cụ thể (2 figure riêng vs 1 figure gộp 4 dataset) để user chọn sau
  khi xem bản nháp — không tự quyết định thay.**

### F6. Không cần sửa
Figure 7 (layer mixing), Figure 8 (geometric diagnostics 6-panel), Figure 11 (transport
plan heatmap) — không có yêu cầu thay đổi từ review.

---

## PHẦN 3 — Sau khi vẽ lại xong

1. Cập nhật mọi `\ref{}` trong `.tex` trỏ đúng label mới của các figure đã gộp (F4, F5).
2. Cập nhật caption Figure 2 khớp lịch trình mới (F2).
3. Nếu V1 xác nhận motivation Section 3.4 cần viết lại, sửa cùng lúc với F2 — không nộp
   bản có hình mới nhưng text cũ mâu thuẫn.
4. Chạy lại kiểm tra brace-balance / figure-environment-balance trên `.tex` sau khi đổi
   `\label`/`\ref` (dễ gây lỗi compile nếu label bị trùng hoặc ref trỏ sai).
5. Báo cáo lại cho user: kết quả V1–V5 (đặc biệt V1, V3, V4 — có phát hiện vấn đề gì không),
   và gửi kèm các file hình mới để user duyệt trước khi chèn chính thức vào bài.