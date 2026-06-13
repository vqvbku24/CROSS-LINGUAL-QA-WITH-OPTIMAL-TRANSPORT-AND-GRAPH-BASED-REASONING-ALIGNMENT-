**Nhiệm vụ:**
Hãy viết cho tôi một module Python tên là `squad_parallel_loader.py` dùng để tạo PyTorch `Dataset` và `DataLoader` phục vụ cho việc huấn luyện mô hình học sâu với dữ liệu song song (Parallel Data) Anh - Việt.

**Nguồn dữ liệu:**
1. Tập tiếng Anh: SQuaD2.0
2. Tập tiếng Việt: `AIForge/vietnamese-squad`

**Yêu cầu cốt lõi về Logic Dóng hàng (Alignment Logic):**
Tập tiếng Việt là bản dịch của tập tiếng Anh, nhưng để đảm bảo an toàn tuyệt đối, KHÔNG được map theo thứ tự dòng (zip). Bắt buộc phải dóng hàng (align) dựa trên cột `id`:
- Biến tập tiếng Anh thành một Dictionary với key là `id` để tra cứu với độ phức tạp O(1).
- Duyệt qua tập tiếng Việt, lấy `id` đi tra cứu trong Dictionary tiếng Anh. Nếu khớp, đưa cặp (English_item, Vietnamese_item) này vào danh sách dữ liệu song song.

**Yêu cầu xử lý trong hàm `__getitem__`:**
Đối với mỗi cặp song song lấy được, thực hiện các bước sau:
1. Lấy ra `question` và `context` của cả bản EN và VI (Lưu ý dùng `.get()` để phòng hờ trường hợp tên cột bị viết hoa chữ cái đầu như `Question` hay `Context`).
2. Tokenize nhánh EN (Question + Context) bằng biến `self.tokenizer` được truyền vào từ class. Yêu cầu: `truncation=True`, `padding="max_length"`, `return_tensors="pt"`.
3. Tokenize nhánh VI tương tự như trên.
4. **Đặc biệt quan trọng:** Tìm vị trí index của token `[SEP]` (sử dụng `self.tokenizer.sep_token_id`) đầu tiên xuất hiện trong chuỗi `input_ids`. Mục đích là để phân tách ranh giới giữa câu hỏi và đoạn văn bản. Lưu index này vào các biến `en_question_end` và `vi_question_end`. Nếu không tìm thấy, trả về 0.

**Định dạng Output của `__getitem__`:**
Hàm phải trả về một dictionary chính xác với các keys sau (tất cả là 1D tensor sau khi bỏ đi chiều batch đầu tiên):
- `"en_input_ids"`
- `"en_attention_mask"`
- `"en_question_end"`
- `"vi_input_ids"`
- `"vi_attention_mask"`
- `"vi_question_end"`

**Yêu cầu về DataLoader:**
Viết thêm một hàm `create_squad_parallel_dataloaders(tokenizer, batch_size=32, max_length=384)` trả về `train_loader` và đối tượng `dataset`. DataLoader BẮT BUỘC phải có `shuffle=True`, `num_workers=4` và `pin_memory=True`.

**Log & Debug:**
Sử dụng thư viện `logging` (mức INFO) để in ra các bước đang chạy (ví dụ: đang tải dữ liệu, đang map ID, và kết quả cuối cùng tìm được bao nhiêu cặp song song). Code cần gọn gàng, có comment giải thích rõ ràng.