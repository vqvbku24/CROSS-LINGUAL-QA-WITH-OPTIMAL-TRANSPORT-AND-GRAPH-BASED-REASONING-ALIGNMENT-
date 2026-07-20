# Spec: Vanilla KD Redesign (VI + AR) và Mở rộng nhánh Arabic

## 0. Nhận xét về bản plan gốc — đọc trước khi code

Bản plan gốc (`arabic/` branch spec) có cấu trúc tốt (tách thư mục, không đụng VI, có smoke test),
nhưng có **5 chỗ cần sửa/bổ sung** trước khi agent bắt tay viết code:

1. **Định nghĩa "Vanilla KD" trong plan gốc đã lỗi thời.** Plan gốc ghi
   *"Vanilla KD (train_baseline hoặc stage2 với lambda_ot=0, lambda_span=0, lambda_margin=0)"*.
   Sau khi review, đã chốt: M1 phải giữ `λ_reg > 0` (không zero nó — xem mục 2) và phải có thêm
   một loss KD thật (`L_kd`, naive index-to-index KL), không chỉ là "tắt hết".
2. **Thứ tự verification sai.** Plan gốc để "kiểm tra ID-alignment ZIZOU↔SQuAD2.0" ở mục
   *Manual Verification* cuối cùng. Đây phải là **gate chặn đầu tiên** — nếu ID space lệch,
   toàn bộ `squad_parallel_loader_ar.py` phải viết lại theo hướng khác.
3. **Thiếu xử lý `_normalize_answer` cho Arabic** (diacritics/tashkeel, định quán từ "ال").
   Nếu không xử lý, EM tiếng Arabic sẽ bị đánh giá sai lệch một cách âm thầm.
4. **"Static Alignment (OT)" / M2 đang bị đặt tên gây hiểu lầm** — đây là ablation nội bộ
   (config M2 của chính framework), không phải reimplementation của Sherborne et al. 2023.
   Không cần train lại, nhưng **label/log phải phản ánh đúng bản chất** để nhất quán với cách
   xử lý Vanilla KD (xem mục 3).
5. **M1 cần chạy lại cho CẢ VI lẫn AR**, không chỉ AR — để hai ngôn ngữ dùng chung một định
   nghĩa "Vanilla KD". Điều này có nghĩa là **phải sửa `train_stage2.py` (VI)** — mâu thuẫn với
   ràng buộc "không sửa file VI" trong plan gốc. Xem mục 1 (Blocking Approval Gate).

---

## 1. Blocking Approval Gate — cần xác nhận từ người dùng trước khi agent code

> [!WARNING]
> Việc thêm `L_kd` cho M1-VI đòi hỏi sửa `train_stage2.py` — file thuộc nhánh VI đã "đóng băng"
> theo ràng buộc ban đầu. Đề xuất: sửa theo kiểu **additive/backward-compatible** — thêm flag mới
> `--lambda_kd` (mặc định `0.0`, tắt), không đổi behavior mặc định của các run cũ. Mọi kết quả
> Table 2/4 hiện tại (M0, M2-M5) vẫn tái tạo được y hệt nếu không truyền `--lambda_kd`.
> **Agent KHÔNG tự ý sửa `train_stage2.py` nếu chưa có xác nhận rõ ràng từ người dùng** — dừng
> lại và hỏi trước khi động vào file này.

---

## 2. Loss mới: Naive Index-to-Index KD (`L_kd`)

### File mới (dùng chung cho cả VI và AR): `losses/vanilla_kd_loss.py`

```python
"""
Naive Index-to-Index Knowledge Distillation loss.

Đại diện cho "Vanilla KD" (M1) trong ablation study — một baseline KD
alignment-free: không có ma trận gamma, không projection ngữ nghĩa, chỉ ép
logits theo đúng chỉ số vị trí (index) sau khi crop về cùng độ dài.

Failure mode mong đợi: khi thứ tự từ giữa 2 ngôn ngữ khác nhau (vd. "red apple"
vs "táo đỏ" — tính từ đổi vị trí), việc ép theo index thô sẽ học sai ranh giới
câu trả lời. Đây là baseline được thiết kế để LÀM NỔI BẬT lý do cần ma trận
gamma trong L_span, không phải để mô phỏng một phương pháp cụ thể trong
literature.
"""
import torch
import torch.nn.functional as F


def naive_index_to_index_kd_loss(
    student_start_logits: torch.Tensor,   # [B, L_student]
    student_end_logits: torch.Tensor,     # [B, L_student]
    teacher_start_logits: torch.Tensor,   # [B, L_teacher]
    teacher_end_logits: torch.Tensor,     # [B, L_teacher]
    student_valid_len: torch.Tensor,      # [B] actual non-pad length per example
    teacher_valid_len: torch.Tensor,      # [B] actual non-pad length per example
    student_gold_start: torch.Tensor,     # [B] gold start index (student side)
    student_gold_end: torch.Tensor,       # [B] gold end index (student side)
    temperature: float = 2.0,
):
    """
    QUAN TRỌNG VỀ THỨ TỰ XỬ LÝ (đã review kỹ, không được đổi):
    1. Crop cả hai tensor về min(valid_len) TRƯỚC khi softmax — nếu softmax
       trên full sequence rồi mới crop xác suất, tổng xác suất sau crop sẽ
       không còn bằng 1 và F.kl_div sẽ tính sai.
    2. Loại (mask) các example mà việc crop cắt mất vị trí gold answer thật
       của phía dài hơn — không được âm thầm tính loss trên vị trí sai.
    """
    B = student_start_logits.size(0)
    device = student_start_logits.device

    min_len = torch.minimum(student_valid_len, teacher_valid_len)  # [B]
    max_crop_len = int(min_len.max().item())

    # Mask ví dụ mà crop sẽ cắt mất gold span thật (crop point < gold end)
    valid_mask = (student_gold_end < min_len) & (student_gold_start < min_len)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), valid_mask

    def crop_and_softmax(logits, lens, L):
        # crop theo per-example min_len, pad phần dư với -inf để không ảnh hưởng softmax
        cropped = logits[:, :L].clone()
        arange = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        pad_mask = arange >= lens.unsqueeze(1)                 # [B, L]
        cropped = cropped.masked_fill(pad_mask, float("-inf"))
        return cropped

    s_start = crop_and_softmax(student_start_logits, min_len, max_crop_len)
    s_end = crop_and_softmax(student_end_logits, min_len, max_crop_len)
    t_start = crop_and_softmax(teacher_start_logits, min_len, max_crop_len)
    t_end = crop_and_softmax(teacher_end_logits, min_len, max_crop_len)

    def kd_kl(s_logits, t_logits, mask):
        s_logits, t_logits = s_logits[mask], t_logits[mask]
        log_p_student = F.log_softmax(s_logits / temperature, dim=-1)
        p_teacher = F.softmax(t_logits / temperature, dim=-1)
        # batchmean + T^2 scaling: chuẩn Hinton et al. KD
        return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temperature ** 2)

    loss_start = kd_kl(s_start, t_start, valid_mask)
    loss_end = kd_kl(s_end, t_end, valid_mask)
    return 0.5 * (loss_start + loss_end), valid_mask
```

### Cần thêm vào cả `train_stage2.py` (VI, additive) và `arabic/train_stage2_ar.py` (AR, native)

- CLI flags mới: `--lambda_kd` (default `0.0`), `--kd_temperature` (default `2.0`).
- Trong vòng lặp loss: `L_total += lambda_kd * L_kd_naive` (chỉ cộng nếu `lambda_kd > 0`).
- **Unit test bắt buộc trước khi train thật**: tạo 2 tensor giả với `L_student != L_teacher`,
  verify không lỗi shape, verify tổng softmax sau crop = 1.0, verify các example bị mask đúng
  khi gold span nằm ngoài vùng crop.
- Log tỷ lệ `valid_mask.mean()` mỗi step đầu training — nếu tỷ lệ ví dụ bị loại quá cao
  (vd. > 30%), đây là dấu hiệu phân bố độ dài EN/target quá lệch, cần báo lại trước khi
  train full.

### Cấu hình M1 mới (áp dụng cho cả VI và AR)

```
lambda_qa    = 0.3   (giữ nguyên)
lambda_ot    = 0.0
lambda_span  = 0.0
lambda_margin= 0.0
lambda_reg   = 50.0  (GIỮ NGUYÊN — không zero, xem mục 1)
lambda_kd    = <cần chọn, đề xuất bắt đầu 1.0, xem mục 5>
kd_temperature = 2.0 (giá trị chuẩn trong literature KD, ghi rõ trong Appendix)
```

---

## 3. Đổi nhãn (không đổi số) cho Static Alignment (OT) / M2

Không cần train lại M2 (số liệu VI hiện tại vẫn giữ nguyên). Chỉ cần:

- Trong log/config name của M2: đổi nhãn nội bộ từ `static_ot_baseline` thành
  `ablation_ot_only_ours` để không gây hiểu lầm đây là reimplementation của Sherborne et al.
- Việc sửa **câu chữ trong paper** (không phải code) — đổi mô tả Table 2 dòng "Static Alignment
  (OT)" thành: *"an ablation of our own framework retaining only $\mathcal{L}_{ot}$ and
  $\mathcal{L}_{reg}$ (equivalent to M2 in Table 4)"* — việc này làm ở paper, không phải trong
  spec code này, nhưng ghi lại đây để không quên khi viết lại Section 4.2/4.4.

---

## 4. Thứ tự thực hiện (các gate chặn, làm đúng thứ tự)

| Bước | Việc | Chặn bởi |
|---|---|---|
| 0 | **[BLOCKING]** Xác nhận approval sửa `train_stage2.py` (mục 1) | Người dùng phải confirm |
| 1 | **[BLOCKING]** Kiểm tra ID-alignment `ZIZOUArabic_Squad/train.json` ↔ `Squad2.0/train-v2.0.json` — chạy `arabic/squad_parallel_loader_ar.py`, log số cặp align được / tổng số | Nếu match rate thấp bất thường (đề xuất ngưỡng cảnh báo < 80%), dừng và báo cáo trước khi viết tiếp |
| 2 | Viết + unit-test `losses/vanilla_kd_loss.py` (mục 2) | Bước 0 |
| 3 | Kiểm tra/sửa `_normalize_answer` cho Arabic (diacritics, "ال") — test trên vài cặp ví dụ tay trước khi tin số EM | — |
| 4 | Thêm `--lambda_kd`/`--kd_temperature` vào `train_stage2.py` (VI, additive) | Bước 0, 2 |
| 5 | Build `arabic/` (các file theo plan gốc: `squad_parallel_loader_ar.py`, `xquad_loader_ar.py`, `quick_eval_ar.py`, `generate_preds_ar.py`, `train_stage2_ar.py` — có `lambda_kd` native từ đầu) | Bước 1, 2, 3 |
| 6 | Chạy lại M1-VI với `L_kd` mới (thay số Vanilla KD cũ trong Table 2/4-VI) | Bước 4 |
| 7 | Chạy `run_ar.sh`: Zero-shot → Vanilla KD (mới) → M2 (ablation, dùng chung checkpoint cho cả Table 2 và ablation) → M5 | Bước 5, 6 (để đồng bộ cấu hình) |
| 8 | `run_ablation_ar.sh`: **tái sử dụng checkpoint M2 và M5 từ bước 7**, không train lại | Bước 7 |
| 9 | Compute cost: dùng `measure_compute_cost.py` (không sửa) đo Full FT / Vanilla KD (mới) / Ours | Bước 6 hoặc 7 (tùy ngôn ngữ đo) |

---

## 5. Việc cần người dùng quyết định trước khi agent chạy full training

1. **Giá trị `lambda_kd` và `kd_temperature`** — đề xuất bắt đầu `lambda_kd=1.0`, `T=2.0` (giá trị
   phổ biến trong literature KD), nhưng nên chạy thử nghiệm nhanh (vài trăm step) ở 1-2 giá trị
   khác nhau trước khi commit vào run full, vì đây là hyperparameter mới chưa từng tune.
2. **Ngưỡng cảnh báo ID-alignment** (đề xuất 80% — người dùng có thể muốn ngưỡng khác).
3. **Xác nhận approval sửa `train_stage2.py`** (mục 1) — bắt buộc phải có trước khi agent code
   bước 4.

---

## 6. Không đổi (giữ nguyên theo plan gốc, đã hợp lý)

- Cấu trúc thư mục `arabic/` tách biệt hoàn toàn.
- `measure_compute_cost.py` — không sửa, đã generic đủ dùng.
- Smoke test plan (`squad_parallel_loader_ar.py`, `xquad_loader_ar.py`, `generate_preds_ar.py`)
  — giữ nguyên, chỉ đẩy bước ID-alignment lên làm đầu tiên (mục 4).