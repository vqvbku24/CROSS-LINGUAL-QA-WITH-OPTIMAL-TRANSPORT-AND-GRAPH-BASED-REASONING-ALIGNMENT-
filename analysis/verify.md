# INVESTIGATION_SPEC_HINDI_ANOMALY.md

## Bối cảnh (Context)

Trong Table 9 / Appendix I.1, nhánh Hindi cho thấy XQuAD-hi F1 đạt đỉnh ở epoch 1
(67.09) rồi suy giảm đơn điệu qua epoch 2-4 (65.61 → 62.69 → 55.25), trong khi
SQuAD-EN F1 tăng đơn điệu suốt cùng khoảng epoch (72.61 → 74.13). Pattern này
lặp lại ổn định qua 3 seeds (SD nhỏ), nghĩa là đây là hiện tượng **hệ thống**,
không phải nhiễu ngẫu nhiên của 1 seed.

**Câu hỏi nghiên cứu**: Cơ chế nào gây ra sự suy giảm sớm và đơn điệu này, và
tại sao nó không xảy ra (hoặc xảy ra chậm hơn nhiều) ở Vietnamese/Arabic?

**Nguyên tắc bắt buộc**: Đây là investigation, KHÔNG PHẢI bugfix. Agent không
được thay đổi hành vi training, không train lại model, không tạo checkpoint
mới. Mọi phân tích dưới đây chỉ là suy luận (inference) chạy trên checkpoint
đã lưu sẵn (epoch 1-4, 3 seeds, cho VI/AR/HI). Nếu checkpoint không tồn tại ở
dạng cần thiết (ví dụ cần cả 4 epoch cho VI để so sánh), báo lại cho user thay
vì tự ý train bù.

---

## Priority Table

| # | Giả thuyết | Effort | Bằng chứng nếu đúng | Bằng chứng nếu sai |
|---|---|---|---|---|
| H1 | Tokenization fragmentation: Devanagari bị XLM-R tách subword nhiều/thô hơn Latin script → OT cost matrix khó khớp | Thấp (~30 phút, không cần load model) | HI có subword/word ratio cao hơn rõ rệt so với VI/AR | Ratio tương đương → loại H1 |
| H2 | Transport plan entropy: γ của HI "loãng" hơn / kém coherent hơn ngay từ epoch 1, hoặc entropy tăng nhanh hơn qua epoch so với VI/AR | Trung bình (cần forward pass qua checkpoint đã lưu, không cần backward) | Entropy(γ_HI) cao hơn Entropy(γ_VI, γ_AR) tại cùng epoch, và/hoặc tăng dốc hơn theo epoch | Entropy tương đương giữa 3 ngôn ngữ → loại H2, cơ chế OT không phải nguyên nhân |
| H3 | Representation drift: cosine alignment suy giảm nhanh hơn / norm drift lớn hơn ở HI giữa epoch 1 và epoch 4, so với VI ở cùng khoảng epoch tương đối | Trung bình (tái dùng pipeline Appendix F, đổi input) | Δcosine, Δnorm của HI (ep1→ep4) lớn hơn rõ rệt so với VI (ep1→ep3, checkpoint đã chọn) | Không khác biệt đáng kể → loại H3 |
| H4 (đối chứng) | Data-distributional: IndicSQuAD-hi khác biệt về answer length / answerable ratio / context length so với AIForge-VI, Arabic-SQuAD, khiến pattern là do data-quality chứ không phải typology | Thấp (thống kê mô tả trên dataset, không cần model) | HI có phân phối lệch rõ rệt (answer length, %unanswerable, context length) so với VI/AR | Phân phối tương đương → loại H4, củng cố hướng H1-H3 |

**Thứ tự chạy khuyến nghị**: H1 → H4 trước (rẻ, không cần load model, có thể loại
trừ nhanh các giả thuyết sai), sau đó mới chạy H2 → H3 (cần load checkpoint).

---

## H1: Tokenization Fragmentation Ratio

**Vị trí**: Không sửa code hiện có — viết script mới, độc lập, đặt tại
`analysis/tokenization_fragmentation.py` (agent tự xác nhận thư mục `analysis/`
hoặc tương đương đã tồn tại chưa; nếu chưa, tạo mới, không đụng vào thư mục
`src/` hoặc `training/`).

**Việc cần làm**:
```python
# Load cùng 1 tokenizer XLM-R dùng trong training (agent tự tìm import trong
# code hiện có, KHÔNG hardcode "xlm-roberta-base" nếu repo dùng checkpoint path khác)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(<path_dùng_trong_training>)

def fragmentation_ratio(text: str) -> float:
    words = text.split()  # whitespace tokenization làm proxy cho "word"
    subwords = tokenizer.tokenize(text)
    return len(subwords) / max(len(words), 1)

# Chạy trên toàn bộ validation context của 3 ngôn ngữ:
# - AIForge Vietnamese-SQuAD (validation split)
# - Arabic-SQuAD / ARCD (validation split)
# - IndicSQuAD Hindi (validation split)
# Report: mean, median, std của fragmentation_ratio cho mỗi ngôn ngữ
```

**Output cần tạo**: bảng CSV/markdown `fragmentation_ratio_by_language.csv` với
cột `language, mean_ratio, median_ratio, std_ratio, n_samples`.

**Tiêu chí diễn giải**: nếu HI ratio > VI/AR ratio một cách rõ rệt (ví dụ >20%
cao hơn), đây là bằng chứng ủng hộ H1. Nếu không, loại H1 và ghi rõ trong báo
cáo cuối (không được bỏ qua kết quả âm tính).

---

## H4: Data-Distributional Check (nên chạy song song/trước H1 vì rẻ hơn)

**Vị trí**: cùng thư mục `analysis/`, file `analysis/dataset_distribution_check.py`.

**Việc cần làm**: tính, cho mỗi ngôn ngữ (VI, AR, HI), trên validation split:
- Answer length (số từ, số token) — mean/median/std
- % câu hỏi unanswerable (nếu dataset có field is_impossible/answerable)
- Context length (số token) — mean/median/std
- Answer position trong context (đầu/giữa/cuối, tính theo % vị trí ký tự)

**Output**: bảng `dataset_distribution_by_language.csv`.

**Tiêu chí diễn giải**: nếu HI có answer length hoặc answerable ratio lệch rõ
rệt so với VI/AR, đây là lời giải thích thay thế (không loại trừ H1-H3, có thể
cộng hưởng). Phải báo cáo dù kết quả ủng hộ hay bác bỏ.

---

## H2: Sinkhorn Transport Plan Entropy theo Epoch

**Vị trí**: KHÔNG viết lại thuật toán Sinkhorn. Agent cần tìm hàm hiện có tính
transport plan γ (rất có thể tên dạng `compute_ot_plan`, `sinkhorn_solver`, hoặc
tương tự — tìm bằng cách grep từ khóa `sinkhorn`, `transport_plan`, `gamma` /
`γ` trong repo). Script mới gọi lại hàm đó ở chế độ **inference-only**
(`torch.no_grad()`), KHÔNG chỉnh sửa hàm gốc.

**Việc cần làm**:
```python
# Với mỗi checkpoint đã lưu (epoch 1-4, seed 42/43/44, cho VI và HI):
# 1. Load checkpoint (frozen teacher + LoRA student tại epoch đó)
# 2. Chạy forward pass trên toàn bộ validation set (hoặc subsample n=200 nếu
#    validation set quá lớn để chạy hết trong thời gian hợp lý — ghi rõ n dùng)
# 3. Lấy transport plan gamma cho mỗi example
# 4. Tính Shannon entropy của gamma (chuẩn hóa theo hàng, tức mỗi source token
#    có 1 phân phối xác suất trên target tokens):

import torch

def row_entropy(gamma: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # gamma: [L_src, L_tgt], mỗi hàng đã (hoặc cần) chuẩn hóa thành phân phối xác suất
    row_normalized = gamma / (gamma.sum(dim=-1, keepdim=True) + eps)
    entropy = -(row_normalized * (row_normalized + eps).log()).sum(dim=-1)
    return entropy.mean()  # entropy trung bình trên các source token

# Aggregate: mean entropy per (language, epoch, seed)
```

**Output**: `transport_entropy_by_epoch_language.csv` với cột
`language, epoch, seed, mean_entropy, std_entropy`, kèm 1 line chart (entropy
vs epoch, 1 đường mỗi ngôn ngữ) lưu dạng `.png` hoặc artifact.

**Tiêu chí diễn giải**:
- Nếu Entropy(HI) cao hơn Entropy(VI) rõ rệt tại epoch 1 VÀ/HOẶC tăng dốc hơn
  qua epoch → ủng hộ H2 (OT alignment kém coherent hơn với Hindi).
- Nếu tương đương → loại H2, và điều đó có nghĩa cơ chế suy giảm không nằm ở
  transport plan mà có thể ở domain-anchor (Lreg) hoặc nơi khác — cần nêu rõ
  trong báo cáo, không được suy diễn thêm nếu không có dữ liệu hỗ trợ.

**Ràng buộc thời gian/compute**: nếu forward pass toàn bộ validation set cho
4 epoch × 3 seed × 2 ngôn ngữ quá tốn thời gian, agent được phép subsample
nhưng PHẢI báo cáo n và cách sample (random seed cố định để tái lập).

---

## H3: Representation Drift (tái dùng pipeline Appendix F)

**Vị trí**: repo đã có sẵn code sinh Figure 5 (Appendix F) — cosine alignment,
Euclidean distance, norm change, before/after Stage-2. Agent cần tìm file đó
(grep theo tên biến/hàm liên quan đến `cosine_similarity`, `anisotropy`, hoặc
tên file có "geometric" / "diagnostic"). KHÔNG sửa file gốc — copy logic ra
script mới `analysis/hindi_representation_drift.py`, tham số hóa để nhận input
là (language, epoch, seed) thay vì hardcode Vietnamese.

**Việc cần làm**: chạy lại đúng 4 diagnostic đã có (panel a-d của Figure 5) cho:
- Hindi tại epoch 1 (checkpoint đã chọn) so với epoch 4 (n=50 sampled pairs,
  giống thiết kế gốc)
- Vietnamese tại epoch 3 (checkpoint đã chọn) — dùng lại số liệu đã có nếu còn,
  không cần chạy lại nếu Appendix F đã lưu kết quả thô

**Output**: bảng so sánh side-by-side HI vs VI cho 4 chỉ số: cosine similarity
before/after, Euclidean distance before/after, norm change, kèm paired t-test
+ Wilcoxon giống format đã dùng trong Appendix F (giữ nguyên chuẩn thống kê đã
thiết lập, không đổi phương pháp).

**Tiêu chí diễn giải**: nếu HI cho thấy cosine alignment giảm mạnh hơn / norm
drift lớn hơn (theo effect size, không chỉ p-value vì n=50 nhỏ) so với VI ở
cùng độ dài huấn luyện tương đối → ủng hộ H3. Nếu không → loại H3.

---

## No-Touch Zones

- KHÔNG sửa training loop, loss function, hoặc bất kỳ file nào ảnh hưởng đến
  cách checkpoint được tạo ra (đây là investigation trên checkpoint đã có sẵn).
- KHÔNG train lại hoặc tạo checkpoint mới dưới bất kỳ hình thức nào.
- KHÔNG sửa các script đã dùng để tạo Table 2, 4, 5, 8, 9, 12 hoặc Figure 4, 5,
  7 trong bài đã nộp — nếu cần tái sử dụng logic, COPY sang file mới trong
  `analysis/`.
- KHÔNG tự động kết luận nguyên nhân nếu chỉ có 1/4 giả thuyết được test —
  phải chạy đủ H1, H4 trước (rẻ), sau đó H2, H3 nếu còn thời gian, và báo cáo
  đầy đủ kết quả kể cả khi tất cả đều "không ủng hộ giả thuyết nào" (kết quả âm
  tính vẫn phải được viết vào output).

## Output tổng hợp cần agent tạo cuối cùng

Một file `HINDI_ANOMALY_FINDINGS.md` tóm tắt:
1. Bảng kết quả 4 giả thuyết (ủng hộ / không ủng hộ / không đủ dữ liệu)
2. Với mỗi giả thuyết: số liệu cụ thể, không diễn giải quá mức
3. Kết luận: giả thuyết nào (nếu có) có bằng chứng mạnh nhất, và có cần thêm
   phân tích nào không được liệt kê ở đây hay không (agent được phép đề xuất
   thêm hướng nếu phát hiện điều bất thường trong lúc chạy)
4. Rõ ràng ghi nhận nếu kết quả không đủ để kết luận (không "cố" tìm ra câu
   trả lời đẹp — mục tiêu là evidence, không phải narrative)