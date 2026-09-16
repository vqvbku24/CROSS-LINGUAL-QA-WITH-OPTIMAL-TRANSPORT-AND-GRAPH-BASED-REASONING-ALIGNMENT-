# Spec: Retrain MAD-X trên SQuAD 2.0 (Baseline riêng, thư mục độc lập)

**Mục tiêu:** Sửa lỗi so sánh không công bằng hiện tại — dòng "MAD-X" trong Table 2 hiện dùng adapter **published, train trên SQuAD 1.1** (không có unanswerable), trong khi được chấm điểm bằng protocol SQuAD 2.0 của bạn (answerable+unanswerable jointly). Đây là nguyên nhân điểm thấp bất thường bạn quan sát được, không phải bug inference.

Spec này hướng dẫn **retrain riêng 1 MAD-X task adapter trên SQuAD 2.0**, để có 1 con số MAD-X thực sự so sánh được cạnh "Ours".

---

## 0. Khác biệt kiến trúc quan trọng — KHÔNG dùng chung hạ tầng với InfoXLM/MINOTAUR

MAD-X (Pfeiffer et al., 2020) có kiến trúc **hoàn toàn khác** framework chính của bạn:
- Backbone **frozen hoàn toàn** (không fine-tune, không LoRA).
- Chỉ train **task adapter** (bottleneck nhỏ, reduction factor 16 → dim 48 với hidden=768) chồng lên **language adapter** (đã pretrain sẵn qua MLM, reduction factor 2 → dim 384, frozen).
- **Không có teacher-student, không có LoRA, không có domain anchor, không có OT.** Cơ chế "bảo vệ nguồn" của MAD-X là **modularity cứng**: task adapter + English language adapter không hề bị đụng tới khi chuyển sang tiếng Việt (chỉ swap language adapter, task adapter giữ nguyên) — tức Δsrc ≈ 0 **theo thiết kế**, không phải nhờ regularization học được như $\mathcal{L}_{domain}$ của bạn. Đây là điểm đáng nêu trong Discussion: 2 triết lý khác nhau để đạt cùng mục tiêu (source preservation) — soft constraint (của bạn, linh hoạt hơn nhưng cần tune λ_reg) vs. hard modularity (MAD-X, "miễn phí" nhưng cứng nhắc, không thể tiếp tục thích nghi với đặc trưng bề mặt của ngôn ngữ đích ngoài những gì language adapter đã học sẵn).

**Do đó:** baseline này **KHÔNG import** `shared/train_loop_base.py` (vốn thiết kế cho vòng lặp LoRA 3-forward-pass). Chỉ tái sử dụng: (a) đúng file data SQuAD 2.0 gốc, (b) `shared/eval_harness.py` để tính EM/F1/Δsrc theo cùng công thức, (c) cùng quy ước xử lý câu hỏi không có câu trả lời.

## 1. Thư mục

```
baselines/
└── madx_retrained/
    ├── README.md
    ├── config.yaml
    ├── train_task_adapter.py      # train task adapter trên SQuAD 2.0
    ├── eval_zero_shot.py          # swap language adapter EN→VI, chạy shared/eval_harness.py
    └── outputs/
        ├── checkpoints/
        ├── logs/
        └── results.json           # cùng format results.json đã dùng cho InfoXLM/MINOTAUR
```

## 2. Cài đặt & Adapter IDs (đã xác nhận tồn tại trên Hub)

Dùng thư viện `adapters` (Poth et al., 2023 — kế thừa `adapter-transformers`):
```bash
pip install adapters --break-system-packages
```

```python
from adapters import AutoAdapterModel
from adapters.composition import Stack

model = AutoAdapterModel.from_pretrained("xlm-roberta-base")

# Language adapters — ĐÃ CÓ SẴN trên Hub, không tự train
en_lang = model.load_adapter("AdapterHub/xlm-roberta-base-en-wiki_pfeiffer", set_active=False)
vi_lang = model.load_adapter("AdapterHub/xlm-roberta-base-vi-wiki_pfeiffer", set_active=False)  # đã xác nhận tồn tại, config "pfeiffer", đúng backbone xlm-roberta-base

# Task adapter — PHẢI TỰ TRAIN (đây là phần việc chính của spec này)
model.add_adapter("qa_task", config="pfeiffer")   # reduction factor mặc định của "pfeiffer" preset = 16, khớp đúng dim=48 (768/16) như bài gốc
model.add_qa_head("qa_task", num_labels=2)

model.active_adapters = Stack(en_lang, "qa_task")
model.train_adapter("qa_task")   # freeze mọi thứ khác — chỉ task adapter + QA head được update
```

**⚠️ Bước bắt buộc trước khi chạy full training:** verify `en_lang` load thành công (không 404) bằng cách chạy thử `model.load_adapter(...)` riêng lẻ trước — dù đã đoán đúng pattern tên theo repo tiếng Việt, vẫn nên xác nhận trực tiếp trên Hub UI (`https://huggingface.co/AdapterHub`, tìm `xlm-roberta-base-en-wiki_pfeiffer`) trước khi tốn compute train.

## 3. Huấn luyện Task Adapter trên SQuAD 2.0

- **Data:** dùng đúng file `train-v2.0.json` bạn đang dùng cho Stage 1 teacher của framework chính (cùng nguồn, đảm bảo không lệch data).
- **Xử lý unanswerable:** thư viện `adapters`/`transformers` mặc định hướng câu trả lời null về vị trí token `<s>` (CLS-equivalent của XLM-R) — **trùng khớp sẵn** với convention bạn đã dùng ("Unanswerable question boundaries are structurally directed to the special token ⟨s⟩ at index 0"). Không cần custom thêm, nhưng **phải test 1 batch nhỏ để xác nhận thật sự khớp** trước khi train full, vì đôi khi thư viện định nghĩa offset khác đi giữa các phiên bản.
- **Hyperparameter** (theo đúng bài gốc Pfeiffer et al. 2020, Section 5.2): learning rate 1e-4 cho adapter (khác biệt so với 2e-5 của LoRA framework chính — đây là **giá trị đúng cho MAD-X**, không copy nhầm từ config LoRA của bạn), batch size theo compute cho phép, chọn checkpoint theo validation performance.
- **KHÔNG thêm** bất kỳ domain-consistency loss hay margin nào — giữ MAD-X nguyên bản, đúng nguyên tắc đã thống nhất (không cho baseline bên ngoài "mượn" cơ chế riêng của bạn).

## 4. Zero-shot Transfer & Checkpoint Selection

Sau khi có task adapter tốt nhất trên English dev set:
```python
model.active_adapters = Stack(vi_lang, "qa_task")  # chỉ swap language adapter, task adapter giữ nguyên
# chạy shared/eval_harness.py để tính EM/F1 trên XQuAD-vi, MLQA-vi
```

**2 cách chọn checkpoint — nên báo cáo cả hai, dùng cho 2 mục đích khác nhau:**

| Cách chọn checkpoint | Dùng khi nào | Ghi chú |
|---|---|---|
| **(a) Target-informed** (giống hệt cách bạn đang chọn checkpoint cho "Ours" — best joint target-val, trong 1 F1 so với teacher) | **Bảng chính (Table 2)**, để so sánh đầu-đối-đầu công bằng, cùng 1 protocol chọn checkpoint cho mọi phương pháp | Không cho MAD-X "thiệt" vì bị áp rule khắt khe hơn "Ours" |
| **(b) English-only** (chỉ nhìn English dev set, đúng tinh thần zero-shot nghiêm ngặt của MAD-X gốc — không hề "liếc" sang tiếng Việt) | **Appendix/phụ**, để thể hiện mức độ "thuần zero-shot" thật sự của MAD-X | Cho thấy chênh lệch bao nhiêu là do riêng quy ước chọn checkpoint |

Dùng (a) làm số chính thức trong Table 2, (b) là optional bonus cho Appendix L nếu muốn phân tích sâu hơn.

## 5. Báo cáo — thêm 2 dòng riêng biệt vào Table 2, không thay dòng cũ

```json
{
  "config_name": "madx_squad1.1_published",    // GIỮ NGUYÊN dòng hiện có — không xóa
  "note": "as-published adapter, protocol mismatch (no unanswerable handling)"
}
{
  "config_name": "madx_squad2.0_retrained",    // DÒNG MỚI từ spec này
  "seed": 42,
  "language": "vi",
  "checkpoint_selection": "target_informed",   // hoặc "english_only" cho bản phụ
  "xquad": {"em": 0.0, "f1": 0.0},
  "mlqa": {"em": 0.0, "f1": 0.0},
  "squad_en": {"pre_f1": null, "post_f1": 0.0, "delta_src": 0.0},
  "trainable_params_task_adapter_only": "~0.9M (reduction=16, 12 layers, XLM-R-base)"
}
```

Giữ **cả 2 dòng** trong bảng (không xóa dòng cũ) — việc để cạnh nhau tự nó chứng minh trực tiếp giả thuyết "điểm thấp là do thiếu unanswerable-handling", biến từ suy đoán thành bằng chứng thực nghiệm.

## 6. Δsrc của MAD-X — lưu ý khi diễn giải, không phải lỗi

Vì task adapter + English language adapter **không hề bị cập nhật** khi swap sang tiếng Việt (chỉ đổi language adapter), Δsrc của MAD-X sẽ ≈ 0 **gần như tuyệt đối, theo cấu trúc**, không phải nhờ học được như $\mathcal{L}_{domain}$ của bạn. Khi viết vào bài, tránh diễn giải "MAD-X cũng preserve source tốt như Ours" theo kiểu so ngang hàng — nên ghi rõ đây là 2 cơ chế khác bản chất (modularity cứng vs. regularization mềm), có đánh đổi khác nhau (MAD-X cần adapter ngôn ngữ pretrain sẵn cho từng target language, không tận dụng được unlabeled target text như Stage 2 của bạn).

## 7. Caveat bắt buộc ghi vào README.md / Appendix L

> "Chúng tôi retrain task adapter của MAD-X (Pfeiffer et al., 2020) trên SQuAD 2.0 — cùng dữ liệu và cùng quy ước xử lý câu hỏi không trả lời được với Stage 1 teacher của framework chính — nhằm loại bỏ nhiễu do lệch protocol so với con số published (SQuAD 1.1, không unanswerable) đã báo cáo trước đó. Chúng tôi giữ nguyên kiến trúc MAD-X gốc (language + invertible + task adapter, backbone frozen), không thêm domain-consistency anchor hay margin regularization của framework chính. Lưu ý: theo thiết kế modular của MAD-X, task adapter tiếng Anh không hề tiếp xúc với văn bản tiếng Việt dưới bất kỳ hình thức nào (kể cả unlabeled) — nghiêm ngặt hơn protocol *target-label-free* của chúng tôi (vốn có thấy văn bản tiếng Việt unlabeled qua parallel loader) — trong khi cơ chế bảo vệ nguồn ngữ của MAD-X là modularity cứng (không cập nhật trọng số) chứ không phải regularization học được."

## 8. Checklist thực hiện

1. [ ] Verify `AdapterHub/xlm-roberta-base-en-wiki_pfeiffer` load được (không 404).
2. [ ] Verify convention null-answer của thư viện `adapters`/QA head khớp với `<s>` index 0 bằng 1 batch test nhỏ.
3. [ ] Train task adapter trên SQuAD 2.0 (lr=1e-4, theo đúng Pfeiffer et al. 2020 Section 5.2), chọn checkpoint theo English dev set trước.
4. [ ] Swap sang `vi_lang`, chạy `shared/eval_harness.py` trên XQuAD-vi/MLQA-vi.
5. [ ] Áp dụng rule chọn checkpoint (a) target-informed cho số chính thức Table 2; (b) English-only cho Appendix nếu muốn.
6. [ ] Ghi số trainable params riêng của task adapter (~0.9M) — không gộp chung dòng "2.65M" của LoRA để tránh gây hiểu lầm cùng mức tham số.
7. [ ] Thêm 2 dòng (published vs retrained) song song vào Table 2, giữ nguyên dòng cũ.
8. [ ] Viết README.md với đúng đoạn caveat ở mục 7.