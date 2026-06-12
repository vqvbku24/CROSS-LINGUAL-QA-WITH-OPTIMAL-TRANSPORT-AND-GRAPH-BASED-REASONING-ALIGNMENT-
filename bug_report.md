markdown_content = """# Spec Hướng Dẫn AI Agent: Sửa Đổi Kiến Trúc Sang LoRA & Triển Khai Ablation Study (Stage 2)

## 1. Bối Cảnh & Vấn Đề Hiện Tại (Root Cause Analysis)
Trong pha huấn luyện Stage 2 trước đó (Full-Parameter Fine-tuning), hệ thống gặp hiện tượng **đóng băng metrics (VI EM đứng im ở 48.89%, EN EM đứng im ở 75.00%)**. 
- **Nguyên nhân cốt lõi:** XLM-RoBERTa sử dụng một **Shared Backbone** duy nhất cho cả hai ngôn ngữ. Lệnh gọi `requires_grad_(False)` nhằm mục đích đóng băng nhánh English (Teacher) đã vô tình đóng băng toàn bộ Backbone của cả nhánh Tiếng Việt (Student).
- **Hệ quả:** Mô hình không thể học được bất kỳ sự dóng hàng (alignment) nào từ $L_{\text{ot}}$ và $L_{\text{cons}}$. Trọng số duy nhất thay đổi là QA Head bị bào mòn bởi `weight_decay`, làm giảm hiệu năng nghiêm trọng.

## 2. Giải Pháp Kiến Trúc: Tích Hợp LoRA (Low-Rank Adaptation)
Để cô lập nhánh học của Student mà không làm ảnh hưởng đến mỏ neo Teacher (bản chất nằm chung một mạng), Agent phải bọc (wrap) Backbone của XLM-R bằng thư viện `peft` (LoRA).
- Toàn bộ tham số gốc của XLM-R sẽ được đóng băng cố định.
- Chỉ các ma trận LoRA bổ sung (gắn vào các tầng Attention và FFN) mới nhận gradient để học cấu trúc Tiếng Việt.
- Sử dụng cơ chế `disable_adapter()` để tạm thời tắt LoRA khi chạy nhánh English, mô phỏng hoàn hảo trạng thái Frozen Teacher nguyên bản.

---

## 3. Danh Sách Các File Cần Sửa Đổi (Step-by-Step Code Changes)

### Tác vụ 1: Sửa đổi lõi mô hình để nhúng LoRA
**File cần chỉnh sửa:** `phase2_model/model_core.py`

Agent cần import và cấu hình `peft` trong hàm khởi tạo của class `CrossLingualOTModel`.

```

```text
SUCCESS

```python
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModel
import torch.nn as nn

# Trong __init__ của CrossLingualOTModel:
# 1. Khởi tạo backbone gốc
base_backbone = AutoModel.from_pretrained(model_name)
self.hidden_size = base_backbone.config.hidden_size

# 2. Cấu hình LoraConfig nâng cao
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["query", "key", "value", "dense"]  # Áp dụng lên Attention và FFN layers
)

# 3. Wrap mô hình gốc thành PEFT model
self.backbone = get_peft_model(base_backbone, lora_config)
self.backbone.print_trainable_parameters()  # In để verify số lượng tham số trainable

# 4. Các Parameter bổ sung như layer_weights và qa_head phải đảm bảo requires_grad=True
self.layer_weights = nn.Parameter(torch.ones(4))

```

---

### Tác vụ 2: Thay đổi luồng huấn luyện hai bước (Two-Pass Forward)

**File cần chỉnh sửa:** `train_stage2.py`

1. **XÓA BỎ HOÀN TOÀN** hàm phá hoại kiến trúc: `freeze_en_backbone(model)`. Không được set `requires_grad_(False)` thủ công trên toàn backbone nữa.
2. **Sửa đổi hàm `stage2_step**` để tận dụng cơ chế tắt/bật adapter của LoRA:

```python
def stage2_step(
    batch: dict,
    model,
    criterion,
    stage2_loss,
    epsilon: float,
    n_iters: int,
    global_step: int,
    device: torch.device,
) -> dict:
    from phase3_loss.losses import (
        sinkhorn_masked, compute_span_loss, compute_cons_loss, gamma_entropy,
        _extract_question_embeddings,
    )

    # ── STEP 2.1: Nhánh Teacher (Tiếng Anh) — NO GRADIENT & TẮT LORA ──
    with torch.no_grad():
        # Phép thuật LoRA: Tạm thời vô hiệu hóa adapter để lấy embedding gốc làm mỏ neo chuẩn
        with model.backbone.disable_adapter():
            en_out = model(batch, branch="en")
            h_en    = en_out["hidden"]       # (B, T_en, H) từ Frozen Base Model
            en_mask = ~en_out["en_pad_mask"]

            # Dự đoán phân phối xác suất ranh giới (Soft Pseudo-labels) từ Teacher
            en_q_emb, en_q_mask = _extract_question_embeddings(h_en, batch["en_question_end"])
            en_start_logits, en_end_logits, _ = criterion.qa_head(h_en, en_q_emb, en_q_mask)
            p_en_start = F.softmax(en_start_logits, dim=-1)
            p_en_end   = F.softmax(en_end_logits, dim=-1)

    # ── STEP 2.2: Nhánh Student (Tiếng Việt) — WITH GRADIENT & BẬT LORA ──
    # Không dùng disable_adapter -> Mạng tự động kích hoạt ma trận LoRA để tính gradient
    vi_out = model(batch, branch="vi")
    h_vi    = vi_out["hidden"]           # (B, T_vi, H) có chứa luồng LoRA weights
    vi_mask = ~vi_out["vi_pad_mask"]

    vi_q_emb, vi_q_mask = _extract_question_embeddings(h_vi, batch["vi_question_end"])
    vi_start_logits, vi_end_logits, _ = criterion.qa_head(h_vi, vi_q_emb, vi_q_mask)

    # ── STEP 2.3: Tính toán hệ thống Loss hình học ──
    gamma_list, L_ot = sinkhorn_masked(h_en, h_vi, en_mask, vi_mask, epsilon=epsilon, n_iters=n_iters)
    L_span = compute_span_loss(gamma_list, p_en_start, p_en_end, vi_start_logits, vi_end_logits, en_mask, vi_mask)
    L_cons = compute_cons_loss(gamma_list, h_en, h_vi, en_mask, vi_mask)

    # Tổng hợp loss kết hợp Curriculum Delay cho Cons và Span
    losses = stage2_loss(L_ot, L_span, L_cons, global_step)

    with torch.no_grad():
        losses["gamma_entropy"] = gamma_entropy(gamma_list)

    return losses

```

3. **Cập nhật Optimizer trong `run_stage2`:** Đảm bảo chỉ truyền các tham số LoRA, `layer_weights` và tham số của `criterion` (QA Head) vào để tối ưu.

```python
# Trong hàm run_stage2() của train_stage2.py
trainable_params = [
    {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": config["stage2_head_lr"]}, # LoRA chỉ có params này requires_grad=True
    {"params": [model.layer_weights], "lr": config["stage2_head_lr"]},
    {"params": list(criterion.parameters()), "lr": config["stage2_head_lr"]},
]
optimizer = AdamW(trainable_params, weight_decay=config["weight_decay"])

```

---

### Tác vụ 3: Đồng bộ hóa module Loss

**File cần kiểm tra/chỉnh sửa:** `phase3_loss/losses.py`

Agent phải đảm bảo file `losses.py` đã export đầy đủ cấu trúc hàm rời (modularized functions) bao gồm:

* `sinkhorn_masked(h_en, h_vi, en_mask, vi_mask, epsilon, n_iters)`
* `compute_span_loss(gamma_list, p_en_start, p_en_end, vi_start_logits, vi_end_logits, en_mask, vi_mask)`
* `compute_cons_loss(gamma_list, h_en, h_vi, en_mask, vi_mask)`
* `gamma_entropy(gamma_list)`
* `Stage2Loss`

*Lưu ý:* Việc trích xuất $\gamma$ từ `sinkhorn_masked` phải tích hợp cơ chế Phạt nặng (`masked_fill` bằng hằng số `1e4` hoặc `-1e8` trong không gian Log) đối với các vị trí Padding của cả EN và VI để triệt tiêu hiện tượng rò rỉ phân phối xác suất sang các token rác.

---

## 4. Ma Trận Đánh Giá Thử Nghiệm (Ablation Study Matrix Specification)

Sau khi hoàn thành sửa đổi code, Agent hoặc Developer sẽ thực thi ma trận huấn luyện 4 pha nghiêm ngặt sau để thu thập số liệu phục vụ bài báo (Paper):

| Cấu hình (Setting) | Hàm Loss Kích Hoạt | CLI Command Thực Thi | Kỳ vọng Kết quả (XQuAD VI EM) |
| --- | --- | --- | --- |
| **Stage 1 (Baseline)** | Supervised EN $L_{\text{qa}} + L_{\text{has\_ans}}$ | `python phase4-evaluation/quick_eval.py` (Chạy zero-shot trên checkpoint Stage 1) | **56.67%** (Mỏ neo cố định) |
| **OT only** | $L_{\text{ot}}$ (Feature Alignment) | `python train_stage2.py --lambda_span 0.0 --lambda_cons 0.0` | Sẽ lớn hơn hoặc dao động quanh mức Baseline |
| **OT + Cons** | $L_{\text{ot}} + \lambda_{\text{cons}} L_{\text{cons}}$ | `python train_stage2.py --lambda_span 0.0 --lambda_cons 0.5` | Đánh giá năng lực kiểm soát không gian Feature |
| **OT + Cons + Span** | $L_{\text{ot}} + \lambda_{\text{cons}} L_{\text{cons}} + \lambda_{\text{span}} L_{\text{span}}$ | `python train_stage2.py --lambda_span 1.0 --lambda_cons 0.5` | **Full Framework** — Đỉnh cao tối ưu của mô hình |

## 5. Tiêu Chí Nghiệm Thu Code (Definition of Done)

1. **Không còn lỗi ImportError:** `train_stage2.py` chạy forward pass mượt mà, phân tách thành công hai tháp qua context manager `disable_adapter()`.
2. **Backbone thực sự chuyển động:** Metrics trên TensorBoard (`Loss/Span`, `Loss/OT`, `Loss/Cons`) phải biến thiên liên tục theo từng step, không được đóng băng hay đứng im một giá trị cố định.
3. **Safety Check hoạt động:** Nếu hiệu năng tiếng Anh (`Eval/SQuAD_EN_EM_Quick`) sụt giảm vượt quá ngưỡng 20% so với baseline Stage 1, hệ thống kích hoạt cơ chế `hard stop` ngay lập tức để bảo vệ tri thức.
"""

with open("agent_code_repair_spec.md", "w", encoding="utf-8") as f:
f.write(markdown_content)

print("SUCCESS")

"""