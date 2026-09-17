# model_core.py
"""
CrossLingualOTModel — Simplified architecture for Sinkhorn OT alignment.

Pipeline:
    1. XLM-R backbone → Mix intermediate layers (6,7,8,9) → H_en, H_vi: (B, T, d=768)
    2. Dynamic Truncation: cut to effective sequence length (max non-PAD tokens in batch)
    3. Cosine distance cost matrix C: (B, T_en, T_vi)
    4. PAD masking on C (set PAD rows/cols to 1e4)

All graph, GAT, subsampling, and FGW components have been removed.
The Sinkhorn OT solver lives in losses.py (computed during loss forward).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType


def get_default_target_layers(config) -> list[int]:
    """
    Tự động chọn target layers ở khoảng độ sâu 50% - 75% của backbone.
    - XLM-R (12 layers): [6, 7, 8, 9]
    - mmBERT (22 layers): [11, 12, 14, 16]
    """
    num_layers = getattr(config, "num_hidden_layers", 12)
    if num_layers == 22:
        return [11, 12, 14, 16]
    elif num_layers == 12:
        return [6, 7, 8, 9]
    else:
        ratios = [0.50, 0.583, 0.667, 0.75]
        return [min(int(num_layers * r), num_layers - 1) for r in ratios]


class CrossLingualOTModel(nn.Module):
    """
    Minimal model: shared backbone → weighted intermediate hidden states + cost matrix.
    Learns to mix intermediate layers via trainable weights (layer_weights: nn.Parameter).

    Dynamic truncation reduces Sinkhorn cost matrix from (B,512,512) to (B,T_en,T_vi)
    where T_en/T_vi = max valid tokens in the current batch — saving O(L²) memory+compute.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        compute_cost_matrix: bool = True,
        target_layers: list[int] | None = None,
    ):
        super().__init__()
        self.compute_cost_matrix = compute_cost_matrix
        # Bắt buộc bật output_hidden_states=True để lấy được các layer ở giữa
        self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.hidden_size = self.backbone.config.hidden_size  # 768 (base) / 1024 (large)
        
        # ── Configurable target layers (default: auto-detected from model depth) ──
        if target_layers is not None:
            self.target_layers = list(target_layers)
        else:
            self.target_layers = get_default_target_layers(self.backbone.config)

        # ── Trainable layer-mixing weights ──
        # Initialized to ones → after softmax → equal weight across target layers.
        self.layer_weights = nn.Parameter(torch.ones(len(self.target_layers)), requires_grad=True)

    def apply_lora(self, target_modules: list[str] | None = None):
        """
        Bọc LoRA vào backbone. 
        CHỈ GỌI hàm này SAU KHI đã load checkpoint của Stage 1 để đảm bảo keys map 1-1.
        Có fail-fast check để đảm bảo LoRA được inject đúng module và có trainable parameters.
        """
        model_type = getattr(self.backbone.config, "model_type", "").lower()
        if target_modules is None:
            if "modernbert" in model_type:
                target_modules = ["Wqkv", "Wo"]
            else:
                target_modules = ["query", "key", "value", "dense"]

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules
        )
        
        try:
            self.backbone = get_peft_model(self.backbone, lora_config)
        except Exception as e:
            err_msg = str(e)
            if "torchao" in err_msg:
                raise RuntimeError(
                    f"LoRA injection failed due to torchao conflict: {err_msg}\n"
                    f"Fix on Colab: Run `!pip uninstall -y torchao` (hoặc `!pip install -U torchao`) và khởi động lại runtime."
                ) from e
            raise RuntimeError(f"Failed to apply LoRA to {model_type}: {err_msg}") from e

        # Fail-fast check: đảm bảo có ít nhất 1 tham số có thể huấn luyện
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        if trainable_params == 0:
            raise RuntimeError(
                f"[LoRA Fail-Fast] 0 trainable parameters found! Target modules {target_modules} "
                f"do not match any module in model_type='{model_type}'."
            )

        self.backbone.hidden_size = self.hidden_size
        if hasattr(self.backbone, "config"):
            self.backbone.config.output_hidden_states = True
        self.backbone.print_trainable_parameters()

    # ──────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────

    def forward(self, batch: dict, branch: str = "both") -> dict:
        """
        Args:
            batch keys:
                en_input_ids, en_attention_mask : (B, L)  — L up to 512
                vi_input_ids, vi_attention_mask : (B, L)
                en_start_position, en_end_position : (B,) — answer span (EN only)
                en_question_end, vi_question_end   : (B,) — index of first [SEP]
            branch: "both" (default, Stage 1 behavior), "en", or "vi"
                "en"  — only process en_input_ids/en_attention_mask
                        returns {"hidden": h_en, "en_pad_mask": ..., "en_seq_len": ...}
                "vi"  — only process vi_input_ids/vi_attention_mask
                        returns {"hidden": h_vi, "vi_pad_mask": ..., "vi_seq_len": ...}
                "both" — full Stage 1 behavior (unchanged)

        Returns:
            branch="both": dict with en_hidden, vi_hidden, cost_matrix, en_pad_mask, vi_pad_mask
            branch="en":   dict with hidden, en_pad_mask, en_seq_len
            branch="vi":   dict with hidden, vi_pad_mask, vi_seq_len
        """
        if branch == "en":
            out = self.backbone(
                input_ids=batch["en_input_ids"],
                attention_mask=batch["en_attention_mask"],
                output_hidden_states=True,
            )
            stacked = torch.stack([out.hidden_states[i] for i in self.target_layers], dim=0)
            weights = torch.softmax(self.layer_weights, dim=0).view(len(self.target_layers), 1, 1, 1)
            H = (stacked * weights).sum(dim=0)
            pad_mask = (batch["en_attention_mask"] == 0)
            return {"hidden": H, "en_pad_mask": pad_mask}

        if branch == "vi":
            out = self.backbone(
                input_ids=batch["vi_input_ids"],
                attention_mask=batch["vi_attention_mask"],
                output_hidden_states=True,
            )
            stacked = torch.stack([out.hidden_states[i] for i in self.target_layers], dim=0)
            weights = torch.softmax(self.layer_weights, dim=0).view(len(self.target_layers), 1, 1, 1)
            H = (stacked * weights).sum(dim=0)
            pad_mask = (batch["vi_attention_mask"] == 0)
            return {"hidden": H, "vi_pad_mask": pad_mask}

        # ── branch="both" — Stage 1 behavior (unchanged) ────────────────
        # ── 1. Shared Backbone ─────────────────────────────────────
        out_en = self.backbone(
            input_ids=batch["en_input_ids"],
            attention_mask=batch["en_attention_mask"],
            output_hidden_states=True,
        )
        out_vi = self.backbone(
            input_ids=batch["vi_input_ids"],
            attention_mask=batch["vi_attention_mask"],
            output_hidden_states=True,
        )

        # ── 2. Mix Intermediate Layers ─────────────────────────────
        stacked_en = torch.stack([out_en.hidden_states[i] for i in self.target_layers], dim=0)
        stacked_vi = torch.stack([out_vi.hidden_states[i] for i in self.target_layers], dim=0)

        # Chuyển weights thành xác suất (tổng = 1) và reshape để nhân với tensor 4D
        weights = torch.softmax(self.layer_weights, dim=0).view(len(self.target_layers), 1, 1, 1)

        # Tính Weighted Sum: (B, L, H)
        H_en = (stacked_en * weights).sum(dim=0)
        H_vi = (stacked_vi * weights).sum(dim=0)

        # ── 3. Dynamic Sequence Truncation ─────────────────────────
        # Tính max token thực tế trong batch (không PAD).
        # Cắt hidden states xuống T_en / T_vi — loại bỏ hoàn toàn các vị trí PAD thừa.
        # Điều này giảm cost matrix từ (B,512,512) → (B,T_en,T_vi):
        #   - Ít FLOPs hơn trong bmm (cosine distance)
        #   - Ít memory hơn trong Sinkhorn (50 iters × B × T_en × T_vi)
        en_seq_len = int(batch["en_attention_mask"].sum(dim=1).max().item())  # T_en
        vi_seq_len = int(batch["vi_attention_mask"].sum(dim=1).max().item())  # T_vi

        H_en = H_en[:, :en_seq_len, :]  # (B, T_en, H)
        H_vi = H_vi[:, :vi_seq_len, :]  # (B, T_vi, H)

        # ── 4. PAD Masking ─────────────────────────────────────────
        en_pad_mask = (batch["en_attention_mask"][:, :en_seq_len] == 0)  # (B, T_en)
        vi_pad_mask = (batch["vi_attention_mask"][:, :vi_seq_len] == 0)  # (B, T_vi)

        result = {
            "en_hidden":   H_en,          # (B, T_en, H)
            "vi_hidden":   H_vi,          # (B, T_vi, H)
            "en_pad_mask": en_pad_mask,   # (B, T_en)
            "vi_pad_mask": vi_pad_mask,   # (B, T_vi)
        }

        # ── 5. Cosine Distance Cost Matrix (Sinkhorn only) ────────
        if self.compute_cost_matrix:
            en_norm = F.normalize(H_en, p=2, dim=-1)   # (B, T_en, H)
            vi_norm = F.normalize(H_vi, p=2, dim=-1)   # (B, T_vi, H)
            C = 1.0 - torch.bmm(en_norm, vi_norm.transpose(1, 2))  # (B, T_en, T_vi)

            # Mask entire rows (EN PAD) and columns (VI PAD)
            C = C.masked_fill(en_pad_mask.unsqueeze(2), 1e4)   # PAD rows  → 1e4
            C = C.masked_fill(vi_pad_mask.unsqueeze(1), 1e4)   # PAD cols  → 1e4

            # NOTE: Do NOT mask shared BPE tokens (numbers, punctuation, "Paris").
            # Sinkhorn has doubly-stochastic marginal constraints — every token must
            # ship exactly 1/L mass. Blocking "Paris_EN"→"Paris_VI" (cost≈0) forces
            # that mass onto unrelated tokens, corrupting their embeddings via L_ot.
            # Shared tokens act as natural zero-cost anchors: they satisfy their
            # marginal cheaply with ∇≈0, freeing other tokens to find semantic matches.

            result["cost_matrix"] = C     # (B, T_en, T_vi)

        return result