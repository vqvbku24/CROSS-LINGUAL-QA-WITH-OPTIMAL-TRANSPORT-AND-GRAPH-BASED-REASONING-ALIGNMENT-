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


class CrossLingualOTModel(nn.Module):
    """
    Minimal model: shared XLM-R backbone → weighted intermediate hidden states + cost matrix.
    Learns to mix layers 6, 7, 8, 9 via trainable weights (layer_weights: nn.Parameter).

    Dynamic truncation reduces Sinkhorn cost matrix from (B,512,512) to (B,T_en,T_vi)
    where T_en/T_vi = max valid tokens in the current batch — saving O(L²) memory+compute.
    """

    def __init__(self, model_name: str = "xlm-roberta-base"):
        super().__init__()
        # Bắt buộc bật output_hidden_states=True để lấy được các layer ở giữa
        self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.hidden_size = self.backbone.config.hidden_size  # 768 (base) / 1024 (large)
        self.backbone.hidden_size = self.hidden_size         # alias for train.py compatibility
        
        # ── Trainable layer-mixing weights for layers 6, 7, 8, 9 ──
        # Initialized to ones → after softmax → equal weight (0.25 each).
        # Optimizer learns which layer combination is best for the QA+OT task.
        self.layer_weights = nn.Parameter(torch.ones(4))

    # ──────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch keys:
                en_input_ids, en_attention_mask : (B, L)  — L up to 512
                vi_input_ids, vi_attention_mask : (B, L)
                en_start_position, en_end_position : (B,) — answer span (EN only)
                en_question_end, vi_question_end   : (B,) — index of first [SEP]

        Returns:
            dict with:
                en_hidden    : (B, T_en, H)    — XLM-R weighted hidden states for EN
                vi_hidden    : (B, T_vi, H)    — XLM-R weighted hidden states for VI
                cost_matrix  : (B, T_en, T_vi) — cosine distance with PAD masking
                en_pad_mask  : (B, T_en)       — True where EN token is PAD
                vi_pad_mask  : (B, T_vi)       — True where VI token is PAD
                en_seq_len   : int             — effective EN length T_en (≤ 512)
                vi_seq_len   : int             — effective VI length T_vi (≤ 512)

        Dynamic Truncation:
            T_en = max non-PAD token count across batch for EN (= attention_mask.sum.max)
            T_vi = same for VI
            Reduces Sinkhorn cost from (B,512,512) → (B,T_en,T_vi). For typical QA
            sequences (~200-350 tokens) this saves ~50-75% of OT compute.
        """
        # ── 1. Shared Backbone ─────────────────────────────────────
        out_en = self.backbone(batch["en_input_ids"], batch["en_attention_mask"])
        out_vi = self.backbone(batch["vi_input_ids"], batch["vi_attention_mask"])

        # ── 2. Mix Intermediate Layers (6, 7, 8, 9) ────────────────
        target_layers = [6, 7, 8, 9]
        
        # Stack 4 layer mục tiêu: (4, B, L, H)
        stacked_en = torch.stack([out_en.hidden_states[i] for i in target_layers], dim=0)
        stacked_vi = torch.stack([out_vi.hidden_states[i] for i in target_layers], dim=0)

        # Chuyển weights thành xác suất (tổng = 1) và reshape để nhân với tensor 4D
        # layer_weights được học bởi optimizer → model tự tìm tổ hợp tốt nhất cho task
        weights = torch.softmax(self.layer_weights, dim=0).view(4, 1, 1, 1)

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

        # ── 4. Cosine Distance Cost Matrix ─────────────────────────
        # C[b,i,j] = 1 - cosine_sim(H_en[b,i], H_vi[b,j])
        en_norm = F.normalize(H_en, p=2, dim=-1)   # (B, T_en, H)
        vi_norm = F.normalize(H_vi, p=2, dim=-1)   # (B, T_vi, H)
        C = 1.0 - torch.bmm(en_norm, vi_norm.transpose(1, 2))  # (B, T_en, T_vi)

        # ── 5. PAD Masking ─────────────────────────────────────────
        # Với truncated sequence, PAD chỉ còn ở cuối 1 số sample (batch không đồng đều)
        en_pad_mask = (batch["en_attention_mask"][:, :en_seq_len] == 0)  # (B, T_en)
        vi_pad_mask = (batch["vi_attention_mask"][:, :vi_seq_len] == 0)  # (B, T_vi)

        # Mask entire rows (EN PAD) and columns (VI PAD)
        C = C.masked_fill(en_pad_mask.unsqueeze(2), 1e4)   # PAD rows  → 1e4
        C = C.masked_fill(vi_pad_mask.unsqueeze(1), 1e4)   # PAD cols  → 1e4

        # NOTE: Do NOT mask shared BPE tokens (numbers, punctuation, "Paris").
        # Sinkhorn has doubly-stochastic marginal constraints — every token must
        # ship exactly 1/L mass. Blocking "Paris_EN"→"Paris_VI" (cost≈0) forces
        # that mass onto unrelated tokens, corrupting their embeddings via L_ot.
        # Shared tokens act as natural zero-cost anchors: they satisfy their
        # marginal cheaply with ∇≈0, freeing other tokens to find semantic matches.

        return {
            "en_hidden":   H_en,          # (B, L, H)
            "vi_hidden":   H_vi,          # (B, L, H)
            "cost_matrix": C,             # (B, L, L)
            "en_pad_mask": en_pad_mask,   # (B, L)
            "vi_pad_mask": vi_pad_mask,   # (B, L)
        }