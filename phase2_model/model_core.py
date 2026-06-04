# model_core.py
"""
CrossLingualOTModel — Simplified architecture for Sinkhorn OT alignment.

Pipeline:
    1. XLM-R backbone → H_en, H_vi: (B, L=512, d=768)
    2. Cosine distance cost matrix C: (B, L, L)
    3. PAD masking on C (set PAD rows/cols to 1e4)

All graph, GAT, subsampling, and FGW components have been removed.
The Sinkhorn OT solver lives in losses.py (computed during loss forward).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.backbone import SharedBackbone


class CrossLingualOTModel(nn.Module):
    """
    Minimal model: shared XLM-R backbone → hidden states + cost matrix.

    No GAT, no subsampling, no FGW.
    Sinkhorn OT is computed inside OTAlignmentLoss.forward() to keep
    the model clean and the transport plan close to the loss computation.
    """

    def __init__(self, model_name: str = "xlm-roberta-base"):
        super().__init__()
        self.backbone = SharedBackbone(model_name)

    # ──────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch keys:
                en_input_ids, en_attention_mask : (B, L)  — L=512
                vi_input_ids, vi_attention_mask : (B, L)
                en_start_position, en_end_position : (B,) — answer span (EN only)
                en_question_end, vi_question_end   : (B,) — index of first [SEP]

        Returns:
            dict with:
                en_hidden    : (B, L, H)   — XLM-R last hidden states for EN
                vi_hidden    : (B, L, H)   — XLM-R last hidden states for VI
                cost_matrix  : (B, L, L)   — cosine distance with PAD masking
                en_pad_mask  : (B, L)      — True where EN token is PAD
                vi_pad_mask  : (B, L)      — True where VI token is PAD
        """
        # ── 1. Shared Backbone ─────────────────────────────────────
        # H_en, H_vi: (B, L, H) where H = 768 for xlm-roberta-base
        H_en = self.backbone(batch["en_input_ids"], batch["en_attention_mask"])
        H_vi = self.backbone(batch["vi_input_ids"], batch["vi_attention_mask"])

        # ── 2. Cosine Distance Cost Matrix ─────────────────────────
        # C[b,i,j] = 1 - cosine_sim(H_en[b,i], H_vi[b,j])
        # Range: [0, 2] for normalized vectors
        en_norm = F.normalize(H_en, p=2, dim=-1)   # (B, L, H), ||v||=1
        vi_norm = F.normalize(H_vi, p=2, dim=-1)   # (B, L, H), ||v||=1
        C = 1.0 - torch.bmm(en_norm, vi_norm.transpose(1, 2))  # (B, L, L)

        # ── 3. PAD Masking ─────────────────────────────────────────
        # Set cost = 1e4 for any pair involving a PAD token.
        # This ensures Sinkhorn assigns ~0 mass to PAD positions.
        en_pad_mask = (batch["en_attention_mask"] == 0)  # (B, L) True = PAD
        vi_pad_mask = (batch["vi_attention_mask"] == 0)  # (B, L) True = PAD

        # Mask entire rows (EN PAD) and columns (VI PAD)
        C = C.masked_fill(en_pad_mask.unsqueeze(2), 1e4)   # PAD rows  → 1e4
        C = C.masked_fill(vi_pad_mask.unsqueeze(1), 1e4)   # PAD cols  → 1e4

        return {
            "en_hidden":   H_en,          # (B, L, H)
            "vi_hidden":   H_vi,          # (B, L, H)
            "cost_matrix": C,             # (B, L, L)
            "en_pad_mask": en_pad_mask,   # (B, L)
            "vi_pad_mask": vi_pad_mask,   # (B, L)
        }