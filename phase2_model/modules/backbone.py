# modules/backbone.py
"""
Shared XLM-RoBERTa backbone for EN and VI.

Simplified: no attention maps (subsampling removed).
Returns only last_hidden_state: (B, L, H).
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class SharedBackbone(nn.Module):
    """XLM-RoBERTa shared encoder cho cả EN và VI."""

    def __init__(self, model_name: str = "xlm-roberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            output_attentions=False,      # ← no attention maps needed
            output_hidden_states=True,     # FIXED: must be True for layer mixing
        )
        self.hidden_size = self.encoder.config.hidden_size  # 768 (base) / 1024 (large)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids     : (B, L)
            attention_mask: (B, L)

        Returns:
            last_hidden_state: (B, L, H) — H = 768 for base, 1024 for large
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state