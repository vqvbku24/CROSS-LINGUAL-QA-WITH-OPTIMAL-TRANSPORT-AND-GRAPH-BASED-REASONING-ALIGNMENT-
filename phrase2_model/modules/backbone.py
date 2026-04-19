# modules/backbone.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SharedBackbone(nn.Module):
    """XLM-RoBERTa shared encoder cho cả EN và VI."""
    
    def __init__(self, model_name: str = "xlm-roberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name, 
            output_attentions=True,
            output_hidden_states=False
        )
        self.hidden_size = self.encoder.config.hidden_size  # 768 (base) / 1024 (large)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Returns:
            last_hidden_state: (B, L, H)
            attn_map: (B, L, L) — mean of layers 7,8,9 across all heads
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Lấy layer 7,8,9 (0-indexed) — giữ thông tin ngữ nghĩa cao
        # out.attentions: tuple of (B, num_heads, L, L) per layer
        attn_layers = torch.stack(out.attentions[7:10])  # (3, B, H, L, L)
        attn_map = attn_layers.mean(dim=0).mean(dim=1)   # (B, L, L)
        
        return out.last_hidden_state, attn_map