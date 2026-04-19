# modules/gat_encoder.py
# NOTE: Dùng PyTorch Geometric (PyG). Nếu không có, Person A cần implement GATConv từ scratch
# pip install torch_geometric

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    # Fallback: Dense attention-based GAT (không cần PyG)

class DenseGATLayer(nn.Module):
    """Dense GAT layer — hoạt động với adjacency matrix thay vì edge_index.
    Dùng khi không có PyG hoặc khi graph nhỏ (K <= 256).
    """
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (N, in_dim)
            adj: (N, N) — attention/adjacency matrix (dùng làm mask)
        Returns:
            out: (N, out_dim)
        """
        N = x.size(0)
        Wh = self.W(x)  # (N, out_dim)
        Wh = Wh.view(N, self.num_heads, self.head_dim)  # (N, H, D)
        
        # Attention scores
        Wh_i = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # (N, N, H, D)
        Wh_j = Wh.unsqueeze(0).expand(N, -1, -1, -1)  # (N, N, H, D)
        e = self.leaky_relu(
            self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1)  # (N, N, H)
        )
        
        # Mask với adj: chỉ attend các node có edge
        mask = (adj > 0).unsqueeze(-1).expand_as(e)  # (N, N, H)
        e = e.masked_fill(~mask, float('-inf'))
        alpha = F.softmax(e, dim=1)  # (N, N, H)
        alpha = self.dropout(alpha)
        
        # Aggregate
        out = (alpha.unsqueeze(-1) * Wh_j).sum(dim=1)  # (N, H, D)
        return out.view(N, -1)  # (N, out_dim)


class GATEncoder(nn.Module):
    """2-layer GAT encoder. Input: node features + adj matrix. Output: embeddings + distance matrix."""
    
    def __init__(self, in_dim: int = 768, hidden_dim: int = 512, 
                 out_dim: int = 256, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            DenseGATLayer(dims[i], dims[i+1], num_heads=num_heads)
            for i in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i+1]) for i in range(num_layers)])
        self.act = nn.GELU()

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor):
        """
        Args:
            node_features: (K, H) — K nodes, H hidden dim
            adj_matrix:    (K, K) — subsampled attention matrix
        Returns:
            node_emb: (K, out_dim)
            D:        (K, K) — pairwise L2 distance matrix (dùng cho FGW)
        """
        x = node_features
        for layer, norm in zip(self.layers, self.norms):
            residual = x if x.shape == (node_features.shape[0], layer.W.out_features) else None
            x = norm(self.act(layer(x, adj_matrix)))
            if residual is not None:
                x = x + residual  # residual connection nếu dim khớp
        
        D = torch.cdist(x, x, p=2)  # (K, K)
        return x, D