# model_core.py
import torch
import torch.nn as nn
from .modules.backbone import SharedBackbone
from .modules.gat_encoder import GATEncoder
from .modules.subsampling import conditional_subsample
from .modules.fgw_solver import fgw_bapg, partial_fgw


class CrossLingualOTModel(nn.Module):
    def __init__(self,
                 model_name: str = "xlm-roberta-base",
                 K: int = 128,
                 gat_hidden: int = 512,
                 gat_out: int = 256,
                 gat_layers: int = 2,
                 fgw_alpha: float = 0.5,
                 fgw_epsilon: float = 0.01,
                 use_partial: bool = True,
                 partial_m: float = 0.85):
        super().__init__()
        self.K = K
        self.fgw_alpha = fgw_alpha
        self.fgw_epsilon = fgw_epsilon
        self.use_partial = use_partial
        self.partial_m = partial_m

        self.backbone = SharedBackbone(model_name)
        self.gat = GATEncoder(
            in_dim=self.backbone.hidden_size,
            hidden_dim=gat_hidden,
            out_dim=gat_out,
            num_layers=gat_layers
        )

    def forward(self, batch: dict) -> dict:
        """
        batch keys:
            en_input_ids, en_attention_mask: (B, L_en)
            vi_input_ids, vi_attention_mask: (B, L_vi)
            en_start_position, en_end_position: (B,)  — answer span (EN only)
            en_question_end: (B,)  — index của [SEP] đầu tiên, tức end of question
        """
        # ── 1. Shared Backbone ────────────────────────────────────────
        en_hidden, en_attn = self.backbone(batch["en_input_ids"], batch["en_attention_mask"])
        vi_hidden, vi_attn = self.backbone(batch["vi_input_ids"], batch["vi_attention_mask"])
        # en_hidden: (B, L, H) | en_attn: (B, L, L)

        batch_gamma    = []
        batch_en_emb   = []
        batch_vi_emb   = []
        batch_D_en     = []
        batch_D_vi     = []
        batch_M        = []
        batch_keep_en  = []   # ← keep_idx_en: token-space indices được giữ lại
        B = en_hidden.size(0)

        for i in range(B):
            # ── 2. Conditional Subsampling ────────────────────────────
            # EN side: dùng soft_boost=10.0 cho answer tokens.
            # Answer tokens KHÔNG bị hard-force vào graph, chỉ được boost
            # attention score → rất likely được chọn bởi top-K nhưng
            # KHÔNG guaranteed. Graph structure gần giống inference.
            q_end = batch["en_question_end"][i].item()
            en_q_idx = list(range(0, q_end + 1))   # [CLS] + question tokens
            en_a_idx = list(range(
                batch["en_start_position"][i].item(),
                batch["en_end_position"][i].item() + 1
            )) if batch["en_start_position"][i].item() > 0 else []

            vi_q_idx = list(range(0, batch["vi_question_end"][i].item() + 1))

            en_sub, en_keep = conditional_subsample(
                en_attn[i], en_q_idx, en_a_idx, K=self.K,
                soft_boost=10.0   # boost, KHÔNG force
            )
            vi_sub, vi_keep = conditional_subsample(
                vi_attn[i], vi_q_idx, [], K=self.K,
                soft_boost=0.0    # inference-like
            )

            # ── 3. GAT Encoder ────────────────────────────────────────
            en_feat = en_hidden[i, en_keep, :]  # (K, H)
            vi_feat = vi_hidden[i, vi_keep, :]  # (K, H)

            en_emb, D_en = self.gat(en_feat, en_sub)
            vi_emb, D_vi = self.gat(vi_feat, vi_sub)

            # ── 4. FGW Solver ─────────────────────────────────────────
            # Feature cost matrix M: cosine distance giữa EN và VI embeddings
            pos = torch.arange(self.K, device=en_emb.device, dtype=torch.float32) / self.K
            pos_cost = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()
            
            en_norm = torch.nn.functional.normalize(en_emb.detach(), dim=-1)
            vi_norm = torch.nn.functional.normalize(vi_emb, dim=-1)
            
            M = (1.0 - torch.mm(en_norm, vi_norm.T)) + 0.1 * pos_cost  # (K, K)

            batch_en_emb.append(en_emb)
            batch_vi_emb.append(vi_emb)
            # Fix Bug #3: Detach D_en/D_vi — Danskin's Theorem.
            # Gradient FGW chỉ chảy qua M (feature cost), KHÔNG qua
            # distance matrices (graph geometry). Nếu không detach,
            # GW term tạo O(K³) gradient chain → explosion (gn_bb 55K+).
            batch_D_en.append(D_en.detach())
            batch_D_vi.append(D_vi.detach())
            batch_M.append(M)
            batch_keep_en.append(en_keep)   # (K,) LongTensor — token indices EN

        # Parallelize FGW solving over the batch to save time
        from concurrent.futures import ThreadPoolExecutor
        
        def solve_fgw(i):
            if self.use_partial:
                return partial_fgw(batch_D_en[i], batch_D_vi[i], m=self.partial_m, nb_dummies=50)
            else:
                return fgw_bapg(batch_D_en[i], batch_D_vi[i], batch_M[i],
                                alpha=self.fgw_alpha, epsilon=self.fgw_epsilon)

        with ThreadPoolExecutor(max_workers=min(B, 8)) as executor:
            batch_gamma = list(executor.map(solve_fgw, range(B)))

        return {
            "gamma"       : torch.stack(batch_gamma),        # (B, K, K)
            "en_node_emb" : torch.stack(batch_en_emb),       # (B, K, out_dim)
            "vi_node_emb" : torch.stack(batch_vi_emb),       # (B, K, out_dim)
            "D_en"        : torch.stack(batch_D_en),         # (B, K, K)
            "D_vi"        : torch.stack(batch_D_vi),         # (B, K, K)
            "M"           : torch.stack(batch_M),            # (B, K, K)
            # keep_idx_en[b][j] = token index gốc (trong [0, L-1]) của node j trong EN graph.
            # losses.py dùng để map en_start/en_end từ token-space [0,511] → graph-space [0,K-1].
            "keep_idx_en" : torch.stack(batch_keep_en),      # (B, K) LongTensor
        }