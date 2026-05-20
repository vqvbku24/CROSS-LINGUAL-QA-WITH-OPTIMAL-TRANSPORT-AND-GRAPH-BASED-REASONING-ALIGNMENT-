# losses.py
"""
Hàm mục tiêu (Loss Functions) cho Cross-Lingual QA with OT & Graph Alignment.

Tổng loss:
    L_total = L_qa + λ_fgw * L_fgw + λ_span * L_span_proj + λ_cons * L_consistency

Chi tiết:
    L_qa          : Cross-entropy span extraction trên EN (supervised).
    L_fgw         : FGW transport cost — cưỡng bức align cấu trúc graph EN ↔ VI.
    L_span_proj   : Pseudo-label QA loss trên VI dùng span projection từ γ.
    L_consistency : KL-divergence giữa logits EN và logits VI — ép hai nhánh nhất quán.

Notes:
    - L_consistency dùng .detach() trên EN logits (stop-gradient) để Teacher
      không bị nhiễu từ VI side (theo ý tưởng Phase 3 trong idea.docx).
    - L_span_proj dùng hard-span pseudo-label decode từ γ (argmax + span constraint).
    - QA Head (start/end) được chia sẻ và apply cho cả EN lẫn VI node embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Helper: Token-space → Graph-space position mapping (Fix Bug #1)
# ──────────────────────────────────────────────────────────────

def _remap_positions_to_graph_space(
    en_start: torch.Tensor,    # (B,) token indices (0-511)
    en_end: torch.Tensor,      # (B,) token indices (0-511)
    keep_idx_en: torch.Tensor, # (B, K) mapping graph node → token index
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Chuyển đổi label từ vị trí token gốc (0-511) sang vị trí node
    trong graph EN sau subsampling (0-K-1).

    Với mỗi sample b:
        - keep_idx_en[b, k] = token index của node thứ k trong graph.
        - Tìm k sao cho keep_idx_en[b, k] == en_start[b] → gs_start[b] = k.
        - Nếu không tìm thấy (token bị loại khi subsampling) → dùng
          nearest-neighbor: chọn node có token index gần nhất với answer token.
          Cách này tốt hơn fallback về 0 vì (0,0) sẽ corrupt toàn bộ label
          khiến loss kẹt ở log(K) ≈ 5.07.

    Args:
        en_start    : (B,) start positions trong token-space
        en_end      : (B,) end   positions trong token-space
        keep_idx_en : (B, K) bảng tra token index → graph node index

    Returns:
        gs_start    : (B,) start positions trong graph-space
        gs_end      : (B,) end   positions trong graph-space
    """
    B, K = keep_idx_en.shape
    device = en_start.device

    # Vectorised nearest-neighbour lookup
    # keep_idx_en: (B, K) — float cast để dùng abs diff
    keep_f = keep_idx_en.float()  # (B, K)

    # start
    s_diff  = (keep_f - en_start.float().unsqueeze(1)).abs()   # (B, K)
    gs_start = s_diff.argmin(dim=1)                             # (B,)

    # end
    e_diff  = (keep_f - en_end.float().unsqueeze(1)).abs()     # (B, K)
    gs_end   = e_diff.argmin(dim=1)                             # (B,)

    # Unanswerable (s=0, e=0) → giữ nguyên (0, 0)
    unanswerable = (en_start == 0) & (en_end == 0)
    gs_start = gs_start.masked_fill(unanswerable, 0)
    gs_end   = gs_end.masked_fill(unanswerable, 0)

    # Đảm bảo gs_start <= gs_end (tránh span bị đảo ngược)
    # BỎ PHẦN NÀY VÌ GRAPH NODES KHÔNG SORT THEO TOKEN INDEX
    # swap_mask = gs_start > gs_end
    # gs_start, gs_end = (
    #     torch.where(swap_mask, gs_end,   gs_start),
    #     torch.where(swap_mask, gs_start, gs_end),
    # )

    return gs_start, gs_end


# ──────────────────────────────────────────────────────────────
# QA Head
# ──────────────────────────────────────────────────────────────

class QAHead(nn.Module):
    """
    Linear head dự đoán start/end span từ node embeddings.
    Tích hợp Cross-Attention layer để nhận question-aware features.

    Thêm has_answer_head: binary classifier để tách unanswerable ra khỏi
    span logits, tránh model collapse (luôn predict CLS là start).
    """

    def __init__(self, hidden_size: int, q_hidden_size: int = 768):
        super().__init__()
        # Cross-Attention layer: Context nodes (Query) attend to Question tokens (Key/Value)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
        self.q_proj = nn.Linear(q_hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.start_proj = nn.Linear(hidden_size, 1)
        self.end_proj   = nn.Linear(hidden_size, 1)

        # has_answer classifier: dùng CLS node embedding (node 0) để predict
        # answerable/unanswerable — tách hoàn toàn khỏi span logits.
        # Input: CLS node embedding (B, H) → Output: (B, 1) logit → sigmoid
        self.has_answer_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, node_emb: torch.Tensor, q_emb: torch.Tensor, q_mask: Optional[torch.Tensor] = None):
        """
        Args:
            node_emb: (B, K, H) - Context nodes (node 0 = CLS)
            q_emb: (B, L_q, H_q) - Question tokens (KHÔNG detach — cần gradient)
            q_mask: (B, L_q) - Padding mask (True = ignore)

        Returns:
            start_logits    : (B, K)
            end_logits      : (B, K)
            has_answer_logit: (B,) — logit trước sigmoid (>0 = answerable)
        """
        q_proj = self.q_proj(q_emb)  # (B, L_q, H)

        # Cross-attention: Context nodes attend to Question tokens
        attn_out, _ = self.cross_attn(
            query=node_emb,
            key=q_proj,
            value=q_proj,
            key_padding_mask=q_mask
        )

        # Residual connection + LayerNorm
        node_emb_out = self.layer_norm(node_emb + attn_out)

        start_logits = self.start_proj(node_emb_out).squeeze(-1)  # (B, K)
        end_logits   = self.end_proj(node_emb_out).squeeze(-1)    # (B, K)

        # has_answer: dùng CLS node (node 0) sau cross-attention
        cls_emb = node_emb_out[:, 0, :]               # (B, H)
        has_answer_logit = self.has_answer_head(cls_emb).squeeze(-1)  # (B,)

        return start_logits, end_logits, has_answer_logit


# ──────────────────────────────────────────────────────────────
# Span Projection: γ → pseudo-label cho VI
# ──────────────────────────────────────────────────────────────

def _decode_vi_span_from_gamma(
    gamma: torch.Tensor,
    en_start: torch.Tensor,
    en_end: torch.Tensor,
    keep_idx_vi: torch.Tensor,
    K: int,
    max_span_len: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dùng transport plan γ để tìm pseudo-label (start, end) trên VI graph.
    Ràng buộc: vi_e >= vi_s và vi_e - vi_s <= max_span_len trong TOKEN SPACE.
    """
    B = gamma.size(0)
    device = gamma.device

    # 1. Tính tổng mass nhận được từ start và end node của EN
    batch_idx = torch.arange(B, device=device)
    start_mass = gamma[batch_idx, en_start, :]  # (B, K)
    end_mass   = gamma[batch_idx, en_end, :]    # (B, K)

    # 2. Tạo ma trận điểm cho mọi cặp (u, v) trong VI graph
    # score_matrix[b, u, v] = start_mass[b, u] + end_mass[b, v]
    score_matrix = start_mass.unsqueeze(2) + end_mass.unsqueeze(1)  # (B, K, K)

    # 3. Ràng buộc Token Space (vi_s <= vi_e và độ dài <= max_span_len)
    tok = keep_idx_vi  # (B, K)
    tok_u = tok.unsqueeze(2)  # (B, K, 1) - start node
    tok_v = tok.unsqueeze(1)  # (B, 1, K) - end node
    tok_dist = tok_v - tok_u  # (B, K, K) - khoảng cách trong token space

    # Mask invalid spans
    invalid_mask = (tok_dist < 0) | (tok_dist > max_span_len)
    score_matrix.masked_fill_(invalid_mask, float('-inf'))

    # 4. Tìm argmax trên 2D (KxK) cho từng sample trong batch
    best_idx = score_matrix.view(B, -1).argmax(dim=1)  # (B,)
    vi_s = best_idx // K
    vi_e = best_idx % K

    return vi_s, vi_e


# ──────────────────────────────────────────────────────────────
# Loss Components
# ──────────────────────────────────────────────────────────────

def qa_loss(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    start_positions: torch.Tensor,
    end_positions: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cross-entropy loss cho span extraction (EN supervised).

    Returns:
        (total, l_start, l_end) — cả 3 để TensorBoard log riêng start/end.
    """
    loss_start = F.cross_entropy(start_logits, start_positions, ignore_index=ignore_index)
    loss_end   = F.cross_entropy(end_logits,   end_positions,   ignore_index=ignore_index)
    return (loss_start + loss_end) / 2.0, loss_start, loss_end


def fgw_alignment_loss(
    gamma: torch.Tensor,
    D_en: torch.Tensor,
    D_vi: torch.Tensor,
    M: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    FGW transport cost — dùng như một regularizer buộc EN graph
    và VI graph phải có cấu trúc tương đồng.

    L_fgw = alpha * GW_cost + (1 - alpha) * W_cost

    Gradient flow (Danskin's Theorem):
        g  (transport plan) : DETACH — coi như hằng số tối ưu, không backprop
                              qua Sinkhorn iterations (tránh chain-rule explosion).
        C1, C2 (geometry)   : DETACH — đã detach từ _patch_model_outputs.
        M  (EN↔VI cost)     : GIỮ GRADIENT — toàn bộ signal FGW dồn vào đây,
                              kéo embedding EN và VI lại gần nhau một cách sạch sẽ.

    Args:
        gamma: (B, K, K) transport plan (từ model_core)
        D_en : (B, K, K) distance matrix EN  [đã detach]
        D_vi : (B, K, K) distance matrix VI  [đã detach]
        M    : (B, K, K) feature cost matrix (cosine dist EN↔VI)  [có grad]
        alpha: weight GW vs Wasserstein

    Returns:
        scalar loss (mean over batch)
    """
    B = gamma.size(0)
    losses = []

    for b in range(B):
        # ── Danskin's Theorem ────────────────────────────────────────────
        # g là nghiệm tối ưu của bài toán OT (Sinkhorn). Theo định lý Danskin,
        # đạo hàm của min_g F(g, θ) theo θ = ∂F/∂θ|_{g=g*}, không cần
        # backprop qua quá trình tìm g*. Detach ngay tại đây.
        g  = gamma[b].detach()  # (K, K) — HẰNG SỐ, không có grad_fn
        # ────────────────────────────────────────────────────────────────
        C1 = D_en[b]    # (K, K) — đã detach từ _patch_model_outputs
        C2 = D_vi[b]    # (K, K) — đã detach từ _patch_model_outputs
        m  = M[b]       # (K, K) — GIỮ GRADIENT (EN↔VI cosine dist)

        # Wasserstein term: <M, g>  — gradient chảy qua m (= M[b])
        w_loss = (m * g).sum()

        # GW term (efficient formulation):
        # C1, C2, g đều là hằng số → gw1, gw2, gw3 không có grad_fn
        # Ngoại trừ w_loss, toàn bộ GW term = hằng số (offset không ảnh hưởng grad)
        p = g.sum(dim=1)  # (K,) marginal EN
        q = g.sum(dim=0)  # (K,) marginal VI

        gw1 = (C1 ** 2 * p.unsqueeze(1) * p.unsqueeze(0)).sum()
        gw2 = (C2 ** 2 * q.unsqueeze(1) * q.unsqueeze(0)).sum()
        gw3 = (C1 @ g @ C2.T * g).sum()
        gw_loss = gw1 + gw2 - 2.0 * gw3

        losses.append(alpha * gw_loss + (1.0 - alpha) * w_loss)

    return torch.stack(losses).mean()


def span_projection_loss(
    vi_start_logits: torch.Tensor,
    vi_end_logits: torch.Tensor,
    gamma: torch.Tensor,
    en_start: torch.Tensor,
    en_end: torch.Tensor,
    keep_idx_vi: torch.Tensor,
    K: int,
    max_span_len: int = 30,
) -> torch.Tensor:
    """
    Pseudo-label QA loss cho VI — span được project từ EN qua γ.

    Chỉ tính loss cho những sample answerable (đã được lọc ở ngoài).
    Sử dụng constraint từ token space thông qua keep_idx_vi.
    """
    with torch.no_grad():
        vi_start_pseudo, vi_end_pseudo = _decode_vi_span_from_gamma(
            gamma, en_start, en_end, keep_idx_vi, K, max_span_len
        )

    loss, _, _ = qa_loss(
        vi_start_logits,
        vi_end_logits,
        vi_start_pseudo,
        vi_end_pseudo,
    )
    return loss


def consistency_loss(
    en_start_logits: torch.Tensor,
    en_end_logits: torch.Tensor,
    vi_start_logits: torch.Tensor,
    vi_end_logits: torch.Tensor,
    gamma: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Transport-Guided Consistency Loss.

    Thay vì KL(VI || EN) trực tiếp — vốn bị structural mismatch do EN và VI
    hoạt động trên 2 graph-space khác nhau (EN có answer-aware subsampling,
    VI không) — ta dùng transport plan γ làm bridge:

        L_cons = T² · KL( softmax(VI/T) || transport(softmax(EN/T), γ) )

    Cụ thể:
        1. Tính EN probability: p_en = softmax(en_logits.detach() / T)
        2. "Transport" p_en sang VI space: p_target = normalize(γᵀ · p_en)
           γᵀ[j, i] = transport mass từ EN node i → VI node j
           → p_target[j] = tổng xác suất EN được transport đến VI node j
        3. KL(VI || p_target) — VI học từ transported EN distribution

    Tại sao hiệu quả:
        - γ đã encode thông tin alignment cấu trúc EN↔VI (từ FGW solver)
        - Target p_target nằm đúng trong VI graph-space → không còn mismatch
        - γ càng tốt → target càng chính xác → cons loss giảm đều
        - Tạo "neo" tự nhiên: loss bị bound bởi chất lượng γ

    QUAN TRỌNG: EN logits vẫn được detach() (stop-gradient Teacher).
    Scale T² giữ nguyên theo Hinton Knowledge Distillation convention.

    Args:
        en_start_logits : (B, K) EN start logits
        en_end_logits   : (B, K) EN end logits
        vi_start_logits : (B, K) VI start logits
        vi_end_logits   : (B, K) VI end logits
        gamma           : (B, K_en, K_vi) transport plan từ FGW solver
        temperature     : nhiệt độ softmax (> 1 để smooth distribution)

    Returns:
        scalar loss
    """
    # ── 1. EN probability distribution (stop-gradient Teacher) ──────────
    en_start_prob = F.softmax(en_start_logits.detach() / temperature, dim=-1)  # (B, K)
    en_end_prob   = F.softmax(en_end_logits.detach()   / temperature, dim=-1)  # (B, K)

    # ── 2. Transport EN distribution → VI space qua γ ──────────────────
    # γ: (B, K_en, K_vi) → γᵀ: (B, K_vi, K_en)
    # γᵀ · p_en → "expected VI probability" dựa trên transport plan
    gamma_T = gamma.detach().transpose(1, 2)  # (B, K_vi, K_en)

    # Normalize γᵀ theo hàng: mỗi VI node nhận tổng mass = 1
    # Tránh division by zero cho VI nodes không nhận mass nào
    gamma_T_norm = gamma_T / (gamma_T.sum(dim=-1, keepdim=True) + 1e-8)

    # Transport: p_target[b, j] = Σ_i γᵀ_norm[b, j, i] · p_en[b, i]
    vi_target_start = torch.bmm(
        gamma_T_norm, en_start_prob.unsqueeze(-1)
    ).squeeze(-1)  # (B, K)
    vi_target_end = torch.bmm(
        gamma_T_norm, en_end_prob.unsqueeze(-1)
    ).squeeze(-1)  # (B, K)

    # Clamp + renormalize để đảm bảo valid probability distribution
    vi_target_start = vi_target_start.clamp(min=1e-8)
    vi_target_end   = vi_target_end.clamp(min=1e-8)
    vi_target_start = vi_target_start / vi_target_start.sum(dim=-1, keepdim=True)
    vi_target_end   = vi_target_end   / vi_target_end.sum(dim=-1, keepdim=True)

    # ── 3. KL(VI || transported_EN) ────────────────────────────────────
    vi_start_log = F.log_softmax(vi_start_logits / temperature, dim=-1)
    vi_end_log   = F.log_softmax(vi_end_logits   / temperature, dim=-1)

    kl_start = F.kl_div(vi_start_log, vi_target_start, reduction="batchmean")
    kl_end   = F.kl_div(vi_end_log,   vi_target_end,   reduction="batchmean")

    # Scale theo T² (theo Knowledge Distillation convention của Hinton)
    return (temperature ** 2) * (kl_start + kl_end) / 2.0


# ──────────────────────────────────────────────────────────────
# Tổng hợp Loss
# ──────────────────────────────────────────────────────────────

class OTAlignmentLoss(nn.Module):
    """
    Tổng hợp tất cả loss components cho Phase 3.

    L_total = L_qa
            + λ_fgw  * L_fgw
            + λ_span * L_span_proj
            + λ_cons * L_consistency

    Cũng expose QAHead để model_core không phải tự tạo.
    """

    def __init__(
        self,
        qa_hidden_size: int = 256,   # = gat_out trong model_core
        K: int = 128,
        lambda_fgw: float = 0.01,
        lambda_span: float = 0.3,
        lambda_cons: float = 0.15,
        fgw_alpha: float = 0.5,
        consistency_temperature: float = 4.0,
        max_span_len: int = 30,
        q_hidden_size: int = 768,
    ):
        """
        Args:
            qa_hidden_size : chiều output của GATEncoder (= gat_out)
            K              : số node sau subsampling (phải khớp với model_core.K)
            lambda_fgw     : trọng số L_fgw
            lambda_span    : trọng số L_span_proj
            lambda_cons    : trọng số L_consistency
            fgw_alpha      : alpha trong FGW (GW vs Wasserstein balance)
            consistency_temperature : nhiệt độ cho KL div
            max_span_len   : max span length khi decode pseudo-label
            q_hidden_size  : hidden size của backbone (thường là 768 cho base, 1024 cho large)
        """
        super().__init__()
        self.K              = K
        self.lambda_fgw     = lambda_fgw
        self.lambda_span    = lambda_span
        self.lambda_cons    = lambda_cons
        self.fgw_alpha      = fgw_alpha
        self.temperature    = consistency_temperature
        self.max_span_len   = max_span_len

        # QA Head dùng chung cho EN và VI, tích hợp Cross-Attention
        self.qa_head = QAHead(qa_hidden_size, q_hidden_size=q_hidden_size)

    def forward(
        self,
        model_outputs: dict,
        batch: dict,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            model_outputs: dict từ CrossLingualOTModel.forward()
                {
                    "gamma"       : (B, K, K),
                    "en_node_emb" : (B, K, out_dim),
                    "vi_node_emb" : (B, K, out_dim),
                    "D_en"        : (B, K, K),
                    "D_vi"        : (B, K, K),
                    "M"           : (B, K, K),
                    "keep_idx_en" : (B, K),  ← token index của từng node EN (Fix Bug #1)
                }
            batch: dict từ DataLoader
                {
                    "en_start_position": (B,) — token-space (0-511)
                    "en_end_position"  : (B,) — token-space (0-511)
                    ...
                }

        Returns:
            dict chứa:
                "total"    : L_total (scalar, có grad_fn để backward)
                "qa"       : L_qa
                "fgw"      : L_fgw
                "span_proj": L_span_proj
                "cons"     : L_consistency
        """
        gamma        = model_outputs["gamma"]        # (B, K, K)
        en_node_emb  = model_outputs["en_node_emb"]  # (B, K, H)
        vi_node_emb  = model_outputs["vi_node_emb"]  # (B, K, H)
        D_en         = model_outputs["D_en"]         # (B, K, K)
        D_vi         = model_outputs["D_vi"]         # (B, K, K)
        M            = model_outputs["M"]            # (B, K, K)
        keep_idx_en  = model_outputs["keep_idx_en"]  # (B, K) — Fix Bug #1

        en_start = batch["en_start_position"]  # (B,) token-space
        en_end   = batch["en_end_position"]    # (B,) token-space
        
        # ── 1. Trích xuất Question embeddings và mask ──────────
        en_hidden = model_outputs.get("en_hidden")  # (B, L_en, H_q)
        vi_hidden = model_outputs.get("vi_hidden")  # (B, L_vi, H_q)
        device = en_node_emb.device

        # Tạo mask cho EN question (KHÔNG detach — cross-attention cần gradient)
        en_q_ends = batch["en_question_end"]
        max_en_q = en_q_ends.max().item() + 1
        en_q_emb = en_hidden[:, :max_en_q, :]          # ← bỏ .detach()
        en_q_mask = torch.arange(max_en_q, device=device).unsqueeze(0) > en_q_ends.unsqueeze(1)

        # Tạo mask cho VI question (KHÔNG detach)
        vi_q_ends = batch["vi_question_end"]
        max_vi_q = vi_q_ends.max().item() + 1
        vi_q_emb = vi_hidden[:, :max_vi_q, :]          # ← bỏ .detach()
        vi_q_mask = torch.arange(max_vi_q, device=device).unsqueeze(0) > vi_q_ends.unsqueeze(1)

        # ── 2. QA Head → logits + has_answer ──────────────────
        en_start_logits, en_end_logits, en_has_ans = self.qa_head(en_node_emb, en_q_emb, en_q_mask)
        vi_start_logits, vi_end_logits, _           = self.qa_head(vi_node_emb, vi_q_emb, vi_q_mask)

        # ── 3. Remap token-space → graph-space ────────────────
        en_start_gs, en_end_gs = _remap_positions_to_graph_space(
            en_start, en_end, keep_idx_en
        )

        # ── 4. L_qa (supervised EN) — dùng graph-space indices ─
        # CẬP NHẬT: Chỉ tính Span Loss cho những mẫu CÓ CÂU TRẢ LỜI.
        # Điều này giúp model không bị ép phải dự đoán CLS (0, 0) trong lúc tìm Span,
        # tránh hiện tượng Model Collapse (luôn dự đoán chuỗi rỗng).
        answerable_mask = batch["en_is_answerable"].bool().to(device)
        
        if answerable_mask.any():
            l_qa, l_qa_start, l_qa_end = qa_loss(
                en_start_logits[answerable_mask], en_end_logits[answerable_mask],
                en_start_gs[answerable_mask], en_end_gs[answerable_mask]
            )
        else:
            # Fallback nếu batch toàn bộ là unanswerable
            l_qa = torch.tensor(0.0, device=device)
            l_qa_start = torch.tensor(0.0, device=device)
            l_qa_end = torch.tensor(0.0, device=device)

        # ── 5. L_has_answer (BCE) ─────────────────────────────
        # Label: 1 nếu answerable (en_is_answerable = 1), 0 nếu không.
        # Tách unanswerable detection ra khỏi span logits.
        has_answer_label = batch["en_is_answerable"].float().to(device)
        
        # CẬP NHẬT: Cho Unanswerable trọng số nhỏ hơn (ví dụ 0.5)
        # Giúp model phạt nặng hơn khi sai ở Answerable, phạt nhẹ hơn khi sai ở Unanswerable.
        # Điều này sẽ ép model bớt thiên vị (bias) về phía Unanswerable.
        # Bạn có thể tune con số 0.5 này (ví dụ 0.3 hoặc 0.1 nếu vẫn còn bias).
        unanswer_weight = 0.3
        sample_weights = torch.where(
            has_answer_label == 1.0, 
            torch.tensor(1.0, device=device), 
            torch.tensor(unanswer_weight, device=device)
        )
        
        l_has_ans = F.binary_cross_entropy_with_logits(
            en_has_ans, has_answer_label, weight=sample_weights
        )

        # ── 6. L_fgw ──────────────────────────────────────────
        l_fgw = fgw_alignment_loss(gamma, D_en, D_vi, M, alpha=self.fgw_alpha)

        # ── 7. L_span_proj — pseudo-label VI ──────────────────
        # CHỈ tính trên answerable samples
        keep_idx_vi = model_outputs.get("keep_idx_vi")
        if answerable_mask.any() and keep_idx_vi is not None:
            l_span = span_projection_loss(
                vi_start_logits[answerable_mask], vi_end_logits[answerable_mask],
                gamma[answerable_mask], en_start_gs[answerable_mask], en_end_gs[answerable_mask],
                keep_idx_vi[answerable_mask], K=self.K, max_span_len=self.max_span_len,
            )
        else:
            l_span = torch.tensor(0.0, device=device)

        # ── 8. L_consistency (Transport-Guided, stop-grad EN) ──
        # Tương tự như L_qa, L_consistency cũng nên được filter theo answerable_mask
        # vì EN không còn học span cho unanswerable nữa, logits lúc này sẽ mang tính random.
        if answerable_mask.any():
            l_cons = consistency_loss(
                en_start_logits[answerable_mask], en_end_logits[answerable_mask],
                vi_start_logits[answerable_mask], vi_end_logits[answerable_mask],
                gamma=gamma[answerable_mask],
                temperature=self.temperature,
            )
        else:
            l_cons = torch.tensor(0.0, device=device)

        # ── 9. Tổng hợp ───────────────────────────────────────
        l_total = (
            l_qa
            + 0.5      * l_has_ans   # trọng số cố định 0.5 (không cần tune)
            + self.lambda_fgw  * l_fgw
            + self.lambda_span * l_span
            + self.lambda_cons * l_cons
        )

        # ── 10. Debug stats — phát hiện sớm CLS collapse ─────────
        # Nếu cls_start_logit_mean ≈ max_start_logit_mean → CLS đang dominate
        # → cần tăng lambda_has_ans hoặc kiểm tra gradient flow.
        # Tất cả đều .detach() — không có chi phí gradient.
        with torch.no_grad():
            cls_start  = en_start_logits[:, 0].mean()            # scalar
            max_start  = en_start_logits.max(dim=1).values.mean()  # scalar
            cls_end    = en_end_logits[:, 0].mean()
            max_end    = en_end_logits.max(dim=1).values.mean()
            # collapse_ratio gần 1.0 → CLS đang thắng hầu hết các samples
            answerable_mask = has_answer_label.bool()
            if answerable_mask.any():
                has_ans_acc = ((en_has_ans[answerable_mask] > 0).float().mean())
            else:
                has_ans_acc = torch.tensor(float('nan'))

        return {
            "total"           : l_total,
            "qa"              : l_qa.detach(),
            "qa_start"        : l_qa_start.detach(),
            "qa_end"          : l_qa_end.detach(),
            "has_ans"         : l_has_ans.detach(),
            "fgw"             : l_fgw.detach(),
            "span_proj"       : l_span.detach(),
            "cons"            : l_cons.detach(),
            # Debug: collapse detector
            "dbg/cls_start"   : cls_start,
            "dbg/max_start"   : max_start,
            "dbg/cls_end"     : cls_end,
            "dbg/max_end"     : max_end,
            "dbg/has_ans_acc" : has_ans_acc,  # accuracy trên answerable samples
        }


# ──────────────────────────────────────────────────────────────
# Quick test: python losses.py
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    B, K, H = 2, 32, 64
    MAX_TOKENS = 512  # token-space size

    # Mock keep_idx_en: mỗi node map tới một token index ngẫu nhiên (không trùng)
    # Đảm bảo token 5 và 10 (answer tokens) nằm trong graph để test remap
    base_idx = torch.stack([
        torch.randperm(MAX_TOKENS)[:K],  # sample 0: chứa token 5, 10
        torch.randperm(MAX_TOKENS)[:K],  # sample 1: unanswerable, không cần
    ])  # (B, K)
    # Ép token 5 → node 3, token 10 → node 7 trong sample 0 (để test)
    base_idx[0, 3] = 5
    base_idx[0, 7] = 10

    # Mock model outputs
    mock_outputs = {
        "gamma"       : torch.rand(B, K, K).softmax(dim=-1),
        "en_node_emb" : torch.randn(B, K, H),
        "vi_node_emb" : torch.randn(B, K, H),
        "D_en"        : torch.rand(B, K, K),
        "D_vi"        : torch.rand(B, K, K),
        "M"           : torch.rand(B, K, K),
        "keep_idx_en" : base_idx,         # (B, K) — Fix Bug #1
        "keep_idx_vi" : base_idx.clone(), # (B, K)
        "en_hidden"   : torch.randn(B, MAX_TOKENS, 768),
        "vi_hidden"   : torch.randn(B, MAX_TOKENS, 768),
    }

    # Mock batch — positions trong token-space (0-511)
    mock_batch = {
        "en_start_position": torch.tensor([5,  0]),   # sample 0: token 5, sample 1: unanswerable
        "en_end_position"  : torch.tensor([10, 0]),
        "en_question_end"  : torch.tensor([12, 10]),
        "vi_question_end"  : torch.tensor([14, 11]),
        "en_is_answerable" : torch.tensor([1,  0]),
    }

    criterion = OTAlignmentLoss(qa_hidden_size=H, K=K)
    losses = criterion(mock_outputs, mock_batch)

    print("=== Loss Components ===")
    for k, v in losses.items():
        print(f"  {k:12s}: {v.item():.6f}")

    # Backward pass
    losses["total"].backward()
    print("\n[OK] Backward pass OK -- gradient flow worked!")
    print("  - L_qa, L_fgw, L_span_proj, L_consistency calculated.")
    print("  - stop-gradient EN logits trong L_consistency: OK")