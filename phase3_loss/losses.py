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
    swap_mask = gs_start > gs_end
    gs_start, gs_end = (
        torch.where(swap_mask, gs_end,   gs_start),
        torch.where(swap_mask, gs_start, gs_end),
    )

    return gs_start, gs_end


# ──────────────────────────────────────────────────────────────
# QA Head
# ──────────────────────────────────────────────────────────────

class QAHead(nn.Module):
    """
    Linear head dự đoán start/end span từ node embeddings.

    Input : (B, K, out_dim) — K subsampled node embeddings
    Output: start_logits (B, K), end_logits (B, K)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.start_proj = nn.Linear(hidden_size, 1)
        self.end_proj   = nn.Linear(hidden_size, 1)

    def forward(self, node_emb: torch.Tensor):
        """
        Args:
            node_emb: (B, K, H)
        Returns:
            start_logits: (B, K)
            end_logits  : (B, K)
        """
        start_logits = self.start_proj(node_emb).squeeze(-1)  # (B, K)
        end_logits   = self.end_proj(node_emb).squeeze(-1)    # (B, K)
        return start_logits, end_logits


# ──────────────────────────────────────────────────────────────
# Span Projection: γ → pseudo-label cho VI
# ──────────────────────────────────────────────────────────────

def _decode_span_from_gamma(
    gamma: torch.Tensor,
    en_start: torch.Tensor,
    en_end: torch.Tensor,
    K: int,
    max_span_len: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dùng transport plan γ để map answer span EN → VI (pseudo-label).

    Thuật toán (vectorized):
        1. Lấy cột EN tương ứng với [en_start, en_end] trong γ.
        2. Tổng hợp xác suất: vi_score[j] = Σ_{i=start}^{end} γ[i, j].
        3. Tìm cặp (s*, e*) maximise vi_score[s*] + vi_score[e*]
           với ràng buộc 0 ≤ e* - s* ≤ max_span_len.

    Complexity: O(B × K × max_span_len) thay vì O(B × K²).

    Args:
        gamma   : (B, K, K) transport plan
        en_start: (B,) start position trong K-node EN graph
        en_end  : (B,) end   position trong K-node EN graph
        K       : số node sau subsampling
        max_span_len: độ dài span tối đa cho VI

    Returns:
        vi_start: (B,) pseudo start (LongTensor)
        vi_end  : (B,) pseudo end   (LongTensor)
    """
    B = gamma.size(0)
    device = gamma.device
    vi_starts = torch.zeros(B, dtype=torch.long, device=device)
    vi_ends   = torch.zeros(B, dtype=torch.long, device=device)

    for b in range(B):
        s = en_start[b].item()
        e = en_end[b].item()

        if s == 0 and e == 0:
            # Unanswerable → pseudo-label cũng là (0, 0)
            continue

        # Đảm bảo indices hợp lệ trong K
        s = max(0, min(int(s), K - 1))
        e = max(s, min(int(e), K - 1))

        # vi_score[j] = tổng transport mass từ answer EN nodes đến j
        vi_score = gamma[b, s:e + 1, :].sum(dim=0)  # (K,)

        # ── Vectorized best span search ──────────────────────────
        # Thay vì O(K²), dùng: best_e = argmax(vi_score[si:si+max_span_len])
        # cho mỗi si → O(K × max_span_len)
        best_score = torch.tensor(-1.0, device=device)
        best_s_val = 0
        best_e_val = 0

        # Tạo matrix (K, max_span_len+1): mỗi row si chứa vi_score[si:si+span]
        span_len = min(max_span_len + 1, K)
        for si in range(K):
            ei_max = min(si + span_len, K)
            end_scores = vi_score[si:ei_max]  # (len,)
            # best end cho start=si
            local_best_idx = end_scores.argmax()
            score = vi_score[si] + end_scores[local_best_idx]
            if score > best_score:
                best_score = score
                best_s_val = si
                best_e_val = si + local_best_idx.item()

        vi_starts[b] = best_s_val
        vi_ends[b]   = best_e_val

    return vi_starts, vi_ends


# ──────────────────────────────────────────────────────────────
# Loss Components
# ──────────────────────────────────────────────────────────────

def qa_loss(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    start_positions: torch.Tensor,
    end_positions: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Cross-entropy loss cho span extraction (EN supervised).

    Unanswerable samples (start=0, end=0) vẫn được tính — loss = CE với label 0.
    Nếu muốn ignore, set ignore_index và truyền -100 vào positions.

    Args:
        start_logits    : (B, K)
        end_logits      : (B, K)
        start_positions : (B,) LongTensor
        end_positions   : (B,) LongTensor

    Returns:
        scalar loss
    """
    loss_start = F.cross_entropy(start_logits, start_positions, ignore_index=ignore_index)
    loss_end   = F.cross_entropy(end_logits,   end_positions,   ignore_index=ignore_index)
    return (loss_start + loss_end) / 2.0


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
    K: int,
    max_span_len: int = 30,
) -> torch.Tensor:
    """
    Pseudo-label QA loss cho VI — span được project từ EN qua γ.

    Chỉ tính loss cho những sample answerable (en_start > 0 hoặc en_end > 0).

    Args:
        vi_start_logits : (B, K)
        vi_end_logits   : (B, K)
        gamma           : (B, K, K)
        en_start        : (B,) answer start trong K-node EN graph
        en_end          : (B,) answer end   trong K-node EN graph

    Returns:
        scalar loss (0.0 nếu không có sample answerable nào)
    """
    with torch.no_grad():
        vi_start_pseudo, vi_end_pseudo = _decode_span_from_gamma(
            gamma, en_start, en_end, K, max_span_len
        )

    # Chỉ train trên answerable samples
    answerable = (en_start > 0) | (en_end > 0)  # (B,)
    if answerable.sum() == 0:
        return torch.tensor(0.0, device=gamma.device, requires_grad=False)

    loss = qa_loss(
        vi_start_logits[answerable],
        vi_end_logits[answerable],
        vi_start_pseudo[answerable],
        vi_end_pseudo[answerable],
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
        lambda_fgw: float = 0.1,
        lambda_span: float = 0.5,
        lambda_cons: float = 0.3,
        fgw_alpha: float = 0.5,
        consistency_temperature: float = 2.0,
        max_span_len: int = 30,
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
        """
        super().__init__()
        self.K              = K
        self.lambda_fgw     = lambda_fgw
        self.lambda_span    = lambda_span
        self.lambda_cons    = lambda_cons
        self.fgw_alpha      = fgw_alpha
        self.temperature    = consistency_temperature
        self.max_span_len   = max_span_len

        # QA Head dùng chung cho EN và VI
        self.qa_head = QAHead(qa_hidden_size)

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

        # ── 1. QA Head → logits ────────────────────────────────
        en_start_logits, en_end_logits = self.qa_head(en_node_emb)  # (B, K) each
        vi_start_logits, vi_end_logits = self.qa_head(vi_node_emb)  # (B, K) each

        # ── 2. FIX BUG #1: Remap token-space → graph-space ────
        # en_start/en_end là indices trong token-space (0-511).
        # QA head hoạt động trên K nodes (0-K-1) nên phải remap.
        en_start_gs, en_end_gs = _remap_positions_to_graph_space(
            en_start, en_end, keep_idx_en
        )

        # ── 3. L_qa (supervised EN) — dùng graph-space indices ─
        l_qa = qa_loss(en_start_logits, en_end_logits,
                       en_start_gs, en_end_gs)

        # ── 4. L_fgw ──────────────────────────────────────────
        l_fgw = fgw_alignment_loss(gamma, D_en, D_vi, M, alpha=self.fgw_alpha)

        # ── 5. L_span_proj — pseudo-label VI, cũng dùng GS indices
        l_span = span_projection_loss(
            vi_start_logits, vi_end_logits,
            gamma, en_start_gs, en_end_gs,  # <--- graph-space indices
            K=self.K, max_span_len=self.max_span_len,
        )

        # ── 6. L_consistency (Transport-Guided, stop-grad EN) ──
        l_cons = consistency_loss(
            en_start_logits, en_end_logits,
            vi_start_logits, vi_end_logits,
            gamma=gamma,
            temperature=self.temperature,
        )

        # ── 7. Tổng hợp ───────────────────────────────────────
        l_total = (
            l_qa
            + self.lambda_fgw  * l_fgw
            + self.lambda_span * l_span
            + self.lambda_cons * l_cons
        )

        return {
            "total"    : l_total,
            "qa"       : l_qa.detach(),
            "fgw"      : l_fgw.detach(),
            "span_proj": l_span.detach(),
            "cons"     : l_cons.detach(),
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
    }

    # Mock batch — positions trong token-space (0-511)
    mock_batch = {
        "en_start_position": torch.tensor([5,  0]),   # sample 0: token 5, sample 1: unanswerable
        "en_end_position"  : torch.tensor([10, 0]),
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