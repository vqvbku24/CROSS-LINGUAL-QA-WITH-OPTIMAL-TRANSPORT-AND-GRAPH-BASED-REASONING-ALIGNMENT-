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

    Thuật toán:
        1. Lấy cột EN tương ứng với [en_start, en_end] trong γ.
        2. Tổng hợp xác suất: vi_score[j] = Σ_{i=start}^{end} γ[i, j].
        3. Tìm cặp (s*, e*) maximise vi_score[s*] + vi_score[e*]
           với ràng buộc 0 ≤ e* - s* ≤ max_span_len.

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
    vi_starts, vi_ends = [], []

    for b in range(B):
        s = en_start[b].item()
        e = en_end[b].item()

        if s == 0 and e == 0:
            # Unanswerable → pseudo-label cũng là (0, 0)
            vi_starts.append(0)
            vi_ends.append(0)
            continue

        # Đảm bảo indices hợp lệ trong K
        s = max(0, min(int(s), K - 1))
        e = max(s, min(int(e), K - 1))

        # vi_score[j] = tổng transport mass từ answer EN nodes đến j
        span_rows = gamma[b, s:e + 1, :]  # (span_len, K)
        vi_score  = span_rows.sum(dim=0)  # (K,)

        # Tìm (start*, end*) tối ưu theo vi_score với ràng buộc span length
        best_score = -1.0
        best_s, best_e = 0, 0
        for si in range(K):
            for ei in range(si, min(si + max_span_len + 1, K)):
                score = vi_score[si].item() + vi_score[ei].item()
                if score > best_score:
                    best_score = score
                    best_s, best_e = si, ei

        vi_starts.append(best_s)
        vi_ends.append(best_e)

    device = gamma.device
    return (
        torch.tensor(vi_starts, dtype=torch.long, device=device),
        torch.tensor(vi_ends,   dtype=torch.long, device=device),
    )


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

    Args:
        gamma: (B, K, K) transport plan (từ model_core)
        D_en : (B, K, K) distance matrix EN
        D_vi : (B, K, K) distance matrix VI
        M    : (B, K, K) feature cost matrix (cosine dist EN↔VI)
        alpha: weight GW vs Wasserstein

    Returns:
        scalar loss (mean over batch)
    """
    B = gamma.size(0)
    losses = []

    for b in range(B):
        g  = gamma[b]   # (K, K)
        C1 = D_en[b]    # (K, K)
        C2 = D_vi[b]    # (K, K)
        m  = M[b]       # (K, K)

        # Wasserstein term: <M, gamma>
        w_loss = (m * g).sum()

        # GW term (efficient formulation):
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
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    KL-divergence consistency regularizer giữa EN và VI predictions.

    QUAN TRỌNG: EN logits được detach() (stop-gradient) để Teacher không bị
    nhiễu từ VI side — chỉ VI nhánh được update từ loss này.

    L_cons = KL( softmax(VI/T) || softmax(EN/T).detach() )
           = 0.5 * KL_start + 0.5 * KL_end

    Args:
        temperature: nhiệt độ softmax (> 1 để smooth distribution)

    Returns:
        scalar loss
    """
    # Stop-gradient trên EN side (Teacher)
    en_start_soft = F.log_softmax(en_start_logits.detach() / temperature, dim=-1)
    en_end_soft   = F.log_softmax(en_end_logits.detach()   / temperature, dim=-1)

    vi_start_soft = F.log_softmax(vi_start_logits / temperature, dim=-1)
    vi_end_soft   = F.log_softmax(vi_end_logits   / temperature, dim=-1)

    # KL(VI || EN) — VI học từ EN distribution
    kl_start = F.kl_div(vi_start_soft, en_start_soft.exp(), reduction="batchmean")
    kl_end   = F.kl_div(vi_end_soft,   en_end_soft.exp(),   reduction="batchmean")

    # Scale theo T^2 (theo Knowledge Distillation convention của Hinton)
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
        K: int = 160,
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
                    "D_en"        : (B, K, K),   ← cần thêm vào model_core
                    "D_vi"        : (B, K, K),   ← cần thêm vào model_core
                    "M"           : (B, K, K),   ← cần thêm vào model_core
                }
            batch: dict từ DataLoader
                {
                    "en_start_position": (B,),
                    "en_end_position"  : (B,),
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
        gamma       = model_outputs["gamma"]        # (B, K, K)
        en_node_emb = model_outputs["en_node_emb"]  # (B, K, H)
        vi_node_emb = model_outputs["vi_node_emb"]  # (B, K, H)
        D_en        = model_outputs["D_en"]         # (B, K, K)
        D_vi        = model_outputs["D_vi"]         # (B, K, K)
        M           = model_outputs["M"]            # (B, K, K)

        en_start = batch["en_start_position"]  # (B,)
        en_end   = batch["en_end_position"]    # (B,)

        # ── 1. QA Head → logits ────────────────────────────────
        en_start_logits, en_end_logits = self.qa_head(en_node_emb)  # (B, K) each
        vi_start_logits, vi_end_logits = self.qa_head(vi_node_emb)  # (B, K) each

        # ── 2. L_qa (supervised EN) ────────────────────────────
        # Map en_start/en_end từ token-space sang K-node space
        # (trong subsampling, keep_idx giữ thứ tự, nên dùng trực tiếp;
        #  nếu cần map chính xác, truyền keep_idx_en từ model_core)
        # Clamp để tránh index-out-of-range
        en_start_clamped = en_start.clamp(0, self.K - 1)
        en_end_clamped   = en_end.clamp(0, self.K - 1)

        l_qa = qa_loss(en_start_logits, en_end_logits,
                       en_start_clamped, en_end_clamped)

        # ── 3. L_fgw ──────────────────────────────────────────
        l_fgw = fgw_alignment_loss(gamma, D_en, D_vi, M, alpha=self.fgw_alpha)

        # ── 4. L_span_proj (pseudo-label VI) ──────────────────
        l_span = span_projection_loss(
            vi_start_logits, vi_end_logits,
            gamma, en_start_clamped, en_end_clamped,
            K=self.K, max_span_len=self.max_span_len,
        )

        # ── 5. L_consistency (KL EN↔VI, stop-grad EN) ─────────
        l_cons = consistency_loss(
            en_start_logits, en_end_logits,
            vi_start_logits, vi_end_logits,
            temperature=self.temperature,
        )

        # ── 6. Tổng hợp ───────────────────────────────────────
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

    # Mock model outputs
    mock_outputs = {
        "gamma"       : torch.rand(B, K, K).softmax(dim=-1),
        "en_node_emb" : torch.randn(B, K, H),
        "vi_node_emb" : torch.randn(B, K, H),
        "D_en"        : torch.rand(B, K, K),
        "D_vi"        : torch.rand(B, K, K),
        "M"           : torch.rand(B, K, K),
    }

    # Mock batch
    mock_batch = {
        "en_start_position": torch.tensor([5, 0]),   # sample 1 answerable, sample 2 không
        "en_end_position"  : torch.tensor([10, 0]),
    }

    criterion = OTAlignmentLoss(qa_hidden_size=H, K=K)
    losses = criterion(mock_outputs, mock_batch)

    print("=== Loss Components ===")
    for k, v in losses.items():
        print(f"  {k:12s}: {v.item():.6f}")

    # Backward pass
    losses["total"].backward()
    print("\n✅ Backward pass OK — gradient flow hoạt động!")
    print("  • L_qa, L_fgw, L_span_proj, L_consistency đều tính được.")
    print("  • stop-gradient EN logits trong L_consistency: OK")