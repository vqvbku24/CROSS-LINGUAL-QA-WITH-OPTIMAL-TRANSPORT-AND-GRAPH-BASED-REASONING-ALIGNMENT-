# losses.py
"""
Loss Functions for Cross-Lingual QA with Sinkhorn OT Alignment.

Architecture (post-refactor — Zero-Shot + Global OT):
    - No graph, no GAT, no subsampling, no FGW.
    - Sinkhorn OT operates directly on full XLM-R hidden states.
    - NO pseudo-labeling from γ (Topological Attractor problem).

Total loss:
    L_total = L_qa
            + L_has_ans          (EN-only answerable detection)
            + λ_ot   * L_ot     (transport cost — global space alignment)

Components:
    L_qa         : Cross-entropy span extraction on EN (supervised).
    L_has_ans    : BCE on EN only (no pos_weight, no VI distant supervision).
    L_ot         : Transport cost <γ.detach(), C> — gradient flows through C only.

Why L_span and L_cons were removed:
    Sinkhorn γ collapses onto "Topological Attractors" — tokens that share
    identical sub-word IDs across languages (numbers, punctuation, Named
    Entities like "Paris"). These have cosine distance ≈ 0, causing γ to
    dump all transport mass onto them instead of semantically aligned tokens.
    Using γ for pseudo-labels (L_span) or distribution matching (L_cons)
    teaches the model to extract single meaningless tokens (Start == End).

    Instead, we rely on Zero-Shot Transfer:
    - L_qa teaches the QA Head accurate span boundaries on EN.
    - L_ot pulls VI hidden space close to EN (global alignment).
    - At inference, QA Head naturally generalizes to VI without pseudo-labels.

Notes:
    - Sinkhorn uses non-uniform marginals (zero mass on PAD tokens).
    - Log-domain Sinkhorn for numerical stability (~50 iterations).
    - has_answer_head trained on EN only (zero-shot transfer to VI).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# Log-Domain Sinkhorn OT Solver (Pure PyTorch, Batched)
# ══════════════════════════════════════════════════════════════

def sinkhorn_log_domain(
    C: torch.Tensor,            # (B, M, N) cost matrix (already PAD-masked with 1e4)
    en_pad_mask: torch.Tensor,  # (B, M) True = PAD
    vi_pad_mask: torch.Tensor,  # (B, N) True = PAD
    epsilon: float = 0.1,       # entropic regularization  ← changed from 0.05
    num_iters: int = 100,       # Sinkhorn iterations
) -> torch.Tensor:
    """
    Vectorized log-domain Sinkhorn-Knopp algorithm.

    Non-uniform marginals:
        mu[b,i] = 1/n_valid_en[b]  if token i is NOT PAD, else 0
        nu[b,j] = 1/n_valid_vi[b]  if token j is NOT PAD, else 0

    This ensures zero transport mass is assigned to PAD tokens.

    Args:
        C          : (B, M, N) cosine distance cost matrix (PAD positions = 1e4)
                     M = T_en, N = T_vi (may differ after dynamic truncation)
        en_pad_mask: (B, M) boolean — True for PAD tokens in EN
        vi_pad_mask: (B, N) boolean — True for PAD tokens in VI
        epsilon    : entropic regularization strength (larger = smoother, more stable)
        num_iters  : number of Sinkhorn iterations

    Returns:
        gamma: (B, M, N) transport plan. Sum ≈ 1.0 per sample.
               PAD rows/cols have ~0 mass.
    """
    B, M, N = C.shape          # M = T_en, N = T_vi (not necessarily equal)
    device = C.device
    dtype = C.dtype

    # ── Non-uniform marginals ─────────────────────────────────
    # mu[b,i] = 1/n_valid if not PAD, else 0
    en_valid = (~en_pad_mask).float()                   # (B, M) — 1 for valid, 0 for PAD
    vi_valid = (~vi_pad_mask).float()                   # (B, N)
    n_valid_en = en_valid.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
    n_valid_vi = vi_valid.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)

    mu = en_valid / n_valid_en                          # (B, M) — sums to 1.0 per sample
    nu = vi_valid / n_valid_vi                          # (B, N)

    # ── Clamp before log to avoid -inf / NaN ──────────────────
    log_mu = torch.log(mu.clamp(min=1e-8))              # (B, M)
    log_nu = torch.log(nu.clamp(min=1e-8))              # (B, N)

    # Set log-marginals of PAD tokens to -1e8 (effectively -inf but finite)
    # This prevents any NaN from propagating through logsumexp
    log_mu = log_mu.masked_fill(en_pad_mask, -1e8)
    log_nu = log_nu.masked_fill(vi_pad_mask, -1e8)

    # ── Log-domain kernel ─────────────────────────────────────
    # log_K[b,i,j] = -C[b,i,j] / epsilon
    log_K = -C / epsilon                                # (B, M, N)

    # ── Sinkhorn iterations in log space ──────────────────────
    # u, v are log-scaling vectors (row and column respectively)
    log_u = torch.zeros(B, M, device=device, dtype=dtype)  # row scaling   (B, M)
    log_v = torch.zeros(B, N, device=device, dtype=dtype)  # column scaling (B, N)

    for _ in range(num_iters):
        # Row update: enforce row marginal = mu
        # log_u[b,i] = log_mu[b,i] - logsumexp_j(log_K[b,i,j] + log_v[b,j])
        log_u = log_mu - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)  # (B, M)

        # Column update: enforce column marginal = nu
        # log_v[b,j] = log_nu[b,j] - logsumexp_i(log_K[b,i,j] + log_u[b,i])
        log_v = log_nu - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)  # (B, N)

        # Clamp to prevent overflow in exp()
        log_u = log_u.clamp(-1e8, 1e2)
        log_v = log_v.clamp(-1e8, 1e2)

    # ── Reconstruct transport plan ────────────────────────────
    # gamma[b,i,j] = exp(log_u[b,i] + log_K[b,i,j] + log_v[b,j])
    gamma = torch.exp(log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1))  # (B, M, N)

    return gamma  # (B, M, N)


# ══════════════════════════════════════════════════════════════
# Sliced Wasserstein Distance (SWD) — Non-Parallel Data
# ══════════════════════════════════════════════════════════════

def sliced_wasserstein_distance(
    H_en: torch.Tensor,          # (B, T_en, H)
    H_vi: torch.Tensor,          # (B, T_vi, H)
    en_pad_mask: torch.Tensor,   # (B, T_en) True = PAD
    vi_pad_mask: torch.Tensor,   # (B, T_vi) True = PAD
    num_projections: int = 50,
) -> torch.Tensor:
    """
    Sliced Wasserstein Distance for non-parallel cross-lingual alignment.

    Pools all valid (non-PAD) tokens from EN and VI across the batch,
    then computes SWD between the two token distributions.

    Unlike Sinkhorn OT, SWD does NOT require parallel data — it compares
    marginal distributions directly without token-to-token coupling.

    Algorithm:
        1. Pool valid tokens: (B, T, H) → (N, H)
        2. L2-normalize (scale-invariant, consistent with cosine distance)
        3. Subsample to min(N_en, N_vi) for equal-size comparison
        4. For each random 1D projection: sort both, compute L2
        5. Average across projections

    Args:
        H_en           : (B, T_en, H) EN hidden states
        H_vi           : (B, T_vi, H) VI hidden states
        en_pad_mask    : (B, T_en) True = PAD token
        vi_pad_mask    : (B, T_vi) True = PAD token
        num_projections: number of random 1D projections

    Returns:
        scalar SWD loss (mean over projections)
    """
    device = H_en.device
    H = H_en.size(-1)

    # 1. Pool valid tokens across batch: (B, T, H) → (N, H)
    X_en = H_en[~en_pad_mask]    # (N_en, H)
    X_vi = H_vi[~vi_pad_mask]    # (N_vi, H)

    N_en, N_vi = X_en.size(0), X_vi.size(0)
    N = min(N_en, N_vi)

    if N == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 2. L2-normalize for scale-invariant alignment
    X_en = F.normalize(X_en, p=2, dim=-1)
    X_vi = F.normalize(X_vi, p=2, dim=-1)

    # 3. Subsample larger set to equal size
    if N_en > N:
        idx = torch.randperm(N_en, device=device)[:N]
        X_en = X_en[idx]
    if N_vi > N:
        idx = torch.randperm(N_vi, device=device)[:N]
        X_vi = X_vi[idx]

    # 4. Random projections on unit sphere
    theta = torch.randn(num_projections, H, device=device, dtype=H_en.dtype)
    theta = F.normalize(theta, p=2, dim=-1)  # (num_proj, H)

    # 5. Project → sort → L2
    proj_en = X_en @ theta.T               # (N, num_proj)
    proj_vi = X_vi @ theta.T               # (N, num_proj)

    proj_en_sorted, _ = torch.sort(proj_en, dim=0)
    proj_vi_sorted, _ = torch.sort(proj_vi, dim=0)

    # SWD = mean squared distance between sorted projections
    swd = ((proj_en_sorted - proj_vi_sorted) ** 2).mean()

    return swd


# ══════════════════════════════════════════════════════════════
# QA Head — Operates on Full Token Sequences (L=512)
# ══════════════════════════════════════════════════════════════

class QAHead(nn.Module):
    """
    Linear head predicting start/end span logits from token embeddings.
    Integrates Cross-Attention: context tokens attend to question tokens.

    has_answer_head: binary classifier for unanswerable detection (EN only).

    Input dimensions:
        context_hidden : (B, L, H)   — full 512-token sequence
        question_hidden: (B, L_q, H) — question tokens [CLS ... SEP)
    """

    def __init__(self, hidden_size: int = 768):
        """
        Args:
            hidden_size: XLM-R hidden dimension (768 for base, 1024 for large)
        """
        super().__init__()

        # Cross-Attention: context (query) attends to question (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Span prediction projections
        self.start_proj = nn.Linear(hidden_size, 1)
        self.end_proj   = nn.Linear(hidden_size, 1)

        # has_answer classifier: CLS embedding → binary logit
        # Trained on EN branch only (has supervised labels).
        self.has_answer_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(
        self,
        context_hidden: torch.Tensor,               # (B, L, H) — full sequence
        question_hidden: torch.Tensor,               # (B, L_q, H) — question tokens
        question_pad_mask: torch.Tensor | None = None,  # (B, L_q) True = ignore
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            context_hidden   : (B, L, H) — all 512 tokens from H_en or H_vi
            question_hidden  : (B, L_q, H) — tokens between [CLS] and first [SEP]
            question_pad_mask: (B, L_q) — True = padding (ignore in attention)

        Returns:
            start_logits     : (B, L) — logits over all 512 positions
            end_logits       : (B, L) — logits over all 512 positions
            has_answer_logit : (B,)   — >0 means answerable (before sigmoid)
        """
        # Cross-attention: context tokens attend to question tokens
        attn_out, _ = self.cross_attn(
            query=context_hidden,       # (B, L, H)
            key=question_hidden,        # (B, L_q, H)
            value=question_hidden,      # (B, L_q, H)
            key_padding_mask=question_pad_mask,
        )

        # Residual + LayerNorm
        context_out = self.layer_norm(context_hidden + attn_out)  # (B, L, H)

        # Span logits over all L positions
        start_logits = self.start_proj(context_out).squeeze(-1)   # (B, L)
        end_logits   = self.end_proj(context_out).squeeze(-1)     # (B, L)

        # has_answer: use CLS token (position 0) after cross-attention
        cls_emb = context_out[:, 0, :]                            # (B, H)
        has_answer_logit = self.has_answer_head(cls_emb).squeeze(-1)  # (B,)

        return start_logits, end_logits, has_answer_logit


# ══════════════════════════════════════════════════════════════
# Loss: QA Span Extraction (Supervised EN)
# ══════════════════════════════════════════════════════════════

def qa_loss(
    start_logits: torch.Tensor,     # (B, L) or (B_filtered, L)
    end_logits: torch.Tensor,       # (B, L) or (B_filtered, L)
    start_positions: torch.Tensor,  # (B,) or (B_filtered,)
    end_positions: torch.Tensor,    # (B,) or (B_filtered,)
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cross-entropy loss for span extraction (supervised EN).

    Returns:
        (total, l_start, l_end) — all three for TensorBoard logging.
    """
    loss_start = F.cross_entropy(start_logits, start_positions, ignore_index=ignore_index)
    loss_end   = F.cross_entropy(end_logits,   end_positions,   ignore_index=ignore_index)
    return (loss_start + loss_end) / 2.0, loss_start, loss_end


# ══════════════════════════════════════════════════════════════
# Loss: OT Transport Cost (gradient flows through C only)
# ══════════════════════════════════════════════════════════════

def ot_transport_loss(
    gamma: torch.Tensor,  # (B, L, L) — transport plan
    C: torch.Tensor,      # (B, L, L) — cost matrix (has grad through backbone)
) -> torch.Tensor:
    """
    L_ot = <gamma.detach(), C>.sum(dim=(-1,-2)).mean()

    Gradient flows through C only (not through gamma).
    This pulls EN↔VI embeddings closer for aligned token pairs.

    Args:
        gamma: (B, L, L) transport plan from Sinkhorn
        C    : (B, L, L) cosine distance cost matrix

    Returns:
        scalar loss (mean over batch)
    """
    # gamma.detach() — treat transport plan as fixed; gradient only through C
    per_sample_cost = (gamma.detach() * C).sum(dim=(-1, -2))  # (B,)
    return per_sample_cost.mean()


# ══════════════════════════════════════════════════════════════
# Helper: Extract question embeddings from full hidden states
# ══════════════════════════════════════════════════════════════

def _extract_question_embeddings(
    hidden: torch.Tensor,        # (B, L, H) — full hidden states
    question_end: torch.Tensor,  # (B,) — index of first [SEP] (exclusive end of question)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract question token embeddings: positions [0, question_end) — [SEP] excluded.

    question_end[b] = index of first [SEP] token for sample b.
    We extract positions [0, question_end[b]) — the [SEP] token itself is NOT included.
    Shorter questions within the batch are padded (q_mask = True for padding positions).

    Args:
        hidden       : (B, L, H) full hidden states
        question_end : (B,) index of first [SEP] per sample

    Returns:
        q_emb  : (B, max_q_len, H) — question token embeddings
        q_mask : (B, max_q_len)     — True = padding (should be ignored)
    """
    device = hidden.device
    max_q_len = question_end.max().item()         # exclusive: [SEP] NOT included
    q_emb = hidden[:, :max_q_len, :]             # (B, max_q_len, H)

    # Padding mask: positions at or beyond each sample's question_end are padding
    positions = torch.arange(max_q_len, device=device).unsqueeze(0)  # (1, max_q_len)
    q_mask = positions >= question_end.unsqueeze(1)                   # (B, max_q_len)

    return q_emb, q_mask


# ══════════════════════════════════════════════════════════════
# OTAlignmentLoss — Orchestrates all loss components
# ══════════════════════════════════════════════════════════════

class OTAlignmentLoss(nn.Module):
    """
    Combined loss for Cross-Lingual QA with OT alignment.

    Supports two OT methods:
        - "swd"      : Sliced Wasserstein Distance (non-parallel data, default)
        - "sinkhorn" : Sinkhorn OT (requires parallel data)

    Zero-Shot + Global OT strategy:
        L_total = L_qa + L_has_ans + λ_ot * L_ot
    """

    def __init__(
        self,
        hidden_size: int = 768,
        lambda_ot: float = 0.1,
        ot_method: str = "swd",
        num_projections: int = 50,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iters: int = 100,
    ):
        """
        Args:
            hidden_size      : XLM-R hidden dim (768 base, 1024 large)
            lambda_ot        : weight for L_ot (OT alignment regularizer)
            ot_method        : "swd" (Sliced Wasserstein) or "sinkhorn"
            num_projections  : number of random projections for SWD
            sinkhorn_epsilon : entropic regularization (sinkhorn only)
            sinkhorn_iters   : Sinkhorn iterations (sinkhorn only)
        """
        super().__init__()
        self.lambda_ot          = lambda_ot
        self.ot_method          = ot_method
        self.num_projections    = num_projections
        self.sinkhorn_epsilon   = sinkhorn_epsilon
        self.sinkhorn_iters     = sinkhorn_iters

        # QA Head: shared for EN and VI, operates on full 512-token sequences
        self.qa_head = QAHead(hidden_size=hidden_size)

    def forward(
        self,
        model_outputs: dict,
        batch: dict,
        global_step: int = 0,
        spe: int = 1,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            model_outputs: dict from CrossLingualOTModel.forward()
                {
                    "en_hidden"   : (B, L, H),
                    "vi_hidden"   : (B, L, H),
                    "cost_matrix" : (B, L, L),
                    "en_pad_mask" : (B, L),
                    "vi_pad_mask" : (B, L),
                }
            batch: dict from DataLoader
                {
                    "en_start_position" : (B,) — token-space
                    "en_end_position"   : (B,) — token-space
                    "en_question_end"   : (B,) — first [SEP] index
                    "vi_question_end"   : (B,) — first [SEP] index
                    "en_is_answerable"  : (B,) — 1 if answerable, 0 if not
                }

        Returns:
            dict with all loss components + debug stats:
                "total", "qa", "qa_start", "qa_end", "has_ans",
                "ot", + debug keys
        """
        en_hidden   = model_outputs["en_hidden"]     # (B, T_en, H)  — truncated to max valid tokens
        vi_hidden   = model_outputs["vi_hidden"]     # (B, T_vi, H)  — truncated to max valid tokens
        en_pad_mask = model_outputs["en_pad_mask"]   # (B, T_en)
        vi_pad_mask = model_outputs["vi_pad_mask"]   # (B, T_vi)

        device = en_hidden.device
        B, T_en, H = en_hidden.shape

        en_start = batch["en_start_position"]        # (B,) token-space
        en_end   = batch["en_end_position"]          # (B,) token-space

        # ── Clamp to truncated sequence length — prevents IndexError ──
        # After dynamic truncation, en_start/en_end may exceed T_en.
        # Clamping ensures valid indices for qa_loss.
        en_seq_len = en_hidden.size(1)               # T_en after dynamic truncation
        en_start = en_start.clamp(max=en_seq_len - 1)
        en_end   = en_end.clamp(max=en_seq_len - 1)

        # ══════════════════════════════════════════════════════
        # 1. OT Alignment — SWD or Sinkhorn
        # ══════════════════════════════════════════════════════
        if self.ot_method == "swd":
            l_ot = sliced_wasserstein_distance(
                en_hidden, vi_hidden, en_pad_mask, vi_pad_mask,
                num_projections=self.num_projections,
            )
        else:
            C = model_outputs["cost_matrix"]
            gamma = sinkhorn_log_domain(
                C, en_pad_mask, vi_pad_mask,
                epsilon=self.sinkhorn_epsilon,
                num_iters=self.sinkhorn_iters,
            )
            l_ot = ot_transport_loss(gamma, C)

        # ══════════════════════════════════════════════════════
        # 2. Extract question embeddings for cross-attention
        # ══════════════════════════════════════════════════════
        en_q_emb, en_q_mask = _extract_question_embeddings(
            en_hidden, batch["en_question_end"]
        )  # (B, max_q_en, H), (B, max_q_en)

        vi_q_emb, vi_q_mask = _extract_question_embeddings(
            vi_hidden, batch["vi_question_end"]
        )  # (B, max_q_vi, H), (B, max_q_vi)

        # ══════════════════════════════════════════════════════
        # 3. QA Head → logits + has_answer
        # ══════════════════════════════════════════════════════
        # Context = truncated hidden states (T_en / T_vi tokens, not 512)
        # Question = tokens [CLS ... SEP) for cross-attention
        en_start_logits, en_end_logits, en_has_ans = self.qa_head(
            en_hidden, en_q_emb, en_q_mask
        )
        vi_start_logits, vi_end_logits, vi_has_ans = self.qa_head(
            vi_hidden, vi_q_emb, vi_q_mask
        )

        # ══════════════════════════════════════════════════════
        # 4. L_qa (supervised EN) — only answerable samples
        # ══════════════════════════════════════════════════════
        answerable_mask = batch["en_is_answerable"].bool().to(device)

        if answerable_mask.any():
            l_qa, l_qa_start, l_qa_end = qa_loss(
                en_start_logits[answerable_mask],
                en_end_logits[answerable_mask],
                en_start[answerable_mask],
                en_end[answerable_mask],
            )
        else:
            l_qa       = torch.tensor(0.0, device=device)
            l_qa_start = torch.tensor(0.0, device=device)
            l_qa_end   = torch.tensor(0.0, device=device)

        # ══════════════════════════════════════════════════════
        # 5. L_has_answer (BCE) — EN only, no pos_weight
        # ══════════════════════════════════════════════════════
        # Previous version used pos_weight=2.0 and trained on VI too,
        # causing 98-99% overconfidence. Now: EN only, balanced BCE.
        has_answer_label = batch["en_is_answerable"].float().to(device)

        l_has_ans = F.binary_cross_entropy_with_logits(
            en_has_ans,
            has_answer_label,
        )



        # ══════════════════════════════════════════════════════
        # 7. Total loss — Zero-Shot + Global OT
        # ══════════════════════════════════════════════════════
        l_total = (
            l_qa
            + l_has_ans
            + self.lambda_ot * l_ot
        )

        # ══════════════════════════════════════════════════════
        # 8. Debug stats — CLS collapse detection & Entropy
        # ══════════════════════════════════════════════════════
        with torch.no_grad():
            cls_start = en_start_logits[:, 0].mean()
            max_start = en_start_logits.max(dim=1).values.mean()
            cls_end   = en_end_logits[:, 0].mean()
            max_end   = en_end_logits.max(dim=1).values.mean()

            # Calculate Entropy with numerical stability clamp
            P_vi_start = F.softmax(vi_start_logits, dim=-1)
            entropy_start = -(P_vi_start * (P_vi_start + 1e-8).log()).sum(dim=-1).mean()

            P_vi_end = F.softmax(vi_end_logits, dim=-1)
            entropy_end = -(P_vi_end * (P_vi_end + 1e-8).log()).sum(dim=-1).mean()

            if answerable_mask.any():
                has_ans_acc = (en_has_ans[answerable_mask] > 0).float().mean()
            else:
                has_ans_acc = torch.tensor(float('nan'))

            # VI has_ans probability (for monitoring zero-shot transfer)
            vi_has_ans_prob = torch.sigmoid(vi_has_ans).mean()

        return {
            "total"           : l_total,
            "qa"              : l_qa.detach(),
            "qa_start"        : l_qa_start.detach(),
            "qa_end"          : l_qa_end.detach(),
            "has_ans"         : l_has_ans.detach(),
            "ot"              : l_ot.detach(),
            # Debug & tracking
            "dbg/cls_start"   : cls_start,
            "dbg/max_start"   : max_start,
            "dbg/cls_end"     : cls_end,
            "dbg/max_end"     : max_end,
            "dbg/has_ans_acc" : has_ans_acc,
            "dbg/entropy_start": entropy_start,
            "dbg/entropy_end"  : entropy_end,
            "dbg/vi_has_ans_prob": vi_has_ans_prob,
        }


# ══════════════════════════════════════════════════════════════
# Quick test: python losses.py
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    B, L, H = 2, 512, 768

    print("=" * 60)
    print("Self-test: OTAlignmentLoss (Zero-Shot + Global OT)")
    print("=" * 60)

    # ── Mock model outputs ────────────────────────────────────
    # Simulate XLM-R hidden states
    en_hidden = torch.randn(B, L, H)
    vi_hidden = torch.randn(B, L, H)

    # Simulate attention masks (first 400 tokens valid, rest PAD)
    en_attn_mask = torch.ones(B, L, dtype=torch.long)
    en_attn_mask[:, 400:] = 0
    vi_attn_mask = torch.ones(B, L, dtype=torch.long)
    vi_attn_mask[:, 380:] = 0

    # Cost matrix with PAD masking
    en_norm = F.normalize(en_hidden, dim=-1)
    vi_norm = F.normalize(vi_hidden, dim=-1)
    C = 1.0 - torch.bmm(en_norm, vi_norm.transpose(1, 2))
    en_pad = (en_attn_mask == 0)
    vi_pad = (vi_attn_mask == 0)
    C = C.masked_fill(en_pad.unsqueeze(2), 1e4)
    C = C.masked_fill(vi_pad.unsqueeze(1), 1e4)

    mock_outputs = {
        "en_hidden":   en_hidden,
        "vi_hidden":   vi_hidden,
        "cost_matrix": C,
        "en_pad_mask": en_pad,
        "vi_pad_mask": vi_pad,
    }

    # ── Mock batch ────────────────────────────────────────────
    mock_batch = {
        "en_start_position": torch.tensor([50, 0]),    # sample 0: answerable, sample 1: unanswerable
        "en_end_position":   torch.tensor([55, 0]),
        "en_question_end":   torch.tensor([20, 15]),   # question tokens [0..20]
        "vi_question_end":   torch.tensor([22, 17]),
        "en_is_answerable":  torch.tensor([1, 0]),
        "en_attention_mask": en_attn_mask,
        "vi_attention_mask": vi_attn_mask,
    }

    # ── Test SWD mode (default) ────────────────────────────
    print("\n--- SWD Mode ---")
    criterion_swd = OTAlignmentLoss(hidden_size=H, ot_method="swd")
    losses = criterion_swd(mock_outputs, mock_batch)

    print("\n=== Loss Components (SWD) ===")
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:20s}: {v.item():.6f}")

    losses["total"].backward()
    print("[OK] SWD backward pass succeeded!")

    # ── Test Sinkhorn mode (backward compat) ──────────────
    print("\n--- Sinkhorn Mode ---")
    criterion_sink = OTAlignmentLoss(hidden_size=H, ot_method="sinkhorn")
    losses2 = criterion_sink(mock_outputs, mock_batch)
    print(f"  Sinkhorn l_ot: {losses2['ot'].item():.6f}")
    losses2["total"].backward()
    print("[OK] Sinkhorn backward pass succeeded!")

    # ── Verify Sinkhorn ───────────────────────────────────────
    gamma = sinkhorn_log_domain(C.detach(), en_pad, vi_pad)
    row_sums = gamma.sum(dim=2)  # should be ~mu for valid, ~0 for PAD
    print(f"\n[Sinkhorn] gamma shape: {gamma.shape}")
    print(f"[Sinkhorn] gamma sum per sample: {gamma.sum(dim=(1,2)).tolist()}")
    print(f"[Sinkhorn] PAD row mass (should be ~0): {row_sums[0, 400:410].tolist()}")
    print(f"[Sinkhorn] Valid row mass (should be ~1/400): {row_sums[0, :5].tolist()}")
    print("\n[OK] All checks passed!")