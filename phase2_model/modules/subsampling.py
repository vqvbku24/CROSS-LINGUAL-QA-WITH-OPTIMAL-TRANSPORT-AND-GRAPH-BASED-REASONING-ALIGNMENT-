# modules/subsampling.py
import torch


def conditional_subsample(
    attention_map: torch.Tensor,
    question_indices: list,
    answer_indices: list,
    K: int = 128,
    soft_boost: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Giảm graph từ N nodes xuống K nodes.

    Hai chế độ tùy soft_boost:

      soft_boost == 0 (DEFAULT — dùng cho INFERENCE + VI training):
        - Hard-force question_indices vào graph.
        - answer_indices được ignore (truyền [] khi inference).
        - Phần còn lại chọn top-K attention.

      soft_boost > 0 (dùng cho EN TRAINING):
        - Hard-force question_indices.
        - answer_indices KHÔNG bị force, nhưng attention score được
          nhân boost_factor → rất likely (nhưng KHÔNG guaranteed) được
          chọn bởi top-K.
        - Tại sao: training graph gần giống inference graph (cùng dùng
          attention-based selection), nhưng answer tokens vẫn thường
          nằm trong graph → labels sạch hơn pure nearest-neighbor.

    Args:
        attention_map:    (N, N) — attention matrix của 1 sample
        question_indices: list[int] — indices của question tokens (bắt buộc giữ)
        answer_indices:   list[int] — indices để boost (KHÔNG hard-force)
        K:                int — số node sau subsampling
        soft_boost:       float — hệ số boost cho answer tokens.
                          0 = không boost (inference mode).
                          >0 = boost attention score (training mode). Khuyến nghị: 10.0

    Returns:
        sub_matrix: (K, K)
        keep_idx:   (K,) LongTensor
    """
    device = attention_map.device
    N = attention_map.shape[0]

    # ── Hard-force: chỉ question tokens ──────────────────────────
    forced = list(dict.fromkeys(question_indices))  # dedup, preserve order
    if len(forced) > K:
        forced = forced[:K]  # cắt question nếu quá dài (hiếm)

    forced_tensor = torch.tensor(forced, device=device, dtype=torch.long)
    num_needed = K - len(forced)

    # ── Attention score để chọn phần còn lại ─────────────────────
    attn_score = attention_map.sum(dim=0).clone()  # (N,)

    # ── Soft-boost: tăng score answer tokens TRƯỚC khi loại forced ─
    # Phải boost trước vì answer có thể trùng question → bị set -inf
    # trước khi kịp boost.
    if soft_boost > 0 and len(answer_indices) > 0:
        boost_idx = torch.tensor(answer_indices, device=device, dtype=torch.long)
        attn_score[boost_idx] = attn_score[boost_idx] * soft_boost

    # Loại forced (question tokens) khỏi top-k selection
    attn_score[forced_tensor] = float('-inf')

    # ── Top-K selection ──────────────────────────────────────────
    if num_needed > 0:
        available = (attn_score != float('-inf')).sum().item()
        _, topk_idx = torch.topk(attn_score, k=min(num_needed, available))
        keep_idx = torch.cat([forced_tensor, topk_idx])
    else:
        keep_idx = forced_tensor

    # Pad nếu thiếu (edge case: N quá nhỏ)
    if len(keep_idx) < K:
        remaining = torch.tensor([j for j in range(N) if j not in keep_idx.tolist()],
                                  device=device, dtype=torch.long)
        keep_idx = torch.cat([keep_idx, remaining[:K - len(keep_idx)]])

    sub_matrix = attention_map[keep_idx][:, keep_idx]  # (K, K)
    return sub_matrix, keep_idx