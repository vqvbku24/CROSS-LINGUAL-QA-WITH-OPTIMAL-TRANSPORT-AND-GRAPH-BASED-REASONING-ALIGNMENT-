# modules/subsampling.py
import torch


def conditional_subsample(
    attention_map: torch.Tensor,
    question_indices: list,
    answer_indices: list,
    K: int = 160
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Giảm graph từ N nodes xuống K nodes, đảm bảo giữ lại:
      - Tất cả question_indices (bao gồm [CLS])
      - Tất cả answer_indices (answer span)
      - Top-(K - len(forced)) nodes theo attention score

    Args:
        attention_map:    (N, N) — attention matrix của 1 sample
        question_indices: list[int] — indices của question tokens (bắt buộc giữ)
        answer_indices:   list[int] — indices của answer span (bắt buộc giữ)
        K:                int — số node sau subsampling

    Returns:
        sub_matrix: (K, K)
        keep_idx:   (K,) LongTensor
    
    Raises:
        ValueError nếu len(forced) > K
    """
    device = attention_map.device
    N = attention_map.shape[0]

    forced = list(dict.fromkeys(question_indices + answer_indices))  # dedup, preserve order
    if len(forced) > K:
        raise ValueError(f"Số forced tokens ({len(forced)}) > K ({K}). Tăng K lên.")

    forced_tensor = torch.tensor(forced, device=device, dtype=torch.long)
    num_needed = K - len(forced)

    # Score = tổng attention nhận được (column sum = "được chú ý nhiều")
    # Dùng column sum thay vì row sum để đo importance tốt hơn
    attn_score = attention_map.sum(dim=0).clone()  # (N,)
    attn_score[forced_tensor] = float('-inf')       # loại forced khỏi topk

    if num_needed > 0:
        # Chỉ chọn từ các token không bị mask (attention_mask)
        _, topk_idx = torch.topk(attn_score, k=min(num_needed, (attn_score != float('-inf')).sum()))
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