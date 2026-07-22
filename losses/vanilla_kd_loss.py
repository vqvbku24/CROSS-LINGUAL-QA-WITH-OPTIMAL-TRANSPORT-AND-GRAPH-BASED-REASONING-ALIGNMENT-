"""
Naive Index-to-Index Knowledge Distillation loss.

Đại diện cho "Vanilla KD" (M1) trong ablation study — một baseline KD
alignment-free: không có ma trận gamma, không projection ngữ nghĩa, chỉ ép
logits theo đúng chỉ số vị trí (index) sau khi crop về cùng độ dài.

Failure mode mong đợi: khi thứ tự từ giữa 2 ngôn ngữ khác nhau (vd. "red apple"
vs "táo đỏ" — tính từ đổi vị trí), việc ép theo index thô sẽ học sai ranh giới
câu trả lời. Đây là baseline được thiết kế để LÀM NỔI BẬT lý do cần ma trận
gamma trong L_span, không phải để mô phỏng một phương pháp cụ thể trong
literature.

Dùng chung cho cả VI và AR nhánh.
"""
import torch
import torch.nn.functional as F


def naive_index_to_index_kd_loss(
    student_start_logits: torch.Tensor,   # [B, L_student]
    student_end_logits: torch.Tensor,     # [B, L_student]
    teacher_start_logits: torch.Tensor,   # [B, L_teacher]
    teacher_end_logits: torch.Tensor,     # [B, L_teacher]
    student_valid_len: torch.Tensor,      # [B] actual non-pad length per example
    teacher_valid_len: torch.Tensor,      # [B] actual non-pad length per example
    student_gold_start: torch.Tensor,     # [B] gold start index (student side)
    student_gold_end: torch.Tensor,       # [B] gold end index (student side)
    temperature: float = 2.0,
):
    """
    QUAN TRỌNG VỀ THỨ TỰ XỬ LÝ (đã review kỹ, không được đổi):
    1. Crop cả hai tensor về min(valid_len) TRƯỚC khi softmax — nếu softmax
       trên full sequence rồi mới crop xác suất, tổng xác suất sau crop sẽ
       không còn bằng 1 và F.kl_div sẽ tính sai.
    2. Loại (mask) các example mà việc crop cắt mất vị trí gold answer thật
       của phía dài hơn — không được âm thầm tính loss trên vị trí sai.

    Returns:
        loss (scalar Tensor), valid_mask (BoolTensor [B])
    """
    B = student_start_logits.size(0)
    device = student_start_logits.device

    min_len = torch.minimum(student_valid_len, teacher_valid_len)  # [B]
    max_crop_len = int(min_len.max().item())

    # Mask ví dụ mà crop sẽ cắt mất gold span thật (crop point < gold end)
    valid_mask = (student_gold_end < min_len) & (student_gold_start < min_len)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), valid_mask

    def crop_and_mask(logits, lens, L):
        """Crop to L, mask out positions >= per-example valid len with a large
        negative constant (-1e9) instead of -inf.
        Using -inf causes softmax to return nan when ALL positions in a row are -inf
        (0/0 = nan). -1e9 is large enough to zero out softmax probabilities while
        remaining numerically stable.
        """
        cropped = logits[:, :L].clone()
        # Replace any pre-existing -inf values (e.g. from QA head question masking)
        # with -1e9 to avoid nan in softmax
        cropped = torch.nan_to_num(cropped, nan=0.0, posinf=1e9, neginf=-1e9)
        arange = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        pad_mask = arange >= lens.unsqueeze(1)                 # [B, L]
        cropped = cropped.masked_fill(pad_mask, -1e9)
        return cropped

    s_start = crop_and_mask(student_start_logits, min_len, max_crop_len)
    s_end   = crop_and_mask(student_end_logits,   min_len, max_crop_len)
    t_start = crop_and_mask(teacher_start_logits, min_len, max_crop_len)
    t_end   = crop_and_mask(teacher_end_logits,   min_len, max_crop_len)

    def kd_kl(s_logits, t_logits, mask):
        s_logits, t_logits = s_logits[mask], t_logits[mask]
        log_p_student = F.log_softmax(s_logits / temperature, dim=-1)
        p_teacher     = F.softmax(t_logits / temperature, dim=-1)
        # Safety: if NaN survives (e.g. all-zero rows after masking), zero out
        log_p_student = torch.nan_to_num(log_p_student, nan=0.0, neginf=-1e9)
        p_teacher     = torch.nan_to_num(p_teacher, nan=0.0)
        # batchmean + T^2 scaling: standard Hinton et al. KD
        return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temperature ** 2)

    loss_start = kd_kl(s_start, t_start, valid_mask)
    loss_end   = kd_kl(s_end,   t_end,   valid_mask)
    return 0.5 * (loss_start + loss_end), valid_mask
