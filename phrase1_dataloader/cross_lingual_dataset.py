# cross_lingual_dataset.py
# 
# GIAO CHO: B
# NHIỆM VỤ: Implement CrossLingualQADataset + DataLoader (Task 3.1 – 3.4)
# DEPENDENCY: process_qa_sample từ preprocess_qa_sample.py (của A)
#   → Trong lúc A chưa xong, dùng mock_process_qa_sample từ data_setup.py
# 

import torch
from torch.utils.data import Dataset, DataLoader
import random

# Sau khi A merge: đổi lại import bên dưới
from preprocess_qa_sample import process_qa_sample
# from data_setup import mock_process_qa_sample as process_qa_sample  # ← dùng tạm khi dev


# 
# Task 3.1 – 3.3: CrossLingualQADataset

class CrossLingualQADataset(Dataset):
    def __init__(self, teacher_ds, student_ds, tokenizer, max_length=512, doc_stride=128):
        """
        Args:
            teacher_ds  : HuggingFace dataset tiếng Anh (SQuAD 2.0).
                          Mỗi sample có các field: "question", "context", "answers".
            student_ds  : HuggingFace dataset tiếng Việt (ViQuAD).
                          Mỗi sample có các field: "question", "context".
                          KHÔNG có "answers" (unanswerable từ góc nhìn của model).
            tokenizer   : XLM-R fast tokenizer (use_fast=True).
            max_length  : Độ dài tối đa chuỗi token.
            doc_stride  : Bước trượt sliding window.
        """
        self.teacher_ds = teacher_ds
        self.student_ds = student_ds
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride

        # TODO 3.1 — Tính và lưu độ dài dataset
        # Dùng min(len(teacher_ds), len(student_ds)) để tránh index out of range
        # khi hai tập có kích thước khác nhau.
        self.length = None  # TODO: thay bằng giá trị đúng

    def __len__(self):
        # TODO 3.2 — Trả về self.length
        raise NotImplementedError

    def __getitem__(self, idx):
        # ------------------------------------------------------------------
        # TODO 3.3a — Lấy English sample theo idx
        # ------------------------------------------------------------------
        # en_sample = self.teacher_ds[idx]
        # Các field cần dùng: en_sample["question"], en_sample["context"], en_sample["answers"]
        # Lưu ý: answers có dạng {"text": [...], "answer_start": [...]}
        en_sample = None  # TODO

        # ------------------------------------------------------------------
        # TODO 3.3b — Random pairing cho Vietnamese sample
        # ------------------------------------------------------------------
        # Phase 1: random để đơn giản.
        # TODO Phase 3: thay bằng semantic similarity matching.
        vi_idx    = random.randint(0, len(self.student_ds) - 1)
        vi_sample = None  # TODO: lấy self.student_ds[vi_idx]

        # ------------------------------------------------------------------
        # TODO 3.3c — Gọi process_qa_sample cho EN (có answer)
        # ------------------------------------------------------------------
        # Truyền đầy đủ keyword args để tránh nhầm thứ tự positional:
        #
        # en_ids, en_mask, en_start, en_end = process_qa_sample(
        #     question   = en_sample["question"],
        #     context    = en_sample["context"],
        #     answer     = en_sample["answers"],
        #     tokenizer  = self.tokenizer,
        #     max_length = self.max_length,
        #     doc_stride = self.doc_stride,
        # )
        en_ids, en_mask, en_start, en_end = None, None, None, None  # TODO

        # ------------------------------------------------------------------
        # TODO 3.3d — Gọi process_qa_sample cho VI (không có answer)
        # ------------------------------------------------------------------
        # Truyền answer=None để hàm của A xử lý unanswerable case.
        #
        # vi_ids, vi_mask, _, _ = process_qa_sample(
        #     question   = vi_sample["question"],
        #     context    = vi_sample["context"],
        #     answer     = None,
        #     tokenizer  = self.tokenizer,
        #     max_length = self.max_length,
        #     doc_stride = self.doc_stride,
        # )
        vi_ids, vi_mask = None, None  # TODO

        return {
            "en_input_ids"       : en_ids,
            "en_attention_mask"  : en_mask,
            "en_start_position"  : en_start,
            "en_end_position"    : en_end,
            "vi_input_ids"       : vi_ids,
            "vi_attention_mask"  : vi_mask,
        }


# ============================================================
# Task 3.4: collate_fn cho DataLoader
# ============================================================
def collate_fn(batch):
    """
    Stack list of dicts thành một dict of tensors.

    Args:
        batch: list[dict] — output của __getitem__, mỗi phần tử là 1 sample.

    Returns:
        dict[str, Tensor] — mỗi value có shape [batch_size, max_length]
                            (hoặc [batch_size] với position scalars).

    TODO 3.4: Dùng torch.stack để gom từng key lại.
    Gợi ý:
        return {
            k: torch.stack([item[k] for item in batch])
            for k in batch[0].keys()
        }
    """
    raise NotImplementedError


# ============================================================
# Smoke test — chạy thẳng file này để kiểm tra end-to-end
# ============================================================
if __name__ == "__main__":
    from data_setup import teacher_dataset, student_dataset, tokenizer

    # Dùng subset nhỏ để test nhanh
    import torch
    from torch.utils.data import Subset

    small_teacher = Subset(teacher_dataset, range(20))
    small_student = Subset(student_dataset, range(20))

    dataset    = CrossLingualQADataset(small_teacher, small_student, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    batch = next(iter(dataloader))

    print("DataLoader ready!")
    print("Batch keys     :", list(batch.keys()))
    print("Batch shapes   :", {k: tuple(v.shape) for k, v in batch.items()})

    # Kỳ vọng shape:
    #   en_input_ids / en_attention_mask / vi_input_ids / vi_attention_mask : (4, 512)
    #   en_start_position / en_end_position                                  : (4,)

    # Sanity check: start <= end với mọi sample trong batch
    assert (batch["en_start_position"] <= batch["en_end_position"]).all(), \
        "start_position > end_position — kiểm tra lại process_qa_sample!"
    print("start <= end passed!")