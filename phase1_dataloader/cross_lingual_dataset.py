# cross_lingual_dataset.py
import inspect
import random
from importlib import import_module
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset


# ──────────────────────────────────────────────────────────────
# Helpers nội bộ
# ──────────────────────────────────────────────────────────────

def _import_attr(module_names: list[str], attr_name: str):
    last_error = None
    for module_name in module_names:
        try:
            module = import_module(module_name)
            return getattr(module, attr_name)
        except (ImportError, AttributeError) as exc:
            last_error = exc
    raise ImportError(f"Không tìm thấy '{attr_name}': {last_error}") from last_error


def _resolve_process_fn(process_fn: Optional[Callable] = None) -> Callable:
    """Ưu tiên: hàm thật → mock. Cho phép team chạy độc lập."""
    if process_fn is not None:
        return process_fn

    # Thử import hàm thật trước
    for mod in ("phase1_dataloader.process_qa_sample", "process_qa_sample"):
        try:
            return _import_attr([mod], "process_qa_sample")
        except ImportError:
            continue

    # Fallback: mock
    for mod in ("phase1_dataloader.data_setup", "data_setup"):
        try:
            return _import_attr([mod], "mock_process_qa_sample")
        except ImportError:
            continue

    raise ImportError("Không tìm thấy process_qa_sample lẫn mock_process_qa_sample.")


def _to_long_tensor(value) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.long)


def _call_process_fn(process_fn: Callable, **kwargs) -> tuple:
    """Gọi process_fn, chỉ truyền những kwargs mà hàm đó chấp nhận."""
    sig = inspect.signature(process_fn)
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if accepts_var_kw:
        return process_fn(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return process_fn(**filtered)


# ──────────────────────────────────────────────────────────────
# Semantic pairing helper
# ──────────────────────────────────────────────────────────────

def _build_vi_index(student_ds) -> dict[str, list[int]]:
    """
    Xây dựng bucket theo title/topic nếu dataset có trường 'title'.
    Fallback: một bucket duy nhất chứa tất cả indices.

    Returns:
        dict[topic_key → list[int]]  (topic_key = '' nếu không có title)
    """
    buckets: dict[str, list[int]] = {}
    for i in range(len(student_ds)):
        try:
            key = student_ds[i].get("title", "")
        except Exception:
            key = ""
        buckets.setdefault(key, []).append(i)
    return buckets


def _pair_vi_index(
    en_sample: dict,
    vi_buckets: dict[str, list[int]],
    en_idx: int,
    student_ds_len: int,
    strategy: str = "topic",
) -> int:
    """
    Chọn VI index sao cho có tương đồng về topic với EN sample.

    strategy:
        "topic"  — ưu tiên cùng title/topic; fallback modulo nếu không khớp.
        "modulo" — luôn dùng en_idx % len(student_ds) (đơn giản, ổn định).
        "random" — giữ behaviour cũ (không khuyến nghị).
    """
    if strategy == "random":
        return random.randint(0, student_ds_len - 1)

    if strategy == "modulo":
        return en_idx % student_ds_len

    # strategy == "topic"
    en_title = en_sample.get("title", "")
    if en_title and en_title in vi_buckets:
        bucket = vi_buckets[en_title]
        return random.choice(bucket)

    # Không tìm thấy topic khớp → fallback modulo (deterministic, không random)
    return en_idx % student_ds_len


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class CrossLingualQADataset(Dataset):
    def __init__(
        self,
        teacher_ds,
        student_ds,
        tokenizer,
        max_length: int = 512,
        doc_stride: int = 128,
        process_fn: Optional[Callable] = None,
        pairing_strategy: str = "topic",
    ):
        """
        Args:
            teacher_ds        : Dataset tiếng Anh (SQuAD v2).
            student_ds        : Dataset tiếng Việt (ViQuAD).
            tokenizer         : XLM-R fast tokenizer (use_fast=True).
            max_length        : Độ dài tối đa của chuỗi token.
            doc_stride        : Bước trượt sliding window.
            process_fn        : Hàm xử lý 1 QA sample. None → auto-resolve.
            pairing_strategy  : "topic" | "modulo" | "random"
                                "topic"  — khớp theo title/domain (tốt nhất cho L_consistency)
                                "modulo" — deterministic, dùng khi không có title
                                "random" — behaviour cũ (không khuyến nghị)
        """
        if len(teacher_ds) == 0 or len(student_ds) == 0:
            raise ValueError("teacher_ds và student_ds phải có ít nhất 1 sample.")

        self.teacher_ds       = teacher_ds
        self.student_ds       = student_ds
        self.tokenizer        = tokenizer
        self.max_length       = max_length
        self.doc_stride       = doc_stride
        self.process_fn       = _resolve_process_fn(process_fn)
        self.pairing_strategy = pairing_strategy
        self.length           = min(len(teacher_ds), len(student_ds))

        # Xây bucket một lần duy nhất
        self._vi_buckets = _build_vi_index(student_ds)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} vượt quá dataset (length={self.length}).")

        en_sample = self.teacher_ds[idx]

        # FIX: pairing có nghĩa ngữ nghĩa thay vì random hoàn toàn
        vi_idx = _pair_vi_index(
            en_sample=en_sample,
            vi_buckets=self._vi_buckets,
            en_idx=idx,
            student_ds_len=len(self.student_ds),
            strategy=self.pairing_strategy,
        )
        vi_sample = self.student_ds[vi_idx]

        # Xử lý format answer từ HuggingFace datasets
        # (field "answers" trong SQuAD có dạng {"text": [...], "answer_start": [...]})
        en_answer = en_sample.get("answers") or en_sample.get("answer")

        # Tokenize EN (có answer span)
        en_ids, en_mask, en_start, en_end, en_qend = _call_process_fn(
            self.process_fn,
            question=en_sample["question"],
            context=en_sample["context"],
            answer=en_answer,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            doc_stride=self.doc_stride,
        )

        # Tokenize VI (không có answer — truyền None)
        vi_ids, vi_mask, _, _, vi_qend = _call_process_fn(
            self.process_fn,
            question=vi_sample["question"],
            context=vi_sample["context"],
            answer=None,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            doc_stride=self.doc_stride,
        )

        return {
            "en_input_ids":       _to_long_tensor(en_ids),
            "en_attention_mask":  _to_long_tensor(en_mask),
            "en_start_position":  _to_long_tensor(en_start),
            "en_end_position":    _to_long_tensor(en_end),
            "en_question_end":    _to_long_tensor(en_qend),   # ← FIX: thêm mới
            "vi_input_ids":       _to_long_tensor(vi_ids),
            "vi_attention_mask":  _to_long_tensor(vi_mask),
            "vi_question_end":    _to_long_tensor(vi_qend),   # ← thêm để đối xứng
        }


# ──────────────────────────────────────────────────────────────
# Collate
# ──────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack list[dict[str, Tensor]] thành dict[str, Tensor]."""
    if not batch:
        raise ValueError("batch rỗng, không thể torch.stack.")
    return {
        key: torch.stack([_to_long_tensor(item[key]) for item in batch], dim=0)
        for key in batch[0].keys()
    }


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def create_dataloader(
    teacher_ds,
    student_ds,
    tokenizer,
    batch_size: int = 4,
    shuffle: bool = True,
    max_length: int = 512,
    doc_stride: int = 128,
    process_fn: Optional[Callable] = None,
    pairing_strategy: str = "topic",
    **dataloader_kwargs,
) -> DataLoader:
    """
    Tạo DataLoader sẵn sàng cho Phase 1.

    Args:
        pairing_strategy: "topic" | "modulo" | "random"
                          Khuyến nghị: "topic" để L_consistency học mapping hợp lý.
    """
    dataset = CrossLingualQADataset(
        teacher_ds=teacher_ds,
        student_ds=student_ds,
        tokenizer=tokenizer,
        max_length=max_length,
        doc_stride=doc_stride,
        process_fn=process_fn,
        pairing_strategy=pairing_strategy,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )


# ──────────────────────────────────────────────────────────────
# Smoke test — python cross_lingual_dataset.py
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch.utils.data import Subset

    # FIX: lazy import — không load dataset ở module level
    from data_setup import get_setup_objects

    teacher_dataset, student_dataset, tokenizer = get_setup_objects()

    small_teacher = Subset(teacher_dataset, range(20))
    small_student = Subset(student_dataset, range(20))

    dataloader = create_dataloader(
        teacher_ds=small_teacher,
        student_ds=small_student,
        tokenizer=tokenizer,
        batch_size=4,
        shuffle=True,
        pairing_strategy="modulo",  # dùng modulo vì Subset nhỏ, không có title đủ
    )

    batch = next(iter(dataloader))

    print("DataLoader ready!")
    print("Batch keys   :", list(batch.keys()))
    print("Batch shapes :", {k: tuple(v.shape) for k, v in batch.items()})

    # Assertions
    assert "en_question_end" in batch, "FAIL: thiếu en_question_end"
    assert "vi_question_end" in batch, "FAIL: thiếu vi_question_end"
    assert (batch["en_start_position"] <= batch["en_end_position"]).all(), \
        "FAIL: start_position > end_position"
    assert (batch["en_question_end"] >= 0).all(), \
        "FAIL: en_question_end âm"

    # Kiểm tra en_question_end < max_length
    assert (batch["en_question_end"] < 512).all(), \
        "FAIL: en_question_end >= max_length"

    print("\n✅ Tất cả assertions PASSED!")
    print("  • en_question_end: có mặt và hợp lệ")
    print("  • start <= end: OK")
    print("  • pairing_strategy=modulo: OK")
