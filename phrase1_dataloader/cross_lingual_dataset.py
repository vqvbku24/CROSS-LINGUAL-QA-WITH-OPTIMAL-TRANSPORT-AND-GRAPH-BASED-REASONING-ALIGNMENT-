"""Task B: Dataset va DataLoader cho cross-lingual QA."""

import inspect
import random
from importlib import import_module

import torch
from torch.utils.data import DataLoader, Dataset


def _import_attr(module_names, attr_name):
    last_error = None
    for module_name in module_names:
        try:
            module = import_module(module_name)
            return getattr(module, attr_name)
        except (ImportError, AttributeError) as exc:
            last_error = exc
    if last_error is None:
        raise ImportError(f"Khong tim thay {attr_name}.")
    raise ImportError(f"Khong tim thay {attr_name}: {last_error}") from last_error


def _resolve_process_fn(process_fn=None):
    if process_fn is not None:
        return process_fn

    try:
        return _import_attr(
            ("phrase1_dataloader.process_qa_sample", "process_qa_sample"),
            "process_qa_sample",
        )
    except ImportError:
        return _import_attr(
            ("phrase1_dataloader.data_setup", "data_setup"),
            "mock_process_qa_sample",
        )


def _to_long_tensor(value):
    return torch.as_tensor(value, dtype=torch.long)


def _call_process_fn(process_fn, **kwargs):
    signature = inspect.signature(process_fn)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_kwargs:
        return process_fn(**kwargs)

    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return process_fn(**filtered_kwargs)


class CrossLingualQADataset(Dataset):
    def __init__(
        self,
        teacher_ds,
        student_ds,
        tokenizer,
        max_length=512,
        doc_stride=128,
        process_fn=None,
    ):
        """
        Args:
            teacher_ds  : Dataset tieng Anh (SQuAD v2).
            student_ds  : Dataset tieng Viet (ViQuAD).
            tokenizer   : XLM-R fast tokenizer (use_fast=True).
            max_length  : Do dai toi da cua chuoi token.
            doc_stride  : Buoc truot sliding window.
            process_fn  : Ham xu ly 1 QA sample. Neu None se uu tien ham that,
                          fallback sang mock de code cua B co the chay doc lap.
        """
        self.teacher_ds = teacher_ds
        self.student_ds = student_ds
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.process_fn = _resolve_process_fn(process_fn)
        self.length = min(len(self.teacher_ds), len(self.student_ds))

        if self.length == 0:
            raise ValueError("teacher_ds va student_ds phai deu co it nhat 1 sample.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} vuot qua dataset co do dai {self.length}.")

        en_sample = self.teacher_ds[idx]

        vi_idx = random.randint(0, len(self.student_ds) - 1)
        vi_sample = self.student_ds[vi_idx]

        en_ids, en_mask, en_start, en_end = _call_process_fn(
            self.process_fn,
            question=en_sample["question"],
            context=en_sample["context"],
            answer=en_sample.get("answers"),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            doc_stride=self.doc_stride,
        )

        vi_ids, vi_mask, _, _ = _call_process_fn(
            self.process_fn,
            question=vi_sample["question"],
            context=vi_sample["context"],
            answer=None,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            doc_stride=self.doc_stride,
        )

        return {
            "en_input_ids": _to_long_tensor(en_ids),
            "en_attention_mask": _to_long_tensor(en_mask),
            "en_start_position": _to_long_tensor(en_start),
            "en_end_position": _to_long_tensor(en_end),
            "vi_input_ids": _to_long_tensor(vi_ids),
            "vi_attention_mask": _to_long_tensor(vi_mask),
        }


def collate_fn(batch):
    """Stack list[dict[str, Tensor]] thanh dict[str, Tensor]."""
    if not batch:
        raise ValueError("batch rong, khong the torch.stack.")

    return {
        key: torch.stack([_to_long_tensor(item[key]) for item in batch], dim=0)
        for key in batch[0].keys()
    }


def create_dataloader(
    teacher_ds,
    student_ds,
    tokenizer,
    batch_size=4,
    shuffle=True,
    max_length=512,
    doc_stride=128,
    process_fn=None,
    **dataloader_kwargs,
):
    """
    Tao DataLoader san sang cho phase 1.

    Mac dinh:
        - batch_size=4
        - shuffle=True
        - collate_fn=collate_fn
    """
    dataset = CrossLingualQADataset(
        teacher_ds=teacher_ds,
        student_ds=student_ds,
        tokenizer=tokenizer,
        max_length=max_length,
        doc_stride=doc_stride,
        process_fn=process_fn,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )


def _load_setup_objects():
    setup_module = None
    for module_name in ("phrase1_dataloader.data_setup", "data_setup"):
        try:
            setup_module = import_module(module_name)
            break
        except ImportError:
            continue

    if setup_module is None:
        raise ImportError("Khong import duoc data_setup.")

    return setup_module.teacher_dataset, setup_module.student_dataset, setup_module.tokenizer


if __name__ == "__main__":
    from torch.utils.data import Subset

    teacher_dataset, student_dataset, tokenizer = _load_setup_objects()

    small_teacher = Subset(teacher_dataset, range(20))
    small_student = Subset(student_dataset, range(20))

    dataloader = create_dataloader(
        teacher_ds=small_teacher,
        student_ds=small_student,
        tokenizer=tokenizer,
        batch_size=4,
        shuffle=True,
    )
    batch = next(iter(dataloader))

    print("DataLoader ready!")
    print("Batch keys   :", list(batch.keys()))
    print("Batch shapes :", {k: tuple(v.shape) for k, v in batch.items()})

    assert (batch["en_start_position"] <= batch["en_end_position"]).all(), (
        "start_position > end_position — kiem tra lai process_qa_sample!"
    )
    print("start <= end passed!")
