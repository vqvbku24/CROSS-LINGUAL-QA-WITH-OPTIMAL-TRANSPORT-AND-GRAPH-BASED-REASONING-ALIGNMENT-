# data/xquad_loader.py
"""
XQuAD Dataloader for Stage 2 Teacher-Student Sinkhorn Alignment.

XQuAD VI: 1190 parallel (EN, VI) QA pairs.
  train: first 1010 pairs (85%) — used for Stage 2 training
  val:   last  180 pairs (15%) — used ONLY for early stopping eval

Key constraints:
  - Split is index-based (deterministic, not random).
  - EN and VI are padded independently to their own batch-max length (≤ 384).
  - No vi_start/end_positions in batch — pseudo-labels come from γ.
  - ViQuAD dataset is NEVER loaded here.
"""

import json
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_dataloader.process_qa_sample import process_qa_sample


# ──────────────────────────────────────────────────────────────
# Parse XQuAD VI JSON into parallel (EN, VI) pairs
# ──────────────────────────────────────────────────────────────

def load_xquad_pairs(xquad_vi_path: str, xquad_en_path: str | None = None) -> list[dict]:
    """
    Parse XQuAD VI JSON (SQuAD format) into a list of parallel pairs.

    XQuAD shares QA IDs across languages. Since we only have xquad.vi.json,
    we use the VI file for both context and question (VI side), and use the
    EN SQuAD dev set to find matching EN context/question by ID.

    If xquad_en_path is None, we derive EN pairs from SQuAD dev-v2.0.json
    using the shared XQuAD question IDs (XQuAD is a subset of SQuAD dev).

    Returns:
        list of {
            "id":           str,
            "question_en":  str,
            "context_en":   str,
            "answer_en":    dict  — {"text": [...], "answer_start": [...]}
            "question_vi":  str,
            "context_vi":   str,
            "answer_vi":    dict
        }
    """
    # Load VI data
    with open(xquad_vi_path, "r", encoding="utf-8") as f:
        vi_data = json.load(f)

    vi_by_id = {}
    for article in vi_data["data"]:
        for para in article["paragraphs"]:
            ctx_vi = para["context"]
            for qa in para["qas"]:
                qid = qa["id"]
                if qa.get("answers") and len(qa["answers"]) > 0:
                    first = qa["answers"][0]
                    ans_vi = {
                        "text": [first["text"]],
                        "answer_start": [int(first["answer_start"])],
                    }
                else:
                    ans_vi = {"text": [], "answer_start": []}
                vi_by_id[qid] = {
                    "question_vi": qa["question"],
                    "context_vi": ctx_vi,
                    "answer_vi": ans_vi,
                }

    # Load EN data (SQuAD dev-v2.0 is the EN side of XQuAD)
    en_by_id = {}
    if xquad_en_path and os.path.exists(xquad_en_path):
        with open(xquad_en_path, "r", encoding="utf-8") as f:
            en_data = json.load(f)
        for article in en_data["data"]:
            for para in article["paragraphs"]:
                ctx_en = para["context"]
                for qa in para["qas"]:
                    qid = qa["id"]
                    if qid in vi_by_id:  # only XQuAD IDs
                        if qa.get("answers") and len(qa["answers"]) > 0:
                            first = qa["answers"][0]
                            ans_en = {
                                "text": [first["text"]],
                                "answer_start": [int(first["answer_start"])],
                            }
                        else:
                            ans_en = {"text": [], "answer_start": []}
                        en_by_id[qid] = {
                            "question_en": qa["question"],
                            "context_en": ctx_en,
                            "answer_en": ans_en,
                        }

    # Build aligned pairs (preserve order from VI file for deterministic split)
    pairs = []
    for article in vi_data["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                qid = qa["id"]
                if qid not in vi_by_id:
                    continue
                vi_info = vi_by_id[qid]
                if qid in en_by_id:
                    en_info = en_by_id[qid]
                else:
                    # Fallback: use EN question from VI file (XQuAD sometimes has EN q)
                    # or skip — only include pairs where we have both sides
                    continue
                pairs.append({
                    "id": qid,
                    **en_info,
                    **vi_info,
                })

    return pairs


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class XQuADDataset(Dataset):
    """
    Tokenizes XQuAD parallel pairs on-the-fly.

    Each item returns pre-tokenized tensors for EN and VI.
    Padding is handled by the collate_fn (pad to batch-max, not global max).

    Args:
        pairs     : list of parallel pair dicts (from load_xquad_pairs)
        tokenizer : HuggingFace fast tokenizer (XLM-R)
        max_length: max token length per sequence (default 384)
    """

    def __init__(self, pairs: list[dict], tokenizer, max_length: int = 384):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]

        # Tokenize EN (with answer span positions)
        en_ids, en_mask, en_start, en_end, en_q_end = process_qa_sample(
            question=pair["question_en"],
            context=pair["context_en"],
            answer=pair["answer_en"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            doc_stride=128,
        )

        # Tokenize VI (no answer positions — pseudo-labels come from γ)
        vi_ids, vi_mask, _, _, vi_q_end = process_qa_sample(
            question=pair["question_vi"],
            context=pair["context_vi"],
            answer=None,  # no VI ground-truth labels
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            doc_stride=128,
        )

        return {
            "en_input_ids":       en_ids,       # [max_length]
            "en_attention_mask":  en_mask,       # [max_length]
            "en_start_positions": en_start,      # scalar
            "en_end_positions":   en_end,        # scalar
            "en_question_end":    en_q_end,      # scalar — index of first [SEP]
            "vi_input_ids":       vi_ids,        # [max_length]
            "vi_attention_mask":  vi_mask,       # [max_length]
            "vi_question_end":    vi_q_end,      # scalar
        }


# ──────────────────────────────────────────────────────────────
# Collate fn: pad EN and VI independently to batch-max length
# ──────────────────────────────────────────────────────────────

def xquad_collate_fn(batch: list[dict]) -> dict:
    """
    Pad EN and VI sequences independently to their own batch-max length.
    L_en and L_vi will differ within a batch — that's expected.

    Padding token id = 1 (XLM-R pad token).
    """
    PAD_ID = 1  # XLM-R pad token id

    def _pad_field(tensors: list[torch.Tensor], pad_val: int) -> torch.Tensor:
        """Right-pad 1D tensors to the max length in the list."""
        max_len = max(t.size(0) for t in tensors)
        padded = []
        for t in tensors:
            diff = max_len - t.size(0)
            if diff > 0:
                t = torch.cat([t, torch.full((diff,), pad_val, dtype=t.dtype)])
            padded.append(t)
        return torch.stack(padded, dim=0)

    en_ids  = _pad_field([b["en_input_ids"]      for b in batch], PAD_ID)
    en_mask = _pad_field([b["en_attention_mask"]  for b in batch], 0)
    vi_ids  = _pad_field([b["vi_input_ids"]       for b in batch], PAD_ID)
    vi_mask = _pad_field([b["vi_attention_mask"]  for b in batch], 0)

    return {
        "en_input_ids":       en_ids,
        "en_attention_mask":  en_mask,
        "en_start_positions": torch.stack([b["en_start_positions"] for b in batch]),
        "en_end_positions":   torch.stack([b["en_end_positions"]   for b in batch]),
        "en_question_end":    torch.stack([b["en_question_end"]    for b in batch]),
        "en_is_answerable":   torch.ones(len(batch), dtype=torch.long), # XQuAD is 100% answerable
        "vi_input_ids":       vi_ids,
        "vi_attention_mask":  vi_mask,
        "vi_question_end":    torch.stack([b["vi_question_end"]    for b in batch]),
    }


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def create_xquad_dataloaders(
    root_dir: str,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 384,
    num_workers: int = 0,
    train_size: int = 1010,   # first 1010 pairs (85%)
) -> tuple[DataLoader, DataLoader, list[dict]]:
    """
    Build train and val DataLoaders from XQuAD VI data.

    Split is index-based (deterministic):
        train: pairs[:train_size]
        val:   pairs[train_size:]

    Args:
        root_dir   : project root directory
        tokenizer  : HuggingFace fast tokenizer
        batch_size : training batch size
        max_length : max token length (≤ 384 per spec)
        num_workers: DataLoader workers (0 = main process)
        train_size : number of training pairs (default 1010)

    Returns:
        train_loader : DataLoader (shuffled)
        val_loader   : DataLoader (fixed order)
        val_pairs    : raw val pair dicts (for string-level EM eval)
    """
    xquad_vi_path = os.path.join(root_dir, "dataset", "xquad.vi.json")
    xquad_en_path = os.path.join(root_dir, "dataset", "xquad.en.json")

    if not os.path.exists(xquad_vi_path):
        raise FileNotFoundError(f"XQuAD VI not found: {xquad_vi_path}")

    all_pairs = load_xquad_pairs(xquad_vi_path, xquad_en_path)

    if len(all_pairs) < train_size:
        raise ValueError(
            f"Expected at least {train_size} XQuAD pairs, got {len(all_pairs)}. "
            f"Check that xquad_en_path ({xquad_en_path}) contains matching EN entries."
        )

    train_pairs = all_pairs[:train_size]
    val_pairs   = all_pairs[train_size:]

    # Verify no overlap
    train_ids = {p["id"] for p in train_pairs}
    val_ids   = {p["id"] for p in val_pairs}
    assert len(train_ids & val_ids) == 0, "Data leakage: val IDs found in train split"

    train_ds = XQuADDataset(train_pairs, tokenizer, max_length=max_length)
    val_ds   = XQuADDataset(val_pairs,   tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,           # train: shuffled
        collate_fn=xquad_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,          # val: fixed order always
        collate_fn=xquad_collate_fn,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader, val_pairs
