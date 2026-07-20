# arabic/data/xquad_loader_ar.py
"""
XQuAD-AR Dataloader for Stage 2 Arabic Training.

XQuAD AR: parallel (EN, AR) QA pairs from xquad.ar.json.
  - If local file exists: use it directly.
  - Fallback: download from HuggingFace (xquad, 'ar' config).

Split:
  train: first 1010 pairs (85%) — Stage 2 training
  val:   remaining pairs  (15%) — early stopping eval only

No AR ground-truth labels used during training (zero-shot).
"""

import json
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from phase1_dataloader.process_qa_sample import process_qa_sample


# ──────────────────────────────────────────────────────────────
# Parse helpers
# ──────────────────────────────────────────────────────────────

def _parse_squad_json(path: str) -> dict:
    """Parse a SQuAD-format JSON. Returns {id -> qa_info}."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    by_id = {}
    for article in data['data']:
        for para in article['paragraphs']:
            ctx = para['context']
            for qa in para['qas']:
                qid = qa['id']
                answers = qa.get('answers', [])
                if answers:
                    first = answers[0]
                    ans = {'text': [first['text']], 'answer_start': [int(first['answer_start'])]}
                else:
                    ans = {'text': [], 'answer_start': []}
                by_id[qid] = {
                    'question': qa['question'],
                    'context': ctx,
                    'answer': ans,
                }
    return by_id


def _download_xquad_ar() -> list:
    """
    Download XQuAD-ar from HuggingFace and return as list of
    {'question_ar', 'context_ar', 'answer_ar'} dicts (id unused after download).
    """
    from datasets import load_dataset
    ds = load_dataset('xquad', 'xquad.ar', split='validation', trust_remote_code=True)
    rows = []
    for ex in ds:
        answers = ex.get('answers', {})
        texts = answers.get('text', [])
        starts = answers.get('answer_start', [])
        rows.append({
            'id': ex.get('id', ''),
            'question_ar': ex['question'],
            'context_ar': ex['context'],
            'answer_ar': {
                'text': list(texts),
                'answer_start': [int(s) for s in starts],
            },
        })
    return rows


def load_xquad_ar_pairs(root_dir: str) -> list:
    """
    Load XQuAD AR + EN pairs.

    Priority:
      1. dataset/xquad.ar.json (local)
      2. HuggingFace download

    Returns list of dicts with keys:
      id, question_en, context_en, answer_en,
           question_ar, context_ar, answer_ar
    """
    ar_path = os.path.join(root_dir, 'dataset', 'xquad.ar.json')
    en_path = os.path.join(root_dir, 'dataset', 'xquad.en.json')

    # ── AR side ──
    if os.path.exists(ar_path):
        ar_by_id = _parse_squad_json(ar_path)
        # Rename keys to ar_*
        ar_by_id = {
            k: {
                'question_ar': v['question'],
                'context_ar': v['context'],
                'answer_ar': v['answer'],
            }
            for k, v in ar_by_id.items()
        }
    else:
        print(f"[xquad_loader_ar] {ar_path} not found — downloading from HuggingFace ...")
        rows = _download_xquad_ar()
        ar_by_id = {}
        for r in rows:
            qid = r.pop('id', None) or str(len(ar_by_id))
            ar_by_id[qid] = {k: v for k, v in r.items()}
        # Save locally for next run
        os.makedirs(os.path.join(root_dir, 'dataset'), exist_ok=True)
        _save_hf_to_squad_json(ar_by_id, ar_path)
        print(f"[xquad_loader_ar] Saved to {ar_path}")

    # ── EN side ──
    en_by_id = {}
    if os.path.exists(en_path):
        raw = _parse_squad_json(en_path)
        en_by_id = {
            k: {
                'question_en': v['question'],
                'context_en': v['context'],
                'answer_en': v['answer'],
            }
            for k, v in raw.items()
        }

    # ── Merge ──
    pairs = []
    for qid, ar_info in ar_by_id.items():
        if qid in en_by_id:
            pairs.append({'id': qid, **en_by_id[qid], **ar_info})
        # If no EN side: skip (need EN for training branch)

    print(f"[xquad_loader_ar] Loaded {len(pairs)} EN-AR pairs")
    return pairs


def _save_hf_to_squad_json(ar_by_id: dict, save_path: str):
    """Save HF-downloaded AR data as SQuAD-format JSON for caching."""
    articles = []
    qas = []
    for qid, info in ar_by_id.items():
        answers = info['answer_ar']
        qas.append({
            'id': qid,
            'question': info['question_ar'],
            'answers': [
                {'text': answers['text'][i], 'answer_start': answers['answer_start'][i]}
                for i in range(len(answers.get('text', [])))
            ],
        })
    articles.append({'title': 'xquad_ar', 'paragraphs': [{'context': '', 'qas': qas}]})
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({'version': '1.0', 'data': articles}, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class XQuADDatasetAR(Dataset):
    """
    Tokenizes XQuAD EN-AR parallel pairs on-the-fly.
    Mirrors XQuADDataset (VI) but with 'ar_*' keys.
    """

    def __init__(self, pairs: list, tokenizer, max_length: int = 384):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]

        # EN branch: tokenize with answer span positions
        en_ids, en_mask, en_start, en_end, en_q_end = process_qa_sample(
            question=pair['question_en'],
            context=pair['context_en'],
            answer=pair['answer_en'],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            doc_stride=128,
        )

        # AR branch: no answer positions (pseudo-labels come from γ)
        ar_ids, ar_mask, _, _, ar_q_end = process_qa_sample(
            question=pair['question_ar'],
            context=pair['context_ar'],
            answer=None,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            doc_stride=128,
        )

        return {
            'en_input_ids':       en_ids,
            'en_attention_mask':  en_mask,
            'en_start_positions': en_start,
            'en_end_positions':   en_end,
            'en_question_end':    en_q_end,
            'ar_input_ids':       ar_ids,
            'ar_attention_mask':  ar_mask,
            'ar_question_end':    ar_q_end,
        }


# ──────────────────────────────────────────────────────────────
# Collate fn
# ──────────────────────────────────────────────────────────────

def xquad_ar_collate_fn(batch: list) -> dict:
    """Pad EN and AR independently to batch-max length."""
    PAD_ID = 1  # XLM-R pad token id

    def _pad(tensors, pad_val):
        max_len = max(t.size(0) for t in tensors)
        return torch.stack([
            torch.cat([t, torch.full((max_len - t.size(0),), pad_val, dtype=t.dtype)])
            if t.size(0) < max_len else t
            for t in tensors
        ])

    return {
        'en_input_ids':       _pad([b['en_input_ids']      for b in batch], PAD_ID),
        'en_attention_mask':  _pad([b['en_attention_mask']  for b in batch], 0),
        'en_start_positions': torch.stack([b['en_start_positions'] for b in batch]),
        'en_end_positions':   torch.stack([b['en_end_positions']   for b in batch]),
        'en_question_end':    torch.stack([b['en_question_end']    for b in batch]),
        'en_is_answerable':   torch.ones(len(batch), dtype=torch.long),  # XQuAD fully answerable
        'ar_input_ids':       _pad([b['ar_input_ids']       for b in batch], PAD_ID),
        'ar_attention_mask':  _pad([b['ar_attention_mask']   for b in batch], 0),
        'ar_question_end':    torch.stack([b['ar_question_end']    for b in batch]),
    }


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def create_xquad_ar_dataloaders(
    root_dir: str,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 384,
    num_workers: int = 0,
    train_size: int = 1010,
):
    """
    Build train and val DataLoaders from XQuAD AR data.

    Returns:
        train_loader, val_loader, val_pairs (raw dicts for string-level EM eval)
    """
    all_pairs = load_xquad_ar_pairs(root_dir)

    if len(all_pairs) < train_size:
        print(
            f"[WARN] Only {len(all_pairs)} XQuAD-AR pairs available "
            f"(expected >= {train_size}). Using all for val only."
        )
        train_pairs = []
        val_pairs = all_pairs
    else:
        train_pairs = all_pairs[:train_size]
        val_pairs   = all_pairs[train_size:]

    # Verify no overlap
    if train_pairs:
        train_ids = {p['id'] for p in train_pairs}
        val_ids   = {p['id'] for p in val_pairs}
        assert len(train_ids & val_ids) == 0, "Data leakage: val IDs found in train split"

    train_ds = XQuADDatasetAR(train_pairs, tokenizer, max_length=max_length)
    val_ds   = XQuADDatasetAR(val_pairs,   tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=xquad_ar_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=xquad_ar_collate_fn,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader, val_pairs


if __name__ == '__main__':
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pairs = load_xquad_ar_pairs(ROOT)
    print(f"Total XQuAD AR pairs: {len(pairs)}")
    if pairs:
        p = pairs[0]
        print(f"  EN question: {p['question_en'][:80]}")
        print(f"  AR question: {p['question_ar'][:80]}")
