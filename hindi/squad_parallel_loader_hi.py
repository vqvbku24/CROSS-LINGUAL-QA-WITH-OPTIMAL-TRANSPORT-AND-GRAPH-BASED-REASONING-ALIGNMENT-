"""
squad_parallel_loader_hi.py
Parallel EN-HI dataloader for Stage 2 Hindi training.

IndicSQuAD format: flat JSON list
  Each entry is a dict with keys: id, title, context, question, answers
  answers = {'text': [...], 'answer_start': [...]}
  IDs match SQuAD2.0 train IDs at 100%.

Alignment strategy: ID-based (same as VI/AR branch).
  - Load IndicSQuAD entries, build {id -> HI qa info} mapping
  - Align with SQuAD2.0 EN by shared ID
  - Only include pairs with both EN and HI sides
"""
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────

def load_indic_squad_hindi(hi_path: str) -> dict:
    """
    Load IndicSQuAD Hindi JSON (flat list format).

    Returns:
        dict {id -> {'question': str, 'context': str, 'answer': dict}}
    """
    with open(hi_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    hi_dict = {}

    for entry in raw:
        qid = entry.get('id', '')
        if not qid:
            continue
        answers = entry.get('answers', {})
        texts   = answers.get('text', [])
        starts  = answers.get('answer_start', [])

        if texts and len(texts) > 0:
            answer_dict = {
                'text': [texts[0]],
                'answer_start': [int(starts[0])],
            }
        else:
            answer_dict = {'text': [], 'answer_start': []}

        hi_dict[qid] = {
            'id': qid,
            'question': entry['question'],
            'context': entry['context'],
            'answer': answer_dict,
        }

    logger.info(f"Loaded {len(hi_dict)} HI QA pairs from IndicSQuAD")
    return hi_dict


def load_squad_en(en_path: str) -> dict:
    """Load SQuAD2.0 EN training data. Returns {id -> en_qa_info}."""
    en_dict = {}
    with open(en_path, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)['data']

    for article in squad_data:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                answers = qa.get('answers', [])
                if answers and len(answers) > 0:
                    first = answers[0]
                    answer_dict = {
                        'text': [first['text']],
                        'answer_start': [int(first['answer_start'])],
                    }
                else:
                    answer_dict = {'text': [], 'answer_start': []}
                en_dict[qa['id']] = {
                    'id': qa['id'],
                    'question': qa['question'],
                    'context': context,
                    'answer': answer_dict,
                }

    logger.info(f"Loaded {len(en_dict)} EN QA pairs from SQuAD2.0")
    return en_dict


def build_parallel_data(en_dict: dict, hi_dict: dict) -> list:
    """Align EN and HI by shared QA id. Returns list of {'en': ..., 'hi': ...}."""
    parallel_data = []
    for qid, hi_item in hi_dict.items():
        if qid in en_dict:
            parallel_data.append({
                'en': en_dict[qid],
                'hi': hi_item,
            })

    match_rate = len(parallel_data) / max(len(hi_dict), 1) * 100
    logger.info(
        f"ID alignment: {len(parallel_data)}/{len(hi_dict)} pairs aligned "
        f"({match_rate:.1f}%)"
    )
    if match_rate < 80.0:
        logger.warning(
            f"ALERT: Match rate {match_rate:.1f}% is below 80% threshold! "
            "Check dataset integrity."
        )

    return parallel_data


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class SquadParallelDatasetHI(Dataset):
    """
    PyTorch Dataset for parallel EN-HI data.
    Mirrors SquadParallelDatasetAR (AR) but with 'hi_*' keys.
    No HI ground-truth labels used during training (zero-shot transfer).
    """

    def __init__(self, parallel_data, tokenizer, max_length=384):
        self.parallel_data = parallel_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.parallel_data)

    def __getitem__(self, idx):
        import sys, os
        # Import from parent project
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from phase1_dataloader.process_qa_sample import process_qa_sample

        item = self.parallel_data[idx]
        en_item = item['en']
        hi_item = item['hi']

        en_question = en_item.get('question', '')
        en_context  = en_item.get('context', '')
        en_answer   = en_item.get('answer')
        is_answerable = (
            en_answer is not None
            and len(en_answer.get('answer_start', [])) > 0
        )

        # Tokenize EN branch (with answer span positions)
        en_input_ids, en_attention_mask, en_start_position, en_end_position, en_question_end = (
            process_qa_sample(
                question=en_question,
                context=en_context,
                answer=en_answer if is_answerable else None,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                doc_stride=128,
            )
        )
        en_is_answerable = torch.tensor(1 if is_answerable else 0, dtype=torch.long)

        # Tokenize HI branch (no answer labels — pseudo-labels come from γ)
        hi_question = hi_item.get('question', '')
        hi_context  = hi_item.get('context', '')
        hi_encoding = self.tokenizer(
            text=hi_question,
            text_pair=hi_context,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        hi_input_ids      = hi_encoding['input_ids'].squeeze(0)
        hi_attention_mask = hi_encoding['attention_mask'].squeeze(0)

        # Find first [SEP] position in HI (= question boundary)
        sep_id = self.tokenizer.sep_token_id

        def find_sep_idx(input_ids, sep_token_id):
            matches = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
            return matches[0].item() if len(matches) > 0 else 0

        hi_question_end = find_sep_idx(hi_input_ids, sep_id)

        return {
            'en_input_ids':      en_input_ids,
            'en_attention_mask': en_attention_mask,
            'en_start_position': en_start_position,
            'en_end_position':   en_end_position,
            'en_is_answerable':  en_is_answerable,
            'en_question_end':   en_question_end,
            'hi_input_ids':      hi_input_ids,
            'hi_attention_mask': hi_attention_mask,
            'hi_question_end':   torch.tensor(hi_question_end, dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def create_squad_parallel_dataloaders_hi(
    tokenizer,
    en_path: str,
    hi_path: str,
    batch_size: int = 32,
    max_length: int = 384,
    distributed: bool = False,
):
    """
    Build DataLoader from aligned EN-HI parallel data.

    Returns:
        (train_loader, dataset, sampler_or_None)
    """
    logger.info(f"Loading EN data from {en_path} ...")
    en_dict = load_squad_en(en_path)

    logger.info(f"Loading HI data from {hi_path} ...")
    hi_dict = load_indic_squad_hindi(hi_path)

    logger.info("Aligning EN-HI pairs by ID ...")
    parallel_data = build_parallel_data(en_dict, hi_dict)

    if len(parallel_data) == 0:
        raise RuntimeError(
            "No aligned EN-HI pairs found. "
            "Check dataset IDs — IndicSQuAD should share IDs with SQuAD2.0."
        )

    dataset = SquadParallelDatasetHI(
        parallel_data=parallel_data,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=True)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, dataset, sampler


# ──────────────────────────────────────────────────────────────
# Smoke test (run directly)
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys, os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ROOT)

    en_path = os.path.join(ROOT, 'dataset', 'Squad2.0', 'train-v2.0.json')
    hi_path = os.path.join(ROOT, 'dataset', 'IndicSQuAD', 'train_hindi.json')

    en_dict = load_squad_en(en_path)
    hi_dict = load_indic_squad_hindi(hi_path)
    parallel = build_parallel_data(en_dict, hi_dict)
    print(f"\n=== Smoke Test ===")
    print(f"Total aligned pairs: {len(parallel)}")
    if parallel:
        p = parallel[0]
        print(f"Sample EN question: {p['en']['question'][:80]}")
        print(f"Sample HI question: {p['hi']['question'][:80]}")
        print(f"EN answer: {p['en']['answer']}")
        print(f"HI answer: {p['hi']['answer']}")
    print("=== DONE ===")
