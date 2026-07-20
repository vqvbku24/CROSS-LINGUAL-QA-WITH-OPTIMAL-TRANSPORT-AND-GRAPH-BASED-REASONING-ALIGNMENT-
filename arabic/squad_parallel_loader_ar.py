"""
squad_parallel_loader_ar.py
Parallel EN-AR dataloader for Stage 2 Arabic training.

ZIZOUArabic_Squad format: pandas-serialized JSON
  data['data'] = {str_index: python_dict_repr_string}
  Each article dict has keys: title, paragraphs (SQuAD format)
  IDs overlap ~98.8% with SQuAD2.0 train IDs.

Alignment strategy: ID-based (same as VI branch).
  - Load ZIZOUArabic_Squad articles, parse via ast.literal_eval
  - Build {id -> AR qa info} mapping
  - Align with SQuAD2.0 EN by shared ID
  - Only include pairs with both EN and AR sides
"""
import ast
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────

def load_zizou_arabic(ar_path: str) -> dict:
    """
    Load ZIZOUArabic_Squad JSON (pandas-serialized format).

    Returns:
        dict {id -> {'question': str, 'context': str, 'answer': dict}}
    """
    with open(ar_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    ar_dict = {}
    parse_errors = 0

    for idx_str, article_str in raw['data'].items():
        try:
            article = ast.literal_eval(article_str)
        except Exception as e:
            parse_errors += 1
            logger.warning(f"Parse error at index {idx_str}: {e}")
            continue

        for para in article.get('paragraphs', []):
            context = para['context']
            for qa in para.get('qas', []):
                qid = qa.get('id', '')
                if not qid:
                    continue
                answers = qa.get('answers', [])
                if answers and len(answers) > 0:
                    first = answers[0]
                    answer_dict = {
                        'text': [first['text']],
                        'answer_start': [int(first['answer_start'])],
                    }
                else:
                    answer_dict = {'text': [], 'answer_start': []}

                ar_dict[qid] = {
                    'id': qid,
                    'question': qa['question'],
                    'context': context,
                    'answer': answer_dict,
                }

    if parse_errors > 0:
        logger.warning(f"Total parse errors: {parse_errors}")

    logger.info(f"Loaded {len(ar_dict)} AR QA pairs from ZIZOUArabic_Squad")
    return ar_dict


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


def build_parallel_data(en_dict: dict, ar_dict: dict) -> list:
    """Align EN and AR by shared QA id. Returns list of {'en': ..., 'ar': ...}."""
    parallel_data = []
    for qid, ar_item in ar_dict.items():
        if qid in en_dict:
            parallel_data.append({
                'en': en_dict[qid],
                'ar': ar_item,
            })

    match_rate = len(parallel_data) / max(len(ar_dict), 1) * 100
    logger.info(
        f"ID alignment: {len(parallel_data)}/{len(ar_dict)} pairs aligned "
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

class SquadParallelDatasetAR(Dataset):
    """
    PyTorch Dataset for parallel EN-AR data.
    Mirrors SquadParallelDataset (VI) but with 'ar_*' keys.
    No AR ground-truth labels used during training (zero-shot transfer).
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
        ar_item = item['ar']

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

        # Tokenize AR branch (no answer labels — pseudo-labels come from γ)
        ar_question = ar_item.get('question', '')
        ar_context  = ar_item.get('context', '')
        ar_encoding = self.tokenizer(
            text=ar_question,
            text_pair=ar_context,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        ar_input_ids      = ar_encoding['input_ids'].squeeze(0)
        ar_attention_mask = ar_encoding['attention_mask'].squeeze(0)

        # Find first [SEP] position in AR (= question boundary)
        sep_id = self.tokenizer.sep_token_id

        def find_sep_idx(input_ids, sep_token_id):
            matches = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
            return matches[0].item() if len(matches) > 0 else 0

        ar_question_end = find_sep_idx(ar_input_ids, sep_id)

        return {
            'en_input_ids':      en_input_ids,
            'en_attention_mask': en_attention_mask,
            'en_start_position': en_start_position,
            'en_end_position':   en_end_position,
            'en_is_answerable':  en_is_answerable,
            'en_question_end':   en_question_end,
            'ar_input_ids':      ar_input_ids,
            'ar_attention_mask': ar_attention_mask,
            'ar_question_end':   torch.tensor(ar_question_end, dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def create_squad_parallel_dataloaders_ar(
    tokenizer,
    en_path: str,
    ar_path: str,
    batch_size: int = 32,
    max_length: int = 384,
    distributed: bool = False,
):
    """
    Build DataLoader from aligned EN-AR parallel data.

    Returns:
        (train_loader, dataset, sampler_or_None)
    """
    logger.info(f"Loading EN data from {en_path} ...")
    en_dict = load_squad_en(en_path)

    logger.info(f"Loading AR data from {ar_path} ...")
    ar_dict = load_zizou_arabic(ar_path)

    logger.info("Aligning EN-AR pairs by ID ...")
    parallel_data = build_parallel_data(en_dict, ar_dict)

    if len(parallel_data) == 0:
        raise RuntimeError(
            "No aligned EN-AR pairs found. "
            "Check dataset IDs — ZIZOUArabic_Squad should share IDs with SQuAD2.0."
        )

    dataset = SquadParallelDatasetAR(
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
    ar_path = os.path.join(ROOT, 'dataset', 'ZIZOUArabic_Squad', 'train.json')

    en_dict = load_squad_en(en_path)
    ar_dict = load_zizou_arabic(ar_path)
    parallel = build_parallel_data(en_dict, ar_dict)
    print(f"\n=== Smoke Test ===")
    print(f"Total aligned pairs: {len(parallel)}")
    if parallel:
        p = parallel[0]
        print(f"Sample EN question: {p['en']['question'][:80]}")
        print(f"Sample AR question: {p['ar']['question'][:80]}")
        print(f"EN answer: {p['en']['answer']}")
        print(f"AR answer: {p['ar']['answer']}")
    print("=== DONE ===")
