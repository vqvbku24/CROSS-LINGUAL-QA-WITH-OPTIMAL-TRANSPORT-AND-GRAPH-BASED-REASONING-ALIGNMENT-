#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate EM and F1 on XQuAD-EN by answer categories:
  - Number
  - Date
  - Person
  - Location
  - Organization
  - Long answer (>5 tokens)
  - Short answer (≤2 tokens)

Supports:
  1. Evaluating pre-generated prediction JSON files (No PyTorch/GPU required).
  2. Loading a model checkpoint (Stage 1 / Stage 2) and running GPU inference.
"""

import os
import sys
import json
import re
import string
import argparse
import collections
from tqdm import tqdm

# Add project root to path to resolve local imports (e.g. phase2_model, phase1_dataloader)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())

# ─────────────────────────────────────────────────────────────
# 1. SQuAD JSON File Loader (Pure Python)
# ─────────────────────────────────────────────────────────────

def load_squad_data_pure(file_path: str) -> list[dict]:
    """Pure Python parser for SQuAD format files (does not import torch)."""
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    parsed_data = []
    for article in dataset["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                if qa.get("answers") and len(qa["answers"]) > 0:
                    first = qa["answers"][0]
                    answer_dict = {
                        "text": [first["text"]],
                        "answer_start": [int(first["answer_start"])],
                    }
                else:
                    answer_dict = {"text": [], "answer_start": []}

                parsed_data.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": context,
                    "answer": answer_dict,
                })
    return parsed_data

# ─────────────────────────────────────────────────────────────
# 2. Standard English SQuAD Metrics
# ─────────────────────────────────────────────────────────────

def _normalize_answer(s: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = " ".join(s.split())
    return s

def _exact_match_score(prediction: str, ground_truths: list[str]) -> bool:
    """Return True if normalized prediction matches any ground truth."""
    pred_norm = _normalize_answer(prediction)
    return any(_normalize_answer(gt) == pred_norm for gt in ground_truths)

def _f1_score(prediction: str, ground_truths: list[str]) -> float:
    """Calculate the max F1 score between prediction and ground truths."""
    def compute_f1(pred, gt):
        pred_toks = _normalize_answer(pred).split()
        gt_toks = _normalize_answer(gt).split()
        common = collections.Counter(pred_toks) & collections.Counter(gt_toks)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
            
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gt_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if not ground_truths:
        return 1.0 if not prediction else 0.0

    max_f1 = 0.0
    for gt in ground_truths:
        f1 = compute_f1(prediction, gt)
        if f1 > max_f1:
            max_f1 = f1
    return max_f1

# ─────────────────────────────────────────────────────────────
# 3. Answer Type Classification Heuristics (Rule-based)
# ─────────────────────────────────────────────────────────────

def clean_word(w):
    return re.sub(r'[^\w\s]', '', w).lower().strip()

def classify_sample(question: str, answer_text: str) -> list[str]:
    """
    Classify a question-answer pair into one or more categories based on
    heuristics on the question and the ground truth answer text.
    """
    q_lower = question.lower()
    ans_lower = answer_text.lower()
    
    ans_words = [clean_word(w) for w in answer_text.split() if clean_word(w)]
    num_words = len(ans_words)
    
    categories = []
    
    # 1. Length-based categories
    if num_words <= 2:
        categories.append("Short answer (≤2 tokens)")
    if num_words > 5:
        categories.append("Long answer (>5 tokens)")
        
    contains_digits = bool(re.search(r'\d', ans_lower))
    
    # 2. Date heuristics
    months = {
        'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    }
    date_words = {
        'century', 'centuries', 'decade', 'decades', 'year', 'years', 'month', 'months', 'week', 'weeks', 'day', 'days',
        'bc', 'ad', 'bce', 'ce', 'era', 'epoch', 'period', 'millennium', 'millennia', 'date', 'calendar',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
    }
    has_month = any(m in ans_words for m in months)
    has_date_word = any(dw in ans_words for dw in date_words)
    has_year = bool(re.search(r'\b(1\d{3}|20\d{2})\b', ans_lower))
    is_when_q = any(wq in q_lower for wq in ['when', 'what year', 'what date', 'which year', 'which month', 'how long ago'])
    
    is_date = False
    if is_when_q:
        is_date = True
    elif has_month or has_date_word:
        is_date = True
    elif has_year and not any(currency in ans_lower for currency in ['$', '€', '£', '¥', 'percent', '%']):
        is_date = True
    elif re.search(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', ans_lower):
        is_date = True
        
    if is_date:
        categories.append("Date")
        
    # 3. Number heuristics (excluding Date matches)
    number_words = {
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
        'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
        'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand',
        'million', 'billion', 'trillion', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth',
        'seventh', 'eighth', 'ninth', 'tenth', 'last', 'percent', 'percentage', 'half', 'quarter',
        'double', 'triple', 'dozen', 'score'
    }
    is_how_many_q = any(hm in q_lower for hm in ['how many', 'how much', 'how far', 'how tall', 'how wide', 'how deep', 'how heavy', 'how long'])
    
    is_number = False
    if not is_date:
        if contains_digits:
            is_number = True
        elif any(nw in ans_words for nw in number_words):
            is_number = True
        elif is_how_many_q:
            is_number = True
            
    if is_number:
        categories.append("Number")
        
    # 4. Person heuristics
    person_words = {
        'mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'lady', 'lord', 'king', 'queen', 'prince', 'princess',
        'emperor', 'empress', 'pope', 'bishop', 'st', 'saint', 'president', 'governor', 'mayor',
        'minister', 'general', 'admiral', 'colonel', 'captain', 'brother', 'sister', 'father', 'mother',
        'uncle', 'aunt', 'nephew', 'niece', 'cousin', 'grandp', 'grandm', 'son', 'daughter',
        'khan', 'tsar', 'czar', 'duke', 'duchess', 'baron', 'baroness', 'lordship'
    }
    is_who_q = any(wq in q_lower for wq in ['who', 'whom', 'whose'])
    has_capital = any(w[0].isupper() for w in answer_text.split() if w and w[0].isalpha())
    
    is_person = False
    if is_who_q:
        org_keywords = {
            'company', 'corporation', 'corp', 'association', 'organization', 'union', 'party', 'club', 'team',
            'department', 'ministry', 'agency', 'foundation', 'institute', 'university', 'college', 'school',
            'government', 'parliament', 'congress', 'senate', 'army', 'navy', 'police', 'force', 'alliance',
            'commission', 'board', 'museum', 'group'
        }
        if not any(ok in ans_words for ok in org_keywords):
            is_person = True
    elif any(pw in ans_words for pw in person_words):
        is_person = True
    elif has_capital and not is_date and not is_number:
        words = answer_text.split()
        if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words if w and w[0].isalpha()):
            loc_keywords = {
                'river', 'mountain', 'lake', 'ocean', 'sea', 'city', 'state', 'country', 'island', 'continent',
                'street', 'road', 'avenue', 'park', 'desert', 'valley', 'forest', 'station', 'airport', 'harbor'
            }
            org_keywords = {
                'company', 'corporation', 'corp', 'association', 'organization', 'union', 'party', 'club', 'team',
                'department', 'ministry', 'agency', 'foundation', 'institute', 'university', 'college', 'school',
                'government', 'parliament', 'congress', 'senate', 'army', 'navy', 'police', 'force', 'alliance',
                'commission', 'board', 'museum', 'group'
            }
            if not any(lk in ans_words for lk in loc_keywords) and not any(ok in ans_words for ok in org_keywords):
                is_person = True

    if is_person:
        categories.append("Person")
        
    # 5. Location heuristics
    loc_keywords = {
        'river', 'mountain', 'lake', 'ocean', 'sea', 'city', 'state', 'country', 'island', 'continent',
        'street', 'road', 'avenue', 'park', 'desert', 'valley', 'forest', 'station', 'airport', 'harbor',
        'port', 'capital', 'region', 'district', 'border', 'town', 'village', 'county', 'province',
        'republic', 'kingdom', 'empire', 'gulf', 'bay', 'canal', 'channel', 'mount', 'peak', 'range',
        'coast', 'shore', 'archipelago', 'peninsula', 'cape', 'hill', 'canyon', 'museum',
        'temple', 'church', 'cathedral', 'mosque', 'palace', 'castle', 'bridge', 'tower', 'hotel',
        'theatre', 'theater', 'building', 'house', 'square', 'garden', 'stadium', 'arena', 'hall',
        'monument', 'memorial', 'shrine', 'library', 'hospital', 'prison', 'jail', 'cemetery',
        'campus', 'office', 'headquarters', 'hq', 'site', 'location', 'place', 'area', 'zone',
        'north', 'south', 'east', 'west', 'northern', 'southern', 'eastern', 'western'
    }
    is_where_q = any(wq in q_lower for wq in ['where', 'which country', 'which city', 'which state', 'which place', 'what country', 'what city', 'what state', 'what island', 'what mountain', 'what river'])
    
    is_location = False
    if is_where_q:
        is_location = True
    elif any(lk in ans_words for lk in loc_keywords):
        is_location = True
    elif has_capital and not is_date and not is_number and not is_person:
        if q_lower.startswith('where') or 'located' in q_lower or 'situated' in q_lower:
            is_location = True

    if is_location:
        categories.append("Location")
        
    # 6. Organization heuristics
    org_keywords = {
        'company', 'corporation', 'corp', 'association', 'organization', 'union', 'party', 'club', 'team',
        'department', 'ministry', 'agency', 'foundation', 'institute', 'university', 'college', 'school',
        'government', 'parliament', 'congress', 'senate', 'army', 'navy', 'police', 'force', 'alliance',
        'commission', 'board', 'museum', 'group', 'ltd', 'limited', 'inc', 'incorporated', 'co', 'firm',
        'syndicate', 'consortium', 'league', 'federation', 'confederation', 'council', 'committee',
        'academy', 'bank', 'court', 'assembly', 'administration', 'bureau', 'authority', 'office',
        'band', 'dynasty'
    }
    is_which_org_q = any(wq in q_lower for wq in ['which company', 'which organization', 'which group', 'which team', 'which university', 'which school', 'which association', 'which agency', 'which department', 'which union',
                                                  'what company', 'what organization', 'what group', 'what team', 'what university', 'what school', 'what association', 'what agency', 'what department', 'what union'])
    
    is_org = False
    if is_which_org_q:
        is_org = True
    elif any(ok in ans_words for ok in org_keywords):
        is_org = True
    elif has_capital and not is_date and not is_number and not is_person and not is_location:
        is_org = True

    if is_org:
        categories.append("Organization")
        
    return categories

# ─────────────────────────────────────────────────────────────
# 4. Model Inference and Prediction Generation (Lazy Imports)
# ─────────────────────────────────────────────────────────────

class _InferenceDataset:
    """Lightweight dataset that tokenizes all samples upfront for DataLoader batching."""
    def __init__(self, data, tokenizer, max_length):
        self.samples = []  # list of (qid, input_ids, attn_mask, q_end_val)
        print(f"Pre-tokenizing {len(data)} samples...")
        for item in tqdm(data, desc="Tokenizing"):
            from phase1_dataloader.process_qa_sample import process_qa_sample
            input_ids, attn_mask, _, _, q_end = process_qa_sample(
                question=item["question"],
                context=item["context"],
                answer=None,
                tokenizer=tokenizer,
                max_length=max_length,
                doc_stride=128,
            )
            self.samples.append((item["id"], input_ids, attn_mask, q_end.item()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        qid, input_ids, attn_mask, q_end_val = self.samples[idx]
        return qid, input_ids, attn_mask, q_end_val


def _collate_fn(batch):
    """Custom collate: stack tensors, keep qids and q_end_vals as lists."""
    import torch
    qids, all_ids, all_masks, all_qends = zip(*batch)
    return (
        list(qids),
        torch.stack(all_ids),
        torch.stack(all_masks),
        list(all_qends),
    )


def run_model_inference(args, device_str):
    """Run model inference on XQuAD-EN using all available GPUs (DataParallel)."""
    # Lazy imports to support running dry-run or prediction mode without torch
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from phase2_model.model_core import CrossLingualOTModel
    from phase3_loss.losses import OTAlignmentLoss

    device = torch.device(device_str)
    print(f"Initializing tokenizer and model backbone: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = CrossLingualOTModel(model_name=args.model_name).to(device)
    criterion = OTAlignmentLoss(hidden_size=model.hidden_size).to(device)

    # 1. Load Stage 1 checkpoint if specified
    if args.stage1_ckpt:
        print(f"Loading Stage 1 base checkpoint from: {args.stage1_ckpt}")
        ckpt_stage1 = torch.load(args.stage1_ckpt, map_location=device)
        model.load_state_dict(ckpt_stage1.get("model_state", ckpt_stage1), strict=False)
        if "criterion_state" in ckpt_stage1 and ckpt_stage1["criterion_state"] is not None:
            criterion.load_state_dict(ckpt_stage1["criterion_state"], strict=False)

        print("Applying LoRA layers...")
        model.apply_lora()
        model.to(device)

    # 2. Load the main checkpoint
    print(f"Loading checkpoint weights from: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    if "criterion_state" in ckpt and ckpt["criterion_state"] is not None:
        criterion.load_state_dict(ckpt["criterion_state"], strict=False)
        print("Loaded QA Head weights successfully!")
    else:
        print("Warning: No criterion_state found in checkpoint.")

    model.eval()
    criterion.eval()

    # 3. Wrap with DataParallel if multiple GPUs are available
    num_gpus = torch.cuda.device_count() if device_str.startswith("cuda") else 0
    if num_gpus > 1:
        gpu_ids = list(range(num_gpus))
        print(f"Using DataParallel across {num_gpus} GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
        criterion = nn.DataParallel(criterion, device_ids=gpu_ids)
    else:
        print(f"Using single device: {device_str}")

    # 4. Build dataset and DataLoader
    print(f"Loading SQuAD/XQuAD-EN data from: {args.squad_file}")
    data = load_squad_data_pure(args.squad_file)
    print(f"Loaded {len(data)} samples for evaluation.")

    dataset = _InferenceDataset(data, tokenizer, args.max_length)
    batch_size = args.batch_size * max(1, num_gpus)  # scale batch with GPU count
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_fn,
    )
    print(f"Effective batch size: {batch_size} ({args.batch_size} per GPU x {max(1, num_gpus)} GPU(s))")

    predictions = {}
    MAX_ANSWER_LEN = 30

    with torch.no_grad():
        for qids, input_ids, attn_mask, q_end_vals in tqdm(loader, desc="Running Inference"):
            input_ids = input_ids.to(device)   # [B, L]
            attn_mask = attn_mask.to(device)   # [B, L]

            # Backbone forward pass
            out = model.module.backbone(input_ids, attn_mask) if isinstance(model, nn.DataParallel) else model.backbone(input_ids, attn_mask)
            target_layers = [6, 7, 8, 9]
            stacked = torch.stack([out.hidden_states[l] for l in target_layers], dim=0)
            raw_weights = model.module.layer_weights if isinstance(model, nn.DataParallel) else model.layer_weights
            weights = torch.softmax(raw_weights, dim=0).view(4, 1, 1, 1)
            hidden = (stacked * weights).sum(dim=0)  # [B, L, H]

            # Process each sample individually for span decoding (q_end varies per sample)
            for b_idx, (qid, q_end_val) in enumerate(zip(qids, q_end_vals)):
                h = hidden[b_idx].unsqueeze(0)           # [1, L, H]
                ids_b = input_ids[b_idx].unsqueeze(0)    # [1, L]
                mask_b = attn_mask[b_idx].unsqueeze(0)   # [1, L]

                # Question embeddings
                q_emb = h[:, :q_end_val, :]
                q_mask = torch.zeros(1, q_end_val, dtype=torch.bool, device=device)

                # QA head — use .module if wrapped
                qa_head = criterion.module.qa_head if isinstance(criterion, nn.DataParallel) else criterion.qa_head
                start_logits, end_logits, has_ans_logit = qa_head(h, q_emb, q_mask)

                # Mask padding and question tokens
                padding_mask = (mask_b[0] == 0)
                start_logits[0].masked_fill_(padding_mask, float('-inf'))
                end_logits[0].masked_fill_(padding_mask, float('-inf'))

                question_mask = torch.arange(start_logits.size(1), device=device) <= q_end_val
                question_mask[0] = False
                start_logits[0].masked_fill_(question_mask, float('-inf'))
                end_logits[0].masked_fill_(question_mask, float('-inf'))

                # Decode span
                start_idx = start_logits[0].argmax().item()
                end_logits_masked = end_logits[0].clone()
                end_logits_masked[:start_idx] = float('-inf')
                end_logits_masked[start_idx + MAX_ANSWER_LEN:] = float('-inf')
                end_idx = end_logits_masked.argmax().item()

                is_answerable_pred = has_ans_logit.item() > 0
                if start_idx == 0 or not is_answerable_pred:
                    pred_span = ""
                else:
                    pred_ids = ids_b[0][start_idx: end_idx + 1]
                    pred_span = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

                predictions[qid] = pred_span

    if args.save_preds:
        pred_out_file = args.save_preds
        with open(pred_out_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)
        print(f"Predictions saved to {pred_out_file}")

    return predictions

# ─────────────────────────────────────────────────────────────
# 5. Evaluation and Tabulation
# ─────────────────────────────────────────────────────────────

def evaluate_pred_dict(predictions, squad_data):
    """
    Evaluates predictions dictionary against squad_data.
    Classifies each sample into categories and aggregates EM/F1 metrics.
    """
    categories_def = [
        "Number",
        "Date",
        "Person",
        "Location",
        "Organization",
        "Long answer (>5 tokens)",
        "Short answer (≤2 tokens)",
        "All"
    ]
    
    cat_metrics = {cat: {"em": 0.0, "f1": 0.0, "count": 0} for cat in categories_def}

    for item in squad_data:
        qid = item["id"]
        question = item["question"]
        ground_truths = item["answer"].get("text", [])
        
        first_gt = ground_truths[0] if ground_truths else ""
        
        sample_cats = classify_sample(question, first_gt)
        sample_cats.append("All")
        
        pred = predictions.get(qid, "")
        
        em = 100.0 if _exact_match_score(pred, ground_truths) else 0.0
        f1 = 100.0 * _f1_score(pred, ground_truths)
        
        for cat in sample_cats:
            if cat in cat_metrics:
                cat_metrics[cat]["em"] += em
                cat_metrics[cat]["f1"] += f1
                cat_metrics[cat]["count"] += 1

    results = {}
    for cat, stats in cat_metrics.items():
        count = stats["count"]
        if count > 0:
            results[cat] = {
                "em": stats["em"] / count,
                "f1": stats["f1"] / count,
                "count": count
            }
        else:
            results[cat] = {
                "em": 0.0,
                "f1": 0.0,
                "count": 0
            }
            
    return results

def print_comparison_table(results_dict, squad_total):
    """
    Prints a comparison table.
    results_dict: { model_name: { category_name: { "em": val, "f1": val, "count": val } } }
    """
    model_names = list(results_dict.keys())
    categories = [
        "Number",
        "Date",
        "Person",
        "Location",
        "Organization",
        "Long answer (>5 tokens)",
        "Short answer (≤2 tokens)",
        "All"
    ]

    print("\n" + "=" * 80)
    print(f"📊 XQuAD-EN EVALUATION REPORT BY ANSWER TYPE (Total samples: {squad_total})")
    print("=" * 80)

    header = f"║ {'Answer Type':<26} ║"
    for name in model_names:
        header += f" {name:<20} ║"
    print(header)
    
    sub_header = f"║ {'':<26} ║"
    for _ in model_names:
        sub_header += f" {'EM %':<8} | {'F1 %':<8} ║"
    print(sub_header)
    print("╠" + "═" * 28 + "╬" + "═" * 22 * len(model_names) + "╣")

    for cat in categories:
        count = 0
        for name in model_names:
            if cat in results_dict[name]:
                count = results_dict[name][cat]["count"]
                break
        
        row = f"║ {f'{cat} ({count})':<26} ║"
        for name in model_names:
            if cat in results_dict[name]:
                em = results_dict[name][cat]["em"]
                f1 = results_dict[name][cat]["f1"]
                row += f" {em:>7.2f} | {f1:>7.2f} ║"
            else:
                row += f" {'N/A':>7} | {'N/A':>7} ║"
        print(row)

    print("=" * 80 + "\n")

# ─────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate XQuAD-EN answers by category type")
    
    # Mode A: Model Checkpoint evaluation
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint model file")
    parser.add_argument("--stage1_ckpt", type=str, default=None, help="Path to Stage 1 base checkpoint")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Pretrained model name")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum token length")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size per GPU")
    parser.add_argument("--save_preds", type=str, default=None, help="Path to save predictions to JSON")
    
    # Mode B: Pre-computed Prediction Files comparison
    parser.add_argument("--pred_files", type=str, nargs="+", default=None, help="Space-separated paths to predictions JSON files")
    parser.add_argument("--names", type=str, nargs="+", default=None, help="Names for prediction files in the comparison table")
    
    # General Args
    parser.add_argument("--squad_file", type=str, default="dataset/xquad.en.json", help="Path to XQuAD English file")
    parser.add_argument("--dry_run", action="store_true", help="Print dataset answer category distribution without model/predictions evaluation")
    
    args = parser.parse_args()

    if not os.path.exists(args.squad_file):
        print(f"Error: Dataset file not found at {args.squad_file}")
        sys.exit(1)

    print(f"Loading ground truth dataset from: {args.squad_file}")
    squad_data = load_squad_data_pure(args.squad_file)
    print(f"Successfully loaded {len(squad_data)} question-answer pairs.")

    # 1. Dry Run Mode
    if args.dry_run:
        print("\n=== DRY RUN: ANSWER CATEGORIES DISTRIBUTION ===")
        counts = collections.Counter()
        example_samples = collections.defaultdict(list)
        
        for item in squad_data:
            gt = item["answer"].get("text", [None])[0]
            cats = classify_sample(item["question"], gt) if gt else ["Unanswerable"]
            counts["All"] += 1
            for cat in cats:
                counts[cat] += 1
                if len(example_samples[cat]) < 3:
                    example_samples[cat].append((item["question"], gt))
                    
        for cat in sorted(counts.keys()):
            pct = (counts[cat] / len(squad_data)) * 100
            print(f"- {cat:<26}: {counts[cat]:>4} samples ({pct:>5.1f}%)")
            if cat != "All" and cat in example_samples:
                print("  Examples:")
                for q, a in example_samples[cat]:
                    print(f"    Q: {q}")
                    print(f"    A: {a}")
        sys.exit(0)

    # 2. Mode B: Evaluate Predictions File(s)
    if args.pred_files is not None:
        if args.names and len(args.names) != len(args.pred_files):
            print("Error: Length of --names must match length of --pred_files")
            sys.exit(1)
            
        results_dict = {}
        for idx, pred_file in enumerate(args.pred_files):
            name = args.names[idx] if args.names else os.path.basename(pred_file)
            print(f"Reading predictions from: {pred_file}")
            with open(pred_file, "r", encoding="utf-8") as f:
                predictions = json.load(f)
            
            if isinstance(predictions, list):
                predictions = {item["id"]: item.get("answer", "") for item in predictions}
            elif isinstance(predictions, dict):
                first_val = next(iter(predictions.values()))
                if isinstance(first_val, dict):
                    predictions = {qid: item.get("answer", "") for qid, item in predictions.items()}
            
            results_dict[name] = evaluate_pred_dict(predictions, squad_data)
            
        print_comparison_table(results_dict, len(squad_data))
        sys.exit(0)

    # 3. Mode A: Checkpoint Evaluation Mode
    if args.ckpt is None:
        print("Error: Please provide either --pred_files OR --ckpt for model evaluation.")
        parser.print_help()
        sys.exit(1)

    # We only check for GPU/Torch inside run_model_inference
    import torch
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(f"Detected {num_gpus} GPU(s):")
        for i, name in enumerate(gpu_names):
            print(f"  cuda:{i} — {name}")
        device_str = "cuda"
    else:
        print("No GPU detected. Running on CPU.")
        device_str = "cpu"

    predictions = run_model_inference(args, device_str)
    
    results = evaluate_pred_dict(predictions, squad_data)
    
    model_name_label = os.path.basename(args.ckpt)
    results_dict = {model_name_label: results}
    
    print_comparison_table(results_dict, len(squad_data))

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
