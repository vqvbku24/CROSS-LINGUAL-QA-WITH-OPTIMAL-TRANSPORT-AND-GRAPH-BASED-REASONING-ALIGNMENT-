# arabic/phase4_evaluation/quick_eval_ar.py
"""
Quick EM and F1 evaluation for Arabic (XQuAD-ar and MLQA-ar).

Mirrors quick_eval.py (VI) but:
  - Adds Arabic-specific _normalize_answer_ar() handling:
      * Remove tashkeel (diacritics): harakat characters U+064B..U+065F + tatweel U+0640
      * Remove al-ta'rif prefix "ال" if it would reduce to shorter form
      * Standard: lowercase, strip punctuation, collapse whitespace
  - quick_em_f1_xquad_ar(): eval on XQuAD val_pairs (ar_* keys)
  - quick_em_f1_mlqa_ar():  eval on MLQA JSON (test-context-ar-question-ar.json)
"""

import torch
import os
import sys
import collections
import re
import string

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from phase1_dataloader.process_qa_sample import process_qa_sample
from gpu_utils import get_model


# ──────────────────────────────────────────────────────────────
# Arabic-aware normalization (Step 3 of spec.md)
# ──────────────────────────────────────────────────────────────

# Arabic diacritics (tashkeel): U+064B (fathatan) to U+065F (superscript alef)
# plus U+0640 (tatweel/kashida)
_ARABIC_DIACRITICS = re.compile(r'[\u064B-\u065F\u0640]')

# Common Arabic definite article forms
_AL_PREFIX = re.compile(r'^(ال|ال)')

# Punctuation table (same as English branch)
_PUNCT_TABLE = str.maketrans('', '', string.punctuation + '،؛؟٪')  # also Arabic punct


def _normalize_answer_ar(s: str) -> str:
    """
    Arabic-specific normalization:
    1. Remove tashkeel (diacritics / harakat)
    2. Remove tatweel (elongation mark ـ)
    3. Normalize alef variants (أ إ آ ٱ → ا)
    4. Normalize ta marbuta (ة → ه)  — common Arabic eval normalization
    5. Remove definite article "ال" prefix from each token
    6. Strip Arabic + Latin punctuation, lowercase, collapse whitespace
    """
    # 1-2. Remove tashkeel + tatweel
    s = _ARABIC_DIACRITICS.sub('', s)

    # 3. Normalize alef variants
    s = re.sub(r'[أإآٱ]', 'ا', s)

    # 4. Normalize ta marbuta
    s = s.replace('ة', 'ه')

    # 5. Remove "ال" prefix word-by-word
    tokens = s.split()
    tokens = [_AL_PREFIX.sub('', t) for t in tokens]
    s = ' '.join(tokens)

    # 6. Strip punctuation, lowercase, collapse whitespace
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = ' '.join(s.split())
    return s


# ──────────────────────────────────────────────────────────────
# Generic EM / F1 helpers (same logic as quick_eval.py)
# ──────────────────────────────────────────────────────────────

def _exact_match_score_ar(prediction: str, ground_truths: list) -> bool:
    pred_norm = _normalize_answer_ar(prediction)
    return any(_normalize_answer_ar(gt) == pred_norm for gt in ground_truths)


def _f1_score_ar(prediction: str, ground_truths: list) -> float:
    def compute_f1(pred, gt):
        pred_toks = _normalize_answer_ar(pred).split()
        gt_toks   = _normalize_answer_ar(gt).split()
        common    = collections.Counter(pred_toks) & collections.Counter(gt_toks)
        num_same  = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_toks)
        recall    = num_same / len(gt_toks)
        return (2 * precision * recall) / (precision + recall)

    if not ground_truths:
        return 1.0 if not prediction else 0.0
    return max(compute_f1(prediction, gt) for gt in ground_truths)


# ──────────────────────────────────────────────────────────────
# Shared forward pass helper (to avoid duplication)
# ──────────────────────────────────────────────────────────────

def _run_inference(model, criterion, tokenizer, question, context, device, max_length=384, max_answer_len=30):
    """Run one QA example through the model. Returns predicted span string."""
    input_ids, attn_mask, _, _, q_end = process_qa_sample(
        question=question,
        context=context,
        answer=None,
        tokenizer=tokenizer,
        max_length=max_length,
        doc_stride=128,
    )
    input_ids = input_ids.unsqueeze(0).to(device)
    attn_mask = attn_mask.unsqueeze(0).to(device)
    q_end_val = q_end.item()

    base_model = get_model(model)
    out = base_model.backbone(input_ids, attn_mask)
    target_layers = [6, 7, 8, 9]
    stacked = torch.stack([out.hidden_states[i] for i in target_layers], dim=0)
    weights = torch.softmax(base_model.layer_weights, dim=0).view(4, 1, 1, 1)
    hidden  = (stacked * weights).sum(dim=0)

    q_emb  = hidden[:, :q_end_val, :]
    q_mask = torch.zeros(1, q_end_val, dtype=torch.bool, device=device)

    start_logits, end_logits, has_ans_logit = criterion.qa_head(hidden, q_emb, q_mask)

    # Mask padding
    padding_mask = (attn_mask[0] == 0)
    start_logits[0].masked_fill_(padding_mask, float('-inf'))
    end_logits[0].masked_fill_(padding_mask, float('-inf'))

    # Mask question tokens (unmask [CLS] for unanswerable)
    question_mask = torch.arange(start_logits.size(1), device=device) <= q_end_val
    question_mask[0] = False
    start_logits[0].masked_fill_(question_mask, float('-inf'))
    end_logits[0].masked_fill_(question_mask, float('-inf'))

    start_idx = start_logits[0].argmax().item()
    end_logits_masked = end_logits[0].clone()
    end_logits_masked[:start_idx] = float('-inf')
    end_logits_masked[start_idx + max_answer_len:] = float('-inf')
    end_idx = end_logits_masked.argmax().item()

    pred_ids  = input_ids[0][start_idx: end_idx + 1]
    pred_span = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
    is_answerable_pred = has_ans_logit.item() > 0
    return pred_span, is_answerable_pred


# ──────────────────────────────────────────────────────────────
# XQuAD-ar evaluation
# ──────────────────────────────────────────────────────────────

def quick_em_f1_xquad_ar(model, criterion, tokenizer, val_pairs, device, max_length=384, max_answer_len=30):
    """
    Exact-match and F1 on XQuAD-ar val split.

    val_pairs: list of dicts with keys question_ar, context_ar, answer_ar
    Returns: (em_score, f1_score) as percentages
    """
    model.eval()
    criterion.eval()
    correct = 0
    total_f1 = 0.0
    total = 0

    with torch.no_grad():
        for pair in val_pairs:
            question_ar  = pair['question_ar']
            context_ar   = pair['context_ar']
            ground_truths = pair['answer_ar'].get('text', [])

            pred_span, is_answerable_pred = _run_inference(
                model, criterion, tokenizer,
                question=question_ar, context=context_ar,
                device=device, max_length=max_length, max_answer_len=max_answer_len,
            )

            total += 1
            f1_current = 0.0
            is_correct = False
            if len(ground_truths) == 0:
                is_correct = not is_answerable_pred
                f1_current = 1.0 if not is_answerable_pred else 0.0
            else:
                if not is_answerable_pred:
                    is_correct = False
                    f1_current = 0.0
                else:
                    is_correct = _exact_match_score_ar(pred_span, ground_truths)
                    f1_current = _f1_score_ar(pred_span, ground_truths)
                    
            if is_correct:
                correct += 1
            total_f1 += f1_current

    em = (correct / total * 100) if total > 0 else 0.0
    f1 = (total_f1 / total * 100) if total > 0 else 0.0
    return em, f1


# ──────────────────────────────────────────────────────────────
# MLQA-ar evaluation (full JSON file)
# ──────────────────────────────────────────────────────────────

def quick_em_f1_mlqa_ar(model, criterion, tokenizer, mlqa_ar_path, device, max_length=384, max_answer_len=30, n_samples=-1):
    """
    Exact-match and F1 on MLQA-ar (test-context-ar-question-ar.json).

    n_samples: if > 0, evaluate on first n_samples only (for quick dev check).
    Returns: (em_score, f1_score) as percentages
    """
    import json

    with open(mlqa_ar_path, 'r', encoding='utf-8') as f:
        mlqa_data = json.load(f)['data']

    model.eval()
    criterion.eval()
    correct = 0
    total_f1 = 0.0
    total = 0

    with torch.no_grad():
        for article in mlqa_data:
            if n_samples > 0 and total >= n_samples:
                break
            for para in article['paragraphs']:
                if n_samples > 0 and total >= n_samples:
                    break
                context = para['context']
                for qa in para['qas']:
                    if n_samples > 0 and total >= n_samples:
                        break
                    question      = qa['question']
                    ground_truths = [a['text'] for a in qa.get('answers', [])]

                    pred_span, is_answerable_pred = _run_inference(
                        model, criterion, tokenizer,
                        question=question, context=context,
                        device=device, max_length=max_length, max_answer_len=max_answer_len,
                    )

                    total += 1
                    f1_current = 0.0
                    is_correct = False
                    if len(ground_truths) == 0:
                        is_correct = not is_answerable_pred
                        f1_current = 1.0 if not is_answerable_pred else 0.0
                    else:
                        if not is_answerable_pred:
                            is_correct = False
                            f1_current = 0.0
                        else:
                            is_correct = _exact_match_score_ar(pred_span, ground_truths)
                            f1_current = _f1_score_ar(pred_span, ground_truths)
                            
                    if is_correct:
                        correct += 1
                    total_f1 += f1_current

                    if total % 500 == 0:
                        print(f"  MLQA-ar: {total} samples | EM={correct/total*100:.2f}% | F1={total_f1/total*100:.2f}%")

    em = (correct / total * 100) if total > 0 else 0.0
    f1 = (total_f1 / total * 100) if total > 0 else 0.0
    return em, f1


if __name__ == "__main__":
    import argparse
    from phase1_dataloader.process_qa_sample import load_squad_data
    from phase2_model.model_core import CrossLingualOTModel
    from phase3_loss.losses import OTAlignmentLoss
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Quick Evaluation Runner for Arabic")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (Stage 1 or Stage 2)")
    parser.add_argument("--stage1_ckpt", type=str, default=None, help="Path to Stage 1 base checkpoint")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to evaluation JSON file")
    parser.add_argument("--n_samples", type=int, default=0, help="Number of samples to evaluate (0 for all)")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Model name")
    parser.add_argument("--max_length", type=int, default=384, help="Max length")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = CrossLingualOTModel(model_name=args.model_name).to(device)
    criterion = OTAlignmentLoss(hidden_size=model.hidden_size).to(device)

    print(f"Loading checkpoint from: {args.ckpt}")
    
    if args.stage1_ckpt:
        print(f"Loading Stage 1 base from: {args.stage1_ckpt}")
        ckpt_stage1 = torch.load(args.stage1_ckpt, map_location=device)
        if "model_state" in ckpt_stage1:
            model.load_state_dict(ckpt_stage1["model_state"], strict=False)
            if "criterion_state" in ckpt_stage1 and ckpt_stage1["criterion_state"] is not None:
                criterion.load_state_dict(ckpt_stage1["criterion_state"], strict=False)
        else:
            model.load_state_dict(ckpt_stage1, strict=False)
        
        print("Applying LoRA to model...")
        model.apply_lora()
        model.to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
        if "criterion_state" in ckpt and ckpt["criterion_state"] is not None:
            criterion.load_state_dict(ckpt["criterion_state"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    print("Checkpoint loaded.")

    print(f"Loading data from {args.eval_file}...")
    data = load_squad_data(args.eval_file)
    if args.n_samples > 0:
        data = data[:args.n_samples]
    print(f"Loaded {len(data)} samples for evaluation.")

    model.eval()
    criterion.eval()
    correct = 0
    total_f1 = 0.0
    total = 0

    with torch.no_grad():
        for i, item in enumerate(data):
            question = item["question"]
            context = item["context"]
            ground_truths = item["answer"].get("text", [])

            pred_span, is_answerable_pred = _run_inference(
                model, criterion, tokenizer,
                question=question, context=context,
                device=device, max_length=args.max_length, max_answer_len=30
            )

            total += 1
            f1_current = 0.0
            is_correct = False
            if len(ground_truths) == 0:
                is_correct = not is_answerable_pred
                f1_current = 1.0 if not is_answerable_pred else 0.0
            else:
                if not is_answerable_pred:
                    is_correct = False
                    f1_current = 0.0
                else:
                    is_correct = _exact_match_score_ar(pred_span, ground_truths)
                    f1_current = _f1_score_ar(pred_span, ground_truths)
                    
            if is_correct:
                correct += 1
            total_f1 += f1_current

            if (i + 1) % 100 == 0 or (i + 1) == len(data):
                print(f"Processed {i+1}/{len(data)} | Current EM: {correct/total*100:.2f}% | Current F1: {total_f1/total*100:.2f}%")

    final_em = (correct / total * 100) if total > 0 else 0.0
    final_f1 = (total_f1 / total * 100) if total > 0 else 0.0
    
    print(f"\n========================================")
    print(f"Evaluation Complete!")
    print(f"File: {args.eval_file}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Samples: {total}")
    print(f"Exact Match (EM): {final_em:.2f}%")
    print(f"F1 Score: {final_f1:.2f}%")
    print(f"========================================")
    print(f"========================================")
