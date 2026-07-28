"""
madx_baseline.py
================
MAD-X zero-shot cross-lingual QA baseline (Pfeiffer et al., 2020).

Pipeline
--------
1. Load xlm-roberta-base + pre-trained English QA adapter from AdapterHub
   (task: squad1) + language adapters for en / ar / hi / vi.
2. Evaluate each target language on:
     - MLQA  : test-context-{lang}-question-{lang}.json
     - XQuAD : xquad.{lang}.json   (ar, hi, vi)
3. Report F1 / EM using the same evaluation logic as mlqa_evaluation_v1.py.

Requirements
------------
    pip install "adapters>=0.2.0" transformers accelerate

Usage
-----
    python madx_baseline.py \
        --mlqa_dir  dataset/MLQA \
        --xquad_dir dataset \
        --output    madx_results.json
"""

import argparse
import json
import os
import sys
import re
import string
import unicodedata
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoConfig
from adapters import AutoAdapterModel, AdapterConfig
from adapters.composition import Stack


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers  (mirrors mlqa_evaluation_v1.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

PUNCT = {chr(i) for i in range(sys.maxunicode)
         if unicodedata.category(chr(i)).startswith('P')}.union(string.punctuation)
WHITESPACE_LANGS = ['en', 'es', 'hi', 'vi', 'de', 'ar']
MIXED_SEGMENTATION_LANGS = ['zh']


def whitespace_tokenize(text):
    return text.split()


def mixed_segmentation(text):
    segs_out = []
    temp_str = ""
    for char in text:
        if re.search(r'[\u4e00-\u9fa5]', char) or char in PUNCT:
            if temp_str:
                segs_out.extend(whitespace_tokenize(temp_str))
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char
    if temp_str:
        segs_out.extend(whitespace_tokenize(temp_str))
    return segs_out


def normalize_answer(s, lang):
    def remove_articles(text, lang):
        if lang == 'en':
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        elif lang == 'es':
            return re.sub(r'\b(un|una|unos|unas|el|la|los|las)\b', ' ', text)
        elif lang == 'hi':
            return text
        elif lang == 'vi':
            return re.sub(r'\b(cua|la|cai|chiec|nhung)\b', ' ', text)
        elif lang == 'de':
            return re.sub(r'\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b', ' ', text)
        elif lang == 'ar':
            return re.sub(r'\sal^|al', ' ', text)
        elif lang == 'zh':
            return text
        else:
            raise Exception(f'Unknown Language {lang}')

    def white_space_fix(text, lang):
        if lang in WHITESPACE_LANGS:
            tokens = whitespace_tokenize(text)
        elif lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
        else:
            raise Exception(f'Unknown Language {lang}')
        return ' '.join(t for t in tokens if t.strip())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in PUNCT)

    return white_space_fix(remove_articles(remove_punc(s.lower()), lang), lang)


def f1_score(prediction, ground_truth, lang):
    pred_toks = normalize_answer(prediction, lang).split()
    gt_toks   = normalize_answer(ground_truth, lang).split()
    common    = Counter(pred_toks) & Counter(gt_toks)
    num_same  = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall    = num_same / len(gt_toks)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction, ground_truth, lang):
    return float(normalize_answer(prediction, lang) == normalize_answer(ground_truth, lang))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, lang):
    return max(metric_fn(prediction, gt, lang) for gt in ground_truths)


def evaluate_squad_format(dataset_json, predictions, lang):
    """Evaluate predictions dict {id->str} against a SQuAD-format dataset."""
    f1 = em = total = 0
    for article in dataset_json['data']:
        for para in article['paragraphs']:
            for qa in para['qas']:
                total += 1
                qid = qa['id']
                ground_truths = [a['text'] for a in qa['answers']]
                pred = predictions.get(qid, "")
                em += metric_max_over_ground_truths(exact_match_score, pred, ground_truths, lang)
                f1 += metric_max_over_ground_truths(f1_score,          pred, ground_truths, lang)
    return {
        'f1':          round(100.0 * f1 / total, 2),
        'exact_match': round(100.0 * em / total, 2),
        'total':       total,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

LANG_ADAPTER_IDS = {
    'en': 'en/wiki@ukp',
    'ar': 'ar/wiki@ukp',
    'hi': 'hi/wiki@ukp',
    'vi': 'vi/wiki@ukp',
}

QA_TASK_ADAPTER_ID = 'AdapterHub/m2qa-xlm-roberta-base-mad-x-2-qa-head'


def build_madx_model(device):
    """
    Build XLM-R + MAD-X adapters:
      - Language adapters: en, ar, hi, vi  (seq_bn, reduction_factor=2)
      - Task adapter    : squad1 QA head   (trained on English SQuAD v1)
    Returns (model, tokenizer).
    """
    print("Loading xlm-roberta-base ...")
    config    = AutoConfig.from_pretrained('xlm-roberta-base')
    model     = AutoAdapterModel.from_pretrained('xlm-roberta-base', config=config)
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    lang_cfg = AdapterConfig.load('seq_bn', reduction_factor=2)

    print("Loading language adapters from AdapterHub ...")
    for lang, adapter_id in LANG_ADAPTER_IDS.items():
        print(f"  {lang}: {adapter_id}")
        model.load_adapter(adapter_id, config=lang_cfg, load_as=lang)

    print(f"Loading QA task adapter: {QA_TASK_ADAPTER_ID}")
    model.load_adapter(QA_TASK_ADAPTER_ID, load_as='qa_squad')

    model.eval()
    model.to(device)
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Span extraction
# ──────────────────────────────────────────────────────────────────────────────

MAX_ANSWER_LEN = 30


def extract_span(start_logits, end_logits, input_ids, tokenizer, q_end_idx):
    """
    Given start/end logit vectors, extract the best answer span.
    Masks question tokens, [CLS], and padding.
    """
    seq_len = start_logits.size(0)
    device  = start_logits.device

    # Positions to suppress: question area + [CLS] + padding (id=1 for XLM-R)
    pos_idx      = torch.arange(seq_len, device=device)
    question_pos = (pos_idx <= q_end_idx)  # includes [CLS]
    pad_pos      = (input_ids == 1)
    mask         = question_pos | pad_pos

    s = start_logits.masked_fill(mask, float('-inf'))
    e = end_logits.masked_fill(mask, float('-inf'))

    start_idx = s.argmax().item()

    e_copy = e.clone()
    e_copy[:start_idx]                    = float('-inf')
    e_copy[start_idx + MAX_ANSWER_LEN:]   = float('-inf')
    end_idx = e_copy.argmax().item()

    span_ids = input_ids[start_idx: end_idx + 1]
    return tokenizer.decode(span_ids, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def predict_on_dataset(model, tokenizer, dataset_json, tgt_lang, device,
                       max_length=384, stride=128, verbose=True):
    """
    Run MAD-X inference over a SQuAD-format dataset JSON.
    Sets active_adapters = Stack(tgt_lang, 'qa_squad').
    Returns dict {qid -> predicted_answer_str}.
    """
    model.active_adapters = Stack(tgt_lang, 'qa_squad')
    predictions = {}
    total = 0

    with torch.no_grad():
        for article in dataset_json['data']:
            for para in article['paragraphs']:
                context = para['context']
                for qa in para['qas']:
                    qid      = qa['id']
                    question = qa['question']

                    # Sliding-window tokenisation (same as generate_mlqa_preds.py)
                    inputs = tokenizer(
                        question,
                        context,
                        max_length=max_length,
                        truncation='only_second',
                        stride=stride,
                        return_overflowing_tokens=True,
                        padding='max_length',
                        return_tensors='pt',
                    )

                    # Find question-end position (first [SEP])
                    sep_id  = tokenizer.sep_token_id
                    sep_pos = (inputs['input_ids'][0] == sep_id).nonzero(as_tuple=True)[0]
                    q_end   = sep_pos[0].item() if len(sep_pos) > 0 else 0

                    best_score  = float('-inf')
                    best_answer = ""

                    for w in range(inputs['input_ids'].size(0)):
                        in_ids   = inputs['input_ids'][w].to(device)
                        attn_msk = inputs['attention_mask'][w].to(device)

                        out = model(
                            input_ids=in_ids.unsqueeze(0),
                            attention_mask=attn_msk.unsqueeze(0),
                        )
                        # AutoAdapterModel with QA head exposes start/end logits
                        s_logits = out.start_logits[0]
                        e_logits = out.end_logits[0]

                        # Score = best start logit (simple heuristic, same as framework)
                        s_masked = s_logits.masked_fill(
                            (in_ids == 1) |
                            (torch.arange(len(in_ids), device=device) <= q_end),
                            float('-inf')
                        )
                        start_idx = s_masked.argmax().item()
                        score = s_logits[start_idx].item() + e_logits[start_idx].item()

                        if score > best_score:
                            best_score  = score
                            best_answer = extract_span(
                                s_logits, e_logits, in_ids, tokenizer, q_end
                            )

                    predictions[qid] = best_answer
                    total += 1
                    if verbose and total % 500 == 0:
                        print(f"    [{tgt_lang.upper()}] {total} samples processed ...")

    return predictions


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

TARGET_LANGS = ['ar', 'hi', 'vi']

MLQA_FILES = {
    'ar': 'test-context-ar-question-ar.json',
    'hi': 'test-context-hi-question-hi.json',
    'vi': 'test-context-vi-question-vi.json',
}

XQUAD_FILES = {
    'ar': 'xquad.ar.json',
    'hi': 'xquad.hi.json',
    'vi': 'xquad.vi.json',
}


def main():
    parser = argparse.ArgumentParser(
        description='MAD-X zero-shot QA baseline (ar/hi/vi) on MLQA + XQuAD'
    )
    parser.add_argument('--mlqa_dir',   default='dataset/MLQA',
                        help='Directory with MLQA test JSON files')
    parser.add_argument('--xquad_dir',  default='dataset',
                        help='Directory with xquad.{lang}.json files')
    parser.add_argument('--output',     default='madx_results.json',
                        help='Output JSON for results')
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--stride',     type=int, default=128)
    parser.add_argument('--save_preds', action='store_true',
                        help='Also save per-language prediction JSON files')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model, tokenizer = build_madx_model(device)

    all_results = {}

    for lang in TARGET_LANGS:
        print(f"\n{'='*60}")
        print(f"  Language: {lang.upper()}")
        print(f"{'='*60}")
        lang_results = {}

        # ── MLQA ──────────────────────────────────────────────────────
        mlqa_path = os.path.join(args.mlqa_dir, MLQA_FILES[lang])
        if os.path.exists(mlqa_path):
            print(f"\n[MLQA] {mlqa_path}")
            with open(mlqa_path, encoding='utf-8') as f:
                mlqa_json = json.load(f)
            preds = predict_on_dataset(
                model, tokenizer, mlqa_json, lang, device,
                max_length=args.max_length, stride=args.stride,
            )
            if args.save_preds:
                pred_path = args.output.replace('.json', f'_mlqa_{lang}_preds.json')
                with open(pred_path, 'w', encoding='utf-8') as f:
                    json.dump(preds, f, ensure_ascii=False, indent=2)
            m = evaluate_squad_format(mlqa_json, preds, lang)
            lang_results['mlqa'] = m
            print(f"  -> F1={m['f1']:.2f}  EM={m['exact_match']:.2f}  (n={m['total']})")
        else:
            print(f"[MLQA] Not found, skipping: {mlqa_path}")

        # ── XQuAD ─────────────────────────────────────────────────────
        xquad_path = os.path.join(args.xquad_dir, XQUAD_FILES[lang])
        if os.path.exists(xquad_path):
            print(f"\n[XQuAD] {xquad_path}")
            with open(xquad_path, encoding='utf-8') as f:
                xquad_json = json.load(f)
            preds = predict_on_dataset(
                model, tokenizer, xquad_json, lang, device,
                max_length=args.max_length, stride=args.stride,
            )
            if args.save_preds:
                pred_path = args.output.replace('.json', f'_xquad_{lang}_preds.json')
                with open(pred_path, 'w', encoding='utf-8') as f:
                    json.dump(preds, f, ensure_ascii=False, indent=2)
            m = evaluate_squad_format(xquad_json, preds, lang)
            lang_results['xquad'] = m
            print(f"  -> F1={m['f1']:.2f}  EM={m['exact_match']:.2f}  (n={m['total']})")
        else:
            print(f"[XQuAD] Not found, skipping: {xquad_path}")

        all_results[lang] = lang_results

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  MAD-X BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"{'Lang':<6} {'Dataset':<8} {'F1':>8} {'EM':>8} {'N':>7}")
    print('-' * 42)
    for lang, res in all_results.items():
        for dset, m in res.items():
            print(f"{lang:<6} {dset:<8} {m['f1']:>8.2f} {m['exact_match']:>8.2f} {m['total']:>7}")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved -> {args.output}")


if __name__ == '__main__':
    main()
