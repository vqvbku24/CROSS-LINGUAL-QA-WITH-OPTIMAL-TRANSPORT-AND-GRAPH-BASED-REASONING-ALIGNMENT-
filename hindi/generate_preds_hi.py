# hindi/generate_preds_hi.py
"""
Generate predictions JSON for Hindi QA evaluation.

Works with both XQuAD-hi and MLQA-hi JSON files (SQuAD format).
Uses sliding window inference to handle long contexts.

Usage:
    python hindi/generate_preds_hi.py \\
        --stage1_ckpt checkpoints/stage1_squad_best.pt \\
        --ckpt checkpoint_stage2_hi/stage2_best.pt \\
        --eval_file dataset/MLQA/test-context-hi-question-hi.json \\
        --output_pred_file hindi_mlqa_preds.json

    # Zero-shot (stage1 only, no stage2):
    python hindi/generate_preds_hi.py \\
        --ckpt checkpoints/stage1_squad_best.pt \\
        --eval_file dataset/MLQA/test-context-hi-question-hi.json \\
        --output_pred_file hindi_zeroshot_preds.json \\
        --zero_shot
"""
import json
import torch
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from phase2_model.model_core import CrossLingualOTModel
from phase3_loss.losses import OTAlignmentLoss
from phase1_dataloader.process_qa_sample import process_qa_sample


def generate_predictions(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 1. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model     = CrossLingualOTModel(model_name=args.model_name).to(device)
    criterion = OTAlignmentLoss(hidden_size=model.hidden_size).to(device)

    # 2. Load Stage 1 base
    if args.stage1_ckpt:
        print(f'Loading Stage 1 base from: {args.stage1_ckpt}')
        ckpt_s1 = torch.load(args.stage1_ckpt, map_location=device)
        model.load_state_dict(ckpt_s1.get('model_state', ckpt_s1), strict=False)
        if 'criterion_state' in ckpt_s1 and ckpt_s1['criterion_state'] is not None:
            criterion.load_state_dict(ckpt_s1['criterion_state'], strict=False)

        if not args.zero_shot:
            print('Applying LoRA to model...')
            model.apply_lora()
            model.to(device)

    # 3. Load Stage 2 LoRA weights (unless zero-shot mode)
    if not args.zero_shot:
        print(f'Loading Stage 2 weights from {args.ckpt} ...')
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt.get('model_state', ckpt), strict=False)
        if 'criterion_state' in ckpt and ckpt['criterion_state'] is not None:
            criterion.load_state_dict(ckpt['criterion_state'], strict=False)
            print('Loaded QA Head weights from Stage 2.')
        else:
            print('WARNING: No criterion_state in Stage 2 checkpoint.')
    else:
        print('Zero-shot mode: using Stage 1 checkpoint only.')
        if args.stage1_ckpt is None:
            # --ckpt IS the stage1 ckpt in zero-shot mode
            print(f'Loading Stage 1 from: {args.ckpt}')
            ckpt_s1 = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(ckpt_s1.get('model_state', ckpt_s1), strict=False)
            if 'criterion_state' in ckpt_s1 and ckpt_s1['criterion_state'] is not None:
                criterion.load_state_dict(ckpt_s1['criterion_state'], strict=False)

    model.eval()
    criterion.eval()

    # 4. Load eval file
    print(f'Reading eval data from {args.eval_file} ...')
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)['data']

    predictions = {}
    total_samples = 0

    # 5. Sliding window inference
    print('Generating predictions...')
    with torch.no_grad():
        for article in eval_data:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    qa_id    = qa['id']
                    question = qa['question']

                    inputs = tokenizer(
                        question,
                        context,
                        max_length=args.max_length,
                        truncation='only_second',
                        stride=128,
                        return_overflowing_tokens=True,
                        padding='max_length',
                        return_tensors='pt',
                    )

                    num_windows = inputs['input_ids'].size(0)
                    sep_positions = (inputs['input_ids'][0] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                    q_end_val = sep_positions[0].item() if len(sep_positions) > 0 else 0

                    best_score = float('-inf')
                    best_span  = ''

                    for w in range(num_windows):
                        input_ids = inputs['input_ids'][w].unsqueeze(0).to(device)
                        attn_mask = inputs['attention_mask'][w].unsqueeze(0).to(device)

                        out = model.backbone(input_ids, attn_mask)
                        stacked = torch.stack([out.hidden_states[i] for i in [6, 7, 8, 9]], dim=0)
                        weights = torch.softmax(model.layer_weights, dim=0).view(4, 1, 1, 1)
                        hidden  = (stacked * weights).sum(dim=0)

                        q_emb  = hidden[:, :q_end_val, :]
                        q_mask = torch.zeros(1, q_end_val, dtype=torch.bool, device=device)

                        start_logits, end_logits, _ = criterion.qa_head(hidden, q_emb, q_mask)

                        # Mask padding
                        padding_mask = (attn_mask[0] == 0)
                        start_logits[0].masked_fill_(padding_mask, float('-inf'))
                        end_logits[0].masked_fill_(padding_mask, float('-inf'))

                        # Mask question tokens, keep [CLS] masked (force answerable for HI generation)
                        question_mask = torch.arange(start_logits.size(1), device=device) <= q_end_val
                        question_mask[0] = True
                        start_logits[0].masked_fill_(question_mask, float('-inf'))
                        end_logits[0].masked_fill_(question_mask, float('-inf'))

                        MAX_ANSWER_LEN = 30
                        start_idx = start_logits[0].argmax().item()
                        end_logits_masked = end_logits[0].clone()
                        end_logits_masked[:start_idx] = float('-inf')
                        end_logits_masked[start_idx + MAX_ANSWER_LEN:] = float('-inf')
                        end_idx = end_logits_masked.argmax().item()

                        score = start_logits[0][start_idx].item() + end_logits[0][end_idx].item()
                        if score > best_score:
                            best_score = score
                            pred_ids   = input_ids[0][start_idx: end_idx + 1]
                            best_span  = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

                    predictions[qa_id] = best_span
                    total_samples += 1

                    if total_samples % 500 == 0:
                        print(f'  Processed {total_samples} samples...')

    # 6. Save predictions
    with open(args.output_pred_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    print(f'Saved {total_samples} predictions to {args.output_pred_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate HI QA predictions')
    parser.add_argument('--ckpt',             type=str, required=True,
                        help='Stage 2 checkpoint (or Stage 1 if --zero_shot)')
    parser.add_argument('--stage1_ckpt',      type=str, default=None,
                        help='Stage 1 checkpoint (required when loading Stage 2 LoRA)')
    parser.add_argument('--eval_file',        type=str, required=True,
                        help='XQuAD-hi or MLQA-hi JSON file')
    parser.add_argument('--output_pred_file', type=str, default='hi_preds.json')
    parser.add_argument('--model_name',       type=str, default='xlm-roberta-base')
    parser.add_argument('--max_length',       type=int, default=384)
    parser.add_argument('--zero_shot',        action='store_true',
                        help='Use Stage 1 only (no LoRA, no Stage 2 weights)')
    args = parser.parse_args()
    generate_predictions(args)
