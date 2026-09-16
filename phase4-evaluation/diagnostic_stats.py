import torch
import os
import sys
import numpy as np
import torch.nn.functional as F
import collections

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_dataloader.process_qa_sample import load_squad_data, process_qa_sample
from gpu_utils import get_model
from phase2_model.model_core import CrossLingualOTModel
from phase3_loss.losses import OTAlignmentLoss
from transformers import AutoTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser(description="Diagnostic Evaluation Runner")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--stage1_ckpt", type=str, default=None, help="Path to Stage 1 base checkpoint (for LoRA)")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to evaluation SQuAD JSON file (e.g. XQuAD VI)")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of samples to evaluate for stats")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Model name")
    parser.add_argument("--max_length", type=int, default=384, help="Max length")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = CrossLingualOTModel(model_name=args.model_name).to(device)
    criterion = OTAlignmentLoss(hidden_size=model.hidden_size).to(device)

    # Load logic... (same as quick_eval.py)
    if args.stage1_ckpt:
        ckpt_stage1 = torch.load(args.stage1_ckpt, map_location=device)
        model.load_state_dict(ckpt_stage1.get("model_state", ckpt_stage1), strict=False)
        if "criterion_state" in ckpt_stage1 and ckpt_stage1["criterion_state"] is not None:
            criterion.load_state_dict(ckpt_stage1["criterion_state"], strict=False)
        model.apply_lora()
        model.to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    if "criterion_state" in ckpt and ckpt["criterion_state"] is not None:
        criterion.load_state_dict(ckpt["criterion_state"], strict=False)

    data = load_squad_data(args.eval_file)
    if args.n_samples > 0:
        data = data[:args.n_samples]
    
    model.eval()
    criterion.eval()

    null_predictions = 0
    total = 0
    has_ans_scores = []
    start_entropies = []
    end_entropies = []
    pred_lengths = []

    print(f"Running inference on {len(data)} samples...")
    with torch.no_grad():
        for i, item in enumerate(data):
            question = item["question"]
            context = item["context"]
            
            input_ids, attn_mask, _, _, q_end = process_qa_sample(
                question=question, context=context, answer=None, tokenizer=tokenizer,
                max_length=args.max_length, doc_stride=128,
            )
            input_ids = input_ids.unsqueeze(0).to(device)
            attn_mask = attn_mask.unsqueeze(0).to(device)
            q_end_val = q_end.item()

            base_model = get_model(model)
            out = base_model.backbone(input_ids, attn_mask)
            target_layers = [6, 7, 8, 9]
            stacked = torch.stack([out.hidden_states[l] for l in target_layers], dim=0)
            weights = torch.softmax(base_model.layer_weights, dim=0).view(4, 1, 1, 1)
            hidden = (stacked * weights).sum(dim=0)

            q_emb = hidden[:, :q_end_val, :]
            q_mask = torch.zeros(1, q_end_val, dtype=torch.bool, device=device)

            start_logits, end_logits, has_ans_logit = criterion.qa_head(hidden, q_emb, q_mask)

            # Masking
            padding_mask = (attn_mask[0] == 0)
            start_logits[0].masked_fill_(padding_mask, float('-inf'))
            end_logits[0].masked_fill_(padding_mask, float('-inf'))

            question_mask = torch.arange(start_logits.size(1), device=device) <= q_end_val
            question_mask[0] = False 
            start_logits[0].masked_fill_(question_mask, float('-inf'))
            end_logits[0].masked_fill_(question_mask, float('-inf'))

            # Entropy calculation (excluding masked positions)
            def compute_entropy(logits, mask):
                l_masked = logits.clone()
                l_masked.masked_fill_(mask, float('-inf'))
                probs = F.softmax(l_masked, dim=-1)
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
                return entropy.item()

            full_mask = padding_mask | question_mask
            start_entropies.append(compute_entropy(start_logits[0], full_mask))
            end_entropies.append(compute_entropy(end_logits[0], full_mask))

            has_ans_scores.append(has_ans_logit.item())

            MAX_ANSWER_LEN = 30
            start_idx = start_logits[0].argmax().item()
            end_logits_masked = end_logits[0].clone()
            end_logits_masked[:start_idx] = float('-inf')
            end_logits_masked[start_idx + MAX_ANSWER_LEN:] = float('-inf')
            end_idx = end_logits_masked.argmax().item()

            is_answerable_pred = has_ans_logit.item() > 0
            
            if not is_answerable_pred:
                null_predictions += 1
            else:
                pred_lengths.append(end_idx - start_idx + 1)
                
            total += 1
            
            if (i+1) % 100 == 0:
                print(f"Processed {i+1}/{len(data)}...")

    # Calculate stats
    null_rate = (null_predictions / total) * 100
    avg_null_score = np.mean(has_ans_scores)
    avg_start_ent = np.mean(start_entropies)
    avg_end_ent = np.mean(end_entropies)
    avg_len = np.mean(pred_lengths) if pred_lengths else 0.0

    print("\n" + "="*50)
    print("DIAGNOSTIC STATISTICS (XQuAD Calibration Check)")
    print("="*50)
    print(f"Checkpoint : {args.ckpt}")
    print(f"Dataset    : {args.eval_file}")
    print(f"Total Samples Processed: {total}")
    print("-" * 50)
    print(f"1. Null Prediction Rate : {null_rate:.2f}%")
    print(f"   (If this is high on XQuAD, calibration drifted!)")
    print(f"2. Average has_ans logit: {avg_null_score:.4f}")
    print(f"   (Values < 0 mean model defaults to Unanswerable)")
    print(f"3. Avg Start Entropy    : {avg_start_ent:.4f}")
    print(f"   Avg End Entropy      : {avg_end_ent:.4f}")
    print(f"   (Higher entropy means flatter distribution/lower confidence)")
    print(f"4. Avg Prediction Length: {avg_len:.2f} tokens")
    print("="*50)

if __name__ == "__main__":
    main()
