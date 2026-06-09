"""
Inference pipeline: model → JSON predictions.

Post-refactor: no graph, no subsampling, no GAT.
Uses full 512-token hidden states with QA head for span prediction.
"""

import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2_model.model_core import CrossLingualOTModel
from phase3_loss.losses import OTAlignmentLoss
from phase1_dataloader.process_qa_sample import process_qa_sample, load_squad_data


def extract_ground_truth(item):
    """Extract ground truth answer, handling SQuAD2 is_impossible."""
    if item.get("is_impossible", False):
        return ""

    val = item.get("answer") or item.get("answers")
    if val is None:
        return ""

    if isinstance(val, str):
        return val.strip()

    if isinstance(val, dict):
        texts = val.get("text", [])
        return texts[0].strip() if texts else ""

    if isinstance(val, list):
        return val[0].strip() if val else ""

    return ""


def find_best_span(start_logits, end_logits, seq_len, max_span_len, question_end, attention_mask=None):
    """Find best answer span in context region (after first [SEP]) using greedy decoding with constraint."""
    context_start = question_end + 1

    start_logits_masked = start_logits.clone()
    start_logits_masked[:context_start] = float('-inf')
    if attention_mask is not None:
        pad_mask = (attention_mask == 0)
        start_logits_masked[pad_mask] = float('-inf')

    start_idx = start_logits_masked.argmax().item()

    end_logits_masked = end_logits.clone()
    end_logits_masked[:start_idx] = float('-inf')
    end_logits_masked[start_idx + max_span_len:] = float('-inf')
    if attention_mask is not None:
        end_logits_masked[pad_mask] = float('-inf')

    end_idx = end_logits_masked.argmax().item()

    best_span_score = start_logits[start_idx].item() + end_logits[end_idx].item()
    found_context_span = (start_idx >= context_start) and (end_idx >= start_idx)

    return start_idx, end_idx, best_span_score, found_context_span


def main():
    parser = argparse.ArgumentParser(description="Inference: Cross-Lingual QA → JSON")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--input_file",  type=str, required=True,
                        help="JSON file with test data (SQuAD format)")
    parser.add_argument("--output_file", type=str, default="phase4-evaluation/predictions.json",
                        help="Output JSON file")
    parser.add_argument("--model_name",  type=str, default="xlm-roberta-base",
                        help="Base model name")
    parser.add_argument("--max_span_len", type=int, default=30,
                        help="Maximum span length (tokens)")
    parser.add_argument("--debug", action="store_true", help="Print details for first 5 samples")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    print("Loading data...")
    try:
        dataset = load_squad_data(args.input_file)
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    print(f"Loaded {len(dataset)} questions.")

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = CrossLingualOTModel(model_name=args.model_name).to(device)

    criterion = OTAlignmentLoss(
        hidden_size=model.backbone.hidden_size,
    ).to(device)

    # Load weights
    model_state = checkpoint["model_state"]
    model_state_clean = {k.replace("module.", ""): v for k, v in model_state.items()}
    model.load_state_dict(model_state_clean, strict=False)

    # Load QA Head
    criterion_state = checkpoint["criterion_state"]
    criterion_state_clean = {k.replace("module.", ""): v for k, v in criterion_state.items()}
    criterion.load_state_dict(criterion_state_clean, strict=True)

    print("✅ Model and QA Head loaded successfully!")

    model.eval()
    criterion.eval()

    print(f"\nRunning inference on {len(dataset)} questions...")
    results = []

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Predicting"):
        question = item["question"]
        context = item["context"]
        ground_truth = extract_ground_truth(item)

        input_ids, attention_mask, _, _, question_end = process_qa_sample(
            question=question,
            context=context,
            answer=None,
            tokenizer=tokenizer,
            max_length=512,
            doc_stride=128
        )

        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        question_end = question_end.item()   # index của [SEP] sau question
        seq_len = input_ids.shape[1]

        with torch.no_grad():
            # Get backbone output
            outputs = model.backbone(input_ids, attention_mask)
            target_layers = [6, 7, 8, 9]
            stacked = torch.stack([outputs.hidden_states[i] for i in target_layers], dim=0)  # (4, 1, L, H)
            weights = torch.softmax(model.layer_weights, dim=0).view(4, 1, 1, 1)
            hidden = (stacked * weights).sum(dim=0)   # (1, L, H)

            # === FIXED: Extract question tokens (before [SEP]) ===
            q_end = question_end
            q_emb = hidden[:, :q_end, :]                    # Question part only
            q_mask = torch.zeros(1, q_end, dtype=torch.bool, device=device)

            # QA Head
            start_logits, end_logits, has_answer_logit = criterion.qa_head(
                hidden, q_emb, q_mask
            )
            start_logits = start_logits.squeeze(0)
            end_logits = end_logits.squeeze(0)

            # Find best span
            best_s, best_e, best_span_score, found_context_span = find_best_span(
                start_logits, end_logits, seq_len, args.max_span_len, question_end, attention_mask[0]
            )

            is_ans = has_answer_logit.item() > 0.0

            if is_ans and found_context_span:
                pred_ids = input_ids[0, best_s : best_e + 1]
                predicted_answer = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            else:
                predicted_answer = ""

        results.append({
            "id": item.get("id", str(idx)),
            "question": question,
            "answer": predicted_answer,
            "ground_truth": ground_truth,
        })

        if args.debug and idx < 5:
            has_ans_prob = torch.sigmoid(has_answer_logit).item()
            print(f"\n[DEBUG #{idx+1}]")
            print(f"  Q       : {question[:70]}...")
            print(f"  GT      : '{ground_truth}'")
            print(f"  Pred    : '{predicted_answer}'")
            print(f"  has_ans : {has_ans_prob:.3f} → {'ANSWERABLE' if has_ans_prob > 0.5 else 'UNANSWERABLE'}")
            print(f"  span    : [{best_s}:{best_e}] score={best_span_score:.3f}")

    # Save
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    empty_pred = sum(1 for r in results if not r["answer"])
    print(f"\n✅ Inference completed!")
    print(f"   Predictions saved to: {args.output_file}")
    print(f"   Empty predictions: {empty_pred}/{len(results)}")


if __name__ == "__main__":
    main()