"""
Inference pipeline: model → JSON predictions.

Post-refactor: no graph, no subsampling, no GAT.
Uses full 512-token hidden states with QA head for span prediction.
Span search operates directly in token-space (no graph-space remapping).
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
    """
    Read ground truth flexibly, supporting 3 formats:
      1. answer is string        -> "Thái Bình Dương"
      2. answer is SQuAD dict    -> {"text": [...], "answer_start": [...]}
      3. answers is SQuAD dict   -> {"text": [...], "answer_start": [...]}
    For is_impossible=True → return "" (SQuAD2 standard).
    """
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


def find_best_span(start_logits, end_logits, seq_len, max_span_len, question_end):
    """
    Find optimal span (s, e) in CONTEXT region (after question).
    Operates directly in token-space (no graph node remapping needed).

    Args:
        start_logits  : (L,) start logits for all positions
        end_logits    : (L,) end logits for all positions
        seq_len       : total sequence length
        max_span_len  : maximum span length in tokens
        question_end  : index of first [SEP] (question boundary)

    Returns:
        best_s, best_e, best_score, found_context_span
    """
    best_score = float('-inf')
    best_s, best_e = 0, 0
    found_context_span = False

    # Context starts after [SEP] (question_end + 1) and second [SEP] (+1)
    context_start = question_end + 2  # skip [SEP] after question

    for s in range(context_start, seq_len):
        if start_logits[s].item() < -1e3:  # skip PAD positions
            break
        for e in range(s, min(s + max_span_len + 1, seq_len)):
            if end_logits[e].item() < -1e3:  # skip PAD positions
                break
            score = start_logits[s].item() + end_logits[e].item()
            if score > best_score:
                best_score = score
                best_s, best_e = s, e
                found_context_span = True

    return best_s, best_e, best_score, found_context_span


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

    print("Loading tokenizer (xlm-roberta-base)...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

    print("Loading data...")
    try:
        dataset = load_squad_data(args.input_file)
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    print(f"Loaded {len(dataset)} questions.")
    sample_gt = extract_ground_truth(dataset[0])
    print(f"🔍 Sanity check ground_truth[0]: '{sample_gt}'")

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})

    # Model: simplified — no GAT, no K param
    model = CrossLingualOTModel(
        model_name="xlm-roberta-base",
    ).to(device)

    criterion = OTAlignmentLoss(
        hidden_size=model.backbone.hidden_size,
    ).to(device)

    # Load model weights
    model_state = checkpoint["model_state"]
    model_state_clean = {k.replace("module.", ""): v for k, v in model_state.items()}
    load_result = model.load_state_dict(model_state_clean, strict=False)

    print(f"\n[MODEL WEIGHTS CHECK]")
    missing = load_result.missing_keys
    print(f" - Missing keys: {len(missing)}")
    if len(missing) > 0:
        print(f" ⚠️ WARNING: Missing keys: {missing[:5]}")
    else:
        print(f" ✅ All backbone weights loaded successfully.")

    # Load criterion (QA Head) weights
    try:
        criterion_state = checkpoint["criterion_state"]
        criterion_state_clean = {k.replace("module.", ""): v for k, v in criterion_state.items()}
        criterion.load_state_dict(criterion_state_clean, strict=True)
        print(" ✅ QA Head & HasAnswer Head loaded successfully!")
    except RuntimeError as e:
        print(f"\n❌ ERROR: Checkpoint structure mismatch!\n{e}")
        sys.exit(1)

    model.eval()
    criterion.eval()

    print(f"\nRunning inference on {len(dataset)} questions...")
    results = []

    for item in tqdm(dataset, desc="Predicting"):
        question     = item["question"]
        context      = item["context"]
        ground_truth = extract_ground_truth(item)

        input_ids, attention_mask, _, _, question_end = process_qa_sample(
            question=question, context=context, answer=None,
            tokenizer=tokenizer, max_length=512, doc_stride=128
        )

        input_ids      = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        question_end   = question_end.item()
        seq_len        = input_ids.shape[1]

        with torch.no_grad():
            # Get hidden states from backbone
            hidden = model.backbone(input_ids, attention_mask)  # (1, L, H)

            # Extract question embeddings
            q_emb = hidden[:, :question_end + 1, :]
            q_mask = torch.zeros(1, question_end + 1, dtype=torch.bool, device=device)

            # QA Head prediction
            start_logits, end_logits, has_answer_logit = criterion.qa_head(
                hidden, q_emb, q_mask
            )
            start_logits = start_logits.squeeze(0)  # (L,)
            end_logits   = end_logits.squeeze(0)    # (L,)

            # Find best span in context region
            best_s, best_e, best_span_score, found_context_span = find_best_span(
                start_logits, end_logits, seq_len, args.max_span_len, question_end
            )

            # Answerable decision
            is_ans = has_answer_logit.item() > 0

            if is_ans and found_context_span:
                pred_ids = input_ids[0, best_s : best_e + 1]
                predicted_answer = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            else:
                predicted_answer = ""

        results.append({
            "id":           item.get("id", str(len(results))),
            "question":     question,
            "answer":       predicted_answer,
            "ground_truth": ground_truth,
        })

        if args.debug and len(results) <= 5:
            has_ans_prob = torch.sigmoid(has_answer_logit).item()
            print(f"\n[DEBUG #{len(results)}]")
            print(f"  Q              : {question[:70]}")
            print(f"  GT             : '{ground_truth}'")
            print(f"  Pred           : '{predicted_answer}'")
            print(f"  has_answer_prob: {has_ans_prob:.3f} ({'answerable' if has_ans_prob > 0.5 else 'unanswerable'})")
            print(f"  best_s={best_s}  best_e={best_e}  score={best_span_score:.3f}")

    # Write results
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    empty_pred  = sum(1 for r in results if not r["answer"])
    empty_truth = sum(1 for r in results if not r["ground_truth"])
    print(f"\nQuick stats:")
    print(f"  Empty predictions  : {empty_pred}/{len(results)} ({empty_pred/len(results)*100:.1f}%)")
    print(f"  Empty ground truth : {empty_truth}/{len(results)} ({empty_truth/len(results)*100:.1f}%)")
    print(f"\nDone! Saved to: {args.output_file}")


if __name__ == "__main__":
    main()