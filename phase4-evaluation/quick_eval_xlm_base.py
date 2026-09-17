# phase4_evaluation/quick_eval.py
"""
Quick EM and F1 evaluation — checks has_answer accuracy on dev set.

Post-refactor: no graph, no subsampling, no GAT.
Uses full 512-token hidden states directly with QA head.
"""
import torch
import os
import sys
import collections

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_dataloader.process_qa_sample import load_squad_data, process_qa_sample
from gpu_utils import get_model


def _normalize_answer(s: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    import re, string
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
        # If there are no ground truths, F1 is 1.0 if prediction is empty, else 0.0
        return 1.0 if not prediction else 0.0

    max_f1 = 0.0
    for gt in ground_truths:
        f1 = compute_f1(prediction, gt)
        if f1 > max_f1:
            max_f1 = f1
    return max_f1


def quick_em_f1(model, criterion, tokenizer, dev_file, n_samples=200, device="cuda"):
    """
    Quick has_answer accuracy on dev set evaluating both EM and F1.

    Returns:
        (em_score, f1_score)
    """
    data = load_squad_data(dev_file)[:n_samples]
    model.eval()
    criterion.eval()
    correct = 0
    total_f1 = 0.0

    with torch.no_grad():
        for item in data:
            is_answerable = len(item["answer"].get("answer_start", [])) > 0

            input_ids, attn_mask, _, _, q_end = process_qa_sample(
                item["question"], item["context"], None, tokenizer, 512, 128
            )
            input_ids = input_ids.unsqueeze(0).to(device)
            attn_mask = attn_mask.unsqueeze(0).to(device)
            q_end_val = q_end.item()

            # Get hidden states from backbone (no GAT, no subsampling)
            base_model = get_model(model)
            out = base_model.backbone(input_ids, attn_mask)
            target_layers = [6, 7, 8, 9]
            stacked = torch.stack([out.hidden_states[i] for i in target_layers], dim=0)  # (4, 1, L, H)
            weights = torch.softmax(base_model.layer_weights, dim=0).view(4, 1, 1, 1)
            hidden = (stacked * weights).sum(dim=0)   # (1, L, H)

            # Extract question embeddings for cross-attention (exclude [SEP])
            q_emb = hidden[:, :q_end_val, :]     # (1, L_q, H)
            q_mask = torch.zeros(1, q_end_val, dtype=torch.bool, device=device)

            # QA head: context=full sequence, question=q tokens
            start_logits, end_logits, has_ans_logit = criterion.qa_head(hidden, q_emb, q_mask)
            
            # Mask out padding tokens
            padding_mask = (attn_mask[0] == 0)
            start_logits[0].masked_fill_(padding_mask, float('-inf'))
            end_logits[0].masked_fill_(padding_mask, float('-inf'))

            # Mask out question tokens (từ index 0 đến q_end_val)
            question_mask = torch.arange(start_logits.size(1), device=device) <= q_end_val
            # UNMASK index 0 ([CLS] token) so the model can predict "unanswerable"
            question_mask[0] = False 
            start_logits[0].masked_fill_(question_mask, float('-inf'))
            end_logits[0].masked_fill_(question_mask, float('-inf'))

            # Mask end positions: only allow [start_idx, start_idx + MAX_ANSWER_LEN]
            MAX_ANSWER_LEN = 30
            start_idx = start_logits[0].argmax().item()
            end_logits_masked = end_logits[0].clone()
            end_logits_masked[:start_idx] = float('-inf')
            end_logits_masked[start_idx + MAX_ANSWER_LEN:] = float('-inf')
            end_idx = end_logits_masked.argmax().item()

            pred_answerable = has_ans_logit.item() > 0

            # Decode span if predicted answerable
            if not pred_answerable:
                pred_span = ""
            else:
                pred_ids = input_ids[0][start_idx: end_idx + 1]
                pred_span = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

            ground_truths = item["answer"].get("text", [])
            
            if not is_answerable:
                # For unanswerable questions, correct if predicted span is empty
                if pred_span == "":
                    correct += 1
                    total_f1 += 1.0
            else:
                # For answerable questions, check Exact Match and F1
                if _exact_match_score(pred_span, ground_truths):
                    correct += 1
                total_f1 += _f1_score(pred_span, ground_truths)

    n_total = len(data)
    return (correct / n_total * 100) if n_total > 0 else 0.0, (total_f1 / n_total * 100) if n_total > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# S2-04 Extension: XQuAD VI Evaluation (exact match & F1 on VI spans)
# ──────────────────────────────────────────────────────────────

def quick_em_f1_xquad_vi(
    model,
    criterion,
    tokenizer,
    val_pairs: list[dict],
    device,
    max_length: int = 384,
    max_answer_len: int = 30,
):
    """
    Exact-match and F1 evaluation on XQuAD VI val split.

    Returns:
        (em_score, f1_score)
    """
    from phase1_dataloader.process_qa_sample import process_qa_sample

    model.eval()
    criterion.eval()
    correct = 0
    total_f1 = 0.0
    total   = 0

    with torch.no_grad():
        for pair in val_pairs:
            question_vi  = pair["question_vi"]
            context_vi   = pair["context_vi"]
            answer_vi    = pair["answer_vi"]
            ground_truths = answer_vi.get("text", [])

            # Tokenize VI input
            input_ids, attn_mask, _, _, q_end = process_qa_sample(
                question=question_vi,
                context=context_vi,
                answer=None,       # no labels needed for eval
                tokenizer=tokenizer,
                max_length=max_length,
                doc_stride=128,
            )
            input_ids = input_ids.unsqueeze(0).to(device)   # (1, L)
            attn_mask = attn_mask.unsqueeze(0).to(device)   # (1, L)
            q_end_val = q_end.item()

            # Hidden states via shared backbone (layers 6-9 weighted mix)
            base_model = get_model(model)
            out = base_model.backbone(input_ids, attn_mask)
            target_layers = [6, 7, 8, 9]
            stacked = torch.stack(
                [out.hidden_states[i] for i in target_layers], dim=0
            )  # (4, 1, L, H)
            weights = torch.softmax(base_model.layer_weights, dim=0).view(4, 1, 1, 1)
            hidden = (stacked * weights).sum(dim=0)   # (1, L, H)

            # Question embeddings for cross-attention (exclude [SEP])
            q_emb  = hidden[:, :q_end_val, :]    # (1, L_q, H)
            q_mask = torch.zeros(1, q_end_val, dtype=torch.bool, device=device)

            # QA head: predict start/end logits
            start_logits, end_logits, _ = criterion.qa_head(hidden, q_emb, q_mask)

            # Mask out padding tokens
            padding_mask = (attn_mask[0] == 0)
            start_logits[0].masked_fill_(padding_mask, float("-inf"))
            end_logits[0].masked_fill_(padding_mask, float("-inf"))

            # Mask out question tokens (từ index 0 đến q_end_val)
            question_mask = torch.arange(start_logits.size(1), device=device) <= q_end_val
            # UNMASK index 0 ([CLS] token) so the model can predict "unanswerable"
            question_mask[0] = False 
            start_logits[0].masked_fill_(question_mask, float('-inf'))
            end_logits[0].masked_fill_(question_mask, float('-inf'))

            # Constrain end to [start, start + max_answer_len]
            start_idx = start_logits[0].argmax().item()
            end_logits_masked = end_logits[0].clone()
            end_logits_masked[:start_idx] = float("-inf")
            end_logits_masked[start_idx + max_answer_len:] = float("-inf")
            end_idx = end_logits_masked.argmax().item()

            # Decode predicted span
            pred_ids   = input_ids[0][start_idx: end_idx + 1]
            pred_span  = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

            total += 1
            if _exact_match_score(pred_span, ground_truths):
                correct += 1
            total_f1 += _f1_score(pred_span, ground_truths)

    return (correct / total * 100) if total > 0 else 0.0, (total_f1 / total * 100) if total > 0 else 0.0


if __name__ == "__main__":
    import argparse
    from phase2_model.model_core import CrossLingualOTModel
    from phase3_loss.losses import OTAlignmentLoss
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Quick Evaluation Runner")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (Stage 1 or Stage 2)")
    parser.add_argument("--stage1_ckpt", type=str, default=None, help="Path to Stage 1 base checkpoint (Required if --ckpt is a Stage 2 LoRA checkpoint)")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to evaluation SQuAD JSON file")
    parser.add_argument("--n_samples", type=int, default=180, help="Number of samples to evaluate")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Model name")
    parser.add_argument("--max_length", type=int, default=384, help="Max length")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Initialize model and criterion
    model = CrossLingualOTModel(model_name=args.model_name).to(device)
    criterion = OTAlignmentLoss(hidden_size=model.hidden_size).to(device)

    print(f"Loading checkpoint from: {args.ckpt}")
    if not os.path.exists(args.ckpt):
        # Fallback helper: check if maybe it's in checkpoint/best.pt
        alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoint", "best.pt")
        if os.path.exists(alt_path):
            print(f"Checkpoint {args.ckpt} not found. Falling back to {alt_path}")
            args.ckpt = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    # If stage1_ckpt is provided, load it first, then apply LoRA, then load the Stage 2 ckpt
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

    # Load the main checkpoint (either Stage 1 or Stage 2 LoRA)
    ckpt = torch.load(args.ckpt, map_location=device)
    
    # Check if checkpoint has dict keys or is direct state dict
    if "model_state" in ckpt:
        # Load model_state. strict=False allows loading LoRA weights smoothly
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

    # Evaluate exact match and F1
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

            # Tokenize
            input_ids, attn_mask, _, _, q_end = process_qa_sample(
                question=question,
                context=context,
                answer=None,
                tokenizer=tokenizer,
                max_length=args.max_length,
                doc_stride=128,
            )
            input_ids = input_ids.unsqueeze(0).to(device)
            attn_mask = attn_mask.unsqueeze(0).to(device)
            q_end_val = q_end.item()

            # Forward backbone
            base_model = get_model(model)
            out = base_model.backbone(input_ids, attn_mask)
            target_layers = [6, 7, 8, 9]
            stacked = torch.stack([out.hidden_states[l] for l in target_layers], dim=0)
            weights = torch.softmax(base_model.layer_weights, dim=0).view(4, 1, 1, 1)
            hidden = (stacked * weights).sum(dim=0)

            # Question embeddings (exclude [SEP])
            q_emb = hidden[:, :q_end_val, :]
            q_mask = torch.zeros(1, q_end_val, dtype=torch.bool, device=device)

            # Predict logits
            start_logits, end_logits, has_ans_logit = criterion.qa_head(hidden, q_emb, q_mask)

            # Mask out padding tokens
            padding_mask = (attn_mask[0] == 0)
            start_logits[0].masked_fill_(padding_mask, float('-inf'))
            end_logits[0].masked_fill_(padding_mask, float('-inf'))

            # Mask out question tokens (từ index 0 đến q_end_val)
            question_mask = torch.arange(start_logits.size(1), device=device) <= q_end_val
            # UNMASK index 0 ([CLS] token) so the model can predict "unanswerable"
            question_mask[0] = False 
            start_logits[0].masked_fill_(question_mask, float('-inf'))
            end_logits[0].masked_fill_(question_mask, float('-inf'))

            # Decode span
            MAX_ANSWER_LEN = 30
            start_idx = start_logits[0].argmax().item()
            end_logits_masked = end_logits[0].clone()
            end_logits_masked[:start_idx] = float('-inf')
            end_logits_masked[start_idx + MAX_ANSWER_LEN:] = float('-inf')
            end_idx = end_logits_masked.argmax().item()

            pred_ids = input_ids[0][start_idx: end_idx + 1]
            pred_span = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

            is_answerable_pred = has_ans_logit.item() > 0
            
            # Logic tính toán correct (EM) và điểm F1 hiện tại
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
                    is_correct = _exact_match_score(pred_span, ground_truths)
                    f1_current = _f1_score(pred_span, ground_truths)
            
            if is_correct:
                correct += 1
            total_f1 += f1_current
            total += 1

            if (i + 1) % 20 == 0 or (i + 1) == len(data):
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