# phase4_evaluation/quick_eval.py
"""
Quick EM evaluation — checks has_answer accuracy on dev set.

Post-refactor: no graph, no subsampling, no GAT.
Uses full 512-token hidden states directly with QA head.
"""
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_dataloader.process_qa_sample import load_squad_data, process_qa_sample


def quick_em(model, criterion, tokenizer, dev_file, n_samples=200, device="cuda"):
    """
    Quick has_answer accuracy on dev set.

    Args:
        model     : CrossLingualOTModel (backbone only)
        criterion : OTAlignmentLoss (contains QA head)
        tokenizer : XLM-R tokenizer
        dev_file  : path to SQuAD2.0 dev JSON
        n_samples : number of samples to evaluate
        device    : torch device

    Returns:
        accuracy percentage (float)
    """
    data = load_squad_data(dev_file)[:n_samples]
    model.eval()
    criterion.eval()
    correct = 0

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
            out = model.backbone(input_ids, attn_mask)
            target_layers = [6, 7, 8, 9]
            stacked = torch.stack([out.hidden_states[i] for i in target_layers], dim=0)  # (4, 1, L, H)
            weights = torch.softmax(model.layer_weights, dim=0).view(4, 1, 1, 1)
            hidden = (stacked * weights).sum(dim=0)   # (1, L, H)

            # Extract question embeddings for cross-attention
            q_emb = hidden[:, :q_end_val + 1, :]     # (1, L_q, H)
            q_mask = torch.zeros(1, q_end_val + 1, dtype=torch.bool, device=device)

            # QA head: context=full sequence, question=q tokens
            start_logits, end_logits, has_ans_logit = criterion.qa_head(hidden, q_emb, q_mask)
            
            # Mask end positions: only allow [start_idx, start_idx + MAX_ANSWER_LEN]
            MAX_ANSWER_LEN = 30
            start_idx = start_logits[0].argmax().item()
            end_logits_masked = end_logits[0].clone()
            end_logits_masked[:start_idx] = float('-inf')
            end_logits_masked[start_idx + MAX_ANSWER_LEN:] = float('-inf')
            end_idx = end_logits_masked.argmax().item()

            pred_answerable = has_ans_logit.item() > 0

            if pred_answerable == is_answerable:
                correct += 1

    return correct / len(data) * 100


# ──────────────────────────────────────────────────────────────
# S2-04 Extension: XQuAD VI Evaluation (exact match on VI spans)
# quick_em() above is NOT modified — extend-only per spec.
# ──────────────────────────────────────────────────────────────

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


def quick_em_xquad_vi(
    model,
    criterion,
    tokenizer,
    val_pairs: list[dict],
    device,
    max_length: int = 384,
    max_answer_len: int = 30,
) -> float:
    """
    Exact-match evaluation on XQuAD VI val split.

    Runs span extraction on VI input using the shared backbone + QA head.
    The predicted span string is compared to ground-truth VI answers.

    Args:
        model      : CrossLingualOTModel (backbone)
        criterion  : OTAlignmentLoss (contains qa_head)
        tokenizer  : XLM-R tokenizer
        val_pairs  : list of dicts from load_xquad_pairs (val split)
                     each dict has: "question_vi", "context_vi", "answer_vi"
        device     : torch device
        max_length : max token length (must match training max_length)
        max_answer_len: max span length in tokens

    Returns:
        EM score (float, 0–100)
    """
    from phase1_dataloader.process_qa_sample import process_qa_sample

    model.eval()
    criterion.eval()
    correct = 0
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
            out = model.backbone(input_ids, attn_mask)
            target_layers = [6, 7, 8, 9]
            stacked = torch.stack(
                [out.hidden_states[i] for i in target_layers], dim=0
            )  # (4, 1, L, H)
            weights = torch.softmax(model.layer_weights, dim=0).view(4, 1, 1, 1)
            hidden = (stacked * weights).sum(dim=0)   # (1, L, H)

            # Question embeddings for cross-attention
            q_emb  = hidden[:, :q_end_val + 1, :]    # (1, L_q, H)
            q_mask = torch.zeros(1, q_end_val + 1, dtype=torch.bool, device=device)

            # QA head: predict start/end logits
            start_logits, end_logits, _ = criterion.qa_head(hidden, q_emb, q_mask)

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

    return (correct / total * 100) if total > 0 else 0.0

