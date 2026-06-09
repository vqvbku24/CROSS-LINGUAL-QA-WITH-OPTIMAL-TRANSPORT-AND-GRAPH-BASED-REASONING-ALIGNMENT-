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
