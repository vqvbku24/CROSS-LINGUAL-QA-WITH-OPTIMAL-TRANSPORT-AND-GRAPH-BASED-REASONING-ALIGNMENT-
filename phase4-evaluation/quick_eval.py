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
            hidden = model.backbone(input_ids, attn_mask)  # (1, L, H)

            # Extract question embeddings for cross-attention
            q_emb = hidden[:, :q_end_val + 1, :]     # (1, L_q, H)
            q_mask = torch.zeros(1, q_end_val + 1, dtype=torch.bool, device=device)

            # QA head: context=full sequence, question=q tokens
            _, _, has_ans_logit = criterion.qa_head(hidden, q_emb, q_mask)
            pred_answerable = has_ans_logit.item() > 0

            if pred_answerable == is_answerable:
                correct += 1

    return correct / len(data) * 100
