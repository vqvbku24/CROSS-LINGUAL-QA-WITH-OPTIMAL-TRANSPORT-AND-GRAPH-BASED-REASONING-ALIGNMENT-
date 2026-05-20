# phase4_evaluation/quick_eval.py
import torch
import os
import sys

# Đảm bảo import được các module khác
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_dataloader.process_qa_sample import load_squad_data, process_qa_sample
from phase2_model.modules.subsampling import conditional_subsample

def quick_em(model, criterion, tokenizer, dev_file, n_samples=200, device="cuda"):
    data = load_squad_data(dev_file)[:n_samples]
    model.eval(); criterion.eval()
    correct = 0
    with torch.no_grad():
        for item in data:
            # FIX: load_squad_data không có trường "is_impossible", ta dùng len(answer_start)
            is_answerable = len(item["answer"].get("answer_start", [])) > 0
            
            input_ids, attn_mask, _, _, q_end = process_qa_sample(
                item["question"], item["context"], None, tokenizer, 512, 128
            )
            input_ids = input_ids.unsqueeze(0).to(device)
            attn_mask = attn_mask.unsqueeze(0).to(device)
            q_end = q_end.item()

            hidden, attn = model.backbone(input_ids, attn_mask)
            q_idx = list(range(q_end + 1))
            sub_matrix, keep_idx = conditional_subsample(attn[0], q_idx, [], K=model.K)
            feat = hidden[0, keep_idx, :]
            node_emb, _ = model.gat(feat, sub_matrix)
            q_emb  = hidden[:, :q_end+1, :]
            q_mask = torch.zeros(1, q_end+1, dtype=torch.bool, device=device)

            _, _, has_ans_logit = criterion.qa_head(node_emb.unsqueeze(0), q_emb, q_mask)
            pred_answerable = has_ans_logit.item() > 0

            # EM chỉ trên has_answer (nhanh, không cần decode span)
            if pred_answerable == is_answerable:
                correct += 1

    return correct / len(data) * 100
