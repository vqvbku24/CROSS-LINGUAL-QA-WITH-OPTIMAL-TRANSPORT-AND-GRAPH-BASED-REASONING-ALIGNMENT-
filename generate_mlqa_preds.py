# generate_mlqa_preds.py
import json
import torch
import argparse
import os
import sys

# Thêm đường dẫn gốc để import được các module của dự án
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
from phase2_model.model_core import CrossLingualOTModel
from phase3_loss.losses import OTAlignmentLoss
from phase1_dataloader.process_qa_sample import process_qa_sample

def generate_predictions(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = CrossLingualOTModel(model_name=args.model_name).to(device)
    criterion = OTAlignmentLoss(hidden_size=model.hidden_size).to(device)

    # 2. Load Stage 1 (Base/Teacher) if provided
    if args.stage1_ckpt:
        print(f"Loading Stage 1 base from: {args.stage1_ckpt}")
        ckpt_stage1 = torch.load(args.stage1_ckpt, map_location=device)
        
        # Load Backbone
        model.load_state_dict(ckpt_stage1.get("model_state", ckpt_stage1), strict=False)
        
        # Load QA Head (Vô cùng quan trọng)
        if "criterion_state" in ckpt_stage1 and ckpt_stage1["criterion_state"] is not None:
            criterion.load_state_dict(ckpt_stage1["criterion_state"], strict=False)
        
        print("Applying LoRA to model...")
        model.apply_lora()
        model.to(device)

    # 3. Load Stage 2 (LoRA Checkpoint)
    print(f"Loading Stage 2 weights from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device)
    
    # Load Backbone LoRA weights
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    
    # Load QA Head Stage 2 (Vô cùng quan trọng để không bị rớt điểm)
    if "criterion_state" in ckpt and ckpt["criterion_state"] is not None:
        criterion.load_state_dict(ckpt["criterion_state"], strict=False)
        print("Successfully loaded QA Head weights!")
    else:
        print("WARNING: No criterion_state found in Stage 2 checkpoint!")
    
    model.eval()
    criterion.eval()

    # 4. Đọc file MLQA JSON nguyên bản
    print(f"Reading MLQA data from {args.eval_file}...")
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        mlqa_data = json.load(f)['data']

    predictions = {}
    total_samples = 0
    
    # 5. Quét từng câu hỏi để trích xuất đáp án
    print("Generating predictions...")
    with torch.no_grad():
        for article in mlqa_data:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    qa_id = qa['id']
                    question = qa['question']

                    # Tokenize
                    input_ids, attn_mask, _, _, q_end = process_qa_sample(
                        question=question, context=context, answer=None,
                        tokenizer=tokenizer, max_length=args.max_length, doc_stride=128
                    )
                    input_ids = input_ids.unsqueeze(0).to(device)
                    attn_mask = attn_mask.unsqueeze(0).to(device)
                    q_end_val = q_end.item()

                    # Forward pass
                    out = model.backbone(input_ids, attn_mask)
                    stacked = torch.stack([out.hidden_states[i] for i in [6, 7, 8, 9]], dim=0)
                    weights = torch.softmax(model.layer_weights, dim=0).view(4, 1, 1, 1)
                    hidden = (stacked * weights).sum(dim=0)

                    q_emb = hidden[:, :q_end_val, :]
                    q_mask = torch.zeros(1, q_end_val, dtype=torch.bool, device=device)

                    start_logits, end_logits, _ = criterion.qa_head(hidden, q_emb, q_mask)

                    # Masking padding
                    padding_mask = (attn_mask[0] == 0)
                    start_logits[0].masked_fill_(padding_mask, float('-inf'))
                    end_logits[0].masked_fill_(padding_mask, float('-inf'))
                    
                    # Masking question tokens
                    question_mask = torch.arange(start_logits.size(1), device=device) <= q_end_val
                    
                    # ĐỔI THÀNH TRUE ĐỂ MASK LUÔN [CLS], ÉP MODEL PHẢI TRẢ LỜI
                    question_mask[0] = True 
                    
                    start_logits[0].masked_fill_(question_mask, float('-inf'))
                    end_logits[0].masked_fill_(question_mask, float('-inf'))

                    # Trích xuất ranh giới
                    MAX_ANSWER_LEN = 30
                    start_idx = start_logits[0].argmax().item()
                    end_logits_masked = end_logits[0].clone()
                    end_logits_masked[:start_idx] = float('-inf')
                    end_logits_masked[start_idx + MAX_ANSWER_LEN:] = float('-inf')
                    end_idx = end_logits_masked.argmax().item()

                    pred_ids = input_ids[0][start_idx: end_idx + 1]
                    pred_span = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

                    predictions[qa_id] = pred_span
                    total_samples += 1
                    
                    if total_samples % 500 == 0:
                        print(f"  Processed {total_samples} samples...")

    # 6. Lưu ra file JSON
    with open(args.output_pred_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    print(f"✅ Saved {total_samples} predictions to {args.output_pred_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--stage1_ckpt", type=str, default=None)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--output_pred_file", type=str, default="mlqa_preds.json")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--max_length", type=int, default=384)
    args = parser.parse_args()
    generate_predictions(args)