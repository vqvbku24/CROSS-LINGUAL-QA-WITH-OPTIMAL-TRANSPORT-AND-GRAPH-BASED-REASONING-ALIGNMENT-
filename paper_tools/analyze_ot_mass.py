import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from data.xquad_loader import load_xquad_pairs
from phase1_dataloader.process_qa_sample import process_qa_sample
from phase2_model.model_core import CrossLingualOTModel
from phase3_loss.losses import OTAlignmentLoss, sinkhorn_masked
from gpu_utils import get_model
from paper_tools.exporter import _is_stage2_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze OT Mass on Answer Tokens")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_vi", type=str, default="dataset/xquad.vi.json")
    parser.add_argument("--dataset_en", type=str, default="dataset/xquad.en.json")
    parser.add_argument("--limit", type=int, default=180, help="Number of samples to evaluate")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load checkpoint and model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config", {})
    model_name = config.get("model_name", "xlm-roberta-base")
    epsilon = config.get("epsilon", 0.03)
    sinkhorn_iters = config.get("sinkhorn_iters", 100)
    
    is_stage2 = _is_stage2_checkpoint(ckpt)
    
    model = CrossLingualOTModel(model_name=model_name, compute_cost_matrix=False)
    criterion = OTAlignmentLoss(hidden_size=model.hidden_size)

    if is_stage2:
        stage1_path = config.get("stage1_ckpt", "")
        if stage1_path and not os.path.isabs(stage1_path):
            stage1_path = os.path.join(str(Path(__file__).parent.parent), stage1_path)
        if stage1_path and os.path.exists(stage1_path):
            s1 = torch.load(stage1_path, map_location=device)
            model.load_state_dict(s1["model_state"], strict=True)
        model.apply_lora()
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        model.load_state_dict(ckpt["model_state"], strict=True)
        
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # 2. Load dataset
    print(f"Loading pairs from {args.dataset_vi}")
    all_pairs = load_xquad_pairs(args.dataset_vi, args.dataset_en)
    
    # Standard split: last 180 are val
    val_pairs = all_pairs[1010:]
    if args.limit and args.limit < len(val_pairs):
        val_pairs = val_pairs[:args.limit]
        
    print(f"Analyzing {len(val_pairs)} validation pairs...")

    masses = []
    expected_masses = []

    with torch.no_grad():
        for pair in tqdm(val_pairs):
            # Tokenize EN
            en_ids, en_mask, en_start, en_end, _ = process_qa_sample(
                question=pair["question_en"],
                context=pair["context_en"],
                answer=pair["answer_en"],
                tokenizer=tokenizer, max_length=384, doc_stride=128
            )
            
            # Tokenize VI WITH TRUE ANSWER to get vi_start, vi_end
            vi_ids, vi_mask, vi_start, vi_end, _ = process_qa_sample(
                question=pair["question_vi"],
                context=pair["context_vi"],
                answer=pair["answer_vi"],
                tokenizer=tokenizer, max_length=384, doc_stride=128
            )
            
            if en_start == 0 and en_end == 0: continue # no answer
            if vi_start == 0 and vi_end == 0: continue # no answer
            
            batch = {
                "en_input_ids": en_ids.unsqueeze(0).to(device),
                "en_attention_mask": en_mask.unsqueeze(0).to(device),
                "vi_input_ids": vi_ids.unsqueeze(0).to(device),
                "vi_attention_mask": vi_mask.unsqueeze(0).to(device)
            }
            
            # Forward
            if is_stage2:
                with get_model(model).backbone.disable_adapter():
                    en_out = model(batch, branch="en")
                    vi_out = model(batch, branch="vi")
            else:
                en_out = model(batch, branch="en")
                vi_out = model(batch, branch="vi")
                
            h_en = en_out["hidden"]
            m_en = ~en_out["en_pad_mask"]
            h_vi = vi_out["hidden"]
            m_vi = ~vi_out["vi_pad_mask"]
            
            # OT
            gamma_list, _ = sinkhorn_masked(h_en, h_vi, m_en, m_vi, epsilon=epsilon, n_iters=sinkhorn_iters)
            g = gamma_list[0].cpu().numpy() # [N_en, N_vi]
            
            en_seq_len = g.shape[0]
            vi_seq_len = g.shape[1]
            
            # Ensure within bounds
            es = min(en_start.item(), en_seq_len - 1)
            ee = min(en_end.item(), en_seq_len - 1)
            vs = min(vi_start.item(), vi_seq_len - 1)
            ve = min(vi_end.item(), vi_seq_len - 1)
            
            if es <= ee and vs <= ve:
                mass = g[es:ee+1, vs:ve+1].sum()
                expected_mass = ((ee - es + 1) * (ve - vs + 1)) / (en_seq_len * vi_seq_len)
                
                masses.append(mass)
                expected_masses.append(expected_mass)

    if not masses:
        print("No valid answer spans found.")
        return

    avg_mass = np.mean(masses)
    avg_expected = np.mean(expected_masses)
    ratio = avg_mass / avg_expected if avg_expected > 0 else 0

    print("-" * 50)
    print(f"RESULTS ACROSS {len(masses)} SAMPLES")
    print("-" * 50)
    print(f"Avg mass in answer box:       {avg_mass:.6f}")
    print(f"Expected mass (uniform):      {avg_expected:.6f}")
    print(f"Concentration Ratio:          {ratio:.2f}x")
    print("-" * 50)
    if ratio < 1.5:
        print("[WARNING] The OT plan is NOT effectively routing mass to the target answer tokens.")
        print("          This confirms the necessity of TASKAWARE_OT_SPEC (Span Alignment Loss).")
    else:
        print("[OK] The OT plan routes significantly more mass to the target answer tokens.")

if __name__ == "__main__":
    main()
