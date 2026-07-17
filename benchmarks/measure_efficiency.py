"""
Usage:
    python benchmarks/measure_efficiency.py --variant full_ft
    python benchmarks/measure_efficiency.py --variant vanilla_kd
    python benchmarks/measure_efficiency.py --variant coordinated
"""
import os
import sys
import argparse
import json
import time
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from peft import LoraConfig, get_peft_model
from squad_parallel_loader import create_squad_parallel_dataloaders
from phase3_loss.losses import sinkhorn_masked, compute_span_loss, compute_pure_margin_loss, compute_reg_loss

RESULTS_PATH = "results/efficiency_benchmark.json"
BATCH_SIZE = 128
SEQ_LEN = 256
MIN_LEN = 16
N_WARMUP = 5
N_TRIALS = 20

def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_fixed_batch(dataloader, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, device="cuda"):
    """
    Pull ONE real batch from the existing dataloader, truncate/pad to seq_len,
    move to device, and reuse this exact same batch object across all 3 variants.
    """
    batch = next(iter(dataloader))
    
    # Prefix keys could be en_ or vi_, we'll just truncate all tensor values
    # Dynamic sequence truncation based on en_attention_mask if available, else attention_mask
    mask_key = "en_attention_mask" if "en_attention_mask" in batch else "attention_mask"
    
    max_valid_len = batch[mask_key].sum(dim=1).max().item() if mask_key in batch else seq_len
    actual_seq_len = max(max_valid_len, MIN_LEN)
    actual_seq_len = min(actual_seq_len, seq_len)
    
    truncated_batch = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if v.dim() > 1:
                truncated_batch[k] = v[:batch_size, :actual_seq_len].to(device)
            else:
                truncated_batch[k] = v[:batch_size].to(device)
        else:
            truncated_batch[k] = v
            
    return truncated_batch

def compute_vanilla_kd_loss(student_outputs, teacher_outputs, temperature=2.0):
    loss_qa = student_outputs.loss if hasattr(student_outputs, "loss") and student_outputs.loss is not None else 0.0
    
    student_start_log_probs = F.log_softmax(student_outputs.start_logits / temperature, dim=-1)
    teacher_start_probs = F.softmax(teacher_outputs.start_logits / temperature, dim=-1)
    loss_start = F.kl_div(student_start_log_probs, teacher_start_probs, reduction='batchmean') * (temperature ** 2)

    student_end_log_probs = F.log_softmax(student_outputs.end_logits / temperature, dim=-1)
    teacher_end_probs = F.softmax(teacher_outputs.end_logits / temperature, dim=-1)
    loss_end = F.kl_div(student_end_log_probs, teacher_end_probs, reduction='batchmean') * (temperature ** 2)

    return loss_qa + (loss_start + loss_end) / 2.0

def forward_backward_pass(variant, model, teacher_model, batch):
    # Adapt batch keys for standard HF model if using parallel dataloader
    hf_batch = {
        "input_ids": batch.get("en_input_ids", batch.get("input_ids")),
        "attention_mask": batch.get("en_attention_mask", batch.get("attention_mask")),
        "start_positions": batch.get("en_start_position", batch.get("start_positions")),
        "end_positions": batch.get("en_end_position", batch.get("end_positions")),
        "output_hidden_states": True
    }
    
    if variant == "full_ft":
        outputs = model(**hf_batch)
        loss = outputs.loss
        loss.backward()
        
    elif variant == "vanilla_kd":
        with torch.no_grad():
            teacher_outputs = teacher_model(**hf_batch)
        student_outputs = model(**hf_batch)
        
        loss = compute_vanilla_kd_loss(student_outputs, teacher_outputs)
        loss.backward()
        
    elif variant == "coordinated":
        with torch.no_grad():
            teacher_outputs = teacher_model(**hf_batch)
        student_outputs = model(**hf_batch)
        
        loss_qa = student_outputs.loss if student_outputs.loss is not None else torch.tensor(0.0, device=model.device, requires_grad=True)
        
        student_hidden = student_outputs.hidden_states[-1]
        teacher_hidden = teacher_outputs.hidden_states[-1]
        
        en_mask = ~hf_batch["attention_mask"].bool()
        
        # OT (Sinkhorn)
        # sinkhorn_masked returns gamma_list, C_list
        gamma_list, C_list = sinkhorn_masked(
            teacher_hidden, student_hidden, en_mask, en_mask, epsilon=0.03, n_iters=100
        )
        # Compute L_ot as sum of transport costs
        loss_ot = torch.stack([(g.detach() * c).sum() for g, c in zip(gamma_list, C_list)]).mean() if gamma_list else torch.tensor(0.0, device=model.device, requires_grad=True)
        
        # Span Loss
        loss_span = compute_span_loss(
            gamma_list, 
            student_outputs.start_logits, student_outputs.end_logits,
            teacher_outputs.start_logits, teacher_outputs.end_logits,
            en_mask, en_mask
        )
        
        # L2 Regularization
        loss_reg = compute_reg_loss(student_hidden, teacher_hidden, en_mask)
        
        # Margin Loss
        en_start = hf_batch["start_positions"]
        en_end = hf_batch["end_positions"]
        loss_margin = compute_pure_margin_loss(
            en_start, en_end,
            student_outputs.start_logits, student_outputs.end_logits,
            teacher_outputs.start_logits, teacher_outputs.end_logits,
            en_mask, en_mask
        )
        
        loss_total = loss_qa + loss_ot + loss_span + loss_reg + loss_margin
        loss_total.backward()

def measure_peak_vram(variant, model, teacher_model, batch, device="cuda") -> float:
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    forward_backward_pass(variant, model, teacher_model, batch)
    model.zero_grad(set_to_none=True)
    
    torch.cuda.synchronize(device)

    return torch.cuda.max_memory_allocated(device) / 1e9  # GB

def measure_throughput(variant, model, teacher_model, batch, n_warmup=N_WARMUP, n_trials=N_TRIALS, device="cuda") -> dict:
    model.train()
    for _ in range(n_warmup):
        forward_backward_pass(variant, model, teacher_model, batch)
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)

    per_trial_samples_per_sec = []
    # Deduce batch size actually used
    hf_batch_size = batch.get("en_input_ids", batch.get("input_ids")).shape[0]

    for _ in range(n_trials):
        torch.cuda.synchronize(device)
        start = time.perf_counter()

        forward_backward_pass(variant, model, teacher_model, batch)
        model.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        per_trial_samples_per_sec.append(hf_batch_size / elapsed)

    import statistics
    return {
        "mean_samples_per_sec": statistics.mean(per_trial_samples_per_sec),
        "std_samples_per_sec": statistics.stdev(per_trial_samples_per_sec) if n_trials > 1 else 0.0,
        "n_trials": n_trials,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["full_ft", "vanilla_kd", "coordinated"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(device) if device == "cuda" else "CPU"

    # Construct model variants
    model_name = "xlm-roberta-base"
    
    teacher_model = None
    if args.variant == "full_ft":
        model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
        model.train()
        for p in model.parameters():
            p.requires_grad = True
    else:
        teacher_model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
            
        base_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        config = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "key", "value", "dense"], lora_dropout=0.05)
        model = get_peft_model(base_model, config).to(device)
        model.train()

    # Dataloader
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # create_squad_parallel_dataloaders will use defaults or local paths. We might need a small dummy batch if paths don't exist, but assuming data is in dataset/
    try:
        dataloader, _, _ = create_squad_parallel_dataloaders(tokenizer, batch_size=BATCH_SIZE, max_length=SEQ_LEN)
        batch = get_fixed_batch(dataloader, device=device)
    except Exception as e:
        print(f"Error loading real dataloader, falling back to dummy batch. Error: {e}")
        # Dummy batch for testing if real data missing
        batch = {
            "en_input_ids": torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN)).to(device),
            "en_attention_mask": torch.ones(BATCH_SIZE, SEQ_LEN).to(device),
            "en_start_position": torch.zeros(BATCH_SIZE, dtype=torch.long).to(device),
            "en_end_position": torch.ones(BATCH_SIZE, dtype=torch.long).to(device)
        }

    trainable_params = count_trainable_params(model)
    print(f"[{args.variant}] Trainable params: {trainable_params}")
    
    if device == "cuda":
        peak_vram_gb = measure_peak_vram(args.variant, model, teacher_model, batch, device=device)
        throughput = measure_throughput(args.variant, model, teacher_model, batch, device=device)
    else:
        peak_vram_gb = 0.0
        throughput = measure_throughput(args.variant, model, teacher_model, batch, n_trials=2, device=device)

    result = {
        "variant": args.variant,
        "gpu": gpu_name,
        "trainable_params": trainable_params,
        "peak_vram_gb": round(peak_vram_gb, 2),
        "throughput_samples_per_sec_mean": round(throughput["mean_samples_per_sec"], 1),
        "throughput_samples_per_sec_std": round(throughput["std_samples_per_sec"], 2),
        "n_trials": throughput["n_trials"],
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
    }

    print(json.dumps(result, indent=2))

    os.makedirs("results", exist_ok=True)
    existing = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            try:
                existing = json.load(f)
            except:
                existing = []
    
    existing = [r for r in existing if r.get("variant") != args.variant]
    existing.append(result)
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)

if __name__ == "__main__":
    main()
