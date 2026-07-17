BENCHMARK_SPEC_EFFICIENCY_TABLE.md
Objective
Replace the fabricated/estimated numbers in tab:complexity (Trainable Params, Peak VRAM,
Throughput) with numbers measured directly via PyTorch instrumentation, run on a single,
consistent GPU (final reported numbers must come from A100 — see Priority table).
Priority Table
PriorityTaskWhyP0Implement measure_trainable_params()Cheapest, no GPU dependency, must be correct first — sanity-checks the claimed 3.2M for OursP0Implement measure_peak_vram() with proper reset_peak_memory_stats + synchronizeCurrent table's VRAM numbers have no instrumentation behind themP0Implement measure_throughput() with warmup + synchronizeSame as above — no warmup means numbers are noiseP1Wire up all 3 variants (Full FT, Vanilla KD, Coordinated) through the SAME entrypoint using existing model class + config flagsEnsures apples-to-apples comparison, not 3 separately-run guessesP1Fix batch used for measurement to a single real batch (see below), reused across all 3 variantsFairness — different batches = different padding = different VRAM/throughputP2Add GPU name + n_trials + mean±std to outputPrevents future you from mixing T4 dev numbers into the A100 paper tableP2Dump results to results/efficiency_benchmark.jsonTraceability — paper table should cite this file, not be typed by handFile Location
Create new file: benchmarks/measure_efficiency.py
(Do not modify existing model/training files except to import from them — see No-Touch Zones.)

Problem (current state)
tab:complexity in the paper has no corresponding script or log in the repo. Numbers appear
to be either estimated by hand or copied from an unrelated run. There is no evidence of:

torch.cuda.reset_peak_memory_stats() being called before measurement
warmup steps before timing
torch.cuda.synchronize() calls around the timed region
a fixed, shared batch across the 3 variants
Exact Implementation
# benchmarks/measure_efficiency.py
"""
Usage:
    python benchmarks/measure_efficiency.py --variant full_ft
    python benchmarks/measure_efficiency.py --variant vanilla_kd
    python benchmarks/measure_efficiency.py --variant coordinated

Run each variant as a SEPARATE process (not in a loop in the same process) —
CUDA peak memory stats and cached allocators leak across models otherwise and
will corrupt the VRAM numbers for whichever variant runs second/third.
"""
import argparse
import json
import time
import torch

# TODO(coding agent): import your existing model builder + config here, e.g.:
# from src.model import build_model
# from src.config import get_config
# from src.data import get_dataloader

RESULTS_PATH = "results/efficiency_benchmark.json"
BATCH_SIZE = 128
SEQ_LEN = 256
N_WARMUP = 5
N_TRIALS = 20


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_fixed_batch(dataloader, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, device="cuda"):
    """
    Pull ONE real batch from the existing dataloader, truncate/pad to seq_len,
    move to device, and reuse this exact same batch object across all 3 variants.
    Do NOT resample per variant — that breaks the fairness of the comparison.
    """
    batch = next(iter(dataloader))
    # TODO(coding agent): adapt keys to your actual batch dict
    # (e.g. input_ids, attention_mask, start_positions, end_positions, labels)
    batch = {k: v[:batch_size, :seq_len].to(device) if torch.is_tensor(v) else v
             for k, v in batch.items()}
    return batch


def measure_peak_vram(model, batch, device="cuda") -> float:
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    out = model(**batch)
    loss = out.loss if hasattr(out, "loss") else out["loss"]
    loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)

    return torch.cuda.max_memory_allocated(device) / 1e9  # GB


def measure_throughput(model, batch, n_warmup=N_WARMUP, n_trials=N_TRIALS,
                        device="cuda") -> dict:
    model.train()
    for _ in range(n_warmup):
        out = model(**batch)
        loss = out.loss if hasattr(out, "loss") else out["loss"]
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)

    per_trial_samples_per_sec = []
    batch_size = next(iter(batch.values())).shape[0]

    for _ in range(n_trials):
        torch.cuda.synchronize(device)
        start = time.perf_counter()

        out = model(**batch)
        loss = out.loss if hasattr(out, "loss") else out["loss"]
        loss.backward()
        model.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        per_trial_samples_per_sec.append(batch_size / elapsed)

    import statistics
    return {
        "mean_samples_per_sec": statistics.mean(per_trial_samples_per_sec),
        "std_samples_per_sec": statistics.stdev(per_trial_samples_per_sec),
        "n_trials": n_trials,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                         choices=["full_ft", "vanilla_kd", "coordinated"])
    args = parser.parse_args()

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(device)

    # TODO(coding agent): replace with your actual config + model construction,
    # using the same flags mentioned in memory (e.g. use_lora=True/False,
    # use_kd=True/False) to select the variant.
    # config = get_config(variant=args.variant)
    # model = build_model(config).to(device)
    # dataloader = get_dataloader(config)

    # batch = get_fixed_batch(dataloader, device=device)

    trainable_params = count_trainable_params(model)
    peak_vram_gb = measure_peak_vram(model, batch, device=device)
    throughput = measure_throughput(model, batch, device=device)

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

    import os
    os.makedirs("results", exist_ok=True)
    existing = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            existing = json.load(f)
    existing = [r for r in existing if r["variant"] != args.variant]  # replace stale entry
    existing.append(result)
    with open(RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)


if __name__ == "__main__":
    main()
No-Touch Zones
Do NOT modify existing model class, training loop, or config files — only import from them.
Do NOT modify losses.py or anything touched by prior specs (BUGFIX_SPEC_LREG.md,ABLATION_SPEC_MARGIN.md, TASKAWARE_OT_SPEC.md).
Do NOT run all 3 variants in the same Python process / same script invocation — CUDA memory
stats are process-global and will contaminate results across variants.
Do NOT use results/efficiency_benchmark.json entries produced on T4 as final paper numbers —
the JSON should carry the "gpu" field precisely so this is auditable. Only A100 entries go
in the paper table.
Acceptance Criteria
Running the script 3 times (once per --variant) on the same GPU in the same session producesresults/efficiency_benchmark.json with 3 entries, each carrying the same "gpu" string.
trainable_params for coordinated variant should land close to the claimed 3.2M — if it's
off by more than ~10%, flag it (likely a LoRA target_modules / rank mismatch, not a benchmark
script bug).
throughput_samples_per_sec_std should be reported in the paper table as mean ± std, not
just mean, once you have real numbers — a single unqualified number is part of what made the
current table look fabricated.

đây là script để làm cái table 6, bạn có thể note kĩ lại cho agent là các framework bỏ cái gì lấy cái gì rồi chỗ nào bạn muốn kĩ hơn có thể làm lại không