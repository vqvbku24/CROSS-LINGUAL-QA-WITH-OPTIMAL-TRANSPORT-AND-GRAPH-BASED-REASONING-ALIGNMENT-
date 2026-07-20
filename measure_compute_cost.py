"""
Compute-cost measurement snippet for Appendix B / Table 6.

Usage: wrap your existing training step (forward + backward + optimizer.step())
for each of the three configurations (Full Fine-Tuning, Vanilla KD, Ours) and
call `measure(step_fn, n_warmup=5, n_measure=30)` for each. Report the mean
wall-clock time/step and peak VRAM in Table 6.

Only requires a few dozen steps -- no full training run needed, since per-step
cost is deterministic given fixed batch size / seq length (already fixed in
your setup: batch size 128, seq length 256).
"""
import time
import torch


def measure(step_fn, n_warmup: int = 5, n_measure: int = 30, device: int = 0):
    """
    step_fn: a zero-argument callable that runs ONE training step
              (forward + backward + optimizer.step() + optimizer.zero_grad()),
              using a fixed batch already moved to device.
    Returns: dict with mean/std step time (seconds) and peak VRAM (GB).
    """
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    # Warmup (exclude from timing -- first steps include CUDA kernel compilation,
    # cudnn autotune, and allocator warmup, which is not representative)
    for _ in range(n_warmup):
        step_fn()
    torch.cuda.synchronize(device)

    times = []
    for _ in range(n_measure):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        step_fn()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    mean_t = sum(times) / len(times)
    std_t = (sum((t - mean_t) ** 2 for t in times) / len(times)) ** 0.5

    return {
        "mean_step_time_s": round(mean_t, 4),
        "std_step_time_s": round(std_t, 4),
        "peak_vram_gb": round(peak_bytes / (1024 ** 3), 2),
        "throughput_samples_per_s": round(128 / mean_t, 2),  # batch size 128
    }


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print(__doc__)
    print("Example (pseudo-code, adapt step_fn to your actual training loop):")
    print(
        """
    def step_fn_full_ft():
        optimizer.zero_grad()
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optimizer.step()

    result = measure(step_fn_full_ft)
    print(result)
    # -> fill into Table 6: Framework Variant | Trainable Params | Peak VRAM (GB) | Throughput (samples/s)
    """
    )