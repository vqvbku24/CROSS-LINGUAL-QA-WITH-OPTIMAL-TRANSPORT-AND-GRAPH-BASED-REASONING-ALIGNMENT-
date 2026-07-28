"""
export_many_samples.py — Run paper_tools/exporter.py once per sample_index,
saving each sample's output into its own subdirectory:

    <output_root>/sample_0000/hidden_before.npy, hidden_after.npy, gamma_fig5.npy,
                                langs.npy, labels.npy, ...
    <output_root>/sample_0001/...
    ...

This layout is exactly what aggregate_alignment_stats.py expects
(--root_dir <output_root>).

exporter.py itself is NOT modified — this just calls it as a subprocess with a
different --output_dir per sample, so it stays safe to use for the original
single-sample Figure 5 workflow too.

Usage:
    python export_many_samples.py \
        --checkpoint checkpoints/stage2_lora_best.pt \
        --dataset    dataset/xquad.vi.json \
        --output_root paper_tools/export_multi \
        --sample_indices 0-49          # range
        # or: --sample_indices 0,3,7,12  # explicit list
        --alpha 0.5

Then:
    python aggregate_alignment_stats.py --root_dir paper_tools/export_multi
"""
import argparse
import subprocess
import sys
from pathlib import Path


def parse_sample_indices(spec: str):
    """Accepts '0-49' (inclusive range) or '0,3,7,12' (explicit comma list)."""
    spec = spec.strip()
    if '-' in spec and ',' not in spec:
        start, end = spec.split('-')
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in spec.split(',') if x.strip() != '']


def main():
    parser = argparse.ArgumentParser(
        description="Run exporter.py across many samples for aggregate alignment stats")
    parser.add_argument("--exporter_path", type=str, default="paper_tools/exporter.py",
                         help="Path to exporter.py")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True,
                         help="Root directory; one subdirectory per sample will be created here")
    parser.add_argument("--sample_indices", type=str, required=True,
                         help="Range '0-49' or comma list '0,3,7,12'")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--stage", type=str, default="auto", choices=["1", "2", "auto"])
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                         help="Passed through to exporter.py. Required for Stage 2 "
                              "checkpoints so the backbone doesn't silently fall back "
                              "to random weights.")
    parser.add_argument("--skip_existing", action="store_true",
                         help="Skip a sample if its output subdirectory already has hidden_before.npy")
    parser.add_argument("--continue_on_error", action="store_true",
                         help="Keep going if one sample fails (e.g. out-of-range index) "
                              "instead of stopping the whole run")
    parser.add_argument("--num_gpus", type=int, default=1,
                         help="Number of GPUs to run in parallel. Will set CUDA_VISIBLE_DEVICES cyclically.")
    args = parser.parse_args()

    indices = parse_sample_indices(args.sample_indices)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {len(indices)} samples to '{output_root}' using {args.num_gpus} GPUs...")
    n_ok, n_skipped, n_failed = 0, 0, 0

    import concurrent.futures
    import queue
    import os

    # Read available GPUs from environment, or fallback to 0...N-1
    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if available_gpus.strip():
        gpu_list = [g.strip() for g in available_gpus.split(",") if g.strip()]
        # Take up to num_gpus from the available list
        gpu_list = gpu_list[:args.num_gpus]
    else:
        gpu_list = [str(i) for i in range(args.num_gpus)]

    if len(gpu_list) < args.num_gpus:
        print(f"[warn] Requested {args.num_gpus} GPUs but only {len(gpu_list)} are available in CUDA_VISIBLE_DEVICES.")
        args.num_gpus = len(gpu_list)

    gpu_queue = queue.Queue()
    for gpu_id in gpu_list:
        gpu_queue.put(gpu_id)

    def process_sample(idx):
        sample_dir = output_root / f"sample_{idx:04d}"

        if args.skip_existing and (sample_dir / "hidden_before.npy").exists():
            return idx, 'skipped', ""

        cmd = [
            sys.executable, args.exporter_path,
            "--checkpoint", args.checkpoint,
            "--dataset", args.dataset,
            "--output_dir", str(sample_dir),
            "--sample_index", str(idx),
            "--alpha", str(args.alpha),
            "--stage", args.stage,
        ]
        if args.stage1_checkpoint:
            cmd += ["--stage1_checkpoint", args.stage1_checkpoint]

        gpu_id = gpu_queue.get()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                return idx, 'failed', result.stdout + "\n" + result.stderr
            return idx, 'ok', ""
        finally:
            gpu_queue.put(gpu_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_gpus) as executor:
        futures = {executor.submit(process_sample, idx): idx for idx in indices}
        for future in concurrent.futures.as_completed(futures):
            idx, status, err_output = future.result()
            if status == 'skipped':
                print(f"[skip] sample_{idx:04d} already exported")
                n_skipped += 1
            elif status == 'ok':
                print(f"[ok] sample_{idx:04d} exported successfully")
                n_ok += 1
            elif status == 'failed':
                print(f"[error] sample_{idx:04d} failed!")
                print(err_output)
                n_failed += 1
                if not args.continue_on_error:
                    print("Stopping due to error. Pass --continue_on_error to skip failures.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    sys.exit(1)

    print(f"\nDone. ok={n_ok}, skipped={n_skipped}, failed={n_failed}, "
          f"total requested={len(indices)}")
    print(f"Next: python aggregate_alignment_stats.py --root_dir {output_root}")


if __name__ == '__main__':
    main()