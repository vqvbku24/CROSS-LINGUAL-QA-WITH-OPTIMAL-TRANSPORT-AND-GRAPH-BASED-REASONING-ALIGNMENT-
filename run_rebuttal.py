#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_rebuttal.py

MASTER RUNNER for ACL 2025 Rebuttal Experiments.
Coordinates Tier 1 (Ablation M3-AR, M3-HI) and Tier 2 (mmBERT Generalization).

Usage:
  # Run Tier 1: M3 Arabic & Hindi + XQuAD ablation table
  python run_rebuttal.py --tier1

  # Run Tier 2: mmBERT smoke test + Stage 1 EN + M2-VI + M5-VI + mmBERT table
  python run_rebuttal.py --tier2

  # Run all rebuttal experiments sequentially:
  python run_rebuttal.py --all

  # Generate summary tables only from existing checkpoints:
  python run_rebuttal.py --eval_only

  # Dry run (print commands without running):
  python run_rebuttal.py --tier1 --dry_run
"""

import os
import sys
import time
import argparse
import subprocess
import shutil
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Ensure console handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def log_msg(msg: str, log_file=None):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    formatted = f"[{ts}] {msg}"
    try:
        print(formatted, flush=True)
    except UnicodeEncodeError:
        print(formatted.encode("ascii", errors="replace").decode("ascii"), flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")


def check_and_download_stage1_ckpt(log_file=None):
    ckpt_path = os.path.join(BASE_DIR, "checkpoints", "stage1_squad_best.pt")
    if os.path.exists(ckpt_path):
        log_msg(f"Stage 1 checkpoint verified: {ckpt_path}", log_file)
        return ckpt_path

    log_msg("Stage 1 checkpoint not found locally. Attempting to download from HF Hub...", log_file)
    os.makedirs(os.path.join(BASE_DIR, "checkpoints"), exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        hf_token = os.environ.get("HF_TOKEN", None)
        token_file = os.path.join(BASE_DIR, ".hf_token")
        if not hf_token and os.path.exists(token_file):
            with open(token_file, "r") as f:
                hf_token = f.read().strip()

        downloaded_path = hf_hub_download(
            repo_id="vinhvo1205/Sinkhorn_2_stages",
            filename="checkpoints/stage1_squad_best.pt",
            token=hf_token
        )
        shutil.copy(downloaded_path, ckpt_path)
        log_msg(f"Successfully downloaded Stage 1 checkpoint to: {ckpt_path}", log_file)
        return ckpt_path
    except Exception as e:
        log_msg(f"Warning: Could not download stage1 checkpoint automatically: {e}", log_file)
        log_msg("Please make sure 'checkpoints/stage1_squad_best.pt' is present.", log_file)
        return ckpt_path


def get_gpu_runner(single_gpu_force: bool = False) -> tuple[str, list[str]]:
    """
    Detects whether to use torchrun (multi-GPU) or python (single-GPU / Colab).
    Returns (runner_type, base_cmd_prefix).
    """
    try:
        import torch
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n_gpus = 0

    if n_gpus > 1 and not single_gpu_force:
        return "torchrun", ["torchrun", f"--nproc_per_node={n_gpus}"]
    else:
        return "python", [sys.executable]


def run_command(cmd_list: list[str], dry_run: bool = False, log_file: str = None) -> int:
    cmd_str = " ".join(str(c) for c in cmd_list)
    log_msg(f"EXEC: {cmd_str}", log_file)
    if dry_run:
        return 0

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    start_time = time.time()
    process = subprocess.Popen(
        cmd_list,
        cwd=BASE_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    with open(log_file, "a", encoding="utf-8") if log_file else open(os.devnull, "w") as lf:
        for line in process.stdout:
            print(line, end="", flush=True)
            lf.write(line)

    process.wait()
    duration = time.time() - start_time
    log_msg(f"Command completed with code {process.returncode} in {duration:.1f}s", log_file)
    return process.returncode


# ──────────────────────────────────────────────────────────────
# TIER 1: Ablation Completion (M3-AR, M3-HI)
# ──────────────────────────────────────────────────────────────

def run_tier1(args, log_file: str):
    log_msg("\n" + "=" * 70, log_file)
    log_msg("  STARTING TIER 1: Vá lỗ hổng Ablation (Arabic & Hindi M3)", log_file)
    log_msg("=" * 70, log_file)

    st1_ckpt = check_and_download_stage1_ckpt(log_file)
    _, runner = get_gpu_runner(args.single_gpu)

    # 1. Train M3-AR (Arabic: OT + Span, Margin = 0)
    out_dir_ar = os.path.join(BASE_DIR, "checkpoint_stage2_ar", "m3_ot_span")
    best_ckpt_ar = os.path.join(out_dir_ar, "stage2_ar_best.pt")

    if args.skip_existing and os.path.exists(best_ckpt_ar):
        log_msg(f"Skip existing: Found M3-AR checkpoint at {best_ckpt_ar}", log_file)
    else:
        log_msg("▶️ Training M3-AR (Arabic: OT=0.5, Span=1.0, Margin=0.0)...", log_file)
        cmd_ar = runner + [
            os.path.join(BASE_DIR, "arabic", "train_stage2_ar.py"),
            "--stage1_ckpt", st1_ckpt,
            "--output_dir", out_dir_ar,
            "--lambda_ot", "0.5",
            "--lambda_span", "1.0",
            "--lambda_margin", "0.0",
            "--lambda_reg", "50.0",
            "--batch_size", str(args.batch_size_stage2),
            "--max_epochs", str(args.stage2_epochs),
            "--seed", str(args.seed),
        ]
        ret = run_command(cmd_ar, dry_run=args.dry_run, log_file=log_file)
        if ret != 0:
            log_msg("❌ ERROR: Training M3-AR failed!", log_file)
            if not args.ignore_errors:
                return ret

    # 2. Train M3-HI (Hindi: OT + Span, Margin = 0)
    out_dir_hi = os.path.join(BASE_DIR, "checkpoint_stage2_hi", "m3_ot_span")
    best_ckpt_hi = os.path.join(out_dir_hi, "stage2_hi_best.pt")

    if args.skip_existing and os.path.exists(best_ckpt_hi):
        log_msg(f"Skip existing: Found M3-HI checkpoint at {best_ckpt_hi}", log_file)
    else:
        log_msg("▶️ Training M3-HI (Hindi: OT=0.5, Span=1.0, Margin=0.0)...", log_file)
        cmd_hi = runner + [
            os.path.join(BASE_DIR, "hindi", "train_stage2_hi.py"),
            "--stage1_ckpt", st1_ckpt,
            "--output_dir", out_dir_hi,
            "--lambda_ot", "0.5",
            "--lambda_span", "1.0",
            "--lambda_margin", "0.0",
            "--lambda_reg", "50.0",
            "--batch_size", str(args.batch_size_stage2),
            "--max_epochs", str(args.stage2_epochs),
            "--seed", str(args.seed),
        ]
        ret = run_command(cmd_hi, dry_run=args.dry_run, log_file=log_file)
        if ret != 0:
            log_msg("❌ ERROR: Training M3-HI failed!", log_file)
            if not args.ignore_errors:
                return ret

    # 3. Generate Summary Ablation Table
    log_msg("📊 Generating Tier 1 Summary Table (XQuAD M1-M5)...", log_file)
    cmd_eval = [sys.executable, os.path.join(BASE_DIR, "rebuttal_eval_summary.py"), "--tier1"]
    run_command(cmd_eval, dry_run=args.dry_run, log_file=log_file)

    log_msg("✅ TIER 1 FINISHED!", log_file)
    return 0


# ──────────────────────────────────────────────────────────────
# TIER 2: mmBERT Modern Backbone Generalization
# ──────────────────────────────────────────────────────────────

def run_tier2(args, log_file: str):
    log_msg("\n" + "=" * 70, log_file)
    log_msg("  STARTING TIER 2: mmBERT Generalization (22 Layers)", log_file)
    log_msg("=" * 70, log_file)

    model_name = "jhu-clsp/mmBERT-base"
    _, runner = get_gpu_runner(args.single_gpu)

    # 1. Smoke test
    if not args.skip_smoke_test:
        log_msg("▶️ [Step 1/5] Running mmBERT Compatibility & Smoke Test Suite...", log_file)
        smoke_cmd = [
            sys.executable,
            os.path.join(BASE_DIR, "test_mmbert_smoke.py"),
            "--model_name", model_name,
            "--device", "cuda" if not args.cpu else "cpu"
        ]
        ret = run_command(smoke_cmd, dry_run=args.dry_run, log_file=log_file)
        if ret != 0:
            log_msg("❌ ERROR: mmBERT Smoke Test failed! Check environment.", log_file)
            if not args.ignore_errors:
                return ret
    else:
        log_msg("⏩ Skipped smoke test (--skip_smoke_test specified)", log_file)

    # 2. Stage 1 EN training on SQuAD 2.0
    st1_mmbert_ckpt = os.path.join(BASE_DIR, "checkpoints", "stage1_squad_best_mmbert.pt")
    if args.skip_existing and os.path.exists(st1_mmbert_ckpt):
        log_msg(f"Skip existing: Found mmBERT Stage 1 checkpoint at {st1_mmbert_ckpt}", log_file)
    else:
        log_msg("▶️ [Step 2/5] Training Stage 1 EN on SQuAD 2.0 (mmBERT)...", log_file)
        st1_cmd = runner + [
            os.path.join(BASE_DIR, "train_stage1.py"),
            "--model_name", model_name,
            "--checkpoint_name", "stage1_squad_best_mmbert.pt",
            "--epochs", str(args.stage1_epochs),
            "--batch_size", str(args.batch_size_stage1),
            "--lr", "2e-5",
            "--seed", str(args.seed),
        ]
        ret = run_command(st1_cmd, dry_run=args.dry_run, log_file=log_file)
        if ret != 0:
            log_msg("❌ ERROR: mmBERT Stage 1 training failed!", log_file)
            if not args.ignore_errors:
                return ret

    # 3. Stage 2 M2-VI (OT only: Span=0, Margin=0)
    out_dir_m2 = os.path.join(BASE_DIR, "checkpoint_stage2_mmbert_vi_m2")
    best_m2_ckpt = os.path.join(out_dir_m2, "stage2_best.pt")
    if args.skip_existing and os.path.exists(best_m2_ckpt):
        log_msg(f"Skip existing: Found mmBERT M2 checkpoint at {best_m2_ckpt}", log_file)
    else:
        log_msg("▶️ [Step 3/5] Training Stage 2 mmBERT M2-VI (OT only)...", log_file)
        m2_cmd = runner + [
            os.path.join(BASE_DIR, "train_stage2.py"),
            "--model_name", model_name,
            "--stage1_ckpt", st1_mmbert_ckpt,
            "--output_dir", out_dir_m2,
            "--lambda_ot", "0.5",
            "--lambda_span", "0.0",
            "--lambda_margin", "0.0",
            "--batch_size", str(args.batch_size_stage2),
            "--max_epochs", str(args.stage2_epochs),
            "--seed", str(args.seed),
        ]
        ret = run_command(m2_cmd, dry_run=args.dry_run, log_file=log_file)
        if ret != 0:
            log_msg("❌ ERROR: mmBERT M2 training failed!", log_file)
            if not args.ignore_errors:
                return ret

    # 4. Stage 2 M5-VI (Ours Full: OT=0.5, Span=1.0, Margin=1.0)
    out_dir_m5 = os.path.join(BASE_DIR, "checkpoint_stage2_mmbert_vi_m5")
    best_m5_ckpt = os.path.join(out_dir_m5, "stage2_best.pt")
    if args.skip_existing and os.path.exists(best_m5_ckpt):
        log_msg(f"Skip existing: Found mmBERT M5 checkpoint at {best_m5_ckpt}", log_file)
    else:
        log_msg("▶️ [Step 4/5] Training Stage 2 mmBERT M5-VI (Full Coordinated - Ours)...", log_file)
        m5_cmd = runner + [
            os.path.join(BASE_DIR, "train_stage2.py"),
            "--model_name", model_name,
            "--stage1_ckpt", st1_mmbert_ckpt,
            "--output_dir", out_dir_m5,
            "--lambda_ot", "0.5",
            "--lambda_span", "1.0",
            "--lambda_margin", "1.0",
            "--batch_size", str(args.batch_size_stage2),
            "--max_epochs", str(args.stage2_epochs),
            "--seed", str(args.seed),
        ]
        ret = run_command(m5_cmd, dry_run=args.dry_run, log_file=log_file)
        if ret != 0:
            log_msg("❌ ERROR: mmBERT M5 training failed!", log_file)
            if not args.ignore_errors:
                return ret

    # 5. Generate mmBERT Comparison Table
    log_msg("📊 [Step 5/5] Generating Tier 2 Summary Table (mmBERT)...", log_file)
    cmd_eval = [sys.executable, os.path.join(BASE_DIR, "rebuttal_eval_summary.py"), "--tier2"]
    run_command(cmd_eval, dry_run=args.dry_run, log_file=log_file)

    log_msg("✅ TIER 2 FINISHED!", log_file)
    return 0


# ──────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Master Rebuttal Execution Script")
    parser.add_argument("--tier1", action="store_true", help="Execute Tier 1: Train M3-AR, M3-HI and evaluate")
    parser.add_argument("--tier2", action="store_true", help="Execute Tier 2: mmBERT pipeline (Smoke, S1, M2, M5, eval)")
    parser.add_argument("--all", action="store_true", help="Execute Tier 1 and Tier 2 end-to-end")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate and display tables without training")
    
    # Training configurations
    parser.add_argument("--batch_size_stage1", type=int, default=16, help="Batch size for Stage 1 (default: 16)")
    parser.add_argument("--batch_size_stage2", type=int, default=32, help="Batch size for Stage 2 (default: 32)")
    parser.add_argument("--stage1_epochs", type=int, default=5, help="Epochs for Stage 1 mmBERT (default: 5)")
    parser.add_argument("--stage2_epochs", type=int, default=6, help="Epochs for Stage 2 (default: 6)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Controls
    parser.add_argument("--skip_smoke_test", action="store_true", help="Skip mmBERT smoke test")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip stages where checkpoint already exists")
    parser.add_argument("--single_gpu", action="store_true", help="Force single-GPU python runner instead of torchrun")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--dry_run", action="store_true", help="Print all commands without executing them")
    parser.add_argument("--ignore_errors", action="store_true", help="Continue to next step even if a step fails")

    args = parser.parse_args()

    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    log_file = os.path.join(BASE_DIR, "logs", f"rebuttal_{get_timestamp()}.log")

    log_msg("=" * 70, log_file)
    log_msg("  ACL 2025 REBUTTAL MASTER RUNNER", log_file)
    log_msg(f"  Project Directory: {BASE_DIR}", log_file)
    log_msg(f"  Log File: {log_file}", log_file)
    log_msg("=" * 70, log_file)

    if args.eval_only:
        log_msg("Running Evaluation & Table Generation Only...", log_file)
        cmd_eval = [sys.executable, os.path.join(BASE_DIR, "rebuttal_eval_summary.py"), "--all"]
        return run_command(cmd_eval, dry_run=args.dry_run, log_file=log_file)

    run_t1 = args.tier1 or args.all
    run_t2 = args.tier2 or args.all

    if not run_t1 and not run_t2:
        log_msg("No target specified. Please specify --tier1, --tier2, --all, or --eval_only.", log_file)
        parser.print_help()
        return 1

    if run_t1:
        ret = run_tier1(args, log_file)
        if ret != 0 and not args.ignore_errors:
            log_msg(f"Tier 1 halted with exit code {ret}", log_file)
            return ret

    if run_t2:
        ret = run_tier2(args, log_file)
        if ret != 0 and not args.ignore_errors:
            log_msg(f"Tier 2 halted with exit code {ret}", log_file)
            return ret

    log_msg("\n" + "=" * 70, log_file)
    log_msg("🎉 ALL REQUESTED REBUTTAL EXPERIMENTS COMPLETED SUCCESSFULLY!", log_file)
    log_msg(f"Full execution log saved to: {log_file}", log_file)
    log_msg("=" * 70, log_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
