import os
import subprocess
import argparse

def run_cmd(cmd, exit_on_fail=True):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        if exit_on_fail:
            print(f"Error executing command. Exiting.")
            exit(1)
        else:
            print(f"[Warn] Command failed, but continuing anyway.")
            return False
    return True

def run_pipeline(lang, checkpoint, dataset, output_root, stage1_checkpoint, prefix, num_gpus):
    print(f"\n{'='*50}\nStarting pipeline for {lang.upper()}\n{'='*50}")
    
    # 1. Export samples
    export_cmd = (
        f"python paper_tools/export_many_samples.py "
        f"--checkpoint {checkpoint} "
        f"--dataset {dataset} "
        f"--output_root {output_root} "
        f"--sample_indices 0-49 "
        f"--stage1_checkpoint {stage1_checkpoint} "
        f"--num_gpus {num_gpus}"
    )
    run_cmd(export_cmd)
    
    # 2. Aggregate stats
    agg_cmd = (
        f"python paper_tools/aggregate_alignment_stats.py "
        f"--root_dir {output_root} "
        f"--output_prefix {prefix}"
    )
    run_cmd(agg_cmd)
    
    # 3. Layer and Anisotropy diagnostics
    diag_cmd = (
        f"python paper_tools/layer_and_anisotropy_diagnostics.py "
        f"--root_dir {output_root} "
        f"--output_prefix {prefix}"
    )
    run_cmd(diag_cmd)

def main():
    parser = argparse.ArgumentParser(description="Run full diagnostics pipeline for all languages")
    
    # Checkpoints
    parser.add_argument("--ckpt_vi", type=str, default="checkpoints/stage2_lora_best.pt", help="Vietnamese Stage 2 checkpoint")
    parser.add_argument("--ckpt_ar", type=str, default="checkpoints/stage2_ar_best.pt", help="Arabic Stage 2 checkpoint")
    parser.add_argument("--ckpt_hi", type=str, default="checkpoints/stage2_hi_best.pt", help="Hindi Stage 2 checkpoint")
    parser.add_argument("--ckpt_stage1", type=str, default="checkpoints/stage1_squad_best.pt", help="Stage 1 Teacher checkpoint")
    
    # Datasets
    parser.add_argument("--data_vi", type=str, default="dataset/xquad.vi.json")
    parser.add_argument("--data_ar", type=str, default="dataset/xquad.ar.json")
    parser.add_argument("--data_hi", type=str, default="dataset/xquad.hi.json")
    
    # Hardware
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to run in parallel")
    
    args = parser.parse_args()
    
    # Output roots and prefixes
    out_vi = "paper_tools/export_multi_vi"
    out_ar = "paper_tools/export_multi_ar"
    out_hi = "paper_tools/export_multi_hi"
    
    prefix_vi = "alignment_stats"
    prefix_ar = "alignment_stats_ar"
    prefix_hi = "alignment_stats_hi"
    
    # Run pipelines
    if os.path.exists(args.ckpt_vi):
        run_pipeline("Vietnamese", args.ckpt_vi, args.data_vi, out_vi, args.ckpt_stage1, prefix_vi, args.num_gpus)
        # Generate summary figure for VI
        print("\nGenerating Figure 5 (Vietnamese Summary)...")
        success = run_cmd(f"python paper_tools/make_summary_figure.py --prefix {prefix_vi} --output_pdf paper_tools/figures/figure_alignment_summary.pdf", exit_on_fail=False)
        if not success:
            print("[Note] Plotting failed (likely missing matplotlib/pandas). The CSVs are saved in root, you can plot them locally!")
    else:
        print(f"Skipping VI: Checkpoint {args.ckpt_vi} not found.")

    run_ar = os.path.exists(args.ckpt_ar)
    run_hi = os.path.exists(args.ckpt_hi)
    
    if run_ar:
        run_pipeline("Arabic", args.ckpt_ar, args.data_ar, out_ar, args.ckpt_stage1, prefix_ar, args.num_gpus)
    else:
        print(f"Skipping AR: Checkpoint {args.ckpt_ar} not found.")
        
    if run_hi:
        run_pipeline("Hindi", args.ckpt_hi, args.data_hi, out_hi, args.ckpt_stage1, prefix_hi, args.num_gpus)
    else:
        print(f"Skipping HI: Checkpoint {args.ckpt_hi} not found.")
        
    if run_ar and run_hi:
        # Generate appendix figure for AR and HI
        print("\nGenerating Appendix Figure 6 (Arabic & Hindi Summary)...")
        success = run_cmd(f"python paper_tools/make_appendix_figure.py --prefix_ar {prefix_ar} --prefix_hi {prefix_hi} --output_pdf paper_tools/figures/figure_appendix_ar_hi.pdf", exit_on_fail=False)
        if not success:
            print("[Note] Plotting failed (likely missing matplotlib/pandas). CSVs are saved, you can copy them and plot locally!")
        
    print("\nAll diagnostics completed successfully!")

if __name__ == '__main__':
    main()
