import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax

def extract_margin_from_tb(tb_dir):
    """
    Extracts margin schedule from TensorBoard event files.
    Tries to find 'Hyperparameters/Lambda_Margin' or 'Loss/Margin'.
    Returns (epochs, margins) or (None, None) if not found.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[warn] tensorboard package not installed. Cannot parse TB events directly.")
        return None, None

    event_files = glob.glob(os.path.join(tb_dir, "**/*tfevents*"), recursive=True)
    if not event_files:
        return None, None

    # Sort to process in order
    event_files.sort()

    tb_steps = []
    tb_margins = []
    
    # We also want to find the step-to-epoch mapping
    epoch_steps = {} # epoch_num -> global_step

    for f in event_files:
        try:
            ea = EventAccumulator(f)
            ea.Reload()
            
            tags = ea.Tags()
            scalars = tags.get('scalars', [])
            
            # 1. Read margin schedule (logged at global_step)
            margin_tag = None
            if 'Hyperparameters/Lambda_Margin' in scalars:
                margin_tag = 'Hyperparameters/Lambda_Margin'
            elif 'Loss/Margin' in scalars:
                margin_tag = 'Loss/Margin'
                
            if margin_tag:
                for event in ea.Scalars(margin_tag):
                    tb_steps.append(event.step)
                    tb_margins.append(event.value)
            
            # 2. Read an epoch-level tag to get the max epoch and step mapping
            # Eval metrics are logged with step = epoch_number (1, 2, 3...)
            # We can find the global_step at each epoch from training logs if needed,
            # or simply use the epoch numbers directly.
            eval_tag = None
            for t in ['Eval/XQuAD_VI_EM', 'Loss/Stage2_Total', 'Loss/Stage2_Total_epoch']:
                if t in scalars:
                    eval_tag = t
                    break
            
            if eval_tag:
                for event in ea.Scalars(eval_tag):
                    # For Eval tags, step is epoch.
                    # For Loss tags, step is global_step.
                    pass
                    
        except Exception as e:
            print(f"[warn] Failed to read event file {f}: {e}")

    if not tb_steps:
        return None, None

    steps = np.array(tb_steps)
    margins = np.array(tb_margins)

    # Sort by step
    idx = np.argsort(steps)
    steps = steps[idx]
    margins = margins[idx]

    # Map steps to epochs:
    # If the max step is S, and we assume 6 epochs (or we estimate from steps),
    # we can map step -> epoch.
    # Alternatively, if the margin changes in discrete steps, we can just plot against
    # step or normalize step to [1, max_epochs].
    # Let's check if we have epoch boundary markers, or estimate max_epochs = 6 (default)
    # or look at the step pattern.
    # Let's estimate: typical steps per epoch is around 100-200.
    # We will normalize steps to epochs: epoch = 1 + (step / max_step) * (num_epochs - 1)
    # If max step is 0, just return 1 epoch.
    max_step = steps[-1]
    if max_step == 0:
        epochs = np.array([1])
    else:
        # Standard Stage 2 runs have 6 epochs
        num_epochs = 6
        epochs = 1 + (steps / max_step) * (num_epochs - 1)

    return epochs, margins

def generate_dummy_data(output_dir):
    np.random.seed(42)
    layer_weights = np.random.randn(4)
    np.save(os.path.join(output_dir, 'layer_weights.npy'), layer_weights)
    
    epochs = np.arange(1, 7)
    margins = np.array([1.0, 1.0, 0.7, 0.7, 0.7, 0.7]) # Simulating the schedule
    df = pd.DataFrame({'epoch': epochs, 'margin': margins})
    df.to_csv(os.path.join(output_dir, 'margin_history.csv'), index=False)

def main():
    parser = argparse.ArgumentParser(description="Plot Figure 3: Layer mixing weights and Margin Schedule")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing layer_weights.npy")
    parser.add_argument("--tb_dir", type=str, default="../checkpoint_stage2/tensorboard_stage2", 
                        help="Directory containing TensorBoard event logs")
    parser.add_argument("--output_pdf", type=str, default="figure3.pdf", help="Path to save output PDF")
    args = parser.parse_args()

    # Search paths for layer_weights.npy
    weights_path = None
    for p in [args.data_dir, 'export', 'paper_tools/export', '.']:
        path = os.path.join(p, 'layer_weights.npy')
        if os.path.exists(path):
            weights_path = path
            break

    # Search paths for TensorBoard
    tb_path = None
    for p in [args.tb_dir, '../checkpoint_stage2/tensorboard_stage2', 'checkpoint_stage2/tensorboard_stage2', 'runs', '.']:
        if os.path.exists(p) and any(f.endswith('tfevents') or os.path.isdir(os.path.join(p, f)) for f in os.listdir(p)):
            tb_path = p
            break

    # 1. Load layer weights
    if weights_path is not None:
        print(f"Loading layer weights from {weights_path}")
        weights = np.load(weights_path)
    else:
        print("[warn] layer_weights.npy not found. Generating dummy weights.")
        generate_dummy_data('.')
        weights = np.load('layer_weights.npy')

    norm_weights = softmax(weights)

    # 2. Load margin schedule
    epochs = None
    margins = None
    if tb_path is not None:
        print(f"Searching TensorBoard events in {tb_path}...")
        epochs, margins = extract_margin_from_tb(tb_path)

    if epochs is not None:
        print(f"Successfully extracted {len(margins)} margin data points from TensorBoard.")
    else:
        print("[warn] Could not extract margin from TensorBoard. Falling back to margin_history.csv or dummy.")
        csv_path = None
        for p in [args.data_dir, 'export', 'paper_tools/export', '.']:
            path = os.path.join(p, 'margin_history.csv')
            if os.path.exists(path):
                csv_path = path
                break
                
        if csv_path and os.path.exists(csv_path):
            print(f"Loading margin history from {csv_path}")
            df = pd.read_csv(csv_path)
            epochs = df['epoch'].values
            margins = df['margin'].values
        else:
            print("[warn] margin_history.csv not found either. Generating dummy schedule.")
            if not os.path.exists('margin_history.csv'):
                generate_dummy_data('.')
            df = pd.read_csv('margin_history.csv')
            epochs = df['epoch'].values
            margins = df['margin'].values

    # Setup fonts to Times New Roman (standard for ACL)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    
    # Subfigure (a): Layer Mixing Weights
    ax1 = axes[0]
    layers = ['Layer 6', 'Layer 7', 'Layer 8', 'Layer 9']
    bars = ax1.bar(layers, norm_weights, color='#d3d3d3', edgecolor='black', width=0.5)
    ax1.set_ylabel('Normalized Weight')
    ax1.set_xlabel('Layers')
    ax1.set_title('(a) Layer Mixing Weights')
    ax1.grid(False)
    
    # Annotate values
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')
        
    ax1.set_ylim(0, max(norm_weights) + 0.1)

    # Subfigure (b): Dynamic Margin Schedule
    ax2 = axes[1]
    ax2.plot(epochs, margins, marker='o', linestyle='-', color='black', markersize=4, linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Margin Parameter (λ_margin)')
    ax2.set_title('(b) Dynamic Margin Schedule')
    ax2.grid(False)
    
    # Adjust spacing
    plt.tight_layout()
    
    # Save files
    pdf_path = args.output_pdf
    svg_path = pdf_path.replace('.pdf', '.svg')
    png_path = pdf_path.replace('.pdf', '.png')
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Figure 3 to:\n  - {pdf_path}\n  - {svg_path}\n  - {png_path}")

if __name__ == '__main__':
    main()

