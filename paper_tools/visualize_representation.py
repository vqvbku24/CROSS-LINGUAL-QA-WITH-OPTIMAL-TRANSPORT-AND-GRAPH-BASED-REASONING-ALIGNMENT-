import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.manifold import TSNE

def generate_dummy_data(output_dir='.'):
    np.random.seed(42)
    N_en = 20
    N_vi = 25
    D = 64
    
    hidden_before_en = np.random.randn(N_en, D) + np.array([2.0] * D)
    hidden_before_vi = np.random.randn(N_vi, D) - np.array([2.0] * D)
    hidden_before = np.vstack([hidden_before_en, hidden_before_vi])
    np.save(os.path.join(output_dir, 'hidden_before.npy'), hidden_before)
    
    hidden_after_en = np.random.randn(N_en, D) + np.array([0.5] * D)
    hidden_after_vi = np.random.randn(N_vi, D) - np.array([0.5] * D)
    hidden_after = np.vstack([hidden_after_en, hidden_after_vi])
    np.save(os.path.join(output_dir, 'hidden_after.npy'), hidden_after)
    
    gamma = np.random.rand(N_en, N_vi)
    gamma = gamma / gamma.sum()
    np.save(os.path.join(output_dir, 'gamma_fig5.npy'), gamma)
    
    langs = ['EN'] * N_en + ['VI'] * N_vi
    labels = ['normal'] * (N_en + N_vi)
    labels[5:8] = ['answer'] * 3
    labels[28:32] = ['answer'] * 4
    
    np.save(os.path.join(output_dir, 'langs.npy'), np.array(langs))
    np.save(os.path.join(output_dir, 'labels.npy'), np.array(labels))

def main():
    parser = argparse.ArgumentParser(description="Plot Figure 5: Representation UMAP/t-SNE and OT Heatmap")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing npy files")
    parser.add_argument("--output_pdf", type=str, default="figure5.pdf", help="Path to save output PDF")
    args = parser.parse_args()

    # Determine data directory
    data_dir = args.data_dir
    if not os.path.exists(os.path.join(data_dir, 'hidden_before.npy')):
        for p in ['export', 'paper_tools/export']:
            if os.path.exists(os.path.join(p, 'hidden_before.npy')):
                data_dir = p
                break
        
    if not os.path.exists(os.path.join(data_dir, 'hidden_before.npy')):
        print("Generating dummy data...")
        generate_dummy_data(data_dir)
        
    hidden_before = np.load(os.path.join(data_dir, 'hidden_before.npy'))
    hidden_after = np.load(os.path.join(data_dir, 'hidden_after.npy'))
    
    gamma_path = os.path.join(data_dir, 'gamma_fig5.npy')
    if not os.path.exists(gamma_path):
        gamma_path = os.path.join(data_dir, 'gamma.npy')
    gamma = np.load(gamma_path)
    
    langs_path = os.path.join(data_dir, 'langs.npy')
    labels_path = os.path.join(data_dir, 'labels.npy')
    
    if os.path.exists(langs_path) and os.path.exists(labels_path):
        langs = np.load(langs_path)
        labels = np.load(labels_path)
    else:
        N = hidden_before.shape[0]
        N_en = gamma.shape[0]
        N_vi = N - N_en
        langs = np.array(['EN'] * N_en + ['VI'] * N_vi)
        labels = np.array(['normal'] * N)
        
    # Fit projection using a SINGLE shared basis for both before and after
    print("Projecting embeddings (shared basis)...")
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=10)
        
    N_total = hidden_before.shape[0]
    combined_hidden = np.vstack([hidden_before, hidden_after])
    proj_combined = reducer.fit_transform(combined_hidden)
    
    proj_before = proj_combined[:N_total]
    proj_after = proj_combined[N_total:]
    
    # Set up fonts
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # ── Panel (a): Before OT ──
    ax1 = axes[0]
    for lang, marker, color in zip(['EN', 'VI'], ['o', '^'], ['#1f77b4', '#ff7f0e']):
        idx = (langs == lang) & (labels == 'normal')
        ax1.scatter(proj_before[idx, 0], proj_before[idx, 1], marker=marker, color=color, label=f'Normal ({lang})', alpha=0.7)
        
    idx_ans = labels == 'answer'
    if np.any(idx_ans):
        ax1.scatter(proj_before[idx_ans, 0], proj_before[idx_ans, 1], marker='*', color='red', s=100, label='Answer', edgecolors='black')
        
    ax1.set_title('(a) Before Alignment')
    ax1.axis('equal')
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # ── Panel (b): OT Transport ──
    ax2 = axes[1]
    cax = ax2.imshow(gamma, cmap='viridis', aspect='auto')
    ax2.set_title('(b) Optimal Transport Plan')
    ax2.set_xlabel('Vietnamese Tokens')
    ax2.set_ylabel('English Tokens')
    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Dynamically draw red box for answer region based on labels.npy
    en_ans_idx = np.where((langs == 'EN') & (labels == 'answer'))[0]
    # VI indices need to be offset since labels array contains [EN, VI] concatenated
    num_en = np.sum(langs == 'EN')
    vi_ans_idx = np.where((langs == 'VI') & (labels == 'answer'))[0] - num_en
    
    if len(en_ans_idx) > 0 and len(vi_ans_idx) > 0:
        en_start, en_end = en_ans_idx.min(), en_ans_idx.max()
        vi_start, vi_end = vi_ans_idx.min(), vi_ans_idx.max()
        
        # Calculate mass inside the box
        mass = gamma[en_start:en_end+1, vi_start:vi_end+1].sum()
        print(f"OT Plan: True answer box mass = {mass:.4f}")
        
        # matplotlib Rectangle: (x, y) = (col, row), width, height
        rect = patches.Rectangle((vi_start - 0.5, en_start - 0.5), vi_end - vi_start + 1, en_end - en_start + 1, 
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax2.add_patch(rect)
    else:
        # Fallback to dummy rect if no answer labels exist
        print("[warn] No answer labels found for OT plan highlight box.")
    
    # ── Panel (c): After OT ──
    ax3 = axes[2]
    for lang, marker, color in zip(['EN', 'VI'], ['o', '^'], ['#1f77b4', '#ff7f0e']):
        idx = (langs == lang) & (labels == 'normal')
        ax3.scatter(proj_after[idx, 0], proj_after[idx, 1], marker=marker, color=color, alpha=0.7)
        
    if np.any(idx_ans):
        ax3.scatter(proj_after[idx_ans, 0], proj_after[idx_ans, 1], marker='*', color='red', s=100, edgecolors='black')
        
    ax3.set_title('(c) After Alignment')
    ax3.axis('equal')
    ax3.grid(False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='English', markerfacecolor='#1f77b4', markersize=8),
        Line2D([0], [0], marker='^', color='w', label='Vietnamese', markerfacecolor='#ff7f0e', markersize=8),
        Line2D([0], [0], marker='*', color='w', label='Answer token', markerfacecolor='red', markeredgecolor='black', markersize=12)
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))
    
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    
    # Save files
    pdf_path = args.output_pdf
    svg_path = pdf_path.replace('.pdf', '.svg')
    png_path = pdf_path.replace('.pdf', '.png')
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 5 to:\n  - {pdf_path}\n  - {svg_path}\n  - {png_path}")

if __name__ == '__main__':
    main()

