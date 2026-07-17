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
    parser.add_argument("--center", dest="center", action="store_true", default=True,
                         help="Mean-center EN and VI to remove language bias (default: on)")
    parser.add_argument("--no-center", dest="center", action="store_false",
                         help="Disable mean-centering (shows raw language gap)")
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
        
    # Optional: Mean Center to remove Language Bias
    if args.center:
        print("Applying mean-centering to remove XLM-R language gap...")
        en_idx = (langs == 'EN')
        vi_idx = (langs == 'VI')
        
        # Calculate centroids from the 'before' state to be a fixed reference
        en_centroid = hidden_before[en_idx].mean(axis=0)
        vi_centroid = hidden_before[vi_idx].mean(axis=0)
        
        # Shift everything to origin
        hidden_before[en_idx] -= en_centroid
        hidden_before[vi_idx] -= vi_centroid
        
        hidden_after[en_idx] -= en_centroid
        hidden_after[vi_idx] -= vi_centroid

    # ── Quantitative alignment metric (gamma-weighted EN-VI answer distance) ──
    # This is a much more reliable signal than eyeballing the UMAP/t-SNE plot,
    # since those methods distort absolute distances. Compute it in the ORIGINAL
    # (centered) space, restricted to the answer-token rows, weighted by the OT plan.
    en_ans_mask_full = (langs == 'EN') & (labels == 'answer')
    vi_ans_mask_full = (langs == 'VI') & (labels == 'answer')
    num_en_total = int(np.sum(langs == 'EN'))

    if np.any(en_ans_mask_full) and np.any(vi_ans_mask_full):
        en_ans_rows = np.where(en_ans_mask_full)[0]
        vi_ans_rows = np.where(vi_ans_mask_full)[0] - num_en_total  # index into gamma's VI axis

        def gamma_weighted_distance(hidden):
            en_vecs = hidden[en_ans_rows]
            vi_vecs = hidden[num_en_total:][vi_ans_rows]
            sub_gamma = gamma[np.ix_(en_ans_rows, vi_ans_rows)]
            w = sub_gamma / (sub_gamma.sum() + 1e-12)
            diffs = en_vecs[:, None, :] - vi_vecs[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)
            return float((dists * w).sum())

        dist_before = gamma_weighted_distance(hidden_before)
        dist_after = gamma_weighted_distance(hidden_after)
        print(f"[metric] Gamma-weighted EN-VI answer-token distance — "
              f"before: {dist_before:.4f}, after: {dist_after:.4f} "
              f"({'closer' if dist_after < dist_before else 'farther'} by "
              f"{abs(dist_before - dist_after):.4f})")
    else:
        en_ans_rows = vi_ans_rows = np.array([], dtype=int)
        print("[warn] No answer-labeled rows found, skipping alignment metric.")

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

    # Draw thin lines connecting the top-weighted EN-VI answer-token correspondences
    # (per the OT plan). This is the visual evidence of alignment that a global
    # UMAP/t-SNE view can hide when the semantic shift is small relative to spread.
    def draw_correspondences(ax, proj):
        if en_ans_rows.size == 0 or vi_ans_rows.size == 0:
            return
        sub_gamma = gamma[np.ix_(en_ans_rows, vi_ans_rows)]
        top_vi_per_en = sub_gamma.argmax(axis=1)
        max_w = sub_gamma.max()
        for i, en_row in enumerate(en_ans_rows):
            vi_row_local = top_vi_per_en[i]
            vi_row_global = num_en_total + vi_ans_rows[vi_row_local]
            w = sub_gamma[i, vi_row_local]
            ax.plot([proj[en_row, 0], proj[vi_row_global, 0]],
                    [proj[en_row, 1], proj[vi_row_global, 1]],
                    color='red', linewidth=0.8, alpha=0.3 + 0.5 * (w / (max_w + 1e-12)),
                    zorder=1)

    draw_correspondences(ax1, proj_before)

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

    draw_correspondences(ax3, proj_after)

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