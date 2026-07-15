import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

def generate_dummy_data(output_dir='.'):
    np.random.seed(42)
    gamma = np.random.rand(15, 20)
    gamma = gamma / gamma.sum()
    np.save(os.path.join(output_dir, 'gamma.npy'), gamma)
    
    with open(os.path.join(output_dir, 'english_tokens.txt'), 'w') as f:
        f.write('\n'.join([f'en_tok_{i}' for i in range(15)]))
        
    with open(os.path.join(output_dir, 'vietnamese_tokens.txt'), 'w') as f:
        f.write('\n'.join([f'vi_tok_{i}' for i in range(20)]))
        
    langs = ['EN'] * 15 + ['VI'] * 20
    labels = ['normal'] * 35
    labels[5:8] = ['answer'] * 3
    labels[23:27] = ['answer'] * 4
    np.save(os.path.join(output_dir, 'langs.npy'), np.array(langs))
    np.save(os.path.join(output_dir, 'labels.npy'), np.array(labels))

def main():
    parser = argparse.ArgumentParser(description="Plot Figure 2: OT Heatmap")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing npy/txt files")
    parser.add_argument("--output_pdf", type=str, default="figure2_ot_heatmap.pdf", help="Path to save output PDF")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(os.path.join(data_dir, 'gamma.npy')):
        for p in ['export', 'paper_tools/export']:
            if os.path.exists(os.path.join(p, 'gamma.npy')):
                data_dir = p
                break

    if not os.path.exists(os.path.join(data_dir, 'gamma.npy')):
        print("Generating dummy data...")
        generate_dummy_data(data_dir)
        
    gamma = np.load(os.path.join(data_dir, 'gamma.npy'))
    
    en_tokens = None
    en_tok_path = os.path.join(data_dir, 'english_tokens.txt')
    if os.path.exists(en_tok_path):
        with open(en_tok_path, 'r', encoding='utf-8') as f:
            en_tokens = [line.strip() for line in f.readlines()]
            
    vi_tokens = None
    vi_tok_path = os.path.join(data_dir, 'vietnamese_tokens.txt')
    if os.path.exists(vi_tok_path):
        with open(vi_tok_path, 'r', encoding='utf-8') as f:
            vi_tokens = [line.strip() for line in f.readlines()]

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10

    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Heatmap
    cax = ax.imshow(gamma, cmap='viridis', aspect='auto')
    fig.colorbar(cax, ax=ax)
    
    # Ticks
    if vi_tokens is not None and len(vi_tokens) == gamma.shape[1]:
        ax.set_xticks(np.arange(len(vi_tokens)))
        ax.set_xticklabels(vi_tokens, rotation=45, ha='right')
    else:
        ax.set_xlabel('Vietnamese tokens')
        
    if en_tokens is not None and len(en_tokens) == gamma.shape[0]:
        ax.set_yticks(np.arange(len(en_tokens)))
        ax.set_yticklabels(en_tokens)
    else:
        ax.set_ylabel('English tokens')
        
    ax.grid(False)
    
    # Dynamically draw red box for answer region based on labels.npy
    langs_path = os.path.join(data_dir, 'langs.npy')
    labels_path = os.path.join(data_dir, 'labels.npy')
    
    if os.path.exists(langs_path) and os.path.exists(labels_path):
        langs = np.load(langs_path)
        labels = np.load(labels_path)
        
        en_ans_idx = np.where((langs == 'EN') & (labels == 'answer'))[0]
        num_en = np.sum(langs == 'EN')
        vi_ans_idx = np.where((langs == 'VI') & (labels == 'answer'))[0] - num_en
        
        if len(en_ans_idx) > 0 and len(vi_ans_idx) > 0:
            en_start, en_end = en_ans_idx.min(), en_ans_idx.max()
            vi_start, vi_end = vi_ans_idx.min(), vi_ans_idx.max()
            
            # calculate and print mass
            mass = gamma[en_start:en_end+1, vi_start:vi_end+1].sum()
            print(f"OT Plan Box Mass: {mass:.4f}")
            
            rect = patches.Rectangle((vi_start - 0.5, en_start - 0.5), vi_end - vi_start + 1, en_end - en_start + 1, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
    else:
        print("[warn] langs.npy or labels.npy not found. Highlight box not drawn.")

    plt.tight_layout()
    
    # Save files
    pdf_path = args.output_pdf
    svg_path = pdf_path.replace('.pdf', '.svg')
    png_path = pdf_path.replace('.pdf', '.png')
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2 to:\n  - {pdf_path}\n  - {svg_path}\n  - {png_path}")

if __name__ == '__main__':
    main()

