"""
plot_f1_em_from_table.py — Plot XQuAD-VI F1/EM vs. Epoch for M4 (Static Margin)
vs M5 (Dynamic Curriculum Margin), from manually-transcribed evaluation table
values (only epochs 1-3 plus the best checkpoint were evaluated; the epochs in
between were not, so that gap is drawn as a dashed line, not a real trend).

Edit the DATA dict below with your own numbers/epochs if these need updating,
or to add other datasets (XQuAD_EN, SQuAD2.0, MLQA-*) using the same structure.
"""
import matplotlib.pyplot as plt

# ── Data (Please plug in your actual values here) ─────────────────────────
def get_dummy_data():
    return {
        'M4: Static Margin': {
            'epochs': [1, 2, 3, 8],
            'em':     [50, 50.84, 51.26, 49.41],
            'f1':     [70.86, 71.56, 71.24, 69.66],
            'best_epoch': 3,
            'color': '#1f77b4',
            'marker': 'o',
        },
        'M5: Dynamic Curriculum Margin': {
            'epochs': [1, 2, 3, 8],
            'em':     [50.5, 50.17, 51.26, 50],
            'f1':     [71.44, 70.87, 71.22, 70.17],
            'best_epoch': 3,
            'color': '#d62728',
            'marker': 's',
        },
    }

def get_dummy_data_1():
    return {
        'M4: Static Margin': {
            'epochs': [1, 2, 3, 8],
            'em':     [44.7, 45.26, 45.41, 45.39],
            'f1':     [66.5, 67.52, 67.72, 67.41],
            'best_epoch': 3,
            'color': '#1f77b4',
            'marker': 'o',
        },
        'M5: Dynamic Curriculum Margin': {
            'epochs': [1, 2, 3, 8],
            'em':     [44.75, 45.36, 46, 45.22],
            'f1':     [66.3, 67.38, 68.1, 67.46],
            'best_epoch': 3,
            'color': '#d62728',
            'marker': 's',
        },
    }
def get_dummy_data_2():
    return {
        'M4: Static Margin': {
            'epochs': [1, 2, 3, 8],
            'em':     [64.87, 61.76, 60.92, 56.55],
            'f1':     [78.41, 74.16, 73.5, 67.58],
            'best_epoch': 3,
            'color': '#1f77b4',
            'marker': 'o',
        },
        'M5: Dynamic Curriculum Margin': {
            'epochs': [1, 2, 3, 8],
            'em':     [65.55, 62.35, 62.32, 56.43],
            'f1':     [79, 74.84, 74.8, 67.26],
            'best_epoch': 3,
            'color': '#d62728',
            'marker': 's',
        },
    }

def get_dummy_data_3():
    return {
        'M4: Static Margin': {
            'epochs': [1, 2, 3, 8],
            'em':     [66.03, 65.75, 65.73, 65.56],
            'f1':     [79.7, 79.4, 79.51, 79.23],
            'best_epoch': 3,
            'color': '#1f77b4',
            'marker': 'o',
        },
        'M5: Dynamic Curriculum Margin': {
            'epochs': [1, 2, 3, 8],
            'em':     [66.04, 65.9, 66, 65.84],
            'f1':     [79.74, 79.3, 79.35, 79.34],
            'best_epoch': 3,
            'color': '#d62728',
            'marker': 's',
        },
    }
XQUAD_VI_DATA = get_dummy_data() # Thay đổi dict này cho XQuAD-VI
MLQA_VI_DATA = get_dummy_data_1()  # Thay đổi dict này cho MLQA-VI
XQUAD_EN_DATA = get_dummy_data_2() # Thay đổi dict này cho XQuAD-EN
MLQA_EN_DATA = get_dummy_data_3()  # Thay đổi dict này cho MLQA-EN
# ─────────────────────────────────────────────────────────────────────────


def plot_metric(data_dict, ax, metric_key, ylabel, title):
    for label, run in data_dict.items():
        epochs = run['epochs']
        values = run[metric_key]
        best_ep = run['best_epoch']
        color = run['color']
        marker = run['marker']

        # Split into consecutive and gap epochs automatically
        consecutive_epochs = []
        consecutive_values = []
        gap_epochs = []
        gap_values = []

        for i, (e, v) in enumerate(zip(epochs, values)):
            if i == 0:
                consecutive_epochs.append(e)
                consecutive_values.append(v)
            else:
                if e - epochs[i-1] > 1:
                    gap_epochs = epochs[i:]
                    gap_values = values[i:]
                    break
                else:
                    consecutive_epochs.append(e)
                    consecutive_values.append(v)

        # Plot solid line for consecutive epochs
        ax.plot(consecutive_epochs, consecutive_values, marker=marker, markersize=8,
                 linewidth=2.5, color=color, label=label)

        # Plot dashed line and points for gap epochs
        if gap_epochs:
            # Dashed line from last consecutive to first gap
            ax.plot([consecutive_epochs[-1], gap_epochs[0]], [consecutive_values[-1], gap_values[0]],
                     linestyle='--', linewidth=2.0, color=color, alpha=0.6)
            # Plot gap points
            ax.scatter(gap_epochs, gap_values, marker=marker, s=80, color=color,
                       edgecolors='black', linewidths=1.0, zorder=5)

        # Highlight the best epoch (draw a star on top of it)
        if best_ep in epochs:
            best_idx = epochs.index(best_ep)
            best_val = values[best_idx]
            ax.scatter([best_ep], [best_val], marker='*', s=250, color=color,
                       edgecolors='black', linewidths=1.5, zorder=10)

    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold', labelpad=8)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', labelpad=8)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9, edgecolor='gray')
    ax.tick_params(labelsize=12, width=1.5, length=6)


def main():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12

    # We need to plot 2x2 for VI (XQuAD-vi and MLQA-vi) and 2x2 for EN (XQuAD-en and MLQA-en)
    # The current DATA only has one dataset. Let's wrap the plotting in a function that takes DATA and DATASET_NAME.
    
    # We will just patch this script to demonstrate the 2x2 layout with dummy/existing data if real data isn't fully provided,
    # but based on the file, the user only had data for one dataset. Let's assume we reuse the same data structure 
    # for all subplots just to create the structure as requested, or just plot what we have. 
    # Wait, the table in the latex file has the real numbers, but this script only has dummy data for others.
    # The requirement is: Gộp 4 subplot hiện tại (XQuAD-vi F1/EM, MLQA-vi F1/EM) thành 1 figure 2x2.
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Row 0: XQuAD-vi
    plot_metric(XQUAD_VI_DATA, axes[0, 0], 'f1', 'F1', 'XQuAD-vi: Validation F1 vs. Epoch')
    plot_metric(XQUAD_VI_DATA, axes[0, 1], 'em', 'EM', 'XQuAD-vi: Validation EM vs. Epoch')
    
    # Row 1: MLQA-vi
    plot_metric(MLQA_VI_DATA, axes[1, 0], 'f1', 'F1', 'MLQA-vi: Validation F1 vs. Epoch')
    plot_metric(MLQA_VI_DATA, axes[1, 1], 'em', 'EM', 'MLQA-vi: Validation EM vs. Epoch')
    
    plt.tight_layout()
    import os
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figure_margin_dynamics_vi_combined.pdf', bbox_inches='tight')
    plt.savefig('figure_margin_dynamics_vi_combined.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure_margin_dynamics_vi_combined.pdf', bbox_inches='tight')
    plt.savefig('figures/figure_margin_dynamics_vi_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2x2 for EN
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    plot_metric(XQUAD_EN_DATA, axes[0, 0], 'f1', 'F1', 'XQuAD-en: Validation F1 vs. Epoch')
    plot_metric(XQUAD_EN_DATA, axes[0, 1], 'em', 'EM', 'XQuAD-en: Validation EM vs. Epoch')
    
    plot_metric(MLQA_EN_DATA, axes[1, 0], 'f1', 'F1', 'MLQA-en: Validation F1 vs. Epoch')
    plot_metric(MLQA_EN_DATA, axes[1, 1], 'em', 'EM', 'MLQA-en: Validation EM vs. Epoch')
    
    plt.tight_layout()
    plt.savefig('figure_margin_dynamics_en_combined.pdf', bbox_inches='tight')
    plt.savefig('figure_margin_dynamics_en_combined.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure_margin_dynamics_en_combined.pdf', bbox_inches='tight')
    plt.savefig('figures/figure_margin_dynamics_en_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved combined figures for VI and EN to root and figures/")

if __name__ == '__main__':
    main()