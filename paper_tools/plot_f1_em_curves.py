"""
plot_f1_em_from_table.py — Plot XQuAD-VI F1/EM vs. Epoch for M4 (Static Margin)
vs M5 (Dynamic Curriculum Margin), from manually-transcribed evaluation table
values (only epochs 1-3 plus the best checkpoint were evaluated; the epochs in
between were not, so that gap is drawn as a dashed line, not a real trend).

Edit the DATA dict below with your own numbers/epochs if these need updating,
or to add other datasets (XQuAD_EN, SQuAD2.0, MLQA-*) using the same structure.
"""
import matplotlib.pyplot as plt

# ── Data (from your table) ──────────────────────────────────────────────
# 'epochs' are the actual epoch numbers evaluated (not necessarily consecutive).
# The last entry is the "best" checkpoint (selected by a criterion other than
# XQuAD-VI alone — noted explicitly on the plot to avoid the point looking
# like an error, since for M4 it's actually LOWER than epochs 1-3).
DATA = {
    'M4: Static Margin': {
        'epochs': [1, 2, 3, 6],
        'em':     [44.7, 45.26, 45.41, 45.39],
        'f1':     [66.5, 67.52, 67.72, 67.41],
        'best_epoch': 6,
        'color': '#1f77b4',
        'marker': 'o',
    },
    'M5: Dynamic Curriculum Margin': {
        'epochs': [1, 2, 3, 7],
        'em':     [44.75, 45.37, 46, 45.89],
        'f1':     [66.3, 67.38, 68.1, 67.46],
        'best_epoch': 7,
        'color': '#d62728',
        'marker': 's',
    },
}
DATASET_NAME = 'MLQA-vi'
OUTPUT_PDF = 'figure_f1_em_mlqa_vi.pdf'
# ─────────────────────────────────────────────────────────────────────────


def plot_metric(ax, metric_key, ylabel, title):
    for label, run in DATA.items():
        epochs = run['epochs']
        values = run[metric_key]
        best_ep = run['best_epoch']
        color = run['color']
        marker = run['marker']

        # Solid line through consecutively-evaluated epochs (here: 1,2,3)
        consecutive_epochs, consecutive_values = [], []
        gap_start = None
        for e, v in zip(epochs, values):
            if e == best_ep:
                gap_start = (consecutive_epochs[-1], consecutive_values[-1]) if consecutive_epochs else None
                continue
            consecutive_epochs.append(e)
            consecutive_values.append(v)

        ax.plot(consecutive_epochs, consecutive_values, marker=marker, markersize=5,
                 linewidth=1.6, color=color, label=label)

        # Highlight epoch 3 — the true best epoch overall (across all 8) — with
        # a bold black outline, so the peak is visually obvious without reading
        # the caption.
        if 3 in consecutive_epochs:
            peak_idx = consecutive_epochs.index(3)
            ax.scatter([consecutive_epochs[peak_idx]], [consecutive_values[peak_idx]],
                       marker=marker, s=110, facecolors=color, edgecolors='black',
                       linewidths=1.8, zorder=6)

        # Dashed line + distinct marker for the best-checkpoint point, since
        # epochs in between were never evaluated (no real trend to imply)
        if gap_start is not None:
            best_val = values[epochs.index(best_ep)]
            ax.plot([gap_start[0], best_ep], [gap_start[1], best_val],
                     linestyle='--', linewidth=1.2, color=color, alpha=0.6)
            ax.scatter([best_ep], [best_val], marker='*', s=180, color=color,
                       edgecolors='black', linewidths=0.7, zorder=5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')


def main():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    plot_metric(axes[0], 'f1', 'F1', f'{DATASET_NAME}: Validation F1 vs. Epoch')
    plot_metric(axes[1], 'em', 'EM', f'{DATASET_NAME}: Validation EM vs. Epoch')

    fig.text(0.5, -0.04,
              f'Epoch 3 is the best epoch overall (across all 8 training epochs) '
              f'for both runs on {DATASET_NAME}. * = best checkpoint among the '
              'remaining post-peak epochs (4-8), shown as a single representative '
              'point instead of plotting all 5 (hence the dashed connector, since '
              'the epochs in between were not individually evaluated on this plot).',
              ha='center', fontsize=8, style='italic')

    plt.tight_layout()
    png_path = OUTPUT_PDF.replace('.pdf', '.png')
    svg_path = OUTPUT_PDF.replace('.pdf', '.svg')
    plt.savefig(OUTPUT_PDF, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"Saved:\n  - {OUTPUT_PDF}\n  - {png_path}\n  - {svg_path}")


if __name__ == '__main__':
    main()