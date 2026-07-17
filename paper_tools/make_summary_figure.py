"""
make_summary_figure.py — Combine all the alignment diagnostics computed by
aggregate_alignment_stats.py and layer_and_anisotropy_diagnostics.py into ONE
multi-panel summary figure for the paper.

Design choices made specifically to preempt common reviewer objections:
  - Every paired comparison shows individual-token lines (not just group
    means +/- SD), because SD across different examples/tokens is a
    different (and usually much larger) source of variance than the
    within-pair change, and a bar+errorbar plot can visually hide a real,
    significant paired effect. Panel (d) was changed from bar+errorbar to a
    slopegraph for exactly this reason.
  - Every panel reports BOTH a paired t-test and a Wilcoxon signed-rank test
    p-value directly on the figure, since relying on just one invites the
    "why not the other test" question, and the two can disagree.
  - The largest-magnitude outlier in the distance panel is labeled with its
    source example directory, so "what is that outlier" has an immediate,
    checkable answer instead of requiring a follow-up question.

Reads:
    <prefix>.csv            (from aggregate_alignment_stats.py; needs the
                              'example' column added there)
    <prefix>_anisotropy.csv (from layer_and_anisotropy_diagnostics.py)
    <prefix>_layers.csv     (from layer_and_anisotropy_diagnostics.py)

Usage:
    python aggregate_alignment_stats.py --root_dir paper_tools/export_multi --output_prefix alignment_stats
    python layer_and_anisotropy_diagnostics.py --root_dir paper_tools/export_multi --output_prefix alignment_stats
    python make_summary_figure.py --prefix alignment_stats --output_pdf figure_alignment_summary.pdf
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def stat_pair(a, b):
    """Paired t-test + Wilcoxon signed-rank test; returns formatted string."""
    a, b = np.asarray(a), np.asarray(b)
    parts = []
    try:
        t, pt = stats.ttest_rel(a, b)
        parts.append(f"paired t: p={pt:.2g}")
    except Exception:
        pass
    try:
        if len(a) >= 6:
            w, pw = stats.wilcoxon(a, b)
            parts.append(f"Wilcoxon: p={pw:.2g}")
    except Exception:
        pass
    return "\n".join(parts)


def stat_one_sample(a, mu=0.0):
    a = np.asarray(a)
    try:
        t, p = stats.ttest_1samp(a, mu)
        return f"one-sample t: p={p:.2g}"
    except Exception:
        return ""


def annotate_stats(ax, text, loc='lower right'):
    if not text:
        return
    positions = {
        'lower right': dict(x=0.97, y=0.03, ha='right', va='bottom'),
        'upper right': dict(x=0.97, y=0.97, ha='right', va='top'),
        'lower left': dict(x=0.03, y=0.03, ha='left', va='bottom'),
        'upper left': dict(x=0.03, y=0.97, ha='left', va='top'),
    }
    pos = positions[loc]
    ax.text(pos['x'], pos['y'], text, transform=ax.transAxes,
            ha=pos['ha'], va=pos['va'], fontsize=7,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))


def paired_slopegraph(ax, before, after, ylabel, title, higher_is_better,
                       x0=0, x1=1, labels=('Before', 'After'), example_ids=None,
                       annotate_outlier=False):
    before = np.asarray(before)
    after = np.asarray(after)
    for i, (b, a) in enumerate(zip(before, after)):
        if higher_is_better is None:
            color = 'gray'
        else:
            improved = (a > b) if higher_is_better else (a < b)
            color = '#2ca02c' if improved else '#d62728'
        ax.plot([x0, x1], [b, a], color=color, alpha=0.35, linewidth=1)
    ax.scatter([x0] * len(before), before, color='#1f77b4', s=18, zorder=3)
    ax.scatter([x1] * len(after), after, color='#ff7f0e', s=18, zorder=3)
    ax.plot([x0, x1], [before.mean(), after.mean()], color='black',
             linewidth=2.2, zorder=4, marker='D', markersize=5)

    if annotate_outlier and example_ids is not None:
        diffs = after - before
        idx = int(np.argmax(np.abs(diffs)))
        ax.annotate(str(example_ids[idx]), xy=(x1, after[idx]),
                    xytext=(x1 + 0.15, after[idx]),
                    fontsize=6.5, color='dimgray', va='center')

    if title:
        n_improved = None
        if higher_is_better is not None:
            n_improved = int(((after > before) if higher_is_better else (after < before)).sum())
        title_full = f"{title}\n(n={len(before)}" + (f", {n_improved}/{len(before)} improved)" if n_improved is not None else ")")
        ax.set_title(title_full, fontsize=9.5)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)


def main():
    parser = argparse.ArgumentParser(description="Build the alignment-diagnostics summary figure")
    parser.add_argument("--prefix", type=str, default="alignment_stats",
                         help="Shared --output_prefix used by the two stats scripts")
    parser.add_argument("--output_pdf", type=str, default="figure_alignment_summary.pdf")
    args = parser.parse_args()

    main_df = pd.read_csv(f"{args.prefix}.csv")
    aniso_df = pd.read_csv(f"{args.prefix}_anisotropy.csv")
    layers_df = pd.read_csv(f"{args.prefix}_layers.csv")

    example_ids = main_df['example'].values if 'example' in main_df.columns else None

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 9

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8))

    # (a) Anisotropy control: OT-weighted cosine vs. random-token control
    ax = axes[0, 0]
    data = [aniso_df['random_control_cosine'].values, aniso_df['ot_weighted_cosine'].values]
    bp = ax.boxplot(data, tick_labels=['Random VI\ntoken', 'OT-weighted\nVI counterpart'],
                     patch_artist=True, widths=0.5, showfliers=False)
    for patch, color in zip(bp['boxes'], ['#aec7e8', '#2ca02c']):
        patch.set_facecolor(color)
    ax.set_ylabel('Cosine similarity to EN answer token')
    ax.set_title(f"(a) Anisotropy control (n={len(aniso_df)})", fontsize=9.5)
    loc_a = 'upper left' if aniso_df['ot_weighted_cosine'].mean() < aniso_df['random_control_cosine'].mean() else 'lower right'
    annotate_stats(ax, stat_pair(aniso_df['ot_weighted_cosine'], aniso_df['random_control_cosine']), loc=loc_a)

    # (b) Euclidean distance, before vs after (paired)
    ax = axes[0, 1]
    paired_slopegraph(ax, main_df['dist_before'], main_df['dist_after'],
                       'OT-weighted Euclidean distance', '(b) Distance: before vs after',
                       higher_is_better=False, example_ids=example_ids, annotate_outlier=True)
    annotate_stats(ax, stat_pair(main_df['dist_after'], main_df['dist_before']), loc='upper left')

    # (c) Cosine alignment, before vs after (paired)
    ax = axes[0, 2]
    paired_slopegraph(ax, main_df['cos_before'], main_df['cos_after'],
                       'Cosine similarity (EN, OT-weighted VI centroid)',
                       '(c) Angular alignment: before vs after', higher_is_better=True)
    annotate_stats(ax, stat_pair(main_df['cos_after'], main_df['cos_before']), loc='lower right')

    # (d) Vector norm change, EN and VI — paired slopegraph (not bar+SD: between-example
    # SD is much larger than the within-pair change and would visually hide a real,
    # significant paired effect).
    ax = axes[1, 0]
    paired_slopegraph(ax, main_df['en_norm_before'], main_df['en_norm_after'],
                       'Vector norm ||x||', '', higher_is_better=None,
                       x0=0, x1=1, labels=('EN\nbefore', 'EN\nafter'))
    for b, a in zip(main_df['vi_norm_before'], main_df['vi_norm_after']):
        ax.plot([3, 4], [b, a], color='gray', alpha=0.35, linewidth=1)
    ax.scatter([3] * len(main_df), main_df['vi_norm_before'], color='#1f77b4', s=18, zorder=3)
    ax.scatter([4] * len(main_df), main_df['vi_norm_after'], color='#ff7f0e', s=18, zorder=3)
    ax.plot([3, 4], [main_df['vi_norm_before'].mean(), main_df['vi_norm_after'].mean()],
            color='black', linewidth=2.2, zorder=4, marker='D', markersize=5)
    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels(['EN\nbefore', 'EN\nafter', 'VI\nbefore', 'VI\nafter'])
    ax.set_xlim(-0.5, 4.5)
    en_p = stat_pair(main_df['en_norm_after'], main_df['en_norm_before'])
    vi_p = stat_pair(main_df['vi_norm_after'], main_df['vi_norm_before'])
    ax.set_title(f"(d) Magnitude (norm) change (n={len(main_df)})", fontsize=9.5)
    annotate_stats(ax, f"EN: {en_p}\nVI: {vi_p}", loc='lower left')

    # (e) Per-layer alignment (cosine) vs. learned layer weight
    ax = axes[1, 1]
    ax2 = ax.twinx()
    layer_labels = [str(int(l)) for l in layers_df['layer']]
    bar = ax.bar(layer_labels, layers_df['mean_cosine'], color='#9467bd', alpha=0.8,
                 label='Cosine alignment', edgecolor='black', linewidth=0.6)
    line = ax2.plot(layer_labels, layers_df['layer_weight_softmax'], color='#d62728',
                     marker='o', linewidth=2, label='Learned layer weight (softmax)')
    ax.set_xlabel('Transformer layer')
    ax.set_ylabel('Mean cosine alignment', color='#9467bd')
    ax2.set_ylabel('Learned layer weight', color='#d62728')
    n_layer_tokens = int(layers_df['n'].dropna().iloc[0]) if 'n' in layers_df.columns and len(layers_df['n'].dropna()) else None
    title_e = '(e) Per-layer alignment vs. learned weight'
    if n_layer_tokens:
        title_e += f" (n={n_layer_tokens})"
    ax.set_title(title_e, fontsize=9.5)
    lines_labels = [bar, line[0]]
    ax.legend(lines_labels, [l.get_label() for l in lines_labels], fontsize=7, loc='lower right')

    # (f) Common-mode shift histogram
    ax = axes[1, 2]
    ax.hist(main_df['common_mode_shift_cosine'], bins=min(15, max(5, len(main_df) // 2)),
            color='#8c564b', edgecolor='black', alpha=0.85)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(main_df['common_mode_shift_cosine'].mean(), color='red', linewidth=1.8,
               label=f"mean={main_df['common_mode_shift_cosine'].mean():.2f}")
    ax.set_xlabel('Cosine(EN shift, OT-weighted VI shift)')
    ax.set_ylabel('Count')
    ax.set_title(f"(f) Common-mode drift (n={len(main_df)})", fontsize=9.5)
    ax.legend(fontsize=8, loc='upper left')
    annotate_stats(ax, stat_one_sample(main_df['common_mode_shift_cosine']), loc='upper right')

    fig.suptitle('Cross-lingual alignment diagnostics: Euclidean distance vs. angular alignment',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    png_path = args.output_pdf.replace('.pdf', '.png')
    svg_path = args.output_pdf.replace('.pdf', '.svg')
    plt.savefig(args.output_pdf, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"Saved summary figure to:\n  - {args.output_pdf}\n  - {png_path}\n  - {svg_path}")


if __name__ == '__main__':
    main()