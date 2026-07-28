"""
make_appendix_figure.py — Combine alignment diagnostics for Arabic and Hindi 
into ONE multi-panel summary figure for the appendix.

Plots panels (a, b, c, f) from the main text summary figure for both languages.
Layout: 2 rows (Arabic, Hindi) x 4 columns (Anisotropy, Distance, Cosine, Common-mode).

Usage:
    python make_appendix_figure.py --prefix_ar alignment_stats_ar --prefix_hi alignment_stats_hi --output_pdf figure_appendix_ar_hi.pdf
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


def plot_language_row(axes, prefix, lang_name, row_idx):
    """
    Plots the 4 panels for a given language.
    axes: list of 4 matplotlib axes objects.
    row_idx: 0 for top row (Arabic), 1 for bottom row (Hindi).
    """
    try:
        main_df = pd.read_csv(f"{prefix}.csv")
        aniso_df = pd.read_csv(f"{prefix}_anisotropy.csv")
    except Exception as e:
        print(f"Warning: Could not load data for {prefix} ({e}). Skipping this row.")
        for ax in axes:
            ax.set_visible(False)
        return

    example_ids = main_df['example'].values if 'example' in main_df.columns else None
    
    letters = [['a', 'b', 'c', 'd'], ['e', 'f', 'g', 'h']][row_idx]

    # 1. Anisotropy control
    ax = axes[0]
    data = [aniso_df['uniform_control_cosine'].values, aniso_df['ot_weighted_cosine'].values]
    bp = ax.boxplot(data, tick_labels=[f'Uniform {lang_name}\ncontrol', f'OT-weighted\n{lang_name} counterpart'],
                     patch_artist=True, widths=0.5, showfliers=False)
    for patch, color in zip(bp['boxes'], ['#aec7e8', '#2ca02c']):
        patch.set_facecolor(color)
    ax.set_ylabel('Cosine sim. to EN answer')
    ax.set_title(f"({letters[0]}) {lang_name} Anisotropy control (n={len(aniso_df)})", fontsize=9.5)
    loc_a = 'upper left' if aniso_df['ot_weighted_cosine'].mean() < aniso_df['uniform_control_cosine'].mean() else 'lower right'
    annotate_stats(ax, stat_pair(aniso_df['ot_weighted_cosine'], aniso_df['uniform_control_cosine']), loc=loc_a)

    # 2. Euclidean distance, before vs after
    ax = axes[1]
    paired_slopegraph(ax, main_df['dist_before'], main_df['dist_after'],
                       f'OT-weighted Euclidean distance', f'({letters[1]}) {lang_name} Distance: before/after',
                       higher_is_better=False, example_ids=example_ids, annotate_outlier=True)
    annotate_stats(ax, stat_pair(main_df['dist_after'], main_df['dist_before']), loc='upper left')

    # 3. Cosine alignment, before vs after
    ax = axes[2]
    paired_slopegraph(ax, main_df['cos_before'], main_df['cos_after'],
                       f'Cosine sim. (EN, {lang_name} centroid)',
                       f'({letters[2]}) {lang_name} Angular alignment', higher_is_better=True)
    annotate_stats(ax, stat_pair(main_df['cos_after'], main_df['cos_before']), loc='lower right')

    # 4. Common-mode shift histogram
    ax = axes[3]
    ax.hist(main_df['common_mode_shift_cosine'], bins=min(15, max(5, len(main_df) // 2)),
            color='#8c564b', edgecolor='black', alpha=0.85)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(main_df['common_mode_shift_cosine'].mean(), color='red', linewidth=1.8,
               label=f"mean={main_df['common_mode_shift_cosine'].mean():.2f}")
    ax.set_xlabel(f'Cosine(EN shift, {lang_name} shift)')
    ax.set_ylabel('Count')
    ax.set_title(f"({letters[3]}) {lang_name} Common-mode drift", fontsize=9.5)
    ax.legend(fontsize=8, loc='upper left')
    annotate_stats(ax, stat_one_sample(main_df['common_mode_shift_cosine']), loc='upper right')


def main():
    parser = argparse.ArgumentParser(description="Build the appendix alignment-diagnostics summary figure for Arabic & Hindi")
    parser.add_argument("--prefix_ar", type=str, default="alignment_stats_ar",
                         help="Shared prefix for Arabic (e.g. alignment_stats_ar)")
    parser.add_argument("--prefix_hi", type=str, default="alignment_stats_hi",
                         help="Shared prefix for Hindi (e.g. alignment_stats_hi)")
    parser.add_argument("--output_pdf", type=str, default="figure_alignment_appendix.pdf")
    args = parser.parse_args()

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 9

    # 2 rows (Arabic, Hindi), 4 columns (Anisotropy, Distance, Cosine, Common-mode)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.8))

    plot_language_row(axes[0], args.prefix_ar, "AR", 0)
    plot_language_row(axes[1], args.prefix_hi, "HI", 1)

    fig.suptitle('Appendix: Cross-lingual alignment diagnostics (Arabic & Hindi)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    png_path = args.output_pdf.replace('.pdf', '.png')
    svg_path = args.output_pdf.replace('.pdf', '.svg')
    plt.savefig(args.output_pdf, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    print(f"Saved appendix summary figure to:\n  - {args.output_pdf}\n  - {png_path}\n  - {svg_path}")


if __name__ == '__main__':
    main()
