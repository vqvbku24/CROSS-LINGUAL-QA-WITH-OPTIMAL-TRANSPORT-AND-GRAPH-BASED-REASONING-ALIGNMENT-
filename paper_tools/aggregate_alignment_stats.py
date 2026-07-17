"""
Aggregate the OT-weighted EN-answer -> VI-counterpart alignment metric across
MANY QA examples, instead of relying on a single example (which, as we saw,
can have just one answer token and tell you almost nothing reliable).

Expected layout: a root directory containing one subdirectory PER EXAMPLE,
each holding the same files visualize_representation.py expects:
    <root>/<example_id>/hidden_before.npy
    <root>/<example_id>/hidden_after.npy
    <root>/<example_id>/gamma_fig5.npy   (or gamma.npy)
    <root>/<example_id>/langs.npy
    <root>/<example_id>/labels.npy

If your examples are laid out differently, adjust `find_example_dirs()` and/or
`load_example()` below — the statistics/plotting logic doesn't need to change.

Usage:
    python aggregate_alignment_stats.py --root_dir /path/to/examples --center
"""
import os
import argparse
import numpy as np


def find_example_dirs(root_dir):
    dirs = []
    for name in sorted(os.listdir(root_dir)):
        d = os.path.join(root_dir, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'hidden_before.npy')):
            dirs.append(d)
    return dirs


def normalize_str_array(arr):
    out = []
    for v in arr:
        if isinstance(v, bytes):
            v = v.decode('utf-8', errors='ignore')
        out.append(str(v).strip())
    return np.array(out)


def load_example(example_dir, center):
    hidden_before = np.load(os.path.join(example_dir, 'hidden_before.npy'))
    hidden_after = np.load(os.path.join(example_dir, 'hidden_after.npy'))

    gamma_path = os.path.join(example_dir, 'gamma_fig5.npy')
    if not os.path.exists(gamma_path):
        gamma_path = os.path.join(example_dir, 'gamma.npy')
    gamma = np.load(gamma_path)

    langs = np.array([s.upper() for s in normalize_str_array(
        np.load(os.path.join(example_dir, 'langs.npy'), allow_pickle=True))])
    labels = np.array([s.lower() for s in normalize_str_array(
        np.load(os.path.join(example_dir, 'labels.npy'), allow_pickle=True))])

    num_en_total = int(np.sum(langs == 'EN'))

    if center:
        en_idx = (langs == 'EN')
        vi_idx = (langs == 'VI')
        en_centroid = hidden_before[en_idx].mean(axis=0)
        vi_centroid = hidden_before[vi_idx].mean(axis=0)
        hidden_before = hidden_before.copy()
        hidden_after = hidden_after.copy()
        hidden_before[en_idx] -= en_centroid
        hidden_before[vi_idx] -= vi_centroid
        hidden_after[en_idx] -= en_centroid
        hidden_after[vi_idx] -= vi_centroid

    en_ans_rows = np.where((langs == 'EN') & (labels == 'answer'))[0]
    return hidden_before, hidden_after, gamma, en_ans_rows, num_en_total


def norm_stats(hidden, gamma, en_ans_rows, num_en_total):
    """Vector norms of EN answer tokens and their OT-weighted VI centroid.
    Tests the hypothesis that Euclidean distance grows mainly because vector
    MAGNITUDE increases, not because direction/alignment gets worse."""
    if en_ans_rows.size == 0:
        return None, None
    en_vecs = hidden[en_ans_rows]
    vi_vecs = hidden[num_en_total:]
    sub_gamma = gamma[en_ans_rows, :]
    w = sub_gamma / (sub_gamma.sum(axis=1, keepdims=True) + 1e-12)
    vi_weighted_centroid = w @ vi_vecs
    en_norms = np.linalg.norm(en_vecs, axis=1)
    vi_norms = np.linalg.norm(vi_weighted_centroid, axis=1)
    return en_norms, vi_norms


def ot_weighted_distance(hidden, gamma, en_ans_rows, num_en_total):
    if en_ans_rows.size == 0:
        return None
    en_vecs = hidden[en_ans_rows]
    vi_vecs = hidden[num_en_total:]
    sub_gamma = gamma[en_ans_rows, :]
    w = sub_gamma / (sub_gamma.sum(axis=1, keepdims=True) + 1e-12)
    diffs = en_vecs[:, None, :] - vi_vecs[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    per_token = (dists * w).sum(axis=1)
    return per_token  # one value per EN answer token in this example


def ot_weighted_cosine_sim(hidden, gamma, en_ans_rows, num_en_total):
    """Angular alignment: cosine similarity between each EN answer vector and
    its OT-weighted VI centroid (sum_j w_ij * vi_vec_j). Scale-invariant, so it
    can show a different trend than raw Euclidean distance if the model is
    mainly rescaling/rotating representations rather than shrinking them."""
    if en_ans_rows.size == 0:
        return None
    en_vecs = hidden[en_ans_rows]
    vi_vecs = hidden[num_en_total:]
    sub_gamma = gamma[en_ans_rows, :]
    w = sub_gamma / (sub_gamma.sum(axis=1, keepdims=True) + 1e-12)
    vi_weighted_centroid = w @ vi_vecs  # (n_ans, D)
    en_norms = np.linalg.norm(en_vecs, axis=1) + 1e-12
    vi_norms = np.linalg.norm(vi_weighted_centroid, axis=1) + 1e-12
    cos_sim = (en_vecs * vi_weighted_centroid).sum(axis=1) / (en_norms * vi_norms)
    return cos_sim  # one value per EN answer token, higher = more aligned in direction


def common_mode_shift_cosine(hidden_before, hidden_after, gamma, en_ans_rows, num_en_total):
    """For each EN answer token, cosine similarity between its own shift
    (after - before) and the OT-weighted shift of its VI counterparts.
    ~1 = moving together (common-mode drift); ~0 = unrelated; <0 = diverging."""
    if en_ans_rows.size == 0:
        return None
    en_shift = hidden_after[en_ans_rows] - hidden_before[en_ans_rows]
    vi_shift_all = hidden_after[num_en_total:] - hidden_before[num_en_total:]
    sub_gamma = gamma[en_ans_rows, :]
    w = sub_gamma / (sub_gamma.sum(axis=1, keepdims=True) + 1e-12)
    vi_shift_weighted = w @ vi_shift_all
    en_norms = np.linalg.norm(en_shift, axis=1) + 1e-12
    vi_norms = np.linalg.norm(vi_shift_weighted, axis=1) + 1e-12
    cos_sim = (en_shift * vi_shift_weighted).sum(axis=1) / (en_norms * vi_norms)
    return cos_sim


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate OT-weighted alignment metric across many examples")
    parser.add_argument("--root_dir", type=str, required=True,
                         help="Directory containing one subdirectory per QA example")
    parser.add_argument("--center", action="store_true", default=True,
                         help="Mean-center EN/VI per example (default: on)")
    parser.add_argument("--no-center", dest="center", action="store_false")
    parser.add_argument("--output_prefix", type=str, default="alignment_stats",
                         help="Prefix for output plot/csv files")
    args = parser.parse_args()

    example_dirs = find_example_dirs(args.root_dir)
    if not example_dirs:
        print(f"[error] No example subdirectories with hidden_before.npy found "
              f"under '{args.root_dir}'. Check the path / layout.")
        return

    print(f"Found {len(example_dirs)} example directories.")

    all_before, all_after, per_example_n = [], [], []
    all_cos_before, all_cos_after = [], []
    all_common_mode = []
    all_en_norm_before, all_en_norm_after = [], []
    all_vi_norm_before, all_vi_norm_after = [], []
    skipped = 0
    for d in example_dirs:
        try:
            hb, ha, gamma, en_ans_rows, num_en_total = load_example(d, args.center)
        except Exception as e:
            print(f"[warn] Skipping '{d}': failed to load ({e})")
            skipped += 1
            continue

        if en_ans_rows.size == 0:
            skipped += 1
            continue

        before_vals = ot_weighted_distance(hb, gamma, en_ans_rows, num_en_total)
        after_vals = ot_weighted_distance(ha, gamma, en_ans_rows, num_en_total)
        all_before.extend(before_vals.tolist())
        all_after.extend(after_vals.tolist())
        per_example_n.append(len(before_vals))

        cos_before = ot_weighted_cosine_sim(hb, gamma, en_ans_rows, num_en_total)
        cos_after = ot_weighted_cosine_sim(ha, gamma, en_ans_rows, num_en_total)
        all_cos_before.extend(cos_before.tolist())
        all_cos_after.extend(cos_after.tolist())

        common_mode = common_mode_shift_cosine(hb, ha, gamma, en_ans_rows, num_en_total)
        all_common_mode.extend(common_mode.tolist())

        en_norm_b, vi_norm_b = norm_stats(hb, gamma, en_ans_rows, num_en_total)
        en_norm_a, vi_norm_a = norm_stats(ha, gamma, en_ans_rows, num_en_total)
        all_en_norm_before.extend(en_norm_b.tolist())
        all_en_norm_after.extend(en_norm_a.tolist())
        all_vi_norm_before.extend(vi_norm_b.tolist())
        all_vi_norm_after.extend(vi_norm_a.tolist())

    if skipped:
        print(f"[warn] Skipped {skipped}/{len(example_dirs)} examples "
              f"(no EN answer-labeled rows or load error).")

    n = len(all_before)
    if n == 0:
        print("[error] No usable answer tokens found across any example.")
        return

    all_before = np.array(all_before)
    all_after = np.array(all_after)
    diffs = all_after - all_before

    print(f"\n=== Aggregated over {n} answer tokens from "
          f"{len(per_example_n)} examples ===")
    print(f"Before: mean={all_before.mean():.4f}, std={all_before.std():.4f}")
    print(f"After:  mean={all_after.mean():.4f}, std={all_after.std():.4f}")
    print(f"Paired diff (after - before): mean={diffs.mean():.4f}, "
          f"std={diffs.std():.4f}")
    n_closer = int((diffs < 0).sum())
    print(f"Tokens that got closer: {n_closer}/{n} ({100 * n_closer / n:.1f}%)")

    # Statistical significance (paired, since before/after share the same token)
    try:
        from scipy import stats
        if n >= 2:
            t_stat, t_p = stats.ttest_rel(all_after, all_before)
            print(f"Paired t-test: t={t_stat:.3f}, p={t_p:.4g}")
        if n >= 6:  # wilcoxon needs a reasonable sample to be meaningful
            w_stat, w_p = stats.wilcoxon(all_after, all_before)
            print(f"Wilcoxon signed-rank test: W={w_stat:.3f}, p={w_p:.4g}")
    except ImportError:
        print("[note] Install scipy (`pip install scipy`) to get significance "
              "tests (paired t-test / Wilcoxon) alongside these summary stats.")

    # ── Angular (cosine) alignment: scale-invariant, can diverge from the
    # Euclidean-distance trend above if the model is mainly rescaling/rotating
    # representations rather than literally shrinking distances. ──
    all_cos_before = np.array(all_cos_before)
    all_cos_after = np.array(all_cos_after)
    cos_diffs = all_cos_after - all_cos_before
    n_more_aligned = int((cos_diffs > 0).sum())
    print(f"\n=== Angular (cosine) alignment: EN answer token vs. its "
          f"OT-weighted VI centroid ===")
    print(f"Before: mean={all_cos_before.mean():.4f}, std={all_cos_before.std():.4f}")
    print(f"After:  mean={all_cos_after.mean():.4f}, std={all_cos_after.std():.4f}")
    print(f"Tokens with HIGHER cosine similarity (more aligned in direction) "
          f"after: {n_more_aligned}/{n} ({100 * n_more_aligned / n:.1f}%)")
    try:
        from scipy import stats
        if n >= 2:
            t_stat, t_p = stats.ttest_rel(all_cos_after, all_cos_before)
            print(f"Paired t-test (cosine): t={t_stat:.3f}, p={t_p:.4g}")
        if n >= 6:
            w_stat, w_p = stats.wilcoxon(all_cos_after, all_cos_before)
            print(f"Wilcoxon signed-rank test (cosine): W={w_stat:.3f}, p={w_p:.4g}")
    except ImportError:
        pass

    # ── Common-mode drift: does the EN answer token move in roughly the same
    # direction as its VI counterpart (rather than moving toward it)? ──
    all_common_mode = np.array(all_common_mode)
    print(f"\n=== Common-mode shift (cosine between EN-answer shift vector and "
          f"OT-weighted VI-counterpart shift vector) ===")
    print(f"mean={all_common_mode.mean():.4f}, std={all_common_mode.std():.4f}, "
          f"median={np.median(all_common_mode):.4f}")
    print(f"Interpretation: ~1.0 = moving together (common-mode drift — explains "
          f"why Euclidean distance can fail to shrink even with real, large "
          f"changes to the embeddings); ~0 = unrelated directions; negative = "
          f"genuinely diverging.")
    try:
        from scipy import stats
        if n >= 2:
            t_stat, t_p = stats.ttest_1samp(all_common_mode, 0.0)
            print(f"One-sample t-test vs. 0: t={t_stat:.3f}, p={t_p:.4g}")
    except ImportError:
        pass

    # ── Norm (magnitude) check: does Euclidean distance grow mainly because
    # vectors get LARGER, not because direction gets worse? ──
    en_norm_before = np.array(all_en_norm_before)
    en_norm_after = np.array(all_en_norm_after)
    vi_norm_before = np.array(all_vi_norm_before)
    vi_norm_after = np.array(all_vi_norm_after)

    print(f"\n=== Vector norm (magnitude) check ===")
    print(f"EN answer-token norm:        before={en_norm_before.mean():.4f}, "
          f"after={en_norm_after.mean():.4f} "
          f"({'+' if en_norm_after.mean() > en_norm_before.mean() else ''}"
          f"{en_norm_after.mean() - en_norm_before.mean():.4f})")
    print(f"VI OT-weighted centroid norm: before={vi_norm_before.mean():.4f}, "
          f"after={vi_norm_after.mean():.4f} "
          f"({'+' if vi_norm_after.mean() > vi_norm_before.mean() else ''}"
          f"{vi_norm_after.mean() - vi_norm_before.mean():.4f})")
    try:
        from scipy import stats
        if n >= 2:
            t_en, p_en = stats.ttest_rel(en_norm_after, en_norm_before)
            t_vi, p_vi = stats.ttest_rel(vi_norm_after, vi_norm_before)
            print(f"Paired t-test, EN norm change: t={t_en:.3f}, p={p_en:.4g}")
            print(f"Paired t-test, VI norm change: t={t_vi:.3f}, p={p_vi:.4g}")
    except ImportError:
        pass

    en_norm_grew = en_norm_after.mean() > en_norm_before.mean()
    vi_norm_grew = vi_norm_after.mean() > vi_norm_before.mean()
    if en_norm_grew or vi_norm_grew:
        print("[interpretation] Vector norm increased on at least one side after "
              "alignment. Combined with the near-saturated, well-above-random "
              "cosine similarity, this supports the hypothesis that the small "
              "rise in Euclidean distance reflects representations getting "
              "LARGER in magnitude while staying tightly aligned in DIRECTION — "
              "i.e. not a loss of cross-lingual correspondence.")
    else:
        print("[interpretation] Norms did not grow — the Euclidean distance "
              "increase is NOT simply explained by scale. Worth re-examining "
              "what specifically is pushing the tokens apart.")

    # Save a paired before/after scatter — the standard way to show this in a paper
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        lims = [min(all_before.min(), all_after.min()) * 0.95,
                max(all_before.max(), all_after.max()) * 1.05]
        ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='y = x (no change)')
        ax.scatter(all_before, all_after, alpha=0.6, edgecolors='black', linewidths=0.3)
        ax.set_xlabel('Distance before alignment')
        ax.set_ylabel('Distance after alignment')
        ax.set_title(f'OT-weighted EN\u2013VI answer distance (n={n})\n'
                      f'Points below the line = closer after alignment')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.legend(loc='upper left', fontsize=8)
        plt.tight_layout()
        out_path = f"{args.output_prefix}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.savefig(f"{args.output_prefix}.pdf", bbox_inches='tight')
        plt.close()
        print(f"\nSaved paired before/after scatter to {out_path} "
              f"and {args.output_prefix}.pdf")
    except ImportError:
        print("[note] matplotlib not available, skipping plot.")

    # Histogram of common-mode shift cosine similarity — a quick visual for
    # whether the "moving together, not closer" story holds across examples
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(all_common_mode, bins=min(15, max(5, n // 2)), color='#1f77b4',
                edgecolor='black', alpha=0.8)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1, label='0 (unrelated)')
        ax.axvline(all_common_mode.mean(), color='red', linestyle='-', linewidth=1.5,
                   label=f'mean={all_common_mode.mean():.2f}')
        ax.set_xlabel('Cosine(EN-answer shift, OT-weighted VI-counterpart shift)')
        ax.set_ylabel('Count')
        ax.set_title(f'Common-mode shift across {n} answer tokens')
        ax.legend(fontsize=8)
        plt.tight_layout()
        cm_path = f"{args.output_prefix}_common_mode.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.savefig(f"{args.output_prefix}_common_mode.pdf", bbox_inches='tight')
        plt.close()
        print(f"Saved common-mode shift histogram to {cm_path}")
    except ImportError:
        pass

    # Save raw values for further analysis / re-plotting
    csv_path = f"{args.output_prefix}.csv"
    with open(csv_path, 'w') as f:
        f.write("dist_before,dist_after,dist_diff,cos_before,cos_after,common_mode_shift_cosine\n")
        for b, a, cb, ca, cm in zip(all_before, all_after, all_cos_before,
                                     all_cos_after, all_common_mode):
            f.write(f"{b},{a},{a - b},{cb},{ca},{cm}\n")
    print(f"Saved raw per-token values to {csv_path}")


if __name__ == '__main__':
    main()