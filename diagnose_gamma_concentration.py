"""
diagnose_gamma_concentration.py

Panel (a) in the anisotropy-control figure shows OT-weighted "expected
pairwise cosine" pinned at ~1.0 with near-zero variance across all 50
examples, for VI, AR, and HI alike -- noticeably tighter than the
cosine-of-centroid metric (Panel c/g) computed on the SAME hidden states.
That's either (a) genuinely sharp, correct OT correspondence, or (b) gamma
collapsing onto a VI token that is a near-duplicate of the EN vector for
reasons unrelated to translation (indexing bug, fallback data, etc.).

This script distinguishes the two by looking, per EN answer token, at the
VI token gamma weights MOST HEAVILY (argmax), and reporting:
  - row_entropy   : Shannon entropy of the gamma row (near 0 = one-hot)
  - max_weight    : the largest single weight in that row
  - cos_to_argmax : cosine(en_vec, vi_vecs[argmax])
  - dist_to_argmax: Euclidean distance(en_vec, vi_vecs[argmax])

If cos_to_argmax ~ 1.0 AND dist_to_argmax ~ 0, the "match" is a near-
duplicate vector -- almost certainly a data/export bug, not a real
cross-lingual finding. If cos_to_argmax ~ 1.0 but dist_to_argmax is
large (i.e. same direction, very different magnitude/position), that's
a more plausible genuine finding (same direction, different vector).

Usage:
    python diagnose_gamma_concentration.py /path/to/export_multi
"""
import os
import sys
import numpy as np


def load_common(example_dir):
    gamma_path = os.path.join(example_dir, 'gamma_fig5.npy')
    if not os.path.exists(gamma_path):
        gamma_path = os.path.join(example_dir, 'gamma.npy')
    gamma = np.load(gamma_path)
    langs = np.load(os.path.join(example_dir, 'langs.npy'), allow_pickle=True)
    labels = np.load(os.path.join(example_dir, 'labels.npy'), allow_pickle=True)
    langs = np.array([str(x).strip().upper() for x in langs])
    labels = np.array([str(x).strip().lower() for x in labels])
    num_en_total = int(np.sum(langs == 'EN'))
    en_ans_rows = np.where((langs == 'EN') & (labels == 'answer'))[0]
    return gamma, en_ans_rows, num_en_total


def diagnose(root_dir, top_n=10):
    dirs = sorted(d for d in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, d)))
    rows = []
    for name in dirs:
        d = os.path.join(root_dir, name)
        hidden_after_path = os.path.join(d, 'hidden_after.npy')
        if not os.path.exists(hidden_after_path):
            continue
        try:
            hidden_after = np.load(hidden_after_path)
            gamma, en_ans_rows, num_en_total = load_common(d)
        except Exception as e:
            print(f"[warn] skipping {name}: {e}")
            continue
        if en_ans_rows.size == 0:
            continue

        en_vecs = hidden_after[en_ans_rows]
        vi_vecs = hidden_after[num_en_total:]
        sub_gamma = gamma[en_ans_rows, :]
        w = sub_gamma / (sub_gamma.sum(axis=1, keepdims=True) + 1e-12)

        for i in range(en_vecs.shape[0]):
            row = w[i]
            p = row / (row.sum() + 1e-12)
            entropy = float(-(p * np.log(p + 1e-12)).sum())
            j = int(np.argmax(row))
            max_w = float(row[j])
            en_v, vi_v = en_vecs[i], vi_vecs[j]
            cos = float(en_v @ vi_v /
                        (np.linalg.norm(en_v) * np.linalg.norm(vi_v) + 1e-12))
            dist = float(np.linalg.norm(en_v - vi_v))
            rows.append((name, entropy, max_w, j, cos, dist))

    if not rows:
        print(f"[error] No usable examples found under '{root_dir}'.")
        return

    entropies = np.array([r[1] for r in rows])
    coss = np.array([r[4] for r in rows])
    dists = np.array([r[5] for r in rows])

    rows_sorted = sorted(rows, key=lambda r: r[5])  # smallest distance first
    print(f"{'example':<15}{'entropy':<10}{'max_w':<10}{'argmax_j':<10}"
          f"{'cos':<12}{'dist':<10}")
    for r in rows_sorted[:top_n]:
        print(f"{r[0]:<15}{r[1]:<10.4f}{r[2]:<10.4f}{r[3]:<10}"
              f"{r[4]:<12.6f}{r[5]:<10.6f}")

    print(f"\nn = {len(rows)}")
    print(f"Mean gamma row entropy: {entropies.mean():.4f} "
          f"(near 0 = one-hot/collapsed, higher = spread out over many VI tokens)")
    print(f"Mean cosine to argmax match: {coss.mean():.6f}, min={coss.min():.6f}")
    print(f"Mean distance to argmax match: {dists.mean():.4f}, min={dists.min():.4f}")

    near_dup = int((dists < 0.05).sum())
    print(f"\nExamples where the argmax VI match is a near-duplicate of the "
          f"EN vector (dist < 0.05): {near_dup}/{len(rows)}")
    if near_dup > 0:
        print("[flag] Near-duplicate matches found -- check whether the VI "
              "slice accidentally includes EN vectors (indexing/ordering "
              "bug), or whether hidden_after.npy is a fallback copy for "
              "these examples, before trusting Panel (a)'s p-value.")
    elif entropies.mean() < 0.3:
        print("[flag] Gamma rows are heavily collapsed (low entropy) onto a "
              "single VI token, but that token is NOT a near-duplicate of "
              "the EN vector -- worth manually inspecting a few of these "
              "argmax matches (decode the token IDs) to confirm they are "
              "plausible translations, not a systematic artifact (e.g. "
              "always matching position 0, or a [CLS]/[SEP]-like token).")
    else:
        print("[ok] Gamma rows are reasonably spread out and matches are not "
              "near-duplicates -- Panel (a)'s near-1.0 cosine is more likely "
              "a genuine (if surprisingly sharp) finding. Still worth a "
              "manual spot-check of a few examples before writing this up.")


if __name__ == '__main__':
    diagnose(sys.argv[1] if len(sys.argv) > 1 else 'paper_tools/export_multi')
