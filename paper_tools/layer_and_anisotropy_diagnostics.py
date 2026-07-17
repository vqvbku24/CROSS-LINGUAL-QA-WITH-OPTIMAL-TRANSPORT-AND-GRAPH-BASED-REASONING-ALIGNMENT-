"""
layer_and_anisotropy_diagnostics.py

Two follow-up diagnostics on top of aggregate_alignment_stats.py:

(1) ANISOTROPY CONTROL
    Contextual embedding spaces (BERT/XLM-R family) are known to be anisotropic:
    almost ANY two token vectors have high cosine similarity because a single
    dominant direction eats most of the norm (Ethayarajh 2019). Our earlier
    "angular alignment" metric (~0.993, barely moving) could just be measuring
    this artifact rather than real EN-VI correspondence. This diagnostic
    computes a control: cosine similarity between the EN answer token and
    RANDOM (non-corresponding) VI tokens from the same sentence, and compares
    it to the OT-weighted correspondence cosine. If they're statistically
    indistinguishable, the 0.993 finding is mostly/entirely an anisotropy
    artifact, not evidence of meaningful cross-lingual alignment.

(2) PER-LAYER ALIGNMENT vs. LEARNED LAYER WEIGHTS
    exporter.py saves hidden_en_layer{6,7,8,9}.npy / hidden_vi_layer{6,7,8,9}.npy
    (LoRA-active / "after" state only — no frozen counterpart is saved per
    layer, so this is a snapshot comparison across layers, not a before/after
    comparison) plus layer_weights.npy, the model's learned softmax mixing
    weights over those 4 layers. This computes OT-weighted distance/cosine
    PER LAYER, aggregated over all examples, and checks whether the layer the
    model learned to weight most heavily is also the layer with the tightest
    cross-lingual alignment — a plausible mechanism for the EM/F1 gain that
    doesn't require the final-layer answer-token distance to shrink.

Usage:
    python layer_and_anisotropy_diagnostics.py --root_dir paper_tools/export_multi
"""
import os
import argparse
import numpy as np

LAYER_INDICES = [6, 7, 8, 9]


def normalize_str_array(arr):
    out = []
    for v in arr:
        if isinstance(v, bytes):
            v = v.decode('utf-8', errors='ignore')
        out.append(str(v).strip())
    return np.array(out)


def find_example_dirs(root_dir):
    dirs = []
    for name in sorted(os.listdir(root_dir)):
        d = os.path.join(root_dir, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'hidden_before.npy')):
            dirs.append(d)
    return dirs


def load_common(example_dir):
    gamma_path = os.path.join(example_dir, 'gamma_fig5.npy')
    if not os.path.exists(gamma_path):
        gamma_path = os.path.join(example_dir, 'gamma.npy')
    gamma = np.load(gamma_path)

    langs = np.array([s.upper() for s in normalize_str_array(
        np.load(os.path.join(example_dir, 'langs.npy'), allow_pickle=True))])
    labels = np.array([s.lower() for s in normalize_str_array(
        np.load(os.path.join(example_dir, 'labels.npy'), allow_pickle=True))])
    num_en_total = int(np.sum(langs == 'EN'))
    en_ans_rows = np.where((langs == 'EN') & (labels == 'answer'))[0]
    return gamma, en_ans_rows, num_en_total


def ot_weighted_stats(en_vecs, vi_vecs, sub_gamma):
    """Returns (distance, cosine_to_weighted_centroid) per EN answer row."""
    w = sub_gamma / (sub_gamma.sum(axis=1, keepdims=True) + 1e-12)
    vi_weighted_centroid = w @ vi_vecs
    dists = np.linalg.norm(en_vecs - vi_weighted_centroid, axis=1)
    en_norms = np.linalg.norm(en_vecs, axis=1) + 1e-12
    vc_norms = np.linalg.norm(vi_weighted_centroid, axis=1) + 1e-12
    cos = (en_vecs * vi_weighted_centroid).sum(axis=1) / (en_norms * vc_norms)
    return dists, cos


def random_control_cosine(en_vecs, vi_vecs, rng, n_draws=20):
    """Cosine similarity between each EN vector and n_draws random (unrelated)
    VI vectors from the same sentence, averaged. This is the anisotropy
    baseline: if real alignment carries no signal beyond anisotropy, OT-weighted
    cosine and this control should look statistically the same."""
    n_vi = vi_vecs.shape[0]
    draws = min(n_draws, n_vi)
    out = np.zeros(en_vecs.shape[0])
    en_norms = np.linalg.norm(en_vecs, axis=1) + 1e-12
    for i, en_vec in enumerate(en_vecs):
        idx = rng.choice(n_vi, size=draws, replace=False)
        sampled = vi_vecs[idx]
        cos = (sampled @ en_vec) / (np.linalg.norm(sampled, axis=1) * en_norms[i] + 1e-12)
        out[i] = cos.mean()
    return out


def diagnostic_anisotropy(example_dirs, seed=0, output_prefix="alignment_stats"):
    rng = np.random.default_rng(seed)
    ot_cos_all, control_cos_all = [], []
    skipped = 0
    for d in example_dirs:
        try:
            hidden_after = np.load(os.path.join(d, 'hidden_after.npy'))
            gamma, en_ans_rows, num_en_total = load_common(d)
        except Exception as e:
            print(f"[warn] Skipping '{d}' for anisotropy check: {e}")
            skipped += 1
            continue
        if en_ans_rows.size == 0:
            skipped += 1
            continue

        en_vecs = hidden_after[en_ans_rows]
        vi_vecs = hidden_after[num_en_total:]
        sub_gamma = gamma[en_ans_rows, :]

        _, ot_cos = ot_weighted_stats(en_vecs, vi_vecs, sub_gamma)
        control_cos = random_control_cosine(en_vecs, vi_vecs, rng)

        ot_cos_all.extend(ot_cos.tolist())
        control_cos_all.extend(control_cos.tolist())

    if skipped:
        print(f"[warn] Skipped {skipped}/{len(example_dirs)} examples for anisotropy check.")

    ot_cos_all = np.array(ot_cos_all)
    control_cos_all = np.array(control_cos_all)
    n = len(ot_cos_all)

    csv_path = f"{output_prefix}_anisotropy.csv"
    with open(csv_path, 'w') as f:
        f.write("ot_weighted_cosine,random_control_cosine\n")
        for a, b in zip(ot_cos_all, control_cos_all):
            f.write(f"{a},{b}\n")
    print(f"Saved anisotropy control raw values to {csv_path}")

    print(f"\n=== (1) Anisotropy control (n={n} answer tokens) ===")
    print(f"OT-weighted correspondence cosine: mean={ot_cos_all.mean():.4f}, "
          f"std={ot_cos_all.std():.4f}")
    print(f"Random-VI-token control cosine:    mean={control_cos_all.mean():.4f}, "
          f"std={control_cos_all.std():.4f}")
    diff = ot_cos_all - control_cos_all
    print(f"Difference (OT-weighted - random control): mean={diff.mean():.4f}, "
          f"std={diff.std():.4f}")
    try:
        from scipy import stats
        if n >= 2:
            t_stat, t_p = stats.ttest_rel(ot_cos_all, control_cos_all)
            print(f"Paired t-test (OT-weighted vs. random control): t={t_stat:.3f}, p={t_p:.4g}")
    except ImportError:
        print("[note] Install scipy for a significance test here.")

    if abs(diff.mean()) < 0.01:
        print("[interpretation] OT-weighted cosine is barely distinguishable from "
              "a RANDOM Vietnamese token in the same sentence. This strongly "
              "suggests the ~0.99 angular-alignment number is dominated by "
              "embedding-space anisotropy, not real cross-lingual correspondence "
              "— it would be misleading to cite it as evidence of alignment.")
    elif diff.mean() > 0:
        print(f"[interpretation] OT-weighted cosine is meaningfully higher than "
              f"the random-token control (by {diff.mean():.4f} on average), so "
              f"the correspondence does carry some real signal above the "
              f"anisotropy floor — worth reporting alongside the raw cosine value.")
    else:
        print(f"[interpretation] OT-weighted cosine is meaningfully LOWER than "
              f"the random-token control (by {abs(diff.mean()):.4f} on average) "
              f"— the OT-predicted VI counterpart is actually LESS similar in "
              f"direction than a random token, which is unexpected and worth "
              f"investigating (e.g. check whether gamma/sinkhorn is well-behaved, "
              f"not just near-uniform or degenerate).")


def diagnostic_layers(example_dirs, output_prefix="alignment_stats"):
    per_layer_dist = {idx: [] for idx in LAYER_INDICES}
    per_layer_cos = {idx: [] for idx in LAYER_INDICES}
    layer_weight_samples = []
    skipped = 0
    used = 0

    for d in example_dirs:
        try:
            gamma, en_ans_rows, num_en_total = load_common(d)
        except Exception as e:
            print(f"[warn] Skipping '{d}' for layer analysis: {e}")
            skipped += 1
            continue
        if en_ans_rows.size == 0:
            skipped += 1
            continue

        lw_path = os.path.join(d, 'layer_weights.npy')
        if os.path.exists(lw_path):
            layer_weight_samples.append(np.load(lw_path))

        any_layer_found = False
        for idx in LAYER_INDICES:
            en_path = os.path.join(d, f'hidden_en_layer{idx}.npy')
            vi_path = os.path.join(d, f'hidden_vi_layer{idx}.npy')
            if not (os.path.exists(en_path) and os.path.exists(vi_path)):
                continue
            any_layer_found = True
            h_en = np.load(en_path)
            h_vi = np.load(vi_path)

            en_ans_local = en_ans_rows[en_ans_rows < h_en.shape[0]]
            if en_ans_local.size == 0:
                continue
            en_vecs = h_en[en_ans_local]
            sub_gamma = gamma[en_ans_local, :h_vi.shape[0]]

            dists, cos = ot_weighted_stats(en_vecs, h_vi, sub_gamma)
            per_layer_dist[idx].extend(dists.tolist())
            per_layer_cos[idx].extend(cos.tolist())

        if any_layer_found:
            used += 1
        else:
            skipped += 1

    if skipped:
        print(f"[warn] {skipped}/{len(example_dirs)} examples had no usable "
              f"per-layer files (hidden_en_layer*.npy / hidden_vi_layer*.npy).")

    if used == 0:
        print("\n=== (2) Per-layer alignment ===")
        print("[error] No per-layer hidden-state files found in any example. "
              "Make sure exporter.py's per-layer saving step ran (it saves "
              "hidden_en_layer{6,7,8,9}.npy / hidden_vi_layer{6,7,8,9}.npy).")
        return

    print(f"\n=== (2) Per-layer alignment (from {used} examples) ===")
    print(f"{'Layer':<8}{'n':<6}{'Mean dist':<12}{'Mean cosine':<12}")
    layer_summary = {}
    for idx in LAYER_INDICES:
        d_arr = np.array(per_layer_dist[idx])
        c_arr = np.array(per_layer_cos[idx])
        if d_arr.size == 0:
            print(f"{idx:<8}{'0':<6}{'--':<12}{'--':<12}")
            continue
        layer_summary[idx] = (d_arr.mean(), c_arr.mean())
        print(f"{idx:<8}{d_arr.size:<6}{d_arr.mean():<12.4f}{c_arr.mean():<12.4f}")

    if layer_weight_samples:
        lw_stack = np.array(layer_weight_samples)
        if not np.allclose(lw_stack, lw_stack[0], atol=1e-4):
            print("[note] layer_weights.npy differs slightly across examples "
                  "(unexpected if it's a single global parameter) — using the mean.")
        lw_mean = lw_stack.mean(axis=0)
        lw_softmax = np.exp(lw_mean) / np.exp(lw_mean).sum()
        print(f"\nLearned layer weights (raw, averaged): "
              f"{dict(zip(LAYER_INDICES, np.round(lw_mean, 4)))}")
        print(f"Learned layer weights (softmax):        "
              f"{dict(zip(LAYER_INDICES, np.round(lw_softmax, 4)))}")

        csv_path = f"{output_prefix}_layers.csv"
        with open(csv_path, 'w') as f:
            f.write("layer,n,mean_dist,mean_cosine,layer_weight_raw,layer_weight_softmax\n")
            for i, idx in enumerate(LAYER_INDICES):
                if idx in layer_summary:
                    mean_dist, mean_cos = layer_summary[idx]
                    n_tok = len(per_layer_dist[idx])
                else:
                    mean_dist, mean_cos, n_tok = '', '', 0
                f.write(f"{idx},{n_tok},{mean_dist},{mean_cos},"
                        f"{lw_mean[i]},{lw_softmax[i]}\n")
        print(f"Saved per-layer summary to {csv_path}")

        if layer_summary:
            best_dist_layer = min(layer_summary, key=lambda k: layer_summary[k][0])
            best_cos_layer = max(layer_summary, key=lambda k: layer_summary[k][1])
            heaviest_layer = LAYER_INDICES[int(np.argmax(lw_mean))]
            print(f"\nTightest alignment by distance: layer {best_dist_layer}")
            print(f"Tightest alignment by cosine:    layer {best_cos_layer}")
            print(f"Most heavily-weighted layer:     layer {heaviest_layer}")
            if heaviest_layer in (best_dist_layer, best_cos_layer):
                print("[interpretation] The model's learned layer-mixing weight "
                      "favors the SAME layer that shows the tightest cross-"
                      "lingual alignment — consistent with 'the model learned "
                      "to lean on the layer where EN/VI are already best aligned' "
                      "as (part of) the mechanism behind the EM/F1 gain.")
            else:
                print("[interpretation] The most heavily-weighted layer is NOT "
                      "the most cross-lingually aligned one by these metrics — "
                      "the EM/F1 gain is more likely driven by something other "
                      "than raw representation-space alignment at these layers "
                      "(e.g., QA-head calibration, or alignment structure these "
                      "distance/cosine summaries don't capture).")
    else:
        print("[warn] No layer_weights.npy found in any example directory.")


def main():
    parser = argparse.ArgumentParser(
        description="Anisotropy control + per-layer alignment diagnostics")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_prefix", type=str, default="alignment_stats",
                         help="Prefix for saved CSVs (matches aggregate_alignment_stats.py "
                              "so make_summary_figure.py can find everything)")
    args = parser.parse_args()

    example_dirs = find_example_dirs(args.root_dir)
    if not example_dirs:
        print(f"[error] No example subdirectories found under '{args.root_dir}'.")
        return
    print(f"Found {len(example_dirs)} example directories.")

    diagnostic_anisotropy(example_dirs, seed=args.seed, output_prefix=args.output_prefix)
    diagnostic_layers(example_dirs, output_prefix=args.output_prefix)


if __name__ == '__main__':
    main()