"""
list_tb_tags.py — List every scalar tag found in a TensorBoard log directory.

Run this FIRST on both your M4 and M5 logdirs to find the exact tag names for
F1 and EM (they vary by project — e.g. 'eval/f1', 'val_f1', 'f1_score', ...).
Then pass those exact names to plot_f1_em_curves.py.

Usage:
    python list_tb_tags.py --logdir runs/M4_static_margin
    python list_tb_tags.py --logdir runs/M5_dynamic_curriculum
"""
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True,
                         help="Directory containing the TensorBoard event file(s) for one run")
    args = parser.parse_args()

    ea = EventAccumulator(args.logdir, size_guidance={'scalars': 0, 'tensors': 0})
    ea.Reload()
    tags_dict = ea.Tags()
    scalar_tags = tags_dict.get('scalars', [])
    tensor_tags = tags_dict.get('tensors', [])

    if not scalar_tags and not tensor_tags:
        print(f"[error] No scalar (or tensor-based scalar) tags found under "
              f"'{args.logdir}'. Check the path — it should point directly at "
              f"the folder containing the 'events.out.tfevents.*' file (or its "
              f"parent if TensorBoard nested it).")
        return

    if scalar_tags:
        print(f"Found {len(scalar_tags)} old-style scalar tags in '{args.logdir}':")
        for t in sorted(scalar_tags):
            events = ea.Scalars(t)
            print(f"  {t}  (n={len(events)}, step range {events[0].step}-{events[-1].step})")

    if tensor_tags:
        print(f"\nFound {len(tensor_tags)} tensor-based scalar tags in '{args.logdir}' "
              f"(these work the same way — just a newer TB wire format):")
        for t in sorted(tensor_tags):
            events = ea.Tensors(t)
            print(f"  {t}  (n={len(events)}, step range {events[0].step}-{events[-1].step})")


if __name__ == '__main__':
    main()