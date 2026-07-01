"""Inspect the JAGO neighborhood-completion dataset before training.

Loads patches, generates samples, and reports per-split statistics: sample
counts, hidden-cell counts, context-cell counts, ring-neighbour coverage, and
a worked example for one sample.

Usage:
    python scripts/inspect_neighborhood_dataset.py \
        --patch-root /scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/patches \
        [--samples-per-patch 5] [--mask-radius-um 100] [--seed 0]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jago_gnn.dataset import build_cell_type_vocab, find_patch_files, load_patch, split_by_slide
from jago_gnn.neighborhood_dataset import NeighborhoodCompletionDataset

TYPE_ID_LABELS = {
    "0": "No label", "1": "Neoplastic", "2": "Inflammatory",
    "3": "Connective", "4": "Necrotic", "5": "Non-neoplastic epithelial",
}


def human_readable(raw: str) -> str:
    label = TYPE_ID_LABELS.get(str(raw))
    return f"type_{raw} ({label})" if label else str(raw)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect JAGO neighborhood-completion dataset.")
    p.add_argument("--patch-root", required=True, type=Path)
    p.add_argument("--samples-per-patch", type=int, default=5)
    p.add_argument("--mask-radius-um", type=float, default=100.0)
    p.add_argument("--min-hidden-cells", type=int, default=5)
    p.add_argument("--min-context-cells", type=int, default=20)
    p.add_argument("--max-patches", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _report_split(name: str, ds: NeighborhoodCompletionDataset, num_classes: int, idx_to_class: dict) -> None:
    samples = ds.samples
    if not samples:
        print(f"\n{name}: 0 samples (skipped {ds.n_skipped}).")
        return

    n_hidden = np.array([s["n_hidden"] for s in samples])
    n_context = np.array([s["n_context"] for s in samples])
    n_edges = np.array([s["edge_index"].shape[1] for s in samples])
    has_ring = np.array([s["ring_frac"] is not None for s in samples])

    print(f"\n{name}: {len(samples)} samples ({ds.n_skipped} skipped)")
    print(f"  Hidden cells   — mean: {n_hidden.mean():.1f}, min: {n_hidden.min()}, max: {n_hidden.max()}")
    print(f"  Context cells  — mean: {n_context.mean():.1f}, min: {n_context.min()}, max: {n_context.max()}")
    print(f"  Context edges  — mean: {n_edges.mean():.1f}, min: {n_edges.min()}, max: {n_edges.max()}")
    print(f"  Ring coverage  — {has_ring.mean():.1%} of samples have ≥1 ring-neighbour cell")

    # Mean composition of hidden regions
    all_comp = np.stack([s["target_composition"].numpy() for s in samples], axis=0)
    print(f"  Mean hidden composition:")
    for cid in sorted(idx_to_class):
        print(f"    {human_readable(idx_to_class[cid])}: {all_comp[:, cid].mean():.3f}")

    # log1p count distribution
    all_counts = np.array([s["target_count"].item() for s in samples])
    print(f"  log1p(n_hidden) — mean: {all_counts.mean():.3f}, min: {all_counts.min():.3f}, max: {all_counts.max():.3f}")
    print(f"  n_hidden        — mean ≈ {np.expm1(all_counts.mean()):.1f} cells")


def main() -> None:
    args = parse_args()

    records = find_patch_files(args.patch_root)
    print(f"Found {len(records)} patch file(s) under {args.patch_root}")

    if args.max_patches is not None:
        records = records[: args.max_patches]
        print(f"Limiting to first {len(records)} patch(es) (--max-patches).")

    patches, n_skipped = [], 0
    for rec in records:
        try:
            patches.append(load_patch(rec))
        except (ValueError, FileNotFoundError) as exc:
            print(f"  Warning: skipping {rec['cells_path']}: {exc}")
            n_skipped += 1

    if not patches:
        print("ERROR: no valid patches loaded.")
        sys.exit(1)

    print(f"Loaded {len(patches)} patch(es) ({n_skipped} skipped), "
          f"across {len({p['slide_id'] for p in patches})} slide(s).")

    vocab = build_cell_type_vocab(patches)
    num_classes = len(vocab) - 1
    idx_to_class = {idx: name for name, idx in vocab.items() if name != "<MASK>"}

    print(f"\nCell type vocab ({num_classes} classes):")
    for name, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        if name != "<MASK>":
            print(f"  {idx}: {human_readable(name)}")

    train_patches, val_patches, test_patches = split_by_slide(patches, seed=args.seed)

    ds_kwargs = dict(
        vocab=vocab,
        samples_per_patch=args.samples_per_patch,
        mask_radius_um=args.mask_radius_um,
        min_hidden_cells=args.min_hidden_cells,
        min_context_cells=args.min_context_cells,
        seed=args.seed,
        deterministic=True,
    )

    print(f"\nBuilding datasets (radius={args.mask_radius_um} µm, "
          f"{args.samples_per_patch} samples/patch, "
          f"min_hidden={args.min_hidden_cells}, min_context={args.min_context_cells}) ...")

    train_ds = NeighborhoodCompletionDataset(train_patches, **ds_kwargs)
    val_ds = NeighborhoodCompletionDataset(val_patches, **ds_kwargs)
    test_ds = NeighborhoodCompletionDataset(test_patches, **ds_kwargs)

    _report_split("TRAIN", train_ds, num_classes, idx_to_class)
    _report_split("VAL", val_ds, num_classes, idx_to_class)
    _report_split("TEST", test_ds, num_classes, idx_to_class)

    # Worked example
    all_samples = train_ds.samples or val_ds.samples or test_ds.samples
    if all_samples:
        ex = all_samples[0]
        print(f"\nWorked example — first sample:")
        print(f"  slide_id={ex['slide_id']}  patch_id={ex['patch_id']}  sample_idx={ex['sample_idx']}")
        print(f"  center=({ex['center_x']:.1f}, {ex['center_y']:.1f}) µm  "
              f"n_hidden={ex['n_hidden']}  n_context={ex['n_context']}")
        print(f"  context edge_index shape: {tuple(ex['edge_index'].shape)}  (symmetrized)")
        print(f"  node feature shape: {tuple(ex['x'].shape)}  (= [x_norm, y_norm, one_hot({num_classes})])")
        print(f"  target_count (log1p): {ex['target_count'].item():.3f}  "
              f"≈ {np.expm1(ex['target_count'].item()):.1f} hidden cells")
        print(f"  target_composition:")
        comp = ex["target_composition"].numpy()
        for cid in sorted(idx_to_class):
            bar = "#" * int(comp[cid] * 30)
            print(f"    {human_readable(idx_to_class[cid]):<40s}: {comp[cid]:.3f}  {bar}")
        print(f"  has ring neighbours: {ex['ring_frac'] is not None}")


if __name__ == "__main__":
    main()
