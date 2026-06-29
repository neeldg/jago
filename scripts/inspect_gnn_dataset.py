"""Inspect the JAGO masked-cell GNN dataset before training.

Loads every patch under --patch-root, reports the cell-type vocabulary,
per-patch node/edge counts, the train/val/test slide split, and a worked
example of one masked patch -- useful for sanity-checking the dataset
pipeline without spending time on a full training run.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jago_gnn.dataset import (
    MASK_TOKEN,
    MaskedCellDataset,
    build_cell_type_vocab,
    find_patch_files,
    load_patch,
    split_by_slide,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the JAGO masked-cell GNN dataset before training."
    )
    parser.add_argument("--patch-root", required=True, type=Path)
    parser.add_argument("--max-patches", required=False, type=int, default=None)
    parser.add_argument("--mask-rate", required=False, type=float, default=0.2)
    parser.add_argument("--seed", required=False, type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = find_patch_files(args.patch_root)
    print(f"Found {len(records)} patch file(s) under {args.patch_root}")

    if args.max_patches is not None:
        records = records[: args.max_patches]
        print(f"Loading first {len(records)} patch(es) (--max-patches).")

    patches = []
    n_skipped = 0
    for record in records:
        try:
            patches.append(load_patch(record))
        except (ValueError, FileNotFoundError) as exc:
            print(f"Warning: skipping {record['cells_path']}: {exc}")
            n_skipped += 1

    if not patches:
        raise ValueError(f"No valid patches could be loaded from {args.patch_root}.")

    n_slides = len({p["slide_id"] for p in patches})
    n_nodes = [len(p["cell_ids"]) for p in patches]
    n_edges = [p["edge_index"].shape[1] // 2 for p in patches]  # symmetrized, so /2 = undirected count

    print(f"Loaded {len(patches)} patch(es) ({n_skipped} skipped) across {n_slides} slide(s).")
    print(f"Nodes per patch: mean={np.mean(n_nodes):.1f}, min={min(n_nodes)}, max={max(n_nodes)}")
    print(f"Undirected edges per patch: mean={np.mean(n_edges):.1f}, min={min(n_edges)}, max={max(n_edges)}")

    vocab = build_cell_type_vocab(patches)
    print(f"\nCell type vocab ({len(vocab) - 1} classes, plus {MASK_TOKEN}):")
    for cell_type, idx in vocab.items():
        print(f"  {idx}: {cell_type}")

    all_types = np.concatenate([p["cell_type_raw"] for p in patches])
    unique, counts = np.unique(all_types, return_counts=True)
    print("\nCell type counts across all loaded patches:")
    for cell_type, count in sorted(zip(unique, counts), key=lambda kv: -kv[1]):
        print(f"  {cell_type}: {count} ({count / len(all_types):.2%})")

    train_patches, val_patches, test_patches = split_by_slide(patches, seed=args.seed)
    print("\nTrain/val/test split by slide_id:")
    for name, split_patches in [("train", train_patches), ("val", val_patches), ("test", test_patches)]:
        split_slides = sorted({p["slide_id"] for p in split_patches})
        print(f"  {name}: {len(split_patches)} patches, {len(split_slides)} slides -> {split_slides}")

    print("\nWorked example: masking the first loaded patch")
    example_dataset = MaskedCellDataset(patches[:1], vocab, mask_rate=args.mask_rate, seed=args.seed)
    example = example_dataset[0]
    print(f"  slide_id={example['slide_id']}, patch_id={example['patch_id']}")
    print(f"  node feature shape: {tuple(example['x'].shape)} (= [x_norm, y_norm, one_hot(type incl. MASK)])")
    print(f"  edge_index shape: {tuple(example['edge_index'].shape)} (symmetrized)")
    print(f"  n_nodes={example['n_nodes']}, n_masked={int(example['mask'].sum())} "
          f"({int(example['mask'].sum()) / example['n_nodes']:.1%})")


if __name__ == "__main__":
    main()
