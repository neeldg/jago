"""Summarize per-patch JAGO stats into one slide-level architecture profile table.

For each slide directory under --patches-root, reads its patch_stats.csv (as
produced by compute_patch_stats.py) and collapses it into a single row of
mean/median/std summary statistics plus a handful of friendlier named
architecture scores.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# friendly_name -> source column in patch_stats.csv
FRIENDLY_SCORE_COLUMNS = {
    "tumor_rich_mean": "cell_type_frac__neoplastic",
    "stromal_rich_mean": "cell_type_frac__connective",
    "immune_rich_mean": "cell_type_frac__inflammatory",
    "tumor_immune_contact_mean": "edge_type_frac__inflammatory-neoplastic",
    "tumor_stroma_contact_mean": "edge_type_frac__connective-neoplastic",
    "immune_stroma_contact_mean": "edge_type_frac__connective-inflammatory",
    "same_type_clustering_mean": "same_type_edge_fraction",
    "mixed_architecture_mean": "mixed_type_edge_fraction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize JAGO patch_stats.csv files into one slide-level architecture profile table."
    )
    parser.add_argument(
        "--patches-root",
        required=True,
        type=Path,
        help="Path to a directory containing one subdirectory per slide, each with a patch_stats.csv.",
    )
    parser.add_argument(
        "--rankings-root",
        required=False,
        type=Path,
        default=None,
        help="Optional path to a directory containing one ranking subdirectory per slide.",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Path to write the output slide-level profile CSV."
    )
    return parser.parse_args()


def summarize_slide(slide_id: str, patch_stats: pd.DataFrame) -> dict:
    numeric_stats = patch_stats.select_dtypes(include=np.number)

    row = {"slide_id": slide_id, "n_patches": len(patch_stats)}

    for column in numeric_stats.columns:
        row[f"{column}__mean"] = numeric_stats[column].mean()
        row[f"{column}__median"] = numeric_stats[column].median()
        row[f"{column}__std"] = numeric_stats[column].std()

    for friendly_name, source_column in FRIENDLY_SCORE_COLUMNS.items():
        if source_column in numeric_stats.columns:
            row[friendly_name] = numeric_stats[source_column].mean()
        else:
            print(
                f"Warning: slide '{slide_id}' has no column '{source_column}' "
                f"(needed for '{friendly_name}'); this cell/edge type combination "
                f"may not be present in this slide. Setting it to NaN."
            )
            row[friendly_name] = np.nan

    return row


def main() -> None:
    args = parse_args()

    if not args.patches_root.exists():
        raise FileNotFoundError(f"Patches root not found: {args.patches_root}")

    if args.rankings_root is not None and not args.rankings_root.exists():
        raise FileNotFoundError(f"Rankings root not found: {args.rankings_root}")

    slide_dirs = sorted(p for p in args.patches_root.iterdir() if p.is_dir())
    if not slide_dirs:
        raise FileNotFoundError(f"No slide subdirectories found under {args.patches_root}")

    rows = []
    for slide_dir in slide_dirs:
        slide_id = slide_dir.name
        patch_stats_path = slide_dir / "patch_stats.csv"

        if not patch_stats_path.exists():
            print(f"Skipping '{slide_id}': {patch_stats_path} not found.")
            continue

        patch_stats = pd.read_csv(patch_stats_path)
        rows.append(summarize_slide(slide_id, patch_stats))

    profiles = pd.DataFrame(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(args.out, index=False)

    print(f"Number of slides summarized: {len(rows)}")
    print(f"Output path: {args.out}")


if __name__ == "__main__":
    main()
