"""Rank JAGO graph patches by biologically meaningful architecture scores.

Reads the per-patch stats CSV produced by compute_patch_stats.py and derives
a handful of named "architecture scores" (e.g. how tumor-rich a patch is, or
how much tumor-immune contact it shows), then writes top-N rankings for each
score plus one combined scores CSV.
"""

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_BASE_COLUMNS = ["patch_id"]

# score_name -> source column in patch-stats CSV
SCORE_DEFINITIONS = {
    "tumor_rich_score": "cell_type_frac__neoplastic",
    "stromal_rich_score": "cell_type_frac__connective",
    "immune_rich_score": "cell_type_frac__inflammatory",
    "tumor_immune_contact_score": "edge_type_frac__inflammatory-neoplastic",
    "tumor_stroma_contact_score": "edge_type_frac__connective-neoplastic",
    "immune_stroma_contact_score": "edge_type_frac__connective-inflammatory",
    "same_type_clustering_score": "same_type_edge_fraction",
    "mixed_architecture_score": "mixed_type_edge_fraction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank JAGO graph patches by biologically meaningful architecture scores."
    )
    parser.add_argument(
        "--patch-stats", required=True, type=Path, help="Path to patch stats CSV from compute_patch_stats.py."
    )
    parser.add_argument(
        "--out-dir", required=True, type=Path, help="Directory to write ranked CSVs into."
    )
    parser.add_argument(
        "--top-n", required=False, type=int, default=10, help="Number of top patches to keep per score."
    )
    return parser.parse_args()


def load_patch_stats(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Patch stats CSV not found: {path}")

    patch_stats = pd.read_csv(path)

    missing = [col for col in REQUIRED_BASE_COLUMNS if col not in patch_stats.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {missing}. "
            f"Required columns are: {REQUIRED_BASE_COLUMNS}"
        )

    return patch_stats


def main() -> None:
    args = parse_args()

    patch_stats = load_patch_stats(args.patch_stats)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    combined = patch_stats.copy()
    available_scores = []

    for score_name, source_column in SCORE_DEFINITIONS.items():
        if source_column not in patch_stats.columns:
            print(
                f"Skipping '{score_name}': column '{source_column}' not found in "
                f"{args.patch_stats} (this cell/edge type combination may not be "
                f"present in this dataset)."
            )
            continue

        ranked = patch_stats.copy()
        ranked[score_name] = ranked[source_column]
        ranked = ranked.sort_values(score_name, ascending=False)

        top_n = ranked.head(args.top_n)
        top_n.to_csv(args.out_dir / f"top_{score_name}.csv", index=False)

        combined[score_name] = patch_stats[source_column]
        available_scores.append(score_name)

        top_5_ids = ranked["patch_id"].head(5).tolist()
        print(f"Top 5 patches for '{score_name}': {top_5_ids}")

    combined.to_csv(args.out_dir / "patch_architecture_scores.csv", index=False)

    if not available_scores:
        print(
            "Warning: none of the expected score source columns were found in "
            f"{args.patch_stats}. patch_architecture_scores.csv was written with no score columns added."
        )


if __name__ == "__main__":
    main()
