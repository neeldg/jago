"""Score and rank batch5 JAGO graph patches by architecture.

Reads patch_<id>_cells.csv / patch_<id>_edges.csv pairs from a patches
root directory, computes per-patch architecture scores, and writes a
combined scores CSV plus top-N ranking CSVs for each score.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TYPE_MAP = {
    "type_0": "nolabe",
    "type_1": "neoplastic",
    "type_2": "inflammatory",
    "type_3": "connective",
    "type_4": "necrotic",
    "type_5": "non_neoplastic_epithelial",
}

CELL_COLUMNS = ["cell_id", "x_um", "y_um", "cell_type"]
EDGE_COLUMNS = [
    "source_cell_id",
    "target_cell_id",
    "distance_um",
    "source_cell_type",
    "target_cell_type",
]

# ranking name -> column in the combined scores table
RANKING_COLUMNS = {
    "tumor_rich_score": "tumor_frac",
    "immune_rich_score": "immune_frac",
    "stromal_rich_score": "stroma_frac",
    "necrotic_rich_score": "necrotic_frac",
    "mixed_architecture_score": "mixed_type_edge_fraction",
    "tumor_immune_contact_score": "tumor_immune_contact",
    "tumor_stroma_contact": "tumor_stroma_contact",
    "immune_stroma_contact": "immune_stroma_contact",
    "n_cells": "n_cells",
    "mean_degree": "mean_degree",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score and rank batch5 JAGO graph patches."
    )
    parser.add_argument(
        "--patches-root", required=True, type=Path,
        help="Directory containing patch_<id>_cells.csv/_edges.csv files.",
    )
    parser.add_argument(
        "--outdir", required=True, type=Path,
        help="Directory to write the scores CSV and ranking CSVs into.",
    )
    parser.add_argument(
        "--top-n", required=False, type=int, default=25,
        help="Number of top patches to keep per ranking CSV.",
    )
    return parser.parse_args()


def find_cell_files(patches_root):
    return sorted(patches_root.glob("*/*_cells.csv"))


def load_table(path, required_columns):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    table = pd.read_csv(path, dtype=str)

    missing = [c for c in required_columns if c not in table.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {missing}. "
            f"Required columns are: {required_columns}"
        )

    return table


def cell_type_entropy(readable_types):
    counts = readable_types.value_counts()
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def edge_pair_fraction(edges, readable_source, readable_target, type_a, type_b):
    n_edges = len(edges)
    if n_edges == 0:
        return 0.0

    forward = (readable_source == type_a) & (readable_target == type_b)
    backward = (readable_source == type_b) & (readable_target == type_a)
    return float((forward | backward).sum() / n_edges)


def compute_patch_score(patch_id, cells, edges):
    n_cells = len(cells)
    n_edges = len(edges)
    mean_degree = (2 * n_edges / n_cells) if n_cells > 0 else 0.0

    readable_cell_type = cells["cell_type"].map(lambda t: TYPE_MAP.get(t, t))
    cell_type_counts = readable_cell_type.value_counts()

    def cell_frac(name):
        return float(cell_type_counts.get(name, 0) / n_cells) if n_cells > 0 else 0.0

    tumor_frac = cell_frac("neoplastic")
    immune_frac = cell_frac("inflammatory")
    stroma_frac = cell_frac("connective")
    necrotic_frac = cell_frac("necrotic")
    epithelial_frac = cell_frac("non_neoplastic_epithelial")
    nolabe_frac = cell_frac("nolabe")

    entropy = cell_type_entropy(readable_cell_type) if n_cells > 0 else 0.0

    if n_edges > 0:
        readable_source = edges["source_cell_type"].map(lambda t: TYPE_MAP.get(t, t))
        readable_target = edges["target_cell_type"].map(lambda t: TYPE_MAP.get(t, t))

        same_type_mask = readable_source == readable_target
        same_type_edge_fraction = float(same_type_mask.sum() / n_edges)
        mixed_type_edge_fraction = 1.0 - same_type_edge_fraction

        tumor_immune_contact = edge_pair_fraction(
            edges, readable_source, readable_target, "neoplastic", "inflammatory"
        )
        tumor_stroma_contact = edge_pair_fraction(
            edges, readable_source, readable_target, "neoplastic", "connective"
        )
        immune_stroma_contact = edge_pair_fraction(
            edges, readable_source, readable_target, "inflammatory", "connective"
        )
    else:
        same_type_edge_fraction = 0.0
        mixed_type_edge_fraction = 0.0
        tumor_immune_contact = 0.0
        tumor_stroma_contact = 0.0
        immune_stroma_contact = 0.0

    return {
        "patch_id": patch_id,
        "n_cells": n_cells,
        "n_edges": n_edges,
        "mean_degree": mean_degree,
        "tumor_frac": tumor_frac,
        "immune_frac": immune_frac,
        "stroma_frac": stroma_frac,
        "necrotic_frac": necrotic_frac,
        "epithelial_frac": epithelial_frac,
        "nolabe_frac": nolabe_frac,
        "cell_type_entropy": entropy,
        "same_type_edge_fraction": same_type_edge_fraction,
        "mixed_type_edge_fraction": mixed_type_edge_fraction,
        "tumor_immune_contact": tumor_immune_contact,
        "tumor_stroma_contact": tumor_stroma_contact,
        "immune_stroma_contact": immune_stroma_contact,
    }


def main():
    args = parse_args()

    if not args.patches_root.exists():
        raise FileNotFoundError(f"Patches root not found: {args.patches_root}")

    cell_files = find_cell_files(args.patches_root)
    if not cell_files:
        raise FileNotFoundError(
            f"No */*_cells.csv files found in {args.patches_root}"
        )

    rows = []
    for cells_path in cell_files:
        edges_path = Path(str(cells_path).replace("_cells.csv", "_edges.csv"))

        slide_id = cells_path.parent.name
        patch_name = cells_path.name[: -len("_cells.csv")]
        patch_id = f"{slide_id}_{patch_name}"

        cells = load_table(cells_path, CELL_COLUMNS)
        edges = load_table(edges_path, EDGE_COLUMNS)

        rows.append(compute_patch_score(patch_id, cells, edges))

    scores = pd.DataFrame(rows)

    args.outdir.mkdir(parents=True, exist_ok=True)
    scores_path = args.outdir / "batch5_patch_architecture_scores.csv"
    scores.to_csv(scores_path, index=False)

    for ranking_name, column in RANKING_COLUMNS.items():
        ranked = scores.sort_values(column, ascending=False)
        top_n = ranked.head(args.top_n)
        out_path = args.outdir / f"top_{ranking_name}.csv"
        top_n.to_csv(out_path, index=False)
        print(f"Top 5 patches for '{ranking_name}': "
              f"{top_n['patch_id'].head(5).tolist()}")

    print(f"Number of patches scored: {len(rows)}")
    print(f"Output path: {scores_path}")


if __name__ == "__main__":
    main()
