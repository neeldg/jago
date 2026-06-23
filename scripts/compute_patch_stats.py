"""Compute per-patch summary statistics for JAGO graph patches.

Reads a patch manifest (as produced by make_graph_patches.py) and, for each
patch, loads its cells/edges CSVs and computes the same family of graph
statistics as compute_graph_stats.py, then writes one row per patch to a
single output CSV.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_CELL_COLUMNS = ["cell_id", "x_um", "y_um", "cell_type"]
REQUIRED_EDGE_COLUMNS = [
    "source_cell_id",
    "target_cell_id",
    "distance_um",
    "source_cell_type",
    "target_cell_type",
]
REQUIRED_MANIFEST_COLUMNS = [
    "patch_id",
    "center_x_um",
    "center_y_um",
    "x_min_um",
    "x_max_um",
    "y_min_um",
    "y_max_um",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-patch summary statistics for JAGO graph patches."
    )
    parser.add_argument(
        "--patch-dir",
        required=True,
        type=Path,
        help="Directory containing patch_<id>_cells.csv and patch_<id>_edges.csv files.",
    )
    parser.add_argument(
        "--manifest", required=True, type=Path, help="Path to patch_manifest.csv."
    )
    parser.add_argument(
        "--type-map",
        required=False,
        type=Path,
        default=None,
        help="Optional JSON file mapping type ids (e.g. 'type_1') to readable names.",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Path to write the output per-patch stats CSV."
    )
    return parser.parse_args()


def load_table(path: Path, required_columns: list, dtype=str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    table = pd.read_csv(path, dtype=dtype)

    missing = [col for col in required_columns if col not in table.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {missing}. "
            f"Required columns are: {required_columns}"
        )

    return table


def load_type_map(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Type map not found: {path}")

    with path.open() as f:
        type_map = json.load(f)

    if not isinstance(type_map, dict):
        raise ValueError(f"Type map {path} must be a JSON object of type id -> readable name.")

    return type_map


def edge_type_key(type_a: str, type_b: str, type_map: dict = None) -> str:
    if type_map:
        type_a = type_map.get(type_a, type_a)
        type_b = type_map.get(type_b, type_b)
    return "-".join(sorted([type_a, type_b]))


def compute_patch_stats(cells: pd.DataFrame, edges: pd.DataFrame, type_map: dict = None) -> dict:
    n_cells = len(cells)
    n_edges = len(edges)
    mean_degree = (2 * n_edges / n_cells) if n_cells > 0 else 0.0

    if type_map:
        readable_cell_type = cells["cell_type"].map(lambda t: type_map.get(t, t))
    else:
        readable_cell_type = cells["cell_type"]

    cell_type_counts = readable_cell_type.value_counts().to_dict()
    cell_type_fractions = (
        {k: v / n_cells for k, v in cell_type_counts.items()} if n_cells > 0 else {}
    )

    if n_edges > 0:
        edge_type_keys = edges.apply(
            lambda row: edge_type_key(row["source_cell_type"], row["target_cell_type"], type_map),
            axis=1,
        )
        edge_type_counts = edge_type_keys.value_counts().to_dict()
        edge_type_fractions = {k: v / n_edges for k, v in edge_type_counts.items()}

        same_type_mask = edges["source_cell_type"] == edges["target_cell_type"]
        same_type_edge_fraction = same_type_mask.sum() / n_edges
        mixed_type_edge_fraction = 1 - same_type_edge_fraction

        distances = edges["distance_um"].astype(float)
        mean_edge_distance_um = float(distances.mean())
        median_edge_distance_um = float(distances.median())
    else:
        edge_type_fractions = {}
        same_type_edge_fraction = 0.0
        mixed_type_edge_fraction = 0.0
        mean_edge_distance_um = 0.0
        median_edge_distance_um = 0.0

    return {
        "n_cells": n_cells,
        "n_edges": n_edges,
        "mean_degree": mean_degree,
        "cell_type_fractions": cell_type_fractions,
        "edge_type_fractions": edge_type_fractions,
        "same_type_edge_fraction": same_type_edge_fraction,
        "mixed_type_edge_fraction": mixed_type_edge_fraction,
        "mean_edge_distance_um": mean_edge_distance_um,
        "median_edge_distance_um": median_edge_distance_um,
    }


def main() -> None:
    args = parse_args()

    manifest = load_table(args.manifest, REQUIRED_MANIFEST_COLUMNS, dtype=None)
    type_map = load_type_map(args.type_map) if args.type_map else None

    per_patch_stats = []
    all_cell_types = set()
    all_edge_types = set()

    for _, manifest_row in manifest.iterrows():
        patch_id = manifest_row["patch_id"]

        cells_path = args.patch_dir / f"{patch_id}_cells.csv"
        edges_path = args.patch_dir / f"{patch_id}_edges.csv"

        cells = load_table(cells_path, REQUIRED_CELL_COLUMNS, dtype=str)
        edges = load_table(edges_path, REQUIRED_EDGE_COLUMNS, dtype=str)

        stats = compute_patch_stats(cells, edges, type_map)
        all_cell_types.update(stats["cell_type_fractions"].keys())
        all_edge_types.update(stats["edge_type_fractions"].keys())

        per_patch_stats.append(
            {
                "patch_id": patch_id,
                "center_x_um": manifest_row["center_x_um"],
                "center_y_um": manifest_row["center_y_um"],
                "x_min_um": manifest_row["x_min_um"],
                "x_max_um": manifest_row["x_max_um"],
                "y_min_um": manifest_row["y_min_um"],
                "y_max_um": manifest_row["y_max_um"],
                "stats": stats,
            }
        )

    cell_type_columns = sorted(all_cell_types)
    edge_type_columns = sorted(all_edge_types)

    rows = []
    for entry in per_patch_stats:
        stats = entry["stats"]
        row = {
            "patch_id": entry["patch_id"],
            "center_x_um": entry["center_x_um"],
            "center_y_um": entry["center_y_um"],
            "x_min_um": entry["x_min_um"],
            "x_max_um": entry["x_max_um"],
            "y_min_um": entry["y_min_um"],
            "y_max_um": entry["y_max_um"],
            "n_cells": stats["n_cells"],
            "n_edges": stats["n_edges"],
            "mean_degree": stats["mean_degree"],
            "same_type_edge_fraction": stats["same_type_edge_fraction"],
            "mixed_type_edge_fraction": stats["mixed_type_edge_fraction"],
            "mean_edge_distance_um": stats["mean_edge_distance_um"],
            "median_edge_distance_um": stats["median_edge_distance_um"],
        }

        for cell_type in cell_type_columns:
            row[f"cell_type_frac__{cell_type}"] = stats["cell_type_fractions"].get(cell_type, 0.0)

        for edge_type in edge_type_columns:
            row[f"edge_type_frac__{edge_type}"] = stats["edge_type_fractions"].get(edge_type, 0.0)

        rows.append(row)

    out_df = pd.DataFrame(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print(f"Number of patches processed: {len(rows)}")


if __name__ == "__main__":
    main()
