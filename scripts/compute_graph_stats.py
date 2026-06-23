"""Compute summary statistics for a JAGO spatial cell graph.

Reads a cell table CSV and its corresponding radius-graph edge list CSV and
writes a JSON summary covering graph size, degree, cell/edge type
composition, and edge distance statistics.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_CELL_COLUMNS = ["cell_id", "x_um", "y_um", "cell_type"]
REQUIRED_EDGE_COLUMNS = [
    "source_cell_id",
    "target_cell_id",
    "distance_um",
    "source_cell_type",
    "target_cell_type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute summary statistics for a JAGO spatial cell graph."
    )
    parser.add_argument(
        "--cells", required=True, type=Path, help="Path to input cell table CSV."
    )
    parser.add_argument(
        "--edges", required=True, type=Path, help="Path to input radius-graph edge list CSV."
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Path to write the output stats JSON."
    )
    return parser.parse_args()


def load_table(path: Path, required_columns: list) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    table = pd.read_csv(path, dtype=str)

    missing = [col for col in required_columns if col not in table.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {missing}. "
            f"Required columns are: {required_columns}"
        )

    return table


def edge_type_key(type_a: str, type_b: str) -> str:
    return "-".join(sorted([type_a, type_b]))


def compute_stats(cells: pd.DataFrame, edges: pd.DataFrame) -> dict:
    n_cells = len(cells)
    n_edges = len(edges)
    mean_degree = (2 * n_edges / n_cells) if n_cells > 0 else 0.0

    cell_type_counts = cells["cell_type"].value_counts().to_dict()
    cell_type_fractions = (
        {k: v / n_cells for k, v in cell_type_counts.items()} if n_cells > 0 else {}
    )

    if n_edges > 0:
        edge_type_keys = edges.apply(
            lambda row: edge_type_key(row["source_cell_type"], row["target_cell_type"]),
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
        edge_type_counts = {}
        edge_type_fractions = {}
        same_type_edge_fraction = 0.0
        mixed_type_edge_fraction = 0.0
        mean_edge_distance_um = 0.0
        median_edge_distance_um = 0.0

    return {
        "n_cells": n_cells,
        "n_edges": n_edges,
        "mean_degree": mean_degree,
        "cell_type_counts": cell_type_counts,
        "cell_type_fractions": cell_type_fractions,
        "edge_type_counts": edge_type_counts,
        "edge_type_fractions": edge_type_fractions,
        "same_type_edge_fraction": same_type_edge_fraction,
        "mixed_type_edge_fraction": mixed_type_edge_fraction,
        "mean_edge_distance_um": mean_edge_distance_um,
        "median_edge_distance_um": median_edge_distance_um,
    }


def print_summary(stats: dict, out_path: Path) -> None:
    print(f"Number of cells: {stats['n_cells']}")
    print(f"Number of edges: {stats['n_edges']}")
    print(f"Mean degree: {stats['mean_degree']:.4f}")
    print("Cell type counts:")
    for cell_type, count in stats["cell_type_counts"].items():
        fraction = stats["cell_type_fractions"][cell_type]
        print(f"  {cell_type}: {count} ({fraction:.2%})")
    print("Edge type counts:")
    for edge_type, count in stats["edge_type_counts"].items():
        fraction = stats["edge_type_fractions"][edge_type]
        print(f"  {edge_type}: {count} ({fraction:.2%})")
    print(f"Same-type edge fraction: {stats['same_type_edge_fraction']:.4f}")
    print(f"Mixed-type edge fraction: {stats['mixed_type_edge_fraction']:.4f}")
    print(f"Mean edge distance (um): {stats['mean_edge_distance_um']:.4f}")
    print(f"Median edge distance (um): {stats['median_edge_distance_um']:.4f}")
    print(f"Output path: {out_path}")


def main() -> None:
    args = parse_args()

    cells = load_table(args.cells, REQUIRED_CELL_COLUMNS)
    edges = load_table(args.edges, REQUIRED_EDGE_COLUMNS)

    stats = compute_stats(cells, edges)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(stats, f, indent=2)

    print_summary(stats, args.out)


if __name__ == "__main__":
    main()
