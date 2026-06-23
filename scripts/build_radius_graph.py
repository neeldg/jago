"""Build an undirected spatial radius graph from a Hover-Net cell table.

Reads a cell table CSV (cell_id, x_um, y_um, cell_type) and connects every
pair of cells whose centroids fall within --radius-um of each other.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REQUIRED_COLUMNS = ["cell_id", "x_um", "y_um", "cell_type"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an undirected radius graph from a Hover-Net cell table."
    )
    parser.add_argument(
        "--cells", required=True, type=Path, help="Path to input cell table CSV."
    )
    parser.add_argument(
        "--radius-um",
        required=True,
        type=float,
        help="Radius (in microns) within which two cells are connected by an edge.",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Path to write the output edge list CSV."
    )
    return parser.parse_args()


def load_cells(cells_path: Path) -> pd.DataFrame:
    if not cells_path.exists():
        raise FileNotFoundError(f"Cell table not found: {cells_path}")

    cells = pd.read_csv(cells_path, dtype={"cell_id": str})

    missing = [col for col in REQUIRED_COLUMNS if col not in cells.columns]
    if missing:
        raise ValueError(
            f"Cell table {cells_path} is missing required column(s): {missing}. "
            f"Required columns are: {REQUIRED_COLUMNS}"
        )

    cells["cell_id"] = cells["cell_id"].astype(str)
    return cells


def build_radius_graph(cells: pd.DataFrame, radius_um: float) -> pd.DataFrame:
    coords = cells[["x_um", "y_um"]].to_numpy(dtype=float)

    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=radius_um, output_type="ndarray")

    if pairs.size == 0:
        return pd.DataFrame(
            columns=[
                "source_cell_id",
                "target_cell_id",
                "distance_um",
                "source_cell_type",
                "target_cell_type",
            ]
        )

    source_idx = pairs[:, 0]
    target_idx = pairs[:, 1]

    distances = np.linalg.norm(coords[source_idx] - coords[target_idx], axis=1)

    cell_id = cells["cell_id"].to_numpy()
    cell_type = cells["cell_type"].to_numpy()

    edges = pd.DataFrame(
        {
            "source_cell_id": cell_id[source_idx],
            "target_cell_id": cell_id[target_idx],
            "distance_um": distances,
            "source_cell_type": cell_type[source_idx],
            "target_cell_type": cell_type[target_idx],
        }
    )
    return edges


def main() -> None:
    args = parse_args()

    cells = load_cells(args.cells)
    edges = build_radius_graph(cells, args.radius_um)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(args.out, index=False)

    num_cells = len(cells)
    num_edges = len(edges)
    mean_degree = (2 * num_edges / num_cells) if num_cells > 0 else 0.0

    print(f"Number of cells: {num_cells}")
    print(f"Radius (um): {args.radius_um}")
    print(f"Number of edges: {num_edges}")
    print(f"Mean degree: {mean_degree:.4f}")
    print(f"Output path: {args.out}")


if __name__ == "__main__":
    main()
