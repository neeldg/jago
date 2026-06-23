"""Tile a JAGO spatial cell graph into fixed-size sliding-window patches.

Slides a square window across the slide's coordinate space, saving one
cells/edges CSV pair per window that contains enough cells. Useful for
turning a whole-slide graph into a set of fixed-size patches for downstream
model training or visualization.
"""

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tile a JAGO spatial cell graph into sliding-window patches."
    )
    parser.add_argument("--cells", required=True, type=Path, help="Path to input cell table CSV.")
    parser.add_argument(
        "--edges", required=True, type=Path, help="Path to input radius-graph edge list CSV."
    )
    parser.add_argument(
        "--out-dir", required=True, type=Path, help="Directory to write patch CSVs and manifest into."
    )
    parser.add_argument(
        "--window-um", required=False, type=float, default=500.0, help="Patch width in microns."
    )
    parser.add_argument(
        "--stride-um",
        required=False,
        type=float,
        default=500.0,
        help="Sliding-window stride in microns.",
    )
    parser.add_argument(
        "--min-cells",
        required=False,
        type=int,
        default=100,
        help="Minimum number of cells required to save a patch.",
    )
    parser.add_argument(
        "--max-patches",
        required=False,
        type=int,
        default=None,
        help="Optional maximum number of patches to save.",
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


def tile_starts(min_val: float, max_val: float, stride_um: float) -> np.ndarray:
    if max_val <= min_val:
        return np.array([min_val])

    starts = np.arange(min_val, max_val, stride_um)
    if len(starts) == 0:
        starts = np.array([min_val])
    return starts


def main() -> None:
    args = parse_args()

    cells = load_table(args.cells, REQUIRED_CELL_COLUMNS)
    edges = load_table(args.edges, REQUIRED_EDGE_COLUMNS)

    cells["x_um"] = cells["x_um"].astype(float)
    cells["y_um"] = cells["y_um"].astype(float)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    x_starts = tile_starts(cells["x_um"].min(), cells["x_um"].max(), args.stride_um)
    y_starts = tile_starts(cells["y_um"].min(), cells["y_um"].max(), args.stride_um)

    manifest_rows = []
    n_saved = 0

    for y_min_um in y_starts:
        for x_min_um in x_starts:
            if args.max_patches is not None and n_saved >= args.max_patches:
                break

            x_max_um = x_min_um + args.window_um
            y_max_um = y_min_um + args.window_um

            in_window = (
                (cells["x_um"] >= x_min_um)
                & (cells["x_um"] < x_max_um)
                & (cells["y_um"] >= y_min_um)
                & (cells["y_um"] < y_max_um)
            )
            patch_cells = cells.loc[in_window]

            if len(patch_cells) < args.min_cells:
                continue

            patch_cell_ids = set(patch_cells["cell_id"])
            patch_edges = edges.loc[
                edges["source_cell_id"].isin(patch_cell_ids)
                & edges["target_cell_id"].isin(patch_cell_ids)
            ]

            n_saved += 1
            patch_id = f"patch_{n_saved:06d}"

            patch_cells.to_csv(args.out_dir / f"{patch_id}_cells.csv", index=False)
            patch_edges.to_csv(args.out_dir / f"{patch_id}_edges.csv", index=False)

            manifest_rows.append(
                {
                    "patch_id": patch_id,
                    "center_x_um": x_min_um + args.window_um / 2,
                    "center_y_um": y_min_um + args.window_um / 2,
                    "x_min_um": x_min_um,
                    "x_max_um": x_max_um,
                    "y_min_um": y_min_um,
                    "y_max_um": y_max_um,
                    "n_cells": len(patch_cells),
                    "n_edges": len(patch_edges),
                }
            )
        else:
            continue
        break

    manifest = pd.DataFrame(
        manifest_rows,
        columns=[
            "patch_id",
            "center_x_um",
            "center_y_um",
            "x_min_um",
            "x_max_um",
            "y_min_um",
            "y_max_um",
            "n_cells",
            "n_edges",
        ],
    )
    manifest.to_csv(args.out_dir / "patch_manifest.csv", index=False)

    print(f"Number of patches saved: {n_saved}")


if __name__ == "__main__":
    main()
