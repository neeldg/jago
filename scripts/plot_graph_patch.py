"""Plot a local patch of a JAGO spatial cell graph.

Crops a square window out of a cell table and its radius-graph edge list,
then renders the cells (colored by type) and, optionally, the edges that
connect them — useful for sanity-checking a graph without rendering an
entire whole-slide graph at once.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
        description="Plot a local patch of a JAGO spatial cell graph."
    )
    parser.add_argument("--cells", required=True, type=Path, help="Path to input cell table CSV.")
    parser.add_argument(
        "--edges", required=True, type=Path, help="Path to input radius-graph edge list CSV."
    )
    parser.add_argument(
        "--type-map",
        required=False,
        type=Path,
        default=None,
        help="Optional JSON file mapping type ids (e.g. 'type_1') to readable names.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Path to write the output PNG.")
    parser.add_argument(
        "--center-x-um",
        required=False,
        type=float,
        default=None,
        help="x_um coordinate of the patch center. Defaults to the median x_um of all cells.",
    )
    parser.add_argument(
        "--center-y-um",
        required=False,
        type=float,
        default=None,
        help="y_um coordinate of the patch center. Defaults to the median y_um of all cells.",
    )
    parser.add_argument(
        "--window-um",
        required=False,
        type=float,
        default=500.0,
        help="Width (and height) in microns of the square patch window. Defaults to 500.",
    )
    parser.add_argument(
        "--max-edges",
        required=False,
        type=int,
        default=20000,
        help="Maximum number of edges to plot. Excess edges are randomly subsampled.",
    )
    parser.add_argument(
        "--draw-edges",
        action="store_true",
        help="If set, draw edges faintly behind the cell scatter points.",
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


def load_type_map(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Type map not found: {path}")

    with path.open() as f:
        type_map = json.load(f)

    if not isinstance(type_map, dict):
        raise ValueError(f"Type map {path} must be a JSON object of type id -> readable name.")

    return type_map


def select_patch(cells: pd.DataFrame, center_x_um: float, center_y_um: float, window_um: float) -> pd.DataFrame:
    half_window = window_um / 2
    x_um = cells["x_um"].astype(float)
    y_um = cells["y_um"].astype(float)

    in_window = (
        (x_um >= center_x_um - half_window)
        & (x_um <= center_x_um + half_window)
        & (y_um >= center_y_um - half_window)
        & (y_um <= center_y_um + half_window)
    )
    return cells.loc[in_window].copy()


def main() -> None:
    args = parse_args()

    cells = load_table(args.cells, REQUIRED_CELL_COLUMNS)
    edges = load_table(args.edges, REQUIRED_EDGE_COLUMNS)
    type_map = load_type_map(args.type_map) if args.type_map else None

    cells["x_um"] = cells["x_um"].astype(float)
    cells["y_um"] = cells["y_um"].astype(float)

    center_x_um = args.center_x_um if args.center_x_um is not None else cells["x_um"].median()
    center_y_um = args.center_y_um if args.center_y_um is not None else cells["y_um"].median()

    patch_cells = select_patch(cells, center_x_um, center_y_um, args.window_um)
    patch_cell_ids = set(patch_cells["cell_id"])

    patch_edges = edges.loc[
        edges["source_cell_id"].isin(patch_cell_ids) & edges["target_cell_id"].isin(patch_cell_ids)
    ].copy()

    if len(patch_edges) > args.max_edges:
        patch_edges = patch_edges.sample(n=args.max_edges, random_state=None)

    if type_map:
        plot_cell_type = patch_cells["cell_type"].map(lambda t: type_map.get(t, t))
    else:
        plot_cell_type = patch_cells["cell_type"]

    fig, ax = plt.subplots(figsize=(8, 8))

    if args.draw_edges and len(patch_edges) > 0:
        coords_by_id = patch_cells.set_index("cell_id")[["x_um", "y_um"]]
        source_coords = coords_by_id.loc[patch_edges["source_cell_id"]].to_numpy()
        target_coords = coords_by_id.loc[patch_edges["target_cell_id"]].to_numpy()

        segments_x = np.column_stack([source_coords[:, 0], target_coords[:, 0], np.full(len(patch_edges), np.nan)]).ravel()
        segments_y = np.column_stack([source_coords[:, 1], target_coords[:, 1], np.full(len(patch_edges), np.nan)]).ravel()

        ax.plot(segments_x, segments_y, color="gray", alpha=0.2, linewidth=0.5, zorder=1)

    for cell_type, group in patch_cells.groupby(plot_cell_type):
        ax.scatter(group["x_um"], group["y_um"], label=cell_type, s=10, zorder=2)

    ax.invert_yaxis()
    ax.set_xlabel("x_um")
    ax.set_ylabel("y_um")
    ax.set_aspect("equal")
    ax.set_title(f"JAGO graph patch (center=({center_x_um:.1f}, {center_y_um:.1f}), window={args.window_um}um)")
    ax.legend(markerscale=2, fontsize="small", loc="best")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Number of cells plotted: {len(patch_cells)}")
    print(f"Number of edges plotted: {len(patch_edges)}")
    print(f"Output path: {args.out}")


if __name__ == "__main__":
    main()
