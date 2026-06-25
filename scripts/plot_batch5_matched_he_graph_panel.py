"""Plot JAGO cell graphs that share the exact field of view as their H&E crops.

The previous representative-patch panel plotted each graph using the median
cell coordinates and a default window size, so the rendered field of view
did not line up with the corresponding H&E crop. This script instead reads
each patch's exact x/y bounds from patch_manifest.csv and uses those same
bounds (and equal aspect ratio, inverted y-axis, no autoscaling) for both the
standalone graph PNG and the combined H&E/graph panel, so the two images
correspond 1:1.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

DEFAULT_PATCHES_ROOT = Path(
    "/scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/patches"
)
DEFAULT_HE_DIR = Path(
    "/scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/figures/he_patches_named"
)
DEFAULT_GRAPH_OUT_DIR = Path(
    "/scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/figures/ranked_patches_matched"
)
DEFAULT_OUT_PNG = Path(
    "/scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/figures/he_graph_pair_panel_matched.png"
)
DEFAULT_OUT_PDF = Path(
    "/scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/figures/he_graph_pair_panel_matched.pdf"
)

TYPE_MAP = {
    "type_0": "nolabe",
    "type_1": "neoplastic",
    "type_2": "inflammatory",
    "type_3": "connective",
    "type_4": "necrotic",
    "type_5": "non_neoplastic_epithelial",
}

TYPE_COLORS = {
    "neoplastic": "red",
    "inflammatory": "blue",
    "connective": "green",
    "necrotic": "purple",
    "non_neoplastic_epithelial": "orange",
    "nolabe": "gray",
}

TYPE_LEGEND_LABELS = {
    "neoplastic": "Neoplastic",
    "inflammatory": "Inflammatory",
    "connective": "Connective",
    "necrotic": "Necrotic",
    "non_neoplastic_epithelial": "Non-neoplastic epithelial",
    "nolabe": "No label",
}

ITEMS = [
    {
        "label": "A. Tumor-rich",
        "key": "tumor_rich",
        "slide_id": "TCGA-3C-AALK-01Z-00-DX1.4E6EB156-BB19-410F-878F-FC0EA7BD0B53",
        "patch_id": "patch_000032",
        "he": "tumor_rich_he.png",
    },
    {
        "label": "B. Immune-rich",
        "key": "immune_rich",
        "slide_id": "TCGA-3C-AALI-01Z-00-DX2.CF4496E0-AB52-4F3E-BDF5-C34833B91B7C",
        "patch_id": "patch_000029",
        "he": "immune_rich_he.png",
    },
    {
        "label": "C. Stromal-rich",
        "key": "stromal_rich",
        "slide_id": "TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291",
        "patch_id": "patch_000028",
        "he": "stromal_rich_he.png",
    },
    {
        "label": "D. Necrotic-rich",
        "key": "necrotic_rich",
        "slide_id": "TCGA-3C-AALK-01Z-00-DX1.4E6EB156-BB19-410F-878F-FC0EA7BD0B53",
        "patch_id": "patch_000041",
        "he": "necrotic_rich_he.png",
    },
    {
        "label": "E. Mixed architecture",
        "key": "mixed_architecture",
        "slide_id": "TCGA-3C-AALI-01Z-00-DX2.CF4496E0-AB52-4F3E-BDF5-C34833B91B7C",
        "patch_id": "patch_000016",
        "he": "mixed_architecture_he.png",
    },
    {
        "label": "F. Tumor-immune contact",
        "key": "tumor_immune_contact",
        "slide_id": "TCGA-3C-AALI-01Z-00-DX2.CF4496E0-AB52-4F3E-BDF5-C34833B91B7C",
        "patch_id": "patch_000045",
        "he": "tumor_immune_contact_he.png",
    },
]

TITLE_TEXT = "JAGO Batch 5: Matched H&E Crops and Cell Graphs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot JAGO cell graphs matched to the field of view of their H&E crops."
    )
    parser.add_argument("--patches-root", required=False, type=Path, default=DEFAULT_PATCHES_ROOT)
    parser.add_argument("--he-dir", required=False, type=Path, default=DEFAULT_HE_DIR)
    parser.add_argument("--graph-out-dir", required=False, type=Path, default=DEFAULT_GRAPH_OUT_DIR)
    parser.add_argument("--out-png", required=False, type=Path, default=DEFAULT_OUT_PNG)
    parser.add_argument("--out-pdf", required=False, type=Path, default=DEFAULT_OUT_PDF)
    return parser.parse_args()


def load_manifest_row(patches_root: Path, slide_id: str, patch_id: str) -> pd.Series:
    manifest_path = patches_root / slide_id / "patch_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    matches = manifest.loc[manifest["patch_id"] == patch_id]
    if len(matches) == 0:
        raise ValueError(f"patch_id '{patch_id}' not found in {manifest_path}")

    return matches.iloc[0]


def load_patch_graph(patches_root: Path, slide_id: str, patch_id: str):
    patch_dir = patches_root / slide_id

    cells_path = patch_dir / f"{patch_id}_cells.csv"
    edges_path = patch_dir / f"{patch_id}_edges.csv"

    if not cells_path.exists():
        raise FileNotFoundError(f"Cells file not found: {cells_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_path}")

    cells = pd.read_csv(cells_path, dtype={"cell_id": str})
    edges = pd.read_csv(edges_path, dtype={"source_cell_id": str, "target_cell_id": str})

    return cells, edges


def render_matched_graph(ax, cells: pd.DataFrame, edges: pd.DataFrame, bounds: tuple) -> None:
    x_min_um, x_max_um, y_min_um, y_max_um = bounds

    coords_by_id = cells.set_index("cell_id")[["x_um", "y_um"]]

    if len(edges) > 0:
        source_coords = coords_by_id.loc[edges["source_cell_id"]].to_numpy()
        target_coords = coords_by_id.loc[edges["target_cell_id"]].to_numpy()

        for (sx, sy), (tx, ty) in zip(source_coords, target_coords):
            ax.plot([sx, tx], [sy, ty], color="gray", alpha=0.2, linewidth=0.5, zorder=1)

    readable_cell_type = cells["cell_type"].map(lambda t: TYPE_MAP.get(t, t))
    for cell_type, group in cells.groupby(readable_cell_type):
        color = TYPE_COLORS.get(cell_type, "black")
        ax.scatter(group["x_um"], group["y_um"], s=8, color=color, zorder=2)

    # Set limits explicitly and disable autoscale so the field of view
    # exactly matches the H&E crop's manifest bounds, not the data extent.
    ax.set_xlim(x_min_um, x_max_um)
    ax.set_ylim(y_max_um, y_min_um)
    ax.set_autoscale_on(False)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def save_individual_graph(cells, edges, bounds, out_path: Path) -> None:
    x_min_um, x_max_um, y_min_um, y_max_um = bounds
    width_um = x_max_um - x_min_um
    height_um = y_max_um - y_min_um
    aspect = height_um / width_um if width_um > 0 else 1.0

    fig_w = 6.0
    fig_h = max(fig_w * aspect, 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    render_matched_graph(ax, cells, edges, bounds)
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_legend_handles():
    return [
        mlines.Line2D(
            [], [], marker="o", linestyle="None", markersize=7, color=color,
            label=TYPE_LEGEND_LABELS[cell_type],
        )
        for cell_type, color in TYPE_COLORS.items()
    ]


def main() -> None:
    args = parse_args()

    resolved_items = []
    for item in ITEMS:
        row = load_manifest_row(args.patches_root, item["slide_id"], item["patch_id"])
        bounds = (
            float(row["x_min_um"]),
            float(row["x_max_um"]),
            float(row["y_min_um"]),
            float(row["y_max_um"]),
        )

        cells, edges = load_patch_graph(args.patches_root, item["slide_id"], item["patch_id"])

        he_path = args.he_dir / item["he"]
        if not he_path.exists():
            raise FileNotFoundError(f"Missing H&E image: {he_path}")

        graph_out_path = args.graph_out_dir / f"{item['key']}_{item['patch_id']}.png"
        save_individual_graph(cells, edges, bounds, graph_out_path)
        print(f"Saved matched graph: {graph_out_path}")

        resolved_items.append(
            {
                "label": item["label"],
                "he_path": he_path,
                "cells": cells,
                "edges": edges,
                "bounds": bounds,
            }
        )

    n_rows = len(resolved_items)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 5 * n_rows), facecolor="white")

    fig.suptitle(TITLE_TEXT, fontsize=18, y=0.995)

    col_header_y = 1.0 - 0.012
    fig.text(0.28, col_header_y, "H&E crop", ha="center", fontsize=14, fontweight="bold")
    fig.text(0.78, col_header_y, "JAGO cell graph", ha="center", fontsize=14, fontweight="bold")

    for row_idx, entry in enumerate(resolved_items):
        he_ax = axes[row_idx, 0]
        graph_ax = axes[row_idx, 1]

        he_image = Image.open(entry["he_path"]).convert("RGB")
        he_ax.imshow(he_image)
        he_ax.set_xticks([])
        he_ax.set_yticks([])
        he_ax.set_ylabel(entry["label"], fontsize=12, rotation=0, ha="right", va="center", labelpad=40)

        render_matched_graph(graph_ax, entry["cells"], entry["edges"], entry["bounds"])

    legend_handles = build_legend_handles()
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0.035, 1, 0.97])

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=300, bbox_inches="tight", facecolor="white")
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved panel: {args.out_png}")
    print(f"Saved panel: {args.out_pdf}")


if __name__ == "__main__":
    main()
