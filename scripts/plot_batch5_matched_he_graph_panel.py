"""This is the canonical JAGO H&E ↔ graph figure script. It guarantees
1:1 physical-coordinate comparison between H&E crops and cell graphs.

Reads each patch's exact x/y bounds from patch_manifest.csv and uses those
same bounds (equal aspect ratio, inverted y-axis, no autoscaling) for both
the H&E crop and the cell graph, with both displayed on micron axes so the
two panels are a true 1:1 physical-coordinate comparison rather than one
being a plain pixel image and the other a coordinate plot. Every patch is
validated against the JAGO figure standards (matching bounds, equal aspect,
micron axes, a 100 um scale bar, and human-readable cell type labels) before
the figure is saved, raising a clear error if any standard is not met.
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
DEFAULT_METADATA_PATH = Path(
    "/scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/figures/"
    "he_graph_pair_panel_matched_metadata.csv"
)

# Tolerance (in microns) used when validating that rendered axis limits and
# imshow extents exactly match the manifest-derived patch bounds.
BOUNDS_ATOL_UM = 1e-6

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

TITLE_TEXT = "JAGO batch5: matched H&E crops and cell graphs"
SCALE_BAR_UM = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot JAGO cell graphs matched to the field of view of their H&E crops."
    )
    parser.add_argument("--patches-root", required=False, type=Path, default=DEFAULT_PATCHES_ROOT)
    parser.add_argument("--he-dir", required=False, type=Path, default=DEFAULT_HE_DIR)
    parser.add_argument("--graph-out-dir", required=False, type=Path, default=DEFAULT_GRAPH_OUT_DIR)
    parser.add_argument("--out-png", required=False, type=Path, default=DEFAULT_OUT_PNG)
    parser.add_argument("--out-pdf", required=False, type=Path, default=DEFAULT_OUT_PDF)
    parser.add_argument("--metadata-out", required=False, type=Path, default=DEFAULT_METADATA_PATH)
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


def get_patch_bounds(row: pd.Series) -> tuple:
    direct_cols = ["x_min_um", "x_max_um", "y_min_um", "y_max_um"]
    if all(col in row.index and pd.notna(row[col]) for col in direct_cols):
        return (
            float(row["x_min_um"]),
            float(row["x_max_um"]),
            float(row["y_min_um"]),
            float(row["y_max_um"]),
        )

    center_cols = ["center_x_um", "center_y_um", "window_um"]
    if all(col in row.index and pd.notna(row[col]) for col in center_cols):
        center_x_um = float(row["center_x_um"])
        center_y_um = float(row["center_y_um"])
        half_window_um = float(row["window_um"]) / 2
        return (
            center_x_um - half_window_um,
            center_x_um + half_window_um,
            center_y_um - half_window_um,
            center_y_um + half_window_um,
        )

    raise ValueError(
        "Manifest row has neither x_min_um/x_max_um/y_min_um/y_max_um nor "
        "center_x_um/center_y_um/window_um columns needed to compute patch bounds."
    )


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


def add_scale_bar(ax, bounds: tuple, scale_um: float = SCALE_BAR_UM) -> None:
    x_min_um, x_max_um, y_min_um, y_max_um = bounds
    width_um = x_max_um - x_min_um
    height_um = y_max_um - y_min_um

    margin_x_um = width_um * 0.06
    margin_y_um = height_um * 0.08

    x_end = x_max_um - margin_x_um
    x_start = x_end - scale_um
    y_bar = y_max_um - margin_y_um

    # Data coordinates are microns, so a 100 um bar is just a 100-unit line.
    ax.plot(
        [x_start, x_end], [y_bar, y_bar],
        color="black", linewidth=3, solid_capstyle="butt", zorder=4,
    )
    ax.text(
        (x_start + x_end) / 2, y_bar - height_um * 0.02, f"{scale_um:g} µm",
        color="black", fontsize=9, ha="center", va="bottom", zorder=4,
    )


def style_micron_axes(ax, bounds: tuple) -> None:
    x_min_um, x_max_um, y_min_um, y_max_um = bounds

    # Set limits explicitly and disable autoscale so the field of view
    # exactly matches the manifest bounds, not the data/image extent.
    ax.set_xlim(x_min_um, x_max_um)
    ax.set_ylim(y_max_um, y_min_um)
    ax.set_autoscale_on(False)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)", fontsize=9)
    ax.set_ylabel("y (µm)", fontsize=9)
    ax.tick_params(labelsize=7)

    add_scale_bar(ax, bounds)


def render_matched_graph(ax, cells: pd.DataFrame, edges: pd.DataFrame, bounds: tuple) -> None:
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

    style_micron_axes(ax, bounds)


def render_he_crop(ax, he_image: Image.Image, bounds: tuple) -> None:
    x_min_um, x_max_um, y_min_um, y_max_um = bounds
    ax.imshow(he_image, extent=[x_min_um, x_max_um, y_max_um, y_min_um])
    style_micron_axes(ax, bounds)


def validate_cell_types_readable(cells: pd.DataFrame, edges: pd.DataFrame, context: str) -> None:
    """Enforce that every cell/edge type label is human-readable, not raw type_N."""
    mapped_cell_types = set(cells["cell_type"].map(lambda t: TYPE_MAP.get(t, t)).unique())
    mapped_edge_types = set(edges["source_cell_type"].map(lambda t: TYPE_MAP.get(t, t)).unique())
    mapped_edge_types |= set(edges["target_cell_type"].map(lambda t: TYPE_MAP.get(t, t)).unique())

    raw_leftover = sorted(
        t for t in (mapped_cell_types | mapped_edge_types) if str(t).startswith("type_")
    )
    if raw_leftover:
        raise ValueError(
            f"{context}: found non-human-readable cell type label(s) {raw_leftover}. "
            f"TYPE_MAP must cover every type id present in the data."
        )


def _has_equal_aspect(ax) -> bool:
    aspect = ax.get_aspect()
    if isinstance(aspect, str):
        return aspect == "equal"
    return float(aspect) == 1.0


def _has_scale_bar(ax, scale_um: float = SCALE_BAR_UM, atol: float = 1e-6) -> bool:
    has_line = any(
        len(set(line.get_ydata())) == 1
        and abs(abs(line.get_xdata()[-1] - line.get_xdata()[0]) - scale_um) < max(atol, scale_um * 1e-6)
        for line in ax.lines
    )
    has_label = any(f"{scale_um:g} µm" in text.get_text() for text in ax.texts)
    return has_line and has_label


def validate_graph_axes(ax, bounds: tuple, context: str) -> None:
    x_min_um, x_max_um, y_min_um, y_max_um = bounds

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    if not (abs(x_lim[0] - x_min_um) < BOUNDS_ATOL_UM and abs(x_lim[1] - x_max_um) < BOUNDS_ATOL_UM):
        raise ValueError(f"{context}: graph x-limits {x_lim} do not match patch bounds ({x_min_um}, {x_max_um}).")
    if not (abs(y_lim[0] - y_max_um) < BOUNDS_ATOL_UM and abs(y_lim[1] - y_min_um) < BOUNDS_ATOL_UM):
        raise ValueError(
            f"{context}: graph y-limits {y_lim} do not match inverted patch bounds ({y_max_um}, {y_min_um})."
        )
    if not _has_equal_aspect(ax):
        raise ValueError(f"{context}: graph axes must use an equal aspect ratio.")
    if "µm" not in ax.get_xlabel() or "µm" not in ax.get_ylabel():
        raise ValueError(f"{context}: graph axes must display x/y in microns (µm).")
    if not _has_scale_bar(ax):
        raise ValueError(f"{context}: graph panel is missing a {SCALE_BAR_UM:g} µm scale bar.")


def validate_he_axes(ax, bounds: tuple, context: str) -> None:
    x_min_um, x_max_um, y_min_um, y_max_um = bounds

    if not ax.images:
        raise ValueError(f"{context}: H&E axes have no imshow image to validate extent against.")

    extent = ax.images[-1].get_extent()
    expected_extent = (x_min_um, x_max_um, y_max_um, y_min_um)
    if any(abs(a - b) > BOUNDS_ATOL_UM for a, b in zip(extent, expected_extent)):
        raise ValueError(f"{context}: H&E imshow extent {extent} does not match expected {expected_extent}.")

    if not _has_equal_aspect(ax):
        raise ValueError(f"{context}: H&E axes must use an equal aspect ratio.")
    if "µm" not in ax.get_xlabel() or "µm" not in ax.get_ylabel():
        raise ValueError(f"{context}: H&E axes must display x/y in microns (µm).")
    if not _has_scale_bar(ax):
        raise ValueError(f"{context}: H&E panel is missing a {SCALE_BAR_UM:g} µm scale bar.")


def validate_patch_panel(he_ax, graph_ax, bounds: tuple, context: str) -> None:
    """Enforce the JAGO figure standards: matching bounds, equal aspect,
    micron axes, and a scale bar on both the H&E and graph panels."""
    validate_he_axes(he_ax, bounds, context)
    validate_graph_axes(graph_ax, bounds, context)


def save_individual_graph(cells, edges, bounds, out_path: Path) -> None:
    x_min_um, x_max_um, y_min_um, y_max_um = bounds
    width_um = x_max_um - x_min_um
    height_um = y_max_um - y_min_um
    aspect = height_um / width_um if width_um > 0 else 1.0

    fig_w = 6.0
    fig_h = max(fig_w * aspect, 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    render_matched_graph(ax, cells, edges, bounds)
    validate_graph_axes(ax, bounds, context=f"standalone graph {out_path.name}")
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
    metadata_rows = []
    for item in ITEMS:
        context = f"{item['label']} ({item['slide_id']}/{item['patch_id']})"

        row = load_manifest_row(args.patches_root, item["slide_id"], item["patch_id"])
        bounds = get_patch_bounds(row)
        x_min_um, x_max_um, y_min_um, y_max_um = bounds

        cells_path = args.patches_root / item["slide_id"] / f"{item['patch_id']}_cells.csv"
        edges_path = args.patches_root / item["slide_id"] / f"{item['patch_id']}_edges.csv"
        cells, edges = load_patch_graph(args.patches_root, item["slide_id"], item["patch_id"])
        validate_cell_types_readable(cells, edges, context=context)

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
                "context": context,
            }
        )

        metadata_rows.append(
            {
                "label": item["label"],
                "slide_id": item["slide_id"],
                "patch_id": item["patch_id"],
                "x_min_um": x_min_um,
                "x_max_um": x_max_um,
                "y_min_um": y_min_um,
                "y_max_um": y_max_um,
                "window_um": x_max_um - x_min_um,
                "he_file": str(he_path),
                "cells_file": str(cells_path),
                "edges_file": str(edges_path),
            }
        )

    n_rows = len(resolved_items)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5.5 * n_rows), facecolor="white")

    # Reserve a generous top margin so the suptitle and column headers each
    # get their own clear band of space, and generous hspace so each row's
    # micron tick labels don't collide with the row above/below it.
    fig.subplots_adjust(top=0.93, bottom=0.06, hspace=0.55, wspace=0.3)

    fig.suptitle(TITLE_TEXT, fontsize=20, y=0.985)

    col_header_y = 0.945
    fig.text(0.28, col_header_y, "H&E crop", ha="center", fontsize=15, fontweight="bold")
    fig.text(0.74, col_header_y, "JAGO cell graph", ha="center", fontsize=15, fontweight="bold")

    for row_idx, entry in enumerate(resolved_items):
        he_ax = axes[row_idx, 0]
        graph_ax = axes[row_idx, 1]

        he_image = Image.open(entry["he_path"]).convert("RGB")
        render_he_crop(he_ax, he_image, entry["bounds"])
        he_ax.text(
            -0.32, 0.5, entry["label"], transform=he_ax.transAxes,
            fontsize=12, fontweight="bold", ha="right", va="center",
        )

        render_matched_graph(graph_ax, entry["cells"], entry["edges"], entry["bounds"])

        validate_patch_panel(he_ax, graph_ax, entry["bounds"], context=entry["context"])

    legend_handles = build_legend_handles()
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=11,
    )

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=300, bbox_inches="tight", facecolor="white")
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved panel: {args.out_png}")
    print(f"Saved panel: {args.out_pdf}")

    metadata = pd.DataFrame(
        metadata_rows,
        columns=[
            "label", "slide_id", "patch_id",
            "x_min_um", "x_max_um", "y_min_um", "y_max_um", "window_um",
            "he_file", "cells_file", "edges_file",
        ],
    )
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.metadata_out, index=False)
    print(f"Saved metadata: {args.metadata_out}")


if __name__ == "__main__":
    main()
