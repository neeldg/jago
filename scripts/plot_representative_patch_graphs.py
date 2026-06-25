"""Plot a 2x3 panel of representative JAGO graph patches.

For each architecture category (tumor-rich, immune-rich, stromal-rich,
necrotic-rich, mixed architecture, tumor-immune contact), takes the
top-ranked patch from its ranking CSV and plots its cell graph, then
arranges all six into one labeled panel figure.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

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

# (panel title, ranking csv filename)
CATEGORIES = [
    ("Tumor-rich", "top_tumor_rich_score.csv"),
    ("Immune-rich", "top_immune_rich_score.csv"),
    ("Stromal-rich", "top_stromal_rich_score.csv"),
    ("Necrotic-rich", "top_necrotic_rich_score.csv"),
    ("Mixed architecture", "top_mixed_architecture_score.csv"),
    ("Tumor-immune contact", "top_tumor_immune_contact_score.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a 2x3 panel of representative JAGO graph patches."
    )
    parser.add_argument(
        "--patches-root", required=True, type=Path,
        help="Directory containing <slide_id>/patch_<n>_cells.csv and _edges.csv files.",
    )
    parser.add_argument(
        "--rankings-root", required=True, type=Path,
        help="Directory containing the top_<score>.csv ranking files.",
    )
    parser.add_argument(
        "--outdir", required=True, type=Path,
        help="Directory to write the output PNG/PDF panel into.",
    )
    return parser.parse_args()


def parse_patch_id(patch_id: str):
    if "_patch_" not in patch_id:
        raise ValueError(f"patch_id '{patch_id}' does not contain '_patch_'.")

    slide_id, _, suffix = patch_id.partition("_patch_")
    patch_name = "patch_" + suffix
    return slide_id, patch_name


def short_slide_id(slide_id: str, max_len: int = 16) -> str:
    if len(slide_id) <= max_len:
        return slide_id
    return slide_id[:max_len] + "..."


def load_top_patch(rankings_root: Path, ranking_filename: str):
    ranking_path = rankings_root / ranking_filename
    if not ranking_path.exists():
        raise FileNotFoundError(f"Ranking file not found: {ranking_path}")

    ranking = pd.read_csv(ranking_path)
    if len(ranking) == 0:
        raise ValueError(f"Ranking file {ranking_path} has no rows.")

    return ranking.iloc[0]["patch_id"]


def load_patch_graph(patches_root: Path, patch_id: str):
    slide_id, patch_name = parse_patch_id(patch_id)
    patch_dir = patches_root / slide_id

    cells_path = patch_dir / f"{patch_name}_cells.csv"
    edges_path = patch_dir / f"{patch_name}_edges.csv"

    if not cells_path.exists():
        raise FileNotFoundError(f"Cells file not found: {cells_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_path}")

    cells = pd.read_csv(cells_path, dtype={"cell_id": str})
    edges = pd.read_csv(edges_path, dtype={"source_cell_id": str, "target_cell_id": str})

    return slide_id, patch_name, cells, edges


def plot_patch_panel(ax, slide_id, patch_name, cells, edges, category_title):
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

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    patch_number = patch_name.replace("patch_", "")
    title = (
        f"{category_title}\n{short_slide_id(slide_id)} | patch {patch_number}\n"
        f"n_cells={len(cells)}, n_edges={len(edges)}"
    )
    ax.set_title(title, fontsize=9)


def main() -> None:
    args = parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(15, 11))

    for ax, (category_title, ranking_filename) in zip(axes.flat, CATEGORIES):
        patch_id = load_top_patch(args.rankings_root, ranking_filename)
        slide_id, patch_name, cells, edges = load_patch_graph(args.patches_root, patch_id)
        plot_patch_panel(ax, slide_id, patch_name, cells, edges, category_title)
        print(f"{category_title}: {patch_id} (n_cells={len(cells)}, n_edges={len(edges)})")

    legend_handles = [
        mlines.Line2D(
            [], [], marker="o", linestyle="None", markersize=6, color=color, label=cell_type
        )
        for cell_type, color in TYPE_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    args.outdir.mkdir(parents=True, exist_ok=True)
    png_path = args.outdir / "representative_patch_graphs.png"
    pdf_path = args.outdir / "representative_patch_graphs.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Output path: {png_path}")
    print(f"Output path: {pdf_path}")


if __name__ == "__main__":
    main()
