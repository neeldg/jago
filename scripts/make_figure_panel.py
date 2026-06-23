"""Assemble a labeled 2x3 figure panel from JAGO patch figure PNGs.

Loads six named patch figures (one per architecture score) from a directory
and arranges them into a single labeled panel suitable for a paper figure.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

PANEL_FILENAMES = [
    "tumor_rich_patch_000042.png",
    "stromal_rich_patch_000049.png",
    "immune_rich_patch_000002.png",
    "tumor_immune_contact_patch_000015.png",
    "tumor_stroma_contact_patch_000043.png",
    "mixed_architecture_patch_000034.png",
]
PANEL_LABELS = ["A", "B", "C", "D", "E", "F"]
N_ROWS = 2
N_COLS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble a labeled 2x3 figure panel from JAGO patch figure PNGs."
    )
    parser.add_argument(
        "--image-dir", required=True, type=Path, help="Directory containing the patch figure PNGs."
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Path to write the output panel PNG."
    )
    parser.add_argument(
        "--title", required=False, type=str, default=None, help="Optional overall figure title."
    )
    return parser.parse_args()


def load_panel_images(image_dir: Path) -> list:
    images = []
    for filename in PANEL_FILENAMES:
        image_path = image_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Expected panel image not found: {image_path}")
        images.append(Image.open(image_path))
    return images


def main() -> None:
    args = parse_args()

    images = load_panel_images(args.image_dir)

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(15, 10))

    for ax, image, label in zip(axes.flat, images, PANEL_LABELS):
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            color="black",
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
        )

    if args.title:
        fig.suptitle(args.title, fontsize=18)

    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Output path: {args.out}")


if __name__ == "__main__":
    main()
