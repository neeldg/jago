"""Stitch paired H&E crops and JAGO cell graph images into one panel.

For each representative architecture category, lays out the matching H&E
crop next to its JAGO cell graph rendering in a two-column, publication-style
figure, and saves both a 300 dpi PNG and a PDF version.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_HE_DIR = Path(
    "/scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/figures/he_patches_named"
)
DEFAULT_GRAPH_DIR = Path(
    "/scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/figures/ranked_patches_clean"
)
DEFAULT_OUT_PATH = Path(
    "/scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/figures/he_graph_pair_panel.png"
)

ITEMS = [
    {
        "label": "A. Tumor-rich",
        "he": "tumor_rich_he.png",
        "graph": "tumor_rich_patch_000032.png",
    },
    {
        "label": "B. Immune-rich",
        "he": "immune_rich_he.png",
        "graph": "immune_rich_patch_000029.png",
    },
    {
        "label": "C. Stromal-rich",
        "he": "stromal_rich_he.png",
        "graph": "stromal_rich_patch_000028.png",
    },
    {
        "label": "D. Necrotic-rich",
        "he": "necrotic_rich_he.png",
        "graph": "necrotic_rich_patch_000041.png",
    },
    {
        "label": "E. Mixed architecture",
        "he": "mixed_architecture_he.png",
        "graph": "mixed_architecture_patch_000016.png",
    },
    {
        "label": "F. Tumor-immune contact",
        "he": "tumor_immune_contact_he.png",
        "graph": "tumor_immune_contact_patch_000045.png",
    },
]

TITLE_TEXT = "JAGO Batch 5: H&E Crops vs. Cell Graph Architecture Patches"
HE_HEADER_TEXT = "H&E crop"
GRAPH_HEADER_TEXT = "JAGO cell graph"

IMAGE_SIZE = 360
TITLE_HEIGHT = 56
HEADER_HEIGHT = 42
LABEL_HEIGHT = 42
PAD = 18
PAIR_GAP = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch paired H&E crops and JAGO cell graph images into one panel."
    )
    parser.add_argument("--he-dir", required=False, type=Path, default=DEFAULT_HE_DIR)
    parser.add_argument("--graph-dir", required=False, type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--out", required=False, type=Path, default=DEFAULT_OUT_PATH)
    return parser.parse_args()


def load_fonts():
    try:
        title_font = ImageFont.truetype("Arial.ttf", 28)
        header_font = ImageFont.truetype("Arial.ttf", 24)
        label_font = ImageFont.truetype("Arial.ttf", 22)
    except Exception:
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    return title_font, header_font, label_font


def load_square(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)
    return img


def build_panel(he_dir: Path, graph_dir: Path, out_path: Path) -> None:
    resolved_items = []
    for item in ITEMS:
        he_path = he_dir / item["he"]
        graph_path = graph_dir / item["graph"]

        if not he_path.exists():
            raise FileNotFoundError(f"Missing H&E image: {he_path}")
        if not graph_path.exists():
            raise FileNotFoundError(f"Missing graph image: {graph_path}")

        resolved_items.append((item["label"], he_path, graph_path))

    title_font, header_font, label_font = load_fonts()

    rows = len(resolved_items)
    panel_w = PAD * 3 + IMAGE_SIZE * 2 + PAIR_GAP
    panel_h = (
        PAD * 2
        + TITLE_HEIGHT
        + HEADER_HEIGHT
        + rows * (LABEL_HEIGHT + IMAGE_SIZE + PAD)
    )

    canvas = Image.new("RGB", (panel_w, panel_h), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((PAD, PAD), TITLE_TEXT, fill="black", font=title_font)

    x_he = PAD
    x_graph = PAD + IMAGE_SIZE + PAIR_GAP
    y_header = PAD + TITLE_HEIGHT

    draw.text((x_he, y_header), HE_HEADER_TEXT, fill="black", font=header_font)
    draw.text((x_graph, y_header), GRAPH_HEADER_TEXT, fill="black", font=header_font)

    y = y_header + HEADER_HEIGHT

    for label, he_path, graph_path in resolved_items:
        draw.text((PAD, y), label, fill="black", font=label_font)
        y_img = y + LABEL_HEIGHT

        he_img = load_square(he_path, IMAGE_SIZE)
        graph_img = load_square(graph_path, IMAGE_SIZE)

        canvas.paste(he_img, (x_he, y_img))
        canvas.paste(graph_img, (x_graph, y_img))

        y = y_img + IMAGE_SIZE + PAD

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(300, 300))

    pdf_path = out_path.with_suffix(".pdf")
    canvas.save(pdf_path, "PDF", resolution=300.0)

    print(f"Saved {out_path}")
    print(f"Saved {pdf_path}")


def main() -> None:
    args = parse_args()
    build_panel(args.he_dir, args.graph_dir, args.out)


if __name__ == "__main__":
    main()
