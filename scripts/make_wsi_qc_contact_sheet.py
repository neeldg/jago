"""Build a QC contact sheet of whole-slide image thumbnails.

Scans a directory for .svs files, renders a low-resolution thumbnail of each
via OpenSlide, arranges them into a labeled grid contact sheet (PNG + PDF),
and writes a wsi_qc_candidates.csv stub for manual slide QC triage.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import openslide
import pandas as pd

THUMBNAIL_SIZE = (256, 256)

QC_CSV_COLUMNS = [
    "slide_id",
    "wsi_path",
    "qc_status",
    "stain_type",
    "qc_reason",
    "notes",
    "use_for_training",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a QC contact sheet of whole-slide image thumbnails."
    )
    parser.add_argument("--wsi-dir", required=True, type=Path, help="Directory containing .svs files.")
    parser.add_argument("--outdir", required=True, type=Path, help="Directory to write outputs into.")
    parser.add_argument(
        "--max-slides", required=False, type=int, default=50, help="Maximum number of slides to include."
    )
    return parser.parse_args()


def find_svs_files(wsi_dir: Path, max_slides: int) -> list:
    svs_paths = sorted(wsi_dir.glob("*.svs"))
    if not svs_paths:
        raise FileNotFoundError(f"No .svs files found in {wsi_dir}")
    return svs_paths[:max_slides]


def load_thumbnail(svs_path: Path):
    slide = openslide.OpenSlide(str(svs_path))
    thumbnail = slide.get_thumbnail(THUMBNAIL_SIZE)
    slide.close()
    return thumbnail


def build_contact_sheet(slide_ids: list, thumbnails: list, out_png: Path, out_pdf: Path) -> None:
    n = len(thumbnails)
    n_cols = max(1, math.ceil(math.sqrt(n)))
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3 * n_cols, 3.3 * n_rows), facecolor="white"
    )
    axes_flat = axes.flat if n > 1 else [axes]

    for ax, slide_id, thumbnail in zip(axes_flat, slide_ids, thumbnails):
        ax.imshow(thumbnail)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(slide_id, fontsize=7, wrap=True)

    for ax in list(axes_flat)[n:]:
        ax.axis("off")

    fig.suptitle("WSI QC Contact Sheet", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_qc_candidates(slide_ids: list, svs_paths: list) -> pd.DataFrame:
    rows = [
        {
            "slide_id": slide_id,
            "wsi_path": str(svs_path),
            "qc_status": "unreviewed",
            "stain_type": "unreviewed",
            "qc_reason": "",
            "notes": "",
            "use_for_training": "",
        }
        for slide_id, svs_path in zip(slide_ids, svs_paths)
    ]
    return pd.DataFrame(rows, columns=QC_CSV_COLUMNS)


def main() -> None:
    args = parse_args()

    if not args.wsi_dir.exists():
        raise FileNotFoundError(f"WSI directory not found: {args.wsi_dir}")

    svs_paths = find_svs_files(args.wsi_dir, args.max_slides)
    slide_ids = [p.stem for p in svs_paths]

    thumbnails = []
    for svs_path in svs_paths:
        print(f"Loading thumbnail: {svs_path.name}")
        thumbnails.append(load_thumbnail(svs_path))

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_png = args.outdir / "wsi_qc_contact_sheet.png"
    out_pdf = args.outdir / "wsi_qc_contact_sheet.pdf"
    build_contact_sheet(slide_ids, thumbnails, out_png, out_pdf)

    qc_candidates = build_qc_candidates(slide_ids, svs_paths)
    qc_csv_path = args.outdir / "wsi_qc_candidates.csv"
    qc_candidates.to_csv(qc_csv_path, index=False)

    print(f"Number of slides: {len(svs_paths)}")
    print(f"Saved contact sheet: {out_png}")
    print(f"Saved contact sheet: {out_pdf}")
    print(f"Saved QC candidates: {qc_csv_path}")


if __name__ == "__main__":
    main()
