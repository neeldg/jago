"""Extract real H&E image crops from a whole-slide .svs file for JAGO graph patches.

Reads patch geometry from a patch_manifest.csv (as produced by
make_graph_patches.py), converts each patch's micron-space bounding box into
pixel coordinates, and pulls the corresponding crop out of the source
whole-slide image via OpenSlide.
"""

import argparse
from pathlib import Path

import openslide
import pandas as pd

REQUIRED_MANIFEST_COLUMNS = [
    "patch_id",
    "center_x_um",
    "center_y_um",
    "x_min_um",
    "x_max_um",
    "y_min_um",
    "y_max_um",
    "n_cells",
    "n_edges",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract H&E image crops from a .svs slide for JAGO graph patches."
    )
    parser.add_argument("--slide", required=True, type=Path, help="Path to the whole-slide .svs file.")
    parser.add_argument(
        "--manifest", required=True, type=Path, help="Path to patch_manifest.csv."
    )
    parser.add_argument(
        "--patch-ids",
        required=True,
        type=str,
        help="Comma-separated patch ids to extract, e.g. patch_000042,patch_000049.",
    )
    parser.add_argument(
        "--mpp", required=False, type=float, default=0.2525, help="Microns per pixel of the slide."
    )
    parser.add_argument(
        "--out-dir", required=True, type=Path, help="Directory to write crop PNGs and metadata into."
    )
    return parser.parse_args()


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    manifest = pd.read_csv(path)

    missing = [col for col in REQUIRED_MANIFEST_COLUMNS if col not in manifest.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {missing}. "
            f"Required columns are: {REQUIRED_MANIFEST_COLUMNS}"
        )

    return manifest


def main() -> None:
    args = parse_args()

    if not args.slide.exists():
        raise FileNotFoundError(f"Slide not found: {args.slide}")

    manifest = load_manifest(args.manifest)
    requested_ids = [patch_id.strip() for patch_id in args.patch_ids.split(",") if patch_id.strip()]

    missing_ids = sorted(set(requested_ids) - set(manifest["patch_id"]))
    if missing_ids:
        raise ValueError(f"Patch id(s) not found in manifest {args.manifest}: {missing_ids}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    slide = openslide.OpenSlide(str(args.slide))
    manifest_by_id = manifest.set_index("patch_id")

    metadata_rows = []

    for patch_id in requested_ids:
        row = manifest_by_id.loc[patch_id]

        x_px = round(row["x_min_um"] / args.mpp)
        y_px = round(row["y_min_um"] / args.mpp)
        w_px = round((row["x_max_um"] - row["x_min_um"]) / args.mpp)
        h_px = round((row["y_max_um"] - row["y_min_um"]) / args.mpp)

        print(f"Extracting {patch_id}: region=({x_px}, {y_px}), size=({w_px}, {h_px})")

        region = slide.read_region((x_px, y_px), 0, (w_px, h_px))
        rgb_image = region.convert("RGB")

        output_path = args.out_dir / f"{patch_id}_he.png"
        rgb_image.save(output_path)

        metadata_rows.append(
            {
                "patch_id": patch_id,
                "x_px": x_px,
                "y_px": y_px,
                "w_px": w_px,
                "h_px": h_px,
                "output_path": str(output_path),
            }
        )

    slide.close()

    metadata = pd.DataFrame(
        metadata_rows, columns=["patch_id", "x_px", "y_px", "w_px", "h_px", "output_path"]
    )
    metadata.to_csv(args.out_dir / "crop_metadata.csv", index=False)

    print(f"Extracted {len(metadata_rows)} patch crop(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
