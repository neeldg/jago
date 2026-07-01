"""Build a masked H&E inpainting dataset directly from QC-passed WSIs.

No Hover-Net outputs, cell graphs, or pre-computed annotations are required.
The entire pipeline runs on raw WSI files, making it possible to use the full
QC-passed slide repository immediately.

Pipeline
--------
1. Read a text file of QC-passed WSI paths.
2. Open each WSI with OpenSlide.
3. Build a low-resolution tissue mask to avoid blank / glass regions.
4. Sample tissue-containing patches at the target resolution.
5. Create a binary inpainting mask (circle or rectangle) per patch.
6. Save: original patch PNG, masked patch PNG, binary mask PNG.
7. Append metadata to metadata.csv (incremental — safe to resume on Sherlock).

Downstream use
--------------
- Train an H&E inpainting / generative model directly on these patch triplets.
- Later, run Hover-Net on generated images and compare cell graphs against
  real hidden regions as biological validation.

Dependencies
------------
  openslide-python  (pip install openslide-python)
  Pillow, numpy, pandas  (standard scientific stack)
  On Sherlock: module load biology openslide

Usage
-----
    python scripts/build_wsi_inpainting_dataset.py \\
        --wsi-list   /path/to/qc_passed_wsis.txt \\
        --outdir     /scratch/groups/ccurtis2/neeldg/jago/inpainting_v1 \\
        --patches-per-slide 50 \\
        --patch-size 512 \\
        --wsi-level  1 \\
        --mask-type  circle \\
        --seed       42

    # Resume a partial run (appends to existing metadata.csv):
    python scripts/build_wsi_inpainting_dataset.py ... --resume

    # Debug on two slides only:
    python scripts/build_wsi_inpainting_dataset.py ... --max-slides 2 --patches-per-slide 5
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import openslide
    from openslide import OpenSlide, OpenSlideError
except ImportError:
    sys.exit(
        "openslide-python is required.\n"
        "  pip install openslide-python\n"
        "  On Sherlock: module load biology openslide && pip install openslide-python\n"
        "  See https://openslide.org/api/python/"
    )


# ---------------------------------------------------------------------------
# Metadata schema
# ---------------------------------------------------------------------------

METADATA_FIELDS = [
    "slide_id",
    "wsi_path",
    "patch_id",
    "x_wsi",
    "y_wsi",
    "wsi_level",
    "level_downsample",
    "mpp",
    "patch_size_px",
    "tissue_fraction",
    "orig_path",
    "masked_path",
    "mask_path",
    "mask_type",
    "mask_center_x",
    "mask_center_y",
    "mask_radius_px",
    "mask_width_px",
    "mask_height_px",
]


# ---------------------------------------------------------------------------
# WSI utilities
# ---------------------------------------------------------------------------

def get_slide_id(wsi_path: Path) -> str:
    """Derive a stable slide identifier from the filename stem."""
    return wsi_path.stem


def get_slide_mpp(slide: OpenSlide) -> float:
    """Return mean MPP from OpenSlide properties, or NaN if unavailable."""
    try:
        mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, "nan"))
        mpp_y = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, "nan"))
        if math.isnan(mpp_x) or math.isnan(mpp_y):
            return float("nan")
        return (mpp_x + mpp_y) / 2.0
    except (ValueError, TypeError):
        return float("nan")


def select_level(slide: OpenSlide, wsi_level: int, target_mpp: float) -> int:
    """Return the WSI pyramid level to use for extraction.

    If --target-mpp is given and native MPP is known, the closest level is
    chosen automatically.  Otherwise wsi_level is used (clamped to range).
    """
    native_mpp = get_slide_mpp(slide)
    if target_mpp is not None and not math.isnan(native_mpp):
        downsample_needed = target_mpp / native_mpp
        level = slide.get_best_level_for_downsample(downsample_needed)
    else:
        level = wsi_level
    return max(0, min(level, slide.level_count - 1))


# ---------------------------------------------------------------------------
# Tissue detection
# ---------------------------------------------------------------------------

def _rgb_to_sat_diff_and_gray(rgb_arr: np.ndarray):
    """Return (sat_diff, gray) for an RGB uint8 array.

    sat_diff = max(R,G,B) - min(R,G,B)  — high values indicate staining
    gray     = ITU-R BT.601 luma         — low values indicate dense tissue
    """
    r = rgb_arr[..., 0].astype(np.float32)
    g = rgb_arr[..., 1].astype(np.float32)
    b = rgb_arr[..., 2].astype(np.float32)
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    sat_diff = cmax - cmin
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return sat_diff, gray


def build_tissue_mask_lores(
    slide: OpenSlide,
    thumbnail_max_dim: int = 2000,
) -> tuple:
    """Build a boolean tissue mask at thumbnail resolution.

    Uses a dual criterion:
      - High colour saturation  (H&E staining: haematoxylin / eosin)
      - Dark luma               (dense nuclei, necrotic regions)

    This avoids glass, blank fields, and processing artefacts.

    Returns
    -------
    tissue : (H, W) bool array
    scale_x : thumbnail pixel → level-0 pixel, x-axis
    scale_y : thumbnail pixel → level-0 pixel, y-axis
    """
    thumb = slide.get_thumbnail((thumbnail_max_dim, thumbnail_max_dim))
    thumb_rgb = np.array(thumb.convert("RGB"))
    H, W = thumb_rgb.shape[:2]

    scale_x = slide.dimensions[0] / W
    scale_y = slide.dimensions[1] / H

    sat_diff, gray = _rgb_to_sat_diff_and_gray(thumb_rgb)
    # Thresholds: sat_diff > 12 catches stained tissue; gray < 220 catches
    # dark dense regions that may have low saturation (e.g., necrosis core).
    tissue = (sat_diff > 12.0) | (gray < 220.0)
    return tissue.astype(bool), float(scale_x), float(scale_y)


def patch_tissue_fraction(patch_rgb: np.ndarray) -> float:
    """Fraction of patch pixels classified as tissue."""
    sat_diff, gray = _rgb_to_sat_diff_and_gray(patch_rgb)
    tissue = (sat_diff > 12.0) | (gray < 220.0)
    return float(tissue.mean())


# ---------------------------------------------------------------------------
# Patch position sampling
# ---------------------------------------------------------------------------

def sample_patch_positions(
    tissue_mask: np.ndarray,
    scale_x: float,
    scale_y: float,
    patch_size_px: int,
    level_ds: float,
    n_candidates: int,
    rng: np.random.Generator,
    wsi_dims: tuple,
) -> list:
    """Sample candidate patch positions in level-0 WSI coordinates.

    Positions are drawn from tissue pixels in the thumbnail, then mapped back
    to full-resolution WSI coordinates.  Boundary constraints ensure the patch
    footprint lies entirely within the slide.

    Parameters
    ----------
    tissue_mask   : (H, W) bool array at thumbnail resolution
    scale_x/y     : thumbnail_px → level-0_px conversion factors
    patch_size_px : patch dimensions at the extraction level
    level_ds      : downsample factor of the extraction level
    n_candidates  : how many positions to attempt
    rng           : numpy random Generator (for reproducibility)
    wsi_dims      : (width, height) of the slide at level 0

    Returns
    -------
    List of (x_wsi, y_wsi) tuples in level-0 pixel space.
    """
    H, W = tissue_mask.shape
    wsi_w, wsi_h = wsi_dims

    # Patch footprint in level-0 pixels
    fp_x = patch_size_px * level_ds
    fp_y = patch_size_px * level_ds

    # Corresponding margin in thumbnail pixels
    margin_x = math.ceil(fp_x / scale_x)
    margin_y = math.ceil(fp_y / scale_y)

    # Tissue pixels that still have room for a full patch
    ys, xs = np.where(tissue_mask)
    valid = (xs + margin_x < W) & (ys + margin_y < H)
    valid_xs = xs[valid]
    valid_ys = ys[valid]

    if len(valid_xs) == 0:
        return []

    n = min(n_candidates, len(valid_xs))
    chosen_idx = rng.choice(len(valid_xs), size=n, replace=False)

    positions = []
    for i in chosen_idx:
        tx, ty = int(valid_xs[i]), int(valid_ys[i])
        # Map to level-0 coordinates; clamp so patch stays inside slide
        wsi_x = int(tx * scale_x)
        wsi_y = int(ty * scale_y)
        wsi_x = min(wsi_x, max(0, wsi_w - int(fp_x)))
        wsi_y = min(wsi_y, max(0, wsi_h - int(fp_y)))
        positions.append((wsi_x, wsi_y))

    return positions


# ---------------------------------------------------------------------------
# Mask creation and application
# ---------------------------------------------------------------------------

def make_circle_mask(patch_size: int, cx: int, cy: int, radius_px: int) -> np.ndarray:
    """Return uint8 array (patch_size × patch_size): 255 inside circle, 0 outside."""
    Y, X = np.ogrid[:patch_size, :patch_size]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    return (dist <= radius_px).astype(np.uint8) * 255


def make_rectangle_mask(
    patch_size: int, cx: int, cy: int, width_px: int, height_px: int
) -> np.ndarray:
    """Return uint8 array (patch_size × patch_size): 255 inside rectangle, 0 outside."""
    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    x0 = max(0, cx - width_px // 2)
    y0 = max(0, cy - height_px // 2)
    x1 = min(patch_size, cx + width_px // 2)
    y1 = min(patch_size, cy + height_px // 2)
    mask[y0:y1, x0:x1] = 255
    return mask


def apply_mask(
    patch_rgb: np.ndarray,
    mask: np.ndarray,
    fill: tuple = (255, 255, 255),
) -> np.ndarray:
    """Return a copy of patch_rgb with masked pixels replaced by fill colour.

    The default fill is white (255, 255, 255), which represents blank glass —
    the natural H&E background — and is the standard convention for H&E
    inpainting datasets.
    """
    out = patch_rgb.copy()
    out[mask > 0] = fill
    return out


def compute_mask_center(
    patch_size: int,
    center_mode: str,
    radius_px: int,
    width_px: int,
    height_px: int,
    rng: np.random.Generator,
) -> tuple:
    """Return (cx, cy) for the mask centre.

    center_mode="center"  → always the patch centre (deterministic, mask always
                            fully within bounds).
    center_mode="random"  → uniform random position within the inner region,
                            keeping the mask entirely inside the patch.
    """
    if center_mode == "center":
        cx = patch_size // 2
        cy = patch_size // 2
    else:
        # Inset so the mask never clips the patch boundary
        inset_x = max(width_px // 2, radius_px) + 1
        inset_y = max(height_px // 2, radius_px) + 1
        lo_x = max(1, inset_x)
        hi_x = max(lo_x + 1, patch_size - inset_x)
        lo_y = max(1, inset_y)
        hi_y = max(lo_y + 1, patch_size - inset_y)
        cx = int(rng.integers(lo_x, hi_x))
        cy = int(rng.integers(lo_y, hi_y))
    return cx, cy


# ---------------------------------------------------------------------------
# Per-slide processing
# ---------------------------------------------------------------------------

def _fmt(value: float) -> str:
    """Format a float for CSV; return empty string for NaN."""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(round(value, 4))


def process_slide(
    wsi_path: Path,
    args: argparse.Namespace,
    slide_idx: int,
    csv_writer: csv.DictWriter,
    csv_fh,
    outdir: Path,
) -> int:
    """Process one WSI: sample patches, create masks, save files, write metadata.

    Returns the number of patch triplets successfully saved.
    """
    slide_id = get_slide_id(wsi_path)
    # Per-slide RNG: deterministic and independent of other slides' outcomes
    rng = np.random.default_rng(args.seed + slide_idx * 1000003)

    try:
        slide = OpenSlide(str(wsi_path))
    except (OpenSlideError, Exception) as exc:
        print("  [WARN] Cannot open {}: {}".format(wsi_path.name, exc))
        return 0

    with slide:
        level = select_level(slide, args.wsi_level, args.target_mpp)
        level_ds = slide.level_downsamples[level]
        native_mpp = get_slide_mpp(slide)
        extraction_mpp = (
            native_mpp * level_ds if not math.isnan(native_mpp) else float("nan")
        )

        # Low-res tissue mask
        try:
            tissue_mask, scale_x, scale_y = build_tissue_mask_lores(
                slide, args.thumbnail_size
            )
        except Exception as exc:
            print("  [WARN] Tissue mask failed for {}: {}".format(slide_id, exc))
            return 0

        tissue_coverage = float(tissue_mask.mean())
        if tissue_coverage < 0.02:
            print(
                "  [WARN] {} appears mostly blank ({:.1%} tissue in thumbnail),"
                " skipping.".format(slide_id, tissue_coverage)
            )
            return 0

        # Candidate sampling positions (level-0 coords)
        candidates = sample_patch_positions(
            tissue_mask, scale_x, scale_y,
            args.patch_size, level_ds,
            args.n_candidates, rng,
            slide.dimensions,
        )
        if not candidates:
            print("  [WARN] No valid candidate positions for {}".format(slide_id))
            return 0

        # Output directory for this slide's images
        slide_dir = outdir / "images" / slide_id
        slide_dir.mkdir(parents=True, exist_ok=True)

        # Mask geometry parameters
        radius_px = int(args.patch_size * args.mask_radius_frac)
        width_px = int(args.patch_size * args.mask_width_frac)
        height_px = int(args.patch_size * args.mask_height_frac)

        n_saved = 0
        for wsi_x, wsi_y in candidates:
            if n_saved >= args.patches_per_slide:
                break

            # Extract patch
            try:
                region = slide.read_region(
                    (wsi_x, wsi_y), level,
                    (args.patch_size, args.patch_size),
                )
                patch_rgb = np.array(region.convert("RGB"))
            except Exception as exc:
                print(
                    "  [WARN] read_region failed at ({},{}) for {}: {}".format(
                        wsi_x, wsi_y, slide_id, exc
                    )
                )
                continue

            # Tissue fraction filter
            t_frac = patch_tissue_fraction(patch_rgb)
            if t_frac < args.min_tissue_frac:
                continue

            # Mask centre and binary mask
            cx, cy = compute_mask_center(
                args.patch_size, args.mask_center_mode,
                radius_px, width_px, height_px, rng,
            )
            if args.mask_type == "circle":
                mask = make_circle_mask(args.patch_size, cx, cy, radius_px)
                mask_radius_out = float(radius_px)
                mask_width_out = float("nan")
                mask_height_out = float("nan")
            else:
                mask = make_rectangle_mask(args.patch_size, cx, cy, width_px, height_px)
                mask_radius_out = float("nan")
                mask_width_out = float(width_px)
                mask_height_out = float(height_px)

            masked_rgb = apply_mask(patch_rgb, mask)

            # File paths (relative to outdir for portability)
            patch_id = "patch_{:05d}".format(n_saved + 1)
            orig_rel = "images/{}/{}_orig.png".format(slide_id, patch_id)
            masked_rel = "images/{}/{}_masked.png".format(slide_id, patch_id)
            mask_rel = "images/{}/{}_mask.png".format(slide_id, patch_id)

            try:
                Image.fromarray(patch_rgb).save(outdir / orig_rel)
                Image.fromarray(masked_rgb).save(outdir / masked_rel)
                Image.fromarray(mask, mode="L").save(outdir / mask_rel)
            except Exception as exc:
                print(
                    "  [WARN] Save failed for {} / {}: {}".format(
                        slide_id, patch_id, exc
                    )
                )
                continue

            # Write metadata row immediately so partial runs aren't lost
            csv_writer.writerow({
                "slide_id": slide_id,
                "wsi_path": str(wsi_path.resolve()),
                "patch_id": patch_id,
                "x_wsi": wsi_x,
                "y_wsi": wsi_y,
                "wsi_level": level,
                "level_downsample": _fmt(level_ds),
                "mpp": _fmt(extraction_mpp),
                "patch_size_px": args.patch_size,
                "tissue_fraction": _fmt(t_frac),
                "orig_path": orig_rel,
                "masked_path": masked_rel,
                "mask_path": mask_rel,
                "mask_type": args.mask_type,
                "mask_center_x": cx,
                "mask_center_y": cy,
                "mask_radius_px": _fmt(mask_radius_out),
                "mask_width_px": _fmt(mask_width_out),
                "mask_height_px": _fmt(mask_height_out),
            })
            csv_fh.flush()

            n_saved += 1

    return n_saved


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build masked H&E inpainting dataset from QC-passed WSIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--wsi-list", required=True, type=Path,
                   help="Text file: one WSI path per line (# lines are comments).")
    p.add_argument("--outdir", required=True, type=Path,
                   help="Root output directory.")

    # Patch sampling
    p.add_argument("--patches-per-slide", type=int, default=50,
                   help="Target number of valid patches per slide.")
    p.add_argument("--n-candidates", type=int, default=500,
                   help="Candidate positions evaluated per slide "
                        "(must be >= patches-per-slide).")
    p.add_argument("--patch-size", type=int, default=512,
                   help="Patch width and height in pixels at the extraction level.")
    p.add_argument("--min-tissue-frac", type=float, default=0.5,
                   help="Minimum tissue fraction [0,1] for a patch to be accepted.")

    # WSI resolution
    p.add_argument("--wsi-level", type=int, default=1,
                   help="WSI pyramid level to extract from (0 = highest resolution). "
                        "Typical TCGA-BRCA: level 0 ≈ 40x, level 1 ≈ 20x.")
    p.add_argument("--target-mpp", type=float, default=None,
                   help="Target microns-per-pixel (e.g. 0.5 for ~20x). "
                        "When set, overrides --wsi-level by selecting the closest level.")
    p.add_argument("--thumbnail-size", type=int, default=2000,
                   help="Maximum thumbnail dimension used for tissue detection.")

    # Masking
    p.add_argument("--mask-type", choices=["circle", "rectangle"], default="circle",
                   help="Shape of the inpainting mask.")
    p.add_argument("--mask-center-mode", choices=["center", "random"], default="center",
                   help="'center' places the mask at the patch centre (always in-bounds). "
                        "'random' samples a position within the inner patch region.")
    p.add_argument("--mask-radius-frac", type=float, default=0.25,
                   help="Circle mask radius as a fraction of patch size.")
    p.add_argument("--mask-width-frac", type=float, default=0.4,
                   help="Rectangle mask width as a fraction of patch size.")
    p.add_argument("--mask-height-frac", type=float, default=0.4,
                   help="Rectangle mask height as a fraction of patch size.")

    # Run control
    p.add_argument("--seed", type=int, default=0,
                   help="Global random seed for reproducible sampling.")
    p.add_argument("--max-slides", type=int, default=None,
                   help="Process at most this many slides (for debugging).")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Append to existing metadata.csv and skip slides whose "
                        "output directory already contains the target patch count.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.wsi_list.exists():
        sys.exit("WSI list not found: {}".format(args.wsi_list))

    if args.n_candidates < args.patches_per_slide:
        args.n_candidates = args.patches_per_slide * 10
        print(
            "Warning: --n-candidates raised to {} "
            "(10× patches-per-slide).".format(args.n_candidates)
        )

    # Load and sort WSI paths for determinism
    wsi_paths = []
    with open(args.wsi_list) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                wsi_paths.append(Path(line))

    if not wsi_paths:
        sys.exit("No WSI paths found in {}".format(args.wsi_list))

    wsi_paths.sort()
    if args.max_slides is not None:
        wsi_paths = wsi_paths[: args.max_slides]

    print("WSI paths to process : {}".format(len(wsi_paths)))
    print("Output directory     : {}".format(args.outdir))
    print(
        "Patches/slide={} | patch_size={}px | level={} | target_mpp={}".format(
            args.patches_per_slide, args.patch_size,
            args.wsi_level, args.target_mpp,
        )
    )
    print(
        "mask_type={} | mask_center_mode={} | min_tissue_frac={} | seed={}".format(
            args.mask_type, args.mask_center_mode, args.min_tissue_frac, args.seed,
        )
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.outdir / "metadata.csv"

    # Append to existing CSV when resuming; start fresh otherwise.
    if args.resume and metadata_path.exists():
        csv_mode = "a"
        write_header = False
        # Collect slide IDs already completed so we can skip them.
        import csv as _csv
        already_done = set()
        with open(metadata_path, newline="") as _fh:
            for row in _csv.DictReader(_fh):
                already_done.add(row["slide_id"])
        print("Resume mode: {} slide(s) already in metadata.csv.".format(
            len(already_done)))
    else:
        csv_mode = "w"
        write_header = True
        already_done = set()

    total_patches = 0
    n_slides_done = 0
    t0 = time.time()

    with open(metadata_path, csv_mode, newline="") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=METADATA_FIELDS)
        if write_header:
            writer.writeheader()

        for slide_idx, wsi_path in enumerate(wsi_paths):
            slide_id = get_slide_id(wsi_path)

            if slide_id in already_done:
                print("[{}/{}] Skip (already done): {}".format(
                    slide_idx + 1, len(wsi_paths), slide_id))
                continue

            if not wsi_path.exists():
                print("[{}/{}] MISSING: {}".format(
                    slide_idx + 1, len(wsi_paths), wsi_path))
                continue

            # Skip slides where we already have enough patches on disk
            if args.resume:
                slide_dir = args.outdir / "images" / slide_id
                if slide_dir.exists():
                    n_existing = len(list(slide_dir.glob("*_orig.png")))
                    if n_existing >= args.patches_per_slide:
                        print("[{}/{}] Skip ({}  patches on disk): {}".format(
                            slide_idx + 1, len(wsi_paths), n_existing, slide_id))
                        total_patches += n_existing
                        n_slides_done += 1
                        continue

            print("[{}/{}] {}".format(slide_idx + 1, len(wsi_paths), slide_id))
            t_slide = time.time()

            n = process_slide(
                wsi_path, args, slide_idx, writer, csv_fh, args.outdir
            )
            elapsed = time.time() - t_slide
            total_patches += n
            n_slides_done += 1
            print("  -> {} patch(es) in {:.1f}s".format(n, elapsed))

    total_elapsed = time.time() - t0
    print(
        "\nFinished: {} patch(es) from {}/{} slide(s) in {:.0f}s.".format(
            total_patches, n_slides_done, len(wsi_paths), total_elapsed,
        )
    )
    print("Metadata: {}".format(metadata_path))
    print("Images:   {}".format(args.outdir / "images"))


if __name__ == "__main__":
    main()
