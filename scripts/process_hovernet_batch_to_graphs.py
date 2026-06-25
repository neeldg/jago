"""Batch-process a directory of Hover-Net JSON outputs into JAGO graphs.

For each Hover-Net inference JSON file, runs the full JAGO pipeline:
parse_hovernet_json.py -> build_radius_graph.py -> make_graph_patches.py ->
compute_patch_stats.py -> rank_patches.py, producing cell tables, radius
graphs, graph patches, patch stats, and architecture rankings per slide.
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-process Hover-Net JSON outputs into JAGO graphs, patches, stats, and rankings."
    )
    parser.add_argument(
        "--json-dir", required=True, type=Path, help="Directory containing Hover-Net inference JSON files."
    )
    parser.add_argument(
        "--out-root", required=True, type=Path, help="Root directory to write all outputs into."
    )
    parser.add_argument(
        "--type-map", required=True, type=Path, help="Path to the type id -> readable name JSON map."
    )
    parser.add_argument("--mpp", required=False, type=float, default=0.2525, help="Microns per pixel.")
    parser.add_argument("--radius-um", required=False, type=float, default=50.0, help="Graph radius in microns.")
    parser.add_argument("--window-um", required=False, type=float, default=500.0, help="Patch window size in microns.")
    parser.add_argument("--stride-um", required=False, type=float, default=500.0, help="Patch sliding-window stride in microns.")
    parser.add_argument("--min-cells", required=False, type=int, default=100, help="Minimum cells per patch.")
    parser.add_argument("--max-patches", required=False, type=int, default=50, help="Maximum patches per slide.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Reprocess a slide even if its patch_stats.csv already exists."
    )
    return parser.parse_args()


def run_step(command: list) -> None:
    print(f"$ {' '.join(str(part) for part in command)}", flush=True)
    subprocess.run(command, check=True)


def process_slide(json_path: Path, args: argparse.Namespace) -> None:
    slide_id = json_path.stem

    cell_tables_dir = args.out_root / "cell_tables"
    graphs_dir = args.out_root / "graphs"
    patches_dir = args.out_root / "patches" / slide_id
    rankings_dir = args.out_root / "rankings" / slide_id

    cell_tables_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)
    rankings_dir.mkdir(parents=True, exist_ok=True)

    patch_stats_path = patches_dir / "patch_stats.csv"
    if patch_stats_path.exists() and not args.overwrite:
        print(
            f"Skipping {slide_id}: {patch_stats_path} already exists (use --overwrite to reprocess).",
            flush=True,
        )
        return

    cells_path = cell_tables_dir / f"{slide_id}_cells.csv"
    edges_path = graphs_dir / f"{slide_id}_edges_{int(args.radius_um)}um.csv"

    run_step(
        [
            sys.executable,
            str(SCRIPTS_DIR / "parse_hovernet_json.py"),
            "--json", str(json_path),
            "--slide-id", slide_id,
            "--mpp", str(args.mpp),
            "--out", str(cells_path),
        ]
    )

    run_step(
        [
            sys.executable,
            str(SCRIPTS_DIR / "build_radius_graph.py"),
            "--cells", str(cells_path),
            "--radius-um", str(args.radius_um),
            "--out", str(edges_path),
        ]
    )

    run_step(
        [
            sys.executable,
            str(SCRIPTS_DIR / "make_graph_patches.py"),
            "--cells", str(cells_path),
            "--edges", str(edges_path),
            "--out-dir", str(patches_dir),
            "--window-um", str(args.window_um),
            "--stride-um", str(args.stride_um),
            "--min-cells", str(args.min_cells),
            "--max-patches", str(args.max_patches),
        ]
    )

    run_step(
        [
            sys.executable,
            str(SCRIPTS_DIR / "compute_patch_stats.py"),
            "--patch-dir", str(patches_dir),
            "--manifest", str(patches_dir / "patch_manifest.csv"),
            "--type-map", str(args.type_map),
            "--out", str(patch_stats_path),
        ]
    )

    run_step(
        [
            sys.executable,
            str(SCRIPTS_DIR / "rank_patches.py"),
            "--patch-stats", str(patch_stats_path),
            "--out-dir", str(rankings_dir),
            "--top-n", "10",
        ]
    )


def main() -> None:
    args = parse_args()

    if not args.json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {args.json_dir}")

    json_paths = sorted(args.json_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No .json files found in {args.json_dir}")

    args.out_root.mkdir(parents=True, exist_ok=True)

    for json_path in json_paths:
        print(f"=== Processing slide: {json_path.stem} ===", flush=True)
        process_slide(json_path, args)

    print(f"Processed {len(json_paths)} slide(s) into {args.out_root}")


if __name__ == "__main__":
    main()
