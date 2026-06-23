"""Parse Hover-Net inference JSON output into a flat cell table CSV.

Hover-Net inference output is a JSON file with a `nuc` dict mapping each
nucleus id to a record containing at least a pixel `centroid` (and usually a
`type` class id). This script flattens that into a cell table CSV with
columns: cell_id, x_um, y_um, cell_type.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse a Hover-Net inference JSON file into a cell table CSV."
    )
    parser.add_argument(
        "--json", required=True, type=Path, help="Path to Hover-Net inference JSON output."
    )
    parser.add_argument(
        "--slide-id",
        required=True,
        type=str,
        help="Slide identifier, prefixed onto each cell_id to keep ids unique across slides.",
    )
    parser.add_argument(
        "--mpp",
        required=True,
        type=float,
        help="Microns per pixel, used to convert pixel centroids into microns.",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Path to write the output cell table CSV."
    )
    return parser.parse_args()


def load_nuclei(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"Hover-Net JSON not found: {json_path}")

    with json_path.open() as f:
        data = json.load(f)

    nuc = data.get("nuc")
    if nuc is None:
        raise ValueError(f"Hover-Net JSON {json_path} has no 'nuc' key.")

    return nuc


def build_cell_table(nuc: dict, slide_id: str, mpp: float) -> pd.DataFrame:
    rows = []
    for nuc_id, record in nuc.items():
        centroid = record.get("centroid")
        if centroid is None:
            raise ValueError(f"Nucleus {nuc_id} is missing a 'centroid' field.")

        x_px, y_px = centroid
        cell_type = record.get("type_name")
        if cell_type is None:
            type_id = record.get("type", "unknown")
            cell_type = f"type_{type_id}"

        rows.append(
            {
                "cell_id": f"{slide_id}_{nuc_id}",
                "x_um": float(x_px) * mpp,
                "y_um": float(y_px) * mpp,
                "cell_type": str(cell_type),
            }
        )

    cells = pd.DataFrame(rows, columns=["cell_id", "x_um", "y_um", "cell_type"])
    cells["cell_id"] = cells["cell_id"].astype(str)
    return cells


def main() -> None:
    args = parse_args()

    nuc = load_nuclei(args.json)
    cells = build_cell_table(nuc, args.slide_id, args.mpp)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.out, index=False)

    print(f"Number of cells: {len(cells)}")
    print(f"Output path: {args.out}")


if __name__ == "__main__":
    main()
