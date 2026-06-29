"""Dataset utilities for the JAGO masked-cell-type GNN.

Loads JAGO graph patches (a `*_cells.csv` + matching `*_edges.csv` pair per
patch), builds a shared cell-type vocabulary, and exposes a
`torch.utils.data.Dataset` that randomly masks a fraction of cell-type labels
per patch for self-supervised masked-cell-type prediction.

Column names across JAGO outputs are not perfectly consistent (different
pipeline versions / manual exports), so every column lookup goes through
`resolve_column`, which accepts a list of likely candidate names and raises a
clear error listing what was searched for and what was actually found.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

X_COLUMN_CANDIDATES = ["x_um", "x", "centroid_x", "global_x"]
Y_COLUMN_CANDIDATES = ["y_um", "y", "centroid_y", "global_y"]
CELL_ID_COLUMN_CANDIDATES = ["cell_id", "id", "node_id"]
CELL_TYPE_COLUMN_CANDIDATES = ["cell_type", "type", "predicted_type", "type_id"]
SOURCE_COLUMN_CANDIDATES = ["source_cell_id", "source", "src", "cell_i"]
TARGET_COLUMN_CANDIDATES = ["target_cell_id", "target", "dst", "cell_j"]

MASK_TOKEN = "<MASK>"


def resolve_column(columns, candidates, role: str, path: Path) -> str:
    """Find which of `candidates` is present in `columns` (case-insensitive)."""
    lower_to_actual = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]
    raise ValueError(
        f"Could not find a '{role}' column in {path}. "
        f"Looked for any of {candidates}, but the file only has columns: {list(columns)}."
    )


def infer_slide_and_patch_id(cells_path: Path, patch_root: Path) -> tuple:
    """Infer (slide_id, patch_id) from a cells CSV path.

    Handles both the nested JAGO layout
    (<patch_root>/<slide_id>/patch_<n>_cells.csv) and the flat
    "<slide_id>_patch_<n>_cells.csv" naming convention.
    """
    patch_name = cells_path.name[: -len("_cells.csv")]

    try:
        relative_parts = cells_path.relative_to(patch_root).parts
    except ValueError:
        relative_parts = (cells_path.name,)

    if len(relative_parts) >= 2:
        slide_id = relative_parts[-2]
        patch_id = patch_name
    elif "_patch_" in patch_name:
        slide_id, _, suffix = patch_name.partition("_patch_")
        patch_id = "patch_" + suffix
    else:
        slide_id = "unknown_slide"
        patch_id = patch_name

    return slide_id, patch_id


def find_patch_files(patch_root: Path) -> list:
    """Recursively find every (cells_path, edges_path, slide_id, patch_id)."""
    patch_root = Path(patch_root)
    if not patch_root.exists():
        raise FileNotFoundError(f"Patch root not found: {patch_root}")

    cells_paths = sorted(patch_root.rglob("*_cells.csv"))
    if not cells_paths:
        raise FileNotFoundError(f"No *_cells.csv files found under {patch_root}")

    records = []
    for cells_path in cells_paths:
        edges_path = Path(str(cells_path).replace("_cells.csv", "_edges.csv"))
        slide_id, patch_id = infer_slide_and_patch_id(cells_path, patch_root)
        records.append(
            {
                "slide_id": slide_id,
                "patch_id": patch_id,
                "cells_path": cells_path,
                "edges_path": edges_path,
            }
        )
    return records


def load_patch(record: dict) -> dict:
    """Load one patch's cells/edges CSVs into raw numpy arrays.

    Returns a dict with: slide_id, patch_id, cell_ids (list[str]), x, y
    (float arrays), cell_type_raw (str array), edge_index (int64 [2, E],
    already symmetrized so both directions of every edge are present).
    """
    cells_path = record["cells_path"]
    edges_path = record["edges_path"]

    cells_df = pd.read_csv(cells_path)
    if len(cells_df) == 0:
        raise ValueError(f"Cells file has no rows: {cells_path}")

    x_col = resolve_column(cells_df.columns, X_COLUMN_CANDIDATES, "x coordinate", cells_path)
    y_col = resolve_column(cells_df.columns, Y_COLUMN_CANDIDATES, "y coordinate", cells_path)
    type_col = resolve_column(cells_df.columns, CELL_TYPE_COLUMN_CANDIDATES, "cell type", cells_path)

    try:
        id_col = resolve_column(cells_df.columns, CELL_ID_COLUMN_CANDIDATES, "cell id", cells_path)
        cell_ids = cells_df[id_col].astype(str).tolist()
    except ValueError:
        # No id column: fall back to row position as the id.
        cell_ids = [str(i) for i in range(len(cells_df))]

    x = cells_df[x_col].astype(float).to_numpy()
    y = cells_df[y_col].astype(float).to_numpy()
    cell_type_raw = cells_df[type_col].astype(str).to_numpy()

    id_to_idx = {cell_id: i for i, cell_id in enumerate(cell_ids)}

    edge_index = _load_edges(edges_path, id_to_idx)

    return {
        "slide_id": record["slide_id"],
        "patch_id": record["patch_id"],
        "cell_ids": cell_ids,
        "x": x,
        "y": y,
        "cell_type_raw": cell_type_raw,
        "edge_index": edge_index,
    }


def _load_edges(edges_path: Path, id_to_idx: dict) -> np.ndarray:
    if not edges_path.exists():
        return np.zeros((2, 0), dtype=np.int64)

    edges_df = pd.read_csv(edges_path)
    if len(edges_df) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    try:
        source_col = resolve_column(edges_df.columns, SOURCE_COLUMN_CANDIDATES, "edge source", edges_path)
        target_col = resolve_column(edges_df.columns, TARGET_COLUMN_CANDIDATES, "edge target", edges_path)
    except ValueError:
        if len(edges_df) == 0:
            return np.zeros((2, 0), dtype=np.int64)
        raise

    source_ids = edges_df[source_col].astype(str).tolist()
    target_ids = edges_df[target_col].astype(str).tolist()

    src_idx = []
    dst_idx = []
    unmapped = []
    for s_id, t_id in zip(source_ids, target_ids):
        s_idx = id_to_idx.get(s_id)
        t_idx = id_to_idx.get(t_id)
        if s_idx is None:
            unmapped.append(s_id)
            continue
        if t_idx is None:
            unmapped.append(t_id)
            continue
        src_idx.append(s_idx)
        dst_idx.append(t_idx)

    if unmapped:
        sample = unmapped[:5]
        raise ValueError(
            f"{edges_path}: {len(unmapped)} edge endpoint id(s) did not match any "
            f"cell id in the corresponding cells file. Example unmatched id(s): {sample}."
        )

    forward = np.array([src_idx, dst_idx], dtype=np.int64)
    backward = np.array([dst_idx, src_idx], dtype=np.int64)
    # Symmetrize so mean-aggregation message passing sees both directions,
    # regardless of whether the source CSV stored directed or undirected
    # (single i<j) edges.
    edge_index = np.concatenate([forward, backward], axis=1)
    return edge_index


def build_cell_type_vocab(patches: list) -> dict:
    """Build a deterministic {readable_type_string: class_id} vocab.

    The MASK_TOKEN is always appended last so its index equals len(vocab).
    """
    distinct_types = set()
    for patch in patches:
        distinct_types.update(patch["cell_type_raw"].tolist())

    sorted_types = sorted(distinct_types)
    vocab = {cell_type: idx for idx, cell_type in enumerate(sorted_types)}
    vocab[MASK_TOKEN] = len(vocab)
    return vocab


def split_by_slide(patches: list, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 0) -> tuple:
    """Split patches into train/val/test by slide_id (not by individual patch)."""
    slide_ids = sorted({p["slide_id"] for p in patches})
    rng = random.Random(seed)
    rng.shuffle(slide_ids)

    n_slides = len(slide_ids)
    if n_slides == 1:
        print("Warning: only 1 unique slide_id found; all patches go to train, val/test are empty.")
        train_slides, val_slides, test_slides = set(slide_ids), set(), set()
    elif n_slides == 2:
        print("Warning: only 2 unique slide_ids found; using train/val split, test is empty.")
        train_slides, val_slides, test_slides = {slide_ids[0]}, {slide_ids[1]}, set()
    else:
        n_train = max(1, round(n_slides * train_frac))
        n_val = max(1, round(n_slides * val_frac))
        n_train = min(n_train, n_slides - 2)
        n_val = min(n_val, n_slides - n_train - 1)
        train_slides = set(slide_ids[:n_train])
        val_slides = set(slide_ids[n_train:n_train + n_val])
        test_slides = set(slide_ids[n_train + n_val:])

    train = [p for p in patches if p["slide_id"] in train_slides]
    val = [p for p in patches if p["slide_id"] in val_slides]
    test = [p for p in patches if p["slide_id"] in test_slides]
    return train, val, test


class MaskedCellDataset(Dataset):
    """Per-patch masked-cell-type prediction examples.

    Node features are [x_norm, y_norm, one_hot(cell_type or MASK)].
    Labels are the original (unmasked) cell-type class id for every node;
    callers should only compute loss/metrics on positions where mask==True.
    """

    def __init__(self, patches: list, vocab: dict, mask_rate: float = 0.2, seed: int = 0, deterministic: bool = False):
        if not 0.0 < mask_rate < 1.0:
            raise ValueError(f"mask_rate must be in (0, 1), got {mask_rate}")

        self.patches = patches
        self.vocab = vocab
        self.mask_rate = mask_rate
        self.deterministic = deterministic
        self.num_classes = len(vocab) - 1  # excludes the MASK token itself
        self.feature_dim = 2 + len(vocab)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.patches)

    def _mask_indicator(self, patch: dict) -> np.ndarray:
        n_nodes = len(patch["cell_ids"])
        if self.deterministic:
            rng = np.random.default_rng(abs(hash((patch["slide_id"], patch["patch_id"]))) % (2 ** 32))
        else:
            rng = self._rng

        n_masked = max(1, int(round(n_nodes * self.mask_rate)))
        n_masked = min(n_masked, n_nodes - 1) if n_nodes > 1 else n_nodes
        masked_positions = rng.choice(n_nodes, size=n_masked, replace=False)

        mask = np.zeros(n_nodes, dtype=bool)
        mask[masked_positions] = True
        return mask

    def __getitem__(self, idx: int) -> dict:
        patch = self.patches[idx]
        n_nodes = len(patch["cell_ids"])

        x = patch["x"].astype(np.float32)
        y = patch["y"].astype(np.float32)
        x_norm = _min_max_normalize(x)
        y_norm = _min_max_normalize(y)

        labels = np.array(
            [self.vocab.get(t, self.vocab[MASK_TOKEN]) for t in patch["cell_type_raw"]], dtype=np.int64
        )
        mask = self._mask_indicator(patch)

        mask_class_id = self.vocab[MASK_TOKEN]
        num_total_classes = len(self.vocab)
        one_hot = np.zeros((n_nodes, num_total_classes), dtype=np.float32)
        visible_class_ids = np.where(mask, mask_class_id, labels)
        one_hot[np.arange(n_nodes), visible_class_ids] = 1.0

        node_features = np.concatenate(
            [x_norm[:, None], y_norm[:, None], one_hot], axis=1
        ).astype(np.float32)

        return {
            "x": torch.from_numpy(node_features),
            "edge_index": torch.from_numpy(patch["edge_index"]).long(),
            "labels": torch.from_numpy(labels),
            "mask": torch.from_numpy(mask),
            "n_nodes": n_nodes,
            "slide_id": patch["slide_id"],
            "patch_id": patch["patch_id"],
        }


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    v_min = values.min()
    v_max = values.max()
    denom = v_max - v_min
    if denom < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - v_min) / denom).astype(np.float32)


def collate_patches(batch: list) -> dict:
    """Collate a list of per-patch examples into one offset-indexed graph batch."""
    x_list = []
    edge_index_list = []
    labels_list = []
    mask_list = []
    batch_index_list = []
    slide_ids = []
    patch_ids = []

    offset = 0
    for batch_pos, example in enumerate(batch):
        n_nodes = example["n_nodes"]

        x_list.append(example["x"])
        labels_list.append(example["labels"])
        mask_list.append(example["mask"])
        batch_index_list.append(torch.full((n_nodes,), batch_pos, dtype=torch.long))
        slide_ids.append(example["slide_id"])
        patch_ids.append(example["patch_id"])

        edge_index = example["edge_index"]
        if edge_index.numel() > 0:
            edge_index_list.append(edge_index + offset)

        offset += n_nodes

    x = torch.cat(x_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    mask = torch.cat(mask_list, dim=0)
    batch_index = torch.cat(batch_index_list, dim=0)
    edge_index = (
        torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros((2, 0), dtype=torch.long)
    )

    return {
        "x": x,
        "edge_index": edge_index,
        "labels": labels,
        "mask": mask,
        "batch_index": batch_index,
        "n_nodes_total": offset,
        "slide_ids": slide_ids,
        "patch_ids": patch_ids,
    }
