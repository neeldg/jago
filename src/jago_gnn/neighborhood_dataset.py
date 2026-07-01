"""Dataset for JAGO neighborhood completion (v1).

For each graph patch we randomly centre a circular spatial mask on an
existing cell and remove every cell inside the circle from the input graph.
The model sees the surrounding *context* graph and must predict:

  - cell-type composition fractions inside the hidden region (length-C vector)
  - log1p(number of hidden cells) as a scalar count target

Column-name resolution, file discovery, and slide-level splitting are all
delegated to dataset.py so this module stays focused on the spatial masking
logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from jago_gnn.dataset import (
    _min_max_normalize,
    build_cell_type_vocab,
    find_patch_files,
    load_patch,
    split_by_slide,
)

__all__ = [
    "NeighborhoodCompletionDataset",
    "collate_neighborhood",
    "build_cell_type_vocab",
    "find_patch_files",
    "load_patch",
    "split_by_slide",
]


def _build_sample(
    patch: dict,
    vocab: dict,
    num_classes: int,
    mask_radius_um: float,
    sample_idx: int,
    rng: np.random.Generator,
    min_hidden_cells: int,
    min_context_cells: int,
) -> dict | None:
    """Attempt to build one neighborhood-completion sample from a patch.

    Returns None if the sampled circle doesn't meet the minimum cell counts.
    """
    x = patch["x"]
    y = patch["y"]
    cell_type_raw = patch["cell_type_raw"]
    orig_ei = patch["edge_index"]   # (2, E), symmetrized
    n_total = len(patch["cell_ids"])

    center_idx = rng.integers(0, n_total)
    cx, cy = float(x[center_idx]), float(y[center_idx])

    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    hidden_local = np.where(dist <= mask_radius_um)[0]
    context_local = np.where(dist > mask_radius_um)[0]

    if len(hidden_local) < min_hidden_cells or len(context_local) < min_context_cells:
        return None

    n_hidden = len(hidden_local)
    n_context = len(context_local)

    # Build boolean index maps for fast vectorised edge operations.
    context_bool = np.zeros(n_total, dtype=bool)
    context_bool[context_local] = True
    hidden_bool = np.zeros(n_total, dtype=bool)
    hidden_bool[hidden_local] = True

    # Re-index edges that stay within the context subgraph.
    remap = np.full(n_total, -1, dtype=np.int64)
    remap[context_local] = np.arange(n_context, dtype=np.int64)

    if orig_ei.shape[1] > 0:
        src, dst = orig_ei[0], orig_ei[1]
        keep = context_bool[src] & context_bool[dst]
        if keep.any():
            context_ei = np.stack([remap[src[keep]], remap[dst[keep]]], axis=0).astype(np.int64)
        else:
            context_ei = np.zeros((2, 0), dtype=np.int64)

        # Ring neighbours: context cells with at least one edge into the hidden region.
        ring_edge = context_bool[src] & hidden_bool[dst]
        if ring_edge.any():
            ring_ctx_idx = np.unique(remap[src[ring_edge]])
            ring_mask = np.zeros(n_context, dtype=bool)
            ring_mask[ring_ctx_idx] = True
        else:
            ring_mask = np.zeros(n_context, dtype=bool)
    else:
        context_ei = np.zeros((2, 0), dtype=np.int64)
        ring_mask = np.zeros(n_context, dtype=bool)

    def _type_fractions(indices: np.ndarray) -> np.ndarray:
        comp = np.zeros(num_classes, dtype=np.float32)
        for t in cell_type_raw[indices]:
            cid = vocab.get(t)
            if cid is not None and cid < num_classes:
                comp[cid] += 1.0
        total = comp.sum()
        return comp / total if total > 0 else comp

    target_comp = _type_fractions(hidden_local)
    context_frac = _type_fractions(context_local)
    ring_idx = np.where(ring_mask)[0]
    ring_frac = _type_fractions(context_local[ring_idx]) if ring_idx.size > 0 else None

    # Node features for context cells: [x_norm, y_norm, one_hot(cell_type)].
    # Coordinates are normalised within the *full* patch so the model can
    # read spatial position relative to the patch boundary.
    x_norm_all = _min_max_normalize(x)
    y_norm_all = _min_max_normalize(y)
    x_ctx = x_norm_all[context_local]
    y_ctx = y_norm_all[context_local]

    one_hot = np.zeros((n_context, num_classes), dtype=np.float32)
    for i, t in enumerate(cell_type_raw[context_local]):
        cid = vocab.get(t)
        if cid is not None and cid < num_classes:
            one_hot[i, cid] = 1.0

    node_features = np.concatenate([x_ctx[:, None], y_ctx[:, None], one_hot], axis=1).astype(np.float32)

    return {
        "x": torch.from_numpy(node_features),
        "edge_index": torch.from_numpy(context_ei).long(),
        "target_composition": torch.from_numpy(target_comp),
        "target_count": torch.tensor(float(np.log1p(n_hidden)), dtype=torch.float32),
        "context_frac": torch.from_numpy(context_frac),
        "ring_frac": torch.from_numpy(ring_frac) if ring_frac is not None else None,
        "n_context": n_context,
        "n_hidden": n_hidden,
        "slide_id": patch["slide_id"],
        "patch_id": patch["patch_id"],
        "sample_idx": sample_idx,
        "center_x": cx,
        "center_y": cy,
    }


class NeighborhoodCompletionDataset(Dataset):
    """Precomputed circular-mask neighborhood-completion examples.

    All samples are generated once at __init__ time (not on-the-fly per
    epoch) so iteration is fast and the val/test sets are deterministic by
    default.  Training variation comes from model dropout, not data
    re-sampling.

    Attributes
    ----------
    samples : list[dict]
        Precomputed sample dicts ready for __getitem__.
    num_classes : int
        Number of real cell-type classes (MASK token excluded; not used here).
    feature_dim : int
        Width of each node feature vector (2 + num_classes).
    n_skipped : int
        Samples discarded for not meeting minimum cell-count thresholds.
    """

    def __init__(
        self,
        patches: list,
        vocab: dict,
        samples_per_patch: int = 5,
        mask_radius_um: float = 100.0,
        min_hidden_cells: int = 5,
        min_context_cells: int = 20,
        seed: int = 0,
        deterministic: bool = True,
    ):
        self.patches = patches
        self.vocab = vocab
        # vocab includes MASK_TOKEN as the last entry; exclude it here.
        self.num_classes = len(vocab) - 1
        self.feature_dim = 2 + self.num_classes
        self.samples_per_patch = samples_per_patch
        self.mask_radius_um = mask_radius_um
        self.min_hidden_cells = min_hidden_cells
        self.min_context_cells = min_context_cells
        self.deterministic = deterministic
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)

        self.samples, self.n_skipped = self._build_all_samples()

    def _build_all_samples(self) -> tuple[list, int]:
        samples = []
        n_skipped = 0
        for patch in self.patches:
            if len(patch["cell_ids"]) == 0:
                continue
            for s_idx in range(self.samples_per_patch):
                if self.deterministic:
                    seed_val = abs(hash((patch["slide_id"], patch["patch_id"], s_idx))) % (2 ** 32)
                    rng = np.random.default_rng(seed_val)
                else:
                    rng = self._rng

                sample = _build_sample(
                    patch, self.vocab, self.num_classes,
                    self.mask_radius_um, s_idx, rng,
                    self.min_hidden_cells, self.min_context_cells,
                )
                if sample is None:
                    n_skipped += 1
                else:
                    samples.append(sample)
        return samples, n_skipped

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_neighborhood(batch: list) -> dict:
    """Collate variable-size context graphs into one offset-indexed batch.

    ring_frac entries may be None (when a sample has no ring-neighbour cells);
    they are kept as a Python list rather than stacked into a tensor so
    callers can handle missing values explicitly.
    """
    x_list, ei_list, batch_idx_list = [], [], []
    target_comp_list, target_count_list = [], []
    context_frac_list, ring_frac_list = [], []
    metadata = []

    offset = 0
    for pos, sample in enumerate(batch):
        n = sample["n_context"]
        x_list.append(sample["x"])
        target_comp_list.append(sample["target_composition"])
        target_count_list.append(sample["target_count"])
        context_frac_list.append(sample["context_frac"])
        ring_frac_list.append(sample["ring_frac"])   # Tensor or None
        batch_idx_list.append(torch.full((n,), pos, dtype=torch.long))

        ei = sample["edge_index"]
        if ei.numel() > 0:
            ei_list.append(ei + offset)

        metadata.append({
            "slide_id": sample["slide_id"],
            "patch_id": sample["patch_id"],
            "sample_idx": sample["sample_idx"],
            "center_x": sample["center_x"],
            "center_y": sample["center_y"],
            "n_hidden": sample["n_hidden"],
            "n_context": n,
        })
        offset += n

    return {
        "x": torch.cat(x_list, dim=0),
        "edge_index": torch.cat(ei_list, dim=1) if ei_list else torch.zeros((2, 0), dtype=torch.long),
        "batch_index": torch.cat(batch_idx_list, dim=0),
        "n_graphs": len(batch),
        "target_composition": torch.stack(target_comp_list, dim=0),
        "target_count": torch.stack(target_count_list, dim=0),
        "context_frac": torch.stack(context_frac_list, dim=0),
        "ring_frac": ring_frac_list,
        "metadata": metadata,
    }
