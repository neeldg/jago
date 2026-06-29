"""Train the first JAGO masked-cell-type GNN (self-supervised, node-level).

Randomly masks a fraction of cell-type labels per graph patch and trains a
pure-PyTorch GraphSAGE-style model to predict the masked types from the
surrounding (visible) cell types, coordinates, and graph structure.

v0.1 adds: optional class-balanced weighted loss, confusion matrices for
val/test, a manual macro-F1 fallback when sklearn isn't available, simple
non-learned baselines (majority class / training-frequency random /
neighbor majority), and human-readable per-class accuracy reporting for the
standard HoverNet 6-class scheme.

Usage:
    python src/jago_gnn/train_masked_cell.py \
        --patch-root /scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/patches \
        --outdir /scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/gnn_v0
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jago_gnn.dataset import (
    MASK_TOKEN,
    MaskedCellDataset,
    build_cell_type_vocab,
    collate_patches,
    find_patch_files,
    load_patch,
    split_by_slide,
)
from jago_gnn.model import MaskedCellGNN

try:
    from sklearn.metrics import f1_score, confusion_matrix

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Standard HoverNet 6-class type_id scheme, used only for human-readable
# logging when the dataset's cell-type column happens to be numeric type_ids.
TYPE_ID_LABELS = {
    "0": "No label",
    "1": "Neoplastic",
    "2": "Inflammatory",
    "3": "Connective",
    "4": "Necrotic",
    "5": "Non-neoplastic epithelial",
}


def human_readable_class_name(raw_class_name: str) -> str:
    """Map a HoverNet-style numeric type_id to a readable name, if recognized."""
    label = TYPE_ID_LABELS.get(str(raw_class_name))
    if label is None:
        return str(raw_class_name)
    return f"type_{raw_class_name} ({label})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the first JAGO masked-cell-type GNN."
    )
    parser.add_argument("--patch-root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--epochs", required=False, type=int, default=50)
    parser.add_argument("--batch-size", required=False, type=int, default=8)
    parser.add_argument("--hidden-dim", required=False, type=int, default=64)
    parser.add_argument("--num-layers", required=False, type=int, default=3)
    parser.add_argument("--mask-rate", required=False, type=float, default=0.2)
    parser.add_argument("--lr", required=False, type=float, default=1e-3)
    parser.add_argument("--seed", required=False, type=int, default=0)
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        default=False,
        help="Use class-balanced weighted cross-entropy, weighted by global "
        "training-label frequency (inverse-frequency 'balanced' weights).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_all_patches(patch_root: Path) -> list:
    records = find_patch_files(patch_root)
    patches = []
    n_skipped = 0
    for record in records:
        try:
            patches.append(load_patch(record))
        except (ValueError, FileNotFoundError) as exc:
            print(f"Warning: skipping {record['cells_path']}: {exc}")
            n_skipped += 1

    if not patches:
        raise ValueError(
            f"No valid patches could be loaded from {patch_root} ({n_skipped} skipped)."
        )

    print(f"Loaded {len(patches)} patch(es), skipped {n_skipped}.")
    return patches


def compute_label_counts(patches: list, vocab: dict, num_classes: int) -> np.ndarray:
    """Count occurrences of each (non-MASK) class across a set of patches."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for patch in patches:
        for cell_type in patch["cell_type_raw"]:
            class_id = vocab.get(cell_type)
            if class_id is not None and class_id < num_classes:
                counts[class_id] += 1
    return counts


def class_weights_from_counts(counts: np.ndarray) -> torch.Tensor:
    """Inverse-frequency 'balanced' weights: weight_c = n_samples / (n_classes * count_c)."""
    safe_counts = np.clip(counts, 1.0, None)
    total = safe_counts.sum()
    num_classes = safe_counts.shape[0]
    weights = total / (num_classes * safe_counts)
    return torch.tensor(weights, dtype=torch.float32)


def macro_f1_manual(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> float:
    """Pure-NumPy macro F1 (matches sklearn's average='macro', zero_division=0)."""
    f1_scores = []
    for class_id in range(num_classes):
        tp = int(np.sum((preds == class_id) & (labels == class_id)))
        fp = int(np.sum((preds == class_id) & (labels != class_id)))
        fn = int(np.sum((preds != class_id) & (labels == class_id)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores)) if f1_scores else float("nan")


def compute_macro_f1(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> float:
    if labels.size == 0:
        return float("nan")
    if SKLEARN_AVAILABLE:
        return float(
            f1_score(labels, preds, average="macro", zero_division=0, labels=list(range(num_classes)))
        )
    return macro_f1_manual(labels, preds, num_classes)


def compute_confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    if SKLEARN_AVAILABLE:
        return confusion_matrix(labels, preds, labels=list(range(num_classes)))
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, preds):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def save_confusion_matrix_csv(
    path: Path, labels: np.ndarray, preds: np.ndarray, idx_to_class: dict, num_classes: int
) -> None:
    cm = compute_confusion_matrix(labels, preds, num_classes)
    class_names = [idx_to_class.get(i, str(i)) for i in range(num_classes)]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class\\pred_class"] + class_names)
        for row_name, row in zip(class_names, cm):
            writer.writerow([row_name] + row.tolist())


def log_per_class_accuracy(log_fn, title: str, per_class_acc: dict, idx_to_class: dict) -> None:
    log_fn(title)
    for class_id in sorted(idx_to_class):
        class_name = idx_to_class[class_id]
        acc = per_class_acc.get(class_id)
        acc_str = f"{acc:.4f}" if acc is not None else "n/a (not present in this split)"
        log_fn(f"    {human_readable_class_name(class_name)}: {acc_str}")


def majority_class_baseline_metrics(loader, majority_class: int, num_classes: int) -> dict:
    """Always predict the single most frequent training class."""
    all_preds, all_labels = [], []
    for batch in loader:
        labels = batch["labels"].numpy()
        mask = batch["mask"].numpy()
        if mask.sum() == 0:
            continue
        masked_labels = labels[mask]
        preds = np.full(masked_labels.shape, majority_class, dtype=np.int64)
        all_preds.append(preds)
        all_labels.append(masked_labels)

    if not all_labels:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return {"accuracy": float((preds == labels).mean()), "macro_f1": compute_macro_f1(labels, preds, num_classes)}


def random_frequency_baseline_metrics(loader, class_probs: np.ndarray, num_classes: int, seed: int) -> dict:
    """Sample predictions from the training label-frequency distribution."""
    rng = np.random.default_rng(seed)
    classes = np.arange(num_classes)
    all_preds, all_labels = [], []
    for batch in loader:
        labels = batch["labels"].numpy()
        mask = batch["mask"].numpy()
        n_masked = int(mask.sum())
        if n_masked == 0:
            continue
        preds = rng.choice(classes, size=n_masked, p=class_probs)
        all_preds.append(preds)
        all_labels.append(labels[mask])

    if not all_labels:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return {"accuracy": float((preds == labels).mean()), "macro_f1": compute_macro_f1(labels, preds, num_classes)}


def neighbor_majority_baseline_metrics(loader, majority_class: int, num_classes: int) -> dict:
    """Predict the majority type among each masked node's visible (unmasked) neighbors.

    Falls back to the global majority class when a masked node has no visible
    neighbors. Edges never cross patches within a batch (collate only offsets
    indices within each patch), so this can run directly on the batched graph.
    """
    all_preds, all_labels = [], []
    for batch in loader:
        edge_index = batch["edge_index"].numpy()
        labels = batch["labels"].numpy()
        mask = batch["mask"].numpy()
        masked_idx = np.where(mask)[0]
        if masked_idx.size == 0:
            continue

        visible = ~mask
        neighbor_labels = {}
        src, dst = edge_index[0], edge_index[1]
        for s, d in zip(src, dst):
            if visible[s]:
                neighbor_labels.setdefault(d, []).append(labels[s])

        preds = np.full(masked_idx.shape, majority_class, dtype=np.int64)
        for pos, node in enumerate(masked_idx):
            votes = neighbor_labels.get(node)
            if votes:
                values, counts = np.unique(votes, return_counts=True)
                preds[pos] = values[np.argmax(counts)]

        all_preds.append(preds)
        all_labels.append(labels[masked_idx])

    if not all_labels:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return {"accuracy": float((preds == labels).mean()), "macro_f1": compute_macro_f1(labels, preds, num_classes)}


def run_epoch(model, loader, optimizer, loss_fn, device, num_classes, train: bool):
    model.train(mode=train)

    total_loss = 0.0
    total_masked = 0
    all_preds = []
    all_labels = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            if mask.sum().item() == 0:
                continue

            logits = model(x, edge_index)
            masked_logits = logits[mask]
            masked_labels = labels[mask]
            loss = loss_fn(masked_logits, masked_labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n_masked = masked_labels.size(0)
            total_loss += loss.item() * n_masked
            total_masked += n_masked

            preds = masked_logits.argmax(dim=1)
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(masked_labels.detach().cpu().numpy())

    if total_masked == 0:
        return {"loss": float("nan"), "acc": float("nan"), "macro_f1": float("nan"), "per_class_acc": {}, "preds": np.array([]), "labels": np.array([])}

    avg_loss = total_loss / total_masked
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = float((preds == labels).mean())
    macro_f1 = compute_macro_f1(labels, preds, num_classes)

    per_class_acc = {}
    for class_id in sorted(set(labels.tolist())):
        class_mask = labels == class_id
        per_class_acc[class_id] = float((preds[class_mask] == labels[class_mask]).mean())

    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "per_class_acc": per_class_acc,
        "preds": preds,
        "labels": labels,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.outdir.mkdir(parents=True, exist_ok=True)
    log_path = args.outdir / "train_log.txt"
    metrics_path = args.outdir / "metrics.csv"
    best_model_path = args.outdir / "best_model.pt"
    baseline_metrics_path = args.outdir / "baseline_metrics.csv"
    confusion_matrix_val_path = args.outdir / "confusion_matrix_val.csv"
    confusion_matrix_test_path = args.outdir / "confusion_matrix_test.csv"

    log_lines = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    log(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Args: {vars(args)}")
    if not SKLEARN_AVAILABLE:
        log("Warning: sklearn not available; macro F1 and confusion matrices will be computed with pure NumPy fallbacks.")

    patches = load_all_patches(args.patch_root)
    vocab = build_cell_type_vocab(patches)
    idx_to_class = {idx: name for name, idx in vocab.items() if name != MASK_TOKEN}
    log(f"Cell type vocab ({len(vocab) - 1} classes + MASK): {vocab}")

    train_patches, val_patches, test_patches = split_by_slide(patches, seed=args.seed)
    log(
        f"Split by slide_id: train={len(train_patches)} patches "
        f"({len({p['slide_id'] for p in train_patches})} slides), "
        f"val={len(val_patches)} patches ({len({p['slide_id'] for p in val_patches})} slides), "
        f"test={len(test_patches)} patches ({len({p['slide_id'] for p in test_patches})} slides)."
    )

    train_ds = MaskedCellDataset(train_patches, vocab, mask_rate=args.mask_rate, seed=args.seed, deterministic=False)
    val_ds = MaskedCellDataset(val_patches, vocab, mask_rate=args.mask_rate, seed=args.seed, deterministic=True)
    test_ds = MaskedCellDataset(test_patches, vocab, mask_rate=args.mask_rate, seed=args.seed, deterministic=True)

    train_label_counts = compute_label_counts(train_patches, vocab, train_ds.num_classes)
    if train_label_counts.sum() > 0:
        majority_class = int(np.argmax(train_label_counts))
        class_probs = train_label_counts / train_label_counts.sum()
    else:
        majority_class = 0
        class_probs = np.full(train_ds.num_classes, 1.0 / train_ds.num_classes)

    log("Training-set class distribution (global label frequency; used for class weights & baselines):")
    for class_id in sorted(idx_to_class):
        log(f"    {human_readable_class_name(idx_to_class[class_id])}: {int(train_label_counts[class_id])}")
    log(f"  Majority class: {human_readable_class_name(idx_to_class[majority_class])}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_patches)
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_patches)
        if len(val_ds) > 0 else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_patches)
        if len(test_ds) > 0 else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    model = MaskedCellGNN(
        in_dim=train_ds.feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=train_ds.num_classes,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.use_class_weights:
        class_weights = class_weights_from_counts(train_label_counts).to(device)
        weight_str = ", ".join(
            f"{human_readable_class_name(idx_to_class[i])}={w:.3f}" for i, w in enumerate(class_weights.tolist())
        )
        log(f"Using class-balanced weighted cross-entropy. Weights: {weight_str}")
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    metrics_rows = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, loss_fn, device, train_ds.num_classes, train=True)

        if val_loader is not None:
            val_metrics = run_epoch(model, val_loader, optimizer, loss_fn, device, train_ds.num_classes, train=False)
        else:
            val_metrics = {"loss": float("nan"), "acc": float("nan"), "macro_f1": float("nan"), "per_class_acc": {}, "preds": np.array([]), "labels": np.array([])}

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        for class_id, class_name in idx_to_class.items():
            row[f"val_acc__{class_name}"] = val_metrics["per_class_acc"].get(class_id, float("nan"))
        metrics_rows.append(row)

        log(
            f"Epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['acc']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        compare_loss = val_metrics["loss"] if val_loader is not None else train_metrics["loss"]
        if compare_loss < best_val_loss:
            best_val_loss = compare_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "in_dim": train_ds.feature_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "num_classes": train_ds.num_classes,
                    "args": vars(args),
                    "epoch": epoch,
                },
                best_model_path,
            )
            log(f"  -> saved new best model (loss={compare_loss:.4f}) to {best_model_path}")

    if test_loader is not None:
        test_metrics = run_epoch(model, test_loader, optimizer, loss_fn, device, train_ds.num_classes, train=False)
        log(
            f"Final test | test_loss={test_metrics['loss']:.4f} | "
            f"test_acc={test_metrics['acc']:.4f} | test_macro_f1={test_metrics['macro_f1']:.4f}"
        )
        test_row = {
            "epoch": "test",
            "train_loss": float("nan"),
            "val_loss": test_metrics["loss"],
            "val_acc": test_metrics["acc"],
            "val_macro_f1": test_metrics["macro_f1"],
        }
        for class_id, class_name in idx_to_class.items():
            test_row[f"val_acc__{class_name}"] = test_metrics["per_class_acc"].get(class_id, float("nan"))
        metrics_rows.append(test_row)
    else:
        log("Test split is empty; skipping final test evaluation.")
        test_metrics = None

    if val_loader is not None:
        log_per_class_accuracy(log, "Per-class accuracy (final validation epoch):", val_metrics["per_class_acc"], idx_to_class)
    if test_metrics is not None:
        log_per_class_accuracy(log, "Per-class accuracy (test):", test_metrics["per_class_acc"], idx_to_class)

    if val_loader is not None and val_metrics["labels"].size > 0:
        save_confusion_matrix_csv(confusion_matrix_val_path, val_metrics["labels"], val_metrics["preds"], idx_to_class, train_ds.num_classes)
        log(f"Saved val confusion matrix: {confusion_matrix_val_path}")
    if test_metrics is not None and test_metrics["labels"].size > 0:
        save_confusion_matrix_csv(confusion_matrix_test_path, test_metrics["labels"], test_metrics["preds"], idx_to_class, train_ds.num_classes)
        log(f"Saved test confusion matrix: {confusion_matrix_test_path}")

    baseline_rows = []
    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        if loader is None:
            continue
        majority_metrics = majority_class_baseline_metrics(loader, majority_class, train_ds.num_classes)
        baseline_rows.append({"baseline": "majority_class", "split": split_name, **majority_metrics})

        random_metrics = random_frequency_baseline_metrics(loader, class_probs, train_ds.num_classes, seed=args.seed)
        baseline_rows.append({"baseline": "random_by_training_frequency", "split": split_name, **random_metrics})

        neighbor_metrics = neighbor_majority_baseline_metrics(loader, majority_class, train_ds.num_classes)
        baseline_rows.append({"baseline": "neighbor_majority", "split": split_name, **neighbor_metrics})

    if baseline_rows:
        with baseline_metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["baseline", "split", "accuracy", "macro_f1"])
            writer.writeheader()
            writer.writerows(baseline_rows)
        log(f"Saved baseline metrics: {baseline_metrics_path}")
        log("Baseline metrics:")
        for row in baseline_rows:
            log(f"    baseline={row['baseline']:<28s} split={row['split']:<5s} acc={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f}")
    else:
        log("No val/test splits available; skipping baseline metrics.")

    fieldnames = list(metrics_rows[0].keys()) if metrics_rows else []
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)
    log(f"Saved metrics: {metrics_path}")

    with log_path.open("w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"Saved training log: {log_path}")


if __name__ == "__main__":
    main()
