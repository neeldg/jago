"""Train the first JAGO masked-cell-type GNN (self-supervised, node-level).

Randomly masks a fraction of cell-type labels per graph patch and trains a
pure-PyTorch GraphSAGE-style model to predict the masked types from the
surrounding (visible) cell types, coordinates, and graph structure.

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
    from sklearn.metrics import f1_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


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


def run_epoch(model, loader, optimizer, loss_fn, device, train: bool):
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

    if SKLEARN_AVAILABLE:
        macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    else:
        macro_f1 = float("nan")

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

    log_lines = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    log(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Args: {vars(args)}")
    if not SKLEARN_AVAILABLE:
        log("Warning: sklearn not available; macro F1 will be reported as NaN.")

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
    loss_fn = nn.CrossEntropyLoss()

    metrics_rows = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)

        if val_loader is not None:
            val_metrics = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)
        else:
            val_metrics = {"loss": float("nan"), "acc": float("nan"), "macro_f1": float("nan"), "per_class_acc": {}}

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
        test_metrics = run_epoch(model, test_loader, optimizer, loss_fn, device, train=False)
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
