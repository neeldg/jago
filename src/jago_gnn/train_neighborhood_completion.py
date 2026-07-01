"""Train JAGO neighborhood completion v1 (self-supervised, graph-level).

For each context graph (cells outside a circular spatial mask) the model
predicts:
  1. cell-type composition fractions inside the hidden region (KL divergence)
  2. log1p count of hidden cells (Huber loss)

Three non-learned baselines are reported for reference:
  - global_train_dist   : always predict the training-set mean composition
  - context_dist        : predict each sample's own context composition
  - ring_neighbor       : predict composition of ring-adjacent context cells,
                          falling back to context_dist when no ring cells exist

Usage:
    python src/jago_gnn/train_neighborhood_completion.py \
        --patch-root /scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/patches \
        --outdir     /scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/neighborhood_v1
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jago_gnn.dataset import build_cell_type_vocab, find_patch_files, load_patch, split_by_slide
from jago_gnn.model import NeighborhoodCompletionGNN
from jago_gnn.neighborhood_dataset import NeighborhoodCompletionDataset, collate_neighborhood

TYPE_ID_LABELS = {
    "0": "No label",
    "1": "Neoplastic",
    "2": "Inflammatory",
    "3": "Connective",
    "4": "Necrotic",
    "5": "Non-neoplastic epithelial",
}


def human_readable_class_name(raw: str) -> str:
    label = TYPE_ID_LABELS.get(str(raw))
    return f"type_{raw} ({label})" if label is not None else str(raw)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train JAGO neighborhood completion v1.")
    p.add_argument("--patch-root", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--mask-radius-um", type=float, default=100.0)
    p.add_argument("--samples-per-patch", type=int, default=5)
    p.add_argument("--min-hidden-cells", type=int, default=5)
    p.add_argument("--min-context-cells", type=int, default=20)
    p.add_argument("--count-loss-weight", type=float, default=0.1)
    p.add_argument("--use-virtual-node", action="store_true", default=False)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_all_patches(patch_root: Path) -> list:
    records = find_patch_files(patch_root)
    patches, n_skipped = [], 0
    for rec in records:
        try:
            patches.append(load_patch(rec))
        except (ValueError, FileNotFoundError) as exc:
            print(f"Warning: skipping {rec['cells_path']}: {exc}")
            n_skipped += 1
    if not patches:
        raise ValueError(f"No valid patches could be loaded from {patch_root}.")
    print(f"Loaded {len(patches)} patch(es), skipped {n_skipped}.")
    return patches


# ---------------------------------------------------------------------------
# Per-epoch training / evaluation
# ---------------------------------------------------------------------------

def _empty_result(num_classes: int) -> dict:
    return {
        "loss": float("nan"),
        "comp_mae": float("nan"),
        "per_class_comp_mae": np.full(num_classes, float("nan")),
        "count_mae": float("nan"),
        "count_rmse": float("nan"),
        "records": [],
    }


def run_epoch(
    model: NeighborhoodCompletionGNN,
    loader: DataLoader,
    optimizer,
    loss_fn_comp: nn.Module,
    loss_fn_count: nn.Module,
    count_loss_weight: float,
    device: torch.device,
    num_classes: int,
    train: bool,
) -> dict:
    model.train(mode=train)
    total_loss = 0.0
    n_samples = 0
    all_pred_comp, all_true_comp = [], []
    all_pred_count, all_true_count = [], []
    all_metadata = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_index = batch["batch_index"].to(device)
            n_graphs = batch["n_graphs"]
            target_comp = batch["target_composition"].to(device)   # (B, C)
            target_count = batch["target_count"].to(device)        # (B,)

            comp_logits, count_pred = model(x, edge_index, batch_index, n_graphs)

            loss_comp = loss_fn_comp(F.log_softmax(comp_logits, dim=-1), target_comp)
            loss_count = loss_fn_count(count_pred, target_count)
            loss = loss_comp + count_loss_weight * loss_count

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            B = n_graphs
            total_loss += loss.item() * B
            n_samples += B

            pred_comp = F.softmax(comp_logits, dim=-1).detach().cpu().numpy()
            all_pred_comp.append(pred_comp)
            all_true_comp.append(target_comp.detach().cpu().numpy())
            all_pred_count.append(count_pred.detach().cpu().numpy())
            all_true_count.append(target_count.detach().cpu().numpy())
            all_metadata.extend(batch["metadata"])

    if n_samples == 0:
        return _empty_result(num_classes)

    pred_comp = np.concatenate(all_pred_comp, axis=0)   # (N, C)
    true_comp = np.concatenate(all_true_comp, axis=0)
    pred_count = np.concatenate(all_pred_count, axis=0)
    true_count = np.concatenate(all_true_count, axis=0)

    abs_comp_err = np.abs(pred_comp - true_comp)        # (N, C)
    records = [
        {
            **meta,
            "true_comp": true_comp[i],
            "pred_comp": pred_comp[i],
            "true_count": float(true_count[i]),
            "pred_count": float(pred_count[i]),
        }
        for i, meta in enumerate(all_metadata)
    ]

    return {
        "loss": total_loss / n_samples,
        "comp_mae": float(abs_comp_err.mean()),
        "per_class_comp_mae": abs_comp_err.mean(axis=0),               # (C,)
        "count_mae": float(np.abs(pred_count - true_count).mean()),
        "count_rmse": float(np.sqrt(np.mean((pred_count - true_count) ** 2))),
        "records": records,
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def evaluate_baselines(
    loader: DataLoader,
    train_global_comp: np.ndarray,
    train_mean_count: float,
    num_classes: int,
) -> dict:
    """Return metrics for three composition baselines on one split."""
    all_true_comp, all_true_count = [], []
    all_context_frac, all_ring_pred = [], []

    for batch in loader:
        true_comp = batch["target_composition"].numpy()     # (B, C)
        true_count = batch["target_count"].numpy()          # (B,)
        context_frac = batch["context_frac"].numpy()        # (B, C)
        ring_fracs = batch["ring_frac"]                     # list[Tensor | None]
        B = batch["n_graphs"]

        all_true_comp.append(true_comp)
        all_true_count.append(true_count)
        all_context_frac.append(context_frac)

        ring_preds = []
        for i in range(B):
            rf = ring_fracs[i]
            ring_preds.append(rf.numpy() if rf is not None else context_frac[i])
        all_ring_pred.append(np.stack(ring_preds, axis=0))

    true_comp = np.concatenate(all_true_comp, axis=0)
    true_count = np.concatenate(all_true_count, axis=0)
    context_frac = np.concatenate(all_context_frac, axis=0)
    ring_pred = np.concatenate(all_ring_pred, axis=0)
    N = len(true_comp)
    global_pred = np.tile(train_global_comp, (N, 1))
    count_baseline = np.full(N, train_mean_count)

    results = {}
    for name, comp_pred in [
        ("global_train_dist", global_pred),
        ("context_dist", context_frac),
        ("ring_neighbor", ring_pred),
    ]:
        abs_err = np.abs(comp_pred - true_comp)
        results[name] = {
            "comp_mae": float(abs_err.mean()),
            "per_class_comp_mae": abs_err.mean(axis=0),
            "count_mae": float(np.abs(count_baseline - true_count).mean()),
            "count_rmse": float(np.sqrt(np.mean((count_baseline - true_count) ** 2))),
        }
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_predictions_csv(path: Path, records: list, idx_to_class: dict, num_classes: int) -> None:
    class_names = [idx_to_class.get(i, str(i)) for i in range(num_classes)]
    fields = (
        ["slide_id", "patch_id", "sample_idx",
         "center_x", "center_y", "n_hidden", "n_context",
         "true_count", "pred_count"]
        + [f"true_comp_{n}" for n in class_names]
        + [f"pred_comp_{n}" for n in class_names]
    )
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in records:
            row = {
                "slide_id": rec["slide_id"],
                "patch_id": rec["patch_id"],
                "sample_idx": rec["sample_idx"],
                "center_x": round(rec["center_x"], 2),
                "center_y": round(rec["center_y"], 2),
                "n_hidden": rec["n_hidden"],
                "n_context": rec["n_context"],
                "true_count": round(rec["true_count"], 4),
                "pred_count": round(rec["pred_count"], 4),
            }
            for i, n in enumerate(class_names):
                row[f"true_comp_{n}"] = round(float(rec["true_comp"][i]), 4)
                row[f"pred_comp_{n}"] = round(float(rec["pred_comp"][i]), 4)
            w.writerow(row)


def save_baseline_metrics_csv(
    path: Path,
    baseline_results: dict,
    idx_to_class: dict,
    num_classes: int,
) -> None:
    class_names = [idx_to_class.get(i, str(i)) for i in range(num_classes)]
    per_class_fields = [f"comp_mae__{n}" for n in class_names]
    fields = ["baseline", "split", "comp_mae", "count_mae", "count_rmse"] + per_class_fields
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (baseline_name, split_name), metrics in baseline_results.items():
            row = {
                "baseline": baseline_name,
                "split": split_name,
                "comp_mae": round(metrics["comp_mae"], 4),
                "count_mae": round(metrics["count_mae"], 4),
                "count_rmse": round(metrics["count_rmse"], 4),
            }
            for i, n in enumerate(class_names):
                row[f"comp_mae__{n}"] = round(float(metrics["per_class_comp_mae"][i]), 4)
            w.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.outdir.mkdir(parents=True, exist_ok=True)
    log_path = args.outdir / "train_log.txt"
    metrics_path = args.outdir / "metrics.csv"
    best_model_path = args.outdir / "best_model.pt"
    baseline_metrics_path = args.outdir / "baseline_metrics.csv"
    predictions_val_path = args.outdir / "predictions_val.csv"
    predictions_test_path = args.outdir / "predictions_test.csv"

    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    log(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Args: {vars(args)}")

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------
    patches = load_all_patches(args.patch_root)
    vocab = build_cell_type_vocab(patches)
    # vocab includes MASK_TOKEN; num_classes excludes it.
    num_classes = len(vocab) - 1
    idx_to_class = {idx: name for name, idx in vocab.items() if name not in ("<MASK>",)}
    log(f"Cell type vocab ({num_classes} classes): { {k: v for k, v in vocab.items() if k != '<MASK>'} }")

    train_patches, val_patches, test_patches = split_by_slide(patches, seed=args.seed)
    log(
        f"Split: train={len(train_patches)} patches "
        f"({len({p['slide_id'] for p in train_patches})} slides), "
        f"val={len(val_patches)} ({len({p['slide_id'] for p in val_patches})} slides), "
        f"test={len(test_patches)} ({len({p['slide_id'] for p in test_patches})} slides)."
    )

    ds_kwargs = dict(
        vocab=vocab,
        samples_per_patch=args.samples_per_patch,
        mask_radius_um=args.mask_radius_um,
        min_hidden_cells=args.min_hidden_cells,
        min_context_cells=args.min_context_cells,
        seed=args.seed,
    )
    train_ds = NeighborhoodCompletionDataset(train_patches, deterministic=False, **ds_kwargs)
    val_ds = NeighborhoodCompletionDataset(val_patches, deterministic=True, **ds_kwargs)
    test_ds = NeighborhoodCompletionDataset(test_patches, deterministic=True, **ds_kwargs)

    log(
        f"Samples — train: {len(train_ds)} (skipped {train_ds.n_skipped}), "
        f"val: {len(val_ds)} (skipped {val_ds.n_skipped}), "
        f"test: {len(test_ds)} (skipped {test_ds.n_skipped})."
    )
    if len(train_ds) == 0:
        raise ValueError(
            "Training set is empty after applying min_hidden_cells / min_context_cells filters. "
            "Try reducing --mask-radius-um or the minimum cell counts."
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_neighborhood)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_neighborhood) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_neighborhood) if test_ds else None

    # -----------------------------------------------------------------------
    # Training-set composition statistics (used by baselines)
    # -----------------------------------------------------------------------
    all_train_comp = np.stack([s["target_composition"].numpy() for s in train_ds.samples], axis=0)
    train_global_comp = all_train_comp.mean(axis=0)     # (C,)
    train_mean_count = float(np.mean([s["target_count"].item() for s in train_ds.samples]))
    log("Training-set mean hidden composition:")
    for cid in sorted(idx_to_class):
        log(f"    {human_readable_class_name(idx_to_class[cid])}: {train_global_comp[cid]:.3f}")
    log(f"  Mean log1p(n_hidden): {train_mean_count:.3f}  (≈ {np.expm1(train_mean_count):.1f} cells)")

    # -----------------------------------------------------------------------
    # Model, optimiser, loss
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    model = NeighborhoodCompletionGNN(
        in_dim=train_ds.feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        use_virtual_node=args.use_virtual_node,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {total_params:,}  (virtual_node={args.use_virtual_node})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # KLDivLoss expects log-probabilities as input; batchmean divides by B.
    loss_fn_comp = nn.KLDivLoss(reduction="batchmean")
    loss_fn_count = nn.HuberLoss()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    metrics_rows: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = 0

    class_names = [idx_to_class.get(i, str(i)) for i in range(num_classes)]

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(
            model, train_loader, optimizer,
            loss_fn_comp, loss_fn_count, args.count_loss_weight,
            device, num_classes, train=True,
        )
        if val_loader is not None:
            val_m = run_epoch(
                model, val_loader, optimizer,
                loss_fn_comp, loss_fn_count, args.count_loss_weight,
                device, num_classes, train=False,
            )
        else:
            val_m = _empty_result(num_classes)

        row: dict = {
            "epoch": epoch,
            "train_loss": round(train_m["loss"], 5),
            "train_comp_mae": round(train_m["comp_mae"], 5),
            "train_count_mae": round(train_m["count_mae"], 5),
            "val_loss": round(val_m["loss"], 5),
            "val_comp_mae": round(val_m["comp_mae"], 5),
            "val_count_mae": round(val_m["count_mae"], 5),
            "val_count_rmse": round(val_m["count_rmse"], 5),
        }
        for i, cn in enumerate(class_names):
            row[f"val_comp_mae__{cn}"] = round(float(val_m["per_class_comp_mae"][i]), 5)
        metrics_rows.append(row)

        log(
            f"Epoch {epoch:04d} | "
            f"train_loss={train_m['loss']:.4f} comp_mae={train_m['comp_mae']:.4f} count_mae={train_m['count_mae']:.4f} | "
            f"val_loss={val_m['loss']:.4f} comp_mae={val_m['comp_mae']:.4f} count_mae={val_m['count_mae']:.4f} rmse={val_m['count_rmse']:.4f}"
        )

        compare = val_m["loss"] if val_loader is not None else train_m["loss"]
        if compare < best_val_loss:
            best_val_loss = compare
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "in_dim": train_ds.feature_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "num_classes": num_classes,
                    "use_virtual_node": args.use_virtual_node,
                    "args": vars(args),
                    "epoch": epoch,
                },
                best_model_path,
            )
            log(f"  -> best model saved (val_loss={compare:.4f}) at epoch {epoch}")

    log(f"Training complete. Best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f}")

    # -----------------------------------------------------------------------
    # Final evaluation on best checkpoint
    # -----------------------------------------------------------------------
    log(f"Reloading best checkpoint (epoch {best_epoch}) for final evaluation.")
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    final_val_records, final_test_records = [], []

    if val_loader is not None:
        final_val_m = run_epoch(
            model, val_loader, optimizer,
            loss_fn_comp, loss_fn_count, args.count_loss_weight,
            device, num_classes, train=False,
        )
        final_val_records = final_val_m["records"]
        log(
            f"Final val (best ckpt) | comp_mae={final_val_m['comp_mae']:.4f} "
            f"count_mae={final_val_m['count_mae']:.4f} count_rmse={final_val_m['count_rmse']:.4f}"
        )
        log("Per-class composition MAE (val):")
        for cid in sorted(idx_to_class):
            log(f"    {human_readable_class_name(idx_to_class[cid])}: {final_val_m['per_class_comp_mae'][cid]:.4f}")

    if test_loader is not None:
        final_test_m = run_epoch(
            model, test_loader, optimizer,
            loss_fn_comp, loss_fn_count, args.count_loss_weight,
            device, num_classes, train=False,
        )
        final_test_records = final_test_m["records"]
        log(
            f"Final test (best ckpt) | comp_mae={final_test_m['comp_mae']:.4f} "
            f"count_mae={final_test_m['count_mae']:.4f} count_rmse={final_test_m['count_rmse']:.4f}"
        )
        log("Per-class composition MAE (test):")
        for cid in sorted(idx_to_class):
            log(f"    {human_readable_class_name(idx_to_class[cid])}: {final_test_m['per_class_comp_mae'][cid]:.4f}")

        # Append test row to metrics_rows for metrics.csv
        test_row: dict = {
            "epoch": "test",
            "train_loss": float("nan"),
            "train_comp_mae": float("nan"),
            "train_count_mae": float("nan"),
            "val_loss": round(final_test_m["loss"], 5),
            "val_comp_mae": round(final_test_m["comp_mae"], 5),
            "val_count_mae": round(final_test_m["count_mae"], 5),
            "val_count_rmse": round(final_test_m["count_rmse"], 5),
        }
        for i, cn in enumerate(class_names):
            test_row[f"val_comp_mae__{cn}"] = round(float(final_test_m["per_class_comp_mae"][i]), 5)
        metrics_rows.append(test_row)
    else:
        log("Test split is empty; skipping final test evaluation.")

    # -----------------------------------------------------------------------
    # Baselines
    # -----------------------------------------------------------------------
    all_baseline_results: dict = {}
    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        if loader is None:
            continue
        bl = evaluate_baselines(loader, train_global_comp, train_mean_count, num_classes)
        for bname, bmetrics in bl.items():
            all_baseline_results[(bname, split_name)] = bmetrics

    if all_baseline_results:
        save_baseline_metrics_csv(baseline_metrics_path, all_baseline_results, idx_to_class, num_classes)
        log(f"Saved baseline metrics: {baseline_metrics_path}")
        log("Baseline summary:")
        for (bname, split), bm in all_baseline_results.items():
            log(f"    {bname:<22s}  split={split:<5s}  comp_mae={bm['comp_mae']:.4f}  count_mae={bm['count_mae']:.4f}")

    # -----------------------------------------------------------------------
    # Predictions CSVs
    # -----------------------------------------------------------------------
    if final_val_records:
        save_predictions_csv(predictions_val_path, final_val_records, idx_to_class, num_classes)
        log(f"Saved val predictions ({len(final_val_records)} rows): {predictions_val_path}")
    if final_test_records:
        save_predictions_csv(predictions_test_path, final_test_records, idx_to_class, num_classes)
        log(f"Saved test predictions ({len(final_test_records)} rows): {predictions_test_path}")

    # -----------------------------------------------------------------------
    # metrics.csv and train_log.txt
    # -----------------------------------------------------------------------
    if metrics_rows:
        with metrics_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            w.writeheader()
            w.writerows(metrics_rows)
        log(f"Saved metrics: {metrics_path}")

    with log_path.open("w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"Saved training log: {log_path}")


if __name__ == "__main__":
    main()
