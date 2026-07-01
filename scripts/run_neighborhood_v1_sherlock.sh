#!/bin/bash
#SBATCH --job-name=jago_nbhd_v1
#SBATCH --partition=ccurtis2
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=jago_nbhd_v1_%j.out
#SBATCH --error=jago_nbhd_v1_%j.err

# JAGO neighborhood-completion GNN v1 — Sherlock training job.
# Submit with: sbatch scripts/run_neighborhood_v1_sherlock.sh
# Adjust --partition / --gpus / --mem above to match your allocation.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_ROOT="/scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/patches"
OUTDIR="/scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/neighborhood_v1"

# Activate your environment here, e.g.:
# module load python/3.10
# conda activate jago

cd "$REPO_ROOT"

echo "=== Inspecting dataset before training ==="
python scripts/inspect_neighborhood_dataset.py \
    --patch-root "$PATCH_ROOT" \
    --mask-radius-um 100 \
    --samples-per-patch 5 \
    --seed 0

echo ""
echo "=== Starting training ==="
python src/jago_gnn/train_neighborhood_completion.py \
    --patch-root         "$PATCH_ROOT" \
    --outdir             "$OUTDIR" \
    --epochs             100 \
    --batch-size         16 \
    --hidden-dim         64 \
    --num-layers         3 \
    --mask-radius-um     100.0 \
    --samples-per-patch  5 \
    --min-hidden-cells   5 \
    --min-context-cells  20 \
    --count-loss-weight  0.1 \
    --use-virtual-node \
    --lr                 1e-3 \
    --seed               0

echo ""
echo "Outputs written to $OUTDIR"
