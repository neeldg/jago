#!/bin/bash
#SBATCH --job-name=jago_gnn_v0
#SBATCH --partition=ccurtis2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=jago_gnn_v0_%j.out
#SBATCH --error=jago_gnn_v0_%j.err

# First JAGO masked-cell-type GNN training run (v0) on Sherlock.
# Adjust --partition/--gpus/--mem above to match your allocation before submitting.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_ROOT="/scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/patches"
OUTDIR="/scratch/groups/ccurtis2/neeldg/jago/outputs/batch5_jago/gnn_v0"

# Activate the JAGO conda/virtual environment here, e.g.:
# module load python/3.9
# source activate jago

cd "$REPO_ROOT"

python src/jago_gnn/train_masked_cell.py \
  --patch-root "$PATCH_ROOT" \
  --outdir "$OUTDIR" \
  --epochs 50 \
  --batch-size 8 \
  --hidden-dim 64 \
  --num-layers 3 \
  --mask-rate 0.2 \
  --lr 1e-3 \
  --seed 0
