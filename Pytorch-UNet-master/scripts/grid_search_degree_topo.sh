#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
IMAGES_DIR="${IMAGES_DIR:-/root/autodl-tmp/datasets/DRIVE_prepared/train/imgs}"
MASKS_DIR="${MASKS_DIR:-/root/autodl-tmp/datasets/DRIVE_prepared/train/masks}"
TEST_IMAGES_DIR="${TEST_IMAGES_DIR:-/root/autodl-tmp/datasets/DRIVE_prepared/test/imgs}"
TEST_MASKS_DIR="${TEST_MASKS_DIR:-/root/autodl-tmp/datasets/DRIVE_prepared/test/masks}"
EPOCHS="${EPOCHS:-30}"
WEIGHTS="${WEIGHTS:-0.001 0.005 0.01 0.02 0.05}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/grid_search_degree_topo}"

mkdir -p "$RESULTS_DIR"
RESULTS_CSV="$RESULTS_DIR/results.csv"
echo "weight,epochs,dice,precision,recall,checkpoint" > "$RESULTS_CSV"

cd "$PROJECT_ROOT"

for weight in $WEIGHTS; do
  run_id="deg_${weight//./p}"
  checkpoint_dir="$RESULTS_DIR/$run_id/checkpoints"
  mkdir -p "$checkpoint_dir"

  echo "=== Running weight=$weight ==="
  WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" train.py \
    --epochs "$EPOCHS" \
    --batch-size 8 \
    --learning-rate 0.001 \
    --classes 1 \
    --scale 1.0 \
    --validation 10 \
    --num-workers 0 \
    --seed 0 \
    --crop-size 256 \
    --images-dir "$IMAGES_DIR" \
    --masks-dir "$MASKS_DIR" \
    --checkpoint-dir "$checkpoint_dir" \
    --disable-cp-topo-loss \
    --degree-topo-weight "$weight" \
    --degree-skeleton-iters 20 \
    --degree-hist-sigma 0.5

  metrics_json="$RESULTS_DIR/$run_id/metrics.json"
  "$PYTHON_BIN" evaluate_checkpoint.py \
    --checkpoint "$checkpoint_dir/checkpoint_epoch${EPOCHS}.pth" \
    --images-dir "$TEST_IMAGES_DIR" \
    --masks-dir "$TEST_MASKS_DIR" \
    --scale 1.0 \
    --classes 1 > "$metrics_json"

  python3 - <<PY
import json
from pathlib import Path
metrics = json.loads(Path("$metrics_json").read_text())
with open("$RESULTS_CSV", "a", encoding="utf-8") as f:
    f.write(f"{float('$weight')},{int('$EPOCHS')},{metrics['dice']},{metrics['precision']},{metrics['recall']},{metrics['checkpoint']}\n")
print(metrics)
PY
done

echo "Saved results to $RESULTS_CSV"
