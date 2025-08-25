#!/usr/bin/env bash
# Run LoRA fine-tuning and then merge adapters into standalone weights.
# This script assumes you've already run the initial pipeline steps:
#   1) Crawl  2) Process  3) Index  4) Prepare dataset
#
# Usage:
#   ./scripts/finetune_and_merge.sh
#   ./scripts/finetune_and_merge.sh --help
#
# Notes:
# - Activate your virtual environment first:
#     python -m venv .venv && source .venv/bin/activate
#     pip install -U pip setuptools wheel
#     pip install -r requirements.txt
# - Configuration is read from configs/config.yaml.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"

print_help() {
  cat <<EOF
Usage: $(basename "$0")

Runs:
  - LoRA fine-tuning with MLX
  - Merge LoRA into standalone weights

Pre-reqs:
  - Completed initial pipeline (crawl, process, index, prepare-dataset)
  - configs/config.yaml set appropriately

Options:
  -h, --help    Show this help

Notes:
  - Uses 'python -m src.training.finetune_mlx' for finetune
  - Uses 'python -m src.training.finetune_mlx --merge-only' for merge
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Try --help for usage."
      exit 1
      ;;
  esac
done

log() {
  echo "[finetune_and_merge] $*"
}

# Basic environment info
log "Repository root: $REPO_ROOT"
log "Python: $("$PYTHON_BIN" --version 2>&1 | tr -d '\n')"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  log "WARNING: No active virtual environment detected (VIRTUAL_ENV is empty)."
  log "Proceeding anyway. It's recommended to 'source .venv/bin/activate' first."
else
  log "Using virtual environment at: $VIRTUAL_ENV"
fi

# Step: Finetune (LoRA, MLX)
log "Step 1/2: Starting LoRA finetuning (MLX)"
"$PYTHON_BIN" -m src.training.finetune_mlx
log "Step 1/2: Finetune complete."

# Step: Merge adapters into standalone weights
log "Step 2/2: Merging LoRA adapter into standalone weights"
"$PYTHON_BIN" -m src.training.finetune_mlx --merge-only
log "Step 2/2: Merge complete."

log "Outputs:"
log "- LoRA adapters: models/adapters/ (as configured in configs/config.yaml)"
log "- Merged weights: models/merges/ (as configured in configs/config.yaml)"

exit 0
