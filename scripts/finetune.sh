#!/usr/bin/env bash
# Run LoRA fine-tuning, optionally merge adapters into standalone weights.
# This script assumes you've already run the initial pipeline steps:
#   1) Crawl  2) Process  3) Index  4) Prepare dataset
#
# Usage:
#   ./scripts/finetune.sh
#   ./scripts/finetune.sh --merge
#   ./scripts/finetune.sh --debug
#   ./scripts/finetune.sh --help
#
# Logging:
# - Default log level is INFO.
# - Pass --debug to enable DEBUG logs across Python steps (sets LOG_LEVEL=DEBUG).
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
DEBUG=0
MERGE=0

print_help() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs:
  - LoRA fine-tuning with MLX
  - Optional: Merge LoRA into standalone weights (with -m|--merge)

Pre-reqs:
  - Completed initial pipeline (crawl, process, index, prepare-dataset)
  - configs/config.yaml set appropriately

Options:
  -m, --merge    Merge adapters into standalone weights after finetune
  -d, --debug    Enable DEBUG logging (exports LOG_LEVEL=DEBUG)
  -h, --help     Show this help

Notes:
  - Uses 'python -m src.training.finetune_mlx' for finetune
  - Uses 'python -m src.training.finetune_mlx --merge-only' for merge
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      print_help
      exit 0
      ;;
    -d | --debug)
      DEBUG=1
      shift
      ;;
    -m | --merge)
      MERGE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Try --help for usage."
      exit 1
      ;;
  esac
done

log() {
  echo "[finetune] $*"
}

# Logging level (propagate to Python)
if [[ $DEBUG -eq 1 ]]; then
  export LOG_LEVEL=DEBUG
  log "Debug logging enabled (LOG_LEVEL=DEBUG)"
fi

# Basic environment info
log "Repository root: $REPO_ROOT"
log "Python: $("$PYTHON_BIN" --version 2>&1 | tr -d '\n')"
if [[ -z ${VIRTUAL_ENV:-} ]]; then
  log "WARNING: No active virtual environment detected (VIRTUAL_ENV is empty)."
  log "Proceeding anyway. It's recommended to 'source .venv/bin/activate' first."
else
  log "Using virtual environment at: $VIRTUAL_ENV"
fi

# Step: Finetune (LoRA, MLX)
TOTAL=$((1 + MERGE))
STEP=1
log "Step $STEP/$TOTAL: Starting LoRA finetuning (MLX)"
"$PYTHON_BIN" -m src.training.finetune_mlx
log "Step $STEP/$TOTAL: Finetune complete."

if [[ $MERGE -eq 1 ]]; then
  STEP=$((STEP + 1))
  # Step: Merge adapters into standalone weights
  log "Step $STEP/$TOTAL: Merging LoRA adapter into standalone weights"
  "$PYTHON_BIN" -m src.training.finetune_mlx --merge-only
  log "Step $STEP/$TOTAL: Merge complete."
else
  log "Merge step skipped (use --merge to enable)."
fi

log "Outputs:"
log "- LoRA adapters: models/adapters/ (as configured in configs/config.yaml)"
if [[ $MERGE -eq 1 ]]; then
  log "- Merged weights: models/merges/ (as configured in configs/config.yaml)"
else
  log "- Merged weights: (skipped; run with --merge to produce)"
fi

exit 0
