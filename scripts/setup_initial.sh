#!/usr/bin/env bash
# Setup the initial pipeline steps for MDE.Learn.Chatbot
# Runs: 1) Crawl  2) Process  3) Index  4) Prepare dataset
# Usage:
#   ./scripts/setup_initial.sh
#   ./scripts/setup_initial.sh --debug
#   ./scripts/setup_initial.sh --help
#
# Logging:
# - Default log level is INFO.
# - Pass --debug to enable DEBUG logs across Python steps (sets LOG_LEVEL=DEBUG).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WITH_DATASET=1
PYTHON_BIN="${PYTHON:-python}"
DEBUG=0

print_help() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs the initial pipeline steps as documented in README:
  1. Crawl docs
  2. Process HTML to chunks
  3. Build Chroma index
  4. Prepare dataset JSONL for finetuning

Options:
  -d, --debug     Enable DEBUG logging (exports LOG_LEVEL=DEBUG)
  -h, --help      Show this help

Notes:
- Ensure your virtual environment is activated before running:
    python -m venv .venv && source .venv/bin/activate
    pip install -U pip setuptools wheel
    pip install -r requirements.txt
- Configuration is read from configs/config.yaml.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    -d|--debug)
      DEBUG=1
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
  echo "[setup_initial] $*"
}

# Logging level (propagate to Python)
if [[ "$DEBUG" -eq 1 ]]; then
  export LOG_LEVEL=DEBUG
  log "Debug logging enabled (LOG_LEVEL=DEBUG)"
fi

# Basic environment info
log "Repository root: $REPO_ROOT"
log "Python: $("$PYTHON_BIN" --version 2>&1 | tr -d '\n')"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  log "WARNING: No active virtual environment detected (VIRTUAL_ENV is empty)."
  log "Proceeding anyway. It's recommended to 'source .venv/bin/activate' first."
else
  log "Using virtual environment at: $VIRTUAL_ENV"
fi

# Step 1: Crawl
log "Step 1/4: Crawl MDE docs (robots-aware, filtered path)"
"$PYTHON_BIN" -m src.crawler.crawler

# Step 2: Process
log "Step 2/4: Process HTML to clean text and chunks"
"$PYTHON_BIN" -m src.processing.process

# Step 3: Index
log "Step 3/4: Build Chroma index with Sentence-Transformers embeddings"
"$PYTHON_BIN" -m src.indexing.build_index

# Step 4: Prepare dataset
log "Step 4/4: Prepare fine-tune dataset (JSONL {\"text\": ...})"
"$PYTHON_BIN" -m src.training.prepare_dataset

log "Initial setup complete."
log "Artifacts:"
log "- Raw HTML:        data/raw/html/ (ignored by git)"
log "- Processed:       data/processed/ (ignored)"
log "- Chroma index:    data/index/chroma/ (ignored)"
log "- Datasets:        data/datasets/ (ignored)"

exit 0
