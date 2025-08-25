#!/usr/bin/env bash
# Clean / Reset generated artifacts for MDE.Learn.Chatbot
#
# Usage:
#   ./scripts/clean_reset.sh                # interactive confirm (INFO logs)
#   ./scripts/clean_reset.sh --yes          # no prompt
#   ./scripts/clean_reset.sh --dry-run      # show what would be removed
#   ./scripts/clean_reset.sh --debug        # verbose script output
#   ./scripts/clean_reset.sh --help
#
# Removes:
#   - data/raw
#   - data/processed
#   - data/index/chroma
#   - data/datasets
#   - models/adapters
#   - models/merges
#   - outputs

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

YES=0
DRY_RUN=0
DEBUG=0

print_help() {
  cat << 'EOF'
Clean / Reset generated artifacts.

Options:
  -y, --yes       Proceed without confirmation prompt
  -n, --dry-run   Show what would be removed without deleting
  -d, --debug     Verbose script output
  -h, --help      Show this help

Targets:
  data/raw
  data/processed
  data/index/chroma
  data/datasets
  models/adapters
  models/merges
  outputs
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y | --yes)
      YES=1
      shift
      ;;
    -n | --dry-run)
      DRY_RUN=1
      shift
      ;;
    -d | --debug)
      DEBUG=1
      shift
      ;;
    -h | --help)
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

if [[ "$DEBUG" -eq 1 ]]; then
  set -x
fi

log() { echo "[clean_reset] $*"; }

TARGETS=(
  "data/raw"
  "data/processed"
  "data/index/chroma"
  "data/datasets"
  "models/adapters"
  "models/merges"
  "outputs"
)

log "Repository root: $REPO_ROOT"
log "Targets to remove:"
for t in "${TARGETS[@]}"; do
  echo "  - $t"
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "Dry run enabled. No changes will be made."
fi

if [[ "$YES" -ne 1 ]]; then
  echo
  read -r -p "Type 'yes' to confirm deletion: " REPLY
  if [[ "$REPLY" != "yes" ]]; then
    log "Aborted."
    exit 0
  fi
fi

EXIT_CODE=0
for t in "${TARGETS[@]}"; do
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "Would remove: $t"
  else
    if [[ -e "$t" ]]; then
      log "Removing: $t"
      rm -rf "$t" || EXIT_CODE=$?
    else
      log "Skip (not found): $t"
    fi
  fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "Dry run complete."
else
  if [[ "$EXIT_CODE" -eq 0 ]]; then
    log "Cleanup complete."
  else
    log "Cleanup finished with errors (exit=$EXIT_CODE)."
  fi
fi

exit "$EXIT_CODE"
