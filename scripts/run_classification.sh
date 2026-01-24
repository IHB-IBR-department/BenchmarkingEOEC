#!/usr/bin/env bash
set -euo pipefail

# Run from repo root. Activate the venv before calling this script:
#   source venv/bin/activate
#
# Examples:
#   bash run_classification.sh
#   bash run_classification.sh --with-permutation

DATA_ROOT="${DATA_ROOT:-$HOME/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data}"
ATLAS="${ATLAS:-Schaefer200}"
STRATEGY="${STRATEGY:-1}"
GSR="${GSR:-GSR}"
GLASSO_DIR="${GLASSO_DIR:-$DATA_ROOT/glasso_precomputed_fc}"

run_permutation=false
if [[ "${1:-}" == "--with-permutation" ]]; then
  run_permutation=true
fi

PYTHONPATH=. python -m benchmarking.ml.pipeline \
  --atlas "$ATLAS" --strategy "$STRATEGY" --gsr "$GSR" \
  --data-path "$DATA_ROOT" \
  --precomputed-glasso "$GLASSO_DIR"

if [[ "$run_permutation" == "true" ]]; then
  PYTHONPATH=. python -m benchmarking.ml.pipeline \
    --atlas "$ATLAS" --strategy "$STRATEGY" --gsr "$GSR" \
    --data-path "$DATA_ROOT" \
    --skip-glasso \
    --n-permutations 100
fi
