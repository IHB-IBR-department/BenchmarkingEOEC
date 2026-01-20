#!/usr/bin/env bash
set -euo pipefail

# Run from repo root. Activate the venv before calling this script:
#   source venv/bin/activate
#
# Examples:
#   bash run_classification.sh
#   bash run_classification.sh --with-permutation

run_permutation=false
if [[ "${1:-}" == "--with-permutation" ]]; then
  run_permutation=true
fi

PYTHONPATH=. python -m benchmarking.cross_site --config configs/cross_site_quick_classification.yaml
PYTHONPATH=. python -m benchmarking.few_shot --config configs/few_shot_quick_classification.yaml

if [[ "$run_permutation" == "true" ]]; then
  PYTHONPATH=. python -m benchmarking.cross_site --config configs/cross_site_corr_schaefer_perm.yaml
fi
