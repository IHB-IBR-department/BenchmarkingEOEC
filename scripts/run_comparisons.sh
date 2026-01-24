#!/usr/bin/env bash
set -euo pipefail

# Run from repo root. Activate the venv before calling this script:
#   source venv/bin/activate
#
# Examples:
#   bash run_comparisons.sh
#   bash run_comparisons.sh --compare P0001 P0003
#   bash run_comparisons.sh --factor gsr GSR noGSR
#   bash run_comparisons.sh --no-factor --compare P0001 P0003
#   bash run_comparisons.sh --direction china2ihb
#   bash run_comparisons.sh --pipeline-dir results/pipelines/Schaefer200_strategy-1_GSR

DIRECTION="ihb2china"
PIPELINE_DIR="results/pipelines/Schaefer200_strategy-1_GSR"

run_factor=true
factor_name="gsr"
factor_level_a="GSR"
factor_level_b="noGSR"

run_compare=false
pipe_a=""
pipe_b=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-factor)
      run_factor=false
      shift
      ;;
    --direction)
      DIRECTION="${2:-}"
      if [[ -z "$DIRECTION" ]]; then
        echo "Usage: bash run_comparisons.sh --direction <ihb2china|china2ihb>" >&2
        exit 1
      fi
      shift 2
      ;;
    --pipeline-dir)
      PIPELINE_DIR="${2:-}"
      if [[ -z "$PIPELINE_DIR" ]]; then
        echo "Usage: bash run_comparisons.sh --pipeline-dir <results/pipelines/...>" >&2
        exit 1
      fi
      shift 2
      ;;
    --factor)
      factor_name="${2:-}"
      factor_level_a="${3:-}"
      factor_level_b="${4:-}"
      if [[ -z "$factor_name" || -z "$factor_level_a" || -z "$factor_level_b" ]]; then
        echo "Usage: bash run_comparisons.sh --factor <name> <level_a> <level_b>" >&2
        exit 1
      fi
      run_factor=true
      shift 4
      ;;
    --compare)
      pipe_a="${2:-}"
      pipe_b="${3:-}"
      if [[ -z "$pipe_a" || -z "$pipe_b" ]]; then
        echo "Usage: bash run_comparisons.sh --compare P0001 P0002" >&2
        exit 1
      fi
      run_compare=true
      shift 3
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

TEST_OUTPUTS="${PIPELINE_DIR}/cross_site_${DIRECTION}_test_outputs.csv"
ABBREV="${ABBREV:-${PIPELINE_DIR}/pipeline_abbreviations.csv}"

if [[ "$run_factor" == "true" ]]; then
  PYTHONPATH=. python -m benchmarking.ml.pipeline_comparisons factor \
    --test-outputs "$TEST_OUTPUTS" \
    --factor "$factor_name" --level-a "$factor_level_a" --level-b "$factor_level_b" \
    --n-permutations 100 --n-bootstrap 100
fi

if [[ "$run_compare" == "true" ]]; then
  PYTHONPATH=. python -m benchmarking.ml.pipeline_comparisons compare \
    --test-outputs "$TEST_OUTPUTS" \
    --abbrev "$ABBREV" \
    --pipeline-a "$pipe_a" --pipeline-b "$pipe_b"
fi
