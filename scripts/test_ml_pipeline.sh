#!/bin/bash
set -e
source venv/bin/activate
export PYTHONPATH=.

echo "Running with config file: configs/ml_single.yaml"
python -m benchmarking.ml.pipeline --config configs/ml_single.yaml
