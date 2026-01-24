#!/bin/bash
# run_hcpex_analysis.sh
# 
# Specifically runs QC-FC and ML pipelines for the HCPex atlas,
# applying the dimension alignment fix (373 ROIs).

set -e

source venv/bin/activate
export PYTHONPATH=.

echo "============================================================"
echo "Starting HCPex-Specific Analysis"
echo "Date: $(date)"
echo "============================================================"

# 1. Run QC-FC for HCPex
echo ""
echo "------------------------------------------------------------"
echo "1. Running QC-FC Analysis (HCPex)"
echo "Config: configs/qc_fc_hcpex_only.yaml"
echo "------------------------------------------------------------"
# This script uses the fix to align standard/AROMA strategies to 373 ROIs
python -m benchmarking.qc_fc --config configs/qc_fc_hcpex_only.yaml

# 2. Run ML Pipelines for HCPex
echo ""
echo "------------------------------------------------------------"
echo "2. Running ML Pipelines (HCPex)"
echo "Config: configs/ml_hcpex_only.yaml"
echo "------------------------------------------------------------"
# The ML pipeline now automatically detects HCPex and aligns ROIs
python -m benchmarking.run_ml_pipelines --config configs/ml_hcpex_only.yaml

echo ""
echo "============================================================"
echo "HCPex Analysis Complete!"
echo "QC-FC Results: results/qcfc/qc_fc_hcpex_fixed.csv"
echo "ML Results: results/pipelines/HCPex_strategy-*"
echo "============================================================"
