#!/bin/bash
# run_full_analysis.sh
#
# Runs full ICC and QC-FC analyses for all atlases and sites.
#
# Usage:
#   bash run_full_analysis.sh

set -e  # Exit on error

# Activate environment
source venv/bin/activate
export PYTHONPATH=.

echo "============================================================"
echo "Starting Full Benchmarking Analysis"
echo "Date: $(date)"
echo "============================================================"

# 1. Run ICC Analysis
echo ""
echo "------------------------------------------------------------"
echo "1. Running ICC Analysis (China Test-Retest)"
echo "Config: configs/icc_full.yaml"
echo "------------------------------------------------------------"
python -m benchmarking.icc --config configs/icc_full.yaml

# 2. Run QC-FC Analysis
echo ""
echo "------------------------------------------------------------"
echo "2. Running QC-FC Analysis (Motion-FC Correlations)"
echo "Config: configs/qc_fc_full.yaml"
echo "------------------------------------------------------------"
python -m benchmarking.qc_fc --config configs/qc_fc_full.yaml

echo ""
echo "============================================================"
echo "Analysis Complete!"
echo "ICC Results: results/icc_results/icc_summary_full.csv"
echo "QC-FC Results: results/qcfc/qc_fc_full.csv"
echo "============================================================"
