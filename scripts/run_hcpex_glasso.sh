#!/bin/bash
# Precompute glasso for all HCPex time series (IHB and China)
#
# Usage:
#   bash run_hcpex_glasso.sh
#
# This will process all HCPex time series with automatic HCPex preprocessing

set -e

DATA_ROOT="${DATA_ROOT:-$HOME/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/glasso_precomputed_fc}"

echo "=================================================="
echo "Precomputing glasso for HCPex atlas"
echo "=================================================="
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Activate virtual environment
source venv/bin/activate

# Process IHB HCPex data
echo "Processing IHB HCPex data..."
PYTHONPATH=. python -m data_utils.preprocessing.precompute_glasso \
  --input "$DATA_ROOT/timeseries_ihb/HCPex" \
  --output-dir "$OUTPUT_DIR" \
  --print-timing

echo ""
echo "Processing China HCPex data..."
PYTHONPATH=. python -m data_utils.preprocessing.precompute_glasso \
  --input "$DATA_ROOT/timeseries_china/HCPex" \
  --output-dir "$OUTPUT_DIR" \
  --print-timing

echo ""
echo "=================================================="
echo "HCPex glasso precomputation complete!"
echo "=================================================="
echo "Output locations:"
echo "  $OUTPUT_DIR/ihb/HCPex/"
echo "  $OUTPUT_DIR/china/HCPex/"
