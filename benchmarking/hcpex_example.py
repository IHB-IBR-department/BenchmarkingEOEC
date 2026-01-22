#!/usr/bin/env python3
"""
Example usage of HCPex preprocessing for consistent FC computation.

This demonstrates how to preprocess HCPex time series data from both sites
to ensure consistent ROI dimensions before functional connectivity estimation.
"""

import numpy as np
from pathlib import Path

from benchmarking.hcpex_preprocess import preprocess_hcpex_timeseries


def main():
    # Setup paths
    data_root = Path.home() / "Yandex.Disk.localized/IHB/OpenCloseBenchmark_data"
    mask_path = data_root / "coverage/hcp_mask.npy"

    # Example: Load and preprocess HCPex data for a specific pipeline
    strategy = "1"
    gsr = "GSR"

    print(f"Processing HCPex data for strategy-{strategy}_{gsr}")
    print("=" * 70)

    # Load IHB data
    ihb_close_file = data_root / f"timeseries_ihb/HCPex/ihb_close_HCPex_strategy-{strategy}_{gsr}.npy"
    ihb_open_file = data_root / f"timeseries_ihb/HCPex/ihb_open_HCPex_strategy-{strategy}_{gsr}.npy"

    ihb_close_raw = np.load(ihb_close_file)
    ihb_open_raw = np.load(ihb_open_file)

    print(f"\nIHB raw shapes:")
    print(f"  Close: {ihb_close_raw.shape}")
    print(f"  Open:  {ihb_open_raw.shape}")

    # Load China data
    china_close_file = data_root / f"timeseries_china/HCPex/china_close_HCPex_strategy-{strategy}_{gsr}.npy"
    china_open_file = data_root / f"timeseries_china/HCPex/china_open_HCPex_strategy-{strategy}_{gsr}.npy"

    china_close_raw = np.load(china_close_file)
    china_open_raw = np.load(china_open_file)

    print(f"\nChina raw shapes:")
    print(f"  Close: {china_close_raw.shape}")
    print(f"  Open:  {china_open_raw.shape}")

    # Preprocess to consistent dimensions
    print("\n" + "=" * 70)
    print("Preprocessing to consistent dimensions...")
    print("=" * 70)

    ihb_close = preprocess_hcpex_timeseries(ihb_close_raw, site="ihb", mask_path=mask_path)
    ihb_open = preprocess_hcpex_timeseries(ihb_open_raw, site="ihb", mask_path=mask_path)

    china_close = preprocess_hcpex_timeseries(china_close_raw, site="china", mask_path=mask_path)
    china_open = preprocess_hcpex_timeseries(china_open_raw, site="china", mask_path=mask_path)

    print(f"\nIHB preprocessed shapes:")
    print(f"  Close: {ihb_close.shape}")
    print(f"  Open:  {ihb_open.shape}")

    print(f"\nChina preprocessed shapes:")
    print(f"  Close: {china_close.shape}")
    print(f"  Open:  {china_open.shape}")

    # Verify consistent ROI count
    print("\n" + "=" * 70)
    print("Verification:")
    print("=" * 70)
    print(f"✓ IHB ROI count:   {ihb_close.shape[2]}")
    print(f"✓ China ROI count: {china_close.shape[2]}")
    print(f"✓ Match: {ihb_close.shape[2] == china_close.shape[2]}")

    print("\nNow you can compute FC matrices with consistent dimensions!")
    print("\nExample (using nilearn):")
    print("  from nilearn.connectome import ConnectivityMeasure")
    print("  conn = ConnectivityMeasure(kind='correlation')")
    print("  ihb_fc = conn.fit_transform(ihb_close)  # shape: (84, 373, 373)")
    print("  china_fc = conn.fit_transform(china_close[:,:,:,0])  # shape: (48, 373, 373)")


if __name__ == "__main__":
    main()
