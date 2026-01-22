#!/usr/bin/env python3
"""
Preprocess HCPex time series data for consistent dimensions across sites.

HCPex atlas has varying dimensions (421-423 ROIs) across sites because nilearn
filters out some ROIs during extraction. This module provides functions to:
1. Adjust the 426-dimensional mask to match actual data dimensions
2. Apply the adjusted mask to get consistent ROI counts across sites
3. Ensure both IHB and China data have the same ROI subset for FC computation

Usage:
    from benchmarking.hcpex_preprocess import preprocess_hcpex_timeseries

    # Load raw time series (varying dimensions)
    ihb_data = np.load("ihb_close_HCPex_strategy-1_GSR.npy")  # (84, 120, 423)
    china_data = np.load("china_close_HCPex_strategy-1_GSR.npy")  # (48, 240, 421, 2)

    # Preprocess to consistent dimensions
    ihb_masked = preprocess_hcpex_timeseries(ihb_data, site="ihb", mask_path="coverage/hcp_mask.npy")
    china_masked = preprocess_hcpex_timeseries(china_data, site="china", mask_path="coverage/hcp_mask.npy")

    # Now both have same ROI dimension
    print(ihb_masked.shape)    # (84, 120, 373)
    print(china_masked.shape)  # (48, 240, 373, 2)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


def load_skipped_rois(skipped_file: Path) -> list[int]:
    """Load skipped ROI indices from text file.

    Args:
        skipped_file: Path to text file with comma-separated ROI indices

    Returns:
        Sorted list of skipped ROI indices
    """
    if not skipped_file.exists():
        raise FileNotFoundError(f"Skipped ROIs file not found: {skipped_file}")

    with open(skipped_file) as f:
        content = f.read().strip()
        indices = [int(x.strip()) for x in content.split(',') if x.strip()]

    return sorted(indices)


def adjust_mask_for_site(
    full_mask: np.ndarray,
    skipped_indices: list[int],
) -> np.ndarray:
    """Adjust 426-dimensional mask to match actual data dimensions.

    The full HCPex atlas has 426 ROIs, but nilearn removes some ROIs during
    extraction. This function removes the skipped ROI indices from the mask
    to create a mask that matches the actual data dimensions.

    Args:
        full_mask: Boolean mask of shape (426,) where True = bad ROI
        skipped_indices: List of ROI indices that were skipped by nilearn

    Returns:
        Adjusted mask with skipped indices removed, shape (426 - len(skipped_indices),)
    """
    if full_mask.shape[0] != 426:
        raise ValueError(f"Expected full_mask shape (426,), got {full_mask.shape}")

    # Create mask for indices to keep (not skipped)
    keep_mask = np.ones(426, dtype=bool)
    keep_mask[skipped_indices] = False

    # Remove skipped indices from the mask
    adjusted_mask = full_mask[keep_mask]

    return adjusted_mask


def apply_roi_mask(
    timeseries: np.ndarray,
    roi_mask: np.ndarray,
) -> np.ndarray:
    """Apply ROI mask to time series data.

    Args:
        timeseries: Time series array with ROIs in the last axis before sessions
            - 3D: (n_subjects, n_timepoints, n_rois)
            - 4D: (n_subjects, n_timepoints, n_rois, n_sessions)
        roi_mask: Boolean mask where True = bad ROI (to exclude)

    Returns:
        Masked time series with bad ROIs removed
    """
    n_rois_data = timeseries.shape[2]
    n_rois_mask = roi_mask.shape[0]

    if n_rois_data != n_rois_mask:
        raise ValueError(
            f"ROI dimension mismatch: data has {n_rois_data} ROIs, "
            f"mask has {n_rois_mask} elements"
        )

    # Keep only good ROIs (where mask is False)
    good_rois = ~roi_mask

    if timeseries.ndim == 3:
        # (n_subjects, n_timepoints, n_rois)
        return timeseries[:, :, good_rois]
    elif timeseries.ndim == 4:
        # (n_subjects, n_timepoints, n_rois, n_sessions)
        return timeseries[:, :, good_rois, :]
    else:
        raise ValueError(f"Expected 3D or 4D array, got {timeseries.ndim}D")


def preprocess_hcpex_timeseries(
    timeseries: np.ndarray,
    site: Literal["ihb", "china"],
    mask_path: str | Path,
    coverage_dir: str | Path | None = None,
) -> np.ndarray:
    """Preprocess HCPex time series to consistent ROI dimensions.

    This is the main function to use. It automatically detects whether the data
    is from AROMA (426 ROIs) or standard strategies (423/421 ROIs) and applies
    the appropriate masking.

    Args:
        timeseries: Raw time series data
            - Standard IHB: (n_subjects, n_timepoints, 423)
            - Standard China: (n_subjects, n_timepoints, 421, 2)
            - AROMA IHB/China: (n_subjects, n_timepoints, 426) or (n_subjects, n_timepoints, 426, 2)
        site: Site identifier ("ihb" or "china")
        mask_path: Path to hcp_mask.npy (426-dimensional mask)
        coverage_dir: Directory containing skipped ROIs files (optional)
            If None, infers from mask_path parent directory

    Returns:
        Masked time series with consistent ROI dimensions (373 ROIs)
            - IHB: (n_subjects, n_timepoints, 373)
            - China: (n_subjects, n_timepoints, 373, 2)
    """
    mask_path = Path(mask_path)

    # Load full 426-dimensional mask
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    full_mask = np.load(mask_path)

    if full_mask.shape[0] != 426:
        raise ValueError(f"Expected mask shape (426,), got {full_mask.shape}")

    actual_n_rois = timeseries.shape[2]

    # Check if this is AROMA data (already has 426 ROIs)
    if actual_n_rois == 426:
        # AROMA strategies: apply mask directly without adjustment
        masked_timeseries = apply_roi_mask(timeseries, full_mask)
        return masked_timeseries

    # Standard strategies: adjust mask for site-specific skipped ROIs
    # Determine coverage directory
    if coverage_dir is None:
        coverage_dir = mask_path.parent
    else:
        coverage_dir = Path(coverage_dir)

    # Load site-specific skipped ROIs
    skipped_file = coverage_dir / f"{site}_skipped_rois_HCPex.txt"
    skipped_indices = load_skipped_rois(skipped_file)

    # Adjust mask to match data dimensions
    adjusted_mask = adjust_mask_for_site(full_mask, skipped_indices)

    # Verify dimensions match
    expected_n_rois = 426 - len(skipped_indices)
    if actual_n_rois != expected_n_rois:
        raise ValueError(
            f"Dimension mismatch for {site}: "
            f"expected {expected_n_rois} ROIs (426 - {len(skipped_indices)} skipped), "
            f"got {actual_n_rois} ROIs in data"
        )

    # Apply mask
    masked_timeseries = apply_roi_mask(timeseries, adjusted_mask)

    return masked_timeseries


def verify_consistent_dimensions(
    ihb_data: np.ndarray,
    china_data: np.ndarray,
) -> tuple[bool, str]:
    """Verify that IHB and China data have consistent ROI dimensions.

    Args:
        ihb_data: IHB time series (n_subjects, n_timepoints, n_rois)
        china_data: China time series (n_subjects, n_timepoints, n_rois, n_sessions)

    Returns:
        Tuple of (success, message)
    """
    ihb_n_rois = ihb_data.shape[2]
    china_n_rois = china_data.shape[2]

    if ihb_n_rois == china_n_rois:
        return True, f"✓ Consistent dimensions: both have {ihb_n_rois} ROIs"
    else:
        return False, f"✗ Inconsistent dimensions: IHB has {ihb_n_rois} ROIs, China has {china_n_rois} ROIs"


if __name__ == "__main__":
    # Test script
    import sys

    print("=" * 70)
    print("Testing HCPex preprocessing")
    print("=" * 70)

    data_root = Path.home() / "Yandex.Disk.localized/IHB/OpenCloseBenchmark_data"

    # Load raw data
    print("\n1. Loading raw time series...")
    ihb_file = data_root / "timeseries_ihb/HCPex/ihb_close_HCPex_strategy-1_GSR.npy"
    china_file = data_root / "timeseries_china/HCPex/china_close_HCPex_strategy-1_GSR.npy"

    ihb_raw = np.load(ihb_file)
    china_raw = np.load(china_file)

    print(f"   IHB raw shape:   {ihb_raw.shape}")
    print(f"   China raw shape: {china_raw.shape}")

    # Preprocess
    print("\n2. Preprocessing with HCPex mask...")
    mask_path = data_root / "coverage/hcp_mask.npy"

    try:
        ihb_masked = preprocess_hcpex_timeseries(ihb_raw, site="ihb", mask_path=mask_path)
        china_masked = preprocess_hcpex_timeseries(china_raw, site="china", mask_path=mask_path)

        print(f"   IHB masked shape:   {ihb_masked.shape}")
        print(f"   China masked shape: {china_masked.shape}")

        # Verify consistency
        print("\n3. Verifying consistency...")
        success, message = verify_consistent_dimensions(ihb_masked, china_masked)
        print(f"   {message}")

        if success:
            print("\n" + "=" * 70)
            print("SUCCESS: HCPex preprocessing working correctly!")
            print("=" * 70)
            sys.exit(0)
        else:
            print("\n" + "=" * 70)
            print("ERROR: Dimensions are not consistent!")
            print("=" * 70)
            sys.exit(1)

    except Exception as e:
        print(f"\n   ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
