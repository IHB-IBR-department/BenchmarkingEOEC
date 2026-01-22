#!/usr/bin/env python3
"""
Create HCPex atlas mask for consistent ROI dimensions across sites.

HCPex atlas has varying dimensions (421-426 ROIs) across sites because nilearn
filters out ROIs that don't pass quality checks. This script creates a unified
mask that identifies ROIs to exclude based on:

1. Low coverage (< threshold) in IHB data
2. ROIs explicitly skipped by nilearn for IHB (ihb_skipped_rois_HCPex.txt)
3. ROIs explicitly skipped by nilearn for China (china_skipped_rois_HCPex.txt)

The mask is saved as hcp_mask.npy in the coverage folder and can be used to
ensure consistent dimensions for functional connectivity matrices across sites.

Usage:
    python -m benchmarking.create_hcpex_mask --threshold 0.1

The mask is a boolean array where True indicates a "bad" ROI to exclude.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmarking.project import resolve_data_root


def load_coverage(coverage_dir: Path, threshold: float = 0.1) -> np.ndarray:
    """Load IHB HCPex parcel coverage and identify low-coverage ROIs.

    Args:
        coverage_dir: Path to coverage folder
        threshold: Coverage threshold (ROIs below this are marked bad)

    Returns:
        Boolean mask where True = bad ROI (low coverage)
    """
    coverage_file = coverage_dir / "ihb_HCPex_parcel_coverage.npy"
    if not coverage_file.exists():
        raise FileNotFoundError(f"Coverage file not found: {coverage_file}")

    coverage = np.load(coverage_file)
    low_coverage_mask = coverage < threshold

    n_low = low_coverage_mask.sum()
    print(f"Low coverage ROIs (< {threshold}): {n_low} / {len(coverage)}")

    return low_coverage_mask


def load_skipped_rois(skipped_file: Path) -> set[int]:
    """Load ROI indices from skipped ROIs text file.

    Supports both formats:
    - One ROI per line
    - Comma-separated ROIs on one or multiple lines

    Args:
        skipped_file: Path to text file with ROI indices

    Returns:
        Set of ROI indices (0-based) that were skipped
    """
    if not skipped_file.exists():
        print(f"Warning: Skipped ROIs file not found: {skipped_file}")
        return set()

    skipped = set()
    with open(skipped_file) as f:
        content = f.read().strip()
        if not content or content.startswith("#"):
            return skipped

        # Split by commas and newlines to handle both formats
        tokens = content.replace('\n', ',').split(',')
        for token in tokens:
            token = token.strip()
            if token and not token.startswith("#"):
                try:
                    roi_idx = int(token)
                    skipped.add(roi_idx)
                except ValueError:
                    print(f"Warning: Invalid ROI index in {skipped_file}: {token}")

    print(f"Loaded {len(skipped)} skipped ROIs from {skipped_file.name}: {sorted(skipped)}")
    return skipped


def create_combined_mask(
    coverage_mask: np.ndarray,
    ihb_skipped: set[int],
    china_skipped: set[int],
) -> np.ndarray:
    """Combine coverage-based and explicitly skipped ROIs into final mask.

    Args:
        coverage_mask: Boolean mask from coverage thresholding
        ihb_skipped: Set of ROI indices skipped for IHB
        china_skipped: Set of ROI indices skipped for China

    Returns:
        Combined boolean mask where True = bad ROI
    """
    final_mask = coverage_mask.copy()
    n_rois = len(final_mask)

    # Mark IHB skipped ROIs
    for roi_idx in ihb_skipped:
        if 0 <= roi_idx < n_rois:
            final_mask[roi_idx] = True
        else:
            print(f"Warning: IHB skipped ROI {roi_idx} out of range [0, {n_rois})")

    # Mark China skipped ROIs
    for roi_idx in china_skipped:
        if 0 <= roi_idx < n_rois:
            final_mask[roi_idx] = True
        else:
            print(f"Warning: China skipped ROI {roi_idx} out of range [0, {n_rois})")

    return final_mask


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    """Save mask as numpy array.

    Args:
        mask: Boolean mask array
        output_path: Path to save mask (including filename)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, mask.astype(bool))
    print(f"\nSaved mask to: {output_path}")
    print(f"Mask shape: {mask.shape}")
    print(f"Bad ROIs: {mask.sum()} / {len(mask)} ({100 * mask.mean():.2f}%)")
    print(f"Good ROIs: {(~mask).sum()} / {len(mask)} ({100 * (~mask).mean():.2f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create HCPex mask for consistent ROI dimensions across sites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default threshold (0.1) and auto-detect data root
    python -m benchmarking.create_hcpex_mask

    # Custom threshold
    python -m benchmarking.create_hcpex_mask --threshold 0.15

    # Specify data root explicitly
    python -m benchmarking.create_hcpex_mask --data-root /path/to/data
        """,
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to OpenCloseBenchmark_data (optional if OPEN_CLOSE_BENCHMARK_DATA is set)",
    )
    parser.add_argument(
        "--coverage-dir",
        default=None,
        help="Path to coverage folder (defaults to <data-root>/coverage)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Coverage threshold for bad ROIs (default: 0.1)",
    )
    parser.add_argument(
        "--output-name",
        default="hcp_mask.npy",
        help="Output filename (default: hcp_mask.npy)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve paths
    data_root = resolve_data_root(args.data_root)
    coverage_dir = Path(args.coverage_dir) if args.coverage_dir else Path(data_root) / "coverage"

    if args.threshold <= 0 or args.threshold >= 1:
        raise ValueError("threshold must be in (0, 1)")

    print("=" * 70)
    print("Creating HCPex atlas mask")
    print("=" * 70)
    print(f"Data root: {data_root}")
    print(f"Coverage dir: {coverage_dir}")
    print(f"Coverage threshold: {args.threshold}")
    print()

    # Step 1: Load coverage-based mask
    print("Step 1: Loading coverage data...")
    coverage_mask = load_coverage(coverage_dir, args.threshold)
    print()

    # Step 2: Load skipped ROIs from both sites
    print("Step 2: Loading skipped ROI lists...")
    ihb_skipped_file = coverage_dir / "ihb_skipped_rois_HCPex.txt"
    china_skipped_file = coverage_dir / "china_skipped_rois_HCPex.txt"

    ihb_skipped = load_skipped_rois(ihb_skipped_file)
    china_skipped = load_skipped_rois(china_skipped_file)

    all_skipped = ihb_skipped | china_skipped
    print(f"Total unique skipped ROIs: {len(all_skipped)}")
    print()

    # Step 3: Combine masks
    print("Step 3: Creating combined mask...")
    final_mask = create_combined_mask(coverage_mask, ihb_skipped, china_skipped)
    print()

    # Step 4: Save mask
    print("Step 4: Saving mask...")
    output_path = coverage_dir / args.output_name
    save_mask(final_mask, output_path)

    print()
    print("=" * 70)
    print("Mask creation complete!")
    print("=" * 70)

    # Print breakdown
    n_coverage = coverage_mask.sum()
    n_skipped_only = final_mask.sum() - n_coverage
    print(f"\nBreakdown:")
    print(f"  - Bad from low coverage: {n_coverage}")
    print(f"  - Bad from skipped ROIs only: {n_skipped_only}")
    print(f"  - Total bad ROIs: {final_mask.sum()}")
    print(f"  - Total good ROIs: {(~final_mask).sum()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
