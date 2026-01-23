#!/usr/bin/env python3
"""
Create bad-ROI masks from parcel coverage arrays (IHB + China).

A parcel is marked "bad" if coverage is below a threshold in either site.
Outputs one mask per atlas: "<atlas>_bad_parcels.npy".

Also prints per-atlas coverage overlap stats for IHB vs China.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from data_utils.paths import resolve_data_root


DEFAULT_ATLASES = ("AAL", "Schaefer200", "Brainnetome", "HCPex")


def load_coverage(coverage_dir: Path, site: str, atlas: str) -> np.ndarray:
    path = coverage_dir / f"{site}_{atlas}_parcel_coverage.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing coverage file: {path}")
    return np.asarray(np.load(path), dtype=float)


def compute_bad_mask(china: np.ndarray, ihb: np.ndarray, threshold: float) -> np.ndarray:
    if china.shape != ihb.shape:
        raise ValueError(f"Coverage shapes differ: china={china.shape} ihb={ihb.shape}")
    return (china < threshold) | (ihb < threshold)


def compute_coverage_stats(china: np.ndarray, ihb: np.ndarray, threshold: float) -> dict[str, float]:
    if china.shape != ihb.shape:
        raise ValueError(f"Coverage shapes differ: china={china.shape} ihb={ihb.shape}")
    ihb_good = ihb >= threshold
    china_good = china >= threshold
    both_good = ihb_good & china_good
    ihb_only = ihb_good & ~china_good
    china_only = china_good & ~ihb_good
    both_bad = ~ihb_good & ~china_good
    total = float(ihb_good.size)
    return {
        "n_rois": float(ihb_good.size),
        "ihb_good": float(ihb_good.sum()),
        "china_good": float(china_good.sum()),
        "both_good": float(both_good.sum()),
        "ihb_only": float(ihb_only.sum()),
        "china_only": float(china_only.sum()),
        "both_bad": float(both_bad.sum()),
        "ihb_good_pct": 100.0 * ihb_good.mean(),
        "china_good_pct": 100.0 * china_good.mean(),
        "both_good_pct": 100.0 * both_good.mean(),
        "ihb_only_pct": 100.0 * ihb_only.mean(),
        "china_only_pct": 100.0 * china_only.mean(),
        "both_bad_pct": 100.0 * both_bad.mean(),
    }


def save_mask(mask: np.ndarray, output_dir: Path, atlas: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{atlas}_bad_parcels.npy"
    np.save(out_path, mask.astype(bool))
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create bad-ROI masks from parcel coverage arrays.",
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
        "--output-dir",
        default="results/coverage_masks",
        help="Output directory for bad-ROI masks (default: results/coverage_masks)",
    )
    parser.add_argument(
        "--atlases",
        nargs="+",
        default=list(DEFAULT_ATLASES),
        help="Atlases to process (default: AAL Schaefer200 Brainnetome HCPex)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data_root = resolve_data_root(args.data_root)
    coverage_dir = Path(args.coverage_dir) if args.coverage_dir else Path(data_root) / "coverage"
    output_dir = Path(args.output_dir)

    if args.threshold <= 0 or args.threshold >= 1:
        raise ValueError("threshold must be in (0, 1)")

    for atlas in args.atlases:
        china = load_coverage(coverage_dir, "china", atlas)
        ihb = load_coverage(coverage_dir, "ihb", atlas)
        stats = compute_coverage_stats(china, ihb, args.threshold)
        mask = compute_bad_mask(china, ihb, args.threshold)
        out_path = save_mask(mask, output_dir, atlas)
        pct_bad = float(np.mean(mask) * 100.0)
        print(
            f"{atlas}: ihb_good={stats['ihb_good']:.0f} ({stats['ihb_good_pct']:.2f}%), "
            f"china_good={stats['china_good']:.0f} ({stats['china_good_pct']:.2f}%), "
            f"both_good={stats['both_good']:.0f} ({stats['both_good_pct']:.2f}%), "
            f"ihb_only={stats['ihb_only']:.0f} ({stats['ihb_only_pct']:.2f}%), "
            f"china_only={stats['china_only']:.0f} ({stats['china_only_pct']:.2f}%), "
            f"both_bad={stats['both_bad']:.0f} ({stats['both_bad_pct']:.2f}%)"
        )
        print(f"{atlas}: {mask.sum()} / {mask.size} bad ({pct_bad:.2f}%) -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
