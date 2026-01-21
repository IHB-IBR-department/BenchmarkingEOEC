"""
Aggregate IHB (St. Petersburg) per-subject time series CSVs into numpy arrays.

Usage:
    # Standard denoising pipelines (strategies 1-6)
    python -m benchmarking.aggregate_ihb

    # With custom paths
    python -m benchmarking.aggregate_ihb --input-dir <path> --output-dir <path>

    # Specific atlases only
    python -m benchmarking.aggregate_ihb --atlases Schaefer200 AAL

    # AROMA denoising pipelines
    python -m benchmarking.aggregate_ihb --aroma

This script reads individual subject CSV files and aggregates them into
numpy arrays with consistent subject ordering. This format is more efficient
for batch processing and ensures reproducible subject ordering across analyses.

Input structure (per-subject CSVs):
    OpenCloseIHB_outputs/sub-XXX/time-series/{atlas}/
        sub-XXX_task-rest_run-1_time-series_{atlas}_strategy-{1-6}_{GSR,noGSR}.csv
        sub-XXX_task-rest_run-2_time-series_{atlas}_strategy-{1-6}_{GSR,noGSR}.csv

Output structure (aggregated arrays):
    timeseries_ihb/{atlas}/
        ihb_close_{atlas}_strategy-{1-6}_{GSR,noGSR}.npy  # (n_subjects, n_timepoints, n_rois)
        ihb_open_{atlas}_strategy-{1-6}_{GSR,noGSR}.npy
        subject_order.txt  # Canonical subject ordering

Run mapping:
    - run-1 = closed (eyes closed)
    - run-2 = open (eyes open)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


# Run to condition mapping
RUN_TO_CONDITION = {
    'run-1': 'close',
    'run-2': 'open',
}


def find_subjects(input_dir: Path) -> list[str]:
    """Find all subject directories in sorted order."""
    subjects = sorted([
        d.name for d in input_dir.iterdir()
        if d.is_dir() and d.name.startswith('sub-')
    ])
    return subjects


def find_atlases(input_dir: Path, subjects: list[str]) -> list[str]:
    """Find available atlases from first subject."""
    if not subjects:
        return []

    first_sub = input_dir / subjects[0] / 'time-series'
    if not first_sub.exists():
        return []

    atlases = sorted([
        d.name for d in first_sub.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])
    return atlases


def parse_filename(filename: str) -> Optional[dict]:
    """
    Parse time series filename into components.

    Example: sub-001_task-rest_run-1_time-series_Schaefer200_strategy-1_noGSR.csv
    Returns: {'subject': 'sub-001', 'run': 'run-1', 'atlas': 'Schaefer200',
              'strategy': '1', 'gsr': 'noGSR'}
    """
    parts = filename.replace('.csv', '').split('_')

    result = {}
    for part in parts:
        if part.startswith('sub-'):
            result['subject'] = part
        elif part.startswith('run-'):
            result['run'] = part
        elif part.startswith('strategy-'):
            result['strategy'] = part.replace('strategy-', '')
        elif part in ('GSR', 'noGSR'):
            result['gsr'] = part

    # Find atlas (comes after 'time-series')
    try:
        ts_idx = parts.index('time-series')
        if ts_idx + 1 < len(parts):
            result['atlas'] = parts[ts_idx + 1]
    except ValueError:
        pass

    return result if len(result) >= 4 else None


def load_timeseries_csv(filepath: Path) -> np.ndarray:
    """Load time series from CSV, handling header row."""
    df = pd.read_csv(filepath, header=0)
    return df.values.astype(np.float32)


def aggregate_pipeline(
    input_dir: Path,
    subjects: list[str],
    atlas: str,
    strategy: str,
    gsr: str,
    run: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Aggregate time series for one pipeline configuration.

    Returns:
        data: (n_subjects, n_timepoints, n_rois)
        valid_subjects: list of subjects that had valid data
    """
    all_data = []
    valid_subjects = []

    for subject in subjects:
        # Construct expected filename
        filename = f"{subject}_task-rest_{run}_time-series_{atlas}_strategy-{strategy}_{gsr}.csv"
        filepath = input_dir / subject / 'time-series' / atlas / filename

        if not filepath.exists():
            print(f"  Warning: Missing {filepath.name}")
            continue

        try:
            ts = load_timeseries_csv(filepath)
            all_data.append(ts)
            valid_subjects.append(subject)
        except Exception as e:
            print(f"  Error loading {filepath.name}: {e}")
            continue

    if not all_data:
        return np.array([]), valid_subjects

    # Stack into (n_subjects, n_timepoints, n_rois)
    return np.stack(all_data, axis=0), valid_subjects


def aggregate_standard(
    input_dir: Path,
    output_dir: Path,
    atlases: Optional[list[str]] = None,
    strategies: Optional[list[int]] = None,
    gsr_options: Optional[list[str]] = None,
):
    """
    Aggregate standard denoising pipelines (strategies 1-6).
    """
    subjects = find_subjects(input_dir)
    print(f"Found {len(subjects)} subjects")

    if atlases is None:
        atlases = find_atlases(input_dir, subjects)
    print(f"Atlases: {atlases}")

    if strategies is None:
        strategies = [1, 2, 3, 4, 5, 6]

    if gsr_options is None:
        gsr_options = ['GSR', 'noGSR']

    output_dir = Path(output_dir)

    for atlas in atlases:
        atlas_dir = output_dir / atlas
        atlas_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing atlas: {atlas}")

        for strategy in strategies:
            for gsr in gsr_options:
                for run, condition in RUN_TO_CONDITION.items():
                    print(f"  strategy-{strategy}, {gsr}, {condition}...")

                    data, _ = aggregate_pipeline(
                        input_dir=input_dir,
                        subjects=subjects,
                        atlas=atlas,
                        strategy=str(strategy),
                        gsr=gsr,
                        run=run,
                    )

                    if data.size == 0:
                        print(f"    No valid data!")
                        continue

                    # Save array
                    out_name = f"ihb_{condition}_{atlas}_strategy-{strategy}_{gsr}.npy"
                    np.save(atlas_dir / out_name, data)
                    print(f"    Saved {out_name}: {data.shape}")

        # Save subject order (same for all pipelines)
        subject_order_file = atlas_dir / 'subject_order.txt'
        with open(subject_order_file, 'w') as f:
            f.write('\n'.join(subjects))
        print(f"  Saved subject_order.txt ({len(subjects)} subjects)")


def aggregate_aroma(
    input_dir: Path,
    output_dir: Path,
    atlases: Optional[list[str]] = None,
    gsr_options: Optional[list[str]] = None,
):
    """
    Aggregate AROMA denoising pipelines.

    AROMA strategies:
        - aggrDenoised: aggressive denoising
        - nonaggrDenoised: non-aggressive (soft) denoising
    """
    # AROMA has nested structure: input_dir/sub-XXX/time-series/{atlas}/
    subjects = find_subjects(input_dir)
    print(f"Found {len(subjects)} subjects (AROMA)")

    if atlases is None:
        # Check first subject for atlases
        first_sub = input_dir / subjects[0] / 'time-series'
        if first_sub.exists():
            atlases = sorted([d.name for d in first_sub.iterdir() if d.is_dir()])
        else:
            atlases = ['Schaefer200']
    print(f"Atlases: {atlases}")

    if gsr_options is None:
        gsr_options = ['GSR', 'noGSR']

    aroma_types = ['aggrDenoised', 'nonaggrDenoised']
    aroma_short = {'aggrDenoised': 'aggr', 'nonaggrDenoised': 'nonaggr'}

    output_dir = Path(output_dir)

    for atlas in atlases:
        atlas_dir = output_dir / atlas
        atlas_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing atlas: {atlas} (AROMA)")

        for aroma_type in aroma_types:
            for gsr in gsr_options:
                for run, condition in RUN_TO_CONDITION.items():
                    print(f"  AROMA_{aroma_type}, {gsr}, {condition}...")

                    all_data = []
                    valid_subjects = []

                    for subject in subjects:
                        # AROMA filename pattern
                        filename = f"{subject}_task-rest_{run}_time-series_{atlas}_strategy-AROMA_{aroma_type}_{gsr}.csv"
                        filepath = input_dir / subject / 'time-series' / atlas / filename

                        if not filepath.exists():
                            # Try alternate path structure
                            alt_filepath = input_dir / subject / 'time-series' / 'time-series' / atlas / filename
                            if alt_filepath.exists():
                                filepath = alt_filepath
                            else:
                                continue

                        try:
                            ts = load_timeseries_csv(filepath)
                            all_data.append(ts)
                            valid_subjects.append(subject)
                        except Exception as e:
                            print(f"    Error: {subject}: {e}")
                            continue

                    if not all_data:
                        print(f"    No valid data!")
                        continue

                    data = np.stack(all_data, axis=0)

                    # Save with short name
                    short = aroma_short[aroma_type]
                    out_name = f"ihb_{condition}_{atlas}_strategy-AROMA_{short}_{gsr}.npy"
                    np.save(atlas_dir / out_name, data)
                    print(f"    Saved {out_name}: {data.shape}")

        # Save subject order
        subject_order_file = atlas_dir / 'subject_order_aroma.txt'
        with open(subject_order_file, 'w') as f:
            f.write('\n'.join(subjects))


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate IHB (St. Petersburg) per-subject time series into numpy arrays',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/OpenCloseIHB_outputs',
        help='Input directory with per-subject CSVs',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/timeseries_ihb',
        help='Output directory for aggregated arrays',
    )
    parser.add_argument(
        '--aroma',
        action='store_true',
        help='Aggregate AROMA data instead of standard pipelines',
    )
    parser.add_argument(
        '--aroma-input-dir',
        type=str,
        default='~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/OpenCloseIHB_aroma/time-series',
        help='Input directory for AROMA data',
    )
    parser.add_argument(
        '--atlases',
        type=str,
        nargs='+',
        default=None,
        help='Atlases to process (default: auto-detect)',
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    print("=" * 60)
    print("IHB Time Series Aggregation")
    print("=" * 60)

    if args.aroma:
        aroma_dir = Path(args.aroma_input_dir).expanduser()
        print(f"Input (AROMA): {aroma_dir}")
        print(f"Output: {output_dir}")
        aggregate_aroma(aroma_dir, output_dir, atlases=args.atlases)
    else:
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        aggregate_standard(input_dir, output_dir, atlases=args.atlases)

    print("\nDone!")


if __name__ == '__main__':
    main()
