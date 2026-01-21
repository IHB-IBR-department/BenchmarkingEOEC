"""
Aggregate China (Beijing) time series with per-subject session mapping.

China data has 3 runs per subject with variable session-to-condition mapping:
- Pattern A (23 subjects): run-1=closed, run-2=open, run-3=closed
- Pattern B (25 subjects): run-1=closed, run-2=closed, run-3=open

This script reads BeijingEOEC.csv to determine which run corresponds to
open/closed for each subject, then aggregates consistently.

Output format:
- close: 4D array (n_subjects, n_timepoints, n_rois, 2) with both closed sessions
  - close[:,:,:,0] = first closed session (always complete)
  - close[:,:,:,1] = second closed session
- open: 3D array (n_subjects, n_timepoints, n_rois)

Incomplete scans (zero-padded):
- sub-2021733 run-2 (open): 239/240 TRs - 1 TR missing (0.4%), minimal impact
- sub-3258811 run-3 (close2): 53/240 TRs - 187 TRs missing (78%), SIGNIFICANT
  WARNING: sub-3258811's second closed session is severely truncated.
  Consider excluding this subject's close2 data from analyses.

Usage:
    python -m benchmarking.aggregate_china
    python -m benchmarking.aggregate_china --aroma
    python -m benchmarking.aggregate_china --exclude sub-3258811  # if needed
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def load_session_mapping(csv_path: Path) -> pd.DataFrame:
    """
    Load BeijingEOEC.csv and create run-to-condition mapping.

    Returns DataFrame with columns:
        - subject: 'sub-XXXXXXX'
        - open_run: 'run-2' or 'run-3'
        - closed_runs: ['run-1', 'run-3'] or ['run-1', 'run-2']
    """
    df = pd.read_csv(csv_path)

    mapping = []
    for _, row in df.iterrows():
        subject = f"sub-{row['SubjectID']}"
        sessions = {
            'run-1': row['Session_1'],
            'run-2': row['Session_2'],
            'run-3': row['Session_3'],
        }

        open_run = [r for r, s in sessions.items() if s == 'open'][0]
        closed_runs = sorted([r for r, s in sessions.items() if s == 'closed'])

        mapping.append({
            'subject': subject,
            'open_run': open_run,
            'closed_run1': closed_runs[0],  # Always run-1
            'closed_run2': closed_runs[1],  # run-2 or run-3
        })

    return pd.DataFrame(mapping)


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


def load_timeseries_csv(filepath: Path, expected_timepoints: int = 240) -> np.ndarray:
    """
    Load time series from CSV, handling header row.

    If the time series is shorter than expected, pad with zeros.
    This handles incomplete scans:
    - sub-2021733 run-2 (open): 239/240 TRs (1 TR missing, minimal impact)
    - sub-3258811 run-3 (close2): 53/240 TRs (78% missing, significant padding)

    Note: sub-3258811's second closed session is severely truncated.
    The close2 data for this subject (close[:,:,:,1]) should be used with caution
    or excluded from analyses requiring complete data.

    Parameters
    ----------
    filepath : Path
        Path to CSV file
    expected_timepoints : int
        Expected number of timepoints (default: 240 for China data)

    Returns
    -------
    data : np.ndarray
        Time series with shape (expected_timepoints, n_rois), zero-padded if needed
    """
    df = pd.read_csv(filepath, header=0)
    data = df.values.astype(np.float32)

    actual_timepoints = data.shape[0]

    if actual_timepoints < expected_timepoints:
        # Pad with zeros for consistency
        n_pad = expected_timepoints - actual_timepoints
        padding = np.zeros((n_pad, data.shape[1]), dtype=np.float32)
        data = np.vstack([data, padding])
        pct_missing = (n_pad / expected_timepoints) * 100
        print(f"      Padded {filepath.parent.parent.parent.name}: {actual_timepoints} -> {expected_timepoints} ({pct_missing:.1f}% zeros)")

    return data


def aggregate_china_standard(
    input_dir: Path,
    output_dir: Path,
    mapping_csv: Path,
    atlases: Optional[list[str]] = None,
    strategies: Optional[list[int]] = None,
    gsr_options: Optional[list[str]] = None,
    exclude_subjects: Optional[list[str]] = None,
):
    """
    Aggregate China standard denoising pipelines.

    Output format:
    - close: 4D array (n_subjects, n_timepoints, n_rois, 2) with both closed sessions
    - open: 3D array (n_subjects, n_timepoints, n_rois)

    Parameters
    ----------
    exclude_subjects : list
        Subjects to exclude
    """
    # Load session mapping
    mapping_df = load_session_mapping(mapping_csv)
    print(f"Loaded session mapping for {len(mapping_df)} subjects")

    subjects = find_subjects(input_dir)
    print(f"Found {len(subjects)} subject directories")

    # Filter to subjects in mapping
    mapping_subjects = set(mapping_df['subject'])
    subjects = [s for s in subjects if s in mapping_subjects]
    print(f"Matched {len(subjects)} subjects with mapping")

    # Exclude problematic subjects
    if exclude_subjects:
        subjects = [s for s in subjects if s not in exclude_subjects]
        print(f"After exclusions: {len(subjects)} subjects")

    if atlases is None:
        atlases = find_atlases(input_dir, subjects)
    print(f"Atlases: {atlases}")

    if strategies is None:
        strategies = [1, 2, 3, 4, 5, 6]

    if gsr_options is None:
        gsr_options = ['GSR', 'noGSR']

    output_dir = Path(output_dir)

    # Build subject-to-mapping lookup
    subj_map = {row['subject']: row for _, row in mapping_df.iterrows()}

    for atlas in atlases:
        atlas_dir = output_dir / atlas
        atlas_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing atlas: {atlas}")

        for strategy in strategies:
            for gsr in gsr_options:
                print(f"  strategy-{strategy}, {gsr}...")

                # Collect data for each condition
                close_data = []  # Will be 4D: (n_subjects, n_timepoints, n_rois, 2)
                open_data = []   # Will be 3D: (n_subjects, n_timepoints, n_rois)
                valid_subjects = []

                for subject in subjects:
                    if subject not in subj_map:
                        continue

                    smap = subj_map[subject]

                    # Build file paths
                    def get_filepath(run):
                        filename = f"{subject}_task-rest_{run}_time-series_{atlas}_strategy-{strategy}_{gsr}.csv"
                        return input_dir / subject / 'time-series' / atlas / filename

                    # Load files
                    close1_path = get_filepath(smap['closed_run1'])
                    close2_path = get_filepath(smap['closed_run2'])
                    open_path = get_filepath(smap['open_run'])

                    if not close1_path.exists() or not close2_path.exists() or not open_path.exists():
                        print(f"    Warning: Missing files for {subject}")
                        continue

                    try:
                        close1_ts = load_timeseries_csv(close1_path)  # (n_timepoints, n_rois)
                        close2_ts = load_timeseries_csv(close2_path)  # (n_timepoints, n_rois)
                        open_ts = load_timeseries_csv(open_path)      # (n_timepoints, n_rois)

                        # Stack both closed sessions along last axis: (n_timepoints, n_rois, 2)
                        close_stacked = np.stack([close1_ts, close2_ts], axis=-1)

                        close_data.append(close_stacked)
                        open_data.append(open_ts)
                        valid_subjects.append(subject)

                    except Exception as e:
                        print(f"    Error loading {subject}: {e}")
                        continue

                if not close_data:
                    print(f"    No valid data!")
                    continue

                # Stack across subjects
                # close_arr: (n_subjects, n_timepoints, n_rois, 2)
                # open_arr: (n_subjects, n_timepoints, n_rois)
                close_arr = np.stack(close_data, axis=0)
                open_arr = np.stack(open_data, axis=0)

                np.save(atlas_dir / f"china_close_{atlas}_strategy-{strategy}_{gsr}.npy", close_arr)
                np.save(atlas_dir / f"china_open_{atlas}_strategy-{strategy}_{gsr}.npy", open_arr)
                print(f"    Saved close: {close_arr.shape}, open: {open_arr.shape}")

        # Save subject order
        subject_order_file = atlas_dir / 'subject_order_china.txt'
        with open(subject_order_file, 'w') as f:
            f.write('\n'.join(valid_subjects))
        print(f"  Saved subject_order_china.txt ({len(valid_subjects)} subjects)")


def aggregate_china_aroma(
    input_dir: Path,
    output_dir: Path,
    mapping_csv: Path,
    atlases: Optional[list[str]] = None,
    gsr_options: Optional[list[str]] = None,
    exclude_subjects: Optional[list[str]] = None,
):
    """
    Aggregate China AROMA pipelines.

    Output format:
    - close: 4D array (n_subjects, n_timepoints, n_rois, 2) with both closed sessions
    - open: 3D array (n_subjects, n_timepoints, n_rois)
    """
    mapping_df = load_session_mapping(mapping_csv)
    print(f"Loaded session mapping for {len(mapping_df)} subjects")

    subjects = find_subjects(input_dir)
    print(f"Found {len(subjects)} subject directories (AROMA)")

    mapping_subjects = set(mapping_df['subject'])
    subjects = [s for s in subjects if s in mapping_subjects]

    if exclude_subjects:
        subjects = [s for s in subjects if s not in exclude_subjects]
    print(f"Processing {len(subjects)} subjects")

    if atlases is None:
        first_sub = input_dir / subjects[0] / 'time-series'
        atlases = sorted([d.name for d in first_sub.iterdir() if d.is_dir()])
    print(f"Atlases: {atlases}")

    if gsr_options is None:
        gsr_options = ['GSR', 'noGSR']

    aroma_types = ['aggrDenoised', 'nonaggrDenoised']
    aroma_short = {'aggrDenoised': 'aggr', 'nonaggrDenoised': 'nonaggr'}

    output_dir = Path(output_dir)
    subj_map = {row['subject']: row for _, row in mapping_df.iterrows()}

    for atlas in atlases:
        atlas_dir = output_dir / atlas
        atlas_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing atlas: {atlas} (AROMA)")

        for aroma_type in aroma_types:
            for gsr in gsr_options:
                print(f"  AROMA_{aroma_type}, {gsr}...")

                close_data = []  # 4D: (n_subjects, n_timepoints, n_rois, 2)
                open_data = []   # 3D: (n_subjects, n_timepoints, n_rois)
                valid_subjects = []

                for subject in subjects:
                    if subject not in subj_map:
                        continue

                    smap = subj_map[subject]

                    def get_filepath(run):
                        filename = f"{subject}_task-rest_{run}_time-series_{atlas}_strategy-AROMA_{aroma_type}_{gsr}.csv"
                        return input_dir / subject / 'time-series' / atlas / filename

                    close1_path = get_filepath(smap['closed_run1'])
                    close2_path = get_filepath(smap['closed_run2'])
                    open_path = get_filepath(smap['open_run'])

                    if not close1_path.exists() or not close2_path.exists() or not open_path.exists():
                        continue

                    try:
                        close1_ts = load_timeseries_csv(close1_path)
                        close2_ts = load_timeseries_csv(close2_path)
                        open_ts = load_timeseries_csv(open_path)

                        # Stack both closed sessions: (n_timepoints, n_rois, 2)
                        close_stacked = np.stack([close1_ts, close2_ts], axis=-1)

                        close_data.append(close_stacked)
                        open_data.append(open_ts)
                        valid_subjects.append(subject)
                    except Exception as e:
                        print(f"    Error: {subject}: {e}")
                        continue

                if not close_data:
                    print(f"    No valid data!")
                    continue

                close_arr = np.stack(close_data, axis=0)
                open_arr = np.stack(open_data, axis=0)

                short = aroma_short[aroma_type]
                np.save(atlas_dir / f"china_close_{atlas}_strategy-AROMA_{short}_{gsr}.npy", close_arr)
                np.save(atlas_dir / f"china_open_{atlas}_strategy-AROMA_{short}_{gsr}.npy", open_arr)
                print(f"    Saved: close={close_arr.shape}, open={open_arr.shape}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate China (Beijing) time series with session mapping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/OpenCloseChina_ts',
        help='Input directory with per-subject CSVs',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/timeseries_china',
        help='Output directory for aggregated arrays',
    )
    parser.add_argument(
        '--mapping-csv',
        type=str,
        default='~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/BeijingEOEC.csv',
        help='Path to BeijingEOEC.csv session mapping',
    )
    parser.add_argument(
        '--aroma',
        action='store_true',
        help='Aggregate AROMA data',
    )
    parser.add_argument(
        '--aroma-input-dir',
        type=str,
        default='~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/OpenCloseChina_aroma',
        help='Input directory for AROMA data',
    )
    parser.add_argument(
        '--atlases',
        type=str,
        nargs='+',
        default=None,
        help='Atlases to process (default: auto-detect)',
    )
    parser.add_argument(
        '--exclude',
        type=str,
        nargs='*',
        default=[],  # sub-2021733 now handled via padding
        help='Subjects to exclude',
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    mapping_csv = Path(args.mapping_csv).expanduser()

    print("=" * 60)
    print("China Time Series Aggregation")
    print("=" * 60)
    print(f"Output format: close=(n_subjects, n_timepoints, n_rois, 2), open=(n_subjects, n_timepoints, n_rois)")
    print(f"Excluding: {args.exclude}")

    if args.aroma:
        aroma_dir = Path(args.aroma_input_dir).expanduser()
        print(f"Input (AROMA): {aroma_dir}")
        print(f"Output: {output_dir}")
        aggregate_china_aroma(
            aroma_dir, output_dir, mapping_csv,
            atlases=args.atlases,
            exclude_subjects=args.exclude,
        )
    else:
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        aggregate_china_standard(
            input_dir, output_dir, mapping_csv,
            atlases=args.atlases,
            exclude_subjects=args.exclude,
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
