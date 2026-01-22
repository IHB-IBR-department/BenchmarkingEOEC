#!/usr/bin/env python
"""
Script to create ihb_rmsd.csv and china_rmsd.csv from _rmsd.json files
"""
import json
import pandas as pd
from pathlib import Path


def create_rmsd_csv(subject_order_file, outputs_dir, output_csv, dataset_name):
    """
    Create RMSD CSV file from JSON files

    Args:
        subject_order_file: Path to subject order file
        outputs_dir: Path to outputs directory containing RMSD JSON files
        output_csv: Path to output CSV file
        dataset_name: Name of dataset for logging (e.g., "IHB", "China")
    """
    # Read subject order
    with open(subject_order_file, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} dataset")
    print(f"{'='*60}")
    print(f"Found {len(subjects)} subjects in {subject_order_file.name}")

    # Collect data
    data = []
    missing_subjects = []

    for subject in subjects:
        rmsd_file = outputs_dir / subject / f"{subject}_rmsd.json"

        if rmsd_file.exists():
            with open(rmsd_file, 'r') as f:
                rmsd_data = json.load(f)

            row = {'subject': subject}

            # Add all runs found in the JSON file
            for run_key in sorted(rmsd_data.keys()):
                row[run_key] = rmsd_data[run_key]

            data.append(row)
        else:
            missing_subjects.append(subject)
            print(f"Warning: RMSD file not found for {subject}")

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_csv, index=False)

    print(f"\nCreated CSV with {len(df)} subjects")
    print(f"Output saved to: {output_csv}")

    if missing_subjects:
        print(f"\nMissing RMSD files for {len(missing_subjects)} subjects:")
        for sub in missing_subjects[:10]:  # Show first 10
            print(f"  - {sub}")
        if len(missing_subjects) > 10:
            print(f"  ... and {len(missing_subjects) - 10} more")

    # Display first few rows
    print(f"\nFirst 5 rows of the CSV:")
    print(df.head())

    return df


if __name__ == "__main__":
    # Define base paths
    data_dir = Path.home() / "Yandex.Disk.localized/IHB/OpenCloseBenchmark_data"

    # Process IHB dataset
    ihb_subject_order = data_dir / "timeseries_ihb" / "subject_order.txt"
    ihb_outputs_dir = data_dir / "OpenCloseIHB_outputs"
    ihb_output_csv = data_dir / "timeseries_ihb" / "ihb_rmsd.csv"

    df_ihb = create_rmsd_csv(
        ihb_subject_order,
        ihb_outputs_dir,
        ihb_output_csv,
        "IHB"
    )

    # Process China dataset
    china_subject_order = data_dir / "timeseries_china" / "subject_order_china.txt"
    china_outputs_dir = data_dir / "OpenCloseChina_ts"
    china_output_csv = data_dir / "timeseries_china" / "china_rmsd.csv"

    df_china = create_rmsd_csv(
        china_subject_order,
        china_outputs_dir,
        china_output_csv,
        "China"
    )

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"IHB dataset: {len(df_ihb)} subjects, columns: {list(df_ihb.columns)}")
    print(f"China dataset: {len(df_china)} subjects, columns: {list(df_china.columns)}")
