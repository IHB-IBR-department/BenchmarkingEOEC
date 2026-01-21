#!/usr/bin/env python3
"""
Script to fix file naming in OpenCloseIHB_outputs folder.
Adds '_noGSR' suffix to CSV files that don't have GSR or noGSR suffix.
"""

import os
from pathlib import Path

def fix_filenames(base_path):
    """
    Find and rename CSV files that don't have _GSR or _noGSR suffix.
    """
    base_path = Path(base_path).expanduser()

    # Find all CSV files
    csv_files = list(base_path.rglob("*.csv"))

    files_to_rename = []
    files_already_correct = []

    for filepath in csv_files:
        filename = filepath.name

        # Check if file already has correct naming
        if filename.endswith("_GSR.csv") or filename.endswith("_noGSR.csv"):
            files_already_correct.append(filepath)
        else:
            # File needs to be renamed
            files_to_rename.append(filepath)

    print(f"Total CSV files found: {len(csv_files)}")
    print(f"Files with correct naming: {len(files_already_correct)}")
    print(f"Files to rename: {len(files_to_rename)}")

    if files_to_rename:
        print("\nSample files to rename:")
        for f in files_to_rename[:10]:
            print(f"  {f.name}")

        response = input("\nProceed with renaming? (yes/no): ")

        if response.lower() == 'yes':
            renamed_count = 0
            errors = []

            for filepath in files_to_rename:
                try:
                    # Create new filename by replacing .csv with _noGSR.csv
                    new_name = filepath.name.replace(".csv", "_noGSR.csv")
                    new_path = filepath.parent / new_name

                    # Rename the file
                    filepath.rename(new_path)
                    renamed_count += 1

                    if renamed_count % 100 == 0:
                        print(f"Renamed {renamed_count} files...")

                except Exception as e:
                    errors.append((filepath, str(e)))

            print(f"\nSuccessfully renamed {renamed_count} files")

            if errors:
                print(f"\nErrors encountered: {len(errors)}")
                for filepath, error in errors[:10]:
                    print(f"  {filepath}: {error}")
        else:
            print("Renaming cancelled")
    else:
        print("\nNo files need renaming!")

if __name__ == "__main__":
    base_path = "~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/OpenCloseIHB_outputs"
    fix_filenames(base_path)
