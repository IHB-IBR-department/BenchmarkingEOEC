#!/usr/bin/env python3
"""
Sort precomputed glasso matrices into the site/atlas hierarchy.

The script renames files like
`china_close_AAL_strategy-1_GSR_glasso.npy`
into subdirectories under the glasso directory:
`<glasso_dir>/china/AAL/...`.

Example:
  PYTHONPATH=. python -m data_utils.preprocessing.migrations.sort_glasso \
    --glasso-dir ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/glasso_precomputed_fc \
    --dry-run

The default `--glasso-dir` is `<data_root>/glasso_precomputed_fc`. Run without
`--dry-run` to perform the move for real.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_utils.fc import _parse_strategy_path
from data_utils.paths import resolve_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reorganize glasso_precomputed_fc into china/atlas and ihb/atlas folders."
    )
    parser.add_argument(
        "--glasso-dir",
        default=None,
        help="Folder with glasso files (default: <data_root>/glasso_precomputed_fc).",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="OpenCloseBenchmark_data root path (optional).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just print planned moves without renaming files.",
    )
    return parser.parse_args()


def sort_glasso(glasso_dir: Path, *, dry_run: bool = False) -> None:
    files = sorted(glasso_dir.glob("*.npy"))
    if not files:
        print(f"No .npy files found in {glasso_dir}")
        return

    for path in files:
        try:
            site, atlas = _parse_strategy_path(path)
        except ValueError as exc:
            print(f"Skip {path.name}: {exc}")
            continue

        dest_dir = glasso_dir / site / atlas
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            print(f"Unable to create {dest_dir}: {exc}")
            sys.exit(1)

        dest_path = dest_dir / path.name
        if dest_path.exists():
            print(f"{dest_path} already exists; skipping {path.name}")
            continue

        print(f"{path} -> {dest_path}")
        if not dry_run:
            path.rename(dest_path)


def main() -> int:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    glasso_dir = Path(args.glasso_dir).expanduser() if args.glasso_dir else data_root / "glasso_precomputed_fc"
    if not glasso_dir.exists():
        print(f"Glasso directory not found: {glasso_dir}")
        return 1

    sort_glasso(glasso_dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
