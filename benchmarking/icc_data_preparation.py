#!/usr/bin/env python3
"""
Prepare FC matrices for ICC workflows with optional subject filtering.

Skips FC types that already exist on disk. If a glasso precompute folder is
provided, the script will reuse the matching precomputed glasso matrix instead
of recomputing it.

This script is intended for the China dataset only (two closed sessions).
Defaults to `<data_root>/timeseries_china` as input and
`<data_root>/icc_precomputed_fc` as output when no paths are provided.

Examples:
  PYTHONPATH=. python -m benchmarking.icc_data_preparation \
    --input /path/to/china_close_AAL_strategy-1_GSR.npy \
    --output-dir /path/to/icc_precomputed_fc \
    --glasso-dir /path/to/glasso_precomputed_fc \
    --print-timing

  PYTHONPATH=. python -m benchmarking.icc_data_preparation \
    --input-dir /path/to/timeseries_china \
    --output-dir /path/to/icc_precomputed_fc \
    --glasso-dir /path/to/glasso_precomputed_fc \
    --strategy 1 \
    --gsr GSR \
    --atlas AAL
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path
from typing import Iterable
import shutil

import numpy as np

from benchmarking.fc import compute_fc_from_strategy_file
from benchmarking.project import resolve_data_root


def default_input_path(data_root: Path) -> Path:
    return data_root / "timeseries_china" / "AAL" / "china_close_AAL_strategy-1_GSR.npy"


def load_subject_order(strategy_path: Path) -> list[str] | None:
    name = strategy_path.name
    if name.startswith("china_") or "timeseries_china" in strategy_path.parts:
        candidates = ["subject_order_china.txt"]
    elif name.startswith("ihb_") or "timeseries_ihb" in strategy_path.parts:
        candidates = ["subject_order.txt"]
    else:
        candidates = ["subject_order_china.txt", "subject_order.txt"]

    for candidate in candidates:
        path = strategy_path.parent / candidate
        if path.exists():
            return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return None


def parse_atlas(strategy_path: Path) -> str:
    name = strategy_path.name
    match = re.search(r"_(?P<atlas>[^_]+)_strategy-", name)
    if match:
        return match.group("atlas")
    parent_name = strategy_path.parent.name
    if parent_name and parent_name not in ("timeseries_ihb", "timeseries_china"):
        return parent_name
    raise ValueError(f"Unable to parse atlas from strategy file: {strategy_path}")


def is_china_series(strategy_path: Path) -> bool:
    name = strategy_path.name
    return name.startswith("china_") or "timeseries_china" in strategy_path.parts


def find_precomputed_glasso(glasso_dir: Path, stem: str) -> Path | None:
    if not glasso_dir.exists():
        raise FileNotFoundError(f"Glasso directory not found: {glasso_dir}")
    matches = list(glasso_dir.rglob(f"{stem}_glasso.npy"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple precomputed glasso files found: {matches}")
    return matches[0]


def filter_subjects(
    timeseries: np.ndarray,
    subject_order: list[str] | None,
    drop_subjects: Iterable[str],
) -> tuple[np.ndarray, list[str] | None, list[str]]:
    drop_list = [s for s in drop_subjects if s]
    if not drop_list:
        return timeseries, subject_order, []
    if subject_order is None:
        raise FileNotFoundError(
            "Subject order file not found; cannot drop subjects without ordering."
        )
    if len(subject_order) != timeseries.shape[0]:
        raise ValueError(
            f"Subject order length {len(subject_order)} does not match data ({timeseries.shape[0]})."
        )
    drop_set = set(drop_list)
    keep_idx = [idx for idx, sid in enumerate(subject_order) if sid not in drop_set]
    dropped = [sid for sid in subject_order if sid in drop_set]
    if not dropped:
        return timeseries, subject_order, []
    filtered = timeseries[keep_idx]
    kept_subjects = [subject_order[idx] for idx in keep_idx]
    return filtered, kept_subjects, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FC matrices for a time-series .npy file with ICC-friendly filtering.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to time-series .npy (default: china_close_AAL_strategy-1_GSR.npy under data root).",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Folder with time-series .npy files to process (recursive).",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to OpenCloseBenchmark_data (optional if OPEN_CLOSE_BENCHMARK_DATA is set).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output base directory (default: <data_root>/icc_precomputed_fc).",
    )
    parser.add_argument(
        "--glasso-dir",
        default=None,
        help="Folder to search for precomputed glasso matrices (optional).",
    )
    parser.add_argument(
        "--atlas",
        default=None,
        help="Filter to a single atlas name (e.g., AAL).",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Filter to a single strategy ID (matches 'strategy-{id}' in filename).",
    )
    parser.add_argument(
        "--gsr",
        default=None,
        help="Filter to a single GSR option: GSR or noGSR.",
    )
    parser.add_argument(
        "--coverage",
        default="china",
        help="Coverage source for ROI filtering (default: china).",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.1,
        help="Coverage threshold for ROI filtering (default: 0.1).",
    )
    parser.add_argument(
        "--drop-subject",
        action="append",
        default=None,
        help="Subject ID to drop (repeatable). Defaults to sub-3258811 if unset.",
    )
    parser.add_argument(
        "--print-timing",
        action="store_true",
        help="Print computation time per FC type.",
    )
    return parser.parse_args()


def normalize_coverage_arg(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in ("none", "off", "false", "no"):
        return None
    return value


def normalize_gsr(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    lowered = cleaned.lower()
    if lowered in ("gsr",):
        return "GSR"
    if lowered in ("nogsr", "no_gsr", "no-gsr"):
        return "noGSR"
    if cleaned in ("GSR", "noGSR"):
        return cleaned
    raise ValueError("gsr must be 'GSR' or 'noGSR'.")


def copy_precomputed_glasso(precomputed: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(precomputed, output_path)


def process_series_folder(
    input_dir: Path,
    output_dir: Path,
    *,
    data_root: str | Path | None,
    coverage: str | None,
    coverage_threshold: float,
    drop_subjects: list[str],
    glasso_dir: Path | None,
    print_timing: bool,
    strategy: str | None,
    gsr: str | None,
    atlas: str | None,
) -> int:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.rglob("*.npy"))
    if not files:
        print("No .npy files found.")
        return 0

    processed = 0
    skipped = 0
    copied = 0
    failed = 0

    coverage_arg = normalize_coverage_arg(coverage)
    gsr_value = normalize_gsr(gsr)
    if coverage_arg is not None:
        if coverage_threshold <= 0 or coverage_threshold >= 1:
            raise ValueError("coverage_threshold must be in (0, 1)")

    name_map = {
        "corr": "corr",
        "partial": "pc",
        "tangent": "tang",
        "glasso": "glasso",
    }

    china_files = [path for path in files if is_china_series(path)]
    if not china_files:
        raise ValueError("No China time-series files found. This script expects China data.")

    atlas_value = atlas.strip() if atlas else None

    for path in china_files:
        if "_close_" not in path.name:
            skipped += 1
            continue
        if atlas_value is not None and parse_atlas(path) != atlas_value:
            skipped += 1
            continue
        if strategy is not None and f"strategy-{strategy}_" not in path.name:
            skipped += 1
            continue
        if gsr_value is not None and not path.name.endswith(f"_{gsr_value}.npy"):
            skipped += 1
            continue
        try:
            atlas = parse_atlas(path)
            atlas_out = output_dir / atlas
            atlas_out.mkdir(parents=True, exist_ok=True)

            out_paths = {
                key: atlas_out / f"{path.stem}_{suffix}.npy"
                for key, suffix in name_map.items()
            }
            missing_kinds = [key for key, out_path in out_paths.items() if not out_path.exists()]

            if "glasso" in missing_kinds and glasso_dir:
                precomputed = find_precomputed_glasso(glasso_dir, path.stem)
                if precomputed is not None:
                    copy_precomputed_glasso(precomputed, out_paths["glasso"])
                    missing_kinds.remove("glasso")
                    copied += 1
                    print(f"Copied glasso: {out_paths['glasso']}")

            if not missing_kinds:
                skipped += 1
                continue

            subject_order = load_subject_order(path)
            timeseries = np.load(path)
            timeseries, _, dropped = filter_subjects(
                timeseries,
                subject_order,
                drop_subjects,
            )
            if dropped:
                print(f"{path.name}: dropped {', '.join(dropped)}")

            with tempfile.TemporaryDirectory() as tmp_dir:
                filtered_path = Path(tmp_dir) / path.name
                np.save(filtered_path, timeseries)
                fc = compute_fc_from_strategy_file(
                    filtered_path,
                    tangent_connectivity=None,
                    kinds=missing_kinds,
                    coverage=coverage_arg,
                    coverage_threshold=coverage_threshold,
                    data_path=data_root,
                    print_timing=print_timing,
                )

            for key in missing_kinds:
                if key not in fc:
                    raise KeyError(f"Missing FC output for {key}")
                out_path = out_paths[key]
                np.save(out_path, fc[key])
                if print_timing:
                    print(f"Saved {name_map[key]}: {out_path} {fc[key].shape}")
                else:
                    print(f"Saved {name_map[key]}: {out_path}")
            processed += 1

        except Exception as exc:
            failed += 1
            print(f"Failed {path}: {exc}")
            continue

    print(
        f"Done. Processed={processed} skipped={skipped} copied_glasso={copied} failed={failed}"
    )
    return 0


def main() -> int:
    args = parse_args()

    data_root = resolve_data_root(args.data_root)
    if args.input_dir and args.input:
        raise ValueError("Use only one of --input or --input-dir.")
    output_dir_base = Path(args.output_dir).expanduser() if args.output_dir else Path(data_root) / "icc_precomputed_fc"
    if args.input_dir is None and args.input is None:
        return process_series_folder(
            Path(data_root) / "timeseries_china",
            output_dir_base,
            data_root=args.data_root,
            coverage=args.coverage,
            coverage_threshold=args.coverage_threshold,
            drop_subjects=args.drop_subject if args.drop_subject is not None else ["sub-3258811"],
            glasso_dir=Path(args.glasso_dir).expanduser() if args.glasso_dir else None,
            print_timing=args.print_timing,
            strategy=args.strategy,
            gsr=args.gsr,
            atlas=args.atlas,
        )
    if args.input_dir:
        return process_series_folder(
            Path(args.input_dir).expanduser(),
            output_dir_base,
            data_root=args.data_root,
            coverage=args.coverage,
            coverage_threshold=args.coverage_threshold,
            drop_subjects=args.drop_subject if args.drop_subject is not None else ["sub-3258811"],
            glasso_dir=Path(args.glasso_dir).expanduser() if args.glasso_dir else None,
            print_timing=args.print_timing,
            strategy=args.strategy,
            gsr=args.gsr,
            atlas=args.atlas,
        )

    input_path = Path(args.input).expanduser() if args.input else default_input_path(data_root)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input time-series file: {input_path}")
    if not is_china_series(input_path):
        raise ValueError("This script supports China time-series inputs only.")

    atlas = parse_atlas(input_path)
    output_dir = output_dir_base / atlas
    output_dir.mkdir(parents=True, exist_ok=True)

    name_map = {
        "corr": "corr",
        "partial": "pc",
        "tangent": "tang",
        "glasso": "glasso",
    }
    out_paths = {
        key: output_dir / f"{input_path.stem}_{suffix}.npy"
        for key, suffix in name_map.items()
    }
    missing_kinds = [key for key, path in out_paths.items() if not path.exists()]
    if not missing_kinds:
        print("All requested FC outputs already exist. Skipping computation.")
        return 0

    if args.coverage_threshold <= 0 or args.coverage_threshold >= 1:
        raise ValueError("coverage_threshold must be in (0, 1)")

    drop_subjects = args.drop_subject if args.drop_subject is not None else ["sub-3258811"]
    subject_order = load_subject_order(input_path)

    timeseries = np.load(input_path)
    timeseries, subject_order, dropped = filter_subjects(
        timeseries,
        subject_order,
        drop_subjects,
    )

    if dropped:
        print(f"Dropped subjects: {', '.join(dropped)}")
    elif drop_subjects:
        print("No matching subjects found to drop.")

    coverage_arg = normalize_coverage_arg(args.coverage)
    glasso_path = None
    if "glasso" in missing_kinds and args.glasso_dir:
        glasso_path = find_precomputed_glasso(
            Path(args.glasso_dir).expanduser(),
            input_path.stem,
        )
        if glasso_path is not None:
            print(f"Using precomputed glasso: {glasso_path}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        filtered_path = Path(tmp_dir) / input_path.name
        np.save(filtered_path, timeseries)
        fc = compute_fc_from_strategy_file(
            filtered_path,
            tangent_connectivity=None,
            kinds=missing_kinds,
            coverage=coverage_arg,
            coverage_threshold=args.coverage_threshold,
            data_path=args.data_root,
            print_timing=args.print_timing,
            glasso=glasso_path,
        )

    for key in missing_kinds:
        if key not in fc:
            raise KeyError(f"Missing FC output for {key}")
        out_path = out_paths[key]
        np.save(out_path, fc[key])
        print(f"Saved {name_map[key]} FC: {out_path} {fc[key].shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
