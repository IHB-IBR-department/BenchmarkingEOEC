#!/usr/bin/env python3
"""
Prepare FC matrices for ICC workflows with optional subject filtering.

Skips FC types that already exist on disk. If a glasso precompute folder is
provided, the script will reuse the matching precomputed glasso matrix instead
of recomputing it. Glasso is never computed by this script: if a matching
precomputed file is missing, glasso output is skipped.

This script is intended for the China dataset only (two closed sessions).
Defaults to `<data_root>/timeseries_china` as input and
`<data_root>/icc_precomputed_fc` as output when no paths are provided.

Examples:
  PYTHONPATH=. python -m benchmarking.icc_data_preparation \
    --input /path/to/china_close_AAL_strategy-1_GSR.npy \
    --output-dir /path/to/icc_precomputed_fc \
    --print-timing

  PYTHONPATH=. python -m benchmarking.icc_data_preparation \
    --atlas AAL \
    --input-dir /path/to/timeseries_china/AAL \
    --output-dir /path/to/icc_precomputed_fc \
    --kinds corr pc tang glasso

  # Overwrite existing outputs for a single atlas
  PYTHONPATH=. python -m benchmarking.icc_data_preparation \
    --atlas AAL \
    --input-dir /path/to/timeseries_china/AAL \
    --output-dir /path/to/icc_precomputed_fc \
    --kinds corr pc tang glasso \
    --overwrite

  # Schaefer200 example with local data layout
  PYTHONPATH=. python -m benchmarking.icc_data_preparation \
    --atlas Schaefer200 \
    --input-dir ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/timeseries_china/Schaefer200 \
    --output-dir ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/icc_precomputed_fc \
    --kinds corr pc tang glasso

Note:
  Folder mode computes all close strategies for the atlas; --strategy/--gsr are ignored.
  Existing outputs are skipped by default; use --overwrite to recompute.
  Default coverage is IHB to keep edge counts consistent with precomputed glasso.

Glasso lookup:
  By default, glasso is searched under:
    <data_root>/glasso_precomputed_fc/china/<atlas>/
  and copied as: <stem>_glasso.npy → <output_dir>/<atlas>/<stem>_glasso.npy

Glasso-only example (copy + drop subject, with explicit precomputed path):
  PYTHONPATH=. python -m benchmarking.icc_data_preparation \
    --atlas AAL \
    --glasso-dir ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/glasso_precomputed_fc \
    --output-dir ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/icc_precomputed_fc \
    --kinds glasso \
    --drop-subject sub-3258811
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path
from typing import Iterable
import shutil

import numpy as np
from tqdm import tqdm

from benchmarking.fc import compute_fc_from_strategy_file
from benchmarking.hcpex_preprocess import preprocess_hcpex_timeseries
from benchmarking.project import resolve_data_root


def default_input_path(data_root: Path) -> Path:
    return data_root / "timeseries_china" / "AAL" / "china_close_AAL_strategy-1_GSR.npy"


def load_subject_order(strategy_path: Path) -> list[str] | None:
    path = strategy_path.parent / "subject_order_china.txt"
    if not path.exists():
        return None
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


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


def preprocess_hcpex_if_needed(
    timeseries: np.ndarray,
    atlas: str,
    data_root: str | Path | None,
) -> np.ndarray:
    """Apply HCPex preprocessing if the atlas is HCPex.

    For HCPex atlas, reduces China data from 421 ROIs to 373 ROIs by applying
    the HCPex mask. Other atlases are returned unchanged.

    Args:
        timeseries: Time series data (China format)
        atlas: Atlas name
        data_root: Data root directory

    Returns:
        Preprocessed time series (or original if not HCPex)
    """
    if atlas.upper() != "HCPEX":
        return timeseries

    # Apply HCPex preprocessing for China data
    try:
        data_root_path = Path(resolve_data_root(data_root))
        mask_path = data_root_path / "coverage" / "hcp_mask.npy"

        if not mask_path.exists():
            print(f"Warning: HCPex mask not found at {mask_path}, skipping preprocessing")
            return timeseries

        preprocessed = preprocess_hcpex_timeseries(
            timeseries,
            site="china",
            mask_path=mask_path,
        )
        print(f"   HCPex preprocessing: {timeseries.shape} -> {preprocessed.shape}")
        return preprocessed

    except Exception as e:
        print(f"Warning: HCPex preprocessing failed: {e}, using original data")
        return timeseries


def find_precomputed_glasso(glasso_dir: Path, stem: str) -> Path | None:
    if not glasso_dir.exists():
        raise FileNotFoundError(f"Glasso directory not found: {glasso_dir}")
    candidate = glasso_dir / f"{stem}_glasso.npy"
    if candidate.exists():
        return candidate
    matches = list(glasso_dir.rglob(f"{stem}_glasso.npy"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple precomputed glasso files found: {matches}")
    return matches[0]


def default_glasso_root(data_root: Path) -> Path:
    return data_root / "glasso_precomputed_fc"


def glasso_search_dir(glasso_root: Path, *, site: str, atlas: str) -> Path:
    """
    Return the directory where this script searches for precomputed glasso.

    Expected layout:
      <glasso_root>/<site>/<atlas>/*.npy
    """
    lowered = site.strip().lower()
    return glasso_root / lowered / atlas


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
    keep_idx, dropped = compute_keep_indices(subject_order, drop_list)
    if len(subject_order) != timeseries.shape[0]:
        raise ValueError(
            f"Subject order length {len(subject_order)} does not match data ({timeseries.shape[0]})."
        )
    if not dropped:
        return timeseries, subject_order, []
    filtered = timeseries[keep_idx]
    kept_subjects = [subject_order[idx] for idx in keep_idx]
    return filtered, kept_subjects, dropped


def compute_keep_indices(
    subject_order: list[str],
    drop_subjects: Iterable[str],
) -> tuple[list[int], list[str]]:
    drop_list = [s for s in drop_subjects if s]
    drop_set = set(drop_list)
    keep_idx = [idx for idx, sid in enumerate(subject_order) if sid not in drop_set]
    dropped = [sid for sid in subject_order if sid in drop_set]
    return keep_idx, dropped


def filter_precomputed_glasso(
    glasso: np.ndarray,
    subject_order: list[str],
    drop_subjects: Iterable[str],
) -> tuple[np.ndarray, list[str]]:
    keep_idx, dropped = compute_keep_indices(subject_order, drop_subjects)
    if not dropped:
        return glasso, []
    if glasso.shape[0] == len(subject_order):
        return glasso[keep_idx], dropped
    if glasso.shape[0] == len(keep_idx):
        return glasso, dropped
    raise ValueError(
        f"Glasso subject count {glasso.shape[0]} does not match subject order "
        f"({len(subject_order)}) or filtered count ({len(keep_idx)})."
    )


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
        help="Root folder with precomputed glasso matrices (default: <data_root>/glasso_precomputed_fc).",
    )
    parser.add_argument(
        "--atlas",
        default=None,
        help="Atlas name (e.g., AAL). Required for folder mode.",
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
        default="ihb",
        help="Coverage source for ROI filtering (default: ihb, to match glasso).",
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even if they already exist (default: skip existing).",
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=None,
        help="FC kinds to compute: corr, pc/partial, tang/tangent, glasso (default: all).",
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


def normalize_kinds(values: list[str] | None) -> list[str] | None:
    """
    Normalize FC-kind aliases to `compute_fc_from_strategy_file` keys.

    Returns list of keys from: corr, partial, tangent, glasso.
    """
    if values is None:
        return None
    if not values:
        return None
    mapping = {
        "corr": "corr",
        "correlation": "corr",
        "pc": "partial",
        "partial": "partial",
        "partial_corr": "partial",
        "partial-corr": "partial",
        "tang": "tangent",
        "tangent": "tangent",
        "glasso": "glasso",
    }
    allowed = {"corr", "partial", "tangent", "glasso"}
    normalized: list[str] = []
    for raw in values:
        key = mapping.get(str(raw).strip().lower())
        if key is None or key not in allowed:
            raise ValueError(
                f"Unknown FC kind: {raw}. Allowed: corr, pc/partial, tang/tangent, glasso."
            )
        if key not in normalized:
            normalized.append(key)
    return normalized


def copy_precomputed_glasso(
    precomputed: Path,
    output_path: Path,
    *,
    subject_order: list[str] | None,
    drop_subjects: Iterable[str],
) -> list[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drop_list = [s for s in drop_subjects if s]
    if not drop_list:
        shutil.copy2(precomputed, output_path)
        return []
    if subject_order is None:
        raise FileNotFoundError(
            "Subject order file not found; cannot drop subjects without ordering."
        )
    glasso = np.load(precomputed)
    filtered, dropped = filter_precomputed_glasso(glasso, subject_order, drop_list)
    np.save(output_path, filtered)
    return dropped


def process_series_folder(
    input_dir: Path,
    output_dir: Path,
    *,
    data_root: str | Path | None,
    coverage: str | None,
    coverage_threshold: float,
    drop_subjects: list[str],
    glasso_root: Path,
    print_timing: bool,
    strategy: str | None,
    gsr: str | None,
    atlas: str | None,
    kinds: list[str] | None = None,
    overwrite: bool = False,
) -> int:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if atlas is None or str(atlas).strip() == "":
        raise ValueError("--atlas is required in folder mode.")

    files = sorted(input_dir.rglob("*.npy"))
    if not files:
        print("No .npy files found.")
        return 0

    processed = 0
    skipped = 0
    copied = 0
    failed = 0

    coverage_arg = normalize_coverage_arg(coverage)
    if strategy is not None or gsr is not None:
        print("Folder mode: strategy/gsr filters are ignored; computing all close strategies.")
    if coverage_arg is not None:
        if coverage_threshold <= 0 or coverage_threshold >= 1:
            raise ValueError("coverage_threshold must be in (0, 1)")

    name_map_all = {
        "corr": "corr",
        "partial": "pc",
        "tangent": "tang",
        "glasso": "glasso",
    }
    kinds_normalized = normalize_kinds(kinds)
    if kinds_normalized is None:
        name_map = name_map_all
    else:
        name_map = {key: name_map_all[key] for key in kinds_normalized}

    china_files = [path for path in files if is_china_series(path)]
    if not china_files:
        raise ValueError("No China time-series files found. This script expects China data.")

    atlas_value = atlas.strip()
    precomputed_glasso_dir = glasso_search_dir(glasso_root, site="china", atlas=atlas_value)

    candidates = [
        path
        for path in china_files
        if "_close_" in path.name and parse_atlas(path) == atlas_value
    ]

    for path in tqdm(candidates, desc=f"China close ({atlas_value})", unit="file"):
        try:
            atlas = parse_atlas(path)
            atlas_out = output_dir / atlas
            atlas_out.mkdir(parents=True, exist_ok=True)

            out_paths = {
                key: atlas_out / f"{path.stem}_{suffix}.npy"
                for key, suffix in name_map.items()
            }
            if overwrite:
                missing_kinds = list(name_map.keys())
            else:
                missing_kinds = [key for key, out_path in out_paths.items() if not out_path.exists()]

            subject_order = load_subject_order(path)
            timeseries = np.load(path)

            # Apply HCPex preprocessing if needed (before subject filtering)
            timeseries = preprocess_hcpex_if_needed(timeseries, atlas, data_root)

            timeseries, _, dropped = filter_subjects(
                timeseries,
                subject_order,
                drop_subjects,
            )
            if dropped:
                print(f"{path.name}: dropped {', '.join(dropped)}")

            glasso_copied = False
            if "glasso" not in missing_kinds and drop_subjects:
                existing_glasso = out_paths.get("glasso")
                if existing_glasso is not None and existing_glasso.exists():
                    if subject_order is None:
                        raise FileNotFoundError(
                            "Subject order file not found; cannot drop subjects without ordering."
                        )
                    expected_keep, _ = compute_keep_indices(subject_order, drop_subjects)
                    glasso = np.load(existing_glasso)
                    if glasso.shape[0] == len(subject_order):
                        filtered, glasso_dropped = filter_precomputed_glasso(
                            glasso, subject_order, drop_subjects
                        )
                        np.save(existing_glasso, filtered)
                        glasso_copied = True
                        copied += 1
                        if glasso_dropped:
                            print(f"{path.name}: dropped {', '.join(glasso_dropped)} from glasso")
                        print(f"Updated glasso: {existing_glasso}")
                    elif glasso.shape[0] != len(expected_keep):
                        raise ValueError(
                            f"Glasso subject count {glasso.shape[0]} does not match subject order "
                            f"({len(subject_order)}) or filtered count ({len(expected_keep)})."
                        )
            if "glasso" in missing_kinds:
                precomputed = find_precomputed_glasso(precomputed_glasso_dir, path.stem)
                if precomputed is None:
                    print(f"Glasso not found for {path.stem} in {precomputed_glasso_dir}; skipping glasso.")
                    missing_kinds.remove("glasso")
                else:
                    glasso_copied = True
                    glasso_dropped = copy_precomputed_glasso(
                        precomputed,
                        out_paths["glasso"],
                        subject_order=subject_order,
                        drop_subjects=drop_subjects,
                    )
                    missing_kinds.remove("glasso")
                    copied += 1
                    if glasso_dropped:
                        print(f"{path.name}: dropped {', '.join(glasso_dropped)} from glasso")
                    print(f"Copied glasso: {out_paths['glasso']}")

            if not missing_kinds:
                if glasso_copied:
                    processed += 1
                else:
                    skipped += 1
                continue

            with tempfile.TemporaryDirectory() as tmp_dir:
                filtered_path = Path(tmp_dir) / path.name
                np.save(filtered_path, timeseries)
                fc = compute_fc_from_strategy_file(
                    filtered_path,
                    tangent_connectivity=None,
                    kinds=[k for k in missing_kinds if k != "glasso"],
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
    glasso_root = Path(args.glasso_dir).expanduser() if args.glasso_dir else default_glasso_root(data_root)
    if args.input_dir is None and args.input is None:
        if not args.atlas:
            raise ValueError("--atlas is required when running on a folder (default timeseries_china).")
        return process_series_folder(
            Path(data_root) / "timeseries_china" / args.atlas,
            output_dir_base,
            data_root=args.data_root,
            coverage=args.coverage,
            coverage_threshold=args.coverage_threshold,
            drop_subjects=args.drop_subject if args.drop_subject is not None else ["sub-3258811"],
            glasso_root=glasso_root,
            print_timing=args.print_timing,
            strategy=args.strategy,
            gsr=args.gsr,
            atlas=args.atlas,
            kinds=args.kinds,
            overwrite=args.overwrite,
        )
    if args.input_dir:
        if not args.atlas:
            raise ValueError("--atlas is required when running on a folder.")
        return process_series_folder(
            Path(args.input_dir).expanduser(),
            output_dir_base,
            data_root=args.data_root,
            coverage=args.coverage,
            coverage_threshold=args.coverage_threshold,
            drop_subjects=args.drop_subject if args.drop_subject is not None else ["sub-3258811"],
            glasso_root=glasso_root,
            print_timing=args.print_timing,
            strategy=args.strategy,
            gsr=args.gsr,
            atlas=args.atlas,
            kinds=args.kinds,
            overwrite=args.overwrite,
        )

    input_path = Path(args.input).expanduser() if args.input else default_input_path(data_root)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input time-series file: {input_path}")
    if not is_china_series(input_path):
        raise ValueError("This script supports China time-series inputs only.")

    atlas = parse_atlas(input_path)
    output_dir = output_dir_base / atlas
    output_dir.mkdir(parents=True, exist_ok=True)

    name_map_all = {
        "corr": "corr",
        "partial": "pc",
        "tangent": "tang",
        "glasso": "glasso",
    }
    kinds_normalized = normalize_kinds(args.kinds)
    if kinds_normalized is None:
        name_map = name_map_all
    else:
        name_map = {key: name_map_all[key] for key in kinds_normalized}
    out_paths = {
        key: output_dir / f"{input_path.stem}_{suffix}.npy"
        for key, suffix in name_map.items()
    }
    if args.overwrite:
        missing_kinds = list(name_map.keys())
    else:
        missing_kinds = [key for key, path in out_paths.items() if not path.exists()]
        if not missing_kinds:
            print("All requested FC outputs already exist. Skipping computation.")
            return 0

    if args.coverage_threshold <= 0 or args.coverage_threshold >= 1:
        raise ValueError("coverage_threshold must be in (0, 1)")

    drop_subjects = args.drop_subject if args.drop_subject is not None else ["sub-3258811"]
    subject_order = load_subject_order(input_path)

    timeseries = np.load(input_path)

    # Apply HCPex preprocessing if needed (before subject filtering)
    timeseries = preprocess_hcpex_if_needed(timeseries, atlas, args.data_root)

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

    if "glasso" in missing_kinds:
        precomputed_dir = glasso_search_dir(glasso_root, site="china", atlas=atlas)
        precomputed = find_precomputed_glasso(precomputed_dir, input_path.stem)
        if precomputed is None:
            print(f"Glasso not found for {input_path.stem} in {precomputed_dir}; skipping glasso.")
            missing_kinds.remove("glasso")
        else:
            glasso_dropped = copy_precomputed_glasso(
                precomputed,
                out_paths["glasso"],
                subject_order=subject_order,
                drop_subjects=drop_subjects,
            )
            missing_kinds.remove("glasso")
            if glasso_dropped:
                print(f"{input_path.name}: dropped {', '.join(glasso_dropped)} from glasso")
            print(f"Copied glasso: {out_paths['glasso']}")
    elif drop_subjects:
        existing_glasso = out_paths.get("glasso")
        if existing_glasso is not None and existing_glasso.exists():
            if subject_order is None:
                raise FileNotFoundError(
                    "Subject order file not found; cannot drop subjects without ordering."
                )
            expected_keep, _ = compute_keep_indices(subject_order, drop_subjects)
            glasso = np.load(existing_glasso)
            if glasso.shape[0] == len(subject_order):
                filtered, glasso_dropped = filter_precomputed_glasso(
                    glasso, subject_order, drop_subjects
                )
                np.save(existing_glasso, filtered)
                if glasso_dropped:
                    print(f"{input_path.name}: dropped {', '.join(glasso_dropped)} from glasso")
                print(f"Updated glasso: {existing_glasso}")
            elif glasso.shape[0] != len(expected_keep):
                raise ValueError(
                    f"Glasso subject count {glasso.shape[0]} does not match subject order "
                    f"({len(subject_order)}) or filtered count ({len(expected_keep)})."
                )

    if not missing_kinds:
        return 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        filtered_path = Path(tmp_dir) / input_path.name
        np.save(filtered_path, timeseries)
        fc = compute_fc_from_strategy_file(
            filtered_path,
            tangent_connectivity=None,
            kinds=[k for k in missing_kinds if k != "glasso"],
            coverage=coverage_arg,
            coverage_threshold=args.coverage_threshold,
            data_path=args.data_root,
            print_timing=args.print_timing,
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
