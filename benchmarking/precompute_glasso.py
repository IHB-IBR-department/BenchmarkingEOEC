#!/usr/bin/env python3
"""
Precompute glasso FC matrices from time-series files in a folder tree.

Skips outputs that already exist and preserves the input folder structure
under the output directory.

Examples:
  # Precompute glasso for a time-series tree (default coverage=ihb)
  PYTHONPATH=. python -m benchmarking.precompute_glasso \
    --input-dir ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/timeseries_ihb/AAL \
    --output-dir ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/glasso_precomputed_fc \
    --print-timing

  # Precompute glasso for a single file
  PYTHONPATH=. python -m benchmarking.precompute_glasso \
    --input ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/timeseries_ihb/Schaefer200/ihb_close_Schaefer200_strategy-1_GSR.npy \
    --output-dir ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/glasso_precomputed_fc \
    --print-timing

  # Use an explicit coverage source and threshold
  PYTHONPATH=. python -m benchmarking.precompute_glasso \
    --input-dir ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/timeseries_china \
    --output-dir ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/glasso_precomputed_fc \
    --coverage china \
    --coverage-threshold 0.1
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np

from benchmarking.fc import ConnectomeTransformer, _resolve_coverage_mask
from benchmarking.hcpex_preprocess import preprocess_hcpex_timeseries
from benchmarking.project import resolve_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute glasso FC matrices for all .npy time-series files in a folder.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="File or folder with time-series .npy files.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Deprecated alias for --input.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output base directory (default: <input_dir_parent>/glasso_precomputed_fc).",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to OpenCloseBenchmark_data (optional if OPEN_CLOSE_BENCHMARK_DATA is set).",
    )
    parser.add_argument(
        "--coverage",
        default="ihb",
        help="Coverage source for ROI filtering (site/both/ihb/china/path). Default: ihb.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.1,
        help="Coverage threshold for ROI filtering (default: 0.1).",
    )
    parser.add_argument(
        "--glasso-lambda",
        type=float,
        default=0.03,
        help="L1 regularization parameter for glasso (default: 0.03).",
    )
    parser.add_argument(
        "--no-vectorize",
        action="store_true",
        help="Save full matrices instead of vectorized upper triangle.",
    )
    parser.add_argument(
        "--keep-diagonal",
        action="store_true",
        help="Keep diagonal when vectorizing (default: discard).",
    )
    parser.add_argument(
        "--print-timing",
        action="store_true",
        help="Print computation time per file.",
    )
    return parser.parse_args()


def normalize_coverage_arg(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in ("none", "off", "false", "no"):
        return None
    return value


def iter_timeseries_files(input_path: Path, output_dir: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix != ".npy":
            return []
        if output_dir in input_path.parents:
            return []
        if input_path.stem.endswith("_glasso"):
            return []
        return [input_path]

    files = []
    for path in sorted(input_path.rglob("*.npy")):
        if output_dir in path.parents:
            continue
        if path.stem.endswith("_glasso"):
            continue
        files.append(path)
    return files


def parse_atlas_and_site(path: Path) -> tuple[str | None, str | None]:
    """Extract atlas name and site from time series file path.

    Returns:
        Tuple of (atlas, site) where each can be None if not found
    """
    name = path.name

    # Try to extract from filename pattern: {site}_{condition}_{atlas}_strategy-...
    match = re.match(r"^(ihb|china)_(?:close|open)\d*_(?P<atlas>[^_]+)_strategy-", name)
    if match:
        return match.group("atlas"), match.group(1)

    # Try to extract atlas from parent directory
    atlas = None
    if path.parent.name not in ("timeseries_ihb", "timeseries_china"):
        atlas = path.parent.name

    # Try to extract site from path
    site = None
    if name.startswith("ihb_"):
        site = "ihb"
    elif name.startswith("china_"):
        site = "china"
    elif "timeseries_ihb" in path.parts:
        site = "ihb"
    elif "timeseries_china" in path.parts:
        site = "china"

    return atlas, site


def preprocess_timeseries_if_needed(
    timeseries: np.ndarray,
    path: Path,
    data_root: str | None,
) -> tuple[np.ndarray, bool]:
    """Apply HCPex preprocessing if the file is for HCPex atlas.

    Args:
        timeseries: Raw time series data
        path: Path to the time series file
        data_root: Data root directory

    Returns:
        Tuple of (preprocessed timeseries, was_hcpex_preprocessed)
        The boolean flag indicates whether HCPex preprocessing was applied
    """
    atlas, site = parse_atlas_and_site(path)

    if atlas is None or site is None:
        return timeseries, False

    if atlas.upper() != "HCPEX":
        return timeseries, False

    if site not in ("ihb", "china"):
        print(f"Warning: Unknown site '{site}' for HCPex preprocessing, skipping")
        return timeseries, False

    # Apply HCPex preprocessing
    try:
        data_root_path = resolve_data_root(data_root)
        mask_path = Path(data_root_path) / "coverage" / "hcp_mask.npy"

        if not mask_path.exists():
            print(f"Warning: HCPex mask not found at {mask_path}, skipping preprocessing")
            return timeseries, False

        preprocessed = preprocess_hcpex_timeseries(
            timeseries,
            site=site,
            mask_path=mask_path,
        )
        print(f"   HCPex preprocessing: {timeseries.shape} -> {preprocessed.shape}")
        return preprocessed, True

    except Exception as e:
        print(f"Warning: HCPex preprocessing failed: {e}, using original data")
        return timeseries, False


def compute_glasso(
    timeseries: np.ndarray,
    *,
    vectorize: bool,
    discard_diagonal: bool,
    glasso_lambda: float,
) -> np.ndarray:
    transformer = ConnectomeTransformer(
        kind="glasso",
        vectorize=vectorize,
        discard_diagonal=discard_diagonal,
        glasso_lambda=glasso_lambda,
    )
    if timeseries.ndim == 4:
        sessions = [timeseries[..., idx] for idx in range(timeseries.shape[-1])]
        outputs = [transformer.fit_transform(ts) for ts in sessions]
        return np.stack(outputs, axis=-1)
    return transformer.fit_transform(timeseries)


def main() -> int:
    args = parse_args()

    if args.input and args.input_dir:
        raise ValueError("Use only one of --input or --input-dir.")

    input_value = args.input or args.input_dir
    if not input_value:
        raise ValueError("Missing --input (or --input-dir).")

    input_path = Path(input_value).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    input_root = input_path if input_path.is_dir() else input_path.parent

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else input_root.parent / "glasso_precomputed_fc"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage_arg = normalize_coverage_arg(args.coverage)
    if coverage_arg is not None:
        if args.coverage_threshold <= 0 or args.coverage_threshold >= 1:
            raise ValueError("coverage_threshold must be in (0, 1)")

    files = iter_timeseries_files(input_path, output_dir)
    if not files:
        print("No .npy files found.")
        return 0

    processed = 0
    skipped = 0
    failed = 0

    for path in files:
        rel = path.relative_to(input_root)
        out_dir = output_dir / rel.parent
        out_path = out_dir / f"{path.stem}_glasso.npy"

        if out_path.exists():
            skipped += 1
            continue

        try:
            timeseries = np.load(path)
            if timeseries.ndim not in (3, 4):
                print(f"Skip {path} (unexpected shape {timeseries.shape})")
                skipped += 1
                continue

            # Apply HCPex preprocessing if needed (before coverage masking)
            # HCPex preprocessing already includes masking, so skip coverage mask if applied
            timeseries, hcpex_preprocessed = preprocess_timeseries_if_needed(timeseries, path, args.data_root)

            # Only apply coverage masking if not HCPex (HCPex preprocessing already includes masking)
            if coverage_arg is not None and not hcpex_preprocessed:
                mask = _resolve_coverage_mask(
                    strategy_path=path,
                    coverage=coverage_arg,
                    coverage_threshold=args.coverage_threshold,
                    data_path=args.data_root,
                )
                if mask is not None:
                    if mask.shape[0] != timeseries.shape[2]:
                        raise ValueError(
                            f"Coverage mask length {mask.shape[0]} does not match ROI count {timeseries.shape[2]}"
                        )
                    if not mask.any():
                        raise ValueError("Coverage mask removed all ROIs.")
                    if timeseries.ndim == 4:
                        timeseries = timeseries[:, :, mask, :]
                    else:
                        timeseries = timeseries[:, :, mask]

            start = time.perf_counter()
            output = compute_glasso(
                timeseries,
                vectorize=not args.no_vectorize,
                discard_diagonal=not args.keep_diagonal,
                glasso_lambda=args.glasso_lambda,
            )
            elapsed = time.perf_counter() - start

            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_path, output)
            processed += 1
            if args.print_timing:
                print(f"{path} -> {out_path} ({output.shape}) in {elapsed:.3f}s")
            else:
                print(f"Saved {out_path} ({output.shape})")
        except Exception as exc:
            failed += 1
            print(f"Failed {path}: {exc}")

    print(f"Done. Processed={processed} skipped={skipped} failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
