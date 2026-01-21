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

  # Use an explicit coverage source and threshold
  PYTHONPATH=. python -m benchmarking.precompute_glasso \
    --input-dir ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/timeseries_china \
    --output-dir ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/glasso_precomputed_fc \
    --coverage china \
    --coverage-threshold 0.1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from benchmarking.fc import ConnectomeTransformer, _resolve_coverage_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute glasso FC matrices for all .npy time-series files in a folder.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder with time-series .npy files (searched recursively).",
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


def iter_timeseries_files(input_dir: Path, output_dir: Path) -> list[Path]:
    files = []
    for path in sorted(input_dir.rglob("*.npy")):
        if output_dir in path.parents:
            continue
        if path.stem.endswith("_glasso"):
            continue
        files.append(path)
    return files


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

    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else input_dir.parent / "glasso_precomputed_fc"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage_arg = normalize_coverage_arg(args.coverage)
    if coverage_arg is not None:
        if args.coverage_threshold <= 0 or args.coverage_threshold >= 1:
            raise ValueError("coverage_threshold must be in (0, 1)")

    files = iter_timeseries_files(input_dir, output_dir)
    if not files:
        print("No .npy files found.")
        return 0

    processed = 0
    skipped = 0
    failed = 0

    for path in files:
        rel = path.relative_to(input_dir)
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

            if coverage_arg is not None:
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
