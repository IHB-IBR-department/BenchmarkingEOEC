#!/usr/bin/env python3
"""
Edge-wise ICC computation from precomputed FC matrices.

Expected inputs are vectorized connectomes with a session axis, typically
produced by `benchmarking/icc_data_preparation.py` and stored under
`icc_precomputed_fc`.

Supported ICC variants (per ICCcomputation.md):
- ICC(3,1): two-way mixed, consistency (default)
- ICC(2,1): two-way random, absolute agreement
- ICC(1,1): one-way random, consistency with session effects in error

Examples:
  # Summary only (default: no edge-wise outputs saved)
  PYTHONPATH=. python -m benchmarking.icc_computation --input ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/icc_precomputed_fc/AAL --icc icc31 icc21 --summary-json icc_results/icc_summary.json

  # Save edge-wise ICC outputs for all files under icc_precomputed_fc
  PYTHONPATH=. python -m benchmarking.icc_computation --input ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/icc_precomputed_fc --save-edgewise

  # Save edge-wise ICC outputs for a single atlas subtree
  PYTHONPATH=. python -m benchmarking.icc_computation --input ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/icc_precomputed_fc/AAL --icc icc31 icc21 --save-edgewise

  # Save edge-wise ICC(1,1) only
  PYTHONPATH=. python -m benchmarking.icc_computation --input ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/icc_precomputed_fc/AAL --icc icc11 --save-edgewise

  # Save edge-wise ICC(3,1) with masking (group-averaged edge-weight percentile)
  PYTHONPATH=. python -m benchmarking.icc_computation --input ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/icc_precomputed_fc/AAL --icc icc31 --mask --mask-percentile 98 --discard-diagonal --save-edgewise
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from nilearn.connectome import sym_matrix_to_vec

from benchmarking.project import resolve_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute edge-wise ICC from precomputed FC matrices.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="File or directory with FC matrices (default: <data_root>/icc_precomputed_fc).",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Deprecated alias for --input.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for edge-wise ICC vectors (default: ./icc_results).",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to OpenCloseBenchmark_data (optional if OPEN_CLOSE_BENCHMARK_DATA is set).",
    )
    parser.add_argument(
        "--icc",
        nargs="+",
        default=["icc31"],
        help="ICC variants to compute: icc31, icc21, icc11 (default: icc31).",
    )
    parser.add_argument(
        "--discard-diagonal",
        action="store_true",
        help="Discard diagonal when vectorizing 4D matrices.",
    )
    parser.add_argument(
        "--pattern",
        default="*.npy",
        help="Glob pattern for input files (default: *.npy).",
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Apply ICC masking using group-averaged absolute edge weights.",
    )
    parser.add_argument(
        "--mask-percentile",
        type=float,
        default=98,
        help="Percentile for ICC masking threshold (default: 98).",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Write per-file average ICC summaries (masked + unmasked) to JSON.",
    )
    parser.add_argument(
        "--save-edgewise",
        action="store_true",
        help="Save edge-wise ICC vectors (default: disabled).",
    )
    return parser.parse_args()


def normalize_icc_list(items: Iterable[str]) -> list[str]:
    allowed = {"icc31", "icc21", "icc11"}
    normalized = []
    for item in items:
        item = item.lower().strip()
        if item not in allowed:
            raise ValueError(f"Unknown ICC variant: {item}. Allowed: {sorted(allowed)}")
        normalized.append(item)
    return sorted(set(normalized), key=normalized.index)


def parse_strategy_from_name(path: Path) -> dict[str, str] | None:
    stem = path.stem
    if "_strategy-" not in stem:
        return None
    prefix, tail = stem.split("_strategy-", 1)
    tail_parts = tail.split("_")
    if len(tail_parts) < 3:
        return None
    strategy = tail_parts[0]
    gsr = tail_parts[1]
    fc = "_".join(tail_parts[2:])

    prefix_parts = prefix.split("_")
    if len(prefix_parts) < 3:
        return None
    site = prefix_parts[0]
    condition = prefix_parts[1]
    atlas = "_".join(prefix_parts[2:])

    return {
        "site": site,
        "condition": condition,
        "atlas": atlas,
        "strategy": strategy,
        "gsr": gsr,
        "fc": fc,
    }


def vectorize_sessions(arr: np.ndarray, discard_diagonal: bool) -> np.ndarray:
    if arr.ndim == 3:
        return arr
    if arr.ndim != 4:
        raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}")
    n_subjects, _, _, n_sessions = arr.shape
    edges = []
    for idx in range(n_sessions):
        edges.append(sym_matrix_to_vec(arr[:, :, :, idx], discard_diagonal=discard_diagonal))
    return np.stack(edges, axis=-1).reshape(n_subjects, -1, n_sessions)


def compute_icc_edgewise(data: np.ndarray, icc_kind: str) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError(f"Expected data shape (n_subjects, n_edges, n_sessions), got {data.shape}")
    n_subjects, n_edges, n_sessions = data.shape
    if n_sessions < 2:
        raise ValueError("ICC requires at least 2 sessions.")

    y = np.asarray(data, dtype=np.float64).transpose(0, 2, 1)  # (n, k, e)
    valid_pair = np.all(np.isfinite(y), axis=1)  # (n, e)
    if not np.all(valid_pair):
        y = np.where(valid_pair[:, None, :], y, np.nan)

    n_eff = valid_pair.sum(axis=0).astype(float)  # (e,)
    k = float(n_sessions)

    subject_mean = np.nanmean(y, axis=1)  # (n, e)
    session_mean = np.nanmean(y, axis=0)  # (k, e)
    grand_mean = np.nanmean(y, axis=(0, 1))  # (e,)

    with np.errstate(invalid="ignore", divide="ignore"):
        msr_num = k * np.nansum((subject_mean - grand_mean) ** 2, axis=0)
        msr = msr_num / (n_eff - 1)

        msc_num = n_eff * np.nansum((session_mean - grand_mean) ** 2, axis=0)
        msc = msc_num / (k - 1)

        msw_num = np.nansum((y - subject_mean[:, None, :]) ** 2, axis=(0, 1))
        msw = msw_num / (n_eff * (k - 1))

        resid = y - subject_mean[:, None, :] - session_mean[None, :, :] + grand_mean[None, None, :]
        mse_num = np.nansum(resid ** 2, axis=(0, 1))
        mse = mse_num / ((n_eff - 1) * (k - 1))

        if icc_kind == "icc31":
            denom = msr + (k - 1) * mse
            icc = (msr - mse) / denom
        elif icc_kind == "icc21":
            denom = msr + (k - 1) * mse + (k * (msc - mse) / n_eff)
            icc = (msr - mse) / denom
        elif icc_kind == "icc11":
            denom = msr + (k - 1) * msw
            icc = (msr - msw) / denom
        else:
            raise ValueError(f"Unknown ICC kind: {icc_kind}")

    icc = np.where(n_eff >= 2, icc, np.nan)
    return icc.astype(np.float32)


def compute_icc_mask(data: np.ndarray, percentile: float = 98.0) -> np.ndarray:
    """
    Compute a boolean mask over edges using group-averaged absolute weights.

    Parameters
    ----------
    data : np.ndarray
        Vectorized correlation data with shape (n_subjects, n_edges, n_sessions).
    percentile : float, default=98.0
        Percentile level used to derive the mask from edge means.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (n_edges,). False means the edge's mean absolute
        weight (averaged across subjects and sessions) falls below the global
        percentile threshold.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected data shape (n_subjects, n_edges, n_sessions), got {data.shape}")
    abs_data = np.abs(data)
    edge_means = abs_data.mean(axis=(0, 2))
    global_threshold = np.percentile(abs_data, percentile)
    return edge_means >= global_threshold


def _safe_nanmean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    if not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmean(values))


def compute_icc_summary(
    data: np.ndarray,
    icc_list: Iterable[str],
    percentile: float = 98.0,
    mask: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    if mask is None:
        mask = compute_icc_mask(data, percentile=percentile)
    summary: dict[str, dict[str, float]] = {}
    for icc_kind in icc_list:
        icc_vec = compute_icc_edgewise(data, icc_kind)
        summary[icc_kind] = {
            "mean": _safe_nanmean(icc_vec),
            "mean_masked": _safe_nanmean(icc_vec[mask]),
        }
    return summary


def _summary_key(entry: dict[str, str]) -> tuple[str, str, str, str]:
    atlas = entry["atlas"]
    strategy = f"strategy-{entry['strategy']}"
    gsr = entry["gsr"]
    fc = entry["fc"]
    return atlas, strategy, gsr, fc


def main() -> int:
    args = parse_args()
    icc_list = normalize_icc_list(args.icc)

    if args.input and args.input_dir:
        raise ValueError("Use only one of --input or --input-dir.")

    data_root = resolve_data_root(args.data_root)
    input_value = args.input or args.input_dir
    input_path = Path(input_value).expanduser() if input_value else Path(data_root) / "icc_precomputed_fc"
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path.cwd() / "icc_results"

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        files = [input_path]
        base_dir = input_path.parent
    else:
        files = sorted(input_path.rglob(args.pattern))
        base_dir = input_path
    if not files:
        print("No files found.")
        return 0

    if not args.save_edgewise and not args.summary_json:
        print("Nothing to do: use --save-edgewise and/or --summary-json.")
        return 0

    processed = 0
    skipped = 0
    failed = 0
    summaries: dict[str, object] = {}

    for path in files:
        if path.is_dir():
            continue
        rel = path.relative_to(base_dir)
        out_dir = None
        if args.save_edgewise:
            out_dir = output_dir / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)

        try:
            arr = np.load(path)
            if arr.ndim not in (3, 4):
                print(f"Skip {path} (unexpected shape {arr.shape})")
                skipped += 1
                continue

            data = vectorize_sessions(arr, discard_diagonal=args.discard_diagonal)
            if data.shape[-1] < 2:
                print(f"Skip {path} (needs >=2 sessions, got {data.shape[-1]})")
                skipped += 1
                continue

            mask = None
            if args.mask or args.summary_json:
                mask = compute_icc_mask(data, percentile=args.mask_percentile)

            if args.summary_json:
                summary_icc_list = icc_list
                if "icc11" not in summary_icc_list:
                    summary_icc_list = summary_icc_list + ["icc11"]
                summary = compute_icc_summary(
                    data,
                    summary_icc_list,
                    percentile=args.mask_percentile,
                    mask=mask,
                )
                meta = parse_strategy_from_name(path)
                if meta is None:
                    summaries[str(rel)] = {
                        "n_subjects": int(data.shape[0]),
                        "n_edges": int(data.shape[1]),
                        "n_sessions": int(data.shape[2]),
                        "mask_percentile": float(args.mask_percentile),
                        "icc": summary,
                    }
                else:
                    atlas, strategy, gsr, fc = _summary_key(meta)
                    atlas_entry = summaries.setdefault(atlas, {})
                    strategy_entry = atlas_entry.setdefault(strategy, {})
                    gsr_entry = strategy_entry.setdefault(gsr, {})
                    gsr_entry[fc] = {
                        "site": meta["site"],
                        "condition": meta["condition"],
                        "n_subjects": int(data.shape[0]),
                        "n_edges": int(data.shape[1]),
                        "n_sessions": int(data.shape[2]),
                        "mask_percentile": float(args.mask_percentile),
                        "icc": summary,
                    }

            if args.save_edgewise:
                for icc_kind in icc_list:
                    mask_tag = ""
                    if args.mask:
                        percentile_label = str(args.mask_percentile).replace(".", "p")
                        mask_tag = f"_mask{percentile_label}"
                    out_path = out_dir / f"{path.stem}_{icc_kind}{mask_tag}.npy"
                    if out_path.exists():
                        skipped += 1
                        continue
                    icc = compute_icc_edgewise(data, icc_kind)
                    if args.mask:
                        icc = np.where(mask, icc, np.nan)
                    np.save(out_path, icc)
                    processed += 1
                    print(f"Saved {out_path} ({icc.shape})")
        except Exception as exc:
            failed += 1
            print(f"Failed {path}: {exc}")

    if args.summary_json and summaries:
        summary_path = Path(args.summary_json).expanduser()
        if summary_path.suffix.lower() != ".json":
            summary_path = summary_path / "icc_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=2)
        print(f"Saved summary JSON to {summary_path}")

    print(f"Done. Processed={processed} skipped={skipped} failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
