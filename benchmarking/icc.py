#!/usr/bin/env python3
"""
ICC End-to-End Runner (China close test-retest)
==============================================

Computes ICC summaries from precomputed, vectorized FC data.

YAML config (top-level keys, similar to qc_fc.py):
  data_source:
    fc: "~/.../icc_precomputed_fc"
  atlases: [AAL, Schaefer200, Brainnetome, HCPex]
  strategies: all
  fc_types: [corr, pc, tang, glasso]
  icc: [icc31, icc21, icc11]
  drop_subjects: [sub-3258811]
  mask: true
  mask_percentile: 95
  save_edgewise: false
  edgewise_output_dir: "icc_results"  # writes *_edgewise_icc_all.pkl (+ *_masked.pkl when mask=true)
  output: "icc_results/icc_summary.csv"
  pattern: "*.npy"

Usage:
  source venv/bin/activate && PYTHONPATH=. python -m benchmarking.icc --config configs/icc_atlas.yaml
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from benchmarking.icc_data_preparation import normalize_kinds


# =============================================================================
# Core ICC computation
# =============================================================================

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
    fc = tail_parts[-1]
    gsr = tail_parts[-2]
    strategy = "_".join(tail_parts[:-2])
    if not strategy:
        return None

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


def compute_icc_mask(data: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """
    Compute a boolean mask over edges using group-averaged absolute weights.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected data shape (n_subjects, n_edges, n_sessions), got {data.shape}")
    abs_data = np.abs(data)
    edge_means = abs_data.mean(axis=(0, 2))
    global_threshold = np.percentile(edge_means, percentile)
    return edge_means >= global_threshold


def _safe_nanmean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    if not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmean(values))


def _safe_nanstd(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    if not np.isfinite(values).any():
        return float("nan")
    return float(np.nanstd(values))


def compute_icc_summary(
    data: np.ndarray,
    icc_list: Iterable[str],
    percentile: float = 95.0,
    mask: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    if mask is None:
        mask = compute_icc_mask(data, percentile=percentile)
    summary: dict[str, dict[str, float]] = {}
    for icc_kind in icc_list:
        icc_vec = compute_icc_edgewise(data, icc_kind)
        summary[icc_kind] = {
            "mean": _safe_nanmean(icc_vec),
            "std": _safe_nanstd(icc_vec),
            "mean_masked": _safe_nanmean(icc_vec[mask]),
            "std_masked": _safe_nanstd(icc_vec[mask]),
        }
    return summary


def _filter_subjects_by_order(
    data: np.ndarray,
    subject_order: list[str],
    drop_subjects: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    drop_list = [s for s in drop_subjects if s]
    if not drop_list:
        return data, subject_order, []

    drop_set = set(drop_list)
    present = [sid for sid in subject_order if sid in drop_set]
    expected_len = len(subject_order) - len(present)
    if data.shape[0] != len(subject_order):
        if expected_len == data.shape[0]:
            return data, subject_order, []
        raise ValueError(
            f"Subject order length {len(subject_order)} does not match data ({data.shape[0]})."
        )

    keep_idx = [idx for idx, sid in enumerate(subject_order) if sid not in drop_set]
    dropped = [sid for sid in subject_order if sid in drop_set]
    if not dropped:
        return data, subject_order, []
    filtered = data[keep_idx]
    kept_subjects = [subject_order[idx] for idx in keep_idx]
    return filtered, kept_subjects, dropped


def _find_subject_order_path(
    *,
    atlas: str,
    fc_dir: Path | None,
) -> Path | None:
    candidates: list[Path] = []
    if fc_dir is not None:
        candidates.append(fc_dir / atlas / "subject_order_china.txt")
        candidates.append(fc_dir / "subject_order_china.txt")

    for path in candidates:
        if path.exists():
            return path
    return None


def _build_subject_order_lookup(
    *,
    fc_dir: Path | None,
) -> Callable[[Path], list[str] | None]:
    cache: dict[str, list[str] | None] = {}

    def lookup(path: Path) -> list[str] | None:
        meta = parse_strategy_from_name(path)
        if meta is None:
            return None
        atlas = meta["atlas"]
        if atlas in cache:
            return cache[atlas]
        order_path = _find_subject_order_path(
            atlas=atlas, fc_dir=fc_dir
        )
        if order_path is None:
            cache[atlas] = None
            return None
        order = [line.strip() for line in order_path.read_text().splitlines() if line.strip()]
        cache[atlas] = order
        return order

    return lookup


def run_icc_computation(
    *,
    input_path: Path,
    icc_list: list[str],
    pattern: str = "*.npy",
    mask: bool = False,
    mask_percentile: float = 95.0,
    save_edgewise: bool = False,
    edgewise_output_dir: Path | None = None,
    allowed_atlases: list[str] | None = None,
    allowed_strategies: list[str] | None = None,
    fc_suffixes: list[str] | None = None,
    drop_subjects: list[str] | None = None,
    subject_order_lookup: Callable[[Path], list[str] | None] | None = None,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.rglob(pattern))
    files = [path for path in files if path.is_file()]

    if fc_suffixes:
        files = [
            path for path in files
            if any(path.name.endswith(f"_{suffix}.npy") for suffix in fc_suffixes)
        ]
    if allowed_atlases:
        atlas_set = set(allowed_atlases)
        filtered = []
        for path in files:
            meta = parse_strategy_from_name(path)
            if meta is None:
                continue
            if meta["atlas"] in atlas_set:
                filtered.append(path)
        files = filtered
    if allowed_strategies:
        strategy_set = set(allowed_strategies)
        filtered = []
        for path in files:
            meta = parse_strategy_from_name(path)
            if meta is None:
                continue
            if meta["strategy"] in strategy_set:
                filtered.append(path)
        files = filtered

    summary_icc_list = list(icc_list)
    if "icc11" not in summary_icc_list:
        summary_icc_list.append("icc11")
    base_columns = ["site", "condition", "atlas", "strategy", "gsr", "fc_type"]
    stat_columns: list[str] = []
    for icc_kind in summary_icc_list:
        stat_columns.extend(
            [
                f"{icc_kind}_mean",
                f"{icc_kind}_std",
                f"{icc_kind}_mean_masked",
                f"{icc_kind}_std_masked",
            ]
        )

    if not files:
        return pd.DataFrame(columns=base_columns + stat_columns)

    summary_rows: list[dict[str, object]] = []
    edgewise_all: dict[tuple[str, str, str], dict[str, dict[str, np.ndarray]]] = {}
    edgewise_masked: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    drop_list = [str(x) for x in (drop_subjects or []) if str(x).strip()]
    if drop_list and subject_order_lookup is None:
        raise ValueError("drop_subjects provided but no subject_order_lookup available.")

    for path in tqdm(files, desc="Computing ICC", unit="file"):
        if save_edgewise and edgewise_output_dir is None:
            raise ValueError("edgewise_output_dir is required when save_edgewise=True")

        arr = np.load(path)
        if arr.ndim != 3:
            print(f"Skip {path} (expected vectorized shape (n_subjects, n_edges, n_sessions), got {arr.shape})")
            continue

        if drop_list:
            subject_order = subject_order_lookup(path) if subject_order_lookup else None
            if subject_order is not None:
                arr, _, _ = _filter_subjects_by_order(arr, subject_order, drop_list)

        data = arr
        if data.shape[-1] < 2:
            continue

        edge_mask = compute_icc_mask(data, percentile=mask_percentile)
        summary = compute_icc_summary(
            data,
            summary_icc_list,
            percentile=mask_percentile,
            mask=edge_mask,
        )
        meta = parse_strategy_from_name(path)
        if meta is None:
            print(f"Skip summary row for {path} (unable to parse strategy metadata).")
        else:
            row = {
                "site": meta["site"],
                "condition": meta["condition"],
                "atlas": meta["atlas"],
                "strategy": meta["strategy"],
                "gsr": meta["gsr"],
                "fc_type": meta["fc"],
            }
            for icc_kind in summary_icc_list:
                stats = summary.get(icc_kind)
                if stats is None:
                    continue
                row[f"{icc_kind}_mean"] = stats.get("mean", float("nan"))
                row[f"{icc_kind}_std"] = stats.get("std", float("nan"))
                row[f"{icc_kind}_mean_masked"] = stats.get("mean_masked", float("nan"))
                row[f"{icc_kind}_std_masked"] = stats.get("std_masked", float("nan"))
            summary_rows.append(row)

        if save_edgewise:
            meta = parse_strategy_from_name(path)
            if meta is None:
                print(f"Skip edgewise output for {path} (unable to parse strategy metadata).")
                continue
            key = (meta["atlas"], meta["strategy"], meta["gsr"])
            fc_type = meta["fc"]
            key_bucket = edgewise_all.setdefault(key, {})
            fc_bucket = key_bucket.setdefault(fc_type, {})
            masked_icc_bucket = None
            if mask and edge_mask is not None:
                masked_bucket = edgewise_masked.setdefault(key, {})
                masked_fc_bucket = masked_bucket.setdefault(
                    fc_type,
                    {"mask": edge_mask, "icc": {}},
                )
                masked_fc_bucket["mask"] = edge_mask
                masked_icc_bucket = masked_fc_bucket.setdefault("icc", {})
            for icc_kind in icc_list:
                icc_vec = compute_icc_edgewise(data, icc_kind)
                fc_bucket[icc_kind] = icc_vec
                if masked_icc_bucket is not None:
                    masked_icc_bucket[icc_kind] = icc_vec[edge_mask]

    if save_edgewise and edgewise_output_dir is not None and edgewise_all:
        edgewise_output_dir.mkdir(parents=True, exist_ok=True)
        for (atlas, strategy, gsr), fc_map in edgewise_all.items():
            base_name = f"{atlas}_strategy-{strategy}_{gsr}_edgewise_icc"
            all_path = edgewise_output_dir / f"{base_name}_all.pkl"
            with all_path.open("wb") as handle:
                pickle.dump(fc_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if mask:
                masked_map = edgewise_masked.get((atlas, strategy, gsr))
                if masked_map:
                    masked_path = edgewise_output_dir / f"{base_name}_masked.pkl"
                    with masked_path.open("wb") as handle:
                        pickle.dump(masked_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pd.DataFrame(summary_rows, columns=base_columns + stat_columns)


# =============================================================================
# Config-driven runner
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ICC preparation + computation from a YAML config.",
    )
    parser.add_argument("--config", required=True, help="Path to ICC YAML config.")
    return parser.parse_args()


def _as_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "all":
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping/dict at top level.")
    return cfg


def _resolve_output_path(output_value: str | Path | None) -> Path:
    output = Path(output_value or "icc_results/icc_summary.csv").expanduser()
    if output.suffix.lower() == ".csv":
        return output
    if output.suffix:
        return output.with_suffix(".csv")
    return output / "icc_summary.csv"


def run_icc_from_config(config: dict[str, Any]) -> pd.DataFrame:
    data_source = config.get("data_source") or {}
    fc_value = data_source.get("fc")
    if not fc_value:
        raise ValueError("data_source.fc is required and must point to precomputed ICC FC data.")
    fc_dir = Path(fc_value).expanduser()

    atlases = _as_list(config.get("atlases")) or ["AAL", "Schaefer200", "Brainnetome", "HCPex"]
    strategies = _as_list(config.get("strategies"))
    if strategies is not None:
        strategies = [str(s) for s in strategies]
    fc_types = _as_list(config.get("fc_types")) or ["corr", "pc", "tang", "glasso"]
    kinds = normalize_kinds([str(x) for x in fc_types] if fc_types is not None else None)
    suffix_map = {"corr": "corr", "partial": "pc", "tangent": "tang", "glasso": "glasso"}
    fc_suffixes = [suffix_map[k] for k in (kinds or suffix_map.keys())]

    icc_list = normalize_icc_list(_as_list(config.get("icc") or ["icc31", "icc21", "icc11"]) or [])
    drop_subjects = config.get("drop_subjects") or ["sub-3258811"]
    drop_subjects = [str(x) for x in drop_subjects if str(x).strip()]

    mask = bool(config.get("mask", True))
    mask_percentile = float(config.get("mask_percentile", 95.0))
    save_edgewise = bool(config.get("save_edgewise", False))
    edgewise_output_dir = Path(config.get("edgewise_output_dir") or "icc_results").expanduser()
    pattern = str(config.get("pattern") or "*.npy")

    subject_order_lookup = _build_subject_order_lookup(fc_dir=fc_dir)

    return run_icc_computation(
        input_path=fc_dir,
        icc_list=icc_list,
        pattern=pattern,
        mask=mask,
        mask_percentile=mask_percentile,
        save_edgewise=save_edgewise,
        edgewise_output_dir=edgewise_output_dir if save_edgewise else None,
        allowed_atlases=atlases,
        allowed_strategies=strategies,
        fc_suffixes=fc_suffixes,
        drop_subjects=drop_subjects,
        subject_order_lookup=subject_order_lookup,
    )


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = _load_config(config_path)

    print(f"Config: {config_path}")
    print(f"Atlases: {config.get('atlases', ['all'])}")
    print(f"FC types: {config.get('fc_types', ['corr', 'pc', 'tang', 'glasso'])}")

    df = run_icc_from_config(config)
    output_path = _resolve_output_path(config.get("output"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print(f"Total pipelines: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
