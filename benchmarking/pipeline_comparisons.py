#!/usr/bin/env python3
"""
Run pipeline comparison tests from precomputed cross-site outputs.

This script is a thin CLI wrapper around benchmarking.stats functions.
It operates on CSVs produced by benchmarking/cross_site.py with
save_test_outputs: true.
It prints results to stdout and writes a JSON file next to the input
test-outputs CSV (override with --output).

Output
------
- Default: `{test_outputs_stem}_<suffix>.json` next to the input CSV
- Override: `--output path/to/results.json`

Examples
--------
# Factor-level test (GSR vs noGSR) for one direction
python benchmarking/pipeline_comparisons.py factor \
  --test-outputs results/cross_site_full_test_outputs.csv \
  --factor gsr --level-a GSR --level-b noGSR \
  --train-site ihb --test-site china

# Pipeline A vs B (by abbrev)
python benchmarking/pipeline_comparisons.py compare \
  --test-outputs results/cross_site_full_test_outputs.csv \
  --abbrev results/cross_site_full_pipeline_abbreviations.csv \
  --pipeline-a P0001 --pipeline-b P0002
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from benchmarking import stats


def _as_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, default=str)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _default_output_path(args: argparse.Namespace) -> Path:
    base = Path(args.test_outputs)
    stem = base.stem
    if stem.endswith("_test_outputs"):
        stem = stem[: -len("_test_outputs")]

    if args.command == "factor":
        if args.train_site and args.test_site:
            direction = f"train_{args.train_site}_test_{args.test_site}"
        else:
            direction = "all_directions"
        suffix = f"factor_{args.factor}_{args.level_a}_vs_{args.level_b}_{direction}"
    else:
        suffix = f"compare_{args.pipeline_a}_vs_{args.pipeline_b}"

    name = f"{stem}_{_safe_slug(suffix)}.json"
    return base.parent / name


def _save_results(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_as_json(data) + "\n", encoding="utf-8")
    print(f"Results saved to: {output_path}")


def _run_factor_tests(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    if args.train_site and args.test_site:
        result = stats.factor_level_randomization_test(
            df,
            factor=args.factor,
            level_a=args.level_a,
            level_b=args.level_b,
            metric=args.metric,
            train_site=args.train_site,
            test_site=args.test_site,
            n_permutations=args.n_permutations,
            n_bootstrap=args.n_bootstrap,
            random_state=args.random_state,
            alternative=args.alternative,
        )
        results.append(result)
        return results

    if "train_site" not in df.columns or "test_site" not in df.columns:
        raise ValueError("train_site/test_site columns missing; please pass --train-site and --test-site")

    directions: List[Tuple[str, str]] = sorted(
        df[["train_site", "test_site"]].drop_duplicates().itertuples(index=False, name=None)
    )
    for train_site, test_site in directions:
        result = stats.factor_level_randomization_test(
            df,
            factor=args.factor,
            level_a=args.level_a,
            level_b=args.level_b,
            metric=args.metric,
            train_site=train_site,
            test_site=test_site,
            n_permutations=args.n_permutations,
            n_bootstrap=args.n_bootstrap,
            random_state=args.random_state,
            alternative=args.alternative,
        )
        results.append(result)

    return results


def _run_pipeline_compare(
    df: pd.DataFrame,
    abbrev_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return stats.compare_pipelines(
        df,
        args.pipeline_a,
        args.pipeline_b,
        abbrev_df=abbrev_df,
        alternative=args.alternative,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pipeline comparison tests from precomputed cross-site outputs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    factor = subparsers.add_parser("factor", help="Factor-level paired randomization tests")
    factor.add_argument("--test-outputs", required=True, help="Path to *_test_outputs.csv")
    factor.add_argument("--factor", required=True, help="Factor name (e.g., gsr, fc_type, atlas)")
    factor.add_argument("--level-a", required=True, help="Factor level A (treated as better if delta > 0)")
    factor.add_argument("--level-b", required=True, help="Factor level B")
    factor.add_argument("--metric", default="log_loss", help="Metric: log_loss, brier, acc")
    factor.add_argument("--train-site", default=None, help="Train site (optional)")
    factor.add_argument("--test-site", default=None, help="Test site (optional)")
    factor.add_argument("--n-permutations", type=int, default=100)
    factor.add_argument("--n-bootstrap", type=int, default=100)
    factor.add_argument("--random-state", type=int, default=42)
    factor.add_argument("--alternative", choices=["two-sided", "greater", "less"], default="two-sided")
    factor.add_argument("--output", default=None, help="Optional output JSON path")

    compare = subparsers.add_parser("compare", help="Pipeline A vs B (Exact McNemar + DeLong)")
    compare.add_argument("--test-outputs", required=True, help="Path to *_test_outputs.csv")
    compare.add_argument("--abbrev", required=True, help="Path to *_pipeline_abbreviations.csv")
    compare.add_argument("--pipeline-a", required=True, help="Pipeline A abbrev (e.g., P0001)")
    compare.add_argument("--pipeline-b", required=True, help="Pipeline B abbrev (e.g., P0002)")
    compare.add_argument("--alternative", choices=["two-sided", "greater", "less"], default="two-sided")
    compare.add_argument("--output", default=None, help="Optional output JSON path")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.test_outputs)

    if args.command == "factor":
        results = _run_factor_tests(df, args)
        for result in results:
            print(_as_json(result))
        output_path = Path(args.output) if args.output else _default_output_path(args)
        _save_results(results, output_path)
        return 0

    if args.command == "compare":
        abbrev_df = pd.read_csv(args.abbrev)
        result = _run_pipeline_compare(df, abbrev_df, args)
        print(_as_json(result))
        output_path = Path(args.output) if args.output else _default_output_path(args)
        _save_results(result, output_path)
        return 0

    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    sys.exit(main())
