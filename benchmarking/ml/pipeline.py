#!/usr/bin/env python3
"""
Unified ML Pipeline
===================

Single entry point for running complete ML evaluation on a preprocessing pipeline.

For a given configuration (atlas, strategy, gsr), runs:
1. Cross-site validation: China → IHB
2. Cross-site validation: IHB → China
3. Few-shot domain adaptation (N repeats)

All with leakage-safe tangent computation and consistent IHB coverage masking.
All FC data is VECTORIZED throughout the pipeline.

Usage
-----
    # Run with precomputed glasso (Schaefer200, strategy-1, GSR)
    python -m benchmarking.ml.pipeline --atlas Schaefer200 --strategy 1 --gsr GSR \\
        --precomputed-glasso ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/glasso_precomputed_fc

    # Skip glasso (faster)
    python -m benchmarking.ml.pipeline --atlas Schaefer200 --strategy 1 --gsr GSR --skip-glasso

Author: BenchmarkingEOEC Team
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from data_utils.timeseries import (
    load_site_timeseries,
    load_ihb_coverage_mask,
    load_site_precomputed_glasso,
)
from data_utils.fc import compute_fc_vectorized, get_fc_types_to_compute
from benchmarking.ml.stats import run_classification_with_permutation


# =============================================================================
# Configuration and Results Dataclasses
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for a single preprocessing pipeline."""

    atlas: str  # 'Schaefer200', 'AAL', 'Brainnetome', 'HCPex'
    strategy: Union[int, str]  # 1-6, 'AROMA_aggr', 'AROMA_nonaggr'
    gsr: str  # 'GSR' or 'noGSR'
    fc_types: Optional[List[str]] = None  # ['corr', 'partial', 'tangent', 'glasso']

    # Glasso options
    skip_glasso: bool = False
    precomputed_glasso_dir: Optional[str] = None
    glasso_lambda: float = 0.03

    # Few-shot settings
    n_few_shot: int = 10  # Number of IHB subjects for few-shot training
    n_repeats: int = 10  # Number of random splits

    # Model settings
    model: str = "logreg"
    model_params: Optional[Dict[str, Any]] = None
    pca_components: float = 0.95

    # Coverage
    coverage_threshold: float = 0.1

    # Paths
    data_path: Optional[str] = None
    output_dir: Optional[str] = None

    # Execution
    n_permutations: int = 0  # 0 = skip permutation test
    random_state: int = 42
    save_test_outputs: bool = True

    def __post_init__(self):
        if self.fc_types is None:
            self.fc_types = ["corr", "partial", "tangent", "glasso"]


@dataclass
class PipelineResults:
    """Results from running the full pipeline."""

    config: PipelineConfig

    # Cross-site results (both directions)
    cross_site_china2ihb: pd.DataFrame = field(default_factory=pd.DataFrame)
    cross_site_ihb2china: pd.DataFrame = field(default_factory=pd.DataFrame)
    cross_site_china2ihb_outputs: pd.DataFrame = field(default_factory=pd.DataFrame)
    cross_site_ihb2china_outputs: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Few-shot results
    few_shot: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Combined summary
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)


# =============================================================================
# Few-Shot Split Generation
# =============================================================================

def generate_few_shot_splits(
    n_subjects: int,
    n_few_shot: int,
    n_repeats: int,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Pre-generate random splits ONCE for fair comparison across FC types.
    """
    rng = np.random.RandomState(random_state)
    all_subjects = np.arange(n_subjects)

    splits = []
    for repeat in range(n_repeats):
        shuffled = rng.permutation(all_subjects)
        splits.append({
            "repeat": repeat,
            "train_subjects": shuffled[:n_few_shot],
            "test_subjects": shuffled[n_few_shot:],
        })
    return splits


def subjects_to_mask(subjects: np.ndarray, n_subjects: int) -> np.ndarray:
    """
    Convert subject indices to a boolean mask for samples.

    Data is ordered as [EC_all, EO_all], so:
    - samples 0..n_subjects-1 are EC for subjects 0..n_subjects-1
    - samples n_subjects..2*n_subjects-1 are EO for subjects 0..n_subjects-1

    Subject i has samples at indices i (EC) and i+n_subjects (EO).
    """
    n_samples = n_subjects * 2
    mask = np.zeros(n_samples, dtype=bool)
    for subj in subjects:
        mask[subj] = True  # EC sample
        mask[subj + n_subjects] = True  # EO sample
    return mask


# =============================================================================
# Main Pipeline Function
# =============================================================================

def run_full_pipeline(config: PipelineConfig) -> PipelineResults:
    """
    Run complete ML evaluation for a single preprocessing pipeline.

    All FC is computed as VECTORIZED data (n_samples, n_edges).
    """
    print(f"\n{'='*60}")
    print(f"Running pipeline: {config.atlas}_strategy-{config.strategy}_{config.gsr}")
    print(f"{'='*60}")

    # =========================================================================
    # 1. Load Timeseries Data
    # =========================================================================
    print("\n[1/4] Loading timeseries data...")

    # Load IHB coverage mask (used for both sites)
    coverage_mask = load_ihb_coverage_mask(
        atlas=config.atlas,
        data_path=config.data_path,
        threshold=config.coverage_threshold,
    )
    n_good_rois = int(np.sum(coverage_mask))
    n_edges = n_good_rois * (n_good_rois - 1) // 2
    print(f"  Coverage mask: {n_good_rois}/{len(coverage_mask)} good ROIs → {n_edges} edges")

    # Load timeseries for both sites
    china_ts, china_y = load_site_timeseries(
        "china", config.atlas, config.strategy, config.gsr, config.data_path
    )
    ihb_ts, ihb_y = load_site_timeseries(
        "ihb", config.atlas, config.strategy, config.gsr, config.data_path
    )

    n_china_subjects = china_ts.shape[0] // 2
    n_ihb_subjects = ihb_ts.shape[0] // 2
    print(f"  China: {china_ts.shape[0]} samples ({n_china_subjects} subjects), {china_ts.shape[1]} TRs")
    print(f"  IHB: {ihb_ts.shape[0]} samples ({n_ihb_subjects} subjects), {ihb_ts.shape[1]} TRs")

    # =========================================================================
    # 2. Determine FC types and load precomputed glasso if available
    # =========================================================================
    print("\n[2/4] Preparing FC computation...")

    fc_types = get_fc_types_to_compute(config.fc_types, config.skip_glasso)
    print(f"  FC types: {fc_types}")

    # Load precomputed glasso if available and needed
    precomputed_glasso = {}
    if "glasso" in fc_types and config.precomputed_glasso_dir:
        print(f"  Loading precomputed glasso...")
        try:
            precomputed_glasso["china"] = load_site_precomputed_glasso(
                "china", config.atlas, config.strategy, config.gsr, config.data_path
            )
            precomputed_glasso["ihb"] = load_site_precomputed_glasso(
                "ihb", config.atlas, config.strategy, config.gsr, config.data_path
            )
            print(f"    china: {precomputed_glasso['china'].shape}, ihb: {precomputed_glasso['ihb'].shape}")
        except FileNotFoundError as e:
            warnings.warn(f"Precomputed glasso not found: {e}. Will compute on-the-fly (slow!).")
            precomputed_glasso = {}
    elif "glasso" in fc_types:
        print("  WARNING: glasso will be computed on-the-fly (slow!)")

    # =========================================================================
    # 3. Run Cross-Site Validation (Both Directions)
    # =========================================================================
    print("\n[3/4] Running cross-site validation...")

    cross_site_results = {}

    directions = [
        ("china2ihb", china_ts, china_y, ihb_ts, ihb_y, "china", "ihb"),
        ("ihb2china", ihb_ts, ihb_y, china_ts, china_y, "ihb", "china"),
    ]

    for direction, ts_train, y_train, ts_test, y_test, train_site, test_site in directions:
        print(f"\n  Direction: {direction}")
        results_rows = []
        output_rows = []

        for fc_type in fc_types:
            print(f"    FC type: {fc_type}...", end=" ", flush=True)

            try:
                # Handle glasso separately (precomputed or compute)
                if fc_type == "glasso" and precomputed_glasso:
                    X_train = precomputed_glasso[train_site]
                    X_test = precomputed_glasso[test_site]
                else:
                    # Compute FC for train site (fit tangent reference)
                    X_train, tangent_transformer = compute_fc_vectorized(
                        ts_train,
                        fc_type=fc_type,
                        coverage_mask=coverage_mask,
                        glasso_lambda=config.glasso_lambda,
                        tangent_transformer=None,
                    )

                    # Compute FC for test site (use train's tangent reference)
                    X_test, _ = compute_fc_vectorized(
                        ts_test,
                        fc_type=fc_type,
                        coverage_mask=coverage_mask,
                        glasso_lambda=config.glasso_lambda,
                        tangent_transformer=tangent_transformer,
                    )

                # Run classification (data is already vectorized)
                clf_result = run_classification_with_permutation(
                    X_train, y_train, X_test, y_test,
                    pca_components=config.pca_components,
                    random_state=config.random_state,
                    n_permutations=config.n_permutations,
                    model=config.model,
                    model_params=config.model_params,
                    return_test_outputs=config.save_test_outputs,
                    vectorize=False,  # Already vectorized!
                )

                print(f"acc={clf_result['test_acc']:.3f}, auc={clf_result.get('test_auc', 'N/A')}")

                # Collect results
                results_rows.append({
                    "direction": direction,
                    "train_site": train_site,
                    "test_site": test_site,
                    "atlas": config.atlas,
                    "fc_type": fc_type,
                    "strategy": config.strategy,
                    "gsr": config.gsr,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "train_acc": clf_result.get("train_acc"),
                    "test_acc": clf_result.get("test_acc"),
                    "train_auc": clf_result.get("train_auc"),
                    "test_auc": clf_result.get("test_auc"),
                    "train_brier": clf_result.get("train_brier"),
                    "test_brier": clf_result.get("test_brier"),
                    "n_pca_components": clf_result.get("n_pca_components"),
                    "p_value": clf_result.get("p_value"),
                })

                # Collect per-sample outputs
                if config.save_test_outputs and "test_y_pred" in clf_result:
                    test_y_pred = clf_result["test_y_pred"]
                    test_score = clf_result.get("test_score") or [None] * len(y_test)
                    test_p_positive = clf_result.get("test_p_positive") or [None] * len(y_test)

                    n_test_subjects = len(y_test) // 2
                    for i in range(len(y_test)):
                        test_subject = i if i < n_test_subjects else i - n_test_subjects
                        output_rows.append({
                            "direction": direction,
                            "train_site": train_site,
                            "test_site": test_site,
                            "fc_type": fc_type,
                            "sample_index": i,
                            "test_subject": test_subject,
                            "y_true": int(y_test[i]),
                            "y_pred": int(test_y_pred[i]),
                            "y_score": test_score[i] if test_score else None,
                            "p_positive": test_p_positive[i] if test_p_positive else None,
                        })

            except Exception as e:
                print(f"FAILED: {e}")
                results_rows.append({
                    "direction": direction,
                    "train_site": train_site,
                    "test_site": test_site,
                    "atlas": config.atlas,
                    "fc_type": fc_type,
                    "strategy": config.strategy,
                    "gsr": config.gsr,
                    "error": str(e),
                })

        cross_site_results[direction] = {
            "results": pd.DataFrame(results_rows),
            "outputs": pd.DataFrame(output_rows) if output_rows else pd.DataFrame(),
        }

    # =========================================================================
    # 4. Run Few-Shot Domain Adaptation
    # =========================================================================
    print("\n[4/4] Running few-shot validation...")

    # Generate splits ONCE (critical for fair comparison)
    splits = generate_few_shot_splits(
        n_subjects=n_ihb_subjects,
        n_few_shot=config.n_few_shot,
        n_repeats=config.n_repeats,
        random_state=config.random_state,
    )
    print(f"  Generated {len(splits)} splits ({config.n_few_shot} few-shot subjects each)")

    few_shot_rows = []

    for fc_type in fc_types:
        print(f"  FC type: {fc_type}...", end=" ", flush=True)

        try:
            # Compute FC for China (always in train) - fit tangent reference
            if fc_type == "glasso" and precomputed_glasso:
                china_fc = precomputed_glasso["china"]
                ihb_fc = precomputed_glasso["ihb"]
                tangent_transformer = None
            else:
                china_fc, tangent_transformer = compute_fc_vectorized(
                    china_ts,
                    fc_type=fc_type,
                    coverage_mask=coverage_mask,
                    glasso_lambda=config.glasso_lambda,
                    tangent_transformer=None,
                )

                # Compute FC for IHB using China's tangent reference
                ihb_fc, _ = compute_fc_vectorized(
                    ihb_ts,
                    fc_type=fc_type,
                    coverage_mask=coverage_mask,
                    glasso_lambda=config.glasso_lambda,
                    tangent_transformer=tangent_transformer,
                )

            # Run all repeats with same FC
            accs = []
            for split in splits:
                train_mask = subjects_to_mask(split["train_subjects"], n_ihb_subjects)
                test_mask = subjects_to_mask(split["test_subjects"], n_ihb_subjects)

                # Concatenate vectorized FC: China + few-shot IHB
                X_train = np.concatenate([china_fc, ihb_fc[train_mask]], axis=0)
                y_train = np.concatenate([china_y, ihb_y[train_mask]], axis=0)

                X_test = ihb_fc[test_mask]
                y_test = ihb_y[test_mask]

                # Run classification
                clf_result = run_classification_with_permutation(
                    X_train, y_train, X_test, y_test,
                    pca_components=config.pca_components,
                    random_state=config.random_state,
                    n_permutations=0,
                    model=config.model,
                    model_params=config.model_params,
                    vectorize=False,  # Already vectorized!
                )

                accs.append(clf_result.get("test_acc", 0))

                few_shot_rows.append({
                    "repeat": split["repeat"],
                    "fc_type": fc_type,
                    "atlas": config.atlas,
                    "strategy": config.strategy,
                    "gsr": config.gsr,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "n_china": len(china_y),
                    "n_ihb_fewshot": int(train_mask.sum()),
                    "train_acc": clf_result.get("train_acc"),
                    "test_acc": clf_result.get("test_acc"),
                    "train_auc": clf_result.get("train_auc"),
                    "test_auc": clf_result.get("test_auc"),
                    "train_brier": clf_result.get("train_brier"),
                    "test_brier": clf_result.get("test_brier"),
                    "n_pca_components": clf_result.get("n_pca_components"),
                })

            print(f"acc={np.mean(accs):.3f} ± {np.std(accs):.3f}")

        except Exception as e:
            print(f"FAILED: {e}")
            for split in splits:
                few_shot_rows.append({
                    "repeat": split["repeat"],
                    "fc_type": fc_type,
                    "atlas": config.atlas,
                    "strategy": config.strategy,
                    "gsr": config.gsr,
                    "error": str(e),
                })

    few_shot_df = pd.DataFrame(few_shot_rows)

    # =========================================================================
    # 5. Create Combined Summary
    # =========================================================================
    summary_rows = []
    for fc_type in fc_types:
        row = {
            "atlas": config.atlas,
            "strategy": config.strategy,
            "gsr": config.gsr,
            "fc_type": fc_type,
        }

        # Cross-site metrics
        for direction in ["china2ihb", "ihb2china"]:
            df = cross_site_results[direction]["results"]
            fc_row = df[df["fc_type"] == fc_type]
            if not fc_row.empty and "test_acc" in fc_row.columns:
                row[f"{direction}_acc"] = fc_row.iloc[0].get("test_acc")
                row[f"{direction}_auc"] = fc_row.iloc[0].get("test_auc")

        # Few-shot metrics
        fs_fc = few_shot_df[few_shot_df["fc_type"] == fc_type]
        if not fs_fc.empty and "test_acc" in fs_fc.columns:
            valid_acc = fs_fc["test_acc"].dropna()
            if len(valid_acc) > 0:
                row["few_shot_acc_mean"] = valid_acc.mean()
                row["few_shot_acc_std"] = valid_acc.std()
                if "test_auc" in fs_fc.columns:
                    valid_auc = fs_fc["test_auc"].dropna()
                    if len(valid_auc) > 0:
                        row["few_shot_auc_mean"] = valid_auc.mean()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # =========================================================================
    # 6. Save Results
    # =========================================================================
    if config.output_dir:
        output_dir = Path(config.output_dir)
        pipeline_dir = output_dir / f"{config.atlas}_strategy-{config.strategy}_{config.gsr}"
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        # Cross-site results
        for direction in ["china2ihb", "ihb2china"]:
            cross_site_results[direction]["results"].to_csv(
                pipeline_dir / f"cross_site_{direction}_results.csv", index=False
            )
            if not cross_site_results[direction]["outputs"].empty:
                cross_site_results[direction]["outputs"].to_csv(
                    pipeline_dir / f"cross_site_{direction}_test_outputs.csv", index=False
                )

        # Few-shot results
        few_shot_df.to_csv(pipeline_dir / "few_shot_results.csv", index=False)

        # Summary
        summary_df.to_csv(pipeline_dir / "summary.csv", index=False)

        print(f"\nResults saved to: {pipeline_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    return PipelineResults(
        config=config,
        cross_site_china2ihb=cross_site_results["china2ihb"]["results"],
        cross_site_ihb2china=cross_site_results["ihb2china"]["results"],
        cross_site_china2ihb_outputs=cross_site_results["china2ihb"]["outputs"],
        cross_site_ihb2china_outputs=cross_site_results["ihb2china"]["outputs"],
        few_shot=few_shot_df,
        summary=summary_df,
    )


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run full ML pipeline for a single preprocessing configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--atlas", required=True,
        choices=["AAL", "Schaefer200", "Brainnetome", "HCPex"],
    )
    parser.add_argument(
        "--strategy", required=True,
        help="Denoising strategy (1-6, AROMA_aggr, AROMA_nonaggr)",
    )
    parser.add_argument(
        "--gsr", required=True, choices=["GSR", "noGSR"],
    )
    parser.add_argument(
        "--fc-types", nargs="+", default=["corr", "partial", "tangent", "glasso"],
    )
    parser.add_argument("--skip-glasso", action="store_true")
    parser.add_argument("--precomputed-glasso", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/pipelines")
    parser.add_argument("--n-few-shot", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-permutations", type=int, default=0)
    parser.add_argument("--pca-components", type=float, default=0.95)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    try:
        strategy = int(args.strategy)
    except ValueError:
        strategy = args.strategy

    config = PipelineConfig(
        atlas=args.atlas,
        strategy=strategy,
        gsr=args.gsr,
        fc_types=args.fc_types,
        skip_glasso=args.skip_glasso,
        precomputed_glasso_dir=args.precomputed_glasso,
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_few_shot=args.n_few_shot,
        n_repeats=args.n_repeats,
        n_permutations=args.n_permutations,
        pca_components=args.pca_components,
        random_state=args.random_state,
    )

    run_full_pipeline(config)


if __name__ == "__main__":
    main()
