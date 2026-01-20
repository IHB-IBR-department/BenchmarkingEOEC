#!/usr/bin/env python3
"""
Few-Shot Domain Adaptation (Scheme B)
=====================================

This script implements few-shot domain adaptation experiments where:
- Training uses Beijing data + a small subset of IHB subjects
- Testing uses the remaining IHB subjects

This tests whether adding a few labeled samples from the target domain
improves cross-site generalization (practical scenario).

Experimental Design:
-------------------
Random Subsampling Strategy (n_repeats iterations):
  Per repeat:
    Training:  Beijing (92 scans) + 10 random IHB subjects (20 scans)
    Testing:   Remaining 74 IHB subjects (148 scans)

CRITICAL: Pre-generate splits ONCE to ensure all pipelines are evaluated
on identical train/test splits for fair comparison.

Usage:
------
    # Quick test
    python benchmarking/few_shot.py --config configs/few_shot_quick.yaml

    # Full experiment (192 pipelines × 10 repeats = 1920 experiments)
    python benchmarking/few_shot.py --config configs/few_shot_full.yaml

Output:
-------
- {output}_results.csv: Per-pipeline per-repeat results
- {output}_summary.csv: Aggregated statistics

Author: BenchmarkingEOEC Team
"""

import os
import argparse
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Import data loading functions from cross_site module
from benchmarking.cross_site import load_fc_data
# Import classification with permutation testing from stats module
from benchmarking.stats import run_classification_with_permutation


# =============================================================================
# Split Generation
# =============================================================================

def generate_splits(
    n_subjects: int,
    n_few_shot: int,
    n_repeats: int,
    random_state: int = 42
) -> List[Dict[str, np.ndarray]]:
    """
    Pre-generate random splits ONCE to ensure identical splits for all pipelines.

    This is CRITICAL for fair comparison between pipelines - all pipelines
    must be evaluated on the exact same train/test splits.

    Parameters
    ----------
    n_subjects : int
        Total number of IHB subjects (84)
    n_few_shot : int
        Number of subjects to use for few-shot training (10)
    n_repeats : int
        Number of random subsampling iterations (10)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    list of dict
        Each dict contains:
        - 'train_subjects': indices of IHB subjects for training
        - 'test_subjects': indices of IHB subjects for testing
        - 'repeat': repeat index
    """
    rng = np.random.RandomState(random_state)
    all_subjects = np.arange(n_subjects)

    splits = []
    for repeat in range(n_repeats):
        # Randomly shuffle and split
        shuffled = rng.permutation(all_subjects)
        train_subjects = shuffled[:n_few_shot]
        test_subjects = shuffled[n_few_shot:]

        splits.append({
            'repeat': repeat,
            'train_subjects': train_subjects,
            'test_subjects': test_subjects
        })

    return splits


def subjects_to_sample_indices(subjects: np.ndarray) -> np.ndarray:
    """
    Convert subject indices to sample indices.

    Each subject has 2 samples (EO and EC), so subject i corresponds to
    samples 2*i (EC) and 2*i+1 (EO).

    Parameters
    ----------
    subjects : np.ndarray
        Subject indices

    Returns
    -------
    np.ndarray
        Sample indices
    """
    # Each subject i has samples at indices 2*i (EC) and 2*i+1 (EO)
    sample_indices = []
    for subj in subjects:
        sample_indices.extend([2 * subj, 2 * subj + 1])
    return np.array(sorted(sample_indices))


# =============================================================================
# Pipeline Processing
# =============================================================================

def process_pipeline_single_split(
    pipeline_config: Dict[str, Any],
    split: Dict[str, Any],
    beijing_X: np.ndarray,
    beijing_y: np.ndarray,
    ihb_X: np.ndarray,
    ihb_y: np.ndarray
) -> Dict[str, Any]:
    """
    Run single pipeline on a single split.

    Parameters
    ----------
    pipeline_config : dict
        Pipeline configuration (atlas, fc_type, strategy, gsr, etc.)
    split : dict
        Split with train_subjects and test_subjects indices
    beijing_X, beijing_y : np.ndarray
        Beijing FC data and labels
    ihb_X, ihb_y : np.ndarray
        IHB FC data and labels

    Returns
    -------
    dict
        Results for this pipeline × split combination
    """
    # Get sample indices from subject indices
    train_sample_idx = subjects_to_sample_indices(split['train_subjects'])
    test_sample_idx = subjects_to_sample_indices(split['test_subjects'])

    # Build training set: Beijing + IHB few-shot
    X_train = np.concatenate([beijing_X, ihb_X[train_sample_idx]], axis=0)
    y_train = np.concatenate([beijing_y, ihb_y[train_sample_idx]], axis=0)

    # Test set: IHB holdout
    X_test = ihb_X[test_sample_idx]
    y_test = ihb_y[test_sample_idx]

    # Run classification with optional permutation testing
    clf_results = run_classification_with_permutation(
        X_train, y_train,
        X_test, y_test,
        pca_components=pipeline_config.get('pca_components', 0.95),
        random_state=pipeline_config.get('random_state', 42),
        n_permutations=pipeline_config.get('n_permutations', 0),
        model=pipeline_config.get('model', 'logreg'),
        model_params=pipeline_config.get('model_params'),
        scale=pipeline_config.get('scale'),
        return_probabilities=pipeline_config.get('return_probabilities', False),
    )

    return {
        'atlas': pipeline_config['atlas'],
        'fc_type': pipeline_config['fc_type'],
        'strategy': pipeline_config['strategy'],
        'gsr': pipeline_config['gsr'],
        'repeat': split['repeat'],
        'n_train': len(y_train),
        'n_test': len(y_test),
        'n_beijing': len(beijing_y),
        'p_value': clf_results.get('p_value'),
        'n_ihb_fewshot': len(train_sample_idx),
        **clf_results
    }


def process_pipeline(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run single pipeline across all pre-generated splits.

    Parameters
    ----------
    config : dict
        Pipeline configuration including:
        - atlas, fc_type, strategy, gsr
        - data_path, pca_components, random_state
        - splits: pre-generated splits (list of dicts)

    Returns
    -------
    list of dict
        Results for each split
    """
    results = []

    try:
        # Load Beijing data (training base)
        beijing_X, beijing_y = load_fc_data(
            site='china',
            fc_type=config['fc_type'],
            atlas=config['atlas'],
            strategy=config['strategy'],
            gsr=config['gsr'],
            data_path=config.get('data_path')
        )

        # Load IHB data (few-shot + test)
        ihb_X, ihb_y = load_fc_data(
            site='ihb',
            fc_type=config['fc_type'],
            atlas=config['atlas'],
            strategy=config['strategy'],
            gsr=config['gsr'],
            data_path=config.get('data_path')
        )

        # Run on each split
        for split in config['splits']:
            result = process_pipeline_single_split(
                config, split, beijing_X, beijing_y, ihb_X, ihb_y
            )
            result['error'] = None
            results.append(result)

    except Exception as e:
        # If loading fails, mark all splits as failed
        for split in config['splits']:
            results.append({
                'atlas': config['atlas'],
                'fc_type': config['fc_type'],
                'strategy': config['strategy'],
                'gsr': config['gsr'],
                'repeat': split['repeat'],
                'n_train': None,
                'n_test': None,
                'n_beijing': None,
                'n_ihb_fewshot': None,
                'train_acc': None,
                'test_acc': None,
                'train_auc': None,
                'test_auc': None,
                'train_brier': None,
                'test_brier': None,
                'n_pca_components': None,
                'p_value': None,
                'error': str(e)
            })
        warnings.warn(f"Pipeline failed: {config['atlas']}/{config['fc_type']}/{config['strategy']}/{config['gsr']} - {e}")

    return results


def generate_pipeline_configs(
    config: Dict[str, Any],
    splits: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Generate all pipeline configurations from YAML config.

    Parameters
    ----------
    config : dict
        Parsed YAML config
    splits : list
        Pre-generated splits (same for all pipelines)

    Returns
    -------
    list
        List of pipeline config dicts
    """
    pipeline_configs = []

    for atlas, fc_type, strategy, gsr in product(
        config['atlases'],
        config['fc_types'],
        config['strategies'],
        config['gsr_options']
    ):
        pipeline_configs.append({
            'atlas': atlas,
            'fc_type': fc_type,
            'strategy': strategy,
            'gsr': gsr,
            'data_path': config.get('data_path'),
            'pca_components': config.get('pca_components', 0.95),
            'random_state': config.get('random_state', 42),
            'n_permutations': config.get('n_permutations', 0),
            'model': (config.get('model') or {}).get('name', 'logreg') if isinstance(config.get('model'), dict) else config.get('model', 'logreg'),
            'model_params': (config.get('model') or {}).get('params') if isinstance(config.get('model'), dict) else config.get('model_params'),
            'scale': (config.get('model') or {}).get('scale') if isinstance(config.get('model'), dict) else config.get('scale'),
            'return_probabilities': config.get('return_probabilities', False),
            'splits': splits  # Same splits for all pipelines!
        })

    return pipeline_configs


def run_all_pipelines(
    config: Dict[str, Any],
    splits: List[Dict[str, Any]],
    n_workers: Optional[int] = None
) -> pd.DataFrame:
    """
    Run all pipeline experiments with multiprocessing.

    All pipelines receive the SAME splits for fair comparison.

    Parameters
    ----------
    config : dict
        Parsed YAML config
    splits : list
        Pre-generated splits
    n_workers : int, optional
        Number of parallel workers

    Returns
    -------
    pd.DataFrame
        Results for all pipelines × all splits
    """
    pipeline_configs = generate_pipeline_configs(config, splits)
    n_workers = n_workers or config.get('n_workers', 1)
    n_repeats = len(splits)

    print(f"Running {len(pipeline_configs)} pipelines × {n_repeats} repeats = {len(pipeline_configs) * n_repeats} experiments")
    print(f"Using {n_workers} workers...")

    all_results = []

    if n_workers == 1:
        # Sequential execution
        for cfg in tqdm(pipeline_configs, desc="Pipelines"):
            all_results.extend(process_pipeline(cfg))
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_pipeline, cfg): cfg for cfg in pipeline_configs}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Pipelines"):
                all_results.extend(future.result())

    return pd.DataFrame(all_results)


# =============================================================================
# Results Aggregation
# =============================================================================

def compute_summary(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute summary statistics from results.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe

    Returns
    -------
    dict
        Summary dataframes by different groupings
    """
    # Filter out failed experiments
    df_valid = df[df['error'].isna()].copy()

    if df_valid.empty:
        return {}

    # Overall summary
    overall = df_valid.agg({
        'test_acc': ['mean', 'std', 'min', 'max', 'count'],
        'test_auc': ['mean', 'std', 'min', 'max'],
        'test_brier': ['mean', 'std', 'min', 'max'],
    }).round(4)

    # Summary by FC type (aggregated across all repeats)
    fc_summary = df_valid.groupby('fc_type').agg({
        'test_acc': ['mean', 'std', 'min', 'max'],
        'test_auc': ['mean', 'std'],
        'test_brier': ['mean', 'std'],
    }).round(4)
    fc_summary.columns = [
        'test_acc_mean', 'test_acc_std', 'test_acc_min', 'test_acc_max',
        'test_auc_mean', 'test_auc_std',
        'test_brier_mean', 'test_brier_std',
    ]
    fc_summary = fc_summary.reset_index()

    # Summary by atlas
    atlas_summary = df_valid.groupby('atlas').agg({
        'test_acc': ['mean', 'std'],
        'test_auc': ['mean', 'std'],
        'test_brier': ['mean', 'std'],
    }).round(4)
    atlas_summary.columns = [
        'test_acc_mean', 'test_acc_std',
        'test_auc_mean', 'test_auc_std',
        'test_brier_mean', 'test_brier_std',
    ]
    atlas_summary = atlas_summary.reset_index()

    # Summary by GSR
    gsr_summary = df_valid.groupby('gsr').agg({
        'test_acc': ['mean', 'std'],
        'test_auc': ['mean', 'std'],
        'test_brier': ['mean', 'std'],
    }).round(4)
    gsr_summary.columns = [
        'test_acc_mean', 'test_acc_std',
        'test_auc_mean', 'test_auc_std',
        'test_brier_mean', 'test_brier_std',
    ]
    gsr_summary = gsr_summary.reset_index()

    # Summary by pipeline (mean across repeats)
    pipeline_summary = df_valid.groupby(['atlas', 'fc_type', 'strategy', 'gsr']).agg({
        'test_acc': ['mean', 'std'],
        'test_auc': ['mean', 'std'],
        'test_brier': ['mean', 'std'],
        'train_acc': ['mean']
    }).round(4)
    pipeline_summary.columns = [
        'test_acc_mean', 'test_acc_std',
        'test_auc_mean', 'test_auc_std',
        'test_brier_mean', 'test_brier_std',
        'train_acc_mean',
    ]
    pipeline_summary = pipeline_summary.reset_index()

    return {
        'by_fc_type': fc_summary,
        'by_atlas': atlas_summary,
        'by_gsr': gsr_summary,
        'by_pipeline': pipeline_summary
    }


def save_results(
    df: pd.DataFrame,
    output_path: str,
    summaries: Dict[str, pd.DataFrame],
    splits: List[Dict[str, Any]]
) -> None:
    """
    Save results, summaries, and splits to files.

    Parameters
    ----------
    df : pd.DataFrame
        Full results
    output_path : str
        Base output path
    summaries : dict
        Summary dataframes
    splits : list
        Pre-generated splits (for reproducibility)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save full results
    results_file = str(output_path).replace('.csv', '_results.csv')
    df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")

    # Save splits for reproducibility
    splits_file = str(output_path).replace('.csv', '_splits.yaml')
    splits_data = [
        {
            'repeat': s['repeat'],
            'train_subjects': s['train_subjects'].tolist(),
            'test_subjects': s['test_subjects'].tolist()
        }
        for s in splits
    ]
    with open(splits_file, 'w') as f:
        yaml.dump(splits_data, f)
    print(f"Splits saved to: {splits_file}")

    # Save summaries
    for name, summary_df in summaries.items():
        if not summary_df.empty:
            summary_file = str(output_path).replace('.csv', f'_summary_{name}.csv')
            summary_df.to_csv(summary_file, index=False)
            print(f"Summary ({name}) saved to: {summary_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Few-Shot Domain Adaptation (Scheme B)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--n-workers', '-j',
        type=int,
        default=None,
        help='Number of parallel workers (overrides config)'
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config).expanduser()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Config loaded from: {config_path}")
    print(f"Data path: {config.get('data_path') or '(default/env)'}")
    print(f"Few-shot subjects: {config['n_few_shot']}")
    print(f"Repeats: {config['n_repeats']}")
    print(f"Atlases: {config['atlases']}")
    print(f"FC types: {config['fc_types']}")
    print(f"Strategies: {config['strategies']}")
    print(f"GSR options: {config['gsr_options']}")
    if isinstance(config.get("model"), dict):
        print(
            "Model: "
            f"{config['model'].get('name', 'logreg')} "
            f"(scale={config['model'].get('scale', '(auto)')}, params={config['model'].get('params', {})})"
        )
    else:
        print(f"Model: {config.get('model', 'logreg')}")

    # Generate splits ONCE (critical for fair comparison)
    n_ihb_subjects = 84  # IHB has 84 subjects
    splits = generate_splits(
        n_subjects=n_ihb_subjects,
        n_few_shot=config['n_few_shot'],
        n_repeats=config['n_repeats'],
        random_state=config.get('random_state', 42)
    )
    print(f"\nGenerated {len(splits)} splits:")
    for s in splits[:3]:
        print(f"  Repeat {s['repeat']}: train={len(s['train_subjects'])} subjects, test={len(s['test_subjects'])} subjects")
    if len(splits) > 3:
        print(f"  ... and {len(splits) - 3} more")

    # Run pipelines
    n_workers = args.n_workers or config.get('n_workers', 1)
    df = run_all_pipelines(config, splits, n_workers=n_workers)

    # Check for errors
    n_errors = df['error'].notna().sum()
    if n_errors > 0:
        print(f"\nWarning: {n_errors} experiments failed!")
        failed = df[df['error'].notna()][['atlas', 'fc_type', 'strategy', 'gsr', 'repeat', 'error']].drop_duplicates()
        print(failed.head(10))

    # Compute and save results
    summaries = compute_summary(df)
    save_results(df, config['output'], summaries, splits)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY BY FC TYPE")
    print("=" * 60)
    if 'by_fc_type' in summaries and not summaries['by_fc_type'].empty:
        print(summaries['by_fc_type'].to_string(index=False))

    print("\n" + "=" * 60)
    print("SUMMARY BY ATLAS")
    print("=" * 60)
    if 'by_atlas' in summaries and not summaries['by_atlas'].empty:
        print(summaries['by_atlas'].to_string(index=False))

    # Overall stats
    df_valid = df[df['error'].isna()]
    if not df_valid.empty:
        print("\n" + "=" * 60)
        print("OVERALL STATISTICS")
        print("=" * 60)
        print(f"Mean test accuracy: {df_valid['test_acc'].mean():.4f} ± {df_valid['test_acc'].std():.4f}")
        print(f"Best pipeline mean: {df_valid.groupby(['atlas', 'fc_type', 'strategy', 'gsr'])['test_acc'].mean().max():.4f}")


if __name__ == '__main__':
    main()
