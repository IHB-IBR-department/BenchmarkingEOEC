"""
Unified ML Pipeline
===================

This module implements the unified machine learning pipeline for cross-site
fMRI classification. It handles:
1. Time series loading (IHB + China)
2. Leakage-safe FC computation (Tangent reference fitted on train only)
3. Direct cross-site validation (both directions)
4. Few-shot domain adaptation
5. Statistical testing (Permutation tests vs chance)
6. Results aggregation and saving

Usage:
    python -m benchmarking.ml.pipeline --atlas AAL --strategy 1 --gsr GSR
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_utils.timeseries import (
    load_timeseries,
    load_ihb_coverage_mask,
    load_precomputed_glasso,
    check_precomputed_glasso_available,
)
from data_utils.fc import compute_fc_train_test
from benchmarking.ml.stats import run_classification_with_permutation


@dataclass
class PipelineConfig:
    """Configuration for a single preprocessing pipeline."""
    atlas: str
    strategy: int | str
    gsr: str
    fc_types: list[str] = field(default_factory=lambda: ['corr', 'partial', 'tangent', 'glasso'])
    models: list[str] = field(default_factory=lambda: ['logreg'])

    # Glasso options
    skip_glasso: bool = False
    precomputed_glasso_dir: str | None = None
    glasso_lambda: float = 0.03

    # Few-shot settings
    n_few_shot: int = 10
    n_repeats: int = 10

    # Model settings
    model_params: dict = field(default_factory=dict)
    pca_components: float = 0.95

    # Coverage
    coverage_threshold: float = 0.1

    # Paths
    data_path: str | None = None
    output_dir: str = 'results/pipelines'

    # Execution
    n_permutations: int = 0
    random_state: int = 42
    save_test_outputs: bool = True
    n_workers: int = 1
    scale: bool = True


@dataclass
class PipelineResults:
    """Results from running the full pipeline."""
    config: PipelineConfig
    cross_site_china2ihb: pd.DataFrame
    cross_site_ihb2china: pd.DataFrame
    cross_site_china2ihb_outputs: pd.DataFrame
    cross_site_ihb2china_outputs: pd.DataFrame
    few_shot: pd.DataFrame
    few_shot_summary: pd.DataFrame
    summary: pd.DataFrame


def generate_few_shot_splits(
    n_subjects: int,
    n_few_shot: int,
    n_repeats: int,
    random_state: int = 42,
) -> list[dict]:
    """Pre-generate random splits for few-shot learning."""
    rng = np.random.RandomState(random_state)
    all_subjects = np.arange(n_subjects)

    splits = []
    for repeat in range(n_repeats):
        shuffled = rng.permutation(all_subjects)
        splits.append({
            'repeat': repeat,
            'train_subjects': shuffled[:n_few_shot],
            'test_subjects': shuffled[n_few_shot:]
        })
    return splits


def _subjects_to_indices(subject_indices: np.ndarray, samples_per_subject: int) -> np.ndarray:
    """Convert subject indices to sample indices (assuming sequential block layout)."""
    # If samples_per_subject = 2, subject 0 -> [0, 1], subject 1 -> [2, 3]
    indices = []
    for subj in subject_indices:
        for i in range(samples_per_subject):
            indices.append(subj * samples_per_subject + i)
    return np.array(indices)


def run_full_pipeline(config: PipelineConfig) -> PipelineResults:
    """Run complete ML evaluation for one preprocessing configuration."""
    
    # 1. Load Timeseries Data
    print(f"Loading timeseries for {config.atlas} strategy-{config.strategy} {config.gsr}...")
    
    # IHB: close + open
    ihb_close = load_timeseries('ihb', 'close', config.atlas, config.strategy, config.gsr, config.data_path)
    ihb_open = load_timeseries('ihb', 'open', config.atlas, config.strategy, config.gsr, config.data_path)
    ihb_ts = np.concatenate([ihb_close, ihb_open], axis=0)
    # IHB Labels: first half close (0), second half open (1)
    ihb_n_sub = len(ihb_close)
    ihb_y = np.array([0] * ihb_n_sub + [1] * ihb_n_sub)
    
    # China: close (session 0) + open
    china_close = load_timeseries('china', 'close', config.atlas, config.strategy, config.gsr, config.data_path)
    china_open = load_timeseries('china', 'open', config.atlas, config.strategy, config.gsr, config.data_path)
    china_ts = np.concatenate([china_close, china_open], axis=0)
    # China Labels
    china_n_sub = len(china_close)
    china_y = np.array([0] * china_n_sub + [1] * china_n_sub)

    # Load Coverage Mask (IHB)
    coverage_mask = load_ihb_coverage_mask(config.atlas, config.data_path, config.coverage_threshold)

    # 2. Handle Glasso Loading (if needed)
    glasso_ihb = None
    glasso_china = None
    
    fc_types = config.fc_types[:]
    if config.skip_glasso and 'glasso' in fc_types:
        fc_types.remove('glasso')
    
    if 'glasso' in fc_types and config.precomputed_glasso_dir:
        try:
            print("Loading precomputed glasso...")
            # Load IHB
            g_ihb_c = load_precomputed_glasso('ihb', 'close', config.atlas, config.strategy, config.gsr, config.precomputed_glasso_dir)
            g_ihb_o = load_precomputed_glasso('ihb', 'open', config.atlas, config.strategy, config.gsr, config.precomputed_glasso_dir)
            glasso_ihb = np.concatenate([g_ihb_c, g_ihb_o], axis=0)
            
            # Load China
            g_china_c = load_precomputed_glasso('china', 'close', config.atlas, config.strategy, config.gsr, config.precomputed_glasso_dir)
            g_china_o = load_precomputed_glasso('china', 'open', config.atlas, config.strategy, config.gsr, config.precomputed_glasso_dir)
            glasso_china = np.concatenate([g_china_c, g_china_o], axis=0)
        except FileNotFoundError as e:
            print(f"Warning: Precomputed glasso not found ({e}). Skipping glasso.")
            if 'glasso' in fc_types:
                fc_types.remove('glasso')

    # 3. Cross-Site Validation
    print("Running cross-site validation...")
    cross_site_results = {}

    for direction in ['china2ihb', 'ihb2china']:
        if direction == 'china2ihb':
            ts_train, y_train = china_ts, china_y
            ts_test, y_test = ihb_ts, ihb_y
            g_train, g_test = glasso_china, glasso_ihb
        else:
            ts_train, y_train = ihb_ts, ihb_y
            ts_test, y_test = china_ts, china_y
            g_train, g_test = glasso_ihb, glasso_china

        results_rows = []
        output_rows = []

        # Compute all FC types
        fc_data = compute_fc_train_test(
            ts_train, ts_test,
            kinds=fc_types,
            coverage_mask=coverage_mask,
            glasso_train=g_train,
            glasso_test=g_test,
            glasso_lambda=config.glasso_lambda,
        )

        for fc_type, (X_train, X_test) in fc_data.items():
            for model_name in config.models:
                clf_res = run_classification_with_permutation(
                    X_train, y_train, X_test, y_test,
                    pca_components=config.pca_components,
                    random_state=config.random_state,
                    n_permutations=config.n_permutations,
                    model=model_name,
                    model_params=config.model_params,
                    scale=config.scale,
                    return_test_outputs=config.save_test_outputs,
                    vectorize=False, # Already vectorized by compute_fc_train_test
                )

                # Metadata
                row = {
                    'direction': direction,
                    'atlas': config.atlas,
                    'fc_type': fc_type,
                    'strategy': config.strategy,
                    'gsr': config.gsr,
                    'n_train': len(y_train),
                    'n_test': len(y_test),
                    'model': model_name,
                    'model_params': json.dumps(config.model_params, sort_keys=True),
                }
                # Extract scalar metrics
                for k, v in clf_res.items():
                    if k not in ['test_y_pred', 'test_score', 'test_p_positive', 'model', 'model_params']:
                        row[k] = v
                results_rows.append(row)

                # Extract per-sample outputs
                if config.save_test_outputs and 'test_y_pred' in clf_res:
                    n_samples = len(y_test)
                    n_sub_test = n_samples // 2
                    test_subjects = np.concatenate([np.arange(n_sub_test), np.arange(n_sub_test)])
                    
                    for i in range(n_samples):
                        output_rows.append({
                            'direction': direction,
                            'atlas': config.atlas,
                            'fc_type': fc_type,
                            'strategy': config.strategy,
                            'gsr': config.gsr,
                            'model': model_name,
                            'model_params': json.dumps(config.model_params, sort_keys=True),
                            'train_site': 'china' if direction == 'china2ihb' else 'ihb',
                            'test_site': 'ihb' if direction == 'china2ihb' else 'china',
                            'sample_index': i,
                            'test_subject': test_subjects[i],
                            'condition': 'close' if i < n_sub_test else 'open',
                            'y_true': int(y_test[i]),
                            'y_pred': int(clf_res['test_y_pred'][i]),
                            'y_score': clf_res['test_score'][i] if clf_res['test_score'] is not None else None,
                            'p_positive': clf_res['test_p_positive'][i] if clf_res['test_p_positive'] is not None else None,
                        })

        cross_site_results[direction] = {
            'results': pd.DataFrame(results_rows),
            'outputs': pd.DataFrame(output_rows)
        }

    # 4. Few-Shot Domain Adaptation (Train: China + k IHB, Test: remaining IHB)
    print(f"Running few-shot ({config.n_few_shot} subjects, {config.n_repeats} repeats)...")
    
    splits = generate_few_shot_splits(ihb_n_sub, config.n_few_shot, config.n_repeats, config.random_state)
    few_shot_rows = []

    # Prepare IHB data specifically for splitting
    # Note: ihb_ts is [Close(0..N), Open(0..N)]. 
    # We need to pick both Close and Open for training subjects.
    
    for split in tqdm(splits, desc="Few-shot repeats"):
        # Map subject indices to sample indices
        # Samples 0..83 are Close, 84..167 are Open (for 84 subjects)
        # Train indices: subject i -> samples i AND i + n_sub
        train_sub_idx = split['train_subjects']
        test_sub_idx = split['test_subjects']
        
        train_indices = np.concatenate([train_sub_idx, train_sub_idx + ihb_n_sub])
        test_indices = np.concatenate([test_sub_idx, test_sub_idx + ihb_n_sub])
        
        # Build training set: All China + Few-shot IHB
        # Note: China and IHB have different TRs (240 vs 120), so we cannot concat raw arrays.
        # We must create a list of arrays for ComputeFC
        
        # China part (convert to list of arrays)
        ts_train_list = list(china_ts)
        
        # IHB part (append selected subjects)
        ihb_train_subset = ihb_ts[train_indices]
        ts_train_list.extend(list(ihb_train_subset))
        
        y_train_fs = np.concatenate([china_y, ihb_y[train_indices]], axis=0)
        
        # Test set: Remaining IHB
        ts_test_fs = ihb_ts[test_indices]
        y_test_fs = ihb_y[test_indices]
        
        # Handle Glasso slicing
        g_train_fs, g_test_fs = None, None
        if glasso_ihb is not None and glasso_china is not None:
            g_train_fs = np.concatenate([glasso_china, glasso_ihb[train_indices]], axis=0)
            g_test_fs = glasso_ihb[test_indices]

        # Compute FC
        fc_data = compute_fc_train_test(
            ts_train_list, ts_test_fs,
            kinds=fc_types,
            coverage_mask=coverage_mask,
            glasso_train=g_train_fs,
            glasso_test=g_test_fs,
            glasso_lambda=config.glasso_lambda,
        )

        for fc_type, (X_train, X_test) in fc_data.items():
            for model_name in config.models:
                # Skip permutation for few-shot to save time
                clf_res = run_classification_with_permutation(
                    X_train, y_train_fs, X_test, y_test_fs,
                    pca_components=config.pca_components,
                    random_state=config.random_state,
                    n_permutations=0,
                    model=model_name,
                    model_params=config.model_params,
                    scale=config.scale,
                    vectorize=False,
                )
                
                row = {
                    'repeat': split['repeat'],
                    'fc_type': fc_type,
                    'model': model_name,
                    'atlas': config.atlas,
                    'strategy': config.strategy,
                    'gsr': config.gsr,
                    'n_train': len(y_train_fs),
                    'n_test': len(y_test_fs),
                    'n_china': len(china_y),
                    'n_ihb_fewshot': len(train_indices),
                }
                for k, v in clf_res.items():
                    if k not in ['test_y_pred', 'test_score', 'test_p_positive', 'model', 'model_params']:
                        row[k] = v
                few_shot_rows.append(row)

    few_shot_df = pd.DataFrame(few_shot_rows)
    
    # Aggregate few-shot summary
    if not few_shot_df.empty:
        few_shot_summary = few_shot_df.groupby(['fc_type', 'model']).agg({
            'test_acc': ['mean', 'std'],
            'test_auc': ['mean', 'std'],
            'test_brier': ['mean', 'std'],
        }).round(4)
        # Flatten columns
        few_shot_summary.columns = [f"few_shot_{c[0]}_{c[1]}" for c in few_shot_summary.columns]
        few_shot_summary = few_shot_summary.reset_index()
    else:
        few_shot_summary = pd.DataFrame()

    # 5. Create Combined Summary
    summary_rows = []
    for fc_type in fc_types:
        for model_name in config.models:
            row = {
                'atlas': config.atlas,
                'strategy': config.strategy,
                'gsr': config.gsr,
                'fc_type': fc_type,
                'model': model_name,
            }
            
            # Cross-site
            for d in ['china2ihb', 'ihb2china']:
                df = cross_site_results[d]['results']
                if not df.empty:
                    fc_row = df[(df['fc_type'] == fc_type) & (df['model'] == model_name)]
                    if not fc_row.empty:
                        rec = fc_row.iloc[0]
                        row[f'{d}_acc'] = rec.get('test_acc')
                        row[f'{d}_auc'] = rec.get('test_auc')
                        row[f'{d}_pval'] = rec.get('p_value')

            # Few-shot
            if not few_shot_summary.empty:
                fs_row = few_shot_summary[(few_shot_summary['fc_type'] == fc_type) & (few_shot_summary['model'] == model_name)]
                if not fs_row.empty:
                    rec = fs_row.iloc[0]
                    row['few_shot_acc_mean'] = rec.get('few_shot_test_acc_mean')
                    row['few_shot_acc_std'] = rec.get('few_shot_test_acc_std')
                    row['few_shot_auc_mean'] = rec.get('few_shot_test_auc_mean')

            summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)

    # 6. Save Results
    if config.output_dir:
        output_path = Path(config.output_dir)
        pipeline_dir = output_path / f"{config.atlas}_strategy-{config.strategy}_{config.gsr}"
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        cross_site_results['china2ihb']['results'].to_csv(pipeline_dir / "cross_site_china2ihb_results.csv", index=False)
        cross_site_results['china2ihb']['outputs'].to_csv(pipeline_dir / "cross_site_china2ihb_test_outputs.csv", index=False)
        
        cross_site_results['ihb2china']['results'].to_csv(pipeline_dir / "cross_site_ihb2china_results.csv", index=False)
        cross_site_results['ihb2china']['outputs'].to_csv(pipeline_dir / "cross_site_ihb2china_test_outputs.csv", index=False)
        
        few_shot_df.to_csv(pipeline_dir / "few_shot_results.csv", index=False)
        summary_df.to_csv(pipeline_dir / "summary.csv", index=False)
        
        print(f"Results saved to: {pipeline_dir}")

    return PipelineResults(
        config=config,
        cross_site_china2ihb=cross_site_results['china2ihb']['results'],
        cross_site_ihb2china=cross_site_results['ihb2china']['results'],
        cross_site_china2ihb_outputs=cross_site_results['china2ihb']['outputs'],
        cross_site_ihb2china_outputs=cross_site_results['ihb2china']['outputs'],
        few_shot=few_shot_df,
        few_shot_summary=few_shot_summary,
        summary=summary_df,
    )


import yaml

def main():
    parser = argparse.ArgumentParser(description='Unified ML Pipeline for EOEC Benchmarking')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    
    # CLI args (override config if provided)
    parser.add_argument('--atlas', help='Atlas name')
    parser.add_argument('--strategy', help='Denoising strategy')
    parser.add_argument('--gsr', choices=['GSR', 'noGSR'])
    parser.add_argument('--fc-types', nargs='+', help='FC types (e.g. corr partial)')
    
    parser.add_argument('--skip-glasso', action='store_true', help='Skip glasso computation')
    parser.add_argument('--precomputed-glasso', type=str, help='Path to precomputed glasso dir')
    
    parser.add_argument('--models', nargs='+', help='List of models (e.g. logreg svm_rbf)')
    parser.add_argument('--n-permutations', type=int, help='Number of permutations for significance testing')
    
    parser.add_argument('--data-path', type=str, help='Data root directory')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--no-scale', action='store_true', help='Disable StandardScaler')
    
    args = parser.parse_args()

    # Load config from YAML if provided
    config_dict = {}
    if args.config:
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)

    # Helper to resolve value from args > config > default
    def get_val(arg_val, config_key, default=None):
        if arg_val is not None and arg_val is not False: # Explicit CLI arg (excluding False for store_true)
             return arg_val
        
        # Traverse config dict
        keys = config_key.split('.')
        val = config_dict
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default

    def to_list(val):
        if val is None: return []
        return [val] if isinstance(val, str) else list(val)

    # Single run
    cfg = PipelineConfig(
        atlas=get_val(args.atlas, 'pipeline.atlas'),
        strategy=get_val(args.strategy, 'pipeline.strategy'),
        gsr=get_val(args.gsr, 'pipeline.gsr'),
        fc_types=to_list(get_val(args.fc_types, 'pipeline.fc_types', ['corr', 'partial', 'tangent', 'glasso'])),
        models=to_list(get_val(args.models, 'pipeline.models', ['logreg'])),
        skip_glasso=get_val(args.skip_glasso, 'pipeline.skip_glasso', False),
        precomputed_glasso_dir=get_val(args.precomputed_glasso, 'pipeline.precomputed_glasso_dir'),
        n_permutations=get_val(args.n_permutations, 'pipeline.n_permutations', 0),
        data_path=get_val(args.data_path, 'data.data_path'),
        output_dir=get_val(args.output_dir, 'output.output_dir', 'results/pipelines'),
        n_few_shot=get_val(None, 'few_shot.n_few_shot', 10),
        n_repeats=get_val(None, 'few_shot.n_repeats', 10),
        scale=not get_val(args.no_scale, 'pipeline.no_scale', False),
    )
    
    results = run_full_pipeline(cfg)
    print("\nSUMMARY:")
    print(results.summary.to_string(index=False))


if __name__ == '__main__':
    main()
