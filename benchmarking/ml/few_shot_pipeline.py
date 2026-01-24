"""
Few-Shot Domain Adaptation Pipeline
===================================

This module implements the few-shot domain adaptation logic separated from the 
main cross-site pipeline. It allows running only the few-shot evaluation with 
an arbitrary number of repeats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_utils.timeseries import (
    load_timeseries,
    load_ihb_coverage_mask,
    load_precomputed_glasso,
)
from data_utils.hcpex import preprocess_hcpex_timeseries
from data_utils.fc import compute_fc_train_test
from benchmarking.ml.stats import run_classification_with_permutation
from benchmarking.ml.pipeline import PipelineConfig, generate_few_shot_splits

def run_few_shot_only(config: PipelineConfig) -> pd.DataFrame:
    """Run ONLY the few-shot evaluation for one preprocessing configuration, merging with existing results."""
    
    # Define output path
    output_path = Path(config.output_dir) / f"{config.atlas}_strategy-{config.strategy}_{config.gsr}"
    output_file = output_path / "few_shot_results.csv"
    output_path.mkdir(parents=True, exist_ok=True)

    # 0. Check for existing results
    existing_df = pd.DataFrame()
    completed_repeats = set()
    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file)
            if not existing_df.empty:
                completed_repeats = set(existing_df['repeat'].unique())
                print(f"   Found {len(completed_repeats)} existing repeats in {output_file.name}")
        except Exception as e:
            print(f"   Warning: Could not read existing file {output_file}: {e}")

    # Determine which repeats to run
    # We want to reach config.n_repeats total.
    target_repeats = list(range(config.n_repeats))
    repeats_to_run = [r for r in target_repeats if r not in completed_repeats]

    if not repeats_to_run:
        print(f"   All {config.n_repeats} repeats already completed. Skipping.")
        return existing_df

    # 1. Load Timeseries Data
    print(f"Loading timeseries for {config.atlas} strategy-{config.strategy} {config.gsr}...")
    
    ihb_close = load_timeseries('ihb', 'close', config.atlas, config.strategy, config.gsr, config.data_path)
    ihb_open = load_timeseries('ihb', 'open', config.atlas, config.strategy, config.gsr, config.data_path)
    ihb_ts = np.concatenate([ihb_close, ihb_open], axis=0)
    ihb_n_sub = len(ihb_close)
    ihb_y = np.array([0] * ihb_n_sub + [1] * ihb_n_sub)
    
    china_close = load_timeseries('china', 'close', config.atlas, config.strategy, config.gsr, config.data_path)
    china_open = load_timeseries('china', 'open', config.atlas, config.strategy, config.gsr, config.data_path)
    china_ts = np.concatenate([china_close, china_open], axis=0)
    china_y = np.array([0] * len(china_close) + [1] * len(china_close))

    if config.atlas.upper() == 'HCPEX':
        from data_utils.paths import resolve_data_root
        data_root = resolve_data_root(config.data_path)
        mask_path = data_root / "coverage" / "hcp_mask.npy"
        ihb_ts = preprocess_hcpex_timeseries(ihb_ts, "ihb", mask_path)
        china_ts = preprocess_hcpex_timeseries(china_ts, "china", mask_path)
        coverage_mask = None
    else:
        coverage_mask = load_ihb_coverage_mask(config.atlas, config.data_path, config.coverage_threshold)

    # 2. Handle Glasso
    glasso_ihb = None
    glasso_china = None
    fc_types = config.fc_types[:]
    if 'glasso' in fc_types and config.precomputed_glasso_dir:
        try:
            g_ihb_c = load_precomputed_glasso('ihb', 'close', config.atlas, config.strategy, config.gsr, config.precomputed_glasso_dir)
            g_ihb_o = load_precomputed_glasso('ihb', 'open', config.atlas, config.strategy, config.gsr, config.precomputed_glasso_dir)
            glasso_ihb = np.concatenate([g_ihb_c, g_ihb_o], axis=0)
            g_china_c = load_precomputed_glasso('china', 'close', config.atlas, config.strategy, config.gsr, config.precomputed_glasso_dir)
            g_china_o = load_precomputed_glasso('china', 'open', config.atlas, config.strategy, config.gsr, config.precomputed_glasso_dir)
            glasso_china = np.concatenate([g_china_c, g_china_o], axis=0)
        except FileNotFoundError:
            if 'glasso' in fc_types: fc_types.remove('glasso')

    # 3. Few-Shot Domain Adaptation
    print(f"Running {len(repeats_to_run)} new few-shot repeats (target total: {config.n_repeats})...")
    
    # Generate ALL target splits
    all_splits = generate_few_shot_splits(ihb_n_sub, config.n_few_shot, config.n_repeats, config.random_state)
    
    # Select only the missing ones
    splits_to_run = [all_splits[r] for r in repeats_to_run]
    
    new_rows = []

    for split in tqdm(splits_to_run, desc="Few-shot repeats"):
        train_sub_idx = split['train_subjects']
        test_sub_idx = split['test_subjects']
        
        train_indices = np.concatenate([train_sub_idx, train_sub_idx + ihb_n_sub])
        test_indices = np.concatenate([test_sub_idx, test_sub_idx + ihb_n_sub])
        
        ts_train_list = list(china_ts)
        ihb_train_subset = ihb_ts[train_indices]
        ts_train_list.extend(list(ihb_train_subset))
        y_train_fs = np.concatenate([china_y, ihb_y[train_indices]], axis=0)
        
        ts_test_fs = ihb_ts[test_indices]
        y_test_fs = ihb_y[test_indices]
        
        g_train_fs, g_test_fs = None, None
        if glasso_ihb is not None and glasso_china is not None:
            g_train_fs = np.concatenate([glasso_china, glasso_ihb[train_indices]], axis=0)
            g_test_fs = glasso_ihb[test_indices]

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
                }
                for k, v in clf_res.items():
                    if k not in ['test_y_pred', 'test_score', 'test_p_positive', 'model', 'model_params']:
                        row[k] = v
                new_rows.append(row)

    # 4. Merge and Save
    new_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Sort for consistency
    combined_df = combined_df.sort_values(['repeat', 'fc_type', 'model']).reset_index(drop=True)
    
    combined_df.to_csv(output_file, index=False)
    print(f"Few-shot results updated in: {output_file} (Total repeats: {len(combined_df['repeat'].unique())})")

    return combined_df
