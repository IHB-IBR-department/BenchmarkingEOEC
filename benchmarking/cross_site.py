#!/usr/bin/env python3
"""
Honest Cross-Site Validation (Scheme A)
=======================================

This script implements cross-site generalization experiments where:
- Training is performed on one site (scanner)
- Testing is performed on a different site (scanner)

This provides TRUE cross-site validation without any data leakage,
addressing Reviewer 1's concern about the "cross-site validation" misnomer.

Experimental Design:
-------------------
Direction 1: Beijing (Siemens 3T) → IHB RAS (Philips 3T)
Direction 2: IHB RAS (Philips 3T) → Beijing (Siemens 3T)

When both datasets are specified for train_on and test_on in config,
bidirectional experiments are automatically run.

Pipeline Parameters:
-------------------
- Atlases: AAL (116), Schaefer200 (200), Brainnetome (246), HCPex (426)
- FC types: corr, partial, tangent, glasso
- Denoising: Strategies 1-6 (24P, aCompCor variants, etc.)
- GSR: with/without global signal regression

Usage:
------
    # Quick test
    python benchmarking/cross_site.py --config configs/cross_site_quick.yaml

    # Full experiment (all 384 combinations)
    python benchmarking/cross_site.py --config configs/cross_site_full.yaml

Output:
-------
- {output}_results.csv: Per-pipeline results
- {output}_summary.csv: Aggregated statistics by direction, FC type, etc.

Author: BenchmarkingEOEC Team
"""

import os
import argparse
import warnings
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Import classification with permutation testing from stats module
from benchmarking.stats import run_classification_with_permutation
from benchmarking.project import resolve_data_root, standard_fc_path, glasso_path


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_fc_standard(
    site: str,
    fc_type: str,
    atlas: str,
    strategy: int,
    gsr: str,
    data_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load standard FC matrices (corr, partial, tangent) for a site.

    Parameters
    ----------
    site : str
        Site identifier: 'china' (Beijing) or 'ihb'
    fc_type : str
        FC type: 'corr', 'partial', or 'tangent'
    atlas : str
        Atlas name: 'AAL', 'Schaefer200', 'Brainnetome', 'HCPex'
    strategy : int
        Denoising strategy (1-6)
    gsr : str
        'GSR' or 'noGSR'
    data_path : str
        Base path to OpenCloseBenchmark_data

    Returns
    -------
    X : np.ndarray
        FC matrices, shape (n_samples, n_rois, n_rois)
    y : np.ndarray
        Labels, shape (n_samples,) - 0=EC (closed), 1=EO (open)
    """
    data_path = resolve_data_root(data_path)

    if site == 'ihb':
        # IHB has simple structure: ihb_{close/open}_{fc}_{atlas}_strategy-{n}_{gsr}.npy
        close_file = standard_fc_path(
            data_root=data_path,
            site="ihb",
            session_or_condition="close",
            fc_type=fc_type,
            atlas=atlas,
            strategy=strategy,
            gsr=gsr,
        )
        open_file = standard_fc_path(
            data_root=data_path,
            site="ihb",
            session_or_condition="open",
            fc_type=fc_type,
            atlas=atlas,
            strategy=strategy,
            gsr=gsr,
        )

        X_close = np.load(close_file)  # (84, n_rois, n_rois)
        X_open = np.load(open_file)    # (84, n_rois, n_rois)

    elif site == 'china':
        # Beijing has complex structure with multiple sessions
        # Load close1 for EC
        close_file = standard_fc_path(
            data_root=data_path,
            site="china",
            session_or_condition="close1",
            fc_type=fc_type,
            atlas=atlas,
            strategy=strategy,
            gsr=gsr,
        )
        X_close = np.load(close_file)  # (46, n_rois, n_rois)

        # Load open2 and open3, merge with subject indices
        open2_file = standard_fc_path(
            data_root=data_path,
            site="china",
            session_or_condition="open2",
            fc_type=fc_type,
            atlas=atlas,
            strategy=strategy,
            gsr=gsr,
        )
        open3_file = standard_fc_path(
            data_root=data_path,
            site="china",
            session_or_condition="open3",
            fc_type=fc_type,
            atlas=atlas,
            strategy=strategy,
            gsr=gsr,
        )

        X_open2 = np.load(open2_file)  # (21, n_rois, n_rois)
        X_open3 = np.load(open3_file)  # (25, n_rois, n_rois)

        # Subject indices for merging open sessions
        sub_idx = {
            2: [1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 20, 22, 24, 26, 28, 30, 33, 35, 37, 39, 41, 42, 43, 44, 46],
            3: [0, 2, 4, 6, 8, 10, 12, 14, 16, 19, 21, 25, 27, 29, 31, 32, 34, 36, 38, 40, 45]
        }

        # Merge open sessions
        n_rois = X_open2.shape[1]
        X_open = np.zeros((47, n_rois, n_rois), dtype=X_open2.dtype)
        X_open[sub_idx[3]] = X_open2
        X_open[sub_idx[2]] = X_open3

        # Remove subject 23 (missing data)
        X_open = np.delete(X_open, 23, axis=0)  # (46, n_rois, n_rois)

    else:
        raise ValueError(f"Unknown site: {site}. Must be 'china' or 'ihb'")

    # Combine EC and EO
    n_subjects = X_close.shape[0]
    X = np.concatenate([X_close, X_open], axis=0)  # (n_samples, n_rois, n_rois)
    y = np.array([0] * n_subjects + [1] * n_subjects)  # 0=EC, 1=EO

    return X, y


def load_fc_glasso(
    site: str,
    atlas: str,
    strategy: int,
    gsr: str,
    data_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Glasso FC matrices for a site.

    Glasso files contain BOTH conditions combined:
    - First half: Eyes Closed (EC)
    - Second half: Eyes Open (EO)

    Parameters
    ----------
    site : str
        Site identifier: 'china' or 'ihb'
    atlas : str
        Atlas name
    strategy : int
        Denoising strategy (1-6)
    gsr : str
        'GSR' or 'noGSR'
    data_path : str
        Base path to OpenCloseBenchmark_data

    Returns
    -------
    X : np.ndarray
        FC matrices, shape (n_samples, n_rois, n_rois)
    y : np.ndarray
        Labels, shape (n_samples,) - 0=EC, 1=EO
    """
    data_path = resolve_data_root(data_path)
    glasso_file = glasso_path(data_root=data_path, site=site, atlas=atlas, strategy=strategy, gsr=gsr)

    X = np.load(glasso_file)  # (n_subjects*2, n_rois, n_rois)
    n_subjects = X.shape[0] // 2

    # First half = EC, second half = EO
    y = np.array([0] * n_subjects + [1] * n_subjects)

    return X, y


def load_fc_data(
    site: str,
    fc_type: str,
    atlas: str,
    strategy: int,
    gsr: str,
    data_path: str,
    bad_roi_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load FC matrices for a single pipeline configuration.

    Dispatches to appropriate loader based on FC type.

    Parameters
    ----------
    site : str
        Site identifier: 'china' or 'ihb'
    fc_type : str
        FC type: 'corr', 'partial', 'tangent', or 'glasso'
    atlas : str
        Atlas name
    strategy : int
        Denoising strategy (1-6)
    gsr : str
        'GSR' or 'noGSR'
    data_path : str
        Base path to OpenCloseBenchmark_data

    Returns
    -------
    X : np.ndarray
        FC matrices, shape (n_samples, n_rois, n_rois)
    y : np.ndarray
        Labels, shape (n_samples,) - 0=EC, 1=EO
    """
    if fc_type == 'glasso':
        X, y = load_fc_glasso(site, atlas, strategy, gsr, data_path)
    else:
        X, y = load_fc_standard(site, fc_type, atlas, strategy, gsr, data_path)

    if bad_roi_mask is not None:
        X = apply_bad_roi_mask(X, bad_roi_mask)

    return X, y


# =============================================================================
# Coverage-based ROI masking
# =============================================================================

_BAD_ROI_MASK_CACHE: Dict[tuple, np.ndarray] = {}


def apply_bad_roi_mask(X: np.ndarray, bad_roi_mask: np.ndarray) -> np.ndarray:
    """
    Zero out FC edges that touch a bad ROI (rows/cols for bad ROIs).
    """
    bad_roi_mask = np.asarray(bad_roi_mask, dtype=bool).reshape(-1)
    if X.shape[1] != bad_roi_mask.shape[0]:
        raise ValueError(f"Bad ROI mask length {bad_roi_mask.shape[0]} does not match FC size {X.shape[1]}")
    if not bad_roi_mask.any():
        return X
    X = np.asarray(X)
    X[:, bad_roi_mask, :] = 0.0
    X[:, :, bad_roi_mask] = 0.0
    return X


def load_bad_roi_mask(
    atlas: str,
    *,
    data_path: Optional[str],
    threshold: float,
    mask_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Load or compute bad-ROI mask for an atlas.

    If mask_dir is provided, loads `<atlas>_bad_parcels.npy` from that folder.
    Otherwise computes from coverage arrays in `<data_root>/coverage`.
    """
    if threshold <= 0 or threshold >= 1:
        raise ValueError("coverage threshold must be in (0, 1)")
    cache_key = (atlas, float(threshold), str(mask_dir), str(data_path))
    cached = _BAD_ROI_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if mask_dir:
        mask_path = Path(mask_dir) / f"{atlas}_bad_parcels.npy"
        if not mask_path.exists():
            raise FileNotFoundError(f"Bad ROI mask not found: {mask_path}")
        mask = np.load(mask_path).astype(bool)
        _BAD_ROI_MASK_CACHE[cache_key] = mask
        return mask

    data_root = resolve_data_root(data_path)
    coverage_dir = Path(data_root) / "coverage"
    china_path = coverage_dir / f"china_{atlas}_parcel_coverage.npy"
    ihb_path = coverage_dir / f"ihb_{atlas}_parcel_coverage.npy"
    if not china_path.exists() or not ihb_path.exists():
        raise FileNotFoundError(f"Missing coverage files for atlas {atlas} in {coverage_dir}")

    china = np.load(china_path).astype(float)
    ihb = np.load(ihb_path).astype(float)
    if china.shape != ihb.shape:
        raise ValueError(f"Coverage shape mismatch for atlas {atlas}: {china.shape} vs {ihb.shape}")

    mask = (china < threshold) | (ihb < threshold)
    _BAD_ROI_MASK_CACHE[cache_key] = mask
    return mask


def prepare_bad_roi_masks(config: Dict[str, Any]) -> None:
    """
    Precompute bad-ROI masks per atlas and store them in a folder.

    This updates config['coverage_mask']['mask_dir'] when enabled and no
    mask_dir is provided, to avoid recomputing per pipeline.
    """
    coverage_cfg = config.get("coverage_mask") or {}
    if not coverage_cfg.get("enabled", False):
        return

    mask_dir = coverage_cfg.get("mask_dir")
    if mask_dir:
        return

    threshold = float(coverage_cfg.get("threshold", 0.1))
    if threshold <= 0 or threshold >= 1:
        raise ValueError("coverage threshold must be in (0, 1)")
    output_path = Path(config["output"])
    mask_dir_path = output_path.parent / "coverage_masks"
    mask_dir_path.mkdir(parents=True, exist_ok=True)

    for atlas in config.get("atlases", []):
        mask = load_bad_roi_mask(
            atlas,
            data_path=config.get("data_path"),
            threshold=threshold,
            mask_dir=None,
        )
        np.save(mask_dir_path / f"{atlas}_bad_parcels.npy", mask.astype(bool))

    coverage_cfg["mask_dir"] = str(mask_dir_path)
    coverage_cfg["threshold"] = threshold
    config["coverage_mask"] = coverage_cfg


# =============================================================================
# Pipeline Processing
# =============================================================================

def process_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run single pipeline experiment: load data, train, evaluate.

    Parameters
    ----------
    config : dict
        Pipeline configuration with keys:
        - train_site: Site for training ('china' or 'ihb')
        - test_site: Site for testing ('china' or 'ihb')
        - atlas: Atlas name
        - fc_type: FC type
        - strategy: Denoising strategy
        - gsr: GSR option
        - data_path: Base data path
        - pca_components: PCA variance to retain
        - random_state: Random seed
        - n_permutations: Number of permutations for statistical test (0 to skip)

    Returns
    -------
    dict
        Results with all config params plus:
        - n_train, n_test: Sample counts
        - train_acc, test_acc: Accuracies
        - n_pca_components: Number of PCA components
        - p_value: Permutation test p-value (None if n_permutations=0)
        - error: Error message if failed (None otherwise)
    """
    result = {
        'train_site': config['train_site'],
        'test_site': config['test_site'],
        'atlas': config['atlas'],
        'fc_type': config['fc_type'],
        'strategy': config['strategy'],
        'gsr': config['gsr'],
        'model': config.get('model'),
        'model_params': config.get('model_params'),
        'n_train': None,
        'n_test': None,
        'train_acc': None,
        'test_acc': None,
        'train_auc': None,
        'test_auc': None,
        'train_brier': None,
        'test_brier': None,
        'test_y_pred': None,
        'test_score': None,
        'test_p_positive': None,
        'n_pca_components': None,
        'p_value': None,
        'error': None
    }

    try:
        bad_roi_mask = None
        if config.get("coverage_mask_enabled"):
            bad_roi_mask = load_bad_roi_mask(
                config["atlas"],
                data_path=config.get("data_path"),
                threshold=config.get("coverage_mask_threshold", 0.1),
                mask_dir=config.get("coverage_mask_dir"),
            )

        # Load training data
        X_train, y_train = load_fc_data(
            site=config['train_site'],
            fc_type=config['fc_type'],
            atlas=config['atlas'],
            strategy=config['strategy'],
            gsr=config['gsr'],
            data_path=config['data_path'],
            bad_roi_mask=bad_roi_mask,
        )

        # Load test data
        X_test, y_test = load_fc_data(
            site=config['test_site'],
            fc_type=config['fc_type'],
            atlas=config['atlas'],
            strategy=config['strategy'],
            gsr=config['gsr'],
            data_path=config['data_path'],
            bad_roi_mask=bad_roi_mask,
        )

        result['n_train'] = len(y_train)
        result['n_test'] = len(y_test)

        # Run classification with optional permutation testing
        clf_results = run_classification_with_permutation(
            X_train, y_train,
            X_test, y_test,
            pca_components=config.get('pca_components', 0.95),
            random_state=config.get('random_state', 42),
            n_permutations=config.get('n_permutations', 0),
            model=config.get('model', 'logreg'),
            model_params=config.get('model_params'),
            scale=config.get('scale'),
            return_test_outputs=config.get('save_test_outputs', False),
        )

        result.update(clf_results)

    except Exception as e:
        result['error'] = str(e)
        warnings.warn(f"Pipeline failed: {config} - {e}")

    return result


def generate_pipeline_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate all pipeline configurations from YAML config.

    If config specifies both sites for train_on and test_on,
    generates bidirectional experiments (excluding same-site pairs).

    Parameters
    ----------
    config : dict
        Parsed YAML config with:
        - train_on: List of sites for training
        - test_on: List of sites for testing
        - atlases: List of atlases
        - fc_types: List of FC types
        - strategies: List of strategies
        - gsr_options: List of GSR options
        - data_path, pca_components, random_state

    Returns
    -------
    list
        List of pipeline config dicts
    """
    train_sites = config['train_on'] if isinstance(config['train_on'], list) else [config['train_on']]
    test_sites = config['test_on'] if isinstance(config['test_on'], list) else [config['test_on']]

    pipeline_configs = []
    coverage_cfg = config.get("coverage_mask") or {}
    coverage_enabled = bool(coverage_cfg.get("enabled", False))
    coverage_threshold = float(coverage_cfg.get("threshold", 0.1))
    coverage_mask_dir = coverage_cfg.get("mask_dir")

    for train_site, test_site, atlas, fc_type, strategy, gsr in product(
        train_sites,
        test_sites,
        config['atlases'],
        config['fc_types'],
        config['strategies'],
        config['gsr_options']
    ):
        # Skip same-site pairs (this is cross-site validation)
        if train_site == test_site:
            continue

        pipeline_configs.append({
            'train_site': train_site,
            'test_site': test_site,
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
            'save_test_outputs': bool(config.get('save_test_outputs', False) or config.get('return_probabilities', False)),
            'coverage_mask_enabled': coverage_enabled,
            'coverage_mask_threshold': coverage_threshold,
            'coverage_mask_dir': coverage_mask_dir,
        })

    return pipeline_configs


def run_all_pipelines(
    config: Dict[str, Any],
    n_workers: Optional[int] = None
) -> pd.DataFrame:
    """
    Run all pipeline experiments with multiprocessing.

    Parameters
    ----------
    config : dict
        Parsed YAML config
    n_workers : int, optional
        Number of parallel workers. If None, uses config value or defaults to 1.

    Returns
    -------
    pd.DataFrame
        Results for all pipelines
    """
    prepare_bad_roi_masks(config)
    pipeline_configs = generate_pipeline_configs(config)
    n_workers = n_workers or config.get('n_workers', 1)

    print(f"Running {len(pipeline_configs)} pipeline configurations with {n_workers} workers...")

    results = []

    if n_workers == 1:
        # Sequential execution (useful for debugging)
        for cfg in tqdm(pipeline_configs, desc="Pipelines"):
            results.append(process_pipeline(cfg))
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_pipeline, cfg): cfg for cfg in pipeline_configs}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Pipelines"):
                results.append(future.result())

    return pd.DataFrame(results)


# =============================================================================
# Results Aggregation
# =============================================================================

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics from results.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe

    Returns
    -------
    pd.DataFrame
        Summary statistics grouped by direction, FC type, atlas
    """
    # Filter out failed pipelines
    df_valid = df[df['error'].isna()].copy()

    if df_valid.empty:
        return pd.DataFrame()

    # Add direction column
    df_valid['direction'] = df_valid['train_site'] + ' → ' + df_valid['test_site']

    # Summary by FC type
    fc_summary = df_valid.groupby(['direction', 'fc_type']).agg({
        'test_acc': ['mean', 'std'],
        'test_auc': ['mean', 'std'],
        'test_brier': ['mean', 'std'],
    }).round(4)
    fc_summary.columns = [
        'test_acc_mean', 'test_acc_std',
        'test_auc_mean', 'test_auc_std',
        'test_brier_mean', 'test_brier_std',
    ]
    fc_summary = fc_summary.reset_index()

    # Summary by atlas
    atlas_summary = df_valid.groupby(['direction', 'atlas']).agg({
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
    gsr_summary = df_valid.groupby(['direction', 'gsr']).agg({
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

    return {
        'by_fc_type': fc_summary,
        'by_atlas': atlas_summary,
        'by_gsr': gsr_summary
    }


def save_results(
    df: pd.DataFrame,
    output_path: str,
    summaries: Dict[str, pd.DataFrame]
) -> None:
    """
    Save results and summaries to CSV files.

    Parameters
    ----------
    df : pd.DataFrame
        Full results
    output_path : str
        Base output path (without extension)
    summaries : dict
        Summary dataframes
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save full results
    results_file = str(output_path).replace('.csv', '_results.csv')
    df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")

    # Save summaries
    for name, summary_df in summaries.items():
        if not summary_df.empty:
            summary_file = str(output_path).replace('.csv', f'_summary_{name}.csv')
            summary_df.to_csv(summary_file, index=False)
            print(f"Summary ({name}) saved to: {summary_file}")


def build_test_outputs_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand per-pipeline test outputs (predictions/scores) into long-form rows.

    This enables paired pipeline comparison tests (e.g., Exact McNemar for
    accuracy and DeLong for ROC-AUC) by keeping per-test-sample outputs.
    """
    if df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for _, row in df[df["error"].isna()].iterrows():
        test_y_pred = row.get("test_y_pred")
        if test_y_pred is None or (isinstance(test_y_pred, float) and pd.isna(test_y_pred)):
            continue

        y_pred = np.asarray(test_y_pred, dtype=int)
        n_test = int(y_pred.shape[0])
        if n_test % 2 != 0:
            raise ValueError(f"Expected even n_test for EO/EC pairing, got {n_test}")

        n_subjects = n_test // 2
        y_true = np.concatenate([np.zeros(n_subjects, dtype=int), np.ones(n_subjects, dtype=int)])
        test_subject = np.tile(np.arange(n_subjects, dtype=int), 2)

        test_score_raw = row.get("test_score")
        if test_score_raw is None or (isinstance(test_score_raw, float) and pd.isna(test_score_raw)):
            y_score = np.full(n_test, np.nan, dtype=float)
        else:
            y_score = np.asarray(test_score_raw, dtype=float)

        test_p_raw = row.get("test_p_positive")
        if test_p_raw is None or (isinstance(test_p_raw, float) and pd.isna(test_p_raw)):
            p_positive = np.full(n_test, np.nan, dtype=float)
        else:
            p_positive = np.asarray(test_p_raw, dtype=float)

        if y_true.shape[0] != n_test:
            raise ValueError(f"Expected y_true length {n_test}, got {y_true.shape[0]}")
        if y_score.shape[0] != n_test:
            raise ValueError(f"Expected test_score length {n_test}, got {y_score.shape[0]}")
        if p_positive.shape[0] != n_test:
            raise ValueError(f"Expected test_p_positive length {n_test}, got {p_positive.shape[0]}")

        model_params = row.get("model_params") or {}
        if isinstance(model_params, dict):
            model_params = json.dumps(model_params, sort_keys=True)

        base = {
            "train_site": row.get("train_site"),
            "test_site": row.get("test_site"),
            "direction": f"{row.get('train_site')} → {row.get('test_site')}",
            "atlas": row.get("atlas"),
            "fc_type": row.get("fc_type"),
            "strategy": row.get("strategy"),
            "gsr": row.get("gsr"),
            "model": row.get("model"),
            "model_params": model_params,
        }

        for i in range(n_test):
            rows.append(
                {
                    **base,
                    "sample_index": int(i),
                    "test_subject": int(test_subject[i]),
                    "y_true": int(y_true[i]),
                    "y_pred": int(y_pred[i]),
                    "y_score": float(y_score[i]),
                    "p_positive": float(p_positive[i]) if not np.isnan(p_positive[i]) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def normalize_model_params(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return json.dumps({})
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def build_pipeline_abbrev_map(df_test_outputs: pd.DataFrame) -> pd.DataFrame:
    """
    Create a stable pipeline abbreviation (column name) for each unique pipeline spec.

    Abbreviations are short IDs (P0001, P0002, ...) to keep wide CSVs manageable.
    Direction is intentionally excluded so the same pipeline maps to the same abbrev.
    """
    if df_test_outputs.empty:
        return pd.DataFrame()

    key_cols = [
        "atlas",
        "fc_type",
        "strategy",
        "gsr",
        "model",
        "model_params",
    ]

    pipelines = (
        df_test_outputs[key_cols]
        .drop_duplicates()
        .sort_values(key_cols, kind="mergesort")
        .reset_index(drop=True)
    )

    pipelines["abbrev"] = [f"P{i + 1:04d}" for i in range(len(pipelines))]
    pipelines["spec"] = (
        "atlas=" + pipelines["atlas"].astype(str)
        + ", fc_type=" + pipelines["fc_type"].astype(str)
        + ", strategy=" + pipelines["strategy"].astype(str)
        + ", gsr=" + pipelines["gsr"].astype(str)
        + ", model=" + pipelines["model"].astype(str)
        + ", model_params=" + pipelines["model_params"].astype(str)
    )

    return pipelines[["abbrev", *key_cols, "spec"]]


def save_pipeline_predictions(
    df_test_outputs: pd.DataFrame,
    pipeline_map: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Save wide prediction matrices for each cross-site direction.

    Output shape (per direction):
    - rows: test samples
    - columns: `sub_id`, `true_state`, then one column per pipeline abbrev
    """
    if df_test_outputs.empty or pipeline_map.empty:
        return

    key_cols = [
        "atlas",
        "fc_type",
        "strategy",
        "gsr",
        "model",
        "model_params",
    ]

    df_m = df_test_outputs.merge(pipeline_map[["abbrev", *key_cols]], on=key_cols, how="left")
    if df_m["abbrev"].isna().any():
        raise ValueError("Failed to assign pipeline abbreviations to some test-output rows.")

    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_m["train_site"].nunique() == 1 and df_m["test_site"].nunique() == 1:
        grouped = [((df_m["train_site"].iloc[0], df_m["test_site"].iloc[0]), df_m)]
    else:
        grouped = df_m.groupby(["train_site", "test_site"], sort=True)

    for (train_site, test_site), df_dir in grouped:
        wide = (
            df_dir.pivot_table(
                index=["test_subject", "y_true"],
                columns="abbrev",
                values="y_pred",
                aggfunc="first",
            )
            .reset_index()
            .rename(columns={"test_subject": "sub_id", "y_true": "true_state"})
        )

        # Stable row/column order
        wide = wide.sort_values(["sub_id", "true_state"], kind="mergesort")
        fixed_cols = ["sub_id", "true_state"]
        pipeline_cols = sorted([c for c in wide.columns if c not in fixed_cols])
        wide = wide[fixed_cols + pipeline_cols]
        wide["sub_id"] = wide["sub_id"].astype(int)
        wide["true_state"] = wide["true_state"].astype(int)
        for c in pipeline_cols:
            wide[c] = wide[c].astype("Int64")

        if df_m["train_site"].nunique() == 1 and df_m["test_site"].nunique() == 1:
            preds_file = output_dir / "pipeline_predictions.csv"
        else:
            preds_file = output_dir / f"pipeline_predictions_{train_site}_to_{test_site}.csv"
        wide.to_csv(preds_file, index=False)
        print(f"Pipeline predictions saved to: {preds_file}")


def direction_slug(train_site: str, test_site: str) -> str:
    return f"{train_site}2{test_site}"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Honest Cross-Site Validation (Scheme A)',
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
    print(f"Train on: {config['train_on']}")
    print(f"Test on: {config['test_on']}")
    print(f"Atlases: {config['atlases']}")
    print(f"FC types: {config['fc_types']}")
    print(f"Strategies: {config['strategies']}")
    print(f"GSR options: {config['gsr_options']}")
    print(
        "Save per-sample test outputs: "
        f"{bool(config.get('save_test_outputs', False) or config.get('return_probabilities', False))}"
    )
    if isinstance(config.get("model"), dict):
        print(
            "Model: "
            f"{config['model'].get('name', 'logreg')} "
            f"(scale={config['model'].get('scale', '(auto)')}, params={config['model'].get('params', {})})"
        )
    else:
        print(f"Model: {config.get('model', 'logreg')}")

    # Run pipelines
    n_workers = args.n_workers or config.get('n_workers', 1)
    df = run_all_pipelines(config, n_workers=n_workers)

    # Check for errors
    n_errors = df['error'].notna().sum()
    if n_errors > 0:
        print(f"\nWarning: {n_errors} pipelines failed!")
        print(df[df['error'].notna()][['train_site', 'test_site', 'atlas', 'fc_type', 'strategy', 'gsr', 'error']])

    # Drop per-sample columns from the main results CSV
    df_results = df.drop(columns=["test_y_pred", "test_score", "test_p_positive"], errors="ignore")
    if "model_params" in df_results.columns:
        df_results["model_params"] = df_results["model_params"].apply(normalize_model_params)

    output_base = Path(config["output"]).expanduser()
    if output_base.parent.name == "cross_site":
        output_root = output_base.parent
    else:
        output_root = output_base.parent / "cross_site"

    # Optionally build per-sample test outputs for paired comparisons (McNemar, DeLong)
    df_test_outputs = None
    if config.get("save_test_outputs", False) or config.get("return_probabilities", False):
        df_test_outputs = build_test_outputs_df(df)

    for (train_site, test_site), df_dir in df_results.groupby(["train_site", "test_site"], sort=True):
        direction_dir = output_root / direction_slug(train_site, test_site)
        direction_dir.mkdir(parents=True, exist_ok=True)
        output_path = direction_dir / output_base.name

        df_dir_out = df_dir.copy()
        pipeline_map = None

        if df_test_outputs is not None:
            df_test_dir = df_test_outputs[
                (df_test_outputs["train_site"] == train_site)
                & (df_test_outputs["test_site"] == test_site)
            ]
            test_outputs_file = str(output_path).replace(".csv", "_test_outputs.csv")
            df_test_dir.to_csv(test_outputs_file, index=False)
            print(f"Test outputs saved to: {test_outputs_file}")

            pipeline_map = build_pipeline_abbrev_map(df_test_dir)
            if not pipeline_map.empty:
                pipeline_map_file = str(output_path).replace(".csv", "_pipeline_abbreviations.csv")
                pipeline_map.to_csv(pipeline_map_file, index=False)
                print(f"Pipeline abbreviations saved to: {pipeline_map_file}")
                key_cols = ["atlas", "fc_type", "strategy", "gsr", "model", "model_params"]
                df_dir_out = df_dir_out.merge(
                    pipeline_map[["abbrev", *key_cols]],
                    on=key_cols,
                    how="left",
                ).rename(columns={"abbrev": "pipeline_id"})
                save_pipeline_predictions(df_test_dir, pipeline_map, str(output_path))

        summaries = compute_summary(df_dir_out)
        save_results(df_dir_out, str(output_path), summaries)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY BY FC TYPE")
    print("=" * 60)
    summaries_all = compute_summary(df_results)
    if 'by_fc_type' in summaries_all and not summaries_all['by_fc_type'].empty:
        print(summaries_all['by_fc_type'].to_string(index=False))


if __name__ == '__main__':
    main()
