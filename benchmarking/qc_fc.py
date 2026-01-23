"""
QC-FC Analysis Module
=====================

Computes Quality Control - Functional Connectivity (QC-FC) correlations.

QC-FC measures the correlation between motion artifacts (mean RMS) and
functional connectivity edges. High QC-FC values indicate motion contamination
in the FC estimates.

Core function:
    qc_fc_edges(fc_vec, rms_vec) - Compute edge-wise correlation with motion

Approach:
    FC matrices are computed/loaded per CONDITION (close/open).
    RMS values are matched to FC by mapping runs to conditions:
    - IHB: run-1=close, run-2=open
    - China: Variable mapping from BeijingEOEC.csv
      - close session 0 → run-1 (all subjects)
      - close session 1 → run-2 or run-3 (depends on subject)
      - open → run-2 or run-3 (depends on subject)

Usage:
    python -m benchmarking.qc_fc --config configs/qc_fc_atlas.yaml

Author: BenchmarkingEOEC Team
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats
from nilearn.connectome import sym_matrix_to_vec
import yaml
from tqdm import tqdm

from data_utils.paths import resolve_data_root
from data_utils.fc import ConnectomeTransformer


# =============================================================================
# Core QC-FC computation
# =============================================================================

def qc_fc_edges(
    fc_vec: np.ndarray,
    rms_vec: np.ndarray,
    return_edge_correlations: bool = False,
) -> dict:
    """
    Compute QC-FC: correlation of each edge with motion (RMS).

    For each edge in the FC matrix, computes Pearson correlation with
    the motion metric (mean RMS) across subjects.

    Parameters
    ----------
    fc_vec : np.ndarray
        Vectorized FC matrix with shape (n_subjects, n_edges).
        Each row is the upper triangle of one subject's FC matrix.
    rms_vec : np.ndarray
        Motion metric (mean RMS) for each subject, shape (n_subjects,).
        Must be aligned with fc_vec rows.
    return_edge_correlations : bool, default=False
        If True, include the full vector of edge correlations in output.

    Returns
    -------
    dict with keys:
        - mean_abs_r: Mean of |correlation| across all edges
        - std_abs_r: Std of |correlation| across edges
        - median_abs_r: Median of |correlation| across edges
        - frac_significant: Fraction of edges with p < 0.05 (uncorrected)
        - n_edges: Number of edges
        - n_subjects: Number of subjects
        - edge_correlations: (optional) Array of correlations for each edge

    Notes
    -----
    QC-FC is computed as Pearson correlation between each edge and RMS.
    Mean |r| close to 0 indicates minimal motion contamination.
    High mean |r| or high fraction of significant edges indicates
    motion artifacts in FC estimates.
    """
    if fc_vec.ndim != 2:
        raise ValueError(f"fc_vec must be 2D, got shape {fc_vec.shape}")

    n_subjects, n_edges = fc_vec.shape

    if rms_vec.shape[0] != n_subjects:
        raise ValueError(
            f"rms_vec length {rms_vec.shape[0]} does not match "
            f"n_subjects {n_subjects}"
        )

    # Compute correlation for each edge
    edge_correlations = np.zeros(n_edges, dtype=np.float64)
    edge_pvalues = np.zeros(n_edges, dtype=np.float64)

    for i in range(n_edges):
        edge_values = fc_vec[:, i]

        # Skip if edge has no variance
        if np.std(edge_values) < 1e-10 or np.std(rms_vec) < 1e-10:
            edge_correlations[i] = 0.0
            edge_pvalues[i] = 1.0
            continue

        r, p = stats.pearsonr(edge_values, rms_vec)
        edge_correlations[i] = r if np.isfinite(r) else 0.0
        edge_pvalues[i] = p if np.isfinite(p) else 1.0

    # Compute summary statistics
    abs_correlations = np.abs(edge_correlations)

    result = {
        'mean_abs_r': float(np.mean(abs_correlations)),
        'std_abs_r': float(np.std(abs_correlations)),
        'median_abs_r': float(np.median(abs_correlations)),
        'frac_significant': float(np.mean(edge_pvalues < 0.05)),
        'n_edges': n_edges,
        'n_subjects': n_subjects,
    }

    if return_edge_correlations:
        result['edge_correlations'] = edge_correlations

    return result


# =============================================================================
# RMS loading and mapping utilities
# =============================================================================

def load_rms_data(
    site: Literal["china", "ihb"],
    data_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load RMS data for a site.

    Parameters
    ----------
    site : 'china' or 'ihb'
    data_path : str, optional

    Returns
    -------
    pd.DataFrame with columns: subject, run-1, run-2, [run-3]
    """
    data_root = resolve_data_root(data_path)

    if site == "ihb":
        rms_file = data_root / "timeseries_ihb" / "ihb_rmsd.csv"
    else:
        rms_file = data_root / "timeseries_china" / "china_rmsd.csv"

    if not rms_file.exists():
        raise FileNotFoundError(
            f"RMS file not found: {rms_file}\n"
            f"Run create_ihb_rmsd_csv.py to generate it."
        )

    return pd.read_csv(rms_file)


def load_china_session_mapping(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load China session mapping from BeijingEOEC.csv.

    Returns DataFrame with columns:
        - subject: 'sub-XXXXXXX'
        - open_run: which run is open ('run-2' or 'run-3')
        - closed_run1: first closed run (always 'run-1')
        - closed_run2: second closed run ('run-2' or 'run-3')
    """
    data_root = resolve_data_root(data_path)
    csv_path = data_root / "BeijingEOEC.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Session mapping not found: {csv_path}")

    df = pd.read_csv(csv_path)

    mapping = []
    for _, row in df.iterrows():
        subject = f"sub-{row['SubjectID']}"
        sessions = {
            'run-1': row['Session_1'],
            'run-2': row['Session_2'],
            'run-3': row['Session_3'],
        }

        open_run = [r for r, s in sessions.items() if s == 'open'][0]
        closed_runs = sorted([r for r, s in sessions.items() if s == 'closed'])

        mapping.append({
            'subject': subject,
            'open_run': open_run,
            'closed_run1': closed_runs[0],  # Always run-1
            'closed_run2': closed_runs[1],  # run-2 or run-3
        })

    return pd.DataFrame(mapping)


def get_rms_for_condition_ihb(
    rms_df: pd.DataFrame,
    condition: Literal["close", "open"],
) -> np.ndarray:
    """
    Get RMS values for IHB matched to condition.

    IHB mapping: run-1=close, run-2=open

    Parameters
    ----------
    rms_df : pd.DataFrame
        RMS data with columns: subject, run-1, run-2
    condition : 'close' or 'open'

    Returns
    -------
    np.ndarray with shape (n_subjects,)
    """
    run = 'run-1' if condition == 'close' else 'run-2'
    return rms_df[run].values


def get_rms_for_condition_china(
    rms_df: pd.DataFrame,
    condition: Literal["close", "open"],
    session_idx: int = 0,
    data_path: Optional[str] = None,
) -> np.ndarray:
    """
    Get RMS values for China matched to condition.

    China has variable run-to-condition mapping per subject.

    For 'close' condition:
        - session_idx=0: first closed session (always run-1)
        - session_idx=1: second closed session (run-2 or run-3 per subject)

    For 'open' condition:
        - The single open run (run-2 or run-3 per subject)

    Parameters
    ----------
    rms_df : pd.DataFrame
        RMS data with columns: subject, run-1, run-2, run-3
    condition : 'close' or 'open'
    session_idx : int
        For 'close', which session (0=first, 1=second). Ignored for 'open'.
    data_path : str, optional

    Returns
    -------
    np.ndarray with shape (n_subjects,)
    """
    # Load session mapping
    session_mapping = load_china_session_mapping(data_path)

    # Create subject-to-mapping lookup
    subj_map = {row['subject']: row for _, row in session_mapping.iterrows()}

    rms_values = []
    for _, row in rms_df.iterrows():
        subject = row['subject']
        if subject not in subj_map:
            raise ValueError(f"Subject {subject} not in session mapping")

        smap = subj_map[subject]

        if condition == 'open':
            run = smap['open_run']
        else:
            # close condition
            if session_idx == 0:
                run = smap['closed_run1']  # always run-1
            else:
                run = smap['closed_run2']  # run-2 or run-3

        rms_values.append(row[run])

    return np.array(rms_values, dtype=np.float64)


# =============================================================================
# FC loading utilities
# =============================================================================

def _build_strategy_string(strategy: Union[int, str]) -> str:
    """
    Build strategy string for filename.

    Handles both numeric strategies (1-6) and AROMA strategies.
    AROMA strategies: 'AROMA_aggr' -> 'AROMA_aggr', 'AROMA_nonaggr' -> 'AROMA_nonaggr'
    """
    if isinstance(strategy, int):
        return str(strategy)
    return str(strategy)


def load_fc_for_qcfc(
    site: Literal["china", "ihb"],
    condition: Literal["close", "open"],
    atlas: str,
    strategy: Union[int, str],
    gsr: str,
    fc_type: str,
    session_idx: int = 0,
    coverage_mask: Optional[np.ndarray] = None,
    data_path: Optional[str] = None,
) -> np.ndarray:
    """
    Load or compute FC for QC-FC analysis.

    For glasso: loads precomputed FC (computed with IHB coverage mask).
    For corr/partial/tangent: loads timeseries and computes FC.

    Parameters
    ----------
    site : 'china' or 'ihb'
    condition : 'close' or 'open'
    atlas : str
    strategy : int or str
        Numeric (1-6) or AROMA strategy ('AROMA_aggr', 'AROMA_nonaggr')
    gsr : str
    fc_type : str
        'corr', 'partial', 'tangent', or 'glasso'
    session_idx : int
        For China 'close', which session (0 or 1). Ignored otherwise.
    coverage_mask : np.ndarray, optional
        Boolean mask of good ROIs (used for corr/partial/tangent)
        Note: glasso is precomputed with IHB coverage already applied
    data_path : str, optional

    Returns
    -------
    fc_vec : np.ndarray with shape (n_subjects, n_edges)
    """
    data_root = resolve_data_root(data_path)
    strategy_str = _build_strategy_string(strategy)

    if fc_type == 'glasso':
        # Load precomputed glasso (already computed with IHB coverage)
        # Path: glasso_precomputed_fc/{site}/{atlas}/{site}_{condition}_{atlas}_strategy-{strategy}_{gsr}_glasso.npy
        glasso_dir = data_root / "glasso_precomputed_fc" / site / atlas
        filename = f"{site}_{condition}_{atlas}_strategy-{strategy_str}_{gsr}_glasso.npy"
        filepath = glasso_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Precomputed glasso not found: {filepath}")

        fc_vec = np.load(filepath)

        # China close has 2 sessions
        if site == "china" and condition == "close" and fc_vec.ndim == 3:
            fc_vec = fc_vec[:, :, session_idx]

        return fc_vec

    else:
        # Load timeseries and compute FC
        ts_dir = data_root / f"timeseries_{site}" / atlas
        filename = f"{site}_{condition}_{atlas}_strategy-{strategy_str}_{gsr}.npy"
        filepath = ts_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Timeseries not found: {filepath}")

        ts = np.load(filepath)

        # China close has 2 sessions: (n_subjects, n_timepoints, n_rois, 2)
        if site == "china" and condition == "close" and ts.ndim == 4:
            ts = ts[:, :, :, session_idx]

        # Apply coverage mask before FC computation
        if coverage_mask is not None:
            ts = ts[:, :, coverage_mask]

        # Compute FC
        kind_map = {
            'corr': 'corr',
            'partial': 'partial',
            'tangent': 'tangent',
        }

        if fc_type not in kind_map:
            raise ValueError(f"Unknown fc_type: {fc_type}")

        transformer = ConnectomeTransformer(
            kind=kind_map[fc_type],
            vectorize=True,
            discard_diagonal=True,
        )

        return transformer.fit_transform(ts)


# =============================================================================
# Main QC-FC computation per pipeline
# =============================================================================

def compute_qcfc_for_pipeline(
    site: Literal["china", "ihb"],
    atlas: str,
    strategy: Union[int, str],
    gsr: str,
    fc_type: str,
    coverage_mask: Optional[np.ndarray] = None,
    data_path: Optional[str] = None,
    return_edge_correlations: bool = False,
) -> dict:
    """
    Compute QC-FC for a single pipeline configuration.

    Loads FC per-condition, matches RMS from corresponding runs,
    and computes QC-FC correlations.

    Parameters
    ----------
    site : 'china' or 'ihb'
    atlas : str
    strategy : int or str
    gsr : str
    fc_type : str
        'corr', 'partial', 'tangent', or 'glasso'
    coverage_mask : np.ndarray, optional
    data_path : str, optional
    return_edge_correlations : bool, default=False
        If True, include edge-wise correlations in the result

    Returns
    -------
    dict with QC-FC results including:
        - mean_abs_r, std_abs_r, frac_significant
        - per-condition results
        - pipeline metadata
        - edge_correlations (optional): dict of condition -> array of correlations
    """
    # Load RMS data
    rms_df = load_rms_data(site, data_path)

    results = {
        'site': site,
        'atlas': atlas,
        'strategy': strategy,
        'gsr': gsr,
        'fc_type': fc_type,
        'conditions': {},
    }

    if return_edge_correlations:
        results['edge_correlations'] = {}

    all_mean_abs_r = []
    all_frac_sig = []

    # For IHB: close (1 session) + open (1 session) = 2 comparisons
    # For China: close_s0 + close_s1 + open = 3 comparisons
    if site == 'ihb':
        condition_sessions = [('close', 0), ('open', 0)]
    else:
        condition_sessions = [('close', 0), ('close', 1), ('open', 0)]

    for condition, session_idx in condition_sessions:
        cond_key = f"{condition}" if session_idx == 0 else f"{condition}_s{session_idx}"

        try:
            # Load FC
            fc_vec = load_fc_for_qcfc(
                site, condition, atlas, strategy, gsr, fc_type,
                session_idx=session_idx,
                coverage_mask=coverage_mask,
                data_path=data_path,
            )

            # Get matched RMS
            if site == 'ihb':
                rms_vec = get_rms_for_condition_ihb(rms_df, condition)
            else:
                rms_vec = get_rms_for_condition_china(
                    rms_df, condition, session_idx, data_path
                )

            # Compute QC-FC
            qcfc_result = qc_fc_edges(fc_vec, rms_vec, return_edge_correlations=return_edge_correlations)

            # Store edge correlations separately if requested
            if return_edge_correlations and 'edge_correlations' in qcfc_result:
                results['edge_correlations'][cond_key] = qcfc_result.pop('edge_correlations')

            results['conditions'][cond_key] = qcfc_result
            all_mean_abs_r.append(qcfc_result['mean_abs_r'])
            all_frac_sig.append(qcfc_result['frac_significant'])

        except Exception as e:
            results['conditions'][cond_key] = {'error': str(e)}

    # Aggregate across conditions
    if all_mean_abs_r:
        results['mean_abs_r'] = float(np.mean(all_mean_abs_r))
        results['std_abs_r_across_conditions'] = float(np.std(all_mean_abs_r))
        results['frac_significant'] = float(np.mean(all_frac_sig))
    else:
        results['mean_abs_r'] = np.nan
        results['std_abs_r_across_conditions'] = np.nan
        results['frac_significant'] = np.nan

    return results


def process_pipeline_wrapper(args: tuple) -> dict:
    """Wrapper for multiprocessing."""
    # args = (site, atlas, strategy, gsr, fc_type, coverage_mask, data_path, return_edge_correlations)
    return compute_qcfc_for_pipeline(*args)


# =============================================================================
# Config-driven execution
# =============================================================================

def load_coverage_mask(
    atlas: str,
    data_path: Optional[str] = None,
    threshold: float = 0.1,
) -> np.ndarray:
    """Load IHB coverage mask."""
    data_root = resolve_data_root(data_path)
    coverage_file = data_root / "coverage" / f"ihb_{atlas}_parcel_coverage.npy"

    if not coverage_file.exists():
        raise FileNotFoundError(f"Coverage file not found: {coverage_file}")

    coverage = np.load(coverage_file).astype(float)
    return coverage >= threshold


def _save_edge_correlations(
    edge_correlations: dict,
    site: str,
    atlas: str,
    strategy: Union[int, str],
    gsr: str,
    fc_type: str,
    output_dir: str,
) -> None:
    """
    Save edge-wise correlations to CSV file.

    Parameters
    ----------
    edge_correlations : dict
        Mapping of condition -> array of edge correlations
    site, atlas, strategy, gsr, fc_type : str
        Pipeline identifiers for filename
    output_dir : str
        Directory to save the CSV file
    """
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    # Build filename
    filename = f"{site}_{atlas}_strategy-{strategy}_{gsr}_{fc_type}_edge_correlations.csv"
    filepath = output_path / filename

    # Create DataFrame with edge correlations for each condition
    df_data = {}
    for cond_key, corr_array in edge_correlations.items():
        df_data[cond_key] = corr_array

    # Add edge index
    n_edges = len(next(iter(edge_correlations.values())))
    df_data['edge_idx'] = np.arange(n_edges)

    df = pd.DataFrame(df_data)

    # Reorder columns: edge_idx first
    cols = ['edge_idx'] + [c for c in df.columns if c != 'edge_idx']
    df = df[cols]

    df.to_csv(filepath, index=False)


def run_qcfc_from_config(config: dict) -> pd.DataFrame:
    """
    Run QC-FC analysis from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration with keys:
        - data_path: str
        - sites: list[str]
        - atlases: list[str]
        - strategies: list[int or str]
        - gsr_options: list[str]
        - fc_types: list[str]
        - use_coverage_mask: bool
        - coverage_threshold: float
        - n_workers: int
        - output: str
        - save_edge_correlations: bool (save per-edge correlations as CSV)
        - edge_correlations_dir: str (directory for edge correlation CSVs)

    Returns
    -------
    pd.DataFrame with QC-FC results for all pipelines
    """
    data_path = config.get('data_path')
    sites = config.get('sites', ['ihb', 'china'])
    atlases = config.get('atlases', ['AAL', 'Schaefer200', 'Brainnetome', 'HCPex'])
    strategies = config.get('strategies', [1, 2, 3, 4, 5, 6])
    gsr_options = config.get('gsr_options', ['GSR', 'noGSR'])
    fc_types = config.get('fc_types', ['corr', 'partial', 'tangent', 'glasso'])
    use_coverage_mask = config.get('use_coverage_mask', True)  # Default: use IHB coverage
    coverage_threshold = config.get('coverage_threshold', 0.1)
    n_workers = config.get('n_workers', 1)
    save_edge_correlations = config.get('save_edge_correlations', False)
    edge_correlations_dir = config.get('edge_correlations_dir', 'results/qcfc/edge_correlations')

    # Generate pipeline configurations
    pipeline_configs = []

    # Pre-load coverage masks for each atlas
    coverage_masks = {}
    if use_coverage_mask:
        for atlas in atlases:
            try:
                coverage_masks[atlas] = load_coverage_mask(atlas, data_path, coverage_threshold)
            except FileNotFoundError:
                print(f"Warning: Coverage mask not found for {atlas}, skipping masking")
                coverage_masks[atlas] = None

    for site in sites:
        for atlas in atlases:
            coverage_mask = coverage_masks.get(atlas) if use_coverage_mask else None

            # All strategies (1-6 numeric, AROMA_aggr, AROMA_nonaggr)
            for strategy in strategies:
                for gsr in gsr_options:
                    for fc_type in fc_types:
                        pipeline_configs.append((
                            site, atlas, strategy, gsr, fc_type,
                            coverage_mask, data_path, save_edge_correlations
                        ))

    print(f"Running QC-FC for {len(pipeline_configs)} pipeline configurations")
    print(f"Using {n_workers} workers")
    if save_edge_correlations:
        edge_corr_path = Path(edge_correlations_dir).expanduser()
        edge_corr_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving edge correlations to: {edge_corr_path}")

    # Run with progress bar
    results = []

    if n_workers == 1:
        pbar = tqdm(pipeline_configs, desc="Computing QC-FC", unit="pipeline")
        for args in pbar:
            site, atlas, strategy, gsr, fc_type = args[0], args[1], args[2], args[3], args[4]
            pbar.set_postfix_str(f"{site}/{atlas}/s{strategy}/{gsr}/{fc_type}")
            try:
                result = process_pipeline_wrapper(args)

                # Save edge correlations if requested
                if save_edge_correlations and 'edge_correlations' in result:
                    _save_edge_correlations(
                        result['edge_correlations'],
                        site, atlas, strategy, gsr, fc_type,
                        edge_correlations_dir
                    )
                    del result['edge_correlations']  # Don't keep in memory

                results.append(result)
            except Exception as e:
                results.append({
                    'site': site,
                    'atlas': atlas,
                    'strategy': strategy,
                    'gsr': gsr,
                    'fc_type': fc_type,
                    'error': str(e),
                    'mean_abs_r': np.nan,
                    'frac_significant': np.nan,
                })
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_pipeline_wrapper, args): args
                for args in pipeline_configs
            }

            pbar = tqdm(as_completed(futures), total=len(pipeline_configs),
                       desc="Computing QC-FC", unit="pipeline")
            for future in pbar:
                args = futures[future]
                site, atlas, strategy, gsr, fc_type = args[0], args[1], args[2], args[3], args[4]
                pbar.set_postfix_str(f"{site}/{atlas}/s{strategy}/{gsr}/{fc_type}")
                try:
                    result = future.result()

                    # Save edge correlations if requested
                    if save_edge_correlations and 'edge_correlations' in result:
                        _save_edge_correlations(
                            result['edge_correlations'],
                            site, atlas, strategy, gsr, fc_type,
                            edge_correlations_dir
                        )
                        del result['edge_correlations']

                    results.append(result)
                except Exception as e:
                    results.append({
                        'site': site,
                        'atlas': atlas,
                        'strategy': strategy,
                        'gsr': gsr,
                        'fc_type': fc_type,
                        'error': str(e),
                        'mean_abs_r': np.nan,
                        'frac_significant': np.nan,
                    })

    # Convert to DataFrame
    rows = []
    for result in results:
        row = {
            'site': result['site'],
            'atlas': result['atlas'],
            'strategy': result['strategy'],
            'gsr': result['gsr'],
            'fc_type': result['fc_type'],
            'mean_abs_r': result.get('mean_abs_r', np.nan),
            'std_abs_r_across_conditions': result.get('std_abs_r_across_conditions', np.nan),
            'frac_significant': result.get('frac_significant', np.nan),
        }

        # Add per-condition results
        conditions = result.get('conditions', {})
        for cond_name, cond_result in conditions.items():
            if isinstance(cond_result, dict) and 'mean_abs_r' in cond_result:
                row[f'{cond_name}_mean_abs_r'] = cond_result['mean_abs_r']
                row[f'{cond_name}_frac_sig'] = cond_result['frac_significant']

        rows.append(row)

    df = pd.DataFrame(rows)

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Compute QC-FC (motion-FC correlations) for fMRI pipelines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Override output path from config',
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override output if specified
    if args.output:
        config['output'] = args.output

    print("=" * 60)
    print("QC-FC Analysis")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Sites: {config.get('sites', ['ihb', 'china'])}")
    print(f"Atlases: {config.get('atlases', ['all'])}")
    print(f"FC types: {config.get('fc_types', ['corr', 'partial', 'tangent', 'glasso'])}")

    # Run analysis
    df = run_qcfc_from_config(config)

    # Save results
    output_path = Path(config.get('output', 'results/qc_fc_results.csv')).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")
    print(f"Total pipelines: {len(df)}")

    # Print summary
    print("\nSummary by FC type:")
    summary = df.groupby('fc_type')['mean_abs_r'].agg(['mean', 'std', 'min', 'max'])
    print(summary.to_string())

    print("\nDone!")


if __name__ == '__main__':
    main()
