"""
Timeseries Loading Module
=========================

Functions for loading preprocessed fMRI timeseries data and coverage masks.

Data Structure
--------------
IHB (St. Petersburg):
    timeseries_ihb/{Atlas}/{site}_{condition}_{Atlas}_strategy-{N}_{GSR}.npy
    Shape: (84, 120, n_rois)

China (Beijing):
    timeseries_china/{Atlas}/{site}_{condition}_{Atlas}_strategy-{N}_{GSR}.npy
    Close shape: (48, 240, n_rois, 2) - 4D with 2 sessions (session 0 used in ML)
    Open shape: (48, 240, n_rois) - 3D

Author: BenchmarkingEOEC Team
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

from data_utils.paths import resolve_data_root

Site = Literal["china", "ihb"]
Condition = Literal["close", "open"]


def timeseries_path(
    *,
    data_root: Union[str, Path],
    site: Site,
    condition: Condition,
    atlas: str,
    strategy: Union[int, str],
    gsr: str,
) -> Path:
    """
    Build path to a timeseries file.

    Parameters
    ----------
    data_root : str or Path
        Base data directory (OpenCloseBenchmark_data)
    site : 'china' or 'ihb'
    condition : 'close' or 'open'
    atlas : str
        Atlas name (AAL, Schaefer200, Brainnetome, HCPex)
    strategy : int or str
        Denoising strategy (1-6, AROMA_aggr, AROMA_nonaggr)
    gsr : str
        'GSR' or 'noGSR'

    Returns
    -------
    Path
        Full path to the timeseries file
    """
    data_root = resolve_data_root(data_root)
    ts_dir = data_root / f"timeseries_{site}" / atlas
    filename = f"{site}_{condition}_{atlas}_strategy-{strategy}_{gsr}.npy"
    return ts_dir / filename


def load_timeseries(
    site: Site,
    condition: Condition,
    atlas: str,
    strategy: Union[int, str],
    gsr: str,
    data_path: Optional[str] = None,
) -> np.ndarray:
    """
    Load timeseries for a site/condition.

    Parameters
    ----------
    site : 'china' or 'ihb'
    condition : 'close' or 'open'
    atlas : str
        Atlas name
    strategy : int or str
        Denoising strategy
    gsr : str
        'GSR' or 'noGSR'
    data_path : str, optional
        Base data path (uses default if None)

    Returns
    -------
    ts : np.ndarray
        Shape: (n_subjects, n_timepoints, n_rois)
        For China 'close', returns session 0 only (use icc_data_preparation for 2-session ICC)
    """
    filepath = timeseries_path(
        data_root=data_path,
        site=site,
        condition=condition,
        atlas=atlas,
        strategy=strategy,
        gsr=gsr,
    )

    if not filepath.exists():
        raise FileNotFoundError(f"Timeseries file not found: {filepath}")

    ts = np.load(filepath)

    # China close has 2 sessions - use first only (second has data quality issues)
    if site == "china" and condition == "close" and ts.ndim == 4:
        ts = ts[:, :, :, 0]

    return ts


def load_site_timeseries(
    site: Site,
    atlas: str,
    strategy: Union[int, str],
    gsr: str,
    data_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load both close and open timeseries for a site, stacked as [EC..., EO...].

    Parameters
    ----------
    site : 'china' or 'ihb'
    atlas : str
    strategy : int or str
    gsr : str
    data_path : str, optional

    Returns
    -------
    ts : np.ndarray
        Shape: (n_subjects*2, n_timepoints, n_rois)
        First half: EC (close), second half: EO (open)
    y : np.ndarray
        Labels: [0]*n_subjects + [1]*n_subjects (0=EC, 1=EO)
    """
    ts_close = load_timeseries(site, "close", atlas, strategy, gsr, data_path)
    ts_open = load_timeseries(site, "open", atlas, strategy, gsr, data_path)

    n_subjects = ts_close.shape[0]

    # Stack: [EC_subjects..., EO_subjects...]
    ts = np.concatenate([ts_close, ts_open], axis=0)
    y = np.array([0] * n_subjects + [1] * n_subjects, dtype=np.int32)

    return ts, y


def load_ihb_coverage_mask(
    atlas: str,
    data_path: Optional[str] = None,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Load IHB coverage mask (always used for both sites).

    IHB has lower coverage in some ROIs due to scanner differences.
    Using IHB coverage ensures consistent masking across both sites.

    Parameters
    ----------
    atlas : str
        Atlas name
    data_path : str, optional
        Base data path
    threshold : float, default=0.1
        Coverage threshold (ROIs with coverage < threshold are marked bad)

    Returns
    -------
    mask : np.ndarray
        Boolean mask: True = good ROI, False = bad ROI
    """
    data_root = resolve_data_root(data_path)
    coverage_file = data_root / "coverage" / f"ihb_{atlas}_parcel_coverage.npy"

    if not coverage_file.exists():
        raise FileNotFoundError(f"Coverage file not found: {coverage_file}")

    coverage = np.load(coverage_file).astype(float)
    return coverage >= threshold


def get_n_good_rois(atlas: str, data_path: Optional[str] = None, threshold: float = 0.1) -> int:
    """Get number of good ROIs after IHB coverage masking."""
    mask = load_ihb_coverage_mask(atlas, data_path, threshold)
    return int(np.sum(mask))


def get_n_edges(n_rois: int) -> int:
    """Get number of edges (upper triangle without diagonal) for n_rois."""
    return n_rois * (n_rois - 1) // 2


# =============================================================================
# Precomputed Glasso Loading
# =============================================================================

def glasso_precomputed_path(
    *,
    data_root: Union[str, Path],
    site: Site,
    condition: Condition,
    atlas: str,
    strategy: Union[int, str],
    gsr: str,
) -> Path:
    """Build path to a precomputed glasso file (glasso_precomputed_fc/{site}/{atlas})."""
    data_root = resolve_data_root(data_root)
    glasso_dir = data_root / "glasso_precomputed_fc" / site / atlas
    filename = f"{site}_{condition}_{atlas}_strategy-{strategy}_{gsr}_glasso.npy"
    return glasso_dir / filename


def load_precomputed_glasso(
    site: Site,
    condition: Condition,
    atlas: str,
    strategy: Union[int, str],
    gsr: str,
    data_path: Optional[str] = None,
) -> np.ndarray:
    """
    Load precomputed glasso FC (vectorized).

    Parameters
    ----------
    site : 'china' or 'ihb'
    condition : 'close' or 'open'
    atlas : str
    strategy : int or str
    gsr : str
    data_path : str, optional

    Returns
    -------
    glasso : np.ndarray
        Shape: (n_subjects, n_edges) - vectorized upper triangle
    """
    filepath = glasso_precomputed_path(
        data_root=data_path,
        site=site,
        condition=condition,
        atlas=atlas,
        strategy=strategy,
        gsr=gsr,
    )

    if not filepath.exists():
        raise FileNotFoundError(f"Precomputed glasso not found: {filepath}")

    glasso = np.load(filepath)

    # China close has 2 sessions - use first only
    if site == "china" and condition == "close" and glasso.ndim == 3:
        glasso = glasso[:, :, 0]

    return glasso


def load_site_precomputed_glasso(
    site: Site,
    atlas: str,
    strategy: Union[int, str],
    gsr: str,
    data_path: Optional[str] = None,
) -> np.ndarray:
    """
    Load precomputed glasso for both conditions, stacked as [EC..., EO...].

    Returns
    -------
    glasso : np.ndarray
        Shape: (n_subjects*2, n_edges)
    """
    glasso_close = load_precomputed_glasso(site, "close", atlas, strategy, gsr, data_path)
    glasso_open = load_precomputed_glasso(site, "open", atlas, strategy, gsr, data_path)
    return np.concatenate([glasso_close, glasso_open], axis=0)


def check_precomputed_glasso_available(
    atlas: str,
    strategy: Union[int, str],
    gsr: str,
    data_path: Optional[str] = None,
) -> bool:
    """Check if precomputed glasso is available for both sites."""
    try:
        for site in ["china", "ihb"]:
            for condition in ["close", "open"]:
                filepath = glasso_precomputed_path(
                    data_root=data_path,
                    site=site,
                    condition=condition,
                    atlas=atlas,
                    strategy=strategy,
                    gsr=gsr,
                )
                if not filepath.exists():
                    return False
        return True
    except Exception:
        return False
