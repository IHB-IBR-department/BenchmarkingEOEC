"""
Functional Connectivity Computation Module
==========================================

This module provides tools for computing functional connectivity (FC) matrices
from fMRI time series data.

Connectivity Types
------------------
- 'corr': Pearson correlation
- 'partial': Partial correlation (regularized via nilearn)
- 'tangent': Tangent space projection (Varoquaux et al., 2010)
- 'glasso': Graphical Lasso regularized partial correlation

Leakage-Safe Design
-------------------
The ConnectomeTransformer class provides a scikit-learn-style fit/transform API
to prevent data leakage in cross-validation:

    # CORRECT: fit on train, transform both
    transformer = ConnectomeTransformer(kind='tangent')
    X_train_fc = transformer.fit_transform(ts_train)
    X_test_fc = transformer.transform(ts_test)

    # WRONG: fitting on all data leaks test information into reference
    X_all_fc = transformer.fit_transform(ts_all)  # Don't do this before CV split!

For tangent space, the reference matrix (geometric mean of covariances) is
computed during fit() from training data only.

Glasso Implementation Note
--------------------------
We use gglasso instead of sklearn.covariance.GraphicalLassoCV because:
1. sklearn's implementation can crash on ill-conditioned data
2. gglasso is faster for single-subject estimation
3. We use a fixed lambda (L1=0.03) following Peterson et al. (2023)

References
----------
- Varoquaux et al. (2010). Detection of brain functional-connectivity difference
  in post-stroke patients using group-level covariance modeling. MICCAI.
- Peterson et al. (2023). Regularized partial correlation provides reliable
  functional connectivity estimates. bioRxiv.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Literal, Optional
import numpy as np
from scipy import stats
from nilearn.connectome import ConnectivityMeasure, sym_matrix_to_vec

from benchmarking.project import resolve_data_root

# gglasso import (used for glasso instead of sklearn due to stability issues)
try:
    from gglasso.problem import glasso_problem
    HAS_GGLASSO = True
except ImportError:
    HAS_GGLASSO = False


ConnectivityKind = Literal['corr', 'partial', 'tangent', 'glasso']


class ConnectomeTransformer:
    """
    Leakage-safe functional connectivity transformer with fit/transform API.

    This class computes FC matrices from time series while ensuring no data
    leakage when used in cross-validation. The reference matrix for tangent
    space projection is computed only during fit().

    Parameters
    ----------
    kind : {'corr', 'partial', 'tangent', 'glasso'}
        Type of connectivity to compute:
        - 'corr': Pearson correlation
        - 'partial': Partial correlation (nilearn regularized)
        - 'tangent': Tangent space projection (requires fit for reference)
        - 'glasso': Graphical Lasso regularized partial correlation
    vectorize : bool, default=False
        If True, return vectorized upper triangle of FC matrices.
        Output shape becomes (n_subjects, n_edges) instead of (n_subjects, n_rois, n_rois).
    discard_diagonal : bool, default=True
        If vectorize=True, whether to discard the diagonal elements.
    glasso_lambda : float, default=0.03
        L1 regularization parameter for graphical lasso.

    Attributes
    ----------
    connectivity_measure_ : ConnectivityMeasure or None
        Fitted nilearn ConnectivityMeasure (for 'corr', 'partial', 'tangent').
    is_fitted_ : bool
        Whether the transformer has been fitted.

    Examples
    --------
    >>> # Leakage-safe cross-validation
    >>> transformer = ConnectomeTransformer(kind='tangent', vectorize=True)
    >>> X_train = transformer.fit_transform(ts_train)  # Fit reference on train
    >>> X_test = transformer.transform(ts_test)        # Transform test using train reference
    """

    def __init__(
        self,
        kind: ConnectivityKind = 'corr',
        vectorize: bool = False,
        discard_diagonal: bool = True,
        glasso_lambda: float = 0.03,
    ):
        self.kind = kind
        self.vectorize = vectorize
        self.discard_diagonal = discard_diagonal
        self.glasso_lambda = glasso_lambda

        self.connectivity_measure_: Optional[ConnectivityMeasure] = None
        self.is_fitted_: bool = False

    def _validate_input(self, timeseries: np.ndarray) -> np.ndarray:
        """Validate and convert input to proper shape."""
        if isinstance(timeseries, list):
            timeseries = np.array(timeseries)

        if timeseries.ndim != 3:
            raise ValueError(
                f"Input timeseries should have shape (n_subjects, n_timepoints, n_nodes), "
                f"but got {timeseries.shape}"
            )
        return timeseries

    def _vectorize(self, conn: np.ndarray) -> np.ndarray:
        """Vectorize FC matrices to upper triangle."""
        return sym_matrix_to_vec(conn, discard_diagonal=self.discard_diagonal)

    def fit(self, timeseries: np.ndarray) -> 'ConnectomeTransformer':
        """
        Fit the transformer on training time series.

        For tangent space, this computes the reference matrix (geometric mean
        of covariances) from the training data.

        Parameters
        ----------
        timeseries : np.ndarray
            Time series data with shape (n_subjects, n_timepoints, n_nodes).

        Returns
        -------
        self : ConnectomeTransformer
            Fitted transformer.
        """
        timeseries = self._validate_input(timeseries)

        if self.kind == 'glasso':
            # Glasso doesn't need fitting (computed per-subject)
            self.is_fitted_ = True
            return self

        # Map kind names to nilearn kinds
        nilearn_kind_map = {
            'corr': 'correlation',
            'partial': 'partial correlation',
            'tangent': 'tangent',
        }

        if self.kind not in nilearn_kind_map:
            raise ValueError(f"Unknown connectivity kind: {self.kind}")

        nilearn_kind = nilearn_kind_map[self.kind]

        # Create and fit the ConnectivityMeasure
        # For tangent, this computes the reference matrix from training data
        self.connectivity_measure_ = ConnectivityMeasure(
            kind=nilearn_kind,
            standardize=False,
        )
        self.connectivity_measure_.fit(timeseries)
        self.is_fitted_ = True

        return self

    def transform(self, timeseries: np.ndarray) -> np.ndarray:
        """
        Transform time series to connectivity matrices.

        For tangent space, uses the reference matrix computed during fit().
        This ensures no leakage from test data.

        Parameters
        ----------
        timeseries : np.ndarray
            Time series data with shape (n_subjects, n_timepoints, n_nodes).

        Returns
        -------
        conn : np.ndarray
            Connectivity matrices. Shape is (n_subjects, n_rois, n_rois) if
            vectorize=False, or (n_subjects, n_edges) if vectorize=True.
        """
        if not self.is_fitted_:
            raise RuntimeError("Transformer must be fitted before transform(). Call fit() first.")

        timeseries = self._validate_input(timeseries)

        if self.kind == 'glasso':
            conn = self._compute_glasso(timeseries)
        else:
            # Use the fitted ConnectivityMeasure
            conn = self.connectivity_measure_.transform(timeseries)

        if self.vectorize:
            conn = self._vectorize(conn)

        return conn

    def fit_transform(self, timeseries: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters
        ----------
        timeseries : np.ndarray
            Time series data with shape (n_subjects, n_timepoints, n_nodes).

        Returns
        -------
        conn : np.ndarray
            Connectivity matrices.
        """
        return self.fit(timeseries).transform(timeseries)

    def _compute_glasso(self, timeseries: np.ndarray) -> np.ndarray:
        """
        Compute graphical lasso FC for each subject.

        Uses gglasso library instead of sklearn for better stability and speed.
        """
        if not HAS_GGLASSO:
            raise ImportError(
                "gglasso is required for glasso connectivity. "
                "Install with: pip install gglasso"
            )

        n_subjects, _, n_nodes = timeseries.shape
        conn = np.zeros((n_subjects, n_nodes, n_nodes), dtype=np.float64)

        for sub in range(n_subjects):
            conn[sub] = graphical_lasso_single(
                timeseries[sub],
                lambda1=self.glasso_lambda,
            )

        return conn


def graphical_lasso_single(data: np.ndarray, lambda1: float = 0.03) -> np.ndarray:
    """
    Compute L1-regularized partial correlation matrix for a single subject.

    Implementation based on Peterson et al. (2023) using gglasso library.
    We use gglasso instead of sklearn's GraphicalLassoCV because:
    1. sklearn can crash on ill-conditioned covariance matrices
    2. gglasso is more stable and faster for single-subject estimation
    3. Fixed lambda avoids CV overhead

    Parameters
    ----------
    data : np.ndarray
        Time series for one subject with shape (n_timepoints, n_nodes).
    lambda1 : float, default=0.03
        L1 regularization parameter. Higher values = sparser solution.
        Default 0.03 follows Peterson et al. (2023).

    Returns
    -------
    partial_corr : np.ndarray
        Regularized partial correlation matrix with shape (n_nodes, n_nodes).
        Diagonal is set to zero.

    References
    ----------
    Peterson, K. L., Sanchez-Romero, R., Mill, R. D., & Cole, M. W. (2023).
    Regularized partial correlation provides reliable functional connectivity
    estimates while correcting for widespread confounding.
    bioRxiv. https://doi.org/10.1101/2023.09.16.558065
    """
    # Transpose to (n_nodes, n_timepoints) for covariance computation
    data = data.T
    n_timepoints = data.shape[1]

    # Z-score each node's time series
    data_scaled = stats.zscore(data, axis=1)

    # Handle constant time series (zscore returns nan)
    data_scaled = np.nan_to_num(data_scaled, nan=0.0)

    # Estimate empirical covariance
    emp_cov = np.cov(data_scaled, rowvar=True)

    # Ensure covariance is well-conditioned
    if not np.isfinite(emp_cov).all():
        raise ValueError("Covariance matrix contains non-finite values")

    # Run graphical lasso
    glasso = glasso_problem(
        emp_cov,
        n_timepoints,
        reg_params={'lambda1': lambda1},
        latent=False,
        do_scaling=False,
    )
    glasso.solve(verbose=False)

    # Extract precision matrix
    precision = np.squeeze(glasso.solution.precision_)

    # Transform precision to partial correlation
    # partial_corr[i,j] = -precision[i,j] / sqrt(precision[i,i] * precision[j,j])
    diag = np.diag(precision)
    # Avoid division by zero
    diag = np.where(diag > 0, diag, 1e-10)
    denom = np.atleast_2d(1.0 / np.sqrt(diag))
    partial_corr = -precision * denom * denom.T

    # Zero diagonal
    np.fill_diagonal(partial_corr, 0)

    return partial_corr


# =============================================================================
# Convenience helper for time-series files
# =============================================================================

_COVERAGE_CACHE: dict[tuple[str, str, str], np.ndarray] = {}


def _parse_strategy_path(strategy_path: Path) -> tuple[str, str]:
    name = strategy_path.name
    match = re.match(r"^(ihb|china)_(?:close|open)\d*_(?P<atlas>[^_]+)_strategy-", name)
    if match:
        return match.group(1), match.group("atlas")

    site = None
    if name.startswith("ihb_"):
        site = "ihb"
    elif name.startswith("china_"):
        site = "china"
    elif "timeseries_ihb" in strategy_path.parts:
        site = "ihb"
    elif "timeseries_china" in strategy_path.parts:
        site = "china"

    atlas_match = re.search(r"_(?P<atlas>[^_]+)_strategy-", name)
    atlas = atlas_match.group("atlas") if atlas_match else None
    if atlas is None:
        parent_name = strategy_path.parent.name
        if parent_name and parent_name not in ("timeseries_ihb", "timeseries_china"):
            atlas = parent_name

    if site and atlas:
        return site, atlas

    raise ValueError(f"Unable to parse site/atlas from strategy file: {strategy_path}")


def _load_parcel_coverage(
    *,
    site: str,
    atlas: str,
    data_path: str | Path | None,
) -> np.ndarray:
    data_root = resolve_data_root(data_path)
    coverage_path = Path(data_root) / "coverage" / f"{site}_{atlas}_parcel_coverage.npy"
    cache_key = (site, atlas, str(coverage_path))
    cached = _COVERAGE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not coverage_path.exists():
        raise FileNotFoundError(f"Missing coverage file: {coverage_path}")
    coverage = np.load(coverage_path).astype(float).reshape(-1)
    _COVERAGE_CACHE[cache_key] = coverage
    return coverage


def _resolve_coverage_mask(
    *,
    strategy_path: Path,
    coverage: str | Path | np.ndarray | bool | None,
    coverage_threshold: float,
    data_path: str | Path | None,
) -> Optional[np.ndarray]:
    if coverage is None or coverage is False:
        return None
    if coverage_threshold <= 0 or coverage_threshold >= 1:
        raise ValueError("coverage_threshold must be in (0, 1)")

    if coverage is True:
        coverage = "site"

    if isinstance(coverage, np.ndarray):
        return np.asarray(coverage, dtype=float).reshape(-1) >= coverage_threshold

    if isinstance(coverage, (str, Path)):
        coverage_key = str(coverage).strip()
        coverage_key_lower = coverage_key.lower()

        if coverage_key_lower in ("site", "auto", "true"):
            site, atlas = _parse_strategy_path(strategy_path)
            values = _load_parcel_coverage(site=site, atlas=atlas, data_path=data_path)
            return values >= coverage_threshold

        if coverage_key_lower in ("both", "join"):
            _, atlas = _parse_strategy_path(strategy_path)
            china = _load_parcel_coverage(site="china", atlas=atlas, data_path=data_path)
            ihb = _load_parcel_coverage(site="ihb", atlas=atlas, data_path=data_path)
            if china.shape != ihb.shape:
                raise ValueError(f"Coverage shape mismatch for atlas {atlas}: {china.shape} vs {ihb.shape}")
            return (china >= coverage_threshold) & (ihb >= coverage_threshold)

        if coverage_key_lower in ("ihb", "china"):
            _, atlas = _parse_strategy_path(strategy_path)
            values = _load_parcel_coverage(site=coverage_key_lower, atlas=atlas, data_path=data_path)
            return values >= coverage_threshold

        coverage_path = Path(coverage).expanduser()
        if not coverage_path.exists():
            raise FileNotFoundError(f"Coverage file not found: {coverage_path}")
        values = np.load(coverage_path).astype(float).reshape(-1)
        return values >= coverage_threshold

    raise TypeError("coverage must be a path, array, string option, or bool.")


def _expected_edge_count(n_rois: int, discard_diagonal: bool) -> int:
    if discard_diagonal:
        return (n_rois * (n_rois - 1)) // 2
    return (n_rois * (n_rois + 1)) // 2


def _load_precomputed_glasso(
    glasso: str | Path | np.ndarray,
    *,
    expected_n_subjects: int,
    expected_n_rois: int,
    expected_n_sessions: int,
    vectorize: bool,
    discard_diagonal: bool,
) -> np.ndarray:
    if isinstance(glasso, (str, Path)):
        glasso_path = Path(glasso).expanduser()
        if not glasso_path.exists():
            raise FileNotFoundError(f"Precomputed glasso not found: {glasso_path}")
        glasso_arr = np.load(glasso_path)
    else:
        glasso_arr = np.asarray(glasso)

    n_edges = _expected_edge_count(expected_n_rois, discard_diagonal)

    if vectorize:
        if glasso_arr.ndim == 2:
            if expected_n_sessions != 1:
                raise ValueError(
                    "Precomputed glasso must include session axis for multi-session data."
                )
            if glasso_arr.shape != (expected_n_subjects, n_edges):
                raise ValueError(
                    f"Precomputed glasso shape {glasso_arr.shape} does not match "
                    f"({expected_n_subjects}, {n_edges})."
                )
        elif glasso_arr.ndim == 3:
            if glasso_arr.shape != (expected_n_subjects, n_edges, expected_n_sessions):
                raise ValueError(
                    f"Precomputed glasso shape {glasso_arr.shape} does not match "
                    f"({expected_n_subjects}, {n_edges}, {expected_n_sessions})."
                )
        else:
            raise ValueError("Precomputed glasso must be 2D or 3D when vectorized.")
    else:
        if glasso_arr.ndim == 3:
            if expected_n_sessions != 1:
                raise ValueError(
                    "Precomputed glasso must include session axis for multi-session data."
                )
            if glasso_arr.shape != (expected_n_subjects, expected_n_rois, expected_n_rois):
                raise ValueError(
                    f"Precomputed glasso shape {glasso_arr.shape} does not match "
                    f"({expected_n_subjects}, {expected_n_rois}, {expected_n_rois})."
                )
        elif glasso_arr.ndim == 4:
            if glasso_arr.shape != (
                expected_n_subjects,
                expected_n_rois,
                expected_n_rois,
                expected_n_sessions,
            ):
                raise ValueError(
                    f"Precomputed glasso shape {glasso_arr.shape} does not match "
                    f"({expected_n_subjects}, {expected_n_rois}, {expected_n_rois}, {expected_n_sessions})."
                )
        else:
            raise ValueError("Precomputed glasso must be 3D or 4D when not vectorized.")

    return glasso_arr


def compute_fc_from_strategy_file(
    strategy_path: str | Path,
    *,
    tangent_connectivity: ConnectivityMeasure | ConnectomeTransformer | None = None,
    kinds: Optional[list[str]] = None,
    vectorize: bool = True,
    discard_diagonal: bool = True,
    glasso_lambda: float = 0.03,
    glasso: str | Path | np.ndarray | None = None,
    skip_glasso: bool = False,
    coverage: str | Path | np.ndarray | bool | None = None,
    coverage_threshold: float = 0.1,
    data_path: str | Path | None = None,
    print_timing: bool = False,
) -> dict[str, np.ndarray]:
    """
    Load a time-series .npy file and compute corr/partial/tangent/glasso FC.

    Parameters
    ----------
    strategy_path : str or Path
        Path to a .npy time-series file. Expected shapes:
        - (n_subjects, n_timepoints, n_nodes)
        - (n_subjects, n_timepoints, n_nodes, n_sessions)
    tangent_connectivity : ConnectivityMeasure or ConnectomeTransformer, optional
        Pre-fitted tangent estimator to avoid leakage. If None, tangent FC is
        fit on the provided data (fit + transform).
    kinds : list[str], optional
        Subset of FC types to compute. Supported values:
        'corr', 'partial' (or 'pc'), 'tangent' (or 'tang'), 'glasso'.
        Defaults to all.
    vectorize : bool, default=True
        If True, return vectorized upper triangle for each FC type.
    discard_diagonal : bool, default=True
        If vectorize=True, whether to discard diagonal elements.
    glasso_lambda : float, default=0.03
        L1 regularization parameter for graphical lasso.
    glasso : str, Path, np.ndarray, optional
        Precomputed glasso matrix or path to one. When provided, glasso is not
        recomputed and the array is returned as-is. It must match the expected
        output shape for the current vectorization and sessions.
    skip_glasso : bool, default=False
        If True, skip glasso computation and omit it from outputs.
    coverage : str, Path, np.ndarray, bool, optional
        Coverage source for ROI filtering. Options:
        - "site"/True: load coverage for the parsed site and atlas
        - "both"/"join": combine IHB + China coverage (keep ROIs good in both)
        - "ihb"/"china": force a specific site coverage
        - path/array: load coverage values directly
    coverage_threshold : float, default=0.1
        ROIs with coverage < threshold are discarded before FC estimation.
    data_path : str or Path, optional
        Base data root for coverage files (uses OPEN_CLOSE_BENCHMARK_DATA if unset).
    print_timing : bool, default=False
        If True, prints computation time per FC type.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of FC type to matrices. For 4D inputs, outputs are stacked
        along the last axis as sessions. If coverage is provided, FC is
        computed only on ROIs that pass the coverage threshold.
    """
    strategy_path = Path(strategy_path).expanduser()
    timeseries = np.load(strategy_path)

    if timeseries.ndim not in (3, 4):
        raise ValueError(
            f"Expected timeseries with 3 or 4 dimensions, got shape {timeseries.shape}"
        )

    coverage_mask = _resolve_coverage_mask(
        strategy_path=strategy_path,
        coverage=coverage,
        coverage_threshold=coverage_threshold,
        data_path=data_path,
    )
    if coverage_mask is not None:
        if coverage_mask.shape[0] != timeseries.shape[2]:
            raise ValueError(
                f"Coverage mask length {coverage_mask.shape[0]} does not match ROI count {timeseries.shape[2]}"
            )
        if not coverage_mask.any():
            raise ValueError("Coverage mask removed all ROIs.")
        if timeseries.ndim == 4:
            timeseries = timeseries[:, :, coverage_mask, :]
        elif timeseries.ndim == 3:
            timeseries = timeseries[:, :, coverage_mask]

    if timeseries.ndim == 4:
        sessions = [timeseries[..., idx] for idx in range(timeseries.shape[-1])]
    else:
        sessions = [timeseries]

    if skip_glasso and glasso is not None:
        raise ValueError("skip_glasso cannot be used with precomputed glasso.")

    kind_map = {
        "corr": "corr",
        "partial": "partial",
        "pc": "partial",
        "tangent": "tangent",
        "tang": "tangent",
        "glasso": "glasso",
    }
    default_order = ["corr", "partial", "tangent", "glasso"]
    if kinds is None:
        requested_kinds = default_order[:]
    else:
        requested_kinds = []
        for kind in kinds:
            if kind not in kind_map:
                raise ValueError(f"Unknown FC kind: {kind}")
            requested_kinds.append(kind_map[kind])
        requested_kinds = [kind for kind in default_order if kind in set(requested_kinds)]

    if skip_glasso:
        requested_kinds = [kind for kind in requested_kinds if kind != "glasso"]

    if not requested_kinds:
        raise ValueError("No FC kinds requested.")

    if glasso is not None and "glasso" not in requested_kinds:
        raise ValueError("glasso provided but glasso is not requested in kinds.")

    glasso_precomputed = None
    if glasso is not None:
        glasso_precomputed = _load_precomputed_glasso(
            glasso,
            expected_n_subjects=timeseries.shape[0],
            expected_n_rois=timeseries.shape[2],
            expected_n_sessions=len(sessions),
            vectorize=vectorize,
            discard_diagonal=discard_diagonal,
        )

    def compute_tangent(ts_session: np.ndarray) -> np.ndarray:
        if tangent_connectivity is None:
            transformer = ConnectomeTransformer(
                kind='tangent',
                vectorize=vectorize,
                discard_diagonal=discard_diagonal,
            )
            return transformer.fit_transform(ts_session)

        if isinstance(tangent_connectivity, ConnectomeTransformer):
            if tangent_connectivity.kind != 'tangent':
                raise ValueError("tangent_connectivity must be kind='tangent'.")
            if not tangent_connectivity.is_fitted_:
                raise ValueError("tangent_connectivity must be fitted before use.")
            if tangent_connectivity.vectorize != vectorize:
                raise ValueError("tangent_connectivity vectorize setting does not match.")
            if tangent_connectivity.discard_diagonal != discard_diagonal:
                raise ValueError("tangent_connectivity discard_diagonal setting does not match.")
            return tangent_connectivity.transform(ts_session)

        if isinstance(tangent_connectivity, ConnectivityMeasure):
            if getattr(tangent_connectivity, 'kind', None) != 'tangent':
                raise ValueError("tangent_connectivity must be kind='tangent'.")
            conn = tangent_connectivity.transform(ts_session)
            if vectorize:
                conn = sym_matrix_to_vec(conn, discard_diagonal=discard_diagonal)
            return conn

        raise TypeError(
            "tangent_connectivity must be a ConnectivityMeasure, ConnectomeTransformer, or None."
        )

    compute_glasso = "glasso" in requested_kinds and glasso_precomputed is None
    compute_kinds = [kind for kind in requested_kinds if kind != "glasso"]
    if compute_glasso:
        compute_kinds.append("glasso")

    outputs: dict[str, list[np.ndarray]] = {key: [] for key in compute_kinds}
    timings: dict[str, float] = {key: 0.0 for key in compute_kinds}

    for ts_session in sessions:
        if "corr" in outputs:
            start = time.perf_counter()
            outputs['corr'].append(
                ConnectomeTransformer(
                    kind='corr',
                    vectorize=vectorize,
                    discard_diagonal=discard_diagonal,
                ).fit_transform(ts_session)
            )
            timings['corr'] += time.perf_counter() - start

        if "partial" in outputs:
            start = time.perf_counter()
            outputs['partial'].append(
                ConnectomeTransformer(
                    kind='partial',
                    vectorize=vectorize,
                    discard_diagonal=discard_diagonal,
                ).fit_transform(ts_session)
            )
            timings['partial'] += time.perf_counter() - start

        if "tangent" in outputs:
            start = time.perf_counter()
            outputs['tangent'].append(compute_tangent(ts_session))
            timings['tangent'] += time.perf_counter() - start

        if "glasso" in outputs:
            start = time.perf_counter()
            outputs['glasso'].append(
                ConnectomeTransformer(
                    kind='glasso',
                    vectorize=vectorize,
                    discard_diagonal=discard_diagonal,
                    glasso_lambda=glasso_lambda,
                ).fit_transform(ts_session)
            )
            timings['glasso'] += time.perf_counter() - start

    results: dict[str, np.ndarray] = {}
    for key, values in outputs.items():
        results[key] = values[0] if len(values) == 1 else np.stack(values, axis=-1)

    if glasso_precomputed is not None:
        results['glasso'] = glasso_precomputed

    if print_timing:
        session_note = ""
        if len(sessions) > 1:
            session_note = f" (sum over {len(sessions)} sessions)"
        for key in requested_kinds:
            if key == "glasso":
                if glasso_precomputed is not None:
                    print(f"glasso time: 0.000s (precomputed){session_note}")
                elif compute_glasso:
                    print(f"glasso time: {timings['glasso']:.3f}s{session_note}")
                else:
                    print(f"glasso time: skipped{session_note}")
            else:
                print(f"{key} time: {timings[key]:.3f}s{session_note}")

    return results


# =============================================================================
# Classification-ready FC computation (leakage-safe)
# =============================================================================

def compute_fc_vectorized(
    ts: np.ndarray,
    *,
    fc_type: str,
    coverage_mask: Optional[np.ndarray] = None,
    glasso_lambda: float = 0.03,
    tangent_transformer: Optional['ConnectomeTransformer'] = None,
) -> tuple[np.ndarray, Optional['ConnectomeTransformer']]:
    """
    Compute VECTORIZED FC for a single site's timeseries.

    All FC is returned as vectorized (2D: n_samples, n_edges).
    For tangent, can use a pre-fitted transformer for leakage control.

    Parameters
    ----------
    ts : np.ndarray
        Timeseries, shape (n_samples, n_timepoints, n_rois)
    fc_type : str
        FC type: 'corr', 'partial', 'tangent', 'glasso'
    coverage_mask : np.ndarray, optional
        Boolean mask of good ROIs (True = keep)
    glasso_lambda : float
        L1 regularization for glasso
    tangent_transformer : ConnectomeTransformer, optional
        Pre-fitted tangent transformer (for leakage-safe test computation).
        If provided with fc_type='tangent', uses this transformer.
        If None and fc_type='tangent', fits a new transformer.

    Returns
    -------
    fc : np.ndarray
        Vectorized FC, shape (n_samples, n_edges)
    transformer : ConnectomeTransformer or None
        Fitted tangent transformer (only for tangent, None otherwise).
        Use this to transform test data with the same reference.
    """
    # Apply coverage mask BEFORE FC computation
    if coverage_mask is not None:
        coverage_mask = np.asarray(coverage_mask, dtype=bool)
        if coverage_mask.shape[0] != ts.shape[2]:
            raise ValueError(
                f"Coverage mask length {coverage_mask.shape[0]} does not match "
                f"ROI count {ts.shape[2]}"
            )
        ts = ts[:, :, coverage_mask]

    kind_map = {
        'corr': 'corr',
        'partial': 'partial',
        'tangent': 'tangent',
        'glasso': 'glasso',
    }
    if fc_type not in kind_map:
        raise ValueError(f"Unknown fc_type: {fc_type}. Valid: {list(kind_map.keys())}")

    kind = kind_map[fc_type]

    if kind == 'tangent':
        if tangent_transformer is not None:
            # Use pre-fitted transformer (leakage-safe for test data)
            if not tangent_transformer.is_fitted_:
                raise ValueError("tangent_transformer must be fitted")
            fc = tangent_transformer.transform(ts)
            return fc, tangent_transformer
        else:
            # Fit new transformer (for training data)
            transformer = ConnectomeTransformer(
                kind='tangent',
                vectorize=True,
                discard_diagonal=True,
            )
            fc = transformer.fit_transform(ts)
            return fc, transformer
    else:
        # For corr/partial/glasso, no cross-sample reference needed
        transformer = ConnectomeTransformer(
            kind=kind,
            vectorize=True,
            discard_diagonal=True,
            glasso_lambda=glasso_lambda,
        )
        fc = transformer.fit_transform(ts)
        return fc, None


def compute_fc_train_test(
    ts_train: np.ndarray,
    ts_test: np.ndarray,
    *,
    fc_type: str,
    coverage_mask: Optional[np.ndarray] = None,
    glasso_lambda: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute VECTORIZED FC for train and test data with leakage-safe tangent.

    CRITICAL: For tangent, reference is fitted on ts_train ONLY.

    Parameters
    ----------
    ts_train : np.ndarray
        Training timeseries, shape (n_train, n_timepoints, n_rois)
    ts_test : np.ndarray
        Test timeseries, shape (n_test, n_timepoints, n_rois)
        NOTE: Can have different n_timepoints than ts_train.
    fc_type : str
        FC type: 'corr', 'partial', 'tangent', 'glasso'
    coverage_mask : np.ndarray, optional
        Boolean mask of good ROIs (True = keep)
    glasso_lambda : float
        L1 regularization for glasso

    Returns
    -------
    X_train : np.ndarray
        Training FC, shape (n_train, n_edges) - VECTORIZED
    X_test : np.ndarray
        Test FC, shape (n_test, n_edges) - VECTORIZED
    """
    # Compute FC for train (and get tangent transformer if applicable)
    X_train, tangent_transformer = compute_fc_vectorized(
        ts_train,
        fc_type=fc_type,
        coverage_mask=coverage_mask,
        glasso_lambda=glasso_lambda,
        tangent_transformer=None,  # Fit new for train
    )

    # Compute FC for test (use train's tangent transformer for leakage control)
    X_test, _ = compute_fc_vectorized(
        ts_test,
        fc_type=fc_type,
        coverage_mask=coverage_mask,
        glasso_lambda=glasso_lambda,
        tangent_transformer=tangent_transformer,  # Use train's reference
    )

    return X_train, X_test


def get_fc_types_to_compute(
    requested_fc_types: list[str],
    skip_glasso: bool = False,
) -> list[str]:
    """
    Filter FC types based on skip_glasso option.

    Parameters
    ----------
    requested_fc_types : list[str]
        Requested FC types (e.g., ['corr', 'partial', 'tangent', 'glasso'])
    skip_glasso : bool
        If True, remove 'glasso' from the list

    Returns
    -------
    list[str]
        Filtered FC types
    """
    valid_types = {'corr', 'partial', 'tangent', 'glasso'}
    fc_types = [fc for fc in requested_fc_types if fc in valid_types]

    if skip_glasso:
        fc_types = [fc for fc in fc_types if fc != 'glasso']

    return fc_types


# =============================================================================
# Legacy function for backward compatibility
# =============================================================================

def get_connectome(
    timeseries: np.ndarray,
    conn_type: str = 'corr',
    vectorize: bool = False,
) -> np.ndarray:
    """
    Compute a connectivity matrix from a given timeseries.

    This is a legacy wrapper around ConnectomeTransformer for backward
    compatibility. For new code, prefer using ConnectomeTransformer directly
    for leakage-safe tangent embedding.

    Parameters
    ----------
    timeseries : np.ndarray
        The input timeseries to compute the connectivity matrix from.
        Input shape: (n_subjects, n_timepoints, n_nodes)
    conn_type : str
        The type of connectivity to compute.
        Options: 'corr', 'partial_corr', 'tang', 'glasso'.
    vectorize : bool, default=False
        If True, return vectorized upper triangle.

    Returns
    -------
    conn : np.ndarray
        The computed connectivity matrix.
    """
    # Map legacy names to new names
    kind_map = {
        'corr': 'corr',
        'partial_corr': 'partial',
        'tang': 'tangent',
        'glasso': 'glasso',
    }

    if conn_type not in kind_map:
        raise ValueError(
            f"Unknown conn_type: {conn_type}. "
            f"Valid options: {list(kind_map.keys())}"
        )

    transformer = ConnectomeTransformer(
        kind=kind_map[conn_type],
        vectorize=vectorize,
    )
    return transformer.fit_transform(timeseries)


# Legacy alias for backward compatibility
def graphicalLasso(data: np.ndarray, L1: float = 0.03) -> np.ndarray:
    """
    Legacy wrapper for graphical_lasso_single.

    Deprecated: Use graphical_lasso_single() instead.
    """
    return graphical_lasso_single(data, lambda1=L1)
