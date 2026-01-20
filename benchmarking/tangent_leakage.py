"""
Tangent Space Leakage Validation Experiment
============================================

This script validates whether computing the tangent space reference matrix
on all subjects (including test) causes significant data leakage.

Background
----------
For tangent space projection (nilearn's ConnectivityMeasure(kind='tangent')),
a reference matrix is computed as the geometric mean of all input covariance
matrices. If this reference includes test subjects, information may leak from
test to training data.

Experimental Design
-------------------
We compare two conditions:

1. LEAK (current approach):
   - Compute tangent FC on ALL subjects first
   - Then split for cross-validation
   - Reference matrix contains information from test subjects

2. NO_LEAK (proper approach):
   - For each CV fold, fit reference on TRAIN subjects only
   - Transform TEST subjects using the train-only reference
   - No information leakage possible

Cross-Validation Protocol
-------------------------
- 3-fold GroupKFold (subject-level splitting)
- Each subject appears in exactly one test fold
- Both EO and EC scans from same subject are in same fold
- ~28 subjects per test fold (for 84 total subjects)

Interpretation
--------------
- Δ < 1%: Leakage effect negligible, main results valid
- Δ 1-3%: Small but measurable effect, cross-site design justified
- Δ > 3%: Substantial leakage, cross-site design essential

Usage
-----
    # Quick test (1 fold)
    python tangent_leakage.py --config configs/tangent_leakage_quick.yaml

    # Full experiment (3 folds)
    python tangent_leakage.py --config configs/tangent_leakage_full.yaml

    # Custom parameters
    python tangent_leakage.py --n_splits 5 --pca_components 0.95

Authors: Medvedeva et al., 2025
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from nilearn.connectome import ConnectivityMeasure, sym_matrix_to_vec
import warnings

warnings.filterwarnings('ignore')


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_timeseries(data_path: str) -> tuple:
    """
    Load and prepare time series data.

    Parameters
    ----------
    data_path : str
        Path to directory containing ihb_close_*.npy and ihb_open_*.npy

    Returns
    -------
    timeseries : ndarray (n_samples, n_timepoints, n_rois)
    subjects : ndarray (n_samples,) - subject ID for each sample
    labels : ndarray (n_samples,) - 0=EC, 1=EO
    """
    data_path = Path(data_path).expanduser()

    # Find time series files
    close_files = list(data_path.glob('*_close_*.npy'))
    open_files = list(data_path.glob('*_open_*.npy'))

    if not close_files or not open_files:
        raise FileNotFoundError(f"No time series files found in {data_path}")

    # Load data
    ts_close = np.load(close_files[0])  # (n_subjects, n_timepoints, n_rois)
    ts_open = np.load(open_files[0])

    n_subjects = ts_close.shape[0]
    n_timepoints = ts_close.shape[1]
    n_rois = ts_close.shape[2]

    print(f"Loaded data: {n_subjects} subjects, {n_timepoints} timepoints, {n_rois} ROIs")
    print(f"  Close file: {close_files[0].name}")
    print(f"  Open file: {open_files[0].name}")

    # Interleave: [sub0_EC, sub0_EO, sub1_EC, sub1_EO, ...]
    timeseries = np.empty((n_subjects * 2, n_timepoints, n_rois), dtype=np.float32)
    timeseries[0::2] = ts_close  # EC at even indices
    timeseries[1::2] = ts_open   # EO at odd indices

    # Subject IDs: [0,0,1,1,2,2,...,83,83]
    subjects = np.repeat(np.arange(n_subjects), 2)

    # Labels: [0,1,0,1,...] = [EC,EO,EC,EO,...]
    labels = np.tile([0, 1], n_subjects)

    return timeseries, subjects, labels


def run_classification(X_train, X_test, y_train, y_test,
                       pca_components=0.95, random_state=42):
    """
    Run PCA + Logistic Regression classification.

    Parameters
    ----------
    X_train, X_test : ndarray
        Feature vectors (flattened FC matrices)
    y_train, y_test : ndarray
        Labels
    pca_components : float or int
        Number of PCA components or variance ratio
    random_state : int
        Random seed

    Returns
    -------
    dict with train_acc, test_acc, n_components
    """
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA (fit on train only)
    pca = PCA(n_components=pca_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Logistic Regression
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train_pca, y_train)

    train_acc = clf.score(X_train_pca, y_train)
    test_acc = clf.score(X_test_pca, y_test)

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'n_components': X_train_pca.shape[1]
    }


def generate_cv_splits(subjects, labels, n_splits=3):
    """
    Generate CV splits once to ensure identical splits for both conditions.

    Parameters
    ----------
    subjects : ndarray
        Subject IDs for each sample
    labels : ndarray
        Labels for each sample
    n_splits : int
        Number of CV folds

    Returns
    -------
    list of tuples (train_idx, test_idx)
    """
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(np.zeros(len(subjects)), labels, groups=subjects))
    return splits


def experiment_leak(timeseries, subjects, labels, cv_splits,
                    pca_components=0.95, random_state=42):
    """
    Condition LEAK: Compute tangent FC on ALL subjects, then split.

    This is the potentially problematic approach where the reference
    matrix includes test subjects.

    Parameters
    ----------
    cv_splits : list of tuples
        Pre-generated (train_idx, test_idx) splits

    Returns
    -------
    list of dicts with fold results
    """
    print("\n" + "="*60)
    print("CONDITION: LEAK (reference computed on ALL subjects)")
    print("="*60)

    # Compute tangent FC on ALL subjects (potential leakage!)
    print("Computing tangent FC on all subjects...")
    tang_all = ConnectivityMeasure(kind='tangent')
    fc_all = tang_all.fit_transform(timeseries)
    fc_all_vec = sym_matrix_to_vec(fc_all, discard_diagonal=True)
    print(f"FC shape: {fc_all_vec.shape}")

    results = []

    for fold, (train_idx, test_idx) in enumerate(cv_splits):
        X_train, X_test = fc_all_vec[train_idx], fc_all_vec[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        metrics = run_classification(X_train, X_test, y_train, y_test,
                                     pca_components=pca_components,
                                     random_state=random_state)

        n_train_subj = len(np.unique(subjects[train_idx]))
        n_test_subj = len(np.unique(subjects[test_idx]))

        fold_result = {
            'fold': fold + 1,
            'n_train_subj': n_train_subj,
            'n_test_subj': n_test_subj,
            'n_train_samples': len(train_idx),
            'n_test_samples': len(test_idx),
            **metrics
        }
        results.append(fold_result)

        print(f"  Fold {fold+1}: train_acc={metrics['train_acc']:.4f}, test_acc={metrics['test_acc']:.4f} "
              f"(train: {n_train_subj} subj, test: {n_test_subj} subj, PCA: {metrics['n_components']} comp)")

    return results


def experiment_no_leak(timeseries, subjects, labels, cv_splits,
                       pca_components=0.95, random_state=42):
    """
    Condition NO_LEAK: Fit tangent reference on TRAIN only.

    This is the proper approach where no information from test
    subjects is used in computing the reference matrix.

    Parameters
    ----------
    cv_splits : list of tuples
        Pre-generated (train_idx, test_idx) splits (same as LEAK condition)

    Returns
    -------
    list of dicts with fold results
    """
    print("\n" + "="*60)
    print("CONDITION: NO_LEAK (reference computed on TRAIN only)")
    print("="*60)

    results = []

    for fold, (train_idx, test_idx) in enumerate(cv_splits):
        ts_train, ts_test = timeseries[train_idx], timeseries[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Fit tangent reference on TRAIN subjects only
        print(f"  Fold {fold+1}: Fitting tangent on train subjects...")
        tang = ConnectivityMeasure(kind='tangent')
        fc_train = tang.fit_transform(ts_train)

        # Transform TEST using TRAIN reference (no leakage)
        fc_test = tang.transform(ts_test)

        # Vectorize
        fc_train_vec = sym_matrix_to_vec(fc_train, discard_diagonal=True)
        fc_test_vec = sym_matrix_to_vec(fc_test, discard_diagonal=True)

        metrics = run_classification(fc_train_vec, fc_test_vec, y_train, y_test,
                                     pca_components=pca_components,
                                     random_state=random_state)

        n_train_subj = len(np.unique(subjects[train_idx]))
        n_test_subj = len(np.unique(subjects[test_idx]))

        fold_result = {
            'fold': fold + 1,
            'n_train_subj': n_train_subj,
            'n_test_subj': n_test_subj,
            'n_train_samples': len(train_idx),
            'n_test_samples': len(test_idx),
            **metrics
        }
        results.append(fold_result)

        print(f"  Fold {fold+1}: train_acc={metrics['train_acc']:.4f}, test_acc={metrics['test_acc']:.4f} "
              f"(train: {n_train_subj} subj, test: {n_test_subj} subj, PCA: {metrics['n_components']} comp)")

    return results


def create_results_table(leak_results, no_leak_results):
    """
    Create a pandas DataFrame with results for all folds.

    Returns
    -------
    df_folds : DataFrame with per-fold results
    df_summary : DataFrame with aggregated results
    """
    import pandas as pd

    # Per-fold results
    rows = []
    for leak, no_leak in zip(leak_results, no_leak_results):
        rows.append({
            'fold': leak['fold'],
            'condition': 'LEAK',
            'n_train_subj': leak['n_train_subj'],
            'n_test_subj': leak['n_test_subj'],
            'n_train_samples': leak['n_train_samples'],
            'n_test_samples': leak['n_test_samples'],
            'train_acc': leak['train_acc'],
            'test_acc': leak['test_acc'],
            'n_pca_components': leak['n_components']
        })
        rows.append({
            'fold': no_leak['fold'],
            'condition': 'NO_LEAK',
            'n_train_subj': no_leak['n_train_subj'],
            'n_test_subj': no_leak['n_test_subj'],
            'n_train_samples': no_leak['n_train_samples'],
            'n_test_samples': no_leak['n_test_samples'],
            'train_acc': no_leak['train_acc'],
            'test_acc': no_leak['test_acc'],
            'n_pca_components': no_leak['n_components']
        })

    df_folds = pd.DataFrame(rows)

    # Aggregated summary
    leak_test_accs = [r['test_acc'] for r in leak_results]
    leak_train_accs = [r['train_acc'] for r in leak_results]
    no_leak_test_accs = [r['test_acc'] for r in no_leak_results]
    no_leak_train_accs = [r['train_acc'] for r in no_leak_results]

    summary_rows = [
        {
            'condition': 'LEAK',
            'train_acc_mean': np.mean(leak_train_accs),
            'train_acc_std': np.std(leak_train_accs),
            'test_acc_mean': np.mean(leak_test_accs),
            'test_acc_std': np.std(leak_test_accs),
        },
        {
            'condition': 'NO_LEAK',
            'train_acc_mean': np.mean(no_leak_train_accs),
            'train_acc_std': np.std(no_leak_train_accs),
            'test_acc_mean': np.mean(no_leak_test_accs),
            'test_acc_std': np.std(no_leak_test_accs),
        },
        {
            'condition': 'DIFFERENCE',
            'train_acc_mean': np.mean(leak_train_accs) - np.mean(no_leak_train_accs),
            'train_acc_std': np.nan,
            'test_acc_mean': np.mean(leak_test_accs) - np.mean(no_leak_test_accs),
            'test_acc_std': np.nan,
        }
    ]
    df_summary = pd.DataFrame(summary_rows)

    return df_folds, df_summary


def print_results(leak_results, no_leak_results):
    """Print formatted results and interpretation."""
    import pandas as pd

    df_folds, df_summary = create_results_table(leak_results, no_leak_results)

    print("\n" + "="*60)
    print("RESULTS - PER FOLD")
    print("="*60)
    print(df_folds.to_string(index=False))

    print("\n" + "="*60)
    print("RESULTS - AGGREGATED")
    print("="*60)
    print(df_summary.to_string(index=False))

    # Extract values for interpretation
    leak_mean = df_summary[df_summary['condition'] == 'LEAK']['test_acc_mean'].values[0]
    leak_std = df_summary[df_summary['condition'] == 'LEAK']['test_acc_std'].values[0]
    no_leak_mean = df_summary[df_summary['condition'] == 'NO_LEAK']['test_acc_mean'].values[0]
    no_leak_std = df_summary[df_summary['condition'] == 'NO_LEAK']['test_acc_std'].values[0]
    diff = df_summary[df_summary['condition'] == 'DIFFERENCE']['test_acc_mean'].values[0]

    print("\n" + "-"*60)
    print("INTERPRETATION")
    print("-"*60)

    abs_diff = abs(diff)
    if abs_diff < 0.01:
        interpretation = "NEGLIGIBLE"
        detail = "Leakage effect < 1%. Main results are valid."
    elif abs_diff < 0.03:
        interpretation = "SMALL"
        detail = "Leakage effect 1-3%. Cross-site design is justified."
    else:
        interpretation = "SUBSTANTIAL"
        detail = "Leakage effect > 3%. Cross-site design is essential."

    print(f"\nLeakage effect: {interpretation}")
    print(f"{detail}")

    n_folds = len(leak_results)
    print("\n" + "-"*60)
    print("FOR MANUSCRIPT (Supplementary Materials)")
    print("-"*60)
    print(f"""
To assess potential data leakage in tangent space projection, we compared
classification accuracy under two conditions: (1) reference matrix computed
on all subjects before cross-validation split (LEAK), and (2) reference matrix
computed only on training subjects within each fold (NO_LEAK).

Using {n_folds}-fold subject-level cross-validation on the IHB dataset:
- LEAK: {leak_mean:.3f} ± {leak_std:.3f}
- NO_LEAK: {no_leak_mean:.3f} ± {no_leak_std:.3f}
- Difference: {diff*100:+.2f} percentage points

The {interpretation.lower()} difference ({abs_diff*100:.2f}%) suggests that {detail.lower()}
In our main cross-site analysis, tangent reference was computed exclusively
on the training site, eliminating any possibility of leakage.
""")

    return df_folds, df_summary, interpretation


def main():
    parser = argparse.ArgumentParser(
        description='Tangent Space Leakage Validation Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--data_path', type=str,
                        default='~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/schafer200_time-series/ihb',
                        help='Path to time series data')
    parser.add_argument('--n_splits', type=int, default=3,
                        help='Number of CV folds (default: 3)')
    parser.add_argument('--pca_components', type=float, default=0.95,
                        help='PCA variance ratio or n_components (default: 0.95)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (optional)')

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        data_path = config.get('data_path', args.data_path)
        n_splits = config.get('n_splits', args.n_splits)
        pca_components = config.get('pca_components', args.pca_components)
        random_state = config.get('random_state', args.random_state)
        output = config.get('output', args.output)
    else:
        data_path = args.data_path
        n_splits = args.n_splits
        pca_components = args.pca_components
        random_state = args.random_state
        output = args.output

    print("="*60)
    print("TANGENT SPACE LEAKAGE VALIDATION EXPERIMENT")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data path: {data_path}")
    print(f"  CV folds: {n_splits}")
    print(f"  PCA components: {pca_components}")
    print(f"  Random state: {random_state}")

    # Load data
    print("\n" + "-"*60)
    print("Loading data...")
    print("-"*60)
    timeseries, subjects, labels = load_timeseries(data_path)

    # Generate CV splits ONCE to ensure identical splits for both conditions
    print("\n" + "-"*60)
    print("Generating CV splits...")
    print("-"*60)
    cv_splits = generate_cv_splits(subjects, labels, n_splits=n_splits)
    print(f"Generated {len(cv_splits)} folds (same splits used for both LEAK and NO_LEAK)")

    # Run experiments with IDENTICAL splits
    leak_results = experiment_leak(timeseries, subjects, labels,
                                   cv_splits=cv_splits,
                                   pca_components=pca_components,
                                   random_state=random_state)

    no_leak_results = experiment_no_leak(timeseries, subjects, labels,
                                         cv_splits=cv_splits,
                                         pca_components=pca_components,
                                         random_state=random_state)

    # Print results
    df_folds, df_summary, interpretation = print_results(leak_results, no_leak_results)

    # Save results if output specified
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save per-fold results
        folds_path = output_path.with_name(output_path.stem + '_folds.csv')
        df_folds.to_csv(folds_path, index=False)
        print(f"\nPer-fold results saved to: {folds_path}")

        # Save summary
        summary_path = output_path.with_name(output_path.stem + '_summary.csv')
        df_summary.to_csv(summary_path, index=False)
        print(f"Summary results saved to: {summary_path}")

    return df_folds, df_summary, interpretation


if __name__ == '__main__':
    main()
