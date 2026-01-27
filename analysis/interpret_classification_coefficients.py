"""
Classification Coefficient Interpretation via Subsampling Stability
====================================================================

This script identifies robust biomarkers for Eyes Open (EO) vs Eyes Closed (EC)
classification by analyzing the stability of logistic regression coefficients
across multiple random subsamples of the pooled dataset (IHB + China).

Approach
--------
1. Pool subjects from both sites (132 total: 84 IHB + 48 China)
2. Precompute correlation and tangent FC for ALL subjects once
   - Tangent reference (Fréchet mean) fitted on full dataset
3. Repeatedly subsample 80% of subjects (stratified by site, without replacement)
4. For each subsample: select precomputed FC, fit classifier, backproject weights
5. Assess stability: edges with consistent sign (≥80%) and high rank (≤500) are "stable"

This precomputation approach provides ~20x speedup over computing FC per subsample.

Key Design Choices
------------------
- Subject-level sampling: Both EO and EC scans included when subject selected
- Same subsample used for both FC types to enable direct comparison
- Strategy 3 (aCompCor(50%)+12P) used as best-performing denoising approach
- Schaefer200 atlas with Yeo 7-network parcellation for interpretability

Usage
-----
    # Quick test (5 iterations)
    python -m analysis.interpret_classification_coefficients --n-subsamples 5

    # Full analysis (1000 iterations, recommended)
    python -m analysis.interpret_classification_coefficients --n-subsamples 1000

    # Or using the activation script
    source venv/bin/activate && PYTHONPATH=. python analysis/interpret_classification_coefficients.py --n-subsamples 1000

Outputs
-------
Results are saved to results/interpretation/subsample/:
- edge_stability_stats_{corr,tangent}.csv: Full stability metrics per edge
- stable_edges_{corr,tangent}.csv: Filtered stable edges
- stable_edges_EO_{corr,tangent}.csv: Edges favoring Eyes Open
- stable_edges_EC_{corr,tangent}.csv: Edges favoring Eyes Closed

Figures are saved to results/figures/:
- stability_volcano_{corr,tangent}.png: Sign consistency vs mean weight
- heatmap_stable_{EO,EC}_count_{corr,tangent}.png: Network-level edge counts
- heatmap_stable_{EO,EC}_weight_{corr,tangent}.png: Network-level importance
- fc_comparison_scatter.png: Correlation vs tangent weight comparison

References
----------
See Methods.md Section 5 for detailed methodology description.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from tqdm import tqdm

# --- Configuration ---
ATLAS = "Schaefer200"
STRATEGY = "3"  # Best performing: aCompCor(50%)+12P
GSR = "GSR"
FC_TYPES = ["corr", "tangent"]  # Both FC types
DATA_ROOT = Path("~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data").expanduser()
OUTPUT_DIR = Path("results/interpretation/subsample")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Subsampling parameters
SUBSAMPLE_FRACTION = 0.8
N_SUBSAMPLES = 1000  # Default, can be overridden via CLI

# Define plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)


def load_network_labels():
    """Load Schaefer labels and parse network assignment."""
    labels_path = DATA_ROOT / "schaefer_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    df = pd.read_csv(labels_path)
    # Expected format like: 7Networks_LH_Vis_1
    # We want 'Vis'
    def parse_net(name):
        parts = name.split('_')
        if len(parts) >= 3:
            return parts[2]
        return "Unknown"

    df['network'] = df['name'].apply(parse_net)
    return df


def get_masked_network_mapping(schaefer_df):
    """Apply coverage mask to identify which networks remain."""
    # We use IHB coverage mask as per standard pipeline
    coverage_path = DATA_ROOT / "coverage" / "ihb_Schaefer200_parcel_coverage.npy"
    coverage = np.load(coverage_path)
    mask = coverage >= 0.1

    kept_indices = np.where(mask)[0]
    # schaefer_df is already masked to 182 ROIs
    kept_networks = schaefer_df['network'].tolist()
    kept_names = schaefer_df['name'].tolist()
    return kept_networks, kept_names, mask, kept_indices


def load_subject_level_data(mask):
    """
    Load timeseries at subject level with paired EO/EC scans.
    Returns:
        subjects_data: list of dicts with 'close', 'open', 'site' keys
        subject_ids: list of subject identifiers
    """
    print(f"Loading timeseries for {ATLAS} strategy-{STRATEGY} {GSR}...")

    subjects_data = []
    subject_ids = []

    # --- Load China ---
    china_close = np.load(DATA_ROOT / f"timeseries_china/{ATLAS}/china_close_{ATLAS}_strategy-{STRATEGY}_{GSR}.npy")
    china_open = np.load(DATA_ROOT / f"timeseries_china/{ATLAS}/china_open_{ATLAS}_strategy-{STRATEGY}_{GSR}.npy")

    # Use only first session of closed (4D -> 3D)
    china_close = china_close[:, :, :, 0]

    # Apply mask
    china_close = china_close[:, :, mask]
    china_open = china_open[:, :, mask]

    n_china = len(china_close)
    for i in range(n_china):
        subjects_data.append({
            'close': china_close[i],  # (n_timepoints, n_rois)
            'open': china_open[i],
            'site': 'china'
        })
        subject_ids.append(f"china_{i:03d}")

    # --- Load IHB ---
    ihb_close = np.load(DATA_ROOT / f"timeseries_ihb/{ATLAS}/ihb_close_{ATLAS}_strategy-{STRATEGY}_{GSR}.npy")
    ihb_open = np.load(DATA_ROOT / f"timeseries_ihb/{ATLAS}/ihb_open_{ATLAS}_strategy-{STRATEGY}_{GSR}.npy")

    ihb_close = ihb_close[:, :, mask]
    ihb_open = ihb_open[:, :, mask]

    n_ihb = len(ihb_close)
    for i in range(n_ihb):
        subjects_data.append({
            'close': ihb_close[i],
            'open': ihb_open[i],
            'site': 'ihb'
        })
        subject_ids.append(f"ihb_{i:03d}")

    print(f"Loaded {n_china} China + {n_ihb} IHB = {len(subjects_data)} total subjects")
    return subjects_data, subject_ids


def stratified_subsample(site_labels, fraction, rng):
    """
    Subsample subjects stratified by site (without replacement).
    Returns indices of selected subjects.
    """
    # Separate by site
    china_indices = np.where(np.array(site_labels) == 'china')[0]
    ihb_indices = np.where(np.array(site_labels) == 'ihb')[0]

    # Sample fraction from each site
    n_china_sample = int(len(china_indices) * fraction)
    n_ihb_sample = int(len(ihb_indices) * fraction)

    selected_china = rng.choice(china_indices, size=n_china_sample, replace=False)
    selected_ihb = rng.choice(ihb_indices, size=n_ihb_sample, replace=False)

    return np.concatenate([selected_china, selected_ihb])


def precompute_all_fc(subjects_data):
    """
    Precompute correlation and tangent FC for ALL subjects.
    Tangent reference is computed from the full dataset.

    Returns:
        fc_corr_close: (n_subjects, n_edges) correlation FC for closed eyes
        fc_corr_open: (n_subjects, n_edges) correlation FC for open eyes
        fc_tang_close: (n_subjects, n_edges) tangent FC for closed eyes
        fc_tang_open: (n_subjects, n_edges) tangent FC for open eyes
        site_labels: list of site labels per subject
    """
    from data_utils.fc import ConnectomeTransformer

    n_subjects = len(subjects_data)

    # Build list of all timeseries
    ts_close_list = [subj['close'] for subj in subjects_data]
    ts_open_list = [subj['open'] for subj in subjects_data]
    all_ts_list = ts_close_list + ts_open_list

    # Get site labels
    site_labels = [subj['site'] for subj in subjects_data]

    n_rois = subjects_data[0]['close'].shape[1]
    n_edges = n_rois * (n_rois - 1) // 2
    print(f"  Precomputing FC for {n_subjects} subjects, {n_rois} ROIs, {n_edges} edges...")

    # Compute correlation FC
    print("  Computing correlation FC...")
    corr_transformer = ConnectomeTransformer(kind='corr', vectorize=True)
    corr_transformer.fit(all_ts_list)
    fc_corr_close = corr_transformer.transform(ts_close_list)
    fc_corr_open = corr_transformer.transform(ts_open_list)

    # Compute tangent FC (fitted on ALL data)
    print("  Computing tangent FC (fitting on full dataset)...")
    tang_transformer = ConnectomeTransformer(kind='tangent', vectorize=True)
    tang_transformer.fit(all_ts_list)
    fc_tang_close = tang_transformer.transform(ts_close_list)
    fc_tang_open = tang_transformer.transform(ts_open_list)

    print(f"  Done. Shapes: corr={fc_corr_close.shape}, tang={fc_tang_close.shape}")

    return fc_corr_close, fc_corr_open, fc_tang_close, fc_tang_open, site_labels


def get_fc_for_subsample(fc_close, fc_open, selected_indices):
    """
    Get precomputed FC matrices for a subsample.
    Returns: X (n_samples, n_edges), y (n_samples,)
    """
    fc_c = fc_close[selected_indices]
    fc_o = fc_open[selected_indices]

    X = np.concatenate([fc_c, fc_o], axis=0)
    y = np.array([0] * len(fc_c) + [1] * len(fc_o))

    return X, y


def fit_and_backproject(X, y):
    """
    Fit pipeline and backproject weights to edge space.
    Returns: edge weights (n_edges,)
    """
    scaler = StandardScaler()
    pca = PCA(n_components=0.95, random_state=42)
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)

    X_scaled = scaler.fit_transform(X)
    X_pca = pca.fit_transform(X_scaled)
    clf.fit(X_pca, y)

    # Backproject: coef @ components
    w_scaled = clf.coef_ @ pca.components_
    return w_scaled.flatten()


def subsampling_analysis(fc_data, site_labels, n_subsamples):
    """
    Run subsampling analysis for both FC types using precomputed data.

    Parameters:
        fc_data: dict with 'corr' and 'tangent' keys, each containing (fc_close, fc_open) tuple
        site_labels: list of site labels per subject
        n_subsamples: number of subsampling iterations

    Returns: dict with weights and ranks for each FC type
    """
    print(f"Running {n_subsamples} subsampling iterations...")

    results = {fc_type: {'weights': [], 'ranks': []} for fc_type in FC_TYPES}

    rng = np.random.default_rng(seed=42)

    for _ in tqdm(range(n_subsamples), desc="Subsampling iterations"):
        # 1. Stratified subject subsampling (same subjects for both FC types)
        selected_indices = stratified_subsample(
            site_labels, SUBSAMPLE_FRACTION, rng
        )

        # 2. Get precomputed FC for each type and fit model
        for fc_type in FC_TYPES:
            fc_close, fc_open = fc_data[fc_type]
            X, y = get_fc_for_subsample(fc_close, fc_open, selected_indices)

            w = fit_and_backproject(X, y)

            results[fc_type]['weights'].append(w)

            # Compute ranks (1 = highest magnitude)
            abs_w = np.abs(w)
            ranks = np.argsort(np.argsort(-abs_w)) + 1
            results[fc_type]['ranks'].append(ranks)

    # Stack results
    for fc_type in FC_TYPES:
        results[fc_type]['weights'] = np.vstack(results[fc_type]['weights'])
        results[fc_type]['ranks'] = np.vstack(results[fc_type]['ranks'])

    return results


def aggregate_and_filter(weights_stack, ranks_stack, kept_networks, kept_names, kept_indices, fc_type):
    """Compute stability metrics and filter edges."""
    # 1. Sign Consistency
    pos_frac = (weights_stack > 0).mean(axis=0)
    neg_frac = (weights_stack < 0).mean(axis=0)
    sign_consistency = np.maximum(pos_frac, neg_frac)

    # Dominant Direction (+1 or -1)
    dominant_sign = np.where(pos_frac > neg_frac, 1, -1)

    # 2. Mean Metrics
    mean_weight = weights_stack.mean(axis=0)
    mean_rank = ranks_stack.mean(axis=0)

    # 3. Build DataFrame
    n_rois = len(kept_networks)
    pairs = list(combinations(range(n_rois), 2))

    data = []
    for idx, (i, j) in enumerate(pairs):
        net_i = kept_networks[i]
        net_j = kept_networks[j]
        name_i = kept_names[i]
        name_j = kept_names[j]

        # Sort for symmetry
        nets = sorted([net_i, net_j])
        network_pair = f"{nets[0]}-{nets[1]}"

        data.append({
            'roi_idx_i': kept_indices[i],
            'roi_idx_j': kept_indices[j],
            'roi_name_i': name_i,
            'roi_name_j': name_j,
            'network_i': net_i,
            'network_j': net_j,
            'network_pair': network_pair,
            'mean_weight': mean_weight[idx],
            'mean_rank': mean_rank[idx],
            'sign_consistency': sign_consistency[idx],
            'dominant_sign': dominant_sign[idx],
            'fc_type': fc_type
        })

    df = pd.DataFrame(data)
    return df


def visualize_stability_volcano(df, fc_type, output_dir):
    """Volcano plot: Stability vs Magnitude."""
    plt.figure(figsize=(10, 8))

    # Plot all edges
    sns.scatterplot(
        data=df,
        x='mean_weight',
        y='sign_consistency',
        hue='mean_rank',
        palette='viridis_r',
        alpha=0.5,
        s=20
    )

    # Highlight relevant edges
    relevant = df[(df['sign_consistency'] >= 0.8) & (df['mean_rank'] <= 500)]
    plt.scatter(relevant['mean_weight'], relevant['sign_consistency'],
                color='red', s=5, label='Top Stable')

    plt.axhline(0.8, color='red', linestyle='--', linewidth=1)

    plt.title(f"Edge Stability Analysis - {fc_type.upper()} (Subsampling n={len(df)})")
    plt.xlabel("Mean Weight (Direction)")
    plt.ylabel("Sign Consistency")
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / f"stability_volcano_{fc_type}.png", dpi=300)
    plt.close()


def visualize_network_matrix_lower(edge_df, metric, title, filename, fmt=".1f"):
    """Create 7x7 heatmap (Lower Triangle Only)."""
    agg = edge_df.groupby(['network_i', 'network_j'])[metric].sum().reset_index()

    networks = ['Vis', 'SomMot', 'DorsAttn', 'VentAttn', 'Limbic', 'Cont', 'Default']
    matrix = np.zeros((7, 7))
    mask = np.zeros((7, 7), dtype=bool)

    for _, row in agg.iterrows():
        n1, n2 = row['network_i'], row['network_j']
        if n1 in networks and n2 in networks:
            i, j = networks.index(n1), networks.index(n2)
            matrix[i, j] = row[metric]
            matrix[j, i] = row[metric]

    mask[np.triu_indices_from(mask, k=1)] = True

    plt.figure(figsize=(9, 8))
    sns.heatmap(matrix, xticklabels=networks, yticklabels=networks,
                cmap="RdBu_r", center=0, annot=True, fmt=fmt,
                square=True, linewidths=.5, mask=mask)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def visualize_fc_comparison(df_corr, df_tang, output_dir):
    """Compare stable edges between corr and tangent."""
    # Filter relevant edges
    rel_corr = set(df_corr[(df_corr['sign_consistency'] >= 0.8) &
                           (df_corr['mean_rank'] <= 500)].index)
    rel_tang = set(df_tang[(df_tang['sign_consistency'] >= 0.8) &
                           (df_tang['mean_rank'] <= 500)].index)

    # Create comparison scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(df_corr['mean_weight'], df_tang['mean_weight'], alpha=0.3, s=10)
    plt.xlabel('Mean Weight (Correlation)')
    plt.ylabel('Mean Weight (Tangent)')
    plt.title('Edge Weights: Correlation vs Tangent')

    # Add diagonal
    lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'r--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "fc_comparison_scatter.png", dpi=300)
    plt.close()

    # Print overlap statistics
    overlap = rel_corr & rel_tang
    print(f"\nFC Type Comparison:")
    print(f"  Stable edges (corr): {len(rel_corr)}")
    print(f"  Stable edges (tang): {len(rel_tang)}")
    print(f"  Overlap: {len(overlap)}")


def main(n_subsamples=N_SUBSAMPLES):
    print("1. Loading labels...")
    labels_df = load_network_labels()
    kept_nets, kept_names, mask, kept_indices = get_masked_network_mapping(labels_df)

    print("2. Loading subject-level data...")
    subjects_data, _ = load_subject_level_data(mask)

    print("3. Precomputing FC (correlation and tangent) for all subjects...")
    fc_corr_close, fc_corr_open, fc_tang_close, fc_tang_open, site_labels = precompute_all_fc(subjects_data)

    # Package FC data for subsampling
    fc_data = {
        'corr': (fc_corr_close, fc_corr_open),
        'tangent': (fc_tang_close, fc_tang_open),
    }

    print("4. Running Subsampling Analysis...")
    results = subsampling_analysis(fc_data, site_labels, n_subsamples)

    print("5. Aggregating Results...")
    dfs = {}
    for fc_type in FC_TYPES:
        df = aggregate_and_filter(
            results[fc_type]['weights'],
            results[fc_type]['ranks'],
            kept_nets, kept_names, kept_indices,
            fc_type
        )
        dfs[fc_type] = df

        # Save full stats
        df.to_csv(OUTPUT_DIR / f"edge_stability_stats_{fc_type}.csv", index=False)

        # Filter and save relevant edges
        relevant_df = df[(df['sign_consistency'] >= 0.8) & (df['mean_rank'] <= 500)].copy()
        print(f"  {fc_type.upper()}: Found {len(relevant_df)} relevant stable edges.")

        # Separate EO and EC
        eo_relevant = relevant_df[relevant_df['mean_weight'] > 0].copy()
        ec_relevant = relevant_df[relevant_df['mean_weight'] < 0].copy()

        eo_relevant.to_csv(OUTPUT_DIR / f"stable_edges_EO_{fc_type}.csv", index=False)
        ec_relevant.to_csv(OUTPUT_DIR / f"stable_edges_EC_{fc_type}.csv", index=False)
        relevant_df.to_csv(OUTPUT_DIR / f"stable_edges_{fc_type}.csv", index=False)

    print("6. Visualizing...")
    for fc_type in FC_TYPES:
        df = dfs[fc_type]
        visualize_stability_volcano(df, fc_type, FIGURES_DIR)

        # Filter relevant
        relevant_df = df[(df['sign_consistency'] >= 0.8) & (df['mean_rank'] <= 500)].copy()
        eo_relevant = relevant_df[relevant_df['mean_weight'] > 0].copy()
        ec_relevant = relevant_df[relevant_df['mean_weight'] < 0].copy()

        if not eo_relevant.empty:
            eo_relevant['count'] = 1
            visualize_network_matrix_lower(
                eo_relevant, 'count',
                f"Count of Stable Edges Favoring Eyes Open ({fc_type.upper()})",
                FIGURES_DIR / f"heatmap_stable_EO_count_{fc_type}.png",
                fmt=".0f"
            )
            eo_relevant['abs_weight'] = eo_relevant['mean_weight'].abs()
            visualize_network_matrix_lower(
                eo_relevant, 'abs_weight',
                f"Total Importance Favoring Eyes Open ({fc_type.upper()})",
                FIGURES_DIR / f"heatmap_stable_EO_weight_{fc_type}.png",
                fmt=".2f"
            )

        if not ec_relevant.empty:
            ec_relevant['count'] = 1
            visualize_network_matrix_lower(
                ec_relevant, 'count',
                f"Count of Stable Edges Favoring Eyes Closed ({fc_type.upper()})",
                FIGURES_DIR / f"heatmap_stable_EC_count_{fc_type}.png",
                fmt=".0f"
            )
            ec_relevant['abs_weight'] = ec_relevant['mean_weight'].abs()
            visualize_network_matrix_lower(
                ec_relevant, 'abs_weight',
                f"Total Importance Favoring Eyes Closed ({fc_type.upper()})",
                FIGURES_DIR / f"heatmap_stable_EC_weight_{fc_type}.png",
                fmt=".2f"
            )

    # FC comparison
    visualize_fc_comparison(dfs['corr'], dfs['tangent'], FIGURES_DIR)

    # --- Print Summary Conclusions ---
    print("\n--- CONCLUSIONS: Top Network Pairs ---")

    def print_top_pairs(sub_df, label, fc_type):
        if sub_df.empty:
            print(f"\n{label} ({fc_type}): None found.")
            return
        agg = sub_df.groupby('network_pair')['mean_weight'].apply(
            lambda x: x.abs().sum()
        ).sort_values(ascending=False)
        print(f"\n{label} ({fc_type}, by Total Importance):")
        for pair, val in agg.head(5).items():
            count = len(sub_df[sub_df['network_pair'] == pair])
            print(f"  - {pair}: SumWeight={val:.3f}, Count={count}")

    for fc_type in FC_TYPES:
        df = dfs[fc_type]
        relevant_df = df[(df['sign_consistency'] >= 0.8) & (df['mean_rank'] <= 500)]
        eo_relevant = relevant_df[relevant_df['mean_weight'] > 0]
        ec_relevant = relevant_df[relevant_df['mean_weight'] < 0]

        print_top_pairs(eo_relevant, "Top EO-Dominant Pairs", fc_type)
        print_top_pairs(ec_relevant, "Top EC-Dominant Pairs", fc_type)

    print(f"\nDone! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classification coefficient interpretation via subsampling")
    parser.add_argument("--n-subsamples", type=int, default=N_SUBSAMPLES,
                        help="Number of subsampling iterations")
    args = parser.parse_args()

    main(n_subsamples=args.n_subsamples)
