import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from data_utils.fc import ConnectomeTransformer
from data_utils.timeseries import load_ihb_coverage_mask

def main():
    data_root = Path("~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data").expanduser()
    ts_dir = data_root / "timeseries_ihb" / "AAL"
    
    strategies = ["1", "2", "3", "4", "5", "6", "AROMA_aggr", "AROMA_nonaggr"]
    gsr_options = ["noGSR", "GSR"]
    conditions = ["close", "open"]
    
    # Load coverage mask to ensure consistency (106 ROIs)
    mask = load_ihb_coverage_mask("AAL", data_path=str(data_root), threshold=0.1)
    
    labels = []
    print("Loading and computing Pearson FC matrices...")
    
    # List to store (n_pipelines, n_subjects, n_edges)
    all_pipeline_data = []
    
    for cond in conditions:
        for strat in strategies:
            for gsr in gsr_options:
                label = f"{cond}_{strat}_{gsr}"
                
                filename = f"ihb_{cond}_AAL_strategy-{strat}_{gsr}.npy"
                path = ts_dir / filename
                if path.exists():
                    # Load timeseries: (subjects, timepoints, rois)
                    ts = np.load(path)
                    # Apply mask: (subjects, timepoints, 106)
                    ts_masked = ts[:, :, mask]
                    # Compute Pearson Correlation
                    transformer = ConnectomeTransformer(kind='corr', vectorize=True)
                    fc_corr = transformer.fit_transform(ts_masked)
                    all_pipeline_data.append(fc_corr)
                    labels.append(label)
                else:
                    print(f"Warning: Timeseries file not found: {path}")

    n_pipelines = len(all_pipeline_data)
    if n_pipelines == 0:
        print("No pipelines found.")
        return

    # Convert to array: (pipelines, subjects, edges)
    all_data_arr = np.array(all_pipeline_data)
    n_subs = all_data_arr.shape[1]
    
    print(f"Found {n_pipelines} Pearson correlation configurations.")
    print(f"Computing subject-wise correlations for {n_subs} subjects...")
    
    # Array to store correlation matrices for each subject
    sub_corr_matrices = []
    
    for s in tqdm(range(n_subs), desc="Subjects"):
        # Extract features for this subject across all pipelines: (pipelines, edges)
        sub_features = all_data_arr[:, s, :]
        
        # Compute pairwise correlation between pipelines for this subject
        sub_corr = np.corrcoef(sub_features)
        sub_corr_matrices.append(sub_corr)
        
    # Average across subjects
    avg_corr_matrix = np.mean(sub_corr_matrices, axis=0)
    
    # Visualization
    plt.figure(figsize=(30, 28))
    
    sns.heatmap(avg_corr_matrix, 
                xticklabels=labels, 
                yticklabels=labels, 
                cmap='RdYlBu_r', 
                vmin=0.5, vmax=1.0, 
                annot=True, 
                fmt=".2f",
                annot_kws={"size": 8},
                cbar=False)
    
    plt.title("IHB AAL: Consistency of Pearson Correlation across Denoising Strategies\n(Subject-wise Correlation averaged across subjects)", fontsize=24)
    plt.xticks(rotation=40, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    output_fig = Path("results/figures/pearson_strategy_consistency_heatmap.png")
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_fig}")

if __name__ == "__main__":
    main()
