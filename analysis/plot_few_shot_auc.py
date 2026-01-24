import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

STRATEGY_MAP = {
    '1': '24P',
    '2': 'aCompCor(5)+12P',
    '3': 'aCompCor(50%)+12P',
    '4': 'aCompCor(5)+24P',
    '5': 'aCompCor(50%)+24P',
    '6': 'a/tCompCor(50%)+24P',
    'AROMA_aggr': 'AROMA Aggressive',
    'AROMA_nonaggr': 'AROMA Non-Aggressive'
}

def main():
    results_dir = Path("results/ml")
    all_files = list(results_dir.glob("*/few_shot_results.csv"))
    
    if not all_files:
        print("No few_shot_results.csv files found.")
        return

    print(f"Found {len(all_files)} files. Aggregating...")
    
    dfs = []
    for f in tqdm(all_files, desc="Loading data"):
        df = pd.read_csv(f)
        dfs.append(df)
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # 1. Filter for Logistic Regression ('logreg')
    plot_df = full_df[full_df['model'] == 'logreg'].copy()
    
    if plot_df.empty:
        print("No data found for model='logreg'.")
        return

    # 2. Map strategy names and normalize FC type labels
    plot_df['strategy_name'] = plot_df['strategy'].astype(str).map(STRATEGY_MAP)
    fc_name_map = {
        'corr': 'Pearson Correlation',
        'partial': 'Partial Correlation',
        'tangent': 'Tangent Space',
        'glasso': 'Graphical Lasso'
    }
    plot_df['fc_type_name'] = plot_df['fc_type'].map(fc_name_map)
    
    # Sort strategies for consistent Y-axis
    order = list(STRATEGY_MAP.values())
    
    sns.set_theme(style="whitegrid")
    
    metrics = [
        ('test_auc', 'AUC', 'few_shot_performance_auc_facet.png'),
        ('test_acc', 'Accuracy', 'few_shot_performance_acc_facet.png')
    ]
    
    for col_metric, title_suffix, filename in metrics:
        print(f"Plotting {title_suffix}...")
        
        # Create FacetGrid
        g = sns.catplot(
            data=plot_df, 
            x=col_metric, y="strategy_name", 
            hue="gsr", 
            col="atlas", row="fc_type_name",
            kind="box", 
            order=order,
            height=5, aspect=1.2,
            palette="muted",
            margin_titles=True,
            sharex=True,
            showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"4"}
        )
        
        # Remove redundant y-axis labels
        g.set_axis_labels(title_suffix, "Denoising Strategy")
        g.set_titles(col_template="{col_name} Atlas", row_template="{row_name}")
        g.add_legend(title="GSR Option")
        
        # Aesthetic cleanup: only show Y label on leftmost
        for ax in g.axes.flat:
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel("Strategy")
            else:
                ax.set_ylabel("")
        
        plt.subplots_adjust(top=0.95)
        g.fig.suptitle(f"Few-Shot Adaptation Performance: {title_suffix} Across All Pipelines (Logistic Regression)", fontsize=20)
        
        output_path = Path("results/figures") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        g.savefig(output_path, dpi=300, bbox_inches='tight')
        
        print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
