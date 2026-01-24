import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Strategy map matching the publication image labels
STRATEGY_MAP_PUB = {
    '1': '24P',
    '2': 'aCompCor+12P',
    '3': 'aCompCor50+12P',
    '4': 'aCompCor+24P',
    '5': 'aCompCor50+24P',
    '6': 'a/tCompCor50+24P',
    'AROMA_aggr': 'AROMA Aggressive',
    'AROMA_nonaggr': 'AROMA Non-Aggressive'
}

# FC type map matching publication titles
FC_NAME_MAP = {
    'corr': 'Pearson Correlation',
    'partial': 'Partial Correlation',
    'glasso': 'Regularized Partial Correlation',
    'tangent': 'Tangent Correlation'
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
    
    # Filter for Logistic Regression
    plot_df = full_df[full_df['model'] == 'logreg'].copy()
    
    # Create the exact labels used in the publication (Strategy + GSR)
    def get_pub_label(row):
        strat = STRATEGY_MAP_PUB.get(str(row['strategy']), str(row['strategy']))
        if row['gsr'] == 'GSR':
            return f"{strat} GSR"
        return strat

    plot_df['strategy_gsr'] = plot_df.apply(get_pub_label, axis=1)
    
    # Define the exact order from the publication image (top to bottom)
    # The image has GSR variants on top, then non-GSR
    # Order: a/t...GSR -> 24P GSR -> a/t... -> 24P
    base_order = [
        'a/tCompCor50+24P', 'aCompCor50+24P', 'aCompCor50+12P', 
        'aCompCor+24P', 'aCompCor+12P', '24P',
        'AROMA Aggressive', 'AROMA Non-Aggressive' # Adding AROMA at the bottom of each block
    ]
    
    order = [f"{s} GSR" for s in base_order] + base_order
    
    # Filter out entries that might not be in our data but are in our 'order' list
    actual_labels = plot_df['strategy_gsr'].unique()
    final_order = [o for s in order if (o := s) in actual_labels]

    # Color palette (Set3 or custom to match the colorful nature of the original)
    palette = sns.color_palette("husl", len(final_order))
    color_dict = dict(zip(final_order, palette))

    # Define order
    atlases = ['AAL', 'Schaefer200', 'Brainnetome', 'HCPex']
    fc_types = ['corr', 'partial', 'glasso', 'tangent']
    
    metrics = [
        ('test_acc', 'Accuracy', 'few_shot_publication_accuracy.png'),
        ('test_auc', 'ROC-AUC', 'few_shot_publication_auc.png')
    ]
    
    for col_metric, title_suffix, filename in metrics:
        print(f"Plotting {title_suffix}...")
        
        # Set up the figure
        fig, axes = plt.subplots(len(fc_types), len(atlases), figsize=(20, 24), sharex=True)
        
        for row_idx, fc in enumerate(fc_types):
            # Add big red title for each FC section
            fig.text(0.5, 0.92 - (row_idx * 0.215), FC_NAME_MAP[fc], 
                     ha='center', fontsize=18, color='firebrick', fontweight='bold')
            
            for col_idx, atlas in enumerate(atlases):
                ax = axes[row_idx, col_idx]
                
                # Subset data
                sub_df = plot_df[(plot_df['fc_type'] == fc) & (plot_df['atlas'] == atlas)]
                
                if sub_df.empty:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                    continue
                    
                sns.boxplot(
                    data=sub_df, x=col_metric, y='strategy_gsr',
                    order=final_order, palette=color_dict,
                    ax=ax, linewidth=1.5, fliersize=4,
                    showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"4"},
                    hue='strategy_gsr', legend=False # Added hue to fix future warning
                )
                
                # Subplot titles (only for top row)
                if row_idx == 0:
                    ax.set_title(atlas, fontsize=14)
                else:
                    ax.set_title("")
                    
                # Remove labels except for edges
                ax.set_xlabel("")
                if col_idx == 0:
                    ax.set_ylabel("") 
                else:
                    ax.set_ylabel("")
                    ax.set_yticklabels([])
                
                # Dynamic X limits
                if title_suffix == 'Accuracy':
                    ax.set_xlim(0.5, 0.9)
                    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
                else:
                    ax.set_xlim(0.5, 1.0)
                    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        # Shared X label at bottom
        fig.text(0.5, 0.08, title_suffix, ha='center', fontsize=16)
        
        # Overall Title
        plt.suptitle(f"Few-Shot Adaptation Performance ({title_suffix}) for Logistic Regression\nTargeting IHB RAS data (50 random splits)", 
                     fontsize=22, y=0.96)
        
        plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.1, wspace=0.1, hspace=0.3)
        
        output_path = Path("results/figures") / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Publication-style plot saved to: {output_path}")

if __name__ == "__main__":
    main()
