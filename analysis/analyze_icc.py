import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import statsmodels.api as sm
from statsmodels.formula.api import ols

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

def perform_anova(df, output_path, metrics=['icc31_mean_masked', 'icc21_mean_masked', 'icc11_mean_masked']):
    """
    Performs an N-way ANOVA to determine the relative importance of factors for multiple ICC metrics.
    Saves results to a Markdown file.
    """
    with open(output_path, 'w') as f:
        f.write(f"# ICC Factor Importance Analysis\n\n")
        f.write(f"**Date:** {pd.Timestamp.now()}\n\n")
        
        for metric in metrics:
            print(f"\n--- Statistical Analysis (ANOVA) for {metric} ---")
            
            # Rename columns for formula compatibility
            data = df.rename(columns={
                'strategy': 'Strategy',
                'gsr': 'GSR',
                'fc_type': 'FCType',
                'atlas': 'Atlas',
                metric: 'ICC'
            })
            
            # Fit OLS model
            model = ols('ICC ~ C(Strategy) + C(GSR) + C(FCType) + C(Atlas)', data=data).fit()
            
            # Type 2 ANOVA DataFrame
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Calculate Partial Eta-Squared (n2_p)
            ss_resid = anova_table.loc['Residual', 'sum_sq']
            anova_table['n2_p'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + ss_resid)
            
            # Sort by importance (effect size)
            anova_table = anova_table.sort_values('n2_p', ascending=False)
            
            f.write(f"## Analysis for {metric.split('_')[0].upper()}\n\n")
            f.write("### ANOVA Results (Effect Size Ranking)\n")
            f.write(anova_table[['n2_p', 'F', 'PR(>F)']].to_markdown())
            
            f.write("\n\n### OLS Regression Summary\n")
            f.write("```text\n")
            f.write(model.summary().as_text())
            f.write("\n```\n\n")

    print(f"\nANOVA and OLS results saved to: {output_path}")

def save_pipeline_ranking(df, output_path, metrics=['icc31_mean_masked', 'icc21_mean_masked', 'icc11_mean_masked'], top_n=20):
    """
    Saves a ranking of the best individual pipelines based on the primary metric (ICC 3,1).
    """
    print(f"\n--- Generating Pipeline Ranking ---")
    
    primary_metric = metrics[0]
    
    # Select relevant columns
    cols = ['atlas', 'strategy', 'gsr', 'fc_type'] + metrics
    ranking = df[cols].sort_values(primary_metric, ascending=False).head(top_n)
    
    # Map strategy names
    ranking['strategy_full'] = ranking['strategy'].map(STRATEGY_MAP)
    
    # Reorder for display
    display_cols = ['atlas', 'strategy_full', 'gsr', 'fc_type'] + metrics
    
    # Append to the existing markdown report
    with open(output_path, 'a') as f:
        f.write(f"## Top {top_n} Pipelines\n")
        f.write(f"Ranking based on primary metric `{primary_metric}`.\n\n")
        f.write(ranking[display_cols].to_markdown(index=False))
    
    print(f"Pipeline ranking appended to: {output_path}")

def analyze_icc_summary(summary_path):
    print(f"--- ICC Summary Analysis: {summary_path} ---")
    df = pd.read_csv(summary_path)
    df['strategy'] = df['strategy'].astype(str)
    
    # Filter out HCPex
    print("Filtering out 'HCPex' atlas...")
    df = df[df['atlas'] != 'HCPex']
    
    metrics = ['icc31_mean_masked', 'icc21_mean_masked', 'icc11_mean_masked']
    
    # Perform Statistical Analysis and Save
    output_md = Path("results/icc_results/icc_anova_results.md")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    
    perform_anova(df, output_md, metrics)
    save_pipeline_ranking(df, output_md, metrics, top_n=20)
    
    # Print Quick Summaries to Console
    for metric in metrics:
        print(f"\n=== Summary for {metric} ===")
        strat_results = df.groupby('strategy')[metric].mean().reset_index()
        strat_results['strategy_full'] = strat_results['strategy'].map(STRATEGY_MAP)
        print(strat_results.sort_values(metric, ascending=False)[['strategy_full', metric]].to_string(index=False))

        fc_rank = df.groupby('fc_type')[metric].mean().sort_values(ascending=False)
        print(f"\nMean Masked ICC by FC Type:")
        print(fc_rank)

def load_icc_edgewise(edgewise_dir, atlas, fc_types=['corr', 'pc', 'tang', 'glasso']):
    base_path = Path(edgewise_dir) / atlas
    files = list(base_path.glob("*_edgewise_icc_all.pkl"))
    data_rows = []

    FC_NAME_MAP = {
        'corr': 'Pearson Correlation',
        'pc': 'Partial Correlation',
        'tang': 'Tangent Space',
        'glasso': 'Graphical Lasso'
    }

    import re

    for p_all in files:
        # robust parsing using regex
        # Filename example: AAL_strategy-AROMA_nonaggr_GSR_edgewise_icc_all.pkl
        # or Brainnetome_strategy-1_noGSR_edgewise_icc_all.pkl
        
        match = re.search(r"strategy-(.+?)_(GSR|noGSR)_edgewise", p_all.name)
        if not match:
            print(f"Skipping file (naming pattern mismatch): {p_all.name}")
            continue
            
        strategy = match.group(1)
        gsr = match.group(2)
        
        p_masked = base_path / p_all.name.replace("_all.pkl", "_masked.pkl")
        if not p_masked.exists(): continue

        with open(p_all, 'rb') as f: data_all = pickle.load(f)
        with open(p_masked, 'rb') as f: data_masked = pickle.load(f)

        for fc in fc_types:
            if fc not in data_all: continue
            
            # Map internal fc name to display name
            fc_display = FC_NAME_MAP.get(fc, fc)
            
            v_all = np.array(data_all[fc]['icc31']).flatten()
            v_mask = np.array(data_masked[fc]['icc']['icc31']).flatten()
            
            full_name = STRATEGY_MAP.get(strategy, strategy)
            label = f"{full_name}\n({gsr})"
            
            # Subsample for plotting performance if needed (e.g. 5000 points)
            # Brainnetome ~30k edges. 30k * 16 pipelines * 4 FCs = ~2M points. 
            # Violin plot handles this density, but DataFrame creation might be slow.
            # We'll use all for accuracy.
            
            for v in v_all[~np.isnan(v_all)]:
                data_rows.append({'Strategy': label, 'ICC': v, 'Mask': 'Unmasked', 'FC Type': fc_display})
            for v in v_mask[~np.isnan(v_mask)]:
                data_rows.append({'Strategy': label, 'ICC': v, 'Mask': 'Masked', 'FC Type': fc_display})

    return pd.DataFrame(data_rows)

def main():
    icc_summary = "results/icc_results/icc_summary_full.csv"
    edgewise_dir = "results/icc_results/edgewise"
    
    analyze_icc_summary(icc_summary)
    
    # Include HCPex back in the plotting loop
    atlases = ['AAL', 'Schaefer200', 'Brainnetome', 'HCPex']
    
    for atlas in atlases:
        out = Path(f"results/figures/icc_violin_{atlas}_all_fc.png")
        if out.exists():
            print(f"\nPlot already exists, skipping: {out}")
            continue

        print(f"\nProcessing Atlas: {atlas}")
        df = load_icc_edgewise(edgewise_dir, atlas, fc_types=['corr', 'pc', 'tang', 'glasso'])
        
        if not df.empty:
            print(f"Generating multi-row violin plot for {atlas}...")
            sns.set_theme(style='darkgrid', font_scale=1.1)
            
            strategies = sorted(df['Strategy'].unique())
            fc_order = ['Pearson Correlation', 'Partial Correlation', 'Tangent Space', 'Graphical Lasso']
            
            g = sns.catplot(
                data=df, 
                x='Strategy', 
                y='ICC', 
                hue='Mask', 
                row='FC Type',
                kind='violin',
                split=True, 
                inner='quartile', 
                palette='Set2',
                order=strategies,
                row_order=[fc for fc in fc_order if fc in df['FC Type'].unique()],
                height=4, 
                aspect=3,
                sharex=True
            )
            
            g.fig.suptitle(f'ICC Reliability Distribution: Masked vs Unmasked ({atlas})', y=1.02, fontsize=20)
            g.set_axis_labels("", "ICC (3,1)")
            
            for ax in g.axes.flat:
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha('right')
                    
            out = Path(f"results/figures/icc_violin_{atlas}_all_fc.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, bbox_inches='tight')
            print(f"Plot saved to: {out}")
            plt.close('all') # Clear memory
        else:
            print(f"No data found for {atlas}")

if __name__ == "__main__":
    main()