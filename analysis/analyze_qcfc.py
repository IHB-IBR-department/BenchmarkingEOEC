import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

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

def perform_qcfc_anova(df, output_path):
    """
    Performs ANOVA on QC-FC results, treating Condition and Site as factors.
    """
    print("\n--- Statistical Analysis (ANOVA) for QC-FC ---")
    
    # 1. Prepare Long Format Data
    # We want rows for 'Close' and 'Open' for each pipeline
    
    # Average China close sessions for a single 'Close' value per pipeline
    df['close_avg'] = df.apply(
        lambda x: np.nanmean([x['close_mean_abs_r'], x['close_s1_mean_abs_r']]) 
        if x['site'] == 'china' else x['close_mean_abs_r'], axis=1
    )
    
    # Melt to long format
    id_vars = ['site', 'atlas', 'strategy', 'gsr', 'fc_type']
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=['close_avg', 'open_mean_abs_r'],
        var_name='Condition',
        value_name='QC_FC'
    )
    long_df['Condition'] = long_df['Condition'].map({'close_avg': 'Close', 'open_mean_abs_r': 'Open'})
    
    # Drop NaNs
    long_df = long_df.dropna(subset=['QC_FC'])
    
    # Rename columns for formula
    data = long_df.rename(columns={
        'strategy': 'Strategy',
        'gsr': 'GSR',
        'fc_type': 'FCType',
        'atlas': 'Atlas',
        'site': 'Site'
    })
    
    # 2. Fit OLS model
    formula = 'QC_FC ~ C(Strategy) + C(GSR) + C(FCType) + C(Atlas) + C(Site) + C(Condition)'
    model = ols(formula, data=data).fit()
    
    # ANOVA Table
    anova_table = sm.stats.anova_lm(model, typ=2)
    ss_resid = anova_table.loc['Residual', 'sum_sq']
    anova_table['n2_p'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + ss_resid)
    anova_table = anova_table.sort_values('n2_p', ascending=False)
    
    # 3. Save to Markdown
    with open(output_path, 'w') as f:
        f.write("# QC-FC Factor Importance Analysis\n\n")
        f.write(f"**Metric:** Mean Absolute Correlation with Motion (|r|)\n")
        f.write(f"**Date:** {pd.Timestamp.now()}\n\n")
        
        f.write("## ANOVA Results (Effect Size Ranking)\n")
        f.write("Factors are ranked by Partial Eta-Squared ($\\eta_p^2$). Lower QC-FC is better.\n\n")
        f.write(anova_table[['n2_p', 'F', 'PR(>F)']].to_markdown())
        
        f.write("\n\n## OLS Regression Summary\n")
        f.write("```text\n")
        f.write(model.summary().as_text())
        f.write("\n```\n")

    print(f"ANOVA results saved to: {output_path}")
    return long_df

def save_qcfc_ranking(df, output_path, top_n=20):
    """
    Saves ranking of best pipelines for Close and Open conditions.
    """
    with open(output_path, 'a') as f:
        for cond in ['Close', 'Open']:
            f.write(f"\n\n## Top {top_n} Pipelines for {cond} Eyes\n")
            f.write(f"Ranking based on lowest absolute correlation with motion.\n\n")
            
            cond_df = df[df['Condition'] == cond].copy()
            cond_df['strategy_full'] = cond_df['strategy'].map(STRATEGY_MAP)
            
            ranking = cond_df.sort_values('QC_FC').head(top_n)
            cols = ['site', 'atlas', 'strategy_full', 'gsr', 'fc_type', 'QC_FC']
            f.write(ranking[cols].to_markdown(index=False))

def analyze_qcfc_summary(summary_path):
    print(f"--- QC-FC Summary Analysis: {summary_path} ---")
    df = pd.read_csv(summary_path)
    df['strategy'] = df['strategy'].astype(str)
    
    # Exclude HCPex for consistency with ICC report
    print("Filtering out 'HCPex' atlas...")
    df = df[df['atlas'] != 'HCPex']
    
    output_md = Path("results/qcfc/qcfc_anova_results.md")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. ANOVA Analysis
    long_df = perform_qcfc_anova(df, output_md)
    
    # 2. Ranking
    save_qcfc_ranking(long_df, output_md)
    
    # 3. Console Summary
    metrics_to_analyze = [
        ('Close Sessions', 'close_avg'),
        ('Open Sessions', 'open_mean_abs_r')
    ]
    
    # Need to add close_avg to df for console print
    df['close_avg'] = df.apply(
        lambda x: np.nanmean([x['close_mean_abs_r'], x['close_s1_mean_abs_r']]) 
        if x['site'] == 'china' else x['close_mean_abs_r'], axis=1
    )

    for condition_label, col_name in metrics_to_analyze:
        print(f"\n=== {condition_label} QC-FC Analysis ===")
        strat_results = df.groupby('strategy')[col_name].mean().reset_index()
        strat_results['strategy_full'] = strat_results['strategy'].map(STRATEGY_MAP)
        print(strat_results.sort_values(col_name)[['strategy_full', col_name]].to_string(index=False))


def load_qcfc_edgewise(edgewise_root, site, atlas, fc_types=['corr', 'pc', 'tang', 'glasso']):
    """
    Loads edge-wise correlations from the site/atlas subfolders for multiple FC types.
    Returns DataFrame: [Strategy, QC-FC, GSR, FC Type]
    """
    base_path = Path(edgewise_root) / site / atlas
    if not base_path.exists():
        print(f"Edgewise directory not found: {base_path}")
        return pd.DataFrame()

    FC_NAME_MAP = {
        'corr': 'Pearson Correlation',
        'partial': 'Partial Correlation', # note: qcfc code used 'partial' in filename
        'tangent': 'Tangent Space',       # note: qcfc code used 'tangent'
        'glasso': 'Graphical Lasso'
    }
    
    # Need to handle fc_type mapping for filenames. 
    # analyze_qcfc used 'corr', 'partial', 'tangent', 'glasso' in output filenames
    
    data_rows = []

    # Get all csv files
    files = list(base_path.glob("*_edge_correlations.csv"))

    import re

    for f in files:
        # Filename: {site}_{atlas}_strategy-{strategy}_{gsr}_{fc_type}_edge_correlations.csv
        # Example: china_Brainnetome_strategy-AROMA_nonaggr_GSR_glasso_edge_correlations.csv
        
        match = re.search(r"strategy-(.+?)_(GSR|noGSR)_([a-zA-Z]+)_edge", f.name)
        if not match:
            # Try alternate pattern if fc_type has underscores or different structure
            # But standard is simple. Let's debug if needed.
            # Fallback to simple split? No, regex is better.
            print(f"Skipping file (naming pattern mismatch): {f.name}")
            continue
            
        strategy = match.group(1)
        gsr = match.group(2)
        fc_raw = match.group(3)
        
        # Verify normalization
        if fc_raw == 'pc': fc_raw = 'partial' 
        if fc_raw == 'tang': fc_raw = 'tangent'
        
        fc_display = FC_NAME_MAP.get(fc_raw, fc_raw.capitalize())
        full_name = STRATEGY_MAP.get(strategy, strategy)
        
        df_edges = pd.read_csv(f)
        if 'close' not in df_edges.columns: continue
        
        corrs = df_edges['close'].values
        
        for c in corrs[~np.isnan(corrs)]:
            data_rows.append({
                'Strategy': full_name,
                'QC-FC': np.abs(c), 
                'GSR': gsr,
                'FC Type': fc_display
            })

    return pd.DataFrame(data_rows)

def main():
    qcfc_summary = "results/qcfc/qc_fc_full.csv"
    edgewise_root = "results/qcfc/edge_correlations"
    
    analyze_qcfc_summary(qcfc_summary)
    
    atlases = ['AAL', 'Schaefer200', 'Brainnetome', 'HCPex']
    site = "china"
    
    for atlas in atlases:
        out = Path(f"results/figures/qcfc_violin_{site}_{atlas}_all_fc.png")
        if out.exists():
            print(f"\nPlot already exists, skipping: {out}")
            continue

        print(f"\nProcessing Atlas: {atlas}")
        df = load_qcfc_edgewise(edgewise_root, site, atlas)
        
        if not df.empty:
            print(f"Generating multi-row QC-FC violin plot for {atlas}...")
            sns.set_theme(style='darkgrid', font_scale=1.1)
            
            strategies = sorted(df['Strategy'].unique())
            fc_order = ['Pearson Correlation', 'Partial Correlation', 'Tangent Space', 'Graphical Lasso']

            g = sns.catplot(
                data=df, 
                x='Strategy', 
                y='QC-FC', 
                hue='GSR', 
                row='FC Type',
                kind='violin',
                split=True, 
                inner='quartile', 
                palette='husl',
                order=strategies,
                row_order=[fc for fc in fc_order if fc in df['FC Type'].unique()],
                height=4, 
                aspect=3,
                sharex=True
            )
            
            g.fig.suptitle(f'QC-FC |r| Distribution: GSR vs noGSR ({site.upper()}, {atlas})', y=1.02, fontsize=20)
            g.set_axis_labels("", "QC-FC Mean |r|")
            
            for ax in g.axes.flat:
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha('right')
                    
            out = Path(f"results/figures/qcfc_violin_{site}_{atlas}_all_fc.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, bbox_inches='tight')
            print(f"Plot saved to: {out}")
            plt.close('all')
        else:
            print(f"No data found for {atlas}")

if __name__ == "__main__":
    main()
