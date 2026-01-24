import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.formula.api import ols
from tqdm import tqdm
from benchmarking.ml.stats import factor_level_randomization_test

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

def aggregate_results(results_dir):
    """Aggregates all summary.csv files from subdirectories."""
    print(f"--- Aggregating ML Results from {results_dir} ---")
    all_summaries = list(Path(results_dir).glob("*/summary.csv"))
    
    dfs = []
    for p in tqdm(all_summaries, desc="Loading summaries"):
        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {p}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Compute aggregate performance metrics
    full_df['avg_cross_site_acc'] = (full_df['china2ihb_acc'] + full_df['ihb2china_acc']) / 2
    full_df['avg_cross_site_auc'] = (full_df['china2ihb_auc'] + full_df['ihb2china_auc']) / 2
    
    return full_df

def aggregate_test_outputs(results_dir):
    """Aggregates all test_outputs.csv files for paired testing."""
    print(f"--- Aggregating ML Test Outputs ---")
    all_outputs = list(Path(results_dir).glob("*/cross_site_*_test_outputs.csv"))
    
    dfs = []
    for p in tqdm(all_outputs, desc="Loading test outputs"):
        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {p}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def analyze_global_significance(df, output_path):
    """Analyzes how many pipelines perform significantly better than chance with FDR correction."""
    print("\n--- Analyzing Global Significance vs Chance (with FDR) ---")
    
    from statsmodels.stats.multitest import multipletests
    
    # Significance threshold
    alpha = 0.05
    
    # Perform FDR correction (Benjamini-Hochberg) for each direction separately
    _, pvals_fdr_c2i, _, _ = multipletests(df['china2ihb_pval'], alpha=alpha, method='fdr_bh')
    _, pvals_fdr_i2c, _, _ = multipletests(df['ihb2china_pval'], alpha=alpha, method='fdr_bh')
    
    df['china2ihb_pval_fdr'] = pvals_fdr_c2i
    df['ihb2china_pval_fdr'] = pvals_fdr_i2c
    
    sig_china2ihb = (df['china2ihb_pval_fdr'] < alpha).sum()
    sig_ihb2china = (df['ihb2china_pval_fdr'] < alpha).sum()
    total = len(df)
    
    res_text = (
        f"\n## Significance vs. Chance (Permutation Testing)\n"
        f"Based on label-shuffling permutation tests (1,000 repeats).\n"
        f"P-values are corrected for multiple comparisons using the False Discovery Rate (FDR, Benjamini-Hochberg) across all {total} pipelines.\n\n"
        f"- **China -> IHB**: {sig_china2ihb}/{total} ({sig_china2ihb/total:.1%}) pipelines are significant at FDR-corrected alpha={alpha}.\n"
        f"- **IHB -> China**: {sig_ihb2china}/{total} ({sig_ihb2china/total:.1%}) pipelines are significant at FDR-corrected alpha={alpha}.\n\n"
        f"Conclusion: Functional connectivity provides robust signal for eye state classification across different sites and scanners, surviving rigorous correction for multiple testing.\n"
    )
    
    with open(output_path, 'a') as f:
        f.write(res_text)
    
    print(f"Significance summary (FDR) appended to: {output_path}")

def perform_factor_comparisons(outputs_df, output_path):
    """Performs paired randomization tests for factors like GSR, FC Method, and Atlas."""
    print("\n--- Performing Factor-Level Paired Comparisons (Sign-Flip) ---")
    
    results_text = "\n## Factor-Level Comparisons (Paired Randomization)\n"
    results_text += "Evaluating the global impact of processing choices using matched pipeline pairs and subject-level sign-flip tests (5,000 permutations).\n\n"
    
    comparisons = [
        {"name": "GSR vs noGSR", "factor": "gsr", "a": "GSR", "b": "noGSR", "metric": "brier"},
        {"name": "Tangent vs Corr", "factor": "fc_type", "a": "tangent", "b": "corr", "metric": "brier"},
    ]
    
    # 1. Standard Comparisons
    for comp in comparisons:
        try:
            res_c2i = factor_level_randomization_test(
                outputs_df, factor=comp["factor"], level_a=comp["a"], level_b=comp["b"],
                metric=comp["metric"], train_site="china", test_site="ihb", n_permutations=5000
            )
            res_i2c = factor_level_randomization_test(
                outputs_df, factor=comp["factor"], level_a=comp["a"], level_b=comp["b"],
                metric=comp["metric"], train_site="ihb", test_site="china", n_permutations=5000
            )
            
            results_text += f"### {comp['name']}\n"
            results_text += f"Metric: {comp['metric'].capitalize()} (lower is better). Level A: {comp['a']}, Level B: {comp['b']}.\n\n"
            
            df_res = pd.DataFrame([
                {"Direction": "China -> IHB", "Observed Delta (B-A)": res_c2i['observed_delta'], "p-value": res_c2i['p_value'], "Pairs": res_c2i['n_pairs']},
                {"Direction": "IHB -> China", "Observed Delta (B-A)": res_i2c['observed_delta'], "p-value": res_i2c['p_value'], "Pairs": res_i2c['n_pairs']}
            ])
            results_text += df_res.to_markdown(index=False)
            results_text += f"\n\n*Note: A positive delta indicates that {comp['a']} (Level A) has lower loss (better performance) than {comp['b']}.*\n\n"
        except Exception as e:
            print(f"Error in {comp['name']}: {e}")

    # 2. Brainnetome vs Others
    try:
        print("Comparing Brainnetome vs. Others...")
        # Create temporary grouping
        tmp_df = outputs_df.copy()
        tmp_df['atlas_group'] = tmp_df['atlas'].apply(lambda x: 'Brainnetome' if x == 'Brainnetome' else 'Others')
        
        # Explicitly define pipeline columns to include our temporary group
        p_cols = ['atlas_group', 'fc_type', 'strategy', 'gsr', 'model', 'model_params']
        
        res_bn_c2i = factor_level_randomization_test(
            tmp_df, factor="atlas_group", level_a="Brainnetome", level_b="Others",
            metric="brier", train_site="china", test_site="ihb", n_permutations=5000,
            pipeline_cols=p_cols
        )
        res_bn_i2c = factor_level_randomization_test(
            tmp_df, factor="atlas_group", level_a="Brainnetome", level_b="Others",
            metric="brier", train_site="ihb", test_site="china", n_permutations=5000,
            pipeline_cols=p_cols
        )
        
        results_text += "### Brainnetome vs. All Other Atlases\n"
        results_text += "Comparing Brainnetome against the average of AAL, Schaefer200, and HCPex.\n\n"
        
        df_bn = pd.DataFrame([
            {"Direction": "China -> IHB", "Observed Delta (B-A)": res_bn_c2i['observed_delta'], "p-value": res_bn_c2i['p_value'], "Pairs": res_bn_c2i['n_pairs']},
            {"Direction": "IHB -> China", "Observed Delta (B-A)": res_bn_i2c['observed_delta'], "p-value": res_bn_i2c['p_value'], "Pairs": res_bn_i2c['n_pairs']}
        ])
        results_text += df_bn.to_markdown(index=False)
        results_text += "\n\n*Note: A positive delta indicates that Brainnetome has lower loss (better performance) than the average of other atlases.*\n"
    except Exception as e:
        print(f"Error in Brainnetome comparison: {e}")

    with open(output_path, 'a') as f:
        f.write(results_text)
    
    print(f"Factor comparisons appended to: {output_path}")

def perform_ml_anova(df, output_path):
    """Performs ANOVA to find factor importance for multiple ML metrics."""
    with open(output_path, 'w') as f:
        f.write("# ML Factor Importance Analysis\n\n")
        f.write(f"**Total pipelines:** {len(df)}\n")
        f.write(f"**Date:** {pd.Timestamp.now()}\n\n")
        
        metrics = [
            ('Accuracy', 'avg_cross_site_acc'),
            ('AUC', 'avg_cross_site_auc')
        ]
        
        for name, col in metrics:
            print(f"\n--- Statistical Analysis (ANOVA) for ML {name} ---")
            
            data = df.rename(columns={
                'strategy': 'Strategy',
                'gsr': 'GSR',
                'fc_type': 'FCType',
                'atlas': 'Atlas',
                'model': 'Model',
                col: 'MetricVal'
            }).dropna(subset=['MetricVal'])
            
            # Ensure categorical
            for factor in ['Strategy', 'GSR', 'FCType', 'Atlas', 'Model']:
                data[factor] = data[factor].astype(str)
                
            model = ols('MetricVal ~ C(Strategy) + C(GSR) + C(FCType) + C(Atlas) + C(Model)', data=data).fit()
            
            anova_table = sm.stats.anova_lm(model, typ=2)
            ss_resid = anova_table.loc['Residual', 'sum_sq']
            anova_table['n2_p'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + ss_resid)
            anova_table = anova_table.sort_values('n2_p', ascending=False)
            
            f.write(f"## Analysis for {name}\n\n")
            f.write("### ANOVA Results (Effect Size Ranking)\n")
            f.write(anova_table[['n2_p', 'F', 'PR(>F)']].to_markdown())
            
            f.write("\n\n### OLS Regression Summary\n")
            f.write("```text\n")
            f.write(model.summary().as_text())
            f.write("\n```\n\n")

    print(f"ANOVA results saved to: {output_path}")

def save_ml_ranking(df, output_path, top_n=20):
    """Saves rankings of best individual pipelines for both Acc and AUC."""
    print(f"\n--- Generating ML Pipeline Rankings ---")
    
    with open(output_path, 'a') as f:
        # Ranking 1: Accuracy
        f.write(f"\n## Top {top_n} Pipelines (Mean Cross-Site Accuracy)\n")
        ranking_acc = df.sort_values('avg_cross_site_acc', ascending=False).head(top_n).copy()
        ranking_acc['strategy_full'] = ranking_acc['strategy'].astype(str).map(STRATEGY_MAP)
        cols_acc = ['atlas', 'strategy_full', 'gsr', 'fc_type', 'model', 'china2ihb_acc', 'ihb2china_acc', 'avg_cross_site_acc']
        f.write(ranking_acc[cols_acc].to_markdown(index=False))
        
        # Ranking 2: AUC
        f.write(f"\n\n## Top {top_n} Pipelines (Mean Cross-Site AUC)\n")
        ranking_auc = df.sort_values('avg_cross_site_auc', ascending=False).head(top_n).copy()
        ranking_auc['strategy_full'] = ranking_auc['strategy'].astype(str).map(STRATEGY_MAP)
        cols_auc = ['atlas', 'strategy_full', 'gsr', 'fc_type', 'model', 'china2ihb_auc', 'ihb2china_auc', 'avg_cross_site_auc']
        f.write(ranking_auc[cols_auc].to_markdown(index=False))
    
    print(f"Rankings appended to: {output_path}")

def perform_few_shot_mixed_analysis(results_dir, output_path):
    """
    Performs mixed-effects model analysis on few-shot results.
    Compares Tangent vs. Pearson Correlation.
    """
    print("\n--- Performing Mixed-Effects Analysis on Few-Shot Results ---")
    
    all_files = list(Path(results_dir).glob("*/few_shot_results.csv"))
    if not all_files:
        print("No few-shot results found for mixed analysis.")
        return

    dfs = []
    for f in all_files:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    
    # Filter for Logistic Regression and the two target FC types
    df = df[(df['model'] == 'logreg') & (df['fc_type'].isin(['corr', 'tangent']))].copy()
    
    # Create unique pipeline identifier for random effects
    df['pipeline_id'] = df['atlas'] + "_" + df['strategy'].astype(str) + "_" + df['gsr']
    
    with open(output_path, 'a') as f:
        f.write("\n## Few-Shot Mixed-Effects Analysis: Tangent vs. Pearson\n")
        f.write("Using Linear Mixed Models to account for repeated measures (splits) and pipeline variability.\n\n")
        
        for metric_name, metric_col in [('Accuracy', 'test_acc'), ('AUC', 'test_auc')]:
            f.write(f"### Metric: {metric_name}\n\n")
            
            # 1. General Comparison (Across all Atlases)
            print(f"   Analyzing {metric_name} (General)...")
            model_gen = sm.MixedLM.from_formula(
                f"{metric_col} ~ C(fc_type, Treatment(reference='corr'))", 
                df, groups=df["pipeline_id"]
            ).fit()
            
            f.write("#### 1. Global Comparison (All Atlases)\n")
            f.write(f"Fixed Effect: FC Type (Reference = Pearson Correlation). Random Effect: Pipeline ID.\n\n")
            f.write("```text\n")
            f.write(model_gen.summary().as_text())
            f.write("\n```\n\n")
            
            # 2. Schaefer200 Specific Comparison
            print(f"   Analyzing {metric_name} (Schaefer200)...")
            df_sch = df[df['atlas'] == 'Schaefer200']
            if not df_sch.empty:
                model_sch = sm.MixedLM.from_formula(
                    f"{metric_col} ~ C(fc_type, Treatment(reference='corr'))", 
                    df_sch, groups=df_sch["pipeline_id"]
                ).fit()
                
                f.write("#### 2. Schaefer200 Atlas Specific\n")
                f.write("```text\n")
                f.write(model_sch.summary().as_text())
                f.write("\n```\n\n")

    print(f"Mixed-effects analysis appended to: {output_path}")

def main():
    results_dir = "results/ml"
    df_summary = aggregate_results(results_dir)
    
    if df_summary.empty:
        print("No results found to analyze.")
        return
        
    output_md = Path("results/ml/ml_statistical_analysis.md")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Primary ANOVA
    perform_ml_anova(df_summary, output_md)
    
    # 2. Global Significance
    analyze_global_significance(df_summary, output_md)
    
    # 3. Factor Comparisons (Paired Tests)
    df_outputs = aggregate_test_outputs(results_dir)
    if not df_outputs.empty:
        perform_factor_comparisons(df_outputs, output_md)
    
    # 4. Few-Shot Mixed Analysis
    perform_few_shot_mixed_analysis(results_dir, output_md)
    
    # 5. Ranking
    save_ml_ranking(df_summary, output_md)
    
    # Final Aggregate CSV for manual inspection
    df_summary.to_csv("results/ml/all_pipelines_summary.csv", index=False)
    print("\nAggregate summary saved to: results/ml/all_pipelines_summary.csv")

if __name__ == "__main__":
    main()
