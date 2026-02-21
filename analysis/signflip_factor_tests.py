#!/usr/bin/env python3
"""
Subject-Level Sign-Flip Randomization Tests for Pairwise Factor Comparisons
=============================================================================

Complements the LMM omnibus analysis (lmm_factor_analysis.py) with planned
pairwise comparisons that inherently respect the within-subject design.

For each factor (fc_type, atlas, strategy, gsr), all pairwise level
comparisons are tested using subject-level sign-flip randomization
(5,000 permutations) with FDR correction for multiple comparisons.

Usage:
    source venv/bin/activate && PYTHONPATH=. python analysis/signflip_factor_tests.py
"""

import argparse
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from benchmarking.ml.stats import factor_level_randomization_test


RESULTS_DIR = Path("results/ml")
OUTPUT_DIR = Path("results/ml")

N_PERMUTATIONS = 5000
N_BOOTSTRAP = 5000
ALPHA = 0.05


def load_all_test_outputs(results_dir: Path) -> pd.DataFrame:
    """Load and concatenate all cross-site test_outputs CSVs."""
    all_files = sorted(results_dir.glob("*/cross_site_*_test_outputs.csv"))
    if not all_files:
        raise FileNotFoundError(f"No test_outputs files found in {results_dir}")

    dfs = []
    for f in tqdm(all_files, desc="Loading test outputs"):
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  Warning: skipping {f.name}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} rows from {len(all_files)} files")
    return df


def run_all_pairwise_tests(df: pd.DataFrame, metric: str = "brier") -> pd.DataFrame:
    """
    Run sign-flip randomization tests for all pairwise comparisons
    within each factor, for each cross-site direction.
    """
    # Normalize strategy to string to avoid int/str mismatch
    df = df.copy()
    df["strategy"] = df["strategy"].astype(str)

    factors = {
        "fc_type": sorted(df["fc_type"].unique()),
        "atlas": sorted(df["atlas"].unique()),
        "strategy": sorted(df["strategy"].unique()),
        "gsr": sorted(df["gsr"].unique()),
    }
    directions = [
        ("china", "ihb", "China -> IHB"),
        ("ihb", "china", "IHB -> China"),
    ]

    results = []
    total_tests = sum(
        len(list(combinations(levels, 2))) * len(directions)
        for levels in factors.values()
    )
    pbar = tqdm(total=total_tests, desc="Running sign-flip tests")

    for factor, levels in factors.items():
        for level_a, level_b in combinations(levels, 2):
            for train_site, test_site, dir_label in directions:
                try:
                    res = factor_level_randomization_test(
                        df,
                        factor=factor,
                        level_a=level_a,
                        level_b=level_b,
                        metric=metric,
                        train_site=train_site,
                        test_site=test_site,
                        n_permutations=N_PERMUTATIONS,
                        n_bootstrap=N_BOOTSTRAP,
                    )
                    results.append({
                        "factor": factor,
                        "level_a": str(level_a),
                        "level_b": str(level_b),
                        "direction": dir_label,
                        "train_site": train_site,
                        "test_site": test_site,
                        "observed_delta": res["observed_delta"],
                        "p_value": res["p_value"],
                        "n_subjects": res["n_subjects"],
                        "n_pairs": res["n_pairs"],
                        "ci_lower": res.get("ci_lower", np.nan),
                        "ci_upper": res.get("ci_upper", np.nan),
                    })
                except Exception as e:
                    print(f"\n  Warning: {factor} {level_a} vs {level_b} ({dir_label}): {e}")

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(results)


def apply_fdr_correction(results_df: pd.DataFrame) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction within each factor."""
    results_df = results_df.copy()
    results_df["p_fdr"] = np.nan
    results_df["significant"] = False

    for factor in results_df["factor"].unique():
        mask = results_df["factor"] == factor
        pvals = results_df.loc[mask, "p_value"].values
        _, pvals_corrected, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")
        results_df.loc[mask, "p_fdr"] = pvals_corrected
        results_df.loc[mask, "significant"] = pvals_corrected < ALPHA

    return results_df


def write_report(results_df: pd.DataFrame, output_path: Path, metric: str):
    """Write comprehensive report."""
    with open(output_path, "w") as f:
        f.write("# Subject-Level Sign-Flip Randomization Tests\n\n")
        f.write("Pairwise factor-level comparisons using matched pipeline pairs\n")
        f.write("and subject-level sign-flip randomization tests.\n\n")
        f.write(f"- **Metric:** {metric} (lower is better)\n")
        f.write(f"- **Permutations:** {N_PERMUTATIONS:,}\n")
        f.write(f"- **Bootstrap CIs:** {N_BOOTSTRAP:,} resamples, 95% CI\n")
        f.write(f"- **Multiple comparison correction:** FDR (Benjamini-Hochberg) within each factor\n")
        f.write(f"- **Significance threshold:** alpha = {ALPHA}\n\n")
        f.write("**Interpretation:** Observed Delta = mean(loss_B - loss_A) across subjects.\n")
        f.write("Positive delta → Level A has lower loss (better). Negative delta → Level B has lower loss (better).\n\n")
        f.write("---\n\n")

        for factor in ["fc_type", "atlas", "strategy", "gsr"]:
            factor_df = results_df[results_df["factor"] == factor].copy()
            if factor_df.empty:
                continue

            f.write(f"## Factor: {factor}\n\n")

            n_sig = factor_df["significant"].sum()
            n_total = len(factor_df)
            f.write(f"**{n_sig}/{n_total}** comparisons significant after FDR correction.\n\n")

            # Table per direction
            for direction in factor_df["direction"].unique():
                dir_df = factor_df[factor_df["direction"] == direction].copy()
                dir_df = dir_df.sort_values("p_value")

                f.write(f"### {direction}\n\n")
                f.write("| Level A | Level B | Delta (B-A) | 95% CI | p-value | p-FDR | Sig | N subj | N pairs |\n")
                f.write("|---------|---------|------------:|-------:|--------:|------:|:---:|-------:|--------:|\n")

                for _, row in dir_df.iterrows():
                    ci_str = f"[{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}]"
                    p_str = f"{row['p_value']:.4f}" if row['p_value'] >= 0.0005 else f"{row['p_value']:.2e}"
                    p_fdr_str = f"{row['p_fdr']:.4f}" if row['p_fdr'] >= 0.0005 else f"{row['p_fdr']:.2e}"
                    sig_str = "***" if row['p_fdr'] < 0.001 else ("**" if row['p_fdr'] < 0.01 else ("*" if row['p_fdr'] < 0.05 else ""))
                    better = row['level_a'] if row['observed_delta'] > 0 else row['level_b']

                    f.write(
                        f"| {row['level_a']} | {row['level_b']} "
                        f"| {row['observed_delta']:+.4f} "
                        f"| {ci_str} "
                        f"| {p_str} "
                        f"| {p_fdr_str} "
                        f"| {sig_str} "
                        f"| {row['n_subjects']} "
                        f"| {row['n_pairs']} |\n"
                    )

                f.write("\n")

            f.write("---\n\n")

        # Summary of all significant results
        sig_df = results_df[results_df["significant"]].copy()
        f.write("## Summary of Significant Results (FDR-corrected)\n\n")

        if sig_df.empty:
            f.write("No comparisons reached significance after FDR correction.\n")
        else:
            f.write(f"**{len(sig_df)}** significant comparisons out of {len(results_df)} total.\n\n")

            f.write("| Factor | Level A | Level B | Direction | Delta | p-FDR | Better |\n")
            f.write("|--------|---------|---------|-----------|------:|------:|--------|\n")

            for _, row in sig_df.sort_values(["factor", "p_fdr"]).iterrows():
                p_fdr_str = f"{row['p_fdr']:.2e}" if row['p_fdr'] < 0.001 else f"{row['p_fdr']:.4f}"
                better = row['level_a'] if row['observed_delta'] > 0 else row['level_b']
                f.write(
                    f"| {row['factor']} | {row['level_a']} | {row['level_b']} "
                    f"| {row['direction']} | {row['observed_delta']:+.4f} "
                    f"| {p_fdr_str} | {better} |\n"
                )

    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sign-flip pairwise factor tests")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--output", type=str,
                        default=str(OUTPUT_DIR / "signflip_pairwise_tests.md"))
    parser.add_argument("--model-filter", type=str, default="logreg")
    parser.add_argument("--metric", type=str, default="brier",
                        choices=["brier", "log_loss", "error"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_all_test_outputs(results_dir)

    if args.model_filter:
        df = df[df["model"] == args.model_filter]
        print(f"Filtered to model={args.model_filter}: {len(df):,} rows")

    # Unique factor levels
    print("\nFactor levels:")
    for col in ["fc_type", "atlas", "strategy", "gsr"]:
        levels = sorted(df[col].astype(str).unique())
        n_pairs = len(list(combinations(levels, 2)))
        print(f"  {col}: {levels} ({n_pairs} pairwise comparisons)")

    # Run tests
    results_df = run_all_pairwise_tests(df, metric=args.metric)

    # FDR correction
    results_df = apply_fdr_correction(results_df)

    # Save raw results as CSV
    csv_path = output_path.with_suffix(".csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Raw results saved to: {csv_path}")

    # Write report
    write_report(results_df, output_path, metric=args.metric)

    # Console summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for factor in ["fc_type", "atlas", "strategy", "gsr"]:
        fdf = results_df[results_df["factor"] == factor]
        n_sig = fdf["significant"].sum()
        n_total = len(fdf)
        print(f"\n  {factor}: {n_sig}/{n_total} significant (FDR-corrected)")

        sig = fdf[fdf["significant"]].sort_values("observed_delta", ascending=False)
        for _, row in sig.iterrows():
            better = row['level_a'] if row['observed_delta'] > 0 else row['level_b']
            print(f"    {row['level_a']:20s} vs {row['level_b']:20s}  "
                  f"delta={row['observed_delta']:+.4f}  p_fdr={row['p_fdr']:.2e}  "
                  f"[{row['direction']}]  better={better}")


if __name__ == "__main__":
    main()
