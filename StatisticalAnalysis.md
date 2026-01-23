# Statistical Analysis Plan

This document outlines the statistical framework for evaluating and comparing functional connectivity (FC) pipelines across three key dimensions: Reliability (ICC), Quality Control (QC-FC), and Predictive Validity (ML Classification).

## 1. Overview of Metrics

| Dimension | Metric | Interpretation | Unit of Analysis |
|-----------|--------|----------------|------------------|
| **Reliability** | Intraclass Correlation (ICC 3,1) | Higher is better (0-1) | Edge-wise (aggregated to mean per pipeline) |
| **QC-FC** | QC-FC Mean |r| | Lower is better (0-1) | Edge-wise correlation with motion (aggregated to mean per pipeline) |
| **Prediction** | Classification Accuracy / AUC | Higher is better (0.5-1.0) | Subject-level predictions (aggregated to mean per pipeline) |

---

## 2. Primary Research Questions

The analysis is structured by metric type, prioritizing factors that are theoretically expected to drive performance.

### A. Reliability (ICC) & Quality Control (QC-FC)
**Primary Factors:** Denoising Strategy, FC Type
**Secondary Factor:** GSR (Global Signal Regression)
**Note:** Atlas choice is considered a "nuisance" factor for these metrics, as it primarily affects the spatial aggregation scale rather than the fundamental signal quality.

**Questions:**
1.  **Denoising Strategy:** Which strategy (1-6, AROMA) yields the most reliable FC estimates (highest ICC) and best motion artifact removal (lowest QC-FC)?
2.  **FC Type:** Does using Tangent Space or Partial Correlation improve reliability compared to Pearson Correlation?
3.  **GSR:** Does the inclusion of GSR consistently improve motion removal (QC-FC) and does it come at a cost to reliability (ICC)?

### B. Machine Learning Prediction (ML)
**Primary Factors:** Denoising Strategy, FC Type, Atlas
**Secondary Factor:** GSR

**Questions:**
1.  **Predictive Power:** Which combination of Atlas, Denoising Strategy, and FC Type maximizes cross-site classification accuracy?
2.  **Atlas Sensitivity:** Does increasing ROI resolution (e.g., Schaefer200 vs AAL) improve predictive performance?
3.  **Generalization:** Which pipelines offer the best few-shot adaptation performance?

---

## 3. Statistical Comparisons

We employ a hierarchical approach: first assessing global differences across primary factors using omnibus tests, followed by planned post-hoc comparisons.

### A. Omnibus Tests (Global Differences)

- **Test:** Friedman Rank Sum Test.
- **Why:** Non-parametric test suitable for repeated measures (pipelines evaluated on the same subjects/edges).
- **Application:**
    - **ICC & QC-FC:** Test for differences in **Denoising Strategy** (blocking on Atlas, GSR, FC Type).
    - **ML:** Test for differences in **Atlas** (blocking on Strategy, GSR, FC Type).

### B. Post-Hoc Pairwise Comparisons
If the omnibus test is significant ($p < 0.05$), we proceed with pairwise comparisons.

- **Test:** Nemenyi test.
- **Correction:** Implicitly corrects for multiple comparisons.
- **Visualization:** Critical Difference (CD) diagrams.

### C. Factor-Level Comparisons (Effect of Single Choice)

#### 1. GSR vs. noGSR
- **Hypothesis:** GSR improves QC-FC (reduces motion) but may have mixed effects on ICC and ML accuracy.
- **Test:** Wilcoxon Signed-Rank Test (paired).
- **Unit:** The pair of (GSR, noGSR) values for each Strategy/Atlas/FC-Type combination.

#### 2. FC Type Comparison (e.g., Tangent vs. Correlation)
- **Hypothesis:** Tangent space projection improves ML classification accuracy and potentially ICC.
- **Test:** Wilcoxon Signed-Rank Test on distribution of pipeline scores.

---

## 4. Analysis Implementation Details

### ICC & QC-FC Analysis
**Input**: Summary CSVs (`icc_summary.csv`, `qc_fc_results.csv`).

1. **Descriptive Stats**: Compute mean and SD of ICC/QC-FC for each Strategy/FC-Type combination (averaged across Atlases).
2. **Strategy Ranking**: Rank strategies from 1 (best) to N (worst) based on mean ICC and QC-FC.
3. **Correlation**: Compute Spearman correlation between ICC and QC-FC ranks to test if better motion cleaning correlates with higher reliability.

### Machine Learning Analysis
**Input**: Per-sample prediction CSVs (`*_test_outputs.csv`).

1. **Pipeline vs. Chance**: Permutation test (1000 permutations).
2. **Pairwise Model Comparison**:
   - **Accuracy**: Exact McNemar’s test.
   - **AUC**: DeLong’s test.
3. **Few-Shot Learning Curve**: Analyze performance gain as $k$ (target subjects) increases.

---

## 5. Example Code Snippet (Python)

```python
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp

# Load results
df = pd.read_csv("results/icc_results/icc_summary.csv")

# 1. Friedman Test for Strategies (Blocking on Atlas/GSR/FC Type)
# Reshape to (Strategies x Blocks)
pivot_df = df.pivot_table(index=['atlas', 'gsr', 'fc_type'], columns='strategy', values='icc31_mean')
stat, p = friedmanchisquare(*[pivot_df[col] for col in pivot_df.columns])
print(f"Friedman Test (Strategies): Statistic={stat:.3f}, p={p:.3e}")

# 2. Nemenyi Post-hoc
if p < 0.05:
    nemenyi = sp.posthoc_nemenyi_friedman(pivot_df.values)
    print(nemenyi)

# 3. GSR Effect (Wilcoxon)
gsr = df[df['gsr'] == 'GSR']['icc31_mean'].values
nogsr = df[df['gsr'] == 'noGSR']['icc31_mean'].values
w_stat, w_p = wilcoxon(gsr, nogsr)
print(f"GSR vs noGSR: W={w_stat}, p={w_p:.3e}")
```