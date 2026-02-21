# Statistical Analysis Plan

This document outlines the statistical framework for evaluating and comparing functional connectivity (FC) pipelines across Reliability (ICC), Quality Control (QC-FC), and Predictive Validity (ML).

## 1. Primary Metrics

| Dimension | Metric | Interpretation |
|-----------|--------|----------------|
| **Reliability** | Masked ICC (3,1), (2,1), (1,1) | Higher is better (0-1) |
| **QC-FC** | Mean absolute correlation (\|r\|) | Lower is better (0-1) |
| **Prediction** | Cross-Site Accuracy, ROC-AUC, Brier Score | Accuracy/AUC: higher is better; Brier: lower is better |

---

## 2. Factor Importance Analysis (Linear Mixed-Effects Model)

We use a **Linear Mixed-Effects Model (LMM)** to determine which processing choices drive the most variance in classification performance, accounting for the repeated-measures structure of the data (same subjects evaluated under all 192 pipeline configurations).

- **Model:** `BrierLoss ~ C(fc_type) + C(atlas) + C(strategy) + C(gsr) + (1 | test_subject)`
- **Dependent Variable:** **Brier score** (per-sample squared probability error), a strictly proper scoring rule that is continuous on [0, 1] and decomposes additively across samples. Chosen because accuracy is binary at the sample level (unsuitable for linear modeling) and ROC-AUC is a population-level statistic that cannot be decomposed per subject. The Brier score is strongly anti-correlated with both accuracy (r = −0.91) and AUC (r = −0.82).
- **Random Effect:** Test subject as random intercept, capturing within-subject correlations across pipeline configurations.
- **Factor Importance:** Assessed via **Type III Wald chi-square tests** of fixed effects.
- **Rationale for LMM over RM-ANOVA:** (1) 192 within-subject conditions make sphericity untenable, (2) cross-validation produces mildly unbalanced data, (3) Brier score can exhibit non-normal residuals.

### Supplementary: OLS ANOVA

The original OLS ANOVA with **Partial Eta-Squared ($\eta_p^2$)** is retained as a supplementary analysis for comparison. Factor importance rankings are consistent between LMM (Wald chi-square) and OLS ($\eta_p^2$).

### Supplementary: LMM on ROC-AUC

A separate LMM analysis is performed on pipeline-level ROC-AUC with pipeline_group (atlas × strategy × GSR) as random intercept, confirming factor rankings are consistent across metrics.

---

## 3. Pairwise Factor Comparisons (Subject-Level Sign-Flip Randomization)

To evaluate the impact of specific processing choices (e.g., "Does GSR help?", "Is tangent better than correlation?"), we use matched-pair randomization tests for **all pairwise factor-level comparisons**.

- **Method:** **Subject-level sign-flip randomization test (5,000 permutations)**.
- **Unit of Analysis:** Subject-level mean Brier loss.
- **Matching:** Pipelines are paired that differ *only* by the factor of interest (e.g., `AAL_1_GSR_corr` vs `AAL_1_noGSR_corr`). Per-sample losses are computed for each pipeline in the pair, then aggregated to the subject level.
- **Multiple Comparison Correction:** **FDR (Benjamini-Hochberg)** within each factor.
- **Confidence Intervals:** 95% bootstrap CIs (5,000 resamples) on the subject-level mean difference.
- **Tests Performed:**
    - **FC type:** 6 pairwise comparisons × 2 directions = 12 tests
    - **Atlas:** 6 pairwise comparisons × 2 directions = 12 tests
    - **Strategy:** 28 pairwise comparisons × 2 directions = 56 tests
    - **GSR:** 1 comparison × 2 directions = 2 tests
    - **Total:** 82 tests

---

## 4. Pipeline-Specific Comparisons

- **Vs. Chance:** **Permutation Test (1,000 repeats)**. Shuffles training labels to establish empirical p-values for each specific pipeline. P-values corrected using **FDR (Benjamini-Hochberg)**.
- **Vs. Each Other (Few-Shot):** **Linear Mixed-Effects Models (LMM)** comparing metrics while accounting for dependency across the 50 random splits.
    - **Formula:** `Metric ~ FC_Type + (1 | Pipeline_ID)`

---

## 5. Classification Feature Interpretation (Haufe Transformation)

To identify neurophysiologically interpretable biomarkers for EO/EC classification, classifier weights are transformed using the **Haufe forward model** (Haufe et al., 2014).

- **Backward model** (extraction filters): raw logistic regression coefficients back-projected through PCA. Optimized for classification, not neurophysiologically interpretable.
- **Forward model** (activation patterns): **a** = Σ_x **w**, computed efficiently via PCA spectral decomposition: **a**_pca = **λ** ⊙ **w**_pca (exact because discarded PCA dimensions are orthogonal to **w**).
- **Stability Assessment:** 1,000 random 80% subsamples (stratified by site) of the pooled dataset (N = 132).
- **Stable Edge Criteria:** Sign consistency ≥ 80% AND mean importance rank ≤ 500 (top ~3% of edges).
- **Network Aggregation:** Stable edges are aggregated by Yeo 7-network pair assignments for interpretability.

---

## 6. Implementation Details

### Reliability (ICC)
- **Masking:** Only the top 5% of edges by weight are included to avoid the "null edge" bias in sparse connectivity methods.
- **Exclusion:** Primary statistical reports are generated excluding the HCPex atlas to avoid high-dimensionality bias in the OLS models, though figures are provided for all four atlases.

### Quality Control (QC-FC)
- **Aggregation:** For the China dataset, correlations from the two EC sessions are averaged into a single `unified_close` value per pipeline before ANOVA.
- **Condition:** "Open" and "Close" states are treated as a factor in the QC-FC ANOVA to test for state-dependent motion contamination.

### Machine Learning (ML)
- **LMM Dependent Variable:** Brier score for subject-level analysis; ROC-AUC for pipeline-level supplementary analysis.
- **Few-Shot Stability:** Few-shot performance is aggregated over 50 random splits to ensure statistical stability.
- **FDR Correction:** Applied within each factor for sign-flip tests; separately for each cross-site direction for pipeline-level tests.

---

## 7. Key Results Summary

### Factor Importance Ranking (LMM, Brier score, combined directions)

| Factor | Wald χ² | df | p-value |
|--------|--------:|---:|--------:|
| FC type | 1508.13 | 3 | < 10⁻¹⁵ |
| Atlas | 231.59 | 3 | < 10⁻¹⁵ |
| Strategy | 119.78 | 7 | < 10⁻¹⁵ |
| GSR | 4.40 | 1 | 0.036 |

**Ranking: FC type >> Atlas > Strategy > GSR**

### Sign-Flip Pairwise Test Summary

**48 / 82** pairwise comparisons significant after FDR correction.

| Factor | Significant / Total | Key findings |
|--------|-------------------:|:-------------|
| FC type | 10 / 12 | Tangent best (all comparisons significant); partial worst |
| Atlas | 5 / 12 | Significant only in China→IHB direction; AAL best, HCPex worst |
| Strategy | 32 / 56 | AROMA_aggr significantly worst; strategies 4 and 3 generally best |
| GSR | 1 / 2 | GSR significantly better only in IHB→China direction |
