# Statistical Analysis Plan

This document outlines the final statistical framework for evaluating and comparing functional connectivity (FC) pipelines across Reliability (ICC), Quality Control (QC-FC), and Predictive Validity (ML).

## 1. Primary Metrics

| Dimension | Metric | Interpretation |
|-----------|--------|----------------|
| **Reliability** | Masked ICC (3,1), (2,1), (1,1) | Higher is better (0-1) |
| **QC-FC** | Mean absolute correlation (|r|) | Lower is better (0-1) |
| **Prediction** | Cross-Site Accuracy & ROC-AUC | Higher is better (0.5-1.0) |

---

## 2. Factor Importance Analysis (N-way ANOVA)

We use N-way Analysis of Variance (ANOVA) to determine which processing choices drive the most variance in performance.

- **Model:** `Metric ~ Strategy + GSR + FC_Type + Atlas + [Site] + [Condition]`
- **Effect Size:** **Partial Eta-Squared ($\eta_p^2$)** is the primary measure of factor importance.
- **Calculation:** $\eta_p^2 = \frac{SS_{effect}}{SS_{effect} + SS_{error}}$, where $SS_{effect}$ is the sum of squares for the factor and $SS_{error}$ is the residual sum of squares.
- **Interpretation:** $\eta_p^2$ represents the proportion of variance explained by a factor after accounting for other factors in the model. This allows for comparing the relative impact of "Denoising Strategy" vs. "FC Type" across different benchmarking dimensions.

---

## 3. Global Factor Comparisons (Paired Randomization)

To evaluate the global impact of a single processing choice (e.g., "Does GSR help?"), we use matched-pair randomization tests.

- **Method:** **Sign-Flip Randomization Test (5,000 permutations)**.
- **Unit of Analysis:** Subject-level loss (Brier score or Accuracy).
- **Matching:** We pair pipelines that differ *only* by the factor of interest (e.g., `AAL_1_GSR_corr` vs `AAL_1_noGSR_corr`).
- **Tests Performed:**
    1. **GSR vs. noGSR**: Global impact of global signal regression.
    2. **Tangent vs. Corr**: Global impact of Riemannian tangent projection vs. Pearson correlation.
    3. **Brainnetome vs. Others**: Impact of atlas selection (Brainnetome vs. average of others).

---

## 4. Pipeline-Specific Comparisons

- **Vs. Chance:** **Permutation Test (1,000 repeats)**. Shuffles training labels to establish empirical p-values for each specific pipeline. To account for the large number of pipelines (259+), p-values are corrected using the **False Discovery Rate (FDR, Benjamini-Hochberg)**.
- **Vs. Each Other (Few-Shot):** **Linear Mixed-Effects Models (LMM)** are used to compare metrics (Accuracy/AUC) while accounting for the dependency across the 50 random splits.
    - **Formula:** `Metric ~ FC_Type + (1 | Pipeline_ID)`
    - This allows for a robust statistical answer to whether one metric (e.g., Tangent) significantly outperforms another (e.g., Pearson) across the variability of denoising strategies.

---

## 5. Implementation Details

### Reliability (ICC)
- **Masking:** Only the top 5% of edges by weight are included to avoid the "null edge" bias in sparse connectivity methods.
- **Exclusion:** Primary statistical reports are generated excluding the HCPex atlas to avoid high-dimensionality bias in the OLS models, though figures are provided for all four atlases.

### Quality Control (QC-FC)
- **Aggregation:** For the China dataset, correlations from the two EC sessions are averaged into a single `unified_close` value per pipeline before ANOVA.
- **Condition:** "Open" and "Close" states are treated as a factor in the QC-FC ANOVA to test for state-dependent motion contamination.

### Machine Learning (ML)
- **AUC Importance:** ANOVA is performed separately for Accuracy and ROC-AUC, as AUC is often more sensitive to connectivity metric choices.
- **Few-Shot Stability:** Few-shot performance is aggregated over 50 random splits to ensure statistical stability.
- **FDR Correction:** Applied separately for each cross-site direction (China->IHB and IHB->China).