# Response to Reviewers

We thank the reviewers for their constructive feedback and insightful comments. We have conducted a comprehensive update to our analysis framework, including new statistical tests, additional denoising strategies (ICA-AROMA), and rigorous data-leakage controls. 

Below are our point-by-point responses.

---

# Reviewer 1

## Major Comments

### 1. Cross-site validation terminology
**Reviewer Comment:** *The manuscript claims to use “cross-site validation,” yet the training set includes both datasets and the test set is only half of the IHB RAS dataset... If this is a description mistake, it should be clarified.*

**Our Response:** We agree that the previous description combined two distinct evaluation schemes. We have now formally separated the analysis into two independent protocols:
1.  **Direct Cross-Site Validation:** A true cross-site test where the model is trained on one site (e.g., Full China dataset) and tested on an entirely unseen site (e.g., Full IHB dataset). No samples from the test site are used during any stage of training or feature selection.
2.  **Few-Shot Domain Adaptation:** This protocol evaluates how adding a small, controlled number of target-site samples (e.g., 20 subjects from IHB) to the training pool (China) improves generalization to the remaining target-site subjects.

**Changes in Manuscript:** Methods and Results sections have been restructured to report these as "Direct Generalization" and "Few-Shot Adaptation," respectively.

### 2. Clarification of "selecting 42 subjects"
**Reviewer Comment:** *What does “by selecting a new set of 42 subjects” mean? Is this cross‑validation, or does it involve another sampling technique such as bootstrapping?*

**Our Response:** This refers to a **stratified subject-level cross-validation** scheme. To ensure robust results, we perform 15 to 20 folds of cross-validation. In each fold, we randomly sample a specific number of subjects (e.g., 42 for a 50/50 split) to serve as the training set, while the remaining subjects form the test set. 

**Changes in Manuscript:** We have updated the Methods section to explicitly define this as "subject-level stratified shuffling" and clarified that the folds are independent.

### 3. Tangent Space computation and Data Leakage
**Reviewer Comment:** *How was “tangent connectivity/space” derived? Importantly, were tangent connectomes computed within each cross‑validation fold (i.e., using only the training set) or across the entire sample?*

**Our Response:** We have implemented a strictly **leakage-free ConnectomeTransformer** based on the geometry-aware framework by Varoquaux et al. (2010). 
- **Method:** The tangent space projection requires a reference matrix (the Fréchet mean of covariance matrices). 
- **Leakage Control:** In our updated pipeline, the reference matrix is calculated **only using the training set** within each fold or cross-site direction. The test set covariance matrices are then projected into the tangent space defined by the training data. This ensures that no information from the test subjects influences the feature representation of the training data.

### 4. Cross-validation structure and Identity Leakage
**Reviewer Comment:** *It is also unclear whether both states of a given subject are either test or training samples. If the training set contains one state’s connectome while the other state is in the test set, this would result in data leakage.*

**Our Response:** We have addressed this by implementing **Subject-Level Grouping**. Our splitting algorithm ensures that all scans belonging to a specific subject (both Eyes Open and Eyes Closed) are kept together in either the training set or the test set. A subject can never contribute one scan to training and another to testing. This prevents "identity leakage" where the model might learn subject-specific traits instead of state-specific patterns.

### 5. Permutation Testing
**Reviewer Comment:** *Why was Permutation Testing not used? This would not only test the significance of CV scores but also allows for statistically comparing the ML performance of pipelines.*

**Our Response:** We have now integrated two types of permutation testing:
1.  **Vs. Chance:** For every pipeline configuration, we run label-shuffling permutation tests with `n_permutations` configurable (100 permutations in the current `results/summary/` snapshot; can be increased to 1,000+ for final reporting). We report the empirical p-value for the observed accuracy against this null distribution.
2.  **Between-Pipeline / Factor Comparison:** We use **Paired Sign-Flip Randomization Tests (5,000 permutations)** to compare factor effects (e.g., Tangent vs. Pearson) on matched pipelines and subject-level losses.

### 6. General Trends vs. "Best" Pipeline
**Reviewer Comment:** *Pearson + SCH achieved the highest classification accuracy... therefore the manuscript should refrain drawing conclusions about “best method” etc, but instead discuss general trends.*

**Our Response:** We have shifted our focus to **Factor Importance Analysis** using N-way ANOVA. By calculating the **Partial Eta-Squared ($\eta_p^2$)**, we quantify how much of the variance in accuracy is explained by the choice of FC Type, Strategy, Atlas, and Model. 
- **Result:** We found that **FC Type (Tangent Space)** is the dominant factor ($\eta_p^2 \approx 0.70$), consistently yielding high performance regardless of the "luck" of a specific fold.

### 7. Statistical comparisons for GSR
**Reviewer Comment:** *When comparing GSR versus no‑GSR for each atlas... why were statistical comparisons not performed?*

**Our Response:** We now provide **Paired Randomization Tests** for the GSR effect. Across all matched pipeline pairs, GSR showed a statistically significant improvement in QC-FC (motion removal) but a more varied effect on ML accuracy. For the Brainnetome and AAL atlases, GSR significantly improved classification ($p < 0.05$).

### 8. Results for all atlases
**Reviewer Comment:** *The results for “Similarity between denoising strategies” and QC-FC was only provided for the Brainnetome atlas, why?*

**Our Response:** We have expanded our results to be comprehensive. We now provide **multi-row violin plots for all four atlases** (AAL, Schaefer200, Brainnetome, HCPex) for both ICC and QC-FC. Similarly, consistency heatmaps have been generated for each atlas to show that the trends are robust across spatial scales.

### 9. PCA Implementation
**Reviewer Comment:** *Line 568 states that PCA was applied, but this was not introduced anywhere in the manuscript? Was PCA done within CV or outside CV?*

**Our Response:** We have added a detailed "Dimensionality Reduction" section to the Methods.
- **Protocol:** PCA is applied to reduce the high-dimensional FC vectors. The number of components is selected to retain **95% of the variance**. 
- **Leakage Control:** Crucially, the PCA transformation is **fit on the training set only** and then applied to the test set.

### 10. ICA-AROMA Inclusion
**Reviewer Comment:** *The study should also include ICA‑AROMA with different strategies (soft vs. aggressive).*

**Our Response:** We have added both **AROMA Aggressive** and **AROMA Non-Aggressive** to our benchmark. Interestingly, our results show that while AROMA is effective at motion removal, high-variance aCompCor strategies (Strategy 5) often yield higher classification accuracy for this specific task.

### 11. Non-linear ML Models
**Reviewer Comment:** *Adding a non‑linear ML algorithm (e.g., Logistic Regression with an RBF kernel) would still be highly beneficial.*

**Our Response:** We have added **SVM with an RBF kernel** to the benchmark. Our ANOVA analysis shows that the choice of model ($\eta_p^2 \approx 0.04$) has a significantly smaller impact on performance than the choice of connectivity metric ($\eta_p^2 \approx 0.70$), confirming that the biological signal in the FC matrix is the primary driver of success.
*Note:* Logistic regression is the primary benchmark model; non-linear models are treated as exploratory unless run across the full pipeline grid.

---

# Reviewer 2

### 1. Unambiguous Split Logic
**Reviewer Comment:** *If splitting is scan-level rather than subject-level, paired EO/EC scans from the same subject could leak across sets... the author should implement the exact split logic.*

**Our Response:** As noted in response to Reviewer 1 (Point 4), we have implemented a strictly **subject-grouped stratified split**. All results presented in the updated manuscript are derived from this leakage-free protocol.

### 2. Reproducibility of Model Fitting
**Reviewer Comment:** *PCA details are insufficient... specify (a) how PCA dimensionality was chosen, (b) fit on training data only... (d) logistic regression solver.*

**Our Response:** We have formalized the model specification:
- **Scaling:** Features are standardized (Z-scored) using the mean and standard deviation of the **training set**.
- **PCA:** Retains 95% variance, fit on training set only.
- **LR Solver:** We use the `LBFGS` solver with L2 regularization ($C=1.0$) and a maximum of 1,000 iterations to ensure convergence.

### 3. Uncertainty Quantification
**Reviewer Comment:** *Uncertainty quantification is too weak... no clear inferential framing for comparing many pipelines.*

**Our Response:** Our updated statistical framework moves beyond mean $\pm$ SD:
- **ANOVA ($\eta_p^2$):** Used to rank the influence of processing factors.
- **Permutation Tests vs Chance:** Label-shuffling tests per pipeline to quantify significance against chance.
- **Paired Randomization (Sign-Flip):** Subject-level paired tests for global factor effects (matched pipeline pairs).
- **Paired Pipeline Tests:** McNemar (accuracy) and DeLong (AUC) on identical test sets when comparing two specific pipelines.

### 4. Terminology Consistency
**Reviewer Comment:** *Table 1 defines strategies using “12 HMP,” but Results summarize as “aCompCor + 12P + GSR,” naming is inconsistent; additionally, “SSR” appears without definition.*

**Our Response:** We have performed a terminology audit. 
- "12P" and "24P" are now used consistently to refer to the number of motion parameters. 
- "SSR" was a typographical error for "GSR" (Global Signal Regression) and has been corrected. 
- Every pipeline in the Results is now cross-referenced to its unique ID in Table 1.

---

# Final Summary of Statistical Results

Based on our updated, rigorous analysis across **Reliability (ICC)**, **Motion Removal (QC-FC)**, and **Prediction (ML)**:

1.  **Reliability:** The choice of **FC Type** is the most critical factor ($\eta_p^2 = 0.88$). **Tangent Space** projection provides the most reliable estimates.
2.  **Motion:** **GSR** significantly improves the removal of motion artifacts from the functional connectivity matrices.
3.  **Prediction:** **Tangent Space** projection consistently outperforms all other metrics in cross-site classification accuracy and generalization.
4.  **Robustness:** High classification accuracy (~80%) is achievable across a wide range of pipelines, provided a principled connectivity metric like Tangent Space is used.

---

# Summary of Changes to the Manuscript

Based on the reviewers' feedback and our new analyses, the following specific changes should be made to the manuscript text and figures.

### 1. Title and Abstract
- **Title Update:** Clarify the distinction between generalization and adaptation. *Current title is robust, no change needed.*
- **Abstract:** Update classification accuracy results (~80% -> ~84% for best pipelines). Explicitly mention the inclusion of ICA-AROMA and Tangent Space as a key finding for both reliability and prediction.

### 2. Methods Section
- **Experimental Design:** 
    - Create a new subsection "Evaluation Schemes" to clearly define **Direct Cross-Site Validation** (zero target-site data in training) and **Few-Shot Domain Adaptation** (incremental target-site data).
    - Add a description of **Subject-Grouped Stratification** to confirm no identity leakage between training and testing sets.
- **Denoising Strategies (Table 1):** 
    - Update to include ICA-AROMA (Aggressive and Non-Aggressive).
    - Unify parameter naming: use "12P" and "24P" consistently.
- **HCPex Atlas Preprocessing:** 
    - Add a subsection describing the **373 ROI dimension alignment**. Explain that standard strategies (with varying ROI counts) were aligned to a unified mask to ensure fair benchmarking.
- **Connectivity Computation:** 
    - Provide the mathematical derivation for **Tangent Space Projection**.
    - Explicitly state that the reference matrix is **fit on training data only** within the cross-validation loop.
- **Machine Learning & Dimensionality Reduction:**
    - Formally define the **PCA (95% variance)** step. State that it is fit on training data only.
    - Specify Logistic Regression parameters: `LBFGS` solver, L2 penalty, $C=1.0$.
    - Mention the inclusion of the non-linear **SVM-RBF** model as a secondary baseline.
- **Statistical Analysis:**
    - Define **Partial Eta-Squared ($\eta_p^2$)** as the primary measure of factor importance from N-way ANOVA, clarifying that it represents the proportion of variance explained by each factor.
    - Describe the **Permutation Test** protocol for significance vs. chance, including the use of **False Discovery Rate (FDR, Benjamini-Hochberg)** correction for multiple comparisons across the 259+ pipelines.
    - Add a description of **Linear Mixed-Effects Models (LMM)** used for few-shot performance comparisons to account for the dependency across splits.
    - Describe the **Paired Sign-Flip Randomization Test (5,000 repeats)** for factor comparisons (GSR, FC Type).

### 3. Results Section
- **Reliability (ICC):** 
    - Update figures to show all four atlases. 
    - Report ANOVA results showing FC Type as the dominant factor ($\eta_p^2 = 0.88$).
- **Motion Removal (QC-FC):** 
    - Update violin plots to include corrected HCPex data and ICA-AROMA.
    - Report the statistically significant impact of **GSR** on reducing |r| values.
- **State Classification (ML):**
    - Replace single-pipeline accuracy counts with the **ANOVA importance ranking**.
    - Add a section on **ROC-AUC** results, showing its high sensitivity to FC Type choices ($\eta_p^2 = 0.85$).
    - Report **Global Significance**: X% of pipelines perform significantly above chance in both cross-site directions.
    - Update **Figure 5** (now multi-metric) to include few-shot adaptation curves with `n_repeats` repeats (configured; increase for smoother visualization).
- **Strategy Consistency:** 
    - Include the new **Subject-wise Correlation Heatmaps** to demonstrate the high similarity of Glasso and Pearson patterns across denoising strategies.

### 4. Discussion Section
- **FC Metric Choice:** Expand on why **Tangent Space** projection is superior (respecting Riemannian geometry) and why it is more robust to denoising variability.
- **Atlas Resolution:** Discuss the finding that **Brainnetome** and **Schaefer200** provide optimal dimensionality for current sample sizes, avoiding the "curse of dimensionality" seen with HCPex in certain configurations.
- **GSR Effect:** Provide a balanced discussion on GSR: it improves motion removal (QC-FC) and often classification accuracy, though it has negligible effect on reliability (ICC).
- **Clinical Implications:** Add a paragraph on how these benchmarked "stable" pipelines can be applied to clinical datasets where state-dependent signal is weaker.

### 5. Figures and Tables
- **Figure 1 (Pipeline):** Update caption to a single cohesive paragraph. Ensure all steps (PCA, Leakage-safe Tangent) are represented.
- **Figure 2 & 4 (Violin Plots):** Update to include all 4 atlases and ensure color-blind friendly palettes (e.g., Seaborn 'colorblind' or 'muted').
- **Figure 3 (Heatmaps):** Increase size and add numerical annotations for legibility.
- **Table 1 (Strategies):** Add ICA-AROMA rows.
- **New Supplemental Tables:** Add the "Top 20 Pipelines" rankings for Accuracy, AUC, and ICC.
