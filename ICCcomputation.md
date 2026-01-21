## Edge-wise test–retest reliability via Intraclass Correlation Coefficient (ICC)

### Data structure and goal
To quantify test-retest reliability of functional connectivity (FC) edges across repeated acquisitions, we compute the intraclass correlation coefficient (ICC) for each edge independently. ICC measures the proportion of total variability attributable to between-subject differences relative to within-subject variability across repeated measurements. Classical ICC variants differ by (i) the assumed ANOVA model (one-way vs two-way; random vs fixed raters/sessions) and (ii) whether the target is absolute agreement or consistency. :contentReference[oaicite:0]{index=0}

In the China dataset we analyzed two closed-state sessions per subject (k=2), but the implementation supports any k >= 2. Let n be the number of subjects and e be the number of edges. We represent FC as vectorized connectomes, e.g., the upper-triangular edges without diagonal. For each edge, the input is an n x k table:

y_{i1}, y_{i2}, ..., y_{ik}, i=1..n.

---

### ICC variants and interpretation (single-measurement ICCs)
We report ICC as a **single-measurement** reliability index (suffix “1”), because downstream analyses use single-session FC estimates.

#### ICC(1,1): one-way random-effects (classical “one-way” ICC)
ICC(1,1) does **not** model session effects explicitly; any systematic shift between sessions inflates within-subject variability and reduces ICC. :contentReference[oaicite:1]{index=1}

#### ICC(2,1): two-way random-effects, **absolute agreement**
ICC(2,1) treats sessions as a **random sample** from a population of possible sessions/raters and evaluates **absolute agreement**, i.e., systematic session offsets are treated as disagreement and reduce ICC. :contentReference[oaicite:2]{index=2}

#### ICC(3,1): two-way mixed-effects, **consistency**
ICC(3,1) treats the two sessions as **fixed** (these exact sessions) and evaluates **consistency**: it is sensitive to changes in relative subject ordering, while systematic session mean shifts are handled as fixed effects and do not automatically destroy reliability. :contentReference[oaicite:3]{index=3}

**Primary choice in this study.** Because our reliability question concerns repeated measurements obtained with the same protocol in two specific sessions (not a random sample of sessions), and because we primarily target stability of inter-individual differences in FC, we use **ICC(3,1) (consistency)** as the main index. As sensitivity analysis, ICC(2,1) can be additionally reported to quantify **absolute agreement**. :contentReference[oaicite:4]{index=4}

---

### Computation for k sessions (per edge)
For each edge separately (vectorized index), we compute the standard ANOVA mean squares for the balanced two-way design (subjects x sessions). :contentReference[oaicite:5]{index=5}

#### Means
For a given edge:
- Subject means:
  y_{i.} = (1/k) * sum_j y_{ij}
- Session means:
  y_{.j} = (1/n) * sum_i y_{ij}
- Grand mean:
  y_{..} = (1/(n*k)) * sum_i sum_j y_{ij}

#### Mean squares (two-way ANOVA components)
- Subjects (rows):
  MSR = k * sum_i (y_{i.} - y_{..})^2 / (n - 1)
- Sessions (columns):
  MSC = n * sum_j (y_{.j} - y_{..})^2 / (k - 1)
- Within-subject:
  MSW = sum_{i,j} (y_{ij} - y_{i.})^2 / (n * (k - 1))
- Residual:
  MSE = sum_{i,j} (y_{ij} - y_{i.} - y_{.j} + y_{..})^2 / ((n - 1) * (k - 1))

#### ICC formulas for k sessions
- ICC(3,1) (two-way mixed, consistency):
  ICC(3,1) = (MSR - MSE) / (MSR + (k - 1) * MSE)
- ICC(2,1) (two-way random, absolute agreement):
  ICC(2,1) = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
- ICC(1,1) (one-way random, consistency with session effects in error):
  ICC(1,1) = (MSR - MSW) / (MSR + (k - 1) * MSW)
Definitions and relationships among ICC forms follow the classical Shrout & Fleiss taxonomy and subsequent formulations by McGraw & Wong and reporting guidelines. :contentReference[oaicite:6]{index=6}

---

### Practical details for edge-wise vectorized connectomes
1. Vectorized inputs. For each edge m=1..e, we apply the above computations to the k vectors {y_{ij}^{(m)}}_{i=1..n}. This yields an ICC vector of shape (e,). If needed for visualization, the ICC vector can be reconstructed into an ROI x ROI matrix using the same edge-indexing scheme (e.g., upper triangle without diagonal).

2. Missing values. If any missing values occur for a given edge/session/subject, we compute means and sums using complete cases for that edge (pairwise exclusion). The number of valid subjects n_m is tracked per edge when reporting summary statistics.

3. Value range and negative ICC. ICC values are typically between 0 and 1, but can be negative when within-subject variability exceeds between-subject variability (interpreted as no reliability for that edge). We report ICC values without truncation unless explicitly stated. :contentReference[oaicite:7]{index=7}

---

### Implementation in this repo (benchmarking/icc_computation.py)
- Inputs: either a single .npy file or a directory of .npy files.
- Shapes: vectorized data is expected as (n_subjects, n_edges, n_sessions). If a 4D array is provided, it is vectorized across the last axis into that shape. Use --discard-diagonal to match how the edges were vectorized.
- Edge-wise outputs: saved only with --save-edgewise (default: off). Default output directory is ./icc_results in the project folder.
- Summary outputs: --summary-json writes per-file summaries that always include icc11, even if it is not in --icc.
- Summary JSON grouping: if filenames follow the pattern
  site_condition_atlas_strategy-<n>_<GSR|noGSR>_<fc>.npy
  the JSON is grouped by atlas -> strategy -> gsr -> fc. Otherwise, the file is keyed by its relative path.

### Masking (optional)
Masking is controlled by --mask for edge-wise outputs and is always used for summary "mean_masked".
The mask is computed from the input data as follows:
- abs_data = abs(data)
- edge_means = mean(abs_data) across subjects and sessions
- global_threshold = percentile(abs_data, p) across all values
- keep edges where edge_means >= global_threshold
The default percentile is 98 (set by --mask-percentile).

---

### References (for methods section)
- Shrout, P. E., & Fleiss, J. L. (1979). *Intraclass correlations: Uses in assessing rater reliability*. Psychological Bulletin, 86(2), 420–428. :contentReference[oaicite:8]{index=8}  
- McGraw, K. O., & Wong, S. P. (1996). *Forming inferences about some intraclass correlation coefficients*. Psychological Methods, 1(1), 30–46. :contentReference[oaicite:9]{index=9}  
- Koo, T. K., & Li, M. Y. (2016). *A guideline of selecting and reporting intraclass correlation coefficients for reliability research*. Journal of Chiropractic Medicine, 15(2), 155–163. :contentReference[oaicite:10]{index=10}  
- Liljequist, D., Elfving, B., & Skavberg Roaldsen, K. (2019). *Intraclass correlation – A discussion and demonstration of basic features*. PLOS ONE, 14(7): e0219854. :contentReference[oaicite:11]{index=11}
