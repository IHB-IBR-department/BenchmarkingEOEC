# BenchmarkingEOEC

Code for the paper:
Medvedeva, T., Knyazeva, I., Masharipov, R., Korotkov, A., Cherednichenko, D., & Kireev, M. (2025).
Benchmarking resting state fMRI connectivity pipelines for classification: Robust accuracy despite processing variability in cross-site eye state prediction. Neuroscience.
https://doi.org/10.1101/2025.10.20.683049

Preprocessed data for Beijing EO/EC is available here:
https://drive.google.com/file/d/1N1wdbF0tsrAzL9GSgUK_y4xBNDiPLH5X/view?usp=sharing

## Classification pipeline (cross_site and few_shot)

The core pipeline is implemented in `benchmarking/cross_site.py` (cross-site) and
`benchmarking/few_shot.py` (few-shot domain adaptation). The steps are the same for
both schemes, with different train/test definitions.

1) Load FC matrices per site and pipeline
   - FC types: corr, partial, tangent, glasso
   - Atlases: AAL, Schaefer200, Brainnetome, HCPex (see configs)
   - Denoising strategies: 1..6
   - GSR: GSR or noGSR

2) Build labels and sample order
   - Each subject contributes two scans: EC (label 0) and EO (label 1).
   - Data are concatenated as [EC..., EO...] per site so the sample order is
     consistent across pipelines.

3) Vectorize FC matrices to edge features
   - Symmetric FC matrices are vectorized via `sym_matrix_to_vec` (diagonal removed).
   - This yields a high-dimensional edge feature vector per scan.

4) Optional scaling
   - StandardScaler is applied when `scale: true` (auto-enabled for SVM).
   - The scaler is fit on training data only, then applied to test data.

5) PCA dimensionality reduction
   - PCA is fit on training features only and then applied to test features.
   - `pca_components` can be:
     - a float in (0,1] for variance retained (default: 0.95)
     - an int for a fixed number of components
     - 0 or None to disable PCA

6) Classifier
   - Logistic regression (default) or SVM with RBF kernel.
   - Outputs include predicted labels and a continuous score
     (decision_function or predict_proba) when available.

7) Metrics and outputs
   - Metrics: accuracy, ROC-AUC, Brier score.
   - Optional permutation test vs chance when `n_permutations > 0`.
   - Cross-site can also save per-sample outputs for paired comparisons.

## Preventing subject leakage and keeping splits identical

Cross-site (Scheme A):
- Train on one site and test on the other (no subject overlap by design).
- Each direction has a fixed test set, so all pipelines are evaluated on the
  same subjects in the same order.
- PCA and scaling are fit on training data only; test data are transformed only.

Few-shot (Scheme B):
- Subject-level splits are generated once and reused for all pipelines.
- Both EC and EO scans from the same subject stay in the same split.
- Splits are saved to `*_splits.yaml` for reproducibility and auditing.

Permutation test:
- Only `y_train` is permuted; `X_train`, `X_test`, and `y_test` remain fixed,
  avoiding leakage across train/test.

## PCA choice

The default `pca_components: 0.95` retains 95% variance to stabilize training on
high-dimensional FC edge features. This reduces overfitting and improves
numerical stability for linear models. The number of components is stored in
outputs for transparency. You can set a fixed number of components or disable
PCA entirely via config.

## Statistical testing and pipeline comparisons

There are two distinct questions:

1) Pipeline vs chance (per pipeline)
   - Permutation test with label shuffling on the training set.
   - Controlled by `n_permutations` in YAML configs.

2) Pipeline A vs B (paired comparisons on the same test set)
   - Exact McNemar test for accuracy.
   - DeLong test for correlated ROC-AUCs.
   - Requires per-sample outputs saved by cross-site runs.

Factor-level comparisons (GSR, FC type, atlas, strategy) use paired, subject-level
randomization (sign-flip) tests with matched pipelines. This aggregates per-sample
losses to subject-level means before testing.

Required inputs for paired comparisons:
- `results/*_test_outputs.csv` (from cross-site with `save_test_outputs: true`)
- `results/*_pipeline_abbreviations.csv` (maps P0001.. to full pipeline specs)

Helper scripts:
- `benchmarking/pipeline_comparisons.py` (CLI for factor and A vs B tests)
- `run_comparisons.sh` (convenience wrapper)

## Tangent leakage experiment

Tangent FC uses a reference matrix; if computed on all subjects before splitting,
information can leak from test to train. The repo includes an explicit check:

- Script: `benchmarking/tangent_leakage.py`
- Method: compare LEAK (reference on all subjects) vs NO_LEAK (reference fit on
  train subjects only) using GroupKFold with subject-level splits.
- EO and EC scans from the same subject are kept in the same fold.

Configs:
- `configs/tangent_leakage_quick.yaml` (quick sanity check)
- `configs/tangent_leakage_full.yaml` (full analysis)

Outputs:
- `results/tangent_leakage_*_folds.csv`
- `results/tangent_leakage_*_summary.csv`

## Outputs overview

Cross-site:
- `results/*_results.csv` per pipeline results
- `results/*_summary_*.csv` aggregated summaries
- `results/*_test_outputs.csv` per-sample outputs (if enabled)
- `results/*_pipeline_abbreviations.csv` mapping of pipeline abbrev to spec
- `results/pipeline_predictions_<train>_to_<test>.csv` wide prediction matrices

Few-shot:
- `results/*_results.csv` per pipeline per repeat
- `results/*_summary_*.csv` aggregated summaries
- `results/*_splits.yaml` saved subject splits

## Quick usage (examples)

Cross-site (classification + outputs for comparisons):
- `PYTHONPATH=. python -m benchmarking.cross_site --config configs/cross_site_quick_classification.yaml`

Few-shot:
- `PYTHONPATH=. python -m benchmarking.few_shot --config configs/few_shot_quick.yaml`

Permutation demo (correlation only, small test):
- `PYTHONPATH=. python -m benchmarking.cross_site --config configs/cross_site_corr_schaefer_perm.yaml`

Pipeline comparisons (after running a cross-site config with `save_test_outputs: true`):
- Factor-level test (GSR vs noGSR):
  `PYTHONPATH=. python -m benchmarking.pipeline_comparisons factor --test-outputs results/cross_site_quick_classification_test_outputs.csv --factor gsr --level-a GSR --level-b noGSR`
- Pipeline A vs B (by abbreviation):
  `PYTHONPATH=. python -m benchmarking.pipeline_comparisons compare --test-outputs results/cross_site_quick_classification_test_outputs.csv --abbrev results/cross_site_quick_classification_pipeline_abbreviations.csv --pipeline-a P0001 --pipeline-b P0003`
