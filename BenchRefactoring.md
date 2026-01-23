# BenchmarkingEOEC Refactoring Plan (No Backward Compatibility)

This plan assumes the **only supported on-disk data layout** is the one described in `DataDescription.md` (and referenced from `readme.md`). Anything that supports legacy layouts (e.g., `fc_data_china/`, `open2/open3`, `glasso_output/`, etc.) should be removed rather than preserved.

## 1) Goals (Clear, Transparent, Minimal)

1. **Single canonical data model**: everything loads from `OpenCloseBenchmark_data/` as documented in `DataDescription.md`.
2. **No backward compatibility**: delete/replace code paths and docs describing old layouts.
3. **Clear code boundaries**:
   - `data_utils/` (NEW): *all* data I/O + preprocessing utilities (paths, loading, aggregation, precompute).
   - `benchmarking/`: only analysis entrypoints: **ICC**, **QC–FC**, **ML**.
4. **Remove duplication**: one source of truth for path building, strategy parsing, ROI masking (coverage + HCPex).
5. **Repo documentation is consistent**: no “hidden” rules or stale docs that contradict `DataDescription.md`.

## 2) Current Problems Observed (Concrete)

- **Legacy code still present and misleading**
  - `benchmarking/data_utils.py`: contains old China `open2/open3` logic and hard-coded subject index hacks.
  - `benchmarking/metrics.py`: imports the legacy `benchmarking/data_utils.py` and implements old QC–FC/ICC helpers.
  - `benchmarking/cross_site.py` + `benchmarking/few_shot.py`: still contain legacy China `open2/open3` handling and/or rely on `benchmarking/project.py` legacy paths.
- **Multiple conflicting “path truth” modules**
  - `benchmarking/project.py` contains legacy naming and legacy glasso checks (e.g., `open2/open3`), while `benchmarking/timeseries.py` follows the newer `timeseries_{site}/` layout.
  - `benchmarking/timeseries.py` currently builds an incorrect `glasso_precomputed_fc` path (missing `/{site}/{atlas}/`), which contradicts `DataDescription.md`.
- **Docs contradict each other**
  - `.claude/skills/data_handling.md` describes legacy folders (`fc_data_*`, `glasso_output/`) that are not part of the desired canonical layout.
  - `ICCcomputation.md` references a non-existent script (`benchmarking/icc_computation.py`) and contains an outdated masking description.
  - `HCPEX_GLASSO_GUIDE.md` shows an outdated output folder layout for glasso files.
- **Git internals**
  - `.git/logs/refs/remotes/origin/metrics` is a git internal log, not a project file; it should never be referenced in docs/plans.

## 3) Target Repository Structure (Proposed End State)

```
data_utils/
  __init__.py
  paths.py                 # resolve_data_root + filename conventions (DataDescription-only)
  parsing.py               # parse pipeline metadata from filenames
  timeseries.py            # load timeseries_{ihb,china} + subject order
  coverage.py              # load coverage + compute masks (IHB/both/etc)
  hcpex.py                 # HCPex mask + preprocessing helpers
  fc.py                    # ConnectomeTransformer + compute_fc_vectorized
  preprocessing/
    __init__.py
    aggregate_ihb.py        # moved from benchmarking/aggregate_ihb.py (CLI preserved)
    aggregate_china.py      # moved from benchmarking/aggregate_china.py
    precompute_glasso.py    # moved from benchmarking/precompute_glasso.py
    icc_data_preparation.py # moved from benchmarking/icc_data_preparation.py
    create_hcpex_mask.py    # moved from benchmarking/create_hcpex_mask.py

benchmarking/
  __init__.py               # minimal (no data layout rules)
  icc.py                    # ICC analysis entrypoint (CSV + edgewise pickles)
  qc_fc.py                  # QC–FC analysis entrypoint (CSV + optional edgewise outputs)
  ml/
    __init__.py
    pipeline.py             # unified ML entrypoint (cross-site + few-shot)
    stats.py                # permutation tests + paired comparisons
    pipeline_comparisons.py # CLI wrapper around stats (optional)
    tangent_leakage.py      # experiment (optional)

configs/
docs/
  (authoritative docs only)
  archive/                  # old plans/reviewer notes
```

Notes:
- `data_utils/` is the only place allowed to “know” the data layout.
- `benchmarking/` should not contain any aggregation/precompute scripts after the move.
- Keep the CLIs by retaining `python -m ...` entrypoints, just moved under `data_utils.preprocessing`.

## 4) Refactoring Steps (Incremental, Verifiable)

### Phase 0 — Freeze the canonical contract
- Treat `DataDescription.md` as the contract.
- Update `readme.md` to reference only:
  - `benchmarking/icc.py`
  - `benchmarking/qc_fc.py`
  - `benchmarking/ml/pipeline.py`
  - preprocessing CLIs under `data_utils/preprocessing/*` (if you keep them public)

### Phase 1 — Create `data_utils/` package (no behavior changes yet)
- Create `data_utils/` and move code **without changing logic** first:
  - move `benchmarking/timeseries.py` → `data_utils/timeseries.py`
  - move `benchmarking/fc.py` → `data_utils/fc.py`
  - move `benchmarking/hcpex_preprocess.py` → `data_utils/hcpex.py`
  - move `benchmarking/project.py` → `data_utils/paths.py` (and delete legacy-only functions)
- Fix known correctness issues while moving:
  - ensure `glasso_precomputed_fc/{site}/{atlas}/{file}.npy` path construction matches `DataDescription.md`

### Phase 2 — Move preprocessing scripts under `data_utils/preprocessing/`
- Move and re-wire imports:
  - `benchmarking/aggregate_ihb.py` → `data_utils/preprocessing/aggregate_ihb.py`
  - `benchmarking/aggregate_china.py` → `data_utils/preprocessing/aggregate_china.py`
  - `benchmarking/precompute_glasso.py` → `data_utils/preprocessing/precompute_glasso.py`
  - `benchmarking/icc_data_preparation.py` → `data_utils/preprocessing/icc_data_preparation.py`
  - `benchmarking/create_hcpex_mask.py` → `data_utils/preprocessing/create_hcpex_mask.py`
- Decide what to do with one-off migration helpers:
  - `benchmarking/sort_glasso.py`: delete or move to `data_utils/preprocessing/migrations/` and mark as “one-time”.

### Phase 3 — Shrink `benchmarking/` to analysis-only
- Keep:
  - `benchmarking/icc.py`
  - `benchmarking/qc_fc.py`
  - ML under `benchmarking/ml/` (or one `benchmarking/ml.py`)
- Delete (no backward compatibility):
  - `benchmarking/data_utils.py`
  - `benchmarking/metrics.py`
- ML decision:
  - Preferred: delete `benchmarking/cross_site.py` + `benchmarking/few_shot.py` and make `benchmarking/ml/pipeline.py` the only ML entrypoint (YAML-driven).
  - Alternative: keep `cross_site.py`/`few_shot.py` as thin wrappers around the unified pipeline, but they must not implement their own loaders.

### Phase 4 — Update configs + tests
- Standardize YAML keys across scripts (recommended):
  - `data_root` (or `data_path`) everywhere
  - `atlases`, `strategies`, `gsr_options`, `fc_types` everywhere
- Convert/rename old configs to match.
- Ensure minimal tests still run (at least `benchmarking/test_icc_computation.py`).

## 5) File-by-File Action Map (Benchmarking Folder)

**Delete**
- `benchmarking/data_utils.py` (legacy China session hacks + obsolete loaders)
- `benchmarking/metrics.py` (legacy, depends on `benchmarking/data_utils.py`)

**Move to `data_utils/`**
- `benchmarking/project.py` → `data_utils/paths.py` (rewrite to DataDescription-only)
- `benchmarking/timeseries.py` → `data_utils/timeseries.py` (fix glasso path)
- `benchmarking/fc.py` → `data_utils/fc.py`
- `benchmarking/hcpex_preprocess.py` → `data_utils/hcpex.py`
- `benchmarking/create_hcpex_mask.py` → `data_utils/preprocessing/create_hcpex_mask.py`
- `benchmarking/coverage_bad_rois.py` → `data_utils/coverage.py` or `data_utils/preprocessing/coverage_bad_rois.py`
- `benchmarking/aggregate_ihb.py` → `data_utils/preprocessing/aggregate_ihb.py`
- `benchmarking/aggregate_china.py` → `data_utils/preprocessing/aggregate_china.py`
- `benchmarking/precompute_glasso.py` → `data_utils/preprocessing/precompute_glasso.py`
- `benchmarking/icc_data_preparation.py` → `data_utils/preprocessing/icc_data_preparation.py`
- `benchmarking/sort_glasso.py` → delete or move to `data_utils/preprocessing/migrations/`

**Keep in `benchmarking/` (but re-import from `data_utils/`)**
- `benchmarking/icc.py`
- `benchmarking/qc_fc.py`
- ML:
  - `benchmarking/pipeline.py` → `benchmarking/ml/pipeline.py` (preferred move)
  - `benchmarking/stats.py` → `benchmarking/ml/stats.py`
  - `benchmarking/pipeline_comparisons.py` → `benchmarking/ml/pipeline_comparisons.py`
  - `benchmarking/tangent_leakage.py` → `benchmarking/ml/tangent_leakage.py`

**Evaluate / likely delete after ML unification**
- `benchmarking/cross_site.py` (legacy session handling + legacy paths)
- `benchmarking/few_shot.py` (depends on `cross_site.py` loaders)
- `benchmarking/pipeline.py` (keep as the unified entrypoint, but move under `benchmarking/ml/`)

## 6) Markdown Docs Audit (Keep / Update / Archive/Delete)

**Keep (authoritative)**
- `readme.md` (main “how to run” + outputs overview)
- `DataDescription.md` (canonical data contract)
- `PipelineComparisons.md` (paired testing rationale; still useful)
- `HCPEX_CHANGES_SUMMARY.md` (useful, mostly correct)

**Update (content correct direction but out of sync)**
- `ICCcomputation.md`
  - references `benchmarking/icc_computation.py` (doesn’t exist)
  - masking description is outdated (now based on percentile of `edge_means`)
- `HCPEX_GLASSO_GUIDE.md`
  - update output folder layout to `glasso_precomputed_fc/{site}/{atlas}/...`
- `.claude/skills/data_handling.md`
  - currently mixes legacy FC folders with the new `timeseries_*` layout; should match `DataDescription.md` or be removed

**Archive/Delete (internal notes / duplicates / stale)**
- `Plan.md` (superseded by this plan + current code state)
- `MLRefactoring.md` (pipeline is already implemented; archive)
- `CLAUDE.md` (developer-only; also references legacy modules; replace with short `CONTRIBUTING.md` if needed)
- `StudyDescription.md` (high-level narrative duplicate; not a contract)
- `Reviewers.md` (internal revision notes)
- `TangentEval.md` (paths/assumptions are legacy; keep only if updated to current script + data layout)

## 7) Definition of Done (What “Clear and Transparent” Means)

1. `benchmarking/` contains only:
   - `icc.py`, `qc_fc.py`, and ML entrypoint(s)
2. All preprocessing scripts live under `data_utils/` (and are optional to run).
3. No file/path logic references removed layouts (`fc_data_*`, `open2/open3`, `glasso_output/`).
4. `readme.md` and `DataDescription.md` fully match the code and outputs.
5. Outdated docs are either updated or moved under `docs/archive/`.

