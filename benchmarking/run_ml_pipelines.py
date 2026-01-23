#!/usr/bin/env python3
"""
Run ML benchmarking pipelines based on a YAML configuration.
Iterates over atlases, strategies, and GSR options defined in the config.

Supports both batch format (atlases/strategies/gsr_options lists) 
and single-run format (pipeline dictionary).

Usage:
    python -m benchmarking.run_ml_pipelines --config configs/ml_atlas.yaml
    python -m benchmarking.run_ml_pipelines --config configs/ml_single.yaml --force
"""

import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from benchmarking.ml.pipeline import run_full_pipeline, PipelineConfig

def to_list(val):
    if val is None: return []
    return [val] if isinstance(val, str) else list(val)

def main():
    parser = argparse.ArgumentParser(description="Run ML pipelines from config")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--force', action='store_true', help='Force re-run even if results exist')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config_data = yaml.safe_load(f)

    # Determine if this is a single-pipeline config or a batch config
    is_single = 'pipeline' in config_data
    
    if is_single:
        # Single run format
        pipeline_cfg = config_data['pipeline']
        atlases = [pipeline_cfg.get('atlas', 'AAL')]
        strategies = [pipeline_cfg.get('strategy')]
        gsr_options = [pipeline_cfg.get('gsr', 'GSR')]
        defaults = pipeline_cfg
    else:
        # Batch format
        atlases = to_list(config_data.get('atlases', []))
        strategies = to_list(config_data.get('strategies', []))
        gsr_options = to_list(config_data.get('gsr_options', []))
        defaults = config_data.get('pipeline_defaults', {})
        
        # If atlases is empty, check defaults or use AAL
        if not atlases:
            atlases = [defaults.get('atlas', 'AAL')]

    # Global settings
    data_config = config_data.get('data', {})
    output_config = config_data.get('output', {})
    few_shot_config = config_data.get('few_shot', {})
    
    output_base_dir = Path(output_config.get('output_dir', 'results/ml'))

    if not strategies or strategies == [None]:
        raise ValueError("No 'strategy' found in config.")
    if not gsr_options:
        raise ValueError("No 'gsr_options' found in config.")

    # Prepare list of tasks
    tasks = []
    for atlas in atlases:
        for strategy in strategies:
            for gsr in gsr_options:
                tasks.append((atlas, strategy, gsr))

    print(f"Loaded config: {args.config} ({'Single' if is_single else 'Batch'} mode)")
    print(f"Found {len(tasks)} pipeline configurations to run.")

    # Run with progress bar
    pbar = tqdm(tasks, desc="Pipelines", unit="pipe")
    
    for atlas, strategy, gsr in pbar:
        pbar.set_description(f"Running {atlas} strategy-{strategy} {gsr}")
        
        try:
            # Check if results already exist
            pipeline_dir_name = f"{atlas}_strategy-{strategy}_{gsr}"
            pipeline_output_dir = output_base_dir / pipeline_dir_name
            summary_file = pipeline_output_dir / "summary.csv"
            
            if not args.force and summary_file.exists():
                # print(f"Skipping {pipeline_dir_name} (already processed)")
                continue

            # Construct PipelineConfig
            config = PipelineConfig(
                atlas=atlas,
                strategy=strategy,
                gsr=gsr,
                fc_types=to_list(defaults.get('fc_types', ['corr', 'partial', 'tangent', 'glasso'])),
                models=to_list(defaults.get('models', ['logreg'])),
                
                # Glasso
                skip_glasso=defaults.get('skip_glasso', False),
                precomputed_glasso_dir=defaults.get('precomputed_glasso_dir'),
                glasso_lambda=defaults.get('glasso_lambda', 0.03),
                
                # Model settings
                model_params=defaults.get('model_params', {}),
                pca_components=defaults.get('pca_components', 0.95),
                
                # Execution
                n_permutations=defaults.get('n_permutations', 0),
                random_state=defaults.get('random_state', 42),
                save_test_outputs=defaults.get('save_test_outputs', True),
                
                # Data & Output
                data_path=data_config.get('data_path'),
                coverage_threshold=data_config.get('coverage_threshold', 0.1),
                output_dir=str(output_base_dir),
                
                # Few-shot
                n_few_shot=few_shot_config.get('n_few_shot', 10),
                n_repeats=few_shot_config.get('n_repeats', 10),
                
                # Scaling (default True)
                scale=not defaults.get('no_scale', False),
            )
            
            run_full_pipeline(config)
            
        except Exception as e:
            print(f"\nERROR running {atlas} strategy-{strategy} {gsr}: {e}")
            import traceback
            traceback.print_exc()

    print("\nBatch execution complete!")

if __name__ == '__main__':
    main()
