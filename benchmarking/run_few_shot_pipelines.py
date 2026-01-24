#!/usr/bin/env python3
"""
Run ONLY Few-Shot ML benchmarking pipelines based on a YAML configuration.
"""

import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from benchmarking.ml.few_shot_pipeline import run_few_shot_only
from benchmarking.ml.pipeline import PipelineConfig

def to_list(val):
    if val is None: return []
    return [val] if isinstance(val, str) else list(val)

def main():
    parser = argparse.ArgumentParser(description="Run few-shot pipelines from config")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--force', action='store_true', help='Force re-run even if results exist')
    parser.add_argument('--repeats', type=int, help='Override number of repeats')
    args = parser.parse_args()

    with open(args.config) as f:
        config_data = yaml.safe_load(f)

    is_single = 'pipeline' in config_data
    if is_single:
        pipeline_cfg = config_data['pipeline']
        atlases = [pipeline_cfg.get('atlas', 'AAL')]
        strategies = [pipeline_cfg.get('strategy')]
        gsr_options = [pipeline_cfg.get('gsr', 'GSR')]
        defaults = pipeline_cfg
    else:
        atlases = to_list(config_data.get('atlases', []))
        strategies = to_list(config_data.get('strategies', []))
        gsr_options = to_list(config_data.get('gsr_options', []))
        defaults = config_data.get('pipeline_defaults', {})
        if not atlases: atlases = [defaults.get('atlas', 'AAL')]

    data_config = config_data.get('data', {})
    output_config = config_data.get('output', {})
    few_shot_config = config_data.get('few_shot', {})
    output_base_dir = Path(output_config.get('output_dir', 'results/ml'))

    n_repeats = args.repeats if args.repeats else few_shot_config.get('n_repeats', 10)

    tasks = []
    for atlas in atlases:
        for strategy in strategies:
            for gsr in gsr_options:
                tasks.append((atlas, strategy, gsr))

    print(f"Loaded config: {args.config}. Repeats: {n_repeats}")
    pbar = tqdm(tasks, desc="Few-Shot Pipelines", unit="pipe")
    
    for atlas, strategy, gsr in pbar:
        pbar.set_description(f"Few-Shot: {atlas} strategy-{strategy} {gsr}")
        try:
            pipeline_dir = output_base_dir / f"{atlas}_strategy-{strategy}_{gsr}"
            output_file = pipeline_dir / f"few_shot_results_n{n_repeats}.csv"
            if not args.force and output_file.exists(): continue

            config = PipelineConfig(
                atlas=atlas, strategy=strategy, gsr=gsr,
                fc_types=to_list(defaults.get('fc_types', ['corr', 'partial', 'tangent', 'glasso'])),
                models=to_list(defaults.get('models', ['logreg'])),
                skip_glasso=defaults.get('skip_glasso', False),
                precomputed_glasso_dir=defaults.get('precomputed_glasso_dir'),
                data_path=data_config.get('data_path'),
                output_dir=str(output_base_dir),
                n_few_shot=few_shot_config.get('n_few_shot', 10),
                n_repeats=n_repeats,
                scale=not defaults.get('no_scale', False),
            )
            run_few_shot_only(config)
        except Exception as e:
            print(f"\nERROR running few-shot for {atlas} {strategy} {gsr}: {e}")

    print("\nFew-shot execution complete!")

if __name__ == '__main__':
    main()
