# grid_search_mcmc.py

import os
import itertools
from copy import deepcopy

import yaml
import pandas as pd
import arviz as az

from step1_build_subject_matrices import SubjectAggregationConfig, run_step1_subject_aggregation
from step2_hierarchical_model import HierarchicalModelConfig, run_step2_hierarchical

def main():
    yaml_path = os.path.join("yaml", "parameter.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)

    cfg1 = SubjectAggregationConfig(**cfg_all["step1_merge_data"])
    run_step1_subject_aggregation(cfg1)
    cfg2 = HierarchicalModelConfig(**cfg_all["step2_hierarchical"])
    
    base_cfg = cfg2
    base_output_dir = base_cfg.output_dir

    draws_tunes_grid = [(1000, 1000), (1000, 2000), (2000, 2000), (2000, 4000), (4000, 4000), (4000, 8000), (8000, 8000)]         
    chains_grid = [2, 4, 8]
    target_accept_grid = [0.8, 0.9, 0.95]

    for (draws, tune), chains, tacc in itertools.product(
        draws_tunes_grid, chains_grid, target_accept_grid
    ):
        cfg = deepcopy(base_cfg)

        out_subdir = f"grid_d{draws}_t{tune}_c{chains}_ta{str(tacc).replace('.', 'p')}"
        cfg.output_dir = os.path.join(base_output_dir, out_subdir)

        cfg.mcmc_draws = draws
        cfg.mcmc_tune = tune
        cfg.mcmc_chains = chains
        cfg.mcmc_target_accept = float(tacc)

        print(
            f"\n=== Running step2 with draws={draws}, tune={tune}, "
            f"chains={chains}, target_accept={tacc} ==="
        )

        idata, outs = run_step2_hierarchical(cfg)


if __name__ == "__main__":
    main()