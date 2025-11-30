# grid_search_subsample.py

import os
from copy import deepcopy

import yaml

from step1_build_subject_matrices import SubjectAggregationConfig, run_step1_subject_aggregation
from step2_hierarchical_model import HierarchicalModelConfig, run_step2_hierarchical
from step3_compare_and_error_summary import HierarchicalCompareConfig, run_step3_hierarchical_compare
from step4_hierarchical_eval_plots import HierarchicalPlotConfig, run_step4_hierarchical_plots


def frac_to_tag(frac: float) -> str:
    return f"ssf{frac}"


def frac_to_pct_suffix(frac: float) -> str:
    return f"{int(round(100 * frac))}pct"


def main():
    yaml_path = os.path.join("yaml", "parameter.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)

    base_cfg_step1 = SubjectAggregationConfig(**cfg_all["step1_merge_data"])
    base_cfg_step2 = HierarchicalModelConfig(**cfg_all["step2_hierarchical"])
    base_cfg_step3 = HierarchicalCompareConfig(**cfg_all["step3_hierarchical_compare"])
    base_cfg_step4 = HierarchicalPlotConfig(**cfg_all["step4_hierarchical_plots"])

    base_step1_outdir = base_cfg_step1.output_dir
    results_root = os.path.dirname(base_step1_outdir)
    subsample_grid = [0.35, 0.4, 0.45, 0.5]

    for frac in subsample_grid:
        tag = frac_to_tag(frac)
        pct_suffix = frac_to_pct_suffix(frac)

        cfg1 = deepcopy(base_cfg_step1)
        step1_outdir = os.path.join(results_root, tag)
        cfg1.output_dir = step1_outdir
        cfg1.subsample_fraction = frac

        print(f"\n=== Step 1: subsample_fraction={frac} → {step1_outdir} ===")
        run_step1_subject_aggregation(cfg1)

        cfg2 = deepcopy(base_cfg_step2)
        cfg2.input_total_csv = os.path.join(
            step1_outdir,
            f"mmlu_subject_total_questions_by_model_incomplete_{pct_suffix}.csv",
        )
        cfg2.input_correct_csv = os.path.join(
            step1_outdir,
            f"mmlu_subject_num_correct_by_model_incomplete_{pct_suffix}.csv",
        )
        cfg2.output_dir = os.path.join(step1_outdir, "hierarchical")

        print(f"=== Step 2: hierarchical model for subsample_fraction={frac} (inputs from {step1_outdir}) ===")
        _idata, _outs = run_step2_hierarchical(cfg2)

        cfg3 = deepcopy(base_cfg_step3)
        compare_dir = os.path.join(cfg2.output_dir, "compare")
        os.makedirs(compare_dir, exist_ok=True)
        cfg3.true_csv = os.path.join(
            step1_outdir,
            "mmlu_subject_num_correct_by_model.csv",
        )
        cfg3.pred_csv = os.path.join(
            cfg2.output_dir,
            "llm_hierarchical_predictions_counts.csv",
        )
        cfg3.out_compare_csv = os.path.join(
            compare_dir,
            "result_compare_hierarchical.csv",
        )
        cfg3.out_error_csv = os.path.join(
            compare_dir,
            "model_error_summary_hierarchical.csv",
        )

        print(f"=== Step 3: compare predictions for subsample_fraction={frac} ===")
        run_step3_hierarchical_compare(cfg3)

        cfg4 = deepcopy(base_cfg_step4)
        cfg4.total_csv = os.path.join(
            step1_outdir,
            "mmlu_subject_total_questions_by_model.csv",
        )
        cfg4.correct_csv = os.path.join(
            step1_outdir,
            "mmlu_subject_num_correct_by_model.csv",
        )
        cfg4.pred_counts_csv = os.path.join(
            cfg2.output_dir,
            "llm_hierarchical_predictions_counts.csv",
        )
        cfg4.out_dir = os.path.join(cfg2.output_dir, "plots")

        print(f"=== Step 4: plotting for subsample_fraction={frac} ===")
        run_step4_hierarchical_plots(cfg4)


if __name__ == "__main__":
    main()