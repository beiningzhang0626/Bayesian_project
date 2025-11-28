# run_code.py

import argparse
import yaml

from step1_build_subject_matrices import SubjectAggregationConfig, run_step1_subject_aggregation
from step2_hierarchical_model import HierarchicalModelConfig, run_step2_hierarchical
from step3_compare_and_error_summary import HierarchicalCompareConfig, run_step3_hierarchical_compare
from step4_hierarchical_eval_plots import HierarchicalPlotConfig, run_step4_hierarchical_plots


def main():
    p = argparse.ArgumentParser(description="Run Bayesian pipeline.")
    p.add_argument(
        "--config",
        default=r"D:\Bayesian_project\pipeline\yaml\parameter.yaml",
        help="YAML config path",
    )
    p.add_argument(
        "--stop_after",
        choices=["none", "step1", "step2", "step3", "step4"],
        default="step4",
        help="Stop after this step",
    )
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)

    cfg1 = SubjectAggregationConfig(**cfg_all["step1_merge_data"])
    cfg2 = HierarchicalModelConfig(**cfg_all["step2_hierarchical"])
    cfg3 = HierarchicalCompareConfig(**cfg_all["step3_hierarchical_compare"])
    cfg4 = HierarchicalPlotConfig(**cfg_all["step4_hierarchical_plots"])

    print("step1")
    run_step1_subject_aggregation(cfg1)
    if args.stop_after == "step1":
        print("stopped at step1")
        return

    print("step2")
    run_step2_hierarchical(cfg2)
    if args.stop_after == "step2":
        print("stopped at step2")
        return

    print("step3")
    run_step3_hierarchical_compare(cfg3)
    if args.stop_after == "step3":
        print("stopped at step3")
        return

    print("step4")
    run_step4_hierarchical_plots(cfg4)

    print("all done")


if __name__ == "__main__":
    main()
