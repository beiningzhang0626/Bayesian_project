import argparse
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot MAE (sum_abs_error / n_tasks) vs subsampling ratio for multiple models, "
                    "one subplot per model, plus FLOPs vs subsampling ratio."
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        required=True,
        help="List of subsampling ratios (e.g. 0.05 0.1 0.2).",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        required=True,
        help="List of CSV files corresponding to the given ratios.",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        required=True,
        help="Number of tasks/subjects over which sum_abs_error was computed.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="subsample_vs_mae.png",
        help="Output filename for the plot (default: subsample_vs_mae.png).",
    )
    parser.add_argument(
        "--model-filter",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional list of model names to include "
            "(e.g. 'Qwen/Qwen3-32B' 'allenai/OLMo-2-0325-32B'). "
            "If omitted, all models present in the CSVs are plotted."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ratios = args.ratios
    files = args.files
    n_tasks = args.n_tasks

    if len(ratios) != len(files):
        raise ValueError("The number of ratios must match the number of files.")

    model_mae = defaultdict(list)

    for r, path in zip(ratios, files):
        df = pd.read_csv(path)
        if "model" not in df.columns or "sum_abs_error" not in df.columns:
            raise ValueError(
                f"{path} must contain 'model' and 'sum_abs_error' columns."
            )

        if args.model_filter is not None:
            df = df[df["model"].isin(args.model_filter)]

        for _, row in df.iterrows():
            model = row["model"]
            sae = float(row["sum_abs_error"])
            mae = sae / n_tasks
            model_mae[model].append((r, mae))

    if not model_mae:
        raise ValueError("No model data found after applying filters.")

    models = sorted(model_mae.keys())
    n_models = len(models)

    HARDCODED_TOTAL_FLOPS = [
        2.399e+17,
        2.399e+17,
        2.024e+17,
        5.248e+17
    ]

    if HARDCODED_TOTAL_FLOPS and len(HARDCODED_TOTAL_FLOPS) != n_models:
        raise ValueError(
            f"HARDCODED_TOTAL_FLOPS length ({len(HARDCODED_TOTAL_FLOPS)}) "
            f"does not match number of models ({n_models})."
        )

    # Map each model to its total FLOPs (if provided)
    model_to_total_flops = {}
    if HARDCODED_TOTAL_FLOPS:
        for m, fl in zip(models, HARDCODED_TOTAL_FLOPS):
            model_to_total_flops[m] = fl

    # Fix a unique color per model for MAE
    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {m: base_colors[i % len(base_colors)] for i, m in enumerate(models)}

    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(4 * n_models, 4),
        sharey=False,
    )

    if n_models == 1:
        axes = [axes]

    # NOTE: we enumerate to know which subplot (leftmost/rightmost) we're on
    for idx, (ax, model) in enumerate(zip(axes, models)):
        pairs = model_mae[model]
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        x = [p[0] for p in pairs_sorted]
        y_mae = [p[1] for p in pairs_sorted]

        # MAE curve (left y-axis)
        line_mae, = ax.plot(
            x,
            y_mae,
            marker="o",
            color=color_map[model],
            label="MAE",
        )
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("Subsampling ratio")
        ax.grid(True, linestyle="--", alpha=0.5)

        # FLOPs curve (right y-axis), only if we have a hard-coded value for this model
        if model in model_to_total_flops:
            total_flops_full = model_to_total_flops[model]
            y_flops = [total_flops_full * r for r in x]

            ax2 = ax.twinx()
            line_flops, = ax2.plot(
                x,
                y_flops,
                marker="s",
                linestyle="--",
                label="FLOPs (full_eval * ratio)",
            )

            # Only leftmost subplot gets FLOPs y-axis label
            if idx == 0:
                ax.set_ylabel("MAE")
            else:
                ax.set_ylabel("")
            if idx == n_models - 1:
                ax2.set_ylabel("FLOPs (1e17)")
            else:
                ax2.set_ylabel("")

            # Combined legend on left axis
            ax.legend(
                [line_mae, line_flops],
                ["MAE", "FLOPs"],
                fontsize=8,
                loc="best",
            )
        else:
            # Only MAE legend if no FLOPs given
            ax.legend([line_mae], ["MAE"], fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()