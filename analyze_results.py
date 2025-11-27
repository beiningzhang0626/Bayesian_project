#!/usr/bin/env python3
"""
Compare Bayesian predicted accuracies with MMLU ground-truth accuracies.

Inputs
------
1) Ground-truth count CSVs (wide format, one row per model):
   - mmlu_subject_total_questions_by_model.csv
   - mmlu_subject_num_correct_by_model.csv

   Columns:
   model, abstract_algebra, anatomy, ..., world_religions

2) Prediction CSV (long format) from Bayesian script:
   - predicted_acc.csv

   Columns:
   model, subject, pred_mean, pred_lower, pred_upper

Outputs
-------
- metrics printed to stdout (MAE, MSE, correlation, CI coverage)
- per-model comparison plots in out_dir:
    <sanitized_model_name>_acc_comparison.png
- overall scatter + regression line:
    overall_true_vs_pred.png
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def read_wide_counts(total_path: str, correct_path: str) -> pd.DataFrame:
    """Read wide-format count CSVs and return long-format with true accuracies.

    Returns columns: model, subject, total_q, num_correct, acc_true
    """
    df_total = pd.read_csv(total_path)
    df_correct = pd.read_csv(correct_path)

    # Sanity: same models and columns
    assert (df_total["model"] == df_correct["model"]).all(), "Model rows mismatch"
    assert list(df_total.columns) == list(df_correct.columns), "Columns mismatch"

    value_cols = [c for c in df_total.columns if c != "model"]

    # Melt to long format
    df_t_long = df_total.melt(
        id_vars="model", value_vars=value_cols, var_name="subject", value_name="total_q"
    )
    df_c_long = df_correct.melt(
        id_vars="model",
        value_vars=value_cols,
        var_name="subject",
        value_name="num_correct",
    )

    df = pd.merge(df_t_long, df_c_long, on=["model", "subject"], how="inner")
    df["acc_true"] = df["num_correct"] / df["total_q"].replace(0, np.nan)
    return df


def evaluate_predictions(df_merged: pd.DataFrame):
    """Print global metrics comparing pred_mean to acc_true."""
    y_true = df_merged["acc_true"].values
    y_pred = df_merged["pred_mean"].values

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    # CI coverage: fraction of true acc inside [pred_lower, pred_upper]
    inside = (
        (df_merged["acc_true"] >= df_merged["pred_lower"])
        & (df_merged["acc_true"] <= df_merged["pred_upper"])
    ).mean()

    print("=== Global Prediction Metrics ===")
    print(f"MAE:            {mae:.4f}")
    print(f"MSE:            {mse:.4f}")
    print(f"Pearson corr:   {corr:.4f}")
    print(f"95% CI coverage:{inside:.4f}")
    print(f"Num points:     {len(df_merged)}")


def plot_per_model(df_merged: pd.DataFrame, out_dir: str):
    """For each model, plot predicted acc (mean ± CI) vs true acc across subjects."""

    os.makedirs(out_dir, exist_ok=True)
    models = sorted(df_merged["model"].unique())

    for model in models:
        sub = df_merged[df_merged["model"] == model].copy()
        sub = sub.sort_values("subject")

        subjects = sub["subject"].tolist()
        x = np.arange(len(subjects))

        pred_mean = sub["pred_mean"].values
        pred_lower = sub["pred_lower"].values
        pred_upper = sub["pred_upper"].values
        yerr_lower = pred_mean - pred_lower
        yerr_upper = pred_upper - pred_mean
        yerr = np.vstack([yerr_lower, yerr_upper])

        acc_true = sub["acc_true"].values

        plt.figure(figsize=(max(8, len(subjects) * 0.25), 4))
        # Pred with CI
        plt.errorbar(
            x,
            pred_mean,
            yerr=yerr,
            fmt="o",
            capsize=3,
            label="Predicted acc (mean ± 95% CI)",
        )
        # Ground truth
        plt.scatter(x, acc_true, marker="x", s=40, label="Ground truth acc")

        plt.xticks(x, subjects, rotation=90, fontsize=6)
        plt.ylabel("Accuracy")
        plt.xlabel("Subject")
        plt.ylim(0.0, 1.0)
        plt.title(f"Accuracy comparison per subject – {model}")
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(
            out_dir, f"{sanitize_filename(model)}_acc_comparison.png"
        )
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved per-model comparison plot: {fname}")


def plot_overall_scatter(df_merged: pd.DataFrame, out_dir: str):
    """Scatter of true vs predicted acc with fitted regression line."""
    os.makedirs(out_dir, exist_ok=True)
    y_true = df_merged["acc_true"].values
    y_pred = df_merged["pred_mean"].values

    slope, intercept, r, p, stderr = linregress(y_true, y_pred)
    x_line = np.linspace(0, 1, 100)
    y_line = intercept + slope * x_line

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.5, label="(subject, model) pairs")
    plt.plot(x_line, x_line, linestyle="--", label="y = x (perfect)")
    plt.plot(x_line, y_line, label=f"Fit: y={slope:.2f}x+{intercept:.2f}")
    plt.xlabel("True accuracy")
    plt.ylabel("Predicted accuracy (mean)")
    plt.title(f"Overall true vs predicted acc (r={r:.3f})")
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(out_dir, "overall_true_vs_pred.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved overall scatter plot: {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total_csv",
        default="/scratch/dkhasha1/bzhang90/Bayesian_project/results/mmlu_subject_total_questions_by_model.csv",
    )
    parser.add_argument(
        "--correct_csv",
        default="/scratch/dkhasha1/bzhang90/Bayesian_project/results/mmlu_subject_num_correct_by_model.csv",
    )
    parser.add_argument(
        "--pred_csv",
        default="/scratch/dkhasha1/bzhang90/Bayesian_project/results/predicted_acc.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="/scratch/dkhasha1/bzhang90/Bayesian_project/results/plots",
    )
    args = parser.parse_args()

    # 1) True accuracies
    df_true = read_wide_counts(args.total_csv, args.correct_csv)

    # 2) Predictions
    df_pred = pd.read_csv(args.pred_csv)

    # Merge on (model, subject)
    df_merged = pd.merge(
        df_true, df_pred, on=["model", "subject"], how="inner", validate="one_to_one"
    )

    # 3) Metrics
    evaluate_predictions(df_merged)

    # 4) Plots
    plot_per_model(df_merged, args.out_dir)
    plot_overall_scatter(df_merged, args.out_dir)


if __name__ == "__main__":
    main()
