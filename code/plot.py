#!/usr/bin/env python3
"""
Analyze hierarchical Bayesian predictions (predicted correct counts)
against MMLU ground-truth accuracies.

Inputs
------
1) Ground-truth count CSVs (wide format, one row per model):
   - mmlu_subject_total_questions_by_model.csv
   - mmlu_subject_num_correct_by_model.csv

   Columns:
   model, abstract_algebra, anatomy, ..., world_religions

2) Prediction CSV from hierarchical script (wide, big_new_models only):
   - llm_hierarchical_predictions_counts.csv

   Columns:
   model, abstract_algebra, anatomy, ..., world_religions
   (values are E[#correct] on full questions for each subject)

Outputs
-------
- Global metrics printed to stdout (MAE, MSE, correlation)
- Per-model comparison plots in out_dir:
    <sanitized_model_name>_acc_comparison.png
- Overall scatter + regression line:
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
    """Make a model name safe to use as a filename."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def read_wide_counts(total_path: str, correct_path: str) -> pd.DataFrame:
    """
    Read wide-format count CSVs and return long-format with true accuracies.

    Returns columns:
        model, subject, total_q, num_correct, acc_true
    """
    df_total = pd.read_csv(total_path)
    df_correct = pd.read_csv(correct_path)

    # Basic sanity checks
    assert (df_total["model"] == df_correct["model"]).all(), "Model rows mismatch"
    assert list(df_total.columns) == list(df_correct.columns), "Columns mismatch"

    value_cols = [c for c in df_total.columns if c != "model"]

    # Melt to long format
    df_t_long = df_total.melt(
        id_vars="model",
        value_vars=value_cols,
        var_name="subject",
        value_name="total_q",
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


def read_pred_counts(pred_counts_csv: str, total_counts_csv: str) -> pd.DataFrame:
    """
    Read predicted correct counts (wide format) and convert to long format
    with predicted accuracies using the total question counts.

    Returns columns:
        model, subject, pred_correct, total_q, pred_acc
    """
    # Predicted correct counts (big_new_models only)
    df_pred_wide = pd.read_csv(pred_counts_csv)

    if "model" not in df_pred_wide.columns:
        raise ValueError("Prediction CSV must have a 'model' column.")

    pred_subject_cols = [c for c in df_pred_wide.columns if c != "model"]

    df_pred_long = df_pred_wide.melt(
        id_vars="model",
        value_vars=pred_subject_cols,
        var_name="subject",
        value_name="pred_correct",
    )

    # Read total counts to recover per-subject total_q
    df_total = pd.read_csv(total_counts_csv)
    if "model" not in df_total.columns:
        raise ValueError("Total-count CSV must have a 'model' column.")

    # We assume total_q per subject is the same across models.
    subject_cols = [c for c in df_total.columns if c != "model"]
    # Use the first row as reference
    ref = df_total.iloc[0]
    subject_to_total = {subj: int(ref[subj]) for subj in subject_cols}

    # Map totals to prediction rows
    df_pred_long["total_q"] = df_pred_long["subject"].map(subject_to_total).astype(int)
    df_pred_long["pred_acc"] = df_pred_long["pred_correct"] / df_pred_long["total_q"].replace(
        0, np.nan
    )

    return df_pred_long


def evaluate_predictions(df_merged: pd.DataFrame):
    """Print global metrics comparing pred_acc to acc_true."""
    y_true = df_merged["acc_true"].values
    y_pred = df_merged["pred_acc"].values

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    print("=== Global Prediction Metrics (hierarchical Bayes) ===")
    print(f"MAE:          {mae:.4f}")
    print(f"MSE:          {mse:.4f}")
    print(f"Pearson corr: {corr:.4f}")
    print(f"Num points:   {len(df_merged)}")


def plot_per_model(df_merged: pd.DataFrame, out_dir: str):
    """For each model, plot predicted acc vs true acc across subjects."""
    os.makedirs(out_dir, exist_ok=True)
    models = sorted(df_merged["model"].unique())

    for model in models:
        sub = df_merged[df_merged["model"] == model].copy()
        sub = sub.sort_values("subject")

        subjects = sub["subject"].tolist()
        x = np.arange(len(subjects))

        acc_true = sub["acc_true"].values
        acc_pred = sub["pred_acc"].values

        plt.figure(figsize=(max(8, len(subjects) * 0.25), 4))

        # Predicted accuracy (line + circles)
        plt.plot(x, acc_pred, "o-", label="Predicted accuracy")

        # Ground truth accuracy (cross markers)
        plt.scatter(x, acc_true, marker="x", s=40, label="Ground truth accuracy")

        plt.xticks(x, subjects, rotation=90, fontsize=6)
        plt.ylabel("Accuracy")
        plt.xlabel("Subject")
        plt.ylim(0.0, 1.0)
        plt.title(f"Accuracy comparison per subject – {model}")
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(out_dir, f"{sanitize_filename(model)}_acc_comparison.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved per-model comparison plot: {fname}")


def plot_overall_scatter(df_merged: pd.DataFrame, out_dir: str):
    """Scatter of true vs predicted acc with fitted regression line."""
    os.makedirs(out_dir, exist_ok=True)
    y_true = df_merged["acc_true"].values
    y_pred = df_merged["pred_acc"].values

    slope, intercept, r, p, stderr = linregress(y_true, y_pred)
    x_line = np.linspace(0, 1, 100)
    y_line = intercept + slope * x_line

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.5, label="(model, subject) pairs")
    plt.plot(x_line, x_line, linestyle="--", label="y = x (perfect)")
    plt.plot(x_line, y_line, label=f"Fit: y={slope:.2f}x+{intercept:.2f}")
    plt.xlabel("True accuracy")
    plt.ylabel("Predicted accuracy")
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
        default=r"D:\Bayesian_project\results\mmlu_subject_total_questions_by_model.csv",
        help="Path to FULL total question counts (wide format).",
    )
    parser.add_argument(
        "--correct_csv",
        default=r"D:\Bayesian_project\results\mmlu_subject_num_correct_by_model.csv",
        help="Path to FULL correct counts (wide format).",
    )
    parser.add_argument(
        "--pred_counts_csv",
        default=r"D:\Bayesian_project\output\hierarchical\llm_hierarchical_predictions_counts.csv",
        help="Path to hierarchical predicted correct counts (wide format).",
    )
    parser.add_argument(
        "--out_dir",
        default=r"D:\Bayesian_project\output\hierarchical\plots",
        help="Directory to save plots.",
    )
    args = parser.parse_args()

    # 1) True accuracies (all models, all subjects)
    df_true = read_wide_counts(args.total_csv, args.correct_csv)

    # 2) Predicted correct counts -> predicted accuracies (big_new_models only)
    df_pred = read_pred_counts(args.pred_counts_csv, args.total_csv)

    # Merge on (model, subject) – inner join restricts to models that have predictions
    df_merged = pd.merge(
        df_true,
        df_pred[["model", "subject", "pred_correct", "pred_acc"]],
        on=["model", "subject"],
        how="inner",
        validate="one_to_one",
    )

    print(f"Merged rows (model-subject pairs with predictions): {len(df_merged)}")

    # 3) Metrics
    evaluate_predictions(df_merged)

    # 4) Plots
    plot_per_model(df_merged, args.out_dir)
    plot_overall_scatter(df_merged, args.out_dir)


if __name__ == "__main__":
    main()
