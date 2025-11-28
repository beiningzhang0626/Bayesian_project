#!/usr/bin/env python3

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


def clean_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def read_counts(p_tot: str, p_cor: str) -> pd.DataFrame:
    df_t = pd.read_csv(p_tot)
    df_c = pd.read_csv(p_cor)

    assert (df_t["model"] == df_c["model"]).all(), "model rows mismatch"
    assert list(df_t.columns) == list(df_c.columns), "column mismatch"

    cols = [c for c in df_t.columns if c != "model"]
    t_long = df_t.melt(id_vars="model", value_vars=cols, var_name="subject", value_name="total_q")
    c_long = df_c.melt(id_vars="model", value_vars=cols, var_name="subject", value_name="num_correct")

    df = pd.merge(t_long, c_long, on=["model", "subject"], how="inner")
    df["acc_true"] = df["num_correct"] / df["total_q"].replace(0, np.nan)
    return df


def eval_pred(df: pd.DataFrame) -> None:
    y_true = df["acc_true"].values
    y_pred = df["pred_mean"].values

    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1])

    inside = (
        (df["acc_true"] >= df["pred_lower"])
        & (df["acc_true"] <= df["pred_upper"])
    ).mean()

    print("MAE :", f"{mae:.4f}")
    print("MSE :", f"{mse:.4f}")
    print("corr:", f"{corr:.4f}")
    print("cov :", f"{inside:.4f}")
    print("n   :", len(df))


def plot_per_model(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    models = sorted(df["model"].unique())

    for m in models:
        sub = df[df["model"] == m].copy().sort_values("subject")
        subs = sub["subject"].tolist()
        x = np.arange(len(subs))

        mu = sub["pred_mean"].values
        lo = sub["pred_lower"].values
        hi = sub["pred_upper"].values
        yerr = np.vstack([mu - lo, hi - mu])

        y_true = sub["acc_true"].values

        plt.figure(figsize=(max(8, len(subs) * 0.25), 4))
        plt.errorbar(x, mu, yerr=yerr, fmt="o", capsize=3, label="pred")
        plt.scatter(x, y_true, marker="x", s=40, label="true")

        plt.xticks(x, subs, rotation=90, fontsize=6)
        plt.ylabel("acc")
        plt.xlabel("subject")
        plt.ylim(0.0, 1.0)
        plt.title(m)
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(out_dir, f"{clean_name(m)}_acc_comparison.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print("saved per-model:", fname)


def plot_overall(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    y_true = df["acc_true"].values
    y_pred = df["pred_mean"].values

    slope, inter, r, p, se = linregress(y_true, y_pred)
    x_line = np.linspace(0, 1, 100)
    y_line = inter + slope * x_line

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.5, label="pairs")
    plt.plot(x_line, x_line, linestyle="--", label="y=x")
    plt.plot(x_line, y_line, label=f"fit y={slope:.2f}x+{inter:.2f}")
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.title(f"overall (r={r:.3f})")
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(out_dir, "overall_true_vs_pred.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print("saved overall:", fname)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total_csv",default="/scratch/dkhasha1/bzhang90/Bayesian_project/results/mmlu_subject_total_questions_by_model.csv")
    p.add_argument("--correct_csv",default="/scratch/dkhasha1/bzhang90/Bayesian_project/results/mmlu_subject_num_correct_by_model.csv")
    p.add_argument("--pred_csv",default="/scratch/dkhasha1/bzhang90/Bayesian_project/results/predicted_acc.csv")
    p.add_argument("--out_dir",default="/scratch/dkhasha1/bzhang90/Bayesian_project/results/plots")
    args = p.parse_args()

    df_true = read_counts(args.total_csv, args.correct_csv)
    df_pred = pd.read_csv(args.pred_csv)

    df = pd.merge(df_true, df_pred, on=["model", "subject"], how="inner", validate="one_to_one")

    eval_pred(df)
    plot_per_model(df, args.out_dir)
    plot_overall(df, args.out_dir)


if __name__ == "__main__":
    main()
