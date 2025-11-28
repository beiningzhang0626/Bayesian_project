# step4_hierarchical_eval_plots.py
#!/usr/bin/env python3

import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


@dataclass
class HierarchicalPlotConfig:
    total_csv: str
    correct_csv: str
    pred_counts_csv: str
    out_dir: str

    model_column: str = "model"
    overall_scatter_filename: str = "overall_true_vs_pred.png"
    per_model_pattern: str = "{model}_acc_comparison.png"

    make_per_model_plots: bool = True
    make_overall_scatter: bool = True


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def read_wide_counts(total_path: str, correct_path: str, model_col: str = "model") -> pd.DataFrame:
    df_t = pd.read_csv(total_path)
    df_c = pd.read_csv(correct_path)

    if model_col not in df_t.columns or model_col not in df_c.columns:
        raise ValueError(f"CSVs must have '{model_col}' column")

    assert (df_t[model_col] == df_c[model_col]).all(), "model rows mismatch"
    assert list(df_t.columns) == list(df_c.columns), "column mismatch"

    val_cols = [c for c in df_t.columns if c != model_col]

    t_long = df_t.melt(id_vars=model_col, value_vars=val_cols, var_name="subject", value_name="total_q")
    c_long = df_c.melt(id_vars=model_col, value_vars=val_cols, var_name="subject", value_name="num_correct")

    df = pd.merge(t_long, c_long, on=[model_col, "subject"], how="inner")
    df["acc_true"] = df["num_correct"] / df["total_q"].replace(0, np.nan)
    return df


def read_pred_counts(pred_counts_csv: str, total_counts_csv: str, model_col: str = "model") -> pd.DataFrame:
    df_p = pd.read_csv(pred_counts_csv)
    if model_col not in df_p.columns:
        raise ValueError(f"prediction CSV must have '{model_col}' column")

    p_cols = [c for c in df_p.columns if c != model_col]
    p_long = df_p.melt(id_vars=model_col, value_vars=p_cols, var_name="subject", value_name="pred_correct")

    df_t = pd.read_csv(total_counts_csv)
    if model_col not in df_t.columns:
        raise ValueError(f"total CSV must have '{model_col}' column")

    subj_cols = [c for c in df_t.columns if c != model_col]
    ref = df_t.iloc[0]
    subj_to_total = {s: int(ref[s]) for s in subj_cols}

    p_long["total_q"] = p_long["subject"].map(subj_to_total).astype(int)
    p_long["pred_acc"] = p_long["pred_correct"] / p_long["total_q"].replace(0, np.nan)
    return p_long


def evaluate_predictions(df: pd.DataFrame) -> dict:
    y_true = df["acc_true"].values
    y_pred = df["pred_acc"].values

    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1])

    print("metrics")
    print(f"MAE  = {mae:.4f}")
    print(f"MSE  = {mse:.4f}")
    print(f"corr = {corr:.4f}")
    print(f"n    = {len(df)}")

    return {"mae": mae, "mse": mse, "pearson_corr": corr, "num_points": int(len(df))}


def plot_per_model(df: pd.DataFrame, out_dir: str, model_col: str = "model", pattern: str = "{model}_acc_comparison.png"):
    os.makedirs(out_dir, exist_ok=True)
    models = sorted(df[model_col].unique())

    for m in models:
        sub = df[df[model_col] == m].copy().sort_values("subject")
        subs = sub["subject"].tolist()
        x = np.arange(len(subs))

        y_true = sub["acc_true"].values
        y_pred = sub["pred_acc"].values

        plt.figure(figsize=(max(8, len(subs) * 0.25), 4))
        plt.plot(x, y_pred, "o-", label="pred")
        plt.scatter(x, y_true, marker="x", s=40, label="true")

        plt.xticks(x, subs, rotation=90, fontsize=6)
        plt.ylabel("accuracy")
        plt.xlabel("subject")
        plt.ylim(0.0, 1.0)
        plt.title(f"{m}")
        plt.legend()
        plt.tight_layout()

        fname = pattern.format(model=sanitize_filename(str(m)))
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print("saved per-model:", out_path)


def plot_overall_scatter(df: pd.DataFrame, out_dir: str, filename: str = "overall_true_vs_pred.png"):
    os.makedirs(out_dir, exist_ok=True)

    y_true = df["acc_true"].values
    y_pred = df["pred_acc"].values

    slope, intercept, r, p, stderr = linregress(y_true, y_pred)
    x_line = np.linspace(0, 1, 100)
    y_line = intercept + slope * x_line

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.5, label="pairs")
    plt.plot(x_line, x_line, linestyle="--", label="y=x")
    plt.plot(x_line, y_line, label=f"fit y={slope:.2f}x+{intercept:.2f}")
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.title(f"overall (r={r:.3f})")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("saved overall:", out_path)


def run_step4_hierarchical_plots(cfg: HierarchicalPlotConfig) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)

    df_true = read_wide_counts(
        total_path=cfg.total_csv,
        correct_path=cfg.correct_csv,
        model_col=cfg.model_column,
    )

    df_pred = read_pred_counts(
        pred_counts_csv=cfg.pred_counts_csv,
        total_counts_csv=cfg.total_csv,
        model_col=cfg.model_column,
    )

    df_m = pd.merge(
        df_true,
        df_pred[[cfg.model_column, "subject", "pred_correct", "pred_acc"]],
        on=[cfg.model_column, "subject"],
        how="inner",
        validate="one_to_one",
    )

    print("[step4] merged rows:", len(df_m))

    metrics = evaluate_predictions(df_m)

    if cfg.make_per_model_plots:
        plot_per_model(
            df_m,
            out_dir=cfg.out_dir,
            model_col=cfg.model_column,
            pattern=cfg.per_model_pattern,
        )

    if cfg.make_overall_scatter:
        plot_overall_scatter(
            df_m,
            out_dir=cfg.out_dir,
            filename=cfg.overall_scatter_filename,
        )

    return metrics


if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        raise SystemExit("install pyyaml or call run_step4_hierarchical_plots() in code")

    yaml_path = "project.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)

    cfg_step = HierarchicalPlotConfig(**cfg_all["step4_hierarchical_plots"])
    run_step4_hierarchical_plots(cfg_step)
