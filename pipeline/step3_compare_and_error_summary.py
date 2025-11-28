# step3_compare_and_error_summary.py

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class HierarchicalCompareConfig:
    true_csv: str
    pred_csv: str
    out_compare_csv: str
    out_error_csv: str
    model_column: str = "model"
    predict_suffix: str = "_hier"


@dataclass
class CompareOutputs:
    compare_csv_path: str
    error_summary_csv_path: str


def build_compare_dataframe(cfg: HierarchicalCompareConfig) -> pd.DataFrame:
    df_true = pd.read_csv(cfg.true_csv)
    df_pred = pd.read_csv(cfg.pred_csv)

    if cfg.model_column not in df_true.columns:
        raise ValueError(f"no '{cfg.model_column}' column in {cfg.true_csv}")
    if cfg.model_column not in df_pred.columns:
        raise ValueError(f"no '{cfg.model_column}' column in {cfg.pred_csv}")

    models_true = set(df_true[cfg.model_column])
    models_pred = set(df_pred[cfg.model_column])
    common = sorted(models_true & models_pred)

    if not common:
        raise ValueError("no common models between true_csv and pred_csv")

    df_true_sub = (df_true[df_true[cfg.model_column].isin(common)].set_index(cfg.model_column).loc[common])
    df_pred_sub = (df_pred[df_pred[cfg.model_column].isin(common)].set_index(cfg.model_column).loc[common])

    task_cols: List[str] = list(df_true_sub.columns)
    df_pred_sub = df_pred_sub[task_cols]

    rows = []
    for m in common:
        r_true = df_true_sub.loc[m].copy()
        r_true.name = m
        rows.append(r_true)

        r_pred = df_pred_sub.loc[m].copy()
        r_pred.name = f"{m}{cfg.predict_suffix}"
        rows.append(r_pred)

    df_cmp = pd.DataFrame(rows)
    df_cmp.insert(0, cfg.model_column, df_cmp.index)
    df_cmp.reset_index(drop=True, inplace=True)
    return df_cmp


def compute_error_summary(df_cmp: pd.DataFrame, cfg: HierarchicalCompareConfig) -> pd.DataFrame:
    if cfg.model_column not in df_cmp.columns:
        raise ValueError(f"compare df must have '{cfg.model_column}'")

    task_cols = [c for c in df_cmp.columns if c != cfg.model_column]
    df_idx = df_cmp.set_index(cfg.model_column)

    base_models = sorted(m for m in df_idx.index if not str(m).endswith(cfg.predict_suffix))
    out = []

    for base in base_models:
        pred_name = f"{base}{cfg.predict_suffix}"
        if pred_name not in df_idx.index:
            print(f"skip {base}, no pred row")
            continue

        v_true = df_idx.loc[base, task_cols].to_numpy(dtype=float)
        v_pred = df_idx.loc[pred_name, task_cols].to_numpy(dtype=float)
        diff = v_pred - v_true

        out.append(
            {
                cfg.model_column: base,
                "sum_error": float(diff.sum()),
                "sum_abs_error": float(np.abs(diff).sum()),
                "sum_squared_error": float((diff ** 2).sum()),
            }
        )

    if not out:
        raise RuntimeError("no (true, pred) pairs in compare df")

    return pd.DataFrame(out)


def run_step3_hierarchical_compare(cfg: HierarchicalCompareConfig) -> CompareOutputs:
    cmp_dir = os.path.dirname(cfg.out_compare_csv)
    err_dir = os.path.dirname(cfg.out_error_csv)
    if cmp_dir:
        os.makedirs(cmp_dir, exist_ok=True)
    if err_dir and err_dir != cmp_dir:
        os.makedirs(err_dir, exist_ok=True)

    df_cmp = build_compare_dataframe(cfg)
    df_cmp.to_csv(cfg.out_compare_csv, index=False)
    print("saved compare:", cfg.out_compare_csv)

    df_err = compute_error_summary(df_cmp, cfg)
    df_err.to_csv(cfg.out_error_csv, index=False)
    print("saved error:", cfg.out_error_csv)

    return CompareOutputs(
        compare_csv_path=cfg.out_compare_csv,
        error_summary_csv_path=cfg.out_error_csv,
    )


if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        raise SystemExit("install pyyaml or call run_step3_hierarchical_compare() in code")

    yaml_path = "project.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)

    cfg_step = HierarchicalCompareConfig(**cfg_all["step3_hierarchical_compare"])
    run_step3_hierarchical_compare(cfg_step)
