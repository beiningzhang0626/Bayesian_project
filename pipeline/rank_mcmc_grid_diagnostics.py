#!/usr/bin/env python
import os
import re
import argparse

import numpy as np
import pandas as pd

CSV_NAME = "llm_hierarchical_diagnostics_per_model.csv"

RE_GRID = re.compile(
    r"grid_d(?P<draws>\d+)_t(?P<tune>\d+)_c(?P<chains>\d+)_ta(?P<ta>[\dp]+)"
)

def parse_grid_params(dir_name: str):
    m = RE_GRID.search(dir_name)
    if not m:
        return {"draws": None, "tune": None, "chains": None, "target_accept": None}

    draws = int(m.group("draws"))
    tune = int(m.group("tune"))
    chains = int(m.group("chains"))
    ta_str = m.group("ta")
    ta = float(ta_str.replace("p", "."))
    return {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "target_accept": ta,
    }


def summarize_diagnostics(csv_path: str):
    df = pd.read_csv(csv_path)

    required_cols = {
        "theta_rhat",
        "theta_ess_bulk",
        "theta_ess_tail",
        "p_max_rhat",
        "p_min_ess_bulk",
        "p_min_ess_tail",
    }
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"{csv_path} is missing required columns: "
            f"{required_cols - set(df.columns)}"
        )

    max_theta_rhat = df["theta_rhat"].max()
    max_p_rhat = df["p_max_rhat"].max()

    min_theta_ess_bulk = df["theta_ess_bulk"].min()
    min_theta_ess_tail = df["theta_ess_tail"].min()
    min_p_ess_bulk = df["p_min_ess_bulk"].min()
    min_p_ess_tail = df["p_min_ess_tail"].min()

    return {
        "max_theta_rhat": float(max_theta_rhat),
        "max_p_rhat": float(max_p_rhat),
        "min_theta_ess_bulk": float(min_theta_ess_bulk),
        "min_theta_ess_tail": float(min_theta_ess_tail),
        "min_p_ess_bulk": float(min_p_ess_bulk),
        "min_p_ess_tail": float(min_p_ess_tail),
    }


def collect_runs(root_dir: str):
    rows = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if CSV_NAME not in filenames:
            continue

        csv_path = os.path.join(dirpath, CSV_NAME)
        rel_dir = os.path.relpath(dirpath, root_dir)
        base = os.path.basename(dirpath)

        try:
            diag = summarize_diagnostics(csv_path)
        except Exception as e:
            print(f"[WARN] Skipping {csv_path}: {e}")
            continue

        params = parse_grid_params(base)

        row = {
            "run_dir": rel_dir,
            "csv_path": csv_path,
            **params,
            **diag,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Rank grid search iterations by MCMC diagnostics."
    )
    parser.add_argument(
        "root_dir",
        help="Base directory containing grid output subdirectories.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of best runs to print (default: 10; use -1 for all).",
    )
    args = parser.parse_args()

    df = collect_runs(args.root_dir)

    if df.empty:
        print("No diagnostics CSVs found.")
        return

    df_sorted = df.sort_values(
        by=[
            "max_theta_rhat",
            "max_p_rhat",
            "min_theta_ess_bulk",
            "min_theta_ess_tail",
            "min_p_ess_bulk",
            "min_p_ess_tail",
        ],
        ascending=[True, True, False, False, False, False],
    )

    if args.top > 0:
        df_show = df_sorted.head(args.top)
    else:
        df_show = df_sorted

    # Nice compact view
    cols_to_print = [
        "run_dir",
        "draws",
        "tune",
        "chains",
        "target_accept",
        "max_theta_rhat",
        "max_p_rhat",
        "min_theta_ess_bulk",
        "min_theta_ess_tail",
        "min_p_ess_bulk",
        "min_p_ess_tail",
    ]
    cols_to_print = [c for c in cols_to_print if c in df_show.columns]

    print("\nBest→worst grid runs by convergence diagnostics:\n")
    print(
        df_show[cols_to_print].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


if __name__ == "__main__":
    main()