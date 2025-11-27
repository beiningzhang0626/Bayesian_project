import pandas as pd
import numpy as np

# ====== PATHS (edit these) ====================================================
INPUT_CSV  = r"D:\Bayesian_project\output\compare\result_compare_linear.csv"   # compare file
OUTPUT_CSV = r"D:\Bayesian_project\output\compare\model_error_summary_linear.csv"
# =============================================================================


def main():
    # Load the compare CSV
    df = pd.read_csv(INPUT_CSV)

    if "model" not in df.columns:
        raise ValueError("Input CSV must contain a 'model' column.")

    # All task columns (everything except 'model')
    task_cols = [c for c in df.columns if c != "model"]

    # Use index by model name for easier lookup
    df_idx = df.set_index("model")

    # Base model names: rows whose name does NOT end with "_predict"
    base_models = sorted(
        m for m in df_idx.index
        if not str(m).endswith("_predict")
    )

    results = []

    for base in base_models:
        pred_name = f"{base}_predict"

        if pred_name not in df_idx.index:
            # If there is no predicted row for this model, skip it
            print(f"[WARNING] No predicted row found for model '{base}', skipping.")
            continue

        # True and predicted values for all tasks
        true_vals = df_idx.loc[base, task_cols].to_numpy(dtype=float)
        pred_vals = df_idx.loc[pred_name, task_cols].to_numpy(dtype=float)

        diff = pred_vals - true_vals

        sum_error = float(diff.sum())                       # signed sum of errors
        sum_abs_error = float(np.abs(diff).sum())           # sum of absolute errors
        sum_squared_error = float((diff ** 2).sum())        # sum of squared errors

        results.append({
            "model": base,
            "sum_error": sum_error,
            "sum_abs_error": sum_abs_error,
            "sum_squared_error": sum_squared_error,
        })

    if not results:
        raise RuntimeError("No (true, predict) pairs found in the input file.")

    df_errors = pd.DataFrame(results)
    df_errors.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved model error summary to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
