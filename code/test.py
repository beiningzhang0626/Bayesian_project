import pandas as pd

# ======== CHANGE TO YOUR OWN PATHS =========
TRUE_CSV = r"D:\Bayesian_project\results\mmlu_subject_num_correct_by_model.csv"          # Original / true correct counts
PRED_CSV = r"D:\Bayesian_project\output\linear\linear_reg_baseline_pred_correct_counts.csv"  # Predicted correct counts (e.g. from linear regression baseline)
OUT_CSV  = r"D:\Bayesian_project\output\compare\result_compare_linear.csv"               # Output comparison results
# ==========================================

# Read CSVs
df_true = pd.read_csv(TRUE_CSV)
df_pred = pd.read_csv(PRED_CSV)

# Ensure both have a 'model' column
if "model" not in df_true.columns:
    raise ValueError(f"No 'model' column in {TRUE_CSV}")
if "model" not in df_pred.columns:
    raise ValueError(f"No 'model' column in {PRED_CSV}")

# Keep only models that appear in both CSVs (intersection)
models_true = set(df_true["model"])
models_pred = set(df_pred["model"])
common_models = sorted(models_true & models_pred)

if not common_models:
    raise ValueError("No common model names in the two CSVs, please check the files.")

# Restrict to the intersection and sort by model
df_true_sub = df_true[df_true["model"].isin(common_models)].set_index("model").loc[common_models]
df_pred_sub = df_pred[df_pred["model"].isin(common_models)].set_index("model").loc[common_models]

# Ensure column order (except 'model') is the same: reorder pred to follow true's column order
task_cols = [c for c in df_true_sub.columns]   # 'model' is already the index
df_pred_sub = df_pred_sub[task_cols]

# Build a new DataFrame: two rows per model: model, model_predict
rows = []
for m in common_models:
    # First row: true values
    row_true = df_true_sub.loc[m].copy()
    row_true.name = m                      # index = 'model_name'
    rows.append(row_true)

    # Second row: predicted values
    row_pred = df_pred_sub.loc[m].copy()
    row_pred.name = f"{m}_predict"         # index = 'model_name_predict'
    rows.append(row_pred)

# Concatenate into one big DataFrame
df_compare = pd.DataFrame(rows)

# Turn index back into a column named 'model'
df_compare.insert(0, "model", df_compare.index)
df_compare.reset_index(drop=True, inplace=True)

# Save
df_compare.to_csv(OUT_CSV, index=False)
print(f"Saved comparison results to: {OUT_CSV}")
