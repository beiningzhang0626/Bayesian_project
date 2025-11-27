"""
Linear regression baseline for LLM accuracy vs model size.

Goal:
- Use a simple linear model
    acc[m, t] = alpha_t + beta_t * log(size[m]) + error
  to predict accuracy for a set of "big_new_models" on each task.

Inputs:
- TOTAL_CSV   : total number of questions per (model, task)
- CORRECT_CSV : number of correct answers per (model, task)

Outputs:
- OUTPUT_PRED_ACC_CSV    : predicted accuracy per (big_new_model, task)
- OUTPUT_PRED_COUNTS_CSV : predicted correct counts per (big_new_model, task),
                            using the original (full) number of questions per task.
"""

import numpy as np
import pandas as pd
import re

# ---------------------------------------------------------
# 0. Config: paths
# ---------------------------------------------------------

# Total number of questions n[m, t]
TOTAL_CSV = r"D:\Bayesian_project\results\mmlu_subject_total_questions_by_model_incomplete_20pct.csv"
# Number of correct answers k[m, t]
CORRECT_CSV = r"D:\Bayesian_project\results\mmlu_subject_num_correct_by_model_incomplete_20pct.csv"

# Linear regression baseline output: predicted accuracy
OUTPUT_PRED_ACC_CSV = r"D:\Bayesian_project\output\linear\linear_reg_baseline_pred_accuracy.csv"
# Linear regression baseline output: predicted correct counts (using original number of questions)
OUTPUT_PRED_COUNTS_CSV = r"D:\Bayesian_project\output\linear\linear_reg_baseline_pred_correct_counts.csv"

# Keep consistent with previous script: which models are subsampled "large models"
SUBSAMPLE_MODEL_PATTERNS = [
    "OLMo-2-0325-32B",
    "Qwen3-32B",
    "Llama-3.1-70B",
    "gemma-3-27b-pt",
]


# ---------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------

def parse_size(model_name: str) -> float:
    """
    Parse model size (in B) from the model name, for example:
        "meta-llama/Llama-3.1-70B"  -> 70.0
        "Qwen/Qwen3-0.6B-Base"      -> 0.6
        "google/gemma-3-27b-pt"     -> 27.0
    """
    matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*[bB]", model_name)
    if not matches:
        raise ValueError(f"Could not parse model size (in B) from name: {model_name}")
    return float(matches[-1])


def is_big_new_model(model_name: str) -> bool:
    """Return whether this model name appears in the 'subsampled large models' list (substring match)."""
    return any(pat in model_name for pat in SUBSAMPLE_MODEL_PATTERNS)


# ---------------------------------------------------------
# 2. Load data and compute accuracy
# ---------------------------------------------------------

# Read CSV; the first column must be "model"
df_n = pd.read_csv(TOTAL_CSV)
df_k = pd.read_csv(CORRECT_CSV)

if "model" not in df_n.columns or "model" not in df_k.columns:
    raise ValueError("Both CSVs must have a 'model' column as the first column.")

# Set 'model' as index, rows = models, columns = tasks/subjects
df_n = df_n.set_index("model")
df_k = df_k.set_index("model")

# Align rows and columns of both tables
df_k = df_k.loc[df_n.index, df_n.columns]

model_names = df_n.index.to_numpy()   # shape (M,)
task_names = df_n.columns.to_numpy()  # shape (T,)

# Convert to numpy
n = df_n.to_numpy(dtype=float)   # total number of questions
k = df_k.to_numpy(dtype=float)   # number of correct answers

M, T = n.shape
if k.shape != (M, T):
    raise ValueError("Shapes of n and k do not match.")

# Avoid division by 0: set accuracy to NaN where n is 0
with np.errstate(divide="ignore", invalid="ignore"):
    acc = np.where(n > 0, k / n, np.nan)   # shape (M, T)

# Parse model sizes
model_sizes = np.array([parse_size(m) for m in model_names], dtype=float)  # shape (M,)
if np.any(model_sizes <= 0):
    raise ValueError("All model sizes must be positive to take log.")

log_sizes = np.log(model_sizes)

# Mark which models are big_new_models
big_mask = np.array([is_big_new_model(m) for m in model_names])  # shape (M,)
big_new_models = model_names[big_mask]
non_big_mask = ~big_mask

if big_new_models.size == 0:
    raise ValueError("No big_new_models found matching SUBSAMPLE_MODEL_PATTERNS.")

print("Big models to predict (linear regression baseline):")
for m in big_new_models:
    print("  -", m)

# ---------------------------------------------------------
# 3. Use non-big models as training set, fit linear regression per task
#    acc[m,t] = alpha_t + beta_t * log(size[m])
# ---------------------------------------------------------

pred_acc = np.zeros((big_new_models.size, T), dtype=float)  # predicted accuracy

for t_idx in range(T):
    # This column is accuracy for task t
    acc_t = acc[:, t_idx]        # shape (M,)
    n_t = n[:, t_idx]            # shape (M,)

    # Models with valid data (n>0 and acc not NaN)
    valid_mask = (n_t > 0) & ~np.isnan(acc_t)

    # Use only non-big + valid models for regression
    train_mask = valid_mask & non_big_mask
    if train_mask.sum() < 2:
        # If too few models, regression is unstable: fall back to mean accuracy as constant prediction
        if train_mask.sum() == 0:
            # If no non-big models available, use mean of all valid models
            if valid_mask.sum() > 0:
                alpha_t = float(np.nanmean(acc_t[valid_mask]))
            else:
                alpha_t = 0.5  # If there is truly no data, default to 0.5
        else:
            alpha_t = float(np.nanmean(acc_t[train_mask]))
        beta_t = 0.0
    else:
        # Standard OLS: acc_t = alpha_t + beta_t * log_sizes
        x_train = log_sizes[train_mask]
        y_train = acc_t[train_mask]

        X = np.column_stack([np.ones_like(x_train), x_train])  # [1, log(size)]
        # Least squares
        coef, *_ = np.linalg.lstsq(X, y_train, rcond=None)
        alpha_t, beta_t = coef

    # Make predictions for big_new_models
    if big_new_models.size > 0:
        x_big = log_sizes[big_mask]  # log(size) for big models
        y_pred = alpha_t + beta_t * x_big
        # Clip to [0, 1]
        y_pred = np.clip(y_pred, 0.0, 1.0)
        pred_acc[:, t_idx] = y_pred

# ---------------------------------------------------------
# 4. Output predicted accuracy (big_new_models × tasks)
# ---------------------------------------------------------

df_pred_acc = pd.DataFrame(pred_acc, index=big_new_models, columns=task_names)
df_pred_acc.to_csv(OUTPUT_PRED_ACC_CSV, index_label="model")
print(f"\nSaved linear-regression predicted accuracy to: {OUTPUT_PRED_ACC_CSV}")

# ---------------------------------------------------------
# 5. Also output predicted correct counts (for comparison with Bayesian results)
#    Use non-big models as reference to get original question counts per task
# ---------------------------------------------------------

# Use the first non-big model row as reference for "original question counts"
if np.any(non_big_mask):
    full_n_per_task = n[non_big_mask][0]  # shape (T,)
else:
    # Extreme case: all models are big; unlikely in new data, but handle as fallback
    full_n_per_task = n[0]
    print("[WARNING] All models are big_new_models; using first row of n as reference for total questions per task.")

full_n_per_task = full_n_per_task.astype(float)  # (T,)

# Predicted correct counts = predicted accuracy × original question counts
pred_correct_counts = pred_acc * full_n_per_task[None, :]  # shape (num_big, T)
df_pred_counts = pd.DataFrame(
    np.round(pred_correct_counts).astype(int),
    index=big_new_models,
    columns=task_names,
)
df_pred_counts.to_csv(OUTPUT_PRED_COUNTS_CSV, index_label="model")
print(f"Saved linear-regression predicted correct counts to: {OUTPUT_PRED_COUNTS_CSV}")
