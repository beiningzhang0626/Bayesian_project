import os
import pandas as pd

TOTAL_CSV_PATH = r"D:\Bayesian_project\evaluation\mmlu_subject_total_questions_by_model.csv"
CORRECT_CSV_PATH = r"D:\Bayesian_project\evaluation\mmlu_subject_num_correct_by_model.csv"
OUTPUT_DIR = r"D:\Bayesian_project\evaluation\variance_outputs"

total_df = pd.read_csv(TOTAL_CSV_PATH, index_col=0)
correct_df = pd.read_csv(CORRECT_CSV_PATH, index_col=0)

correct_df = correct_df.reindex_like(total_df)

accuracy_df = correct_df.astype(float).where(total_df != 0) / total_df.where(total_df != 0)

var_by_model = accuracy_df.var(axis=0, ddof=1)
var_by_model_df = var_by_model.to_frame(name="variance_accuracy_across_tasks")
var_by_model_df.index.name = total_df.columns.name or "model"

var_by_task = accuracy_df.var(axis=1, ddof=1)
var_by_task_df = var_by_task.to_frame(name="variance_accuracy_across_models")
var_by_task_df.index.name = total_df.index.name or "task"

os.makedirs(OUTPUT_DIR, exist_ok=True)

path_model = os.path.join(OUTPUT_DIR, "accuracy_variance_by_model.csv")
path_task = os.path.join(OUTPUT_DIR, "accuracy_variance_by_task.csv")

var_by_model_df.to_csv(path_model)
var_by_task_df.to_csv(path_task)

print(f"Saved variance by model to: {path_model}")
print(f"Saved variance by task to:  {path_task}")
