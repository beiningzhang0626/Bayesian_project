import os
import glob
import json
import random
from collections import Counter
import pandas as pd

DATA_DIR = r"D:\Bayesian_project\dataset"
OUTPUT_DIR = r"D:\Bayesian_project\result"
SUBSAMPLE_MODEL_PATTERNS = [
    "OLMo-2-0325-32B",
    "Qwen3-32B",
    "Llama-3.1-70B",
    "gemma-3-27b-pt",
]

SUBSAMPLE_FRACTION = 0.2   # e.g. 0.7 = keep 70% per subject
RANDOM_SEED = 42


def model_needs_subsample(model_name: str, file_basename: str) -> bool:
    combined = model_name + " " + file_basename
    return any(pat in combined for pat in SUBSAMPLE_MODEL_PATTERNS)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)
    json_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    subject_total_counts_per_model = {}
    subject_correct_counts_per_model = {}
    per_model_subject_is_correct = {}

    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_basename = os.path.basename(path)
        # Use the model_name in the file as the row index
        model_name = data.get("model_name", file_basename)

        total_counter = Counter()
        correct_counter = Counter()
        subj_to_is_correct_list = {}

        for item in data["individual_results"]:
            subj = item["subject"]
            total_counter[subj] += 1

            # is_correct is usually 0/1 or 0.0/1.0, convert to int
            is_corr = item.get("is_correct", 0)
            try:
                is_corr_int = int(round(float(is_corr)))
            except (TypeError, ValueError):
                is_corr_int = 0  # If something goes wrong, treat as incorrect

            correct_counter[subj] += is_corr_int
            subj_to_is_correct_list.setdefault(subj, []).append(is_corr_int)

        subject_total_counts_per_model[model_name] = total_counter
        subject_correct_counts_per_model[model_name] = correct_counter
        per_model_subject_is_correct[model_name] = subj_to_is_correct_list
    models = list(subject_total_counts_per_model.keys())
    ref_model = models[0]
    ref_subjects = set(subject_total_counts_per_model[ref_model].keys())

    print(f"use {ref_model} as the reference to check subject consistency:\n")

    for m in models:
        s = set(subject_total_counts_per_model[m].keys())
        if s != ref_subjects:
            print(f"[WARNING] subject set of {m} is not the same as {ref_model}:")
            only_in_ref = sorted(ref_subjects - s)
            only_in_m = sorted(s - ref_subjects)
            if only_in_ref:
                print("  Subjects only appearing in the reference model:", only_in_ref)
            if only_in_m:
                print("  Subjects only appearing in the current model:", only_in_m)
        else:
            print(f"[OK] subject set of {m} is exactly the same as {ref_model}.")

    print("check whether the number of questions per subject is consistent")

    ref_counts = subject_total_counts_per_model[ref_model]
    for m in models:
        counts = subject_total_counts_per_model[m]
        diffs = []
        for subj in ref_subjects:
            c_ref = ref_counts[subj]
            c_m = counts.get(subj, 0)
            if c_ref != c_m:
                diffs.append((subj, c_ref, c_m))
        if diffs:
            print(f"[WARNING] {m} has subjects whose question counts differ from {ref_model}:")
            for subj, c_ref, c_m in diffs:
                print(f"  {subj}: reference={c_ref}, current={c_m}")
        else:
            print(f"[OK] For each subject, the number of questions in {m} is also the same as {ref_model}.")
    all_subjects = sorted(
        {subj for counts in subject_total_counts_per_model.values()
         for subj in counts.keys()}
    )
    df_total = pd.DataFrame(index=models, columns=all_subjects)
    df_correct = pd.DataFrame(index=models, columns=all_subjects)

    for m in models:
        total_counts = subject_total_counts_per_model[m]
        correct_counts = subject_correct_counts_per_model[m]
        for subj in all_subjects:
            df_total.loc[m, subj] = total_counts.get(subj, 0)
            df_correct.loc[m, subj] = correct_counts.get(subj, 0)

    df_total = df_total.fillna(0).astype(int)
    df_correct = df_correct.fillna(0).astype(int)
    out_total = os.path.join(OUTPUT_DIR, "mmlu_subject_total_questions_by_model.csv")
    out_correct = os.path.join(OUTPUT_DIR, "mmlu_subject_num_correct_by_model.csv")

    df_total.to_csv(out_total, index_label="model")
    df_correct.to_csv(out_correct, index_label="model")

    subject_total_counts_incomplete = {}
    subject_correct_counts_incomplete = {}

    for m in models:
        needs_subsample = model_needs_subsample(m, "")

        if not needs_subsample:
            # For non-target models, keep full data
            subject_total_counts_incomplete[m] = subject_total_counts_per_model[m]
            subject_correct_counts_incomplete[m] = subject_correct_counts_per_model[m]
            continue
        total_counter_inc = Counter()
        correct_counter_inc = Counter()
        subj_to_list = per_model_subject_is_correct[m]

        for subj, is_list in subj_to_list.items():
            n_total = len(is_list)
            if n_total == 0:
                continue

            k = int(round(SUBSAMPLE_FRACTION * n_total))
            k = max(0, min(n_total, k))  # clamp to [0, n_total]

            if k == 0:
                # all zeros
                total_counter_inc[subj] = 0
                correct_counter_inc[subj] = 0
            else:
                # sample k questions without replacement
                chosen = random.sample(is_list, k)
                total_counter_inc[subj] = k
                correct_counter_inc[subj] = sum(chosen)

        subject_total_counts_incomplete[m] = total_counter_inc
        subject_correct_counts_incomplete[m] = correct_counter_inc
    df_total_inc = pd.DataFrame(index=models, columns=all_subjects)
    df_correct_inc = pd.DataFrame(index=models, columns=all_subjects)

    for m in models:
        total_counts = subject_total_counts_incomplete[m]
        correct_counts = subject_correct_counts_incomplete[m]
        for subj in all_subjects:
            df_total_inc.loc[m, subj] = total_counts.get(subj, 0)
            df_correct_inc.loc[m, subj] = correct_counts.get(subj, 0)

    df_total_inc = df_total_inc.fillna(0).astype(int)
    df_correct_inc = df_correct_inc.fillna(0).astype(int)

    frac_pct = int(round(SUBSAMPLE_FRACTION * 100))
    out_total_inc = os.path.join(
        OUTPUT_DIR, f"mmlu_subject_total_questions_by_model_incomplete_{frac_pct}pct.csv"
    )
    out_correct_inc = os.path.join(
        OUTPUT_DIR, f"mmlu_subject_num_correct_by_model_incomplete_{frac_pct}pct.csv"
    )

    df_total_inc.to_csv(out_total_inc, index_label="model")
    df_correct_inc.to_csv(out_correct_inc, index_label="model")

if __name__ == "__main__":
    main()
