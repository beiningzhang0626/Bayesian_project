#!/usr/bin/env python3
"""
Prettier exploratory plots for the MMLU model×subject matrices produced by Step 1.

Inputs (from Step 1 outputs)
----------------------------
1) Full total question counts:
   D:\Bayesian_project\results\mmlu_subject_total_questions_by_model.csv

2) Full correct counts:
   D:\Bayesian_project\results\mmlu_subject_num_correct_by_model.csv

3) Incomplete total question counts (20% subsample for some big models):
   D:\Bayesian_project\results\mmlu_subject_total_questions_by_model_incomplete_20pct.csv

4) Incomplete correct counts:
   D:\Bayesian_project\results\mmlu_subject_num_correct_by_model_incomplete_20pct.csv

Outputs
-------
All figures are saved under:
   D:\Bayesian_project\output\eda_plots

Figures:
  1) questions_per_subject.png
       – Number of questions per subject (full dataset).

  2) accuracy_heatmap_models_by_subject.png
       – Heatmap of accuracy by model × subject (full dataset).

  3) overall_accuracy_per_model_sorted.png
       – Models sorted by overall accuracy (horizontal bar; colored by family).

  4) subject_difficulty_mean_accuracy.png
       – Subjects sorted by mean accuracy across models (easier ↔ harder).

  5) subsampled_models_accuracy_delta_boxplot.png
       – For subsampled models only, boxplots of
         (accuracy_incomplete − accuracy_full) over subjects.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- File paths (Step 1 outputs) ----------
TOTAL_FULL_CSV = r"D:\Bayesian_project\results\mmlu_subject_total_questions_by_model.csv"
CORRECT_FULL_CSV = r"D:\Bayesian_project\results\mmlu_subject_num_correct_by_model.csv"
TOTAL_INCOMP_CSV = r"D:\Bayesian_project\results\mmlu_subject_total_questions_by_model_incomplete_20pct.csv"
CORRECT_INCOMP_CSV = r"D:\Bayesian_project\results\mmlu_subject_num_correct_by_model_incomplete_20pct.csv"

# Where to save plots
OUT_DIR = r"D:\Bayesian_project\original_dataset_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Global matplotlib style ----------
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


def load_matrices():
    """Load all four matrices as DataFrames indexed by model."""
    df_total_full = pd.read_csv(TOTAL_FULL_CSV).set_index("model")
    df_correct_full = pd.read_csv(CORRECT_FULL_CSV).set_index("model")
    df_total_inc = pd.read_csv(TOTAL_INCOMP_CSV).set_index("model")
    df_correct_inc = pd.read_csv(CORRECT_INCOMP_CSV).set_index("model")

    df_total_full = df_total_full.astype(int)
    df_correct_full = df_correct_full.astype(int)
    df_total_inc = df_total_inc.astype(int)
    df_correct_inc = df_correct_inc.astype(int)

    return df_total_full, df_correct_full, df_total_inc, df_correct_inc


def parse_family(model_name: str) -> str:
    """Rough family name from HF id: 'meta-llama/Llama-3.1-70B' -> 'meta-llama'."""
    return str(model_name).split("/")[0]


# ---------- Figure 1: questions per subject ----------

def plot_questions_per_subject(df_total_full):
    """Total number of questions per subject (using first model as reference)."""
    subject_cols = df_total_full.columns
    ref_counts = df_total_full.iloc[0]

    fig_w = max(10, 0.35 * len(subject_cols))
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    x = np.arange(len(subject_cols))
    ax.bar(x, ref_counts.values, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(subject_cols, rotation=90, fontsize=6)
    ax.set_ylabel("Number of questions")
    ax.set_xlabel("Subject")
    ax.set_title("Total number of questions per subject (full dataset)")
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "questions_per_subject.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------- Figure 2: accuracy heatmap ----------

def plot_accuracy_heatmap(df_total_full, df_correct_full):
    """Heatmap of accuracy by model × subject."""
    acc = df_correct_full / df_total_full.replace(0, np.nan)

    fig_w = max(10, 0.25 * acc.shape[1])
    fig_h = max(6, 0.35 * acc.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(acc.values, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")

    ax.set_xticks(np.arange(acc.shape[1]))
    ax.set_xticklabels(acc.columns, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(acc.shape[0]))
    ax.set_yticklabels(acc.index, fontsize=7)

    ax.set_xlabel("Subject")
    ax.set_ylabel("Model")
    ax.set_title("Accuracy heatmap (full dataset)")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "accuracy_heatmap_models_by_subject.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------- Figure 3: overall accuracy per model (sorted, colored by family) ----------

def plot_overall_accuracy_per_model_sorted(df_total_full, df_correct_full):
    """Horizontal barplot of overall accuracy per model, sorted and colored by family."""
    total_per_model = df_total_full.sum(axis=1)
    correct_per_model = df_correct_full.sum(axis=1)
    overall_acc = (correct_per_model / total_per_model.replace(0, np.nan)).sort_values()

    models = overall_acc.index
    families = [parse_family(m) for m in models]
    unique_fam = list(dict.fromkeys(families))  # preserve order
    cmap = plt.get_cmap("tab10")
    fam_to_color = {fam: cmap(i % 10) for i, fam in enumerate(unique_fam)}
    colors = [fam_to_color[f] for f in families]

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(models) + 2))
    y = np.arange(len(models))
    ax.barh(y, overall_acc.values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=7)
    ax.set_xlabel("Overall accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Overall (question-weighted) accuracy per model")

    # Legend for families
    handles = [plt.Line2D([0], [0], marker="s", color="none",
                          markerfacecolor=fam_to_color[f], markersize=8)
               for f in unique_fam]
    ax.legend(handles, unique_fam, title="Model family", loc="lower right")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "overall_accuracy_per_model_sorted.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------- Figure 4: subject difficulty (mean accuracy across models) ----------

def plot_subject_difficulty(df_total_full, df_correct_full):
    """
    Subjects sorted by mean accuracy across models.
    High mean accuracy = easier; low = harder.
    """
    acc = df_correct_full / df_total_full.replace(0, np.nan)
    mean_acc = acc.mean(axis=0).sort_values()   # hardest (left) -> easiest (right)

    subjects = mean_acc.index
    fig_w = max(10, 0.35 * len(subjects))
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    x = np.arange(len(subjects))
    bars = ax.bar(x, mean_acc.values, color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=90, fontsize=6)
    ax.set_ylabel("Mean accuracy across models")
    ax.set_xlabel("Subject")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Subject difficulty (mean model accuracy)")

    # Light line at overall mean
    overall_mean = float(acc.values.mean())
    ax.axhline(overall_mean, color="black", linestyle="--", linewidth=1)
    ax.text(len(subjects) - 1, overall_mean + 0.01,
            f"overall mean = {overall_mean:.2f}",
            ha="right", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "subject_difficulty_mean_accuracy.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------- Figure 5: subsampled models – accuracy delta boxplot ----------

def plot_subsampled_models_delta(df_total_full, df_correct_full,
                                 df_total_inc, df_correct_inc):
    """
    For models that were subsampled (incomplete total < full total for at
    least one subject), plot boxplots of (acc_incomplete − acc_full)
    across subjects. This shows whether 20% sampling systematically
    over-/under-estimates performance.
    """
    acc_full = df_correct_full / df_total_full.replace(0, np.nan)
    acc_inc = df_correct_inc / df_total_inc.replace(0, np.nan)

    ratio = df_total_inc / df_total_full.replace(0, np.nan)
    subsampled_models = ratio.min(axis=1)[ratio.min(axis=1) < 1.0].index.tolist()

    if not subsampled_models:
        print("No subsampled models detected (ratio == 1 for all models).")
        return

    # Collect per-model accuracy deltas over subjects
    deltas = []
    labels = []
    for m in subsampled_models:
        # Use subjects where both full & incomplete have non-zero total
        total_full = df_total_full.loc[m]
        total_inc = df_total_inc.loc[m]
        mask = (total_full > 0) & (total_inc > 0)

        if mask.sum() == 0:
            continue

        df_m = pd.DataFrame({
            "acc_full": acc_full.loc[m, mask],
            "acc_inc": acc_inc.loc[m, mask],
        }).dropna()

        if df_m.empty:
            continue

        deltas.append(df_m["acc_inc"] - df_m["acc_full"])
        labels.append(m)

    if not deltas:
        print("No valid (model, subject) pairs for subsampled models.")
        return

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(labels)), 4))
    bp = ax.boxplot(deltas, labels=labels, vert=True,
                    patch_artist=True, widths=0.6)

    # Color boxes
    cmap = plt.get_cmap("Set2")
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=cmap(i % cmap.N), alpha=0.8)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Accuracy(incomplete) − Accuracy(full)")
    ax.set_xlabel("Subsampled model")
    ax.set_title("Effect of 20% subsampling on accuracy (per model)")
    plt.xticks(rotation=15, ha="right", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "subsampled_models_accuracy_delta_boxplot.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------- Main ----------

def main():
    df_total_full, df_correct_full, df_total_inc, df_correct_inc = load_matrices()

    # 1 & 2: your original two plots (cleaned up a bit)
    plot_questions_per_subject(df_total_full)
    plot_accuracy_heatmap(df_total_full, df_correct_full)

    # 3–5: redesigned, more informative plots
    plot_overall_accuracy_per_model_sorted(df_total_full, df_correct_full)
    plot_subject_difficulty(df_total_full, df_correct_full)
    plot_subsampled_models_delta(df_total_full, df_correct_full,
                                 df_total_inc, df_correct_inc)

    print("All EDA plots generated.")


if __name__ == "__main__":
    main()
