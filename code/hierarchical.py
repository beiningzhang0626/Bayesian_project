"""
Bayesian hierarchical model for LLM performance prediction.

Setting:
- Multiple models m = 1..M
- Multiple tasks t = 1..T
- For each (m, t): n[m, t] questions, k[m, t] correct
- Each model has a size (number of parameters), size[m]

Goal:
- Fit a hierarchical model:
    logit(p[m, t]) = theta[m] - delta[t]
    theta[m] ~ Normal(mu_theta + beta_size * log(size[m]), sigma_theta)
    delta[t] ~ Normal(mu_delta, sigma_delta)
- Use MCMC to sample posterior over all parameters.
- From posterior, get accuracy per task for a NEW model, including uncertainty.

Dependencies:
    pip install numpy pymc arviz
"""

import os
import numpy as np
import pymc as pm
import arviz as az

# ---------------------------------------------------------
# 0. Config: output directory / text / csv files & (optional) JAX/GPU
# ---------------------------------------------------------

# Root directory where ALL outputs will be saved
OUTPUT_DIR = r"D:\Bayesian_project\output\hierarchical"  # <-- change this

# Where to save posterior summaries (accuracy, intervals, etc.)
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "llm_hierarchical_results.txt")

# Where to save posterior predictive results (big_new_models only, text)
OUTPUT_PRED_TXT = os.path.join(OUTPUT_DIR, "llm_hierarchical_predictions.txt")

# Where to save posterior predictive EXPECTED correct counts
# (rows = big_new_models, columns = tasks; values = predicted #correct)
OUTPUT_PRED_CSV = os.path.join(OUTPUT_DIR, "llm_hierarchical_predictions_counts.csv")

# Try to use JAX / GPU-accelerated sampler if available.
# If you prefer to force classic PyMC sampling on CPU, set this to False.
USE_JAX_SAMPLER = True

# ---------------------------------------------------------
# 1. Real DATA: read from aggregated matrices
# ---------------------------------------------------------

import pandas as pd
import re

# Paths to your matrices:
#   - TOTAL_CSV   : total number of questions per (model, subject)
#   - CORRECT_CSV : number of correct answers per (model, subject)
TOTAL_CSV   = r"D:\Bayesian_project\results\mmlu_subject_total_questions_by_model_incomplete_20pct.csv"
CORRECT_CSV = r"D:\Bayesian_project\results\mmlu_subject_num_correct_by_model_incomplete_20pct.csv"

# Read CSVs; first column "model", remaining columns are subjects
df_n = pd.read_csv(TOTAL_CSV)
df_k = pd.read_csv(CORRECT_CSV)

if "model" not in df_n.columns or "model" not in df_k.columns:
    raise ValueError("Both CSVs must have a 'model' column as the first column.")

# Use 'model' as index so rows = models
df_n = df_n.set_index("model")
df_k = df_k.set_index("model")

# Make sure models and subjects are aligned in both matrices
df_k = df_k.loc[df_n.index, df_n.columns]

# ---- Names for PyMC coords ----
model_names = df_n.index.to_numpy()      # shape (M,)
task_names  = df_n.columns.to_numpy()    # shape (T,)

# ---- n[m, t] and k[m, t] ----
n = df_n.to_numpy(dtype=int)             # total number of questions
k = df_k.to_numpy(dtype=int)             # number of correct answers

M, T = n.shape
assert k.shape == (M, T)

# ---- Parse model sizes (in billions) from model name strings ----
# e.g. "meta-llama/Llama-3.1-70B"  -> 70.0
#      "Qwen/Qwen3-0.6B-Base"     -> 0.6
#      "google/gemma-3-27b-pt"    -> 27.0
def parse_size(model_name: str) -> float:
    matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*[bB]", model_name)
    if not matches:
        raise ValueError(f"Could not parse model size (in B) from name: {model_name}")
    return float(matches[-1])

model_sizes = np.array([parse_size(m) for m in model_names], dtype=float)

# Sanity checks
assert model_sizes.shape == (M,)

# ---- Large models with subsampled tasks (keep consistent with incomplete CSV generation) ----
SUBSAMPLE_MODEL_PATTERNS = [
    "OLMo-2-0325-32B",
    "Qwen3-32B",
    "Llama-3.1-70B",
    "gemma-3-27b-pt",
]

def is_big_new_model(model_name: str) -> bool:
    """Return whether this model appears in the 'subsampled large models' list (substring match)."""
    return any(pat in model_name for pat in SUBSAMPLE_MODEL_PATTERNS)

# The actual big_new_models appearing in THIS CSV
big_new_models = np.array([m for m in model_names if is_big_new_model(m)])


# ---------------------------------------------------------
# 2. Build the hierarchical model in PyMC
# ---------------------------------------------------------

def build_and_sample_model(
    k,
    n,
    model_sizes,
    model_names,
    task_names,
    random_seed=123,
    use_jax_sampler=USE_JAX_SAMPLER,
):
    """
    Build the hierarchical Bayesian model and run MCMC.

    Returns:
        model: the PyMC model object
        idata: ArviZ InferenceData with posterior samples
        sampler_info: short string describing which sampler/backend was used
    """

    coords = {
        "model": model_names,
        "task": task_names,
    }

    with pm.Model(coords=coords) as model:
        # --- Data containers (for convenience & possible future updating) ---
        n_obs = pm.Data("n_obs", n, dims=("model", "task"))
        k_obs = pm.Data("k_obs", k, dims=("model", "task"))
        size = pm.Data("size", model_sizes, dims=("model",))

        # --- Hyperpriors for model ability ---
        mu_theta = pm.Normal("mu_theta", mu=0.0, sigma=2.0)
        sigma_theta = pm.HalfNormal("sigma_theta", sigma=1.0)
        beta_size = pm.Normal("beta_size", mu=0.0, sigma=1.0)

        # --- Hyperpriors for task difficulty ---
        mu_delta = pm.Normal("mu_delta", mu=0.0, sigma=2.0)
        sigma_delta = pm.HalfNormal("sigma_delta", sigma=1.0)

        # --- Model-level abilities: theta[m] ---
        mean_theta = mu_theta + beta_size * pm.math.log(size)
        theta = pm.Normal(
            "theta",
            mu=mean_theta,
            sigma=sigma_theta,
            dims=("model",),
        )

        # --- Task-level difficulties: delta[t] ---
        delta = pm.Normal(
            "delta",
            mu=mu_delta,
            sigma=sigma_delta,
            dims=("task",),
        )

        # --- Linear predictor and probabilities ---
        # logit(p[m, t]) = theta[m] - delta[t]
        logit_p = theta[:, None] - delta[None, :]
        p = pm.Deterministic(
            "p",
            pm.math.sigmoid(logit_p),
            dims=("model", "task"),
        )

        # --- Likelihood: Binomial at (model, task) level ---
        _ = pm.Binomial(
            "k_like",
            n=n_obs,
            p=p,
            observed=k_obs,
            dims=("model", "task"),
        )

        sampler_info = "pm.sample (CPU, PyTensor backend)"

        if use_jax_sampler:
            try:
                # JAX-based NUTS; will use GPU if available.
                from pymc.sampling import jax as pmjax
                import jax

                backend = None
                devices = None
                try:
                    backend = jax.default_backend()
                    devices = jax.devices()
                except Exception:
                    backend = "unknown"
                    devices = []

                gpu_devices = [d for d in devices if getattr(d, "platform", "") == "gpu"]
                if gpu_devices:
                    sampler_info = (
                        f"pymc.sampling.jax.sample_numpyro_nuts "
                        f"(JAX backend={backend}, {len(gpu_devices)} GPU device(s))"
                    )
                else:
                    sampler_info = (
                        f"pymc.sampling.jax.sample_numpyro_nuts "
                        f"(JAX backend={backend}, no GPU detected)"
                    )

                idata = pmjax.sample_numpyro_nuts(
                    draws=2000,
                    tune=2000,
                    chains=4,
                    target_accept=0.9,
                    random_seed=random_seed,
                    chain_method="parallel",
                )

            except Exception as e:
                sampler_info = (
                    "pm.sample (CPU, fell back from JAX sampler: "
                    f"{type(e).__name__}: {e})"
                )
                idata = pm.sample(
                    draws=2000,
                    tune=2000,
                    chains=4,
                    target_accept=0.9,
                    random_seed=random_seed,
                    return_inferencedata=True,
                )
        else:
            idata = pm.sample(
                draws=2000,
                tune=2000,
                chains=4,
                target_accept=0.9,
                random_seed=random_seed,
                return_inferencedata=True,
            )

    return model, idata, sampler_info


# ---------------------------------------------------------
# 3. Posterior analysis helpers
# ---------------------------------------------------------

def summarize_model_task_accuracy(idata):
    """
    Summarize posterior accuracy p[model, task] for each model and task.
    Returns a simple dict: summary[model_name][task_name] = {mean, hdi_3, hdi_97}
    """

    p_posterior = idata.posterior["p"]  # shape: (chain, draw, model, task)
    models = p_posterior.coords["model"].values
    tasks = p_posterior.coords["task"].values

    summary = {}
    for m in models:
        summary[m] = {}
        for t in tasks:
            samples = p_posterior.sel(model=m, task=t).values.flatten()
            mean = samples.mean()
            hdi_low, hdi_high = az.hdi(samples, hdi_prob=0.95)
            summary[m][t] = {
                "mean": float(mean),
                "hdi_3%": float(hdi_low),
                "hdi_97%": float(hdi_high),
            }
    return summary


def summarize_new_model(idata, new_model_name):
    """
    Extract and summarize the posterior accuracy for ONE model on each task.

    Assumes the new model is included in the model_names.
    """
    summary = summarize_model_task_accuracy(idata)
    if new_model_name not in summary:
        raise ValueError(f"Model {new_model_name} not found in posterior coords.")
    return summary[new_model_name]


# ---------------------------------------------------------
# 4. Run everything and save results to text / csv files
# ---------------------------------------------------------

def main():
    # Make sure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Open two files: summary + predictive
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f_summary, \
         open(OUTPUT_PRED_TXT, "w", encoding="utf-8") as f_pred:

        def log(msg=""):
            """Write to summary file + print to screen"""
            msg = str(msg)
            print(msg)
            f_summary.write(msg + "\n")

        def log_pred(msg=""):
            """Write to predictive file + print to screen"""
            msg = str(msg)
            print(msg)
            f_pred.write(msg + "\n")

        log("Fitting hierarchical model with MCMC...")
        model, idata, sampler_info = build_and_sample_model(
            k=k,
            n=n,
            model_sizes=model_sizes,
            model_names=model_names,
            task_names=task_names,
            random_seed=123,
            use_jax_sampler=USE_JAX_SAMPLER,
        )
        log(f"Sampler used: {sampler_info}")
        log("Sampling finished.\n")

        # 2) Summarize posterior accuracy for all models × tasks
        full_summary = summarize_model_task_accuracy(idata)

        log("Posterior accuracy summary (per model & task):")
        for m in model_names:
            log(f"\n=== {m} ===")
            for t in task_names:
                stats = full_summary[m][t]
                log(
                    f"  {t}: mean={stats['mean']:.3f}, "
                    f"95% CI=({stats['hdi_3%']:.3f}, {stats['hdi_97%']:.3f})"
                )

        # 3) Focus on the “subsampled large models” (big_new_models)
        log("\n\nSubsampled big_new_models detected in this CSV:")
        if big_new_models.size == 0:
            log("  [WARNING] No models matched SUBSAMPLE_MODEL_PATTERNS.")
        else:
            for m in big_new_models:
                log(f"  - {m}")

            for new_model_name in big_new_models:
                try:
                    new_model_summary = summarize_new_model(idata, new_model_name)
                except ValueError as e:
                    log(f"[WARNING] {e}")
                    continue

                log(f"\nPosterior accuracy for SUBSAMPLED big_new_model: {new_model_name}")
                for t in task_names:
                    stats = new_model_summary[t]
                    log(
                        f"  {t}: mean={stats['mean']:.3f}, "
                        f"95% CI=({stats['hdi_3%']:.3f}, {stats['hdi_97%']:.3f})"
                    )

        # 4) Posterior predictive for future evaluations (written to OUTPUT_PRED_TXT & CSV)
        p_post = idata.posterior["p"]  # (chain, draw, model, task)
        model_coords = p_post.coords["model"].values

        if big_new_models.size > 0:
            # --- Use models without subsampling as reference to restore original question counts ---
            non_big_mask = np.array([not is_big_new_model(m) for m in model_names])
            if np.any(non_big_mask):
                ref_row = n[non_big_mask][0]   # First non-subsampled row (T,)
            else:
                # Degenerate case: all models are big_new_models
                ref_row = n[0]
                log("[WARNING] All models match SUBSAMPLE_MODEL_PATTERNS; "
                    "using first row of n as reference for total questions per task.")

            N_future_per_task = ref_row.astype(int)  # shape (T,)

            log_pred("Posterior predictive: SUBSAMPLED big_new_models")
            log_pred("Using original number of questions per task (from reference model).")

            # Store predicted E[#correct] per big_new_model × task
            pred_means = np.zeros((len(big_new_models), len(task_names)), dtype=float)

            for i_model, new_model_name in enumerate(big_new_models):
                if new_model_name not in model_coords:
                    log_pred(f"[WARNING] {new_model_name} not in posterior coords, skip predictive.")
                    continue

                m_idx = int(np.where(model_coords == new_model_name)[0][0])

                log_pred(f"\n--- Model: {new_model_name} ---")
                for t_idx, t_name in enumerate(task_names):
                    N_future = int(N_future_per_task[t_idx])
                    if N_future <= 0:
                        log_pred(f"  {t_name}: N_future <= 0, skip.")
                        continue

                    # Extract posterior samples of p
                    p_samples = p_post.isel(model=m_idx, task=t_idx).values.flatten()
                    # Simulate predicted correct counts
                    future_k = np.random.binomial(N_future, p_samples)
                    mean_future = future_k.mean()
                    low, high = np.percentile(future_k, [2.5, 97.5])

                    # Save mean for CSV output
                    pred_means[i_model, t_idx] = mean_future

                    log_pred(
                        f"  {t_name}: E[correct] ≈ {mean_future:.1f} / {N_future}, "
                        f"95% predictive interval = ({low:.0f}, {high:.0f})"
                    )

            # Export predicted correct counts as CSV
            df_pred = pd.DataFrame(pred_means, index=big_new_models, columns=task_names)
            df_pred_int = df_pred.round().astype(int)
            df_pred_int.to_csv(OUTPUT_PRED_CSV, index_label="model")
            log_pred(f"\nPredicted correct-count CSV saved to: {OUTPUT_PRED_CSV}")

        log(f"\nSummary written to: {OUTPUT_TXT}")
        log_pred(f"\nPosterior predictive results written to: {OUTPUT_PRED_TXT}")


if __name__ == "__main__":
    main()
