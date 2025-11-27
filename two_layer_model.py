"""
Bayesian hierarchical model for LLM performance prediction (with family layer).

Setting:
- Multiple models m = 1..M
- Multiple tasks t = 1..T
- Multiple families f = 1..F
- For each (m, t): n[m, t] questions, k[m, t] correct
- Each model has:
    - a size (number of parameters), size[m]
    - a family index f(m)

Goal:
- Extend the previous hierarchical model (global -> model) by adding a
  family-level layer (global -> family -> model), but keep the same aim:
  use limited data from large models to predict their behavior by pooling
  information across tasks, model sizes, and families.

Model:
    Global ability:
        mu_theta ~ Normal(0, 2)
        sigma_family ~ HalfNormal(1)
        sigma_theta ~ HalfNormal(1)
        beta_size ~ Normal(0, 1)

    Family abilities:
        theta_family[f] ~ Normal(mu_theta, sigma_family)

    Model abilities:
        theta[m] ~ Normal(theta_family[f(m)] + beta_size * log(size[m]),
                          sigma_theta)

    Task difficulties:
        mu_delta ~ Normal(0, 2)
        sigma_delta ~ HalfNormal(1)
        delta[t] ~ Normal(mu_delta, sigma_delta)

    Observation model:
        logit(p[m, t]) = theta[m] - delta[t]
        k[m, t] ~ Binomial(n[m, t], p[m, t])

Inference:
- Use MCMC (NumPyro NUTS via JAX; GPU-capable) to sample the posterior.
- From posterior, get accuracy per task for large, partially-evaluated models.

Dependencies:
    pip install numpy pandas "pymc[sampling_jax]" arviz
    # plus GPU-enabled jax & jaxlib (see JAX docs)
"""

import numpy as np
import pandas as pd
import re
import pymc as pm
import pymc.sampling.jax as pmjax   # JAX / NumPyro backend (GPU-capable)
import arviz as az
import os


# ---------------------------------------------------------
# 0. Paths / configuration
# ---------------------------------------------------------

TOTAL_CSV   = r"results/mmlu_subject_total_questions_by_model_incomplete_20pct.csv"
CORRECT_CSV = r"results/mmlu_subject_num_correct_by_model_incomplete_20pct.csv"

OUTPUT_TXT     = r"output/bayesian_hierarchical_large_family_summary.txt"
OUTPUT_NETCDF  = r"output/bayesian_hierarchical_large_family_posterior.nc"

GLOBAL_RANDOM_SEED = 123
np.random.seed(GLOBAL_RANDOM_SEED)

# Large models with subsampled tasks (targets for prediction / emphasis)
SUBSAMPLE_MODEL_PATTERNS = [
    "OLMo-2-0325-32B",
    "Qwen3-32B",
    "Llama-3.1-70B",
    "gemma-3-27b-pt",
]

# Known families (prefixes of model names)
FAMILIES = [
    "meta-llama",
    "allenai/OLMo",
    "Qwen",
    "google/gemma",
]


# ---------------------------------------------------------
# 1. Data loading and preprocessing
# ---------------------------------------------------------

# Read CSVs; first column "model", remaining columns are subjects
df_n = pd.read_csv(TOTAL_CSV)
df_k = pd.read_csv(CORRECT_CSV)

if "model" not in df_n.columns or "model" not in df_k.columns:
    raise ValueError("Both CSVs must have a 'model' column as the first column.")

# Use 'model' as index so rows = models
df_n = df_n.set_index("model")
df_k = df_k.set_index("model")

# Align models and tasks
df_k = df_k.loc[df_n.index, df_n.columns]

model_names = df_n.index.to_numpy()      # shape (M,)
task_names  = df_n.columns.to_numpy()    # shape (T,)

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
assert model_sizes.shape == (M,)


# ---- Parse model families from model names using explicit prefixes ----
def parse_family(model_name: str) -> str:
    """
    Deterministically map each model name to one of the known families.
    Model names are guaranteed to start with the family prefix.
    """
    for fam in FAMILIES:
        if model_name.startswith(fam):
            return fam
    raise ValueError(f"Model name does not match any known family: {model_name}")


family_names_raw = [parse_family(m) for m in model_names]
family_names = np.array(sorted(set(family_names_raw)))
F = len(family_names)

# Map each model to a family index 0..F-1
family_index_per_model = np.array(
    [np.where(family_names == fam)[0][0] for fam in family_names_raw],
    dtype=int
)


# ---- Identify large subsampled models we want to highlight ----
def is_subsampled_model(model_name: str) -> bool:
    """
    Return True if this model name contains any of the subsample patterns.
    """
    return any(pat in model_name for pat in SUBSAMPLE_MODEL_PATTERNS)


subsampled_model_mask = np.array([is_subsampled_model(m) for m in model_names])
subsampled_model_names = model_names[subsampled_model_mask]


# ---------------------------------------------------------
# 2. Build the hierarchical model in PyMC (with family layer)
# ---------------------------------------------------------

def build_and_sample_model(
    k,
    n,
    model_sizes,
    model_names,
    task_names,
    family_names,
    family_index_per_model,
    random_seed=123,
):
    """
    Build the hierarchical Bayesian model with a family layer and run MCMC
    using NumPyro NUTS (JAX backend, GPU capable).

    Returns:
        model: PyMC model object
        idata: ArviZ InferenceData with posterior samples
    """

    coords = {
        "model": model_names,
        "task": task_names,
        "family": family_names,
    }

    with pm.Model(coords=coords) as model:
        # Data
        n_obs = pm.Data("n_obs", n, dims=("model", "task"))
        k_obs = pm.Data("k_obs", k, dims=("model", "task"))
        size  = pm.Data("size", model_sizes, dims=("model",))
        fam_idx = pm.Data("fam_idx", family_index_per_model, dims=("model",))

        # Global hyperpriors for abilities
        mu_theta = pm.Normal("mu_theta", mu=0.0, sigma=2.0)
        sigma_family = pm.HalfNormal("sigma_family", sigma=1.0)
        sigma_theta = pm.HalfNormal("sigma_theta", sigma=1.0)
        beta_size = pm.Normal("beta_size", mu=0.0, sigma=1.0)

        # Family-level abilities
        theta_family = pm.Normal(
            "theta_family",
            mu=mu_theta,
            sigma=sigma_family,
            dims=("family",),
        )

        # Model-level abilities: depends on family + size
        # theta[m] ~ Normal(theta_family[f(m)] + beta_size * log(size[m]),
        #                   sigma_theta)
        mean_theta_model = theta_family[fam_idx] + beta_size * pm.math.log(size)
        theta = pm.Normal(
            "theta",
            mu=mean_theta_model,
            sigma=sigma_theta,
            dims=("model",),
        )

        # Task difficulty hyperpriors
        mu_delta = pm.Normal("mu_delta", mu=0.0, sigma=2.0)
        sigma_delta = pm.HalfNormal("sigma_delta", sigma=1.0)

        # Task-level difficulties
        delta = pm.Normal(
            "delta",
            mu=mu_delta,
            sigma=sigma_delta,
            dims=("task",),
        )

        # Linear predictor and probabilities
        logit_p = theta[:, None] - delta[None, :]
        p = pm.Deterministic(
            "p",
            pm.math.sigmoid(logit_p),
            dims=("model", "task"),
        )

        # Likelihood
        pm.Binomial(
            "k_like",
            n=n_obs,
            p=p,
            observed=k_obs,
            dims=("model", "task"),
        )

        # MCMC: JAX / NumPyro NUTS (GPU capable if jax is configured)
        idata = pmjax.sample_numpyro_nuts(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.9,
            random_seed=random_seed,
        )

    return model, idata


# ---------------------------------------------------------
# 3. Posterior analysis helpers
# ---------------------------------------------------------

def summarize_model_task_accuracy(idata):
    """
    Summarize posterior accuracy p[model, task] for each model and task.

    Returns a dict: summary[model_name][task_name] = {mean, hdi_3%, hdi_97%}
    """
    p_posterior = idata.posterior["p"]  # (chain, draw, model, task)
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


def summarize_specific_models(idata, target_model_names):
    """
    Return summaries for a list of specific model names.
    """
    full = summarize_model_task_accuracy(idata)
    out = {}
    for name in target_model_names:
        if name not in full:
            raise ValueError(f"Model {name} not found in posterior coords.")
        out[name] = full[name]
    return out


# ---------------------------------------------------------
# 4. Main: run everything, print + save to TXT
# ---------------------------------------------------------

def main():
    # Ensure output directory exists
    out_dir = os.path.dirname(OUTPUT_TXT)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Fit hierarchical model with a family layer
    model, idata = build_and_sample_model(
        k=k,
        n=n,
        model_sizes=model_sizes,
        model_names=model_names,
        task_names=task_names,
        family_names=family_names,
        family_index_per_model=family_index_per_model,
        random_seed=GLOBAL_RANDOM_SEED,
    )

    # Optionally save NetCDF
    try:
        idata.to_netcdf(OUTPUT_NETCDF)
    except Exception as e:
        print(f"Warning: could not save NetCDF file: {e}")

    # Collect log lines for TXT
    lines = []

    def log(line: str):
        print(line)
        lines.append(line)

    # Posterior summary per model × task
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

    # Focus on large subsampled models (targets for prediction)
    if len(subsampled_model_names) > 0:
        log("\n\nPosterior accuracy for large subsampled models:")
        subsampled_summary = summarize_specific_models(idata, subsampled_model_names)
        for m in subsampled_model_names:
            log(f"\n=== {m} ===")
            for t in task_names:
                stats = subsampled_summary[m][t]
                log(
                    f"  {t}: mean={stats['mean']:.3f}, "
                    f"95% CI=({stats['hdi_3%']:.3f}, {stats['hdi_97%']:.3f})"
                )
    else:
        log("\n\nNo subsampled models matched the given patterns.")

    # Posterior predictive for future evaluations for these subsampled models
    N_future = 100
    p_post = idata.posterior["p"]  # (chain, draw, model, task)

    if len(subsampled_model_names) > 0:
        log(f"\nPosterior predictive: subsampled models on {N_future} future questions per task")
        for m_name in subsampled_model_names:
            m_idx = int(np.where(p_post.coords["model"].values == m_name)[0][0])
            log(f"\n--- {m_name} ---")
            for t_idx, t_name in enumerate(task_names):
                p_samples = p_post.isel(model=m_idx, task=t_idx).values.flatten()
                future_k = np.random.binomial(N_future, p_samples)
                mean_future = future_k.mean()
                low, high = np.percentile(future_k, [2.5, 97.5])
                log(
                    f"  {t_name}: E[correct] ≈ {mean_future:.1f} / {N_future}, "
                    f"95% predictive interval = ({low:.0f}, {high:.0f})"
                )

    # Save all log lines to TXT
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved summary to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
