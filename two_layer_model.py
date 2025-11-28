import os
import re

import numpy as np
import pandas as pd
import pymc as pm
import pymc.sampling.jax as pmjax
import arviz as az


# paths
CSV_N = r"results/mmlu_subject_total_questions_by_model_incomplete_20pct.csv"
CSV_K = r"results/mmlu_subject_num_correct_by_model_incomplete_20pct.csv"

OUT_TXT = r"output/bayesian_hierarchical_large_family_summary.txt"
OUT_NC = r"output/bayesian_hierarchical_large_family_posterior.nc"
OUT_PRED = r"results/mmlu_subject_predicted_acc.csv"

SEED = 123
np.random.seed(SEED)

BIG_PAT = ["OLMo-2-0325-32B", "Qwen3-32B", "Llama-3.1-70B","gemma-3-27b-pt"]

FAMS = ["meta-llama",
    "allenai/OLMo", "Qwen",
    "google/gemma"
]


def parse_size(name: str) -> float:
    m = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*[bB]", name)
    if not m:
        raise ValueError(f"cannot parse size from: {name}")
    return float(m[-1])


def parse_fam(name: str) -> str:
    for fam in FAMS:
        if name.startswith(fam):
            return fam
    raise ValueError(f"no family for: {name}")


def is_big(name: str) -> bool:
    return any(p in name for p in BIG_PAT)


#load the dataset that is alraedy preprocessed
df_n = pd.read_csv(CSV_N)
df_k = pd.read_csv(CSV_K)

if "model" not in df_n.columns or "model" not in df_k.columns:
    raise ValueError("CSVs need 'model' column")

df_n = df_n.set_index("model")
df_k = df_k.set_index("model")
df_k = df_k.loc[df_n.index, df_n.columns]

m_names = df_n.index.to_numpy()
t_names = df_n.columns.to_numpy()

n = df_n.to_numpy(dtype=int)
k = df_k.to_numpy(dtype=int)

M, T = n.shape
assert k.shape == (M, T)

sizes = np.array([parse_size(m) for m in m_names], dtype=float)
if sizes.shape != (M,):
    raise ValueError("size shape mismatch")

fam_raw = [parse_fam(m) for m in m_names]
fam_unique = np.array(sorted(set(fam_raw)))
F = len(fam_unique)
fam_idx = np.array([np.where(fam_unique == f)[0][0] for f in fam_raw], dtype=int)

big_mask = np.array([is_big(m) for m in m_names])
big_models = m_names[big_mask]


def build_and_sample(k,n,sizes,m_names,t_names,fam_names,fam_idx,seed=123):
    coords = {"model": m_names, "task": t_names, "family": fam_names}

    with pm.Model(coords=coords) as model:
        n_obs = pm.Data("n_obs", n, dims=("model", "task"))
        k_obs = pm.Data("k_obs", k, dims=("model", "task"))
        size = pm.Data("size", sizes, dims=("model",))
        fam = pm.Data("fam_idx", fam_idx, dims=("model",))

        mu_th = pm.Normal("mu_theta", mu=0.0, sigma=2.0)
        sig_fam = pm.HalfNormal("sigma_family", sigma=1.0)
        sig_th = pm.HalfNormal("sigma_theta", sigma=1.0)
        beta = pm.Normal("beta_size", mu=0.0, sigma=1.0)

        th_fam = pm.Normal("theta_family", mu=mu_th, sigma=sig_fam, dims=("family",))

        th_mean = th_fam[fam] + beta * pm.math.log(size)
        theta = pm.Normal("theta", mu=th_mean, sigma=sig_th, dims=("model",))
        mu_d = pm.Normal("mu_delta", mu=0.0, sigma=2.0)
        sig_d = pm.HalfNormal("sigma_delta", sigma=1.0)
        delta = pm.Normal("delta", mu=mu_d, sigma=sig_d, dims=("task",))

        logit_p = theta[:, None] - delta[None, :]
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p), dims=("model", "task"))

        pm.Binomial("k_like", n=n_obs, p=p, observed=k_obs, dims=("model", "task"))

        idata = pmjax.sample_numpyro_nuts(draws=2000,tune=2000,chains=4,target_accept=0.9,random_seed=seed)

    return model, idata


def summarize_p(idata, hdi=0.95):
    p_post = idata.posterior["p"]
    ms = p_post.coords["model"].values
    ts = p_post.coords["task"].values
    out = {}
    for m in ms:
        out[m] = {}
        for t in ts:
            x = p_post.sel(model=m, task=t).values.flatten()
            mu = x.mean()
            lo, hi = az.hdi(x, hdi_prob=hdi)
            out[m][t] = {"mean": float(mu), "hdi_3%": float(lo), "hdi_97%": float(hi)}
    return out


def summarize_subset(idata, names):
    full = summarize_p(idata)
    sub = {}
    for m in names:
        if m not in full:
            raise ValueError(f"model {m} not in posterior coords")
        sub[m] = full[m]
    return sub


def save_pred_csv(idata, out_path):
    p = idata.posterior["p"]
    p_stack = p.stack(sample=("chain", "draw"))

    mu = p_stack.mean(dim="sample")
    lo = p_stack.quantile(0.025, dim="sample")
    hi = p_stack.quantile(0.975, dim="sample")

    ms = p.coords["model"].values
    ts = p.coords["task"].values

    rows = []
    for m in ms:
        for t in ts:
            rows.append(
                {
                    "model": str(m),
                    "subject": str(t),
                    "pred_mean": float(mu.sel(model=m, task=t).values),
                    "pred_lower": float(lo.sel(model=m, task=t).values),
                    "pred_upper": float(hi.sel(model=m, task=t).values),
                }
            )

    df = pd.DataFrame(rows)
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("saved pred csv:", out_path)


def main():
    d = os.path.dirname(OUT_TXT)
    if d:
        os.makedirs(d, exist_ok=True)

    print("fit model")
    model, idata = build_and_sample(k=k,n=n,sizes=sizes,m_names=m_names,t_names=t_names,fam_names=fam_unique,fam_idx=fam_idx,seed=SEED)
    _ = model

    try:
        d_nc = os.path.dirname(OUT_NC)
        if d_nc:
            os.makedirs(d_nc, exist_ok=True)
        idata.to_netcdf(OUT_NC)
        print("saved nc:", OUT_NC)
    except Exception as e:
        print("warn: could not save nc:", e)

    save_pred_csv(idata, OUT_PRED)

    lines = []

    def log(s: str):
        print(s)
        lines.append(s)

    full = summarize_p(idata)
    log("posterior per model/task:")
    for m in m_names:
        log(f"\n[{m}]")
        for t in t_names:
            st = full[m][t]
            log(f"  {t}: mean={st['mean']:.3f}, ci=({st['hdi_3%']:.3f},{st['hdi_97%']:.3f})")

    if len(big_models) > 0:
        log("\nlarge models:")
        sub = summarize_subset(idata, big_models)
        for m in big_models:
            log(f"\n[{m}]")
            for t in t_names:
                st = sub[m][t]
                log(f"  {t}: mean={st['mean']:.3f}, ci=({st['hdi_3%']:.3f},{st['hdi_97%']:.3f})")
    else:
        log("\nno large models matched patterns")

    N_future = 100
    p_post = idata.posterior["p"]

    if len(big_models) > 0:
        log(f"\npredictive for large models, N={N_future}")
        for m in big_models:
            midx = int(np.where(p_post.coords["model"].values == m)[0][0])
            log(f"\n{m}:")
            for j, t in enumerate(t_names):
                ps = p_post.isel(model=midx, task=j).values.flatten()
                fut = np.random.binomial(N_future, ps)
                mu = fut.mean()
                lo, hi = np.percentile(fut, [2.5, 97.5])
                log(f"  {t}: mean={mu:.1f}/{N_future}, band=({lo:.0f},{hi:.0f})")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("saved txt:", OUT_TXT)


if __name__ == "__main__":
    main()
