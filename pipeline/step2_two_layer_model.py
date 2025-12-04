#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Step2

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


@dataclass
class FamilyHierarchicalModelConfig:
    input_total_csv: str
    input_correct_csv: str
    output_dir: str

    results_txt: str = "llm_hierarchical_family_results.txt"
    pred_txt: str = "llm_hierarchical_family_predictions.txt"
    pred_csv: str = "llm_hierarchical_family_predictions_counts.csv"

    subsample_model_patterns: List[str] = field(default_factory=list)
    family_prefixes: List[str] = field(default_factory=list)

    use_jax_sampler: bool = True
    mcmc_draws: int = 2000
    mcmc_tune: int = 2000
    mcmc_chains: int = 4
    mcmc_target_accept: float = 0.9
    mcmc_random_seed: int = 123

    priors_mu_theta_mean: float = 0.0
    priors_mu_theta_sd: float = 2.0
    priors_sigma_family_sd: float = 1.0
    priors_sigma_theta_sd: float = 1.0

    priors_beta_size_mean: float = 0.0
    priors_beta_size_sd: float = 1.0

    priors_mu_delta_mean: float = 0.0
    priors_mu_delta_sd: float = 2.0
    priors_sigma_delta_sd: float = 1.0

    hdi_prob: float = 0.95

    model_size_regex: str = r"([0-9]+(?:\.[0-9]+)?)\s*[bB]"
    log_size: bool = True

    predictive_enabled: bool = True
    predictive_interval_prob: float = 0.95
    predictive_random_seed: Optional[int] = None


@dataclass
class FamilyHierarchicalModelOutputs:
    summary_txt_path: str
    predictive_txt_path: str
    pred_counts_csv_path: Optional[str]
    model_names: np.ndarray
    task_names: np.ndarray


def parse_model_size(name: str, regex: str) -> float:
    m = re.findall(regex, name)
    if not m:
        raise ValueError(f"cannot parse size from: {name}")
    return float(m[-1])


def is_big_new_model(name: str, pats: List[str]) -> bool:
    return any(p in name for p in pats)


def parse_family(name: str, prefixes: List[str]) -> str:
    for fam in prefixes:
        if name.startswith(fam):
            return fam
    raise ValueError(f"no family matched for: {name!r}")


def load_counts_from_csv(csv_n: str, csv_k: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df_n, df_k = pd.read_csv(csv_n), pd.read_csv(csv_k)
    if "model" not in df_n.columns or "model" not in df_k.columns:
        raise ValueError("CSV must have 'model' col")
    df_n, df_k = df_n.set_index("model"), df_k.set_index("model")
    df_k = df_k.loc[df_n.index, df_n.columns]
    m_names, t_names = df_n.index.to_numpy(), df_n.columns.to_numpy()
    n, k = df_n.to_numpy(dtype=int), df_k.to_numpy(dtype=int)
    if k.shape != n.shape:
        raise ValueError(f"shape mismatch n={n.shape}, k={k.shape}")
    return n, k, m_names, t_names


def build_and_sample_family_model(
    n: np.ndarray,
    k: np.ndarray,
    sizes: np.ndarray,
    fam_idx: np.ndarray,
    m_names: np.ndarray,
    t_names: np.ndarray,
    fam_names: np.ndarray,
    cfg: FamilyHierarchicalModelConfig,
) -> Tuple[pm.Model, az.InferenceData, str]:
    coords = {"model": m_names, "task": t_names, "family": fam_names}

    with pm.Model(coords=coords) as model:
        n_obs = pm.Data("n_obs", n, dims=("model", "task"))
        k_obs = pm.Data("k_obs", k, dims=("model", "task"))
        size_data = pm.Data("size", sizes, dims=("model",))
        fam_data = pm.Data("fam_idx", fam_idx, dims=("model",))

        mu_theta = pm.Normal("mu_theta", mu=cfg.priors_mu_theta_mean, sigma=cfg.priors_mu_theta_sd)
        sig_family = pm.HalfNormal("sigma_family", sigma=cfg.priors_sigma_family_sd)
        sig_theta = pm.HalfNormal("sigma_theta", sigma=cfg.priors_sigma_theta_sd)
        beta_size = pm.Normal("beta_size", mu=cfg.priors_beta_size_mean, sigma=cfg.priors_beta_size_sd)

        theta_family = pm.Normal("theta_family", mu=mu_theta, sigma=sig_family, dims=("family",))

        mu_delta = pm.Normal("mu_delta", mu=cfg.priors_mu_delta_mean, sigma=cfg.priors_mu_delta_sd)
        sig_delta = pm.HalfNormal("sigma_delta", sigma=cfg.priors_sigma_delta_sd)
        delta = pm.Normal("delta", mu=mu_delta, sigma=sig_delta, dims=("task",))

        size_arg = pm.math.log(size_data) if cfg.log_size else size_data
        th_mean = theta_family[fam_data] + beta_size * size_arg
        theta = pm.Normal("theta", mu=th_mean, sigma=sig_theta, dims=("model",))

        logit_p = theta[:, None] - delta[None, :]
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p), dims=("model", "task"))

        _ = pm.Binomial("k_like", n=n_obs, p=p, observed=k_obs, dims=("model", "task"))

        info = "pm.sample"
        if cfg.use_jax_sampler:
            try:
                from pymc.sampling import jax as pmjax  # type: ignore
                import jax

                try:
                    backend, devs = jax.default_backend(), jax.devices()
                except Exception:
                    backend, devs = "unknown", []
                gpus = [d for d in devs if getattr(d, "platform", "") == "gpu"]
                info = f"jax nuts ({backend}, {len(gpus)} gpu)" if gpus else f"jax nuts ({backend}, cpu)"
                idata = pmjax.sample_numpyro_nuts(
                    draws=cfg.mcmc_draws,
                    tune=cfg.mcmc_tune,
                    chains=cfg.mcmc_chains,
                    target_accept=cfg.mcmc_target_accept,
                    random_seed=cfg.mcmc_random_seed,
                    chain_method="parallel",
                )
            except Exception as e:
                info = f"pm.sample (fallback: {type(e).__name__})"
                idata = pm.sample(
                    draws=cfg.mcmc_draws,
                    tune=cfg.mcmc_tune,
                    chains=cfg.mcmc_chains,
                    target_accept=cfg.mcmc_target_accept,
                    random_seed=cfg.mcmc_random_seed,
                    return_inferencedata=True,
                )
        else:
            idata = pm.sample(
                draws=cfg.mcmc_draws,
                tune=cfg.mcmc_tune,
                chains=cfg.mcmc_chains,
                target_accept=cfg.mcmc_target_accept,
                random_seed=cfg.mcmc_random_seed,
                return_inferencedata=True,
            )

    return model, idata, info


def summarize_model_task_accuracy(
    idata: az.InferenceData,
    hdi_prob: float,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    p_post = idata.posterior["p"]
    m_names, t_names = p_post.coords["model"].values, p_post.coords["task"].values
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m in m_names:
        out[m] = {}
        for t in t_names:
            x = p_post.sel(model=m, task=t).values.flatten()
            mu = x.mean()
            lo, hi = az.hdi(x, hdi_prob=hdi_prob)
            out[m][t] = {"mean": float(mu), "hdi_3%": float(lo), "hdi_97%": float(hi)}
    return out


def summarize_new_model(
    idata: az.InferenceData,
    name: str,
    hdi_prob: float,
) -> Dict[str, Dict[str, float]]:
    sm = summarize_model_task_accuracy(idata, hdi_prob)
    if name not in sm:
        raise ValueError(f"model {name} not in posterior coords")
    return sm[name]


def run_step2_family_hierarchical(
    cfg: FamilyHierarchicalModelConfig,
) -> Tuple[az.InferenceData, FamilyHierarchicalModelOutputs]:
    os.makedirs(cfg.output_dir, exist_ok=True)

    n, k, m_names, t_names = load_counts_from_csv(cfg.input_total_csv, cfg.input_correct_csv)
    M, T = n.shape
    assert k.shape == (M, T)

    sizes = np.array([parse_model_size(m, cfg.model_size_regex) for m in m_names], dtype=float)
    if sizes.shape != (M,):
        raise ValueError(f"sizes shape mismatch: {sizes.shape} vs M={M}")

    if not cfg.family_prefixes:
        raise ValueError("family_prefixes must not be empty")

    fam_raw = [parse_family(m, cfg.family_prefixes) for m in m_names]
    fam_unique = np.array(sorted(set(fam_raw)))
    fam_idx = np.array([np.where(fam_unique == f)[0][0] for f in fam_raw], dtype=int)

    big_models = np.array([m for m in m_names if is_big_new_model(m, cfg.subsample_model_patterns)])
    big_models_set = set(map(str, big_models))

    sum_txt = os.path.join(cfg.output_dir, cfg.results_txt)
    pred_txt = os.path.join(cfg.output_dir, cfg.pred_txt)
    pred_csv = os.path.join(cfg.output_dir, cfg.pred_csv)

    with open(sum_txt, "w", encoding="utf-8") as fs, open(pred_txt, "w", encoding="utf-8") as fp:
        def log(msg: str = "") -> None:
            msg = str(msg)
            print(msg)
            fs.write(msg + "\n")

        def log_pred(msg: str = "") -> None:
            msg = str(msg)
            print(msg)
            fp.write(msg + "\n")

        log("fit model")
        model, idata, info = build_and_sample_family_model(
            n=n,
            k=k,
            sizes=sizes,
            fam_idx=fam_idx,
            m_names=m_names,
            t_names=t_names,
            fam_names=fam_unique,
            cfg=cfg,
        )
        _ = model

        posterior_nc = os.path.join(cfg.output_dir, "llm_hierarchical_family_posterior.nc")
        az.to_netcdf(idata, posterior_nc)
        log(f"posterior: {posterior_nc}")
        log(f"sampler: {info}")

        summary = az.summary(
            idata,
            var_names=["mu_theta", "sigma_family", "sigma_theta", "beta_size", "mu_delta", "sigma_delta", "theta", "delta"],
            round_to=3,
        )
        diag_txt = os.path.join(cfg.output_dir, "llm_hierarchical_family_diagnostics.txt")
        with open(diag_txt, "w", encoding="utf-8") as fd:
            fd.write(summary.to_string())
        log(f"diag: {diag_txt}")

        max_rhat = float(summary["r_hat"].max())
        min_ess_bulk = float(summary["ess_bulk"].min())
        min_ess_tail = float(summary["ess_tail"].min())
        n_div = int(idata.sample_stats["diverging"].sum().values)
        try:
            bfmi = az.bfmi(idata)
            min_bfmi = float(bfmi.min())
        except Exception:
            min_bfmi = float("nan")

        log(f"conv: rhat_max={max_rhat:.3f}, ess_bulk_min={min_ess_bulk:.1f}, ess_tail_min={min_ess_tail:.1f}, div={n_div}, bfmi_min={min_bfmi:.3f}")

        rhat_ds = az.rhat(idata, var_names=["theta", "p"])
        ess_bulk_ds = az.ess(idata, method="bulk", var_names=["theta", "p"])
        ess_tail_ds = az.ess(idata, method="tail", var_names=["theta", "p"])

        rhat_theta_da, rhat_p_da = rhat_ds["theta"], rhat_ds["p"]
        ess_bulk_theta_da, ess_tail_theta_da = ess_bulk_ds["theta"], ess_tail_ds["theta"]
        ess_bulk_p_da, ess_tail_p_da = ess_bulk_ds["p"], ess_tail_ds["p"]

        rows = []
        for m in rhat_theta_da.coords["model"].values:
            m_str = str(m)
            if m_str not in big_models_set:
                continue
            th_rhat = float(rhat_theta_da.sel(model=m).values)
            th_ess_bulk = float(ess_bulk_theta_da.sel(model=m).values)
            th_ess_tail = float(ess_tail_theta_da.sel(model=m).values)
            rhat_p_m, ess_bulk_p_m, ess_tail_p_m = rhat_p_da.sel(model=m), ess_bulk_p_da.sel(model=m), ess_tail_p_da.sel(model=m)
            p_max_rhat = float(rhat_p_m.max(dim="task").values)
            p_min_ess_bulk = float(ess_bulk_p_m.min(dim="task").values)
            p_min_ess_tail = float(ess_tail_p_m.min(dim="task").values)
            rows.append(
                {
                    "model": m_str,
                    "theta_rhat": th_rhat,
                    "theta_ess_bulk": th_ess_bulk,
                    "theta_ess_tail": th_ess_tail,
                    "p_max_rhat": p_max_rhat,
                    "p_min_ess_bulk": p_min_ess_bulk,
                    "p_min_ess_tail": p_min_ess_tail,
                }
            )

        diag_models_csv = os.path.join(cfg.output_dir, "llm_hierarchical_family_diagnostics_per_model.csv")
        pd.DataFrame(rows).to_csv(diag_models_csv, index=False)
        log(f"diag_per_model: {diag_models_csv}")

        plots_dir = os.path.join(cfg.output_dir, "diagnostics_plots")
        os.makedirs(plots_dir, exist_ok=True)
        log(f"plots: {plots_dir}")

        def _sanitize(name: str) -> str:
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

        def _save_fig(path: str):
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            log(f"plot: {path}")

        try:
            axes = az.plot_trace(idata, var_names=["mu_theta", "sigma_family", "sigma_theta", "beta_size", "mu_delta", "sigma_delta"])
            plt.suptitle("Global params (family)", y=0.975)
            _save_fig(os.path.join(plots_dir, "trace_global_family.png"))
        except Exception as e:
            log(f"plot global fail: {e}")

        try:
            theta_models = list(idata.posterior["theta"].coords["model"].values)
            task_names_list = list(idata.posterior["p"].coords["task"].values)
            theta_sel = [m for m in theta_models if str(m) in big_models_set] or theta_models[:4]
            for m in theta_sel:
                m_str = str(m)
                _ = az.plot_trace(idata, var_names=["theta"], coords={"model": [m]})
                plt.suptitle(f"Theta (family)\n{m_str}", y=0.95)
                _save_fig(os.path.join(plots_dir, f"trace_theta_family_{_sanitize(m_str)}.png"))
            delta_tasks_sel = task_names_list[:4]
            _ = az.plot_trace(idata, var_names=["delta"], coords={"task": delta_tasks_sel})
            plt.suptitle("Delta (family)", y=0.95)
            _save_fig(os.path.join(plots_dir, "trace_delta_family_subset.png"))
        except Exception as e:
            log(f"plot theta/delta fail: {e}")

        _ = summarize_model_task_accuracy(idata, hdi_prob=cfg.hdi_prob)

        saved_pred_csv = None
        if cfg.predictive_enabled and big_models.size > 0:
            p_post = idata.posterior["p"]
            p_models = p_post.coords["model"].values
            mask_non_big = np.array([not is_big_new_model(m, cfg.subsample_model_patterns) for m in m_names])
            if np.any(mask_non_big):
                ref_row = n[mask_non_big][0]
            else:
                ref_row = n[0]
                log_pred("use first row as ref")
            N_ref = ref_row.astype(int)
            rng = np.random.default_rng(cfg.predictive_random_seed) if cfg.predictive_random_seed is not None else np.random.default_rng()
            pm_mat = np.zeros((len(big_models), len(t_names)), dtype=float)
            alpha = 1.0 - cfg.predictive_interval_prob
            q_lo, q_hi = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
            log_pred("posterior predictive:")
            for i_m, name in enumerate(big_models):
                if name not in p_models:
                    log_pred(f"skip {name}")
                    continue
                m_idx = int(np.where(p_models == name)[0][0])
                log_pred(f"[{name}]")
                for j_t, t in enumerate(t_names):
                    N = int(N_ref[j_t])
                    if N <= 0:
                        log_pred(f"  {t}: N<=0")
                        continue
                    p_samp = p_post.isel(model=m_idx, task=j_t).values.flatten()
                    k_future = rng.binomial(N, p_samp)
                    mu = k_future.mean()
                    lo, hi = np.percentile(k_future, [q_lo, q_hi])
                    pm_mat[i_m, j_t] = mu
                    log_pred(f"  {t}: {mu:.1f}/{N}, [{lo:.0f},{hi:.0f}]")
            df_pred = pd.DataFrame(pm_mat, index=big_models, columns=t_names)
            df_pred_int = df_pred.round().astype(int)
            df_pred_int.to_csv(pred_csv, index_label="model")
            saved_pred_csv = pred_csv
            log_pred(f"pred_csv: {pred_csv}")

        log(f"summary_txt: {sum_txt}")
        log_pred(f"pred_txt: {pred_txt}")

    outs = FamilyHierarchicalModelOutputs(
        summary_txt_path=sum_txt,
        predictive_txt_path=pred_txt,
        pred_counts_csv_path=saved_pred_csv,
        model_names=m_names,
        task_names=t_names,
    )
    return idata, outs


if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        raise SystemExit("need pyyaml for CLI")
    yaml_path = "project.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)
    cfg_step = FamilyHierarchicalModelConfig(**cfg_all["step2_family_hierarchical"])
    run_step2_family_hierarchical(cfg_step)
