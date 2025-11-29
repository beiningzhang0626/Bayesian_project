# step2_hierarchical_model.py

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
class HierarchicalModelConfig:
    input_total_csv: str
    input_correct_csv: str
    output_dir: str

    results_txt: str = "llm_hierarchical_results.txt"
    pred_txt: str = "llm_hierarchical_predictions.txt"
    pred_csv: str = "llm_hierarchical_predictions_counts.csv"

    subsample_model_patterns: List[str] = field(default_factory=list)

    use_jax_sampler: bool = True
    mcmc_draws: int = 2000
    mcmc_tune: int = 2000
    mcmc_chains: int = 4
    mcmc_target_accept: float = 0.9
    mcmc_random_seed: int = 123

    priors_mu_theta_mean: float = 0.0
    priors_mu_theta_sd: float = 2.0
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
class HierarchicalModelOutputs:
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


def load_counts_from_csv(csv_n: str, csv_k: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df_n = pd.read_csv(csv_n)
    df_k = pd.read_csv(csv_k)

    if "model" not in df_n.columns or "model" not in df_k.columns:
        raise ValueError("CSV must have 'model' column")

    df_n = df_n.set_index("model")
    df_k = df_k.set_index("model")
    df_k = df_k.loc[df_n.index, df_n.columns]

    m_names = df_n.index.to_numpy()
    t_names = df_n.columns.to_numpy()

    n = df_n.to_numpy(dtype=int)
    k = df_k.to_numpy(dtype=int)
    if k.shape != n.shape:
        raise ValueError(f"shape mismatch n={n.shape}, k={k.shape}")
    return n, k, m_names, t_names


def build_and_sample_model(
    n: np.ndarray,
    k: np.ndarray,
    sizes: np.ndarray,
    m_names: np.ndarray,
    t_names: np.ndarray,
    cfg: HierarchicalModelConfig,
) -> Tuple[pm.Model, az.InferenceData, str]:
    coords = {"model": m_names, "task": t_names}

    with pm.Model(coords=coords) as model:
        n_obs = pm.Data("n_obs", n, dims=("model", "task"))
        k_obs = pm.Data("k_obs", k, dims=("model", "task"))
        size_data = pm.Data("size", sizes, dims=("model",))

        mu_theta = pm.Normal("mu_theta", mu=cfg.priors_mu_theta_mean, sigma=cfg.priors_mu_theta_sd)
        sig_theta = pm.HalfNormal("sigma_theta", sigma=cfg.priors_sigma_theta_sd)
        beta_size = pm.Normal("beta_size", mu=cfg.priors_beta_size_mean, sigma=cfg.priors_beta_size_sd)

        mu_delta = pm.Normal("mu_delta", mu=cfg.priors_mu_delta_mean, sigma=cfg.priors_mu_delta_sd)
        sig_delta = pm.HalfNormal("sigma_delta", sigma=cfg.priors_sigma_delta_sd)

        if cfg.log_size:
            size_arg = pm.math.log(size_data)
        else:
            size_arg = size_data

        th_mean = mu_theta + beta_size * size_arg
        theta = pm.Normal("theta", mu=th_mean, sigma=sig_theta, dims=("model",))
        delta = pm.Normal("delta", mu=mu_delta, sigma=sig_delta, dims=("task",))

        logit_p = theta[:, None] - delta[None, :]
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p), dims=("model", "task"))

        _ = pm.Binomial("k_like", n=n_obs, p=p, observed=k_obs, dims=("model", "task"))

        info = "pm.sample (pytensor)"

        if cfg.use_jax_sampler:
            try:
                from pymc.sampling import jax as pmjax
                import jax

                try:
                    backend = jax.default_backend()
                    devs = jax.devices()
                except Exception:
                    backend, devs = "unknown", []

                gpus = [d for d in devs if getattr(d, "platform", "") == "gpu"]
                if gpus:
                    info = f"jax nuts ({backend}, {len(gpus)} gpu)"
                else:
                    info = f"jax nuts ({backend}, cpu)"

                idata = pmjax.sample_numpyro_nuts(
                    draws=cfg.mcmc_draws,
                    tune=cfg.mcmc_tune,
                    chains=cfg.mcmc_chains,
                    target_accept=cfg.mcmc_target_accept,
                    random_seed=cfg.mcmc_random_seed,
                    chain_method="parallel",
                )
            except Exception as e:
                info = f"pm.sample (fallback from jax: {type(e).__name__})"
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
    m_names = p_post.coords["model"].values
    t_names = p_post.coords["task"].values

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


def run_step2_hierarchical(cfg: HierarchicalModelConfig) -> Tuple[az.InferenceData, HierarchicalModelOutputs]:
    os.makedirs(cfg.output_dir, exist_ok=True)

    n, k, m_names, t_names = load_counts_from_csv(cfg.input_total_csv, cfg.input_correct_csv)
    M, T = n.shape
    assert k.shape == (M, T)

    sizes = np.array([parse_model_size(m, cfg.model_size_regex) for m in m_names], dtype=float)

    big_models = np.array([m for m in m_names if is_big_new_model(m, cfg.subsample_model_patterns)])
    big_models_set = set(map(str, big_models))  # only use big models for diagnostics

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

        #log("fit")
        model, idata, info = build_and_sample_model(n, k, sizes, m_names, t_names, cfg)

        posterior_nc = os.path.join(cfg.output_dir, "llm_hierarchical_posterior.nc")
        az.to_netcdf(idata, posterior_nc)
        log(f"posterior saved: {posterior_nc}")

        summary = az.summary(
            idata,
            var_names=["mu_theta", "sigma_theta", "beta_size", "mu_delta", "sigma_delta", "theta", "delta"],
            round_to=3,
        )
        diag_txt = os.path.join(cfg.output_dir, "llm_hierarchical_diagnostics.txt")
        with open(diag_txt, "w", encoding="utf-8") as fd:
            fd.write(summary.to_string())
        log(f"diagnostics summary saved: {diag_txt}")

        max_rhat = float(summary["r_hat"].max())
        min_ess_bulk = float(summary["ess_bulk"].min())
        min_ess_tail = float(summary["ess_tail"].min())

        n_div = int(idata.sample_stats["diverging"].sum().values)
        try:
            bfmi = az.bfmi(idata)
            min_bfmi = float(bfmi.min())
        except Exception:
            min_bfmi = float("nan")

        log(f"convergence metrics:")
        log(f"  max_rhat      = {max_rhat:.3f}")
        log(f"  min_ess_bulk  = {min_ess_bulk:.1f}")
        log(f"  min_ess_tail  = {min_ess_tail:.1f}")
        log(f"  n_divergences = {n_div}")
        log(f"  min_BFMI      = {min_bfmi:.3f}")

        # Per model diagnostics (restricted to big models)
        rhat_ds = az.rhat(idata, var_names=["theta", "p"])
        ess_bulk_ds = az.ess(idata, method="bulk", var_names=["theta", "p"])
        ess_tail_ds = az.ess(idata, method="tail", var_names=["theta", "p"])

        # DataArrays
        rhat_theta_da = rhat_ds["theta"]          # (model,)
        rhat_p_da = rhat_ds["p"]                  # (model, task)

        ess_bulk_theta_da = ess_bulk_ds["theta"]  # (model,)
        ess_tail_theta_da = ess_tail_ds["theta"]  # (model,)

        ess_bulk_p_da = ess_bulk_ds["p"]          # (model, task)
        ess_tail_p_da = ess_tail_ds["p"]          # (model, task)

        rows = []
        for m in rhat_theta_da.coords["model"].values:
            m_str = str(m)

            # only keep big models in diagnostics
            if m_str not in big_models_set:
                continue

            # theta diagnostics (scalar per model)
            th_rhat = float(rhat_theta_da.sel(model=m).values)
            th_ess_bulk = float(ess_bulk_theta_da.sel(model=m).values)
            th_ess_tail = float(ess_tail_theta_da.sel(model=m).values)

            # p diagnostics aggregated over tasks
            rhat_p_m = rhat_p_da.sel(model=m)             # (task,)
            ess_bulk_p_m = ess_bulk_p_da.sel(model=m)     # (task,)
            ess_tail_p_m = ess_tail_p_da.sel(model=m)     # (task,)

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

        df_models = pd.DataFrame(rows)
        diag_models_csv = os.path.join(cfg.output_dir, "llm_hierarchical_diagnostics_per_model.csv")
        df_models.to_csv(diag_models_csv, index=False)
        log(f"per-model diagnostics saved: {diag_models_csv}")

        plots_dir = os.path.join(cfg.output_dir, "diagnostics_plots")
        os.makedirs(plots_dir, exist_ok=True)
        log(f"saving diagnostic plots under: {plots_dir}")

        def _sanitize(name: str) -> str:
            # safe filename based on model name
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

        def _save_current_fig(path: str):
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            log(f"saved plot: {path}")

        # Global parameters trace plot
        try:
            axes = az.plot_trace(
                idata,
                var_names=["mu_theta", "sigma_theta", "beta_size", "mu_delta", "sigma_delta"],
            )
            plt.suptitle(
                "Global Parameters", y=0.975,
            )

            # Improve x/y labels and legends for trace panels
            if isinstance(axes, np.ndarray):
                axes_arr = np.atleast_2d(axes)
                for ax_row in axes_arr:
                    trace_ax = ax_row[0]  # left column: trace
                    trace_ax.set_xlabel("Parameter Value")
                    trace_ax = ax_row[1]  # left column: trace
                    trace_ax.set_ylabel("Parameter Value")
                    trace_ax.set_xlabel("MCMC Draw")

            _save_current_fig(os.path.join(plots_dir, "trace_global.png"))
        except Exception as e:
            log(f"warn: failed to create global trace plot: {e}")

        # Theta / delta traces, labeled per model / tasks
        try:
            theta_models = list(idata.posterior["theta"].coords["model"].values)
            print(theta_models)
            task_names = list(idata.posterior["p"].coords["task"].values)

            # use only big models for theta traces (fallback to first 4 if none match)
            theta_models_sel = [m for m in theta_models if str(m) in big_models_set]
            if not theta_models_sel:
                theta_models_sel = theta_models[:4]

            # per-model theta trace plots
            for m in theta_models_sel:
                m_str = str(m)
                axes = az.plot_trace(
                    idata,
                    var_names=["theta"],
                    coords={"model": [m]},
                )
                plt.suptitle(
                    f"Theta Trace\n{m_str}",
                    y=0.95,
                )
                if isinstance(axes, np.ndarray):
                    axes_arr = np.atleast_2d(axes)
                    for ax_row in axes_arr:
                        trace_ax = ax_row[0]  # left column: trace
                        trace_ax.set_xlabel("Parameter Value")
                        trace_ax = ax_row[1]  # left column: trace
                        trace_ax.set_ylabel("Parameter Value")
                        trace_ax.set_xlabel("MCMC Draw")

                fname = f"trace_theta_{_sanitize(m_str)}.png"
                _save_current_fig(os.path.join(plots_dir, fname))

            # delta (per-task difficulty) – small subset of tasks, but labeled
            delta_tasks_sel = task_names[:4]
            axes = az.plot_trace(
                idata,
                var_names=["delta"],
                coords={"task": delta_tasks_sel},
            )
            tasks_label = ", ".join(str(t) for t in delta_tasks_sel)
            plt.suptitle(
                f"Delta Trace\nTasks: {tasks_label}",
                y=0.95,
            )
            if isinstance(axes, np.ndarray):
                axes_arr = np.atleast_2d(axes)
                for ax_row in axes_arr:
                    trace_ax = ax_row[0]  # left column: trace
                    trace_ax.set_xlabel("Parameter Value")
                    trace_ax = ax_row[1]  # left column: trace
                    trace_ax.set_ylabel("Parameter Value")
                    trace_ax.set_xlabel("MCMC Draw")
                    
            _save_current_fig(os.path.join(plots_dir, "trace_delta_subset.png"))
        except Exception as e:
            log(f"warn: failed to create theta/delta trace plots: {e}")

        _ = model
        #log(f"sampler: {info}")
        #log("done sampling")

        full_sum = summarize_model_task_accuracy(idata, hdi_prob=cfg.hdi_prob)

        #log("posterior:")
        for m in m_names:
            #log(f"\n[{m}]")
            for t in t_names:
                st = full_sum[m][t]
                #log(f"  {t}: mean={st['mean']:.3f}, ci=({st['hdi_3%']:.3f},{st['hdi_97%']:.3f})")

        #log("\nbig models:")
        if big_models.size == 0:
            #log("  none")
            pass
        else:
            for m in big_models:
                #log(f"  {m}")
                pass

            for name in big_models:
                try:
                    sm = summarize_new_model(idata, name=name, hdi_prob=cfg.hdi_prob)
                except ValueError as e:
                    log(f"warn: {e}")
                    continue
                #log(f"\nsummary for {name}:")
                for t in t_names:
                    st = sm[t]
                    #log(f"  {t}: mean={st['mean']:.3f}, ci=({st['hdi_3%']:.3f},{st['hdi_97%']:.3f})")

        saved_pred_csv = None
        if cfg.predictive_enabled and big_models.size > 0:
            p_post = idata.posterior["p"]
            p_models = p_post.coords["model"].values

            mask_non_big = np.array([not is_big_new_model(m, cfg.subsample_model_patterns) for m in m_names])
            if np.any(mask_non_big):
                ref_row = n[mask_non_big][0]
            else:
                ref_row = n[0]
                log_pred("warn: using first row as ref n")

            N_ref = ref_row.astype(int)

            #log_pred("predictive:")
            if cfg.predictive_random_seed is not None:
                rng = np.random.default_rng(cfg.predictive_random_seed)
            else:
                rng = np.random.default_rng()

            pm_mat = np.zeros((len(big_models), len(t_names)), dtype=float)

            alpha = 1.0 - cfg.predictive_interval_prob
            q_lo = 100.0 * (alpha / 2.0)
            q_hi = 100.0 * (1.0 - alpha / 2.0)

            for i_m, name in enumerate(big_models):
                if name not in p_models:
                    #log_pred(f"skip {name}, not in coords")
                    continue
                m_idx = int(np.where(p_models == name)[0][0])

                #log_pred(f"\n{name}:")
                for j_t, t in enumerate(t_names):
                    N = int(N_ref[j_t])
                    if N <= 0:
                        #log_pred(f"  {t}: N<=0")
                        continue

                    p_samp = p_post.isel(model=m_idx, task=j_t).values.flatten()
                    k_future = rng.binomial(N, p_samp)
                    mu = k_future.mean()
                    lo, hi = np.percentile(k_future, [q_lo, q_hi])

                    pm_mat[i_m, j_t] = mu
                    #log_pred(f"  {t}: mean={mu:.1f}/{N}, band=({lo:.0f},{hi:.0f})")

            df_pred = pd.DataFrame(pm_mat, index=big_models, columns=t_names)
            df_pred_int = df_pred.round().astype(int)
            df_pred_int.to_csv(pred_csv, index_label="model")
            saved_pred_csv = pred_csv
            #log_pred(f"\nsaved: {pred_csv}")

        log(f"\nsummary file: {sum_txt}")
        log_pred(f"\npred file: {pred_txt}")

    outs = HierarchicalModelOutputs(
        summary_txt_path=sum_txt,
        predictive_txt_path=pred_txt,
        pred_counts_csv_path=saved_pred_csv,
        model_names=m_names,
        task_names=t_names,
    )
    return idata, outs