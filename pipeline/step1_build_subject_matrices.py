# step1_build_subject_matrices.py

import os
import glob
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import Counter

import pandas as pd


@dataclass
class SubjectAggregationConfig:
    data_dir: str
    output_dir: str
    subsample_model_patterns: List[str]
    subsample_fraction: float
    random_seed: int = 42
    keep_models: Optional[List[str]] = None
    keep_subjects: Optional[List[str]] = None


@dataclass
class SubjectAggregationOutputs:
    full_total_csv: str
    full_correct_csv: str
    incomplete_total_csv: str
    incomplete_correct_csv: str


def model_needs_subsample(name: str, base: str, pats: List[str]) -> bool:
    s = name + " " + base
    return any(p in s for p in pats)


def run_step1_subject_aggregation(cfg: SubjectAggregationConfig) -> SubjectAggregationOutputs:
    os.makedirs(cfg.output_dir, exist_ok=True)
    random.seed(cfg.random_seed)

    js = sorted(glob.glob(os.path.join(cfg.data_dir, "*.json")))
    if not js:
        raise RuntimeError(f"no json files under {cfg.data_dir}")

    tot_by_model: Dict[str, Counter] = {}
    cor_by_model: Dict[str, Counter] = {}
    hits_by_model: Dict[str, Dict[str, List[int]]] = {}

    for p in js:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)

        base = os.path.basename(p)
        m = d.get("model_name", base)

        c_tot = Counter()
        c_cor = Counter()
        subj_hits: Dict[str, List[int]] = {}

        for r in d["individual_results"]:
            subj = r["subject"]
            c_tot[subj] += 1

            v = r.get("is_correct", 0)
            try:
                v_int = int(round(float(v)))
            except (TypeError, ValueError):
                v_int = 0

            c_cor[subj] += v_int
            subj_hits.setdefault(subj, []).append(v_int)

        tot_by_model[m] = c_tot
        cor_by_model[m] = c_cor
        hits_by_model[m] = subj_hits

    all_models = list(tot_by_model.keys())
    if not all_models:
        raise RuntimeError("no models found")

    all_subsample = [m for m in all_models if model_needs_subsample(m, "", cfg.subsample_model_patterns)]

    if cfg.keep_models:
        keep_set = set(cfg.keep_models)
        kept = [m for m in all_models if m in keep_set]
        if not kept:
            raise RuntimeError("keep_models removed all models")

        dropped_targets = [m for m in all_subsample if m not in kept]
        if dropped_targets:
            raise RuntimeError(
                "these subsampled models were dropped by keep_models, but later steps expect them:\n"
                f"  {dropped_targets}"
            )

        tot_by_model = {m: tot_by_model[m] for m in kept}
        cor_by_model = {m: cor_by_model[m] for m in kept}
        hits_by_model = {m: hits_by_model[m] for m in kept}
        models = kept
    else:
        models = all_models

    if cfg.keep_subjects:
        keep_subj = set(cfg.keep_subjects)

        subj_union = set()
        for c in tot_by_model.values():
            subj_union.update(c.keys())

        missing = sorted(keep_subj - subj_union)
        if missing:
            raise RuntimeError(
                "subjects in keep_subjects not found in data:\n"
                f"  {missing}"
            )

        for m in models:
            c_tot = tot_by_model[m]
            c_cor = cor_by_model[m]
            sh = hits_by_model[m]

            new_tot = Counter({s: c for s, c in c_tot.items() if s in keep_subj})
            new_cor = Counter({s: c for s, c in c_cor.items() if s in keep_subj})
            new_hits = {s: arr for s, arr in sh.items() if s in keep_subj}

            tot_by_model[m] = new_tot
            cor_by_model[m] = new_cor
            hits_by_model[m] = new_hits

    subs = sorted({s for c in tot_by_model.values() for s in c.keys()})
    if not subs:
        raise RuntimeError("no subjects left after filtering")

    ref = models[0]
    ref_sub = set(tot_by_model[ref].keys())

    print("check subject sets")
    for m in models:
        s = set(tot_by_model[m].keys())
        if s != ref_sub:
            only_ref = sorted(ref_sub - s)
            only_cur = sorted(s - ref_sub)
            print("WARN subjects:", m)
            if only_ref:
                print("  only_ref:", only_ref)
            if only_cur:
                print("  only_cur:", only_cur)
        else:
            print("OK subjects:", m)

    print("check question counts")
    ref_cnt = tot_by_model[ref]
    for m in models:
        cnt = tot_by_model[m]
        diff = []
        for subj in ref_sub:
            a = ref_cnt[subj]
            b = cnt.get(subj, 0)
            if a != b:
                diff.append((subj, a, b))
        if diff:
            print("WARN counts:", m)
            for subj, a, b in diff:
                print(f"  {subj}: ref={a} cur={b}")
        else:
            print("OK counts:", m)

    df_tot = pd.DataFrame(index=models, columns=subs)
    df_cor = pd.DataFrame(index=models, columns=subs)
    for m in models:
        t = tot_by_model[m]
        c = cor_by_model[m]
        for subj in subs:
            df_tot.loc[m, subj] = t.get(subj, 0)
            df_cor.loc[m, subj] = c.get(subj, 0)

    df_tot = df_tot.fillna(0).astype(int)
    df_cor = df_cor.fillna(0).astype(int)

    out_tot = os.path.join(cfg.output_dir, "mmlu_subject_total_questions_by_model.csv")
    out_cor = os.path.join(cfg.output_dir, "mmlu_subject_num_correct_by_model.csv")

    df_tot.to_csv(out_tot, index_label="model")
    df_cor.to_csv(out_cor, index_label="model")

    print("saved full:")
    print("  total ->", out_tot)
    print("  correct ->", out_cor)

    inc_tot: Dict[str, Counter] = {}
    inc_cor: Dict[str, Counter] = {}

    for m in models:
        need = model_needs_subsample(m, "", cfg.subsample_model_patterns)

        if not need:
            inc_tot[m] = tot_by_model[m]
            inc_cor[m] = cor_by_model[m]
            continue

        t_inc = Counter()
        c_inc = Counter()
        sh = hits_by_model[m]

        for subj, arr in sh.items():
            n_q = len(arr)
            if n_q == 0:
                continue

            k_q = int(round(cfg.subsample_fraction * n_q))
            k_q = max(0, min(n_q, k_q))

            if k_q == 0:
                t_inc[subj] = 0
                c_inc[subj] = 0
            else:
                pick = random.sample(arr, k_q)
                t_inc[subj] = k_q
                c_inc[subj] = sum(pick)

        inc_tot[m] = t_inc
        inc_cor[m] = c_inc

    df_tot_inc = pd.DataFrame(index=models, columns=subs)
    df_cor_inc = pd.DataFrame(index=models, columns=subs)
    for m in models:
        t = inc_tot[m]
        c = inc_cor[m]
        for subj in subs:
            df_tot_inc.loc[m, subj] = t.get(subj, 0)
            df_cor_inc.loc[m, subj] = c.get(subj, 0)

    df_tot_inc = df_tot_inc.fillna(0).astype(int)
    df_cor_inc = df_cor_inc.fillna(0).astype(int)

    pct = int(round(cfg.subsample_fraction * 100))
    out_tot_inc = os.path.join(cfg.output_dir, f"mmlu_subject_total_questions_by_model_incomplete_{pct}pct.csv")
    out_cor_inc = os.path.join(cfg.output_dir, f"mmlu_subject_num_correct_by_model_incomplete_{pct}pct.csv")

    df_tot_inc.to_csv(out_tot_inc, index_label="model")
    df_cor_inc.to_csv(out_cor_inc, index_label="model")

    print("saved incomplete:")
    print("  total ->", out_tot_inc)
    print("  correct ->", out_cor_inc)
    print("step1 done")

    return SubjectAggregationOutputs(
        full_total_csv=out_tot,
        full_correct_csv=out_cor,
        incomplete_total_csv=out_tot_inc,
        incomplete_correct_csv=out_cor_inc,
    )
