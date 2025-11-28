# step1_build_subject_matrices.py

import os
import glob
import json
import random
from dataclasses import dataclass
from typing import List, Dict
from collections import Counter

import pandas as pd


@dataclass
class SubjectAggregationConfig:
    data_dir: str
    output_dir: str
    subsample_model_patterns: List[str]
    subsample_fraction: float
    random_seed: int = 42


@dataclass
class SubjectAggregationOutputs:
    full_total_csv: str
    full_correct_csv: str
    incomplete_total_csv: str
    incomplete_correct_csv: str


def model_needs_subsample(model_name: str, file_basename: str, patterns: List[str]) -> bool:
    s = model_name + " " + file_basename
    return any(p in s for p in patterns)


def run_step1_subject_aggregation(cfg: SubjectAggregationConfig) -> SubjectAggregationOutputs:
    os.makedirs(cfg.output_dir, exist_ok=True)
    random.seed(cfg.random_seed)

    js_files = sorted(glob.glob(os.path.join(cfg.data_dir, "*.json")))
    if not js_files:
        raise RuntimeError(f"no json files under {cfg.data_dir}")

    tot_by_model: Dict[str, Counter] = {}
    cor_by_model: Dict[str, Counter] = {}
    per_model_subj_correct: Dict[str, Dict[str, List[int]]] = {}

    for p in js_files:
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
                v = int(round(float(v)))
            except (TypeError, ValueError):
                v = 0

            c_cor[subj] += v
            subj_hits.setdefault(subj, []).append(v)

        tot_by_model[m] = c_tot
        cor_by_model[m] = c_cor
        per_model_subj_correct[m] = subj_hits

    models = list(tot_by_model.keys())
    ref = models[0]
    ref_subj = set(tot_by_model[ref].keys())

    print("check subjects")

    for m in models:
        s = set(tot_by_model[m].keys())
        if s != ref_subj:
            only_ref = sorted(ref_subj - s)
            only_m = sorted(s - ref_subj)
            print("warn:", m)
            if only_ref:
                print("  only_ref:", only_ref)
            if only_m:
                print("  only_cur:", only_m)
        else:
            print("ok:", m)

    print("check counts")

    ref_cnt = tot_by_model[ref]
    for m in models:
        cnt = tot_by_model[m]
        diff = []
        for subj in ref_subj:
            a = ref_cnt[subj]
            b = cnt.get(subj, 0)
            if a != b:
                diff.append((subj, a, b))
        if diff:
            print("warn_counts:", m)
            for subj, a, b in diff:
                print(f"  {subj}: {a} vs {b}")
        else:
            print("ok_counts:", m)

    all_subj = sorted({s for c in tot_by_model.values() for s in c.keys()})

    df_tot = pd.DataFrame(index=models, columns=all_subj)
    df_cor = pd.DataFrame(index=models, columns=all_subj)

    for m in models:
        t = tot_by_model[m]
        c = cor_by_model[m]
        for subj in all_subj:
            df_tot.loc[m, subj] = t.get(subj, 0)
            df_cor.loc[m, subj] = c.get(subj, 0)

    df_tot = df_tot.fillna(0).astype(int)
    df_cor = df_cor.fillna(0).astype(int)

    out_tot = os.path.join(cfg.output_dir, "mmlu_subject_total_questions_by_model.csv")
    out_cor = os.path.join(cfg.output_dir, "mmlu_subject_num_correct_by_model.csv")

    df_tot.to_csv(out_tot, index_label="model")
    df_cor.to_csv(out_cor, index_label="model")

    print("saved full")

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
        subj_hits = per_model_subj_correct[m]

        for subj, arr in subj_hits.items():
            n = len(arr)
            if n == 0:
                continue
            k = int(round(cfg.subsample_fraction * n))
            k = max(0, min(n, k))

            if k == 0:
                t_inc[subj] = 0
                c_inc[subj] = 0
            else:
                pick = random.sample(arr, k)
                t_inc[subj] = k
                c_inc[subj] = sum(pick)

        inc_tot[m] = t_inc
        inc_cor[m] = c_inc

    df_tot_inc = pd.DataFrame(index=models, columns=all_subj)
    df_cor_inc = pd.DataFrame(index=models, columns=all_subj)

    for m in models:
        t = inc_tot[m]
        c = inc_cor[m]
        for subj in all_subj:
            df_tot_inc.loc[m, subj] = t.get(subj, 0)
            df_cor_inc.loc[m, subj] = c.get(subj, 0)

    df_tot_inc = df_tot_inc.fillna(0).astype(int)
    df_cor_inc = df_cor_inc.fillna(0).astype(int)

    pct = int(round(cfg.subsample_fraction * 100))
    out_tot_inc = os.path.join(cfg.output_dir, f"mmlu_subject_total_questions_by_model_incomplete_{pct}pct.csv")
    out_cor_inc = os.path.join(cfg.output_dir, f"mmlu_subject_num_correct_by_model_incomplete_{pct}pct.csv")

    df_tot_inc.to_csv(out_tot_inc, index_label="model")
    df_cor_inc.to_csv(out_cor_inc, index_label="model")

    print("saved incomplete")
    print("step1 done")

    return SubjectAggregationOutputs(
        full_total_csv=out_tot,
        full_correct_csv=out_cor,
        incomplete_total_csv=out_tot_inc,
        incomplete_correct_csv=out_cor_inc,
    )
