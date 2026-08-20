from collections import Counter
from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
REPORT_DIR = ROOT / "reports" / "extended_validation_source_selector"
VAL = ROOT / "Val.csv"
CURRENT = ROOT / "reports/selector_family_meta_learner/family_meta_subset_hybrid_choices.csv"
BASELINE_EXP2 = 0.5892166283468145
CURRENT_BEST = 0.6258761046358958
RANDOM_STATE = 139

SOURCES = [
    ("current_family_meta", CURRENT, "prediction"),
    ("exp3_top100", ROOT / "modal_outputs/exp3_top100_crossencoder_rerank/val_predictions.csv", "rerank"),
    ("exp4_pairwise", ROOT / "modal_outputs/exp4_pairwise_hardneg_rerank/val_predictions.csv", "rerank"),
    ("exp5_encoder_exp2", ROOT / "modal_outputs/exp5_encoder_exp2_rerank_eval/val_predictions.csv", "rerank"),
    ("exp9_jina", ROOT / "modal_outputs/exp9_jina_multilingual_reranker/val_predictions.csv", "rerank"),
    ("exp14_qonly", ROOT / "modal_outputs/exp14_qonly_gemma_reranker_recovered", "qonly_bgem3"),
    ("exp8_ghana", ROOT / "modal_outputs/exp8_ghana_grouped_encoder_reranker/ghana_val_predictions.csv", "ghana_rerank"),
    ("exp10_lug_global", ROOT / "modal_outputs/exp10_lug_uga_reranker/lug_val_predictions.csv", "global_rerank"),
    ("exp10_lug_specialized", ROOT / "modal_outputs/exp10_lug_uga_reranker/lug_val_predictions.csv", "lug_rerank"),
    ("exp12_lug_merged", ROOT / "modal_outputs/exp12_lug_e5_merge_rerank/lug_val_predictions.csv", "merged_rerank"),
    ("exp13_lug_selector", ROOT / "reports/exp13_lug_merged_selector/oof_best_selector_choices.csv", "selector_answer"),
    ("local_mt0_generation", ROOT / "local_eval_outputs/val_predictions.csv", "prediction"),
    ("exp1_qa_base", ROOT / "modal_outputs/exp1_query_to_qa_doc/baseline_query_to_qa_doc_val_predictions.csv", "prediction"),
    ("exp1_qa_ft", ROOT / "modal_outputs/exp1_query_to_qa_doc/finetuned_query_to_qa_doc_val_predictions.csv", "prediction"),
    ("exp11_bge_base_top1", ROOT / "modal_outputs/exp11_base_encoder_benchmark/val_base_encoder_predictions.csv", "top1_answer", {"model": "bge_m3_base"}),
    ("exp11_e5_base_top1", ROOT / "modal_outputs/exp11_base_encoder_benchmark/val_base_encoder_predictions.csv", "top1_answer", {"model": "multilingual_e5_base"}),
    ("exp11_e5_large_top1", ROOT / "modal_outputs/exp11_base_encoder_benchmark/val_base_encoder_predictions.csv", "top1_answer", {"model": "multilingual_e5_large"}),
]


def rouge1(pred, ref):
    pred_toks = str(pred).split()
    ref_toks = str(ref).split()
    if not pred_toks or not ref_toks:
        return 0.0
    overlap = sum((Counter(pred_toks) & Counter(ref_toks)).values())
    return float(2 * overlap / (len(pred_toks) + len(ref_toks))) if overlap else 0.0


def jaccard(a, b):
    a = set(str(a).split())
    b = set(str(b).split())
    return len(a & b) / len(a | b) if a and b else 0.0


def load_wide():
    val = pd.read_csv(VAL)[["ID", "subset", "output"]]
    wide = val.copy()
    loaded = []
    for source in SOURCES:
        name, path, col = source[:3]
        filters = source[3] if len(source) > 3 else None
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if filters:
            for fcol, fval in filters.items():
                if fcol not in df.columns:
                    df = df.iloc[0:0]
                    break
                df = df[df[fcol].eq(fval)]
        if col not in df.columns:
            continue
        wide = wide.merge(df[["ID", col]].rename(columns={col: name}), on="ID", how="left")
        wide[name] = wide[name].fillna("")
        loaded.append(name)
    return wide, loaded


def make_model(kind, cat_cols, num_cols):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ]
    )
    if kind == "hgb":
        reg = HistGradientBoostingRegressor(
            max_iter=260,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=RANDOM_STATE,
        )
    elif kind == "extra_trees":
        reg = ExtraTreesRegressor(
            n_estimators=420,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif kind == "random_forest":
        reg = RandomForestRegressor(
            n_estimators=280,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("reg", reg)])


def build_candidates(wide, loaded):
    rows = []
    for source in loaded:
        pred = wide[source].fillna("").astype(str)
        tmp = wide[["ID", "subset", "output"] + loaded].copy()
        tmp["source"] = source
        tmp["prediction"] = pred
        tmp["target_r1"] = [rouge1(p, r) for p, r in zip(pred, wide["output"])]
        tmp["pred_len"] = pred.map(lambda x: len(str(x).split()))
        tmp["char_len"] = pred.map(lambda x: len(str(x)))
        tmp["source_nonempty"] = (pred != "").astype(int)
        sims = []
        for _, row in tmp.iterrows():
            vals = [jaccard(row["prediction"], row[o]) for o in loaded if o != source]
            sims.append(
                {
                    "agreement_max": max(vals) if vals else 0.0,
                    "agreement_mean": float(np.mean(vals)) if vals else 0.0,
                    "agreement_count_ge_080": int(sum(v >= 0.80 for v in vals)),
                    "agreement_count_ge_095": int(sum(v >= 0.95 for v in vals)),
                    "unique_same_text": int(sum(str(row["prediction"]) == str(row[o]) for o in loaded)),
                }
            )
        sim_df = pd.DataFrame(sims)
        for c in sim_df.columns:
            tmp[c] = sim_df[c].to_numpy()
        # Compact pairwise features versus current best.
        if "current_family_meta" in loaded and source != "current_family_meta":
            tmp["jac_vs_current"] = [jaccard(a, b) for a, b in zip(tmp["prediction"], tmp["current_family_meta"])]
            tmp["len_diff_vs_current"] = tmp["pred_len"] - tmp["current_family_meta"].map(lambda x: len(str(x).split()))
        else:
            tmp["jac_vs_current"] = 1.0
            tmp["len_diff_vs_current"] = 0
        rows.append(tmp)
    cand = pd.concat(rows, ignore_index=True)
    return cand


def choose(cand, col):
    return cand.loc[cand.groupby("ID")[col].idxmax()].copy()


def run():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    wide, loaded = load_wide()
    cand = build_candidates(wide, loaded)
    cat_cols = ["subset", "source"]
    banned = {"ID", "output", "prediction", "target_r1"} | set(loaded)
    num_cols = [c for c in cand.columns if c not in banned and c not in cat_cols]
    for c in num_cols:
        cand[c] = pd.to_numeric(cand[c], errors="coerce")

    ids = cand[["ID", "subset"]].drop_duplicates().reset_index(drop=True)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y = cand["target_r1"].to_numpy()
    pred_cols = []
    for kind in ["hgb", "extra_trees", "random_forest"]:
        oof = np.zeros(len(cand), dtype=np.float32)
        for fold, (tr_i, va_i) in enumerate(folds.split(ids["ID"], ids["subset"]), start=1):
            tr_ids = set(ids.iloc[tr_i]["ID"])
            va_ids = set(ids.iloc[va_i]["ID"])
            tr = cand["ID"].isin(tr_ids).to_numpy()
            va = cand["ID"].isin(va_ids).to_numpy()
            model = make_model(kind, cat_cols, num_cols)
            model.fit(cand.loc[tr, cat_cols + num_cols], y[tr])
            oof[va] = model.predict(cand.loc[va, cat_cols + num_cols])
            print(f"{kind} fold {fold} done", flush=True)
        col = f"{kind}_pred_r1"
        cand[col] = oof
        pred_cols.append(col)
    cand["ensemble_pred_r1"] = cand[pred_cols].mean(axis=1)
    cand["weighted_ensemble_pred_r1"] = 0.50 * cand[pred_cols[0]] + 0.30 * cand[pred_cols[1]] + 0.20 * cand[pred_cols[2]]

    results = []
    choices = {}
    for col in pred_cols + ["ensemble_pred_r1", "weighted_ensemble_pred_r1"]:
        chosen = choose(cand, col)
        label = col.replace("_pred_r1", "")
        choices[label] = chosen
        results.append(
            {
                "model": label,
                "score": float(chosen["target_r1"].mean()),
                "gain_vs_exp2": float(chosen["target_r1"].mean() - BASELINE_EXP2),
                "gain_vs_current_best": float(chosen["target_r1"].mean() - CURRENT_BEST),
                "choice_counts": chosen["source"].value_counts().to_dict(),
                "per_subset": chosen.groupby("subset")["target_r1"].mean().round(6).to_dict(),
            }
        )
    best = max(results, key=lambda x: x["score"])
    best_label = best["model"]
    oracle = choose(cand, "target_r1")

    cand.to_csv(REPORT_DIR / "candidate_level_oof_scores.csv", index=False)
    choices[best_label].to_csv(REPORT_DIR / "best_oof_choices.csv", index=False)
    pd.DataFrame(results).sort_values("score", ascending=False).to_csv(REPORT_DIR / "leaderboard.csv", index=False)
    source_scores = []
    for source in loaded:
        source_scores.append({"source": source, "score": float(cand[cand["source"].eq(source)]["target_r1"].mean())})
    summary = {
        "loaded_sources": loaded,
        "source_scores": source_scores,
        "current_best": CURRENT_BEST,
        "oracle": float(oracle["target_r1"].mean()),
        "best": best,
        "results": results,
        "notes": [
            "This includes already-local validation artifacts that may not yet have mirrored test predictions.",
            "No Modal compute was used.",
        ],
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(results).sort_values("score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    run()
