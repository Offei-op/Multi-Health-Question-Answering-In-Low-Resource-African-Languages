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


HERE = Path(__file__).resolve().parent
GATE_SCRIPT = HERE / "local_cluster_fast_gate.py"
if not GATE_SCRIPT.exists():
    GATE_SCRIPT = Path(r"C:\Users\Papa Offei\Documents\lalang\local_cluster_fast_gate.py")

spec = importlib.util.spec_from_file_location("gate", GATE_SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

ROOT = gate.ROOT
REPORT_DIR = ROOT / "reports" / "rich_clean_meta_selector"
RANDOM_STATE = 53
SOURCES = ["top1", "exp2_rerank", "fast_candidate_answer", "noleak_candidate_answer"]


def source_label(source):
    return {
        "top1": "exp2_top1",
        "exp2_rerank": "exp2_rerank",
        "fast_candidate_answer": "cluster_fast",
        "noleak_candidate_answer": "cluster_noleak",
    }[source]


def build_candidates():
    wide = gate.load_table()
    shared_cols = []
    for c in wide.columns:
        if c.startswith("fast_") or c.startswith("noleak_") or c.startswith("jac_") or c.startswith("len_diff_"):
            shared_cols.append(c)
    shared_cols += [
        "ID",
        "subset",
        "hybrid_source",
    ]
    shared_cols = list(dict.fromkeys(c for c in shared_cols if not c.endswith("_r1") and "gain" not in c and "beats" not in c))
    rows = []
    for source in SOURCES:
        src = wide[shared_cols].copy()
        src["source"] = source_label(source)
        src["prediction"] = wide[source].fillna("")
        src["target_r1"] = wide[f"{source}_r1"]
        src["pred_len"] = src["prediction"].map(lambda x: len(str(x).split()))
        src["char_len"] = src["prediction"].map(lambda x: len(str(x)))
        src["source_is_hybrid"] = (src["source"] == wide["hybrid_source"].astype(str)).astype(int)
        for other in SOURCES:
            if other == source:
                continue
            src[f"jac_vs_{source_label(other)}"] = [gate.jaccard(a, b) for a, b in zip(src["prediction"], wide[other])]
            src[f"len_diff_vs_{source_label(other)}"] = src["pred_len"] - wide[other].map(lambda x: len(str(x).split()))
        rows.append(src)
    cand = pd.concat(rows, ignore_index=True)
    cand["source_nonempty"] = (cand["prediction"].astype(str) != "").astype(int)
    return cand, wide


def make_model(kind, cat_cols, num_cols):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ]
    )
    if kind == "hgb":
        reg = HistGradientBoostingRegressor(
            max_iter=280,
            learning_rate=0.035,
            max_leaf_nodes=19,
            l2_regularization=0.07,
            random_state=RANDOM_STATE,
        )
    elif kind == "extra_trees":
        reg = ExtraTreesRegressor(
            n_estimators=420,
            min_samples_leaf=6,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif kind == "random_forest":
        reg = RandomForestRegressor(
            n_estimators=260,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("reg", reg)])


def choose(cand, score_col):
    idx = cand.groupby("ID")[score_col].idxmax()
    return cand.loc[idx].copy()


def run():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cand, wide = build_candidates()
    cat_cols = ["subset", "source", "hybrid_source"]
    banned = {"ID", "prediction", "target_r1"}
    num_cols = [c for c in cand.columns if c not in banned and c not in cat_cols]
    for c in num_cols:
        cand[c] = pd.to_numeric(cand[c], errors="coerce")

    ids = cand[["ID", "subset"]].drop_duplicates().reset_index(drop=True)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y = cand["target_r1"].to_numpy()
    pred_cols = []
    for kind in ["hgb", "extra_trees", "random_forest"]:
        pred = np.zeros(len(cand), dtype=np.float32)
        for fold, (tr_i, va_i) in enumerate(folds.split(ids["ID"], ids["subset"]), start=1):
            tr_ids = set(ids.iloc[tr_i]["ID"])
            va_ids = set(ids.iloc[va_i]["ID"])
            tr = cand["ID"].isin(tr_ids).to_numpy()
            va = cand["ID"].isin(va_ids).to_numpy()
            model = make_model(kind, cat_cols, num_cols)
            model.fit(cand.loc[tr, cat_cols + num_cols], y[tr])
            pred[va] = model.predict(cand.loc[va, cat_cols + num_cols])
            print(f"{kind} fold {fold} done", flush=True)
        col = f"{kind}_pred_r1"
        cand[col] = pred
        pred_cols.append(col)

    cand["ensemble_pred_r1"] = cand[pred_cols].mean(axis=1)
    # This intentionally favors HGB slightly because the fast gate showed it was better calibrated.
    cand["weighted_ensemble_pred_r1"] = 0.50 * cand["hgb_pred_r1"] + 0.30 * cand["extra_trees_pred_r1"] + 0.20 * cand["random_forest_pred_r1"]

    results = []
    choice_map = {}
    for col in pred_cols + ["ensemble_pred_r1", "weighted_ensemble_pred_r1"]:
        chosen = choose(cand, col)
        label = col.replace("_pred_r1", "")
        choice_map[label] = chosen
        results.append(
            {
                "model": label,
                "score": float(chosen["target_r1"].mean()),
                "gain_vs_exp2": float(chosen["target_r1"].mean() - wide["exp2_rerank_r1"].mean()),
                "gain_vs_hybrid": float(chosen["target_r1"].mean() - wide["hybrid_prediction_r1"].mean()),
                "choice_counts": chosen["source"].value_counts().to_dict(),
                "per_subset": chosen.groupby("subset")["target_r1"].mean().round(6).to_dict(),
            }
        )

    best = max(results, key=lambda x: x["score"])
    best_label = best["model"]
    oracle = choose(cand, "target_r1")

    cand.to_csv(REPORT_DIR / "candidate_level_oof_scores.csv", index=False)
    choice_map[best_label].to_csv(REPORT_DIR / "best_oof_choices.csv", index=False)
    pd.DataFrame(results).sort_values("score", ascending=False).to_csv(REPORT_DIR / "leaderboard.csv", index=False)
    summary = {
        "baseline_exp2": float(wide["exp2_rerank_r1"].mean()),
        "baseline_hybrid": float(wide["hybrid_prediction_r1"].mean()),
        "clean_oracle": float(oracle["target_r1"].mean()),
        "best": best,
        "results": results,
        "notes": [
            "All features are drawn from clean deployable sources and OOF validation predictions.",
            "No validation reference-length or target columns are used as features.",
        ],
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(results).sort_values("score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    run()
