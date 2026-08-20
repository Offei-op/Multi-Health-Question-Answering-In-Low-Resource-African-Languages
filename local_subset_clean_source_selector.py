from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "local_deployable_source_selector.py"
if not BASE_SCRIPT.exists():
    BASE_SCRIPT = Path(r"C:\Users\Papa Offei\Documents\lalang\local_deployable_source_selector.py")

spec = importlib.util.spec_from_file_location("base_selector", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

ROOT = base.ROOT
REPORT_DIR = ROOT / "reports" / "subset_clean_source_selector"
CLEAN_SOURCES = {"exp2_top1", "exp2_rerank", "cluster_fast", "cluster_noleak"}


def make_feature_table():
    val = pd.read_csv(base.VAL)
    val["ID"] = val["ID"].astype(str)
    val["subset"] = val["subset"].astype(str)
    val["output"] = val["output"].fillna("").astype(str)
    original = base.SOURCES
    base.SOURCES = [s for s in original if s[0] in CLEAN_SOURCES]
    wide, loaded, missing = base.load_sources(val)
    cand = base.make_candidate_table(wide, loaded)
    return cand, wide, loaded, missing


def fit_predict_subset(grp, kind):
    cat_cols = ["source"]
    exclude = {"ID", "prediction", "target_r1", "subset"}
    num_cols = [c for c in grp.columns if c not in exclude and c not in cat_cols]
    for c in num_cols:
        grp[c] = pd.to_numeric(grp[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = pd.get_dummies(grp[cat_cols + num_cols], columns=cat_cols)
    y = grp["target_r1"].to_numpy()
    ids = grp["ID"].drop_duplicates().to_numpy()
    n_splits = min(5, len(ids))
    pred = np.zeros(len(grp), dtype=np.float32)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=37)
    for tr_i, va_i in kf.split(ids):
        tr_ids = set(ids[tr_i])
        va_ids = set(ids[va_i])
        tr = grp["ID"].isin(tr_ids).to_numpy()
        va = grp["ID"].isin(va_ids).to_numpy()
        if kind == "hgb":
            model = HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.045,
                max_leaf_nodes=15,
                l2_regularization=0.08,
                random_state=37,
            )
        elif kind == "extra_trees":
            model = ExtraTreesRegressor(
                n_estimators=240,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=37,
                n_jobs=-1,
            )
        else:
            raise ValueError(kind)
        model.fit(x.loc[tr], y[tr])
        pred[va] = model.predict(x.loc[va])
    return pred


def choose(cand, col):
    idx = cand.groupby("ID")[col].idxmax()
    return cand.loc[idx].copy()


def run():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cand, wide, loaded, missing = make_feature_table()
    out_parts = []
    subset_results = []
    for subset, grp in cand.groupby("subset", sort=False):
        grp = grp.copy()
        grp["subset_hgb_pred"] = fit_predict_subset(grp, "hgb")
        grp["subset_extra_trees_pred"] = fit_predict_subset(grp, "extra_trees")
        grp["subset_ensemble_pred"] = grp[["subset_hgb_pred", "subset_extra_trees_pred"]].mean(axis=1)
        out_parts.append(grp)
        for col in ["subset_hgb_pred", "subset_extra_trees_pred", "subset_ensemble_pred"]:
            chosen = choose(grp, col)
            subset_results.append(
                {
                    "subset": subset,
                    "model": col,
                    "score": float(chosen["target_r1"].mean()),
                    "rows": int(chosen.shape[0]),
                    "choice_counts": chosen["source"].value_counts().to_dict(),
                }
            )
    out = pd.concat(out_parts, ignore_index=True)
    results = []
    choices = {}
    for col in ["subset_hgb_pred", "subset_extra_trees_pred", "subset_ensemble_pred"]:
        chosen = choose(out, col)
        label = col.replace("_pred", "")
        choices[label] = chosen
        results.append(
            {
                "model": label,
                "score": float(chosen["target_r1"].mean()),
                "choice_counts": chosen["source"].value_counts().to_dict(),
                "per_subset": chosen.groupby("subset")["target_r1"].mean().round(6).to_dict(),
            }
        )
    best = max(results, key=lambda x: x["score"])
    choices[best["model"]].to_csv(REPORT_DIR / "best_oof_choices.csv", index=False)
    out.to_csv(REPORT_DIR / "candidate_level_oof_scores.csv", index=False)
    pd.DataFrame(results).sort_values("score", ascending=False).to_csv(REPORT_DIR / "leaderboard.csv", index=False)
    pd.DataFrame(subset_results).to_csv(REPORT_DIR / "subset_model_scores.csv", index=False)
    summary = {
        "loaded_sources": loaded,
        "missing_sources": missing,
        "best": best,
        "results": results,
        "baseline_exp2_rerank": 0.5892166283468145,
        "clean_global_selector": 0.6041607089218853,
        "clean_oracle": 0.6479322858935932,
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
