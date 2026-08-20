from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


HERE = Path(__file__).resolve().parent
META_SCRIPT = HERE / "local_rich_clean_meta_selector.py"
if not META_SCRIPT.exists():
    META_SCRIPT = Path(r"C:\Users\Papa Offei\Documents\lalang\local_rich_clean_meta_selector.py")

spec = importlib.util.spec_from_file_location("meta", META_SCRIPT)
meta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meta)

ROOT = meta.ROOT
REPORT_DIR = ROOT / "reports" / "residual_clean_override_selector"
BASE_CHOICES = ROOT / "reports/rich_clean_meta_selector/meta_gate_subset_hybrid_choices.csv"
RANDOM_STATE = 67


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
            n_estimators=380,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("reg", reg)])


def run():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cand, wide = meta.build_candidates()
    base = pd.read_csv(BASE_CHOICES)[["ID", "prediction", "target_r1", "selector_family"]].rename(
        columns={"prediction": "base_prediction", "target_r1": "base_r1"}
    )
    cand = cand.merge(base, on="ID", how="left")
    cand["base_prediction"] = cand["base_prediction"].fillna("")
    cand["base_r1"] = cand["base_r1"].fillna(wide["hybrid_prediction_r1"].mean())
    cand["target_gain_vs_base"] = cand["target_r1"] - cand["base_r1"]
    cand["jac_candidate_vs_base"] = [meta.gate.jaccard(a, b) for a, b in zip(cand["prediction"], cand["base_prediction"])]
    cand["len_diff_candidate_vs_base"] = cand["pred_len"] - cand["base_prediction"].map(lambda x: len(str(x).split()))

    cat_cols = ["subset", "source", "hybrid_source", "selector_family"]
    banned = {"ID", "prediction", "target_r1", "base_prediction", "base_r1", "target_gain_vs_base"}
    num_cols = [c for c in cand.columns if c not in banned and c not in cat_cols]
    for c in num_cols:
        cand[c] = pd.to_numeric(cand[c], errors="coerce")

    ids = cand[["ID", "subset"]].drop_duplicates().reset_index(drop=True)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y = cand["target_gain_vs_base"].to_numpy()
    pred_cols = []
    for kind in ["hgb", "extra_trees"]:
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
        col = f"{kind}_pred_gain"
        cand[col] = oof
        pred_cols.append(col)
    cand["ensemble_pred_gain"] = cand[pred_cols].mean(axis=1)

    results = []
    choices = {}
    for col in pred_cols + ["ensemble_pred_gain"]:
        idx = cand.groupby("ID")[col].idxmax()
        best_candidate = cand.loc[idx].copy()
        for scope, threshold in [("zero", 0.0)]:
            use_override = best_candidate[col] > threshold
            score = np.where(use_override, best_candidate["target_r1"], best_candidate["base_r1"])
            label = f"{col}_{scope}"
            choices[label] = best_candidate.assign(use_override=use_override, final_r1=score)
            results.append(
                {
                    "model": col,
                    "scope": scope,
                    "threshold": threshold,
                    "score": float(np.mean(score)),
                    "gain_vs_exp2": float(np.mean(score) - wide["exp2_rerank_r1"].mean()),
                    "gain_vs_base": float(np.mean(score) - best_candidate["base_r1"].mean()),
                    "override_rate": float(use_override.mean()),
                    "override_counts": best_candidate.loc[use_override, "source"].value_counts().to_dict(),
                    "per_subset": pd.DataFrame({"subset": best_candidate["subset"], "r1": score}).groupby("subset")["r1"].mean().round(6).to_dict(),
                }
            )
        for scope in ["sweep"]:
            best = None
            for threshold in np.quantile(best_candidate[col].dropna(), np.linspace(0.05, 0.95, 37)):
                use_override = best_candidate[col] > threshold
                score = np.where(use_override, best_candidate["target_r1"], best_candidate["base_r1"])
                row = {
                    "model": col,
                    "scope": scope,
                    "threshold": float(threshold),
                    "score": float(np.mean(score)),
                    "gain_vs_exp2": float(np.mean(score) - wide["exp2_rerank_r1"].mean()),
                    "gain_vs_base": float(np.mean(score) - best_candidate["base_r1"].mean()),
                    "override_rate": float(use_override.mean()),
                    "override_counts": best_candidate.loc[use_override, "source"].value_counts().to_dict(),
                    "per_subset": pd.DataFrame({"subset": best_candidate["subset"], "r1": score}).groupby("subset")["r1"].mean().round(6).to_dict(),
                }
                if best is None or row["score"] > best["score"]:
                    best = row
            results.append(best)
            use_override = best_candidate[col] > best["threshold"]
            score = np.where(use_override, best_candidate["target_r1"], best_candidate["base_r1"])
            choices[f"{col}_sweep"] = best_candidate.assign(use_override=use_override, final_r1=score)

    leaderboard = pd.DataFrame(results).sort_values("score", ascending=False)
    best = leaderboard.iloc[0].to_dict()
    best_key = f"{best['model']}_{best['scope']}"
    cand.to_csv(REPORT_DIR / "candidate_level_oof_scores.csv", index=False)
    choices[best_key].to_csv(REPORT_DIR / "best_override_choices.csv", index=False)
    leaderboard.to_csv(REPORT_DIR / "leaderboard.csv", index=False)
    summary = {
        "baseline_exp2": float(wide["exp2_rerank_r1"].mean()),
        "base_score": float(pd.read_csv(BASE_CHOICES)["target_r1"].mean()),
        "clean_oracle": float(cand.loc[cand.groupby("ID")["target_r1"].idxmax(), "target_r1"].mean()),
        "best": best,
        "notes": [
            "Zero scope is the cleaner calibrated read.",
            "Sweep scope is diagnostic and may be optimistic because threshold is selected on OOF validation predictions.",
        ],
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    run()
