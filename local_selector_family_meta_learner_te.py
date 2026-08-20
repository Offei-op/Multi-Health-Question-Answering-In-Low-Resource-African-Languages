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
BASE_SCRIPT = HERE / "local_selector_family_meta_learner.py"
if not BASE_SCRIPT.exists():
    BASE_SCRIPT = Path(r"C:\Users\Papa Offei\Documents\lalang\local_selector_family_meta_learner.py")

spec = importlib.util.spec_from_file_location("fam", BASE_SCRIPT)
fam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fam)

ROOT = fam.ROOT
REPORT_DIR = ROOT / "reports" / "selector_family_meta_learner_target_encoded"
RANDOM_STATE = 101


def add_te(train_df, valid_df, smooth=20.0):
    valid_df = valid_df.copy()
    global_mean = float(train_df["target_r1"].mean())
    for keys, name in [
        (["family"], "family"),
        (["subset"], "subset"),
        (["family", "subset"], "family_subset"),
        (["hybrid_source", "family"], "hybrid_family"),
    ]:
        stats = train_df.groupby(keys)["target_r1"].agg(["mean", "count"]).reset_index()
        stats[f"{name}_te"] = (stats["mean"] * stats["count"] + global_mean * smooth) / (stats["count"] + smooth)
        valid_df = valid_df.merge(stats[keys + [f"{name}_te"]], on=keys, how="left")
        valid_df[f"{name}_te"] = valid_df[f"{name}_te"].fillna(global_mean)
    return valid_df


def make_model(kind, cat_cols, num_cols):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ]
    )
    if kind == "hgb":
        reg = HistGradientBoostingRegressor(max_iter=220, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.08, random_state=RANDOM_STATE)
    elif kind == "extra_trees":
        reg = ExtraTreesRegressor(n_estimators=360, min_samples_leaf=8, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)
    elif kind == "random_forest":
        reg = RandomForestRegressor(n_estimators=240, min_samples_leaf=8, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("reg", reg)])


def choose(cand, col):
    return cand.loc[cand.groupby("ID")[col].idxmax()].copy()


def run():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cand0 = fam.load_family_candidates()
    cat_cols = ["subset", "family", "hybrid_source"]
    banned = {"ID", "prediction", "target_r1"} | {n for n, *_ in fam.FAMILY_FILES}
    base_num_cols = [c for c in cand0.columns if c not in banned and c not in cat_cols]
    for c in base_num_cols:
        cand0[c] = pd.to_numeric(cand0[c], errors="coerce")
    ids = cand0[["ID", "subset"]].drop_duplicates().reset_index(drop=True)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y = cand0["target_r1"].to_numpy()
    pred_cols = []
    for kind in ["hgb", "extra_trees", "random_forest"]:
        oof = np.zeros(len(cand0), dtype=np.float32)
        for fold, (tr_i, va_i) in enumerate(folds.split(ids["ID"], ids["subset"]), start=1):
            tr_ids = set(ids.iloc[tr_i]["ID"])
            va_ids = set(ids.iloc[va_i]["ID"])
            tr = cand0["ID"].isin(tr_ids).to_numpy()
            va = cand0["ID"].isin(va_ids).to_numpy()
            train_te = add_te(cand0.loc[tr].copy(), cand0.loc[tr].copy())
            valid_te = add_te(cand0.loc[tr].copy(), cand0.loc[va].copy())
            num_cols = base_num_cols + ["family_te", "subset_te", "family_subset_te", "hybrid_family_te"]
            model = make_model(kind, cat_cols, num_cols)
            model.fit(train_te[cat_cols + num_cols], train_te["target_r1"].to_numpy())
            oof[va] = model.predict(valid_te[cat_cols + num_cols])
            print(f"{kind} fold {fold} done", flush=True)
        col = f"{kind}_te_pred_r1"
        cand0[col] = oof
        pred_cols.append(col)
    cand0["ensemble_te_pred_r1"] = cand0[pred_cols].mean(axis=1)
    cand0["weighted_ensemble_te_pred_r1"] = 0.50 * cand0[pred_cols[0]] + 0.30 * cand0[pred_cols[1]] + 0.20 * cand0[pred_cols[2]]

    results = []
    choices = {}
    for col in pred_cols + ["ensemble_te_pred_r1", "weighted_ensemble_te_pred_r1"]:
        chosen = choose(cand0, col)
        label = col.replace("_pred_r1", "")
        choices[label] = chosen
        results.append(
            {
                "model": label,
                "score": float(chosen["target_r1"].mean()),
                "gain_vs_exp2": float(chosen["target_r1"].mean() - 0.5892166283468145),
                "gain_vs_family_meta": float(chosen["target_r1"].mean() - 0.6258109015467073),
                "choice_counts": chosen["family"].value_counts().to_dict(),
                "per_subset": chosen.groupby("subset")["target_r1"].mean().round(6).to_dict(),
            }
        )
    best = max(results, key=lambda x: x["score"])
    choices[best["model"]].to_csv(REPORT_DIR / "best_oof_choices.csv", index=False)
    cand0.to_csv(REPORT_DIR / "candidate_level_oof_scores.csv", index=False)
    pd.DataFrame(results).sort_values("score", ascending=False).to_csv(REPORT_DIR / "leaderboard.csv", index=False)
    summary = {
        "family_meta_best": 0.6258109015467073,
        "best": best,
        "results": results,
        "notes": ["Target encodings are fold-safe and computed from training IDs only."],
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(results).sort_values("score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    run()
