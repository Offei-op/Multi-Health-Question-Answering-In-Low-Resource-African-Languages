from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
BASE_DIR = ROOT / "reports" / "existing_source_selector"
CANDIDATES = BASE_DIR / "candidate_level_oof_scores.csv"
REPORT_DIR = ROOT / "reports" / "existing_source_selector_target_encoded"


def add_target_encoding(train_df, valid_df, smooth=25.0):
    global_mean = float(train_df["target_r1"].mean())
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    def encode(keys, name):
        stats = train_df.groupby(keys)["target_r1"].agg(["mean", "count"]).reset_index()
        stats[f"{name}_te"] = (stats["mean"] * stats["count"] + global_mean * smooth) / (stats["count"] + smooth)
        cols = keys + [f"{name}_te"]
        return valid_df.merge(stats[cols], on=keys, how="left")[f"{name}_te"].fillna(global_mean).to_numpy()

    valid_df["source_te"] = encode(["source"], "source")
    valid_df["subset_te"] = encode(["subset"], "subset")
    valid_df["source_subset_te"] = encode(["source", "subset"], "source_subset")
    return valid_df


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
            random_state=42,
        )
    elif kind == "extra_trees":
        reg = ExtraTreesRegressor(
            n_estimators=260,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("reg", reg)])


def score_choices(cand, col, label):
    idx = cand.groupby("ID")[col].idxmax()
    chosen = cand.loc[idx].copy()
    return {
        "label": label,
        "rouge1": float(chosen["target_r1"].mean()),
        "choice_counts": chosen["source"].value_counts().to_dict(),
        "per_subset": chosen.groupby("subset")["target_r1"].mean().round(6).to_dict(),
    }, chosen


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cand0 = pd.read_csv(CANDIDATES)
    exclude = {
        "ID",
        "prediction",
        "target_r1",
        "hgb_pred_r1",
        "extra_trees_pred_r1",
        "ridge_pred_r1",
        "ensemble_avg_pred_r1",
    }
    cat_cols = ["subset", "source"]
    base_num_cols = [c for c in cand0.columns if c not in exclude and c not in cat_cols]
    groups = cand0["ID"].to_numpy()
    y = cand0["target_r1"].to_numpy()
    gkf = GroupKFold(n_splits=5)

    results = []
    pred_cols = []
    choice_files = {}
    for kind in ["hgb", "extra_trees"]:
        oof = np.zeros(len(cand0), dtype=np.float32)
        for tr, va in gkf.split(cand0, y, groups):
            train_df = cand0.iloc[tr].copy()
            valid_df = add_target_encoding(train_df, cand0.iloc[va].copy())
            num_cols = base_num_cols + ["source_te", "subset_te", "source_subset_te"]
            model = make_model(kind, cat_cols, num_cols)
            train_te = add_target_encoding(train_df, train_df.copy())
            model.fit(train_te[cat_cols + num_cols], train_te["target_r1"].to_numpy())
            oof[va] = model.predict(valid_df[cat_cols + num_cols])
        col = f"{kind}_te_pred_r1"
        cand0[col] = oof
        pred_cols.append(col)
        result, chosen = score_choices(cand0, col, kind)
        results.append(result)
        choice_files[kind] = chosen

    cand0["ensemble_te_avg_pred_r1"] = cand0[pred_cols].mean(axis=1)
    result, chosen = score_choices(cand0, "ensemble_te_avg_pred_r1", "ensemble_te_avg")
    results.append(result)
    choice_files["ensemble_te_avg"] = chosen

    best_single = 0.5904174317680138
    exp2 = 0.5892166283468145
    leaderboard = pd.DataFrame(
        [
            {
                "label": r["label"],
                "rouge1": r["rouge1"],
                "gain_vs_best_single": r["rouge1"] - best_single,
                "gain_vs_exp2_rerank": r["rouge1"] - exp2,
                "top_choice": max(r["choice_counts"].items(), key=lambda x: x[1])[0],
            }
            for r in results
        ]
    ).sort_values("rouge1", ascending=False)

    best_label = str(leaderboard.iloc[0]["label"])
    cand0.to_csv(REPORT_DIR / "candidate_level_oof_scores.csv", index=False)
    leaderboard.to_csv(REPORT_DIR / "selector_leaderboard.csv", index=False)
    choice_files[best_label].to_csv(REPORT_DIR / "best_oof_choices.csv", index=False)
    summary = {
        "experiment": "existing_source_selector_with_fold_safe_target_encoding",
        "results": results,
        "best": leaderboard.iloc[0].to_dict(),
        "notes": [
            "Target encodings are computed inside each GroupKFold split from training IDs only.",
            "This tests whether source/subset priors improve local no-Modal source selection.",
        ],
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
