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
RICH_SCRIPT = HERE / "local_rich_clean_meta_selector.py"
if not RICH_SCRIPT.exists():
    RICH_SCRIPT = Path(r"C:\Users\Papa Offei\Documents\lalang\local_rich_clean_meta_selector.py")

spec = importlib.util.spec_from_file_location("rich", RICH_SCRIPT)
rich = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rich)

ROOT = rich.ROOT
REPORT_DIR = ROOT / "reports" / "rich_clean_meta_with_regressor"
REG_CHOICES = ROOT / "reports/local_candidate_regressor_submission/val_oof_choices.csv"
RANDOM_STATE = 83


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
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=RANDOM_STATE,
        )
    elif kind == "extra_trees":
        reg = ExtraTreesRegressor(
            n_estimators=380,
            min_samples_leaf=7,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif kind == "random_forest":
        reg = RandomForestRegressor(
            n_estimators=240,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("reg", reg)])


def build_candidates():
    cand, wide = rich.build_candidates()
    reg = pd.read_csv(REG_CHOICES)
    # Keep the same schema as the rich candidates. Text is unavailable in this val artifact, but test-time text exists.
    template_cols = cand.columns
    rows = []
    base = wide[["ID", "subset", "hybrid_source"]].merge(reg, on=["ID", "subset"], how="left")
    for _, r in base.iterrows():
        row = {c: np.nan for c in template_cols}
        row["ID"] = r["ID"]
        row["subset"] = r["subset"]
        row["hybrid_source"] = r["hybrid_source"]
        row["source"] = "local_candidate_regressor"
        row["prediction"] = ""
        row["target_r1"] = r["candidate_r1"]
        row["pred_len"] = 0
        row["char_len"] = 0
        row["source_is_hybrid"] = 0
        row["source_nonempty"] = 1
        row["reg_candidate_rank"] = r["candidate_rank"]
        row["reg_rerank_rank"] = r["rerank_rank"]
        row["reg_pred_hgb"] = r["pred_hgb"]
        rows.append(row)
    reg_cand = pd.DataFrame(rows)
    out = pd.concat([cand, reg_cand], ignore_index=True, sort=False)
    return out, wide


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
                "gain_vs_exp2": float(chosen["target_r1"].mean() - wide["exp2_rerank_r1"].mean()),
                "gain_vs_plain_rich": float(chosen["target_r1"].mean() - 0.6227089712249555),
                "choice_counts": chosen["source"].value_counts().to_dict(),
                "per_subset": chosen.groupby("subset")["target_r1"].mean().round(6).to_dict(),
            }
        )

    best = max(results, key=lambda x: x["score"])
    choices[best["model"]].to_csv(REPORT_DIR / "best_oof_choices.csv", index=False)
    cand.to_csv(REPORT_DIR / "candidate_level_oof_scores.csv", index=False)
    pd.DataFrame(results).sort_values("score", ascending=False).to_csv(REPORT_DIR / "leaderboard.csv", index=False)
    summary = {
        "baseline_exp2": float(wide["exp2_rerank_r1"].mean()),
        "plain_rich_best": 0.6227089712249555,
        "oracle_with_regressor": float(choose(cand, "target_r1")["target_r1"].mean()),
        "regressor_source_score": float(pd.read_csv(REG_CHOICES)["candidate_r1"].mean()),
        "best": best,
        "results": results,
        "notes": [
            "The local candidate regressor validation source has target scores and ranks but no answer text in its val choice file.",
            "It is deployable on test because test_candidate_regressor_choices.csv contains candidate_answer.",
        ],
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(results).sort_values("score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    run()
