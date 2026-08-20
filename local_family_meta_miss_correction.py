from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
REPORT_DIR = ROOT / "reports" / "family_meta_miss_correction"
CAND = ROOT / "reports/selector_family_meta_learner/candidate_level_oof_scores.csv"
BEST = ROOT / "reports/selector_family_meta_learner/family_meta_subset_hybrid_choices.csv"
BASELINE_EXP2 = 0.5892166283468145
CURRENT_BEST = 0.6258761046358958
RANDOM_STATE = 113


def choose(df, col):
    return df.loc[df.groupby("ID")[col].idxmax()].copy()


def make_model(kind, cat_cols, num_cols):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ]
    )
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=RANDOM_STATE,
        )
    elif kind == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=360,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif kind == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=240,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("clf", clf)])


def run():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv(CAND)
    base = pd.read_csv(BEST)[["ID", "subset", "selector", "target_r1", "prediction"]].rename(
        columns={"selector": "base_selector", "target_r1": "base_r1", "prediction": "base_prediction"}
    )
    oracle = choose(cand, "target_r1")[["ID", "family", "target_r1", "prediction"]].rename(
        columns={"family": "oracle_family", "target_r1": "oracle_r1", "prediction": "oracle_prediction"}
    )
    rows = base.merge(oracle, on="ID", how="left")
    rows["oracle_gain_vs_base"] = rows["oracle_r1"] - rows["base_r1"]
    rows["base_is_oracle"] = (rows["oracle_gain_vs_base"].abs() < 1e-12).astype(int)
    rows["should_switch"] = (rows["oracle_gain_vs_base"] > 1e-12).astype(int)

    # Candidate to switch to: highest predicted family-meta score excluding the current base prediction family if possible.
    # This is deployable because it uses model scores already generated in OOF.
    score_col = "ensemble_pred_r1"
    best_pred = choose(cand, score_col)[["ID", "family", "target_r1", "prediction", score_col]].rename(
        columns={"family": "pred_family", "target_r1": "pred_family_r1", "prediction": "pred_prediction"}
    )
    rows = rows.merge(best_pred, on="ID", how="left")
    rows["pred_gain_vs_base"] = rows["pred_family_r1"] - rows["base_r1"]

    feat = rows.merge(
        cand[cand["family"].eq("meta_gate")].drop_duplicates("ID"),
        on=["ID", "subset"],
        how="left",
        suffixes=("", "_meta_gate_row"),
    )
    feat["oracle_gain_band"] = pd.cut(
        feat["oracle_gain_vs_base"],
        bins=[-1, 0, 0.01, 0.05, 0.15, 1.01],
        labels=["none", "tiny", "small", "medium", "large"],
    ).astype(str)

    cat_cols = ["subset", "base_selector", "pred_family", "hybrid_source"]
    banned = {
        "ID",
        "prediction",
        "base_prediction",
        "oracle_prediction",
        "pred_prediction",
        "base_r1",
        "oracle_r1",
        "oracle_gain_vs_base",
        "base_is_oracle",
        "should_switch",
        "pred_family_r1",
        "pred_gain_vs_base",
        "oracle_family",
        "target_r1",
        "family",
        "oracle_gain_band",
    }
    num_cols = [c for c in feat.columns if c not in banned and c not in cat_cols]
    for c in num_cols:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")

    ids = feat[["ID", "subset"]].drop_duplicates().reset_index(drop=True)
    y = feat["should_switch"].to_numpy()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pred_cols = []
    for kind in ["hgb", "extra_trees", "random_forest"]:
        proba = np.zeros(len(feat), dtype=np.float32)
        for fold, (tr, va) in enumerate(folds.split(ids["ID"], ids["subset"]), start=1):
            model = make_model(kind, cat_cols, num_cols)
            model.fit(feat.iloc[tr][cat_cols + num_cols], y[tr])
            proba[va] = model.predict_proba(feat.iloc[va][cat_cols + num_cols])[:, 1]
            print(f"{kind} fold {fold} done", flush=True)
        col = f"{kind}_switch_proba"
        feat[col] = proba
        pred_cols.append(col)
    feat["ensemble_switch_proba"] = feat[pred_cols].mean(axis=1)

    results = []
    for col in pred_cols + ["ensemble_switch_proba"]:
        try:
            auc = float(roc_auc_score(y, feat[col]))
        except Exception:
            auc = None
        for mode in ["switch_to_pred_family", "switch_to_oracle_diagnostic"]:
            for q in np.linspace(0.50, 0.98, 25):
                threshold = float(np.quantile(feat[col], q))
                switch = feat[col] >= threshold
                if mode == "switch_to_pred_family":
                    final_r1 = np.where(switch, feat["pred_family_r1"], feat["base_r1"])
                else:
                    # Upper-bound diagnostic for whether switch detection itself is the bottleneck.
                    final_r1 = np.where(switch, feat["oracle_r1"], feat["base_r1"])
                results.append(
                    {
                        "model": col,
                        "mode": mode,
                        "threshold_quantile": float(q),
                        "threshold": threshold,
                        "score": float(np.mean(final_r1)),
                        "gain_vs_current_best": float(np.mean(final_r1) - CURRENT_BEST),
                        "gain_vs_exp2": float(np.mean(final_r1) - BASELINE_EXP2),
                        "switch_rate": float(switch.mean()),
                        "switch_count": int(switch.sum()),
                        "auc": auc,
                        "per_subset": pd.DataFrame({"subset": feat["subset"], "r1": final_r1}).groupby("subset")["r1"].mean().round(6).to_dict(),
                    }
                )

    lb = pd.DataFrame(results).sort_values("score", ascending=False)
    miss_summary = rows.groupby("subset").agg(
        rows=("ID", "count"),
        base=("base_r1", "mean"),
        oracle=("oracle_r1", "mean"),
        oracle_gain=("oracle_gain_vs_base", "mean"),
        switch_rate=("should_switch", "mean"),
    ).reset_index().sort_values("oracle_gain", ascending=False)

    feat.to_csv(REPORT_DIR / "miss_correction_rows.csv", index=False)
    miss_summary.to_csv(REPORT_DIR / "miss_summary_by_subset.csv", index=False)
    lb.to_csv(REPORT_DIR / "leaderboard.csv", index=False)
    summary = {
        "current_best": CURRENT_BEST,
        "family_oracle": float(rows["oracle_r1"].mean()),
        "oracle_gap_remaining": float(rows["oracle_r1"].mean() - CURRENT_BEST),
        "miss_rate": float(rows["should_switch"].mean()),
        "miss_summary_by_subset": miss_summary.to_dict(orient="records"),
        "best": lb.iloc[0].to_dict(),
        "notes": [
            "switch_to_pred_family is deployable-style and only switches to the model-predicted best family.",
            "switch_to_oracle_diagnostic is not deployable; it tests whether switch detection is useful if target family were known.",
        ],
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(lb.head(20).to_string(index=False))


if __name__ == "__main__":
    run()
