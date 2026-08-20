from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


HERE = Path(__file__).resolve().parent
FAM_SCRIPT = HERE / "local_selector_family_meta_learner.py"
if not FAM_SCRIPT.exists():
    FAM_SCRIPT = Path(r"C:\Users\Papa Offei\Documents\lalang\local_selector_family_meta_learner.py")

spec = importlib.util.spec_from_file_location("fam", FAM_SCRIPT)
fam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fam)

ROOT = fam.ROOT
REPORT_DIR = ROOT / "reports" / "selector_family_classifier"
BASELINE_EXP2 = 0.5892166283468145
CURRENT_BEST = 0.625917  # best deployable-style tiny correction found so far, rounded from report.
RANDOM_STATE = 127


def make_model(kind, cat_cols, num_cols):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ]
    )
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(
            max_iter=240,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=RANDOM_STATE,
        )
    elif kind == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=6,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif kind == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=320,
            min_samples_leaf=7,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("clf", clf)])


def build_row_table():
    cand = fam.load_family_candidates()
    # Actual family scores, one row per ID.
    scores = cand.pivot_table(index=["ID", "subset"], columns="family", values="target_r1", aggfunc="first").reset_index()
    families = [name for name, *_ in fam.FAMILY_FILES]
    scores["oracle_family"] = scores[families].idxmax(axis=1)
    scores["oracle_r1"] = scores[families].max(axis=1)
    scores = scores.rename(columns={name: f"score_{name}" for name in families})

    # Use one representative row per ID for shared features, then add model-prediction features per family.
    rep = cand[cand["family"].eq("meta_gate")].drop_duplicates("ID").copy()
    keep = ["ID", "subset", "hybrid_source"]
    for c in rep.columns:
        if (
            c.startswith("fast_")
            or c.startswith("noleak_")
            or c.startswith("jac_")
            or c.startswith("len_diff_")
            or c in {"agreement_mean", "agreement_max", "pred_len", "char_len"}
        ) and not c.endswith("_r1") and "gain" not in c and "beats" not in c:
            keep.append(c)
    keep = list(dict.fromkeys([c for c in keep if c in rep.columns]))
    row = rep[keep].merge(scores[["ID", "oracle_family", "oracle_r1"] + [f"score_{name}" for name in families]], on="ID", how="left")

    # Add family-meta model scores and agreement stats per family. These are deployable predictions from prior OOF models.
    pred_cols = ["hgb_pred_r1", "extra_trees_pred_r1", "random_forest_pred_r1", "ensemble_pred_r1", "weighted_ensemble_pred_r1"]
    for fam_name in families:
        sub = cand[cand["family"].eq(fam_name)][["ID"] + [c for c in pred_cols if c in cand.columns]].copy()
        sub = sub.rename(columns={c: f"{fam_name}_{c}" for c in pred_cols if c in sub.columns})
        row = row.merge(sub, on="ID", how="left")
    return row, families


def score_predictions(row, pred_family_col, families):
    idx = pd.factorize(row["ID"])[0]
    chosen = []
    for _, r in row.iterrows():
        fam_name = r[pred_family_col]
        score_col = f"score_{fam_name}"
        chosen.append(r[score_col] if score_col in row.columns else r["score_meta_gate"])
    return np.asarray(chosen, dtype=float)


def run():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    row, families = build_row_table()
    le = LabelEncoder()
    y = le.fit_transform(row["oracle_family"].astype(str))
    cat_cols = ["subset", "hybrid_source"]
    banned = {"ID", "oracle_family", "oracle_r1"} | {f"score_{name}" for name in families}
    num_cols = [c for c in row.columns if c not in banned and c not in cat_cols]
    for c in num_cols:
        row[c] = pd.to_numeric(row[c], errors="coerce")

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    pred_family_cols = []
    proba_cols = []
    for kind in ["hgb", "extra_trees", "random_forest"]:
        pred = np.zeros(len(row), dtype=int)
        maxp = np.zeros(len(row), dtype=np.float32)
        for fold, (tr, va) in enumerate(folds.split(row["ID"], row["subset"]), start=1):
            model = make_model(kind, cat_cols, num_cols)
            model.fit(row.iloc[tr][cat_cols + num_cols], y[tr])
            proba = model.predict_proba(row.iloc[va][cat_cols + num_cols])
            # Align class columns for safety.
            classes = model.named_steps["clf"].classes_
            full = np.zeros((len(va), len(le.classes_)), dtype=np.float32)
            for j, cls in enumerate(classes):
                full[:, cls] = proba[:, j]
            pred[va] = full.argmax(axis=1)
            maxp[va] = full.max(axis=1)
            print(f"{kind} fold {fold} done", flush=True)
        col = f"{kind}_pred_family"
        pcol = f"{kind}_max_proba"
        row[col] = le.inverse_transform(pred)
        row[pcol] = maxp
        pred_family_cols.append(col)
        proba_cols.append(pcol)
        chosen = score_predictions(row, col, families)
        results.append(
            {
                "model": kind,
                "score": float(chosen.mean()),
                "gain_vs_exp2": float(chosen.mean() - BASELINE_EXP2),
                "gain_vs_family_meta_subset_hybrid": float(chosen.mean() - 0.6258761046358958),
                "accuracy": float((row[col] == row["oracle_family"]).mean()),
                "choice_counts": row[col].value_counts().to_dict(),
                "per_subset": pd.DataFrame({"subset": row["subset"], "r1": chosen}).groupby("subset")["r1"].mean().round(6).to_dict(),
            }
        )

    # Majority/average-proxy ensemble: use the family whose predicted class is most common, tie by highest model confidence.
    votes = row[pred_family_cols].to_numpy()
    ens = []
    for i, vote_row in enumerate(votes):
        counts = pd.Series(vote_row).value_counts()
        top = counts[counts == counts.max()].index.tolist()
        if len(top) == 1:
            ens.append(top[0])
        else:
            best = None
            for fam_name in top:
                conf = sum(row.loc[i, p] for c, p in zip(pred_family_cols, proba_cols) if row.loc[i, c] == fam_name)
                if best is None or conf > best[1]:
                    best = (fam_name, conf)
            ens.append(best[0])
    row["vote_ensemble_pred_family"] = ens
    chosen = score_predictions(row, "vote_ensemble_pred_family", families)
    results.append(
        {
            "model": "vote_ensemble",
            "score": float(chosen.mean()),
            "gain_vs_exp2": float(chosen.mean() - BASELINE_EXP2),
            "gain_vs_family_meta_subset_hybrid": float(chosen.mean() - 0.6258761046358958),
            "accuracy": float((row["vote_ensemble_pred_family"] == row["oracle_family"]).mean()),
            "choice_counts": row["vote_ensemble_pred_family"].value_counts().to_dict(),
            "per_subset": pd.DataFrame({"subset": row["subset"], "r1": chosen}).groupby("subset")["r1"].mean().round(6).to_dict(),
        }
    )

    lb = pd.DataFrame(results).sort_values("score", ascending=False)
    best_model = lb.iloc[0]["model"]
    pred_col = f"{best_model}_pred_family" if best_model != "vote_ensemble" else "vote_ensemble_pred_family"
    row["best_pred_family"] = row[pred_col]
    row["best_pred_r1"] = score_predictions(row, pred_col, families)
    row.to_csv(REPORT_DIR / "family_classifier_rows.csv", index=False)
    lb.to_csv(REPORT_DIR / "leaderboard.csv", index=False)
    summary = {
        "family_oracle": float(row["oracle_r1"].mean()),
        "current_best": 0.6258761046358958,
        "best": lb.iloc[0].to_dict(),
        "results": results,
        "notes": [
            "This predicts the oracle selector family directly.",
            "Scores are computed from held-out/OFF family outputs; no reference text is used as a prediction feature.",
        ],
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(lb.to_string(index=False))


if __name__ == "__main__":
    run()
