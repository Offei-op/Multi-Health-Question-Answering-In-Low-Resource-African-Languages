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
REPORT_DIR = ROOT / "reports" / "selector_family_meta_learner"
RANDOM_STATE = 97


FAMILY_FILES = [
    ("meta_gate", ROOT / "reports/rich_clean_meta_selector/meta_gate_subset_hybrid_choices.csv", "target_r1", "prediction"),
    ("rich_meta", ROOT / "reports/rich_clean_meta_selector/best_oof_choices.csv", "target_r1", "prediction"),
    ("fast_gate", ROOT / "reports/cluster_fast_gate/best_gate_choices.csv", "target_r1", "prediction"),
    ("residual", ROOT / "reports/residual_clean_override_selector/best_override_choices.csv", "final_r1", "prediction"),
    ("subset_clean", ROOT / "reports/subset_clean_source_selector/best_oof_choices.csv", "target_r1", "prediction"),
    ("global_clean", ROOT / "reports/clean_deployable_source_selector/best_oof_choices.csv", "target_r1", "prediction"),
]


def make_model(kind, cat_cols, num_cols):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ]
    )
    if kind == "hgb":
        reg = HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=RANDOM_STATE,
        )
    elif kind == "extra_trees":
        reg = ExtraTreesRegressor(
            n_estimators=360,
            min_samples_leaf=8,
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


def load_family_candidates():
    row_features = gate.load_table()
    clean_feature_cols = ["ID", "subset", "hybrid_source"]
    for c in row_features.columns:
        if (
            c.startswith("fast_")
            or c.startswith("noleak_")
            or c.startswith("jac_")
            or c.startswith("len_diff_")
        ) and not c.endswith("_r1") and "gain" not in c and "beats" not in c:
            clean_feature_cols.append(c)
    clean_feature_cols = list(dict.fromkeys(clean_feature_cols))
    row_features = row_features[clean_feature_cols].copy()

    wide_preds = None
    frames = []
    for name, path, rcol, pcol in FAMILY_FILES:
        df = pd.read_csv(path)
        df = df[["ID", "subset", rcol, pcol]].rename(columns={rcol: "target_r1", pcol: "prediction"})
        df["family"] = name
        df["prediction"] = df["prediction"].fillna("")
        if wide_preds is None:
            wide_preds = df[["ID", "subset", "prediction"]].rename(columns={"prediction": name})
        else:
            wide_preds = wide_preds.merge(df[["ID", "prediction"]].rename(columns={"prediction": name}), on="ID", how="left")
        frames.append(df)
    cand = pd.concat(frames, ignore_index=True)
    cand = cand.merge(row_features, on=["ID", "subset"], how="left")
    cand = cand.merge(wide_preds, on=["ID", "subset"], how="left")
    cand["pred_len"] = cand["prediction"].map(lambda x: len(str(x).split()))
    cand["char_len"] = cand["prediction"].map(lambda x: len(str(x)))
    for name, _, _, _ in FAMILY_FILES:
        cand[f"jac_vs_{name}"] = [gate.jaccard(a, b) for a, b in zip(cand["prediction"], cand[name])]
        cand[f"len_diff_vs_{name}"] = cand["pred_len"] - cand[name].map(lambda x: len(str(x).split()))
    cand["agreement_mean"] = cand[[f"jac_vs_{n}" for n, *_ in FAMILY_FILES]].mean(axis=1)
    cand["agreement_max"] = cand[[f"jac_vs_{n}" for n, *_ in FAMILY_FILES]].replace(1.0, np.nan).max(axis=1).fillna(1.0)
    return cand


def choose(cand, col):
    idx = cand.groupby("ID")[col].idxmax()
    return cand.loc[idx].copy()


def run():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cand = load_family_candidates()
    cat_cols = ["subset", "family", "hybrid_source"]
    banned = {"ID", "prediction", "target_r1"} | {n for n, *_ in FAMILY_FILES}
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
                "gain_vs_exp2": float(chosen["target_r1"].mean() - 0.5892166283468145),
                "gain_vs_best_family_hybrid": float(chosen["target_r1"].mean() - 0.6238405650326898),
                "choice_counts": chosen["family"].value_counts().to_dict(),
                "per_subset": chosen.groupby("subset")["target_r1"].mean().round(6).to_dict(),
            }
        )

    best = max(results, key=lambda x: x["score"])
    oracle = choose(cand, "target_r1")
    choices[best["model"]].to_csv(REPORT_DIR / "best_oof_choices.csv", index=False)
    cand.to_csv(REPORT_DIR / "candidate_level_oof_scores.csv", index=False)
    pd.DataFrame(results).sort_values("score", ascending=False).to_csv(REPORT_DIR / "leaderboard.csv", index=False)
    summary = {
        "best_family_hybrid": 0.6238405650326898,
        "family_oracle": float(oracle["target_r1"].mean()),
        "best": best,
        "results": results,
        "notes": [
            "This learns to choose among clean selector-family outputs.",
            "Family target scores come from OOF validation choices; no references are used as prediction features.",
        ],
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(results).sort_values("score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    run()
