from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
REPORT_DIR = ROOT / "reports" / "cluster_fast_gate"
VAL = ROOT / "Val.csv"
EXP2 = ROOT / "modal_outputs/exp2_crossencoder_rerank/val_predictions.csv"
CLUSTER_FAST = ROOT / "reports/cluster_aware_selector_experiment/cluster_selector_fast_oof_choices.csv"
CLUSTER_NOLEAK = ROOT / "reports/cluster_aware_selector_experiment/cluster_selector_noleak_oof_choices.csv"
HYBRID = ROOT / "reports/subset_clean_source_selector/hybrid_best_oof_choices.csv"
RANDOM_STATE = 41


def rouge1(pred, ref):
    pred_toks = str(pred).strip().split()
    ref_toks = str(ref).strip().split()
    if not pred_toks or not ref_toks:
        return 0.0
    overlap = sum((Counter(pred_toks) & Counter(ref_toks)).values())
    if overlap == 0:
        return 0.0
    return float(2.0 * overlap / (len(pred_toks) + len(ref_toks)))


def jaccard(a, b):
    a = set(str(a).split())
    b = set(str(b).split())
    return len(a & b) / len(a | b) if a and b else 0.0


def pick_model(df, contains):
    if "model" not in df.columns:
        return df
    hit = df[df["model"].astype(str).str.contains(contains, regex=False)].copy()
    return hit if len(hit) else df.drop_duplicates("ID").copy()


def load_table():
    val = pd.read_csv(VAL)[["ID", "subset", "input", "output"]]
    exp2 = pd.read_csv(EXP2)[["ID", "top1", "rerank"]].rename(columns={"rerank": "exp2_rerank"})
    fast = pick_model(pd.read_csv(CLUSTER_FAST), "hist_gb_fast_depth6")
    noleak = pick_model(pd.read_csv(CLUSTER_NOLEAK), "hist_gb_noleak_depth6")
    hybrid = pd.read_csv(HYBRID)[["ID", "source", "prediction", "target_r1"]].rename(
        columns={"source": "hybrid_source", "prediction": "hybrid_prediction", "target_r1": "hybrid_r1"}
    )

    # Only keep features available at test time. Validation target/reference-derived columns are diagnostic only.
    banned = {
        "target_r1",
        "reference",
        "reference_len",
        "answer_ref_len_ratio_proxy",
        "selector_true_rouge1",
        "oracle_r1",
    }
    fast_cols = [c for c in fast.columns if c not in banned]
    no_cols = [c for c in noleak.columns if c in {"ID", "candidate_answer", "pred", "candidate_answer_len", "query_question_jaccard", "query_answer_jaccard"}]
    fast = fast[fast_cols].drop_duplicates("ID").add_prefix("fast_").rename(columns={"fast_ID": "ID"})
    noleak = noleak[no_cols].drop_duplicates("ID").add_prefix("noleak_").rename(columns={"noleak_ID": "ID"})
    df = val.merge(exp2, on="ID").merge(fast, on="ID", how="left").merge(noleak, on="ID", how="left").merge(hybrid, on="ID", how="left")
    df["fast_candidate_answer"] = df["fast_candidate_answer"].fillna("")
    df["noleak_candidate_answer"] = df["noleak_candidate_answer"].fillna("")
    df["hybrid_prediction"] = df["hybrid_prediction"].fillna(df["exp2_rerank"])
    for col in ["exp2_rerank", "top1", "fast_candidate_answer", "noleak_candidate_answer", "hybrid_prediction"]:
        df[f"{col}_r1"] = [rouge1(p, r) for p, r in zip(df[col], df["output"])]

    df["fast_gain_vs_hybrid"] = df["fast_candidate_answer_r1"] - df["hybrid_prediction_r1"]
    df["fast_gain_vs_exp2"] = df["fast_candidate_answer_r1"] - df["exp2_rerank_r1"]
    df["fast_beats_hybrid"] = (df["fast_gain_vs_hybrid"] > 0).astype(int)

    pred_cols = ["top1", "exp2_rerank", "fast_candidate_answer", "noleak_candidate_answer", "hybrid_prediction"]
    for a in pred_cols:
        df[f"{a}_len"] = df[a].map(lambda x: len(str(x).split()))
    for a, b in [
        ("fast_candidate_answer", "exp2_rerank"),
        ("fast_candidate_answer", "top1"),
        ("fast_candidate_answer", "noleak_candidate_answer"),
        ("fast_candidate_answer", "hybrid_prediction"),
        ("exp2_rerank", "noleak_candidate_answer"),
        ("exp2_rerank", "top1"),
    ]:
        df[f"jac_{a}_vs_{b}"] = [jaccard(x, y) for x, y in zip(df[a], df[b])]
        df[f"len_diff_{a}_vs_{b}"] = df[f"{a}_len"] - df[f"{b}_len"]
    return df


def make_model(kind, cat_cols, num_cols):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ]
    )
    if kind == "hgb":
        reg = HistGradientBoostingRegressor(
            max_iter=240,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=RANDOM_STATE,
        )
    elif kind == "extra_trees":
        reg = ExtraTreesRegressor(
            n_estimators=360,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("reg", reg)])


def score_gate(df, pred_col, threshold=0.0, subsets=None):
    use_fast = df[pred_col] > threshold
    if subsets is not None:
        use_fast &= df["subset"].isin(subsets)
    chosen_r1 = np.where(use_fast, df["fast_candidate_answer_r1"], df["hybrid_prediction_r1"])
    return {
        "score": float(np.mean(chosen_r1)),
        "gain_vs_hybrid": float(np.mean(chosen_r1) - df["hybrid_prediction_r1"].mean()),
        "use_fast_rate": float(use_fast.mean()),
        "use_fast_count": int(use_fast.sum()),
        "per_subset": pd.DataFrame({"subset": df["subset"], "r1": chosen_r1}).groupby("subset")["r1"].mean().round(6).to_dict(),
    }


def run():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_table()
    cat_cols = ["subset", "hybrid_source"]
    banned = {
        "ID",
        "input",
        "output",
        "top1",
        "exp2_rerank",
        "fast_candidate_answer",
        "noleak_candidate_answer",
        "hybrid_prediction",
        "fast_candidate_answer_r1",
        "exp2_rerank_r1",
        "top1_r1",
        "noleak_candidate_answer_r1",
        "hybrid_prediction_r1",
        "fast_gain_vs_hybrid",
        "fast_gain_vs_exp2",
        "fast_beats_hybrid",
    }
    num_cols = [c for c in df.columns if c not in banned and c not in cat_cols]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ids = df[["ID", "subset"]].drop_duplicates().reset_index(drop=True)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y = df["fast_gain_vs_hybrid"].to_numpy()
    pred_cols = []
    for kind in ["hgb", "extra_trees"]:
        oof = np.zeros(len(df), dtype=np.float32)
        for fold, (tr_i, va_i) in enumerate(folds.split(ids["ID"], ids["subset"]), start=1):
            tr_ids = set(ids.iloc[tr_i]["ID"])
            va_ids = set(ids.iloc[va_i]["ID"])
            tr = df["ID"].isin(tr_ids).to_numpy()
            va = df["ID"].isin(va_ids).to_numpy()
            model = make_model(kind, cat_cols, num_cols)
            model.fit(df.loc[tr, cat_cols + num_cols], y[tr])
            oof[va] = model.predict(df.loc[va, cat_cols + num_cols])
            print(f"{kind} fold {fold} done", flush=True)
        col = f"{kind}_pred_gain"
        df[col] = oof
        pred_cols.append(col)
    df["ensemble_pred_gain"] = df[pred_cols].mean(axis=1)

    target_subsets = ["Lug_Uga", "Eng_Uga", "Eng_Ken", "Swa_Ken"]
    results = []
    for col in pred_cols + ["ensemble_pred_gain"]:
        for label, subsets in [("all", None), ("east_africa", target_subsets)]:
            base_result = score_gate(df, col, 0.0, subsets)
            base_result.update({"model": col, "scope": label, "threshold": 0.0})
            results.append(base_result)

        # Diagnostic threshold sweep on the OOF predictions. This is slightly optimistic, so use it to guide the next model.
        for label, subsets in [("all_sweep", None), ("east_africa_sweep", target_subsets)]:
            best = None
            for threshold in np.quantile(df[col].dropna(), np.linspace(0.05, 0.95, 37)):
                res = score_gate(df, col, float(threshold), subsets)
                if best is None or res["score"] > best["score"]:
                    best = res
                    best["threshold"] = float(threshold)
            best.update({"model": col, "scope": label})
            results.append(best)

    leaderboard = pd.DataFrame(results).sort_values("score", ascending=False)
    best = leaderboard.iloc[0].to_dict()
    best_col = best["model"]
    best_scope = best["scope"]
    best_threshold = float(best["threshold"])
    best_subsets = target_subsets if "east_africa" in str(best_scope) else None
    use_fast = df[best_col] > best_threshold
    if best_subsets is not None:
        use_fast &= df["subset"].isin(best_subsets)
    choices = df[["ID", "subset", "hybrid_source", "hybrid_prediction", "fast_candidate_answer", "hybrid_prediction_r1", "fast_candidate_answer_r1"]].copy()
    choices["use_fast"] = use_fast
    choices["prediction"] = np.where(use_fast, choices["fast_candidate_answer"], choices["hybrid_prediction"])
    choices["target_r1"] = np.where(use_fast, choices["fast_candidate_answer_r1"], choices["hybrid_prediction_r1"])

    df.to_csv(REPORT_DIR / "oof_gate_predictions.csv", index=False)
    choices.to_csv(REPORT_DIR / "best_gate_choices.csv", index=False)
    leaderboard.to_csv(REPORT_DIR / "leaderboard.csv", index=False)
    summary = {
        "baseline_exp2": float(df["exp2_rerank_r1"].mean()),
        "baseline_hybrid": float(df["hybrid_prediction_r1"].mean()),
        "cluster_fast": float(df["fast_candidate_answer_r1"].mean()),
        "oracle_fast_vs_hybrid": float(np.maximum(df["fast_candidate_answer_r1"], df["hybrid_prediction_r1"]).mean()),
        "fast_beats_hybrid_rate": float(df["fast_beats_hybrid"].mean()),
        "best": best,
        "target_subsets": target_subsets,
        "notes": [
            "Threshold-sweep rows are diagnostic and may be optimistic because the threshold is chosen on OOF validation predictions.",
            "The zero-threshold rows are the cleaner read of whether predicted gain is calibrated.",
        ],
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(leaderboard.head(20).to_string(index=False))


if __name__ == "__main__":
    run()
