from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
REPORT_DIR = ROOT / "reports" / "predicted_length_cluster_selector"
TRAIN = ROOT / "Train.csv"
VAL = ROOT / "Val.csv"
FEATURES = ROOT / "reports/cluster_aware_selector_experiment/cluster_candidate_features_fullcap80_noleak.csv"
IDCOL = "ID"
GCOL = "subset"
RANDOM_STATE = 19


def token_len(s):
    return len(str(s).split())


def choose_best(df, score_col):
    return df.loc[df.groupby(IDCOL)[score_col].idxmax()].copy()


def add_predicted_lengths(df):
    train = pd.read_csv(TRAIN)
    val = pd.read_csv(VAL)
    train["output_len"] = train["output"].map(token_len).astype(float)
    val["true_output_len"] = val["output"].map(token_len).astype(float)

    pre = ColumnTransformer(
        [
            (
                "text",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=80_000,
                    sublinear_tf=True,
                ),
                "input",
            ),
            ("subset", OneHotEncoder(handle_unknown="ignore"), ["subset"]),
        ]
    )
    length_model = Pipeline(
        [
            ("pre", pre),
            ("reg", Ridge(alpha=6.0, random_state=RANDOM_STATE)),
        ]
    )
    length_model.fit(train[["input", "subset"]], np.log1p(train["output_len"].to_numpy()))
    pred_len = np.expm1(length_model.predict(val[["input", "subset"]]))
    pred_len = np.clip(pred_len, 4, 450)
    length_map = pd.DataFrame({IDCOL: val[IDCOL].astype(str), "pred_ref_len": pred_len, "true_ref_len": val["true_output_len"]})
    df = df.merge(length_map, on=IDCOL, how="left")

    cand_len = pd.to_numeric(df["candidate_answer_len"], errors="coerce").fillna(0).clip(lower=1)
    pred_ref = pd.to_numeric(df["pred_ref_len"], errors="coerce").fillna(cand_len.median()).clip(lower=1)
    df["pred_ref_len"] = pred_ref
    df["len_ratio_pred"] = cand_len / pred_ref
    df["abs_len_ratio_pred_minus_1"] = (df["len_ratio_pred"] - 1.0).abs()
    df["abs_len_diff_pred"] = (cand_len - pred_ref).abs()
    df["log_len_ratio_pred"] = np.log1p(cand_len) - np.log1p(pred_ref)

    val_mae = mean_absolute_error(length_map["true_ref_len"], length_map["pred_ref_len"])
    val_mape = float(np.mean(np.abs(length_map["true_ref_len"] - length_map["pred_ref_len"]) / length_map["true_ref_len"].clip(lower=1)))
    return df, {"val_length_mae_tokens": float(val_mae), "val_length_mape": val_mape}


def make_model(kind, cat_cols, num_cols):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ]
    )
    if kind == "hgb":
        reg = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.06,
            random_state=RANDOM_STATE,
        )
    elif kind == "extra_trees":
        reg = ExtraTreesRegressor(
            n_estimators=320,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("reg", reg)])


def run():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES)
    df[IDCOL] = df[IDCOL].astype(str)
    df, length_metrics = add_predicted_lengths(df)

    text_cols = {IDCOL, "candidate_answer", "target_r1", "pred", "true_ref_len"}
    cat_cols = [GCOL]
    feature_cols = [c for c in df.columns if c not in text_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ids = df[[IDCOL, GCOL]].drop_duplicates().reset_index(drop=True)
    y = df["target_r1"].to_numpy(dtype=np.float32)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    choice_frames = {}
    for kind in ["hgb", "extra_trees"]:
        pred = np.zeros(len(df), dtype=np.float32)
        for fold, (tr_idx, va_idx) in enumerate(folds.split(ids[IDCOL], ids[GCOL]), start=1):
            tr_ids = set(ids.iloc[tr_idx][IDCOL])
            va_ids = set(ids.iloc[va_idx][IDCOL])
            tr_mask = df[IDCOL].isin(tr_ids).to_numpy()
            va_mask = df[IDCOL].isin(va_ids).to_numpy()
            model = make_model(kind, cat_cols, num_cols)
            model.fit(df.loc[tr_mask, cat_cols + num_cols], y[tr_mask])
            pred[va_mask] = model.predict(df.loc[va_mask, cat_cols + num_cols])
            print(f"{kind} fold {fold} done", flush=True)
        col = f"{kind}_pred"
        df[col] = pred
        chosen = choose_best(df, col)
        choice_frames[kind] = chosen
        results.append(
            {
                "model": kind,
                "score": float(chosen["target_r1"].mean()),
                "rmse": float(np.sqrt(np.mean((df[col] - df["target_r1"]) ** 2))),
                "chosen_rows": len(chosen),
                "per_subset": chosen.groupby(GCOL)["target_r1"].mean().round(6).to_dict(),
            }
        )

    df["ensemble_pred"] = df[["hgb_pred", "extra_trees_pred"]].mean(axis=1)
    chosen = choose_best(df, "ensemble_pred")
    choice_frames["ensemble"] = chosen
    results.append(
        {
            "model": "ensemble",
            "score": float(chosen["target_r1"].mean()),
            "rmse": float(np.sqrt(np.mean((df["ensemble_pred"] - df["target_r1"]) ** 2))),
            "chosen_rows": len(chosen),
            "per_subset": chosen.groupby(GCOL)["target_r1"].mean().round(6).to_dict(),
        }
    )

    oracle = choose_best(df, "target_r1")
    rrf = choose_best(df, "rrf")
    best = max(results, key=lambda x: x["score"])
    best_label = best["model"]

    pd.DataFrame(results).sort_values("score", ascending=False).to_csv(REPORT_DIR / "leaderboard.csv", index=False)
    choice_frames[best_label].to_csv(REPORT_DIR / "best_oof_choices.csv", index=False)
    df[[IDCOL, GCOL, "candidate_answer", "target_r1", "pred_ref_len", "len_ratio_pred", "hgb_pred", "extra_trees_pred", "ensemble_pred"]].to_csv(
        REPORT_DIR / "candidate_predictions.csv", index=False
    )

    summary = {
        "length_metrics": length_metrics,
        "oracle": float(oracle["target_r1"].mean()),
        "rrf_baseline": float(rrf["target_r1"].mean()),
        "best": best,
        "results": results,
        "comparison": {
            "exp2_rerank": 0.5892166283468145,
            "cluster_noleak_hgb": 0.5646714819082213,
            "cluster_fullcap_leaky_hgb": 0.6353457041144692,
        },
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
