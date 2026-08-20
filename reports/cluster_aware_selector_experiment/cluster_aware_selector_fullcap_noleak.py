from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
OUT = ROOT / "reports" / "cluster_aware_selector_experiment"
FEATURES = OUT / "cluster_candidate_features.csv"
IDCOL = "ID"
GCOL = "subset"
RANDOM_STATE = 13


USECOLS = [
    "ID",
    "candidate_answer",
    "subset",
    "target_r1",
    "source_count",
    "rrf",
    "best_any_rank",
    "max_any_score",
    "exp2_candidate_rank",
    "exp2_bi_score",
    "exp2_rerank_score",
    "exp2_rerank_rank",
    "src_exp2_bge_top50",
    "src_bge200_q2q",
    "src_tfidf_char_q2q_top50",
    "src_tfidf_word_q2q_top50",
    "src_tfidf_char_q2a_top50",
    "src_tfidf_word_q2a_top50",
    "bge200_q2q_best_rank",
    "bge200_q2q_max_score",
    "tfidf_char_q2q_top50_best_rank",
    "tfidf_word_q2q_top50_best_rank",
    "tfidf_char_q2a_top50_best_rank",
    "tfidf_word_q2a_top50_best_rank",
    "answer_freq_subset",
    "answer_freq_global",
    "candidate_answer_len",
    "candidate_question_len",
    "val_input_len",
    "query_question_jaccard",
    "query_answer_jaccard",
    "log_answer_freq_subset",
    "log_answer_freq_global",
    "inv_best_any_rank",
    "inv_exp2_candidate_rank",
    "inv_exp2_rerank_rank",
    "inv_bge200_q2q_best_rank",
    "inv_tfidf_char_q2q_top50_best_rank",
    "inv_tfidf_word_q2q_top50_best_rank",
    "inv_tfidf_char_q2a_top50_best_rank",
    "inv_tfidf_word_q2a_top50_best_rank",
]


def add_score(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c not in {IDCOL, GCOL, "candidate_answer"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["_fullcap_pool_score"] = (
        df["rrf"].fillna(0)
        + 0.20 * df["source_count"].fillna(0)
        + 0.10 * df["log_answer_freq_subset"].fillna(0)
        + 3.0 * df["inv_exp2_rerank_rank"].fillna(0)
        + 2.0 * df["inv_bge200_q2q_best_rank"].fillna(0)
        + 0.45 * df["query_question_jaccard"].fillna(0)
        + 0.35 * df["query_answer_jaccard"].fillna(0)
    )
    return df


def load_top_per_id(cap: int = 80) -> pd.DataFrame:
    frames = []
    for i, chunk in enumerate(pd.read_csv(FEATURES, usecols=USECOLS, chunksize=250_000), start=1):
        chunk = add_score(chunk)
        frames.append(
            chunk.sort_values([IDCOL, "_fullcap_pool_score"], ascending=[True, False])
            .groupby(IDCOL, sort=False)
            .head(cap)
        )
        print(f"chunk {i} processed; kept chunk top rows={len(frames[-1]):,}", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df = (
        df.sort_values([IDCOL, "_fullcap_pool_score"], ascending=[True, False])
        .groupby(IDCOL, sort=False)
        .head(cap)
        .reset_index(drop=True)
    )
    return df


def choose_best(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return df.loc[df.groupby(IDCOL)[score_col].idxmax()].copy()


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_top_per_id(cap=80)
    df.to_csv(OUT / "cluster_candidate_features_fullcap80_noleak.csv", index=False)
    print(f"noleak frame: {df.shape}; IDs={df[IDCOL].nunique():,}", flush=True)

    text_cols = {IDCOL, "candidate_answer", "target_r1", "pred"}
    feature_cols = [c for c in df.columns if c not in text_cols]
    cat_cols = [GCOL]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    x = df[num_cols + cat_cols].copy()
    x[num_cols] = x[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    x[cat_cols] = enc.fit_transform(x[cat_cols].fillna(""))
    y = df["target_r1"].to_numpy(dtype=np.float32)
    ids = df[[IDCOL, GCOL]].drop_duplicates().reset_index(drop=True)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "hist_gb_noleak_depth6": HistGradientBoostingRegressor(
            max_iter=260,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.04,
            random_state=RANDOM_STATE,
        ),
        "hist_gb_noleak_depth4": HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=RANDOM_STATE,
        ),
    }
    try:
        from lightgbm import LGBMRegressor

        models["lightgbm_noleak"] = LGBMRegressor(
            n_estimators=360,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception as e:
        print(f"LightGBM unavailable: {e}", flush=True)

    leader = []
    oracle = choose_best(df, "target_r1")
    leader.append({"model": "noleak_candidate_pool_oracle", "score": float(oracle["target_r1"].mean()), "rmse": 0.0, "chosen_rows": len(oracle)})
    for score_col in ["_fullcap_pool_score", "rrf", "source_count", "inv_bge200_q2q_best_rank"]:
        chosen = choose_best(df, score_col)
        leader.append({"model": f"heuristic_{score_col}", "score": float(chosen["target_r1"].mean()), "rmse": np.nan, "chosen_rows": len(chosen)})

    choices = []
    subset_rows = []
    for name, model in models.items():
        pred = np.zeros(len(df), dtype=np.float32)
        for fold, (tr_idx, va_idx) in enumerate(folds.split(ids[IDCOL], ids[GCOL]), start=1):
            tr_ids = set(ids.iloc[tr_idx][IDCOL])
            va_ids = set(ids.iloc[va_idx][IDCOL])
            tr_mask = df[IDCOL].isin(tr_ids).to_numpy()
            va_mask = df[IDCOL].isin(va_ids).to_numpy()
            model.fit(x.loc[tr_mask], y[tr_mask])
            pred[va_mask] = model.predict(x.loc[va_mask])
            print(f"{name} fold {fold} done", flush=True)
        scored = df.copy()
        scored["pred"] = pred
        chosen = choose_best(scored, "pred")
        rmse = math.sqrt(mean_squared_error(y, pred))
        leader.append({"model": name, "score": float(chosen["target_r1"].mean()), "rmse": float(rmse), "chosen_rows": len(chosen)})
        chosen["model"] = name
        choices.append(chosen)
        for subset, grp in chosen.groupby(GCOL, sort=True):
            subset_rows.append({"model": name, "subset": subset, "score": float(grp["target_r1"].mean()), "rows": len(grp)})
    return (
        pd.DataFrame(leader).sort_values("score", ascending=False),
        pd.concat(choices, ignore_index=True),
        pd.DataFrame(subset_rows),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    leader, choices, subset = run()
    leader.to_csv(OUT / "cluster_selector_noleak_leaderboard.csv", index=False)
    choices.to_csv(OUT / "cluster_selector_noleak_oof_choices.csv", index=False)
    subset.to_csv(OUT / "cluster_selector_noleak_subset_summary.csv", index=False)
    summary = {"best": leader.iloc[0].to_dict(), "leaderboard": leader.to_dict(orient="records")}
    (OUT / "cluster_selector_noleak_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(leader.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
