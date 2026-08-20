from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

try:
    from lightgbm import LGBMRanker, LGBMRegressor
except Exception:  # pragma: no cover - optional local dependency
    LGBMRanker = None
    LGBMRegressor = None


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
OUT = ROOT / "reports" / "candidate_ranker_current_best_gate"
OUT.mkdir(parents=True, exist_ok=True)

VAL_FEATURES = ROOT / "modal_outputs" / "exp2_val_candidate_scores" / "candidate_tabular_features.csv"
VAL_TEXT = ROOT / "modal_outputs" / "exp2_val_candidate_scores" / "val_candidate_scores.csv"
VAL = ROOT / "Val.csv"
TEST = ROOT / "Test.csv"
TEST_FEATURES = (
    ROOT
    / "modal_outputs"
    / "exp7_cluster_selector_test_predictions_files"
    / "test_candidate_features_scored.csv"
)
CURRENT_BEST_VAL = ROOT / "reports" / "exp7_exp5_length_grouped_submission" / "oof_choices.csv"
CURRENT_BEST_TEST = (
    ROOT
    / "modal_outputs"
    / "exp5_encoder_exp2_test_predictions_files"
    / "submission_exp7_base_exp5_length_grouped.csv"
)
SUB_OUT = (
    ROOT
    / "modal_outputs"
    / "exp7_submissions"
    / "submission_exp7_exp5_candidate_ranker_gate.csv"
)

IDCOL = "ID"
GCOL = "subset"
SEED = 606
N_SPLITS = 5

LEAKAGE_COLUMNS = {
    "candidate_r1",
    "reference_len",
    "answer_ref_len_ratio",
    "is_oracle",
    "reference",
    "output",
    "target_r1",
    "target",
}

BASE_FEATURES = [
    "subset",
    "candidate_rank",
    "bi_rank",
    "rerank_rank",
    "bi_score",
    "rerank_score",
    "rerank_score_max",
    "rerank_score_second",
    "rerank_score_margin_to_best",
    "rerank_score_gap_from_second",
    "rerank_score_delta_to_group_best",
    "bi_score_max",
    "bi_score_delta_to_group_best",
    "is_top1",
    "is_rerank_choice",
    "val_input_len",
    "candidate_question_len",
    "candidate_answer_len",
    "candidate_len_ratio_to_query",
    "query_candidate_question_jaccard",
    "query_candidate_answer_jaccard",
    "query_candidate_best_jaccard",
]


def rouge1_f1(pred: object, ref: object) -> float:
    p = str(pred).strip().split()
    r = str(ref).strip().split()
    if not p or not r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    return float(2 * overlap / (len(p) + len(r)))


_WS = re.compile(r"\s+")


def norm_text(x: object) -> str:
    return _WS.sub(" ", str(x).strip().lower())


def second_largest(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce").dropna().nlargest(2)
    if len(vals) >= 2:
        return float(vals.iloc[-1])
    if len(vals) == 1:
        return float(vals.iloc[0])
    return 0.0


def add_group_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["candidate_rank", "bi_rank", "rerank_rank", "bi_score", "rerank_score"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["candidate_rank"] = out["candidate_rank"].fillna(10000)
    out["bi_rank"] = out["bi_rank"].fillna(out["candidate_rank"]).fillna(10000)
    out["rerank_rank"] = out["rerank_rank"].fillna(10000)
    out["bi_score"] = out["bi_score"].fillna(-1e6)
    out["rerank_score"] = out["rerank_score"].fillna(-1e6)

    out["rerank_score_max"] = out.groupby(IDCOL)["rerank_score"].transform("max")
    out["rerank_score_second"] = out.groupby(IDCOL)["rerank_score"].transform(second_largest)
    out["rerank_score_margin_to_best"] = out["rerank_score_max"] - out["rerank_score"]
    out["rerank_score_gap_from_second"] = out["rerank_score"] - out["rerank_score_second"]
    out["rerank_score_delta_to_group_best"] = out["rerank_score"] - out["rerank_score_max"]
    out["bi_score_max"] = out.groupby(IDCOL)["bi_score"].transform("max")
    out["bi_score_delta_to_group_best"] = out["bi_score"] - out["bi_score_max"]
    out["is_top1"] = (out["candidate_rank"] == 1).astype(int)
    out["is_rerank_choice"] = (out["rerank_rank"] == 1).astype(int)

    out["val_input_len"] = pd.to_numeric(out["val_input_len"], errors="coerce").fillna(0.0)
    out["candidate_question_len"] = pd.to_numeric(out["candidate_question_len"], errors="coerce").fillna(0.0)
    out["candidate_answer_len"] = pd.to_numeric(out["candidate_answer_len"], errors="coerce").fillna(0.0)
    out["candidate_len_ratio_to_query"] = out["candidate_answer_len"] / out["val_input_len"].clip(lower=1)
    out["query_candidate_question_jaccard"] = pd.to_numeric(
        out["query_candidate_question_jaccard"], errors="coerce"
    ).fillna(0.0)
    out["query_candidate_answer_jaccard"] = pd.to_numeric(
        out["query_candidate_answer_jaccard"], errors="coerce"
    ).fillna(0.0)
    out["query_candidate_best_jaccard"] = out[
        ["query_candidate_question_jaccard", "query_candidate_answer_jaccard"]
    ].max(axis=1)

    for col in ["candidate_answer", "candidate_question"]:
        if col not in out:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    out["answer_freq_subset"] = out.groupby([GCOL, "candidate_answer"])[IDCOL].transform("count")
    out["answer_freq_global"] = out.groupby("candidate_answer")[IDCOL].transform("count")
    out["question_freq_subset"] = out.groupby([GCOL, "candidate_question"])[IDCOL].transform("count")
    out["question_freq_global"] = out.groupby("candidate_question")[IDCOL].transform("count")
    out["log_answer_freq_subset"] = np.log1p(out["answer_freq_subset"])
    out["log_answer_freq_global"] = np.log1p(out["answer_freq_global"])
    return out


def load_validation_candidates() -> pd.DataFrame:
    feats = pd.read_csv(VAL_FEATURES)
    df = feats.copy()
    df["candidate_question"] = ""
    df["candidate_answer"] = ""
    df["val_input"] = ""
    df["reference"] = ""
    print("Skipping malformed validation candidate text CSV; using clean tabular candidates only.", flush=True)
    if "val_input_len" not in df:
        df["val_input_len"] = df["val_input"].fillna("").astype(str).str.len()
    df = add_group_features(df)
    return df


def load_test_candidates() -> pd.DataFrame:
    raw = pd.read_csv(TEST_FEATURES)
    df = pd.DataFrame()
    df[IDCOL] = raw[IDCOL]
    df[GCOL] = raw[GCOL]
    df["subset"] = raw[GCOL]
    df["candidate_answer"] = raw["candidate_answer"].fillna("").astype(str)
    df["candidate_question"] = raw["candidate_question"].fillna("").astype(str)
    df["candidate_rank"] = pd.to_numeric(raw["bge_q2q_top50_best_rank"], errors="coerce").fillna(10000)
    df["bi_rank"] = df["candidate_rank"]
    df["rerank_rank"] = pd.to_numeric(raw["exp6_rerank_rank"], errors="coerce").fillna(10000)
    df["bi_score"] = pd.to_numeric(raw["bge_q2q_top50_max_score"], errors="coerce").fillna(-1e6)
    df["rerank_score"] = pd.to_numeric(raw["exp6_rerank_score"], errors="coerce").fillna(-1e6)
    df["val_input_len"] = pd.to_numeric(raw["query_len"], errors="coerce").fillna(0.0)
    df["candidate_question_len"] = pd.to_numeric(raw["candidate_question_len"], errors="coerce").fillna(0.0)
    df["candidate_answer_len"] = pd.to_numeric(raw["candidate_answer_len"], errors="coerce").fillna(0.0)
    df["query_candidate_question_jaccard"] = pd.to_numeric(
        raw["query_question_jaccard"], errors="coerce"
    ).fillna(0.0)
    df["query_candidate_answer_jaccard"] = pd.to_numeric(
        raw["query_answer_jaccard"], errors="coerce"
    ).fillna(0.0)
    return add_group_features(df)


def prep_x(
    df: pd.DataFrame,
    features: list[str],
    enc: OrdinalEncoder | None = None,
) -> tuple[pd.DataFrame, OrdinalEncoder]:
    x = df[features].copy()
    if any(c in LEAKAGE_COLUMNS for c in x.columns):
        bad = sorted(set(x.columns) & LEAKAGE_COLUMNS)
        raise RuntimeError(f"Leakage feature requested: {bad}")
    num_cols = [c for c in features if c != "subset"]
    x[num_cols] = x[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if enc is None:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        x[["subset"]] = enc.fit_transform(x[["subset"]].fillna(""))
    else:
        x[["subset"]] = enc.transform(x[["subset"]].fillna(""))
    return x, enc


def choose_by_score(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    idx = df.groupby(IDCOL)[score_col].idxmax()
    return df.loc[idx].copy()


def choose_rerank(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["_rr_choice"] = np.where(
        temp["rerank_rank"].eq(1),
        1_000_000 + temp["rerank_score"],
        temp["rerank_score"],
    )
    return choose_by_score(temp, "_rr_choice")


def model_specs() -> list[tuple[str, object, bool]]:
    specs: list[tuple[str, object, bool]] = [
        (
            "hgb",
            HistGradientBoostingRegressor(
                max_iter=500,
                learning_rate=0.025,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=SEED,
            ),
            False,
        ),
    ]
    if LGBMRegressor is not None:
        specs.append(
            (
                "lgbm_reg",
                LGBMRegressor(
                    objective="regression",
                    n_estimators=500,
                    learning_rate=0.035,
                    num_leaves=31,
                    min_child_samples=24,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=2.0,
                    random_state=SEED,
                    n_jobs=-1,
                    verbose=-1,
                ),
                False,
            )
        )
    return specs


def fit_predict_model(
    model: object,
    is_ranker: bool,
    tr: pd.DataFrame,
    va: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    x_tr, enc = prep_x(tr, features)
    x_va, _ = prep_x(va, features, enc)
    y_tr = tr["candidate_r1"].to_numpy(dtype=np.float32)
    if is_ranker:
        ordered = tr.sort_values(IDCOL).copy()
        x_tr, enc = prep_x(ordered, features)
        y_tr = ordered["candidate_r1"].to_numpy(dtype=np.float32)
        group = ordered.groupby(IDCOL, sort=False).size().to_numpy()
        model.fit(x_tr, y_tr, group=group)
        x_va, _ = prep_x(va, features, enc)
    else:
        model.fit(x_tr, y_tr)
    return model.predict(x_va)


def score_candidate_choices(choices: pd.DataFrame) -> float:
    return float(choices["candidate_r1"].mean())


def current_best_rows(val_candidates: pd.DataFrame) -> pd.DataFrame:
    cur = pd.read_csv(CURRENT_BEST_VAL)
    val = pd.read_csv(VAL, usecols=[IDCOL, GCOL, "output"])
    cur = cur.merge(val, on=[IDCOL, GCOL], how="left")
    if "target_r1" not in cur:
        cur["target_r1"] = [rouge1_f1(p, r) for p, r in zip(cur["prediction"], cur["output"])]
    cur["prediction_norm"] = cur["prediction"].map(norm_text)

    cand = val_candidates[[IDCOL, "candidate_answer", "candidate_rank", "rerank_rank"]].copy()
    cand["candidate_answer_norm"] = cand["candidate_answer"].map(norm_text)
    matched = cur[[IDCOL, "prediction_norm"]].merge(
        cand,
        left_on=[IDCOL, "prediction_norm"],
        right_on=[IDCOL, "candidate_answer_norm"],
        how="left",
    )
    matched = matched.sort_values([IDCOL, "candidate_rank"]).drop_duplicates(IDCOL)
    out = cur.merge(
        matched[[IDCOL, "candidate_rank", "rerank_rank"]].rename(
            columns={
                "candidate_rank": "current_candidate_rank",
                "rerank_rank": "current_rerank_rank",
            }
        ),
        on=IDCOL,
        how="left",
    )
    out = out.rename(columns={"prediction": "current_prediction", "target_r1": "current_r1"})
    return out[[IDCOL, GCOL, "current_prediction", "current_r1", "current_candidate_rank", "current_rerank_rank"]]


def tune_gate_vs_current(
    pred_df: pd.DataFrame,
    current: pd.DataFrame,
    score_col: str,
    label: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    best = choose_by_score(pred_df, score_col)
    rerank = choose_rerank(pred_df)
    rows = best[
        [
            IDCOL,
            GCOL,
            "candidate_answer",
            "candidate_rank",
            "rerank_rank",
            "candidate_r1",
            score_col,
            "rerank_score",
        ]
    ].merge(
        rerank[[IDCOL, score_col]].rename(columns={score_col: "rerank_choice_pred"}),
        on=IDCOL,
        how="left",
    )
    rows = rows.merge(current, on=[IDCOL, GCOL], how="left")
    rows["pred_margin_vs_rerank_choice"] = rows[score_col] - rows["rerank_choice_pred"]
    rows["actual_gain_vs_current"] = rows["candidate_r1"] - rows["current_r1"]
    rows["same_as_current"] = rows["candidate_answer"].map(norm_text).eq(rows["current_prediction"].map(norm_text))
    rows["current_is_in_exp2_pool"] = rows["current_candidate_rank"].notna()
    rows["ranker_is_candidate_rank1"] = rows["candidate_rank"].eq(1)
    rows["ranker_is_rerank_rank1"] = rows["rerank_rank"].eq(1)
    base_score = float(rows["current_r1"].mean())

    records = []
    thresholds = np.round(np.arange(-0.05, 0.251, 0.005), 3)
    rank_caps = [1, 3, 5, 10, 25, 10000]
    require_pool_opts = [False, True]
    for t in thresholds:
        for cap in rank_caps:
            for require_pool in require_pool_opts:
                mask = (
                    ~rows["same_as_current"]
                    & rows["pred_margin_vs_rerank_choice"].ge(t)
                    & rows["candidate_rank"].le(cap)
                )
                if require_pool:
                    mask = mask & rows["current_is_in_exp2_pool"]
                score = base_score + float(rows.loc[mask, "actual_gain_vs_current"].sum()) / len(rows)
                records.append(
                    {
                        "model": label,
                        "threshold": float(t),
                        "rank_cap": int(cap),
                        "require_current_in_pool": bool(require_pool),
                        "score": score,
                        "gain_vs_current": score - base_score,
                        "switch_n": int(mask.sum()),
                        "mean_gain_switched": float(rows.loc[mask, "actual_gain_vs_current"].mean())
                        if mask.any()
                        else 0.0,
                    }
                )
    lb = pd.DataFrame(records).sort_values(
        ["score", "switch_n", "rank_cap"], ascending=[False, True, True]
    )
    best_gate = lb.iloc[0].to_dict()
    rows["model"] = label
    rows.to_csv(OUT / f"{label}_gate_rows.csv", index=False)
    lb.to_csv(OUT / f"{label}_gate_leaderboard.csv", index=False)
    return best_gate, rows, lb


def train_oof(val_candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    features = [c for c in BASE_FEATURES if c in val_candidates.columns]
    leakage_used = sorted(set(features) & LEAKAGE_COLUMNS)
    if leakage_used:
        raise RuntimeError(f"Leakage columns in features: {leakage_used}")

    ids = val_candidates[[IDCOL, GCOL]].drop_duplicates().reset_index(drop=True)
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    out = val_candidates.copy()
    summaries: list[dict] = []
    current = current_best_rows(val_candidates)
    base_current = float(current["current_r1"].mean())
    base_rerank = score_candidate_choices(choose_rerank(val_candidates))
    base_bi_top1 = score_candidate_choices(val_candidates[val_candidates["candidate_rank"].eq(1)].drop_duplicates(IDCOL))

    fold_map = {}
    for fold, (_, va_idx) in enumerate(folds.split(ids[IDCOL], ids[GCOL]), start=1):
        for row_id in ids.iloc[va_idx][IDCOL]:
            fold_map[row_id] = fold
    out["fold"] = out[IDCOL].map(fold_map)

    for label, base_model, is_ranker in model_specs():
        score_col = f"pred_{label}"
        out[score_col] = np.nan
        print(f"\nTraining OOF {label}...", flush=True)
        for fold, (tr_idx, va_idx) in enumerate(folds.split(ids[IDCOL], ids[GCOL]), start=1):
            tr_ids = set(ids.iloc[tr_idx][IDCOL])
            va_ids = set(ids.iloc[va_idx][IDCOL])
            tr = out[out[IDCOL].isin(tr_ids)].copy()
            va = out[out[IDCOL].isin(va_ids)].copy()
            model = base_model.__class__(**base_model.get_params())
            pred = fit_predict_model(model, is_ranker, tr, va, features)
            out.loc[va.index, score_col] = pred
            print(f"  fold {fold}: {len(tr):,} train pairs -> {len(va):,} val pairs", flush=True)
        choices = choose_by_score(out, score_col)
        gate, _, _ = tune_gate_vs_current(out, current, score_col, label)
        summaries.append(
            {
                "model": label,
                "candidate_choice_score": float(choices["candidate_r1"].mean()),
                "gain_vs_exp2_rerank": float(choices["candidate_r1"].mean() - base_rerank),
                "current_best_oof_score": base_current,
                "gain_if_used_directly_vs_current": float(choices["candidate_r1"].mean() - base_current),
                "best_gate": gate,
            }
        )

    pred_cols = [c for c in out.columns if c.startswith("pred_")]
    out["pred_mean"] = out[pred_cols].mean(axis=1)
    out["pred_max"] = out[pred_cols].max(axis=1)
    for ens_col in ["pred_mean", "pred_max"]:
        label = ens_col.replace("pred_", "ensemble_")
        choices = choose_by_score(out, ens_col)
        gate, _, _ = tune_gate_vs_current(out, current, ens_col, label)
        summaries.append(
            {
                "model": label,
                "candidate_choice_score": float(choices["candidate_r1"].mean()),
                "gain_vs_exp2_rerank": float(choices["candidate_r1"].mean() - base_rerank),
                "current_best_oof_score": base_current,
                "gain_if_used_directly_vs_current": float(choices["candidate_r1"].mean() - base_current),
                "best_gate": gate,
            }
        )

    summary_df = pd.DataFrame(summaries).sort_values(
        by=["current_best_oof_score"], ascending=False
    )
    common = {
        "feature_columns": features,
        "leakage_columns_excluded": sorted(LEAKAGE_COLUMNS),
        "current_best_oof_score": base_current,
        "exp2_pool_rerank_score": base_rerank,
        "exp2_pool_bi_top1_score": base_bi_top1,
        "n_val_ids": int(ids[IDCOL].nunique()),
        "n_val_candidate_rows": int(len(out)),
        "n_current_best_predictions_present_in_exp2_pool": int(current["current_candidate_rank"].notna().sum()),
    }
    return out, summary_df, [common] + summaries


def fit_final_and_predict_test(
    val_candidates: pd.DataFrame,
    test_candidates: pd.DataFrame,
    summary_rows: list[dict],
) -> tuple[pd.DataFrame, dict]:
    features = summary_rows[0]["feature_columns"]
    best_model_name = max(
        summary_rows[1:],
        key=lambda r: (
            r["best_gate"]["gain_vs_current"],
            -r["best_gate"]["switch_n"],
        ),
    )["model"]
    best_gate = next(r["best_gate"] for r in summary_rows[1:] if r["model"] == best_model_name)
    if best_model_name.startswith("ensemble_"):
        train_labels = [name for name, _, _ in model_specs()]
    else:
        train_labels = [best_model_name]

    pred_cols = []
    fitted_meta = []
    for label, base_model, is_ranker in model_specs():
        if label not in train_labels:
            continue
        model = base_model.__class__(**base_model.get_params())
        x_train, enc = prep_x(val_candidates.sort_values(IDCOL), features)
        train = val_candidates.sort_values(IDCOL)
        y = train["candidate_r1"].to_numpy(dtype=np.float32)
        if is_ranker:
            group = train.groupby(IDCOL, sort=False).size().to_numpy()
            model.fit(x_train, y, group=group)
        else:
            model.fit(x_train, y)
        x_test, _ = prep_x(test_candidates, features, enc)
        pred_col = f"final_{label}"
        test_candidates[pred_col] = model.predict(x_test)
        pred_cols.append(pred_col)
        fitted_meta.append({"model": label, "is_ranker": bool(is_ranker)})

    if best_model_name == "ensemble_mean":
        score_col = "final_ensemble_mean"
        test_candidates[score_col] = test_candidates[pred_cols].mean(axis=1)
    elif best_model_name == "ensemble_max":
        score_col = "final_ensemble_max"
        test_candidates[score_col] = test_candidates[pred_cols].max(axis=1)
    else:
        score_col = f"final_{best_model_name}"

    best = choose_by_score(test_candidates, score_col)
    rerank = choose_rerank(test_candidates)
    rows = best[[IDCOL, GCOL, "candidate_answer", "candidate_rank", "rerank_rank", score_col]].merge(
        rerank[[IDCOL, score_col]].rename(columns={score_col: "rerank_choice_pred"}),
        on=IDCOL,
        how="left",
    )
    rows["pred_margin_vs_rerank_choice"] = rows[score_col] - rows["rerank_choice_pred"]
    mask = rows["pred_margin_vs_rerank_choice"].ge(best_gate["threshold"]) & rows["candidate_rank"].le(
        best_gate["rank_cap"]
    )
    rows["use_ranker_override"] = mask

    base = pd.read_csv(CURRENT_BEST_TEST)
    before = base["TargetR1F1"].astype(str).copy()
    pred_map = rows[rows["use_ranker_override"]].set_index(IDCOL)["candidate_answer"].to_dict()
    sub = base.copy()
    sub["TargetR1F1"] = [pred_map.get(i, p) for i, p in zip(sub[IDCOL], sub["TargetR1F1"])]
    sub["TargetRLF1"] = sub["TargetR1F1"]
    sub["TargetLLM"] = sub["TargetR1F1"]
    SUB_OUT.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUB_OUT, index=False)
    rows.to_csv(OUT / "test_ranker_gate_choices.csv", index=False)
    return rows, {
        "selected_model_for_test": best_model_name,
        "selected_gate": best_gate,
        "fitted_models": fitted_meta,
        "submission": str(SUB_OUT),
        "test_override_count": int(mask.sum()),
        "changed_vs_current_best_submission": int((sub["TargetR1F1"].astype(str) != before).sum()),
    }


def main() -> None:
    print("Loading validation candidates...", flush=True)
    val_candidates = load_validation_candidates()
    print(f"Validation candidates: {len(val_candidates):,} rows, {val_candidates[IDCOL].nunique():,} IDs", flush=True)
    oof, summary_df, summary_rows = train_oof(val_candidates)
    oof.to_csv(OUT / "candidate_oof_predictions.csv", index=False)
    summary_df.to_csv(OUT / "model_summary.csv", index=False)

    print("\nLoading test candidates...", flush=True)
    test_candidates = load_test_candidates()
    test_rows, test_summary = fit_final_and_predict_test(val_candidates, test_candidates, summary_rows)

    summary = {
        **summary_rows[0],
        "models": summary_rows[1:],
        "test": test_summary,
        "notes": [
            "This experiment ranks individual candidates, then gates against the current best exp7+exp5 submission.",
            "Features intentionally exclude target/reference/oracle columns; target ROUGE is used only as the training label.",
            "A high validation score from direct candidate selection is not enough here; only the gate-vs-current metric matters.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
