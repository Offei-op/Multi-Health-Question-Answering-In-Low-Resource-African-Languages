from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
OUT = ROOT / "reports" / "local_candidate_regressor_submission"
OUT.mkdir(parents=True, exist_ok=True)

VAL_FEATURES = ROOT / "modal_outputs" / "exp2_val_candidate_scores" / "candidate_tabular_features.csv"
EXP2_VAL_PREDS = ROOT / "modal_outputs" / "exp2_crossencoder_rerank" / "val_predictions.csv"
TEST_FEATURES = (
    ROOT
    / "modal_outputs"
    / "exp7_cluster_selector_test_predictions_files"
    / "test_candidate_features_scored.csv"
)
SUB_EXP6 = ROOT / "modal_outputs" / "exp7_submissions" / "submission_exp6_reranker_trainval_top50cluster.csv"
SUB_OUT = ROOT / "modal_outputs" / "exp7_submissions" / "submission_local_candidate_regressor_hgb.csv"

IDCOL = "ID"
GCOL = "subset"
SEED = 24


FEATURES = [
    "subset",
    "candidate_rank",
    "bi_rank",
    "rerank_rank",
    "bi_score",
    "rerank_score",
    "rerank_score_max",
    "rerank_score_second",
    "rerank_score_margin_to_best",
    "is_top1",
    "val_input_len",
    "candidate_question_len",
    "candidate_answer_len",
    "query_candidate_question_jaccard",
    "query_candidate_answer_jaccard",
]


def rouge1_f1(pred: object, ref: object) -> float:
    p = str(pred).strip().split()
    r = str(ref).strip().split()
    if not p or not r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return float(2 * precision * recall / (precision + recall))


def prep_x(df: pd.DataFrame, fit_encoder: OrdinalEncoder | None = None) -> tuple[pd.DataFrame, OrdinalEncoder]:
    x = df[FEATURES].copy()
    num_cols = [c for c in FEATURES if c != "subset"]
    x[num_cols] = x[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if fit_encoder is None:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        x[["subset"]] = enc.fit_transform(x[["subset"]].fillna(""))
    else:
        enc = fit_encoder
        x[["subset"]] = enc.transform(x[["subset"]].fillna(""))
    return x, enc


def choose_by_score(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return df.loc[df.groupby(IDCOL)[score_col].idxmax()].copy()


def choose_rerank_rows(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["_rerank_choice_score"] = np.where(
        pd.to_numeric(tmp["rerank_rank"], errors="coerce").fillna(10000).eq(1),
        1_000_000 + pd.to_numeric(tmp["rerank_score"], errors="coerce").fillna(-1),
        pd.to_numeric(tmp["rerank_score"], errors="coerce").fillna(-1),
    )
    return choose_by_score(tmp, "_rerank_choice_score")


def tune_margin_gate(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    best = choose_by_score(df, "pred_hgb")
    rerank = choose_rerank_rows(df)
    rows = best[[IDCOL, GCOL, "candidate_r1", "pred_hgb"]].merge(
        rerank[[IDCOL, "candidate_r1", "pred_hgb"]],
        on=IDCOL,
        suffixes=("_best", "_rerank"),
    )
    rows["pred_margin_vs_rerank"] = rows["pred_hgb_best"] - rows["pred_hgb_rerank"]
    rows["actual_gain_vs_rerank"] = rows["candidate_r1_best"] - rows["candidate_r1_rerank"]
    base = float(rows["candidate_r1_rerank"].mean())
    records = []
    for t in np.round(np.arange(-0.05, 0.151, 0.005), 3):
        mask = rows["pred_margin_vs_rerank"] >= t
        score = base + float(rows.loc[mask, "actual_gain_vs_rerank"].sum()) / len(rows)
        records.append(
            {
                "pred_margin_threshold": float(t),
                "score": score,
                "gain_vs_rerank": score - base,
                "switch_n": int(mask.sum()),
                "mean_actual_gain_switched": float(rows.loc[mask, "actual_gain_vs_rerank"].mean()) if mask.any() else 0.0,
            }
        )
    lb = pd.DataFrame(records).sort_values(["score", "switch_n"], ascending=[False, True]).reset_index(drop=True)
    lb.to_csv(OUT / "margin_gate_leaderboard.csv", index=False)
    rows.to_csv(OUT / "val_margin_gate_rows.csv", index=False)
    return lb.iloc[0].to_dict(), rows


def score_choices(choices: pd.DataFrame, val_preds: pd.DataFrame) -> dict:
    out = val_preds[[IDCOL, GCOL]].merge(choices[[IDCOL, "candidate_r1"]], on=IDCOL, how="left")
    out["r1"] = out["candidate_r1"].fillna(0.0)
    return {
        "rouge1": float(out["r1"].mean()),
        "per_subset": out.groupby(GCOL)["r1"].mean().round(6).to_dict(),
    }


def train_oof() -> tuple[pd.DataFrame, dict, OrdinalEncoder]:
    df = pd.read_csv(VAL_FEATURES)
    val_preds = pd.read_csv(EXP2_VAL_PREDS)
    ids = df[[IDCOL, GCOL]].drop_duplicates().reset_index(drop=True)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    df["pred_hgb"] = np.nan

    model_params = dict(
        max_iter=360,
        learning_rate=0.035,
        max_leaf_nodes=31,
        l2_regularization=0.04,
        random_state=SEED,
    )
    for fold, (tr_id_idx, va_id_idx) in enumerate(folds.split(ids, ids[GCOL]), start=1):
        tr_ids = set(ids.iloc[tr_id_idx][IDCOL])
        va_ids = set(ids.iloc[va_id_idx][IDCOL])
        tr = df[df[IDCOL].isin(tr_ids)]
        va = df[df[IDCOL].isin(va_ids)]
        x_tr, enc = prep_x(tr)
        x_va, _ = prep_x(va, enc)
        model = HistGradientBoostingRegressor(**model_params)
        model.fit(x_tr, tr["candidate_r1"].to_numpy(dtype=np.float32))
        df.loc[va.index, "pred_hgb"] = model.predict(x_va)
        print(f"fold {fold}: trained on {len(tr):,}, predicted {len(va):,}", flush=True)

    choices = choose_by_score(df, "pred_hgb")
    metrics = score_choices(choices, val_preds)
    best_gate, gate_rows = tune_margin_gate(df)
    baseline = {
        "top1": float(val_preds.apply(lambda r: rouge1_f1(r["top1"], r["reference"]), axis=1).mean()),
        "rerank": float(val_preds.apply(lambda r: rouge1_f1(r["rerank"], r["reference"]), axis=1).mean()),
    }
    metrics["gain_vs_rerank"] = metrics["rouge1"] - baseline["rerank"]
    metrics["best_margin_gate"] = best_gate
    metrics["baseline"] = baseline
    save_cols = [IDCOL, GCOL, "candidate_rank", "rerank_rank", "candidate_r1", "pred_hgb"]
    choices[save_cols].to_csv(OUT / "val_oof_choices.csv", index=False)

    # Fit final model on all validation rows for test-time use.
    x_all, enc = prep_x(df)
    final = HistGradientBoostingRegressor(**model_params)
    final.fit(x_all, df["candidate_r1"].to_numpy(dtype=np.float32))
    return df, {"metrics": metrics, "model_params": model_params}, (enc, final, best_gate)


def make_test_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df[IDCOL] = raw[IDCOL]
    df[GCOL] = raw[GCOL]
    df["candidate_answer"] = raw["candidate_answer"]
    df["subset"] = raw["subset"]
    df["candidate_rank"] = pd.to_numeric(raw["bge_q2q_top50_best_rank"], errors="coerce").fillna(10000)
    df["bi_rank"] = df["candidate_rank"]
    df["rerank_rank"] = pd.to_numeric(raw["exp6_rerank_rank"], errors="coerce").fillna(10000)
    df["bi_score"] = pd.to_numeric(raw["bge_q2q_top50_max_score"], errors="coerce").fillna(-1)
    df["rerank_score"] = pd.to_numeric(raw["exp6_rerank_score"], errors="coerce").fillna(-1)
    grp = df.groupby(IDCOL)["rerank_score"]
    df["rerank_score_max"] = grp.transform("max")
    # Compute second-best rerank score per row group.
    second = df.groupby(IDCOL)["rerank_score"].transform(
        lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else s.max()
    )
    df["rerank_score_second"] = second
    df["rerank_score_second"] = df["rerank_score_second"].fillna(df["rerank_score_max"])
    df["rerank_score_margin_to_best"] = df["rerank_score_max"] - df["rerank_score"]
    df["is_top1"] = (df["candidate_rank"] == 1).astype(int)
    df["val_input_len"] = pd.to_numeric(raw["query_len"], errors="coerce").fillna(0)
    df["candidate_question_len"] = pd.to_numeric(raw["candidate_question_len"], errors="coerce").fillna(0)
    df["candidate_answer_len"] = pd.to_numeric(raw["candidate_answer_len"], errors="coerce").fillna(0)
    df["query_candidate_question_jaccard"] = pd.to_numeric(raw["query_question_jaccard"], errors="coerce").fillna(0)
    df["query_candidate_answer_jaccard"] = pd.to_numeric(raw["query_answer_jaccard"], errors="coerce").fillna(0)
    return df


def apply_to_test(enc: OrdinalEncoder, model: HistGradientBoostingRegressor, gate: dict) -> dict:
    raw = pd.read_csv(TEST_FEATURES)
    test = make_test_features(raw)
    x_test, _ = prep_x(test, enc)
    test["pred_hgb"] = model.predict(x_test)
    best = choose_by_score(test, "pred_hgb")
    rerank = choose_rerank_rows(test)
    gate_rows = best[[IDCOL, "candidate_answer", "pred_hgb"]].merge(
        rerank[[IDCOL, "candidate_answer", "pred_hgb"]],
        on=IDCOL,
        suffixes=("_best", "_rerank"),
    )
    gate_rows["pred_margin_vs_rerank"] = gate_rows["pred_hgb_best"] - gate_rows["pred_hgb_rerank"]
    gate_rows["use_best"] = gate_rows["pred_margin_vs_rerank"] >= gate["pred_margin_threshold"]
    gate_rows["candidate_answer"] = np.where(
        gate_rows["use_best"], gate_rows["candidate_answer_best"], gate_rows["candidate_answer_rerank"]
    )
    choices = gate_rows[[IDCOL, "candidate_answer", "pred_margin_vs_rerank", "use_best"]].copy()
    choices.to_csv(OUT / "test_candidate_regressor_choices.csv", index=False)

    base = pd.read_csv(SUB_EXP6)
    pred_map = choices.set_index(IDCOL)["candidate_answer"].to_dict()
    sub = base.copy()
    sub["TargetR1F1"] = [pred_map.get(i, p) for i, p in zip(sub[IDCOL], sub["TargetR1F1"])]
    sub["TargetRLF1"] = sub["TargetR1F1"]
    sub["TargetLLM"] = sub["TargetR1F1"]
    sub.to_csv(SUB_OUT, index=False)
    changed = int((sub["TargetR1F1"].astype(str) != base["TargetR1F1"].astype(str)).sum())
    return {"submission": str(SUB_OUT), "changed_vs_exp6": changed, "test_gate_switch_n": int(choices["use_best"].sum())}


def main() -> None:
    _, fit_summary, fitted = train_oof()
    test_summary = apply_to_test(*fitted)
    summary = {**fit_summary, **test_summary}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
