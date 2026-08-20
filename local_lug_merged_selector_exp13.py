from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
IN_CAND = ROOT / "modal_outputs" / "exp12_lug_e5_merge_rerank" / "lug_val_candidate_scores.csv"
IN_PREDS = ROOT / "modal_outputs" / "exp12_lug_e5_merge_rerank" / "lug_val_predictions.csv"
OUT = ROOT / "reports" / "exp13_lug_merged_selector"


def fast_r1_series(preds: pd.Series, refs: pd.Series) -> float:
    vals = []
    for pred, ref in zip(preds.fillna("").astype(str), refs.fillna("").astype(str)):
        pt = pred.strip().split()
        rt = ref.strip().split()
        if not pt or not rt:
            vals.append(0.0)
            continue
        pc = {}
        rc = {}
        for t in pt:
            pc[t] = pc.get(t, 0) + 1
        for t in rt:
            rc[t] = rc.get(t, 0) + 1
        overlap = sum(min(pc.get(t, 0), rc.get(t, 0)) for t in pc)
        if overlap == 0:
            vals.append(0.0)
            continue
        precision = overlap / len(pt)
        recall = overlap / len(rt)
        vals.append(2 * precision * recall / (precision + recall))
    return float(np.mean(vals))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["bge_rank", "e5_rank", "bge_score", "e5_score"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["has_bge"] = out["bge_rank"].notna().astype(float)
    out["has_e5"] = out["e5_rank"].notna().astype(float)
    out["both_sources"] = ((out["has_bge"] == 1) & (out["has_e5"] == 1)).astype(float)
    out["bge_rank_filled"] = out["bge_rank"].fillna(999.0)
    out["e5_rank_filled"] = out["e5_rank"].fillna(999.0)
    out["best_rank"] = out[["bge_rank_filled", "e5_rank_filled"]].min(axis=1)
    out["rank_gap"] = out["bge_rank_filled"] - out["e5_rank_filled"]
    out["bge_score_filled"] = out["bge_score"].fillna(-1.0)
    out["e5_score_filled"] = out["e5_score"].fillna(-1.0)
    out["score_gap"] = out["bge_score_filled"] - out["e5_score_filled"]
    out["candidate_answer"] = out["candidate_answer"].fillna("").astype(str)
    out["candidate_question"] = out["candidate_question"].fillna("").astype(str)
    out["answer_len"] = out["candidate_answer"].str.split().str.len().astype(float)
    out["question_len"] = out["candidate_question"].str.split().str.len().astype(float)
    out["answer_char_len"] = out["candidate_answer"].str.len().astype(float)
    out["question_char_len"] = out["candidate_question"].str.len().astype(float)
    out["rerank_score"] = pd.to_numeric(out["rerank_score"], errors="coerce").fillna(0.0)

    # Per-query relative features are test-time safe.
    group = out.groupby("ID")
    out["rerank_rank"] = group["rerank_score"].rank(method="first", ascending=False)
    out["bge_score_rank"] = group["bge_score_filled"].rank(method="first", ascending=False)
    out["e5_score_rank"] = group["e5_score_filled"].rank(method="first", ascending=False)
    out["rerank_z"] = (out["rerank_score"] - group["rerank_score"].transform("mean")) / (
        group["rerank_score"].transform("std").replace(0, np.nan)
    )
    out["rerank_z"] = out["rerank_z"].fillna(0.0)
    out["candidate_count"] = group["candidate_i"].transform("count").astype(float)
    return out


def pick_by_score(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    idx = df.groupby("ID")[score_col].idxmax()
    return df.loc[idx].copy().sort_values("ID")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv(IN_CAND)
    preds = pd.read_csv(IN_PREDS)
    cand = add_features(cand)

    feature_cols = [
        "rerank_score",
        "rerank_rank",
        "rerank_z",
        "bge_rank_filled",
        "e5_rank_filled",
        "best_rank",
        "rank_gap",
        "bge_score_filled",
        "e5_score_filled",
        "score_gap",
        "has_bge",
        "has_e5",
        "both_sources",
        "answer_len",
        "question_len",
        "answer_char_len",
        "question_char_len",
        "bge_score_rank",
        "e5_score_rank",
        "candidate_count",
    ]

    models = {
        "hgb_l2": HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.035,
            max_iter=350,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=42,
        ),
        "hgb_abs": HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.04,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            random_state=43,
        ),
        "rf": RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=4,
            max_features=0.65,
            n_jobs=-1,
            random_state=44,
        ),
        "ridge": Ridge(alpha=2.0),
    }

    x = cand[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y = cand["rouge1"].astype(float).to_numpy()
    groups = cand["ID"].to_numpy()
    gkf = GroupKFold(n_splits=5)

    oof_scores = {}
    model_rows = []
    for name, model in models.items():
        pred = np.zeros(len(cand), dtype=np.float32)
        fold_mse = []
        for tr, va in gkf.split(x, y, groups):
            model.fit(x.iloc[tr], y[tr])
            p = model.predict(x.iloc[va])
            pred[va] = p
            fold_mse.append(mean_squared_error(y[va], p))
        cand[f"pred_{name}"] = pred
        chosen = pick_by_score(cand, f"pred_{name}")
        score = fast_r1_series(chosen["candidate_answer"], preds.set_index("ID").loc[chosen["ID"], "ref"].reset_index(drop=True))
        oof_scores[name] = {
            "oof_mse": float(np.mean(fold_mse)),
            "chosen_rouge1": score,
            "changed_vs_exp12_rerank": int((chosen["candidate_answer"].to_numpy() != preds.set_index("ID").loc[chosen["ID"], "merged_rerank"].to_numpy()).sum()),
        }
        model_rows.append((name, score))

    pred_cols = [f"pred_{n}" for n in models]
    cand["pred_mean"] = cand[pred_cols].mean(axis=1)
    cand["pred_maxblend"] = 0.72 * cand["pred_mean"] + 0.28 * cand["rerank_score"]
    for score_col in ["pred_mean", "pred_maxblend"]:
        chosen = pick_by_score(cand, score_col)
        oof_scores[score_col] = {
            "chosen_rouge1": fast_r1_series(
                chosen["candidate_answer"], preds.set_index("ID").loc[chosen["ID"], "ref"].reset_index(drop=True)
            ),
            "changed_vs_exp12_rerank": int((chosen["candidate_answer"].to_numpy() != preds.set_index("ID").loc[chosen["ID"], "merged_rerank"].to_numpy()).sum()),
        }

    baseline = {
        "bge_ft_top1": fast_r1_series(preds["bge_top"], preds["ref"]),
        "e5_large_top1": fast_r1_series(preds["e5_top"], preds["ref"]),
        "exp12_merged_rerank": fast_r1_series(preds["merged_rerank"], preds["ref"]),
        "merged_oracle": fast_r1_series(preds["merged_oracle"], preds["ref"]),
    }

    best_name = max(oof_scores, key=lambda k: oof_scores[k]["chosen_rouge1"])
    best_col = best_name if best_name.startswith("pred_") else f"pred_{best_name}"
    if best_name in ("pred_mean", "pred_maxblend"):
        best_col = best_name
    best = pick_by_score(cand, best_col)
    choices = preds.merge(
        best[["ID", "candidate_answer", "sources", "bge_rank", "e5_rank", "rerank_score", "rouge1", best_col]],
        on="ID",
        how="left",
    )
    choices = choices.rename(columns={"candidate_answer": "selector_answer", "rouge1": "selector_true_rouge1", best_col: "selector_score"})
    choices.to_csv(OUT / "oof_best_selector_choices.csv", index=False)
    cand.to_csv(OUT / "candidate_scores_with_oof_predictions.csv", index=False)

    summary = {
        "experiment": "exp13_lug_merged_selector",
        "input_candidates": str(IN_CAND),
        "rows": int(len(cand)),
        "queries": int(cand["ID"].nunique()),
        "features": feature_cols,
        "baseline": baseline,
        "oof_scores": oof_scores,
        "best_selector": best_name,
        "best_selector_rouge1": oof_scores[best_name]["chosen_rouge1"],
        "delta_vs_exp12_merged_rerank": float(oof_scores[best_name]["chosen_rouge1"] - baseline["exp12_merged_rerank"]),
        "remaining_gap_to_oracle": float(baseline["merged_oracle"] - oof_scores[best_name]["chosen_rouge1"]),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
