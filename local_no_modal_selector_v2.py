from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
OUT = ROOT / "reports" / "local_no_modal_selector_v2"
OUT.mkdir(parents=True, exist_ok=True)

EXP2_VAL = ROOT / "modal_outputs" / "exp2_crossencoder_rerank" / "val_predictions.csv"
NOLEAK_CHOICES = ROOT / "reports" / "cluster_aware_selector_experiment" / "cluster_selector_noleak_oof_choices.csv"
EXP7_TEST = ROOT / "modal_outputs" / "exp7_cluster_selector_test_predictions_files"
SUB_EXP6 = ROOT / "modal_outputs" / "exp7_submissions" / "submission_exp6_reranker_trainval_top50cluster.csv"
SUB_EXP7_CLUSTER = ROOT / "modal_outputs" / "exp7_submissions" / "submission_exp7_cluster_selector_trainval.csv"
SUB_EXP7_DENSE = ROOT / "modal_outputs" / "exp7_submissions" / "submission_exp7_dense_top1_trainval.csv"
TEST_SELECTOR_CHOICES = EXP7_TEST / "test_selector_choices.csv"
TEST_RERANKER_CHOICES = EXP7_TEST / "test_reranker_choices.csv"
TEST_CANDIDATES = EXP7_TEST / "test_candidate_features_scored.csv"
TEST = ROOT / "Test.csv"


def tokens(x: object) -> list[str]:
    return str(x).strip().split()


def rouge1_f1(pred: object, ref: object) -> float:
    p = tokens(pred)
    r = tokens(ref)
    if not p or not r:
        return 0.0
    from collections import Counter

    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(p)
    rec = overlap / len(r)
    return float(2 * prec * rec / (prec + rec))


def score_frame(df: pd.DataFrame, pred_col: str = "pred") -> dict:
    rows = []
    for subset, p, ref in zip(df["subset"], df[pred_col], df["reference"]):
        rows.append((subset, rouge1_f1(p, ref)))
    s = pd.DataFrame(rows, columns=["subset", "r1"])
    return {
        "rouge1": float(s["r1"].mean()),
        "per_subset": s.groupby("subset")["r1"].mean().round(6).to_dict(),
    }


def load_val_base() -> pd.DataFrame:
    val = pd.read_csv(EXP2_VAL)
    val["rerank_r1"] = [rouge1_f1(p, r) for p, r in zip(val["rerank"], val["reference"])]
    val["top1_r1"] = [rouge1_f1(p, r) for p, r in zip(val["top1"], val["reference"])]
    return val


def load_choice_model(model: str = "lightgbm_noleak") -> pd.DataFrame:
    usecols = [
        "ID",
        "subset",
        "candidate_answer",
        "target_r1",
        "pred",
        "model",
        "source_count",
        "best_any_rank",
        "max_any_score",
        "rrf",
        "exp2_candidate_rank",
        "exp2_bi_score",
        "exp2_rerank_score",
        "exp2_rerank_rank",
        "bge200_q2q_max_score",
        "query_question_jaccard",
        "query_answer_jaccard",
        "candidate_answer_len",
        "candidate_question_len",
        "val_input_len",
        "_fullcap_pool_score",
    ]
    df = pd.read_csv(NOLEAK_CHOICES, usecols=usecols)
    return df[df["model"].eq(model)].copy()


def eval_cluster_gates() -> tuple[pd.DataFrame, pd.DataFrame]:
    val = load_val_base()
    choice = load_choice_model("lightgbm_noleak")
    m = val[["ID", "subset", "reference", "rerank", "rerank_r1", "top1", "top1_r1"]].merge(choice, on=["ID", "subset"])
    m["choice_gain"] = m["target_r1"] - m["rerank_r1"]
    m["pred_minus_rerank_score"] = m["pred"] - m["exp2_rerank_score"].replace(-1, np.nan)
    m["choice_is_exp2_pool"] = m["exp2_rerank_rank"].lt(10000).astype(int)
    base_score = float(m["rerank_r1"].mean())

    records = []
    # A compact but fairly expressive threshold grid. The goal is a tiny gate, not a broad replacement.
    for pred_t in np.round(np.arange(0.28, 0.62, 0.02), 3):
        for max_rank in [1, 2, 3, 5, 10, 25, 10000]:
            for min_sources in [1, 2, 3, 4]:
                for require_exp2 in [0, 1]:
                    mask = (m["pred"] >= pred_t) & (m["best_any_rank"] <= max_rank) & (m["source_count"] >= min_sources)
                    if require_exp2:
                        mask &= m["choice_is_exp2_pool"].eq(1)
                    if mask.sum() == 0:
                        continue
                    score = base_score + float(m.loc[mask, "choice_gain"].sum()) / len(m)
                    records.append(
                        {
                            "pred_t": pred_t,
                            "max_rank": max_rank,
                            "min_sources": min_sources,
                            "require_exp2_pool": require_exp2,
                            "switch_n": int(mask.sum()),
                            "score": score,
                            "gain": score - float(m["rerank_r1"].mean()),
                            "mean_actual_gain_on_switched": float(m.loc[mask, "choice_gain"].mean()),
                        }
                    )
    lb = pd.DataFrame(records).sort_values(["score", "switch_n"], ascending=[False, True]).reset_index(drop=True)
    lb.to_csv(OUT / "cluster_gate_leaderboard.csv", index=False)
    m.to_csv(OUT / "cluster_gate_val_rows.csv", index=False)
    return lb, m


def apply_cluster_gate(rule: dict) -> Path:
    base = pd.read_csv(SUB_EXP6)
    cluster = pd.read_csv(SUB_EXP7_CLUSTER)
    choices = pd.read_csv(TEST_SELECTOR_CHOICES)
    choices["choice_is_exp6_pool"] = choices["exp6_rerank_rank"].lt(10000).astype(int)

    mask = (
        (choices["selector_pred"] >= rule["pred_t"])
        & (choices["best_any_rank"] <= rule["max_rank"])
        & (choices["source_count"] >= rule["min_sources"])
    )
    if rule["require_exp2_pool"]:
        mask &= choices["choice_is_exp6_pool"].eq(1)
    switched_ids = set(choices.loc[mask, "ID"])

    out = base.copy()
    cluster_map = cluster.set_index("ID")["TargetR1F1"].to_dict()
    out["TargetR1F1"] = [cluster_map.get(i, a) if i in switched_ids else a for i, a in zip(out["ID"], out["TargetR1F1"])]
    out["TargetRLF1"] = out["TargetR1F1"]
    out["TargetLLM"] = out["TargetR1F1"]

    name = (
        f"submission_local_cluster_gate_pred{rule['pred_t']}_rank{rule['max_rank']}"
        f"_src{rule['min_sources']}_exp2{rule['require_exp2_pool']}.csv"
    ).replace(".", "p")
    path = ROOT / "modal_outputs" / "exp7_submissions" / name
    out.to_csv(path, index=False)

    pd.DataFrame({"ID": sorted(switched_ids)}).to_csv(OUT / "test_switched_ids.csv", index=False)
    return path


def submission_diff(a: Path, b: Path) -> int:
    da = pd.read_csv(a)
    db = pd.read_csv(b)
    return int((da["TargetR1F1"].astype(str) != db["TargetR1F1"].astype(str)).sum())


def main() -> None:
    lb, val_rows = eval_cluster_gates()
    base_score = float(val_rows["rerank_r1"].mean())
    top = lb.iloc[0].to_dict()
    # If the best learned gate is negative, still save only the leaderboard/report, no fake-hero submission.
    made_submission = None
    if top["gain"] > 0:
        made_submission = str(apply_cluster_gate(top))

    summary = {
        "base_exp2_rerank_val_r1": base_score,
        "best_gate": top,
        "made_submission": made_submission,
        "diff_vs_exp6_reranker": submission_diff(Path(made_submission), SUB_EXP6) if made_submission else None,
        "diff_vs_exp7_cluster": submission_diff(Path(made_submission), SUB_EXP7_CLUSTER) if made_submission else None,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Outputs saved to {OUT}")


if __name__ == "__main__":
    main()
