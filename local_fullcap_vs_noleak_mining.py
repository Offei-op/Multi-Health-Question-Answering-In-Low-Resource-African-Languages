from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
REPORT_DIR = ROOT / "reports" / "fullcap_vs_noleak_mining"
FULLCAP = ROOT / "reports/cluster_aware_selector_experiment/cluster_selector_fullcap_oof_choices.csv"
NOLEAK = ROOT / "reports/cluster_aware_selector_experiment/cluster_selector_noleak_oof_choices.csv"
EXP2 = ROOT / "modal_outputs/exp2_crossencoder_rerank/val_predictions.csv"


def rouge1(pred, ref):
    pred_toks = str(pred).strip().split()
    ref_toks = str(ref).strip().split()
    if not pred_toks or not ref_toks:
        return 0.0
    overlap = sum((Counter(pred_toks) & Counter(ref_toks)).values())
    return float(2 * overlap / (len(pred_toks) + len(ref_toks))) if overlap else 0.0


def band(x, bins):
    for label, lo, hi in bins:
        if lo <= x < hi:
            return label
    return bins[-1][0]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(FULLCAP)
    n = pd.read_csv(NOLEAK)
    e = pd.read_csv(EXP2)[["ID", "rerank", "reference"]].rename(columns={"rerank": "exp2_rerank"})
    if "model" in f.columns:
        f = f[f["model"].astype(str).eq("hist_gb_fullcap_depth6")].copy()
    if "model" in n.columns:
        n = n[n["model"].astype(str).eq("hist_gb_noleak_depth6")].copy()

    keep_cols = [
        "ID",
        "subset",
        "candidate_answer",
        "target_r1",
        "candidate_answer_len",
        "candidate_question_len",
        "val_input_len",
        "query_question_jaccard",
        "query_answer_jaccard",
        "answer_freq_subset",
        "source_count",
        "rrf",
        "exp2_rerank_score",
        "exp2_rerank_rank",
        "bge200_q2q_best_rank",
        "model",
    ]
    f = f[[c for c in keep_cols if c in f.columns]].rename(columns={c: f"fullcap_{c}" for c in keep_cols if c != "ID" and c in f.columns})
    n = n[[c for c in keep_cols if c in n.columns]].rename(columns={c: f"noleak_{c}" for c in keep_cols if c != "ID" and c in n.columns})
    df = f.merge(n, on="ID", how="inner").merge(e, on="ID", how="left")
    df["subset"] = df["fullcap_subset"]
    df["fullcap_gain_vs_noleak"] = df["fullcap_target_r1"] - df["noleak_target_r1"]
    df["fullcap_gain_vs_exp2"] = df["fullcap_target_r1"] - [rouge1(p, r) for p, r in zip(df["exp2_rerank"], df["reference"])]
    df["same_answer"] = df["fullcap_candidate_answer"].astype(str) == df["noleak_candidate_answer"].astype(str)
    df["fullcap_len_ratio"] = pd.to_numeric(df["fullcap_candidate_answer_len"], errors="coerce") / pd.to_numeric(df["fullcap_val_input_len"], errors="coerce").clip(lower=1)
    df["noleak_len_ratio"] = pd.to_numeric(df["noleak_candidate_answer_len"], errors="coerce") / pd.to_numeric(df["noleak_val_input_len"], errors="coerce").clip(lower=1)
    df["fullcap_answer_len_band"] = pd.to_numeric(df["fullcap_candidate_answer_len"], errors="coerce").fillna(0).map(
        lambda x: band(x, [("000-049", 0, 50), ("050-099", 50, 100), ("100-149", 100, 150), ("150-249", 150, 250), ("250+", 250, 10**9)])
    )
    df["input_len_band"] = pd.to_numeric(df["fullcap_val_input_len"], errors="coerce").fillna(0).map(
        lambda x: band(x, [("000-014", 0, 15), ("015-024", 15, 25), ("025-039", 25, 40), ("040-059", 40, 60), ("060+", 60, 10**9)])
    )

    subset = df.groupby("subset").agg(
        rows=("ID", "count"),
        fullcap=("fullcap_target_r1", "mean"),
        noleak=("noleak_target_r1", "mean"),
        gain=("fullcap_gain_vs_noleak", "mean"),
        changed=("same_answer", lambda s: 1.0 - float(s.mean())),
    ).reset_index().sort_values("gain", ascending=False)

    len_band = df.groupby(["subset", "fullcap_answer_len_band"]).agg(
        rows=("ID", "count"),
        gain=("fullcap_gain_vs_noleak", "mean"),
        fullcap=("fullcap_target_r1", "mean"),
        noleak=("noleak_target_r1", "mean"),
    ).reset_index().sort_values(["subset", "gain"], ascending=[True, False])

    top_wins = df.sort_values("fullcap_gain_vs_noleak", ascending=False).head(300)
    top_losses = df.sort_values("fullcap_gain_vs_noleak", ascending=True).head(300)
    summary = {
        "fullcap": float(df["fullcap_target_r1"].mean()),
        "noleak": float(df["noleak_target_r1"].mean()),
        "gain": float(df["fullcap_gain_vs_noleak"].mean()),
        "same_answer_rate": float(df["same_answer"].mean()),
        "changed_answer_rate": float(1.0 - df["same_answer"].mean()),
        "largest_subset_gains": subset.head(8).to_dict(orient="records"),
        "notes": [
            "Fullcap uses validation reference length, so this is a diagnostic report only.",
            "Rows with large fullcap_gain_vs_noleak show where legal proxy features should be targeted.",
        ],
    }

    df.to_csv(REPORT_DIR / "fullcap_vs_noleak_rows.csv", index=False)
    subset.to_csv(REPORT_DIR / "subset_gain_summary.csv", index=False)
    len_band.to_csv(REPORT_DIR / "length_band_gain_summary.csv", index=False)
    top_wins.to_csv(REPORT_DIR / "top_fullcap_wins.csv", index=False)
    top_losses.to_csv(REPORT_DIR / "top_fullcap_losses.csv", index=False)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(subset.to_string(index=False))


if __name__ == "__main__":
    main()
