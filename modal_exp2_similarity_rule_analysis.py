from pathlib import Path

import modal


APP_NAME = "lalang-exp2-similarity-rule-analysis"
VOLUME_NAME = "lalang-bgem3-rerank"
REMOTE_ROOT = Path("/data")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "tqdm>=4.66.0",
)


@app.function(image=image, timeout=60 * 30, volumes={str(REMOTE_ROOT): volume})
def analyze_similarity_rules():
    import json
    from collections import Counter

    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm

    in_path = REMOTE_ROOT / "exp2_val_candidate_scores" / "val_candidate_scores.csv"
    out_dir = REMOTE_ROOT / "exp2_val_candidate_scores"
    print(f"Reading {in_path}...", flush=True)
    cand = pd.read_csv(in_path)
    print(f"candidate rows={len(cand):,}", flush=True)

    def rouge1(pred, ref):
        pred_toks = str(pred).strip().split()
        ref_toks = str(ref).strip().split()
        if not pred_toks or not ref_toks:
            return 0.0
        overlap = sum((Counter(pred_toks) & Counter(ref_toks)).values())
        if overlap == 0:
            return 0.0
        return float(2.0 * overlap / (len(pred_toks) + len(ref_toks)))

    print("Scoring candidates with fast whitespace ROUGE-1...", flush=True)
    cand["candidate_r1"] = [
        rouge1(a, r) for a, r in tqdm(zip(cand["candidate_answer"], cand["reference"]), total=len(cand))
    ]

    top1 = cand[cand["candidate_rank"] == 1][
        ["ID", "subset", "bi_score", "rerank_score", "candidate_r1", "candidate_answer"]
    ].rename(
        columns={
            "bi_score": "top1_bi_score",
            "rerank_score": "top1_rerank_score",
            "candidate_r1": "top1_r1",
            "candidate_answer": "top1_answer",
        }
    )
    chosen_idx = cand.groupby("ID")["rerank_score"].idxmax()
    chosen = cand.loc[chosen_idx][
        ["ID", "candidate_rank", "bi_score", "rerank_score", "candidate_r1", "candidate_answer"]
    ].rename(
        columns={
            "candidate_rank": "chosen_rank",
            "bi_score": "chosen_bi_score",
            "rerank_score": "chosen_rerank_score",
            "candidate_r1": "chosen_r1",
            "candidate_answer": "chosen_answer",
        }
    )
    oracle_idx = cand.groupby("ID")["candidate_r1"].idxmax()
    oracle = cand.loc[oracle_idx][["ID", "candidate_rank", "candidate_r1"]].rename(
        columns={"candidate_rank": "oracle_rank", "candidate_r1": "oracle_r1"}
    )
    top2 = (
        cand.sort_values(["ID", "rerank_score"], ascending=[True, False])
        .groupby("ID")
        .head(2)
        .groupby("ID")["rerank_score"]
        .agg(["first", "last"])
        .reset_index()
    )
    top2["rerank_margin"] = top2["first"] - top2["last"]

    rows = top1.merge(chosen, on="ID").merge(oracle, on="ID").merge(top2[["ID", "rerank_margin"]], on="ID")
    rows["delta_chosen_vs_top1"] = rows["chosen_r1"] - rows["top1_r1"]
    rows["oracle_gap_after_chosen"] = rows["oracle_r1"] - rows["chosen_r1"]
    rows["bi_gap_top1_minus_chosen"] = rows["top1_bi_score"] - rows["chosen_bi_score"]
    rows["reranker_jump"] = rows["chosen_rank"] > 1
    base = float(rows["chosen_r1"].mean())
    top1_base = float(rows["top1_r1"].mean())
    oracle_base = float(rows["oracle_r1"].mean())

    rules = []

    def add_rule(name, trigger):
        trigger = np.asarray(trigger, dtype=bool)
        if int(trigger.sum()) < 10:
            return
        score = rows["chosen_r1"].to_numpy().copy()
        score[trigger] = rows["top1_r1"].to_numpy()[trigger]
        rules.append(
            {
                "rule": name,
                "triggered": int(trigger.sum()),
                "score": float(score.mean()),
                "gain_vs_chosen": float(score.mean() - base),
                "trigger_delta_chosen_vs_top1": float(rows.loc[trigger, "delta_chosen_vs_top1"].mean()),
                "trigger_hurt_pct": float((rows.loc[trigger, "delta_chosen_vs_top1"] < 0).mean()),
                "trigger_win_pct": float((rows.loc[trigger, "delta_chosen_vs_top1"] > 0).mean()),
                "trigger_top1_r1": float(rows.loc[trigger, "top1_r1"].mean()),
                "trigger_chosen_r1": float(rows.loc[trigger, "chosen_r1"].mean()),
                "trigger_oracle_r1": float(rows.loc[trigger, "oracle_r1"].mean()),
            }
        )

    top1_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97]
    chosen_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    gap_thresholds = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
    margin_thresholds = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
    rank_thresholds = [2, 3, 5, 10, 20]

    for scope in ["__ALL__"] + sorted(rows["subset"].unique()):
        sm = np.ones(len(rows), dtype=bool) if scope == "__ALL__" else (rows["subset"].to_numpy() == scope)
        jump = sm & rows["reranker_jump"].to_numpy()
        for t in top1_thresholds:
            add_rule(f"{scope} jump & top1_bi>={t}", jump & (rows["top1_bi_score"].to_numpy() >= t))
        for t in chosen_thresholds:
            add_rule(f"{scope} jump & chosen_bi<={t}", jump & (rows["chosen_bi_score"].to_numpy() <= t))
        for g in gap_thresholds:
            add_rule(f"{scope} jump & bi_gap>={g}", jump & (rows["bi_gap_top1_minus_chosen"].to_numpy() >= g))
        for r in rank_thresholds:
            add_rule(f"{scope} chosen_rank>={r}", sm & (rows["chosen_rank"].to_numpy() >= r))
        for t in top1_thresholds:
            for g in gap_thresholds:
                add_rule(
                    f"{scope} jump & top1_bi>={t} & bi_gap>={g}",
                    jump
                    & (rows["top1_bi_score"].to_numpy() >= t)
                    & (rows["bi_gap_top1_minus_chosen"].to_numpy() >= g),
                )
        for t in top1_thresholds:
            for m in margin_thresholds:
                add_rule(
                    f"{scope} jump & top1_bi>={t} & margin<={m}",
                    jump
                    & (rows["top1_bi_score"].to_numpy() >= t)
                    & (rows["rerank_margin"].to_numpy() <= m),
                )

    rules_df = pd.DataFrame(rules).sort_values("gain_vs_chosen", ascending=False)

    summary = {
        "experiment": "exp2_similarity_rule_analysis",
        "rows": int(len(rows)),
        "base_chosen_r1_fast": base,
        "top1_r1_fast": top1_base,
        "oracle_r1_fast": oracle_base,
        "best_rule": rules_df.iloc[0].to_dict() if len(rules_df) else None,
    }

    rows.to_csv(out_dir / "val_similarity_rule_rows.csv", index=False)
    rules_df.to_csv(out_dir / "similarity_fallback_rules.csv", index=False)
    (out_dir / "similarity_rule_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main():
    print(analyze_similarity_rules.remote())
