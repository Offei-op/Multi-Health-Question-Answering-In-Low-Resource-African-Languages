from pathlib import Path

import modal


APP_NAME = "lalang-exp2-candidate-feature-export"
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
def export_candidate_features():
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

    def toks(text):
        return str(text).strip().split()

    def rouge1(pred, ref):
        pred_toks = toks(pred)
        ref_toks = toks(ref)
        if not pred_toks or not ref_toks:
            return 0.0
        overlap = sum((Counter(pred_toks) & Counter(ref_toks)).values())
        if overlap == 0:
            return 0.0
        return float(2.0 * overlap / (len(pred_toks) + len(ref_toks)))

    def jaccard(a, b):
        aa = set(toks(a.lower()))
        bb = set(toks(b.lower()))
        if not aa or not bb:
            return 0.0
        return float(len(aa & bb) / len(aa | bb))

    print("Computing labels/features...", flush=True)
    cand["candidate_r1"] = [
        rouge1(a, r) for a, r in tqdm(zip(cand["candidate_answer"], cand["reference"]), total=len(cand))
    ]
    cand["val_input_len"] = cand["val_input"].map(lambda x: len(toks(x)))
    cand["reference_len"] = cand["reference"].map(lambda x: len(toks(x)))
    cand["candidate_question_len"] = cand["candidate_question"].map(lambda x: len(toks(x)))
    cand["candidate_answer_len"] = cand["candidate_answer"].map(lambda x: len(toks(x)))
    cand["query_candidate_question_jaccard"] = [
        jaccard(q, cq) for q, cq in tqdm(zip(cand["val_input"], cand["candidate_question"]), total=len(cand))
    ]
    cand["query_candidate_answer_jaccard"] = [
        jaccard(q, ca) for q, ca in tqdm(zip(cand["val_input"], cand["candidate_answer"]), total=len(cand))
    ]
    cand["answer_ref_len_ratio"] = cand["candidate_answer_len"] / cand["reference_len"].replace(0, np.nan)

    cand["rerank_rank"] = cand.groupby("ID")["rerank_score"].rank(method="first", ascending=False).astype(int)
    cand["bi_rank"] = cand["candidate_rank"].astype(int)
    cand["rerank_score_max"] = cand.groupby("ID")["rerank_score"].transform("max")
    cand["rerank_score_second"] = (
        cand.sort_values(["ID", "rerank_score"], ascending=[True, False])
        .groupby("ID")["rerank_score"]
        .transform(lambda s: s.iloc[1] if len(s) > 1 else s.iloc[0])
    )
    cand["rerank_score_margin_to_best"] = cand["rerank_score_max"] - cand["rerank_score"]
    cand["is_rerank_choice"] = (cand["rerank_rank"] == 1).astype(int)
    cand["is_top1"] = (cand["candidate_rank"] == 1).astype(int)
    cand["is_oracle"] = cand.groupby("ID")["candidate_r1"].transform(lambda s: s == s.max()).astype(int)

    cols = [
        "ID",
        "subset",
        "candidate_rank",
        "bi_rank",
        "rerank_rank",
        "bi_score",
        "rerank_score",
        "rerank_score_max",
        "rerank_score_second",
        "rerank_score_margin_to_best",
        "is_rerank_choice",
        "is_top1",
        "is_oracle",
        "candidate_r1",
        "val_input_len",
        "reference_len",
        "candidate_question_len",
        "candidate_answer_len",
        "answer_ref_len_ratio",
        "query_candidate_question_jaccard",
        "query_candidate_answer_jaccard",
    ]
    out = cand[cols].copy()
    out_path = out_dir / "candidate_tabular_features.csv"
    out.to_csv(out_path, index=False)
    summary = {
        "rows": int(len(out)),
        "ids": int(out["ID"].nunique()),
        "path": str(out_path),
        "base_rerank_r1": float(out.loc[out["is_rerank_choice"] == 1, "candidate_r1"].mean()),
        "base_top1_r1": float(out.loc[out["is_top1"] == 1, "candidate_r1"].mean()),
        "oracle_r1": float(out.groupby("ID")["candidate_r1"].max().mean()),
    }
    (out_dir / "candidate_tabular_features_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main():
    print(export_candidate_features.remote())
