from pathlib import Path

import modal


APP_NAME = "lalang-lug-e5-merge-rerank-exp12"
VOLUME_NAME = "lalang-bgem3-rerank"
REMOTE_ROOT = Path("/data")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "sentence-transformers>=5.1.0",
        "transformers>=4.46.0",
        "peft>=0.12.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.5.0",
        "tqdm>=4.66.0",
        "safetensors>=0.4.3",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 3,
    volumes={str(REMOTE_ROOT): volume},
)
def run(k_bge: int = 50, k_e5: int = 50, batch_size: int = 64, max_seq_length: int = 256):
    import gc
    import json
    import time
    from collections import Counter

    import numpy as np
    import pandas as pd
    import torch
    from peft import PeftModel
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from sklearn.neighbors import NearestNeighbors
    from tqdm.auto import tqdm

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    qcol, acol, gcol, idcol = "input", "output", "subset", "ID"
    subset = "Lug_Uga"
    out_dir = REMOTE_ROOT / "exp12_lug_e5_merge_rerank"
    out_dir.mkdir(parents=True, exist_ok=True)

    def fast_r1(pred, ref):
        pred_toks = str(pred).strip().split()
        ref_toks = str(ref).strip().split()
        if not pred_toks or not ref_toks:
            return 0.0
        pc = Counter(pred_toks)
        rc = Counter(ref_toks)
        overlap = sum(min(pc[t], rc[t]) for t in pc)
        if overlap == 0:
            return 0.0
        precision = overlap / len(pred_toks)
        recall = overlap / len(ref_toks)
        return float(2 * precision * recall / (precision + recall))

    def score_predictions(rows, pred_col, label):
        vals = [fast_r1(r[pred_col], r["ref"]) for r in rows]
        return {
            "label": label,
            "rouge1": float(np.mean(vals)) if vals else 0.0,
            "count": int(len(vals)),
        }

    def candidate_text(q, a):
        return f"Candidate question: {q}\nCandidate answer: {a}"

    def clean(df):
        for c in (idcol, qcol, acol, gcol):
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str).str.strip()
        return df[(df[qcol] != "") & (df[acol] != "") & (df[gcol] != "")].reset_index(drop=True)

    t0 = time.time()
    print("Reading data...", flush=True)
    train = clean(pd.read_csv(REMOTE_ROOT / "Train.csv"))
    val = clean(pd.read_csv(REMOTE_ROOT / "Val.csv"))
    train = train[train[gcol] == subset].reset_index(drop=True)
    val = val[val[gcol] == subset].reset_index(drop=True)
    print(f"{subset}: train={len(train):,} val={len(val):,}", flush=True)

    train_q = train[qcol].tolist()
    train_a = train[acol].tolist()
    val_q = val[qcol].tolist()
    val_a = val[acol].tolist()
    val_ids = val[idcol].tolist()

    def retrieve_with_model(label, model, corpus_texts, query_texts, k):
        print(f"\nRetrieving with {label} k={k}...", flush=True)
        corpus_embs = model.encode(
            corpus_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=128,
            convert_to_numpy=True,
        )
        nn = NearestNeighbors(n_neighbors=min(k, len(train)), metric="cosine").fit(corpus_embs)
        query_embs = model.encode(
            query_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=128,
            convert_to_numpy=True,
        )
        dists, idx_mat = nn.kneighbors(query_embs, n_neighbors=min(k, len(train)))
        out = []
        for drow, irow in zip(dists, idx_mat):
            cands = []
            for rank, (dist, j) in enumerate(zip(drow, irow), start=1):
                cands.append(
                    {
                        "source": label,
                        "source_rank": rank,
                        "source_score": float(1.0 - dist),
                        "train_idx": int(j),
                        "candidate_question": train_q[int(j)],
                        "candidate_answer": train_a[int(j)],
                    }
                )
            out.append(cands)
        return out

    print("\nLoading fine-tuned BGE-M3 encoder adapter...", flush=True)
    bge = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
    bge.max_seq_length = max_seq_length
    inner = bge[0].auto_model
    bge[0].auto_model = PeftModel.from_pretrained(inner, str(REMOTE_ROOT / "bge_m3_adapter"), is_trainable=False)
    bge[0].auto_model.eval()
    bge_cands = retrieve_with_model("bge_ft", bge, train_q, val_q, k_bge)
    try:
        bge.model.cpu()
    except Exception:
        pass
    del bge
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nLoading E5-large base encoder...", flush=True)
    e5 = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda" if torch.cuda.is_available() else "cpu")
    e5.max_seq_length = max_seq_length
    e5_cands = retrieve_with_model(
        "e5_large",
        e5,
        ["passage: " + q for q in train_q],
        ["query: " + q for q in val_q],
        k_e5,
    )
    try:
        e5.model.cpu()
    except Exception:
        pass
    del e5
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nMerging candidates by answer text...", flush=True)
    merged = []
    debug_rows = []
    for row_i, (bid, q, ref, bc, ec) in enumerate(zip(val_ids, val_q, val_a, bge_cands, e5_cands)):
        by_answer = {}
        for c in bc + ec:
            key = c["candidate_answer"].strip()
            if not key:
                continue
            rec = by_answer.setdefault(
                key,
                {
                    "candidate_answer": c["candidate_answer"],
                    "candidate_question": c["candidate_question"],
                    "bge_rank": None,
                    "bge_score": None,
                    "e5_rank": None,
                    "e5_score": None,
                    "sources": set(),
                },
            )
            rec["sources"].add(c["source"])
            if c["source"] == "bge_ft" and (rec["bge_rank"] is None or c["source_rank"] < rec["bge_rank"]):
                rec["bge_rank"] = c["source_rank"]
                rec["bge_score"] = c["source_score"]
                rec["candidate_question"] = c["candidate_question"]
            if c["source"] == "e5_large" and (rec["e5_rank"] is None or c["source_rank"] < rec["e5_rank"]):
                rec["e5_rank"] = c["source_rank"]
                rec["e5_score"] = c["source_score"]
                if rec["bge_rank"] is None:
                    rec["candidate_question"] = c["candidate_question"]
        cs = list(by_answer.values())
        cs.sort(key=lambda x: min(x["bge_rank"] or 9999, x["e5_rank"] or 9999))
        for c in cs:
            c["sources"] = "+".join(sorted(c["sources"]))
        merged.append(cs)

    print("\nLoading current best exp2 reranker...", flush=True)
    reranker = CrossEncoder(
        str(REMOTE_ROOT / "exp2_crossencoder_rerank" / "final"),
        max_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    flat_pairs, row_lens = [], []
    for q, cs in zip(val_q, merged):
        row_lens.append(len(cs))
        flat_pairs.extend([(q, candidate_text(c["candidate_question"], c["candidate_answer"])) for c in cs])
    print(f"Scoring {len(flat_pairs):,} merged candidates...", flush=True)
    scores = reranker.predict(flat_pairs, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    rows = []
    off = 0
    for i, (bid, q, ref, bc, ec, cs, n) in enumerate(zip(val_ids, val_q, val_a, bge_cands, e5_cands, merged, row_lens)):
        row_scores = np.asarray(scores[off : off + n], dtype=np.float32)
        off += n
        best_i = int(np.argmax(row_scores)) if n else 0
        bge_top = bc[0]["candidate_answer"] if bc else ""
        e5_top = ec[0]["candidate_answer"] if ec else ""
        merged_pred = cs[best_i]["candidate_answer"] if cs else ""
        oracle_i = 0
        oracle_r1 = 0.0
        for ci, c in enumerate(cs):
            r1 = fast_r1(c["candidate_answer"], ref)
            if r1 > oracle_r1:
                oracle_r1 = r1
                oracle_i = ci
        oracle_pred = cs[oracle_i]["candidate_answer"] if cs else ""
        rows.append(
            {
                "ID": bid,
                "query": q,
                "ref": ref,
                "bge_top": bge_top,
                "e5_top": e5_top,
                "merged_rerank": merged_pred,
                "merged_oracle": oracle_pred,
                "merged_candidate_count": n,
                "chosen_sources": cs[best_i]["sources"] if cs else "",
                "oracle_sources": cs[oracle_i]["sources"] if cs else "",
                "chosen_rerank_score": float(row_scores[best_i]) if n else 0.0,
                "oracle_r1": float(oracle_r1),
            }
        )
        for ci, c in enumerate(cs):
            debug_rows.append(
                {
                    "ID": bid,
                    "candidate_i": ci,
                    "bge_rank": c["bge_rank"],
                    "bge_score": c["bge_score"],
                    "e5_rank": c["e5_rank"],
                    "e5_score": c["e5_score"],
                    "sources": c["sources"],
                    "rerank_score": float(row_scores[ci]) if n else 0.0,
                    "rouge1": fast_r1(c["candidate_answer"], ref),
                    "chosen": ci == best_i,
                    "oracle": ci == oracle_i,
                    "candidate_question": c["candidate_question"],
                    "candidate_answer": c["candidate_answer"],
                }
            )

    metrics = {
        "bge_ft_top1": score_predictions(rows, "bge_top", "bge_ft_top1"),
        "e5_large_top1": score_predictions(rows, "e5_top", "e5_large_top1"),
        "merged_exp2_rerank": score_predictions(rows, "merged_rerank", "merged_exp2_rerank"),
        "merged_oracle": score_predictions(rows, "merged_oracle", "merged_oracle"),
    }
    summary = {
        "experiment": "exp12_lug_e5_merge_rerank",
        "subset": subset,
        "gpu": "L40S",
        "k_bge": k_bge,
        "k_e5": k_e5,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "seconds": float(time.time() - t0),
        "metrics": metrics,
        "delta_merged_rerank_vs_bge_top1": float(metrics["merged_exp2_rerank"]["rouge1"] - metrics["bge_ft_top1"]["rouge1"]),
        "delta_merged_oracle_vs_bge_top1": float(metrics["merged_oracle"]["rouge1"] - metrics["bge_ft_top1"]["rouge1"]),
    }

    pd.DataFrame(rows).to_csv(out_dir / "lug_val_predictions.csv", index=False)
    pd.DataFrame(debug_rows).to_csv(out_dir / "lug_val_candidate_scores.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(k_bge: int = 50, k_e5: int = 50, batch_size: int = 64, max_seq_length: int = 256):
    run.remote(k_bge=k_bge, k_e5=k_e5, batch_size=batch_size, max_seq_length=max_seq_length)
