from pathlib import Path

import modal


APP_NAME = "lalang-exp2-val-candidate-scores"
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
def score_val_candidates(k: int = 50, batch_size: int = 64):
    import json

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
    out_dir = REMOTE_ROOT / "exp2_val_candidate_scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading Train/Val...", flush=True)
    train = pd.read_csv(REMOTE_ROOT / "Train.csv")
    val = pd.read_csv(REMOTE_ROOT / "Val.csv")
    for df in (train, val):
        for c in (idcol, qcol, gcol):
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str).str.strip()
        df[acol] = df[acol].fillna("").astype(str).str.strip()
    train = train[(train[qcol] != "") & (train[acol] != "") & (train[gcol] != "")].reset_index(drop=True)
    val = val[(val[qcol] != "") & (val[acol] != "") & (val[gcol] != "")].reset_index(drop=True)
    print(f"train={len(train):,} val={len(val):,}", flush=True)

    def candidate_text(q, a):
        return f"Candidate question: {q}\nCandidate answer: {a}"

    print("\nLoading fine-tuned BGE-M3 adapter...", flush=True)
    bi = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
    bi.max_seq_length = 256
    inner = bi[0].auto_model
    bi[0].auto_model = PeftModel.from_pretrained(inner, str(REMOTE_ROOT / "bge_m3_adapter"), is_trainable=False)
    bi[0].auto_model.eval()

    print("\nEncoding train indices per subset...", flush=True)
    indices = {}
    for subset, grp in tqdm(list(train.groupby(gcol)), desc="Index train"):
        embs = bi.encode(
            grp[qcol].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=128,
            convert_to_numpy=True,
        )
        nn = NearestNeighbors(n_neighbors=min(k, len(grp)), metric="cosine").fit(embs)
        indices[subset] = {
            "nn": nn,
            "q": np.array(grp[qcol].astype(str).tolist(), dtype=object),
            "a": np.array(grp[acol].astype(str).tolist(), dtype=object),
        }

    print("\nRetrieving val topK per subset...", flush=True)
    rows = []
    flat_pairs = []
    for subset, grp in tqdm(list(val.groupby(gcol)), desc="Retrieve val"):
        m = indices[subset]
        q_embs = bi.encode(
            grp[qcol].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=128,
            convert_to_numpy=True,
        )
        n_neighbors = min(k, len(m["a"]))
        distances, idx_mat = m["nn"].kneighbors(q_embs, n_neighbors=n_neighbors)
        for row, dists, idxs in zip(grp.itertuples(index=False), distances, idx_mat):
            val_id = getattr(row, idcol)
            val_q = getattr(row, qcol)
            val_ref = getattr(row, acol)
            for rank, (dist, j) in enumerate(zip(dists, idxs), start=1):
                cand_q = str(m["q"][j])
                cand_a = str(m["a"][j])
                rows.append(
                    {
                        "ID": val_id,
                        "subset": subset,
                        "val_input": val_q,
                        "reference": val_ref,
                        "candidate_rank": rank,
                        "bi_score": float(1.0 - dist),
                        "candidate_question": cand_q,
                        "candidate_answer": cand_a,
                    }
                )
                flat_pairs.append((val_q, candidate_text(cand_q, cand_a)))

    print(f"candidate rows={len(rows):,}", flush=True)
    print("\nFreeing encoder before reranker scoring...", flush=True)
    try:
        bi.model.cpu()
    except Exception:
        pass
    del bi
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nLoading exp2 trained reranker...", flush=True)
    reranker = CrossEncoder(
        str(REMOTE_ROOT / "exp2_crossencoder_rerank" / "final"),
        max_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\nScoring validation candidates...", flush=True)
    scores = reranker.predict(flat_pairs, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    out = pd.DataFrame(rows)
    out["rerank_score"] = np.asarray(scores, dtype=np.float32)
    out["rerank_rank"] = out.groupby("ID")["rerank_score"].rank(method="first", ascending=False).astype(int)

    best = out.loc[out.groupby("ID")["rerank_score"].idxmax()].copy()
    best = best[["ID", "subset", "candidate_rank", "bi_score", "rerank_score", "candidate_answer"]].rename(
        columns={
            "candidate_rank": "chosen_candidate_rank",
            "bi_score": "chosen_bi_score",
            "rerank_score": "chosen_rerank_score",
            "candidate_answer": "chosen_answer",
        }
    )
    top2 = (
        out.sort_values(["ID", "rerank_score"], ascending=[True, False])
        .groupby("ID")
        .head(2)
        .groupby("ID")["rerank_score"]
        .agg(["first", "last"])
        .reset_index()
    )
    top2["score_margin_top1_top2"] = top2["first"] - top2["last"]
    best = best.merge(top2[["ID", "score_margin_top1_top2"]], on="ID", how="left")

    cand_path = out_dir / "val_candidate_scores.csv"
    chosen_path = out_dir / "val_chosen_scores.csv"
    summary_path = out_dir / "summary.json"
    out.to_csv(cand_path, index=False)
    best.to_csv(chosen_path, index=False)
    summary = {
        "experiment": "exp2_val_candidate_scores",
        "k": k,
        "val_rows": int(len(val)),
        "candidate_rows": int(len(out)),
        "candidate_scores": str(cand_path),
        "chosen_scores": str(chosen_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(upload: bool = True, k: int = 50, batch_size: int = 64):
    local_root = Path(__file__).resolve().parent
    if upload:
        print(f"Uploading current Train/Val CSVs to Modal volume {VOLUME_NAME}...")
        with volume.batch_upload(force=True) as batch:
            batch.put_file(local_root / "Train.csv", "/Train.csv")
            batch.put_file(local_root / "Val.csv", "/Val.csv")
        print("Upload complete.")
    summary = score_val_candidates.remote(k=k, batch_size=batch_size)
    print(summary)
