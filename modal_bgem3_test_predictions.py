from pathlib import Path

import modal


APP_NAME = "lalang-bgem3-test-predict"
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
def predict_test(k: int = 50, batch_size: int = 64):
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
    out_dir = REMOTE_ROOT / "test_predictions_best_setup"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading Train, Val, Test...", flush=True)
    train = pd.read_csv(REMOTE_ROOT / "Train.csv")
    val = pd.read_csv(REMOTE_ROOT / "Val.csv")
    test = pd.read_csv(REMOTE_ROOT / "Test.csv")
    for df in (train, val, test):
        for c in (idcol, qcol, gcol):
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str).str.strip()
        if acol in df.columns:
            df[acol] = df[acol].fillna("").astype(str).str.strip()

    bank = pd.concat([train, val], ignore_index=True)
    bank = bank[(bank[qcol] != "") & (bank[acol] != "") & (bank[gcol] != "")].reset_index(drop=True)
    test = test[(test[qcol] != "") & (test[gcol] != "")].reset_index(drop=True)
    print(f"bank=train+val={len(bank):,} test={len(test):,}", flush=True)
    print("Bank subsets:", flush=True)
    print(bank[gcol].value_counts().sort_index().to_string(), flush=True)
    print("Test subsets:", flush=True)
    print(test[gcol].value_counts().sort_index().to_string(), flush=True)

    def candidate_text(q, a):
        return f"Candidate question: {q}\nCandidate answer: {a}"

    print("\nLoading fine-tuned BGE-M3 encoder adapter...", flush=True)
    bi = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
    bi.max_seq_length = 256
    inner = bi[0].auto_model
    bi[0].auto_model = PeftModel.from_pretrained(inner, str(REMOTE_ROOT / "bge_m3_adapter"), is_trainable=False)
    bi[0].auto_model.eval()

    print("\nEncoding Train+Val answer bank per subset...", flush=True)
    indices = {}
    for subset, grp in tqdm(list(bank.groupby(gcol)), desc="Index bank"):
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

    print("\nRetrieving test topK per subset...", flush=True)
    test_cands = [[] for _ in range(len(test))]
    pos = {idx: i for i, idx in enumerate(test.index)}
    for subset, grp in tqdm(list(test.groupby(gcol)), desc="Retrieve test"):
        if subset not in indices:
            print(f"WARNING: no bank rows for subset {subset}; leaving blank predictions", flush=True)
            continue
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
        for row_idx, dists, idxs in zip(grp.index, distances, idx_mat):
            picked = []
            for rank, (dist, j) in enumerate(zip(dists, idxs), start=1):
                picked.append(
                    {
                        "rank": rank,
                        "bi_score": float(1.0 - dist),
                        "q": str(m["q"][j]),
                        "a": str(m["a"][j]),
                    }
                )
            test_cands[pos[row_idx]] = picked

    print("\nFreeing encoder before reranker scoring...", flush=True)
    try:
        bi.model.cpu()
    except Exception:
        pass
    del bi
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nLoading trained cross-encoder reranker...", flush=True)
    reranker = CrossEncoder(
        str(REMOTE_ROOT / "exp2_crossencoder_rerank" / "final"),
        max_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\nScoring test candidates...", flush=True)
    flat_pairs, row_lens = [], []
    for q, cs in zip(test[qcol].tolist(), test_cands):
        flat_pairs.extend([(q, candidate_text(c["q"], c["a"])) for c in cs])
        row_lens.append(len(cs))
    scores = reranker.predict(flat_pairs, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    preds, top1_preds, debug_rows = [], [], []
    off = 0
    for row_i, (row, cs, n) in enumerate(zip(test.itertuples(index=False), test_cands, row_lens)):
        if n == 0:
            preds.append("")
            top1_preds.append("")
            continue
        row_scores = np.asarray(scores[off : off + n], dtype=np.float32)
        off += n
        best_i = int(np.argmax(row_scores))
        preds.append(cs[best_i]["a"])
        top1_preds.append(cs[0]["a"])
        for cand_i, c in enumerate(cs):
            debug_rows.append(
                {
                    "ID": getattr(row, idcol),
                    "subset": getattr(row, gcol),
                    "candidate_rank": c["rank"],
                    "bi_score": c["bi_score"],
                    "rerank_score": float(row_scores[cand_i]),
                    "chosen": cand_i == best_i,
                    "candidate_question": c["q"],
                    "candidate_answer": c["a"],
                }
            )

    submission = pd.DataFrame(
        {
            "ID": test[idcol].tolist(),
            "TargetRLF1": preds,
            "TargetR1F1": preds,
            "TargetLLM": preds,
        }
    )
    submission_path = out_dir / "submission_best_bgem3_rerank_train_val.csv"
    debug_path = out_dir / "test_candidate_scores.csv"
    top1_path = out_dir / "submission_encoder_top1_train_val.csv"
    summary_path = out_dir / "summary.json"

    submission.to_csv(submission_path, index=False)
    pd.DataFrame(
        {
            "ID": test[idcol].tolist(),
            "TargetRLF1": top1_preds,
            "TargetR1F1": top1_preds,
            "TargetLLM": top1_preds,
        }
    ).to_csv(top1_path, index=False)
    pd.DataFrame(debug_rows).to_csv(debug_path, index=False)

    summary = {
        "experiment": "test_predictions_best_bgem3_top50_crossencoder_rerank",
        "k": k,
        "bank": "Train.csv + Val.csv",
        "test_rows": int(len(test)),
        "submission": str(submission_path),
        "encoder_top1_submission": str(top1_path),
        "debug_candidates": str(debug_path),
        "blank_predictions": int(sum(1 for p in preds if not p)),
        "changed_by_reranker_vs_encoder_top1": int(sum(p != t for p, t in zip(preds, top1_preds))),
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
        print(f"Uploading current CSVs to Modal volume {VOLUME_NAME}...")
        with volume.batch_upload(force=True) as batch:
            batch.put_file(local_root / "Train.csv", "/Train.csv")
            batch.put_file(local_root / "Val.csv", "/Val.csv")
            batch.put_file(local_root / "Test.csv", "/Test.csv")
        print("Upload complete.")
    summary = predict_test.remote(k=k, batch_size=batch_size)
    print(summary)
