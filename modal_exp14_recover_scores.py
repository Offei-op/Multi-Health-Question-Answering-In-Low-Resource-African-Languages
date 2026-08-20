from pathlib import Path

import modal


APP_NAME = "lalang-exp14-recover-scores"
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
        "datasets>=3.0.0",
        "accelerate>=0.33.0",
        "peft>=0.12.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.5.0",
        "safetensors>=0.4.3",
        "protobuf>=4.25.0",
        "sentencepiece>=0.2.0",
        "tqdm>=4.66.0",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 4,
    volumes={str(REMOTE_ROOT): volume},
)
def recover(score_gemma: bool = False, score_exp2: bool = False):
    import gc
    import json
    from collections import Counter

    import numpy as np
    import pandas as pd
    import torch
    from peft import PeftModel
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from sklearn.neighbors import NearestNeighbors
    from tqdm.auto import tqdm

    qcol, acol, gcol, idcol = "input", "output", "subset", "ID"
    k = 50
    out_dir = REMOTE_ROOT / "exp14_qonly_gemma_reranker"
    report_dir = REMOTE_ROOT / "exp14_qonly_gemma_reranker_recovered"
    report_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    def rouge1_fast(pred, ref):
        pred_toks = str(pred).strip().split()
        ref_toks = str(ref).strip().split()
        if not pred_toks or not ref_toks:
            return 0.0
        overlap = sum((Counter(pred_toks) & Counter(ref_toks)).values())
        if overlap == 0:
            return 0.0
        return float(2.0 * overlap / (len(pred_toks) + len(ref_toks)))

    def score_preds(preds, refs, subs, label):
        rows = [{"subset": s, "rouge1": rouge1_fast(p, r)} for p, r, s in zip(preds, refs, subs)]
        df = pd.DataFrame(rows)
        return {
            "label": label,
            "rouge1": float(df["rouge1"].mean()),
            "per_subset": df.groupby("subset")["rouge1"].mean().round(4).to_dict(),
        }

    def candidate_text_qa(q, a):
        return f"Candidate question: {q}\nCandidate answer: {a}"

    def candidate_text_q(q, _a=None):
        return str(q)

    def zscore(row):
        arr = np.asarray(row, dtype=np.float32)
        sd = float(arr.std())
        if sd < 1e-6:
            return np.zeros_like(arr)
        return (arr - float(arr.mean())) / sd

    print("Reading data...", flush=True)
    train = pd.read_csv(REMOTE_ROOT / "Train.csv")
    val = pd.read_csv(REMOTE_ROOT / "Val.csv")
    for df in (train, val):
        for c in (qcol, acol, gcol, idcol):
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str).str.strip()
    train = train[(train[qcol] != "") & (train[acol] != "")].reset_index(drop=True)
    val = val[(val[qcol] != "") & (val[acol] != "")].reset_index(drop=True)

    print("Rebuilding exp14 candidate pool...", flush=True)
    bi = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
    bi.max_seq_length = 256
    bi[0].auto_model = PeftModel.from_pretrained(
        bi[0].auto_model,
        str(REMOTE_ROOT / "bge_m3_adapter"),
        is_trainable=False,
    )
    bi[0].auto_model.eval()
    indices = {}
    for subset, grp in tqdm(list(train.groupby(gcol)), desc="Index train"):
        embs = bi.encode(
            grp[qcol].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=128,
            convert_to_numpy=True,
        )
        indices[subset] = {
            "nn": NearestNeighbors(n_neighbors=min(k, len(grp)), metric="cosine").fit(embs),
            "q": np.array(grp[qcol].astype(str).tolist(), dtype=object),
            "a": np.array(grp[acol].astype(str).tolist(), dtype=object),
        }

    val_cands = [[] for _ in range(len(val))]
    pos = {idx: i for i, idx in enumerate(val.index)}
    for subset, grp in tqdm(list(val.groupby(gcol)), desc="Retrieve val"):
        m = indices[subset]
        q_embs = bi.encode(
            grp[qcol].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=128,
            convert_to_numpy=True,
        )
        _, idx_mat = m["nn"].kneighbors(q_embs, n_neighbors=min(k, len(m["a"])))
        for row_idx, idxs in zip(grp.index, idx_mat):
            val_cands[pos[row_idx]] = [{"q": str(m["q"][j]), "a": str(m["a"][j])} for j in idxs]
    del bi
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    refs = val[acol].tolist()
    subs = val[gcol].tolist()
    queries = val[qcol].tolist()
    top1 = [cs[0]["a"] if cs else "" for cs in val_cands]

    def score_candidate_pairs(reranker, text_fn, label, batch_size=128):
        flat_pairs, row_lens = [], []
        for q, cs in zip(queries, val_cands):
            flat_pairs.extend([(q, text_fn(c["q"], c["a"])) for c in cs])
            row_lens.append(len(cs))
        print(f"Scoring {len(flat_pairs):,} pairs for {label}...", flush=True)
        raw = reranker.predict(flat_pairs, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        preds, score_rows, off = [], [], 0
        for cs, n in zip(val_cands, row_lens):
            row_scores = np.asarray(raw[off : off + n], dtype=np.float32).reshape(-1)
            off += n
            score_rows.append(row_scores)
            preds.append(cs[int(np.argmax(row_scores))]["a"] if n else "")
        return preds, score_rows

    metrics = {"top1": score_preds(top1, refs, subs, "ft_bgem3_top1")}
    predictions = {
        "ID": val[idcol].tolist(),
        "subset": subs,
        "top1": top1,
        "reference": refs,
    }
    score_rows_by_name = {}

    print("Loading saved q-only BGE-M3 reranker...", flush=True)
    qonly = CrossEncoder(str(out_dir / "qonly_bgem3_final"), max_length=256, device="cuda" if torch.cuda.is_available() else "cpu")
    preds, rows = score_candidate_pairs(qonly, candidate_text_q, "qonly_bgem3", batch_size=128)
    metrics["qonly_bgem3"] = score_preds(preds, refs, subs, "qonly_bgem3")
    predictions["qonly_bgem3"] = preds
    score_rows_by_name["qonly_bgem3"] = rows
    del qonly
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if score_exp2:
        print("Loading exp2 q+a reranker for blend comparison...", flush=True)
        exp2 = CrossEncoder(str(REMOTE_ROOT / "exp2_crossencoder_rerank" / "final"), max_length=512, device="cuda" if torch.cuda.is_available() else "cpu")
        preds, rows = score_candidate_pairs(exp2, candidate_text_qa, "exp2_qa", batch_size=128)
        metrics["exp2_qa_same_candidates"] = score_preds(preds, refs, subs, "exp2_qa_same_candidates")
        predictions["exp2_qa"] = preds
        score_rows_by_name["exp2_qa"] = rows
        del exp2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if score_gemma:
        print("Scoring Gemma q-only and q+a...", flush=True)
        gemma = CrossEncoder(
            "BAAI/bge-reranker-v2-gemma",
            num_labels=1,
            max_length=512,
            trust_remote_code=True,
            automodel_args={"torch_dtype": torch.bfloat16},
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        preds, rows = score_candidate_pairs(gemma, candidate_text_q, "gemma_qonly_zero_shot", batch_size=16)
        metrics["gemma_qonly_zero_shot"] = score_preds(preds, refs, subs, "gemma_qonly_zero_shot")
        predictions["gemma_qonly"] = preds
        score_rows_by_name["gemma_qonly"] = rows
        preds, rows = score_candidate_pairs(gemma, candidate_text_qa, "gemma_qa_zero_shot", batch_size=16)
        metrics["gemma_qa_zero_shot"] = score_preds(preds, refs, subs, "gemma_qa_zero_shot")
        predictions["gemma_qa"] = preds
        score_rows_by_name["gemma_qa"] = rows

    def blend(names, label):
        preds = []
        for i, cs in enumerate(val_cands):
            if not cs:
                preds.append("")
                continue
            total = np.zeros(len(cs), dtype=np.float32)
            for name in names:
                total += zscore(score_rows_by_name[name][i])
            preds.append(cs[int(np.argmax(total))]["a"])
        metrics[label] = score_preds(preds, refs, subs, label)
        predictions[label] = preds

    if score_exp2:
        blend(["exp2_qa", "qonly_bgem3"], "blend_exp2_qa_plus_qonly_bgem3")
    if score_gemma:
        blend(["exp2_qa", "qonly_bgem3", "gemma_qa"], "blend_exp2_qa_qonly_bgem3_gemma_qa")
        blend(["qonly_bgem3", "gemma_qonly"], "blend_qonly_bgem3_plus_gemma_qonly")

    pd.DataFrame(predictions).to_csv(report_dir / "val_predictions.csv", index=False)
    summary = {
        "experiment": "exp14_recovered_scores",
        "source_model_dir": str(out_dir / "qonly_bgem3_final"),
        "metrics": metrics,
        "score_gemma": score_gemma,
        "score_exp2": score_exp2,
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(score_gemma: bool = False, score_exp2: bool = False):
    call = recover.spawn(score_gemma=score_gemma, score_exp2=score_exp2)
    print(f"Spawned exp14 recovery call: {call.object_id}")
