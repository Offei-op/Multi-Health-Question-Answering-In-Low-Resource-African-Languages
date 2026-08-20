from pathlib import Path

import modal


APP_NAME = "lalang-exp5-encoder-exp2-rerank-eval"
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
        "rouge-score>=0.1.2",
        "tqdm>=4.66.0",
        "safetensors>=0.4.3",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 4,
    volumes={str(REMOTE_ROOT): volume},
)
def run_eval(k: int = 50):
    import gc
    import json
    from collections import Counter

    import numpy as np
    import pandas as pd
    import torch
    from peft import PeftModel
    from rouge_score import rouge_scorer
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from sklearn.neighbors import NearestNeighbors
    from tqdm.auto import tqdm

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    qcol, acol, gcol, idcol = "input", "output", "subset", "ID"
    out_dir = REMOTE_ROOT / "exp5_encoder_exp2_rerank_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    class WhitespaceTokenizer:
        def tokenize(self, text):
            return [] if text is None else str(text).strip().split()

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"],
        tokenizer=WhitespaceTokenizer(),
        use_stemmer=False,
    )

    def rouge_scores(pred, ref):
        s = scorer.score(str(ref), str(pred))
        return float(s["rouge1"].fmeasure), float(s["rougeL"].fmeasure)

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

    def candidate_text(q, a):
        return f"Candidate question: {q}\nCandidate answer: {a}"

    print("Reading data...", flush=True)
    train = pd.read_csv(REMOTE_ROOT / "Train.csv")
    val = pd.read_csv(REMOTE_ROOT / "Val.csv")
    for df in (train, val):
        for c in (qcol, acol, gcol):
            df[c] = df[c].fillna("").astype(str).str.strip()
    train = train[(train[qcol] != "") & (train[acol] != "")].reset_index(drop=True)
    val = val[(val[qcol] != "") & (val[acol] != "")].reset_index(drop=True)
    print(f"train={len(train):,} val={len(val):,}", flush=True)

    print("\nLoading exp5 encoder adapter for candidate generation...", flush=True)
    bi = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
    bi.max_seq_length = 256
    inner = bi[0].auto_model
    bi[0].auto_model = PeftModel.from_pretrained(
        inner,
        str(REMOTE_ROOT / "exp5_bgem3_encoder_mining_v2" / "final"),
        is_trainable=False,
    )
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
            "embs": embs,
            "q": np.array(grp[qcol].astype(str).tolist(), dtype=object),
            "a": np.array(grp[acol].astype(str).tolist(), dtype=object),
        }

    print("\nRetrieving val topK from train with exp5 encoder...", flush=True)
    val_cands = [[] for _ in range(len(val))]
    for subset, grp in tqdm(list(val.groupby(gcol)), desc="Retrieve val"):
        m = indices[subset]
        q_embs = bi.encode(
            grp[qcol].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=128,
            convert_to_numpy=True,
        )
        dists, idx_mat = m["nn"].kneighbors(q_embs, n_neighbors=min(k, len(m["a"])))
        sims = 1.0 - dists
        for row_idx, idxs, row_sims in zip(grp.index, idx_mat, sims):
            val_cands[int(row_idx)] = [
                {
                    "q": str(m["q"][int(j)]),
                    "a": str(m["a"][int(j)]),
                    "bi_score": float(sim),
                    "rank": rank,
                }
                for rank, (j, sim) in enumerate(zip(idxs, row_sims), start=1)
            ]

    top1_preds, oracle_preds, oracle_ranks = [], [], []
    for cs, ref in zip(val_cands, val[acol].tolist()):
        if not cs:
            top1_preds.append("")
            oracle_preds.append("")
            oracle_ranks.append(0)
            continue
        labels = [fast_r1(c["a"], ref) for c in cs]
        best = int(np.argmax(labels))
        top1_preds.append(cs[0]["a"])
        oracle_preds.append(cs[best]["a"])
        oracle_ranks.append(best + 1)

    print("\nFreeing encoder before reranker scoring...", flush=True)
    try:
        bi.model.cpu()
    except Exception:
        pass
    del bi
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nLoading exp2 reranker...", flush=True)
    reranker = CrossEncoder(
        str(REMOTE_ROOT / "exp2_crossencoder_rerank" / "final"),
        max_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\nScoring exp5 candidates with exp2 reranker...", flush=True)
    flat_pairs, row_lens = [], []
    for q, cs in zip(val[qcol].tolist(), val_cands):
        flat_pairs.extend([(q, candidate_text(c["q"], c["a"])) for c in cs])
        row_lens.append(len(cs))
    scores = reranker.predict(flat_pairs, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    rerank_preds, chosen_ranks, chosen_scores = [], [], []
    off = 0
    for cs, n in zip(val_cands, row_lens):
        if n == 0:
            rerank_preds.append("")
            chosen_ranks.append(0)
            chosen_scores.append(np.nan)
            continue
        row_scores = scores[off : off + n]
        off += n
        j = int(np.argmax(row_scores))
        rerank_preds.append(cs[j]["a"])
        chosen_ranks.append(j + 1)
        chosen_scores.append(float(row_scores[j]))

    def score_preds(preds, refs, subs, label):
        rows = []
        for p, r, s_name in zip(preds, refs, subs):
            r1, rl = rouge_scores(p, r)
            rows.append({"subset": s_name, "rouge1": r1, "rougeL": rl})
        df = pd.DataFrame(rows)
        per = df.groupby("subset")[["rouge1", "rougeL"]].mean().round(4)
        out = {
            "label": label,
            "rouge1": float(df["rouge1"].mean()),
            "rougeL": float(df["rougeL"].mean()),
            "per_subset": per.to_dict(orient="index"),
        }
        print(json.dumps(out, indent=2), flush=True)
        return out

    refs = val[acol].tolist()
    subs = val[gcol].tolist()
    top1_metrics = score_preds(top1_preds, refs, subs, "exp5_encoder_top1")
    rerank_metrics = score_preds(rerank_preds, refs, subs, "exp5_encoder_exp2_rerank")
    oracle_metrics = score_preds(oracle_preds, refs, subs, "exp5_encoder_oracle_top50")

    pred_rows = []
    for i, (top1, rerank, oracle, ref) in enumerate(zip(top1_preds, rerank_preds, oracle_preds, refs)):
        top1_r1, top1_rl = rouge_scores(top1, ref)
        rerank_r1, rerank_rl = rouge_scores(rerank, ref)
        oracle_r1, oracle_rl = rouge_scores(oracle, ref)
        pred_rows.append(
            {
                "ID": val[idcol].iloc[i],
                "subset": val[gcol].iloc[i],
                "top1": top1,
                "rerank": rerank,
                "oracle": oracle,
                "reference": ref,
                "top1_r1": top1_r1,
                "rerank_r1": rerank_r1,
                "oracle_r1": oracle_r1,
                "top1_rl": top1_rl,
                "rerank_rl": rerank_rl,
                "oracle_rl": oracle_rl,
                "chosen_rank": chosen_ranks[i],
                "oracle_rank": oracle_ranks[i],
                "chosen_score": chosen_scores[i],
            }
        )
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out_dir / "val_predictions.csv", index=False)

    diagnostics = {
        "changed_from_top1_rows": int((pred_df["top1"] != pred_df["rerank"]).sum()),
        "wins_vs_top1_rows": int((pred_df["rerank_r1"] > pred_df["top1_r1"] + 1e-12).sum()),
        "hurts_vs_top1_rows": int((pred_df["rerank_r1"] + 1e-12 < pred_df["top1_r1"]).sum()),
        "exact_oracle_ge_095_and_rerank_lt_050_rows": int(
            ((pred_df["oracle_r1"] >= 0.95) & (pred_df["rerank_r1"] < 0.50)).sum()
        ),
    }
    print("\nDiagnostics:", flush=True)
    print(json.dumps(diagnostics, indent=2), flush=True)

    summary = {
        "experiment": "exp5_encoder_candidates_scored_by_exp2_reranker",
        "gpu": "L40S",
        "k": k,
        "encoder": "exp5_bgem3_encoder_mining_v2/final",
        "reranker": "exp2_crossencoder_rerank/final",
        "top1": top1_metrics,
        "rerank": rerank_metrics,
        "oracle": oracle_metrics,
        "delta_top1_vs_exp2_encoder": top1_metrics["rouge1"] - 0.5395458277838778,
        "delta_oracle_vs_exp2_encoder": oracle_metrics["rouge1"] - 0.6836959744084082,
        "delta_rerank_vs_exp2": rerank_metrics["rouge1"] - 0.5892166283468145,
        "diagnostics": diagnostics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(k: int = 50):
    summary = run_eval.remote(k=k)
    print(summary)
