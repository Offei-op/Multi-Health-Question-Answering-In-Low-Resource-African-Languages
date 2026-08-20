from pathlib import Path

import modal


APP_NAME = "lalang-lug-uga-reranker-exp10"
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
def run_lug_uga_reranker(
    subset_name: str = "Lug_Uga",
    k: int = 50,
    train_pairs_per_query: int = 12,
    epochs: int = 1,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 8e-6,
):
    import gc
    import json
    import random
    import time
    from collections import Counter
    from functools import lru_cache

    import numpy as np
    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import MSELoss
    from sklearn.neighbors import NearestNeighbors
    from tqdm.auto import tqdm

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    qcol, acol, gcol, idcol = "input", "output", "subset", "ID"
    out_dir = REMOTE_ROOT / "exp10_lug_uga_reranker"
    out_dir.mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=500_000)
    def ws_tokens(text):
        return tuple(str(text).strip().split())

    def f1_from_overlap(overlap, pred_len, ref_len):
        if overlap <= 0 or pred_len <= 0 or ref_len <= 0:
            return 0.0
        precision = overlap / pred_len
        recall = overlap / ref_len
        return 2.0 * precision * recall / (precision + recall)

    def lcs_len(a, b):
        if not a or not b:
            return 0
        if len(a) < len(b):
            short, long = a, b
        else:
            short, long = b, a
        prev = [0] * (len(short) + 1)
        for tok in long:
            cur = [0]
            left = 0
            for j, stok in enumerate(short, start=1):
                up = prev[j]
                if tok == stok:
                    val = prev[j - 1] + 1
                else:
                    val = left if left >= up else up
                cur.append(val)
                left = val
            prev = cur
        return prev[-1]

    def rouge_scores(pred, ref):
        pred_toks = ws_tokens(pred)
        ref_toks = ws_tokens(ref)
        if not pred_toks or not ref_toks:
            return 0.0, 0.0
        overlap = sum((Counter(pred_toks) & Counter(ref_toks)).values())
        r1 = f1_from_overlap(overlap, len(pred_toks), len(ref_toks))
        rl = f1_from_overlap(lcs_len(pred_toks, ref_toks), len(pred_toks), len(ref_toks))
        return float(r1), float(rl)

    def target_score(candidate_answer, reference):
        r1, rl = rouge_scores(candidate_answer, reference)
        return np.float32(0.75 * r1 + 0.25 * rl)

    def candidate_text(q, a):
        return f"Candidate question: {q}\nCandidate answer: {a}"

    def score_preds(preds, refs, subs, label):
        rows = []
        for p, r, s in zip(preds, refs, subs):
            r1, rl = rouge_scores(p, r)
            rows.append({"subset": s, "rouge1": r1, "rougeL": rl})
        df = pd.DataFrame(rows)
        out = {
            "label": label,
            "rouge1": float(df["rouge1"].mean()),
            "rougeL": float(df["rougeL"].mean()),
            "per_subset": df.groupby("subset")[["rouge1", "rougeL"]].mean().round(6).to_dict(orient="index"),
        }
        print(json.dumps(out, indent=2), flush=True)
        return out

    print("Reading data...", flush=True)
    train_all = pd.read_csv(REMOTE_ROOT / "Train.csv")
    val_all = pd.read_csv(REMOTE_ROOT / "Val.csv")
    for df in (train_all, val_all):
        for c in (idcol, qcol, acol, gcol):
            df[c] = df[c].fillna("").astype(str).str.strip()
    train = train_all[
        train_all[gcol].eq(subset_name) & train_all[qcol].ne("") & train_all[acol].ne("")
    ].reset_index(drop=True)
    val = val_all[val_all[gcol].eq(subset_name) & val_all[qcol].ne("") & val_all[acol].ne("")].reset_index(drop=True)
    total_val_rows = int(len(val_all[(val_all[qcol].ne("")) & (val_all[acol].ne(""))]))
    print(f"{subset_name} train={len(train):,} val={len(val):,}; all val rows={total_val_rows:,}", flush=True)

    print("\nLoading BGE-M3 encoder adapter for candidate generation...", flush=True)
    bi = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
    bi.max_seq_length = 256
    inner = bi[0].auto_model
    bi[0].auto_model = PeftModel.from_pretrained(inner, str(REMOTE_ROOT / "bge_m3_adapter"), is_trainable=False)
    bi[0].auto_model.eval()

    print("Indexing Lug_Uga train questions...", flush=True)
    train_emb = bi.encode(
        train[qcol].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=128,
        convert_to_numpy=True,
    )
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(train)), metric="cosine").fit(train_emb)

    def retrieve(query_df, leave_self_out=False):
        q_emb = bi.encode(
            query_df[qcol].tolist(),
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=128,
            convert_to_numpy=True,
        )
        n_neighbors = min(k + (1 if leave_self_out else 0), len(train))
        dists, idx_mat = nn.kneighbors(q_emb, n_neighbors=n_neighbors)
        cands = []
        for row_i, row_dists, idxs in tqdm(
            list(zip(query_df.index, dists, idx_mat)), desc="Build candidate lists"
        ):
            picked = []
            for dist, j in zip(row_dists, idxs):
                if leave_self_out and int(j) == int(row_i):
                    continue
                picked.append(
                    {
                        "q": str(train[qcol].iloc[int(j)]),
                        "a": str(train[acol].iloc[int(j)]),
                        "rank": len(picked) + 1,
                        "bi_score": float(1.0 - dist),
                    }
                )
                if len(picked) >= k:
                    break
            cands.append(picked)
        return cands

    print("\nRetrieving train/val candidates...", flush=True)
    train_cands = retrieve(train, leave_self_out=True)
    val_cands = retrieve(val, leave_self_out=False)
    del bi
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nCandidate baseline/oracle:", flush=True)
    top1 = [cs[0]["a"] if cs else "" for cs in val_cands]
    oracle = [max(cs, key=lambda c, ref=ref: target_score(c["a"], ref))["a"] if cs else "" for cs, ref in zip(val_cands, val[acol])]
    top1_metrics = score_preds(top1, val[acol].tolist(), val[gcol].tolist(), "lug_top1")
    oracle_metrics = score_preds(oracle, val[acol].tolist(), val[gcol].tolist(), "lug_oracle")

    print("\nBuilding Lug_Uga reranker pairs...", flush=True)
    rng = np.random.default_rng(seed)
    pair_q, pair_c, pair_y = [], [], []
    for i in tqdm(range(len(train)), desc="Lug reranker pairs"):
        cs = train_cands[int(i)]
        if not cs:
            continue
        labels = np.array([target_score(c["a"], train[acol].iloc[int(i)]) for c in cs], dtype=np.float32)
        order = np.argsort(-labels)
        chosen = []
        chosen.extend(order[: min(4, len(order))].tolist())
        chosen.extend(order[-min(4, len(order)) :].tolist())
        mid_pool = order[4:-4] if len(order) > 8 else order
        if len(mid_pool) > 0:
            n_mid = max(0, train_pairs_per_query - len(set(chosen)))
            chosen.extend(rng.choice(mid_pool, size=min(n_mid, len(mid_pool)), replace=False).tolist())
        seen = set()
        chosen = [x for x in chosen if not (x in seen or seen.add(x))]
        for j in chosen[:train_pairs_per_query]:
            pair_q.append(str(train[qcol].iloc[int(i)]))
            pair_c.append(candidate_text(cs[int(j)]["q"], cs[int(j)]["a"]))
            pair_y.append(float(labels[int(j)]))
    print(f"Lug_Uga reranker pairs={len(pair_y):,}", flush=True)

    ds = Dataset.from_dict({"query": pair_q, "candidate": pair_c, "label": pair_y}).shuffle(seed=seed)
    del pair_q, pair_c, pair_y, train_cands
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def rerank_predictions(reranker_model, label):
        flat, row_lens = [], []
        for q, cs in zip(val[qcol].tolist(), val_cands):
            flat.extend([(q, candidate_text(c["q"], c["a"])) for c in cs])
            row_lens.append(len(cs))
        scores = reranker_model.predict(flat, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        preds, debug_rows, off = [], [], 0
        for row, cs, n in zip(val.itertuples(index=False), val_cands, row_lens):
            if n == 0:
                preds.append("")
                continue
            row_scores = np.asarray(scores[off : off + n], dtype=np.float32)
            off += n
            best = int(np.argmax(row_scores))
            preds.append(cs[best]["a"])
            debug_rows.append(
                {
                    "ID": getattr(row, idcol),
                    "subset": getattr(row, gcol),
                    "label": label,
                    "candidate_rank": cs[best]["rank"],
                    "bi_score": cs[best]["bi_score"],
                    "rerank_score": float(row_scores[best]),
                }
            )
        return preds, debug_rows

    print("\nScoring Lug val with current global exp2 reranker...", flush=True)
    global_reranker = CrossEncoder(
        str(REMOTE_ROOT / "exp2_crossencoder_rerank" / "final"),
        max_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    global_preds, global_debug = rerank_predictions(global_reranker, "global_exp2_reranker")
    global_metrics = score_preds(global_preds, val[acol].tolist(), val[gcol].tolist(), "global_exp2_reranker")
    del global_reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nContinuing exp2 reranker on Lug_Uga pairs...", flush=True)
    reranker = CrossEncoder(
        str(REMOTE_ROOT / "exp2_crossencoder_rerank" / "final"),
        max_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    try:
        reranker.model.gradient_checkpointing_enable()
    except Exception as e:
        print(f"gradient checkpointing not enabled: {e}", flush=True)

    loss = MSELoss(model=reranker)
    args = CrossEncoderTrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.1,
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=["none"],
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        seed=seed,
    )
    trainer = CrossEncoderTrainer(model=reranker, args=args, train_dataset=ds, loss=loss)
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    reranker_dir = out_dir / "reranker_final"
    reranker.save_pretrained(str(reranker_dir))
    print(f"Lug_Uga reranker trained in {train_seconds / 60:.1f} min", flush=True)

    print("\nScoring Lug val with specialized Lug reranker...", flush=True)
    lug_preds, lug_debug = rerank_predictions(reranker, "lug_specialized_reranker")
    lug_metrics = score_preds(lug_preds, val[acol].tolist(), val[gcol].tolist(), "lug_specialized_reranker")

    pred_df = pd.DataFrame(
        {
            "ID": val[idcol].tolist(),
            "subset": val[gcol].tolist(),
            "reference": val[acol].tolist(),
            "top1": top1,
            "oracle": oracle,
            "global_rerank": global_preds,
            "lug_rerank": lug_preds,
        }
    )
    pred_df.to_csv(out_dir / "lug_val_predictions.csv", index=False)
    pd.DataFrame(global_debug + lug_debug).to_csv(out_dir / "lug_val_candidate_scores.csv", index=False)

    subset_delta = lug_metrics["rouge1"] - global_metrics["rouge1"]
    overall_delta_if_swapped = subset_delta * len(val) / total_val_rows
    summary = {
        "experiment": "exp10_lug_uga_reranker",
        "subset": subset_name,
        "gpu": "L40S",
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "all_val_rows": total_val_rows,
        "k": k,
        "epochs": epochs,
        "train_pairs_per_query": train_pairs_per_query,
        "reranker_examples": int(len(ds)),
        "train_seconds": train_seconds,
        "top1": top1_metrics,
        "oracle": oracle_metrics,
        "global_rerank": global_metrics,
        "lug_rerank": lug_metrics,
        "subset_delta_r1": subset_delta,
        "overall_delta_if_only_lug_swapped": overall_delta_if_swapped,
        "reranker_dir": str(reranker_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main():
    local_root = Path(__file__).resolve().parent
    # Data/model artifacts should already be in the shared Modal volume from prior experiments.
    required = ["Train.csv", "Val.csv"]
    missing = [p for p in required if not (REMOTE_ROOT / p)]
    if missing:
        print(f"Will rely on existing volume files; missing check skipped: {missing}")
    call = run_lug_uga_reranker.spawn()
    print(f"Spawned Lug_Uga exp10 reranker call: {call.object_id}")
