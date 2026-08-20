from pathlib import Path

import modal


APP_NAME = "lalang-qonly-gemma-reranker-exp14"
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
        "rouge-score>=0.1.2",
        "tqdm>=4.66.0",
        "safetensors>=0.4.3",
        "protobuf>=4.25.0",
        "sentencepiece>=0.2.0",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 8,
    volumes={str(REMOTE_ROOT): volume},
)
def run_exp14(
    k: int = 50,
    train_pairs_per_query: int = 12,
    epochs: int = 1,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 1e-5,
    score_gemma: bool = True,
    gemma_batch_size: int = 16,
):
    import gc
    import json
    import random
    import time
    from collections import Counter

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
    out_dir = REMOTE_ROOT / "exp14_qonly_gemma_reranker"
    out_dir.mkdir(parents=True, exist_ok=True)

    def rouge1_fast(pred, ref):
        pred_toks = str(pred).strip().split()
        ref_toks = str(ref).strip().split()
        if not pred_toks or not ref_toks:
            return 0.0
        overlap = sum((Counter(pred_toks) & Counter(ref_toks)).values())
        if overlap == 0:
            return 0.0
        return float(2.0 * overlap / (len(pred_toks) + len(ref_toks)))

    def target_score(candidate_answer, reference):
        return np.float32(rouge1_fast(candidate_answer, reference))

    def candidate_text_qa(q, a):
        return f"Candidate question: {q}\nCandidate answer: {a}"

    def candidate_text_q(q, _a=None):
        return str(q)

    def zscore(x):
        arr = np.asarray(x, dtype=np.float32)
        if len(arr) == 0:
            return arr
        sd = float(arr.std())
        if sd < 1e-6:
            return np.zeros_like(arr)
        return (arr - float(arr.mean())) / sd

    def score_preds(preds, refs, subs, label):
        rows = []
        for p, r, s_name in zip(preds, refs, subs):
            rows.append({"subset": s_name, "rouge1": rouge1_fast(p, r)})
        df = pd.DataFrame(rows)
        out = {
            "label": label,
            "rouge1": float(df["rouge1"].mean()),
            "per_subset": df.groupby("subset")["rouge1"].mean().round(4).to_dict(),
        }
        print(json.dumps(out, indent=2), flush=True)
        return out

    def evaluate_candidate_lists(cands, refs, subs, label):
        top1_preds, oracle_preds, rows = [], [], []
        for cs, ref, subset in zip(cands, refs, subs):
            if not cs:
                top1 = oracle = ""
                top1_r1 = oracle_r1 = 0.0
            else:
                top1 = cs[0]["a"]
                oracle = max(cs, key=lambda c: target_score(c["a"], ref))["a"]
                top1_r1 = rouge1_fast(top1, ref)
                oracle_r1 = rouge1_fast(oracle, ref)
            top1_preds.append(top1)
            oracle_preds.append(oracle)
            rows.append({"subset": subset, "top1_r1": top1_r1, "oracle_r1": oracle_r1})
        df = pd.DataFrame(rows)
        out = {
            "label": label,
            "top1_r1": float(df["top1_r1"].mean()),
            "oracle_r1": float(df["oracle_r1"].mean()),
            "per_subset": df.groupby("subset")[["top1_r1", "oracle_r1"]].mean().round(4).to_dict(orient="index"),
        }
        print(json.dumps(out, indent=2), flush=True)
        return out, top1_preds, oracle_preds

    def score_candidate_pairs(reranker, cands, queries, text_fn, label, batch_size_=64):
        flat_pairs, row_lens = [], []
        for q, cs in zip(queries, cands):
            flat_pairs.extend([(q, text_fn(c["q"], c["a"])) for c in cs])
            row_lens.append(len(cs))
        print(f"Scoring {len(flat_pairs):,} pairs for {label}...", flush=True)
        raw = reranker.predict(flat_pairs, batch_size=batch_size_, show_progress_bar=True, convert_to_numpy=True)
        scores_by_row, off = [], 0
        preds = []
        for cs, n in zip(cands, row_lens):
            if n == 0:
                scores_by_row.append(np.asarray([], dtype=np.float32))
                preds.append("")
                continue
            row_scores = np.asarray(raw[off : off + n], dtype=np.float32).reshape(-1)
            off += n
            scores_by_row.append(row_scores)
            preds.append(cs[int(np.argmax(row_scores))]["a"])
        return preds, scores_by_row

    def blend_scores(cands, score_rows, refs, subs, label):
        preds = []
        for cs, rows in zip(cands, score_rows):
            if not cs:
                preds.append("")
                continue
            blended = np.zeros(len(cs), dtype=np.float32)
            for row in rows:
                blended += zscore(row)
            preds.append(cs[int(np.argmax(blended))]["a"])
        return score_preds(preds, refs, subs, label), preds

    print("Reading data...", flush=True)
    train = pd.read_csv(REMOTE_ROOT / "Train.csv")
    val = pd.read_csv(REMOTE_ROOT / "Val.csv")
    for df in (train, val):
        for c in (qcol, acol, gcol, idcol):
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str).str.strip()
    train = train[(train[qcol] != "") & (train[acol] != "")].reset_index(drop=True)
    val = val[(val[qcol] != "") & (val[acol] != "")].reset_index(drop=True)
    print(f"train={len(train):,} val={len(val):,}", flush=True)
    print(train[gcol].value_counts().sort_index().to_string(), flush=True)

    print("\nLoading fine-tuned BGE-M3 encoder for candidate generation...", flush=True)
    bi = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
    bi.max_seq_length = 256
    bi[0].auto_model = PeftModel.from_pretrained(
        bi[0].auto_model,
        str(REMOTE_ROOT / "bge_m3_adapter"),
        is_trainable=False,
    )
    bi[0].auto_model.eval()

    def build_indices(df):
        indices = {}
        for subset, grp in tqdm(list(df.groupby(gcol)), desc="Index train"):
            embs = bi.encode(
                grp[qcol].tolist(),
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=128,
                convert_to_numpy=True,
            )
            nn = NearestNeighbors(n_neighbors=min(k + 1, len(grp)), metric="cosine").fit(embs)
            indices[subset] = {
                "nn": nn,
                "q": np.array(grp[qcol].astype(str).tolist(), dtype=object),
                "a": np.array(grp[acol].astype(str).tolist(), dtype=object),
                "orig_idx": np.array(grp.index.tolist()),
            }
        return indices

    def retrieve_topk(df, indices, leave_self_out=False):
        cands = [[] for _ in range(len(df))]
        pos = {idx: i for i, idx in enumerate(df.index)}
        for subset, grp in tqdm(list(df.groupby(gcol)), desc="Retrieve"):
            m = indices[subset]
            q_embs = bi.encode(
                grp[qcol].tolist(),
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=128,
                convert_to_numpy=True,
            )
            n_neighbors = min(k + (1 if leave_self_out else 0), len(m["a"]))
            dists, idx_mat = m["nn"].kneighbors(q_embs, n_neighbors=n_neighbors)
            for row_idx, row_dists, idxs in zip(grp.index, dists, idx_mat):
                picked = []
                for dist, j in zip(row_dists, idxs):
                    if leave_self_out and m["orig_idx"][j] == row_idx:
                        continue
                    picked.append({"q": str(m["q"][j]), "a": str(m["a"][j]), "bi_score": float(1.0 - dist)})
                    if len(picked) >= k:
                        break
                cands[pos[row_idx]] = picked
        return cands

    train_idx = build_indices(train)
    print("\nRetrieving train topK with self-exclusion...", flush=True)
    train_cands = retrieve_topk(train, train_idx, leave_self_out=True)
    print("\nRetrieving val topK from train...", flush=True)
    val_cands = retrieve_topk(val, train_idx, leave_self_out=False)

    baseline_metrics, val_top1, val_oracle = evaluate_candidate_lists(
        val_cands, val[acol].tolist(), val[gcol].tolist(), f"ft_bgem3_per_subset_top{k}"
    )
    top1_metrics = score_preds(val_top1, val[acol].tolist(), val[gcol].tolist(), "ft_bgem3_top1")
    oracle_metrics = score_preds(val_oracle, val[acol].tolist(), val[gcol].tolist(), "oracle_topk")

    print("\nFreeing bi-encoder before reranking...", flush=True)
    try:
        bi.model.cpu()
    except Exception:
        pass
    del bi
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_metrics = {"top1": top1_metrics, "oracle": oracle_metrics}
    score_tables = {}
    prediction_tables = {
        "ID": val[idcol].tolist(),
        "subset": val[gcol].tolist(),
        "top1": val_top1,
        "oracle": val_oracle,
        "reference": val[acol].tolist(),
    }

    exp2_dir = REMOTE_ROOT / "exp2_crossencoder_rerank" / "final"
    if exp2_dir.exists():
        print("\nScoring existing exp2 q+a reranker on the same candidates...", flush=True)
        current = CrossEncoder(str(exp2_dir), max_length=512, device="cuda" if torch.cuda.is_available() else "cpu")
        preds, score_rows = score_candidate_pairs(current, val_cands, val[qcol].tolist(), candidate_text_qa, "exp2_qa")
        all_metrics["exp2_qa_same_candidates"] = score_preds(preds, val[acol].tolist(), val[gcol].tolist(), "exp2_qa_same_candidates")
        score_tables["exp2_qa"] = score_rows
        prediction_tables["exp2_qa"] = preds
        del current
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"WARNING: missing {exp2_dir}; skipping current reranker scoring.", flush=True)

    print("\nBuilding q-only BGE-M3 cross-encoder regression pairs...", flush=True)
    rng = np.random.default_rng(seed)
    pair_q, pair_c, pair_y = [], [], []
    per_subset_pair_counts = {}
    for subset in sorted(train[gcol].unique()):
        subset_rows = np.where(train[gcol].to_numpy() == subset)[0]
        before = len(pair_y)
        for i in tqdm(subset_rows, desc=f"Pairs {subset}", mininterval=5):
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
                pair_c.append(candidate_text_q(cs[int(j)]["q"]))
                pair_y.append(float(labels[int(j)]))
        per_subset_pair_counts[subset] = len(pair_y) - before
    print("Pair counts:", per_subset_pair_counts, flush=True)
    print(f"Total pairs: {len(pair_y):,}", flush=True)
    print(pd.Series(pair_y).describe().round(4).to_string(), flush=True)
    train_ds = Dataset.from_dict({"query": pair_q, "candidate": pair_c, "label": pair_y}).shuffle(seed=seed)
    del pair_q, pair_c, pair_y, train_cands
    gc.collect()

    print("\nTraining BGE-M3 q-only cross-encoder...", flush=True)
    reranker = CrossEncoder(
        "BAAI/bge-reranker-v2-m3",
        num_labels=1,
        max_length=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    loss = MSELoss(model=reranker)
    args = CrossEncoderTrainingArguments(
        output_dir=str(out_dir / "qonly_bgem3_trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.1,
        bf16=True,
        fp16=False,
        tf32=True,
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=["none"],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=seed,
    )
    trainer = CrossEncoderTrainer(model=reranker, args=args, train_dataset=train_ds, loss=loss)
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    reranker.save_pretrained(str(out_dir / "qonly_bgem3_final"))
    print(f"BGE-M3 q-only reranker trained in {train_seconds / 3600:.2f} hours", flush=True)

    preds, score_rows = score_candidate_pairs(reranker, val_cands, val[qcol].tolist(), candidate_text_q, "qonly_bgem3", batch_size_=128)
    all_metrics["qonly_bgem3"] = score_preds(preds, val[acol].tolist(), val[gcol].tolist(), "qonly_bgem3")
    score_tables["qonly_bgem3"] = score_rows
    prediction_tables["qonly_bgem3"] = preds
    del reranker, trainer, train_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if score_gemma:
        print("\nScoring pretrained BGE Gemma reranker, q-only and q+a...", flush=True)
        gemma = CrossEncoder(
            "BAAI/bge-reranker-v2-gemma",
            num_labels=1,
            max_length=512,
            trust_remote_code=True,
            automodel_args={"torch_dtype": torch.bfloat16},
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        preds, score_rows = score_candidate_pairs(
            gemma, val_cands, val[qcol].tolist(), candidate_text_q, "gemma_qonly_zero_shot", batch_size_=gemma_batch_size
        )
        all_metrics["gemma_qonly_zero_shot"] = score_preds(preds, val[acol].tolist(), val[gcol].tolist(), "gemma_qonly_zero_shot")
        score_tables["gemma_qonly"] = score_rows
        prediction_tables["gemma_qonly"] = preds

        preds, score_rows = score_candidate_pairs(
            gemma, val_cands, val[qcol].tolist(), candidate_text_qa, "gemma_qa_zero_shot", batch_size_=gemma_batch_size
        )
        all_metrics["gemma_qa_zero_shot"] = score_preds(preds, val[acol].tolist(), val[gcol].tolist(), "gemma_qa_zero_shot")
        score_tables["gemma_qa"] = score_rows
        prediction_tables["gemma_qa"] = preds
        del gemma
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "exp2_qa" in score_tables and "qonly_bgem3" in score_tables:
        metric, preds = blend_scores(
            val_cands,
            [score_tables["exp2_qa"], score_tables["qonly_bgem3"]],
            val[acol].tolist(),
            val[gcol].tolist(),
            "blend_exp2_qa_plus_qonly_bgem3",
        )
        all_metrics["blend_exp2_qa_plus_qonly_bgem3"] = metric
        prediction_tables["blend_exp2_qa_plus_qonly_bgem3"] = preds
    if "exp2_qa" in score_tables and "qonly_bgem3" in score_tables and "gemma_qa" in score_tables:
        metric, preds = blend_scores(
            val_cands,
            [score_tables["exp2_qa"], score_tables["qonly_bgem3"], score_tables["gemma_qa"]],
            val[acol].tolist(),
            val[gcol].tolist(),
            "blend_exp2_qa_qonly_bgem3_gemma_qa",
        )
        all_metrics["blend_exp2_qa_qonly_bgem3_gemma_qa"] = metric
        prediction_tables["blend_exp2_qa_qonly_bgem3_gemma_qa"] = preds
    if "qonly_bgem3" in score_tables and "gemma_qonly" in score_tables:
        metric, preds = blend_scores(
            val_cands,
            [score_tables["qonly_bgem3"], score_tables["gemma_qonly"]],
            val[acol].tolist(),
            val[gcol].tolist(),
            "blend_qonly_bgem3_plus_gemma_qonly",
        )
        all_metrics["blend_qonly_bgem3_plus_gemma_qonly"] = metric
        prediction_tables["blend_qonly_bgem3_plus_gemma_qonly"] = preds

    pd.DataFrame(prediction_tables).to_csv(out_dir / "val_predictions.csv", index=False)
    summary = {
        "experiment": "exp14_qonly_bgem3_plus_bge_gemma_zero_shot",
        "gpu": "L40S",
        "k": k,
        "train_pairs_per_query": train_pairs_per_query,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "learning_rate": lr,
        "qonly_train_seconds": train_seconds,
        "candidate_baseline": baseline_metrics,
        "metrics": all_metrics,
        "pair_counts_by_subset": per_subset_pair_counts,
        "notes": [
            "qonly_bgem3 is fine-tuned with candidate question only, labels still come from candidate-answer ROUGE against the reference answer.",
            "gemma scores are pretrained zero-shot reranking passes, not task fine-tuned.",
            "blend metrics are per-row z-score sums over candidate scores.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(upload: bool = False, score_gemma: bool = True):
    local_root = Path(__file__).resolve().parent
    if upload:
        print(f"Uploading data + adapters/models to Modal volume {VOLUME_NAME}...")
        with volume.batch_upload(force=True) as batch:
            batch.put_file(local_root / "Train.csv", "/Train.csv")
            batch.put_file(local_root / "Val.csv", "/Val.csv")
            batch.put_directory(local_root / "Bgem3-finetune" / "bge-m3-health-qa" / "final", "/bge_m3_adapter")
        print("Upload complete.")
    call = run_exp14.spawn(score_gemma=score_gemma)
    print(f"Spawned exp14 q-only + Gemma reranker call: {call.object_id}")
