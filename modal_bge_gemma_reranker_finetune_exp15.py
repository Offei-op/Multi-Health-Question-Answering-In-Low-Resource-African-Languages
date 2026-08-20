from pathlib import Path

import modal


APP_NAME = "lalang-bge-gemma-reranker-finetune-exp15"
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
    timeout=60 * 60 * 10,
    volumes={str(REMOTE_ROOT): volume},
)
def run_gemma_reranker_finetune(
    k: int = 50,
    train_pairs_per_query: int = 12,
    epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 16,
    lr: float = 5e-6,
    max_length: int = 384,
    score_batch_size: int = 16,
):
    import gc
    import json
    import pickle
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
    model_name = "BAAI/bge-reranker-v2-gemma"
    out_dir = REMOTE_ROOT / "exp15_bge_gemma_reranker_finetune"
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

    def candidate_text(q, a):
        return f"Candidate question: {q}\nCandidate answer: {a}"

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

    rng = np.random.default_rng(seed)
    pair_cache = out_dir / f"train_pairs_qa_top{k}_ppq{train_pairs_per_query}_seed{seed}.pkl"
    if pair_cache.exists():
        print(f"\nLoading cached q+a regression pairs from {pair_cache}", flush=True)
        with pair_cache.open("rb") as f:
            cached = pickle.load(f)
        pair_q = cached["pair_q"]
        pair_c = cached["pair_c"]
        pair_y = cached["pair_y"]
        per_subset_pair_counts = cached["per_subset_pair_counts"]
    else:
        print("\nBuilding q+a cross-encoder regression pairs...", flush=True)
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
                    pair_c.append(candidate_text(cs[int(j)]["q"], cs[int(j)]["a"]))
                    pair_y.append(float(labels[int(j)]))
            per_subset_pair_counts[subset] = len(pair_y) - before
        with pair_cache.open("wb") as f:
            pickle.dump(
                {
                    "pair_q": pair_q,
                    "pair_c": pair_c,
                    "pair_y": pair_y,
                    "per_subset_pair_counts": per_subset_pair_counts,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        pd.DataFrame({"query": pair_q[:5000], "candidate": pair_c[:5000], "label": pair_y[:5000]}).to_csv(
            out_dir / "train_pairs_sample.csv", index=False
        )
        volume.commit()
        print(f"Cached q+a regression pairs to {pair_cache}", flush=True)

    print("Pair counts:", per_subset_pair_counts, flush=True)
    print(f"Total pairs: {len(pair_y):,}", flush=True)
    pair_stats = pd.Series(pair_y).describe().round(4).to_dict()
    print(pd.Series(pair_y).describe().round(4).to_string(), flush=True)
    (out_dir / "pair_stats.json").write_text(
        json.dumps(
            {
                "pair_cache": str(pair_cache),
                "pair_counts_by_subset": per_subset_pair_counts,
                "total_pairs": len(pair_y),
                "target_describe": pair_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    train_ds = Dataset.from_dict({"query": pair_q, "candidate": pair_c, "label": pair_y}).shuffle(seed=seed)
    del pair_q, pair_c, pair_y, train_cands
    try:
        bi.model.cpu()
    except Exception:
        pass
    del bi
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nLoading and fine-tuning {model_name}...", flush=True)
    reranker = CrossEncoder(
        model_name,
        num_labels=1,
        max_length=max_length,
        trust_remote_code=True,
        automodel_args={"torch_dtype": torch.bfloat16},
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
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=["none"],
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        seed=seed,
    )
    trainer = CrossEncoderTrainer(model=reranker, args=args, train_dataset=train_ds, loss=loss)
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    print(f"BGE Gemma reranker trained in {train_seconds / 3600:.2f} hours", flush=True)
    reranker.save_pretrained(str(out_dir / "final"))

    print("\nScoring validation candidates with fine-tuned Gemma reranker...", flush=True)
    flat_pairs, row_lens = [], []
    for q, cs in zip(val[qcol].tolist(), val_cands):
        flat_pairs.extend([(q, candidate_text(c["q"], c["a"])) for c in cs])
        row_lens.append(len(cs))
    print(f"Scoring {len(flat_pairs):,} validation pairs...", flush=True)
    scores = reranker.predict(flat_pairs, batch_size=score_batch_size, show_progress_bar=True, convert_to_numpy=True)
    rerank_preds, off = [], 0
    for cs, n in zip(val_cands, row_lens):
        if n == 0:
            rerank_preds.append("")
            continue
        row_scores = np.asarray(scores[off : off + n], dtype=np.float32).reshape(-1)
        off += n
        rerank_preds.append(cs[int(np.argmax(row_scores))]["a"])

    gemma_metrics = score_preds(rerank_preds, val[acol].tolist(), val[gcol].tolist(), "finetuned_bge_gemma_qa")
    pd.DataFrame(
        {
            "ID": val[idcol].tolist(),
            "subset": val[gcol].tolist(),
            "top1": val_top1,
            "rerank": rerank_preds,
            "oracle": val_oracle,
            "reference": val[acol].tolist(),
        }
    ).to_csv(out_dir / "val_predictions.csv", index=False)

    summary = {
        "experiment": "exp15_bge_gemma_reranker_finetune_qa_regression",
        "base_model": model_name,
        "candidate_encoder": "fine_tuned_BAAI/bge-m3_adapter",
        "gpu": "L40S",
        "k": k,
        "train_pairs_per_query": train_pairs_per_query,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "learning_rate": lr,
        "max_length": max_length,
        "pair_cache": str(pair_cache),
        "train_seconds": train_seconds,
        "candidate_baseline": baseline_metrics,
        "top1": top1_metrics,
        "gemma_rerank": gemma_metrics,
        "oracle": oracle_metrics,
        "delta_gemma_vs_top1": gemma_metrics["rouge1"] - top1_metrics["rouge1"],
        "delta_gemma_vs_exp2_reference": gemma_metrics["rouge1"] - 0.5892166283468145,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(
    upload: bool = False,
    k: int = 50,
    train_pairs_per_query: int = 12,
    epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 16,
    lr: float = 5e-6,
    max_length: int = 384,
):
    local_root = Path(__file__).resolve().parent
    if upload:
        print(f"Uploading data + BGE adapter to Modal volume {VOLUME_NAME}...")
        with volume.batch_upload(force=True) as batch:
            batch.put_file(local_root / "Train.csv", "/Train.csv")
            batch.put_file(local_root / "Val.csv", "/Val.csv")
            batch.put_file(local_root / "Test.csv", "/Test.csv")
            batch.put_directory(local_root / "Bgem3-finetune" / "bge-m3-health-qa" / "final", "/bge_m3_adapter")
        print("Upload complete.")
    summary = run_gemma_reranker_finetune.remote(
        k=k,
        train_pairs_per_query=train_pairs_per_query,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        max_length=max_length,
    )
    print(summary)
