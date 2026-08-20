from pathlib import Path

import modal


APP_NAME = "lalang-bgem3-crossencoder-rerank"
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
    )
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 8,
    volumes={str(REMOTE_ROOT): volume},
)
def run_reranker_experiment(
    k: int = 50,
    train_pairs_per_query: int = 12,
    epochs: int = 1,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 1e-5,
):
    import gc
    import json
    import random
    import time

    import numpy as np
    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from rouge_score import rouge_scorer
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
    out_dir = REMOTE_ROOT / "exp2_crossencoder_rerank"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading data...", flush=True)
    train = pd.read_csv(REMOTE_ROOT / "Train.csv")
    val = pd.read_csv(REMOTE_ROOT / "Val.csv")
    test = pd.read_csv(REMOTE_ROOT / "Test.csv")
    for df in (train, val, test):
        for c in (qcol, gcol):
            df[c] = df[c].fillna("").astype(str).str.strip()
        if acol in df.columns:
            df[acol] = df[acol].fillna("").astype(str).str.strip()
    train = train[(train[qcol] != "") & (train[acol] != "")].reset_index(drop=True)
    val = val[(val[qcol] != "") & (val[acol] != "")].reset_index(drop=True)
    print(f"train={len(train):,} val={len(val):,} test={len(test):,}", flush=True)
    print(train[gcol].value_counts().sort_index().to_string(), flush=True)

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

    def target_score(candidate_answer, reference):
        r1, rl = rouge_scores(candidate_answer, reference)
        return np.float32(0.75 * r1 + 0.25 * rl)

    def candidate_text(q, a):
        return f"Candidate question: {q}\nCandidate answer: {a}"

    print("\nLoading fine-tuned BGE-M3 adapter for candidate generation...", flush=True)
    bi = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
    bi.max_seq_length = 256
    inner = bi[0].auto_model
    bi[0].auto_model = PeftModel.from_pretrained(inner, str(REMOTE_ROOT / "bge_m3_adapter"), is_trainable=False)
    bi[0].auto_model.eval()

    def encode_subset_indices(df, name):
        indices = {}
        print(f"Encoding {name} per-subset query indices...", flush=True)
        for subset, grp in tqdm(list(df.groupby(gcol)), desc=f"Index {name}"):
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
                "embs": embs,
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
            _, idx_mat = m["nn"].kneighbors(q_embs, n_neighbors=n_neighbors)
            for row_idx, idxs in zip(grp.index, idx_mat):
                picked = []
                for j in idxs:
                    if leave_self_out and m["orig_idx"][j] == row_idx:
                        continue
                    picked.append(
                        {
                            "q": str(m["q"][j]),
                            "a": str(m["a"][j]),
                        }
                    )
                    if len(picked) >= k:
                        break
                cands[pos[row_idx]] = picked
        return cands

    train_idx = encode_subset_indices(train, "train")
    print("\nRetrieving train topK with self-exclusion...", flush=True)
    train_cands = retrieve_topk(train, train_idx, leave_self_out=True)
    print("\nRetrieving val topK from train...", flush=True)
    val_cands = retrieve_topk(val, train_idx, leave_self_out=False)

    def evaluate_candidate_lists(cands, refs, subs, label):
        rows = []
        top1_preds = []
        oracle_preds = []
        for cs, ref, subset in zip(cands, refs, subs):
            if not cs:
                top1 = ""
                oracle = ""
                top1_r1 = oracle_r1 = 0.0
            else:
                top1 = cs[0]["a"]
                scored = [(target_score(c["a"], ref), c["a"]) for c in cs]
                oracle = max(scored, key=lambda x: x[0])[1]
                top1_r1, _ = rouge_scores(top1, ref)
                oracle_r1, _ = rouge_scores(oracle, ref)
            top1_preds.append(top1)
            oracle_preds.append(oracle)
            rows.append({"subset": subset, "top1_r1": top1_r1, "oracle_r1": oracle_r1})
        df = pd.DataFrame(rows)
        per = df.groupby("subset")[["top1_r1", "oracle_r1"]].mean().round(4)
        out = {
            "label": label,
            "top1_r1": float(df["top1_r1"].mean()),
            "oracle_r1": float(df["oracle_r1"].mean()),
            "per_subset": per.to_dict(orient="index"),
        }
        print(json.dumps(out, indent=2), flush=True)
        return out, top1_preds, oracle_preds

    print("\nCorrected candidate-generation baseline:", flush=True)
    baseline_metrics, val_top1, val_oracle = evaluate_candidate_lists(
        val_cands, val[acol].tolist(), val[gcol].tolist(), f"ft_bgem3_per_subset_top{k}"
    )

    print("\nBuilding cross-encoder regression pairs...", flush=True)
    rng = np.random.default_rng(seed)
    pair_q, pair_c, pair_y = [], [], []
    per_subset_pair_counts = {}
    for subset in sorted(train[gcol].unique()):
        subset_rows = np.where(train[gcol].to_numpy() == subset)[0]
        before = len(pair_y)
        for i in tqdm(subset_rows, desc=f"Pairs {subset}"):
            cs = train_cands[int(i)]
            if not cs:
                continue
            ref = train[acol].iloc[int(i)]
            labels = np.array([target_score(c["a"], ref) for c in cs], dtype=np.float32)
            order = np.argsort(-labels)
            # Always include strongest, weakest, and some middle/hard examples.
            chosen = []
            chosen.extend(order[: min(4, len(order))].tolist())
            chosen.extend(order[-min(4, len(order)) :].tolist())
            mid_pool = order[4:-4] if len(order) > 8 else order
            if len(mid_pool) > 0:
                n_mid = max(0, train_pairs_per_query - len(set(chosen)))
                chosen.extend(rng.choice(mid_pool, size=min(n_mid, len(mid_pool)), replace=False).tolist())
            # Deduplicate while preserving order.
            seen = set()
            chosen = [x for x in chosen if not (x in seen or seen.add(x))]
            for j in chosen[:train_pairs_per_query]:
                pair_q.append(str(train[qcol].iloc[int(i)]))
                pair_c.append(candidate_text(cs[int(j)]["q"], cs[int(j)]["a"]))
                pair_y.append(float(labels[int(j)]))
        per_subset_pair_counts[subset] = len(pair_y) - before
    print("Pair counts:", per_subset_pair_counts, flush=True)
    print(f"Total pairs: {len(pair_y):,}", flush=True)
    s = pd.Series(pair_y)
    print(s.describe().round(4).to_string(), flush=True)
    print(f"targets >= .5: {(s >= .5).mean() * 100:.1f}% | >= .9: {(s >= .9).mean() * 100:.1f}%", flush=True)

    train_ds = Dataset.from_dict({"query": pair_q, "candidate": pair_c, "label": pair_y}).shuffle(seed=seed)
    del pair_q, pair_c, pair_y, train_cands
    gc.collect()

    print("\nFreeing bi-encoder before cross-encoder training...", flush=True)
    try:
        bi.model.cpu()
    except Exception:
        pass
    del bi
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nLoading BGE reranker cross-encoder...", flush=True)
    reranker = CrossEncoder(
        "BAAI/bge-reranker-v2-m3",
        num_labels=1,
        max_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
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
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=["none"],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=seed,
    )
    trainer = CrossEncoderTrainer(model=reranker, args=args, train_dataset=train_ds, loss=loss)
    print("\nStarting cross-encoder training...", flush=True)
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    print(f"Cross-encoder trained in {train_seconds / 3600:.2f} hours", flush=True)
    reranker.save_pretrained(str(out_dir / "final"))

    print("\nScoring validation candidates with trained reranker...", flush=True)
    flat_pairs, row_lens = [], []
    for q, cs in zip(val[qcol].tolist(), val_cands):
        flat_pairs.extend([(q, candidate_text(c["q"], c["a"])) for c in cs])
        row_lens.append(len(cs))
    scores = reranker.predict(flat_pairs, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    rerank_preds, off = [], 0
    for cs, n in zip(val_cands, row_lens):
        if n == 0:
            rerank_preds.append("")
            continue
        row_scores = scores[off : off + n]
        off += n
        rerank_preds.append(cs[int(np.argmax(row_scores))]["a"])

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

    top1_metrics = score_preds(val_top1, val[acol].tolist(), val[gcol].tolist(), "ft_bgem3_top1")
    rerank_metrics = score_preds(rerank_preds, val[acol].tolist(), val[gcol].tolist(), "crossencoder_rerank")
    oracle_metrics = score_preds(val_oracle, val[acol].tolist(), val[gcol].tolist(), "oracle_topk")

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
        "experiment": "ft_bgem3_top50_crossencoder_rouge_regression",
        "gpu": "L40S",
        "k": k,
        "train_pairs_per_query": train_pairs_per_query,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "learning_rate": lr,
        "train_seconds": train_seconds,
        "candidate_baseline": baseline_metrics,
        "top1": top1_metrics,
        "rerank": rerank_metrics,
        "oracle": oracle_metrics,
        "delta_rerank_vs_top1": rerank_metrics["rouge1"] - top1_metrics["rouge1"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(
    upload: bool = True,
    k: int = 50,
    train_pairs_per_query: int = 12,
    epochs: int = 1,
    batch_size: int = 8,
    grad_accum: int = 4,
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
    summary = run_reranker_experiment.remote(
        k=k,
        train_pairs_per_query=train_pairs_per_query,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
    )
    print(summary)
