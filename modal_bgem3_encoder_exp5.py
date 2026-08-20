from pathlib import Path

import modal


APP_NAME = "lalang-bgem3-encoder-exp5"
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
    timeout=60 * 60 * 10,
    volumes={str(REMOTE_ROOT): volume},
)
def run_encoder_exp5(
    dense_k: int = 200,
    lexical_k: int = 50,
    eval_k: int = 50,
    pairs_per_anchor: int = 4,
    epochs: int = 2,
    batch_size: int = 32,
    grad_accum: int = 1,
    lr: float = 1.5e-5,
    max_seq_length: int = 256,
):
    import gc
    import json
    import random
    import time
    from collections import Counter, defaultdict

    import numpy as np
    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from sklearn.feature_extraction.text import TfidfVectorizer
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
    out_dir = REMOTE_ROOT / "exp5_bgem3_encoder_mining_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    class WhitespaceTokenizer:
        def tokenize(self, text):
            return [] if text is None else str(text).strip().split()

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

    print("Reading data...", flush=True)
    train = pd.read_csv(REMOTE_ROOT / "Train.csv")
    val = pd.read_csv(REMOTE_ROOT / "Val.csv")
    for df in (train, val):
        for c in (qcol, acol, gcol):
            df[c] = df[c].fillna("").astype(str).str.strip()
    train = train[(train[qcol] != "") & (train[acol] != "")].reset_index(drop=True)
    val = val[(val[qcol] != "") & (val[acol] != "")].reset_index(drop=True)
    print(f"train={len(train):,} val={len(val):,}", flush=True)
    print(train[gcol].value_counts().sort_index().to_string(), flush=True)

    print("\nLoading baseline BGE-M3 for mining...", flush=True)
    miner = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
    miner.max_seq_length = max_seq_length
    train_queries = train[qcol].tolist()
    train_answers = train[acol].tolist()
    train_subsets = train[gcol].tolist()
    print("Encoding train queries with baseline BGE-M3...", flush=True)
    train_embs = miner.encode(
        train_queries,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("\nBuilding dense and lexical mining pools...", flush=True)
    candidate_pool = [set() for _ in range(len(train))]
    dense_sources = np.zeros(len(train), dtype=np.int32)
    lexical_sources = np.zeros(len(train), dtype=np.int32)

    def add_candidate(anchor_idx, cand_idx, source):
        if anchor_idx == cand_idx:
            return
        candidate_pool[anchor_idx].add(int(cand_idx))
        if source == "dense":
            dense_sources[anchor_idx] += 1
        else:
            lexical_sources[anchor_idx] += 1

    for subset, grp in tqdm(list(train.groupby(gcol)), desc="Dense topK per subset"):
        idxs = grp.index.to_numpy()
        embs = train_embs[idxs]
        nn = NearestNeighbors(n_neighbors=min(dense_k + 1, len(idxs)), metric="cosine").fit(embs)
        _, mat = nn.kneighbors(embs)
        for local_i, row in enumerate(mat):
            anchor = int(idxs[local_i])
            for local_j in row:
                cand = int(idxs[int(local_j)])
                add_candidate(anchor, cand, "dense")

    def lexical_mine(source_name, train_field, char=False):
        for subset, grp in tqdm(list(train.groupby(gcol)), desc=source_name):
            idxs = grp.index.to_numpy()
            texts = grp[train_field].tolist()
            vec = TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                analyzer="char_wb" if char else "word",
                ngram_range=(3, 5) if char else (1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
                norm="l2",
            )
            x = vec.fit_transform(texts)
            y_texts = grp[qcol].tolist()
            y = vec.transform(y_texts)
            nn = NearestNeighbors(n_neighbors=min(lexical_k + 1, len(idxs)), metric="cosine", algorithm="brute")
            nn.fit(x)
            _, mat = nn.kneighbors(y)
            for local_i, row in enumerate(mat):
                anchor = int(idxs[local_i])
                for local_j in row:
                    cand = int(idxs[int(local_j)])
                    add_candidate(anchor, cand, "lexical")

    lexical_mine("TFIDF word q2q mining", qcol, char=False)
    lexical_mine("TFIDF char q2q mining", qcol, char=True)
    lexical_mine("TFIDF word q2a mining", acol, char=False)
    lexical_mine("TFIDF char q2a mining", acol, char=True)

    print("\nScoring mined candidates and building contrastive rows...", flush=True)
    weak_subsets = {"Aka_Gha", "Amh_Eth", "Eng_Gha"}
    boost_subsets = {"Aka_Gha", "Amh_Eth", "Eng_Gha", "Lug_Uga"}
    rows = []
    mining_rows = []
    stats = defaultdict(int)
    rng = np.random.default_rng(seed)

    for i in tqdm(range(len(train)), desc="Build pairs"):
        cands = list(candidate_pool[i])
        if not cands:
            stats["no_candidates"] += 1
            continue
        ref = train_answers[i]
        subset = train_subsets[i]
        scored = [(j, fast_r1(train_answers[j], ref)) for j in cands]
        scored.sort(key=lambda x: -x[1])
        best_j, best_r = scored[0]
        pos_threshold = 0.30 if subset in weak_subsets else 0.40
        neg_threshold = 0.20 if subset in weak_subsets else 0.25

        positives = [(j, r) for j, r in scored if r >= pos_threshold]
        if not positives and best_r >= 0.22:
            positives = [(best_j, best_r)]
        negatives = [(j, r) for j, r in scored if r <= neg_threshold]
        if not negatives:
            # Use the bottom of the retrieved pool as fallback negatives.
            negatives = scored[-min(8, len(scored)) :]
        if not positives or not negatives:
            stats["dropped"] += 1
            continue

        positives = positives[: min(3, len(positives))]
        # Keep hard-ish negatives from retrieved pool, but avoid near-ties.
        filtered_negs = []
        for j, r in negatives:
            if positives[0][1] - r >= 0.08:
                filtered_negs.append((j, r))
        negatives = filtered_negs[: min(6, len(filtered_negs))]
        if not negatives:
            stats["no_margin_negatives"] += 1
            continue

        n_pairs = pairs_per_anchor + (2 if subset in boost_subsets else 0)
        produced = 0
        for pj, pr in positives:
            if produced >= n_pairs:
                break
            neg_order = list(range(len(negatives)))
            rng.shuffle(neg_order)
            for ni in neg_order:
                nj, nr = negatives[ni]
                rows.append(
                    {
                        "anchor": train_queries[i],
                        "positive": train_queries[pj],
                        "negative": train_queries[nj],
                        "subset": subset,
                        "pos_r1": float(pr),
                        "neg_r1": float(nr),
                    }
                )
                produced += 1
                if produced >= n_pairs:
                    break
        mining_rows.append(
            {
                "idx": i,
                "subset": subset,
                "candidate_count": len(cands),
                "best_r1": float(best_r),
                "positives": len(positives),
                "negatives": len(negatives),
                "pairs": produced,
            }
        )
        stats["ok"] += 1

    pair_df = pd.DataFrame(rows)
    mining_df = pd.DataFrame(mining_rows)
    pair_df.to_csv(out_dir / "train_pairs_summary_sample.csv", index=False)
    mining_df.to_csv(out_dir / "mining_anchor_summary.csv", index=False)
    print(f"Training rows: {len(pair_df):,}", flush=True)
    print("Anchor stats:", dict(stats), flush=True)
    print(pair_df.groupby("subset").size().sort_index().to_string(), flush=True)
    print(pair_df[["pos_r1", "neg_r1"]].describe().round(4).to_string(), flush=True)

    shuffled_pairs = pair_df[["anchor", "positive", "negative"]].sample(frac=1.0, random_state=seed)
    train_ds = Dataset.from_dict(
        {
            "anchor": shuffled_pairs["anchor"].astype(str).tolist(),
            "positive": shuffled_pairs["positive"].astype(str).tolist(),
            "negative": shuffled_pairs["negative"].astype(str).tolist(),
        }
    )
    del pair_df, rows, candidate_pool, train_embs, miner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nLoading fresh BGE-M3 + LoRA for training...", flush=True)
    model = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = max_seq_length
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["query", "key", "value", "dense"],
        bias="none",
    )
    model.add_adapter(lora_config)
    try:
        model[0].auto_model.gradient_checkpointing_enable()
    except Exception:
        pass
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    args = SentenceTransformerTrainingArguments(
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
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        seed=seed,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds, loss=train_loss)

    print("\nStarting encoder training...", flush=True)
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    print(f"Encoder trained in {train_seconds / 3600:.2f} hours", flush=True)
    model.save_pretrained(str(out_dir / "final"))

    @torch.no_grad()
    def evaluate_encoder(eval_model, train_df, val_df, k):
        eval_model.eval()
        rows = []
        for subset, tr_grp in tqdm(list(train_df.groupby(gcol)), desc="Eval subsets"):
            va_grp = val_df[val_df[gcol] == subset]
            if va_grp.empty:
                continue
            tr_q = tr_grp[qcol].tolist()
            tr_a = tr_grp[acol].tolist()
            va_q = va_grp[qcol].tolist()
            va_a = va_grp[acol].tolist()
            tr_emb = eval_model.encode(
                tr_q,
                batch_size=128,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            va_emb = eval_model.encode(
                va_q,
                batch_size=128,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            nn = NearestNeighbors(n_neighbors=min(k, len(tr_grp)), metric="cosine").fit(tr_emb)
            _, mat = nn.kneighbors(va_emb)
            for row_idx, idxs in zip(va_grp.index.tolist(), mat):
                ref = val_df.at[row_idx, acol]
                top1_answer = tr_a[int(idxs[0])]
                top1_r1 = fast_r1(top1_answer, ref)
                oracle_r1 = max(fast_r1(tr_a[int(j)], ref) for j in idxs)
                rows.append({"ID": val_df.at[row_idx, idcol], "subset": subset, "top1_r1": top1_r1, "oracle_r1": oracle_r1})
        df = pd.DataFrame(rows)
        per = df.groupby("subset")[["top1_r1", "oracle_r1"]].mean().round(4)
        out = {
            "top1_r1": float(df["top1_r1"].mean()),
            "oracle_r1": float(df["oracle_r1"].mean()),
            "per_subset": per.to_dict(orient="index"),
        }
        df.to_csv(out_dir / "val_encoder_eval_rows.csv", index=False)
        print(json.dumps(out, indent=2), flush=True)
        return out

    print(f"\nEvaluating exp5 encoder top1/oracle@{eval_k}...", flush=True)
    eval_metrics = evaluate_encoder(model, train, val, eval_k)

    summary = {
        "experiment": "exp5_bgem3_encoder_mining_v2",
        "gpu": "L40S",
        "dense_k": dense_k,
        "lexical_k": lexical_k,
        "eval_k": eval_k,
        "pairs_per_anchor": pairs_per_anchor,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "learning_rate": lr,
        "max_seq_length": max_seq_length,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "train_seconds": train_seconds,
        "train_examples": int(len(train_ds)),
        "anchor_stats": dict(stats),
        "eval": eval_metrics,
        "delta_top1_vs_exp2_encoder": eval_metrics["top1_r1"] - 0.5395458277838778,
        "delta_oracle50_vs_exp2_encoder": eval_metrics["oracle_r1"] - 0.6836959744084082,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(
    upload: bool = False,
    dense_k: int = 200,
    lexical_k: int = 50,
    eval_k: int = 50,
    epochs: int = 2,
    batch_size: int = 32,
    grad_accum: int = 1,
):
    local_root = Path(__file__).resolve().parent
    if upload:
        print(f"Uploading Train/Val/Test to Modal volume {VOLUME_NAME}...")
        with volume.batch_upload(force=True) as batch:
            batch.put_file(local_root / "Train.csv", "/Train.csv")
            batch.put_file(local_root / "Val.csv", "/Val.csv")
            batch.put_file(local_root / "Test.csv", "/Test.csv")
        print("Upload complete.")
    summary = run_encoder_exp5.remote(
        dense_k=dense_k,
        lexical_k=lexical_k,
        eval_k=eval_k,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
    )
    print(summary)
