from pathlib import Path

import modal


APP_NAME = "lalang-bgem3-rerank-trainval-exp6"
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
def train_trainval_reranker(
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

    qcol, acol, gcol = "input", "output", "subset"
    out_dir = REMOTE_ROOT / "exp6_crossencoder_rerank_train_val"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading Train + Val...", flush=True)
    train = pd.read_csv(REMOTE_ROOT / "Train.csv")
    val = pd.read_csv(REMOTE_ROOT / "Val.csv")
    for df in (train, val):
        for c in (qcol, acol, gcol):
            df[c] = df[c].fillna("").astype(str).str.strip()
    bank = pd.concat([train, val], ignore_index=True)
    bank = bank[(bank[qcol] != "") & (bank[acol] != "") & (bank[gcol] != "")].reset_index(drop=True)
    print(f"labeled bank=train+val={len(bank):,}", flush=True)
    print(bank[gcol].value_counts().sort_index().to_string(), flush=True)

    class WhitespaceTokenizer:
        def tokenize(self, text):
            return [] if text is None else str(text).strip().split()

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], tokenizer=WhitespaceTokenizer(), use_stemmer=False)

    def target_score(candidate_answer, reference):
        s = scorer.score(str(reference), str(candidate_answer))
        r1 = float(s["rouge1"].fmeasure)
        rl = float(s["rougeL"].fmeasure)
        return np.float32(0.75 * r1 + 0.25 * rl)

    def candidate_text(q, a):
        return f"Candidate question: {q}\nCandidate answer: {a}"

    print("\nLoading fine-tuned BGE-M3 adapter...", flush=True)
    bi = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
    bi.max_seq_length = 256
    bi[0].auto_model = PeftModel.from_pretrained(
        bi[0].auto_model,
        str(REMOTE_ROOT / "bge_m3_adapter"),
        is_trainable=False,
    )
    bi[0].auto_model.eval()

    print("\nEncoding Train+Val per-subset indices...", flush=True)
    indices = {}
    for subset, grp in tqdm(list(bank.groupby(gcol)), desc="Index bank"):
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

    print("\nRetrieving Train+Val topK with self-exclusion...", flush=True)
    bank_cands = [[] for _ in range(len(bank))]
    pos = {idx: i for i, idx in enumerate(bank.index)}
    for subset, grp in tqdm(list(bank.groupby(gcol)), desc="Retrieve bank"):
        m = indices[subset]
        q_embs = bi.encode(
            grp[qcol].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=128,
            convert_to_numpy=True,
        )
        n_neighbors = min(k + 1, len(m["a"]))
        _, idx_mat = m["nn"].kneighbors(q_embs, n_neighbors=n_neighbors)
        for row_idx, idxs in zip(grp.index, idx_mat):
            picked = []
            for j in idxs:
                if m["orig_idx"][j] == row_idx:
                    continue
                picked.append({"q": str(m["q"][j]), "a": str(m["a"][j])})
                if len(picked) >= k:
                    break
            bank_cands[pos[row_idx]] = picked

    print("\nBuilding regression pairs...", flush=True)
    rng = np.random.default_rng(seed)
    pair_q, pair_c, pair_y = [], [], []
    per_subset_pair_counts = {}
    bank_subsets = bank[gcol].to_numpy()
    for subset in sorted(bank[gcol].unique()):
        subset_rows = np.where(bank_subsets == subset)[0]
        before = len(pair_y)
        for i in tqdm(subset_rows, desc=f"Pairs {subset}"):
            cs = bank_cands[int(i)]
            if not cs:
                continue
            ref = bank[acol].iloc[int(i)]
            labels = np.array([target_score(c["a"], ref) for c in cs], dtype=np.float32)
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
                pair_q.append(str(bank[qcol].iloc[int(i)]))
                pair_c.append(candidate_text(cs[int(j)]["q"], cs[int(j)]["a"]))
                pair_y.append(float(labels[int(j)]))
        per_subset_pair_counts[subset] = len(pair_y) - before
    print("Pair counts:", per_subset_pair_counts, flush=True)
    print(f"Total pairs: {len(pair_y):,}", flush=True)
    s = pd.Series(pair_y)
    print(s.describe().round(4).to_string(), flush=True)
    print(f"targets >= .5: {(s >= .5).mean() * 100:.1f}% | >= .9: {(s >= .9).mean() * 100:.1f}%", flush=True)

    train_ds = Dataset.from_dict({"query": pair_q, "candidate": pair_c, "label": pair_y}).shuffle(seed=seed)
    del pair_q, pair_c, pair_y, bank_cands, indices, bi
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
    print("\nStarting Train+Val cross-encoder training...", flush=True)
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    print(f"Cross-encoder trained in {train_seconds / 3600:.2f} hours", flush=True)
    final_dir = out_dir / "final"
    reranker.save_pretrained(str(final_dir))

    summary = {
        "experiment": "exp6_train_val_crossencoder_rouge_regression",
        "gpu": "L40S",
        "k": k,
        "labeled_bank_rows": int(len(bank)),
        "train_pairs_per_query": train_pairs_per_query,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "learning_rate": lr,
        "train_seconds": train_seconds,
        "pairs": int(len(train_ds)),
        "pair_counts_by_subset": per_subset_pair_counts,
        "final_model": str(final_dir),
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
    call = train_trainval_reranker.spawn(
        k=k,
        train_pairs_per_query=train_pairs_per_query,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
    )
    print(f"Spawned Train+Val reranker call: {call.object_id}")
