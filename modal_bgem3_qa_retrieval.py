from pathlib import Path

import modal


APP_NAME = "lalang-bgem3-qa-retrieval"
VOLUME_NAME = "lalang-bgem3-exp1"
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


def remote_path(name: str) -> str:
    return str(REMOTE_ROOT / name)


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 6,
    volumes={str(REMOTE_ROOT): volume},
)
def train_query_to_qa_doc(
    epochs: int = 2,
    batch_size: int = 32,
    grad_accum: int = 2,
    lr: float = 1.5e-5,
    lora_r: int = 32,
    max_seq_length: int = 512,
):
    import gc
    import json
    import os
    import random
    import time
    from collections import defaultdict

    import numpy as np
    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import losses
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
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
    out_dir = REMOTE_ROOT / "exp1_query_to_qa_doc"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading data from Modal volume...", flush=True)
    train_df = pd.read_csv(REMOTE_ROOT / "Train.csv")
    val_df = pd.read_csv(REMOTE_ROOT / "Val.csv")
    for df in (train_df, val_df):
        for c in (qcol, acol, gcol):
            df[c] = df[c].fillna("").astype(str).str.strip()
    train_df = train_df[(train_df[qcol] != "") & (train_df[acol] != "")].reset_index(drop=True)
    val_df = val_df[(val_df[qcol] != "") & (val_df[acol] != "")].reset_index(drop=True)
    print(f"train={len(train_df):,} val={len(val_df):,}", flush=True)
    print(train_df[gcol].value_counts().sort_index().to_string(), flush=True)

    def make_doc(row):
        return f"Question: {row[qcol]}\nAnswer: {row[acol]}"

    train_docs = train_df.apply(make_doc, axis=1).tolist()
    val_queries = val_df[qcol].tolist()
    val_refs = val_df[acol].tolist()
    val_subsets = val_df[gcol].tolist()

    class WhitespaceTokenizer:
        def tokenize(self, text):
            return [] if text is None else str(text).strip().split()

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"],
        tokenizer=WhitespaceTokenizer(),
        use_stemmer=False,
    )

    def rouge1(reference, candidate):
        if not reference or not candidate:
            return 0.0
        return scorer.score(str(reference), str(candidate))["rouge1"].fmeasure

    @torch.no_grad()
    def evaluate(model, label, top_k=20, encode_batch_size=128):
        model.eval()
        print(f"\n=== Evaluating {label} ===", flush=True)
        print("Encoding train QA docs...", flush=True)
        doc_embs = model.encode(
            train_docs,
            batch_size=encode_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        print("Encoding val queries...", flush=True)
        query_embs = model.encode(
            val_queries,
            batch_size=encode_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        rows = []
        predictions = []
        chunk = 256
        for start in tqdm(range(0, len(query_embs), chunk), desc=f"Scoring {label}"):
            end = min(start + chunk, len(query_embs))
            sims = query_embs[start:end] @ doc_embs.T
            top_idx = np.argpartition(-sims, kth=min(top_k, sims.shape[1]) - 1, axis=1)[:, :top_k]
            for local_i, cand_idx in enumerate(top_idx):
                global_i = start + local_i
                cand_idx = cand_idx[np.argsort(-sims[local_i, cand_idx])]
                top1_j = int(cand_idx[0])
                top1_answer = train_df[acol].iloc[top1_j]
                best_oracle = max(rouge1(val_refs[global_i], train_df[acol].iloc[int(j)]) for j in cand_idx)
                r_top1 = rouge1(val_refs[global_i], top1_answer)
                predictions.append(top1_answer)
                rows.append(
                    {
                        "subset": val_subsets[global_i],
                        "top1_r1": r_top1,
                        "oracle20_r1": best_oracle,
                    }
                )
        res = pd.DataFrame(rows)
        per_subset = (
            res.groupby("subset")[["top1_r1", "oracle20_r1"]]
            .mean()
            .sort_index()
        )
        overall = {
            "label": label,
            "top1_r1": float(res["top1_r1"].mean()),
            "oracle20_r1": float(res["oracle20_r1"].mean()),
            "per_subset": per_subset.round(4).to_dict(orient="index"),
        }
        print(json.dumps(overall, indent=2), flush=True)
        pd.DataFrame(
            {
                "ID": val_df["ID"].tolist(),
                "subset": val_subsets,
                "prediction": predictions,
                "reference": val_refs,
            }
        ).to_csv(out_dir / f"{label}_val_predictions.csv", index=False)
        return overall

    print("\nLoading baseline BGE-M3...", flush=True)
    baseline = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
    baseline.max_seq_length = max_seq_length
    baseline_metrics = evaluate(baseline, "baseline_query_to_qa_doc")
    del baseline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nBuilding training pairs: query -> full QA doc...", flush=True)
    train_ds = Dataset.from_dict(
        {
            "anchor": train_df[qcol].tolist(),
            "positive": train_docs,
        }
    ).shuffle(seed=seed)
    print(train_ds, flush=True)

    print("\nLoading trainable BGE-M3 + LoRA...", flush=True)
    model = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = max_seq_length
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        target_modules=["query", "key", "value", "dense"],
        bias="none",
    )
    model.add_adapter(lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    try:
        train_loss = losses.CachedMultipleNegativesRankingLoss(model=model, mini_batch_size=16)
        loss_name = "CachedMultipleNegativesRankingLoss"
    except AttributeError:
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
        loss_name = "MultipleNegativesRankingLoss"
    print(f"Using loss: {loss_name}", flush=True)

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
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        loss=train_loss,
    )

    print("\nStarting training...", flush=True)
    t0 = time.time()
    train_output = trainer.train()
    train_seconds = time.time() - t0
    print(train_output, flush=True)
    print(f"Training finished in {train_seconds / 3600:.2f} hours", flush=True)

    final_dir = out_dir / "final"
    model.save_pretrained(str(final_dir))
    print(f"Saved model to {final_dir}", flush=True)

    ft_metrics = evaluate(model, "finetuned_query_to_qa_doc")
    summary = {
        "experiment": "query_to_qa_doc_bgem3_lora",
        "gpu": "L40S",
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "learning_rate": lr,
        "lora_r": lora_r,
        "max_seq_length": max_seq_length,
        "loss": loss_name,
        "train_seconds": train_seconds,
        "baseline": baseline_metrics,
        "finetuned": ft_metrics,
        "delta_top1_r1": ft_metrics["top1_r1"] - baseline_metrics["top1_r1"],
        "delta_oracle20_r1": ft_metrics["oracle20_r1"] - baseline_metrics["oracle20_r1"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(
    upload: bool = True,
    epochs: int = 2,
    batch_size: int = 32,
    grad_accum: int = 2,
):
    local_root = Path(__file__).resolve().parent
    if upload:
        print(f"Uploading Train.csv and Val.csv to Modal volume {VOLUME_NAME}...")
        with volume.batch_upload(force=True) as batch:
            batch.put_file(local_root / "Train.csv", "/Train.csv")
            batch.put_file(local_root / "Val.csv", "/Val.csv")
        print("Upload complete.")
    summary = train_query_to_qa_doc.remote(
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
    )
    print(summary)
