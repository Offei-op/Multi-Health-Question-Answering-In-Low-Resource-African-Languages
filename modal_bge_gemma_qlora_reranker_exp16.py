from pathlib import Path

import modal


APP_NAME = "lalang-bge-gemma-qlora-reranker-exp16"
VOLUME_NAME = "lalang-bgem3-rerank"
REMOTE_ROOT = Path("/data")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "transformers>=4.46.0",
        "datasets>=3.0.0",
        "accelerate>=0.33.0",
        "peft>=0.12.0",
        "bitsandbytes>=0.46.0",
        "sentence-transformers>=5.1.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.5.0",
        "tqdm>=4.66.0",
        "safetensors>=0.4.3",
        "protobuf>=4.25.0",
        "sentencepiece>=0.2.0",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 5,
    volumes={str(REMOTE_ROOT): volume},
)
def run_gemma_qlora_reranker(
    k: int = 50,
    train_pairs_per_query: int = 12,
    max_steps: int = 1000,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 2e-4,
    max_length: int = 384,
    score_batch_size: int = 128,
    reuse_existing_adapter: bool = False,
    eval_k: int = 10,
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
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import NearestNeighbors
    from tqdm.auto import tqdm
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

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
    out_dir = REMOTE_ROOT / "exp16_bge_gemma_qlora_reranker"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_pair_cache = REMOTE_ROOT / "exp15_bge_gemma_reranker_finetune" / (
        f"train_pairs_qa_top{k}_ppq{train_pairs_per_query}_seed{seed}.pkl"
    )

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

    if not source_pair_cache.exists():
        raise FileNotFoundError(f"Missing cached pair file: {source_pair_cache}")
    print(f"Loading cached q+a pairs from {source_pair_cache}", flush=True)
    with source_pair_cache.open("rb") as f:
        pair_cache = pickle.load(f)
    pair_q = pair_cache["pair_q"]
    pair_c = pair_cache["pair_c"]
    pair_y = pair_cache["pair_y"]
    print(f"Cached pairs: {len(pair_y):,}", flush=True)
    print(pd.Series(pair_y).describe().round(4).to_string(), flush=True)

    train_ds = Dataset.from_dict({"query": pair_q, "candidate": pair_c, "labels": pair_y}).shuffle(seed=seed)
    del pair_q, pair_c, pair_y, pair_cache
    gc.collect()

    print("\nLoading tokenizer/model in 4-bit and attaching LoRA...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    adapter_dir = out_dir / "adapter"
    if reuse_existing_adapter and adapter_dir.exists():
        print(f"Reusing existing adapter at {adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        train_seconds = 0.0
    else:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        def tokenize(batch):
            toks = tokenizer(
                batch["query"],
                batch["candidate"],
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            toks["labels"] = [float(x) for x in batch["labels"]]
            return toks

        print("Tokenizing pairs...", flush=True)
        train_tok = train_ds.map(tokenize, batched=True, batch_size=2048, remove_columns=train_ds.column_names)
        collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

        args = TrainingArguments(
            output_dir=str(out_dir / "trainer"),
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            warmup_ratio=0.05,
            bf16=True,
            fp16=False,
            tf32=True,
            logging_steps=25,
            save_strategy="steps",
            save_steps=max_steps,
            save_total_limit=1,
            report_to=["none"],
            optim="paged_adamw_8bit",
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            seed=seed,
        )
        trainer = Trainer(model=model, args=args, train_dataset=train_tok, data_collator=collator)
        print("\nStarting Gemma QLoRA reranker training...", flush=True)
        t0 = time.time()
        trainer.train()
        train_seconds = time.time() - t0
        print(f"Gemma QLoRA trained in {train_seconds / 3600:.2f} hours", flush=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        volume.commit()
        del train_tok, trainer

    del train_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nBuilding validation candidates with the fine-tuned BGE-M3 encoder...", flush=True)
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
        nn = NearestNeighbors(n_neighbors=min(eval_k, len(grp)), metric="cosine").fit(embs)
        indices[subset] = {
            "nn": nn,
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
        _, idx_mat = m["nn"].kneighbors(q_embs, n_neighbors=min(eval_k, len(m["a"])))
        for row_idx, idxs in zip(grp.index, idx_mat):
            val_cands[pos[row_idx]] = [{"q": str(m["q"][j]), "a": str(m["a"][j])} for j in idxs]

    top1_preds, oracle_preds = [], []
    for cs, ref in zip(val_cands, val[acol].tolist()):
        if not cs:
            top1_preds.append("")
            oracle_preds.append("")
        else:
            top1_preds.append(cs[0]["a"])
            oracle_preds.append(max(cs, key=lambda c: target_score(c["a"], ref))["a"])
    top1_metrics = score_preds(top1_preds, val[acol].tolist(), val[gcol].tolist(), "ft_bgem3_top1")
    oracle_metrics = score_preds(oracle_preds, val[acol].tolist(), val[gcol].tolist(), "oracle_topk")

    try:
        bi.model.cpu()
    except Exception:
        pass
    del bi
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nScoring validation candidates with QLoRA Gemma reranker...", flush=True)
    model.eval()
    flat_q, flat_c, row_lens = [], [], []
    for q, cs in zip(val[qcol].tolist(), val_cands):
        flat_q.extend([q] * len(cs))
        flat_c.extend([candidate_text(c["q"], c["a"]) for c in cs])
        row_lens.append(len(cs))
    scores = []
    for start in tqdm(range(0, len(flat_q), score_batch_size), desc="Score val pairs"):
        batch_q = flat_q[start : start + score_batch_size]
        batch_c = flat_c[start : start + score_batch_size]
        toks = tokenizer(
            batch_q,
            batch_c,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        toks = {kk: vv.to(model.device) for kk, vv in toks.items()}
        with torch.no_grad():
            logits = model(**toks).logits.reshape(-1).float().detach().cpu().numpy()
        scores.extend(logits.tolist())

    rerank_preds, off = [], 0
    scores_arr = np.asarray(scores, dtype=np.float32)
    for cs, n in zip(val_cands, row_lens):
        if n == 0:
            rerank_preds.append("")
            continue
        row_scores = scores_arr[off : off + n]
        off += n
        rerank_preds.append(cs[int(np.argmax(row_scores))]["a"])
    qlora_metrics = score_preds(rerank_preds, val[acol].tolist(), val[gcol].tolist(), "bge_gemma_qlora_qa")

    pd.DataFrame(
        {
            "ID": val[idcol].tolist(),
            "subset": val[gcol].tolist(),
            "top1": top1_preds,
            "rerank": rerank_preds,
            "oracle": oracle_preds,
            "reference": val[acol].tolist(),
        }
    ).to_csv(out_dir / "val_predictions.csv", index=False)

    summary = {
        "experiment": "exp16_bge_gemma_qlora_reranker",
        "base_model": model_name,
        "gpu": "L40S",
        "k": k,
        "eval_k": eval_k,
        "train_pairs_per_query": train_pairs_per_query,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "learning_rate": lr,
        "max_length": max_length,
        "score_batch_size": score_batch_size,
        "reuse_existing_adapter": reuse_existing_adapter,
        "pair_cache": str(source_pair_cache),
        "train_seconds": train_seconds,
        "top1": top1_metrics,
        "gemma_qlora_rerank": qlora_metrics,
        "oracle": oracle_metrics,
        "delta_gemma_qlora_vs_top1": qlora_metrics["rouge1"] - top1_metrics["rouge1"],
        "delta_gemma_qlora_vs_exp2_reference": qlora_metrics["rouge1"] - 0.5892166283468145,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(
    max_steps: int = 1000,
    k: int = 50,
    train_pairs_per_query: int = 12,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 2e-4,
    max_length: int = 384,
    score_batch_size: int = 128,
    reuse_existing_adapter: bool = False,
    eval_k: int = 10,
):
    summary = run_gemma_qlora_reranker.remote(
        k=k,
        train_pairs_per_query=train_pairs_per_query,
        max_steps=max_steps,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        max_length=max_length,
        score_batch_size=score_batch_size,
        reuse_existing_adapter=reuse_existing_adapter,
        eval_k=eval_k,
    )
    print(summary)
