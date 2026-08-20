from pathlib import Path

import modal


APP_NAME = "lalang-qwen3-cluster-listwise-reranker-exp18"
VOLUME_NAME = "lalang-bgem3-rerank"
REMOTE_ROOT = Path("/data")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "transformers>=4.51.0,<5.0.0",
        "accelerate>=1.0.0",
        "peft>=0.15.0",
        "bitsandbytes>=0.46.0",
        "sentence-transformers>=5.1.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "pyarrow>=17.0.0",
        "tqdm>=4.66.0",
        "safetensors>=0.4.3",
        "sentencepiece>=0.2.0",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 12,
    volumes={str(REMOTE_ROOT): volume},
)
def run(
    model_name: str = "Qwen/Qwen3-Reranker-0.6B",
    candidate_k: int = 50,
    eval_k: int = 20,
    group_size: int = 8,
    max_train_groups: int = 12000,
    max_steps: int = 1200,
    grad_accum: int = 8,
    learning_rate: float = 1.0e-4,
    max_length: int = 512,
    encode_batch_size: int = 128,
    score_batch_size: int = 48,
    zero_shot_k: int = 10,
    reuse_group_cache: bool = True,
    reuse_adapter: bool = False,
):
    """Fine-tune Qwen3-Reranker with a listwise loss over answer clusters."""
    import gc
    import json
    import math
    import pickle
    import random
    import re
    import time
    import unicodedata
    from collections import Counter, defaultdict

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from sentence_transformers import SentenceTransformer
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    out_dir = REMOTE_ROOT / "exp18_qwen3_cluster_listwise_reranker"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"groups_k{candidate_k}_g{group_size}_seed{seed}.pkl"
    adapter_dir = out_dir / "adapter"

    whitespace = re.compile(r"\s+")

    def answer_key(text):
        text = unicodedata.normalize("NFKC", str(text))
        return whitespace.sub(" ", text).strip().casefold()

    def clean(df):
        for c in (qcol, acol, gcol, idcol):
            if c in df:
                df[c] = df[c].fillna("").astype(str).str.strip()
        return df[
            df[qcol].ne("") & df[acol].ne("") & df[gcol].ne("")
        ].reset_index(drop=True)

    def rouge1_fast(pred, ref):
        pt = str(pred).strip().split()
        rt = str(ref).strip().split()
        if not pt or not rt:
            return 0.0
        overlap = sum((Counter(pt) & Counter(rt)).values())
        return 0.0 if overlap == 0 else float(2.0 * overlap / (len(pt) + len(rt)))

    def top_indices(scores, k):
        k = min(int(k), len(scores))
        if k == len(scores):
            idx = np.arange(len(scores))
        else:
            idx = np.argpartition(-scores, k - 1)[:k]
        return idx[np.argsort(-scores[idx], kind="stable")]

    def reduce_max_by_cluster(row_scores, cluster_ids, n_clusters):
        out = np.full((row_scores.shape[0], n_clusters), -np.inf, dtype=np.float32)
        for i in range(row_scores.shape[0]):
            np.maximum.at(out[i], cluster_ids, row_scores[i])
        return out

    def candidate_document(answer, related_questions):
        related = "\n".join(f"- {q}" for q in related_questions[:3])
        return f"Candidate answer:\n{answer}\n\nQuestions previously answered this way:\n{related}"

    def metric_bundle(preds, refs, subsets, label):
        df = pd.DataFrame(
            {
                "subset": subsets,
                "rouge1": [rouge1_fast(p, r) for p, r in zip(preds, refs)],
            }
        )
        return {
            "label": label,
            "rouge1": float(df.rouge1.mean()),
            "per_subset": df.groupby("subset").rouge1.mean().round(6).to_dict(),
        }

    print("Reading data...", flush=True)
    train = clean(pd.read_csv(REMOTE_ROOT / "Train.csv"))
    val = clean(pd.read_csv(REMOTE_ROOT / "Val.csv"))
    print(f"train={len(train):,} val={len(val):,}", flush=True)

    if reuse_group_cache and cache_path.exists():
        print(f"Loading cached groups from {cache_path}", flush=True)
        with cache_path.open("rb") as f:
            cached = pickle.load(f)
        train_groups = cached["train_groups"]
        val_groups = cached["val_groups"]
        mining_summary = cached["mining_summary"]
    else:
        print("Loading exp5 fine-tuned BGE-M3 for cluster mining...", flush=True)
        encoder_path = REMOTE_ROOT / "exp5_bgem3_encoder_mining_v2" / "final"
        if not encoder_path.exists():
            raise FileNotFoundError(f"Missing encoder: {encoder_path}")
        encoder = SentenceTransformer(
            "BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu"
        )
        encoder[0].auto_model = PeftModel.from_pretrained(
            encoder[0].auto_model, str(encoder_path), is_trainable=False
        )
        encoder[0].auto_model.eval()
        encoder.max_seq_length = 512

        all_train_groups = []
        val_groups = []
        mining_rows = []

        for subset in sorted(val[gcol].unique()):
            tr = train.loc[train[gcol].eq(subset)].reset_index(drop=True)
            va = val.loc[val[gcol].eq(subset)].reset_index(drop=True)
            keys = tr[acol].map(answer_key)
            cluster_ids, unique_keys = pd.factorize(keys, sort=False)
            cluster_ids = cluster_ids.astype(np.int64, copy=False)
            n_clusters = len(unique_keys)
            first_rows = np.full(n_clusters, -1, dtype=np.int64)
            members = [[] for _ in range(n_clusters)]
            for row_i, cluster_i in enumerate(cluster_ids):
                if first_rows[cluster_i] < 0:
                    first_rows[cluster_i] = row_i
                members[cluster_i].append(row_i)
            answers = tr[acol].to_numpy(dtype=object)[first_rows]
            questions = tr[qcol].to_numpy(dtype=object)
            qa_docs = [
                f"Question: {q}\nAnswer: {a}"
                for q, a in zip(tr[qcol].tolist(), tr[acol].tolist())
            ]

            print(f"\nEncoding {subset}: rows={len(tr):,} clusters={n_clusters:,}", flush=True)
            q_emb = encoder.encode(
                tr[qcol].tolist(), batch_size=encode_batch_size,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=True,
            ).astype(np.float32, copy=False)
            qa_emb = encoder.encode(
                qa_docs, batch_size=encode_batch_size,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=True,
            ).astype(np.float32, copy=False)
            val_emb = encoder.encode(
                va[qcol].tolist(), batch_size=encode_batch_size,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=True,
            ).astype(np.float32, copy=False)

            def make_group(query, ref, row_q_scores, row_qa_scores, row_self=None, is_train=False):
                if row_self is not None:
                    row_q_scores = row_q_scores.copy()
                    row_qa_scores = row_qa_scores.copy()
                    row_q_scores[row_self] = -np.inf
                    row_qa_scores[row_self] = -np.inf
                qmax = np.full(n_clusters, -np.inf, dtype=np.float32)
                qamax = np.full(n_clusters, -np.inf, dtype=np.float32)
                np.maximum.at(qmax, cluster_ids, row_q_scores)
                np.maximum.at(qamax, cluster_ids, row_qa_scores)
                fused = 0.5 * qmax + 0.5 * qamax
                ranked = top_indices(fused, candidate_k)
                cands = []
                for cluster_i in ranked:
                    cluster_i = int(cluster_i)
                    member_idx = np.asarray(members[cluster_i], dtype=np.int64)
                    member_idx = member_idx[np.argsort(-row_q_scores[member_idx], kind="stable")]
                    related = [
                        str(questions[j]) for j in member_idx
                        if row_self is None or int(j) != int(row_self)
                    ][:3]
                    answer = str(answers[cluster_i])
                    cands.append(
                        {
                            "answer": answer,
                            "document": candidate_document(answer, related),
                            "retrieval_score": float(fused[cluster_i]),
                            "label": float(rouge1_fast(answer, ref)),
                            "cluster": cluster_i,
                        }
                    )
                if not is_train:
                    return {"query": str(query), "candidates": cands}

                labels = np.asarray([c["label"] for c in cands], dtype=np.float32)
                if len(labels) < group_size or float(labels.max()) < 0.10:
                    return None
                label_order = np.argsort(-labels)
                chosen = []
                chosen.extend(label_order[:2].tolist())
                chosen.extend(range(min(4, len(cands))))
                hard_pool = [i for i in range(4, min(30, len(cands)))]
                rng = np.random.default_rng(seed + (0 if row_self is None else int(row_self)))
                if hard_pool:
                    rng.shuffle(hard_pool)
                    chosen.extend(hard_pool)
                seen = set()
                chosen = [i for i in chosen if not (i in seen or seen.add(i))]
                if len(chosen) < group_size:
                    chosen.extend(i for i in range(len(cands)) if i not in seen)
                selected = [cands[i] for i in chosen[:group_size]]
                return {"query": str(query), "candidates": selected}

            chunk = 96
            subset_train_groups = []
            for start in tqdm(range(0, len(tr), chunk), desc=f"Mine train {subset}"):
                stop = min(start + chunk, len(tr))
                sq = q_emb[start:stop] @ q_emb.T
                sqa = q_emb[start:stop] @ qa_emb.T
                for local_i in range(stop - start):
                    i = start + local_i
                    group = make_group(
                        tr.at[i, qcol], tr.at[i, acol], sq[local_i], sqa[local_i],
                        row_self=i, is_train=True,
                    )
                    if group is not None:
                        group.update({"subset": subset, "ID": tr.at[i, idcol]})
                        subset_train_groups.append(group)

            for start in tqdm(range(0, len(va), chunk), desc=f"Mine val {subset}"):
                stop = min(start + chunk, len(va))
                sq = val_emb[start:stop] @ q_emb.T
                sqa = val_emb[start:stop] @ qa_emb.T
                for local_i in range(stop - start):
                    i = start + local_i
                    group = make_group(
                        va.at[i, qcol], va.at[i, acol], sq[local_i], sqa[local_i],
                        row_self=None, is_train=False,
                    )
                    group.update(
                        {
                            "subset": subset,
                            "ID": va.at[i, idcol],
                            "reference": va.at[i, acol],
                        }
                    )
                    val_groups.append(group)

            all_train_groups.extend(subset_train_groups)
            mining_rows.append(
                {
                    "subset": subset,
                    "train_rows": len(tr),
                    "clusters": n_clusters,
                    "eligible_train_groups": len(subset_train_groups),
                    "val_groups": len(va),
                }
            )
            del q_emb, qa_emb, val_emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rng = random.Random(seed)
        rng.shuffle(all_train_groups)
        if max_train_groups > 0:
            train_groups = all_train_groups[:max_train_groups]
        else:
            train_groups = all_train_groups
        mining_summary = {
            "rows": mining_rows,
            "eligible_train_groups": len(all_train_groups),
            "selected_train_groups": len(train_groups),
            "val_groups": len(val_groups),
        }
        with cache_path.open("wb") as f:
            pickle.dump(
                {
                    "train_groups": train_groups,
                    "val_groups": val_groups,
                    "mining_summary": mining_summary,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        volume.commit()
        del encoder, all_train_groups
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Mining summary:", json.dumps(mining_summary, indent=2), flush=True)
    print(f"train_groups={len(train_groups):,} val_groups={len(val_groups):,}", flush=True)

    val_groups.sort(key=lambda x: x["ID"])
    refs = [g["reference"] for g in val_groups]
    subsets = [g["subset"] for g in val_groups]
    baseline_preds = [g["candidates"][0]["answer"] for g in val_groups]
    oracle_preds = [
        max(g["candidates"][:eval_k], key=lambda c: c["label"])["answer"]
        for g in val_groups
    ]
    baseline_metrics = metric_bundle(baseline_preds, refs, subsets, "cluster_fusion_qmax_qa_top1")
    oracle_metrics = metric_bundle(oracle_preds, refs, subsets, f"cluster_oracle_top{eval_k}")

    print(f"\nLoading {model_name} in 4-bit...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    no_ids = tokenizer.encode("no", add_special_tokens=False)
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    if not no_ids or not yes_ids:
        raise RuntimeError("Tokenizer did not produce yes/no token IDs")
    token_no, token_yes = int(no_ids[-1]), int(yes_ids[-1])
    print(f"yes token={token_yes} no token={token_no}", flush=True)

    prefix = (
        "<|im_start|>system\n"
        "Judge whether the candidate answer correctly addresses the health question. "
        "The answer must be yes or no.<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    instruction = "Rank answers by how well they answer the query in the same language."

    def prompt_text(query, document):
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"

    def tokenize_prompts(prompts):
        budget = max_length - len(prefix_ids) - len(suffix_ids)
        encoded = []
        for prompt in prompts:
            middle = tokenizer.encode(prompt, add_special_tokens=False)[:budget]
            encoded.append(prefix_ids + middle + suffix_ids)
        width = max(len(x) for x in encoded)
        input_ids = []
        attention = []
        for ids in encoded:
            pad = width - len(ids)
            input_ids.append([tokenizer.pad_token_id] * pad + ids)
            attention.append([0] * pad + [1] * len(ids))
        device = next(model.parameters()).device
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
        }

    def forward_scores(prompts):
        toks = tokenize_prompts(prompts)
        logits = model(**toks, use_cache=False).logits[:, -1, :]
        return logits[:, token_yes] - logits[:, token_no]

    @torch.no_grad()
    def score_groups(groups, k, desc):
        model.eval()
        flat_prompts, row_lengths = [], []
        for group in groups:
            cands = group["candidates"][:k]
            row_lengths.append(len(cands))
            flat_prompts.extend(prompt_text(group["query"], c["document"]) for c in cands)
        scores = []
        for start in tqdm(range(0, len(flat_prompts), score_batch_size), desc=desc):
            batch = flat_prompts[start : start + score_batch_size]
            scores.extend(forward_scores(batch).float().cpu().tolist())
        return np.asarray(scores, dtype=np.float32), row_lengths

    if zero_shot_k > 0:
        zero_k = min(zero_shot_k, eval_k)
        zero_scores, zero_lengths = score_groups(val_groups, zero_k, "Zero-shot Qwen3")
        zero_preds, off = [], 0
        for group, n in zip(val_groups, zero_lengths):
            pick = int(np.argmax(zero_scores[off : off + n]))
            off += n
            zero_preds.append(group["candidates"][pick]["answer"])
        zero_metrics = metric_bundle(zero_preds, refs, subsets, f"qwen3_zero_shot_top{zero_k}")
    else:
        zero_preds = [""] * len(val_groups)
        zero_metrics = None

    if reuse_adapter and adapter_dir.exists():
        print(f"Loading existing adapter from {adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        train_seconds = 0.0
        completed_steps = 0
    else:
        print("Preparing QLoRA adapters...", flush=True)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
        model.train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
        warmup = max(1, int(max_steps * 0.05))

        def lr_factor(step):
            if step < warmup:
                return float(step + 1) / warmup
            progress = (step - warmup) / max(1, max_steps - warmup)
            return max(0.05, 1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        optimizer.zero_grad(set_to_none=True)
        order = list(range(len(train_groups)))
        rng = random.Random(seed)
        rng.shuffle(order)
        cursor = 0
        completed_steps = 0
        micro_step = 0
        running = []
        t0 = time.time()
        progress = tqdm(total=max_steps, desc="Listwise QLoRA steps")
        while completed_steps < max_steps:
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            group = train_groups[order[cursor]]
            cursor += 1
            candidates = group["candidates"][:group_size]
            prompts = [prompt_text(group["query"], c["document"]) for c in candidates]
            labels = torch.tensor(
                [c["label"] for c in candidates],
                dtype=torch.float32,
                device=next(model.parameters()).device,
            )
            scores = forward_scores(prompts).float()
            target = F.softmax(labels / 0.10, dim=0)
            loss = -(target * F.log_softmax(scores / 1.0, dim=0)).sum()
            (loss / grad_accum).backward()
            running.append(float(loss.detach().cpu()))
            micro_step += 1
            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1
                progress.update(1)
                if completed_steps % 25 == 0:
                    progress.set_postfix(
                        loss=f"{np.mean(running[-100:]):.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                if completed_steps % 100 == 0:
                    model.save_pretrained(str(adapter_dir))
                    tokenizer.save_pretrained(str(adapter_dir))
                    (out_dir / "training_state.json").write_text(
                        json.dumps(
                            {
                                "completed_steps": completed_steps,
                                "max_steps": max_steps,
                                "mean_recent_loss": float(np.mean(running[-100:])),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    volume.commit()
        progress.close()
        train_seconds = time.time() - t0
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        volume.commit()
        print(f"Training finished in {train_seconds / 3600:.2f} hours", flush=True)

    ft_scores, ft_lengths = score_groups(val_groups, eval_k, "Fine-tuned Qwen3")
    ft_preds, score_rows, off = [], [], 0
    for group, n in zip(val_groups, ft_lengths):
        row_scores = ft_scores[off : off + n]
        off += n
        pick = int(np.argmax(row_scores))
        ft_preds.append(group["candidates"][pick]["answer"])
        for rank, (candidate, score) in enumerate(
            zip(group["candidates"][:n], row_scores), start=1
        ):
            score_rows.append(
                {
                    "ID": group["ID"],
                    "subset": group["subset"],
                    "candidate_rank": rank,
                    "retrieval_score": candidate["retrieval_score"],
                    "reranker_score": float(score),
                    "candidate_r1": candidate["label"],
                    "answer": candidate["answer"],
                }
            )
    ft_metrics = metric_bundle(ft_preds, refs, subsets, f"qwen3_listwise_top{eval_k}")

    predictions = pd.DataFrame(
        {
            "ID": [g["ID"] for g in val_groups],
            "subset": subsets,
            "retrieval_top1": baseline_preds,
            "zero_shot_qwen3": zero_preds,
            "listwise_qwen3": ft_preds,
            "oracle": oracle_preds,
            "reference": refs,
        }
    )
    predictions.to_csv(out_dir / "val_predictions.csv", index=False)
    pd.DataFrame(score_rows).to_parquet(out_dir / "val_candidate_scores.parquet", index=False)

    summary = {
        "experiment": "exp18_qwen3_cluster_listwise_reranker",
        "base_model": model_name,
        "candidate_k": candidate_k,
        "eval_k": eval_k,
        "group_size": group_size,
        "max_train_groups": max_train_groups,
        "max_steps": max_steps,
        "completed_steps": completed_steps,
        "gradient_accumulation": grad_accum,
        "effective_candidates_per_step": group_size * grad_accum,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "train_seconds": train_seconds,
        "mining": mining_summary,
        "retrieval_top1": baseline_metrics,
        "zero_shot": zero_metrics,
        "listwise_qwen3": ft_metrics,
        "oracle": oracle_metrics,
        "delta_listwise_vs_retrieval": ft_metrics["rouge1"] - baseline_metrics["rouge1"],
        "delta_listwise_vs_zero_shot": (
            None if zero_metrics is None else ft_metrics["rouge1"] - zero_metrics["rouge1"]
        ),
        "delta_listwise_vs_exp2": ft_metrics["rouge1"] - 0.5892166283468145,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    volume.commit()
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


@app.local_entrypoint()
def main(
    model_name: str = "Qwen/Qwen3-Reranker-0.6B",
    candidate_k: int = 50,
    eval_k: int = 20,
    group_size: int = 8,
    max_train_groups: int = 12000,
    max_steps: int = 1200,
    grad_accum: int = 8,
    learning_rate: float = 1.0e-4,
    max_length: int = 512,
    score_batch_size: int = 48,
    zero_shot_k: int = 10,
    reuse_group_cache: bool = True,
    reuse_adapter: bool = False,
):
    summary = run.remote(
        model_name=model_name,
        candidate_k=candidate_k,
        eval_k=eval_k,
        group_size=group_size,
        max_train_groups=max_train_groups,
        max_steps=max_steps,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        max_length=max_length,
        score_batch_size=score_batch_size,
        zero_shot_k=zero_shot_k,
        reuse_group_cache=reuse_group_cache,
        reuse_adapter=reuse_adapter,
    )
    print(summary)
