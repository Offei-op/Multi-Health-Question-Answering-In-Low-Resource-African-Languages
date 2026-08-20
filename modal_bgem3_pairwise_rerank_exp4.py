from pathlib import Path

import modal


APP_NAME = "lalang-bgem3-pairwise-rerank-exp4"
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
def run_pairwise_experiment(
    k: int = 50,
    pairs_per_query: int = 6,
    epochs: int = 1,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 1e-5,
    max_length: int = 512,
):
    import gc
    import json
    import random
    import time
    from collections import Counter

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from rouge_score import rouge_scorer
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from sklearn.neighbors import NearestNeighbors
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

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
    out_dir = REMOTE_ROOT / "exp4_pairwise_hardneg_rerank"
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

    def fast_rouge1_f1(pred, ref):
        pred_toks = str(pred).strip().split()
        ref_toks = str(ref).strip().split()
        if not pred_toks or not ref_toks:
            return 0.0
        pred_counts = Counter(pred_toks)
        ref_counts = Counter(ref_toks)
        overlap = sum(min(pred_counts[t], ref_counts[t]) for t in pred_counts)
        if overlap == 0:
            return 0.0
        precision = overlap / len(pred_toks)
        recall = overlap / len(ref_toks)
        return float(2 * precision * recall / (precision + recall))

    def target_score(candidate_answer, reference):
        return np.float32(fast_rouge1_f1(candidate_answer, reference))

    def fast_r1(pred, ref):
        return fast_rouge1_f1(pred, ref)

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
            dists, idx_mat = m["nn"].kneighbors(q_embs, n_neighbors=n_neighbors)
            sims = 1.0 - dists
            for row_idx, idxs, row_sims in zip(grp.index, idx_mat, sims):
                picked = []
                for j, sim in zip(idxs, row_sims):
                    if leave_self_out and m["orig_idx"][j] == row_idx:
                        continue
                    picked.append(
                        {
                            "q": str(m["q"][j]),
                            "a": str(m["a"][j]),
                            "bi_score": float(sim),
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
        oracle_ranks = []
        for cs, ref, subset in zip(cands, refs, subs):
            if not cs:
                top1 = ""
                oracle = ""
                top1_r1 = oracle_r1 = 0.0
                oracle_rank = 0
            else:
                top1 = cs[0]["a"]
                labels = np.array([target_score(c["a"], ref) for c in cs], dtype=np.float32)
                best_idx = int(np.argmax(labels))
                oracle = cs[best_idx]["a"]
                oracle_rank = best_idx + 1
                top1_r1, _ = rouge_scores(top1, ref)
                oracle_r1, _ = rouge_scores(oracle, ref)
            top1_preds.append(top1)
            oracle_preds.append(oracle)
            oracle_ranks.append(oracle_rank)
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
        return out, top1_preds, oracle_preds, oracle_ranks

    print("\nCorrected candidate-generation baseline:", flush=True)
    baseline_metrics, val_top1, val_oracle, val_oracle_ranks = evaluate_candidate_lists(
        val_cands, val[acol].tolist(), val[gcol].tolist(), f"ft_bgem3_per_subset_top{k}"
    )

    print("\nBuilding weighted pairwise hard-negative examples...", flush=True)
    rng = np.random.default_rng(seed)
    rows = []
    pair_stats = []
    per_subset_pair_counts = {}
    for subset in sorted(train[gcol].unique()):
        subset_rows = np.where(train[gcol].to_numpy() == subset)[0]
        before = len(rows)
        for i in tqdm(subset_rows, desc=f"Pairs {subset}"):
            cs = train_cands[int(i)]
            if len(cs) < 2:
                continue
            query = str(train[qcol].iloc[int(i)])
            ref = str(train[acol].iloc[int(i)])
            labels = np.array([target_score(c["a"], ref) for c in cs], dtype=np.float32)
            order = np.argsort(-labels)
            pos_idx = int(order[0])
            pos_label = float(labels[pos_idx])
            if pos_label <= 0.0:
                continue

            neg_pool = []
            # Encoder-hard negatives: top retrieved candidates that are not the oracle.
            neg_pool.extend([j for j in range(min(8, len(cs))) if j != pos_idx])
            # Label-hard negatives: candidates just below the oracle.
            neg_pool.extend([int(j) for j in order[1 : min(8, len(order))]])
            # Catastrophic contrast: bad candidates against exact-ish positives.
            if pos_label >= 0.90:
                bad = [int(j) for j in order if labels[int(j)] < 0.50 and int(j) != pos_idx]
                neg_pool.extend(bad[:4])
            # Easy anchors keep the score direction sane.
            neg_pool.extend([int(j) for j in order[-min(3, len(order)) :] if int(j) != pos_idx])

            seen = set()
            neg_pool = [j for j in neg_pool if not (j in seen or seen.add(j))]
            if len(neg_pool) > pairs_per_query:
                keep = neg_pool[: max(2, pairs_per_query // 2)]
                rest = neg_pool[max(2, pairs_per_query // 2) :]
                extra = rng.choice(rest, size=pairs_per_query - len(keep), replace=False).tolist()
                neg_pool = keep + [int(x) for x in extra]

            for neg_idx in neg_pool[:pairs_per_query]:
                neg_label = float(labels[int(neg_idx)])
                delta = pos_label - neg_label
                if delta < 0.03:
                    continue
                weight = 1.0 + 3.0 * min(1.0, delta)
                if pos_label >= 0.90 and neg_label < 0.50:
                    weight += 2.0
                if pos_idx >= 5:
                    weight += 0.75
                if subset in {"Lug_Uga", "Eng_Uga", "Swa_Ken", "Eng_Ken"} and pos_label >= 0.90 and neg_label < 0.60:
                    weight += 0.75
                rows.append(
                    {
                        "query": query,
                        "pos": candidate_text(cs[pos_idx]["q"], cs[pos_idx]["a"]),
                        "neg": candidate_text(cs[int(neg_idx)]["q"], cs[int(neg_idx)]["a"]),
                        "weight": float(weight),
                        "delta": float(delta),
                        "pos_label": pos_label,
                        "neg_label": neg_label,
                        "subset": subset,
                        "pos_rank": pos_idx + 1,
                        "neg_rank": int(neg_idx) + 1,
                    }
                )
        per_subset_pair_counts[subset] = len(rows) - before
    pair_df = pd.DataFrame(rows)
    pair_stats.append({"total_pairs": int(len(pair_df))})
    print("Pair counts:", per_subset_pair_counts, flush=True)
    print(f"Total pairwise examples: {len(pair_df):,}", flush=True)
    print(pair_df[["weight", "delta", "pos_label", "neg_label", "pos_rank", "neg_rank"]].describe().round(4).to_string(), flush=True)
    print("Exact-ish hard pairs:", int(((pair_df["pos_label"] >= 0.90) & (pair_df["neg_label"] < 0.50)).sum()), flush=True)
    pair_df.sample(min(20000, len(pair_df)), random_state=seed).to_csv(out_dir / "pair_sample.csv", index=False)

    class PairwiseDataset(Dataset):
        def __init__(self, df):
            self.query = df["query"].tolist()
            self.pos = df["pos"].tolist()
            self.neg = df["neg"].tolist()
            self.weight = df["weight"].astype("float32").to_numpy()

        def __len__(self):
            return len(self.query)

        def __getitem__(self, idx):
            return self.query[idx], self.pos[idx], self.neg[idx], self.weight[idx]

    print("\nFreeing bi-encoder before cross-encoder training...", flush=True)
    try:
        bi.model.cpu()
    except Exception:
        pass
    del bi
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nLoading BGE reranker base for pairwise training...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3", num_labels=1)
    model.to(device)
    model.train()

    ds = PairwiseDataset(pair_df.sample(frac=1.0, random_state=seed).reset_index(drop=True))
    del pair_df, rows, train_cands
    gc.collect()

    def collate(batch):
        q, pos, neg, weight = zip(*batch)
        pos_tok = tokenizer(
            list(q),
            list(pos),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        neg_tok = tokenizer(
            list(q),
            list(neg),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return pos_tok, neg_tok, torch.tensor(weight, dtype=torch.float32)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_update_steps = max(1, (len(loader) * epochs) // grad_accum)
    warmup_steps = max(1, int(total_update_steps * 0.1))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)
    scaler_enabled = False
    optimizer.zero_grad(set_to_none=True)

    print("\nStarting pairwise cross-encoder training...", flush=True)
    print(
        json.dumps(
            {
                "examples": len(ds),
                "epochs": epochs,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "effective_pair_batch": batch_size * grad_accum,
                "total_update_steps": total_update_steps,
                "warmup_steps": warmup_steps,
                "lr": lr,
                "max_length": max_length,
            },
            indent=2,
        ),
        flush=True,
    )
    t0 = time.time()
    running = []
    global_step = 0
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"pairwise epoch {epoch + 1}/{epochs}")
        for step, (pos_tok, neg_tok, weight) in enumerate(pbar, start=1):
            pos_tok = {k: v.to(device, non_blocking=True) for k, v in pos_tok.items()}
            neg_tok = {k: v.to(device, non_blocking=True) for k, v in neg_tok.items()}
            weight = weight.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                pos_scores = model(**pos_tok).logits.view(-1)
                neg_scores = model(**neg_tok).logits.view(-1)
                per_example = F.softplus(-(pos_scores - neg_scores))
                loss = (per_example * weight).mean() / grad_accum
            loss.backward()
            running.append(float(loss.detach().cpu()) * grad_accum)
            if step % grad_accum == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if step % 50 == 0:
                pbar.set_postfix(loss=f"{np.mean(running[-50:]):.4f}", updates=global_step)
    train_seconds = time.time() - t0
    print(f"Pairwise cross-encoder trained in {train_seconds / 3600:.2f} hours", flush=True)

    final_dir = out_dir / "final"
    tokenizer.save_pretrained(str(final_dir))
    model.save_pretrained(str(final_dir))

    print("\nScoring validation candidates with pairwise-trained reranker...", flush=True)
    model.eval()
    reranker = CrossEncoder(str(final_dir), max_length=max_length, device=device)
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

    top1_metrics = score_preds(val_top1, val[acol].tolist(), val[gcol].tolist(), "ft_bgem3_top1")
    rerank_metrics = score_preds(rerank_preds, val[acol].tolist(), val[gcol].tolist(), "pairwise_rerank")
    oracle_metrics = score_preds(val_oracle, val[acol].tolist(), val[gcol].tolist(), "oracle_topk")

    detail_rows = []
    for idx, (top1, pred, oracle, ref, subset, oracle_rank, chosen_rank, chosen_score) in enumerate(
        zip(
            val_top1,
            rerank_preds,
            val_oracle,
            val[acol].tolist(),
            val[gcol].tolist(),
            val_oracle_ranks,
            chosen_ranks,
            chosen_scores,
        )
    ):
        top1_r1, _ = rouge_scores(top1, ref)
        pred_r1, _ = rouge_scores(pred, ref)
        oracle_r1, _ = rouge_scores(oracle, ref)
        detail_rows.append(
            {
                "ID": val[idcol].iloc[idx],
                "subset": subset,
                "top1": top1,
                "rerank": pred,
                "oracle": oracle,
                "reference": ref,
                "top1_r1": top1_r1,
                "rerank_r1": pred_r1,
                "oracle_r1": oracle_r1,
                "oracle_rank": oracle_rank,
                "chosen_rank": chosen_rank,
                "chosen_score": chosen_score,
            }
        )
    pred_df = pd.DataFrame(detail_rows)
    pred_df.to_csv(out_dir / "val_predictions.csv", index=False)

    exact_bad = pred_df[(pred_df["oracle_r1"] >= 0.95) & (pred_df["rerank_r1"] < 0.50)]
    exact_missed = pred_df[(pred_df["oracle_r1"] >= 0.95) & (pred_df["rerank"] != pred_df["oracle"])]
    exact_top1_jumped = pred_df[(pred_df["top1_r1"] >= 0.95) & (pred_df["rerank"] != pred_df["top1"])]
    diagnostics = {
        "exact_oracle_ge_095_and_rerank_lt_050_rows": int(len(exact_bad)),
        "exact_oracle_ge_095_and_rerank_lt_050_gain_if_oracle": float((exact_bad["oracle_r1"] - exact_bad["rerank_r1"]).sum() / len(pred_df)),
        "exact_oracle_ge_095_missed_rows": int(len(exact_missed)),
        "top1_ge_095_jumped_rows": int(len(exact_top1_jumped)),
        "top1_ge_095_jumped_loss_vs_top1": float((exact_top1_jumped["top1_r1"] - exact_top1_jumped["rerank_r1"]).sum() / len(pred_df)),
        "changed_from_top1_rows": int((pred_df["top1"] != pred_df["rerank"]).sum()),
        "wins_vs_top1_rows": int((pred_df["rerank_r1"] > pred_df["top1_r1"] + 1e-12).sum()),
        "hurts_vs_top1_rows": int((pred_df["rerank_r1"] + 1e-12 < pred_df["top1_r1"]).sum()),
    }
    print("\nDiagnostics:", flush=True)
    print(json.dumps(diagnostics, indent=2), flush=True)

    summary = {
        "experiment": "ft_bgem3_top50_pairwise_hardneg_rerank",
        "gpu": "L40S",
        "base_reranker": "BAAI/bge-reranker-v2-m3",
        "k": k,
        "pairs_per_query": pairs_per_query,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "effective_pair_batch": batch_size * grad_accum,
        "learning_rate": lr,
        "max_length": max_length,
        "train_seconds": train_seconds,
        "pair_counts": per_subset_pair_counts,
        "candidate_baseline": baseline_metrics,
        "top1": top1_metrics,
        "rerank": rerank_metrics,
        "oracle": oracle_metrics,
        "delta_rerank_vs_top1": rerank_metrics["rouge1"] - top1_metrics["rouge1"],
        "delta_rerank_vs_exp2": rerank_metrics["rouge1"] - 0.5892166283468145,
        "diagnostics": diagnostics,
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
    pairs_per_query: int = 6,
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
    summary = run_pairwise_experiment.remote(
        k=k,
        pairs_per_query=pairs_per_query,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
    )
    print(summary)
