from pathlib import Path

import modal


APP_NAME = "lalang-stronger-reranker-jina-exp9"
VOLUME_NAME = "lalang-bgem3-rerank"
REMOTE_ROOT = Path("/data")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "sentence-transformers==3.3.1",
        "transformers==4.46.3",
        "huggingface-hub<1.0",
        "datasets==3.2.0",
        "accelerate==1.2.1",
        "peft==0.14.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.5.0",
        "rouge-score>=0.1.2",
        "tqdm>=4.66.0",
        "safetensors>=0.4.3",
        "einops>=0.8.0",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 8,
    volumes={str(REMOTE_ROOT): volume},
)
def run_jina_reranker_experiment(
    model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
    k: int = 50,
    train_pairs_per_query: int = 12,
    epochs: int = 1,
    batch_size: int = 16,
    grad_accum: int = 1,
    lr: float = 1e-5,
):
    import gc
    import json
    import pickle
    import random
    import time

    import numpy as np
    import pandas as pd
    import torch
    from peft import PeftModel
    from collections import Counter
    from functools import lru_cache
    from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer
    from sklearn.neighbors import NearestNeighbors
    from torch.utils.data import DataLoader
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
    out_dir = REMOTE_ROOT / "exp9_jina_multilingual_reranker"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSmoke-loading stronger reranker base before expensive prep: {model_name}", flush=True)
    smoke = CrossEncoder(
        model_name,
        num_labels=1,
        max_length=512,
        trust_remote_code=True,
        automodel_args={"torch_dtype": torch.float32},
    )
    del smoke
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Jina smoke load passed.", flush=True)

    @lru_cache(maxsize=1_000_000)
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

    print("\nLoading fine-tuned BGE-M3 adapter for candidate generation...", flush=True)
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
    print("\nRetrieving train/val candidates...", flush=True)
    train_cands = retrieve_topk(train, train_idx, leave_self_out=True)
    val_cands = retrieve_topk(val, train_idx, leave_self_out=False)

    def evaluate_candidate_lists(cands, refs, subs, label):
        rows, top1_preds, oracle_preds = [], [], []
        for cs, ref, subset in zip(cands, refs, subs):
            if not cs:
                top1 = oracle = ""
                top1_r1 = oracle_r1 = 0.0
            else:
                top1 = cs[0]["a"]
                oracle = max(cs, key=lambda c: target_score(c["a"], ref))["a"]
                top1_r1, _ = rouge_scores(top1, ref)
                oracle_r1, _ = rouge_scores(oracle, ref)
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

    print("\nEvaluating candidate pool baseline/oracle...", flush=True)
    baseline_metrics, val_top1, val_oracle = evaluate_candidate_lists(
        val_cands, val[acol].tolist(), val[gcol].tolist(), f"ft_bgem3_per_subset_top{k}"
    )

    rng = np.random.default_rng(seed)
    pair_cache = out_dir / f"train_pairs_top{k}_ppq{train_pairs_per_query}_seed{seed}.pkl"
    if pair_cache.exists():
        print(f"\nLoading cached regression pairs from {pair_cache}", flush=True)
        with pair_cache.open("rb") as f:
            cached = pickle.load(f)
        pair_q, pair_c, pair_y = cached["pair_q"], cached["pair_c"], cached["pair_y"]
        per_subset_pair_counts = cached["per_subset_pair_counts"]
    else:
        print("\nBuilding regression pairs...", flush=True)
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
                {"pair_q": pair_q, "pair_c": pair_c, "pair_y": pair_y, "per_subset_pair_counts": per_subset_pair_counts},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        volume.commit()
        print(f"Cached regression pairs to {pair_cache}", flush=True)
    print("Pair counts:", per_subset_pair_counts, flush=True)
    print(f"Total pairs: {len(pair_y):,}", flush=True)
    s = pd.Series(pair_y)
    print(s.describe().round(4).to_string(), flush=True)

    print("Creating CrossEncoder InputExamples...", flush=True)
    order = rng.permutation(len(pair_y))
    train_examples = [
        InputExample(texts=[pair_q[int(i)], pair_c[int(i)]], label=float(pair_y[int(i)])) for i in tqdm(order, desc="Examples")
    ]
    del pair_q, pair_c, pair_y, order, train_cands
    try:
        bi.model.cpu()
    except Exception:
        pass
    del bi
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nLoading stronger reranker base: {model_name}", flush=True)
    reranker = CrossEncoder(
        model_name,
        num_labels=1,
        max_length=512,
        trust_remote_code=True,
        automodel_args={"torch_dtype": torch.float32},
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    try:
        reranker.model.gradient_checkpointing_enable()
    except Exception as e:
        print(f"gradient checkpointing not enabled: {e}", flush=True)

    train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=batch_size, num_workers=0)
    warmup_steps = max(100, int(len(train_dataloader) * epochs * 0.1))
    print("\nStarting stronger reranker training...", flush=True)
    t0 = time.time()
    reranker.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        loss_fct=torch.nn.MSELoss(),
        activation_fct=torch.nn.Identity(),
        scheduler="WarmupLinear",
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        weight_decay=0.01,
        output_path=str(out_dir / "final"),
        save_best_model=False,
        use_amp=False,
        show_progress_bar=True,
    )
    train_seconds = time.time() - t0
    print(f"Stronger reranker trained in {train_seconds / 3600:.2f} hours", flush=True)
    reranker.save_pretrained(str(out_dir / "final"))

    print("\nScoring validation candidates...", flush=True)
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
        row_scores = np.asarray(scores[off : off + n], dtype=np.float32)
        off += n
        rerank_preds.append(cs[int(np.argmax(row_scores))]["a"])

    def score_preds(preds, label):
        rows = []
        for p, r, s_name in zip(preds, val[acol].tolist(), val[gcol].tolist()):
            r1, rl = rouge_scores(p, r)
            rows.append({"subset": s_name, "rouge1": r1, "rougeL": rl})
        df = pd.DataFrame(rows)
        out = {
            "label": label,
            "rouge1": float(df["rouge1"].mean()),
            "rougeL": float(df["rougeL"].mean()),
            "per_subset": df.groupby("subset")[["rouge1", "rougeL"]].mean().round(4).to_dict(orient="index"),
        }
        print(json.dumps(out, indent=2), flush=True)
        return out

    top1_metrics = score_preds(val_top1, "ft_bgem3_top1")
    rerank_metrics = score_preds(rerank_preds, "jina_reranker")
    oracle_metrics = score_preds(val_oracle, "oracle_topk")

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
        "experiment": "exp9_jina_multilingual_reranker",
        "base_model": model_name,
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
        "delta_vs_exp2_rerank_r1": rerank_metrics["rouge1"] - 0.5892166283468145,
        "delta_vs_top1": rerank_metrics["rouge1"] - top1_metrics["rouge1"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(upload: bool = False):
    local_root = Path(__file__).resolve().parent
    if upload:
        print(f"Uploading data + BGE adapter to Modal volume {VOLUME_NAME}...")
        with volume.batch_upload(force=True) as batch:
            batch.put_file(local_root / "Train.csv", "/Train.csv")
            batch.put_file(local_root / "Val.csv", "/Val.csv")
            batch.put_directory(local_root / "Bgem3-finetune" / "bge-m3-health-qa" / "final", "/bge_m3_adapter")
        print("Upload complete.")
    call = run_jina_reranker_experiment.spawn()
    print(f"Spawned exp9 Jina reranker call: {call.object_id}")
