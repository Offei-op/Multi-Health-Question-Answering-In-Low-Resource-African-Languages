from pathlib import Path

import modal


APP_NAME = "lalang-ghana-grouped-exp8"
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
    timeout=60 * 60 * 4,
    volumes={str(REMOTE_ROOT): volume},
)
def run_ghana_grouped_experiment(
    k: int = 50,
    dense_k: int = 200,
    lexical_k: int = 50,
    encoder_epochs: int = 1,
    encoder_pairs_per_anchor: int = 6,
    reranker_pairs_per_query: int = 12,
    reranker_epochs: int = 1,
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
    from peft import PeftModel
    from rouge_score import rouge_scorer
    from sentence_transformers import CrossEncoder, SentenceTransformer, losses
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import MSELoss
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
    ghana = {"Aka_Gha", "Eng_Gha"}
    out_dir = REMOTE_ROOT / "exp8_ghana_grouped_encoder_reranker"
    out_dir.mkdir(parents=True, exist_ok=True)

    def norm_space(x):
        return " ".join(str(x).strip().split())

    def fast_r1(pred, ref):
        pred_toks = norm_space(pred).split()
        ref_toks = norm_space(ref).split()
        if not pred_toks or not ref_toks:
            return 0.0
        pc = Counter(pred_toks)
        rc = Counter(ref_toks)
        overlap = sum(min(pc[t], rc[t]) for t in pc)
        if overlap <= 0:
            return 0.0
        precision = overlap / len(pred_toks)
        recall = overlap / len(ref_toks)
        return float(2 * precision * recall / (precision + recall))

    class WhitespaceTokenizer:
        def tokenize(self, text):
            return [] if text is None else str(text).strip().split()

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], tokenizer=WhitespaceTokenizer(), use_stemmer=False)

    def rouge_scores(pred, ref):
        s = scorer.score(str(ref), str(pred))
        return float(s["rouge1"].fmeasure), float(s["rougeL"].fmeasure)

    def target_score(candidate_answer, reference):
        r1, rl = rouge_scores(candidate_answer, reference)
        return np.float32(0.75 * r1 + 0.25 * rl)

    def candidate_text(q, a):
        return f"Candidate question: {q}\nCandidate answer: {a}"

    print("Reading Ghana train/val rows...", flush=True)
    train_all = pd.read_csv(REMOTE_ROOT / "Train.csv")
    val_all = pd.read_csv(REMOTE_ROOT / "Val.csv")
    for df in (train_all, val_all):
        for c in (idcol, qcol, acol, gcol):
            df[c] = df[c].fillna("").astype(str).map(norm_space)
    train = train_all[train_all[gcol].isin(ghana) & (train_all[qcol] != "") & (train_all[acol] != "")].reset_index(drop=True)
    val = val_all[val_all[gcol].isin(ghana) & (val_all[qcol] != "") & (val_all[acol] != "")].reset_index(drop=True)
    print(f"ghana train={len(train):,} val={len(val):,}", flush=True)
    print(train[gcol].value_counts().sort_index().to_string(), flush=True)
    print(val[gcol].value_counts().sort_index().to_string(), flush=True)

    def load_global_encoder(is_trainable=False):
        model = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
        model.max_seq_length = 256
        model[0].auto_model = PeftModel.from_pretrained(
            model[0].auto_model,
            str(REMOTE_ROOT / "bge_m3_adapter"),
            is_trainable=is_trainable,
        )
        if not is_trainable:
            model[0].auto_model.eval()
        return model

    @torch.no_grad()
    def retrieve_with_encoder(model, query_df, bank_df, leave_self_out=False):
        cands = [[] for _ in range(len(query_df))]
        for subset, bank_grp0 in bank_df.groupby(gcol, sort=True):
            query_grp = query_df[query_df[gcol] == subset]
            if query_grp.empty:
                continue
            bank_grp = bank_grp0.reset_index(drop=False).rename(columns={"index": "orig_idx"})
            b_emb = model.encode(
                bank_grp[qcol].tolist(),
                batch_size=128,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            q_emb = model.encode(
                query_grp[qcol].tolist(),
                batch_size=128,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            nn = NearestNeighbors(n_neighbors=min(k + (1 if leave_self_out else 0), len(bank_grp)), metric="cosine").fit(b_emb)
            dists, idxs = nn.kneighbors(q_emb)
            for q_pos, row_i, row_dists, row_idxs in zip(range(len(query_grp)), query_grp.index.tolist(), dists, idxs):
                picked = []
                for dist, j in zip(row_dists, row_idxs):
                    orig_idx = int(bank_grp.at[int(j), "orig_idx"])
                    if leave_self_out and orig_idx == row_i:
                        continue
                    picked.append(
                        {
                            "rank": len(picked) + 1,
                            "bi_score": float(1.0 - dist),
                            "q": str(bank_grp.at[int(j), qcol]),
                            "a": str(bank_grp.at[int(j), acol]),
                        }
                    )
                    if len(picked) >= k:
                        break
                cands[int(row_i)] = picked
        return cands

    def evaluate_candidates(cands, refs, subs, label):
        rows = []
        top1_preds, oracle_preds = [], []
        for cs, ref, subset in zip(cands, refs, subs):
            if not cs:
                top1 = oracle = ""
                top1_r1 = oracle_r1 = 0.0
            else:
                top1 = cs[0]["a"]
                oracle = max(cs, key=lambda c: target_score(c["a"], ref))["a"]
                top1_r1 = fast_r1(top1, ref)
                oracle_r1 = fast_r1(oracle, ref)
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

    print("\nBaseline Ghana global encoder + global reranker...", flush=True)
    global_encoder = load_global_encoder(is_trainable=False)
    base_val_cands = retrieve_with_encoder(global_encoder, val, train, leave_self_out=False)
    base_encoder_metrics, base_top1, base_oracle = evaluate_candidates(
        base_val_cands, val[acol].tolist(), val[gcol].tolist(), "global_encoder_ghana_val"
    )
    del global_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nMining Ghana encoder triplets from global encoder retrieval + lexical pools...", flush=True)
    miner = load_global_encoder(is_trainable=False)
    train_q = train[qcol].tolist()
    train_a = train[acol].tolist()
    train_s = train[gcol].tolist()
    train_emb = miner.encode(train_q, batch_size=128, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    candidate_pool = [set() for _ in range(len(train))]

    def add_cand(i, j):
        if i != j:
            candidate_pool[int(i)].add(int(j))

    for subset, grp in tqdm(list(train.groupby(gcol)), desc="Dense Ghana mine"):
        idxs = grp.index.to_numpy()
        nn = NearestNeighbors(n_neighbors=min(dense_k + 1, len(idxs)), metric="cosine").fit(train_emb[idxs])
        _, mat = nn.kneighbors(train_emb[idxs])
        for local_i, row in enumerate(mat):
            anchor = int(idxs[local_i])
            for local_j in row:
                add_cand(anchor, int(idxs[int(local_j)]))

    def lexical_mine(name, field, char):
        for subset, grp in tqdm(list(train.groupby(gcol)), desc=name):
            idxs = grp.index.to_numpy()
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
            x = vec.fit_transform(grp[field].tolist())
            y = vec.transform(grp[qcol].tolist())
            nn = NearestNeighbors(n_neighbors=min(lexical_k + 1, len(idxs)), metric="cosine", algorithm="brute").fit(x)
            _, mat = nn.kneighbors(y)
            for local_i, row in enumerate(mat):
                anchor = int(idxs[local_i])
                for local_j in row:
                    add_cand(anchor, int(idxs[int(local_j)]))

    lexical_mine("Ghana tfidf word q2q", qcol, False)
    lexical_mine("Ghana tfidf char q2q", qcol, True)
    lexical_mine("Ghana tfidf word q2a", acol, False)
    lexical_mine("Ghana tfidf char q2a", acol, True)

    rows = []
    rng = np.random.default_rng(seed)
    stats = defaultdict(int)
    for i in tqdm(range(len(train)), desc="Build Ghana triplets"):
        scored = [(j, fast_r1(train_a[j], train_a[i])) for j in candidate_pool[i]]
        if not scored:
            stats["no_candidates"] += 1
            continue
        scored.sort(key=lambda x: -x[1])
        positives = [(j, r) for j, r in scored if r >= 0.28]
        if not positives and scored[0][1] >= 0.20:
            positives = [scored[0]]
        negatives = [(j, r) for j, r in scored if r <= 0.20]
        if not negatives:
            negatives = scored[-min(8, len(scored)) :]
        if not positives or not negatives:
            stats["dropped"] += 1
            continue
        produced = 0
        positives = positives[:3]
        negatives = [(j, r) for j, r in negatives if positives[0][1] - r >= 0.08][:8]
        if not negatives:
            stats["no_margin_negatives"] += 1
            continue
        for pj, pr in positives:
            order = np.arange(len(negatives))
            rng.shuffle(order)
            for ni in order:
                nj, nr = negatives[int(ni)]
                rows.append({"anchor": train_q[i], "positive": train_q[pj], "negative": train_q[nj], "subset": train_s[i]})
                produced += 1
                if produced >= encoder_pairs_per_anchor:
                    break
            if produced >= encoder_pairs_per_anchor:
                break
        stats["ok"] += 1
    pair_df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed)
    print(f"Ghana encoder triplets={len(pair_df):,} stats={dict(stats)}", flush=True)
    print(pair_df.groupby("subset").size().sort_index().to_string(), flush=True)
    pair_df.to_csv(out_dir / "encoder_triplets_sample.csv", index=False)
    train_ds = Dataset.from_dict(
        {
            "anchor": pair_df["anchor"].astype(str).tolist(),
            "positive": pair_df["positive"].astype(str).tolist(),
            "negative": pair_df["negative"].astype(str).tolist(),
        }
    )
    del miner, train_emb, candidate_pool, pair_df, rows
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nContinuing global encoder adapter on Ghana...", flush=True)
    encoder = load_global_encoder(is_trainable=True)
    try:
        encoder[0].auto_model.gradient_checkpointing_enable()
    except Exception:
        pass
    enc_loss = losses.MultipleNegativesRankingLoss(model=encoder)
    enc_args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "encoder_trainer"),
        num_train_epochs=encoder_epochs,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        seed=seed,
    )
    enc_trainer = SentenceTransformerTrainer(model=encoder, args=enc_args, train_dataset=train_ds, loss=enc_loss)
    t0 = time.time()
    enc_trainer.train()
    encoder_seconds = time.time() - t0
    encoder_dir = out_dir / "encoder_final"
    encoder.save_pretrained(str(encoder_dir))
    print(f"Ghana encoder trained in {encoder_seconds / 60:.1f} min", flush=True)

    print("\nEvaluating Ghana specialized encoder...", flush=True)
    gh_train_cands = retrieve_with_encoder(encoder, train, train, leave_self_out=True)
    gh_val_cands = retrieve_with_encoder(encoder, val, train, leave_self_out=False)
    gh_encoder_metrics, gh_top1, gh_oracle = evaluate_candidates(
        gh_val_cands, val[acol].tolist(), val[gcol].tolist(), "ghana_encoder_ghana_val"
    )

    print("\nBuilding Ghana reranker pairs...", flush=True)
    pair_q, pair_c, pair_y = [], [], []
    per_subset_pair_counts = {}
    for subset in sorted(train[gcol].unique()):
        idxs = np.where(train[gcol].to_numpy() == subset)[0]
        before = len(pair_y)
        for i in tqdm(idxs, desc=f"Reranker pairs {subset}"):
            cs = gh_train_cands[int(i)]
            if not cs:
                continue
            labels = np.array([target_score(c["a"], train[acol].iloc[int(i)]) for c in cs], dtype=np.float32)
            order = np.argsort(-labels)
            chosen = []
            chosen.extend(order[: min(4, len(order))].tolist())
            chosen.extend(order[-min(4, len(order)) :].tolist())
            mid_pool = order[4:-4] if len(order) > 8 else order
            if len(mid_pool) > 0:
                n_mid = max(0, reranker_pairs_per_query - len(set(chosen)))
                chosen.extend(rng.choice(mid_pool, size=min(n_mid, len(mid_pool)), replace=False).tolist())
            seen = set()
            chosen = [x for x in chosen if not (x in seen or seen.add(x))]
            for j in chosen[:reranker_pairs_per_query]:
                pair_q.append(str(train[qcol].iloc[int(i)]))
                pair_c.append(candidate_text(cs[int(j)]["q"], cs[int(j)]["a"]))
                pair_y.append(float(labels[int(j)]))
        per_subset_pair_counts[subset] = len(pair_y) - before
    print("Ghana reranker pair counts:", per_subset_pair_counts, flush=True)
    print(f"Total Ghana reranker pairs={len(pair_y):,}", flush=True)

    rerank_ds = Dataset.from_dict({"query": pair_q, "candidate": pair_c, "label": pair_y}).shuffle(seed=seed)
    del pair_q, pair_c, pair_y, gh_train_cands
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nContinuing exp2 global reranker on Ghana pairs...", flush=True)
    reranker = CrossEncoder(
        str(REMOTE_ROOT / "exp2_crossencoder_rerank" / "final"),
        max_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    rerank_loss = MSELoss(model=reranker)
    rerank_args = CrossEncoderTrainingArguments(
        output_dir=str(out_dir / "reranker_trainer"),
        num_train_epochs=reranker_epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=8e-6,
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
    rerank_trainer = CrossEncoderTrainer(model=reranker, args=rerank_args, train_dataset=rerank_ds, loss=rerank_loss)
    t0 = time.time()
    rerank_trainer.train()
    reranker_seconds = time.time() - t0
    reranker_dir = out_dir / "reranker_final"
    reranker.save_pretrained(str(reranker_dir))
    print(f"Ghana reranker trained in {reranker_seconds / 60:.1f} min", flush=True)

    def rerank_predictions(reranker_model, cands, label):
        flat_pairs, row_lens = [], []
        for q, cs in zip(val[qcol].tolist(), cands):
            flat_pairs.extend([(q, candidate_text(c["q"], c["a"])) for c in cs])
            row_lens.append(len(cs))
        scores = reranker_model.predict(flat_pairs, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        preds, off = [], 0
        debug_rows = []
        for row, cs, n in zip(val.itertuples(index=False), cands, row_lens):
            if n == 0:
                preds.append("")
                continue
            row_scores = np.asarray(scores[off : off + n], dtype=np.float32)
            off += n
            best_i = int(np.argmax(row_scores))
            preds.append(cs[best_i]["a"])
            for cand_i, c in enumerate(cs):
                debug_rows.append(
                    {
                        "ID": getattr(row, idcol),
                        "subset": getattr(row, gcol),
                        "label": label,
                        "candidate_rank": c["rank"],
                        "bi_score": c["bi_score"],
                        "rerank_score": float(row_scores[cand_i]),
                        "chosen": cand_i == best_i,
                        "candidate_answer": c["a"],
                    }
                )
        return preds, debug_rows

    print("\nScoring Ghana val with global reranker baseline...", flush=True)
    global_reranker = CrossEncoder(
        str(REMOTE_ROOT / "exp2_crossencoder_rerank" / "final"),
        max_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    base_rerank_preds, base_debug = rerank_predictions(global_reranker, base_val_cands, "global_encoder_global_reranker")
    del global_reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nScoring Ghana val with specialized encoder + specialized reranker...", flush=True)
    gh_rerank_preds, gh_debug = rerank_predictions(reranker, gh_val_cands, "ghana_encoder_ghana_reranker")

    def score_preds(preds, label):
        rows = []
        for p, r, s in zip(preds, val[acol].tolist(), val[gcol].tolist()):
            r1, rl = rouge_scores(p, r)
            rows.append({"subset": s, "rouge1": r1, "rougeL": rl})
        df = pd.DataFrame(rows)
        out = {
            "label": label,
            "rouge1": float(df["rouge1"].mean()),
            "rougeL": float(df["rougeL"].mean()),
            "per_subset": df.groupby("subset")[["rouge1", "rougeL"]].mean().round(4).to_dict(orient="index"),
        }
        print(json.dumps(out, indent=2), flush=True)
        return out

    base_rerank_metrics = score_preds(base_rerank_preds, "global_encoder_global_reranker")
    gh_rerank_metrics = score_preds(gh_rerank_preds, "ghana_encoder_ghana_reranker")
    gh_top1_metrics = score_preds(gh_top1, "ghana_encoder_top1")

    pred_df = pd.DataFrame(
        {
            "ID": val[idcol].tolist(),
            "subset": val[gcol].tolist(),
            "reference": val[acol].tolist(),
            "global_top1": base_top1,
            "global_rerank": base_rerank_preds,
            "global_oracle": base_oracle,
            "ghana_top1": gh_top1,
            "ghana_rerank": gh_rerank_preds,
            "ghana_oracle": gh_oracle,
        }
    )
    pred_df.to_csv(out_dir / "ghana_val_predictions.csv", index=False)
    pd.DataFrame(base_debug + gh_debug).to_csv(out_dir / "ghana_val_candidate_scores.csv", index=False)

    summary = {
        "experiment": "exp8_ghana_grouped_encoder_reranker",
        "subsets": sorted(ghana),
        "gpu": "L40S",
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "k": k,
        "dense_k": dense_k,
        "lexical_k": lexical_k,
        "encoder_epochs": encoder_epochs,
        "encoder_pairs_per_anchor": encoder_pairs_per_anchor,
        "encoder_train_seconds": encoder_seconds,
        "encoder_examples": int(len(train_ds)),
        "encoder_dir": str(encoder_dir),
        "reranker_epochs": reranker_epochs,
        "reranker_pairs_per_query": reranker_pairs_per_query,
        "reranker_train_seconds": reranker_seconds,
        "reranker_examples": int(len(rerank_ds)),
        "reranker_dir": str(reranker_dir),
        "global_encoder": base_encoder_metrics,
        "ghana_encoder": gh_encoder_metrics,
        "global_rerank": base_rerank_metrics,
        "ghana_rerank": gh_rerank_metrics,
        "ghana_top1": gh_top1_metrics,
        "delta_ghana_rerank_vs_global_rerank": gh_rerank_metrics["rouge1"] - base_rerank_metrics["rouge1"],
        "delta_ghana_encoder_oracle_vs_global_encoder_oracle": gh_encoder_metrics["oracle_r1"] - base_encoder_metrics["oracle_r1"],
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
        print(f"Uploading CSVs and adapters to Modal volume {VOLUME_NAME}...")
        with volume.batch_upload(force=True) as batch:
            batch.put_file(local_root / "Train.csv", "/Train.csv")
            batch.put_file(local_root / "Val.csv", "/Val.csv")
            batch.put_directory(local_root / "Bgem3-finetune" / "bge-m3-health-qa" / "final", "/bge_m3_adapter")
        print("Upload complete.")
    call = run_ghana_grouped_experiment.spawn()
    print(f"Spawned Ghana grouped exp8 call: {call.object_id}")
