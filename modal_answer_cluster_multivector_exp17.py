from pathlib import Path

import modal


APP_NAME = "lalang-answer-cluster-multivector-exp17"
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
        "accelerate>=0.33.0",
        "peft>=0.12.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "pyarrow>=17.0.0",
        "tqdm>=4.66.0",
        "safetensors>=0.4.3",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 6,
    volumes={str(REMOTE_ROOT): volume},
)
def run(
    model_source: str = "finetuned",
    batch_size: int = 128,
    max_seq_length: int = 512,
    eval_k: int = 100,
    score_chunk_size: int = 128,
):
    """Audit row retrieval against unique-answer, multi-vector retrieval.

    This experiment intentionally does no training and uses only fixed fusion weights.
    It measures whether answer clustering raises candidate recall before a new reranker
    is trained. All retrieval is restricted to the query's subset.
    """
    import json
    import re
    import time
    import unicodedata
    from collections import Counter

    import numpy as np
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer
    from tqdm.auto import tqdm

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    qcol, acol, gcol, idcol = "input", "output", "subset", "ID"
    out_dir = REMOTE_ROOT / "exp17_answer_cluster_multivector"
    out_dir.mkdir(parents=True, exist_ok=True)

    def clean(df):
        for c in (qcol, gcol, idcol):
            df[c] = df[c].fillna("").astype(str).str.strip()
        if acol in df:
            df[acol] = df[acol].fillna("").astype(str).str.strip()
        required = (qcol, gcol, idcol) + ((acol,) if acol in df else ())
        mask = np.ones(len(df), dtype=bool)
        for c in required:
            mask &= df[c].ne("").to_numpy()
        return df.loc[mask].reset_index(drop=True)

    whitespace = re.compile(r"\s+")

    def answer_key(text):
        text = unicodedata.normalize("NFKC", str(text))
        return whitespace.sub(" ", text).strip().casefold()

    def l2_normalize(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, 1e-12)

    def top_indices(scores, k):
        k = min(int(k), scores.shape[0])
        if k == scores.shape[0]:
            idx = np.arange(scores.shape[0])
        else:
            idx = np.argpartition(-scores, k - 1)[:k]
        return idx[np.argsort(-scores[idx], kind="stable")]

    def reduce_max_by_cluster(row_scores, cluster_ids, n_clusters):
        out = np.full((row_scores.shape[0], n_clusters), -np.inf, dtype=np.float32)
        for i in range(row_scores.shape[0]):
            np.maximum.at(out[i], cluster_ids, row_scores[i])
        return out

    train = clean(pd.read_csv(REMOTE_ROOT / "Train.csv"))
    val = clean(pd.read_csv(REMOTE_ROOT / "Val.csv"))
    print(f"train={len(train):,} val={len(val):,} eval_k={eval_k}", flush=True)

    ft_path = REMOTE_ROOT / "exp5_bgem3_encoder_mining_v2" / "final"
    if model_source == "finetuned":
        if not ft_path.exists():
            raise FileNotFoundError(f"Fine-tuned encoder not found: {ft_path}")
        model_name = str(ft_path)
    elif model_source == "base":
        model_name = "BAAI/bge-m3"
    else:
        model_name = model_source

    print(f"Loading encoder: {model_name}", flush=True)
    model = SentenceTransformer(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.max_seq_length = max_seq_length

    cutoffs = [k for k in (1, 10, 50, 100) if k <= eval_k]
    strategies = [
        "row_question",
        "cluster_question_max",
        "cluster_question_centroid",
        "cluster_answer_direct",
        "cluster_qa_max",
        "cluster_fusion_qmax_answer",
        "cluster_fusion_qmax_qa",
        "cluster_fusion_all",
    ]
    result_rows = []
    candidate_rows = []
    cluster_stats = []
    started = time.time()

    for subset in sorted(val[gcol].unique()):
        tr = train.loc[train[gcol].eq(subset)].reset_index(drop=True)
        va = val.loc[val[gcol].eq(subset)].reset_index(drop=True)
        if tr.empty or va.empty:
            continue
        print(f"\n=== {subset}: train={len(tr):,} val={len(va):,} ===", flush=True)

        keys = tr[acol].map(answer_key)
        cluster_ids, unique_keys = pd.factorize(keys, sort=False)
        unique_key_set = set(unique_keys)
        cluster_ids = cluster_ids.astype(np.int64, copy=False)
        n_clusters = len(unique_keys)
        first_rows = np.full(n_clusters, -1, dtype=np.int64)
        member_counts = np.zeros(n_clusters, dtype=np.int64)
        for row_i, cluster_i in enumerate(cluster_ids):
            if first_rows[cluster_i] < 0:
                first_rows[cluster_i] = row_i
            member_counts[cluster_i] += 1
        cluster_answers = tr[acol].to_numpy(dtype=object)[first_rows]
        cluster_stats.append(
            {
                "subset": subset,
                "train_rows": int(len(tr)),
                "unique_answer_clusters": int(n_clusters),
                "duplicate_row_fraction": float(1.0 - n_clusters / len(tr)),
                "largest_cluster": int(member_counts.max()),
                "mean_questions_per_cluster": float(member_counts.mean()),
                "val_exact_answer_in_train_rate": float(
                    va[acol].map(answer_key).isin(unique_key_set).mean()
                ),
            }
        )

        train_questions = tr[qcol].tolist()
        qa_docs = [
            f"Question: {q}\nAnswer: {a}"
            for q, a in zip(tr[qcol].tolist(), tr[acol].tolist())
        ]
        val_questions = va[qcol].tolist()

        print("Encoding questions, QA documents, answers, and validation queries...", flush=True)
        q_row_emb = model.encode(
            train_questions,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32, copy=False)
        qa_row_emb = model.encode(
            qa_docs,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32, copy=False)
        answer_emb = model.encode(
            cluster_answers.tolist(),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32, copy=False)
        val_emb = model.encode(
            val_questions,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32, copy=False)

        centroid = np.zeros((n_clusters, q_row_emb.shape[1]), dtype=np.float32)
        np.add.at(centroid, cluster_ids, q_row_emb)
        centroid /= member_counts[:, None]
        centroid = l2_normalize(centroid).astype(np.float32, copy=False)

        cluster_token_counts = [Counter(str(x).strip().split()) for x in cluster_answers]
        cluster_token_lens = np.array(
            [sum(c.values()) for c in cluster_token_counts], dtype=np.int32
        )

        for start in tqdm(
            range(0, len(va), score_chunk_size),
            desc=f"{subset} scoring",
        ):
            stop = min(start + score_chunk_size, len(va))
            query_emb = val_emb[start:stop]
            row_q_scores = query_emb @ q_row_emb.T
            row_qa_scores = query_emb @ qa_row_emb.T
            qmax = reduce_max_by_cluster(row_q_scores, cluster_ids, n_clusters)
            qamax = reduce_max_by_cluster(row_qa_scores, cluster_ids, n_clusters)
            centroid_scores = query_emb @ centroid.T
            answer_scores = query_emb @ answer_emb.T

            score_maps = {
                "cluster_question_max": qmax,
                "cluster_question_centroid": centroid_scores,
                "cluster_answer_direct": answer_scores,
                "cluster_qa_max": qamax,
                # Fixed, predeclared weights. Embeddings are all cosine-normalized.
                "cluster_fusion_qmax_answer": 0.75 * qmax + 0.25 * answer_scores,
                "cluster_fusion_qmax_qa": 0.50 * qmax + 0.50 * qamax,
                "cluster_fusion_all": 0.50 * qmax + 0.25 * qamax + 0.25 * answer_scores,
            }

            for local_i in range(stop - start):
                val_i = start + local_i
                ref = str(va.at[val_i, acol])
                ref_key = answer_key(ref)
                ref_counter = Counter(ref.strip().split())
                ref_len = sum(ref_counter.values())
                r1_cache = {}

                def cluster_r1(cluster_i):
                    cluster_i = int(cluster_i)
                    if cluster_i not in r1_cache:
                        cand_counter = cluster_token_counts[cluster_i]
                        cand_len = int(cluster_token_lens[cluster_i])
                        if not cand_len or not ref_len:
                            score = 0.0
                        else:
                            overlap = sum(
                                min(count, ref_counter.get(token, 0))
                                for token, count in cand_counter.items()
                            )
                            score = (
                                0.0
                                if overlap == 0
                                else float(2 * overlap / (cand_len + ref_len))
                            )
                        r1_cache[cluster_i] = score
                    return r1_cache[cluster_i]

                row_rank = top_indices(row_q_scores[local_i], eval_k)
                row_cluster_rank = cluster_ids[row_rank]
                rankings = {"row_question": row_cluster_rank}
                ranking_scores = {"row_question": row_q_scores[local_i, row_rank]}
                for strategy, matrix in score_maps.items():
                    rank = top_indices(matrix[local_i], eval_k)
                    rankings[strategy] = rank
                    ranking_scores[strategy] = matrix[local_i, rank]

                for strategy in strategies:
                    rank = rankings[strategy]
                    scores = ranking_scores[strategy]
                    r1s = np.array([cluster_r1(x) for x in rank], dtype=np.float32)
                    row = {
                        "ID": va.at[val_i, idcol],
                        "subset": subset,
                        "strategy": strategy,
                        "top1_r1": float(r1s[0]),
                        "top1_answer": str(cluster_answers[int(rank[0])]),
                        "top1_score": float(scores[0]),
                        "exact_target_available": bool(ref_key in unique_key_set),
                    }
                    for cutoff in cutoffs:
                        prefix = r1s[: min(cutoff, len(r1s))]
                        row[f"oracle{cutoff}_r1"] = float(prefix.max())
                        candidate_keys = [unique_keys[int(x)] for x in rank[:cutoff]]
                        row[f"exact_recall_at_{cutoff}"] = bool(ref_key in candidate_keys)
                    result_rows.append(row)

                    # Persist a compact candidate pool for later reranker work.
                    # Full answer text is stored only for the most useful fixed strategies.
                    if strategy in {
                        "cluster_question_max",
                        "cluster_fusion_all",
                    }:
                        for cand_rank, (cluster_i, score) in enumerate(
                            zip(rank, scores), start=1
                        ):
                            candidate_rows.append(
                                {
                                    "ID": va.at[val_i, idcol],
                                    "subset": subset,
                                    "strategy": strategy,
                                    "candidate_rank": cand_rank,
                                    "candidate_cluster": int(cluster_i),
                                    "score": float(score),
                                    "answer": str(cluster_answers[int(cluster_i)]),
                                }
                            )

        del q_row_emb, qa_row_emb, answer_emb, val_emb, centroid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pd.DataFrame(result_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    cluster_df = pd.DataFrame(cluster_stats)

    metric_cols = ["top1_r1"]
    for cutoff in cutoffs:
        metric_cols.extend([f"oracle{cutoff}_r1", f"exact_recall_at_{cutoff}"])
    leaderboard = (
        results.groupby("strategy", as_index=False)[metric_cols]
        .mean()
        .sort_values([f"oracle{max(cutoffs)}_r1", "top1_r1"], ascending=False)
    )
    per_subset = (
        results.groupby(["strategy", "subset"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["strategy", "subset"])
    )

    baseline = leaderboard.loc[leaderboard.strategy.eq("row_question")].iloc[0]
    for cutoff in cutoffs:
        leaderboard[f"delta_oracle{cutoff}_vs_row"] = (
            leaderboard[f"oracle{cutoff}_r1"] - baseline[f"oracle{cutoff}_r1"]
        )
        leaderboard[f"delta_exact_recall_at_{cutoff}_vs_row"] = (
            leaderboard[f"exact_recall_at_{cutoff}"]
            - baseline[f"exact_recall_at_{cutoff}"]
        )

    results.to_parquet(out_dir / "val_strategy_predictions.parquet", index=False)
    candidate_df.to_parquet(out_dir / "val_candidate_pools.parquet", index=False)
    cluster_df.to_csv(out_dir / "answer_cluster_stats.csv", index=False)
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    per_subset.to_csv(out_dir / "per_subset.csv", index=False)

    summary = {
        "experiment": "exp17_answer_cluster_multivector",
        "model_source": model_source,
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "eval_k": eval_k,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "seconds": float(time.time() - started),
        "cluster_stats": cluster_df.to_dict(orient="records"),
        "leaderboard": leaderboard.to_dict(orient="records"),
        "artifacts": {
            "predictions": str(out_dir / "val_strategy_predictions.parquet"),
            "candidate_pools": str(out_dir / "val_candidate_pools.parquet"),
            "leaderboard": str(out_dir / "leaderboard.csv"),
            "per_subset": str(out_dir / "per_subset.csv"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    volume.commit()
    print("\nLEADERBOARD", flush=True)
    print(leaderboard.round(6).to_string(index=False), flush=True)
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


@app.local_entrypoint()
def main(
    upload: bool = True,
    model_source: str = "finetuned",
    batch_size: int = 128,
    max_seq_length: int = 512,
    eval_k: int = 100,
):
    local_root = Path(__file__).resolve().parent
    if upload:
        print(f"Uploading Train.csv and Val.csv to Modal volume {VOLUME_NAME}...")
        with volume.batch_upload(force=True) as batch:
            batch.put_file(local_root / "Train.csv", "/Train.csv")
            batch.put_file(local_root / "Val.csv", "/Val.csv")
        print("Upload complete.")
    summary = run.remote(
        model_source=model_source,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        eval_k=eval_k,
    )
    print(summary)
