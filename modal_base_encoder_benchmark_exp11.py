from pathlib import Path

import modal


APP_NAME = "lalang-base-encoder-benchmark-exp11"
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
    timeout=60 * 60 * 4,
    volumes={str(REMOTE_ROOT): volume},
)
def run(k: int = 50, batch_size: int = 128, max_seq_length: int = 256):
    import gc
    import json
    import time
    from collections import Counter

    import numpy as np
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import NearestNeighbors
    from tqdm.auto import tqdm

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    qcol, acol, gcol = "input", "output", "subset"
    out_dir = REMOTE_ROOT / "exp11_base_encoder_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    def clean(df):
        for c in (qcol, acol, gcol):
            df[c] = df[c].fillna("").astype(str).str.strip()
        return df[(df[qcol] != "") & (df[acol] != "") & (df[gcol] != "")].reset_index(drop=True)

    train = clean(pd.read_csv(REMOTE_ROOT / "Train.csv"))
    val = clean(pd.read_csv(REMOTE_ROOT / "Val.csv"))
    print(f"train={len(train):,} val={len(val):,} k={k}", flush=True)
    print("Val subsets:", flush=True)
    print(val[gcol].value_counts().sort_index().to_string(), flush=True)

    specs = [
        {
            "label": "bge_m3_base",
            "model": "BAAI/bge-m3",
            "query_prefix": "",
            "passage_prefix": "",
        },
        {
            "label": "multilingual_e5_base",
            "model": "intfloat/multilingual-e5-base",
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
        },
        {
            "label": "multilingual_e5_large",
            "model": "intfloat/multilingual-e5-large",
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
        },
    ]

    all_rows = []
    summary = {
        "experiment": "exp11_base_encoder_benchmark",
        "k": k,
        "max_seq_length": max_seq_length,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "models": {},
    }

    for spec in specs:
        label = spec["label"]
        t0 = time.time()
        print(f"\n=== {label}: {spec['model']} ===", flush=True)
        model = SentenceTransformer(spec["model"], device="cuda" if torch.cuda.is_available() else "cpu")
        model.max_seq_length = max_seq_length

        subset_indices = {}
        for subset, grp in tqdm(list(train.groupby(gcol)), desc=f"{label} index"):
            corpus_texts = [spec["passage_prefix"] + x for x in grp[qcol].tolist()]
            embs = model.encode(
                corpus_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size,
                convert_to_numpy=True,
            )
            nn = NearestNeighbors(n_neighbors=min(k, len(grp)), metric="cosine").fit(embs)
            subset_indices[subset] = {
                "nn": nn,
                "answers": np.array(grp[acol].astype(str).tolist(), dtype=object),
                "queries": np.array(grp[qcol].astype(str).tolist(), dtype=object),
            }

        pred_rows = []
        for subset, grp in tqdm(list(val.groupby(gcol)), desc=f"{label} val"):
            if subset not in subset_indices:
                continue
            m = subset_indices[subset]
            val_texts = [spec["query_prefix"] + x for x in grp[qcol].tolist()]
            q_embs = model.encode(
                val_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size,
                convert_to_numpy=True,
            )
            n_neighbors = min(k, len(m["answers"]))
            distances, idx_mat = m["nn"].kneighbors(q_embs, n_neighbors=n_neighbors)
            for row, dists, idxs in zip(grp.itertuples(index=False), distances, idx_mat):
                ref = getattr(row, acol)
                top_j = int(idxs[0])
                top_answer = str(m["answers"][top_j])
                top_r1 = fast_r1(top_answer, ref)
                best_r1 = top_r1
                best_rank = 1
                best_answer = top_answer
                for rank, j in enumerate(idxs, start=1):
                    cand_answer = str(m["answers"][int(j)])
                    r1 = fast_r1(cand_answer, ref)
                    if r1 > best_r1:
                        best_r1 = r1
                        best_rank = rank
                        best_answer = cand_answer
                pred_rows.append(
                    {
                        "model": label,
                        "ID": getattr(row, "ID"),
                        "subset": getattr(row, gcol),
                        "top1_r1": float(top_r1),
                        "oracle_r1": float(best_r1),
                        "oracle_rank": int(best_rank),
                        "top1_answer": top_answer,
                        "oracle_answer": best_answer,
                    }
                )

        df = pd.DataFrame(pred_rows)
        per_subset = (
            df.groupby("subset")[["top1_r1", "oracle_r1"]]
            .mean()
            .join(df.groupby("subset")["ID"].count().rename("count"))
            .reset_index()
            .sort_values("subset")
        )
        model_summary = {
            "model_name": spec["model"],
            "top1_rouge1": float(df["top1_r1"].mean()),
            "oracle_rouge1": float(df["oracle_r1"].mean()),
            "seconds": float(time.time() - t0),
            "per_subset": {
                row["subset"]: {
                    "top1_rouge1": float(row["top1_r1"]),
                    "oracle_rouge1": float(row["oracle_r1"]),
                    "count": int(row["count"]),
                }
                for _, row in per_subset.iterrows()
            },
        }
        summary["models"][label] = model_summary
        all_rows.extend(pred_rows)

        print(json.dumps({label: model_summary}, indent=2), flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows_df = pd.DataFrame(all_rows)
    rows_df.to_csv(out_dir / "val_base_encoder_predictions.csv", index=False)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    leaderboard = []
    for label, data in summary["models"].items():
        leaderboard.append(
            {
                "model": label,
                "model_name": data["model_name"],
                "top1_rouge1": data["top1_rouge1"],
                "oracle_rouge1": data["oracle_rouge1"],
                "seconds": data["seconds"],
            }
        )
    pd.DataFrame(leaderboard).sort_values("top1_rouge1", ascending=False).to_csv(
        out_dir / "leaderboard.csv", index=False
    )

    volume.commit()
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(k: int = 50, batch_size: int = 128, max_seq_length: int = 256):
    run.remote(k=k, batch_size=batch_size, max_seq_length=max_seq_length)
