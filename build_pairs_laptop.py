"""
Build training_pairs.pkl on the laptop (RTX 4050, 6 GB).

For every training row:
  - Retrieve top-K candidate ANSWERS via BGE-M3 (same-subset, self-excluded)
  - Compute ROUGE-1 F1 of each candidate against THIS row's reference answer
  - Save (query, candidate, rouge_target) triples

Inputs in the same directory:
  - Train.csv

Output:
  - training_pairs.pkl  (input for train_reranker_laptop.py)

Run:
  python build_pairs_laptop.py
"""
from __future__ import annotations

import gc, os, pickle, time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from rouge_score import rouge_scorer

# =============================================================================
# Config
# =============================================================================
HERE      = Path(__file__).resolve().parent
TRAIN_CSV = HERE / "Train.csv"
OUT_PATH  = HERE / "training_pairs.pkl"

QCOL, ACOL, GCOL = "input", "output", "subset"
BI_MODEL_NAME    = "BAAI/bge-m3"
K                = 10            # candidates per training row — keep in sync with training notebook
ENCODE_BATCH     = 32            # encoder batch size — 32 fits comfortably on 6 GB
SEED             = 42

# =============================================================================
# Utilities
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class WhitespaceTokenizer:
    def tokenize(self, t):
        return [] if t is None else str(t).strip().split()

_SCORER = rouge_scorer.RougeScorer(["rouge1"], tokenizer=WhitespaceTokenizer(), use_stemmer=False)
def rouge1(p, r):
    return _SCORER.score(str(r), str(p))["rouge1"].fmeasure

def gpu_status(label=""):
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"  [GPU {label}] {free/1e9:.2f} GB free / {total/1e9:.2f} GB total", flush=True)

def load_csv(path):
    df = pd.read_csv(path).dropna(subset=["input", "output", "subset"])
    for c in [QCOL, GCOL]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df[ACOL] = df[ACOL].fillna("").astype(str).str.strip()
    df = df[(df[QCOL] != "") & (df[ACOL] != "")]
    return df.reset_index(drop=True)

# =============================================================================
# Step 1: encode train questions with BGE-M3 (per-subset NN indices)
# =============================================================================
def encode_subset_indices(df, encoder, k):
    indices = {}
    for g, grp in df.groupby(GCOL):
        embs = encoder.encode(grp[QCOL].tolist(),
                              normalize_embeddings=True,
                              show_progress_bar=False,
                              batch_size=ENCODE_BATCH,
                              convert_to_numpy=True)
        n_fit = min(k + 1, len(grp))             # +1 so we can drop self and still keep k
        nn_ = NearestNeighbors(n_neighbors=n_fit, metric="cosine").fit(embs)
        indices[g] = {
            "nn":   nn_,
            "embs": embs,
            "ans":  np.array(grp[ACOL].astype(str).tolist(), dtype=object),
            "orig_idx": np.array(grp.index.tolist()),
        }
        print(f"  {g}: {len(grp)} rows indexed", flush=True)
    return indices

# =============================================================================
# Step 2: build (q, candidate, rouge_target) pairs with self-exclusion
# =============================================================================
def build_pairs(train_df, indices, k):
    """For every train row: take top-K (skipping self), label each by ROUGE-1 vs ref."""
    pairs_q, pairs_c, pairs_y = [], [], []
    total = len(train_df)
    seen = 0
    t0 = time.time()
    last_print = t0

    for g, grp in train_df.groupby(GCOL):
        m = indices[g]
        dist, idx = m["nn"].kneighbors(m["embs"], n_neighbors=min(k + 1, len(m["ans"])))
        for local_i, (orig_idx, row) in enumerate(grp.iterrows()):
            ref = str(row[ACOL])
            q   = str(row[QCOL])
            picked = 0
            for j in idx[local_i]:
                if m["orig_idx"][j] == orig_idx:   # self-exclusion by original index identity
                    continue
                cand = str(m["ans"][j])
                pairs_q.append(q)
                pairs_c.append(cand)
                pairs_y.append(rouge1(cand, ref))
                picked += 1
                if picked == k:
                    break
            seen += 1

            # Lightweight progress every 5 seconds (rouge1 is the slow part)
            now = time.time()
            if now - last_print > 5:
                rate = seen / (now - t0)
                eta_s = (total - seen) / rate if rate > 0 else 0
                print(f"  {seen}/{total} rows ({rate:.1f}/s, ETA {eta_s/60:.1f} min)", flush=True)
                last_print = now

    return pairs_q, pairs_c, np.array(pairs_y, dtype=np.float32)

# =============================================================================
# Main
# =============================================================================
def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        gpu_status("startup")

    if not TRAIN_CSV.exists():
        raise SystemExit(f"Missing {TRAIN_CSV}")
    train_df = load_csv(TRAIN_CSV)
    print(f"Loaded {len(train_df)} train rows")
    print(train_df[GCOL].value_counts().to_string())

    # Load BGE-M3 only for the encoding pass, then release it
    print(f"\nLoading {BI_MODEL_NAME} ...", flush=True)
    from sentence_transformers import SentenceTransformer
    bi = SentenceTransformer(BI_MODEL_NAME, device=DEVICE)
    gpu_status("after BGE-M3 load")

    print(f"\nEncoding train questions + building per-subset NN indices (K={K})...")
    t0 = time.time()
    indices = encode_subset_indices(train_df, bi, K)
    print(f"  done in {(time.time()-t0)/60:.1f} min")

    # Release the bi-encoder before the ROUGE pass (frees ~2 GB)
    bi.to("cpu")
    del bi
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    gpu_status("after BGE-M3 release")

    print(f"\nBuilding pairs (K={K} per row, self-excluded, ROUGE-1 labels)...")
    tr_q, tr_c, tr_y = build_pairs(train_df, indices, K)
    print(f"\nBuilt {len(tr_q)} pairs")

    # Quick sanity stats — same readout as section 4 of the original notebook
    s = pd.Series(tr_y)
    print("\nTarget ROUGE-1 distribution:")
    print(s.describe().round(4))
    print(f"Targets >= 0.5: {int((tr_y>=0.5).sum())} ({(tr_y>=0.5).mean()*100:.1f}%)")
    print(f"Targets >= 0.9: {int((tr_y>=0.9).sum())} ({(tr_y>=0.9).mean()*100:.1f}%)")

    # Self-leakage check — must be 0 if exclusion worked
    train_outputs = set(train_df[ACOL].astype(str).tolist())
    same_as_self = 0
    own_answer = train_df.set_index(QCOL)[ACOL].astype(str).to_dict()
    for q, c in zip(tr_q, tr_c):
        if own_answer.get(q) == c:
            same_as_self += 1
    print(f"Self-leakage check (pairs where cand == row's own answer): {same_as_self}")
    if same_as_self > 0:
        print("  WARN: non-zero — investigate before training")

    with open(OUT_PATH, "wb") as f:
        pickle.dump({"tr_q": tr_q, "tr_c": tr_c, "tr_y": tr_y}, f)
    print(f"\nSaved {OUT_PATH.name}  ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
