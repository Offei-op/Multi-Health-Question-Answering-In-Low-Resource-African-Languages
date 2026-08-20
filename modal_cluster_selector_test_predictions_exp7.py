from pathlib import Path

import modal


APP_NAME = "lalang-cluster-selector-test-exp7"
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
        "peft>=0.12.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.5.0",
        "lightgbm>=4.5.0",
        "tqdm>=4.66.0",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 6,
    volumes={str(REMOTE_ROOT): volume},
)
def build_selector_submission(
    bge_k: int = 200,
    lexical_k: int = 50,
    cap_per_row: int = 50,
    reranker_batch_size: int = 128,
):
    import json
    import math
    import time
    from collections import Counter, defaultdict

    import numpy as np
    import pandas as pd
    import torch
    from lightgbm import LGBMRegressor
    from peft import PeftModel
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import OrdinalEncoder
    from tqdm.auto import tqdm

    qcol, acol, gcol, idcol = "input", "output", "subset", "ID"
    out_dir = REMOTE_ROOT / "exp7_cluster_selector_test_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    reranker_dir = REMOTE_ROOT / "exp6_crossencoder_rerank_train_val" / "final"
    if not reranker_dir.exists():
        raise FileNotFoundError(f"Train+Val reranker missing: {reranker_dir}")

    seed = 42
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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

    def token_jaccard(a, b):
        aa = set(norm_space(a).lower().split())
        bb = set(norm_space(b).lower().split())
        if not aa or not bb:
            return 0.0
        return len(aa & bb) / len(aa | bb)

    def candidate_text(q, a):
        return f"Candidate question: {q}\nCandidate answer: {a}"

    def make_item():
        return defaultdict(float, {"best_any_rank": 10_000.0})

    def add_candidate(pool, row_i, answer, source, rank, score=0.0, question=""):
        answer = norm_space(answer)
        if not answer:
            return
        item = pool[(int(row_i), answer)]
        item["row_i"] = int(row_i)
        item["candidate_answer"] = answer
        if question and not item.get("candidate_question"):
            item["candidate_question"] = norm_space(question)
        item["source_count"] += 1
        item[f"src_{source}"] = 1
        item[f"{source}_count"] += 1
        item[f"{source}_best_rank"] = min(item[f"{source}_best_rank"] or 10_000, int(rank))
        item[f"{source}_max_score"] = max(item[f"{source}_max_score"], float(score))
        item["best_any_rank"] = min(item["best_any_rank"], int(rank))
        item["max_any_score"] = max(item["max_any_score"], float(score))
        item["rrf"] += 1.0 / (60.0 + int(rank))

    print("Reading data...", flush=True)
    train = pd.read_csv(REMOTE_ROOT / "Train.csv")
    val = pd.read_csv(REMOTE_ROOT / "Val.csv")
    test = pd.read_csv(REMOTE_ROOT / "Test.csv")
    for df in (train, val, test):
        for c in (idcol, qcol, gcol):
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str).map(norm_space)
        if acol in df.columns:
            df[acol] = df[acol].fillna("").astype(str).map(norm_space)
    bank = pd.concat([train, val], ignore_index=True)
    bank = bank[(bank[qcol] != "") & (bank[acol] != "") & (bank[gcol] != "")].reset_index(drop=True)
    test = test[(test[qcol] != "") & (test[gcol] != "")].reset_index(drop=True)
    print(f"bank={len(bank):,} test={len(test):,}", flush=True)

    train_pool = defaultdict(make_item)
    test_pool = defaultdict(make_item)

    print("Loading BGE-M3 adapter for dense candidates...", flush=True)
    bi = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
    bi.max_seq_length = 256
    bi[0].auto_model = PeftModel.from_pretrained(
        bi[0].auto_model,
        str(REMOTE_ROOT / "bge_m3_adapter"),
        is_trainable=False,
    )
    bi[0].auto_model.eval()

    print("Building dense BGE candidates...", flush=True)
    bank_indices = {}
    for subset, bank_grp0 in tqdm(list(bank.groupby(gcol)), desc="Dense subsets"):
        bank_grp = bank_grp0.reset_index(drop=False).rename(columns={"index": "bank_index"})
        bank_emb = bi.encode(
            bank_grp[qcol].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=128,
            convert_to_numpy=True,
        )
        nn = NearestNeighbors(n_neighbors=min(bge_k + 1, len(bank_grp)), metric="cosine", algorithm="brute").fit(bank_emb)
        bank_indices[subset] = (bank_grp, bank_emb, nn)
        dists, idxs = nn.kneighbors(bank_emb, n_neighbors=min(bge_k + 1, len(bank_grp)))
        for row_pos, row_dists, row_idxs in zip(bank_grp["bank_index"].tolist(), dists, idxs):
            picked = 0
            for rank0, (dist, j) in enumerate(zip(row_dists, row_idxs), start=1):
                cand_bank_i = int(bank_grp.at[int(j), "bank_index"])
                if cand_bank_i == int(row_pos):
                    continue
                picked += 1
                source = "bge_q2q_top50" if picked <= 50 else "bge_q2q_top200"
                add_candidate(
                    train_pool,
                    int(row_pos),
                    bank_grp.at[int(j), acol],
                    source,
                    picked,
                    float(1.0 - dist),
                    bank_grp.at[int(j), qcol],
                )
                if picked >= bge_k:
                    break
        test_grp = test[test[gcol] == subset].reset_index(drop=False).rename(columns={"index": "test_index"})
        if not test_grp.empty:
            test_emb = bi.encode(
                test_grp[qcol].tolist(),
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=128,
                convert_to_numpy=True,
            )
            dists, idxs = nn.kneighbors(test_emb, n_neighbors=min(bge_k, len(bank_grp)))
            for row_pos, row_dists, row_idxs in zip(test_grp["test_index"].tolist(), dists, idxs):
                for rank, (dist, j) in enumerate(zip(row_dists, row_idxs), start=1):
                    source = "bge_q2q_top50" if rank <= 50 else "bge_q2q_top200"
                    add_candidate(
                        test_pool,
                        int(row_pos),
                        bank_grp.at[int(j), acol],
                        source,
                        rank,
                        float(1.0 - dist),
                        bank_grp.at[int(j), qcol],
                    )

    print("Freeing dense encoder...", flush=True)
    try:
        bi.model.cpu()
    except Exception:
        pass
    del bi
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def lexical_candidates(pool, query_df, source_name, field, k, char, self_exclude):
        for subset, bank_grp0 in tqdm(list(bank.groupby(gcol)), desc=f"Lexical {source_name}"):
            bank_grp = bank_grp0.reset_index(drop=False).rename(columns={"index": "bank_index"})
            query_grp = query_df[query_df[gcol] == subset].reset_index(drop=False)
            if bank_grp.empty or query_grp.empty:
                continue
            row_col = "index"
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
            x = vec.fit_transform(bank_grp[field].tolist())
            y = vec.transform(query_grp[qcol].tolist())
            nn = NearestNeighbors(n_neighbors=min(k + (1 if self_exclude else 0), len(bank_grp)), metric="cosine", algorithm="brute").fit(x)
            dists, idxs = nn.kneighbors(y)
            for query_row, row_dists, row_idxs in zip(query_grp.itertuples(index=False), dists, idxs):
                row_i = int(getattr(query_row, row_col))
                picked = 0
                for dist, j in zip(row_dists, row_idxs):
                    cand_bank_i = int(bank_grp.at[int(j), "bank_index"])
                    if self_exclude and cand_bank_i == row_i:
                        continue
                    picked += 1
                    add_candidate(
                        pool,
                        row_i,
                        bank_grp.at[int(j), acol],
                        source_name,
                        picked,
                        float(1.0 - dist),
                        bank_grp.at[int(j), qcol],
                    )
                    if picked >= k:
                        break

    print("Building lexical candidates...", flush=True)
    lexical_candidates(train_pool, bank, "tfidf_word_q2q_top50", qcol, lexical_k, False, True)
    lexical_candidates(train_pool, bank, "tfidf_char_q2q_top50", qcol, lexical_k, True, True)
    lexical_candidates(train_pool, bank, "tfidf_word_q2a_top50", acol, lexical_k, False, True)
    lexical_candidates(train_pool, bank, "tfidf_char_q2a_top50", acol, lexical_k, True, True)
    lexical_candidates(test_pool, test, "tfidf_word_q2q_top50", qcol, lexical_k, False, False)
    lexical_candidates(test_pool, test, "tfidf_char_q2q_top50", qcol, lexical_k, True, False)
    lexical_candidates(test_pool, test, "tfidf_word_q2a_top50", acol, lexical_k, False, False)
    lexical_candidates(test_pool, test, "tfidf_char_q2a_top50", acol, lexical_k, True, False)

    ans_freq = bank.groupby([gcol, acol]).size().rename("answer_freq_subset").reset_index()
    global_ans_freq = bank.groupby(acol).size().rename("answer_freq_global").reset_index()

    def pool_to_frame(pool, rows_df, labeled):
        rows = [dict(item) for item in pool.values()]
        df = pd.DataFrame(rows)
        meta_cols = [idcol, qcol, gcol] + ([acol] if labeled else [])
        rename_cols = {"index": "row_i", qcol: "query"}
        if labeled:
            rename_cols[acol] = "reference_output"
        meta = rows_df[meta_cols].reset_index(drop=False).rename(columns=rename_cols)
        df = df.merge(meta, on="row_i", how="left")
        df = df.merge(ans_freq, left_on=[gcol, "candidate_answer"], right_on=[gcol, acol], how="left").drop(columns=[acol], errors="ignore")
        df = df.merge(global_ans_freq, left_on="candidate_answer", right_on=acol, how="left").drop(columns=[acol], errors="ignore")
        df["answer_freq_subset"] = df["answer_freq_subset"].fillna(0)
        df["answer_freq_global"] = df["answer_freq_global"].fillna(0)
        if labeled:
            df["target_r1"] = [fast_r1(a, r) for a, r in zip(df["candidate_answer"], df["reference_output"])]
        df["candidate_answer_len"] = df["candidate_answer"].map(lambda x: len(norm_space(x).split()))
        df["candidate_question_len"] = df["candidate_question"].fillna("").map(lambda x: len(norm_space(x).split()))
        df["query_len"] = df["query"].map(lambda x: len(norm_space(x).split()))
        df["query_question_jaccard"] = [token_jaccard(a, b) for a, b in zip(df["query"], df["candidate_question"].fillna(""))]
        df["query_answer_jaccard"] = [token_jaccard(a, b) for a, b in zip(df["query"], df["candidate_answer"])]
        df["log_answer_freq_subset"] = np.log1p(df["answer_freq_subset"])
        df["log_answer_freq_global"] = np.log1p(df["answer_freq_global"])
        for col in df.columns:
            if col.startswith("src_"):
                df[col] = df[col].fillna(0).astype(int)
        for col in list(df.columns):
            if col.endswith("_best_rank") or col in {"best_any_rank"}:
                df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan).fillna(10_000)
                df[f"inv_{col}"] = 1.0 / (df[col] + 1.0)
            elif col.endswith("_max_score") or col in {"max_any_score"}:
                df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-1.0)
        df["_pre_rerank_pool_score"] = (
            df["rrf"].fillna(0)
            + 0.20 * df["source_count"].fillna(0)
            + 0.10 * df["log_answer_freq_subset"].fillna(0)
            + 2.0 * df.get("inv_bge_q2q_top50_best_rank", 0)
            + 1.0 * df.get("inv_bge_q2q_top200_best_rank", 0)
            + 0.45 * df["query_question_jaccard"].fillna(0)
            + 0.35 * df["query_answer_jaccard"].fillna(0)
        )
        df = df.sort_values(["row_i", "_pre_rerank_pool_score"], ascending=[True, False])
        df = df.groupby("row_i", sort=False).head(cap_per_row).reset_index(drop=True)
        return df

    print("Materializing feature frames...", flush=True)
    train_feat = pool_to_frame(train_pool, bank, labeled=True)
    test_feat = pool_to_frame(test_pool, test, labeled=False)
    print(f"train candidates={len(train_feat):,} test candidates={len(test_feat):,}", flush=True)

    print("Loading Train+Val reranker for candidate scoring...", flush=True)
    reranker = CrossEncoder(str(reranker_dir), max_length=512, device="cuda" if torch.cuda.is_available() else "cpu")

    def score_with_reranker(df, label):
        pairs = [(q, candidate_text(cq, ca)) for q, cq, ca in zip(df["query"], df["candidate_question"].fillna(""), df["candidate_answer"])]
        print(f"Scoring {label} pairs with reranker: {len(pairs):,}", flush=True)
        scores = reranker.predict(pairs, batch_size=reranker_batch_size, show_progress_bar=True, convert_to_numpy=True)
        df["exp6_rerank_score"] = np.asarray(scores, dtype=np.float32)
        df["exp6_rerank_rank"] = df.groupby("row_i")["exp6_rerank_score"].rank(method="first", ascending=False)
        df["inv_exp6_rerank_rank"] = 1.0 / (df["exp6_rerank_rank"] + 1.0)
        max_score = df.groupby("row_i")["exp6_rerank_score"].transform("max")
        df["exp6_rerank_margin_to_best"] = max_score - df["exp6_rerank_score"]
        return df

    train_feat = score_with_reranker(train_feat, "train")
    test_feat = score_with_reranker(test_feat, "test")
    try:
        reranker.model.cpu()
    except Exception:
        pass
    del reranker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    text_cols = {
        "row_i",
        idcol,
        "query",
        "candidate_answer",
        "candidate_question",
        "reference_output",
        "target_r1",
        "pred",
    }
    feature_cols = [c for c in train_feat.columns if c not in text_cols]
    for c in feature_cols:
        if c not in test_feat.columns:
            test_feat[c] = 0
    for c in test_feat.columns:
        if c.startswith("src_") and c not in train_feat.columns:
            train_feat[c] = 0
            feature_cols.append(c)
    cat_cols = [gcol]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    combined_cat = pd.concat([train_feat[cat_cols], test_feat[cat_cols]], axis=0).fillna("")
    enc.fit(combined_cat)
    x_train = train_feat[num_cols + cat_cols].copy()
    x_test = test_feat[num_cols + cat_cols].copy()
    x_train[num_cols] = x_train[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_test[num_cols] = x_test[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_train[cat_cols] = enc.transform(x_train[cat_cols].fillna(""))
    x_test[cat_cols] = enc.transform(x_test[cat_cols].fillna(""))

    print("Training LightGBM selector on Train+Val candidates...", flush=True)
    selector = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    selector.fit(x_train, train_feat["target_r1"].astype(np.float32))
    test_feat["selector_pred"] = selector.predict(x_test)

    def choose(df, score_col):
        idx = df.groupby("row_i")[score_col].idxmax()
        return df.loc[idx].sort_values("row_i").copy()

    selector_choice = choose(test_feat, "selector_pred")
    reranker_choice = choose(test_feat, "exp6_rerank_score")
    dense_choice = choose(test_feat, "inv_bge_q2q_top50_best_rank") if "inv_bge_q2q_top50_best_rank" in test_feat.columns else reranker_choice

    def make_submission(choice, filename):
        sub = pd.DataFrame(
            {
                "ID": choice[idcol].tolist(),
                "TargetRLF1": choice["candidate_answer"].tolist(),
                "TargetR1F1": choice["candidate_answer"].tolist(),
                "TargetLLM": choice["candidate_answer"].tolist(),
            }
        )
        path = out_dir / filename
        sub.to_csv(path, index=False)
        return str(path)

    selector_path = make_submission(selector_choice, "submission_exp7_cluster_selector_trainval.csv")
    reranker_path = make_submission(reranker_choice, "submission_exp6_reranker_trainval_top50cluster.csv")
    dense_path = make_submission(dense_choice, "submission_exp7_dense_top1_trainval.csv")

    train_feat.sample(min(100_000, len(train_feat)), random_state=seed).to_csv(out_dir / "train_candidate_features_sample.csv", index=False)
    test_feat.to_csv(out_dir / "test_candidate_features_scored.csv", index=False)
    selector_choice.to_csv(out_dir / "test_selector_choices.csv", index=False)
    reranker_choice.to_csv(out_dir / "test_reranker_choices.csv", index=False)

    summary = {
        "experiment": "exp7_cluster_selector_trainval_test",
        "reranker": str(reranker_dir),
        "bge_k": bge_k,
        "lexical_k": lexical_k,
        "cap_per_row": cap_per_row,
        "bank_rows": int(len(bank)),
        "test_rows": int(len(test)),
        "train_candidate_rows": int(len(train_feat)),
        "test_candidate_rows": int(len(test_feat)),
        "selector_submission": selector_path,
        "reranker_submission": reranker_path,
        "dense_submission": dense_path,
        "selector_vs_reranker_changed": int((selector_choice["candidate_answer"].to_numpy() != reranker_choice["candidate_answer"].to_numpy()).sum()),
        "blank_selector_predictions": int((selector_choice["candidate_answer"].astype(str).str.len() == 0).sum()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    print("SUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(cap_per_row: int = 50, reranker_batch_size: int = 128):
    call = build_selector_submission.spawn(cap_per_row=cap_per_row, reranker_batch_size=reranker_batch_size)
    print(f"Spawned selector/test prediction call: {call.object_id}")
