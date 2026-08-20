from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
OUT = ROOT / "reports" / "cluster_aware_selector_experiment"
QCOL = "input"
ACOL = "output"
GCOL = "subset"
IDCOL = "ID"
RANDOM_STATE = 13


def norm_space(x: object) -> str:
    return " ".join(str(x).strip().split())


def fast_r1(pred: object, ref: object) -> float:
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


def token_jaccard(a: object, b: object) -> float:
    aa = set(norm_space(a).lower().split())
    bb = set(norm_space(b).lower().split())
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(ROOT / "Train.csv")
    val = pd.read_csv(ROOT / "Val.csv")
    for df in (train, val):
        for c in (QCOL, ACOL, GCOL, IDCOL):
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str).map(norm_space)
    train = train[(train[QCOL] != "") & (train[ACOL] != "")].reset_index(drop=True)
    val = val[(val[QCOL] != "") & (val[ACOL] != "")].reset_index(drop=True)
    return train, val


def add_candidate(pool: dict, row_id: str, answer: str, source: str, rank: int, score: float = 0.0, question: str = "") -> None:
    answer = norm_space(answer)
    if not answer:
        return
    item = pool[(row_id, answer)]
    item["ID"] = row_id
    item["candidate_answer"] = answer
    if question and not item.get("candidate_question"):
        item["candidate_question"] = norm_space(question)
    item["source_count"] += 1
    item[f"src_{source}"] = 1
    item[f"{source}_count"] += 1
    item[f"{source}_best_rank"] = min(item[f"{source}_best_rank"], int(rank))
    item[f"{source}_max_score"] = max(item[f"{source}_max_score"], float(score))
    item["best_any_rank"] = min(item["best_any_rank"], int(rank))
    item["max_any_score"] = max(item["max_any_score"], float(score))
    item["rrf"] += 1.0 / (60.0 + int(rank))


def make_item() -> defaultdict:
    return defaultdict(float, {"best_any_rank": 10_000.0})


def add_exp2_candidates(pool: dict) -> None:
    p = ROOT / "modal_outputs" / "exp2_val_candidate_scores" / "val_candidate_scores.csv"
    usecols = [
        "ID",
        "candidate_rank",
        "bi_score",
        "candidate_question",
        "candidate_answer",
        "rerank_score",
        "rerank_rank",
    ]
    df = pd.read_csv(p, usecols=usecols, encoding="latin1", engine="python", on_bad_lines="skip")
    for r in df.itertuples(index=False):
        add_candidate(
            pool,
            r.ID,
            r.candidate_answer,
            "exp2_bge_top50",
            int(r.candidate_rank),
            float(r.bi_score),
            r.candidate_question,
        )
        item = pool[(r.ID, norm_space(r.candidate_answer))]
        item["exp2_candidate_rank"] = min(item.get("exp2_candidate_rank", 10_000.0), float(r.candidate_rank))
        item["exp2_bi_score"] = max(item.get("exp2_bi_score", -99.0), float(r.bi_score))
        item["exp2_rerank_score"] = max(item.get("exp2_rerank_score", -99.0), float(r.rerank_score))
        item["exp2_rerank_rank"] = min(item.get("exp2_rerank_rank", 10_000.0), float(r.rerank_rank))
        item["src_exp2_reranked"] = 1


def lexical_hits(pool: dict, train: pd.DataFrame, val: pd.DataFrame, source: str, field: str, k: int, char: bool) -> None:
    for subset, val_grp in val.groupby(GCOL, sort=True):
        train_grp = train[train[GCOL] == subset].reset_index(drop=True)
        if train_grp.empty or val_grp.empty:
            continue
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
        x = vec.fit_transform(train_grp[field].tolist())
        y = vec.transform(val_grp[QCOL].tolist())
        nn = NearestNeighbors(n_neighbors=min(k, len(train_grp)), metric="cosine", algorithm="brute").fit(x)
        dists, idxs = nn.kneighbors(y)
        sims = 1.0 - dists
        for val_row, row_idxs, row_sims in zip(val_grp.itertuples(index=False), idxs, sims):
            for rank, (j, sim) in enumerate(zip(row_idxs, row_sims), start=1):
                tr = train_grp.iloc[int(j)]
                add_candidate(pool, val_row.ID, tr[ACOL], source, rank, float(sim), tr[QCOL])


def bge200_hits(pool: dict, train: pd.DataFrame, val: pd.DataFrame) -> dict:
    try:
        import torch
        from peft import PeftModel
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        return {"available": False, "error": f"import failed: {type(e).__name__}: {e}"}

    adapter = ROOT / "Bgem3-finetune" / "bge-m3-health-qa" / "final"
    if not adapter.exists():
        return {"available": False, "error": f"adapter missing: {adapter}"}
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer("BAAI/bge-m3", device=device)
        model.max_seq_length = 256
        model[0].auto_model = PeftModel.from_pretrained(model[0].auto_model, str(adapter), is_trainable=False)
        model[0].auto_model.eval()
        for subset, train_grp0 in train.groupby(GCOL, sort=True):
            val_grp = val[val[GCOL] == subset].reset_index(drop=True)
            train_grp = train_grp0.reset_index(drop=True)
            if train_grp.empty or val_grp.empty:
                continue
            print(f"BGE200 {subset}: train={len(train_grp):,} val={len(val_grp):,}", flush=True)
            tr_emb = model.encode(
                train_grp[QCOL].tolist(),
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=128,
                convert_to_numpy=True,
            )
            va_emb = model.encode(
                val_grp[QCOL].tolist(),
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=128,
                convert_to_numpy=True,
            )
            nn = NearestNeighbors(n_neighbors=min(200, len(train_grp)), metric="cosine", algorithm="brute").fit(tr_emb)
            dists, idxs = nn.kneighbors(va_emb)
            sims = 1.0 - dists
            for val_row, row_idxs, row_sims in zip(val_grp.itertuples(index=False), idxs, sims):
                for rank, (j, sim) in enumerate(zip(row_idxs, row_sims), start=1):
                    tr = train_grp.iloc[int(j)]
                    add_candidate(pool, val_row.ID, tr[ACOL], "bge200_q2q", rank, float(sim), tr[QCOL])
        return {"available": True, "device": device}
    except Exception as e:
        return {"available": False, "error": f"runtime failed: {type(e).__name__}: {e}"}


def build_feature_frame(pool: dict, train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    val_meta = val[[IDCOL, QCOL, ACOL, GCOL]].rename(columns={QCOL: "val_input", ACOL: "reference"})
    ans_freq = train.groupby([GCOL, ACOL]).size().rename("answer_freq_subset").reset_index()
    global_ans_freq = train.groupby(ACOL).size().rename("answer_freq_global").reset_index()
    rows = []
    for item in pool.values():
        rows.append(dict(item))
    cand = pd.DataFrame(rows)
    cand = cand.merge(val_meta, on=IDCOL, how="left")
    cand = cand.merge(ans_freq, left_on=[GCOL, "candidate_answer"], right_on=[GCOL, ACOL], how="left").drop(columns=[ACOL], errors="ignore")
    cand = cand.merge(global_ans_freq, left_on="candidate_answer", right_on=ACOL, how="left").drop(columns=[ACOL], errors="ignore")
    cand["answer_freq_subset"] = cand["answer_freq_subset"].fillna(0)
    cand["answer_freq_global"] = cand["answer_freq_global"].fillna(0)
    cand["target_r1"] = [fast_r1(a, r) for a, r in zip(cand["candidate_answer"], cand["reference"])]
    cand["candidate_answer_len"] = cand["candidate_answer"].map(lambda x: len(norm_space(x).split()))
    cand["candidate_question_len"] = cand["candidate_question"].fillna("").map(lambda x: len(norm_space(x).split()))
    cand["val_input_len"] = cand["val_input"].map(lambda x: len(norm_space(x).split()))
    cand["reference_len"] = cand["reference"].map(lambda x: len(norm_space(x).split()))
    cand["answer_ref_len_ratio_proxy"] = cand["candidate_answer_len"] / cand["reference_len"].clip(lower=1)
    cand["query_question_jaccard"] = [token_jaccard(a, b) for a, b in zip(cand["val_input"], cand["candidate_question"].fillna(""))]
    cand["query_answer_jaccard"] = [token_jaccard(a, b) for a, b in zip(cand["val_input"], cand["candidate_answer"])]
    cand["log_answer_freq_subset"] = np.log1p(cand["answer_freq_subset"])
    cand["log_answer_freq_global"] = np.log1p(cand["answer_freq_global"])
    for col in cand.columns:
        if col.startswith("src_"):
            cand[col] = cand[col].fillna(0).astype(int)
    for col in cand.columns:
        if col.endswith("_best_rank") or col in {"best_any_rank", "exp2_candidate_rank", "exp2_rerank_rank"}:
            cand[col] = cand[col].replace(0, np.nan).fillna(10_000)
            cand[f"inv_{col}"] = 1.0 / (cand[col] + 1.0)
        elif col.endswith("_max_score") or col in {"max_any_score", "exp2_bi_score", "exp2_rerank_score"}:
            cand[col] = cand[col].replace([np.inf, -np.inf], np.nan).fillna(-1.0)
    return cand


def choose_best(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    idx = df.groupby(IDCOL)[score_col].idxmax()
    return df.loc[idx].copy()


def mean_choice_score(choice: pd.DataFrame) -> float:
    return float(choice["target_r1"].mean())


def add_baseline_rows(cand: pd.DataFrame) -> list[dict]:
    rows = []
    for label, mask_col in [
        ("exp2_rerank_existing", "exp2_rerank_rank"),
        ("exp2_encoder_top1_existing", "exp2_candidate_rank"),
    ]:
        if mask_col not in cand.columns:
            continue
        tmp = cand.copy()
        tmp["_baseline_score"] = -tmp[mask_col].fillna(10_000)
        chosen = choose_best(tmp, "_baseline_score")
        rows.append({"model": label, "score": mean_choice_score(chosen), "rmse": np.nan, "candidate_rows": len(cand)})
    oracle = choose_best(cand, "target_r1")
    rows.append({"model": "candidate_pool_oracle", "score": mean_choice_score(oracle), "rmse": 0.0, "candidate_rows": len(cand)})
    return rows


def train_oof(cand: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    drop = {
        IDCOL,
        "candidate_answer",
        "candidate_question",
        "val_input",
        "reference",
        "target_r1",
        "pred",
    }
    feature_cols = [c for c in cand.columns if c not in drop and c != ACOL]
    categorical_cols = [GCOL]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    x = cand[numeric_cols + categorical_cols].copy()
    x[numeric_cols] = x[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x[categorical_cols] = enc.fit_transform(x[categorical_cols].fillna(""))
    y = cand["target_r1"].to_numpy()

    id_subset = cand[[IDCOL, GCOL]].drop_duplicates().reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "hist_gb_cluster": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.035, l2_regularization=0.02, random_state=RANDOM_STATE),
        "extra_trees_cluster": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        "random_forest_cluster": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
    }
    try:
        from lightgbm import LGBMRegressor

        models["lightgbm_cluster"] = LGBMRegressor(
            n_estimators=900,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        pass

    leader = add_baseline_rows(cand)
    oof_choice_frames = []
    subset_rows = []
    for name, model in models.items():
        pred = np.zeros(len(cand), dtype=np.float32)
        for fold, (tr_ids, va_ids) in enumerate(skf.split(id_subset[IDCOL], id_subset[GCOL]), start=1):
            train_ids = set(id_subset.iloc[tr_ids][IDCOL])
            valid_ids = set(id_subset.iloc[va_ids][IDCOL])
            tr_mask = cand[IDCOL].isin(train_ids).to_numpy()
            va_mask = cand[IDCOL].isin(valid_ids).to_numpy()
            model.fit(x.loc[tr_mask], y[tr_mask])
            pred[va_mask] = model.predict(x.loc[va_mask])
            print(f"{name} fold {fold} done", flush=True)
        scored = cand.copy()
        scored["pred"] = pred
        chosen = choose_best(scored, "pred")
        rmse = float(math.sqrt(mean_squared_error(y, pred)))
        leader.append({"model": name, "score": mean_choice_score(chosen), "rmse": rmse, "candidate_rows": len(cand)})
        chosen["model"] = name
        oof_choice_frames.append(chosen)
        for subset, grp in chosen.groupby(GCOL, sort=True):
            subset_rows.append({"model": name, "subset": subset, "score": float(grp["target_r1"].mean()), "rows": len(grp)})

    leader_df = pd.DataFrame(leader).sort_values("score", ascending=False)
    choices_df = pd.concat(oof_choice_frames, ignore_index=True) if oof_choice_frames else pd.DataFrame()
    subset_df = pd.DataFrame(subset_rows).sort_values(["model", GCOL])
    return leader_df, choices_df, subset_df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train, val = load_data()
    pool = defaultdict(make_item)

    print("Loading exp2 top50 candidate scores...", flush=True)
    add_exp2_candidates(pool)
    print(f"candidate keys after exp2: {len(pool):,}", flush=True)

    print("Adding lexical answer-cluster candidates...", flush=True)
    lexical_hits(pool, train, val, "tfidf_word_q2q_top50", QCOL, k=50, char=False)
    lexical_hits(pool, train, val, "tfidf_char_q2q_top50", QCOL, k=50, char=True)
    lexical_hits(pool, train, val, "tfidf_word_q2a_top50", ACOL, k=50, char=False)
    lexical_hits(pool, train, val, "tfidf_char_q2a_top50", ACOL, k=50, char=True)
    print(f"candidate keys after lexical: {len(pool):,}", flush=True)

    print("Adding local BGE200 candidates when available...", flush=True)
    bge_status = bge200_hits(pool, train, val)
    print(f"BGE status: {bge_status}", flush=True)
    print(f"candidate keys after BGE200: {len(pool):,}", flush=True)

    cand = build_feature_frame(pool, train, val)
    cand.to_csv(OUT / "cluster_candidate_features.csv", index=False)
    print(f"feature frame: {cand.shape}", flush=True)

    leader, choices, subset = train_oof(cand)
    leader.to_csv(OUT / "cluster_selector_leaderboard.csv", index=False)
    choices.to_csv(OUT / "cluster_selector_oof_choices.csv", index=False)
    subset.to_csv(OUT / "cluster_selector_subset_summary.csv", index=False)
    summary = {
        "train_rows": len(train),
        "val_rows": len(val),
        "candidate_rows": len(cand),
        "bge_status": bge_status,
        "best": leader.iloc[0].to_dict(),
        "leaderboard": leader.to_dict(orient="records"),
    }
    (OUT / "cluster_selector_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(leader.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
