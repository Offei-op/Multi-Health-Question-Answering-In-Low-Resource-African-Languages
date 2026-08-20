import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
OUT = ROOT / "reports" / "oracle_lift_audit"
QCOL = "input"
ACOL = "output"
GCOL = "subset"
IDCOL = "ID"


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
    return 2 * precision * recall / (precision + recall)


def norm_text(x):
    return str(x).strip()


def load_data():
    train = pd.read_csv(ROOT / "Train.csv")
    val = pd.read_csv(ROOT / "Val.csv")
    for df in (train, val):
        for c in (QCOL, ACOL, GCOL):
            df[c] = df[c].fillna("").map(norm_text)
    train = train[(train[QCOL] != "") & (train[ACOL] != "")].reset_index(drop=True)
    val = val[(val[QCOL] != "") & (val[ACOL] != "")].reset_index(drop=True)
    return train, val


def add_candidate(pool, row_id, source, rank, question, answer):
    answer = norm_text(answer)
    if not answer:
        return
    key = (row_id, answer)
    item = pool.get(key)
    if item is None:
        pool[key] = {
            "ID": row_id,
            "answer": answer,
            "question": norm_text(question),
            "sources": {source},
            "best_rank": int(rank),
            "source_ranks": {source: int(rank)},
        }
    else:
        item["sources"].add(source)
        item["best_rank"] = min(item["best_rank"], int(rank))
        item["source_ranks"][source] = min(item["source_ranks"].get(source, 10**9), int(rank))


def load_bge_oracle():
    p = ROOT / "modal_outputs" / "exp2_val_candidate_scores" / "candidate_tabular_features.csv"
    df = pd.read_csv(p, usecols=["ID", "candidate_r1", "candidate_rank"])
    out = df.groupby("ID")["candidate_r1"].max().rename("bge_q2q_top50").reset_index()
    counts = df.groupby("ID")["candidate_rank"].count().rename("bge_q2q_top50_count").reset_index()
    return out.merge(counts, on="ID"), len(df)


def lexical_retrieve(pool, train, val, source, field, k=50, analyzer="word", ngram_range=(1, 2), char=False):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb" if char else analyzer,
        ngram_range=(3, 5) if char else ngram_range,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
    )
    for subset, val_grp in val.groupby(GCOL, sort=True):
        train_grp = train[train[GCOL] == subset].reset_index(drop=True)
        if train_grp.empty or val_grp.empty:
            continue
        train_texts = train_grp[field].tolist()
        val_texts = val_grp[field if field in val_grp.columns else QCOL].tolist()
        # For query-to-answer retrieval, validation side is still the query.
        if field == ACOL:
            val_texts = val_grp[QCOL].tolist()
        X = vectorizer.fit_transform(train_texts)
        Y = vectorizer.transform(val_texts)
        nn = NearestNeighbors(n_neighbors=min(k, len(train_grp)), metric="cosine", algorithm="brute")
        nn.fit(X)
        _, idx = nn.kneighbors(Y)
        for val_row, idxs in zip(val_grp.itertuples(index=False), idx):
            row_id = getattr(val_row, IDCOL)
            for rank, j in enumerate(idxs, start=1):
                tr = train_grp.iloc[int(j)]
                add_candidate(pool, row_id, source, rank, tr[QCOL], tr[ACOL])


def summarize_pool(pool, val, source_sets, bge_oracle):
    refs = dict(zip(val[IDCOL], val[ACOL]))
    subsets = dict(zip(val[IDCOL], val[GCOL]))
    records = []
    by_id = defaultdict(list)
    for (row_id, answer), item in pool.items():
        r1 = fast_r1(answer, refs[row_id])
        item["r1"] = r1
        by_id[row_id].append(item)

    for row in val[[IDCOL, GCOL]].itertuples(index=False):
        row_id = getattr(row, IDCOL)
        subset = getattr(row, GCOL)
        items = by_id.get(row_id, [])
        all_best = max((x["r1"] for x in items), default=0.0)
        rec = {"ID": row_id, "subset": subset, "union_all": all_best, "candidate_count_all": len(items)}
        for name, sources in source_sets.items():
            best = max((x["r1"] for x in items if x["sources"] & sources), default=0.0)
            cnt = sum(1 for x in items if x["sources"] & sources)
            rec[name] = best
            rec[f"{name}_count"] = cnt
        records.append(rec)

    rows = pd.DataFrame(records).merge(bge_oracle, on="ID", how="left")
    rows["bge_q2q_top50"] = rows["bge_q2q_top50"].fillna(0.0)
    rows["bge_q2q_top50_count"] = rows["bge_q2q_top50_count"].fillna(0.0)
    rows["bge_plus_all_lexical"] = np.maximum(rows["bge_q2q_top50"], rows["lexical_union"])
    rows["bge_plus_q2q_lexical"] = np.maximum(
        rows["bge_q2q_top50"],
        np.maximum(rows["tfidf_word_q2q_top50"], rows["tfidf_char_q2q_top50"]),
    )
    rows["union_all"] = rows["bge_plus_all_lexical"]
    metric_cols = [
        "union_all",
        "bge_plus_all_lexical",
        "bge_plus_q2q_lexical",
        "bge_q2q_top50",
    ] + list(source_sets.keys())
    metric_cols = list(dict.fromkeys(metric_cols))
    overall = []
    for col in metric_cols:
        overall.append(
            {
                "pool": col,
                "oracle_r1": rows[col].mean(),
                "rows_ge_050": int((rows[col] >= 0.50).sum()),
                "rows_ge_080": int((rows[col] >= 0.80).sum()),
                "rows_ge_095": int((rows[col] >= 0.95).sum()),
                "rows_ge_099": int((rows[col] >= 0.99).sum()),
                "mean_candidate_count": rows.get(f"{col}_count", rows["candidate_count_all"]).mean(),
            }
        )
    overall_df = pd.DataFrame(overall).sort_values("oracle_r1", ascending=False)

    subset_rows = []
    for subset, grp in rows.groupby("subset", sort=True):
        for col in metric_cols:
            subset_rows.append({"subset": subset, "pool": col, "oracle_r1": grp[col].mean(), "rows": len(grp)})
    subset_df = pd.DataFrame(subset_rows)

    current = rows["bge_q2q_top50"]
    lift_rows = []
    for col in metric_cols:
        if col == "bge_q2q_top50":
            continue
        lift_rows.append(
            {
                "pool": col,
                "delta_vs_bge_top50": rows[col].mean() - current.mean(),
                "rows_improved": int((rows[col] > current + 1e-12).sum()),
                "rows_worse": int((rows[col] + 1e-12 < current).sum()),
                "sum_gain_over_bge": float((rows[col] - current).clip(lower=0).sum() / len(rows)),
            }
        )
    lift_df = pd.DataFrame(lift_rows).sort_values("delta_vs_bge_top50", ascending=False)

    return rows, overall_df, subset_df, lift_df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, val = load_data()
    print(f"train={len(train):,} val={len(val):,}")
    pool = {}
    bge_oracle, bge_rows = load_bge_oracle()
    print(f"Loaded BGE top50 oracle labels: {bge_rows:,} candidate rows")

    jobs = [
        ("tfidf_word_q2q_top50", QCOL, False, (1, 2)),
        ("tfidf_char_q2q_top50", QCOL, True, (3, 5)),
        ("tfidf_word_q2a_top50", ACOL, False, (1, 2)),
        ("tfidf_char_q2a_top50", ACOL, True, (3, 5)),
    ]
    for source, field, char, ngram in jobs:
        print(f"Retrieving {source}...")
        lexical_retrieve(pool, train, val, source, field, k=50, char=char, ngram_range=ngram)
        print(f"  unique candidates={len(pool):,}")

    source_sets = {
        "tfidf_word_q2q_top50": {"tfidf_word_q2q_top50"},
        "tfidf_char_q2q_top50": {"tfidf_char_q2q_top50"},
        "tfidf_word_q2a_top50": {"tfidf_word_q2a_top50"},
        "tfidf_char_q2a_top50": {"tfidf_char_q2a_top50"},
        "lexical_union": {
            "tfidf_word_q2q_top50",
            "tfidf_char_q2q_top50",
            "tfidf_word_q2a_top50",
            "tfidf_char_q2a_top50",
        },
    }
    rows, overall, subset, lift = summarize_pool(pool, val, source_sets, bge_oracle)
    rows.to_csv(OUT / "oracle_lift_rows.csv", index=False)
    overall.to_csv(OUT / "oracle_lift_overall.csv", index=False)
    subset.to_csv(OUT / "oracle_lift_by_subset.csv", index=False)
    lift.to_csv(OUT / "oracle_lift_vs_bge.csv", index=False)

    summary = {
        "train_rows": len(train),
        "val_rows": len(val),
        "unique_row_answer_candidates": len(pool),
        "overall": overall.to_dict(orient="records"),
        "lift_vs_bge": lift.to_dict(orient="records"),
    }
    (OUT / "oracle_lift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "ORACLE LIFT AUDIT",
        "=" * 90,
        "",
        "Overall oracle ROUGE-1:",
        overall.to_string(index=False),
        "",
        "Lift versus current BGE top50:",
        lift.to_string(index=False),
        "",
        "By subset:",
        subset.pivot(index="subset", columns="pool", values="oracle_r1").round(4).to_string(),
    ]
    (OUT / "oracle_lift_summary.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
