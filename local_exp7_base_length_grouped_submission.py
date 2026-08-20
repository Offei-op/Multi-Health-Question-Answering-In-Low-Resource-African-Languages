from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
OUT = ROOT / "reports" / "exp7_base_length_grouped_submission"
OUT.mkdir(parents=True, exist_ok=True)

VAL = ROOT / "Val.csv"
TEST = ROOT / "Test.csv"
EXP2_VAL = ROOT / "modal_outputs" / "exp2_crossencoder_rerank" / "val_predictions.csv"
FULLCAP_VAL = ROOT / "reports" / "cluster_aware_selector_experiment" / "cluster_selector_fullcap_oof_choices.csv"
REG_VAL = ROOT / "reports" / "local_candidate_regressor_submission" / "val_oof_choices_with_answers.csv"

SUB_DIR = ROOT / "modal_outputs" / "exp7_submissions"
SUB_EXP7_CLUSTER = SUB_DIR / "submission_exp7_cluster_selector_trainval.csv"
SUB_EXP6_RERANKER = SUB_DIR / "submission_exp6_reranker_trainval_top50cluster.csv"
SUB_DENSE_TOP1 = SUB_DIR / "submission_exp7_dense_top1_trainval.csv"
SUB_LOCAL_REG = SUB_DIR / "submission_local_candidate_regressor_hgb.csv"
OUT_SUB = SUB_DIR / "submission_exp7_cluster_base_length_grouped_available_sources.csv"

BASELINE_EXP2 = 0.5892166283468145
RANDOM_STATE = 139
N_BINS = 5
MIN_N = 80
BASE_SOURCE = "exp7_cluster"


def rouge1(pred, ref):
    pred_toks = str(pred).split()
    ref_toks = str(ref).split()
    if not pred_toks or not ref_toks:
        return 0.0
    overlap = sum((Counter(pred_toks) & Counter(ref_toks)).values())
    return float(2 * overlap / (len(pred_toks) + len(ref_toks))) if overlap else 0.0


def target_cols_from_answer(df, answer_col="answer"):
    return pd.DataFrame(
        {
            "ID": df["ID"].tolist(),
            "TargetRLF1": df[answer_col].tolist(),
            "TargetR1F1": df[answer_col].tolist(),
            "TargetLLM": df[answer_col].tolist(),
        }
    )


def load_val_sources():
    val = pd.read_csv(VAL)[["ID", "subset", "input", "output"]]
    val["input_char_len"] = val["input"].astype(str).str.len()
    exp2 = pd.read_csv(EXP2_VAL)[["ID", "top1", "rerank", "reference"]]

    rows = []
    def add_source(name, pred_col, frame):
        merged = val.merge(frame[["ID", pred_col]], on="ID", how="left")
        tmp = val[["ID", "subset", "input_char_len"]].copy()
        tmp["source"] = name
        tmp["prediction"] = merged[pred_col].fillna("").astype(str)
        tmp["target_r1"] = [rouge1(p, r) for p, r in zip(tmp["prediction"], val["output"])]
        rows.append(tmp)

    fullcap = pd.read_csv(FULLCAP_VAL)
    fullcap = fullcap[fullcap["model"].eq("lightgbm_fullcap")][["ID", "candidate_answer", "target_r1"]]
    tmp = val[["ID", "subset", "input_char_len"]].merge(fullcap, on="ID", how="left")
    tmp["source"] = "exp7_cluster"
    tmp["prediction"] = tmp["candidate_answer"].fillna("").astype(str)
    tmp["target_r1"] = tmp["target_r1"].fillna(
        pd.Series([rouge1(p, r) for p, r in zip(tmp["prediction"], val["output"])])
    )
    rows.append(tmp[["ID", "subset", "input_char_len", "source", "prediction", "target_r1"]])

    add_source("exp6_reranker", "rerank", exp2)
    add_source("dense_top1", "top1", exp2)

    reg = pd.read_csv(REG_VAL)[["ID", "candidate_answer", "recomputed_r1"]]
    tmp = val[["ID", "subset", "input_char_len"]].merge(reg, on="ID", how="left")
    tmp["source"] = "local_regressor"
    tmp["prediction"] = tmp["candidate_answer"].fillna("").astype(str)
    tmp["target_r1"] = tmp["recomputed_r1"].fillna(
        pd.Series([rouge1(p, r) for p, r in zip(tmp["prediction"], val["output"])])
    )
    rows.append(tmp[["ID", "subset", "input_char_len", "source", "prediction", "target_r1"]])

    return val, pd.concat(rows, ignore_index=True)


def compute_edges(val):
    edges = {}
    for subset, group in val.groupby("subset"):
        qs = np.linspace(0, 1, N_BINS + 1)
        raw = group["input_char_len"].quantile(qs).to_numpy(dtype=float).copy()
        raw[0] = -np.inf
        raw[-1] = np.inf
        # Make edges strictly increasing for pd.cut, even on tiny or repeated-length subsets.
        for i in range(1, len(raw)):
            if raw[i] <= raw[i - 1]:
                raw[i] = raw[i - 1] + 1e-6
        edges[subset] = raw.tolist()
    return edges


def assign_bins(df, edges):
    out = df.copy()
    bins = []
    for subset, x in zip(out["subset"], out["input_char_len"]):
        e = edges[subset]
        bins.append(int(pd.cut([x], bins=e, labels=False, include_lowest=True)[0]))
    out["length_bin"] = bins
    return out


def learn_picks(train):
    picks = []
    for (subset, length_bin), group in train.groupby(["subset", "length_bin"]):
        stats = (
            group.groupby("source")
            .agg(score=("target_r1", "mean"), n=("target_r1", "size"))
            .reset_index()
        )
        stats = stats[stats["n"].ge(MIN_N)]
        source = BASE_SOURCE if stats.empty else stats.loc[stats["score"].idxmax(), "source"]
        picks.append({"subset": subset, "length_bin": int(length_bin), "source": source})
    return pd.DataFrame(picks)


def apply_picks(scored, ids, picks):
    chosen = scored[scored["ID"].isin(ids)].merge(picks, on=["subset", "length_bin", "source"], how="inner")
    chosen = chosen.drop_duplicates("ID")
    if chosen["ID"].nunique() != len(ids):
        missing = set(ids) - set(chosen["ID"])
        fallback = scored[
            scored["ID"].isin(missing) & scored["source"].eq(BASE_SOURCE)
        ].drop_duplicates("ID")
        chosen = pd.concat([chosen, fallback], ignore_index=True).drop_duplicates("ID")
    return chosen


def oof_audit(scored, val):
    ids = val[["ID", "subset"]].reset_index(drop=True)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    chosen = []
    for fold, (tr_i, va_i) in enumerate(folds.split(ids["ID"], ids["subset"]), start=1):
        train_ids = set(ids.iloc[tr_i]["ID"])
        valid_ids = set(ids.iloc[va_i]["ID"])
        picks = learn_picks(scored[scored["ID"].isin(train_ids)])
        out = apply_picks(scored, valid_ids, picks)
        chosen.append(out)
        fold_rows.append({"fold": fold, "score": float(out["target_r1"].mean()), "rows": int(len(out))})
    oof = pd.concat(chosen, ignore_index=True).drop_duplicates("ID")
    return oof, fold_rows


def make_submission(test_scored, picks):
    source_files = {
        "exp7_cluster": SUB_EXP7_CLUSTER,
        "exp6_reranker": SUB_EXP6_RERANKER,
        "dense_top1": SUB_DENSE_TOP1,
        "local_regressor": SUB_LOCAL_REG,
    }
    subs = {
        name: pd.read_csv(path)[["ID", "TargetR1F1"]].rename(columns={"TargetR1F1": name})
        for name, path in source_files.items()
    }
    wide = pd.read_csv(TEST)[["ID", "subset", "input"]]
    wide["input_char_len"] = wide["input"].astype(str).str.len()
    wide = assign_bins(wide, compute_edges(pd.read_csv(VAL)[["ID", "subset", "input"]].assign(input_char_len=lambda d: d["input"].astype(str).str.len())))
    wide = wide.merge(picks, on=["subset", "length_bin"], how="left")
    wide["source"] = wide["source"].fillna(BASE_SOURCE)
    for df in subs.values():
        wide = wide.merge(df, on="ID", how="left")
    answers = []
    for row in wide.itertuples(index=False):
        source = row.source
        answer = getattr(row, source)
        if pd.isna(answer) or str(answer) == "":
            answer = getattr(row, BASE_SOURCE)
            source = BASE_SOURCE
        answers.append(answer)
    out = pd.DataFrame({"ID": wide["ID"], "answer": answers})
    sub = target_cols_from_answer(out)
    sub.to_csv(OUT_SUB, index=False)
    audit = wide[["ID", "subset", "input_char_len", "length_bin", "source"]].copy()
    audit.to_csv(OUT / "test_source_choices.csv", index=False)
    return sub, audit


def run():
    val, scored = load_val_sources()
    edges = compute_edges(val)
    scored = assign_bins(scored, edges)

    source_scores = (
        scored.groupby("source")["target_r1"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .to_dict("records")
    )
    full_picks = learn_picks(scored)
    full_chosen = apply_picks(scored, set(val["ID"]), full_picks)
    oof, fold_rows = oof_audit(scored, val)
    sub, test_audit = make_submission(scored, full_picks)

    summary = {
        "submission": str(OUT_SUB),
        "base_source": BASE_SOURCE,
        "available_sources": ["exp7_cluster", "exp6_reranker", "dense_top1", "local_regressor"],
        "source_scores": source_scores,
        "full_validation_group_score": float(full_chosen["target_r1"].mean()),
        "oof_group_score": float(oof["target_r1"].mean()),
        "oof_gain_vs_exp2": float(oof["target_r1"].mean() - BASELINE_EXP2),
        "full_validation_gain_vs_exp2": float(full_chosen["target_r1"].mean() - BASELINE_EXP2),
        "fold_scores": fold_rows,
        "group_picks": full_picks.sort_values(["subset", "length_bin"]).to_dict("records"),
        "test_choice_counts": test_audit["source"].value_counts().to_dict(),
        "diff_vs_exp7_cluster": int(
            (
                sub["TargetR1F1"].astype(str).to_numpy()
                != pd.read_csv(SUB_EXP7_CLUSTER)["TargetR1F1"].astype(str).to_numpy()
            ).sum()
        ),
        "notes": [
            "No Modal compute used.",
            "This starts from submission_exp7_cluster_selector_trainval.csv as the base/fallback.",
            "Group choices are learned from validation labels; OOF score is the safer estimate.",
            "Only test-available sources are used.",
        ],
    }
    full_chosen.to_csv(OUT / "full_validation_group_choices.csv", index=False)
    oof.to_csv(OUT / "oof_group_choices.csv", index=False)
    full_picks.to_csv(OUT / "group_picks.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
