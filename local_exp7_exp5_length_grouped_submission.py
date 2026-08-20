from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
OUT = ROOT / "reports" / "exp7_exp5_length_grouped_submission"
OUT.mkdir(parents=True, exist_ok=True)

VAL = ROOT / "Val.csv"
TEST = ROOT / "Test.csv"
EXP7_VAL = ROOT / "reports/cluster_aware_selector_experiment/cluster_selector_fullcap_oof_choices.csv"
EXP5_VAL = ROOT / "modal_outputs/exp5_encoder_exp2_rerank_eval/val_predictions.csv"
SUB_EXP7 = ROOT / "modal_outputs/exp7_submissions/submission_exp7_cluster_selector_trainval.csv"
SUB_EXP5 = ROOT / "modal_outputs/exp5_encoder_exp2_test_predictions_files/submission_exp5_encoder_exp2_rerank_trainval.csv"
OUT_SUB = ROOT / "modal_outputs/exp5_encoder_exp2_test_predictions_files/submission_exp7_base_exp5_length_grouped.csv"

BASELINE_EXP2 = 0.5892166283468145
RANDOM_STATE = 139
BASE_SOURCE = "exp7_cluster"
N_BINS = 5
MIN_N = 80


def rouge1(pred, ref):
    p = str(pred).split()
    r = str(ref).split()
    if not p or not r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    return float(2 * overlap / (len(p) + len(r))) if overlap else 0.0


def compute_edges(val):
    edges = {}
    for subset, group in val.groupby("subset"):
        raw = group["input_char_len"].quantile(np.linspace(0, 1, N_BINS + 1)).to_numpy(dtype=float).copy()
        raw[0] = -np.inf
        raw[-1] = np.inf
        for i in range(1, len(raw)):
            if raw[i] <= raw[i - 1]:
                raw[i] = raw[i - 1] + 1e-6
        edges[subset] = raw.tolist()
    return edges


def assign_bins(df, edges):
    out = df.copy()
    bins = []
    for subset, x in zip(out["subset"], out["input_char_len"]):
        bins.append(int(pd.cut([x], bins=edges[subset], labels=False, include_lowest=True)[0]))
    out["length_bin"] = bins
    return out


def load_val_sources():
    val = pd.read_csv(VAL)[["ID", "subset", "input", "output"]]
    val["input_char_len"] = val["input"].astype(str).str.len()

    exp7 = pd.read_csv(EXP7_VAL)
    exp7 = exp7[exp7["model"].eq("lightgbm_fullcap")][["ID", "candidate_answer", "target_r1"]]
    e7 = val[["ID", "subset", "input_char_len"]].merge(exp7, on="ID", how="left")
    e7["source"] = "exp7_cluster"
    e7["prediction"] = e7["candidate_answer"].fillna("").astype(str)
    e7["target_r1"] = e7["target_r1"].fillna(
        pd.Series([rouge1(p, r) for p, r in zip(e7["prediction"], val["output"])])
    )

    exp5 = pd.read_csv(EXP5_VAL)[["ID", "rerank"]]
    e5 = val[["ID", "subset", "input_char_len", "output"]].merge(exp5, on="ID", how="left")
    e5["source"] = "exp5_encoder_exp2"
    e5["prediction"] = e5["rerank"].fillna("").astype(str)
    e5["target_r1"] = [rouge1(p, r) for p, r in zip(e5["prediction"], e5["output"])]

    scored = pd.concat(
        [
            e7[["ID", "subset", "input_char_len", "source", "prediction", "target_r1"]],
            e5[["ID", "subset", "input_char_len", "source", "prediction", "target_r1"]],
        ],
        ignore_index=True,
    )
    return val, scored


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
        fallback = scored[scored["ID"].isin(missing) & scored["source"].eq(BASE_SOURCE)].drop_duplicates("ID")
        chosen = pd.concat([chosen, fallback], ignore_index=True).drop_duplicates("ID")
    return chosen


def oof_score(scored, val):
    ids = val[["ID", "subset"]].reset_index(drop=True)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fold_scores = []
    for fold, (tr_i, va_i) in enumerate(folds.split(ids["ID"], ids["subset"]), start=1):
        tr_ids = set(ids.iloc[tr_i]["ID"])
        va_ids = set(ids.iloc[va_i]["ID"])
        picks = learn_picks(scored[scored["ID"].isin(tr_ids)])
        chosen = apply_picks(scored, va_ids, picks)
        rows.append(chosen)
        fold_scores.append({"fold": fold, "score": float(chosen["target_r1"].mean())})
    oof = pd.concat(rows, ignore_index=True).drop_duplicates("ID")
    return oof, fold_scores


def make_test_submission(edges, picks):
    test = pd.read_csv(TEST)[["ID", "subset", "input"]]
    test["input_char_len"] = test["input"].astype(str).str.len()
    test = assign_bins(test, edges)
    test = test.merge(picks, on=["subset", "length_bin"], how="left")
    test["source"] = test["source"].fillna(BASE_SOURCE)
    exp7 = pd.read_csv(SUB_EXP7)[["ID", "TargetR1F1"]].rename(columns={"TargetR1F1": "exp7_cluster"})
    exp5 = pd.read_csv(SUB_EXP5)[["ID", "TargetR1F1"]].rename(columns={"TargetR1F1": "exp5_encoder_exp2"})
    test = test.merge(exp7, on="ID", how="left").merge(exp5, on="ID", how="left")
    answers = []
    for row in test.itertuples(index=False):
        ans = getattr(row, row.source)
        if pd.isna(ans) or str(ans) == "":
            ans = row.exp7_cluster
        answers.append(ans)
    sub = pd.DataFrame({"ID": test["ID"], "TargetRLF1": answers, "TargetR1F1": answers, "TargetLLM": answers})
    sub.to_csv(OUT_SUB, index=False)
    test[["ID", "subset", "input_char_len", "length_bin", "source"]].to_csv(OUT / "test_source_choices.csv", index=False)
    return sub, test


def run():
    val, scored = load_val_sources()
    edges = compute_edges(val)
    scored = assign_bins(scored, edges)
    full_picks = learn_picks(scored)
    full = apply_picks(scored, set(val["ID"]), full_picks)
    oof, fold_scores = oof_score(scored, val)
    sub, test_choices = make_test_submission(edges, full_picks)
    exp7 = pd.read_csv(SUB_EXP7)
    summary = {
        "submission": str(OUT_SUB),
        "sources": ["exp7_cluster", "exp5_encoder_exp2"],
        "source_scores": scored.groupby("source")["target_r1"].mean().to_dict(),
        "full_validation_group_score": float(full["target_r1"].mean()),
        "oof_group_score": float(oof["target_r1"].mean()),
        "oof_gain_vs_exp2": float(oof["target_r1"].mean() - BASELINE_EXP2),
        "fold_scores": fold_scores,
        "group_picks": full_picks.sort_values(["subset", "length_bin"]).to_dict("records"),
        "test_choice_counts": test_choices["source"].value_counts().to_dict(),
        "diff_vs_exp7": int((sub["TargetR1F1"].astype(str) != exp7["TargetR1F1"].astype(str)).sum()),
        "notes": [
            "Exp7 validation counterpart is the closest available lightgbm_fullcap OOF source.",
            "This is a conservative exp7-base gate using only exp5 as override.",
        ],
    }
    full_picks.to_csv(OUT / "group_picks.csv", index=False)
    oof.to_csv(OUT / "oof_choices.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
