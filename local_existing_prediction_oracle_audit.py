from collections import Counter
from pathlib import Path
import json

import pandas as pd


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
VAL = ROOT / "Val.csv"
REPORT_DIR = ROOT / "reports" / "existing_prediction_oracle_audit"

SOURCES = [
    ("exp1_baseline_qa_doc", ROOT / "modal_outputs/exp1_query_to_qa_doc/baseline_query_to_qa_doc_val_predictions.csv", ["prediction", "pred", "top1", "answer"]),
    ("exp1_finetuned_qa_doc", ROOT / "modal_outputs/exp1_query_to_qa_doc/finetuned_query_to_qa_doc_val_predictions.csv", ["prediction", "pred", "top1", "answer"]),
    ("exp2_top1", ROOT / "modal_outputs/exp2_crossencoder_rerank/val_predictions.csv", ["top1"]),
    ("exp2_rerank", ROOT / "modal_outputs/exp2_crossencoder_rerank/val_predictions.csv", ["rerank"]),
    ("exp3_top100_rerank", ROOT / "modal_outputs/exp3_top100_crossencoder_rerank/val_predictions.csv", ["rerank"]),
    ("exp4_pairwise", ROOT / "modal_outputs/exp4_pairwise_hardneg_rerank/val_predictions.csv", ["rerank"]),
    ("exp5_encoder_exp2_rerank", ROOT / "modal_outputs/exp5_encoder_exp2_rerank_eval/val_predictions.csv", ["rerank"]),
    ("exp8_ghana_grouped", ROOT / "modal_outputs/exp8_ghana_grouped_encoder_reranker/ghana_val_predictions.csv", ["ghana_rerank", "rerank", "prediction"]),
    ("exp9_jina", ROOT / "modal_outputs/exp9_jina_multilingual_reranker/val_predictions.csv", ["rerank"]),
    ("exp10_lug_global", ROOT / "modal_outputs/exp10_lug_uga_reranker/lug_val_predictions.csv", ["global_rerank", "rerank", "prediction"]),
    ("exp10_lug_specialized", ROOT / "modal_outputs/exp10_lug_uga_reranker/lug_val_predictions.csv", ["lug_rerank", "specialized_rerank"]),
    ("exp12_lug_merged", ROOT / "modal_outputs/exp12_lug_e5_merge_rerank/lug_val_predictions.csv", ["merged_rerank", "rerank"]),
    ("exp13_lug_selector", ROOT / "reports/exp13_lug_merged_selector/oof_best_selector_choices.csv", ["selected_answer", "prediction", "answer"]),
    ("exp14_qonly", ROOT / "modal_outputs/exp14_qonly_gemma_reranker_recovered", ["qonly_bgem3"]),
    ("local_regressor_oof", ROOT / "reports/local_candidate_regressor_submission/val_oof_choices.csv", ["selected_answer", "prediction", "answer"]),
    ("local_regressor_margin_gate", ROOT / "reports/local_candidate_regressor_submission/val_margin_gate_rows.csv", ["selected_answer", "prediction", "answer"]),
    ("cluster_selector_fast", ROOT / "reports/cluster_aware_selector_experiment/cluster_selector_fast_oof_choices.csv", ["selected_answer", "prediction", "answer"]),
    ("cluster_selector_fullcap", ROOT / "reports/cluster_aware_selector_experiment/cluster_selector_fullcap_oof_choices.csv", ["selected_answer", "prediction", "answer"]),
    ("cluster_selector_noleak", ROOT / "reports/cluster_aware_selector_experiment/cluster_selector_noleak_oof_choices.csv", ["selected_answer", "prediction", "answer"]),
]


def rouge1(pred, ref):
    pred_toks = str(pred).strip().split()
    ref_toks = str(ref).strip().split()
    if not pred_toks or not ref_toks:
        return 0.0
    overlap = sum((Counter(pred_toks) & Counter(ref_toks)).values())
    if overlap == 0:
        return 0.0
    return float(2.0 * overlap / (len(pred_toks) + len(ref_toks)))


def find_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    lower = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lower:
            return lower[col.lower()]
    return None


def normalize_source(name, path, candidates, val):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "ID" not in df.columns:
        return None
    pred_col = find_col(df, candidates)
    if pred_col is None:
        return None
    out = val[["ID", "subset", "output"]].merge(df[["ID", pred_col]], on="ID", how="left")
    out = out.rename(columns={pred_col: name})
    out[name] = out[name].fillna("")
    return out[["ID", name]]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(VAL)
    val["ID"] = val["ID"].astype(str)
    val["subset"] = val["subset"].astype(str)
    val["output"] = val["output"].fillna("").astype(str)
    base = val[["ID", "subset", "output"]].copy()

    loaded = []
    missing = []
    for name, path, candidates in SOURCES:
        src = normalize_source(name, path, candidates, val)
        if src is None:
            missing.append({"name": name, "path": str(path)})
            continue
        base = base.merge(src, on="ID", how="left")
        loaded.append(name)

    score_rows = []
    for name in loaded:
        base[name] = base[name].fillna("")
        scores = [rouge1(pred, ref) for pred, ref in zip(base[name], base["output"])]
        base[f"{name}_r1"] = scores
        score_rows.append(
            {
                "source": name,
                "coverage": float((base[name].astype(str) != "").mean()),
                "rouge1": float(pd.Series(scores).mean()),
            }
        )

    pred_cols = loaded
    r1_cols = [f"{name}_r1" for name in pred_cols]
    base["oracle_existing_sources_r1"] = base[r1_cols].max(axis=1)
    base["oracle_existing_source"] = base[r1_cols].idxmax(axis=1).str.replace("_r1", "", regex=False)

    best_source = max(score_rows, key=lambda x: x["rouge1"])
    summary = {
        "loaded_sources": loaded,
        "missing_sources": missing,
        "best_single_source": best_source,
        "oracle_existing_sources_r1": float(base["oracle_existing_sources_r1"].mean()),
        "oracle_gain_vs_best_single": float(base["oracle_existing_sources_r1"].mean() - best_source["rouge1"]),
        "oracle_gain_vs_exp2_rerank": float(base["oracle_existing_sources_r1"].mean() - base["exp2_rerank_r1"].mean()) if "exp2_rerank_r1" in base else None,
    }

    source_scores = pd.DataFrame(score_rows).sort_values("rouge1", ascending=False)
    per_subset = []
    for subset, grp in base.groupby("subset"):
        row = {
            "subset": subset,
            "oracle_existing_sources_r1": float(grp["oracle_existing_sources_r1"].mean()),
        }
        for name in loaded:
            row[name] = float(grp[f"{name}_r1"].mean())
        per_subset.append(row)
    per_subset = pd.DataFrame(per_subset)

    base.to_csv(REPORT_DIR / "existing_prediction_oracle_rows.csv", index=False)
    source_scores.to_csv(REPORT_DIR / "source_scores.csv", index=False)
    per_subset.to_csv(REPORT_DIR / "per_subset_scores.csv", index=False)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(source_scores.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
