from pathlib import Path
import json

import pandas as pd


ROOT = Path(r"C:\Users\Papa Offei\Documents\lalang")
REPORT_DIR = ROOT / "reports" / "test_source_feasibility_audit"

VAL_SOURCES = [
    "exp1_baseline_qa_doc",
    "exp1_finetuned_qa_doc",
    "exp2_top1",
    "exp2_rerank",
    "exp3_top100_rerank",
    "exp4_pairwise",
    "exp5_encoder_exp2_rerank",
    "exp8_ghana_grouped",
    "exp9_jina",
    "exp10_lug_global",
    "exp10_lug_specialized",
    "exp12_lug_merged",
    "exp14_qonly",
]

KNOWN_TEST_EQUIVALENTS = {
    "exp2_top1": [
        ROOT / "modal_outputs/test_predictions_best_setup/submission_encoder_top1_train_val.csv",
        ROOT / "modal_outputs/exp7_submissions/submission_exp7_dense_top1_trainval.csv",
    ],
    "exp2_rerank": [
        ROOT / "modal_outputs/test_predictions_best_setup/submission_best_bgem3_rerank_train_val.csv",
        ROOT / "modal_outputs/exp7_submissions/submission_exp6_reranker_trainval_top50cluster.csv",
    ],
    "local_regressor": [
        ROOT / "modal_outputs/exp7_submissions/submission_local_candidate_regressor_hgb.csv",
    ],
    "cluster_selector": [
        ROOT / "modal_outputs/exp7_submissions/submission_exp7_cluster_selector_trainval.csv",
    ],
}


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in VAL_SOURCES:
        equivalents = KNOWN_TEST_EQUIVALENTS.get(src, [])
        existing = [str(p) for p in equivalents if p.exists()]
        rows.append(
            {
                "validation_source": src,
                "has_known_local_test_equivalent": bool(existing),
                "local_test_files": " | ".join(existing),
                "conversion_without_modal": "yes" if existing else "unknown/no",
            }
        )

    # Also list submission-like files we might be able to use as sources even if not mapped.
    submission_files = []
    for folder in [ROOT / "modal_outputs", ROOT / "reports", ROOT]:
        if folder.exists():
            for path in folder.rglob("*.csv"):
                name = path.name.lower()
                if "submission" in name or "test" in name:
                    try:
                        cols = list(pd.read_csv(path, nrows=0).columns)
                    except Exception:
                        cols = []
                    submission_files.append({"path": str(path), "columns": cols})

    summary = {
        "mapped_sources": rows,
        "submission_like_csv_count": len(submission_files),
        "submission_like_csvs": submission_files,
        "notes": [
            "This audit only maps local files already present. It does not claim unavailable sources are impossible, only that no local test equivalent was found.",
            "Sources like exp3/exp4/exp5/exp9/exp14 may require model inference to convert, which could be local if model artifacts are available, but not from existing submission CSVs.",
        ],
    }
    pd.DataFrame(rows).to_csv(REPORT_DIR / "source_test_feasibility.csv", index=False)
    pd.DataFrame(submission_files).to_csv(REPORT_DIR / "submission_like_files.csv", index=False)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
