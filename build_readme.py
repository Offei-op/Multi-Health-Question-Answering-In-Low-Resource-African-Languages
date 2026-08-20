"""Build the project README from the persisted experiment summaries.

The workspace contains many large local artifacts that are intentionally not
committed to GitHub. This generator keeps the README tied to the JSON result
files that are committed, so the metric appendix can be regenerated after a
new experiment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def number(value: object, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def headline_rows() -> list[tuple[str, str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str, str]] = []

    def add(label: str, source: str, top1: object, rerank: object, oracle: object, delta: object) -> None:
        rows.append((label, source, number(top1), number(rerank), number(oracle), number(delta)))

    exp2 = load_json(ROOT / "modal_outputs/exp2_crossencoder_rerank/summary.json")
    add("Exp 2 — BGE-M3 cross-encoder, top-50", "modal_outputs/exp2_crossencoder_rerank/summary.json", exp2["top1"]["rouge1"], exp2["rerank"]["rouge1"], exp2["oracle"]["rouge1"], exp2["delta_rerank_vs_top1"])

    exp3 = load_json(ROOT / "modal_outputs/exp3_top100_crossencoder_rerank/summary.json")
    add("Exp 3 — BGE-M3 cross-encoder, top-100", "modal_outputs/exp3_top100_crossencoder_rerank/summary.json", exp3["top1"]["rouge1"], exp3["rerank"]["rouge1"], exp3["oracle"]["rouge1"], exp3["delta_rerank_vs_top1"])

    exp5 = load_json(ROOT / "modal_outputs/exp5_encoder_exp2_rerank_eval/summary.json")
    add("Exp 5 — mined encoder candidates + Exp 2 reranker", "modal_outputs/exp5_encoder_exp2_rerank_eval/summary.json", exp5["top1"]["rouge1"], exp5["rerank"]["rouge1"], exp5["oracle"]["rouge1"], exp5["delta_rerank_vs_exp2"])

    exp9 = load_json(ROOT / "modal_outputs/exp9_jina_multilingual_reranker/summary.json")
    add("Exp 9 — Jina multilingual reranker", "modal_outputs/exp9_jina_multilingual_reranker/summary.json", exp9["top1"]["rouge1"], exp9["rerank"]["rouge1"], exp9["oracle"]["rouge1"], exp9["delta_vs_top1"])

    exp17 = load_json(ROOT / "modal_outputs/exp17_answer_cluster_multivector/summary.json")
    best_cluster = max(exp17["leaderboard"], key=lambda row: row.get("top1_r1", float("-inf")))
    add("Exp 17 — answer-cluster multi-vector fusion", "modal_outputs/exp17_answer_cluster_multivector/summary.json", best_cluster["top1_r1"], None, best_cluster["oracle100_r1"], best_cluster["delta_oracle1_vs_row"])

    exp18 = load_json(ROOT / "modal_outputs/exp18_qwen3_cluster_listwise_reranker/summary.json")
    add("Exp 18 — Qwen3 listwise reranker", "modal_outputs/exp18_qwen3_cluster_listwise_reranker/summary.json", exp18["retrieval_top1"]["rouge1"], exp18["listwise_qwen3"]["rouge1"], exp18["oracle"]["rouge1"], exp18["delta_listwise_vs_retrieval"])

    selector = load_json(ROOT / "reports/deployable_source_selector/summary.json")
    add("Local deployable source selector (Extra Trees)", "reports/deployable_source_selector/summary.json", selector["exp2_rerank"], selector["best_selector"]["rouge1"], selector["deployable_oracle"], selector["best_selector"]["gain_vs_exp2_rerank"])

    return rows


def subset_rows() -> list[tuple[str, str, str, str, str]]:
    exp2 = load_json(ROOT / "modal_outputs/exp2_crossencoder_rerank/summary.json")
    rows = []
    for subset, metrics in exp2["top1"]["per_subset"].items():
        rows.append((subset, number(metrics["rouge1"]), number(exp2["rerank"]["per_subset"][subset]["rouge1"]), number(exp2["oracle"]["per_subset"][subset]["rouge1"]), number(exp2["rerank"]["per_subset"][subset]["rouge1"] - metrics["rouge1"])))
    return rows


def inventory() -> tuple[int, int, int, int, int, int]:
    files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    total = sum(p.stat().st_size for p in files)
    return (
        len(files),
        total,
        sum(p.suffix.lower() == ".py" for p in files),
        sum(p.suffix.lower() == ".ipynb" for p in files),
        sum(p.suffix.lower() == ".json" for p in files),
        sum(p.suffix.lower() == ".csv" for p in files),
    )


def write_local_manifest() -> None:
    ignored_suffixes = {".csv", ".safetensors", ".pt", ".pth", ".bin", ".pkl", ".zip", ".parquet", ".onnx", ".model", ".bak", ".docx", ".xlsx", ".html"}
    ignored_names = {"Train.csv", "Val.csv", "Test.csv"}
    artifacts = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.name in {"README.md", "LOCAL_ARTIFACTS.md"}:
            continue
        if path.name in ignored_names or path.suffix.lower() in ignored_suffixes:
            artifacts.append((path.stat().st_size, rel(path)))
    artifacts.sort(reverse=True)
    lines = [
        "# Local artifact manifest",
        "",
        "This file is generated by `python build_readme.py`. It records the large/raw artifacts kept in the local workspace and excluded from the GitHub-ready repository by `.gitignore`.",
        "",
        f"Excluded files: **{len(artifacts):,}**; total size: **{sum(size for size, _ in artifacts) / (1024 ** 3):.2f} GiB**.",
        "",
        "| Size (MiB) | Relative path |",
        "|---:|---|",
    ]
    for size, path in artifacts:
        lines.append(f"| {size / (1024 ** 2):.2f} | `{path}` |")
    (ROOT / "LOCAL_ARTIFACTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    write_local_manifest()
    file_count, total_bytes, py_count, notebook_count, json_count, csv_count = inventory()
    summary_paths = sorted(ROOT.rglob("summary.json"))
    headings = []
    for path in summary_paths:
        data = load_json(path)
        title = data.get("experiment") or rel(path)
        headings.append((title, rel(path), json.dumps(data, indent=2, ensure_ascii=False)))

    lines: list[str] = []
    lines.extend([
        "# Lalang: multilingual health-QA retrieval and reranking",
        "",
        "This repository is the reproducible research record for a multilingual health question-answering retrieval system. It contains the training and evaluation notebooks, Modal GPU experiment scripts, local selector experiments, persisted result summaries, model-card metadata, and the decision trail from dense retrieval through reranking and source selection.",
        "",
        "> **Repository scope.** The complete local workspace is approximately 18 GB. GitHub-ready contents are the source code, notebooks, small metadata/result JSON files, and this generated documentation. Large datasets, model weights, optimizer states, caches, and wide prediction tables remain local and are listed below because GitHub's file/storage limits make committing them directly inappropriate.",
        "",
        "## Research snapshot",
        "",
        "- Data split used by the main retrieval/reranking evaluations: **29,814 training rows** and **6,686 validation rows**.",
        "- Evaluation is disaggregated over eight subsets: `Aka_Gha`, `Amh_Eth`, `Eng_Eth`, `Eng_Gha`, `Eng_Ken`, `Eng_Uga`, `Lug_Uga`, and `Swa_Ken`.",
        "- Primary metrics are ROUGE-1 and ROUGE-L between the selected answer and the reference answer. `top1` is the first retrieved candidate, `rerank` is the learned rerank choice, and `oracle` is the best answer available inside the candidate pool.",
        "- The strongest single practical Modal result in the persisted main track is Exp 3's top-100 cross-encoder reranker: **0.5904 ROUGE-1**, versus **0.5395** for its top-1 candidate baseline and **0.6913** candidate-pool oracle.",
        "- The strongest local source-selection result recorded here is the extended ensemble at **0.6277 ROUGE-1**; its deployable-only Extra Trees selector is **0.6302 ROUGE-1** with a **0.6595** deployable oracle.",
        "",
        "## Main results",
        "",
        "| Experiment | Source summary | Top-1 R1 | Rerank R1 | Oracle R1 | Delta shown |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for label, source, top1, rerank, oracle, delta in headline_rows():
        lines.append(f"| {label} | [`{source}`]({source}) | {top1} | {rerank} | {oracle} | {delta} |")

    lines.extend([
        "",
        "`Delta shown` is the delta reported by that experiment; its denominator varies by experiment, so use the linked raw summary for the exact definition.",
        "",
        "## Exp 2 cross-encoder by subset",
        "",
        "This is the most complete common comparison because it reports top-1, reranked, oracle, ROUGE-1, and ROUGE-L for every validation subset.",
        "",
        "| Subset | Top-1 R1 | Rerank R1 | Oracle R1 | Rerank gain |",
        "|---|---:|---:|---:|---:|",
    ])
    for subset, top1, rerank, oracle, delta in subset_rows():
        lines.append(f"| {subset} | {top1} | {rerank} | {oracle} | {delta} |")

    lines.extend([
        "",
        "## Experiment map",
        "",
        "The table below is generated from every committed `summary.json`. The full JSON payload for each result—including hyperparameters, per-subset metrics, diagnostics, choice counts, and artifact paths—is reproduced in the metric appendix so no persisted summary metric is lost.",
        "",
        "| Experiment / result family | Summary |",
        "|---|---|",
    ])
    for title, source, _ in headings:
        lines.append(f"| {title} | [`{source}`]({source}) |")

    lines.extend([
        "",
        "## Chronological research record",
        "",
        "1. **Data and exploratory analysis.** `EDA.ipynb`, `build_pairs_laptop.py`, `build_pairs_laptop.ipynb`, and the starter/retrieval notebooks establish the `input`/`output`/`subset`/`ID` schema, data splits, candidate construction, and baseline evaluation.",
        "2. **Generation and retrieval baselines.** MT0/MT5 RAG notebooks and `retrieval_baselines.ipynb` establish generation and dense/lexical retrieval baselines.",
        "3. **Exp 1 — query-to-QA-document BGE-M3 LoRA.** Fine-tunes the query encoder with cached multiple-negatives ranking loss; the persisted comparison is baseline versus fine-tuned top-1 and oracle@20 recall.",
        "4. **Exp 2 — top-50 BGE-M3 cross-encoder regression.** Fine-tunes a cross-encoder to predict answer ROUGE and reranks a 50-candidate pool. This is the main practical reranking baseline.",
        "5. **Exp 3 — top-100 cross-encoder.** Tests a larger candidate pool and a faster ROUGE-1 regression setup.",
        "6. **Exp 4 — pairwise hard-negative reranking.** Replaces regression targets with pairwise preference training; diagnostics quantify wins, hurts, exact-oracle misses, and catastrophic jumps.",
        "7. **Exp 5 — BGE-M3 encoder mining.** Improves candidate mining with dense and lexical retrieval, then feeds the resulting pool to the Exp 2 reranker.",
        "8. **Exp 7 — cluster/length-grouped source selection.** Tests answer-cluster retrieval, length-grouped choices, and candidate-source selection for deployable submissions.",
        "9. **Exp 8 — Ghana specialist.** Trains a Ghana-subset specialist encoder/reranker and compares it with the global models.",
        "10. **Exp 9 — Jina multilingual reranker.** Tests `jinaai/jina-reranker-v2-base-multilingual` against the BGE-M3 cross-encoder.",
        "11. **Exp 10–13 — Luganda/Uganda specialization and merging.** Tests a Luganda specialist, E5/BGE candidate merging, and a selector over the merged candidates.",
        "12. **Exp 11 — base encoder benchmark.** Compares `BAAI/bge-m3`, `intfloat/multilingual-e5-base`, and `intfloat/multilingual-e5-large` as retrieval encoders.",
        "13. **Exp 14 — question-only selector.** Tests a selector using question-side features and complements the Exp 2 reranker.",
        "14. **Exp 17 — answer-cluster multi-vector retrieval.** Measures duplicate-answer structure, exact-answer-in-train rates, cluster strategies, exact recall@K, and oracle ROUGE@K.",
        "15. **Exp 18 — Qwen3 listwise reranking.** Tests `Qwen/Qwen3-Reranker-0.6B` on cluster-mined groups; the recorded result is below the cluster-fusion retrieval baseline, which is itself documented in the raw summary.",
        "16. **Local selector track.** The `reports/` summaries record leakage audits, oracle audits, family meta-learners, candidate regressors, fallback rules, clean-source selectors, and the extended ensemble used to choose among existing candidate sources.",
        "",
        "## Evaluation protocol and metric definitions",
        "",
        "- **ROUGE-1 (`rouge1`, R1):** unigram overlap F-measure between a selected/generated answer and the reference answer.",
        "- **ROUGE-L (`rougeL`, RL):** longest-common-subsequence-based overlap score; it is reported for the main answer-ranking experiments where available.",
        "- **Top-1:** score of the first candidate from the retrieval/encoder stage.",
        "- **Rerank:** score of the candidate selected by the learned reranker or selector.",
        "- **Oracle:** the best reference-overlap score available among the evaluated candidate pool. It is an upper bound on ranking quality for that pool, not a deployable score.",
        "- **Exact recall@K:** whether an exact normalized answer match appears in the first K candidates; reported in the answer-cluster experiments.",
        "- **Coverage:** fraction of validation rows for which a source produces a usable candidate; source-selection reports include it where relevant.",
        "- **No-leak / fold-safe results:** reports with these names exclude validation-derived target information from the corresponding training fold; full-cap results are retained as diagnostic/oracle-oriented comparisons and are labeled accordingly.",
        "",
        "## Reproduction",
        "",
        "### Local analysis",
        "",
        "1. Restore the local datasets and model artifacts listed in `LOCAL_ARTIFACTS.md` (this file is generated alongside the README).",
        "2. Use Python 3.12 or the environment used by the notebooks. The Modal scripts declare their cloud dependencies inline; the core stack is PyTorch, Transformers, Sentence Transformers, Datasets, PEFT, pandas, NumPy, scikit-learn, rouge-score, tqdm, and Modal.",
        "3. Run the notebooks in chronological order for exploratory work, or run the corresponding `modal_*.py` and `local_*.py` files for the persisted experiment families.",
        "4. Keep the generated `summary.json` beside each experiment output and run `python build_readme.py` to refresh this README's experiment index, headline tables, and full metric appendix.",
        "",
        "### Modal experiments",
        "",
        "The Modal scripts define the image dependencies, GPU type, volume paths, seeds, candidate K, training hyperparameters, and output locations. Before running them, update the hard-coded local/remote data paths if your Modal volume layout differs. Most persisted cloud runs used an NVIDIA L40S.",
        "",
        "## Repository layout",
        "",
        "| Path | Role |",
        "|---|---|",
        "| `modal_*.py` | Cloud training/evaluation jobs and experiment entry points. |",
        "| `local_*.py` | Local selectors, audits, fallback rules, and submission construction. |",
        "| `*.ipynb` | Exploratory analysis, data construction, retrieval, reranking, and generation notebooks. |",
        "| `modal_outputs/` | Persisted Modal result summaries and selected output artifacts. |",
        "| `reports/` | Local validation reports, leakage/oracle audits, selector leaderboards, and error mining. |",
        "| `Bgem3-finetune/`, `lora_adapter/`, `trainer/`, `mt0-rag-finetuned/` | Fine-tuning metadata, adapters, and model-card files; large weights are local-only. |",
        "| `build_readme.py` | Regenerates this README from the result summaries. |",
        "",
        "## Local-only artifact policy",
        "",
        f"The local workspace inventory at README generation time is **{file_count:,} files / {total_bytes / (1024 ** 3):.2f} GiB**, including {py_count} Python files, {notebook_count} notebooks, {json_count} JSON files, and {csv_count} CSV files. GitHub-ready commits exclude the following classes via `.gitignore`:",
        "",
        "- raw `Train.csv`, `Val.csv`, and `Test.csv` data plus generated CSV prediction tables;",
        "- model weights and training state (`*.safetensors`, `*.pt`, `*.bin`, `*.pkl`);",
        "- parquet candidate pools, ZIP archives, spreadsheets, and rendered review workbooks;",
        "- Python caches and other transient output.",
        "",
        "The exact local paths and sizes are recorded in `LOCAL_ARTIFACTS.md`, which is generated from the workspace inventory. Keep that manifest with the local workspace when moving the full research archive or migrate the excluded artifacts to object storage/model hosting if public release is required.",
        "",
        "## Limitations and research conclusions",
        "",
        "- The best candidate-pool oracle remains materially above the deployable ranker, so retrieval coverage and ranking calibration are still the main bottlenecks.",
        "- Gains vary substantially by language/country subset; the common Exp 2 reranker is strongest on the English/Kenya, English/Uganda, Swahili/Kenya, and Luganda/Uganda subsets, while Akan/Ghana, Amharic/Ethiopia, and English/Ghana remain harder.",
        "- Pairwise hard-negative training and Qwen3 listwise reranking did not beat the common Exp 2 cross-encoder in the persisted comparisons.",
        "- Full-cap, oracle-assisted, and validation-derived source choices are useful diagnostics but should not be presented as leakage-free deployable performance; the reports label the safer no-leak/fold-safe alternatives.",
        "- This is a research artifact, not a clinical decision system. Do not use its predictions for diagnosis, treatment, triage, or patient-specific medical advice.",
        "",
        "## Metric appendix: every persisted summary",
        "",
        "The following sections reproduce every `summary.json` in the workspace at the time this README was generated. These blocks are intentionally verbose: they preserve the complete metric, hyperparameter, diagnostic, and artifact record in one browsable document.",
        "",
    ])

    for title, source, payload in headings:
        safe_title = title.replace("<", "&lt;").replace(">", "&gt;")
        lines.extend([
            f"<details><summary>{safe_title} — <code>{source}</code></summary>",
            "",
            "```json",
            payload,
            "```",
            "",
            "</details>",
            "",
        ])

    lines.extend([
        "## License and data notice",
        "",
        "No license was present in the original workspace. Add a license before redistributing the code. Confirm the competition/dataset terms and the licenses of all base models before publishing the excluded data, weights, or derived artifacts.",
        "",
        "## Citation",
        "",
        "If this research is used, cite the repository and the upstream datasets/models used by the relevant experiment. The exact upstream model identifiers are preserved in the experiment summaries and scripts.",
        "",
    ])

    (ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
