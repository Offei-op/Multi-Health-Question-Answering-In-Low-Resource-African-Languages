# Lalang: multilingual health-QA retrieval and reranking

This repository is the reproducible research record for a multilingual health question-answering retrieval system. It contains the training and evaluation notebooks, Modal GPU experiment scripts, local selector experiments, persisted result summaries, model-card metadata, and the decision trail from dense retrieval through reranking and source selection.

> **Repository scope.** The complete local workspace is approximately 18 GB. GitHub-ready contents are the source code, notebooks, small metadata/result JSON files, and this generated documentation. Large datasets, model weights, optimizer states, caches, and wide prediction tables remain local and are listed below because GitHub's file/storage limits make committing them directly inappropriate.

## Research snapshot

- Data split used by the main retrieval/reranking evaluations: **29,814 training rows** and **6,686 validation rows**.
- Evaluation is disaggregated over eight subsets: `Aka_Gha`, `Amh_Eth`, `Eng_Eth`, `Eng_Gha`, `Eng_Ken`, `Eng_Uga`, `Lug_Uga`, and `Swa_Ken`.
- Primary metrics are ROUGE-1 and ROUGE-L between the selected answer and the reference answer. `top1` is the first retrieved candidate, `rerank` is the learned rerank choice, and `oracle` is the best answer available inside the candidate pool.
- The strongest single practical Modal result in the persisted main track is Exp 3's top-100 cross-encoder reranker: **0.5904 ROUGE-1**, versus **0.5395** for its top-1 candidate baseline and **0.6913** candidate-pool oracle.
- The strongest local source-selection result recorded here is the extended ensemble at **0.6277 ROUGE-1**; its deployable-only Extra Trees selector is **0.6302 ROUGE-1** with a **0.6595** deployable oracle.

## Main results

| Experiment | Source summary | Top-1 R1 | Rerank R1 | Oracle R1 | Delta shown |
|---|---|---:|---:|---:|---:|
| Exp 2 — BGE-M3 cross-encoder, top-50 | [`modal_outputs/exp2_crossencoder_rerank/summary.json`](modal_outputs/exp2_crossencoder_rerank/summary.json) | 0.5395 | 0.5892 | 0.6837 | 0.0497 |
| Exp 3 — BGE-M3 cross-encoder, top-100 | [`modal_outputs/exp3_top100_crossencoder_rerank/summary.json`](modal_outputs/exp3_top100_crossencoder_rerank/summary.json) | 0.5395 | 0.5904 | 0.6913 | 0.0509 |
| Exp 5 — mined encoder candidates + Exp 2 reranker | [`modal_outputs/exp5_encoder_exp2_rerank_eval/summary.json`](modal_outputs/exp5_encoder_exp2_rerank_eval/summary.json) | 0.5464 | 0.5903 | 0.6890 | 0.0011 |
| Exp 9 — Jina multilingual reranker | [`modal_outputs/exp9_jina_multilingual_reranker/summary.json`](modal_outputs/exp9_jina_multilingual_reranker/summary.json) | 0.5395 | 0.5770 | 0.6837 | 0.0375 |
| Exp 17 — answer-cluster multi-vector fusion | [`modal_outputs/exp17_answer_cluster_multivector/summary.json`](modal_outputs/exp17_answer_cluster_multivector/summary.json) | 0.5569 | — | 0.6994 | 0.0100 |
| Exp 18 — Qwen3 listwise reranker | [`modal_outputs/exp18_qwen3_cluster_listwise_reranker/summary.json`](modal_outputs/exp18_qwen3_cluster_listwise_reranker/summary.json) | 0.5569 | 0.5435 | 0.6853 | -0.0134 |
| Local deployable source selector (Extra Trees) | [`reports/deployable_source_selector/summary.json`](reports/deployable_source_selector/summary.json) | 0.5892 | 0.6302 | 0.6595 | 0.0410 |

`Delta shown` is the delta reported by that experiment; its denominator varies by experiment, so use the linked raw summary for the exact definition.

## Exp 2 cross-encoder by subset

This is the most complete common comparison because it reports top-1, reranked, oracle, ROUGE-1, and ROUGE-L for every validation subset.

| Subset | Top-1 R1 | Rerank R1 | Oracle R1 | Rerank gain |
|---|---:|---:|---:|---:|
| Aka_Gha | 0.2903 | 0.3216 | 0.3950 | 0.0313 |
| Amh_Eth | 0.1711 | 0.1907 | 0.3079 | 0.0196 |
| Eng_Eth | 0.5629 | 0.6646 | 0.7506 | 0.1017 |
| Eng_Gha | 0.2831 | 0.3026 | 0.3844 | 0.0195 |
| Eng_Ken | 0.7992 | 0.8199 | 0.9202 | 0.0207 |
| Eng_Uga | 0.8171 | 0.8650 | 0.9380 | 0.0479 |
| Lug_Uga | 0.5587 | 0.6784 | 0.8533 | 0.1197 |
| Swa_Ken | 0.7941 | 0.8311 | 0.9210 | 0.0370 |

## Experiment map

The table below is generated from every committed `summary.json`. The full JSON payload for each result—including hyperparameters, per-subset metrics, diagnostics, choice counts, and artifact paths—is reproduced in the metric appendix so no persisted summary metric is lost.

| Experiment / result family | Summary |
|---|---|
| exp10_lug_uga_reranker | [`modal_outputs/exp10_lug_uga_reranker/summary.json`](modal_outputs/exp10_lug_uga_reranker/summary.json) |
| exp11_base_encoder_benchmark | [`modal_outputs/exp11_base_encoder_benchmark/summary.json`](modal_outputs/exp11_base_encoder_benchmark/summary.json) |
| exp12_lug_e5_merge_rerank | [`modal_outputs/exp12_lug_e5_merge_rerank/summary.json`](modal_outputs/exp12_lug_e5_merge_rerank/summary.json) |
| exp17_answer_cluster_multivector | [`modal_outputs/exp17_answer_cluster_multivector/summary.json`](modal_outputs/exp17_answer_cluster_multivector/summary.json) |
| exp18_qwen3_cluster_listwise_reranker | [`modal_outputs/exp18_qwen3_cluster_listwise_reranker/summary.json`](modal_outputs/exp18_qwen3_cluster_listwise_reranker/summary.json) |
| query_to_qa_doc_bgem3_lora | [`modal_outputs/exp1_query_to_qa_doc/summary.json`](modal_outputs/exp1_query_to_qa_doc/summary.json) |
| ft_bgem3_top50_crossencoder_rouge_regression | [`modal_outputs/exp2_crossencoder_rerank/summary.json`](modal_outputs/exp2_crossencoder_rerank/summary.json) |
| exp2_val_candidate_scores | [`modal_outputs/exp2_val_candidate_scores/summary.json`](modal_outputs/exp2_val_candidate_scores/summary.json) |
| ft_bgem3_top100_crossencoder_fast_rouge1_regression_pairs16 | [`modal_outputs/exp3_top100_crossencoder_rerank/summary.json`](modal_outputs/exp3_top100_crossencoder_rerank/summary.json) |
| ft_bgem3_top50_pairwise_hardneg_rerank | [`modal_outputs/exp4_pairwise_hardneg_rerank/summary.json`](modal_outputs/exp4_pairwise_hardneg_rerank/summary.json) |
| exp5_bgem3_encoder_mining_v2 | [`modal_outputs/exp5_bgem3_encoder_mining_v2/summary.json`](modal_outputs/exp5_bgem3_encoder_mining_v2/summary.json) |
| exp5_encoder_candidates_scored_by_exp2_reranker | [`modal_outputs/exp5_encoder_exp2_rerank_eval/summary.json`](modal_outputs/exp5_encoder_exp2_rerank_eval/summary.json) |
| exp5_encoder_exp2_rerank_test_predictions | [`modal_outputs/exp5_encoder_exp2_test_predictions_files/summary.json`](modal_outputs/exp5_encoder_exp2_test_predictions_files/summary.json) |
| exp8_ghana_grouped_encoder_reranker | [`modal_outputs/exp8_ghana_grouped_encoder_reranker/summary.json`](modal_outputs/exp8_ghana_grouped_encoder_reranker/summary.json) |
| exp9_jina_multilingual_reranker | [`modal_outputs/exp9_jina_multilingual_reranker/summary.json`](modal_outputs/exp9_jina_multilingual_reranker/summary.json) |
| test_predictions_best_bgem3_top50_crossencoder_rerank | [`modal_outputs/test_predictions_best_setup/summary.json`](modal_outputs/test_predictions_best_setup/summary.json) |
| reports/candidate_ranker_current_best_gate/summary.json | [`reports/candidate_ranker_current_best_gate/summary.json`](reports/candidate_ranker_current_best_gate/summary.json) |
| reports/clean_deployable_source_selector/summary.json | [`reports/clean_deployable_source_selector/summary.json`](reports/clean_deployable_source_selector/summary.json) |
| reports/cluster_fast_gate/summary.json | [`reports/cluster_fast_gate/summary.json`](reports/cluster_fast_gate/summary.json) |
| reports/deployable_source_selector/summary.json | [`reports/deployable_source_selector/summary.json`](reports/deployable_source_selector/summary.json) |
| reports/existing_prediction_oracle_audit/summary.json | [`reports/existing_prediction_oracle_audit/summary.json`](reports/existing_prediction_oracle_audit/summary.json) |
| existing_source_candidate_level_selector | [`reports/existing_source_selector/summary.json`](reports/existing_source_selector/summary.json) |
| existing_source_selector_with_fold_safe_target_encoding | [`reports/existing_source_selector_target_encoded/summary.json`](reports/existing_source_selector_target_encoded/summary.json) |
| exp13_lug_merged_selector | [`reports/exp13_lug_merged_selector/summary.json`](reports/exp13_lug_merged_selector/summary.json) |
| exp14_qonly_vs_exp2_selector | [`reports/exp14_qonly_exp2_selector/summary.json`](reports/exp14_qonly_exp2_selector/summary.json) |
| reports/exp7_base_length_grouped_submission/summary.json | [`reports/exp7_base_length_grouped_submission/summary.json`](reports/exp7_base_length_grouped_submission/summary.json) |
| reports/exp7_exp5_length_grouped_submission/summary.json | [`reports/exp7_exp5_length_grouped_submission/summary.json`](reports/exp7_exp5_length_grouped_submission/summary.json) |
| reports/extended_validation_source_selector/summary.json | [`reports/extended_validation_source_selector/summary.json`](reports/extended_validation_source_selector/summary.json) |
| reports/family_meta_miss_correction/summary.json | [`reports/family_meta_miss_correction/summary.json`](reports/family_meta_miss_correction/summary.json) |
| reports/fullcap_vs_noleak_mining/summary.json | [`reports/fullcap_vs_noleak_mining/summary.json`](reports/fullcap_vs_noleak_mining/summary.json) |
| reports/local_candidate_regressor_submission/summary.json | [`reports/local_candidate_regressor_submission/summary.json`](reports/local_candidate_regressor_submission/summary.json) |
| reports/local_no_modal_selector_v2/summary.json | [`reports/local_no_modal_selector_v2/summary.json`](reports/local_no_modal_selector_v2/summary.json) |
| reports/predicted_length_cluster_selector/summary.json | [`reports/predicted_length_cluster_selector/summary.json`](reports/predicted_length_cluster_selector/summary.json) |
| reports/residual_clean_override_selector/summary.json | [`reports/residual_clean_override_selector/summary.json`](reports/residual_clean_override_selector/summary.json) |
| reports/rich_clean_meta_selector/summary.json | [`reports/rich_clean_meta_selector/summary.json`](reports/rich_clean_meta_selector/summary.json) |
| reports/rich_clean_meta_selector_target_encoded/summary.json | [`reports/rich_clean_meta_selector_target_encoded/summary.json`](reports/rich_clean_meta_selector_target_encoded/summary.json) |
| reports/rich_clean_meta_with_regressor/summary.json | [`reports/rich_clean_meta_with_regressor/summary.json`](reports/rich_clean_meta_with_regressor/summary.json) |
| reports/selector_family_classifier/summary.json | [`reports/selector_family_classifier/summary.json`](reports/selector_family_classifier/summary.json) |
| reports/selector_family_hybrid/summary.json | [`reports/selector_family_hybrid/summary.json`](reports/selector_family_hybrid/summary.json) |
| reports/selector_family_meta_learner/summary.json | [`reports/selector_family_meta_learner/summary.json`](reports/selector_family_meta_learner/summary.json) |
| reports/selector_family_meta_learner_target_encoded/summary.json | [`reports/selector_family_meta_learner_target_encoded/summary.json`](reports/selector_family_meta_learner_target_encoded/summary.json) |
| reports/subset_clean_source_selector/summary.json | [`reports/subset_clean_source_selector/summary.json`](reports/subset_clean_source_selector/summary.json) |
| reports/test_source_feasibility_audit/summary.json | [`reports/test_source_feasibility_audit/summary.json`](reports/test_source_feasibility_audit/summary.json) |

## Chronological research record

1. **Data and exploratory analysis.** `EDA.ipynb`, `build_pairs_laptop.py`, `build_pairs_laptop.ipynb`, and the starter/retrieval notebooks establish the `input`/`output`/`subset`/`ID` schema, data splits, candidate construction, and baseline evaluation.
2. **Generation and retrieval baselines.** MT0/MT5 RAG notebooks and `retrieval_baselines.ipynb` establish generation and dense/lexical retrieval baselines.
3. **Exp 1 — query-to-QA-document BGE-M3 LoRA.** Fine-tunes the query encoder with cached multiple-negatives ranking loss; the persisted comparison is baseline versus fine-tuned top-1 and oracle@20 recall.
4. **Exp 2 — top-50 BGE-M3 cross-encoder regression.** Fine-tunes a cross-encoder to predict answer ROUGE and reranks a 50-candidate pool. This is the main practical reranking baseline.
5. **Exp 3 — top-100 cross-encoder.** Tests a larger candidate pool and a faster ROUGE-1 regression setup.
6. **Exp 4 — pairwise hard-negative reranking.** Replaces regression targets with pairwise preference training; diagnostics quantify wins, hurts, exact-oracle misses, and catastrophic jumps.
7. **Exp 5 — BGE-M3 encoder mining.** Improves candidate mining with dense and lexical retrieval, then feeds the resulting pool to the Exp 2 reranker.
8. **Exp 7 — cluster/length-grouped source selection.** Tests answer-cluster retrieval, length-grouped choices, and candidate-source selection for deployable submissions.
9. **Exp 8 — Ghana specialist.** Trains a Ghana-subset specialist encoder/reranker and compares it with the global models.
10. **Exp 9 — Jina multilingual reranker.** Tests `jinaai/jina-reranker-v2-base-multilingual` against the BGE-M3 cross-encoder.
11. **Exp 10–13 — Luganda/Uganda specialization and merging.** Tests a Luganda specialist, E5/BGE candidate merging, and a selector over the merged candidates.
12. **Exp 11 — base encoder benchmark.** Compares `BAAI/bge-m3`, `intfloat/multilingual-e5-base`, and `intfloat/multilingual-e5-large` as retrieval encoders.
13. **Exp 14 — question-only selector.** Tests a selector using question-side features and complements the Exp 2 reranker.
14. **Exp 17 — answer-cluster multi-vector retrieval.** Measures duplicate-answer structure, exact-answer-in-train rates, cluster strategies, exact recall@K, and oracle ROUGE@K.
15. **Exp 18 — Qwen3 listwise reranking.** Tests `Qwen/Qwen3-Reranker-0.6B` on cluster-mined groups; the recorded result is below the cluster-fusion retrieval baseline, which is itself documented in the raw summary.
16. **Local selector track.** The `reports/` summaries record leakage audits, oracle audits, family meta-learners, candidate regressors, fallback rules, clean-source selectors, and the extended ensemble used to choose among existing candidate sources.

## Evaluation protocol and metric definitions

- **ROUGE-1 (`rouge1`, R1):** unigram overlap F-measure between a selected/generated answer and the reference answer.
- **ROUGE-L (`rougeL`, RL):** longest-common-subsequence-based overlap score; it is reported for the main answer-ranking experiments where available.
- **Top-1:** score of the first candidate from the retrieval/encoder stage.
- **Rerank:** score of the candidate selected by the learned reranker or selector.
- **Oracle:** the best reference-overlap score available among the evaluated candidate pool. It is an upper bound on ranking quality for that pool, not a deployable score.
- **Exact recall@K:** whether an exact normalized answer match appears in the first K candidates; reported in the answer-cluster experiments.
- **Coverage:** fraction of validation rows for which a source produces a usable candidate; source-selection reports include it where relevant.
- **No-leak / fold-safe results:** reports with these names exclude validation-derived target information from the corresponding training fold; full-cap results are retained as diagnostic/oracle-oriented comparisons and are labeled accordingly.

## Reproduction

### Local analysis

1. Restore the local datasets and model artifacts listed in `LOCAL_ARTIFACTS.md` (this file is generated alongside the README).
2. Use Python 3.12 or the environment used by the notebooks. The Modal scripts declare their cloud dependencies inline; the core stack is PyTorch, Transformers, Sentence Transformers, Datasets, PEFT, pandas, NumPy, scikit-learn, rouge-score, tqdm, and Modal.
3. Run the notebooks in chronological order for exploratory work, or run the corresponding `modal_*.py` and `local_*.py` files for the persisted experiment families.
4. Keep the generated `summary.json` beside each experiment output and run `python build_readme.py` to refresh this README's experiment index, headline tables, and full metric appendix.

### Modal experiments

The Modal scripts define the image dependencies, GPU type, volume paths, seeds, candidate K, training hyperparameters, and output locations. Before running them, update the hard-coded local/remote data paths if your Modal volume layout differs. Most persisted cloud runs used an NVIDIA L40S.

## Repository layout

| Path | Role |
|---|---|
| `modal_*.py` | Cloud training/evaluation jobs and experiment entry points. |
| `local_*.py` | Local selectors, audits, fallback rules, and submission construction. |
| `*.ipynb` | Exploratory analysis, data construction, retrieval, reranking, and generation notebooks. |
| `modal_outputs/` | Persisted Modal result summaries and selected output artifacts. |
| `reports/` | Local validation reports, leakage/oracle audits, selector leaderboards, and error mining. |
| `Bgem3-finetune/`, `lora_adapter/`, `trainer/`, `mt0-rag-finetuned/` | Fine-tuning metadata, adapters, and model-card files; large weights are local-only. |
| `build_readme.py` | Regenerates this README from the result summaries. |

## Local-only artifact policy

The local workspace inventory at README generation time is **1,354 files / 18.05 GiB**, including 469 Python files, 17 notebooks, 105 JSON files, and 224 CSV files. GitHub-ready commits exclude the following classes via `.gitignore`:

- raw `Train.csv`, `Val.csv`, and `Test.csv` data plus generated CSV prediction tables;
- model weights and training state (`*.safetensors`, `*.pt`, `*.bin`, `*.pkl`);
- parquet candidate pools, ZIP archives, spreadsheets, and rendered review workbooks;
- Python caches and other transient output.

The exact local paths and sizes are recorded in `LOCAL_ARTIFACTS.md`, which is generated from the workspace inventory. Keep that manifest with the local workspace when moving the full research archive or migrate the excluded artifacts to object storage/model hosting if public release is required.

## Limitations and research conclusions

- The best candidate-pool oracle remains materially above the deployable ranker, so retrieval coverage and ranking calibration are still the main bottlenecks.
- Gains vary substantially by language/country subset; the common Exp 2 reranker is strongest on the English/Kenya, English/Uganda, Swahili/Kenya, and Luganda/Uganda subsets, while Akan/Ghana, Amharic/Ethiopia, and English/Ghana remain harder.
- Pairwise hard-negative training and Qwen3 listwise reranking did not beat the common Exp 2 cross-encoder in the persisted comparisons.
- Full-cap, oracle-assisted, and validation-derived source choices are useful diagnostics but should not be presented as leakage-free deployable performance; the reports label the safer no-leak/fold-safe alternatives.
- This is a research artifact, not a clinical decision system. Do not use its predictions for diagnosis, treatment, triage, or patient-specific medical advice.

## Metric appendix: every persisted summary

The following sections reproduce every `summary.json` in the workspace at the time this README was generated. These blocks are intentionally verbose: they preserve the complete metric, hyperparameter, diagnostic, and artifact record in one browsable document.

<details><summary>exp10_lug_uga_reranker — <code>modal_outputs/exp10_lug_uga_reranker/summary.json</code></summary>

```json
{
  "experiment": "exp10_lug_uga_reranker",
  "subset": "Lug_Uga",
  "gpu": "L40S",
  "train_rows": 3383,
  "val_rows": 846,
  "all_val_rows": 6686,
  "k": 50,
  "epochs": 1,
  "train_pairs_per_query": 12,
  "reranker_examples": 40596,
  "train_seconds": 665.4740951061249,
  "top1": {
    "label": "lug_top1",
    "rouge1": 0.5586771911482437,
    "rougeL": 0.5356754361443129,
    "per_subset": {
      "Lug_Uga": {
        "rouge1": 0.558677,
        "rougeL": 0.535675
      }
    }
  },
  "oracle": {
    "label": "lug_oracle",
    "rouge1": 0.8532728829816133,
    "rougeL": 0.8404519295697557,
    "per_subset": {
      "Lug_Uga": {
        "rouge1": 0.853273,
        "rougeL": 0.840452
      }
    }
  },
  "global_rerank": {
    "label": "global_exp2_reranker",
    "rouge1": 0.6793081908518714,
    "rougeL": 0.6608428167278044,
    "per_subset": {
      "Lug_Uga": {
        "rouge1": 0.679308,
        "rougeL": 0.660843
      }
    }
  },
  "lug_rerank": {
    "label": "lug_specialized_reranker",
    "rouge1": 0.6783635334904257,
    "rougeL": 0.6606593801939441,
    "per_subset": {
      "Lug_Uga": {
        "rouge1": 0.678364,
        "rougeL": 0.660659
      }
    }
  },
  "subset_delta_r1": -0.0009446573614457776,
  "overall_delta_if_only_lug_swapped": -0.00011953038106238826,
  "reranker_dir": "/data/exp10_lug_uga_reranker/reranker_final"
}
```

</details>

<details><summary>exp11_base_encoder_benchmark — <code>modal_outputs/exp11_base_encoder_benchmark/summary.json</code></summary>

```json
{
  "experiment": "exp11_base_encoder_benchmark",
  "k": 50,
  "max_seq_length": 256,
  "train_rows": 29814,
  "val_rows": 6686,
  "models": {
    "bge_m3_base": {
      "model_name": "BAAI/bge-m3",
      "top1_rouge1": 0.49846948014743486,
      "oracle_rouge1": 0.6754309123022352,
      "seconds": 44.69831204414368,
      "per_subset": {
        "Aka_Gha": {
          "top1_rouge1": 0.28240396016691854,
          "oracle_rouge1": 0.3934762619285527,
          "count": 1114
        },
        "Amh_Eth": {
          "top1_rouge1": 0.16315041954127774,
          "oracle_rouge1": 0.30830608619241506,
          "count": 462
        },
        "Eng_Eth": {
          "top1_rouge1": 0.5469010440161853,
          "oracle_rouge1": 0.7493461888357091,
          "count": 564
        },
        "Eng_Gha": {
          "top1_rouge1": 0.2825914145840942,
          "oracle_rouge1": 0.3854160357552903,
          "count": 1104
        },
        "Eng_Ken": {
          "top1_rouge1": 0.7810454510329977,
          "oracle_rouge1": 0.9185490348769356,
          "count": 390
        },
        "Eng_Uga": {
          "top1_rouge1": 0.7460903876873639,
          "oracle_rouge1": 0.9355700528534144,
          "count": 1688
        },
        "Lug_Uga": {
          "top1_rouge1": 0.42724912862600295,
          "oracle_rouge1": 0.7990196389982801,
          "count": 846
        },
        "Swa_Ken": {
          "top1_rouge1": 0.7662147975849911,
          "oracle_rouge1": 0.9142539974037579,
          "count": 518
        }
      }
    },
    "multilingual_e5_base": {
      "model_name": "intfloat/multilingual-e5-base",
      "top1_rouge1": 0.4875018420822146,
      "oracle_rouge1": 0.6768323967182054,
      "seconds": 27.46133780479431,
      "per_subset": {
        "Aka_Gha": {
          "top1_rouge1": 0.2767079868610606,
          "oracle_rouge1": 0.39579082218371503,
          "count": 1114
        },
        "Amh_Eth": {
          "top1_rouge1": 0.16591601978603368,
          "oracle_rouge1": 0.3047298029687088,
          "count": 462
        },
        "Eng_Eth": {
          "top1_rouge1": 0.5552252524502024,
          "oracle_rouge1": 0.7486053509022882,
          "count": 564
        },
        "Eng_Gha": {
          "top1_rouge1": 0.28581379153925246,
          "oracle_rouge1": 0.3859043771823967,
          "count": 1104
        },
        "Eng_Ken": {
          "top1_rouge1": 0.7862196892692636,
          "oracle_rouge1": 0.9186749955775178,
          "count": 390
        },
        "Eng_Uga": {
          "top1_rouge1": 0.6999418416962724,
          "oracle_rouge1": 0.9347250035417275,
          "count": 1688
        },
        "Lug_Uga": {
          "top1_rouge1": 0.457489494804358,
          "oracle_rouge1": 0.8170523759466763,
          "count": 846
        },
        "Swa_Ken": {
          "top1_rouge1": 0.7156029521491531,
          "oracle_rouge1": 0.9035290443067695,
          "count": 518
        }
      }
    },
    "multilingual_e5_large": {
      "model_name": "intfloat/multilingual-e5-large",
      "top1_rouge1": 0.5048918014240319,
      "oracle_rouge1": 0.6786129826422103,
      "seconds": 41.293991565704346,
      "per_subset": {
        "Aka_Gha": {
          "top1_rouge1": 0.27409383914192753,
          "oracle_rouge1": 0.39475679843763345,
          "count": 1114
        },
        "Amh_Eth": {
          "top1_rouge1": 0.16054737042820996,
          "oracle_rouge1": 0.3043702739455134,
          "count": 462
        },
        "Eng_Eth": {
          "top1_rouge1": 0.5607431164354776,
          "oracle_rouge1": 0.7489559467258885,
          "count": 564
        },
        "Eng_Gha": {
          "top1_rouge1": 0.28393530157974334,
          "oracle_rouge1": 0.386869525689101,
          "count": 1104
        },
        "Eng_Ken": {
          "top1_rouge1": 0.8036780925811836,
          "oracle_rouge1": 0.9188351175877304,
          "count": 390
        },
        "Eng_Uga": {
          "top1_rouge1": 0.7394486278951249,
          "oracle_rouge1": 0.9359560255542213,
          "count": 1688
        },
        "Lug_Uga": {
          "top1_rouge1": 0.48765316882277043,
          "oracle_rouge1": 0.8249181598082022,
          "count": 846
        },
        "Swa_Ken": {
          "top1_rouge1": 0.7573188242235418,
          "oracle_rouge1": 0.9096388444340126,
          "count": 518
        }
      }
    }
  }
}
```

</details>

<details><summary>exp12_lug_e5_merge_rerank — <code>modal_outputs/exp12_lug_e5_merge_rerank/summary.json</code></summary>

```json
{
  "experiment": "exp12_lug_e5_merge_rerank",
  "subset": "Lug_Uga",
  "gpu": "L40S",
  "k_bge": 50,
  "k_e5": 50,
  "train_rows": 3383,
  "val_rows": 846,
  "seconds": 202.43869137763977,
  "metrics": {
    "bge_ft_top1": {
      "label": "bge_ft_top1",
      "rouge1": 0.5586771911482437,
      "count": 846
    },
    "e5_large_top1": {
      "label": "e5_large_top1",
      "rouge1": 0.48765316882277043,
      "count": 846
    },
    "merged_exp2_rerank": {
      "label": "merged_exp2_rerank",
      "rouge1": 0.6835317759537722,
      "count": 846
    },
    "merged_oracle": {
      "label": "merged_oracle",
      "rouge1": 0.8740366794311258,
      "count": 846
    }
  },
  "delta_merged_rerank_vs_bge_top1": 0.12485458480552858,
  "delta_merged_oracle_vs_bge_top1": 0.3153594882828822
}
```

</details>

<details><summary>exp17_answer_cluster_multivector — <code>modal_outputs/exp17_answer_cluster_multivector/summary.json</code></summary>

```json
{
  "experiment": "exp17_answer_cluster_multivector",
  "model_source": "finetuned",
  "model_name": "/data/exp5_bgem3_encoder_mining_v2/final",
  "max_seq_length": 512,
  "eval_k": 100,
  "train_rows": 29814,
  "val_rows": 6686,
  "seconds": 303.19540429115295,
  "cluster_stats": [
    {
      "subset": "Aka_Gha",
      "train_rows": 4455,
      "unique_answer_clusters": 4432,
      "duplicate_row_fraction": 0.00516273849607185,
      "largest_cluster": 4,
      "mean_questions_per_cluster": 1.0051895306859207,
      "val_exact_answer_in_train_rate": 0.00807899461400359
    },
    {
      "subset": "Amh_Eth",
      "train_rows": 1845,
      "unique_answer_clusters": 1838,
      "duplicate_row_fraction": 0.003794037940379358,
      "largest_cluster": 3,
      "mean_questions_per_cluster": 1.0038084874863982,
      "val_exact_answer_in_train_rate": 0.006493506493506494
    },
    {
      "subset": "Eng_Eth",
      "train_rows": 3915,
      "unique_answer_clusters": 2899,
      "duplicate_row_fraction": 0.259514687100894,
      "largest_cluster": 63,
      "mean_questions_per_cluster": 1.350465677819938,
      "val_exact_answer_in_train_rate": 0.49113475177304966
    },
    {
      "subset": "Eng_Gha",
      "train_rows": 4443,
      "unique_answer_clusters": 4378,
      "duplicate_row_fraction": 0.014629754670267836,
      "largest_cluster": 29,
      "mean_questions_per_cluster": 1.014846962083143,
      "val_exact_answer_in_train_rate": 0.019021739130434784
    },
    {
      "subset": "Eng_Ken",
      "train_rows": 2080,
      "unique_answer_clusters": 984,
      "duplicate_row_fraction": 0.5269230769230769,
      "largest_cluster": 5,
      "mean_questions_per_cluster": 2.113821138211382,
      "val_exact_answer_in_train_rate": 0.882051282051282
    },
    {
      "subset": "Eng_Uga",
      "train_rows": 7623,
      "unique_answer_clusters": 1791,
      "duplicate_row_fraction": 0.7650531286894924,
      "largest_cluster": 68,
      "mean_questions_per_cluster": 4.256281407035176,
      "val_exact_answer_in_train_rate": 0.8981042654028436
    },
    {
      "subset": "Lug_Uga",
      "train_rows": 3383,
      "unique_answer_clusters": 1112,
      "duplicate_row_fraction": 0.671297664794561,
      "largest_cluster": 26,
      "mean_questions_per_cluster": 3.04226618705036,
      "val_exact_answer_in_train_rate": 0.875886524822695
    },
    {
      "subset": "Swa_Ken",
      "train_rows": 2070,
      "unique_answer_clusters": 985,
      "duplicate_row_fraction": 0.5241545893719807,
      "largest_cluster": 5,
      "mean_questions_per_cluster": 2.1015228426395938,
      "val_exact_answer_in_train_rate": 0.8745173745173745
    }
  ],
  "leaderboard": [
    {
      "strategy": "cluster_fusion_all",
      "top1_r1": 0.5505407873740029,
      "oracle1_r1": 0.5505407873740029,
      "exact_recall_at_1": 0.3664373317379599,
      "oracle10_r1": 0.671593760576789,
      "exact_recall_at_10": 0.48130421776847143,
      "oracle50_r1": 0.6950167096591873,
      "exact_recall_at_50": 0.49730780735865987,
      "oracle100_r1": 0.6999194590899132,
      "exact_recall_at_100": 0.5,
      "delta_oracle1_vs_row": 0.0036470993152297737,
      "delta_exact_recall_at_1_vs_row": -0.0032904576727490475,
      "delta_oracle10_vs_row": 0.011705137801507393,
      "delta_exact_recall_at_10_vs_row": 0.012713131917439446,
      "delta_oracle50_vs_row": 0.0060774729437534925,
      "delta_exact_recall_at_50_vs_row": 0.006132216571941351,
      "delta_oracle100_vs_row": 0.004076740107182286,
      "delta_exact_recall_at_100_vs_row": 0.004486987735566883
    },
    {
      "strategy": "cluster_qa_max",
      "top1_r1": 0.5145263657396737,
      "oracle1_r1": 0.5145263657396737,
      "exact_recall_at_1": 0.3179778641938379,
      "oracle10_r1": 0.6664516559938226,
      "exact_recall_at_10": 0.47502243493867785,
      "oracle50_r1": 0.6942300657604381,
      "exact_recall_at_50": 0.4965599760693987,
      "oracle100_r1": 0.6994847653853236,
      "exact_recall_at_100": 0.4994017349685911,
      "delta_oracle1_vs_row": -0.032367322319099445,
      "delta_exact_recall_at_1_vs_row": -0.05174992521687105,
      "delta_oracle10_vs_row": 0.0065630332185410145,
      "delta_exact_recall_at_10_vs_row": 0.006431349087645866,
      "delta_oracle50_vs_row": 0.0052908290450043305,
      "delta_exact_recall_at_50_vs_row": 0.005384385282680204,
      "delta_oracle100_vs_row": 0.0036420464025926913,
      "delta_exact_recall_at_100_vs_row": 0.0038887227041579653
    },
    {
      "strategy": "cluster_fusion_qmax_qa",
      "top1_r1": 0.5568867610279118,
      "oracle1_r1": 0.5568867610279118,
      "exact_recall_at_1": 0.3773556685611726,
      "oracle10_r1": 0.6724563910676856,
      "exact_recall_at_10": 0.4830990128626982,
      "oracle50_r1": 0.6945380122840155,
      "exact_recall_at_50": 0.49670954232725095,
      "oracle100_r1": 0.6994401187661867,
      "exact_recall_at_100": 0.4995513012264433,
      "delta_oracle1_vs_row": 0.00999307296913865,
      "delta_exact_recall_at_1_vs_row": 0.007627879150463646,
      "delta_oracle10_vs_row": 0.012567768292404025,
      "delta_exact_recall_at_10_vs_row": 0.0145079270116662,
      "delta_oracle50_vs_row": 0.005598775568581682,
      "delta_exact_recall_at_50_vs_row": 0.0055339515405324335,
      "delta_oracle100_vs_row": 0.003597399783455857,
      "delta_exact_recall_at_100_vs_row": 0.004038288962010195
    },
    {
      "strategy": "cluster_fusion_qmax_answer",
      "top1_r1": 0.5490412069233862,
      "oracle1_r1": 0.5490412069233862,
      "exact_recall_at_1": 0.36449297038588097,
      "oracle10_r1": 0.6709679482633434,
      "exact_recall_at_10": 0.4811546515106192,
      "oracle50_r1": 0.6942570527952533,
      "exact_recall_at_50": 0.49670954232725095,
      "oracle100_r1": 0.6991111531339957,
      "exact_recall_at_100": 0.49925216871073885,
      "delta_oracle1_vs_row": 0.0021475188646130627,
      "delta_exact_recall_at_1_vs_row": -0.005234819024827975,
      "delta_oracle10_vs_row": 0.011079325488061786,
      "delta_exact_recall_at_10_vs_row": 0.012563565659587217,
      "delta_oracle50_vs_row": 0.005317816079819515,
      "delta_exact_recall_at_50_vs_row": 0.0055339515405324335,
      "delta_oracle100_vs_row": 0.0032684341512648762,
      "delta_exact_recall_at_100_vs_row": 0.003739156446305736
    },
    {
      "strategy": "cluster_question_centroid",
      "top1_r1": 0.5441184733808208,
      "oracle1_r1": 0.5441184733808208,
      "exact_recall_at_1": 0.3659886329644032,
      "oracle10_r1": 0.6667394393922957,
      "exact_recall_at_10": 0.4769667962907568,
      "oracle50_r1": 0.6919609861977168,
      "exact_recall_at_50": 0.49446604845946757,
      "oracle100_r1": 0.6981002247741603,
      "exact_recall_at_100": 0.498055638647921,
      "delta_oracle1_vs_row": -0.0027752146779523867,
      "delta_exact_recall_at_1_vs_row": -0.003739156446305736,
      "delta_oracle10_vs_row": 0.006850816617014144,
      "delta_exact_recall_at_10_vs_row": 0.008375710439724793,
      "delta_oracle50_vs_row": 0.003021749482283065,
      "delta_exact_recall_at_50_vs_row": 0.0032904576727490475,
      "delta_oracle100_vs_row": 0.0022575057914294705,
      "delta_exact_recall_at_100_vs_row": 0.0025426263834879004
    },
    {
      "strategy": "cluster_question_max",
      "top1_r1": 0.5467647553029861,
      "oracle1_r1": 0.5467647553029861,
      "exact_recall_at_1": 0.3694286568950045,
      "oracle10_r1": 0.6680888269874433,
      "exact_recall_at_10": 0.47891115764283576,
      "oracle50_r1": 0.6921658347740507,
      "exact_recall_at_50": 0.49446604845946757,
      "oracle100_r1": 0.698092689268534,
      "exact_recall_at_100": 0.49820520490577325,
      "delta_oracle1_vs_row": -0.00012893275578707009,
      "delta_exact_recall_at_1_vs_row": -0.00029913251570445887,
      "delta_oracle10_vs_row": 0.008200204212161721,
      "delta_exact_recall_at_10_vs_row": 0.010320071791803775,
      "delta_oracle50_vs_row": 0.003226598058616914,
      "delta_exact_recall_at_50_vs_row": 0.0032904576727490475,
      "delta_oracle100_vs_row": 0.0022499702858030934,
      "delta_exact_recall_at_100_vs_row": 0.00269219264134013
    },
    {
      "strategy": "row_question",
      "top1_r1": 0.5468936880587731,
      "oracle1_r1": 0.5468936880587731,
      "exact_recall_at_1": 0.36972778941070894,
      "oracle10_r1": 0.6598886227752816,
      "exact_recall_at_10": 0.468591085851032,
      "oracle50_r1": 0.6889392367154338,
      "exact_recall_at_50": 0.4911755907867185,
      "oracle100_r1": 0.6958427189827309,
      "exact_recall_at_100": 0.4955130122644331,
      "delta_oracle1_vs_row": 0.0,
      "delta_exact_recall_at_1_vs_row": 0.0,
      "delta_oracle10_vs_row": 0.0,
      "delta_exact_recall_at_10_vs_row": 0.0,
      "delta_oracle50_vs_row": 0.0,
      "delta_exact_recall_at_50_vs_row": 0.0,
      "delta_oracle100_vs_row": 0.0,
      "delta_exact_recall_at_100_vs_row": 0.0
    },
    {
      "strategy": "cluster_answer_direct",
      "top1_r1": 0.3796850961701002,
      "oracle1_r1": 0.3796850961701002,
      "exact_recall_at_1": 0.16183069099611128,
      "oracle10_r1": 0.592712848865843,
      "exact_recall_at_10": 0.3725695483099013,
      "oracle50_r1": 0.6628859841640601,
      "exact_recall_at_50": 0.45049356865091233,
      "oracle100_r1": 0.6816554036614912,
      "exact_recall_at_100": 0.4763685312593479,
      "delta_oracle1_vs_row": -0.16720859188867293,
      "delta_exact_recall_at_1_vs_row": -0.20789709841459766,
      "delta_oracle10_vs_row": -0.06717577390943863,
      "delta_exact_recall_at_10_vs_row": -0.09602153754113069,
      "delta_oracle50_vs_row": -0.026053252551373696,
      "delta_exact_recall_at_50_vs_row": -0.040682022135806184,
      "delta_oracle100_vs_row": -0.014187315321239624,
      "delta_exact_recall_at_100_vs_row": -0.0191444810050852
    }
  ],
  "artifacts": {
    "predictions": "/data/exp17_answer_cluster_multivector/val_strategy_predictions.parquet",
    "candidate_pools": "/data/exp17_answer_cluster_multivector/val_candidate_pools.parquet",
    "leaderboard": "/data/exp17_answer_cluster_multivector/leaderboard.csv",
    "per_subset": "/data/exp17_answer_cluster_multivector/per_subset.csv"
  }
}
```

</details>

<details><summary>exp18_qwen3_cluster_listwise_reranker — <code>modal_outputs/exp18_qwen3_cluster_listwise_reranker/summary.json</code></summary>

```json
{
  "experiment": "exp18_qwen3_cluster_listwise_reranker",
  "base_model": "Qwen/Qwen3-Reranker-0.6B",
  "candidate_k": 50,
  "eval_k": 20,
  "group_size": 8,
  "max_train_groups": 12000,
  "max_steps": 400,
  "completed_steps": 400,
  "gradient_accumulation": 8,
  "effective_candidates_per_step": 64,
  "learning_rate": 0.0001,
  "max_length": 512,
  "train_seconds": 1087.6802332401276,
  "mining": {
    "rows": [
      {
        "subset": "Aka_Gha",
        "train_rows": 4455,
        "clusters": 4432,
        "eligible_train_groups": 4414,
        "val_groups": 1114
      },
      {
        "subset": "Amh_Eth",
        "train_rows": 1845,
        "clusters": 1838,
        "eligible_train_groups": 1805,
        "val_groups": 462
      },
      {
        "subset": "Eng_Eth",
        "train_rows": 3915,
        "clusters": 2899,
        "eligible_train_groups": 3913,
        "val_groups": 564
      },
      {
        "subset": "Eng_Gha",
        "train_rows": 4443,
        "clusters": 4378,
        "eligible_train_groups": 4441,
        "val_groups": 1104
      },
      {
        "subset": "Eng_Ken",
        "train_rows": 2080,
        "clusters": 984,
        "eligible_train_groups": 2079,
        "val_groups": 390
      },
      {
        "subset": "Eng_Uga",
        "train_rows": 7623,
        "clusters": 1791,
        "eligible_train_groups": 7618,
        "val_groups": 1688
      },
      {
        "subset": "Lug_Uga",
        "train_rows": 3383,
        "clusters": 1112,
        "eligible_train_groups": 3375,
        "val_groups": 846
      },
      {
        "subset": "Swa_Ken",
        "train_rows": 2070,
        "clusters": 985,
        "eligible_train_groups": 2070,
        "val_groups": 518
      }
    ],
    "eligible_train_groups": 29715,
    "selected_train_groups": 12000,
    "val_groups": 6686
  },
  "retrieval_top1": {
    "label": "cluster_fusion_qmax_qa_top1",
    "rouge1": 0.5568867603221539,
    "per_subset": {
      "Aka_Gha": 0.301964,
      "Amh_Eth": 0.182424,
      "Eng_Eth": 0.575503,
      "Eng_Gha": 0.296658,
      "Eng_Ken": 0.810315,
      "Eng_Uga": 0.828669,
      "Lug_Uga": 0.619814,
      "Swa_Ken": 0.794215
    }
  },
  "zero_shot": null,
  "listwise_qwen3": {
    "label": "qwen3_listwise_top20",
    "rouge1": 0.5435287703870808,
    "per_subset": {
      "Aka_Gha": 0.305426,
      "Amh_Eth": 0.164884,
      "Eng_Eth": 0.653791,
      "Eng_Gha": 0.285826,
      "Eng_Ken": 0.803271,
      "Eng_Uga": 0.82414,
      "Lug_Uga": 0.555588,
      "Swa_Ken": 0.6928
    }
  },
  "oracle": {
    "label": "cluster_oracle_top20",
    "rouge1": 0.6852973708083323,
    "per_subset": {
      "Aka_Gha": 0.395156,
      "Amh_Eth": 0.297236,
      "Eng_Eth": 0.745994,
      "Eng_Gha": 0.383083,
      "Eng_Ken": 0.920801,
      "Eng_Uga": 0.940747,
      "Lug_Uga": 0.871116,
      "Swa_Ken": 0.920176
    }
  },
  "delta_listwise_vs_retrieval": -0.013357989935073089,
  "delta_listwise_vs_zero_shot": null,
  "delta_listwise_vs_exp2": -0.045687857959733735
}
```

</details>

<details><summary>query_to_qa_doc_bgem3_lora — <code>modal_outputs/exp1_query_to_qa_doc/summary.json</code></summary>

```json
{
  "experiment": "query_to_qa_doc_bgem3_lora",
  "gpu": "L40S",
  "epochs": 2,
  "batch_size": 32,
  "gradient_accumulation": 2,
  "effective_batch": 64,
  "learning_rate": 1.5e-05,
  "lora_r": 32,
  "max_seq_length": 512,
  "loss": "CachedMultipleNegativesRankingLoss",
  "train_seconds": 2703.9505562782288,
  "baseline": {
    "label": "baseline_query_to_qa_doc",
    "top1_r1": 0.445531304063734,
    "oracle20_r1": 0.643655891217406,
    "per_subset": {
      "Aka_Gha": {
        "top1_r1": 0.2764,
        "oracle20_r1": 0.3767
      },
      "Amh_Eth": {
        "top1_r1": 0.1384,
        "oracle20_r1": 0.2803
      },
      "Eng_Eth": {
        "top1_r1": 0.4971,
        "oracle20_r1": 0.7127
      },
      "Eng_Gha": {
        "top1_r1": 0.2868,
        "oracle20_r1": 0.3747
      },
      "Eng_Ken": {
        "top1_r1": 0.7283,
        "oracle20_r1": 0.8955
      },
      "Eng_Uga": {
        "top1_r1": 0.6131,
        "oracle20_r1": 0.904
      },
      "Lug_Uga": {
        "top1_r1": 0.4257,
        "oracle20_r1": 0.7223
      },
      "Swa_Ken": {
        "top1_r1": 0.6391,
        "oracle20_r1": 0.8735
      }
    }
  },
  "finetuned": {
    "label": "finetuned_query_to_qa_doc",
    "top1_r1": 0.45403765213713065,
    "oracle20_r1": 0.6548019787184409,
    "per_subset": {
      "Aka_Gha": {
        "top1_r1": 0.278,
        "oracle20_r1": 0.3814
      },
      "Amh_Eth": {
        "top1_r1": 0.146,
        "oracle20_r1": 0.2767
      },
      "Eng_Eth": {
        "top1_r1": 0.4918,
        "oracle20_r1": 0.7237
      },
      "Eng_Gha": {
        "top1_r1": 0.279,
        "oracle20_r1": 0.3732
      },
      "Eng_Ken": {
        "top1_r1": 0.6671,
        "oracle20_r1": 0.9055
      },
      "Eng_Uga": {
        "top1_r1": 0.6539,
        "oracle20_r1": 0.9093
      },
      "Lug_Uga": {
        "top1_r1": 0.4647,
        "oracle20_r1": 0.7744
      },
      "Swa_Ken": {
        "top1_r1": 0.6101,
        "oracle20_r1": 0.8917
      }
    }
  },
  "delta_top1_r1": 0.008506348073396675,
  "delta_oracle20_r1": 0.011146087501034962
}
```

</details>

<details><summary>ft_bgem3_top50_crossencoder_rouge_regression — <code>modal_outputs/exp2_crossencoder_rerank/summary.json</code></summary>

```json
{
  "experiment": "ft_bgem3_top50_crossencoder_rouge_regression",
  "gpu": "L40S",
  "k": 50,
  "train_pairs_per_query": 12,
  "epochs": 1,
  "batch_size": 8,
  "gradient_accumulation": 4,
  "effective_batch": 32,
  "learning_rate": 1e-05,
  "train_seconds": 3914.618010520935,
  "candidate_baseline": {
    "label": "ft_bgem3_per_subset_top50",
    "top1_r1": 0.5395458277838778,
    "oracle_r1": 0.6836959744084082,
    "per_subset": {
      "Aka_Gha": {
        "top1_r1": 0.2903,
        "oracle_r1": 0.395
      },
      "Amh_Eth": {
        "top1_r1": 0.1711,
        "oracle_r1": 0.3079
      },
      "Eng_Eth": {
        "top1_r1": 0.5629,
        "oracle_r1": 0.7506
      },
      "Eng_Gha": {
        "top1_r1": 0.2831,
        "oracle_r1": 0.3844
      },
      "Eng_Ken": {
        "top1_r1": 0.7992,
        "oracle_r1": 0.9202
      },
      "Eng_Uga": {
        "top1_r1": 0.8171,
        "oracle_r1": 0.938
      },
      "Lug_Uga": {
        "top1_r1": 0.5587,
        "oracle_r1": 0.8533
      },
      "Swa_Ken": {
        "top1_r1": 0.7941,
        "oracle_r1": 0.921
      }
    }
  },
  "top1": {
    "label": "ft_bgem3_top1",
    "rouge1": 0.5395458277838778,
    "rougeL": 0.4922840777969995,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.2903,
        "rougeL": 0.173
      },
      "Amh_Eth": {
        "rouge1": 0.1711,
        "rougeL": 0.16
      },
      "Eng_Eth": {
        "rouge1": 0.5629,
        "rougeL": 0.5438
      },
      "Eng_Gha": {
        "rouge1": 0.2831,
        "rougeL": 0.1869
      },
      "Eng_Ken": {
        "rouge1": 0.7992,
        "rougeL": 0.7816
      },
      "Eng_Uga": {
        "rouge1": 0.8171,
        "rougeL": 0.8008
      },
      "Lug_Uga": {
        "rouge1": 0.5587,
        "rougeL": 0.5357
      },
      "Swa_Ken": {
        "rouge1": 0.7941,
        "rougeL": 0.7759
      }
    }
  },
  "rerank": {
    "label": "crossencoder_rerank",
    "rouge1": 0.5892166283468145,
    "rougeL": 0.5397974652880141,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.3216,
        "rougeL": 0.1886
      },
      "Amh_Eth": {
        "rouge1": 0.1907,
        "rougeL": 0.1774
      },
      "Eng_Eth": {
        "rouge1": 0.6646,
        "rougeL": 0.6468
      },
      "Eng_Gha": {
        "rouge1": 0.3026,
        "rougeL": 0.1996
      },
      "Eng_Ken": {
        "rouge1": 0.8199,
        "rougeL": 0.8007
      },
      "Eng_Uga": {
        "rouge1": 0.865,
        "rougeL": 0.8532
      },
      "Lug_Uga": {
        "rouge1": 0.6784,
        "rougeL": 0.6599
      },
      "Swa_Ken": {
        "rouge1": 0.8311,
        "rougeL": 0.8129
      }
    }
  },
  "oracle": {
    "label": "oracle_topk",
    "rouge1": 0.6836959744084082,
    "rougeL": 0.6256092887229702,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.395,
        "rougeL": 0.2291
      },
      "Amh_Eth": {
        "rouge1": 0.3079,
        "rougeL": 0.2791
      },
      "Eng_Eth": {
        "rouge1": 0.7506,
        "rougeL": 0.7314
      },
      "Eng_Gha": {
        "rouge1": 0.3844,
        "rougeL": 0.255
      },
      "Eng_Ken": {
        "rouge1": 0.9202,
        "rougeL": 0.9069
      },
      "Eng_Uga": {
        "rouge1": 0.938,
        "rougeL": 0.9301
      },
      "Lug_Uga": {
        "rouge1": 0.8533,
        "rougeL": 0.8405
      },
      "Swa_Ken": {
        "rouge1": 0.921,
        "rougeL": 0.9075
      }
    }
  },
  "delta_rerank_vs_top1": 0.04967080056293671
}
```

</details>

<details><summary>exp2_val_candidate_scores — <code>modal_outputs/exp2_val_candidate_scores/summary.json</code></summary>

```json
{
  "experiment": "exp2_val_candidate_scores",
  "k": 50,
  "val_rows": 6686,
  "candidate_rows": 334300,
  "candidate_scores": "/data/exp2_val_candidate_scores/val_candidate_scores.csv",
  "chosen_scores": "/data/exp2_val_candidate_scores/val_chosen_scores.csv"
}
```

</details>

<details><summary>ft_bgem3_top100_crossencoder_fast_rouge1_regression_pairs16 — <code>modal_outputs/exp3_top100_crossencoder_rerank/summary.json</code></summary>

```json
{
  "experiment": "ft_bgem3_top100_crossencoder_fast_rouge1_regression_pairs16",
  "gpu": "L40S",
  "k": 100,
  "train_pairs_per_query": 16,
  "epochs": 1,
  "batch_size": 8,
  "gradient_accumulation": 4,
  "effective_batch": 32,
  "learning_rate": 1e-05,
  "train_seconds": 5337.78245139122,
  "candidate_baseline": {
    "label": "ft_bgem3_per_subset_top100",
    "top1_r1": 0.5395458277838778,
    "oracle_r1": 0.6913266408627975,
    "per_subset": {
      "Aka_Gha": {
        "top1_r1": 0.2903,
        "oracle_r1": 0.4044
      },
      "Amh_Eth": {
        "top1_r1": 0.1711,
        "oracle_r1": 0.319
      },
      "Eng_Eth": {
        "top1_r1": 0.5629,
        "oracle_r1": 0.7577
      },
      "Eng_Gha": {
        "top1_r1": 0.2831,
        "oracle_r1": 0.39
      },
      "Eng_Ken": {
        "top1_r1": 0.7992,
        "oracle_r1": 0.9239
      },
      "Eng_Uga": {
        "top1_r1": 0.8171,
        "oracle_r1": 0.9414
      },
      "Lug_Uga": {
        "top1_r1": 0.5587,
        "oracle_r1": 0.873
      },
      "Swa_Ken": {
        "top1_r1": 0.7941,
        "oracle_r1": 0.9235
      }
    }
  },
  "top1": {
    "label": "ft_bgem3_top1",
    "rouge1": 0.5395458277838778,
    "rougeL": 0.5395458277838778,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.2903,
        "rougeL": 0.2903
      },
      "Amh_Eth": {
        "rouge1": 0.1711,
        "rougeL": 0.1711
      },
      "Eng_Eth": {
        "rouge1": 0.5629,
        "rougeL": 0.5629
      },
      "Eng_Gha": {
        "rouge1": 0.2831,
        "rougeL": 0.2831
      },
      "Eng_Ken": {
        "rouge1": 0.7992,
        "rougeL": 0.7992
      },
      "Eng_Uga": {
        "rouge1": 0.8171,
        "rougeL": 0.8171
      },
      "Lug_Uga": {
        "rouge1": 0.5587,
        "rougeL": 0.5587
      },
      "Swa_Ken": {
        "rouge1": 0.7941,
        "rougeL": 0.7941
      }
    }
  },
  "rerank": {
    "label": "crossencoder_rerank",
    "rouge1": 0.5904174317680138,
    "rougeL": 0.5904174317680138,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.3222,
        "rougeL": 0.3222
      },
      "Amh_Eth": {
        "rouge1": 0.1981,
        "rougeL": 0.1981
      },
      "Eng_Eth": {
        "rouge1": 0.6638,
        "rougeL": 0.6638
      },
      "Eng_Gha": {
        "rouge1": 0.3073,
        "rougeL": 0.3073
      },
      "Eng_Ken": {
        "rouge1": 0.7991,
        "rougeL": 0.7991
      },
      "Eng_Uga": {
        "rouge1": 0.8684,
        "rougeL": 0.8684
      },
      "Lug_Uga": {
        "rouge1": 0.6821,
        "rougeL": 0.6821
      },
      "Swa_Ken": {
        "rouge1": 0.8277,
        "rougeL": 0.8277
      }
    }
  },
  "oracle": {
    "label": "oracle_topk",
    "rouge1": 0.6913266408627975,
    "rougeL": 0.6913266408627975,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.4044,
        "rougeL": 0.4044
      },
      "Amh_Eth": {
        "rouge1": 0.319,
        "rougeL": 0.319
      },
      "Eng_Eth": {
        "rouge1": 0.7577,
        "rougeL": 0.7577
      },
      "Eng_Gha": {
        "rouge1": 0.39,
        "rougeL": 0.39
      },
      "Eng_Ken": {
        "rouge1": 0.9239,
        "rougeL": 0.9239
      },
      "Eng_Uga": {
        "rouge1": 0.9414,
        "rougeL": 0.9414
      },
      "Lug_Uga": {
        "rouge1": 0.873,
        "rougeL": 0.873
      },
      "Swa_Ken": {
        "rouge1": 0.9235,
        "rougeL": 0.9235
      }
    }
  },
  "delta_rerank_vs_top1": 0.05087160398413593
}
```

</details>

<details><summary>ft_bgem3_top50_pairwise_hardneg_rerank — <code>modal_outputs/exp4_pairwise_hardneg_rerank/summary.json</code></summary>

```json
{
  "experiment": "ft_bgem3_top50_pairwise_hardneg_rerank",
  "gpu": "L40S",
  "base_reranker": "BAAI/bge-reranker-v2-m3",
  "k": 50,
  "pairs_per_query": 6,
  "epochs": 1,
  "batch_size": 8,
  "gradient_accumulation": 4,
  "effective_pair_batch": 32,
  "learning_rate": 1e-05,
  "max_length": 512,
  "train_seconds": 3271.5812406539917,
  "pair_counts": {
    "Aka_Gha": 23909,
    "Amh_Eth": 10394,
    "Eng_Eth": 21064,
    "Eng_Gha": 24398,
    "Eng_Ken": 11328,
    "Eng_Uga": 27818,
    "Lug_Uga": 16832,
    "Swa_Ken": 11391
  },
  "candidate_baseline": {
    "label": "ft_bgem3_per_subset_top50",
    "top1_r1": 0.5394795443970752,
    "oracle_r1": 0.6840949279607131,
    "per_subset": {
      "Aka_Gha": {
        "top1_r1": 0.2903,
        "oracle_r1": 0.3957
      },
      "Amh_Eth": {
        "top1_r1": 0.1712,
        "oracle_r1": 0.3084
      },
      "Eng_Eth": {
        "top1_r1": 0.5619,
        "oracle_r1": 0.7509
      },
      "Eng_Gha": {
        "top1_r1": 0.2831,
        "oracle_r1": 0.3854
      },
      "Eng_Ken": {
        "top1_r1": 0.7992,
        "oracle_r1": 0.9204
      },
      "Eng_Uga": {
        "top1_r1": 0.8172,
        "oracle_r1": 0.938
      },
      "Lug_Uga": {
        "top1_r1": 0.5587,
        "oracle_r1": 0.8535
      },
      "Swa_Ken": {
        "top1_r1": 0.7941,
        "oracle_r1": 0.9212
      }
    }
  },
  "top1": {
    "label": "ft_bgem3_top1",
    "rouge1": 0.5394795443970752,
    "rougeL": 0.4922645017101416,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.2903,
        "rougeL": 0.173
      },
      "Amh_Eth": {
        "rouge1": 0.1712,
        "rougeL": 0.1603
      },
      "Eng_Eth": {
        "rouge1": 0.5619,
        "rougeL": 0.5429
      },
      "Eng_Gha": {
        "rouge1": 0.2831,
        "rougeL": 0.1869
      },
      "Eng_Ken": {
        "rouge1": 0.7992,
        "rougeL": 0.7816
      },
      "Eng_Uga": {
        "rouge1": 0.8172,
        "rougeL": 0.8009
      },
      "Lug_Uga": {
        "rouge1": 0.5587,
        "rougeL": 0.5357
      },
      "Swa_Ken": {
        "rouge1": 0.7941,
        "rougeL": 0.7759
      }
    }
  },
  "rerank": {
    "label": "pairwise_rerank",
    "rouge1": 0.5712154030803913,
    "rougeL": 0.5232546272454361,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.3049,
        "rougeL": 0.1789
      },
      "Amh_Eth": {
        "rouge1": 0.1655,
        "rougeL": 0.1523
      },
      "Eng_Eth": {
        "rouge1": 0.6622,
        "rougeL": 0.6471
      },
      "Eng_Gha": {
        "rouge1": 0.2811,
        "rougeL": 0.1813
      },
      "Eng_Ken": {
        "rouge1": 0.8038,
        "rougeL": 0.7823
      },
      "Eng_Uga": {
        "rouge1": 0.8548,
        "rougeL": 0.8415
      },
      "Lug_Uga": {
        "rouge1": 0.6383,
        "rougeL": 0.6212
      },
      "Swa_Ken": {
        "rouge1": 0.8162,
        "rougeL": 0.7966
      }
    }
  },
  "oracle": {
    "label": "oracle_topk",
    "rouge1": 0.6840949279607131,
    "rougeL": 0.6231081254679269,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.3957,
        "rougeL": 0.2245
      },
      "Amh_Eth": {
        "rouge1": 0.3084,
        "rougeL": 0.2756
      },
      "Eng_Eth": {
        "rouge1": 0.7509,
        "rougeL": 0.7291
      },
      "Eng_Gha": {
        "rouge1": 0.3854,
        "rougeL": 0.2488
      },
      "Eng_Ken": {
        "rouge1": 0.9204,
        "rougeL": 0.9058
      },
      "Eng_Uga": {
        "rouge1": 0.938,
        "rougeL": 0.93
      },
      "Lug_Uga": {
        "rouge1": 0.8535,
        "rougeL": 0.8393
      },
      "Swa_Ken": {
        "rouge1": 0.9212,
        "rougeL": 0.9065
      }
    }
  },
  "delta_rerank_vs_top1": 0.031735858683316076,
  "delta_rerank_vs_exp2": -0.018001225266423226,
  "diagnostics": {
    "exact_oracle_ge_095_and_rerank_lt_050_rows": 462,
    "exact_oracle_ge_095_and_rerank_lt_050_gain_if_oracle": 0.05501189114785372,
    "exact_oracle_ge_095_missed_rows": 572,
    "top1_ge_095_jumped_rows": 198,
    "top1_ge_095_jumped_loss_vs_top1": 0.018777411582631083,
    "changed_from_top1_rows": 3837,
    "wins_vs_top1_rows": 2152,
    "hurts_vs_top1_rows": 1637
  }
}
```

</details>

<details><summary>exp5_bgem3_encoder_mining_v2 — <code>modal_outputs/exp5_bgem3_encoder_mining_v2/summary.json</code></summary>

```json
{
  "experiment": "exp5_bgem3_encoder_mining_v2",
  "gpu": "L40S",
  "dense_k": 200,
  "lexical_k": 50,
  "eval_k": 50,
  "pairs_per_anchor": 4,
  "epochs": 2,
  "batch_size": 32,
  "grad_accum": 1,
  "effective_batch": 32,
  "learning_rate": 1.5e-05,
  "max_seq_length": 256,
  "lora_r": 32,
  "lora_alpha": 64,
  "lora_dropout": 0.05,
  "train_seconds": 3723.5051488876343,
  "train_examples": 141395,
  "anchor_stats": {
    "ok": 28768,
    "dropped": 1046
  },
  "eval": {
    "top1_r1": 0.5465267817657121,
    "oracle_r1": 0.6888770220521963,
    "per_subset": {
      "Aka_Gha": {
        "top1_r1": 0.295,
        "oracle_r1": 0.4054
      },
      "Amh_Eth": {
        "top1_r1": 0.1781,
        "oracle_r1": 0.312
      },
      "Eng_Eth": {
        "top1_r1": 0.5772,
        "oracle_r1": 0.7539
      },
      "Eng_Gha": {
        "top1_r1": 0.2936,
        "oracle_r1": 0.3906
      },
      "Eng_Ken": {
        "top1_r1": 0.8129,
        "oracle_r1": 0.9221
      },
      "Eng_Uga": {
        "top1_r1": 0.8216,
        "oracle_r1": 0.9386
      },
      "Lug_Uga": {
        "top1_r1": 0.5757,
        "oracle_r1": 0.8668
      },
      "Swa_Ken": {
        "top1_r1": 0.7774,
        "oracle_r1": 0.9198
      }
    }
  },
  "delta_top1_vs_exp2_encoder": 0.0069809539818342925,
  "delta_oracle50_vs_exp2_encoder": 0.005181047643788017
}
```

</details>

<details><summary>exp5_encoder_candidates_scored_by_exp2_reranker — <code>modal_outputs/exp5_encoder_exp2_rerank_eval/summary.json</code></summary>

```json
{
  "experiment": "exp5_encoder_candidates_scored_by_exp2_reranker",
  "gpu": "L40S",
  "k": 50,
  "encoder": "exp5_bgem3_encoder_mining_v2/final",
  "reranker": "exp2_crossencoder_rerank/final",
  "top1": {
    "label": "exp5_encoder_top1",
    "rouge1": 0.5463792018516199,
    "rougeL": 0.49860449347328434,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.2949,
        "rougeL": 0.1769
      },
      "Amh_Eth": {
        "rouge1": 0.1791,
        "rougeL": 0.1668
      },
      "Eng_Eth": {
        "rouge1": 0.5779,
        "rougeL": 0.5592
      },
      "Eng_Gha": {
        "rouge1": 0.2936,
        "rougeL": 0.1957
      },
      "Eng_Ken": {
        "rouge1": 0.813,
        "rougeL": 0.7958
      },
      "Eng_Uga": {
        "rouge1": 0.8206,
        "rougeL": 0.8049
      },
      "Lug_Uga": {
        "rouge1": 0.5757,
        "rougeL": 0.5536
      },
      "Swa_Ken": {
        "rouge1": 0.7771,
        "rougeL": 0.7543
      }
    }
  },
  "rerank": {
    "label": "exp5_encoder_exp2_rerank",
    "rouge1": 0.5902712632484098,
    "rougeL": 0.5407295839099605,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.3247,
        "rougeL": 0.1909
      },
      "Amh_Eth": {
        "rouge1": 0.1899,
        "rougeL": 0.1763
      },
      "Eng_Eth": {
        "rouge1": 0.6691,
        "rougeL": 0.6513
      },
      "Eng_Gha": {
        "rouge1": 0.3012,
        "rougeL": 0.1986
      },
      "Eng_Ken": {
        "rouge1": 0.8153,
        "rougeL": 0.7959
      },
      "Eng_Uga": {
        "rouge1": 0.8645,
        "rougeL": 0.8529
      },
      "Lug_Uga": {
        "rouge1": 0.6877,
        "rougeL": 0.669
      },
      "Swa_Ken": {
        "rouge1": 0.8266,
        "rougeL": 0.8079
      }
    }
  },
  "oracle": {
    "label": "exp5_encoder_oracle_top50",
    "rouge1": 0.6889750171263002,
    "rougeL": 0.627569495359982,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.4054,
        "rougeL": 0.2326
      },
      "Amh_Eth": {
        "rouge1": 0.3118,
        "rougeL": 0.2803
      },
      "Eng_Eth": {
        "rouge1": 0.7539,
        "rougeL": 0.7297
      },
      "Eng_Gha": {
        "rouge1": 0.3906,
        "rougeL": 0.2517
      },
      "Eng_Ken": {
        "rouge1": 0.9221,
        "rougeL": 0.908
      },
      "Eng_Uga": {
        "rouge1": 0.939,
        "rougeL": 0.9314
      },
      "Lug_Uga": {
        "rouge1": 0.8668,
        "rougeL": 0.8549
      },
      "Swa_Ken": {
        "rouge1": 0.9198,
        "rougeL": 0.9044
      }
    }
  },
  "delta_top1_vs_exp2_encoder": 0.006833374067742093,
  "delta_oracle_vs_exp2_encoder": 0.005279042717891924,
  "delta_rerank_vs_exp2": 0.0010546349015952972,
  "diagnostics": {
    "changed_from_top1_rows": 3291,
    "wins_vs_top1_rows": 2024,
    "hurts_vs_top1_rows": 1228,
    "exact_oracle_ge_095_and_rerank_lt_050_rows": 399
  }
}
```

</details>

<details><summary>exp5_encoder_exp2_rerank_test_predictions — <code>modal_outputs/exp5_encoder_exp2_test_predictions_files/summary.json</code></summary>

```json
{
  "experiment": "exp5_encoder_exp2_rerank_test_predictions",
  "k": 50,
  "bank": "Train.csv + Val.csv",
  "encoder": "/data/exp5_bgem3_encoder_mining_v2/final",
  "reranker": "/data/exp2_crossencoder_rerank/final",
  "test_rows": 2618,
  "submission": "/data/exp5_encoder_exp2_test_predictions/submission_exp5_encoder_exp2_rerank_trainval.csv",
  "encoder_top1_submission": "/data/exp5_encoder_exp2_test_predictions/submission_exp5_encoder_top1_trainval.csv",
  "test_predictions": "/data/exp5_encoder_exp2_test_predictions/test_predictions.csv",
  "debug_candidates": "/data/exp5_encoder_exp2_test_predictions/test_candidate_scores.csv",
  "blank_predictions": 0,
  "changed_by_reranker_vs_encoder_top1": 1206,
  "mean_chosen_rank": 9.75057295645531
}
```

</details>

<details><summary>exp8_ghana_grouped_encoder_reranker — <code>modal_outputs/exp8_ghana_grouped_encoder_reranker/summary.json</code></summary>

```json
{
  "experiment": "exp8_ghana_grouped_encoder_reranker",
  "subsets": [
    "Aka_Gha",
    "Eng_Gha"
  ],
  "gpu": "L40S",
  "train_rows": 8898,
  "val_rows": 2218,
  "k": 50,
  "dense_k": 200,
  "lexical_k": 50,
  "encoder_epochs": 1,
  "encoder_pairs_per_anchor": 6,
  "encoder_train_seconds": 843.6410613059998,
  "encoder_examples": 52320,
  "encoder_dir": "/data/exp8_ghana_grouped_encoder_reranker/encoder_final",
  "reranker_epochs": 1,
  "reranker_pairs_per_query": 12,
  "reranker_train_seconds": 1335.3309898376465,
  "reranker_examples": 106776,
  "reranker_dir": "/data/exp8_ghana_grouped_encoder_reranker/reranker_final",
  "global_encoder": {
    "label": "global_encoder_ghana_val",
    "top1_r1": 0.28671556828925526,
    "oracle_r1": 0.38973992919407185,
    "per_subset": {
      "Aka_Gha": {
        "top1_r1": 0.2903,
        "oracle_r1": 0.395
      },
      "Eng_Gha": {
        "top1_r1": 0.2831,
        "oracle_r1": 0.3844
      }
    }
  },
  "ghana_encoder": {
    "label": "ghana_encoder_ghana_val",
    "top1_r1": 0.29205916552922917,
    "oracle_r1": 0.3944582097807091,
    "per_subset": {
      "Aka_Gha": {
        "top1_r1": 0.2939,
        "oracle_r1": 0.4005
      },
      "Eng_Gha": {
        "top1_r1": 0.2902,
        "oracle_r1": 0.3884
      }
    }
  },
  "global_rerank": {
    "label": "global_encoder_global_reranker",
    "rouge1": 0.3120674036588961,
    "rougeL": 0.19395176834560654,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.3217,
        "rougeL": 0.1887
      },
      "Eng_Gha": {
        "rouge1": 0.3024,
        "rougeL": 0.1993
      }
    }
  },
  "ghana_rerank": {
    "label": "ghana_encoder_ghana_reranker",
    "rouge1": 0.3223572650558902,
    "rougeL": 0.2018567776032025,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.3261,
        "rougeL": 0.193
      },
      "Eng_Gha": {
        "rouge1": 0.3186,
        "rougeL": 0.2108
      }
    }
  },
  "ghana_top1": {
    "label": "ghana_encoder_top1",
    "rouge1": 0.29205916552922917,
    "rougeL": 0.18459757426106385,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.2939,
        "rougeL": 0.1765
      },
      "Eng_Gha": {
        "rouge1": 0.2902,
        "rougeL": 0.1928
      }
    }
  },
  "delta_ghana_rerank_vs_global_rerank": 0.01028986139699406,
  "delta_ghana_encoder_oracle_vs_global_encoder_oracle": 0.004718280586637258
}
```

</details>

<details><summary>exp9_jina_multilingual_reranker — <code>modal_outputs/exp9_jina_multilingual_reranker/summary.json</code></summary>

```json
{
  "experiment": "exp9_jina_multilingual_reranker",
  "base_model": "jinaai/jina-reranker-v2-base-multilingual",
  "gpu": "L40S",
  "k": 50,
  "train_pairs_per_query": 12,
  "epochs": 1,
  "batch_size": 16,
  "gradient_accumulation": 1,
  "effective_batch": 16,
  "learning_rate": 1e-05,
  "train_seconds": 5082.837837934494,
  "candidate_baseline": {
    "label": "ft_bgem3_per_subset_top50",
    "top1_r1": 0.5395458277838778,
    "oracle_r1": 0.6836959744084082,
    "per_subset": {
      "Aka_Gha": {
        "top1_r1": 0.2903,
        "oracle_r1": 0.395
      },
      "Amh_Eth": {
        "top1_r1": 0.1711,
        "oracle_r1": 0.3079
      },
      "Eng_Eth": {
        "top1_r1": 0.5629,
        "oracle_r1": 0.7506
      },
      "Eng_Gha": {
        "top1_r1": 0.2831,
        "oracle_r1": 0.3844
      },
      "Eng_Ken": {
        "top1_r1": 0.7992,
        "oracle_r1": 0.9202
      },
      "Eng_Uga": {
        "top1_r1": 0.8171,
        "oracle_r1": 0.938
      },
      "Lug_Uga": {
        "top1_r1": 0.5587,
        "oracle_r1": 0.8533
      },
      "Swa_Ken": {
        "top1_r1": 0.7941,
        "oracle_r1": 0.921
      }
    }
  },
  "top1": {
    "label": "ft_bgem3_top1",
    "rouge1": 0.5395458277838778,
    "rougeL": 0.4922840777969995,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.2903,
        "rougeL": 0.173
      },
      "Amh_Eth": {
        "rouge1": 0.1711,
        "rougeL": 0.16
      },
      "Eng_Eth": {
        "rouge1": 0.5629,
        "rougeL": 0.5438
      },
      "Eng_Gha": {
        "rouge1": 0.2831,
        "rougeL": 0.1869
      },
      "Eng_Ken": {
        "rouge1": 0.7992,
        "rougeL": 0.7816
      },
      "Eng_Uga": {
        "rouge1": 0.8171,
        "rougeL": 0.8008
      },
      "Lug_Uga": {
        "rouge1": 0.5587,
        "rougeL": 0.5357
      },
      "Swa_Ken": {
        "rouge1": 0.7941,
        "rougeL": 0.7759
      }
    }
  },
  "rerank": {
    "label": "jina_reranker",
    "rouge1": 0.5770043163797073,
    "rougeL": 0.5275465017883888,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.3185,
        "rougeL": 0.1888
      },
      "Amh_Eth": {
        "rouge1": 0.182,
        "rougeL": 0.169
      },
      "Eng_Eth": {
        "rouge1": 0.6543,
        "rougeL": 0.6387
      },
      "Eng_Gha": {
        "rouge1": 0.299,
        "rougeL": 0.1955
      },
      "Eng_Ken": {
        "rouge1": 0.7956,
        "rougeL": 0.7731
      },
      "Eng_Uga": {
        "rouge1": 0.8545,
        "rougeL": 0.8412
      },
      "Lug_Uga": {
        "rouge1": 0.6587,
        "rougeL": 0.6413
      },
      "Swa_Ken": {
        "rouge1": 0.7914,
        "rougeL": 0.7695
      }
    }
  },
  "oracle": {
    "label": "oracle_topk",
    "rouge1": 0.6836959744084082,
    "rougeL": 0.6256092887229702,
    "per_subset": {
      "Aka_Gha": {
        "rouge1": 0.395,
        "rougeL": 0.2291
      },
      "Amh_Eth": {
        "rouge1": 0.3079,
        "rougeL": 0.2791
      },
      "Eng_Eth": {
        "rouge1": 0.7506,
        "rougeL": 0.7314
      },
      "Eng_Gha": {
        "rouge1": 0.3844,
        "rougeL": 0.255
      },
      "Eng_Ken": {
        "rouge1": 0.9202,
        "rougeL": 0.9069
      },
      "Eng_Uga": {
        "rouge1": 0.938,
        "rougeL": 0.9301
      },
      "Lug_Uga": {
        "rouge1": 0.8533,
        "rougeL": 0.8405
      },
      "Swa_Ken": {
        "rouge1": 0.921,
        "rougeL": 0.9075
      }
    }
  },
  "delta_vs_exp2_rerank_r1": -0.012212311967107259,
  "delta_vs_top1": 0.03745848859582945
}
```

</details>

<details><summary>test_predictions_best_bgem3_top50_crossencoder_rerank — <code>modal_outputs/test_predictions_best_setup/summary.json</code></summary>

```json
{
  "experiment": "test_predictions_best_bgem3_top50_crossencoder_rerank",
  "k": 50,
  "bank": "Train.csv + Val.csv",
  "test_rows": 2618,
  "submission": "/data/test_predictions_best_setup/submission_best_bgem3_rerank_train_val.csv",
  "encoder_top1_submission": "/data/test_predictions_best_setup/submission_encoder_top1_train_val.csv",
  "debug_candidates": "/data/test_predictions_best_setup/test_candidate_scores.csv",
  "blank_predictions": 0,
  "changed_by_reranker_vs_encoder_top1": 1250
}
```

</details>

<details><summary>reports/candidate_ranker_current_best_gate/summary.json — <code>reports/candidate_ranker_current_best_gate/summary.json</code></summary>

```json
{
  "feature_columns": [
    "subset",
    "candidate_rank",
    "bi_rank",
    "rerank_rank",
    "bi_score",
    "rerank_score",
    "rerank_score_max",
    "rerank_score_second",
    "rerank_score_margin_to_best",
    "rerank_score_gap_from_second",
    "rerank_score_delta_to_group_best",
    "bi_score_max",
    "bi_score_delta_to_group_best",
    "is_top1",
    "is_rerank_choice",
    "val_input_len",
    "candidate_question_len",
    "candidate_answer_len",
    "candidate_len_ratio_to_query",
    "query_candidate_question_jaccard",
    "query_candidate_answer_jaccard",
    "query_candidate_best_jaccard"
  ],
  "leakage_columns_excluded": [
    "answer_ref_len_ratio",
    "candidate_r1",
    "is_oracle",
    "output",
    "reference",
    "reference_len",
    "target",
    "target_r1"
  ],
  "current_best_oof_score": 0.6369591077160984,
  "exp2_pool_rerank_score": 0.5889304525640328,
  "exp2_pool_bi_top1_score": 0.5394795443970752,
  "n_val_ids": 6686,
  "n_val_candidate_rows": 334300,
  "n_current_best_predictions_present_in_exp2_pool": 0,
  "models": [
    {
      "model": "hgb",
      "candidate_choice_score": 0.5903836635712,
      "gain_vs_exp2_rerank": 0.001453211007167221,
      "current_best_oof_score": 0.6369591077160984,
      "gain_if_used_directly_vs_current": -0.04657544414489845,
      "best_gate": {
        "model": "hgb",
        "threshold": -0.05,
        "rank_cap": 1,
        "require_current_in_pool": true,
        "score": 0.6369591077160983,
        "gain_vs_current": 0.0,
        "switch_n": 0,
        "mean_gain_switched": 0.0
      }
    },
    {
      "model": "lgbm_reg",
      "candidate_choice_score": 0.590128490730057,
      "gain_vs_exp2_rerank": 0.0011980381660242623,
      "current_best_oof_score": 0.6369591077160984,
      "gain_if_used_directly_vs_current": -0.04683061698604141,
      "best_gate": {
        "model": "lgbm_reg",
        "threshold": 0.24,
        "rank_cap": 1,
        "require_current_in_pool": false,
        "score": 0.6369593213821809,
        "gain_vs_current": 2.136660826046466e-07,
        "switch_n": 5,
        "mean_gain_switched": 0.0002857142857143058
      }
    },
    {
      "model": "ensemble_mean",
      "candidate_choice_score": 0.590425031573055,
      "gain_vs_exp2_rerank": 0.0014945790090222788,
      "current_best_oof_score": 0.6369591077160984,
      "gain_if_used_directly_vs_current": -0.04653407614304339,
      "best_gate": {
        "model": "ensemble_mean",
        "threshold": 0.23,
        "rank_cap": 1,
        "require_current_in_pool": false,
        "score": 0.6369847476460159,
        "gain_vs_current": 2.5639929917553594e-05,
        "switch_n": 4,
        "mean_gain_switched": 0.04285714285714288
      }
    },
    {
      "model": "ensemble_max",
      "candidate_choice_score": 0.5899544458717647,
      "gain_vs_exp2_rerank": 0.0010239933077319519,
      "current_best_oof_score": 0.6369591077160984,
      "gain_if_used_directly_vs_current": -0.04700466184433372,
      "best_gate": {
        "model": "ensemble_max",
        "threshold": 0.25,
        "rank_cap": 3,
        "require_current_in_pool": false,
        "score": 0.6370144066991146,
        "gain_vs_current": 5.5298983016260905e-05,
        "switch_n": 7,
        "mean_gain_switched": 0.05281842863521275
      }
    }
  ],
  "test": {
    "selected_model_for_test": "ensemble_max",
    "selected_gate": {
      "model": "ensemble_max",
      "threshold": 0.25,
      "rank_cap": 3,
      "require_current_in_pool": false,
      "score": 0.6370144066991146,
      "gain_vs_current": 5.5298983016260905e-05,
      "switch_n": 7,
      "mean_gain_switched": 0.05281842863521275
    },
    "fitted_models": [
      {
        "model": "hgb",
        "is_ranker": false
      },
      {
        "model": "lgbm_reg",
        "is_ranker": false
      }
    ],
    "submission": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_submissions\\submission_exp7_exp5_candidate_ranker_gate.csv",
    "test_override_count": 1,
    "changed_vs_current_best_submission": 1
  },
  "notes": [
    "This experiment ranks individual candidates, then gates against the current best exp7+exp5 submission.",
    "Features intentionally exclude target/reference/oracle columns; target ROUGE is used only as the training label.",
    "A high validation score from direct candidate selection is not enough here; only the gate-vs-current metric matters."
  ]
}
```

</details>

<details><summary>reports/clean_deployable_source_selector/summary.json — <code>reports/clean_deployable_source_selector/summary.json</code></summary>

```json
{
  "loaded_sources": [
    "exp2_top1",
    "exp2_rerank",
    "cluster_fast",
    "cluster_noleak"
  ],
  "missing_sources": [],
  "best_single_source": {
    "source": "exp2_rerank",
    "coverage": 1.0,
    "rouge1": 0.5892166283468145
  },
  "exp2_rerank": 0.5892166283468145,
  "deployable_oracle": 0.6479322858935932,
  "deployable_oracle_gain_vs_exp2": 0.058715657546778655,
  "best_selector": {
    "label": "hgb",
    "rouge1": 0.6041607089218853,
    "gain_vs_exp2_rerank": 0.014944080575070795,
    "gain_vs_best_single": 0.014944080575070795,
    "top_choice": "exp2_rerank"
  },
  "selector_results": [
    {
      "label": "hgb",
      "rouge1": 0.6041607089218853,
      "choice_counts": {
        "exp2_rerank": 3510,
        "cluster_fast": 1869,
        "cluster_noleak": 840,
        "exp2_top1": 467
      },
      "per_subset": {
        "Aka_Gha": 0.322518,
        "Amh_Eth": 0.187013,
        "Eng_Eth": 0.668439,
        "Eng_Gha": 0.304773,
        "Eng_Ken": 0.827397,
        "Eng_Uga": 0.878956,
        "Lug_Uga": 0.761695,
        "Swa_Ken": 0.829163
      }
    },
    {
      "label": "extra_trees",
      "rouge1": 0.6039149386175271,
      "choice_counts": {
        "exp2_rerank": 2831,
        "cluster_fast": 1616,
        "cluster_noleak": 1264,
        "exp2_top1": 975
      },
      "per_subset": {
        "Aka_Gha": 0.32252,
        "Amh_Eth": 0.190135,
        "Eng_Eth": 0.673209,
        "Eng_Gha": 0.306391,
        "Eng_Ken": 0.823044,
        "Eng_Uga": 0.877135,
        "Lug_Uga": 0.760035,
        "Swa_Ken": 0.826486
      }
    },
    {
      "label": "ensemble",
      "rouge1": 0.6035426289913252,
      "choice_counts": {
        "exp2_rerank": 3415,
        "cluster_fast": 1940,
        "cluster_noleak": 919,
        "exp2_top1": 412
      },
      "per_subset": {
        "Aka_Gha": 0.323082,
        "Amh_Eth": 0.190181,
        "Eng_Eth": 0.671435,
        "Eng_Gha": 0.3053,
        "Eng_Ken": 0.826186,
        "Eng_Uga": 0.877588,
        "Lug_Uga": 0.755759,
        "Swa_Ken": 0.827831
      }
    }
  ]
}
```

</details>

<details><summary>reports/cluster_fast_gate/summary.json — <code>reports/cluster_fast_gate/summary.json</code></summary>

```json
{
  "baseline_exp2": 0.5892166283468145,
  "baseline_hybrid": 0.6065232342601322,
  "cluster_fast": 0.5164699179116768,
  "oracle_fast_vs_hybrid": 0.6245958706481922,
  "fast_beats_hybrid_rate": 0.0717918037690697,
  "best": {
    "score": 0.6224808076839063,
    "gain_vs_hybrid": 0.01595757342377413,
    "use_fast_rate": 0.12503739156446306,
    "use_fast_count": 836,
    "per_subset": {
      "Aka_Gha": 0.325431,
      "Amh_Eth": 0.197348,
      "Eng_Eth": 0.685493,
      "Eng_Gha": 0.309539,
      "Eng_Ken": 0.854441,
      "Eng_Uga": 0.901727,
      "Lug_Uga": 0.806415,
      "Swa_Ken": 0.853817
    },
    "model": "hgb_pred_gain",
    "scope": "all_sweep",
    "threshold": 0.004963183600921184
  },
  "target_subsets": [
    "Lug_Uga",
    "Eng_Uga",
    "Eng_Ken",
    "Swa_Ken"
  ],
  "notes": [
    "Threshold-sweep rows are diagnostic and may be optimistic because the threshold is chosen on OOF validation predictions.",
    "The zero-threshold rows are the cleaner read of whether predicted gain is calibrated."
  ]
}
```

</details>

<details><summary>reports/deployable_source_selector/summary.json — <code>reports/deployable_source_selector/summary.json</code></summary>

```json
{
  "loaded_sources": [
    "exp2_top1",
    "exp2_rerank",
    "cluster_fast",
    "cluster_fullcap",
    "cluster_noleak"
  ],
  "missing_sources": [
    {
      "source": "local_regressor",
      "reason": "bad_columns",
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\reports\\local_candidate_regressor_submission\\val_oof_choices.csv",
      "columns": [
        "ID",
        "subset",
        "candidate_rank",
        "rerank_rank",
        "candidate_r1",
        "pred_hgb"
      ]
    },
    {
      "source": "local_regressor_margin",
      "reason": "bad_columns",
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\reports\\local_candidate_regressor_submission\\val_margin_gate_rows.csv",
      "columns": [
        "ID",
        "subset",
        "candidate_r1_best",
        "pred_hgb_best",
        "candidate_r1_rerank",
        "pred_hgb_rerank",
        "pred_margin_vs_rerank",
        "actual_gain_vs_rerank"
      ]
    }
  ],
  "best_single_source": {
    "source": "cluster_fullcap",
    "coverage": 1.0,
    "rouge1": 0.6353457041144692
  },
  "exp2_rerank": 0.5892166283468145,
  "deployable_oracle": 0.6594522709392795,
  "deployable_oracle_gain_vs_exp2": 0.07023564259246495,
  "best_selector": {
    "label": "extra_trees",
    "rouge1": 0.6301936577454038,
    "gain_vs_exp2_rerank": 0.04097702939858927,
    "gain_vs_best_single": -0.005152046369065411,
    "top_choice": "cluster_fullcap"
  },
  "selector_results": [
    {
      "label": "hgb",
      "rouge1": 0.628435960151685,
      "choice_counts": {
        "cluster_fullcap": 4197,
        "exp2_rerank": 1659,
        "cluster_fast": 439,
        "exp2_top1": 203,
        "cluster_noleak": 188
      },
      "per_subset": {
        "Aka_Gha": 0.331688,
        "Amh_Eth": 0.198123,
        "Eng_Eth": 0.674056,
        "Eng_Gha": 0.310832,
        "Eng_Ken": 0.874939,
        "Eng_Uga": 0.906712,
        "Lug_Uga": 0.825412,
        "Swa_Ken": 0.863531
      }
    },
    {
      "label": "extra_trees",
      "rouge1": 0.6301936577454038,
      "choice_counts": {
        "cluster_fullcap": 3189,
        "exp2_rerank": 2103,
        "cluster_fast": 620,
        "exp2_top1": 499,
        "cluster_noleak": 275
      },
      "per_subset": {
        "Aka_Gha": 0.332544,
        "Amh_Eth": 0.193642,
        "Eng_Eth": 0.673945,
        "Eng_Gha": 0.313094,
        "Eng_Ken": 0.87915,
        "Eng_Uga": 0.909146,
        "Lug_Uga": 0.82877,
        "Swa_Ken": 0.867085
      }
    },
    {
      "label": "ensemble",
      "rouge1": 0.62988559302051,
      "choice_counts": {
        "cluster_fullcap": 4138,
        "exp2_rerank": 1654,
        "cluster_fast": 531,
        "exp2_top1": 194,
        "cluster_noleak": 169
      },
      "per_subset": {
        "Aka_Gha": 0.332241,
        "Amh_Eth": 0.195792,
        "Eng_Eth": 0.67393,
        "Eng_Gha": 0.311668,
        "Eng_Ken": 0.878472,
        "Eng_Uga": 0.908989,
        "Lug_Uga": 0.826882,
        "Swa_Ken": 0.869005
      }
    }
  ]
}
```

</details>

<details><summary>reports/existing_prediction_oracle_audit/summary.json — <code>reports/existing_prediction_oracle_audit/summary.json</code></summary>

```json
{
  "loaded_sources": [
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
    "exp14_qonly"
  ],
  "missing_sources": [
    {
      "name": "exp13_lug_selector",
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\reports\\exp13_lug_merged_selector\\oof_best_selector_choices.csv"
    },
    {
      "name": "local_regressor_oof",
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\reports\\local_candidate_regressor_submission\\val_oof_choices.csv"
    },
    {
      "name": "local_regressor_margin_gate",
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\reports\\local_candidate_regressor_submission\\val_margin_gate_rows.csv"
    },
    {
      "name": "cluster_selector_fast",
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\reports\\cluster_aware_selector_experiment\\cluster_selector_fast_oof_choices.csv"
    },
    {
      "name": "cluster_selector_fullcap",
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\reports\\cluster_aware_selector_experiment\\cluster_selector_fullcap_oof_choices.csv"
    },
    {
      "name": "cluster_selector_noleak",
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\reports\\cluster_aware_selector_experiment\\cluster_selector_noleak_oof_choices.csv"
    }
  ],
  "best_single_source": {
    "source": "exp3_top100_rerank",
    "coverage": 1.0,
    "rouge1": 0.5904174317680138
  },
  "oracle_existing_sources_r1": 0.6569692116972645,
  "oracle_gain_vs_best_single": 0.06655177992925077,
  "oracle_gain_vs_exp2_rerank": 0.06775258335044998
}
```

</details>

<details><summary>existing_source_candidate_level_selector — <code>reports/existing_source_selector/summary.json</code></summary>

```json
{
  "experiment": "existing_source_candidate_level_selector",
  "rows": 6686,
  "candidate_rows": 86918,
  "sources": [
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
    "exp14_qonly"
  ],
  "best_single_source": {
    "source": "exp3_top100_rerank",
    "rouge1": 0.5904174317680138
  },
  "exp2_rerank": 0.5892166283468145,
  "oracle_existing_sources": 0.6569692116972645,
  "oracle_gain_vs_best_single": 0.06655177992925077,
  "results": [
    {
      "label": "hgb",
      "rouge1": 0.5973082755294336,
      "choice_counts": {
        "exp4_pairwise": 3938,
        "exp2_rerank": 683,
        "exp1_baseline_qa_doc": 580,
        "exp2_top1": 518,
        "exp3_top100_rerank": 354,
        "exp1_finetuned_qa_doc": 246,
        "exp5_encoder_exp2_rerank": 143,
        "exp9_jina": 104,
        "exp8_ghana_grouped": 61,
        "exp14_qonly": 59
      },
      "per_subset": {
        "Aka_Gha": 0.326009,
        "Amh_Eth": 0.194108,
        "Eng_Eth": 0.668667,
        "Eng_Gha": 0.309374,
        "Eng_Ken": 0.82786,
        "Eng_Uga": 0.870496,
        "Lug_Uga": 0.704131,
        "Swa_Ken": 0.838061
      }
    },
    {
      "label": "extra_trees",
      "rouge1": 0.5978363087630308,
      "choice_counts": {
        "exp4_pairwise": 2067,
        "exp2_top1": 732,
        "exp1_baseline_qa_doc": 721,
        "exp1_finetuned_qa_doc": 518,
        "exp3_top100_rerank": 498,
        "exp5_encoder_exp2_rerank": 439,
        "exp8_ghana_grouped": 433,
        "exp2_rerank": 406,
        "exp14_qonly": 349,
        "exp9_jina": 317,
        "exp10_lug_specialized": 115,
        "exp12_lug_merged": 69,
        "exp10_lug_global": 22
      },
      "per_subset": {
        "Aka_Gha": 0.32426,
        "Amh_Eth": 0.19765,
        "Eng_Eth": 0.672758,
        "Eng_Gha": 0.311612,
        "Eng_Ken": 0.822176,
        "Eng_Uga": 0.868886,
        "Lug_Uga": 0.708365,
        "Swa_Ken": 0.838868
      }
    },
    {
      "label": "ridge",
      "rouge1": 0.5945472115690034,
      "choice_counts": {
        "exp4_pairwise": 3964,
        "exp2_top1": 1083,
        "exp2_rerank": 685,
        "exp1_baseline_qa_doc": 309,
        "exp1_finetuned_qa_doc": 168,
        "exp3_top100_rerank": 148,
        "exp5_encoder_exp2_rerank": 131,
        "exp14_qonly": 85,
        "exp9_jina": 80,
        "exp8_ghana_grouped": 32,
        "exp12_lug_merged": 1
      },
      "per_subset": {
        "Aka_Gha": 0.321602,
        "Amh_Eth": 0.194901,
        "Eng_Eth": 0.667544,
        "Eng_Gha": 0.30704,
        "Eng_Ken": 0.813866,
        "Eng_Uga": 0.865645,
        "Lug_Uga": 0.709478,
        "Swa_Ken": 0.835005
      }
    },
    {
      "label": "ensemble_avg",
      "rouge1": 0.5965919897978431,
      "choice_counts": {
        "exp4_pairwise": 4029,
        "exp2_top1": 971,
        "exp2_rerank": 598,
        "exp1_baseline_qa_doc": 294,
        "exp3_top100_rerank": 209,
        "exp1_finetuned_qa_doc": 172,
        "exp5_encoder_exp2_rerank": 159,
        "exp9_jina": 119,
        "exp8_ghana_grouped": 51,
        "exp14_qonly": 45,
        "exp10_lug_specialized": 24,
        "exp12_lug_merged": 14,
        "exp10_lug_global": 1
      },
      "per_subset": {
        "Aka_Gha": 0.324326,
        "Amh_Eth": 0.196249,
        "Eng_Eth": 0.67141,
        "Eng_Gha": 0.308109,
        "Eng_Ken": 0.819911,
        "Eng_Uga": 0.868369,
        "Lug_Uga": 0.709141,
        "Swa_Ken": 0.83497
      }
    }
  ],
  "best": {
    "label": "extra_trees",
    "rouge1": 0.5978363087630308,
    "gain_vs_best_single": 0.007418876995017087,
    "gain_vs_exp2_rerank": 0.008619680416216302,
    "top_choice": "exp4_pairwise"
  },
  "notes": [
    "OOF split is by ID, so all candidates for a validation row are held out together.",
    "Features use only source name, subset, prediction text shape, and agreement/similarity among candidate predictions.",
    "This does not require Modal; conversion to test requires the same prediction sources to exist for test."
  ]
}
```

</details>

<details><summary>existing_source_selector_with_fold_safe_target_encoding — <code>reports/existing_source_selector_target_encoded/summary.json</code></summary>

```json
{
  "experiment": "existing_source_selector_with_fold_safe_target_encoding",
  "results": [
    {
      "label": "hgb",
      "rouge1": 0.5975208075199556,
      "choice_counts": {
        "exp4_pairwise": 3612,
        "exp8_ghana_grouped": 683,
        "exp1_baseline_qa_doc": 599,
        "exp2_rerank": 372,
        "exp2_top1": 342,
        "exp14_qonly": 281,
        "exp1_finetuned_qa_doc": 256,
        "exp3_top100_rerank": 202,
        "exp10_lug_global": 111,
        "exp5_encoder_exp2_rerank": 83,
        "exp9_jina": 71,
        "exp10_lug_specialized": 47,
        "exp12_lug_merged": 27
      },
      "per_subset": {
        "Aka_Gha": 0.324337,
        "Amh_Eth": 0.198798,
        "Eng_Eth": 0.670226,
        "Eng_Gha": 0.312364,
        "Eng_Ken": 0.817814,
        "Eng_Uga": 0.868009,
        "Lug_Uga": 0.709192,
        "Swa_Ken": 0.839552
      }
    },
    {
      "label": "extra_trees",
      "rouge1": 0.5982064328738139,
      "choice_counts": {
        "exp4_pairwise": 1930,
        "exp1_baseline_qa_doc": 714,
        "exp8_ghana_grouped": 696,
        "exp2_top1": 629,
        "exp1_finetuned_qa_doc": 552,
        "exp3_top100_rerank": 494,
        "exp5_encoder_exp2_rerank": 468,
        "exp9_jina": 416,
        "exp2_rerank": 360,
        "exp14_qonly": 262,
        "exp10_lug_specialized": 101,
        "exp12_lug_merged": 53,
        "exp10_lug_global": 11
      },
      "per_subset": {
        "Aka_Gha": 0.323837,
        "Amh_Eth": 0.19719,
        "Eng_Eth": 0.671906,
        "Eng_Gha": 0.314043,
        "Eng_Ken": 0.82237,
        "Eng_Uga": 0.868782,
        "Lug_Uga": 0.70862,
        "Swa_Ken": 0.84049
      }
    },
    {
      "label": "ensemble_te_avg",
      "rouge1": 0.5983992526904078,
      "choice_counts": {
        "exp4_pairwise": 3319,
        "exp8_ghana_grouped": 710,
        "exp2_top1": 588,
        "exp1_baseline_qa_doc": 583,
        "exp1_finetuned_qa_doc": 348,
        "exp3_top100_rerank": 230,
        "exp14_qonly": 226,
        "exp5_encoder_exp2_rerank": 219,
        "exp9_jina": 163,
        "exp2_rerank": 136,
        "exp10_lug_specialized": 90,
        "exp12_lug_merged": 63,
        "exp10_lug_global": 11
      },
      "per_subset": {
        "Aka_Gha": 0.324961,
        "Amh_Eth": 0.200876,
        "Eng_Eth": 0.669562,
        "Eng_Gha": 0.313059,
        "Eng_Ken": 0.819044,
        "Eng_Uga": 0.869551,
        "Lug_Uga": 0.710076,
        "Swa_Ken": 0.839543
      }
    }
  ],
  "best": {
    "label": "ensemble_te_avg",
    "rouge1": 0.5983992526904078,
    "gain_vs_best_single": 0.007981820922394078,
    "gain_vs_exp2_rerank": 0.009182624343593293,
    "top_choice": "exp4_pairwise"
  },
  "notes": [
    "Target encodings are computed inside each GroupKFold split from training IDs only.",
    "This tests whether source/subset priors improve local no-Modal source selection."
  ]
}
```

</details>

<details><summary>exp13_lug_merged_selector — <code>reports/exp13_lug_merged_selector/summary.json</code></summary>

```json
{
  "experiment": "exp13_lug_merged_selector",
  "input_candidates": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp12_lug_e5_merge_rerank\\lug_val_candidate_scores.csv",
  "rows": 46318,
  "queries": 846,
  "features": [
    "rerank_score",
    "rerank_rank",
    "rerank_z",
    "bge_rank_filled",
    "e5_rank_filled",
    "best_rank",
    "rank_gap",
    "bge_score_filled",
    "e5_score_filled",
    "score_gap",
    "has_bge",
    "has_e5",
    "both_sources",
    "answer_len",
    "question_len",
    "answer_char_len",
    "question_char_len",
    "bge_score_rank",
    "e5_score_rank",
    "candidate_count"
  ],
  "baseline": {
    "bge_ft_top1": 0.5586771911482437,
    "e5_large_top1": 0.48765316882277043,
    "exp12_merged_rerank": 0.6835317759537722,
    "merged_oracle": 0.8740366794311258
  },
  "oof_scores": {
    "hgb_l2": {
      "oof_mse": 0.007497982814614326,
      "chosen_rouge1": 0.6918884229860995,
      "changed_vs_exp12_rerank": 78
    },
    "hgb_abs": {
      "oof_mse": 0.008181270221055498,
      "chosen_rouge1": 0.6836209322698509,
      "changed_vs_exp12_rerank": 105
    },
    "rf": {
      "oof_mse": 0.007425765852377139,
      "chosen_rouge1": 0.685071447157713,
      "changed_vs_exp12_rerank": 77
    },
    "ridge": {
      "oof_mse": 0.008804068430234815,
      "chosen_rouge1": 0.6835961281869238,
      "changed_vs_exp12_rerank": 59
    },
    "pred_mean": {
      "chosen_rouge1": 0.687454813609036,
      "changed_vs_exp12_rerank": 52
    },
    "pred_maxblend": {
      "chosen_rouge1": 0.6878100488311472,
      "changed_vs_exp12_rerank": 49
    }
  },
  "best_selector": "hgb_l2",
  "best_selector_rouge1": 0.6918884229860995,
  "delta_vs_exp12_merged_rerank": 0.008356647032327302,
  "remaining_gap_to_oracle": 0.1821482564450263
}
```

</details>

<details><summary>exp14_qonly_vs_exp2_selector — <code>reports/exp14_qonly_exp2_selector/summary.json</code></summary>

```json
{
  "experiment": "exp14_qonly_vs_exp2_selector",
  "rows": 6686,
  "feature_count": 55,
  "categorical_features": [
    "subset"
  ],
  "numeric_feature_count": 54,
  "results": [
    {
      "label": "hgb",
      "base_exp2_r1": 0.5892166283468145,
      "qonly_r1": 0.5647796047735751,
      "oracle_between_exp2_qonly": 0.6134483465316233,
      "best_threshold": 0.6499999999999999,
      "selected_r1": 0.5896085826957426,
      "gain_vs_exp2": 0.0003919543489280253,
      "qonly_pick_rate": 0.02034101106790308,
      "accuracy": 0.8390667065510021,
      "auc": 0.8680962319087684
    },
    {
      "label": "extra_trees",
      "base_exp2_r1": 0.5892166283468145,
      "qonly_r1": 0.5647796047735751,
      "oracle_between_exp2_qonly": 0.6134483465316233,
      "best_threshold": 0.83,
      "selected_r1": 0.5897932982629549,
      "gain_vs_exp2": 0.0005766699161403599,
      "qonly_pick_rate": 0.009572240502542627,
      "accuracy": 0.8381693090038888,
      "auc": 0.8698233830193256
    },
    {
      "label": "rf",
      "base_exp2_r1": 0.5892166283468145,
      "qonly_r1": 0.5647796047735751,
      "oracle_between_exp2_qonly": 0.6134483465316233,
      "best_threshold": 0.82,
      "selected_r1": 0.5897902140659471,
      "gain_vs_exp2": 0.0005735857191325744,
      "qonly_pick_rate": 0.011067903081064912,
      "accuracy": 0.8393658390667066,
      "auc": 0.8713328163629328
    },
    {
      "label": "logreg",
      "base_exp2_r1": 0.5892166283468145,
      "qonly_r1": 0.5647796047735751,
      "oracle_between_exp2_qonly": 0.6134483465316233,
      "best_threshold": 0.8799999999999999,
      "selected_r1": 0.5899765648245269,
      "gain_vs_exp2": 0.0007599364777123307,
      "qonly_pick_rate": 0.01226443314388274,
      "accuracy": 0.8384684415195932,
      "auc": 0.8688771796368311
    },
    {
      "label": "ensemble_avg",
      "base_exp2_r1": 0.5892166283468145,
      "qonly_r1": 0.5647796047735751,
      "oracle_between_exp2_qonly": 0.6134483465316233,
      "best_threshold": 0.7999999999999999,
      "selected_r1": 0.5898206338728043,
      "gain_vs_exp2": 0.0006040055259897148,
      "qonly_pick_rate": 0.00807657792402034,
      "accuracy": 0.8378701764881843,
      "auc": 0.872758802158816
    }
  ],
  "best": {
    "label": "logreg",
    "base_exp2_r1": 0.5892166283468145,
    "qonly_r1": 0.5647796047735751,
    "oracle_between_exp2_qonly": 0.6134483465316233,
    "best_threshold": 0.8799999999999999,
    "selected_r1": 0.5899765648245269,
    "gain_vs_exp2": 0.0007599364777123307,
    "qonly_pick_rate": 0.01226443314388274,
    "accuracy": 0.8384684415195932,
    "auc": 0.8688771796368311
  },
  "notes": [
    "OOF selector uses only deployable prediction/text-shape features; no reference-derived features are included.",
    "Target is whether q-only prediction has higher ROUGE-1 than exp2 q+a prediction.",
    "Threshold is selected on OOF predictions, so treat gain as exploratory but less leaky than in-sample fitting."
  ]
}
```

</details>

<details><summary>reports/exp7_base_length_grouped_submission/summary.json — <code>reports/exp7_base_length_grouped_submission/summary.json</code></summary>

```json
{
  "submission": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_submissions\\submission_exp7_cluster_base_length_grouped_available_sources.csv",
  "base_source": "exp7_cluster",
  "available_sources": [
    "exp7_cluster",
    "exp6_reranker",
    "dense_top1",
    "local_regressor"
  ],
  "source_scores": [
    {
      "source": "exp7_cluster",
      "target_r1": 0.6363771294109717
    },
    {
      "source": "exp6_reranker",
      "target_r1": 0.5892166283468147
    },
    {
      "source": "dense_top1",
      "target_r1": 0.5395458277838778
    },
    {
      "source": "local_regressor",
      "target_r1": 0.10399207748125318
    }
  ],
  "full_validation_group_score": 0.6375100486856335,
  "oof_group_score": 0.6368311486805152,
  "oof_gain_vs_exp2": 0.047614520333700616,
  "full_validation_gain_vs_exp2": 0.048293420338819004,
  "fold_scores": [
    {
      "fold": 1,
      "score": 0.6437999822173471,
      "rows": 1338
    },
    {
      "fold": 2,
      "score": 0.6411431125863071,
      "rows": 1337
    },
    {
      "fold": 3,
      "score": 0.6302298997718883,
      "rows": 1337
    },
    {
      "fold": 4,
      "score": 0.6366130026017469,
      "rows": 1337
    },
    {
      "fold": 5,
      "score": 0.632364533933935,
      "rows": 1337
    }
  ],
  "group_picks": [
    {
      "subset": "Aka_Gha",
      "length_bin": 0,
      "source": "exp7_cluster"
    },
    {
      "subset": "Aka_Gha",
      "length_bin": 1,
      "source": "exp7_cluster"
    },
    {
      "subset": "Aka_Gha",
      "length_bin": 2,
      "source": "exp7_cluster"
    },
    {
      "subset": "Aka_Gha",
      "length_bin": 3,
      "source": "exp7_cluster"
    },
    {
      "subset": "Aka_Gha",
      "length_bin": 4,
      "source": "exp7_cluster"
    },
    {
      "subset": "Amh_Eth",
      "length_bin": 0,
      "source": "exp7_cluster"
    },
    {
      "subset": "Amh_Eth",
      "length_bin": 1,
      "source": "exp7_cluster"
    },
    {
      "subset": "Amh_Eth",
      "length_bin": 2,
      "source": "exp6_reranker"
    },
    {
      "subset": "Amh_Eth",
      "length_bin": 3,
      "source": "exp6_reranker"
    },
    {
      "subset": "Amh_Eth",
      "length_bin": 4,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Eth",
      "length_bin": 0,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Eth",
      "length_bin": 1,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Eth",
      "length_bin": 2,
      "source": "exp6_reranker"
    },
    {
      "subset": "Eng_Eth",
      "length_bin": 3,
      "source": "exp6_reranker"
    },
    {
      "subset": "Eng_Eth",
      "length_bin": 4,
      "source": "exp6_reranker"
    },
    {
      "subset": "Eng_Gha",
      "length_bin": 0,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Gha",
      "length_bin": 1,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Gha",
      "length_bin": 2,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Gha",
      "length_bin": 3,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Gha",
      "length_bin": 4,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Ken",
      "length_bin": 0,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Ken",
      "length_bin": 1,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Ken",
      "length_bin": 2,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Ken",
      "length_bin": 3,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Ken",
      "length_bin": 4,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Uga",
      "length_bin": 0,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Uga",
      "length_bin": 1,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Uga",
      "length_bin": 2,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Uga",
      "length_bin": 3,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Uga",
      "length_bin": 4,
      "source": "exp7_cluster"
    },
    {
      "subset": "Lug_Uga",
      "length_bin": 0,
      "source": "exp7_cluster"
    },
    {
      "subset": "Lug_Uga",
      "length_bin": 1,
      "source": "exp7_cluster"
    },
    {
      "subset": "Lug_Uga",
      "length_bin": 2,
      "source": "exp7_cluster"
    },
    {
      "subset": "Lug_Uga",
      "length_bin": 3,
      "source": "exp7_cluster"
    },
    {
      "subset": "Lug_Uga",
      "length_bin": 4,
      "source": "exp7_cluster"
    },
    {
      "subset": "Swa_Ken",
      "length_bin": 0,
      "source": "exp7_cluster"
    },
    {
      "subset": "Swa_Ken",
      "length_bin": 1,
      "source": "exp7_cluster"
    },
    {
      "subset": "Swa_Ken",
      "length_bin": 2,
      "source": "exp7_cluster"
    },
    {
      "subset": "Swa_Ken",
      "length_bin": 3,
      "source": "exp7_cluster"
    },
    {
      "subset": "Swa_Ken",
      "length_bin": 4,
      "source": "exp7_cluster"
    }
  ],
  "test_choice_counts": {
    "exp7_cluster": 2560,
    "exp6_reranker": 58
  },
  "diff_vs_exp7_cluster": 11,
  "notes": [
    "No Modal compute used.",
    "This starts from submission_exp7_cluster_selector_trainval.csv as the base/fallback.",
    "Group choices are learned from validation labels; OOF score is the safer estimate.",
    "Only test-available sources are used."
  ]
}
```

</details>

<details><summary>reports/exp7_exp5_length_grouped_submission/summary.json — <code>reports/exp7_exp5_length_grouped_submission/summary.json</code></summary>

```json
{
  "submission": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp5_encoder_exp2_test_predictions_files\\submission_exp7_base_exp5_length_grouped.csv",
  "sources": [
    "exp7_cluster",
    "exp5_encoder_exp2"
  ],
  "source_scores": {
    "exp5_encoder_exp2": 0.5902712632484098,
    "exp7_cluster": 0.6363771294109717
  },
  "full_validation_group_score": 0.637714650600872,
  "oof_group_score": 0.6369591077160984,
  "oof_gain_vs_exp2": 0.047742479369283886,
  "fold_scores": [
    {
      "fold": 1,
      "score": 0.6438033101167855
    },
    {
      "fold": 2,
      "score": 0.6411538283540543
    },
    {
      "fold": 3,
      "score": 0.6307107248393876
    },
    {
      "fold": 4,
      "score": 0.6365644297662189
    },
    {
      "fold": 5,
      "score": 0.6325581264296993
    }
  ],
  "group_picks": [
    {
      "subset": "Aka_Gha",
      "length_bin": 0,
      "source": "exp7_cluster"
    },
    {
      "subset": "Aka_Gha",
      "length_bin": 1,
      "source": "exp7_cluster"
    },
    {
      "subset": "Aka_Gha",
      "length_bin": 2,
      "source": "exp7_cluster"
    },
    {
      "subset": "Aka_Gha",
      "length_bin": 3,
      "source": "exp7_cluster"
    },
    {
      "subset": "Aka_Gha",
      "length_bin": 4,
      "source": "exp7_cluster"
    },
    {
      "subset": "Amh_Eth",
      "length_bin": 0,
      "source": "exp7_cluster"
    },
    {
      "subset": "Amh_Eth",
      "length_bin": 1,
      "source": "exp7_cluster"
    },
    {
      "subset": "Amh_Eth",
      "length_bin": 2,
      "source": "exp5_encoder_exp2"
    },
    {
      "subset": "Amh_Eth",
      "length_bin": 3,
      "source": "exp5_encoder_exp2"
    },
    {
      "subset": "Amh_Eth",
      "length_bin": 4,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Eth",
      "length_bin": 0,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Eth",
      "length_bin": 1,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Eth",
      "length_bin": 2,
      "source": "exp5_encoder_exp2"
    },
    {
      "subset": "Eng_Eth",
      "length_bin": 3,
      "source": "exp5_encoder_exp2"
    },
    {
      "subset": "Eng_Eth",
      "length_bin": 4,
      "source": "exp5_encoder_exp2"
    },
    {
      "subset": "Eng_Gha",
      "length_bin": 0,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Gha",
      "length_bin": 1,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Gha",
      "length_bin": 2,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Gha",
      "length_bin": 3,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Gha",
      "length_bin": 4,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Ken",
      "length_bin": 0,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Ken",
      "length_bin": 1,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Ken",
      "length_bin": 2,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Ken",
      "length_bin": 3,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Ken",
      "length_bin": 4,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Uga",
      "length_bin": 0,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Uga",
      "length_bin": 1,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Uga",
      "length_bin": 2,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Uga",
      "length_bin": 3,
      "source": "exp7_cluster"
    },
    {
      "subset": "Eng_Uga",
      "length_bin": 4,
      "source": "exp7_cluster"
    },
    {
      "subset": "Lug_Uga",
      "length_bin": 0,
      "source": "exp7_cluster"
    },
    {
      "subset": "Lug_Uga",
      "length_bin": 1,
      "source": "exp7_cluster"
    },
    {
      "subset": "Lug_Uga",
      "length_bin": 2,
      "source": "exp7_cluster"
    },
    {
      "subset": "Lug_Uga",
      "length_bin": 3,
      "source": "exp7_cluster"
    },
    {
      "subset": "Lug_Uga",
      "length_bin": 4,
      "source": "exp7_cluster"
    },
    {
      "subset": "Swa_Ken",
      "length_bin": 0,
      "source": "exp7_cluster"
    },
    {
      "subset": "Swa_Ken",
      "length_bin": 1,
      "source": "exp7_cluster"
    },
    {
      "subset": "Swa_Ken",
      "length_bin": 2,
      "source": "exp7_cluster"
    },
    {
      "subset": "Swa_Ken",
      "length_bin": 3,
      "source": "exp7_cluster"
    },
    {
      "subset": "Swa_Ken",
      "length_bin": 4,
      "source": "exp7_cluster"
    }
  ],
  "test_choice_counts": {
    "exp7_cluster": 2560,
    "exp5_encoder_exp2": 58
  },
  "diff_vs_exp7": 17,
  "notes": [
    "Exp7 validation counterpart is the closest available lightgbm_fullcap OOF source.",
    "This is a conservative exp7-base gate using only exp5 as override."
  ]
}
```

</details>

<details><summary>reports/extended_validation_source_selector/summary.json — <code>reports/extended_validation_source_selector/summary.json</code></summary>

```json
{
  "loaded_sources": [
    "current_family_meta",
    "exp3_top100",
    "exp4_pairwise",
    "exp5_encoder_exp2",
    "exp9_jina",
    "exp14_qonly",
    "exp8_ghana",
    "exp10_lug_global",
    "exp10_lug_specialized",
    "exp12_lug_merged",
    "exp13_lug_selector",
    "local_mt0_generation",
    "exp1_qa_base",
    "exp1_qa_ft",
    "exp11_bge_base_top1",
    "exp11_e5_base_top1",
    "exp11_e5_large_top1"
  ],
  "source_scores": [
    {
      "source": "current_family_meta",
      "score": 0.6258985395745735
    },
    {
      "source": "exp3_top100",
      "score": 0.5904174317680138
    },
    {
      "source": "exp4_pairwise",
      "score": 0.5712154030803913
    },
    {
      "source": "exp5_encoder_exp2",
      "score": 0.5902712632484098
    },
    {
      "source": "exp9_jina",
      "score": 0.5770043163797073
    },
    {
      "source": "exp14_qonly",
      "score": 0.5647796047735751
    },
    {
      "source": "exp8_ghana",
      "score": 0.10693814147382058
    },
    {
      "source": "exp10_lug_global",
      "score": 0.08595494009283326
    },
    {
      "source": "exp10_lug_specialized",
      "score": 0.08583540971177088
    },
    {
      "source": "exp12_lug_merged",
      "score": 0.0864893632152096
    },
    {
      "source": "exp13_lug_selector",
      "score": 0.08754675528660487
    },
    {
      "source": "local_mt0_generation",
      "score": 0.18886970512760629
    },
    {
      "source": "exp1_qa_base",
      "score": 0.445531304063734
    },
    {
      "source": "exp1_qa_ft",
      "score": 0.45403765213713065
    },
    {
      "source": "exp11_bge_base_top1",
      "score": 0.49846948014743486
    },
    {
      "source": "exp11_e5_base_top1",
      "score": 0.4875018420822146
    },
    {
      "source": "exp11_e5_large_top1",
      "score": 0.5048918014240319
    }
  ],
  "current_best": 0.6258761046358958,
  "oracle": 0.6738652677123754,
  "best": {
    "model": "ensemble",
    "score": 0.6277010239494406,
    "gain_vs_exp2": 0.038484395602626065,
    "gain_vs_current_best": 0.0018249193135447817,
    "choice_counts": {
      "exp4_pairwise": 2646,
      "current_family_meta": 1048,
      "exp3_top100": 584,
      "exp9_jina": 530,
      "exp14_qonly": 481,
      "exp5_encoder_exp2": 385,
      "exp8_ghana": 266,
      "exp1_qa_base": 180,
      "local_mt0_generation": 163,
      "exp1_qa_ft": 132,
      "exp11_e5_large_top1": 101,
      "exp11_e5_base_top1": 60,
      "exp11_bge_base_top1": 52,
      "exp13_lug_selector": 27,
      "exp12_lug_merged": 18,
      "exp10_lug_specialized": 9,
      "exp10_lug_global": 4
    },
    "per_subset": {
      "Aka_Gha": 0.330326,
      "Amh_Eth": 0.198707,
      "Eng_Eth": 0.684471,
      "Eng_Gha": 0.318674,
      "Eng_Ken": 0.868384,
      "Eng_Uga": 0.904338,
      "Lug_Uga": 0.81362,
      "Swa_Ken": 0.860331
    }
  },
  "results": [
    {
      "model": "hgb",
      "score": 0.6267725269182943,
      "gain_vs_exp2": 0.037555898571479784,
      "gain_vs_current_best": 0.0008964222823985013,
      "choice_counts": {
        "exp4_pairwise": 3597,
        "exp3_top100": 1029,
        "current_family_meta": 1004,
        "exp14_qonly": 302,
        "local_mt0_generation": 164,
        "exp9_jina": 148,
        "exp5_encoder_exp2": 79,
        "exp8_ghana": 76,
        "exp11_e5_large_top1": 72,
        "exp1_qa_base": 66,
        "exp1_qa_ft": 57,
        "exp11_bge_base_top1": 53,
        "exp11_e5_base_top1": 38,
        "exp13_lug_selector": 1
      },
      "per_subset": {
        "Aka_Gha": 0.328736,
        "Amh_Eth": 0.199166,
        "Eng_Eth": 0.685864,
        "Eng_Gha": 0.314757,
        "Eng_Ken": 0.865843,
        "Eng_Uga": 0.90423,
        "Lug_Uga": 0.812321,
        "Swa_Ken": 0.862572
      }
    },
    {
      "model": "extra_trees",
      "score": 0.6258916015796743,
      "gain_vs_exp2": 0.03667497323285973,
      "gain_vs_current_best": 1.5496943778448014e-05,
      "choice_counts": {
        "exp4_pairwise": 1481,
        "current_family_meta": 1218,
        "exp9_jina": 899,
        "exp5_encoder_exp2": 695,
        "exp14_qonly": 573,
        "exp3_top100": 533,
        "exp8_ghana": 337,
        "exp11_bge_base_top1": 186,
        "exp1_qa_base": 181,
        "exp11_e5_large_top1": 172,
        "exp1_qa_ft": 154,
        "local_mt0_generation": 111,
        "exp11_e5_base_top1": 83,
        "exp13_lug_selector": 25,
        "exp12_lug_merged": 17,
        "exp10_lug_specialized": 15,
        "exp10_lug_global": 6
      },
      "per_subset": {
        "Aka_Gha": 0.328701,
        "Amh_Eth": 0.196192,
        "Eng_Eth": 0.682694,
        "Eng_Gha": 0.319421,
        "Eng_Ken": 0.864144,
        "Eng_Uga": 0.902459,
        "Lug_Uga": 0.80937,
        "Swa_Ken": 0.85931
      }
    },
    {
      "model": "random_forest",
      "score": 0.6274743596051062,
      "gain_vs_exp2": 0.03825773125829168,
      "gain_vs_current_best": 0.0015982549692103998,
      "choice_counts": {
        "exp4_pairwise": 1252,
        "current_family_meta": 1188,
        "exp9_jina": 715,
        "exp3_top100": 668,
        "exp5_encoder_exp2": 582,
        "exp14_qonly": 543,
        "exp1_qa_base": 269,
        "exp11_e5_large_top1": 268,
        "exp8_ghana": 247,
        "exp1_qa_ft": 229,
        "local_mt0_generation": 226,
        "exp11_e5_base_top1": 191,
        "exp11_bge_base_top1": 179,
        "exp10_lug_specialized": 39,
        "exp12_lug_merged": 35,
        "exp13_lug_selector": 30,
        "exp10_lug_global": 25
      },
      "per_subset": {
        "Aka_Gha": 0.328492,
        "Amh_Eth": 0.196429,
        "Eng_Eth": 0.681976,
        "Eng_Gha": 0.320296,
        "Eng_Ken": 0.871826,
        "Eng_Uga": 0.904208,
        "Lug_Uga": 0.813883,
        "Swa_Ken": 0.86004
      }
    },
    {
      "model": "ensemble",
      "score": 0.6277010239494406,
      "gain_vs_exp2": 0.038484395602626065,
      "gain_vs_current_best": 0.0018249193135447817,
      "choice_counts": {
        "exp4_pairwise": 2646,
        "current_family_meta": 1048,
        "exp3_top100": 584,
        "exp9_jina": 530,
        "exp14_qonly": 481,
        "exp5_encoder_exp2": 385,
        "exp8_ghana": 266,
        "exp1_qa_base": 180,
        "local_mt0_generation": 163,
        "exp1_qa_ft": 132,
        "exp11_e5_large_top1": 101,
        "exp11_e5_base_top1": 60,
        "exp11_bge_base_top1": 52,
        "exp13_lug_selector": 27,
        "exp12_lug_merged": 18,
        "exp10_lug_specialized": 9,
        "exp10_lug_global": 4
      },
      "per_subset": {
        "Aka_Gha": 0.330326,
        "Amh_Eth": 0.198707,
        "Eng_Eth": 0.684471,
        "Eng_Gha": 0.318674,
        "Eng_Ken": 0.868384,
        "Eng_Uga": 0.904338,
        "Lug_Uga": 0.81362,
        "Swa_Ken": 0.860331
      }
    },
    {
      "model": "weighted_ensemble",
      "score": 0.6273651928913013,
      "gain_vs_exp2": 0.038148564544486785,
      "gain_vs_current_best": 0.0014890882554055018,
      "choice_counts": {
        "exp4_pairwise": 3324,
        "current_family_meta": 1031,
        "exp3_top100": 482,
        "exp14_qonly": 373,
        "exp9_jina": 330,
        "exp5_encoder_exp2": 270,
        "exp8_ghana": 257,
        "local_mt0_generation": 154,
        "exp1_qa_base": 149,
        "exp1_qa_ft": 97,
        "exp11_e5_large_top1": 85,
        "exp11_e5_base_top1": 54,
        "exp11_bge_base_top1": 31,
        "exp13_lug_selector": 25,
        "exp12_lug_merged": 15,
        "exp10_lug_specialized": 6,
        "exp10_lug_global": 3
      },
      "per_subset": {
        "Aka_Gha": 0.330038,
        "Amh_Eth": 0.198803,
        "Eng_Eth": 0.68489,
        "Eng_Gha": 0.317908,
        "Eng_Ken": 0.868384,
        "Eng_Uga": 0.90399,
        "Lug_Uga": 0.813693,
        "Swa_Ken": 0.85872
      }
    }
  ],
  "notes": [
    "This includes already-local validation artifacts that may not yet have mirrored test predictions.",
    "No Modal compute was used."
  ]
}
```

</details>

<details><summary>reports/family_meta_miss_correction/summary.json — <code>reports/family_meta_miss_correction/summary.json</code></summary>

```json
{
  "current_best": 0.6258761046358958,
  "family_oracle": 0.6338465263448811,
  "oracle_gap_remaining": 0.00797042170898532,
  "miss_rate": 0.09303021238408615,
  "miss_summary_by_subset": [
    {
      "subset": "Eng_Gha",
      "rows": 1104,
      "base": 0.31024410250325724,
      "oracle": 0.3227760801747575,
      "oracle_gain": 0.012531977671500294,
      "switch_rate": 0.17391304347826086
    },
    {
      "subset": "Amh_Eth",
      "rows": 462,
      "base": 0.19734811740192876,
      "oracle": 0.20875587024405204,
      "oracle_gain": 0.011407752842123255,
      "switch_rate": 0.13852813852813853
    },
    {
      "subset": "Eng_Eth",
      "rows": 564,
      "base": 0.687572050671966,
      "oracle": 0.6971816547139634,
      "oracle_gain": 0.009609604041997458,
      "switch_rate": 0.07092198581560284
    },
    {
      "subset": "Swa_Ken",
      "rows": 518,
      "base": 0.8626808073933665,
      "oracle": 0.8717965639946128,
      "oracle_gain": 0.009115756601246309,
      "switch_rate": 0.03861003861003861
    },
    {
      "subset": "Lug_Uga",
      "rows": 846,
      "base": 0.8099479551374374,
      "oracle": 0.818462957764543,
      "oracle_gain": 0.0085150026271056,
      "switch_rate": 0.05319148936170213
    },
    {
      "subset": "Aka_Gha",
      "rows": 1114,
      "base": 0.32889562843420017,
      "oracle": 0.33644289271616257,
      "oracle_gain": 0.0075472642819623915,
      "switch_rate": 0.1696588868940754
    },
    {
      "subset": "Eng_Uga",
      "rows": 1688,
      "base": 0.9044176795423533,
      "oracle": 0.9085791936395677,
      "oracle_gain": 0.0041615140972143414,
      "switch_rate": 0.031990521327014215
    },
    {
      "subset": "Eng_Ken",
      "rows": 390,
      "base": 0.8666699440874253,
      "oracle": 0.8702771116669413,
      "oracle_gain": 0.0036071675795161474,
      "switch_rate": 0.046153846153846156
    }
  ],
  "best": {
    "model": "extra_trees_switch_proba",
    "mode": "switch_to_oracle_diagnostic",
    "threshold_quantile": 0.6,
    "threshold": 0.07400687783956528,
    "score": 0.6338465263448811,
    "gain_vs_current_best": 0.00797042170898532,
    "gain_vs_exp2": 0.0446298979980666,
    "switch_rate": 0.3999401734968591,
    "switch_count": 2674,
    "auc": 0.9338606312940637,
    "per_subset": {
      "Aka_Gha": 0.336443,
      "Amh_Eth": 0.208756,
      "Eng_Eth": 0.697182,
      "Eng_Gha": 0.322776,
      "Eng_Ken": 0.870277,
      "Eng_Uga": 0.908579,
      "Lug_Uga": 0.818463,
      "Swa_Ken": 0.871797
    }
  },
  "notes": [
    "switch_to_pred_family is deployable-style and only switches to the model-predicted best family.",
    "switch_to_oracle_diagnostic is not deployable; it tests whether switch detection is useful if target family were known."
  ]
}
```

</details>

<details><summary>reports/fullcap_vs_noleak_mining/summary.json — <code>reports/fullcap_vs_noleak_mining/summary.json</code></summary>

```json
{
  "fullcap": 0.6353457041144692,
  "noleak": 0.5646714819082213,
  "gain": 0.070674222206248,
  "same_answer_rate": 0.6114268620999103,
  "changed_answer_rate": 0.38857313790008974,
  "largest_subset_gains": [
    {
      "subset": "Lug_Uga",
      "rows": 846,
      "fullcap": 0.8388260822890731,
      "noleak": 0.6164620386247086,
      "gain": 0.2223640436643643,
      "changed": 0.5047281323877069
    },
    {
      "subset": "Eng_Ken",
      "rows": 390,
      "fullcap": 0.8893843099255634,
      "noleak": 0.7985629536439506,
      "gain": 0.09082135628161284,
      "changed": 0.2205128205128205
    },
    {
      "subset": "Eng_Uga",
      "rows": 1688,
      "fullcap": 0.9166767035428719,
      "noleak": 0.8320124718808536,
      "gain": 0.08466423166201834,
      "changed": 0.22867298578199047
    },
    {
      "subset": "Swa_Ken",
      "rows": 518,
      "fullcap": 0.8773453144508172,
      "noleak": 0.8006533589071095,
      "gain": 0.07669195554370768,
      "changed": 0.20463320463320467
    },
    {
      "subset": "Eng_Eth",
      "rows": 564,
      "fullcap": 0.6734334319242876,
      "noleak": 0.6402864287584216,
      "gain": 0.033147003165866,
      "changed": 0.34397163120567376
    },
    {
      "subset": "Aka_Gha",
      "rows": 1114,
      "fullcap": 0.3357867952203368,
      "noleak": 0.31645299033045443,
      "gain": 0.019333804889882292,
      "changed": 0.63016157989228
    },
    {
      "subset": "Eng_Gha",
      "rows": 1104,
      "fullcap": 0.3148184095729589,
      "noleak": 0.29720962329437267,
      "gain": 0.017608786278586242,
      "changed": 0.447463768115942
    },
    {
      "subset": "Amh_Eth",
      "rows": 462,
      "fullcap": 0.19081604568533517,
      "noleak": 0.17636796066470992,
      "gain": 0.014448085020625263,
      "changed": 0.43939393939393945
    }
  ],
  "notes": [
    "Fullcap uses validation reference length, so this is a diagnostic report only.",
    "Rows with large fullcap_gain_vs_noleak show where legal proxy features should be targeted."
  ]
}
```

</details>

<details><summary>reports/local_candidate_regressor_submission/summary.json — <code>reports/local_candidate_regressor_submission/summary.json</code></summary>

```json
{
  "metrics": {
    "rouge1": 0.592474046904811,
    "per_subset": {
      "Aka_Gha": 0.322646,
      "Amh_Eth": 0.19166,
      "Eng_Eth": 0.669888,
      "Eng_Gha": 0.30663,
      "Eng_Ken": 0.823142,
      "Eng_Uga": 0.86516,
      "Lug_Uga": 0.691886,
      "Swa_Ken": 0.830541
    },
    "gain_vs_rerank": 0.0032574185579964388,
    "best_margin_gate": {
      "pred_margin_threshold": 0.01,
      "score": 0.592785038263852,
      "gain_vs_rerank": 0.003854585699819224,
      "switch_n": 1501.0,
      "mean_actual_gain_switched": 0.01716972684143335
    },
    "baseline": {
      "top1": 0.5395458277838778,
      "rerank": 0.5892166283468145
    }
  },
  "model_params": {
    "max_iter": 360,
    "learning_rate": 0.035,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.04,
    "random_state": 24
  },
  "submission": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_submissions\\submission_local_candidate_regressor_hgb.csv",
  "changed_vs_exp6": 340,
  "test_gate_switch_n": 340
}
```

</details>

<details><summary>reports/local_no_modal_selector_v2/summary.json — <code>reports/local_no_modal_selector_v2/summary.json</code></summary>

```json
{
  "base_exp2_rerank_val_r1": 0.5892166283468145,
  "best_gate": {
    "pred_t": 0.48,
    "max_rank": 5.0,
    "min_sources": 4.0,
    "require_exp2_pool": 1.0,
    "switch_n": 1030.0,
    "score": 0.5889455275749736,
    "gain": -0.0002711007718408931,
    "mean_actual_gain_on_switched": -0.00175978617526979
  },
  "made_submission": null,
  "diff_vs_exp6_reranker": null,
  "diff_vs_exp7_cluster": null
}
```

</details>

<details><summary>reports/predicted_length_cluster_selector/summary.json — <code>reports/predicted_length_cluster_selector/summary.json</code></summary>

```json
{
  "length_metrics": {
    "val_length_mae_tokens": 31.63275166055124,
    "val_length_mape": 0.5308944949809483
  },
  "oracle": 0.6931894925956986,
  "rrf_baseline": 0.3641091015889087,
  "best": {
    "model": "ensemble",
    "score": 0.5675436634843569,
    "rmse": 0.08737003972008178,
    "chosen_rows": 6686,
    "per_subset": {
      "Aka_Gha": 0.318454,
      "Amh_Eth": 0.188334,
      "Eng_Eth": 0.637917,
      "Eng_Gha": 0.300893,
      "Eng_Ken": 0.804893,
      "Eng_Uga": 0.82876,
      "Lug_Uga": 0.628028,
      "Swa_Ken": 0.804422
    }
  },
  "results": [
    {
      "model": "hgb",
      "score": 0.5652962836774692,
      "rmse": 0.08791255762182762,
      "chosen_rows": 6686,
      "per_subset": {
        "Aka_Gha": 0.316546,
        "Amh_Eth": 0.185692,
        "Eng_Eth": 0.635636,
        "Eng_Gha": 0.298593,
        "Eng_Ken": 0.800917,
        "Eng_Uga": 0.830173,
        "Lug_Uga": 0.623169,
        "Swa_Ken": 0.795584
      }
    },
    {
      "model": "extra_trees",
      "score": 0.5650062954276893,
      "rmse": 0.08826405255958449,
      "chosen_rows": 6686,
      "per_subset": {
        "Aka_Gha": 0.319297,
        "Amh_Eth": 0.184063,
        "Eng_Eth": 0.64497,
        "Eng_Gha": 0.302403,
        "Eng_Ken": 0.802299,
        "Eng_Uga": 0.821803,
        "Lug_Uga": 0.618354,
        "Swa_Ken": 0.803193
      }
    },
    {
      "model": "ensemble",
      "score": 0.5675436634843569,
      "rmse": 0.08737003972008178,
      "chosen_rows": 6686,
      "per_subset": {
        "Aka_Gha": 0.318454,
        "Amh_Eth": 0.188334,
        "Eng_Eth": 0.637917,
        "Eng_Gha": 0.300893,
        "Eng_Ken": 0.804893,
        "Eng_Uga": 0.82876,
        "Lug_Uga": 0.628028,
        "Swa_Ken": 0.804422
      }
    }
  ],
  "comparison": {
    "exp2_rerank": 0.5892166283468145,
    "cluster_noleak_hgb": 0.5646714819082213,
    "cluster_fullcap_leaky_hgb": 0.6353457041144692
  }
}
```

</details>

<details><summary>reports/residual_clean_override_selector/summary.json — <code>reports/residual_clean_override_selector/summary.json</code></summary>

```json
{
  "baseline_exp2": 0.5892166283468145,
  "base_score": 0.6234960658361287,
  "clean_oracle": 0.6479322858935932,
  "best": {
    "model": "extra_trees_pred_gain",
    "scope": "sweep",
    "threshold": 3.2490423564013327e-05,
    "score": 0.6237821868130333,
    "gain_vs_exp2": 0.03456555846621878,
    "gain_vs_base": 0.0002861209769046136,
    "override_rate": 0.12503739156446306,
    "override_counts": {
      "exp2_rerank": 527,
      "cluster_fast": 189,
      "exp2_top1": 78,
      "cluster_noleak": 42
    },
    "per_subset": {
      "Aka_Gha": 0.327202,
      "Amh_Eth": 0.196744,
      "Eng_Eth": 0.686277,
      "Eng_Gha": 0.310199,
      "Eng_Ken": 0.861839,
      "Eng_Uga": 0.902542,
      "Lug_Uga": 0.808277,
      "Swa_Ken": 0.853817
    }
  },
  "notes": [
    "Zero scope is the cleaner calibrated read.",
    "Sweep scope is diagnostic and may be optimistic because threshold is selected on OOF validation predictions."
  ]
}
```

</details>

<details><summary>reports/rich_clean_meta_selector/summary.json — <code>reports/rich_clean_meta_selector/summary.json</code></summary>

```json
{
  "baseline_exp2": 0.5892166283468145,
  "baseline_hybrid": 0.6065232342601322,
  "clean_oracle": 0.6479322858935932,
  "best": {
    "model": "ensemble",
    "score": 0.6227089712249555,
    "gain_vs_exp2": 0.03349234287814096,
    "gain_vs_hybrid": 0.016185736964823283,
    "choice_counts": {
      "exp2_rerank": 2875,
      "cluster_noleak": 1667,
      "cluster_fast": 1323,
      "exp2_top1": 821
    },
    "per_subset": {
      "Aka_Gha": 0.327302,
      "Amh_Eth": 0.191737,
      "Eng_Eth": 0.686277,
      "Eng_Gha": 0.307692,
      "Eng_Ken": 0.861839,
      "Eng_Uga": 0.902542,
      "Lug_Uga": 0.805827,
      "Swa_Ken": 0.85356
    }
  },
  "results": [
    {
      "model": "hgb",
      "score": 0.6221661912064367,
      "gain_vs_exp2": 0.03294956285962214,
      "gain_vs_hybrid": 0.015642956946304465,
      "choice_counts": {
        "exp2_rerank": 2361,
        "exp2_top1": 2235,
        "cluster_fast": 1274,
        "cluster_noleak": 816
      },
      "per_subset": {
        "Aka_Gha": 0.32519,
        "Amh_Eth": 0.193066,
        "Eng_Eth": 0.687772,
        "Eng_Gha": 0.305415,
        "Eng_Ken": 0.862785,
        "Eng_Uga": 0.90315,
        "Lug_Uga": 0.805929,
        "Swa_Ken": 0.85028
      }
    },
    {
      "model": "extra_trees",
      "score": 0.6190059358902364,
      "gain_vs_exp2": 0.029789307543421906,
      "gain_vs_hybrid": 0.012482701630104232,
      "choice_counts": {
        "exp2_rerank": 2855,
        "cluster_noleak": 1654,
        "cluster_fast": 1373,
        "exp2_top1": 804
      },
      "per_subset": {
        "Aka_Gha": 0.324405,
        "Amh_Eth": 0.18765,
        "Eng_Eth": 0.681555,
        "Eng_Gha": 0.307361,
        "Eng_Ken": 0.862271,
        "Eng_Uga": 0.898101,
        "Lug_Uga": 0.800631,
        "Swa_Ken": 0.844119
      }
    },
    {
      "model": "random_forest",
      "score": 0.6213007924297795,
      "gain_vs_exp2": 0.032084164082964994,
      "gain_vs_hybrid": 0.01477755816964732,
      "choice_counts": {
        "exp2_rerank": 2736,
        "cluster_noleak": 1602,
        "cluster_fast": 1279,
        "exp2_top1": 1069
      },
      "per_subset": {
        "Aka_Gha": 0.326695,
        "Amh_Eth": 0.19181,
        "Eng_Eth": 0.684618,
        "Eng_Gha": 0.308225,
        "Eng_Ken": 0.861556,
        "Eng_Uga": 0.902806,
        "Lug_Uga": 0.799035,
        "Swa_Ken": 0.847742
      }
    },
    {
      "model": "ensemble",
      "score": 0.6227089712249555,
      "gain_vs_exp2": 0.03349234287814096,
      "gain_vs_hybrid": 0.016185736964823283,
      "choice_counts": {
        "exp2_rerank": 2875,
        "cluster_noleak": 1667,
        "cluster_fast": 1323,
        "exp2_top1": 821
      },
      "per_subset": {
        "Aka_Gha": 0.327302,
        "Amh_Eth": 0.191737,
        "Eng_Eth": 0.686277,
        "Eng_Gha": 0.307692,
        "Eng_Ken": 0.861839,
        "Eng_Uga": 0.902542,
        "Lug_Uga": 0.805827,
        "Swa_Ken": 0.85356
      }
    },
    {
      "model": "weighted_ensemble",
      "score": 0.6226594566933394,
      "gain_vs_exp2": 0.0334428283465249,
      "gain_vs_hybrid": 0.016136222433207226,
      "choice_counts": {
        "exp2_rerank": 2835,
        "cluster_noleak": 1656,
        "cluster_fast": 1375,
        "exp2_top1": 820
      },
      "per_subset": {
        "Aka_Gha": 0.326495,
        "Amh_Eth": 0.193702,
        "Eng_Eth": 0.687655,
        "Eng_Gha": 0.306398,
        "Eng_Ken": 0.865303,
        "Eng_Uga": 0.903149,
        "Lug_Uga": 0.805651,
        "Swa_Ken": 0.849868
      }
    }
  ],
  "notes": [
    "All features are drawn from clean deployable sources and OOF validation predictions.",
    "No validation reference-length or target columns are used as features."
  ]
}
```

</details>

<details><summary>reports/rich_clean_meta_selector_target_encoded/summary.json — <code>reports/rich_clean_meta_selector_target_encoded/summary.json</code></summary>

```json
{
  "baseline_exp2": 0.5892166283468145,
  "plain_rich_best": 0.6227089712249555,
  "clean_oracle": 0.6479322858935932,
  "best": {
    "model": "weighted_ensemble_te",
    "score": 0.622026867988561,
    "gain_vs_exp2": 0.03281023964174645,
    "gain_vs_plain_rich": -0.0006821032363945045,
    "choice_counts": {
      "exp2_rerank": 2968,
      "cluster_fast": 1490,
      "cluster_noleak": 1459,
      "exp2_top1": 769
    },
    "per_subset": {
      "Aka_Gha": 0.325273,
      "Amh_Eth": 0.193576,
      "Eng_Eth": 0.684562,
      "Eng_Gha": 0.305255,
      "Eng_Ken": 0.8619,
      "Eng_Uga": 0.904307,
      "Lug_Uga": 0.802037,
      "Swa_Ken": 0.854934
    }
  },
  "results": [
    {
      "model": "hgb_te",
      "score": 0.6213138648872539,
      "gain_vs_exp2": 0.03209723654043939,
      "gain_vs_plain_rich": -0.0013951063377015682,
      "choice_counts": {
        "exp2_rerank": 3045,
        "cluster_fast": 1660,
        "exp2_top1": 1028,
        "cluster_noleak": 953
      },
      "per_subset": {
        "Aka_Gha": 0.324759,
        "Amh_Eth": 0.192334,
        "Eng_Eth": 0.683993,
        "Eng_Gha": 0.304143,
        "Eng_Ken": 0.862603,
        "Eng_Uga": 0.903492,
        "Lug_Uga": 0.802653,
        "Swa_Ken": 0.852057
      }
    },
    {
      "model": "extra_trees_te",
      "score": 0.6191689594223867,
      "gain_vs_exp2": 0.02995233107557216,
      "gain_vs_plain_rich": -0.0035400118025687988,
      "choice_counts": {
        "exp2_rerank": 2929,
        "cluster_noleak": 1567,
        "cluster_fast": 1358,
        "exp2_top1": 832
      },
      "per_subset": {
        "Aka_Gha": 0.325104,
        "Amh_Eth": 0.192243,
        "Eng_Eth": 0.682411,
        "Eng_Gha": 0.30711,
        "Eng_Ken": 0.85903,
        "Eng_Uga": 0.898045,
        "Lug_Uga": 0.796691,
        "Swa_Ken": 0.849286
      }
    },
    {
      "model": "random_forest_te",
      "score": 0.6200041503065354,
      "gain_vs_exp2": 0.030787521959720898,
      "gain_vs_plain_rich": -0.002704820918420059,
      "choice_counts": {
        "exp2_rerank": 2774,
        "cluster_noleak": 1525,
        "cluster_fast": 1455,
        "exp2_top1": 932
      },
      "per_subset": {
        "Aka_Gha": 0.325151,
        "Amh_Eth": 0.192999,
        "Eng_Eth": 0.683696,
        "Eng_Gha": 0.305966,
        "Eng_Ken": 0.858728,
        "Eng_Uga": 0.900898,
        "Lug_Uga": 0.798248,
        "Swa_Ken": 0.848716
      }
    },
    {
      "model": "ensemble_te",
      "score": 0.6220058317959013,
      "gain_vs_exp2": 0.03278920344908676,
      "gain_vs_plain_rich": -0.0007031394290542003,
      "choice_counts": {
        "exp2_rerank": 2914,
        "cluster_noleak": 1531,
        "cluster_fast": 1471,
        "exp2_top1": 770
      },
      "per_subset": {
        "Aka_Gha": 0.325553,
        "Amh_Eth": 0.193343,
        "Eng_Eth": 0.685234,
        "Eng_Gha": 0.306371,
        "Eng_Ken": 0.863559,
        "Eng_Uga": 0.904243,
        "Lug_Uga": 0.800083,
        "Swa_Ken": 0.853312
      }
    },
    {
      "model": "weighted_ensemble_te",
      "score": 0.622026867988561,
      "gain_vs_exp2": 0.03281023964174645,
      "gain_vs_plain_rich": -0.0006821032363945045,
      "choice_counts": {
        "exp2_rerank": 2968,
        "cluster_fast": 1490,
        "cluster_noleak": 1459,
        "exp2_top1": 769
      },
      "per_subset": {
        "Aka_Gha": 0.325273,
        "Amh_Eth": 0.193576,
        "Eng_Eth": 0.684562,
        "Eng_Gha": 0.305255,
        "Eng_Ken": 0.8619,
        "Eng_Uga": 0.904307,
        "Lug_Uga": 0.802037,
        "Swa_Ken": 0.854934
      }
    }
  ],
  "notes": [
    "Target encodings are fold-safe: validation fold encodings are computed from training IDs only.",
    "No target/reference-derived features are used at prediction time."
  ]
}
```

</details>

<details><summary>reports/rich_clean_meta_with_regressor/summary.json — <code>reports/rich_clean_meta_with_regressor/summary.json</code></summary>

```json
{
  "baseline_exp2": 0.5892166283468145,
  "plain_rich_best": 0.6227089712249555,
  "oracle_with_regressor": 0.6502489146304556,
  "regressor_source_score": 0.592474046904811,
  "best": {
    "model": "hgb",
    "score": 0.613606420275577,
    "gain_vs_exp2": 0.02438979192876245,
    "gain_vs_plain_rich": -0.009102550949378507,
    "choice_counts": {
      "local_candidate_regressor": 2245,
      "exp2_rerank": 1846,
      "cluster_fast": 1230,
      "exp2_top1": 765,
      "cluster_noleak": 600
    },
    "per_subset": {
      "Aka_Gha": 0.323931,
      "Amh_Eth": 0.191402,
      "Eng_Eth": 0.682935,
      "Eng_Gha": 0.307697,
      "Eng_Ken": 0.852958,
      "Eng_Uga": 0.888439,
      "Lug_Uga": 0.779547,
      "Swa_Ken": 0.842812
    }
  },
  "results": [
    {
      "model": "hgb",
      "score": 0.613606420275577,
      "gain_vs_exp2": 0.02438979192876245,
      "gain_vs_plain_rich": -0.009102550949378507,
      "choice_counts": {
        "local_candidate_regressor": 2245,
        "exp2_rerank": 1846,
        "cluster_fast": 1230,
        "exp2_top1": 765,
        "cluster_noleak": 600
      },
      "per_subset": {
        "Aka_Gha": 0.323931,
        "Amh_Eth": 0.191402,
        "Eng_Eth": 0.682935,
        "Eng_Gha": 0.307697,
        "Eng_Ken": 0.852958,
        "Eng_Uga": 0.888439,
        "Lug_Uga": 0.779547,
        "Swa_Ken": 0.842812
      }
    },
    {
      "model": "extra_trees",
      "score": 0.6028440053895535,
      "gain_vs_exp2": 0.013627377042738975,
      "gain_vs_plain_rich": -0.019864965835401982,
      "choice_counts": {
        "local_candidate_regressor": 2853,
        "exp2_rerank": 1439,
        "cluster_noleak": 1010,
        "exp2_top1": 756,
        "cluster_fast": 628
      },
      "per_subset": {
        "Aka_Gha": 0.324413,
        "Amh_Eth": 0.194604,
        "Eng_Eth": 0.674666,
        "Eng_Gha": 0.309455,
        "Eng_Ken": 0.832688,
        "Eng_Uga": 0.872573,
        "Lug_Uga": 0.742054,
        "Swa_Ken": 0.833463
      }
    },
    {
      "model": "random_forest",
      "score": 0.6058487498293046,
      "gain_vs_exp2": 0.016632121482490048,
      "gain_vs_plain_rich": -0.01686022139565091,
      "choice_counts": {
        "local_candidate_regressor": 2695,
        "exp2_rerank": 1405,
        "cluster_noleak": 969,
        "cluster_fast": 851,
        "exp2_top1": 766
      },
      "per_subset": {
        "Aka_Gha": 0.32592,
        "Amh_Eth": 0.192804,
        "Eng_Eth": 0.67564,
        "Eng_Gha": 0.309842,
        "Eng_Ken": 0.840845,
        "Eng_Uga": 0.875065,
        "Lug_Uga": 0.749879,
        "Swa_Ken": 0.841682
      }
    },
    {
      "model": "ensemble",
      "score": 0.6082279636855679,
      "gain_vs_exp2": 0.01901133533875332,
      "gain_vs_plain_rich": -0.014481007539387636,
      "choice_counts": {
        "local_candidate_regressor": 2628,
        "exp2_rerank": 1696,
        "cluster_noleak": 919,
        "cluster_fast": 854,
        "exp2_top1": 589
      },
      "per_subset": {
        "Aka_Gha": 0.324815,
        "Amh_Eth": 0.194408,
        "Eng_Eth": 0.67926,
        "Eng_Gha": 0.30865,
        "Eng_Ken": 0.841122,
        "Eng_Uga": 0.879708,
        "Lug_Uga": 0.759066,
        "Swa_Ken": 0.841591
      }
    },
    {
      "model": "weighted_ensemble",
      "score": 0.6092445904374335,
      "gain_vs_exp2": 0.020027962090618967,
      "gain_vs_plain_rich": -0.01346438078752199,
      "choice_counts": {
        "local_candidate_regressor": 2537,
        "exp2_rerank": 1819,
        "cluster_fast": 937,
        "cluster_noleak": 859,
        "exp2_top1": 534
      },
      "per_subset": {
        "Aka_Gha": 0.32437,
        "Amh_Eth": 0.193577,
        "Eng_Eth": 0.681325,
        "Eng_Gha": 0.308262,
        "Eng_Ken": 0.842554,
        "Eng_Uga": 0.881949,
        "Lug_Uga": 0.762663,
        "Swa_Ken": 0.840737
      }
    }
  ],
  "notes": [
    "The local candidate regressor validation source has target scores and ranks but no answer text in its val choice file.",
    "It is deployable on test because test_candidate_regressor_choices.csv contains candidate_answer."
  ]
}
```

</details>

<details><summary>reports/selector_family_classifier/summary.json — <code>reports/selector_family_classifier/summary.json</code></summary>

```json
{
  "family_oracle": 0.6338465263448813,
  "current_best": 0.6258761046358958,
  "best": {
    "model": "hgb",
    "score": 0.6237827026919157,
    "gain_vs_exp2": 0.03456607434510117,
    "gain_vs_family_meta_subset_hybrid": -0.002093401943980111,
    "accuracy": 0.8991923422075979,
    "choice_counts": {
      "meta_gate": 6224,
      "rich_meta": 200,
      "fast_gate": 162,
      "global_clean": 83,
      "subset_clean": 13,
      "residual": 4
    },
    "per_subset": {
      "Aka_Gha": 0.326431,
      "Amh_Eth": 0.197009,
      "Eng_Eth": 0.687128,
      "Eng_Gha": 0.308604,
      "Eng_Ken": 0.85699,
      "Eng_Uga": 0.903027,
      "Lug_Uga": 0.810836,
      "Swa_Ken": 0.855613
    }
  },
  "results": [
    {
      "model": "hgb",
      "score": 0.6237827026919157,
      "gain_vs_exp2": 0.03456607434510117,
      "gain_vs_family_meta_subset_hybrid": -0.002093401943980111,
      "accuracy": 0.8991923422075979,
      "choice_counts": {
        "meta_gate": 6224,
        "rich_meta": 200,
        "fast_gate": 162,
        "global_clean": 83,
        "subset_clean": 13,
        "residual": 4
      },
      "per_subset": {
        "Aka_Gha": 0.326431,
        "Amh_Eth": 0.197009,
        "Eng_Eth": 0.687128,
        "Eng_Gha": 0.308604,
        "Eng_Ken": 0.85699,
        "Eng_Uga": 0.903027,
        "Lug_Uga": 0.810836,
        "Swa_Ken": 0.855613
      }
    },
    {
      "model": "extra_trees",
      "score": 0.6073013809165413,
      "gain_vs_exp2": 0.018084752569726747,
      "gain_vs_family_meta_subset_hybrid": -0.018574723719354536,
      "accuracy": 0.8544720311097816,
      "choice_counts": {
        "meta_gate": 5118,
        "fast_gate": 487,
        "rich_meta": 473,
        "global_clean": 386,
        "subset_clean": 195,
        "residual": 27
      },
      "per_subset": {
        "Aka_Gha": 0.323249,
        "Amh_Eth": 0.185497,
        "Eng_Eth": 0.678086,
        "Eng_Gha": 0.305011,
        "Eng_Ken": 0.818883,
        "Eng_Uga": 0.883169,
        "Lug_Uga": 0.768341,
        "Swa_Ken": 0.840299
      }
    },
    {
      "model": "random_forest",
      "score": 0.6129663872728095,
      "gain_vs_exp2": 0.023749758925994957,
      "gain_vs_family_meta_subset_hybrid": -0.012909717363086326,
      "accuracy": 0.8629973078073586,
      "choice_counts": {
        "meta_gate": 5190,
        "fast_gate": 502,
        "rich_meta": 476,
        "global_clean": 364,
        "subset_clean": 133,
        "residual": 21
      },
      "per_subset": {
        "Aka_Gha": 0.323571,
        "Amh_Eth": 0.185374,
        "Eng_Eth": 0.68177,
        "Eng_Gha": 0.305642,
        "Eng_Ken": 0.838543,
        "Eng_Uga": 0.891721,
        "Lug_Uga": 0.782026,
        "Swa_Ken": 0.84246
      }
    },
    {
      "model": "vote_ensemble",
      "score": 0.6130855243921589,
      "gain_vs_exp2": 0.023868896045344368,
      "gain_vs_family_meta_subset_hybrid": -0.012790580243736915,
      "accuracy": 0.8640442716123242,
      "choice_counts": {
        "meta_gate": 5222,
        "fast_gate": 483,
        "rich_meta": 468,
        "global_clean": 361,
        "subset_clean": 131,
        "residual": 21
      },
      "per_subset": {
        "Aka_Gha": 0.323387,
        "Amh_Eth": 0.185456,
        "Eng_Eth": 0.681734,
        "Eng_Gha": 0.305545,
        "Eng_Ken": 0.837801,
        "Eng_Uga": 0.892089,
        "Lug_Uga": 0.782923,
        "Swa_Ken": 0.84246
      }
    }
  ],
  "notes": [
    "This predicts the oracle selector family directly.",
    "Scores are computed from held-out/OFF family outputs; no reference text is used as a prediction feature."
  ]
}
```

</details>

<details><summary>reports/selector_family_hybrid/summary.json — <code>reports/selector_family_hybrid/summary.json</code></summary>

```json
{
  "score": 0.6238405650326898,
  "gain_vs_exp2": 0.03462393668587527,
  "per_subset_selector": [
    {
      "subset": "Aka_Gha",
      "selector": "meta_gate",
      "score": 0.32730234892787186
    },
    {
      "subset": "Amh_Eth",
      "selector": "meta_gate",
      "score": 0.19734811740192876
    },
    {
      "subset": "Eng_Eth",
      "selector": "meta_gate",
      "score": 0.6862770745957062
    },
    {
      "subset": "Eng_Gha",
      "selector": "residual",
      "score": 0.31019895653540625
    },
    {
      "subset": "Eng_Ken",
      "selector": "meta_gate",
      "score": 0.8618394590310063
    },
    {
      "subset": "Eng_Uga",
      "selector": "meta_gate",
      "score": 0.9025423011566358
    },
    {
      "subset": "Lug_Uga",
      "selector": "residual",
      "score": 0.8082768917628919
    },
    {
      "subset": "Swa_Ken",
      "selector": "residual",
      "score": 0.8538173918344738
    }
  ]
}
```

</details>

<details><summary>reports/selector_family_meta_learner/summary.json — <code>reports/selector_family_meta_learner/summary.json</code></summary>

```json
{
  "best_family_hybrid": 0.6238405650326898,
  "family_oracle": 0.6338465263448811,
  "best": {
    "model": "ensemble",
    "score": 0.6258109015467073,
    "gain_vs_exp2": 0.036594273199892724,
    "gain_vs_best_family_hybrid": 0.0019703365140174567,
    "choice_counts": {
      "rich_meta": 1366,
      "meta_gate": 1286,
      "residual": 1216,
      "fast_gate": 1158,
      "subset_clean": 847,
      "global_clean": 813
    },
    "per_subset": {
      "Aka_Gha": 0.328896,
      "Amh_Eth": 0.196405,
      "Eng_Eth": 0.687572,
      "Eng_Gha": 0.310244,
      "Eng_Ken": 0.86667,
      "Eng_Uga": 0.904418,
      "Lug_Uga": 0.809948,
      "Swa_Ken": 0.862681
    }
  },
  "results": [
    {
      "model": "hgb",
      "score": 0.6251550358998074,
      "gain_vs_exp2": 0.035938407552992824,
      "gain_vs_best_family_hybrid": 0.0013144708671175565,
      "choice_counts": {
        "meta_gate": 6330,
        "fast_gate": 189,
        "rich_meta": 144,
        "global_clean": 16,
        "residual": 4,
        "subset_clean": 3
      },
      "per_subset": {
        "Aka_Gha": 0.328327,
        "Amh_Eth": 0.196713,
        "Eng_Eth": 0.686582,
        "Eng_Gha": 0.309615,
        "Eng_Ken": 0.864379,
        "Eng_Uga": 0.903975,
        "Lug_Uga": 0.810072,
        "Swa_Ken": 0.860546
      }
    },
    {
      "model": "extra_trees",
      "score": 0.6247105984503492,
      "gain_vs_exp2": 0.03549397010353461,
      "gain_vs_best_family_hybrid": 0.0008700334176593438,
      "choice_counts": {
        "rich_meta": 1404,
        "meta_gate": 1262,
        "residual": 1190,
        "fast_gate": 1133,
        "global_clean": 859,
        "subset_clean": 838
      },
      "per_subset": {
        "Aka_Gha": 0.327179,
        "Amh_Eth": 0.196449,
        "Eng_Eth": 0.686765,
        "Eng_Gha": 0.310166,
        "Eng_Ken": 0.861601,
        "Eng_Uga": 0.902792,
        "Lug_Uga": 0.810103,
        "Swa_Ken": 0.862038
      }
    },
    {
      "model": "random_forest",
      "score": 0.6249738706870498,
      "gain_vs_exp2": 0.03575724234023525,
      "gain_vs_best_family_hybrid": 0.001133305654359984,
      "choice_counts": {
        "meta_gate": 3227,
        "rich_meta": 1041,
        "fast_gate": 760,
        "residual": 630,
        "subset_clean": 548,
        "global_clean": 480
      },
      "per_subset": {
        "Aka_Gha": 0.32828,
        "Amh_Eth": 0.196099,
        "Eng_Eth": 0.686618,
        "Eng_Gha": 0.310405,
        "Eng_Ken": 0.863961,
        "Eng_Uga": 0.902887,
        "Lug_Uga": 0.809433,
        "Swa_Ken": 0.862038
      }
    },
    {
      "model": "ensemble",
      "score": 0.6258109015467073,
      "gain_vs_exp2": 0.036594273199892724,
      "gain_vs_best_family_hybrid": 0.0019703365140174567,
      "choice_counts": {
        "rich_meta": 1366,
        "meta_gate": 1286,
        "residual": 1216,
        "fast_gate": 1158,
        "subset_clean": 847,
        "global_clean": 813
      },
      "per_subset": {
        "Aka_Gha": 0.328896,
        "Amh_Eth": 0.196405,
        "Eng_Eth": 0.687572,
        "Eng_Gha": 0.310244,
        "Eng_Ken": 0.86667,
        "Eng_Uga": 0.904418,
        "Lug_Uga": 0.809948,
        "Swa_Ken": 0.862681
      }
    },
    {
      "model": "weighted_ensemble",
      "score": 0.6256653228585802,
      "gain_vs_exp2": 0.03644869451176569,
      "gain_vs_best_family_hybrid": 0.0018247578258904218,
      "choice_counts": {
        "rich_meta": 1374,
        "meta_gate": 1286,
        "residual": 1203,
        "fast_gate": 1167,
        "global_clean": 832,
        "subset_clean": 824
      },
      "per_subset": {
        "Aka_Gha": 0.328303,
        "Amh_Eth": 0.197358,
        "Eng_Eth": 0.6878,
        "Eng_Gha": 0.309972,
        "Eng_Ken": 0.866359,
        "Eng_Uga": 0.904584,
        "Lug_Uga": 0.810049,
        "Swa_Ken": 0.861084
      }
    }
  ],
  "notes": [
    "This learns to choose among clean selector-family outputs.",
    "Family target scores come from OOF validation choices; no references are used as prediction features."
  ]
}
```

</details>

<details><summary>reports/selector_family_meta_learner_target_encoded/summary.json — <code>reports/selector_family_meta_learner_target_encoded/summary.json</code></summary>

```json
{
  "family_meta_best": 0.6258109015467073,
  "best": {
    "model": "hgb_te",
    "score": 0.6256555031411847,
    "gain_vs_exp2": 0.036438874794370135,
    "gain_vs_family_meta": -0.00015539840552258877,
    "choice_counts": {
      "meta_gate": 6093,
      "fast_gate": 277,
      "rich_meta": 139,
      "global_clean": 116,
      "subset_clean": 55,
      "residual": 6
    },
    "per_subset": {
      "Aka_Gha": 0.327921,
      "Amh_Eth": 0.197266,
      "Eng_Eth": 0.686287,
      "Eng_Gha": 0.309581,
      "Eng_Ken": 0.865102,
      "Eng_Uga": 0.904495,
      "Lug_Uga": 0.81117,
      "Swa_Ken": 0.863746
    }
  },
  "results": [
    {
      "model": "hgb_te",
      "score": 0.6256555031411847,
      "gain_vs_exp2": 0.036438874794370135,
      "gain_vs_family_meta": -0.00015539840552258877,
      "choice_counts": {
        "meta_gate": 6093,
        "fast_gate": 277,
        "rich_meta": 139,
        "global_clean": 116,
        "subset_clean": 55,
        "residual": 6
      },
      "per_subset": {
        "Aka_Gha": 0.327921,
        "Amh_Eth": 0.197266,
        "Eng_Eth": 0.686287,
        "Eng_Gha": 0.309581,
        "Eng_Ken": 0.865102,
        "Eng_Uga": 0.904495,
        "Lug_Uga": 0.81117,
        "Swa_Ken": 0.863746
      }
    },
    {
      "model": "extra_trees_te",
      "score": 0.6247917023370437,
      "gain_vs_exp2": 0.03557507399022919,
      "gain_vs_family_meta": -0.001019199209663535,
      "choice_counts": {
        "meta_gate": 1235,
        "rich_meta": 1188,
        "fast_gate": 1165,
        "residual": 1138,
        "subset_clean": 980,
        "global_clean": 980
      },
      "per_subset": {
        "Aka_Gha": 0.327377,
        "Amh_Eth": 0.195985,
        "Eng_Eth": 0.687666,
        "Eng_Gha": 0.309372,
        "Eng_Ken": 0.864062,
        "Eng_Uga": 0.902426,
        "Lug_Uga": 0.810373,
        "Swa_Ken": 0.862681
      }
    },
    {
      "model": "random_forest_te",
      "score": 0.6241725689629447,
      "gain_vs_exp2": 0.034955940616130166,
      "gain_vs_family_meta": -0.0016383325837625584,
      "choice_counts": {
        "meta_gate": 2040,
        "fast_gate": 1115,
        "rich_meta": 1044,
        "residual": 851,
        "subset_clean": 850,
        "global_clean": 786
      },
      "per_subset": {
        "Aka_Gha": 0.327492,
        "Amh_Eth": 0.194883,
        "Eng_Eth": 0.685738,
        "Eng_Gha": 0.309395,
        "Eng_Ken": 0.864158,
        "Eng_Uga": 0.902411,
        "Lug_Uga": 0.809099,
        "Swa_Ken": 0.859536
      }
    },
    {
      "model": "ensemble_te",
      "score": 0.6254810698607948,
      "gain_vs_exp2": 0.03626444151398023,
      "gain_vs_family_meta": -0.0003298316859124961,
      "choice_counts": {
        "fast_gate": 1298,
        "rich_meta": 1218,
        "meta_gate": 1206,
        "residual": 1155,
        "global_clean": 922,
        "subset_clean": 887
      },
      "per_subset": {
        "Aka_Gha": 0.328723,
        "Amh_Eth": 0.196978,
        "Eng_Eth": 0.687703,
        "Eng_Gha": 0.309759,
        "Eng_Ken": 0.86433,
        "Eng_Uga": 0.904188,
        "Lug_Uga": 0.809338,
        "Swa_Ken": 0.862681
      }
    },
    {
      "model": "weighted_ensemble_te",
      "score": 0.6251992821957517,
      "gain_vs_exp2": 0.035982653848937174,
      "gain_vs_family_meta": -0.0006116193509555501,
      "choice_counts": {
        "fast_gate": 1298,
        "rich_meta": 1217,
        "meta_gate": 1207,
        "residual": 1151,
        "global_clean": 940,
        "subset_clean": 873
      },
      "per_subset": {
        "Aka_Gha": 0.328372,
        "Amh_Eth": 0.196575,
        "Eng_Eth": 0.686978,
        "Eng_Gha": 0.30941,
        "Eng_Ken": 0.86433,
        "Eng_Uga": 0.903881,
        "Lug_Uga": 0.809332,
        "Swa_Ken": 0.862699
      }
    }
  ],
  "notes": [
    "Target encodings are fold-safe and computed from training IDs only."
  ]
}
```

</details>

<details><summary>reports/subset_clean_source_selector/summary.json — <code>reports/subset_clean_source_selector/summary.json</code></summary>

```json
{
  "loaded_sources": [
    "exp2_top1",
    "exp2_rerank",
    "cluster_fast",
    "cluster_noleak"
  ],
  "missing_sources": [],
  "best": {
    "model": "subset_ensemble",
    "score": 0.605265968813137,
    "choice_counts": {
      "exp2_rerank": 3083,
      "cluster_fast": 1599,
      "cluster_noleak": 1078,
      "exp2_top1": 926
    },
    "per_subset": {
      "Aka_Gha": 0.322923,
      "Amh_Eth": 0.193023,
      "Eng_Eth": 0.67574,
      "Eng_Gha": 0.306609,
      "Eng_Ken": 0.812825,
      "Eng_Uga": 0.877343,
      "Lug_Uga": 0.768819,
      "Swa_Ken": 0.82993
    }
  },
  "results": [
    {
      "model": "subset_hgb",
      "score": 0.6039838451099379,
      "choice_counts": {
        "exp2_rerank": 3113,
        "cluster_fast": 1538,
        "exp2_top1": 1123,
        "cluster_noleak": 912
      },
      "per_subset": {
        "Aka_Gha": 0.31976,
        "Amh_Eth": 0.190433,
        "Eng_Eth": 0.674525,
        "Eng_Gha": 0.304922,
        "Eng_Ken": 0.806571,
        "Eng_Uga": 0.878966,
        "Lug_Uga": 0.766817,
        "Swa_Ken": 0.830103
      }
    },
    {
      "model": "subset_extra_trees",
      "score": 0.6040377810132422,
      "choice_counts": {
        "exp2_rerank": 2928,
        "cluster_fast": 1493,
        "cluster_noleak": 1196,
        "exp2_top1": 1069
      },
      "per_subset": {
        "Aka_Gha": 0.322485,
        "Amh_Eth": 0.192999,
        "Eng_Eth": 0.674018,
        "Eng_Gha": 0.305304,
        "Eng_Ken": 0.818983,
        "Eng_Uga": 0.875799,
        "Lug_Uga": 0.760469,
        "Swa_Ken": 0.833729
      }
    },
    {
      "model": "subset_ensemble",
      "score": 0.605265968813137,
      "choice_counts": {
        "exp2_rerank": 3083,
        "cluster_fast": 1599,
        "cluster_noleak": 1078,
        "exp2_top1": 926
      },
      "per_subset": {
        "Aka_Gha": 0.322923,
        "Amh_Eth": 0.193023,
        "Eng_Eth": 0.67574,
        "Eng_Gha": 0.306609,
        "Eng_Ken": 0.812825,
        "Eng_Uga": 0.877343,
        "Lug_Uga": 0.768819,
        "Swa_Ken": 0.82993
      }
    }
  ],
  "baseline_exp2_rerank": 0.5892166283468145,
  "clean_global_selector": 0.6041607089218853,
  "clean_oracle": 0.6479322858935932
}
```

</details>

<details><summary>reports/test_source_feasibility_audit/summary.json — <code>reports/test_source_feasibility_audit/summary.json</code></summary>

```json
{
  "mapped_sources": [
    {
      "validation_source": "exp1_baseline_qa_doc",
      "has_known_local_test_equivalent": false,
      "local_test_files": "",
      "conversion_without_modal": "unknown/no"
    },
    {
      "validation_source": "exp1_finetuned_qa_doc",
      "has_known_local_test_equivalent": false,
      "local_test_files": "",
      "conversion_without_modal": "unknown/no"
    },
    {
      "validation_source": "exp2_top1",
      "has_known_local_test_equivalent": true,
      "local_test_files": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\test_predictions_best_setup\\submission_encoder_top1_train_val.csv | C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_submissions\\submission_exp7_dense_top1_trainval.csv",
      "conversion_without_modal": "yes"
    },
    {
      "validation_source": "exp2_rerank",
      "has_known_local_test_equivalent": true,
      "local_test_files": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\test_predictions_best_setup\\submission_best_bgem3_rerank_train_val.csv | C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_submissions\\submission_exp6_reranker_trainval_top50cluster.csv",
      "conversion_without_modal": "yes"
    },
    {
      "validation_source": "exp3_top100_rerank",
      "has_known_local_test_equivalent": false,
      "local_test_files": "",
      "conversion_without_modal": "unknown/no"
    },
    {
      "validation_source": "exp4_pairwise",
      "has_known_local_test_equivalent": false,
      "local_test_files": "",
      "conversion_without_modal": "unknown/no"
    },
    {
      "validation_source": "exp5_encoder_exp2_rerank",
      "has_known_local_test_equivalent": false,
      "local_test_files": "",
      "conversion_without_modal": "unknown/no"
    },
    {
      "validation_source": "exp8_ghana_grouped",
      "has_known_local_test_equivalent": false,
      "local_test_files": "",
      "conversion_without_modal": "unknown/no"
    },
    {
      "validation_source": "exp9_jina",
      "has_known_local_test_equivalent": false,
      "local_test_files": "",
      "conversion_without_modal": "unknown/no"
    },
    {
      "validation_source": "exp10_lug_global",
      "has_known_local_test_equivalent": false,
      "local_test_files": "",
      "conversion_without_modal": "unknown/no"
    },
    {
      "validation_source": "exp10_lug_specialized",
      "has_known_local_test_equivalent": false,
      "local_test_files": "",
      "conversion_without_modal": "unknown/no"
    },
    {
      "validation_source": "exp12_lug_merged",
      "has_known_local_test_equivalent": false,
      "local_test_files": "",
      "conversion_without_modal": "unknown/no"
    },
    {
      "validation_source": "exp14_qonly",
      "has_known_local_test_equivalent": false,
      "local_test_files": "",
      "conversion_without_modal": "unknown/no"
    }
  ],
  "submission_like_csv_count": 30,
  "submission_like_csvs": [
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_cluster_selector_test_predictions_files\\test_candidate_features_scored.csv",
      "columns": [
        "best_any_rank",
        "row_i",
        "candidate_answer",
        "candidate_question",
        "source_count",
        "src_bge_q2q_top50",
        "bge_q2q_top50_count",
        "bge_q2q_top50_best_rank",
        "bge_q2q_top50_max_score",
        "max_any_score",
        "rrf",
        "src_tfidf_word_q2q_top50",
        "tfidf_word_q2q_top50_count",
        "tfidf_word_q2q_top50_best_rank",
        "tfidf_word_q2q_top50_max_score",
        "src_tfidf_char_q2q_top50",
        "tfidf_char_q2q_top50_count",
        "tfidf_char_q2q_top50_best_rank",
        "tfidf_char_q2q_top50_max_score",
        "src_tfidf_char_q2a_top50",
        "tfidf_char_q2a_top50_count",
        "tfidf_char_q2a_top50_best_rank",
        "tfidf_char_q2a_top50_max_score",
        "src_tfidf_word_q2a_top50",
        "tfidf_word_q2a_top50_count",
        "tfidf_word_q2a_top50_best_rank",
        "tfidf_word_q2a_top50_max_score",
        "src_bge_q2q_top200",
        "bge_q2q_top200_count",
        "bge_q2q_top200_best_rank",
        "bge_q2q_top200_max_score",
        "ID",
        "query",
        "subset",
        "answer_freq_subset",
        "answer_freq_global",
        "candidate_answer_len",
        "candidate_question_len",
        "query_len",
        "query_question_jaccard",
        "query_answer_jaccard",
        "log_answer_freq_subset",
        "log_answer_freq_global",
        "inv_best_any_rank",
        "inv_bge_q2q_top50_best_rank",
        "inv_tfidf_word_q2q_top50_best_rank",
        "inv_tfidf_char_q2q_top50_best_rank",
        "inv_tfidf_char_q2a_top50_best_rank",
        "inv_tfidf_word_q2a_top50_best_rank",
        "inv_bge_q2q_top200_best_rank",
        "_pre_rerank_pool_score",
        "exp6_rerank_score",
        "exp6_rerank_rank",
        "inv_exp6_rerank_rank",
        "exp6_rerank_margin_to_best",
        "selector_pred"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_cluster_selector_test_predictions_files\\test_reranker_choices.csv",
      "columns": [
        "best_any_rank",
        "row_i",
        "candidate_answer",
        "candidate_question",
        "source_count",
        "src_bge_q2q_top50",
        "bge_q2q_top50_count",
        "bge_q2q_top50_best_rank",
        "bge_q2q_top50_max_score",
        "max_any_score",
        "rrf",
        "src_tfidf_word_q2q_top50",
        "tfidf_word_q2q_top50_count",
        "tfidf_word_q2q_top50_best_rank",
        "tfidf_word_q2q_top50_max_score",
        "src_tfidf_char_q2q_top50",
        "tfidf_char_q2q_top50_count",
        "tfidf_char_q2q_top50_best_rank",
        "tfidf_char_q2q_top50_max_score",
        "src_tfidf_char_q2a_top50",
        "tfidf_char_q2a_top50_count",
        "tfidf_char_q2a_top50_best_rank",
        "tfidf_char_q2a_top50_max_score",
        "src_tfidf_word_q2a_top50",
        "tfidf_word_q2a_top50_count",
        "tfidf_word_q2a_top50_best_rank",
        "tfidf_word_q2a_top50_max_score",
        "src_bge_q2q_top200",
        "bge_q2q_top200_count",
        "bge_q2q_top200_best_rank",
        "bge_q2q_top200_max_score",
        "ID",
        "query",
        "subset",
        "answer_freq_subset",
        "answer_freq_global",
        "candidate_answer_len",
        "candidate_question_len",
        "query_len",
        "query_question_jaccard",
        "query_answer_jaccard",
        "log_answer_freq_subset",
        "log_answer_freq_global",
        "inv_best_any_rank",
        "inv_bge_q2q_top50_best_rank",
        "inv_tfidf_word_q2q_top50_best_rank",
        "inv_tfidf_char_q2q_top50_best_rank",
        "inv_tfidf_char_q2a_top50_best_rank",
        "inv_tfidf_word_q2a_top50_best_rank",
        "inv_bge_q2q_top200_best_rank",
        "_pre_rerank_pool_score",
        "exp6_rerank_score",
        "exp6_rerank_rank",
        "inv_exp6_rerank_rank",
        "exp6_rerank_margin_to_best",
        "selector_pred"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_cluster_selector_test_predictions_files\\test_selector_choices.csv",
      "columns": [
        "best_any_rank",
        "row_i",
        "candidate_answer",
        "candidate_question",
        "source_count",
        "src_bge_q2q_top50",
        "bge_q2q_top50_count",
        "bge_q2q_top50_best_rank",
        "bge_q2q_top50_max_score",
        "max_any_score",
        "rrf",
        "src_tfidf_word_q2q_top50",
        "tfidf_word_q2q_top50_count",
        "tfidf_word_q2q_top50_best_rank",
        "tfidf_word_q2q_top50_max_score",
        "src_tfidf_char_q2q_top50",
        "tfidf_char_q2q_top50_count",
        "tfidf_char_q2q_top50_best_rank",
        "tfidf_char_q2q_top50_max_score",
        "src_tfidf_char_q2a_top50",
        "tfidf_char_q2a_top50_count",
        "tfidf_char_q2a_top50_best_rank",
        "tfidf_char_q2a_top50_max_score",
        "src_tfidf_word_q2a_top50",
        "tfidf_word_q2a_top50_count",
        "tfidf_word_q2a_top50_best_rank",
        "tfidf_word_q2a_top50_max_score",
        "src_bge_q2q_top200",
        "bge_q2q_top200_count",
        "bge_q2q_top200_best_rank",
        "bge_q2q_top200_max_score",
        "ID",
        "query",
        "subset",
        "answer_freq_subset",
        "answer_freq_global",
        "candidate_answer_len",
        "candidate_question_len",
        "query_len",
        "query_question_jaccard",
        "query_answer_jaccard",
        "log_answer_freq_subset",
        "log_answer_freq_global",
        "inv_best_any_rank",
        "inv_bge_q2q_top50_best_rank",
        "inv_tfidf_word_q2q_top50_best_rank",
        "inv_tfidf_char_q2q_top50_best_rank",
        "inv_tfidf_char_q2a_top50_best_rank",
        "inv_tfidf_word_q2a_top50_best_rank",
        "inv_bge_q2q_top200_best_rank",
        "_pre_rerank_pool_score",
        "exp6_rerank_score",
        "exp6_rerank_rank",
        "inv_exp6_rerank_rank",
        "exp6_rerank_margin_to_best",
        "selector_pred"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_submissions\\submission_exp6_reranker_trainval_top50cluster.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_submissions\\submission_exp7_cluster_selector_trainval.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_submissions\\submission_exp7_dense_top1_trainval.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_submissions\\submission_local_candidate_regressor_hgb.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\test_predictions_best_setup\\submission_best_bgem3_rerank_train_val.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\test_predictions_best_setup\\submission_encoder_top1_train_val.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\test_predictions_best_setup\\test_candidate_scores.csv",
      "columns": [
        "ID",
        "subset",
        "candidate_rank",
        "bi_score",
        "rerank_score",
        "chosen",
        "candidate_question",
        "candidate_answer"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\reports\\local_candidate_regressor_submission\\test_candidate_regressor_choices.csv",
      "columns": [
        "ID",
        "candidate_answer",
        "pred_margin_vs_rerank",
        "use_best"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\submission_best_bgem3_rerank_train_val.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\submission_bgem3_retrieval.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\submission_bge_rerank.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\submission_rouge_reranker.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\Test.csv",
      "columns": [
        "ID",
        "input",
        "subset"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\Experiment G - results\\submission_best_per_subset.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\Experiment G - results\\submission_G.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\Experiment G - results\\submission_G_token_fusion.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\reports\\local_candidate_regressor_submission\\test_candidate_regressor_choices.csv",
      "columns": [
        "ID",
        "candidate_answer",
        "pred_margin_vs_rerank",
        "use_best"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_cluster_selector_test_predictions_files\\test_candidate_features_scored.csv",
      "columns": [
        "best_any_rank",
        "row_i",
        "candidate_answer",
        "candidate_question",
        "source_count",
        "src_bge_q2q_top50",
        "bge_q2q_top50_count",
        "bge_q2q_top50_best_rank",
        "bge_q2q_top50_max_score",
        "max_any_score",
        "rrf",
        "src_tfidf_word_q2q_top50",
        "tfidf_word_q2q_top50_count",
        "tfidf_word_q2q_top50_best_rank",
        "tfidf_word_q2q_top50_max_score",
        "src_tfidf_char_q2q_top50",
        "tfidf_char_q2q_top50_count",
        "tfidf_char_q2q_top50_best_rank",
        "tfidf_char_q2q_top50_max_score",
        "src_tfidf_char_q2a_top50",
        "tfidf_char_q2a_top50_count",
        "tfidf_char_q2a_top50_best_rank",
        "tfidf_char_q2a_top50_max_score",
        "src_tfidf_word_q2a_top50",
        "tfidf_word_q2a_top50_count",
        "tfidf_word_q2a_top50_best_rank",
        "tfidf_word_q2a_top50_max_score",
        "src_bge_q2q_top200",
        "bge_q2q_top200_count",
        "bge_q2q_top200_best_rank",
        "bge_q2q_top200_max_score",
        "ID",
        "query",
        "subset",
        "answer_freq_subset",
        "answer_freq_global",
        "candidate_answer_len",
        "candidate_question_len",
        "query_len",
        "query_question_jaccard",
        "query_answer_jaccard",
        "log_answer_freq_subset",
        "log_answer_freq_global",
        "inv_best_any_rank",
        "inv_bge_q2q_top50_best_rank",
        "inv_tfidf_word_q2q_top50_best_rank",
        "inv_tfidf_char_q2q_top50_best_rank",
        "inv_tfidf_char_q2a_top50_best_rank",
        "inv_tfidf_word_q2a_top50_best_rank",
        "inv_bge_q2q_top200_best_rank",
        "_pre_rerank_pool_score",
        "exp6_rerank_score",
        "exp6_rerank_rank",
        "inv_exp6_rerank_rank",
        "exp6_rerank_margin_to_best",
        "selector_pred"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_cluster_selector_test_predictions_files\\test_reranker_choices.csv",
      "columns": [
        "best_any_rank",
        "row_i",
        "candidate_answer",
        "candidate_question",
        "source_count",
        "src_bge_q2q_top50",
        "bge_q2q_top50_count",
        "bge_q2q_top50_best_rank",
        "bge_q2q_top50_max_score",
        "max_any_score",
        "rrf",
        "src_tfidf_word_q2q_top50",
        "tfidf_word_q2q_top50_count",
        "tfidf_word_q2q_top50_best_rank",
        "tfidf_word_q2q_top50_max_score",
        "src_tfidf_char_q2q_top50",
        "tfidf_char_q2q_top50_count",
        "tfidf_char_q2q_top50_best_rank",
        "tfidf_char_q2q_top50_max_score",
        "src_tfidf_char_q2a_top50",
        "tfidf_char_q2a_top50_count",
        "tfidf_char_q2a_top50_best_rank",
        "tfidf_char_q2a_top50_max_score",
        "src_tfidf_word_q2a_top50",
        "tfidf_word_q2a_top50_count",
        "tfidf_word_q2a_top50_best_rank",
        "tfidf_word_q2a_top50_max_score",
        "src_bge_q2q_top200",
        "bge_q2q_top200_count",
        "bge_q2q_top200_best_rank",
        "bge_q2q_top200_max_score",
        "ID",
        "query",
        "subset",
        "answer_freq_subset",
        "answer_freq_global",
        "candidate_answer_len",
        "candidate_question_len",
        "query_len",
        "query_question_jaccard",
        "query_answer_jaccard",
        "log_answer_freq_subset",
        "log_answer_freq_global",
        "inv_best_any_rank",
        "inv_bge_q2q_top50_best_rank",
        "inv_tfidf_word_q2q_top50_best_rank",
        "inv_tfidf_char_q2q_top50_best_rank",
        "inv_tfidf_char_q2a_top50_best_rank",
        "inv_tfidf_word_q2a_top50_best_rank",
        "inv_bge_q2q_top200_best_rank",
        "_pre_rerank_pool_score",
        "exp6_rerank_score",
        "exp6_rerank_rank",
        "inv_exp6_rerank_rank",
        "exp6_rerank_margin_to_best",
        "selector_pred"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_cluster_selector_test_predictions_files\\test_selector_choices.csv",
      "columns": [
        "best_any_rank",
        "row_i",
        "candidate_answer",
        "candidate_question",
        "source_count",
        "src_bge_q2q_top50",
        "bge_q2q_top50_count",
        "bge_q2q_top50_best_rank",
        "bge_q2q_top50_max_score",
        "max_any_score",
        "rrf",
        "src_tfidf_word_q2q_top50",
        "tfidf_word_q2q_top50_count",
        "tfidf_word_q2q_top50_best_rank",
        "tfidf_word_q2q_top50_max_score",
        "src_tfidf_char_q2q_top50",
        "tfidf_char_q2q_top50_count",
        "tfidf_char_q2q_top50_best_rank",
        "tfidf_char_q2q_top50_max_score",
        "src_tfidf_char_q2a_top50",
        "tfidf_char_q2a_top50_count",
        "tfidf_char_q2a_top50_best_rank",
        "tfidf_char_q2a_top50_max_score",
        "src_tfidf_word_q2a_top50",
        "tfidf_word_q2a_top50_count",
        "tfidf_word_q2a_top50_best_rank",
        "tfidf_word_q2a_top50_max_score",
        "src_bge_q2q_top200",
        "bge_q2q_top200_count",
        "bge_q2q_top200_best_rank",
        "bge_q2q_top200_max_score",
        "ID",
        "query",
        "subset",
        "answer_freq_subset",
        "answer_freq_global",
        "candidate_answer_len",
        "candidate_question_len",
        "query_len",
        "query_question_jaccard",
        "query_answer_jaccard",
        "log_answer_freq_subset",
        "log_answer_freq_global",
        "inv_best_any_rank",
        "inv_bge_q2q_top50_best_rank",
        "inv_tfidf_word_q2q_top50_best_rank",
        "inv_tfidf_char_q2q_top50_best_rank",
        "inv_tfidf_char_q2a_top50_best_rank",
        "inv_tfidf_word_q2a_top50_best_rank",
        "inv_bge_q2q_top200_best_rank",
        "_pre_rerank_pool_score",
        "exp6_rerank_score",
        "exp6_rerank_rank",
        "inv_exp6_rerank_rank",
        "exp6_rerank_margin_to_best",
        "selector_pred"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_submissions\\submission_exp6_reranker_trainval_top50cluster.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_submissions\\submission_exp7_cluster_selector_trainval.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_submissions\\submission_exp7_dense_top1_trainval.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\exp7_submissions\\submission_local_candidate_regressor_hgb.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\test_predictions_best_setup\\submission_best_bgem3_rerank_train_val.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\test_predictions_best_setup\\submission_encoder_top1_train_val.csv",
      "columns": [
        "ID",
        "TargetRLF1",
        "TargetR1F1",
        "TargetLLM"
      ]
    },
    {
      "path": "C:\\Users\\Papa Offei\\Documents\\lalang\\modal_outputs\\test_predictions_best_setup\\test_candidate_scores.csv",
      "columns": [
        "ID",
        "subset",
        "candidate_rank",
        "bi_score",
        "rerank_score",
        "chosen",
        "candidate_question",
        "candidate_answer"
      ]
    }
  ],
  "notes": [
    "This audit only maps local files already present. It does not claim unavailable sources are impossible, only that no local test equivalent was found.",
    "Sources like exp3/exp4/exp5/exp9/exp14 may require model inference to convert, which could be local if model artifacts are available, but not from existing submission CSVs."
  ]
}
```

</details>

## License and data notice

No license was present in the original workspace. Add a license before redistributing the code. Confirm the competition/dataset terms and the licenses of all base models before publishing the excluded data, weights, or derived artifacts.

## Citation

If this research is used, cite the repository and the upstream datasets/models used by the relevant experiment. The exact upstream model identifiers are preserved in the experiment summaries and scripts.
