# No-Modal ROUGE Lift Goal Progress - 2026-06-08

Objective: increase current best validation ROUGE-1 by about `+0.04` without using Modal credits, while documenting breakthroughs and failures.

## Baseline

- Current core validation baseline used for comparison: exp2 q+a BGE-M3 reranker, ROUGE-1 `0.5892166`.
- Best single existing local prediction source found so far: exp3 top100 rerank, ROUGE-1 `0.5904174`.

## Breakthroughs

### Existing Prediction Source Oracle

Script: `local_existing_prediction_oracle_audit.py`

Report folder: `reports/existing_prediction_oracle_audit`

Findings:

- Oracle over existing deployable-ish validation prediction sources: `0.6569692`.
- Headroom vs exp2 q+a reranker: `+0.0677526`.
- This proves that the local artifacts contain enough diverse predictions to theoretically exceed the `+0.04` target, but only if source selection can be learned.

### Candidate-Level Existing Source Selector

Script: `local_existing_source_selector.py`

Report folder: `reports/existing_source_selector`

Best OOF result:

- Model: ExtraTrees candidate-level regressor.
- ROUGE-1: `0.5978363`.
- Gain vs exp2 q+a reranker: `+0.0086197`.
- Gain vs best single existing source: `+0.0074189`.

Interpretation:

- This is the strongest honest no-Modal validation gain found so far.
- It captures only a small part of the `+0.0678` oracle headroom.
- The model often chooses `exp4_pairwise` despite weak standalone performance, suggesting it is learning row/source interaction patterns rather than just choosing the strongest global source.

### Target-Encoded Existing Source Selector

Script: `local_existing_source_selector_te.py`

Report folder: `reports/existing_source_selector_target_encoded`

Best OOF result:

- Model: average of HGB and ExtraTrees with fold-safe source/subset target encodings.
- ROUGE-1: `0.5983993`.
- Gain vs exp2 q+a reranker: `+0.0091826`.
- Gain vs best single existing source: `+0.0079818`.

Interpretation:

- This is now the best no-Modal local validation gain.
- Fold-safe source/subset priors help, but only modestly.
- Still far short of the `+0.04` target, so the next question is whether test-time source availability and stronger features can unlock more of the `0.65697` oracle.

### Deployable Source Selector Audit

Script: `local_deployable_source_selector.py`

Report folder: `reports/deployable_source_selector`

Initial result:

- `cluster_fullcap` alone scored `0.6353457`, which is `+0.0461291` vs exp2 q+a reranker.
- A selector over `exp2_top1`, `exp2_rerank`, and cluster selector outputs scored `0.6301937`, `+0.0409770` vs exp2.
- Deployable-source oracle over those same sources was `0.6594523`, `+0.0702356` vs exp2.

Audit verdict:

- This apparent `+0.04` breakthrough is not clean enough to count as achieved.
- The `cluster_fullcap` validation selector used `reference_len` and `answer_ref_len_ratio_proxy`, which depend on the validation reference answer.
- The clean `cluster_noleak` version scores only `0.5646715`, so the fullcap gain is mostly a leakage clue, not a submission-ready result.

What this taught us:

- Candidate selection is highly sensitive to per-row answer shape/length.
- If we can approximate expected answer length or answer-shape confidence legally at test time, there may still be a large selector gain available without retraining neural models.

### Clean Deployable Source Selector

Script: `local_deployable_source_selector.py`

Report folder: `reports/clean_deployable_source_selector`

Clean source set:

- `exp2_top1`
- `exp2_rerank`
- `cluster_fast`
- `cluster_noleak`

Result:

- Clean source oracle: `0.6479323`, `+0.0587157` vs exp2.
- Best learned OOF selector: `0.6041607`, `+0.0149441` vs exp2.

Interpretation:

- This is a real, non-leaky, deployable local gain.
- It proves the clean source set has enough oracle headroom to reach the `+0.04` target.
- The hard part is learning when to trust weak standalone sources, especially `cluster_fast`.

### Subset-Specific Clean Source Selector

Script: `local_subset_clean_source_selector.py`

Report folder: `reports/subset_clean_source_selector`

Result:

- Best subset-trained selector: `0.6052660`, `+0.0160493` vs exp2.
- Per-subset hybrid between global clean selector and subset selector: `0.6065232`, `+0.0173066` vs exp2.

Interpretation:

- Subset-specific training helps, but only modestly.
- The best clean gain is now roughly `+0.0173`.
- Remaining clean oracle headroom is still about `+0.0414`, so the target is selector-limited, not source-limited.

### Cluster-Fast Trust Gate

Script: `local_cluster_fast_gate.py`

Report folder: `reports/cluster_fast_gate`

Result:

- Baseline clean hybrid before gate: `0.6065232`.
- Best zero-threshold gate: `0.622417`.
- Best diagnostic threshold sweep: `0.6224808`.
- Gain vs exp2 q+a reranker: `+0.0332642`.

Interpretation:

- This was the largest honest no-Modal jump after the clean selector.
- The model learned when the globally weak `cluster_fast` source should override the current clean hybrid.
- This validated the earlier observation that `cluster_fast` carries high-value sparse wins, especially in Lug_Uga and East Africa subsets.

### Rich Clean Meta Selector

Script: `local_rich_clean_meta_selector.py`

Report folder: `reports/rich_clean_meta_selector`

Result:

- Best all-source rich meta-selector: `0.6227090`.
- Per-subset hybrid between rich meta, fast gate, and earlier selector families: `0.6238406`.
- Gain vs exp2 q+a reranker: `+0.0346239`.

Interpretation:

- Rich raw candidate features help recover more of the clean oracle, but the model still leaves meaningful headroom.
- A residual override model (`local_residual_clean_override_selector.py`) only improved the subset family hybrid to `0.6237822` before the family-level hybrid, so residual gating is mostly saturated.

### Selector-Family Meta Learner

Script: `local_selector_family_meta_learner.py`

Report folder: `reports/selector_family_meta_learner`

Result:

- Selector-family oracle: `0.6338465`, which would clear the `+0.04` target.
- Learned selector-family meta-learner: `0.6258109`.
- Per-subset hybrid with family meta included: `0.6258761`.
- Gain vs exp2 q+a reranker: `+0.0366595`.

Interpretation:

- This is the best honest no-Modal validation result so far.
- It is still about `0.00334` short of the `+0.04` target.
- The remaining gap is now a meta-selection problem among already-clean selector outputs, not a candidate-source availability problem.

### Fullcap vs Noleak Diagnostic Mining

Script: `local_fullcap_vs_noleak_mining.py`

Report folder: `reports/fullcap_vs_noleak_mining`

One-to-one HGB comparison:

- Leaky fullcap: `0.6353457`.
- Clean noleak: `0.5646715`.
- Gap: `+0.0706742`.
- Changed answer rate: `38.86%`.

Largest subset gaps:

- `Lug_Uga`: `+0.2223640`.
- `Eng_Ken`: `+0.0908214`.
- `Eng_Uga`: `+0.0846642`.
- `Swa_Ken`: `+0.0766920`.

Interpretation:

- The biggest missing clean signal is not Ghana; it is Lug_Uga and the English East Africa subsets.
- Many Lug_Uga losses are short/direct answers where clean noleak chooses longer generic candidates.
- Next legal selector work should focus on source-confidence and answer-shape proxies in those subsets.

## Failures / Low-Yield Routes

### Exp14 q-only vs exp2 binary selector

Report folder: `reports/exp14_qonly_exp2_selector`

- q-only reranker alone: `0.5647796`.
- Perfect exp2/q-only chooser: `0.6134483`.
- Best deployable OOF selector: `0.5899766`.
- Gain vs exp2: `+0.0007599`.

Interpretation:

- q-only has real complementary wins, but final-text-only deployable features barely identify them.
- Keep q-only as a possible feature, but this binary branch is not enough to chase the `+0.04` target.

### Predicted Answer-Length Replacement For Fullcap Leakage

Script: `local_predicted_length_cluster_selector.py`

Report folder: `reports/predicted_length_cluster_selector`

Result:

- Train-only TF-IDF/Ridge answer-length predictor validation MAE: `31.63` tokens.
- Best clean cluster selector with predicted-length features: `0.5675437`.
- Comparison: exp2 q+a reranker is `0.5892166`; leaky fullcap HGB was `0.6353457`.

Interpretation:

- This did not recover the fullcap gain.
- The true reference length signal was sharper than a simple query-to-length prediction.
- Predicted length alone is not worth using as the next submission branch.

### Candidate-Regressor Source As Fifth Source

Scripts:

- `local_reconstruct_candidate_regressor_source.py`
- `local_rich_clean_meta_with_regressor.py`

Report folder: `reports/rich_clean_meta_with_regressor`

Result:

- Local candidate regressor source alone: `0.5924740`.
- Oracle with regressor added to the clean sources: `0.6502489`.
- Best learned selector with regressor source: `0.6136064`.

Interpretation:

- The candidate regressor adds oracle headroom, but the learned selector over-trusts it and hurts the current best.
- Its validation choice file does not include answer text, so it is awkward to use as a text-agreement source.
- Keep it as a possible final test blend source, but it is not improving validation selection yet.

### Miss Detector / Rescue Family Attempt

Script: `local_family_meta_miss_correction.py`

Report folder: `reports/family_meta_miss_correction`

Result:

- Family-meta miss detector AUC: about `0.935`.
- Diagnostic oracle if the correct rescue family were known: `0.6338465`.
- Deployable-style switch-to-predicted-family result: about `0.625917`.

Interpretation:

- We can identify many rows where the current family meta choice is suspicious.
- The hard part is not miss detection; it is choosing the replacement source accurately.
- This route gave only a tiny practical lift and did not meaningfully move the leaderboard target.

### Direct Oracle-Family Classifier

Script: `local_selector_family_classifier.py`

Report folder: `reports/selector_family_classifier`

Result:

- Direct classifier that predicts which selector family should win: best around `0.6237827`.
- This is below the selector-family meta learner and below the current best.

Interpretation:

- Treating the problem as multiclass family classification is too blunt.
- Regression-style candidate scoring remains stronger than direct winner classification.

### Extended Validation Source Selector

Script: `local_extended_validation_source_selector.py`

Report folder: `reports/extended_validation_source_selector`

Added local sources:

- Previous selector-family hybrid.
- Exp3 top100 reranker.
- Exp4 pairwise reranker.
- Exp5 encoder-exp2 rerank.
- Exp9 Jina reranker.
- Exp14 q-only reranker.
- Ghana specialist validation source.
- Lug_Uga specialist/global/merged/selector sources.
- Leftover local mT0 generation validation predictions.
- Exp1 query-to-QA baseline and finetuned validation predictions.
- Exp11 base encoder benchmark top1 answers from BGE-M3 base, multilingual-E5 base, and multilingual-E5 large.

Intermediate results:

- Specialist/all-model per-subset hybrid before adding the leftover weak sources: `0.6278404`.
- Gain vs exp2 q+a reranker: `+0.0386237`.
- Expanded global selector after adding weak generation/base-encoder sources: `0.6277010`.
- Expanded per-subset all-model hybrid: `0.6289118`.
- Gain vs exp2 q+a reranker: `+0.0396952`.

Interpretation:

- The leftover sources are weak globally, but they provide row-level rescue cases.
- Exp11 E5-large top1 is the strongest of the leftover raw sources at `0.5048918`, still far below the current ensemble.
- The improvement comes from diversity, not standalone strength.

### Breakthrough: Length-Grouped Selector Hybrid

Report files:

- `reports/extended_validation_source_selector/length_grouped_selector_hybrid_choices.csv`
- `reports/extended_validation_source_selector/length_grouped_selector_hybrid_summary.json`

Best configuration:

- Grouping feature: validation query/input character length.
- Groups: 5 quantile bins inside each subset.
- Minimum group support per selector: 80 rows.
- Best validation ROUGE-1: `0.6303205`.
- Gain vs exp2 q+a reranker: `+0.0411038`.
- Gain over expanded all-model per-subset hybrid: `+0.0014086`.

Verdict:

- This clears the requested `+0.04` validation target without using Modal credits.
- The useful signal is that selector reliability changes by both subset and query length.
- The result is validation-tuned because it chooses the best selector per subset/length bin using validation labels.
- For a final test submission, this should be treated as a rule family to mirror carefully: compute the same subset + query-length bins on test, then apply the learned selector choices only if all underlying test prediction sources are available.

Best grouped selector choices:

- `Aka_Gha`: switches among expanded random forest, expanded HGB, old all-model hybrid, extended subset hybrid, and expanded all-model hybrid by query-length bin.
- `Amh_Eth`: short queries prefer expanded HGB; mid bins prefer specialist weighted/random forest; longest bin reverts to all-model hybrid.
- `Eng_Eth`: combines old all-model hybrid, specialist models, and expanded ensemble depending on query length.
- `Eng_Gha`: short and mid queries benefit from expanded extra trees / specialist weighted ensemble; longer bins stay with expanded all-model hybrid.
- `Eng_Ken`: length bins rotate between expanded HGB, extended specialist hybrid, specialist extra trees, expanded random forest, and expanded extra trees.
- `Eng_Uga`: length bins use expanded random forest / extra trees plus old specialist/all-model hybrids.
- `Lug_Uga`: mostly keeps old all-model hybrid, with selective expanded ensemble / specialist HGB / expanded random forest bins.
- `Swa_Ken`: mixes expanded HGB, extended specialist hybrid, expanded extra trees, and extended subset hybrid.

## Current Best

- Best validation ROUGE-1 found without Modal before leakage audit: `0.6303205`.
- Baseline exp2 q+a BGE-M3 reranker: `0.5892166`.
- Apparent validation gain: `+0.0411038`.
- Target status after leakage audit: not cleanly achieved. The apparent score is useful diagnostics, but it is validation-tuned.

## Deployment Caveat

The breakthrough uses existing local validation artifacts and validation-tuned group choices. It is excellent evidence that the missing gain is in reranker/selector gating rather than encoder recall, but a leaderboard submission should only use this exact recipe after confirming the equivalent test predictions exist for every selected source. If a source has no test-side equivalent, the grouped rule should fall back to the nearest available selector family.

## Leakage Audit Correction

Script: `local_length_grouped_selector_leakage_audit.py`

Report file: `reports/extended_validation_source_selector/length_grouped_selector_leakage_audit.json`

Findings:

- Winning runtime features were only `subset` and `input_char_len`.
- These features are available at test time.
- The winning runtime feature set does not require `reference`, `output`, or ROUGE at test time.
- However, the reported `0.6303205` score used full validation `target_r1` to choose the best selector inside each subset/length bin.
- That is not direct feature leakage, but it is validation target leakage / model-selection leakage.

Safer estimates:

- Full-validation grouped score: `0.6303205`, `+0.0411038` vs exp2.
- Fixed winning config with group picks learned OOF: `0.6277781`, `+0.0385614` vs exp2.
- Nested config OOF: `0.6268037`, `+0.0375871` vs exp2.

Corrected verdict:

- The grouped selector proves the idea is promising, but the `+0.04` claim should not be treated as honestly achieved yet.
- The honest no-Modal result is still around `+0.0386`, close but not across the requested `+0.04`.
- Next work should chase roughly `+0.0015` more with OOF-safe selector rules or a deployable test-side source, not report the full-validation grouped number as final.

### Target Encoding Attempts

Scripts:

- `local_rich_clean_meta_selector_te.py`
- `local_selector_family_meta_learner_te.py`

Results:

- Rich clean target-encoded selector: best `0.6220269`, below plain rich `0.6227090`.
- Selector-family target-encoded learner: best `0.6256555`, below plain family meta `0.6258109`.

Interpretation:

- Fold-safe target encodings did not unlock the remaining gap.
- The current model already captures most useful source/subset priors.

## Next No-Modal Steps

1. Convert the length-grouped selector hybrid into a test-side recipe by checking which selected source families already have test predictions.
2. For missing test-side source families, define deterministic fallbacks to the nearest available selector family rather than silently dropping rows.
3. Rebuild the same subset + input-character-length bins on test, using train+val/test query lengths only, then apply the learned selector map.
4. Create an audit table that shows source availability and fallback rate by subset/bin before making a final submission.
5. Treat the `0.6303205` validation score as a selector-gating breakthrough, but keep the leaderboard submission conservative unless every selected branch is mirrored.
