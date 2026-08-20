# Promising Experiments Not Yet Converted To Final Test Submission

Baseline for estimates: current best production-style retrieval/rerank/selector pipeline before the new E5-large Lug_Uga work. Overall gain estimates are validation proxies, weighted by Val subset counts where possible. Treat them as decision guidance, not guaranteed leaderboard movement.

| Priority | Todo | Evidence | Estimated Overall ROUGE Gain | Running Conservative Gain | Status |
|---:|---|---|---:|---:|---|
| 1 | Convert Lug_Uga BGE+E5-large merged candidates + local HGB selector into a test submission overlay. | Exp12/13: Lug_Uga global rerank baseline `0.67931`; merged rerank `0.68353`; HGB selector `0.69189`; merged oracle `0.87404`. | `+0.0013` to `+0.0016` | `+0.0015` | Proven on val, not yet test-submitted |
| 2 | Produce two Lug overlay submissions: overlay on current best cluster-selector submission, and overlay on best train+val reranker submission. | Cluster selector helped a bit on LB; Lug selector changes only 78/846 val Lug rows, so it should be safe to A/B without disturbing other subsets. | `+0.0008` to `+0.0016` | `+0.0025` | Needs final test candidate generation |
| 3 | Run exp12-style merged candidate evaluation for Eng_Ken and Eng_Eth using BGE + E5-large, then rerank with current best reranker. | Exp11 base top1: Eng_Ken E5-large beats BGE by `+0.02263`; Eng_Eth by `+0.01384`. Oracle change is tiny, so this is less certain than Lug. | `+0.0002` to `+0.0010` | `+0.0030` | Needs val check before test |
| 4 | If item 3 is positive, train local source-aware candidate selectors for Eng_Ken and Eng_Eth. | Lug selector proved source/rank/rerank features can extract extra value from merged candidates. These subsets have top1 evidence but weaker oracle evidence. | `+0.0002` to `+0.0008` | `+0.0034` | Conditional on item 3 |
| 5 | Try Eng_Gha as a low-risk E5-large candidate-source addition, but only if merged rerank/selector improves on val. | Exp11: E5-large top1 `+0.00134`, oracle `+0.00145` vs BGE base. Small but both positive. | `+0.0000` to `+0.0003` | `+0.0035` | Low priority |
| 6 | Build one final source-aware selector over all available candidate sources for test: BGE rerank, cluster selector, local candidate regressor, Lug E5 selector, and any positive Eng subset selectors. | Previous leaked selector showed large theoretical headroom; honest selector helped modestly. The new source flags and E5 ranks are stronger features than before. | `+0.0010` to `+0.0030` | `+0.0050` | Needs careful OOF validation |
| 7 | Create final test submissions in a small A/B batch, not one giant untested blend. Suggested files: current best, current best + Lug overlay, current best + Lug/Eng overlays, current best + global source-aware selector. | Leaderboard response may differ from val. We need isolate which overlay helps. | Not additive; validation of above | `+0.0050` | Submission management |

## Do Not Spend More Compute On Yet

- More plain reranker fine-tuning: ROUGE regression, pairwise, stronger base, subset-only reranker, and Ghana grouping have all failed to produce the needed jump.
- Replacing BGE-M3 globally with E5-large: Exp11 says E5-large helps mainly Lug_Uga, Eng_Ken, and Eng_Eth. It hurts or barely helps other subsets.
- E5-large top1 as a direct Lug fallback: despite good base-model evidence, in Exp12 E5-large top1 alone was only `0.48765`, far below BGE fine-tuned top1 `0.55868`.

## Current Best Next Action

Generate a test submission overlay for Lug_Uga only:

1. Use Train+Val as answer bank.
2. For test rows where `subset == Lug_Uga`, retrieve BGE fine-tuned top50 and E5-large top50.
3. Deduplicate by answer text.
4. Score merged candidates with current best train+val reranker.
5. Apply the HGB selector trained on all Lug_Uga validation candidate rows from Exp12/13.
6. Leave every non-Lug_Uga row exactly as in the current best submission.

Expected total gain: about `+0.001` to `+0.0016` overall ROUGE, with unusually strong evidence compared with most recent ideas.

## Newly Proven Local Selector Candidate

- Exp14 q-only-vs-exp2 selector: best OOF `logreg` selects q-only at threshold `0.88` with ROUGE-1 `0.589977` vs exp2 `0.589217` (`+0.000760`). This should be considered for final source-aware selector work, but it needs a train+val/test-time q-only reranker artifact before conversion to submission.
