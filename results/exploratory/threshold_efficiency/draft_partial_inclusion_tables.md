# Draft: partial inclusion at the selected threshold

Discussion draft only. The manuscript, SI Appendix, poster, and their PDFs have not been edited.

These results use partial inclusion with the existing Bonferroni-adjusted Clopper–Pearson procedure (`tied_bonferroni`), not the separate ordered-fallback alternative. This choice retains the guarantee simultaneously over all error limits α.

## Algorithm changes

For a nonempty report with top-candidate probability U, answer whenever U>t; when U=t, answer independently with probability γ; otherwise abstain. Empty lists always abstain. Thus γ=1 recovers the original inclusive threshold rule.

1. Fix the original threshold set T and tie-inclusion set Γ={0.25,0.50,0.75,1.00}, independently of the graded selection sample. The number of candidate rules is now M=|T||Γ|=408, rather than 102.
2. Draw one independent Uniform(0,1) value V_i for each graded question. Reuse that question's value across all candidate pairs (t,γ). Define A_i(t,γ)=1 for a nonempty report when U_i>t, or U_i=t and V_i≤γ, and zero otherwise.
3. For every pair compute the actual integer counts N(t,γ)=Σ_i A_i(t,γ) and K(t,γ)=Σ_i A_i(t,γ)(1−Z_i). Retain the same bound formula, now indexed by (t,γ), with Beta quantile at 1−δ/M and M=408. Set the bound to one when K=N. Fractional inclusion weights must not be substituted for binomial counts.
4. Among pairs with N>0 and upper bound at most α, choose the smallest t and then the largest γ at that t. These nested rules maximize the answer rate among qualifying pairs. If no pair qualifies, abstain everywhere.
5. For each new report, use the selected pair and a fresh independent uniform draw to make the decision in Step 2.

The selected-rule guarantee becomes r(t̂(α),γ̂(α))≤α. Here r(t,γ) is error probability conditional on answering under the randomized rule. Its probability averages over future questions and the independent inclusion decision. Confidence is over the graded selection sample and its auxiliary draws; the bounds remain simultaneous over all α, separately for each fixed model, grader, and candidate family. This does not assert that every finite test error is below α.

Integration would also update the notation, proposition and proof, and empirical methods/captions in the SI. The existing deterministic penalty threshold and its associated theory remain applicable as the γ=1 case. The current grader-specific explanations around S8 and numerical comparisons around S9 would need to be regenerated because their values change.

## Prospective Table S8

Primary grader GPT-5.6 Terra; the original primary question partition, with 2,163 selection and 2,163 test questions; confidence 95%; 408 candidate rules. Tie inclusion γ is the probability of answering exactly at t; all nonempty reports above t answer. Upper bound is the simultaneous one-sided bound used for selection. Test error uses the realized integer test counts and pointwise two-sided 95% Clopper–Pearson intervals. A dash denotes no qualifying rule or undefined conditional error. All denotes α=0.20,0.25,0.30,0.35,0.40. Values are rounded for presentation; Gemini's 30.0% bound at α=0.30 is slightly below 30% before rounding.

| Model | α | Threshold t | Tie inclusion γ | Upper bound (%) | Answered n (%) | Test error (%) [95% interval] |
|---|---|---|---|---|---|---|
| Gemini | 0.20 | — | — | — | 0 (0.0) | — |
| Gemini | 0.25 | — | — | — | 0 (0.0) | — |
| Gemini | 0.30 | 0.800 | 0.25 | 30.0 | 743 (34.4) | 22.7 [19.8, 25.9] |
| Gemini | 0.35 | 0.600 | 0.75 | 34.9 | 1,291 (59.7) | 28.6 [26.1, 31.1] |
| Gemini | 0.40 | 0.400 | 0.25 | 38.9 | 1,546 (71.5) | 33.4 [31.0, 35.8] |
| Sonnet | 0.20 | — | — | — | 0 (0.0) | — |
| Sonnet | 0.25 | 0.900 | 0.75 | 24.3 | 44 (2.0) | 20.5 [9.8, 35.3] |
| Sonnet | 0.30 | 0.850 | 0.25 | 29.0 | 119 (5.5) | 15.1 [9.2, 22.8] |
| Sonnet | 0.35 | 0.760 | 1.00 | 32.4 | 234 (10.8) | 21.4 [16.3, 27.2] |
| Sonnet | 0.40 | 0.600 | 0.75 | 39.9 | 485 (22.4) | 31.5 [27.4, 35.9] |
| DeepSeek | All | — | — | — | 0 (0.0) | — |

## Prospective Table S9

The same 100 additional random partitions into threshold-selection and test samples as the current table, using the primary grader. The 50–50 and 70–30 allocations use 2,163 and 3,028 selection questions. Both selected rules are evaluated on the common 1,298 test questions unused by either selection sample. Qualifying gives the count of supported partitions. Answered is the mean answer rate including zero when unsupported. Test error is the median [25th,75th percentiles] among partitions returning answers; every qualifying partition returns answers. These percentiles summarize variation across overlapping partitions, not confidence intervals. A single contributing partition has no percentile interval. A positive answer rate rounding to zero is shown as <0.1. All denotes the same five α values.

| Model | α | 50–50 qualifying /100 | 50–50 answered (%), mean | 50–50 test error (%) | 70–30 qualifying /100 | 70–30 answered (%), mean | 70–30 test error (%) |
|---|---|---|---|---|---|---|---|
| Gemini | 0.20 | 0 | 0.0 | — | 0 | 0.0 | — |
| Gemini | 0.25 | 56 | 10.6 | 18.7 [17.4, 20.4] | 93 | 18.3 | 18.5 [16.3, 20.9] |
| Gemini | 0.30 | 100 | 31.0 | 21.9 [19.9, 24.0] | 100 | 33.7 | 23.7 [22.1, 24.9] |
| Gemini | 0.35 | 100 | 59.5 | 30.0 [28.5, 31.0] | 100 | 62.6 | 30.3 [29.3, 31.4] |
| Gemini | 0.40 | 100 | 70.1 | 33.8 [32.9, 34.9] | 100 | 71.9 | 33.9 [33.0, 35.7] |
| Sonnet | 0.20 | 0 | 0.0 | — | 0 | 0.0 | — |
| Sonnet | 0.25 | 1 | <0.1 | 20.0 | 3 | 0.1 | 23.5 [21.1, 24.0] |
| Sonnet | 0.30 | 47 | 4.1 | 21.7 [20.5, 24.1] | 80 | 7.2 | 22.2 [20.0, 23.8] |
| Sonnet | 0.35 | 94 | 12.7 | 25.5 [22.2, 28.6] | 100 | 16.3 | 28.2 [24.6, 30.6] |
| Sonnet | 0.40 | 100 | 22.5 | 31.8 [29.1, 33.6] | 100 | 24.2 | 32.5 [30.7, 34.5] |
| DeepSeek | All | 0 | 0.0 | — | 0 | 0.0 | — |

## Interpretation and reproducibility

- The primary Gemini result at α=0.40 changes from 63.8% answered and 29.8% test error to 71.5% answered and 33.4% test error. Across additional 50–50 partitions evaluated on the common 1,298 test questions, mean answered rises from 62.7% to 70.1%, and median test error from 30.3% to 33.8%.
- This is not a uniform improvement. In the primary partition, Gemini at α=0.25 loses support, Gemini at α=0.35 answers fewer questions, and Sonnet at α=0.40 answers fewer questions. The larger candidate family entails a stricter per-rule correction.
- Sonnet's supported α=0.25 result in S8 is fragile: only one of the 100 additional 50–50 partitions qualifies. DeepSeek remains unsupported at all tested limits.
- S9 intentionally uses its existing common-test convention. Earlier exploratory report summaries used all 2,163 test questions for the 50–50 allocation, so their rounded means and medians differ slightly.
- All partial-inclusion values here use auxiliary randomization replicate 0, designated before running the comparison. Four other auxiliary randomizations were run separately as a sensitivity check, not pooled into 500 partitions or selected for favorable results. The uniform draws use NumPy SeedSequence([20260914, partition_seed, coin_rep]), with primary partition seed 20260912 and additional partition seeds 20260913–20261012. The random draws are shared across models, graders and allocations within each partition, with disjoint selection and test question draws.
- These are exploratory results on the existing benchmark, not new independent validation data.

## Two-line poster wording

Let $Y$ be the correct answer. The model reports $R=(S,q)$: a candidate list $S=(a_1,\ldots,a_K)$, candidate probabilities $q_j$, and probability $q_\perp$ that none is correct (IDK), with $\sum_jq_j+q_\perp=1$.

A typesetting check using the poster's Helvetica text font, math font, 31.68 bp text size, 39.6 bp leading and 1,560 bp block width produces two lines. Remove the existing forced line breaks and allow natural wrapping. The prose already defines the IDK event, so the separate equality q_perp=Pr_q(Y not in S) is redundant here. No font reduction or box widening is needed.
