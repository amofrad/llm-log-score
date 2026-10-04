# Sensitivity of threshold decisions to a 70–30 split

The primary 50–50 analysis is unchanged. This comparison uses 3,028 questions for threshold selection and 1,298 for testing, versus 2,163 and 2,163. Both allocations use the same question permutation in each split, the same 102 thresholds, confidence 95%, and error limits from 0.20 to 0.40. No new model responses or grades were collected.

All 61,812 original 50–50 bounds and all 3,030 selections/test evaluations were reproduced under both graders across the original split and 100 additional splits. Results below summarize the additional 100 separately from the original split.

## Main findings

**Gemini at α = 0.25.** With the primary grader, qualifying splits increase from 67/100 to 97/100, and mean answered percentage on common test questions increases from 12.3% to 18.7%. With the second grader, qualifying splits increase from 63/100 to 95/100. This is the clearest benefit of providing more threshold-selection data.

**Sonnet at α = 0.30.** With the primary grader, qualifying splits increase from 53/100 to 88/100, and mean answered percentage on common test questions increases from 4.9% to 8.7%. With the second grader, qualifying splits increase from 44/100 to 77/100. This is the clearest benefit of providing more threshold-selection data.

**Some decisions are unchanged.** At α = 0.30 and 0.40, Gemini selects exactly the same threshold under both allocations in every additional split, under both graders. Tighter bounds do not always change the selected decision rule.

**Low error limits remain fragile.** At α = 0.20, Gemini qualifies in only 2/100 splits under either grader with 70–30, and the observed test errors in both qualifying splits exceed 20%. Sonnet at α = 0.25 qualifies in only 7/100 splits with the primary grader and 2/100 with the second. DeepSeek has no qualifying threshold at any tested error limit in any split under either allocation or grader.

**More answers can come with higher error.** In the original split, at α = 0.30, Sonnet’s threshold changes from 0.857 to 0.760 under the primary grader. On the same 1,298 test questions, it answers 57 versus 143 questions (4.4% versus 11.0%), with errors of 19.3% versus 24.5%. Both observed errors are below 30%; the second rule answers more questions. Across additional splits at α = 0.35, Sonnet’s median error on each allocation’s own test set rises from 22.2% to 28.7% as more questions are answered.

**The evaluation tradeoff is visible.** For Gemini at α = 0.30, where the selected threshold is unchanged, the average width of the pointwise 95% test-error interval increases from 6.6 to 8.6 percentage points with the smaller test sample. This is a loss of evaluation precision, not a change in the decision rule.

**Assessment.** Keep 50–50 as the primary analysis and use 70–30 as sensitivity evidence. The results show that selection-sample size materially affects whether a useful threshold can be supported, especially for Gemini at 0.25 and Sonnet at 0.30. They do not establish that 70–30 is universally preferable or that test error should equal the chosen limit.

Occasional support losses also occur. For example, Gemini at α = 0.25 gains support in 31 primary-grader splits but loses it in one; Sonnet at α = 0.30 gains support in 36 and loses it in one. All comparisons are retained in the saved outputs.

## Repeated-split comparison

Qualified counts splits with at least one supported threshold. Answered is the mean percentage answered on the **same 1,298 test questions** under both rules, including zero for unsupported rules. Thus the answer-rate comparison is not due to different test questions. The means are descriptive because the splits overlap. Displayed rates and changes are rounded separately; changes are calculated before rounding.

### GPT-5.6 Terra (primary)

| Model | α | Qualified: 50–50 | Qualified: 70–30 | Answered: 50–50 (%) | Answered: 70–30 (%) | Change (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Gemini | 0.20 | 1/100 | 2/100 | 0.2 | 0.3 | +0.2 |
| Gemini | 0.25 | 67/100 | 97/100 | 12.3 | 18.7 | +6.4 |
| Gemini | 0.30 | 100/100 | 100/100 | 28.2 | 28.2 | +0.0 |
| Gemini | 0.35 | 100/100 | 100/100 | 60.2 | 62.6 | +2.4 |
| Gemini | 0.40 | 100/100 | 100/100 | 62.7 | 62.7 | +0.0 |
| Sonnet | 0.20 | 0/100 | 0/100 | 0.0 | 0.0 | +0.0 |
| Sonnet | 0.25 | 2/100 | 7/100 | 0.1 | 0.2 | +0.2 |
| Sonnet | 0.30 | 53/100 | 88/100 | 4.9 | 8.7 | +3.8 |
| Sonnet | 0.35 | 100/100 | 100/100 | 13.4 | 16.7 | +3.3 |
| Sonnet | 0.40 | 100/100 | 100/100 | 22.3 | 23.5 | +1.2 |
| DeepSeek | 0.20 | 0/100 | 0/100 | 0.0 | 0.0 | +0.0 |
| DeepSeek | 0.25 | 0/100 | 0/100 | 0.0 | 0.0 | +0.0 |
| DeepSeek | 0.30 | 0/100 | 0/100 | 0.0 | 0.0 | +0.0 |
| DeepSeek | 0.35 | 0/100 | 0/100 | 0.0 | 0.0 | +0.0 |
| DeepSeek | 0.40 | 0/100 | 0/100 | 0.0 | 0.0 | +0.0 |

| Model | α | Median test error: 50–50 (%) | Median test error: 70–30 (%) | Test errors above α: 50–50 | Test errors above α: 70–30 |
|---|---:|---:|---:|---:|---:|
| Gemini | 0.20 | 19.9 | 22.4 | 0/1 | 2/2 |
| Gemini | 0.25 | 18.6 | 18.3 | 0/67 | 1/97 |
| Gemini | 0.30 | 20.6 | 21.0 | 0/100 | 0/100 |
| Gemini | 0.35 | 30.1 | 30.3 | 0/100 | 0/100 |
| Gemini | 0.40 | 30.1 | 30.3 | 0/100 | 0/100 |
| Sonnet | 0.20 | — | — | 0/0 | 0/0 |
| Sonnet | 0.25 | 22.2 | 23.5 | 0/2 | 1/7 |
| Sonnet | 0.30 | 22.6 | 22.0 | 0/53 | 0/88 |
| Sonnet | 0.35 | 22.2 | 28.7 | 1/100 | 2/100 |
| Sonnet | 0.40 | 32.1 | 32.3 | 0/100 | 0/100 |
| DeepSeek | 0.20 | — | — | 0/0 | 0/0 |
| DeepSeek | 0.25 | — | — | 0/0 | 0/0 |
| DeepSeek | 0.30 | — | — | 0/0 | 0/0 |
| DeepSeek | 0.35 | — | — | 0/0 | 0/0 |
| DeepSeek | 0.40 | — | — | 0/0 | 0/0 |

Conditional-error summaries use each allocation’s own test set and only splits with returned answers; 0/0 denotes no evaluable error rate. These test-error fractions are not estimates of the guarantee’s failure probability.

### Gemini 3.5 Flash (second)

| Model | α | Qualified: 50–50 | Qualified: 70–30 | Answered: 50–50 (%) | Answered: 70–30 (%) | Change (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Gemini | 0.20 | 1/100 | 2/100 | 0.2 | 0.3 | +0.2 |
| Gemini | 0.25 | 63/100 | 95/100 | 11.5 | 18.0 | +6.6 |
| Gemini | 0.30 | 100/100 | 100/100 | 28.2 | 28.2 | +0.0 |
| Gemini | 0.35 | 100/100 | 100/100 | 60.0 | 62.6 | +2.6 |
| Gemini | 0.40 | 100/100 | 100/100 | 62.7 | 62.7 | +0.0 |
| Sonnet | 0.20 | 0/100 | 0/100 | 0.0 | 0.0 | +0.0 |
| Sonnet | 0.25 | 1/100 | 2/100 | 0.0 | 0.1 | +0.0 |
| Sonnet | 0.30 | 44/100 | 77/100 | 3.8 | 7.2 | +3.4 |
| Sonnet | 0.35 | 97/100 | 100/100 | 11.4 | 13.6 | +2.2 |
| Sonnet | 0.40 | 100/100 | 100/100 | 21.5 | 23.2 | +1.7 |
| DeepSeek | 0.20 | 0/100 | 0/100 | 0.0 | 0.0 | +0.0 |
| DeepSeek | 0.25 | 0/100 | 0/100 | 0.0 | 0.0 | +0.0 |
| DeepSeek | 0.30 | 0/100 | 0/100 | 0.0 | 0.0 | +0.0 |
| DeepSeek | 0.35 | 0/100 | 0/100 | 0.0 | 0.0 | +0.0 |
| DeepSeek | 0.40 | 0/100 | 0/100 | 0.0 | 0.0 | +0.0 |

| Model | α | Median test error: 50–50 (%) | Median test error: 70–30 (%) | Test errors above α: 50–50 | Test errors above α: 70–30 |
|---|---:|---:|---:|---:|---:|
| Gemini | 0.20 | 20.3 | 22.4 | 1/1 | 2/2 |
| Gemini | 0.25 | 18.7 | 18.4 | 0/63 | 1/95 |
| Gemini | 0.30 | 20.8 | 21.0 | 0/100 | 0/100 |
| Gemini | 0.35 | 30.2 | 30.3 | 0/100 | 0/100 |
| Gemini | 0.40 | 30.2 | 30.3 | 0/100 | 0/100 |
| Sonnet | 0.20 | — | — | 0/0 | 0/0 |
| Sonnet | 0.25 | 27.4 | 27.1 | 1/1 | 2/2 |
| Sonnet | 0.30 | 23.8 | 23.0 | 0/44 | 0/77 |
| Sonnet | 0.35 | 22.3 | 23.3 | 1/97 | 2/100 |
| Sonnet | 0.40 | 30.8 | 33.0 | 0/100 | 0/100 |
| DeepSeek | 0.20 | — | — | 0/0 | 0/0 |
| DeepSeek | 0.25 | — | — | 0/0 | 0/0 |
| DeepSeek | 0.30 | — | — | 0/0 | 0/0 |
| DeepSeek | 0.35 | — | — | 0/0 | 0/0 |
| DeepSeek | 0.40 | — | — | 0/0 | 0/0 |

Conditional-error summaries use each allocation’s own test set and only splits with returned answers; 0/0 denotes no evaluable error rate. These test-error fractions are not estimates of the guarantee’s failure probability.

## Original split: numerical comparison

The bounds below are from threshold selection; errors and exact pointwise 95% intervals are from each allocation’s own test questions. Thresholds are rounded to three decimals.

### GPT-5.6 Terra (primary)

| Model | α | Allocation | Threshold | Selection bound (%) | Test answered n (%) | Test error (%) [95% interval] |
|---|---:|---|---:|---:|---:|---|
| Gemini | 0.20 | 50-50 | — | — | 0 (0.0) | — |
| Gemini | 0.20 | 70-30 | — | — | 0 (0.0) | — |
| Gemini | 0.25 | 50-50 | 0.910 | 24.9 | 329 (15.2) | 15.5 [11.8, 19.9] |
| Gemini | 0.25 | 70-30 | 0.857 | 24.2 | 212 (16.3) | 17.5 [12.6, 23.2] |
| Gemini | 0.30 | 50-50 | 0.810 | 29.1 | 625 (28.9) | 18.6 [15.6, 21.8] |
| Gemini | 0.30 | 70-30 | 0.810 | 26.1 | 359 (27.7) | 19.8 [15.8, 24.3] |
| Gemini | 0.35 | 50-50 | 0.460 | 34.6 | 1,339 (61.9) | 28.7 [26.3, 31.2] |
| Gemini | 0.35 | 70-30 | 0.410 | 33.6 | 820 (63.2) | 31.1 [27.9, 34.4] |
| Gemini | 0.40 | 50-50 | 0.410 | 35.2 | 1,381 (63.8) | 29.8 [27.4, 32.2] |
| Gemini | 0.40 | 70-30 | 0.410 | 33.6 | 820 (63.2) | 31.1 [27.9, 34.4] |
| Sonnet | 0.20 | 50-50 | — | — | 0 (0.0) | — |
| Sonnet | 0.20 | 70-30 | — | — | 0 (0.0) | — |
| Sonnet | 0.25 | 50-50 | — | — | 0 (0.0) | — |
| Sonnet | 0.25 | 70-30 | — | — | 0 (0.0) | — |
| Sonnet | 0.30 | 50-50 | 0.857 | 29.0 | 85 (3.9) | 16.5 [9.3, 26.1] |
| Sonnet | 0.30 | 70-30 | 0.760 | 28.0 | 143 (11.0) | 24.5 [17.7, 32.4] |
| Sonnet | 0.35 | 50-50 | 0.760 | 31.2 | 234 (10.8) | 21.4 [16.3, 27.2] |
| Sonnet | 0.35 | 70-30 | 0.710 | 34.8 | 230 (17.7) | 28.3 [22.5, 34.6] |
| Sonnet | 0.40 | 50-50 | 0.560 | 39.7 | 516 (23.9) | 32.0 [28.0, 36.2] |
| Sonnet | 0.40 | 70-30 | 0.560 | 38.1 | 299 (23.0) | 32.4 [27.2, 38.1] |
| DeepSeek | 0.20 | 50-50 | — | — | 0 (0.0) | — |
| DeepSeek | 0.20 | 70-30 | — | — | 0 (0.0) | — |
| DeepSeek | 0.25 | 50-50 | — | — | 0 (0.0) | — |
| DeepSeek | 0.25 | 70-30 | — | — | 0 (0.0) | — |
| DeepSeek | 0.30 | 50-50 | — | — | 0 (0.0) | — |
| DeepSeek | 0.30 | 70-30 | — | — | 0 (0.0) | — |
| DeepSeek | 0.35 | 50-50 | — | — | 0 (0.0) | — |
| DeepSeek | 0.35 | 70-30 | — | — | 0 (0.0) | — |
| DeepSeek | 0.40 | 50-50 | — | — | 0 (0.0) | — |
| DeepSeek | 0.40 | 70-30 | — | — | 0 (0.0) | — |

### Gemini 3.5 Flash (second)

| Model | α | Allocation | Threshold | Selection bound (%) | Test answered n (%) | Test error (%) [95% interval] |
|---|---:|---|---:|---:|---:|---|
| Gemini | 0.20 | 50-50 | — | — | 0 (0.0) | — |
| Gemini | 0.20 | 70-30 | — | — | 0 (0.0) | — |
| Gemini | 0.25 | 50-50 | — | — | 0 (0.0) | — |
| Gemini | 0.25 | 70-30 | 0.857 | 24.4 | 212 (16.3) | 17.5 [12.6, 23.2] |
| Gemini | 0.30 | 50-50 | 0.810 | 29.3 | 625 (28.9) | 18.6 [15.6, 21.8] |
| Gemini | 0.30 | 70-30 | 0.810 | 26.2 | 359 (27.7) | 19.8 [15.8, 24.3] |
| Gemini | 0.35 | 50-50 | 0.460 | 34.7 | 1,339 (61.9) | 28.8 [26.3, 31.3] |
| Gemini | 0.35 | 70-30 | 0.410 | 33.7 | 820 (63.2) | 31.2 [28.1, 34.5] |
| Gemini | 0.40 | 50-50 | 0.410 | 35.3 | 1,381 (63.8) | 29.9 [27.5, 32.4] |
| Gemini | 0.40 | 70-30 | 0.410 | 33.7 | 820 (63.2) | 31.2 [28.1, 34.5] |
| Sonnet | 0.20 | 50-50 | — | — | 0 (0.0) | — |
| Sonnet | 0.20 | 70-30 | — | — | 0 (0.0) | — |
| Sonnet | 0.25 | 50-50 | — | — | 0 (0.0) | — |
| Sonnet | 0.25 | 70-30 | — | — | 0 (0.0) | — |
| Sonnet | 0.30 | 50-50 | 0.890 | 27.0 | 49 (2.3) | 22.4 [11.8, 36.6] |
| Sonnet | 0.30 | 70-30 | 0.760 | 28.7 | 143 (11.0) | 25.2 [18.3, 33.1] |
| Sonnet | 0.35 | 50-50 | 0.760 | 32.2 | 234 (10.8) | 21.8 [16.7, 27.6] |
| Sonnet | 0.35 | 70-30 | 0.760 | 28.7 | 143 (11.0) | 25.2 [18.3, 33.1] |
| Sonnet | 0.40 | 50-50 | 0.610 | 38.8 | 436 (20.2) | 30.3 [26.0, 34.8] |
| Sonnet | 0.40 | 70-30 | 0.560 | 39.0 | 299 (23.0) | 32.8 [27.5, 38.4] |
| DeepSeek | 0.20 | 50-50 | — | — | 0 (0.0) | — |
| DeepSeek | 0.20 | 70-30 | — | — | 0 (0.0) | — |
| DeepSeek | 0.25 | 50-50 | — | — | 0 (0.0) | — |
| DeepSeek | 0.25 | 70-30 | — | — | 0 (0.0) | — |
| DeepSeek | 0.30 | 50-50 | — | — | 0 (0.0) | — |
| DeepSeek | 0.30 | 70-30 | — | — | 0 (0.0) | — |
| DeepSeek | 0.35 | 50-50 | — | — | 0 (0.0) | — |
| DeepSeek | 0.35 | 70-30 | — | — | 0 (0.0) | — |
| DeepSeek | 0.40 | 50-50 | — | — | 0 (0.0) | — |
| DeepSeek | 0.40 | 70-30 | — | — | 0 (0.0) | — |

## Interpretation and reproducibility

- The objective is to support an upper limit on conditional error while answering as many questions as possible. It does not target equality between test error and α.
- More selection data can tighten bounds, but support and the selected threshold can move in either direction in a particular split. The saved paired comparisons include both gains and losses.
- Evaluating both rules on common test questions controls test-set composition. The rules can still answer different subsets of those questions.
- Smaller test sets can make test-error estimates less precise at a fixed threshold. Actual precision also depends on how many answers the selected rule returns.
- These are overlapping retrospective splits of an already examined benchmark, not independent external replications. The individual risk guarantee does not automatically extend to selecting the allocation with the best observed result.

Reproduce from the repository root with `python3 analysis/threshold_split_sensitivity.py` (requires NumPy and SciPy).

Saved outputs: `all_results.csv`, `all_bounds.csv`, `primary_results.csv`, `split_summary.csv`, `paired_comparisons.csv`, `primary_assignments.csv`, `metadata.json`, and the prespecified `protocol.md`.
