# Threshold-efficiency comparison

Specified on 2026-09-14 before the full comparison. This is exploratory work on
an already examined benchmark. The previous discussion included the primary
Holm selections and counts of changed selections over 100 partitions; those
results were already known. This is not a preregistration or fresh validation.
Do not modify the manuscript, SI, poster, or their figures/tables.

## Shared design

- Use the original parsed top-candidate probabilities and stored grades in
  `results/exploratory/alpha_risk_control_{openai,gemini}/primary_inputs.csv`.
  No new calls, grading, answer merging, or normalization.
- Retain Gemini, Sonnet, and DeepSeek; GPT-5.6 Terra is the primary grader.
- Use the saved 102-point grid, delta = 0.05, and alpha = 0.20, 0.25, 0.30,
  0.35, 0.40. Do not select grid points based on test results.
- Retain primary partition seed 20260912 and 100 additional seeds 20260913
  through 20261012. Retain nested 50-50 and 70-30 allocations. Compare methods
  on exactly the same test questions within each allocation. Also retain
  evaluation on the common 1,298 test questions for allocation comparisons.
- Reproduce all saved baseline counts, bounds, selections, and test results
  before interpreting the alternatives. Preserve input hashes.

## Deterministic threshold methods

1. `bonferroni`: current exact binomial upper limits, tail delta / 102.
2. `holm`: apply Holm's step-down procedure to exact one-sided binomial
   p-values for the null that conditional error exceeds the chosen alpha.
3. `fallback`: order thresholds from high to low. Each starts with budget
   delta / 102. On rejection, pass its available budget to the next lower
   threshold. On failure, retain that budget at the failed threshold; the
   next threshold still has its own initial budget. This is a fixed chain
   graphical testing procedure, not an assumption that risk is monotone.
4. `multistart`: start descending fixed-sequence tests at 0.1, 0.2, ..., 1.0,
   each with budget delta / 10; stop each sequence at its first failure.
   Take the union of rejected thresholds. Starts and order are not learned.

Return the lowest rejected threshold. A zero answer count has p-value one;
unsupported methods abstain everywhere. The three testing alternatives have
guarantees separately for each fixed alpha; do not claim the current joint
guarantee over every alpha for them.

## Planned partial-tie extension

If deterministic improvements are limited, evaluate both `tied_bonferroni`
and `tied_fallback` on the fixed 408-rule family: each original threshold t,
and gamma in {1, 0.75, 0.50, 0.25}. The rule answers if U > t, or if U = t
and an independent Uniform(0,1) draw is at most gamma. Empty lists abstain.
Order candidates by increasing t, then decreasing gamma; this is the order
from largest to smallest nested answer sets. No fractional binomial counts.

Generate independent auxiliary uniforms for every question using NumPy
SeedSequence([20260914, partition_seed, coin_rep]), coin_rep = 0, ..., 4.
Use the same draws across models, graders, and allocations in a partition.
The selection and test questions have disjoint draws. Repeat 0 is the primary
randomization; repeats 1-4 assess sensitivity and cannot be selected for
favorable outcomes. Future use requires fresh independent randomization.

Apply Bonferroni over all 408 rules or the same fixed-chain fallback procedure.
The former retains a simultaneous-in-alpha guarantee, now over selection
questions and auxiliary draws. The latter has a fixed-alpha guarantee.

## Reporting and checks

- Save all selected rules, selection counts, adjusted-test evidence, test
  counts, conditional errors and pointwise exact 95% intervals; also save
  primary candidate counts/bounds, paired summaries, and source hashes.
- Summaries of the additional partitions include qualification, mean answer
  rate (zero if unsupported), median conditional test error among answering
  partitions, and observed test errors above alpha. The latter is NOT an
  estimate of the population-guarantee failure probability.
- Report gains and losses; do not pick a method, seed, allocation, or grader
  per result. Keep auxiliary-randomization sensitivity separate from the
  main 100-partition summaries. These overlapping partitions and repeated
  coins are not independent new datasets.
- Check binomial-bound/p-value equivalence, baseline reproduction, nested
  answer sets, gamma=1 recovery of deterministic rules, Holm/fallback
  inclusion of Bonferroni rejections, and invariance of selection to changes
  in test grades. Handle zero answers as undefined conditional error.
- Any later method recommendation needs an explicit guarantee scope and
  independent confirmation before making new validation claims.

Reference: Angelopoulos et al., Learn then Test, Sections 2.3.1-2.3.2:
https://arxiv.org/html/2110.01052v5#S2.SS3
