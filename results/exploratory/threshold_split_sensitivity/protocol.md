# Sensitivity to the allocation of threshold-selection and test questions

Specified on 2026-09-13, before computing the 70–30 results, following the user's
request. This is a sensitivity analysis on an already examined benchmark; it
is not a new external validation. Keep the 50–50 primary analysis, manuscript,
SI, poster, published tables, and figures unchanged.

## Fixed design

- Use the original parsed-report top candidates and stored grades from the
  existing analysis (`primary_inputs.csv`, raw representation). No new model
  calls, grading, answer merging, or probability normalization.
- Include Gemini 3.5 Flash, Claude Sonnet 4.6, and DeepSeek V3.2. Analyze both
  GPT-5.6 Terra (primary grader) and Gemini 3.5 Flash independently.
- Retain the exact 102-point grid saved in the original metadata: the 101
  hundredth-spaced thresholds plus the penalty-6 threshold (about 0.857).
- Retain delta = 0.05 and alpha = 0.20, 0.25, 0.30, 0.35, 0.40. Select the
  smallest threshold whose one-sided Clopper–Pearson upper bound, with tail
  delta / 102, is at most alpha. No supported threshold means abstention on
  every question; conditional error is undefined.
- Sort the same 4,326 question IDs and permute with NumPy's default_rng using
  primary seed 20260912 and 100 additional seeds 20260913 through 20261012.
  Use the same permutation across models, graders, and allocations.
- The first 2,163 questions select the 50–50 rule; the remaining 2,163 test it.
  The first floor(0.7 * 4,326) = 3,028 questions select the 70–30 rule; the
  remaining 1,298 test it. Thus selection samples are nested.
- In addition to each allocation's own test results, evaluate both selected
  rules on the common last 1,298 questions, which neither selection sample
  uses. This compares rules on identical test questions.

## Verification and outputs

Verify source-response hashes against the original analysis metadata. Recompute
all 102 bounds and all five reported alpha levels for the 50–50 method under
both graders in all 101 splits, and require agreement with saved results before
reporting sensitivity findings. Verify that test labels do not affect selection.

Save every bound, selected rule, selection/test count, exact two-sided 95% test
error interval, minimum supported alpha, and primary split assignment. Summarize
the 100 additional splits separately from the primary split. Report qualifying
split counts, mean answer rates including zeros, conditional error among splits
with answers, and paired changes on the common test questions. Include support
gains and losses, rather than retaining only improvements.

Overlapping splits are descriptive sensitivity checks, not 100 independent
experiments. Do not attach an independence-based significance test to these
split summaries. A test error above alpha is a finite-sample observation, not
by itself a violation of the guarantee for the underlying conditional risk.
The guarantee applies separately to a fixed allocation, model, and grader; it
does not automatically cover choosing an allocation after examining outcomes.
