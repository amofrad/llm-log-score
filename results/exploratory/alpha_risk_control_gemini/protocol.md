# Selecting a supported threshold from an error tolerance

Design fixed on 2026-09-12 before computing this extension. This is retrospective
work on an extensively examined benchmark, not fresh external confirmation.
Keep manuscript, SI, existing figures, and previous experiments unchanged.

Amendment on 2026-09-12: after reviewing the original four tabulated targets,
the user requested alpha=.30, .35, and .40. Add these to the primary tables and
100 repeated-split comparisons for every method, model, representation, and
grader. They were already within the saved primary lookup curve. Preserve the
original four results and all method settings, data splits, grids, and bounds.
This is an explicitly subsequent extension, not part of the original target
selection. Simultaneous bounds cover these additional alpha levels without
another multiplicity adjustment or refitting requirement.

## Objective and guarantee

Given alpha, return the smallest candidate threshold whose simultaneous upper
confidence bound on error among returned answers is at most alpha. Use delta=.05.
If none is supported, return no threshold and abstain everywhere; conditional
error is then undefined. The guarantee is simultaneous over all candidate
thresholds and all alpha in (0,1), separately for each model, representation,
method, and split. It is not simultaneous over choosing methods or models.

For M thresholds fixed independently of the certification observations, use
one-sided exact binomial (Clopper-Pearson) bounds with tail delta/M. Zero
certification answers have bound one and cannot support a rule. Conditional
on any separate development data, a union bound gives simultaneous coverage.
This assumes independent questions from the same distribution and treats
recorded grader labels as the correctness outcome. It does not guarantee the
error fraction in every finite test sample or under distribution shift.

## Comparisons

Use three methods, fixed before calculating their results:

1. fixed_full (primary/reference): the existing fixed 102-point grid
   {0,.01,...,1} union {6/7}, using the full 2,163-question certification half.
   This reproduces the earlier strict-threshold experiment, not a new method.
2. isotonic_grid (extension): split that half into 1,081 development and 1,082
   certification questions. Fit weighted isotonic correctness versus raw U
   only on development questions. Candidate cutoffs are zero and the observed
   U values where the fitted step function strictly increases. Pool equal
   adjacent fitted levels when extracting cutoffs. These are all distinct
   finite decisions of the fitted isotonic penalty rule. Do not add cutoffs
   after seeing certification or test labels. Certify every candidate with
   delta divided by this development-fixed grid size.
3. fixed_half (sample-size comparison): use the same 1,082 certification
   questions as method 2 but retain the full fixed 102-point grid. This
   separates the smaller candidate family from reduced certification data.

The extension can reduce the multiple-search penalty, but needs development
data and may omit a useful cutoff. Do not choose the best method afterward and
claim the individual method's guarantee covers that selection.

## Data and splits

Reuse the existing raw parser, fixed non-answer filter, stable top-entry ties,
and stored grade recovery from threshold_selection.py. Candidate selection is
independent of its grade. A returned entry graded not_attempted counts as an
unsuccessful answer, matching earlier raw-report experiments. Processed reports
from calibration_decisions.py are a separate gold-informed sensitivity.

Sort the same 4,326 question IDs and permute with seed 20260912. First 2,163
form the full certification half; last 2,163 are the common final test set.
Within the first half, its first 1,081 permuted IDs are development and its
remaining 1,082 are certification for the two split methods. Use the same IDs
for all models, representations, and graders. Repeat with 100 consecutive
additional seeds. Fit and certify afresh each time. These overlapping splits
describe allocation sensitivity and are not independent replications.

Repeat the entire fixed analysis under both graders, without changing settings.
Raw reports and the primary grader are primary. No new model or grading calls.

## Outputs and interpretation

Originally tabulate alpha=.10, 1/7, .20, .25; the amendment adds .30, .35, .40.
A primary lookup curve uses .05 through .80
in increments of .005, plus 1/7. Since the bounds are simultaneous, the saved
bound table also supports arbitrary alpha without refitting. Include the
unsupported status, selected cutoff, certification counts and upper bound,
test answer count/rate, conditional error, and pointwise two-sided exact 95%
test interval. Compare the unvalidated threshold 1-alpha on the same test IDs.

Report support frequency, mean answer rate including zeros, conditional error
only where answers exist, and the smallest alpha supported by each primary
bound table. Do not interpret failure to certify as proof that no useful
population threshold exists. Save every bound, split metric, input hash,
development grid, primary input/split ID, and code version. Reproduce the
existing primary raw fixed-grid results as a verification check.

Method reference: Angelopoulos et al., Learn then Test, especially simultaneous
risk bounds and data splitting: https://arxiv.org/html/2110.01052v4 .
