# Rich probability-report decision rule: analysis plan

Written on 2026-09-12 before fitting or evaluating the richer rules. This is a
retrospective extension on a benchmark whose aggregate results have already been
examined; it is not a preregistration or a new external validation dataset.

## Question and comparison

Does the probability report contain useful information about correctness beyond
its largest concrete-answer probability? The candidate answer itself stays the
same: select the largest-probability concrete entry using the existing parser,
fixed non-answer strings, and stable entry-order ties. Extract all features before
attaching stored correctness labels. Never use the answer key, question identity,
answer text, graded candidate groups, or other models' outputs as predictors.

Use the three retained models and every available benchmark question. Empty
candidate lists always abstain. Fit a separate correctness predictor per model.
Predictors are top probability U, listed IDK probability, top-two concrete
probability gap (second probability zero for a singleton), log(1 + candidate
count), total listed probability including IDK, normalized entropy of the listed
probability vector, and top probability divided by total concrete probability.
For entropy only, divide entries by their listed total and divide entropy by the
log of the number of positive entries (zero for at most one positive entry).
This feature normalization does not change candidate selection or probabilities
used by the original rule. Do not assign unallocated probability to IDK.

Four prespecified rules:

1. Raw U, without fitting.
2. Gradient boosting using U alone (information ablation).
3. Standardized L2 logistic regression using all features, C=1.
4. Gradient boosting using all features (primary richer rule).

Both boosting rules use identical fixed settings: histogram gradient boosting,
150 iterations, learning rate 0.05, at most 7 leaves, depth at most 3, minimum 40
training observations per leaf, L2 regularization 10, and no early stopping or
class weighting. There is no hyperparameter search or choice of winning learner
using certification or test outcomes. Keep every rule in the results.

## Separation of fitting, certification, and evaluation

Sort question IDs and permute them with a fixed seed, shared across models. Use
50% development, 25% certification, and the remainder for test (2,163 / 1,081 /
1,082 for 4,326 questions). Primary seed: 2026091201. Repeat the entire procedure
for 100 additional consecutive seeds. These overlapping splits describe
sensitivity to allocation; they are not independent replications.

On development questions, obtain three-fold out-of-fold predictions for each
learned rule, and then refit that rule on all nonempty development reports.
Use out-of-fold scores and development labels only to order the fixed threshold
grid {0, .01, ..., 1} union {6/7}. For each target, sort thresholds by ascending
one-sided binomial p-value for the null that conditional error is at least that
target; break ties by larger answered count, then smaller threshold. Final fitted
score functions and all testing orders are fixed before certification labels
are used. Out-of-fold scores determine order only, not the certificate.

On certification questions, use exact binomial upper bounds and fixed-sequence
testing: stop at the first failed test; select the smallest threshold among
preceding passes; abstain everywhere if none pass. Target errors remain 0.10,
1/7, 0.20, and 0.25. The error budget is 0.05/4 per sequence, yielding a 95%
guarantee simultaneously across these four prespecified targets, separately
for each fixed model, rule, and split. It is not simultaneous across rules,
models, or repeated splits. A secondary comparison uses the original
Bonferroni correction over 102 thresholds on the same certification data.

Conditional on the development data, the fitted score and sequence are fixed.
If a sequence certifies any threshold with true conditional error above its
target, it must pass the first such threshold in that sequence. Its exact
binomial test has error probability at most 0.05/4, even though tests share
questions. A union bound over the four targets gives 0.05. The guarantee assumes
independent questions from the same distribution and the recorded grader's
correctness labels; it does not bound every realized finite test error rate.

## Endpoints and reporting

Primary information comparison: test area under the risk-coverage curve (AURC)
for the richer boosting rule minus the U-only boosting rule, conditional on
nonempty reports. Lower is better. Compute the empirical discrete AURC averaged
over all answer counts, with expectation over uniform random ordering inside
score ties. Give paired question-bootstrap intervals (2,000 draws, primary seed
+ 1000), with Bonferroni coverage across the three model comparisons. These
intervals condition on fitted rules and measure test-question sampling only.

Also report test AUROC, Brier score, log loss, error at fixed 10%, 20%, and 40%
answer rates, certified answer rates and errors, and support frequencies across
all repeated splits. Include zeros for unsuccessful certification when
summarizing answer rate; conditional error remains undefined when no questions
are answered. Fixed-coverage diagnostics randomize within boundary ties and
are descriptive, not certified operating rules. Test error intervals for
selected rules are pointwise two-sided 95% exact binomial intervals.

Retain primary per-question scores and split memberships, fitted estimators,
all split results, testing traces, input hashes, software versions, and this
plan. Report disappointing or inconsistent outcomes as well as improvements.
Do not edit the manuscript or existing paper figures during this experiment.

Method reference: Angelopoulos et al., Learn then Test, including split fixed
sequence testing: https://arxiv.org/html/2110.01052v4 (Sections 2.3.3 and Appendix C).
