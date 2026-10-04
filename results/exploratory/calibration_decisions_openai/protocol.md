# Calibration and penalty-based threshold decisions

Design fixed on 2026-09-12 before computing this experiment. This is a
retrospective extension on an already studied benchmark, not preregistration or
new external validation. The objective is expected score under +1 for correct,
-L for incorrect, and 0 for abstention. It is distinct from certifying a fixed
conditional error rate. Do not change the manuscript, SI, or existing figures
during this development step.

## Theory and algorithms

Develop the calibration-to-score identity, the optimal rule based on a
model-specific correctness curve, monotonicity of score-optimal thresholds in
L, and the equivalence of isotonic calibration and empirical threshold-score
maximization. These specialize established calibration and cost-sensitive
classification results; do not claim a new general learning principle.

Use three report decision methods and two external baselines:

1. Original probability threshold L/(1+L).
2. The unified isotonic/direct empirical-score rule (primary). Group tied U
   values; fit weighted isotonic regression to group correctness fractions.
   Include blocks whose fitted correctness is at least L/(1+L). Independently
   maximize empirical correct count - L * incorrect count over all distinct
   training-score thresholds, plus zero and the all-abstain action, breaking
   score ties toward the smallest threshold. Verify both constructions agree.
   When all observed groups qualify, use threshold zero; when none qualify,
   abstain everywhere. For predictions between observed training scores, extend
   the isotonic fit as a right-continuous step function taking the value at the
   greatest observed score no larger than U, clipped at the observed endpoints.
   This convention makes the two routes agree outside the training sample too.
3. A smooth monotone beta-calibration sensitivity: sigmoid(a log U - b log(1-U)
   + c), with a,b >= 0, probabilities clipped to [1e-6,1-1e-6], and mean binary
   log loss plus 0.001/2 times ((a-1)^2+(b-1)^2+c^2). This fixed regularizer
   shrinks toward the identity map. No hyperparameter tuning or method choice
   uses test outcomes.
4. Separately elicited answer-or-abstain prompts at L=0,3,6, averaging their
   existing 50 samples per question. Explicit abstentions and responses graded
   not_attempted both receive zero, as in the existing three-way analysis;
   verify that these plus correct and incorrect counts reconcile to 50.
5. Always abstain, with score zero and answer rate zero.

Penalties 0,3,6 are the tabulated comparisons. Curves use all penalties from
0 through 10 in increments of 0.1, with no selection of favorable penalties.
The empirical maximizer includes all-abstain. Its nonnegative training score
does not guarantee nonnegative test score. Report answer rates as well as
scores so that improvements obtained mainly by abstaining are visible.

## Data and separation

Use the three retained models and the same 4,326 question IDs. Analyze two
representations separately throughout:

- Raw (primary): original parser, fixed non-answer strings, and stable entry
  ties; select the candidate and its probability before recovering its stored
  grade, using threshold_selection.py. No answer key is used to make a future
  decision. As in that existing analysis, a selected string graded not_attempted
  is an unsuccessful returned answer.
- Processed (secondary): retained graded/canonicalized candidates used by
  Fig. 5, with graded not_attempted entries excluded and stable entry ties.
  This is gold-informed benchmark processing, not deployment selection.

Sort IDs and use the same half-split as the earlier threshold experiment:
seed 20260912, first 2,163 questions for fitting, remaining 2,163 for evaluation.
Run 100 additional consecutive seeds. All models, methods, representations,
and graders share split membership. Fit anew on each split; no test label
affects a calibration map or selected threshold. Repeat the full analysis for
the second grader without changing settings. Never pool raw and processed
observations or treat representations/graders as independent replications.

## Endpoints

Report per-question mean decision score, answer rate, error among answered
questions, and selected thresholds. The primary contrast is unified
isotonic/direct score minus the original threshold score on raw reports at
L=3 and L=6 for each of the three models. Use 2,000 paired question-bootstrap
draws on the primary test split, with 99.1667% percentile intervals to adjust
over these six comparisons. These intervals condition on fitted rules and the
observed report per question; they do not measure fitting uncertainty.

Also report pointwise 95% paired intervals against the penalty-prompt mean and
always-abstain. These are secondary comparisons, not a joint guarantee. The
question-level bootstrap retains the 50-sample penalty mean as one paired
observation. Repeated splits describe allocation sensitivity and are not 100
independent experiments. Report all methods and targets, all-abstain outcomes,
and negative as well as positive test scores.

Compute calibration-to-score discrepancies for the original thresholds using
the identity in the theory note. Save full per-question primary inputs and
scores, both threshold constructions, isotonic blocks, beta parameters, every
split's metrics, primary bootstrap contrasts, input hashes, and software
versions. Keep all exploratory outputs outside the paper figure directory.

References: Fawcett and Niculescu-Mizil (2007), PAV and the ROC convex hull,
https://doi.org/10.1007/s10994-007-5011-0; Kull et al. (2017), beta calibration,
https://proceedings.mlr.press/v54/kull17a.html.
