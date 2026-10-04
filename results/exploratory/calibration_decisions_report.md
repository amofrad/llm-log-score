# Calibration and threshold decisions: Routes 1 and 2

Developed on September 12, 2026. The manuscript, SI, and existing paper figures
were not changed. This report evaluates a retrospective extension using the
existing SimpleQA responses.

The two routes yield one coherent procedure: estimate how top-candidate
probabilities relate to correctness, then choose answers according to the
user's penalty. With isotonic calibration, this is exactly equivalent to
maximizing the fitting sample's decision score over thresholds. Held-out
results support useful answering for Gemini at L=3, a smaller and less certain
benefit for Sonnet, and all-abstain for raw DeepSeek reports at L=3 and L=6.
The method does not establish useful answering across every model and penalty.

## How the routes connect

For the original top candidate, let U be its reported probability and Z its
correctness indicator. Estimate the model-specific curve

\[
g(u)=\Pr(Z=1\mid U=u).
\]

Under +1 for correct, −L for incorrect, and zero for abstention, answering has
conditional expected score (1+L)g(u)−L. The optimal decision based on U is

\[
\boxed{\text{answer if }g(U)\ge L/(1+L).}
\]

Route 1 fits a nondecreasing correctness curve and uses this decision rule.
Route 2 directly chooses the threshold maximizing

\[
\widehat V_L(t)=\frac{1}{n}\sum_{i=1}^n
\mathbf1\{U_i\ge t\}\,[Z_i-L(1-Z_i)],
\]

including all-abstain and resolving ties toward more answers. Empty reports
always abstain. A weighted isotonic fit pools adjacent probability groups
until their observed correctness fractions are nondecreasing. Thresholding
these fitted fractions produces the same maximizing training answer set as
Route 2. A specified step extension between observed probabilities makes the
rules agree on new scores too. Numerical cutoffs between adjacent observed
probabilities depend on this convention.

This gives a map from a user's penalty to a model-specific threshold, without
refitting for each penalty. Larger penalties produce weakly higher thresholds.
Representative labeled fitting questions are needed, but neither labels nor
another model call are needed when applying the learned rule to a new report.

The PAV/ROC convex hull connection is established in
[Fawcett and Niculescu-Mizil (2007)](https://doi.org/10.1007/s10994-007-5011-0).
This should be presented as a principled adaptation and empirical evaluation,
not a newly invented general calibration algorithm. The
[theory note](../../analysis/calibration_decisions_theory.md) gives the proofs,
the population assumptions, and a calibration-estimation regret bound.

## What this adds to the existing calibration result

For a threshold t, let c(t) be the fraction answered, r(t) the observed error
rate among answers, and e(t) their mean reported error probability. Exactly,

\[
V_L(t)=c(t)[1-(1+L)r(t)],\qquad
\widetilde V_L(t)-V_L(t)=(1+L)c(t)[r(t)-e(t)].
\]

Here the tilde denotes the decision score implied by the model's stated
probabilities. Thus the solid–dashed gap in Fig. 5 quantifies how much the
report overstates decision value, after accounting for answer rate and penalty.
This identity holds empirically on the same questions and representation;
it does not require calibrated or normalized reports.

At the original threshold t=L/(1+L), the dotted value 1−t is also the
break-even error rate for this score. Being below it can yield positive score
even if reported error is too low. This connects both visual comparisons in
Fig. 5 to the user's decision objective.

Better average calibration does not alone imply a lower optimal threshold.
That ordering follows if one model's correctness curve is pointwise above
another's, with both curves nondecreasing. Calibration also cannot manufacture
accurate candidates: a well-estimated low correctness probability can properly
lead to abstention everywhere.

## Evaluation

The design was fixed before computing this experiment. Each model uses the
same 2,163 fitting questions and 2,163 test questions, followed by 100 additional
fixed half-splits. Raw reports are primary: the parser selects the original
candidate before its stored grade is attached. Gold-informed processed reports,
which match Fig. 5, are analyzed separately. The full experiment is repeated
with the second grader. No test label chooses a method, threshold, or parameter.

The prespecified comparisons use L=0,3,6. Curves show the full L=0–10 range.
Baselines are the original threshold L/(1+L), always abstain, and the existing
answer-or-abstain prompts averaged over 50 samples per question. A fixed smooth
monotone [beta-calibration](https://proceedings.mlr.press/v54/kull17a.html) fit
is a secondary sensitivity, not a replacement selected after seeing results.
Explicit abstentions and graded not_attempted penalty responses receive zero,
following the existing analysis; all 50 outcomes reconcile for every question.

### Primary test results: raw reports, primary grader

Scores below are per question, including abstentions. Zero is the score from
always abstaining. Conditional error is undefined when no answers are returned.

| Model | L | Learned cutoff | Questions answered | Error among answers | Original score | Learned score |
|---|---:|---:|---:|---:|---:|---:|
| Gemini | 3 | 0.95 | 329/2,163 (15.2%) | 15.5% | −0.0791 | +0.0578 |
| Sonnet | 3 | 0.82 | 215/2,163 (9.9%) | 20.5% | −0.0051 | +0.0180 |
| DeepSeek | 3 | All-abstain | 0 | — | −0.4082 | 0 |
| Gemini | 6 | 0.98 | 8/2,163 (0.4%) | 12.5% | −0.0227 | +0.0005 |
| Sonnet | 6 | 0.90 | 48/2,163 (2.2%) | 20.8% | −0.0060 | −0.0102 |
| DeepSeek | 6 | All-abstain | 0 | — | −0.0550 | 0 |

Source: [all primary methods](calibration_decisions_openai/primary_metrics.csv).

The six primary score differences use nominal Bonferroni-adjusted 99.1667%
paired bootstrap intervals. These are approximate intervals conditional on
the fitted rule, not finite-sample risk guarantees.

| Model | L | Learned minus original score | Adjusted interval |
|---|---:|---:|---:|
| Gemini | 3 | +0.1368 | [+0.0712, +0.2042] |
| Sonnet | 3 | +0.0231 | [−0.0026, +0.0493] |
| DeepSeek | 3 | +0.4082 | [+0.3430, +0.4742] |
| Gemini | 6 | +0.0231 | [−0.0351, +0.0865] |
| Sonnet | 6 | −0.0042 | [−0.0180, +0.0139] |
| DeepSeek | 6 | +0.0550 | [+0.0230, +0.0912] |

Gemini at L=3 also has a pointwise 95% score interval of [+0.0324, +0.0814]
against always abstaining. Sonnet's corresponding interval is
[−0.0042, +0.0402]. DeepSeek's improvements over the original threshold come
entirely from abstaining; they do not demonstrate useful answering.

Every L=3 and L=6 learned score exceeds its penalty-prompt mean, with pointwise
paired intervals excluding zero. However, all six penalty-prompt means are
negative, so always abstaining also improves them. That comparison alone is
insufficient evidence for the value of learning a threshold. The stronger
evidence is positive score with nontrivial answering, clearest for Gemini at L=3.
All intervals and secondary contrasts are
[saved here](calibration_decisions_openai/primary_paired_intervals.csv).

### Allocation and method sensitivity

Across the 100 additional splits, the isotonic/direct rule has the following
raw-report results under the primary grader:

| Model | L | Mean test score | Mean answer rate | Splits with positive score |
|---|---:|---:|---:|---:|
| Gemini | 3 | +0.0471 | 18.85% | 100/100 |
| Sonnet | 3 | +0.0143 | 7.97% | 98/100 |
| DeepSeek | 3 | 0 | 0% | 0/100 |
| Gemini | 6 | −0.0011 | 1.42% | 85/100 |
| Sonnet | 6 | +0.0004 | 1.32% | 73/100 |
| DeepSeek | 6 | 0 | 0% | 0/100 |

These overlapping splits describe sensitivity to allocation; they are not
independent replications or a basis for treating 100/100 as a success guarantee.
Gemini's negative mean at L=6 despite mostly positive splits reflects occasional
larger losses. Sparse high-probability groups are especially unstable: the
primary raw fit's highest-probability groups have only ten Gemini observations
and three Sonnet observations, all correct in fitting. Their fitted value of
one does not establish certain correctness on new questions.

Smooth beta calibration produces similar L=3 results: repeated-split mean
scores are +0.0491 for Gemini and +0.0124 for Sonnet. At L=6 its means are
+0.0023 and +0.0030, but answer rates are only 0.35% and 0.68%.
It also abstains on all raw DeepSeek questions at both penalties. This does
not support a broad performance improvement from using a smoother curve.

The second grader preserves the central pattern: repeated-split learned
scores at L=3 are +0.0463 for Gemini, +0.0106 for Sonnet, and zero with
all-abstain for DeepSeek. Exact thresholds can change: primary Sonnet's L=3
cutoff becomes 0.88. This is another reason to avoid interpreting one fitted
cutoff as a model's universal threshold.

Processed reports look more favorable in some cases. At L=3, Gemini's primary
learned score is +0.0989 with 21.4% answered, versus +0.0578 and 15.2% for raw
reports. Processed DeepSeek instead chooses cutoff 0.94 and answers 20 test
questions, but has negative score −0.0111. These differences reinforce the need
to keep gold-informed processing separate from a rule applicable to raw reports.

## Implication for the paper

The defensible contribution is a progression from calibration diagnosis to
decision consequences and then to an implementable selection rule. A user
supplies the cost of an incorrect answer; labeled development questions map
that cost to a model-specific threshold. The observed results show where this
adds decision value and where it mainly recommends abstention.

I would retain the exact calibration-to-score identity and the unified
isotonic/direct procedure as the core development. Gemini at L=3 is the
clearest empirical example, with all penalties and models reported alongside
it. The evidence does not support claiming a broadly superior new algorithm,
guaranteed error control, or useful high-penalty answering for every model.
The earlier difficulty certifying low error therefore remains compatible with
these results; changing the objective permits different decisions, not more
accurate underlying candidates.

If incorporated, the identity follows naturally after the interpretation of
Fig. 5. The procedure and full comparison could then appear in the SI, with
a short main-text statement of the held-out result and its limits. They are
currently separate review artifacts.

## Figures and verification

- [Raw-report decision scores and answer rates](calibration_decisions_openai/decision_scores_raw.pdf)
- [Raw-report fitted calibration maps](calibration_decisions_openai/calibration_maps_raw.pdf)
- [Processed-report decision scores and answer rates](calibration_decisions_openai/decision_scores_processed.pdf)
- [Primary-grader split summary](calibration_decisions_openai/split_summary.csv)
- [Second-grader split summary](calibration_decisions_gemini/split_summary.csv)
- [Fixed protocol](../../analysis/calibration_decisions_protocol.md)
- [Theory and proofs](../../analysis/calibration_decisions_theory.md)

Twenty-five analysis tests passed. Independent checks reconstructed all 180
primary model/representation/method/penalty metrics from saved question-level
inputs. All 4,812 recorded isotonic/direct threshold comparisons agree, and
thresholds are weakly increasing in penalty. The six full-sample processed
calibration comparisons reproduce current Table S8. Fitting and test IDs are
disjoint; changing test labels cannot change either fitted map or its thresholds.
Input hashes, fit parameters, predictions, and method settings are retained.
See the [verification record](calibration_decisions_verification.json).
