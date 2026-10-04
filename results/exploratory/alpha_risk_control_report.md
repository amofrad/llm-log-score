# Error-controlled threshold selection

September 12, 2026. The manuscript, SI, and existing figures are unchanged.
This is a retrospective analysis of the existing SimpleQA reports.

The implementation now takes an error tolerance alpha and returns the lowest
supported reported-probability cutoff, together with its upper error bound.
If the evidence is insufficient, it returns no threshold and abstains
everywhere. At the original 10%–25% tolerances, raw-report support is limited
and Gemini's primary 25% certificate changes under the second grader. The
subsequently requested 30%, 35%, and 40% levels reveal a clearer tradeoff:
Gemini has consistent support, Sonnet becomes more consistently supported at
35%–40%, and DeepSeek remains unsupported. The calibration-guided extension
does not produce a robust improvement over using the full certification sample.

## What a user receives

For each candidate cutoff t, count the returned answers and their errors on
labeled certification questions. Construct one-sided exact binomial upper
bounds, adjusted for searching the candidate family. The lookup is

\[
\boxed{\widehat t(\alpha)=\min\{t:B_{.05}(t)\le\alpha,\ N_t>0\}.}
\]

The bound concerns error among returned answers. At 95% confidence it covers
the candidate thresholds simultaneously, so the same bound table can answer
arbitrary alpha requests without refitting or another correction over alpha.
This confidence scope is separate for each model and method, under independent
questions from the same distribution and the recorded grader's correctness
labels. It does not cover post-hoc choice of the best model/method or guarantee
the error fraction in every finite test sample.

This is an application of established simultaneous risk control; see
[Learn then Test](https://arxiv.org/html/2110.01052v4) and the
[local derivation](../../analysis/alpha_risk_control_theory.md).

## Methods and independent data roles

All methods share 2,163 final test questions. Primary split seed: 20260912.

| Method | Development questions | Certification questions | Candidate cutoffs |
|---|---:|---:|---|
| Full fixed grid — primary reference | 0 | 2,163 | Fixed 102-point grid |
| Learned isotonic grid — extension | 1,081 | 1,082 | Zero and development-fitted isotonic step starts |
| Smaller-sample fixed grid — comparison | 0 | 1,082 | Same fixed 102-point grid |

The first method reproduces our earlier strict-threshold experiment. The new
comparison asks whether the fitted calibration curve can reduce the number of
cutoffs requiring certification. It constructs that smaller family using only
development labels; its subsequent confidence bounds use independent questions.
In the primary raw fits, the learned families have 10, 17, and 9 thresholds for
Gemini, Sonnet, and DeepSeek. The fitted curve need not be correctly calibrated
for independent certification to be valid.

We retained raw and gold-informed processed representations separately and
repeated every analysis under both graders. Another 100 fixed splits describe
allocation sensitivity. No test outcomes select thresholds or tune the methods.
The [protocol](../../analysis/alpha_risk_control_protocol.md) fixes the initial
targets, methods, sample allocation, and confidence scope. Its amendment records
the user's subsequent request for 30%, 35%, and 40%, with methods unchanged.

## Primary results on raw reports

The original error tolerances are 10%, 1/7 (about 14.3%), 20%, and 25%.
The additional requested levels appear below; all original results are retained.

| Model | alpha=10% | alpha=14.3% | alpha=20% | alpha=25% |
|---|---|---|---|---|
| Gemini | No supported cutoff | No supported cutoff | No supported cutoff | t=0.91 |
| Sonnet | No supported cutoff | No supported cutoff | No supported cutoff | No supported cutoff |
| DeepSeek | No supported cutoff | No supported cutoff | No supported cutoff | No supported cutoff |

This table uses the full fixed grid and primary grader. For Gemini at alpha=25%:

- Certification: 56 errors among 327 answers; simultaneous upper bound 24.88%.
- Independent test: 51 errors among 329 answers, or 15.5% error
  (pointwise exact 95% interval: 11.8%–19.9%).
- Questions answered: 329/2,163, or 15.2%.

The cutoff 0.91 returns the same observed Gemini answers as the preceding
penalty-based cutoff 0.95: the raw probabilities at or above 0.90 are only
0.90, 0.95, and 0.98. The numerical cutoff difference does not represent an
accuracy improvement. On future scores between these values, the rules could
make different decisions.

Neither smaller-sample method supports any of the four targets for any raw
model on this primary split. The learned grid improves Gemini's best bound
from 28.31% to 25.78% compared with the same-size fixed grid, but still does
not meet 25%. The full 2,163-question reference performs better here.

Source: [all primary results](alpha_risk_control_openai/primary_results.csv).

## Sensitivity to grading and allocation

Under the second grader, Gemini has 57 rather than 56 certification errors at
t=0.91. Its bound rises from 24.88% to 25.22%, so the primary 25% certificate
no longer passes. None of the raw models is supported at the four targets on
that split under that grader. This is material sensitivity to a single grade,
not robust evidence of satisfying a sharp 25% requirement.

Across the 100 additional splits, the numbers supporting alpha=25% are:

| Model | Full fixed grid | Learned isotonic grid | Smaller-sample fixed grid |
|---|---:|---:|---:|
| Gemini, primary grader | 67/100 | 43/100 | 19/100 |
| Sonnet, primary grader | 2/100 | 3/100 | 2/100 |
| DeepSeek, primary grader | 0/100 | 0/100 | 0/100 |
| Gemini, second grader | 63/100 | 43/100 | 15/100 |
| Sonnet, second grader | 1/100 | 3/100 | 1/100 |
| DeepSeek, second grader | 0/100 | 0/100 | 0/100 |

For primary-grader Gemini at 25%, mean test answer rates, including zeros for
unsupported splits, are 12.30%, 8.28%, and 3.17%, respectively. Thus the learned
grid helps relative to the same certification sample size, but using more
questions for certification remains more effective in this comparison.

At alpha=20%, Gemini is supported in 1/100 full-grid splits and 3/100
learned-grid splits under either grader; the smaller-sample fixed grid supports
none. No raw model is supported at 10% or 14.3% in any of these additional
splits. Sonnet and DeepSeek are also unsupported at 20% throughout.

These overlapping splits are not independent replications. A finite test error
fraction can exceed alpha even for a supported rule: this happens in one
second-grader Sonnet split at 25% for each method. The guarantee concerns the
underlying conditional error rate and uncertainty over certification samples;
it is not a deterministic limit on every test sample.

## What larger tolerances reveal

The primary bound table is valid for arbitrary alpha. The smallest tolerance
it can support with the full fixed grid is:

| Model | Primary grader | Second grader |
|---|---:|---:|
| Gemini | 24.88% | 25.22% |
| Sonnet | 26.95% | 26.95% |
| DeepSeek | 72.36% | 72.36% |

These are minima of the computed upper bounds, not lower bounds on the models'
true achievable error rates. Insufficient data and conservative search
adjustments can prevent certification of a useful rule. Increasing the allowed
error does eventually permit more answers; the saved lookup curves show that
tradeoff over 5%–80%. The original four tolerances remain in the tables along
with the three user-requested levels. The latter were added after reviewing
the first results and are explicitly identified as a subsequent extension.

Processed reports yield more support, but their candidate selection uses
gold-informed processing. On the primary split at 25%, the full grid supports
processed Gemini at t=6/7 (upper bound 24.05%; 462 test answers; 13.4% test
error), and Sonnet at t=0.89 (upper bound 22.61%; 62 answers; 16.1% error).
Neither has primary support at the three stricter targets, and DeepSeek has
none at all four. These benchmark results should remain separate from a
threshold intended for original raw reports.

## Additional requested levels: 30%, 35%, and 40%

The primary raw-report results below use the full fixed grid, primary grader,
and the same 2,163 test questions. The confidence-bound construction is unchanged.

| Model | Allowed error alpha | Cutoff t | Certification upper error bound | Questions answered | Observed test error |
|---|---:|---:|---:|---:|---:|
| Gemini | 30% | 0.81 | 29.15% | 625 (28.9%) | 18.6% |
| Gemini | 35% | 0.46 | 34.61% | 1,339 (61.9%) | 28.7% |
| Gemini | 40% | 0.41 | 35.25% | 1,381 (63.8%) | 29.8% |
| Sonnet | 30% | 6/7 | 28.97% | 85 (3.9%) | 16.5% |
| Sonnet | 35% | 0.76 | 31.23% | 234 (10.8%) | 21.4% |
| Sonnet | 40% | 0.56 | 39.71% | 516 (23.9%) | 32.0% |
| DeepSeek | 30%, 35%, 40% | No supported cutoff | — | 0 | Undefined |

These are maximum allowed errors, not error rates the method tries to attain.
The observed error and confidence bound can both lie below alpha. Cutoffs can
change abruptly because reported probabilities are discrete and answer sets
change in groups. Also, this rule bounds average error across the returned
answers; it does not require each probability group to have at least 1-alpha
correctness. That is why the reported cutoff can be below 1-alpha, as for
Gemini at 35%.

For the full fixed grid, support across the 100 additional splits is:

| Model / grader | alpha=30% | alpha=35% | alpha=40% |
|---|---:|---:|---:|
| Gemini / primary | 100/100 | 100/100 | 100/100 |
| Gemini / second | 100/100 | 100/100 | 100/100 |
| Sonnet / primary | 53/100 | 100/100 | 100/100 |
| Sonnet / second | 44/100 | 97/100 | 100/100 |
| DeepSeek / either | 0/100 | 0/100 | 0/100 |

Primary-grader mean answer rates, including zero for unsupported splits, are
28.3%, 60.4%, and 62.8% for Gemini, and 4.9%, 13.4%, and 22.3% for Sonnet,
at 30%, 35%, and 40%, respectively. These summarize overlapping allocations;
100/100 is not an independent replication count or a certainty guarantee.

The second grader selects the same primary Gemini cutoffs at all three levels,
with test errors 18.6%, 28.8%, and 29.9%. Sonnet's primary cutoffs are 0.89,
0.76, and 0.61 under that grader, with answer rates 2.3%, 10.8%, and 20.2%.
Thus the broader pattern persists, although Sonnet's exact cutoff is less stable.

The learned grid still helps compared with a fixed grid on the same smaller
certification sample. For example, at 30% primary-grader Gemini support rises
from 84/100 to 96/100, and at 35% Sonnet support rises from 59/100 to 79/100.
The full certification sample gives 100/100 in both comparisons. Under the
second grader those learned-grid improvements are 82 to 95 and 49 to 69,
respectively, while the full sample gives 100 and 97. All methods and both
representations remain in the CSVs; no method is selected afterward for a
combined guarantee.

The extension is informative because it identifies where an acceptable error
tolerance leads to a useful number of answers. For Gemini, increasing alpha
from 30% to 35% substantially expands answering, whereas 35% to 40% adds much
less in the primary split. Sonnet benefits at higher tolerances but answers
fewer questions. The limited support at the stricter original tolerances
remains part of the result.

## Interpretation

The formulation answers the user's operational question: given an error
tolerance, it supplies a supported cutoff or reports that available evidence
cannot support one. It does not require assuming reported probabilities are
calibrated, and the actual error curve need not decrease monotonically.

At 10%–25%, the evidence supports only a narrow and grader-sensitive example.
At 30%–40%, support is much more consistent for Gemini and, at the higher
levels, Sonnet. DeepSeek remains unsupported. This gives a useful description
of the model-specific tradeoff between allowed error and answering frequency.
The learned-grid extension reduces the search cost, but using more questions
for certification generally remains preferable in this experiment. These
findings remain compatible with the previous positive decision-score result:
maximizing score can be useful without establishing a strict low-error guarantee.

I would retain this as exploratory validation for now. Its value is a precise
decision procedure and a clear account of when the data support using it. It
does not currently justify a broad claim of reliable low-error deployment.

## Reproduction and artifacts

From the repository directory:

```bash
python analysis/alpha_risk_control.py
python analysis/alpha_risk_control.py --grader gemini
python analysis/alpha_risk_control.py --lookup results/exploratory/alpha_risk_control_openai --alpha 0.25
```

The lookup can be restricted with `--model gemini35flash`; `--method
isotonic_grid` selects the separately evaluated learned-grid method. It reads
certification bounds only and does not use test labels. Defaults are raw
reports and the full fixed-grid method.

- [Primary bounds used by the lookup](alpha_risk_control_openai/primary_bounds.csv)
- [Raw-report tolerance/threshold/answer-rate plot](alpha_risk_control_openai/alpha_threshold_raw.pdf)
- [Processed-report plot](alpha_risk_control_openai/alpha_threshold_processed.pdf)
- [Primary-grader split summary](alpha_risk_control_openai/split_summary.csv)
- [Second-grader split summary](alpha_risk_control_gemini/split_summary.csv)
- [Minimum supported tolerances](alpha_risk_control_openai/minimum_supported_alpha.csv)
- [Theory and guarantee](../../analysis/alpha_risk_control_theory.md)
- [Verification](alpha_risk_control_verification.json)

All 30 analysis tests passed. Independent checks reconstructed 2,589 primary
bounds and 252 selected-rule test summaries. The 12 primary raw reference
rows match the archived strict-threshold experiment. Tests verify that
certification labels cannot change the learned candidate grid, test labels
cannot change certification or decisions, and exact simultaneous coverage
holds in an exhaustively enumerated small example. Source and input hashes
match the saved metadata. Every split's bounds and unsuccessful outcomes are
retained. The alpha extension preserves all original-target results, all
certification bounds, and all primary inputs exactly.
