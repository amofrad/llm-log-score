# Beyond a Single Answer: Proper Scoring of LLM Uncertainty

Code and results for:

> *Beyond a Single Answer: Proper Scoring of LLM Uncertainty*
>
>  Ali Kaazempur-Mofrad, Xiaowu Dai, and Xuming He

## Quick start

Use Python 3.10 or newer.

The graded model outputs behind the paper's figures and tables are included
under `results/`, so reproduction needs no API access:

```bash
pip install -r requirements.txt
python analysis/reproduce_all.py
```

This rebuilds the computed figures, tables, and audits into `results/figures/`
from `results/graded_by_openai/`; Fig. 1 is supplied as a schematic. Use
`--grader gemini` to reproduce the main analyses under the second grader.
The sensitivity analyses use the primary grader only. Bootstrap resampling
uses fixed seeds for reproducibility. The paper analyses include Gemini 3.5 Flash,
Claude Sonnet 4.6, and DeepSeek V3.2.

**Terminology.** A probability report lists candidate answers and IDK with
probabilities. Report-based decisions (**RBD**) threshold its highest candidate
probability. **EPP** asks the model to answer or abstain under payoffs +1 for a
correct answer, −L for an incorrect answer, and 0 for abstention. Top-0.9 sets
are constructed from the report, including IDK in the ranking. Their three
outcomes are coverage (the correct answer is present), miscoverage with IDK,
and miscoverage without IDK. IDK alone does not supply a correct answer.

For an isolated rebuild that keeps all generated intermediates out of the
published results, use:

```bash
python analysis/reproduce_all.py --out-dir /tmp/llm-report-reproduction
```

`--skip-figures` regenerates tables and numerical audits; `--skip-tables`
regenerates figures and their supporting data. Figure 1 is a supplied schematic,
not a computed plot. All analysis commands use the saved outputs; model and
grading APIs are needed only for collecting or grading new responses.

## Repository structure

```
llm-log-score/
├── runner/                  data collection
│   ├── engine.py            API clients, caching, prompts, scoring, grading
│   └── run.py               subcommands: simpleqa, consistency, batch
├── grading/                 post-hoc grading
│   ├── graders.py           both graders and the shared rubric
│   └── grade.py             grade a run tree into a parallel graded tree
├── analysis/                figures and tables
│   ├── common.py            loaders and post-hoc decision helpers
│   ├── make_figures.py
│   ├── make_tables.py
│   ├── comparison_uncertainty.py  Fig. 3C and Table S2 intervals
│   ├── paper_tables.py           all numbered table exports
│   ├── rho_sensitivity.py   Fig. S4 and Table S4
│   ├── rule_sensitivity.py  Fig. S5
│   ├── threshold_calibration.py  Fig. 5 and Table S8
│   ├── threshold_selection.py    Raw parsing and exact binomial bounds
│   ├── alpha_risk_control.py     Error-controlled threshold lookup
│   ├── make_alpha_paper_tables.py Table S9 and archival comparisons
│   ├── threshold_split_sensitivity.py 50–50 versus 70–30 analysis
│   ├── make_split_sensitivity_table.py Table S10
│   └── reproduce_all.py     rebuild computed figures, tables, and audits
├── results/
│   ├── graded_by_openai/    primary grading, including sensitivity runs
│   ├── graded_by_gemini/    second grader
│   └── figures/             the paper's figures, numerical data, and audits
│       └── tables/          Tables 1–2 and S1–S11 as CSV and LaTeX
├── README.md
└── requirements.txt
```

## Data

All experiments use the SimpleQA factuality benchmark: 4,326 short-answer
factual questions with adjudicated gold answers (Wei et al., 2024, "Measuring
short-form factuality in large language models", arXiv:2411.04368). The runner
loads it from the Hugging Face Hub as
[`OpenEvals/SimpleQA`](https://huggingface.co/datasets/OpenEvals/SimpleQA) via
the `datasets` library; no copy is stored in this repository. Question ids
`simpleqa-0` through `simpleqa-4320` are the dataset's `test` split in row
order, and `simpleqa-4321` through `simpleqa-4325` are its five `few_shot`
questions appended after.

## Pipeline

1. **Run.** `runner/run.py simpleqa` collects the probability-report condition
   and error-penalty prompting (EPP; L = 0, 3, 6; 50 samples per question,
   temperature 0 with per-sample seeds) on the 4,326 SimpleQA questions;
   `consistency` collects repeated reports (50 replicates on 1,000 questions);
   `batch` is a batch-API variant. The three models, all accessed through GCP Vertex AI:
   `google/gemini-3.5-flash`, `claude-sonnet-4-6`,
   `deepseek-ai/deepseek-v3.2-maas`.
   Needs gcloud credentials and `--gcp-project`/`--gcp-location`. Responses
   are cached locally (SQLite), so interrupted runs resume. The main experiment
   uses about 2.0M answer-model calls; repeated-report and sensitivity experiments
   add further calls.

2. **Grade.** `grading/grade.py --grader openai --runs <tree> --out <tree>`
   (or `--grader gemini`)
   grades every candidate answer and penalty sample and writes a parallel
   graded tree. Both graders use the same rubric (`grading/graders.py`):
   question + gold answer + one candidate in, CORRECT / INCORRECT /
   NOT_ATTEMPTED out, temperature 0. Unique (question, canonical answer)
   pairs are graded once and checkpointed. The
   OpenAI grader needs `OPENAI_API_KEY`; the Gemini grader needs the GCP
   flags. The runner can also grade inline while collecting (`--use-grader`).

3. **Analyze.** `analysis/reproduce_all.py` runs everything;
   `make_figures.py`, `make_tables.py`, `rho_sensitivity.py`,
   `rule_sensitivity.py`, `threshold_calibration.py`, and `alpha_risk_control.py` also run standalone
   (`--steps` selects individual tables/audits in `make_tables.py`). The analysis reads
   `results/graded_by_openai` or `results/graded_by_gemini`; to point it at
   a graded tree saved somewhere else (for example, one produced by
   `grade.py` under a custom path), set the environment variable
   `GRADED_RESULTS_DIR` to that directory.

   For the main benchmark curves and sets, answers graded correct are merged
   into one entry, non-answers are combined with IDK, and remaining answer
   strings are canonicalized using fixed rules. This processing uses the
   benchmark grades. Threshold decisions and top-*p* sets use retained
   probabilities without renormalization or assigning unallocated probability
   to IDK. Repeated-report divergence and the score-optimal list analysis
   normalize the retained probabilities.

   Threshold calibration compares observed error with mean reported error
   among questions answered at each threshold. Fig. 5 and Table S8 use
   pointwise 95% intervals from 2,000 paired question-level bootstrap resamples;
   empty reports always abstain. These diagnostics are also available with
   `python analysis/threshold_calibration.py --grader gemini`.

   The error-controlled threshold experiment (Tables S9–S10) starts from
   `raw_log_response`. Candidate selection uses the existing parser and fixed
   non-answer strings, before attaching grades; it does not use the answer key.
   Parsed probabilities are kept, and ties follow parsed entry order. Stored
   grades are recovered afterward using source strings and display mappings,
   checking each graded group's total probability and requiring an unambiguous
   grade for the selected candidate. The script raises an error on an unresolved
   grade rather than excluding a question.

   The same 2,163 threshold-selection and 2,163 test questions are used for all models,
   with split seed 20260912. On the threshold-selection half, one-sided exact binomial
   bounds use a Bonferroni correction across the fixed 102-threshold grid.
   For each reported target error rate (0.20, 0.25, 0.30, 0.35, 0.40), the smallest qualifying
   threshold is selected; if none qualifies, the rule abstains on all questions
   and error among returned answers is undefined. The 95% guarantee is simultaneous across
   thresholds and targets, separately for each model and split, under independent
   questions from the same distribution. Test intervals are two-sided exact
   binomial intervals. Another 100 random partitions assess sensitivity to
   which questions are used for threshold selection. Table S10 compares 50–50
   and 70–30 allocations on the common 1,298 test questions in each partition;
   these overlapping partitions are not independent replications. Reproduce with
   `python analysis/alpha_risk_control.py` (or add `--grader gemini`), then
   `python analysis/make_alpha_paper_tables.py` for Table S9. Run
   `python analysis/threshold_split_sensitivity.py` and
   `python analysis/make_split_sensitivity_table.py` for Table S10 after both
   graders' threshold analyses are available.
   The original four-target `threshold_selection.py` outputs remain as an
   archival comparison (`ThresholdSelectionLegacy*`); Tables S9–S10 use the expanded analysis.

### Exploratory richer decision rules

`analysis/rich_decision_rule.py` tests whether IDK probability, the gap between
the leading candidates, candidate count, and the report's probability
distribution improve selection beyond the top probability alone. All features
come from original parsed reports before attaching grades. The candidate answer
stays fixed. A matched learner using only the top probability distinguishes
additional report information from the effect of learning a decision score.
This retrospective experiment is separate from the manuscript figures and the
default paper reproduction command.

Install its additional dependency and run:

```bash
pip install -r analysis/requirements-rich-decision-rule.txt
python analysis/rich_decision_rule.py
```

The [analysis plan](analysis/rich_decision_rule_protocol.md) fixes the features,
learners, question splits, endpoints, and confidence scope. Each split separates
development, certification, and test questions. Fixed-sequence testing uses
an order learned only from development predictions; it stops at the first
failed certification test. The 95% guarantee is simultaneous over four target
error rates, separately for each fixed model, learner, and split, under the
stated distribution assumptions. The 100 additional splits describe sensitivity
to allocation and are not independent replications. All four learners and
unsuccessful selections are retained. Outputs, including per-question scores,
fitted estimators, testing traces, paired comparisons, plots, and input hashes,
are written to `results/exploratory/rich_decision_rule_openai`. Add
`--grader gemini` for the second grader or `--out-dir PATH` for another location.
When reloading feature CSVs to reproduce tree predictions, use pandas
`read_csv(..., float_precision="round_trip")` to preserve values at tree split
boundaries. Re-extracting features from the original responses also preserves
the training representation.

### Exploratory calibration and penalty-based decisions

`analysis/calibration_decisions.py` develops scalar calibration and direct
empirical score optimization under +1 for correct, −L for incorrect, and zero
for abstention. Weighted isotonic calibration and direct threshold optimization
give the same decisions under a shared tie and interpolation convention. A
fixed monotone beta-calibration fit is a smooth sensitivity comparison.

```bash
python analysis/calibration_decisions.py
python analysis/calibration_decisions.py --grader gemini
```

The [protocol](analysis/calibration_decisions_protocol.md) specifies the
2,163/2,163 fit/test split, 100 additional splits, raw and processed
representations, penalties, baselines, and interval scope. The
[theory note](analysis/calibration_decisions_theory.md) derives the connection
and its assumptions. This optimizes decision score; it does not certify a
future conditional error rate. Both successful and unsuccessful decisions,
including all-abstain, are retained.

Outputs go to `results/exploratory/calibration_decisions_openai` or
`calibration_decisions_gemini`, separate from default paper reproduction.
The [results report](results/exploratory/calibration_decisions_report.md)
interprets the comparisons. No extra analysis dependencies are required;
the independent isotonic unit test also uses scikit-learn from
`analysis/requirements-rich-decision-rule.txt`. Use
`read_csv(..., float_precision="round_trip")` when reconstructing decisions
from saved probabilities at threshold boundaries.

### Error-controlled decision rule

`analysis/alpha_risk_control.py` maps a requested error tolerance to the
smallest threshold supported by simultaneous one-sided exact binomial bounds.
It retains the previous fixed-grid method as its primary reference and tests
an isotonic development grid with independent certification. A fixed grid on
the same smaller certification sample separates the grid effect from sample
size. All methods use a common held-out test half. The manuscript uses the
fixed-grid method; the learned-grid comparison is retained as a repository-only
analysis, with formatted results in `results/figures/RiskControlGridComparison.csv`
and `RiskControlGridComparison_rows.tex`.

```bash
python analysis/alpha_risk_control.py
python analysis/alpha_risk_control.py --grader gemini
python analysis/alpha_risk_control.py --lookup results/exploratory/alpha_risk_control_openai --alpha 0.25
```

The lookup reads saved bounds without fitting or using test outcomes. Optional
`--model`, `--method`, and `--representation` select an analysis; defaults use
all three models, the full fixed-grid reference, and raw reports. An unsupported
request returns no threshold. The 95% guarantee covers thresholds and arbitrary
error tolerances simultaneously, separately for each fixed model and method,
under the stated sampling and grading assumptions. It does not cover choosing
the best method or model afterward or guarantee each finite test error fraction.

The [protocol](analysis/alpha_risk_control_protocol.md) fixes the comparisons
and splits; the [theory note](analysis/alpha_risk_control_theory.md) gives the
guarantee. Detailed outputs and the
[results report](results/exploratory/alpha_risk_control_report.md) are retained
in `results/exploratory/alpha_risk_control_{openai,gemini}/` and its parent.
`make_alpha_paper_tables.py` formats the saved results as SI Table S9 in
`results/figures/`; add `--recompute` to rebuild the analysis for both graders.
It also retains the full-test summaries across 100 additional 50–50 partitions
as `RiskControlFullTestPartitions.csv` and `RiskControlFullTestPartitions_rows.tex`,
alongside the repository-only grid comparison.
`python analysis/make_split_sensitivity_table.py` formats Table S10 from the
saved 50–50 versus 70–30 analysis in
`results/exploratory/threshold_split_sensitivity/`, evaluating both rules on
the common 1,298 test questions in each partition. Table S10 reports qualifying
partitions, mean answer rates, and median observed test errors with 25th–75th
percentiles. Error summaries exclude partitions with no returned test answers.
Both formatting commands display the primary grader by default; `--grader gemini`
selects the second grader. Their partition-summary CSVs retain both graders.
The default reproduction command includes both analyses, Tables S9–S10,
and the repository-only comparisons.
The paper tables report 20%, 25%, 30%, 35%, and 40% error tolerances. The
detailed analysis retains the earlier 10% and 1/7 results. The 30%, 35%, and
40% tolerances were added after the initial analysis, using the same methods,
bounds, and 100 additional splits; the protocol records the extension.

## Graded results

Each run directory holds gzip-compressed JSONL, one record per question
(loaders read `.jsonl` and `.jsonl.gz` interchangeably):

- `simpleqa_topp_results.jsonl` -- the probability-report condition:
  `log_candidates_json` (reported candidates with `answer`, `probability`,
  `points`, `grade`), reported IDK probability, token counts, and the rule
  parameters (`log_idk_rule`, `log_idk_rho`, `simpleqa_top_p`). The main runs use
  the logarithmic scoring rule with an additional penalty for an unlisted correct
  answer. The multiplier `log_idk_rho` (ρ) is 0.5, and the additional penalty
  is −log ρ; `raw_log_response` preserves the original response used by the
  threshold-selection experiment. The sensitivity records identify their corresponding multiplier
  or scoring statement.
- `simpleqa_penalty_results.jsonl` -- one answer-or-abstain condition:
  `penalty_value` (L), per-question sample counts, and
  `penalty_distribution_json` (the graded, canonicalized answer distribution).
- `simpleqa_log_consistency_results.jsonl` -- repeated reports, one record
  per (question, replicate).

Paper-analysis run directories, identical in both trees:

| Model | Report + L=3 | L=0 | L=6 | Repeated reports |
|---|---|---|---|---|
| Gemini 3.5 Flash | `gemini35flash` | `gemini35flash_L0` | `gemini35flash_L6` | `gemini35flash_consistency` |
| Claude Sonnet 4.6 | `claudesonnet46` | `claudesonnet46_L0` | `claudesonnet46_L6` | `claudesonnet46_consistency` |
| DeepSeek V3.2 | `deepseekv32` | `deepseekv32_L0` | `deepseekv32_L6` | `deepseekv32_consistency` |

Previously collected Qwen results remain in the graded-data trees for archival
purposes and are excluded from the paper analyses and default reproduction.

The primary grading tree also contains the 1,000-question Gemini sensitivity
runs used in Figs. S4-S5 and Table S4:

| Analysis | Directory | Runs |
|---|---|---|
| Sensitivity to ρ | `rho_sensitivity/` | `gemini35flash_rho10`, `gemini35flash_rho50`, `gemini35flash_rho90` |
| Scoring guidance | `rule_sensitivity/` | `gemini35flash_log`, `gemini35flash_quadratic`, `gemini35flash_brier`, `gemini35flash_linear`, `gemini35flash_unscored` |

The three ρ-sensitivity runs use the main probability-report template at
ρ = 0.1, 0.5, and 0.9; the ρ = 0.5 reports were elicited separately from the
main experiment. The scoring-guidance comparison uses a separate template
with optional IDK and no 90-point minimum, making its logarithmic condition
a separate baseline. All five conditions use the same questions and JSON
return format; four specify a scoring rule, and one provides no scoring rule.

## Figures and tables

Filenames follow the current main-text and SI numbering. Figure 3 is a single
vector PDF with aligned A–C panels, model headings shared across rows, and
one legend above the plots for panels A–B. The three set-outcome columns in
Figure 4 and Figure S1 use coverage, miscoverage with IDK, and miscoverage
without IDK.

| Paper | Files in `results/figures/` |
|---|---|
| Fig. 1 | `F1_Concept.pdf` (supplied schematic) |
| Fig. 2 | `F2_ResponseCounts.pdf`, `F2_ResponseCounts.csv` |
| Fig. 3 | `F3_Frontier.pdf`; `F3_Frontier_curves.csv`, `F3_Frontier.csv` (intervals), `F3_EPP_points.csv`, `matched_relative_accuracy.csv` |
| Fig. 4 | `F4_OutcomeTable_L3.pdf`, `F4_OutcomeTable_L3_set_outcomes.csv`, `F4_OutcomeTable_L3_long.csv` |
| Fig. 5 | `F5_ThresholdCalibration.pdf`, `F5_ThresholdCalibration.csv` |
| Fig. S1 | `FS1_OutcomeTable_L0.pdf`, `FS1_OutcomeTable_L6.pdf`, with corresponding `_set_outcomes.csv` and `_long.csv` files |
| Fig. S2 | `FS2_JSD.pdf`, `FS2_JSD.csv`, `FS2_JSD_by_question.csv` |
| Fig. S3 | `FS3_cumulative.pdf`, `FS3_cumulative.csv` |
| Fig. S4 | `FS4_rho_sensitivity.pdf`, `FS4_rho_sensitivity.csv`, `FS4_rho_summary.csv`, `FS4_rho_paired_comparisons.csv` |
| Fig. S5 | `FS5A_rule_sensitivity.pdf`, `FS5A_rule_sensitivity.csv`, `FS5B_rule_outcome_tables.pdf`, `FS5B_rule_outcome_tables.csv`, `FS5_rule_paired_comparisons.csv` |

### Tables

Every table has a numbered CSV and LaTeX tabular snippet under
[`results/figures/tables/`](results/figures/tables/). CSVs retain analysis
precision and supporting fields; the LaTeX snippets reproduce the displayed
layout, row selection, and rounding. The snippets use the manuscript's macros
and packages. The descriptive tables have versioned templates; numerical
rows are regenerated from the analyses.

| Paper | File stem in `results/figures/tables/` | Contents |
|---|---|---|
| Table 1 | `Table1_DecisionOutcomes` | RBD and EPP outcomes at each L, side by side |
| Table 2 | `Table2_MethodComparison` | RLCR and the probability-report setup for RBD |
| Table S1 | `TableS1_LogLoss` | Miscoverage-without-IDK bounds and realized log loss |
| Table S2 | `TableS2_SetOutcomeIntervals` | Figure 4 row percentages and 95% intervals |
| Table S3 | `TableS3_MatchedAbstention` | Absolute accuracy gains and hallucination reductions |
| Table S4 | `TableS4_IdealListExtent` | Score-optimal list sizes at fixed reported probabilities |
| Table S5 | `TableS5_TokenCosts` | Answer-model token use |
| Table S6 | `TableS6_IDKReliability` | IDK-probability calibration bins |
| Table S7 | `TableS7_TopReliability` | Top-candidate calibration bins |
| Table S8 | `TableS8_ThresholdCalibration` | Calibration at the decision thresholds |
| Table S9 | `TableS9_RiskControl` | Selected thresholds and held-out error |
| Table S10 | `TableS10_SplitSensitivity` | 50–50 versus 70–30 sample allocations |
| Table S11 | `TableS11_Models` | Model names and API identifiers |

`analysis/paper_tables.py` exports these tables from computed results.
`table_bodies.txt` also retains an additional set-source comparison that is
not a manuscript table. Table 1 uses largest-remainder rounding within each
method and L so its displayed outcome proportions total 1.000. Other tables
retain their specified units and precision.

### Numerical results and uncertainty

`analysis/comparison_uncertainty.py` computes Figure 3C and Table S2 from the
saved reports and EPP responses. Both use 2,000 paired question-level bootstrap
resamples, keeping each report and all 50 EPP responses together. The relative
accuracy gain is `100 * (RBD accuracy - EPP accuracy) / EPP accuracy`. Each
resample recomputes the matched-abstention comparison and the EPP denominator.
The intervals are pointwise 95% percentile intervals. In Table S2, each
resample also recomputes the conditional row denominator. The analysis runs
in the default reproduction workflow and can be run separately:

```bash
python analysis/comparison_uncertainty.py
```

Figure 3A–B uses 500 question-level bootstrap resamples for the curve bands.
Table S3 uses its original paired bootstrap for absolute gains; its intervals
are not obtained by rescaling the relative-gain intervals in panel C. At L=0,
Table S3 compares with the curve's lowest attainable abstention rather than
claiming an exact match. L=3 and L=6 comparisons match abstention by interpolation.

The numerical sources also include:

- `headline_report.json`: decision proportions, set outcomes, and reporting-discrepancy calculations.
- `matched_relative_accuracy.csv`: Figure 3C estimates and intervals.
- `set_outcome_intervals.csv`: Figure 4 row percentages and intervals.
- `matched_abstention_gaps.csv`: absolute matched-abstention differences.
- `support_containment.csv`: observed overlap of EPP answers and report candidates.
- `grader_agreement.csv`: agreement between the two graders.
- `threshold_calibration_metadata.json`: settings for Figure 5 and Table S8.

The detailed threshold bounds, selected rules, test metrics, split summaries,
and input hashes for Tables S9–S10 are in
`results/exploratory/alpha_risk_control_openai/` and
`results/exploratory/alpha_risk_control_gemini/`; the allocation comparison is
in `results/exploratory/threshold_split_sensitivity/`. An isolated `--out-dir`
rebuild writes these under its own `intermediate/` directory.

Grading: `graded_by_openai` (`gpt-5.6-terra`, via the OpenAI API) is the
primary grading; `graded_by_gemini` (`google/gemini-3.5-flash`) is the second
grader. They agree on 99.8% of candidate grades, and the top-0.9 coverage outcome
changes on 0.2–0.6% of questions per model (`make_tables.py --steps agreement`).

## Verification

The analysis tests check candidate selection, abstention, denominators,
threshold boundaries, binomial bounds, sample splitting, and calibration
identities. The optional exploratory tests require scikit-learn:

```bash
pip install -r analysis/requirements-rich-decision-rule.txt
python -m unittest discover -s tests -v
```

[`results/figures/reproduction_audit.json`](results/figures/reproduction_audit.json)
records the artifact-to-manuscript checks for this release. Generated PDFs
may differ byte-for-byte because of timestamps or platform fonts; numerical
CSV comparisons are the reproducibility check.
