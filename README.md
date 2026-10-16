# Beyond a Single Answer: Proper Scoring of LLM Uncertainty

Code and results for:

> *Beyond a Single Answer: Proper Scoring of LLM Uncertainty*
>
> Ali Kaazempur-Mofrad, Xiaowu Dai, and Xuming He

## Overview

The paper studies probability reports that list candidate answers and assign
probabilities to each candidate and to IDK, the event that none of the listed
answers is correct. The same report supports report-based decisions (**RBD**)
and top-*p* sets of alternatives without another model query.

We evaluate Gemini 3.5 Flash, Claude Sonnet 4.6, and DeepSeek V3.2 on
4,326 [SimpleQA](https://huggingface.co/datasets/OpenEvals/SimpleQA) questions.
The comparison uses **error-penalty prompting (EPP)**, which assigns +1 to a
correct answer, −L to an incorrect answer, and 0 to abstention.

## Reproduce the paper results

Use Python 3.11 or newer. The graded responses are included, so reproduction
requires no API credentials or new model queries.

```bash
python -m pip install -r requirements.txt
python analysis/reproduce_all.py
```

Outputs are written to [`results/figures/`](results/figures/), including figure
PDFs, supporting numerical results, and numbered table CSVs. Figure 1 is a
supplied schematic; the remaining figures are regenerated from the saved data.
Bootstrap analyses use fixed random seeds. The numerical and plotting packages
are pinned in `requirements.txt` to the versions used to verify reproduction.
Figures 3 and S1 require Arial for their column labels; install that font before
running reproduction. The remaining plots use Matplotlib's bundled fonts.

Useful options:

- `--out-dir PATH`: write the paper outputs to another directory.
- `--grader gemini`: repeat the main analyses with the second grader; the
  sensitivity experiments use the primary grader only.
- `--skip-figures` or `--skip-tables`: regenerate only the requested outputs.

For example, to keep a reproduction separate from the included results:

```bash
python analysis/reproduce_all.py --out-dir /tmp/llm-report-reproduction
```

## Repository contents

```text
llm-log-score/
├── runner/                              Data collection
│   ├── engine.py                        API clients, prompts, scoring, and caching
│   └── run.py                           simpleqa, consistency, and batch commands
├── grading/                             Response grading
│   ├── graders.py                       Graders and shared rubric
│   └── grade.py                         Grade collected responses
├── analysis/                            Figures, tables, and statistical analyses
│   ├── common.py                        Data loaders and decision helpers
│   ├── reproduce_all.py                 Reproduce the paper's computed results
│   ├── make_figures.py                   Main and supplementary figures
│   ├── make_tables.py                    Numerical summaries and audits
│   ├── comparison_uncertainty.py         Figure 2C and Table S2 intervals
│   ├── paper_tables.py                   Numbered table CSV exports
│   ├── rho_sensitivity.py                Figure S4 and Table S4
│   ├── rule_sensitivity.py               Figure S5
│   ├── threshold_calibration.py          Figure 4 and Table S8
│   ├── threshold_selection.py            Error-controlled threshold selection and Table S9
│   └── threshold_split_sensitivity.py    Sample-allocation comparisons and Table S10
├── results/
│   ├── graded_by_openai/                 Primary grading, including sensitivity runs
│   ├── graded_by_gemini/                 Second-grader results
│   └── figures/                         Paper figures and supporting numerical data
│       └── tables/                      Tables 1 and S1–S11 as CSV
├── README.md
└── requirements.txt
```

## Data and experiment setup

The main experiment retains one probability report at ρ = 0.5 and 50 EPP
responses at each L = 0, 3, and 6 for every question and model. The primary
grader is GPT-5.6 Terra; Gemini 3.5 Flash provides a second grading using the
same rubric.

The following run directories appear under both `results/graded_by_openai/`
and `results/graded_by_gemini/`:

| Model | Report and EPP at L = 3 | EPP at L = 0 | EPP at L = 6 | Repeated reports |
|---|---|---|---|---|
| Gemini 3.5 Flash | `gemini35flash` | `gemini35flash_L0` | `gemini35flash_L6` | `gemini35flash_consistency` |
| Claude Sonnet 4.6 | `claudesonnet46` | `claudesonnet46_L0` | `claudesonnet46_L6` | `claudesonnet46_consistency` |
| DeepSeek V3.2 | `deepseekv32` | `deepseekv32_L0` | `deepseekv32_L6` | `deepseekv32_consistency` |

Responses are stored as compressed JSONL files:

- **`simpleqa_topp_results.jsonl.gz`**: candidate answers, probabilities,
  IDK probability, correctness grades, and the original report text.
- **`simpleqa_penalty_results.jsonl.gz`**: EPP response distributions and
  correct, incorrect, and abstention counts for each question.
- **`simpleqa_log_consistency_results.jsonl.gz`**: repeated probability reports,
  indexed by question and replicate.

The repeated-report experiment uses 50 reports per question on a common
1,000-question subset. Gemini's omission-penalty and scoring-rule experiments
use the same subset and are stored under `rho_sensitivity/` and
`rule_sensitivity/` in the primary grading directory.

## Analysis workflow

The reproduction command runs the analyses in order and exports their results.
Threshold-selection intermediates are computed in a temporary directory and
removed after the tables are exported.
Individual components can also be run separately:

| Script | Main outputs |
|---|---|
| `make_figures.py` | Decision curves, set comparisons, and repeated-report figures |
| `make_tables.py` | Decision outcomes, calibration summaries, matched-abstention comparisons, and token costs |
| `comparison_uncertainty.py` | Relative accuracy gains and set-outcome intervals |
| `rho_sensitivity.py`, `rule_sensitivity.py` | Sensitivity to the omission penalty and scoring instructions |
| `threshold_calibration.py` | Reported versus observed error among returned answers |
| `threshold_selection.py` | Threshold selection, held-out evaluation, and Table S9 |
| `threshold_split_sensitivity.py` | Comparisons of 50–50 and 70–30 sample allocations and Table S10 |
| `paper_tables.py` | All numbered table CSVs from computed results |

For example, to regenerate the decision outcomes in Table 1:

```bash
python analysis/make_tables.py --steps tables
python analysis/paper_tables.py
```

The two threshold-analysis scripts also run independently. Each computes its
required inputs, exports its table, and removes temporary intermediates.
Use `--analysis-dir PATH` to retain intermediates for reuse.

Benchmark comparisons merge answer variants using correctness grades and fixed
string rules. The error-controlled analysis instead selects candidates from
original parsed reports before attaching grades. Threshold selection and testing
use separate question samples.

## Figures and tables

Filenames follow the numbering in the paper and SI Appendix.

| Paper figure | PDF |
|---|---|
| Fig. 1 | [F1_Concept.pdf](results/figures/F1_Concept.pdf) |
| Fig. 2 | [F2_Frontier.pdf](results/figures/F2_Frontier.pdf) |
| Fig. 3 | [F3_OutcomeTable_L3.pdf](results/figures/F3_OutcomeTable_L3.pdf) |
| Fig. 4 | [F4_ThresholdCalibration.pdf](results/figures/F4_ThresholdCalibration.pdf) |
| Fig. S1 | [L = 0](results/figures/FS1_OutcomeTable_L0.pdf), [L = 6](results/figures/FS1_OutcomeTable_L6.pdf) |
| Fig. S2 | [FS2_JSD.pdf](results/figures/FS2_JSD.pdf) |
| Fig. S3 | [FS3_cumulative.pdf](results/figures/FS3_cumulative.pdf) |
| Fig. S4 | [FS4_rho_sensitivity.pdf](results/figures/FS4_rho_sensitivity.pdf) |
| Fig. S5 | [Panel A](results/figures/FS5A_rule_sensitivity.pdf), [Panel B](results/figures/FS5B_rule_outcome_tables.pdf) |

Supporting CSVs are stored alongside the figures. Figures 3 and S1 share two
files, with `penalty_level` identifying L = 0, 3, or 6:

- **`F3_FS1_OutcomeTable.csv`** contains cell counts, percentages across all
  questions (`percent`), and percentages within each EPP outcome row
  (`row_percent`), for each model and L. Counts average over the 50 EPP responses
  per question.
- **`F3_FS1_OutcomeTable_by_question.csv`** contains each question's contribution
  to those counts (`weight`), with EPP outcomes and top-p set outcomes identified
  separately.

Tables 1 and S1–S11 are stored once, in
[`results/figures/tables/`](results/figures/tables/). Numbering follows the main
text and SI Appendix. CSVs retain analysis precision and may include diagnostic
columns or additional rows. Table 1 also includes display values rounded by the
largest-remainder method so each three-outcome group totals 1.000.

Rebuild with `python analysis/reproduce_all.py`, or export already computed
results with `python analysis/paper_tables.py`. The source column below identifies
the analysis behind each table; Table S11 uses the shared model configuration.

| Paper table | CSV in `results/figures/tables/` | Contents | Source in `analysis/` |
|---|---|---|---|
| Table 1 | `Table1_DecisionOutcomes.csv` | RBD and EPP outcomes at each L | `make_tables.py` |
| Table S1 | `TableS1_LogLoss.csv` | Miscoverage bounds and realized log loss | `make_tables.py` |
| Table S2 | `TableS2_SetOutcomeIntervals.csv` | Figure 3 row percentages and uncertainty intervals | `comparison_uncertainty.py` |
| Table S3 | `TableS3_MatchedAbstention.csv` | Accuracy gains and hallucination reductions | `make_tables.py` |
| Table S4 | `TableS4_IdealListExtent.csv` | Score-optimal candidate-list sizes | `rho_sensitivity.py` |
| Table S5 | `TableS5_TokenCosts.csv` | Answer-model token use | `make_tables.py` |
| Table S6 | `TableS6_IDKReliability.csv` | IDK-probability calibration bins | `make_tables.py` |
| Table S7 | `TableS7_TopReliability.csv` | Top-candidate calibration bins | `make_tables.py` |
| Table S8 | `TableS8_ThresholdCalibration.csv` | Calibration at decision thresholds | `threshold_calibration.py` |
| Table S9 | `TableS9_RiskControl.csv` | Selected thresholds and held-out error | `threshold_selection.py` |
| Table S10 | `TableS10_SplitSensitivity.csv` | Sensitivity to sample allocation | `threshold_split_sensitivity.py` |
| Table S11 | `TableS11_Models.csv` | Model names and API identifiers | `common.py` |

## Collect or grade new responses

`runner/run.py` collects responses through Google Cloud Vertex AI;
`grading/grade.py` grades them using either the OpenAI or Gemini grader.
Collection requires Google Cloud credentials and project/location settings.
The OpenAI grader uses `OPENAI_API_KEY`; the Gemini grader uses Google Cloud
credentials. The shared grading rubric is in `grading/graders.py`.

Their available options are listed by:

```bash
python runner/run.py --help
python grading/grade.py --help
```
