# Beyond a Single Answer: Proper Scoring of LLM Uncertainty

Code and results accompanying the paper by **Ali Kaazempur-Mofrad, Xiaowu Dai,
and Xuming He**.

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

Use Python 3.10 or newer. The graded responses are included, so reproduction
requires no API credentials or new model queries.

```bash
python -m pip install -r requirements.txt
python analysis/reproduce_all.py
```

Outputs are written to [`results/figures/`](results/figures/), including figure
PDFs, supporting numerical results, and numbered table CSVs. Figure 1 is a
supplied schematic; the remaining figures are regenerated from the saved data.
Bootstrap analyses use fixed random seeds.

Useful options:

- `--out-dir PATH`: write the results and intermediate files to another directory.
- `--grader gemini`: repeat the main analyses with the second grader; the
  sensitivity experiments use the primary grader only.
- `--skip-figures` or `--skip-tables`: regenerate only the requested outputs.

For example, to keep a reproduction separate from the included results:

```bash
python analysis/reproduce_all.py --out-dir /tmp/llm-report-reproduction
```

## Repository contents

| Directory | Contents |
|---|---|
| [`runner/`](runner/) | Model queries and prompts |
| [`grading/`](grading/) | Grading code and shared rubric |
| [`analysis/`](analysis/) | Reproduction scripts, analysis protocols, and theory notes |
| [`results/graded_by_openai/`](results/graded_by_openai/) | Responses graded by GPT-5.6 Terra, the primary grader |
| [`results/graded_by_gemini/`](results/graded_by_gemini/) | Responses graded by Gemini 3.5 Flash |
| [`results/figures/`](results/figures/) | Paper figures and numerical results |
| [`results/figures/tables/`](results/figures/tables/) | Tables 1–2 and S1–S11 as CSV |
| [`tests/`](tests/) | Analysis and result-consistency tests |

The main experiment retains one probability report and 50 EPP responses at
each L = 0, 3, and 6 per question and model. Graded responses are stored as
compressed JSONL files. Repeated-report and sensitivity data are included
in the corresponding grading directories.

## Figures and tables

Filenames follow the numbering in the paper and SI Appendix.

| Paper figure | PDF |
|---|---|
| Fig. 1 | [F1_Concept.pdf](results/figures/F1_Concept.pdf) |
| Fig. 2 | [F2_ResponseCounts.pdf](results/figures/F2_ResponseCounts.pdf) |
| Fig. 3 | [F3_Frontier.pdf](results/figures/F3_Frontier.pdf) |
| Fig. 4 | [F4_OutcomeTable_L3.pdf](results/figures/F4_OutcomeTable_L3.pdf) |
| Fig. 5 | [F5_ThresholdCalibration.pdf](results/figures/F5_ThresholdCalibration.pdf) |
| Fig. S1 | [L = 0](results/figures/FS1_OutcomeTable_L0.pdf), [L = 6](results/figures/FS1_OutcomeTable_L6.pdf) |
| Fig. S2 | [FS2_JSD.pdf](results/figures/FS2_JSD.pdf) |
| Fig. S3 | [FS3_cumulative.pdf](results/figures/FS3_cumulative.pdf) |
| Fig. S4 | [FS4_rho_sensitivity.pdf](results/figures/FS4_rho_sensitivity.pdf) |
| Fig. S5 | [Panel A](results/figures/FS5A_rule_sensitivity.pdf), [Panel B](results/figures/FS5B_rule_outcome_tables.pdf) |

Supporting CSVs are stored alongside the figures. All 13 paper tables are
available in [`results/figures/tables/`](results/figures/tables/), with their
sources listed in the [table manifest](results/figures/tables/manifest.json).
The CSVs retain analysis precision; Table 1 also includes rounded display values.

## Additional analyses

Detailed settings and assumptions are documented with the analyses:

- [Error-controlled decisions](analysis/alpha_risk_control_protocol.md)
- [Sensitivity to threshold-selection sample size](analysis/threshold_split_sensitivity_protocol.md)
- [Exploratory decision rules](analysis/rich_decision_rule_protocol.md)
- [Exploratory calibration and decision scores](analysis/calibration_decisions_protocol.md)

## Collect or grade new responses

The collection and grading tools require credentials for the relevant model
providers. Their available options are listed by:

```bash
python runner/run.py --help
python grading/grade.py --help
```

## Verification

To run the full test suite, including the exploratory analyses:

```bash
python -m pip install -r analysis/requirements-rich-decision-rule.txt
python -m unittest discover -s tests -v
```

The [reproduction audit](results/figures/reproduction_audit.json) records checks
against the manuscript figures and tables.
