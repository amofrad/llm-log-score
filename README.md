# Log-Score Elicitation of Uncertainty in Language Models

Code and results for:

> *Log-Score Elicitation of Uncertainty
in Language Models*
>
>  Ali Kaazempur-Mofrad and Xiaowu Dai

## Quick start

The graded model outputs behind the paper's figures and tables are included
under `results/`, so reproduction needs no API access:

```bash
pip install -r requirements.txt
python analysis/reproduce_all.py
```

This rebuilds every figure and table into `results/figures/` from
`results/graded_by_openai/`. Use `--grader gemini` to rerun everything under
the second grader instead. The bootstrap is seeded, so regenerated numbers
match exactly.

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
│   └── reproduce_all.py     rebuild every paper artifact
├── results/
│   ├── graded_by_openai/    primary grading
│   ├── graded_by_gemini/    second grader
│   └── figures/             the paper's figures, tables, and audits
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

1. **Run.** `runner/run.py simpleqa` collects the uncertainty-report arm and
   the penalty arms (L = 0, 3, 6; 50 samples per question, temperature 0 with
   per-sample seeds) on the 4,326 SimpleQA questions; `consistency` collects
   repeated reports (50 replicates on 1,000 questions); `batch` is a batch-API
   variant. The four models, all accessed through GCP Vertex AI:
   `google/gemini-3.5-flash`, `claude-sonnet-4-6`,
   `deepseek-ai/deepseek-v3.2-maas`, `qwen/qwen3-235b-a22b-instruct-2507-maas`.
   Needs gcloud credentials and `--gcp-project`/`--gcp-location`. Responses
   are cached locally (SQLite), so interrupted runs resume. The full study is
   about 2.6M answer-model calls.

2. **Grade.** `grading/grade.py --grader openai --runs <tree> --out <tree>`
   (or `--grader gemini`)
   grades every candidate answer and penalty sample and writes a parallel
   graded tree. Both graders use the same rubric (`grading/graders.py`):
   question + gold answer + one candidate in, CORRECT / INCORRECT /
   NOT_ATTEMPTED out, temperature 0. Unique (question, canonical answer)
   pairs are graded once (~243k for the full study) and checkpointed. The
   OpenAI grader needs `OPENAI_API_KEY`; the Gemini grader needs the GCP
   flags. The runner can also grade inline while collecting (`--use-grader`).

3. **Analyze.** `analysis/reproduce_all.py` runs everything;
   `make_figures.py` and `make_tables.py` also run standalone (`--steps`
   selects individual tables/audits). The analysis reads
   `results/graded_by_openai` or `results/graded_by_gemini`; to point it at
   a graded tree saved somewhere else (for example, one produced by
   `grade.py` under a custom path), set the environment variable
   `GRADED_RESULTS_DIR` to that directory.

## Graded results

Each run directory holds gzip-compressed JSONL, one record per question
(loaders read `.jsonl` and `.jsonl.gz` interchangeably):

- `simpleqa_topp_results.jsonl` -- the uncertainty-report arm:
  `log_candidates_json` (reported candidates with `answer`, `probability`,
  `points`, `grade`), stated IDK mass, token counts, and the rule parameters
  (`log_idk_rule = "residual"`, `log_idk_rho = 0.5`, `simpleqa_top_p = 0.9`).
- `simpleqa_penalty_results.jsonl` -- one penalty arm: `penalty_value` (L),
  per-question sample counts, and `penalty_distribution_json` (the graded,
  canonicalized answer distribution).
- `simpleqa_log_consistency_results.jsonl` -- repeated reports, one record
  per (question, replicate).

Run directories, identical in both trees:

| Model | Report + L=3 | L=0 | L=6 | Repeated reports |
|---|---|---|---|---|
| Gemini 3.5 Flash | `gemini35flash` | `gemini35flash_L0` | `gemini35flash_L6` | `gemini35flash_consistency` |
| Claude Sonnet 4.6 | `claudesonnet46` | `claudesonnet46_L0` | `claudesonnet46_L6` | `claudesonnet46_consistency` |
| DeepSeek V3.2 | `deepseekv32` | `deepseekv32_L0` | `deepseekv32_L6` | `deepseekv32_consistency` |
| Qwen3 235B-A22B | `qwen3_235b` | `qwen3_235b_L0` | `qwen3_235b_L6` | `qwen3_235b_consistency` |

## Figures and tables

`results/figures/` holds every figure and table in the paper. Apart from
Fig. 1, figures are produced by `make_figures.py` (each with its plotted data
saved as a `.csv`) and the tables and audits by `make_tables.py`;
`reproduce_all.py` runs both.

| Paper | File(s) in `results/figures/` |
|---|---|
| Fig. 1 | `F1_Concept.pdf` (schematic) |
| Fig. 2 | `F2A_Frontier.pdf`, `F2B_Frontier.pdf` |
| Fig. 3 | `F3_OutcomeTable_L3.pdf` |
| Fig. S1 | `FS1_OutcomeTable_L0.pdf`, `FS1_OutcomeTable_L6.pdf` |
| Fig. S2 | `FS2_JSD.pdf` |
| Fig. S3 | `FS3_cumulative.pdf` |
| Fig. S4 | `FS4_response_counts.pdf` |
| Tables 1, 2, and SI table bodies | `table_bodies.txt` |
| Tables S1-S3 | `theory_audit_*.csv` |
| Table S4 | `token_costs.csv` |
| Table S5 | `matched_abstention_gaps.csv` |
| Assumption 1 audit | `support_containment.csv` |
| Numbers quoted in the text | `headline_report.json` |
| Grader agreement | `grader_agreement.csv` |

Grading: `graded_by_openai` (`gpt-5.6-terra`, via the OpenAI API) is the
primary grading; `graded_by_gemini` (`google/gemini-3.5-flash`) is the second
grader used for the agreement audit. The two agree on 99.8% of candidate
grades, and the top-0.9 coverage outcome changes on 0.1-0.6% of questions per
model (`make_tables.py --steps agreement`).
