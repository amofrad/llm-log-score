# Log-Score Elicitation of Uncertainty in Language Models

Code and results for the paper *Log-Score Elicitation of Uncertainty
in Language Models* (Kaazempur-Mofrad and Dai).

## Reproducing the paper

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

## Layout

```
runner/     data collection: engine.py + run.py (subcommands simpleqa,
            consistency, and batch)
grading/    graders.py (both graders and the shared rubric) + grade.py
analysis/   common.py, make_figures.py, make_tables.py, reproduce_all.py
results/    graded_by_openai/ (primary grading), graded_by_gemini/ (second
            grader), figures/ (the paper's figures, tables, and audits)
data/       dataset notes (SimpleQA, loaded from Hugging Face)
```

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

## Paper artifacts

In `results/figures/`: `F2A`/`F2B_Frontier.pdf` (Fig. 2),
`F3_OutcomeTable_L3.pdf` (Fig. 3), and `FS1`-`FS4` (Figs. S1-S4) come from
`make_figures.py`; `F1_Concept.pdf` is a hand-authored illustration.
`make_tables.py` writes `table_bodies.txt` (Tables 1, 2, and the SI table
bodies), `theory_audit_*.csv` (Tables S1-S3), `token_costs.csv` (Table S4),
`matched_abstention_gaps.csv` (Table S5), `support_containment.csv`,
`headline_report.json`, and `grader_agreement.csv`.

On grading: `graded_by_openai` (grader `gpt-5.6-terra`, via the OpenAI API)
is the paper's primary grading; `graded_by_gemini`
(`google/gemini-3.5-flash`) is the second grading used for the agreement
audit. The two agree on 99.8% of candidate grades, and the top-0.9 coverage
outcome changes on 0.1-0.6% of questions per model
(`make_tables.py --steps agreement` recomputes this).
