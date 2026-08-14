# Data

## SimpleQA

All experiments use the SimpleQA factuality benchmark, 4,326 short-answer
factual questions with adjudicated gold answers (Wei et al., 2024,
"Measuring short-form factuality in large language models",
arXiv:2411.04368).

The runner loads the dataset from the Hugging Face Hub as
[`OpenEvals/SimpleQA`](https://huggingface.co/datasets/OpenEvals/SimpleQA)
via the `datasets` library; no local copy is stored in this repository.
Question ids in the result files are `simpleqa-0` through `simpleqa-4325`,
assigned in the dataset's row order.

## Graded results

The graded model outputs used by every figure and table in the paper are
shipped under [`../results/`](../results/) as gzip-compressed JSONL; see the
repository README for the file schema and the run-directory map.
