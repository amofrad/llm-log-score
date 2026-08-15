"""Reproduce every figure and table in the paper (no API access required).

    python analysis/reproduce_all.py                  # primary (OpenAI-graded)
    python analysis/reproduce_all.py --grader gemini  # second-grader replication

Writes into results/figures (or results/figures_gemini for the second
grader): all figure PDFs and their CSV side-outputs, the audit CSVs behind
Tables S1-S3, the matched-abstention gaps behind Table S5, the
support-containment audit, the LaTeX table bodies for Tables 1, 2, S1, S2,
S3, and S5, the values quoted in the text, and the grader-agreement audit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_figures as mf  # noqa: E402
import make_tables as mt  # noqa: E402

# Repeated-elicitation runs for Fig. S2 (1,000 questions x 50 replicates)
CONSISTENCY_RUNS = ",".join([
    "gemini35flash_consistency",
    "claudesonnet46_consistency",
    "deepseekv32_consistency",
    "qwen3_235b_consistency",
])
CONSISTENCY_QUESTION_LIMIT = "1000"
CONSISTENCY_REPEAT_LIMIT = "50"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grader", choices=["openai", "gemini"], default="openai",
                        help="Which graded result tree to analyze (default: openai).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: results/figures, or "
                             "results/figures_gemini for --grader gemini).")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Regenerate only tables and audits.")
    parser.add_argument("--skip-tables", action="store_true",
                        help="Regenerate only figures.")
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    if out_dir is None:
        name = "figures" if args.grader == "openai" else "figures_gemini"
        out_dir = mt.REPO_ROOT / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_tables:
        mt.main(["--grader", args.grader, "--out-dir", str(out_dir)])

    if not args.skip_figures:
        mt.set_results_tree(mt.tree_for_grader(args.grader))
        mf.main([
            "--out-dir", str(out_dir),
            "--consistency-runs", CONSISTENCY_RUNS,
            "--consistency-question-limit", CONSISTENCY_QUESTION_LIMIT,
            "--consistency-repeat-limit", CONSISTENCY_REPEAT_LIMIT,
        ])

    print(f"\nDone. Regenerated paper artifacts in {out_dir}")


if __name__ == "__main__":
    main()
