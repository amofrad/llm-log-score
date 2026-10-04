"""Reproduce computed figures, tables, and audits (no API access required).

    python analysis/reproduce_all.py                  # primary (OpenAI-graded)
    python analysis/reproduce_all.py --grader gemini  # second-grader replication

Writes into results/figures (or results/figures_gemini for the second
grader): computed figure PDFs and their CSV side-outputs, including relative
accuracy gains (Fig. 3C) and set-outcome intervals (Table S2), the audit CSVs behind
Tables S1, S6, and S7, the matched-abstention gaps behind Table S3, the fixed-report
calculation behind Table S4, token costs (Table S5), threshold calibration
(Fig. 5 and Table S8),
error-controlled decisions on raw reports (Tables S9--S10),
observed answer overlap, numbered table CSVs,
the values quoted in the text, and the grader-agreement audit.
Fig. 1 is supplied separately as a schematic.
"""
from __future__ import annotations

import argparse
import sys
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_figures as mf  # noqa: E402
import make_tables as mt  # noqa: E402
import rho_sensitivity as rs  # noqa: E402
import rule_sensitivity as rules  # noqa: E402
import threshold_calibration as calibration  # noqa: E402
import make_alpha_paper_tables as risk_tables  # noqa: E402
import threshold_split_sensitivity as split_sensitivity  # noqa: E402
import make_split_sensitivity_table as split_table  # noqa: E402
import comparison_uncertainty as comparisons  # noqa: E402

# Repeated-elicitation runs for Fig. S2 (1,000 questions x 50 replicates)
CONSISTENCY_RUNS = ",".join([
    "gemini35flash_consistency",
    "claudesonnet46_consistency",
    "deepseekv32_consistency",
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
    elif not args.skip_tables:
        mt.set_results_tree(mt.tree_for_grader(args.grader))
        comparisons.main(["--outdir", str(out_dir)])

    if args.grader == "openai" and not (args.skip_figures and args.skip_tables):
        sensitivity_tree = mt.tree_for_grader(args.grader)
        rs.run(
            sensitivity_tree / "rho_sensitivity",
            out_dir,
            include_figure_outputs=not args.skip_figures,
            include_table_output=not args.skip_tables,
        )
        if not args.skip_figures:
            rules.run(sensitivity_tree / "rule_sensitivity", out_dir)

    if not (args.skip_figures and args.skip_tables):
        calibration.run(
            mt.tree_for_grader(args.grader), out_dir,
            include_figure_outputs=not args.skip_figures,
            include_table_output=not args.skip_tables,
        )

    if not args.skip_tables:
        analysis_dir = out_dir / "intermediate" if args.out_dir else mt.REPO_ROOT / "results/exploratory"
        risk_tables.run(out_dir, recompute=True, primary_grader=args.grader, analysis_dir=analysis_dir)
        sensitivity_dir = analysis_dir / "threshold_split_sensitivity"
        split_sensitivity.run(sensitivity_dir, analysis_dir=analysis_dir)
        split_table.generate(sensitivity_dir, out_dir, primary_grader=args.grader)

    shutil.copy2(HERE / "paper_artifacts.json", out_dir / "artifact_manifest.json")
    schematic = mt.REPO_ROOT / "results/figures/F1_Concept.pdf"
    if not args.skip_figures and schematic.resolve() != (out_dir / schematic.name).resolve():
        shutil.copy2(schematic, out_dir / schematic.name)

    if not args.skip_tables:
        from paper_tables import export
        export(out_dir, primary=args.grader == "openai")

    print(f"\nDone. Regenerated paper artifacts in {out_dir}")


if __name__ == "__main__":
    main()
