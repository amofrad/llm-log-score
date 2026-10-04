"""Generate SI Table S9 and repository-only partition and grid comparisons.

Use --recompute to rebuild both graders' analysis from the stored responses.
Without it, read the saved, verified analysis; no model calls are made.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common

ROOT = Path(__file__).resolve().parents[1]
PAPER_TARGETS = (.20, .25, .30, .35, .40)
MODEL_LABELS = {"gemini35flash": "Gemini", "sonnet46": "Sonnet", "deepseekv32maas": "DeepSeek"}


def one(frame, model, alpha, method):
    rows = frame[(frame.model_key == model) & (frame.representation == "raw")
                 & (frame.method == method) & np.isclose(frame.alpha, alpha, rtol=0, atol=1e-14)]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {model}, {alpha}, {method}; found {len(rows)}")
    return rows.iloc[0]


def alpha_tex(alpha):
    return f"${alpha:.2f}$"


def threshold_tex(value):
    return f"${value:.3f}$"


def rate_tex(value):
    return r"$<0.1$" if 0 < 100 * value < .05 else f"{100 * value:.1f}"


def error_summary(frame, model, alpha):
    rows = frame[(frame.model_key == model) & np.isclose(frame.alpha, alpha, rtol=0, atol=1e-14)]
    assert len(rows) == rows.seed.nunique() == 100
    answered = rows[rows.n_answered > 0]
    assert answered.observed_error.notna().all()
    assert rows.loc[rows.n_answered == 0, "observed_error"].isna().all()
    assert np.allclose(answered.observed_error, answered.n_incorrect / answered.n_answered)
    errors = answered.observed_error
    return {"n_partitions_with_test_answers": len(errors),
            "median_test_error": errors.median(),
            "test_error_q25": errors.quantile(.25),
            "test_error_q75": errors.quantile(.75)}


def error_tex(summary):
    if not summary["n_partitions_with_test_answers"]:
        return "---"
    median = f"{100 * summary['median_test_error']:.1f}"
    if summary["n_partitions_with_test_answers"] == 1:
        return median
    return median + f" [{100 * summary['test_error_q25']:.1f}, {100 * summary['test_error_q75']:.1f}]"


def run(out_dir=None, recompute=False, primary_grader="openai", analysis_dir=None):
    out_dir = Path(out_dir) if out_dir else ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = Path(analysis_dir) if analysis_dir else ROOT / "results/exploratory"
    summaries = {}
    repeated = {}
    for grader in ("openai", "gemini"):
        folder = analysis_dir / f"alpha_risk_control_{grader}"
        if recompute:
            import alpha_risk_control as arc
            arc.run(grader, folder, repeats=100)
        summaries[grader] = pd.read_csv(folder / "split_summary.csv", float_precision="round_trip")
        if not (summaries[grader].n_splits == 100).all():
            raise ValueError("Publication tables require all 100 additional splits")
        all_results = pd.read_csv(folder / "all_results.csv", float_precision="round_trip")
        primary_seed = pd.read_csv(folder / "primary_results.csv", usecols=["seed"]).seed.unique()
        assert len(primary_seed) == 1
        repeated[grader] = all_results[(all_results.method == "fixed_full")
                                      & (all_results.representation == "raw")
                                      & (all_results.seed != primary_seed[0])]
    primary = pd.read_csv(analysis_dir / f"alpha_risk_control_{primary_grader}" / "primary_results.csv",
                          float_precision="round_trip")
    records = {8: [], 9: [], 10: []}
    lines = {8: [], 9: [], 10: []}
    for model in common.MODEL_ORDER:
        if records[8]:
            for table in lines:
                lines[table].append(r"\midrule")
        for i, alpha in enumerate(PAPER_TARGETS):
            label = MODEL_LABELS[model] if i == 0 else ""
            display_row = model != "deepseekv32maas" or i == 0
            prefix = f"{label} & " + ("All" if model == "deepseekv32maas" else alpha_tex(alpha))
            row = one(primary, model, alpha, "fixed_full")
            records[8].append(row.to_dict() | {"grader": primary_grader})
            if row.supported:
                cutoff = threshold_tex(row.threshold)
                bound = f"{100 * row.certification_upper:.1f}"
                error = f"{100 * row.observed_error:.1f} [{100 * row.error_lo:.1f}, {100 * row.error_hi:.1f}]"
            else:
                cutoff, bound, error = "---", "---", "---"
            answered = f"{int(row.n_answered):,} ({100 * row.answer_rate:.1f})"
            if model == "deepseekv32maas" and row.supported:
                raise ValueError("DeepSeek rows can only be combined while all targets are unsupported")
            if display_row:
                lines[8].append(f"{prefix} & {cutoff} & {bound} & {answered} & {error}" + r"\\")

            full = {g: one(summaries[g], model, alpha, "fixed_full") for g in summaries}
            errors = {g: error_summary(repeated[g], model, alpha) for g in summaries}
            for g in summaries:
                assert errors[g]["n_partitions_with_test_answers"] == full[g].n_supported
                if full[g].n_supported:
                    assert np.isclose(errors[g]["median_test_error"], full[g].median_error_when_answering)
            records[9].append({"model_key": model, "alpha": alpha, "display_grader": primary_grader} | {
                f"{g}_{key}": full[g][key] for g in full
                for key in ("n_splits", "n_supported", "mean_answer_rate")} | {
                f"{g}_{key}": value for g, summary in errors.items() for key, value in summary.items()})
            displayed = full[primary_grader]
            values = [str(int(displayed.n_supported)), rate_tex(displayed.mean_answer_rate),
                      error_tex(errors[primary_grader])]
            if model == "deepseekv32maas" and any(row.n_supported for row in full.values()):
                raise ValueError("DeepSeek split results differ; display all targets separately")
            if display_row:
                lines[9].append(prefix + " & " + " & ".join(values) + r"\\")

            paired = {(g, method): one(summaries[g], model, alpha, method)
                      for g in summaries for method in ("fixed_half", "isotonic_grid")}
            records[10].append({"model_key": model, "alpha": alpha} | {
                f"{g}_{method}_{key}": paired[g, method][key]
                for g, method in paired for key in ("n_splits", "n_supported", "mean_answer_rate", "median_grid_size")})
            values = [str(int(row.n_supported)) for row in paired.values()]
            if model == "deepseekv32maas" and any(row.n_supported for row in paired.values()):
                raise ValueError("DeepSeek grid results differ; display all targets separately")
            if display_row:
                lines[10].append(prefix + " & " + " & ".join(values) + r"\\")

    for table in records:
        stem = {8: "TableS9_RiskControl", 9: "RiskControlFullTestPartitions", 10: "RiskControlGridComparison"}[table]
        pd.DataFrame(records[table]).to_csv(out_dir / f"{stem}.csv", index=False)
        (out_dir / f"{stem}_rows.tex").write_text(
            "% Generated by analysis/make_alpha_paper_tables.py; do not edit values manually.\n"
            + "\n".join(lines[table]) + "\n")
    print(f"Wrote SI Table S9 and repository-only partition and grid comparisons to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--grader", choices=("openai", "gemini"), default="openai",
                        help="Grader displayed in Table S9 and the full-test partition summary; CSVs retain both graders' partition summaries.")
    args = parser.parse_args()
    run(args.out_dir, args.recompute, args.grader)
