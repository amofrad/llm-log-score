"""Reproduce Fig. S4 and Table S6 from the three rho-sensitivity runs.

The analysis uses the standard uncertainty-report prompt on the same 1,000
SimpleQA questions at rho = 0.1, 0.5, and 0.9. It writes the figure, its
question-level data, summary statistics, the paired rho contrast, and the
fixed-report calculation to ``results/figures``.

Run: python analysis/rho_sensitivity.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = common.OUTPUTS_DIR / "rho_sensitivity"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "figures"

EXPECTED_QUESTIONS = 1_000
NOMINAL_COVERAGE = 0.9
BOOTSTRAP_SEED = 12
BOOTSTRAP_DRAWS = 2_000

RUNS = {
    0.1: "gemini35flash_rho10",
    0.5: "gemini35flash_rho50",
    0.9: "gemini35flash_rho90",
}
COLORS = {0.1: "#188038", 0.5: "#1a73e8", 0.9: "#e8710a"}


def load_reports(results_dir: Path) -> dict[float, pd.DataFrame]:
    """Load and validate the three matched sensitivity runs."""
    reports = {}
    question_sets = []
    for expected_rho, run_dir in RUNS.items():
        df = common.load_run(run_dir, results_dir=results_dir)
        observed_rho = df["log_idk_rho"].dropna().unique()
        if len(observed_rho) != 1 or not np.isclose(observed_rho[0], expected_rho):
            raise ValueError(
                f"{run_dir} does not contain a single rho={expected_rho:g} run"
            )
        observed_rule = set(df["log_idk_rule"].dropna())
        if observed_rule != {"residual"}:
            raise ValueError(f"{run_dir} is not a residual-log run")
        if df["log_prompt_variant"].notna().any():
            raise ValueError(f"{run_dir} is not a standard-prompt run")
        observed_coverage = df["simpleqa_top_p"].dropna().unique()
        if len(observed_coverage) != 1 or not np.isclose(
            observed_coverage[0], NOMINAL_COVERAGE
        ):
            raise ValueError(
                f"{run_dir} does not use nominal coverage {NOMINAL_COVERAGE:g}"
            )
        reports[expected_rho] = df
        question_sets.append(set(df["question_id"]))

    if any(question_ids != question_sets[0] for question_ids in question_sets[1:]):
        raise ValueError("The rho runs do not contain the same questions")
    if len(question_sets[0]) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"Expected {EXPECTED_QUESTIONS} matched questions; "
            f"found {len(question_sets[0])}"
        )
    return reports


def question_metrics(reports: dict[float, pd.DataFrame]) -> dict[float, pd.DataFrame]:
    """Compute and align the question-level quantities used in Fig. S4."""
    question_ids = set(next(iter(reports.values()))["question_id"])
    order = [
        question_id
        for question_id in common.simpleqa_full_question_order()
        if question_id in question_ids
    ]
    metrics = {}
    for rho, df in reports.items():
        q = common.report_metrics(df, p=NOMINAL_COVERAGE).set_index("question_id")
        metrics[rho] = q.loc[order].reset_index()
    return metrics


def write_question_data(metrics: dict[float, pd.DataFrame], out_dir: Path) -> None:
    rows = []
    for rho, frame in metrics.items():
        part = frame.copy()
        part.insert(1, "rho", rho)
        rows.append(part)
    pd.concat(rows, ignore_index=True).to_csv(
        out_dir / "FS4_rho_sensitivity.csv", index=False
    )


def write_summary(metrics: dict[float, pd.DataFrame], out_dir: Path) -> None:
    rows = []
    rate_columns = ("truth_in_list", "coverage", "coverage_or_idk", "incorrect")
    for rho, q in metrics.items():
        n = len(q)
        row = {
            "rho": rho,
            "n_questions": n,
            "mean_list_size": q["list_size"].mean(),
            "median_list_size": q["list_size"].median(),
            "all_idk_rate": (q["list_size"] == 0).mean(),
            "mean_idk_mass": q["idk_mass"].mean(),
            "median_idk_mass": q["idk_mass"].median(),
            "mean_top_concrete_probability": q["top_concrete_probability"].mean(),
        }
        for column in rate_columns:
            successes = int(q[column].sum())
            low, high = common.wilson_interval(successes, n)
            row[f"{column}_rate"] = successes / n
            row[f"{column}_ci_low"] = low
            row[f"{column}_ci_high"] = high
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "FS4_rho_summary.csv", index=False)


def make_figure(metrics: dict[float, pd.DataFrame], out_dir: Path) -> None:
    """Create the list-size and IDK-mass panels in Fig. S4."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    rhos = list(RUNS)

    max_list_size = max(int(q["list_size"].max()) for q in metrics.values())
    x = np.arange(max_list_size + 1)
    width = 0.8 / len(rhos)
    for index, rho in enumerate(rhos):
        frequencies = metrics[rho]["list_size"].value_counts(normalize=True)
        offsets = x + (index - (len(rhos) - 1) / 2) * width
        axes[0].bar(
            offsets,
            [frequencies.get(k, 0.0) for k in x],
            width=width * 0.94,
            color=COLORS[rho],
            label=f"$\\rho={rho:g}$",
        )
    axes[0].set_xticks(x)
    axes[0].set_xlabel("candidate answers per question")
    axes[0].set_ylabel("fraction of questions")
    axes[0].set_title("List size")
    axes[0].legend(frameon=False)

    n = EXPECTED_QUESTIONS
    for rho in rhos:
        values = np.sort(metrics[rho]["idk_mass"].to_numpy())
        axes[1].step(
            np.r_[0.0, values],
            np.r_[0.0, np.arange(1, n + 1) / n],
            where="post",
            color=COLORS[rho],
            label=f"$\\rho={rho:g}$",
        )
    axes[1].set_xlabel("reported IDK probability")
    axes[1].set_ylabel("cumulative fraction of questions")
    axes[1].set_title("IDK probability")
    axes[1].set_xlim(0, 1)
    axes[1].legend(frameon=False, loc="lower right")

    fig.tight_layout()
    target = out_dir / "FS4_rho_sensitivity.pdf"
    fig.savefig(target)
    plt.close(fig)
    print(f"wrote {target}")


def write_paired_contrast(metrics: dict[float, pd.DataFrame], out_dir: Path) -> None:
    """Compare rho=0.1 with rho=0.9 on matched questions."""
    low = metrics[0.1].set_index("question_id")
    high = metrics[0.9].set_index("question_id")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for metric in common.REPORT_METRICS:
        differences = (
            low[metric].astype(float) - high[metric].astype(float)
        ).to_numpy()
        differences = differences[~np.isnan(differences)]
        samples = np.array(
            [
                differences[rng.integers(0, len(differences), len(differences))].mean()
                for _ in range(BOOTSTRAP_DRAWS)
            ]
        )
        ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
        rows.append(
            {
                "comparison": "rho_0.1_minus_rho_0.9",
                "metric": metric,
                "n_questions": len(differences),
                "mean_rho_0.1": low[metric].mean(),
                "mean_rho_0.9": high[metric].mean(),
                "difference": differences.mean(),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    pd.DataFrame(rows).to_csv(
        out_dir / "FS4_rho_paired_comparisons.csv", index=False
    )


def write_ideal_list_extent(report: pd.DataFrame, out_dir: Path) -> None:
    """Reoptimize each rho=0.5 report with its probability profile fixed."""
    rows = []
    for rho in RUNS:
        optimal_sizes = []
        for candidates in report["candidates"]:
            concrete_masses = sorted(
                (
                    candidate["probability"]
                    for candidate in candidates
                    if candidate["grade"] != "not_attempted"
                    and candidate["probability"] > 0
                ),
                reverse=True,
            )
            idk_mass = sum(
                candidate["probability"]
                for candidate in candidates
                if candidate["grade"] == "not_attempted"
            )
            total_mass = sum(concrete_masses) + idk_mass
            if total_mass <= 0:
                optimal_sizes.append(0)
                continue
            concrete_masses = [mass / total_mass for mass in concrete_masses]
            idk_mass /= total_mass

            best_size, best_score = 0, -np.inf
            for size in range(len(concrete_masses) + 1):
                residual_mass = idk_mass + sum(concrete_masses[size:])
                expected_score = sum(
                    mass * np.log(mass) for mass in concrete_masses[:size]
                )
                if residual_mass > 0:
                    expected_score += residual_mass * np.log(rho * residual_mass)
                if expected_score > best_score + 1e-12:
                    best_size, best_score = size, expected_score
            optimal_sizes.append(best_size)

        optimal_sizes = np.asarray(optimal_sizes)
        rows.append(
            {
                "rho": rho,
                "n_questions": len(optimal_sizes),
                "mean_optimal_list_size": optimal_sizes.mean(),
                "all_idk_rate": (optimal_sizes == 0).mean(),
            }
        )
    pd.DataFrame(rows).to_csv(
        out_dir / "TableS6_ideal_list_extent.csv", index=False
    )


def run(
    results_dir: Path,
    out_dir: Path,
    *,
    include_figure_outputs: bool = True,
    include_table_output: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = load_reports(results_dir)
    if include_figure_outputs:
        metrics = question_metrics(reports)
        write_question_data(metrics, out_dir)
        write_summary(metrics, out_dir)
        write_paired_contrast(metrics, out_dir)
        make_figure(metrics, out_dir)
    if include_table_output:
        write_ideal_list_extent(reports[0.5], out_dir)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    run(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
