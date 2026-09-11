"""Reproduce Fig. S5 from the five scoring-guidance runs.

The five prompts use the same reporting instructions and the same 1,000
SimpleQA questions. They differ only in whether they state a residual-log,
quadratic, Brier, linear, or no scoring rule. Outputs are written to
``results/figures``.

Run: python analysis/rule_sensitivity.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
import numpy as np
import pandas as pd

import common

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = common.OUTPUTS_DIR / "rule_sensitivity"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "figures"

EXPECTED_QUESTIONS = 1_000
NOMINAL_COVERAGE = 0.9
BOOTSTRAP_SEED = 20260829
BOOTSTRAP_DRAWS = 2_000


@dataclass(frozen=True)
class RuleSpec:
    slug: str
    label: str
    run_dir: str
    stored_rule: str
    color: str
    tick: str


RULES = (
    RuleSpec(
        "residual_log",
        "residual-log score",
        "gemini35flash_log",
        "residual",
        "#9334e6",
        "Log\n($\\rho=0.5$)",
    ),
    RuleSpec(
        "quadratic",
        "quadratic score",
        "gemini35flash_quadratic",
        "quadratic",
        "#188038",
        "Quadratic\nscore",
    ),
    RuleSpec(
        "brier",
        "Brier score",
        "gemini35flash_brier",
        "brier",
        "#81c995",
        "Brier\nscore",
    ),
    RuleSpec(
        "linear",
        "linear score",
        "gemini35flash_linear",
        "linear",
        "#d93025",
        "Linear\nscore",
    ),
    RuleSpec(
        "no_rule",
        "no scoring rule",
        "gemini35flash_unscored",
        "unscored",
        "#5f6368",
        "Unspecified",
    ),
)


def load_reports(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and validate the five matched scoring-guidance runs."""
    reports = {}
    question_sets = []
    for spec in RULES:
        df = common.load_run(spec.run_dir, results_dir=results_dir)
        observed_rule = set(df["log_idk_rule"].dropna())
        if observed_rule != {spec.stored_rule}:
            raise ValueError(
                f"{spec.run_dir} has rule metadata {sorted(observed_rule)}, "
                f"expected {spec.stored_rule}"
            )
        observed_prompt = set(df["log_prompt_variant"].dropna())
        if observed_prompt != {"incentive-only"}:
            raise ValueError(f"{spec.run_dir} does not use the shared prompt template")
        observed_coverage = df["simpleqa_top_p"].dropna().unique()
        if len(observed_coverage) != 1 or not np.isclose(
            observed_coverage[0], NOMINAL_COVERAGE
        ):
            raise ValueError(
                f"{spec.run_dir} does not use nominal coverage {NOMINAL_COVERAGE:g}"
            )
        if spec.slug == "residual_log":
            observed_rho = df["log_idk_rho"].dropna().unique()
            if len(observed_rho) != 1 or not np.isclose(observed_rho[0], 0.5):
                raise ValueError(f"{spec.run_dir} is not the rho=0.5 residual run")
        reports[spec.slug] = df
        question_sets.append(set(df["question_id"]))

    if any(question_ids != question_sets[0] for question_ids in question_sets[1:]):
        raise ValueError("The scoring-guidance runs do not contain the same questions")
    if len(question_sets[0]) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"Expected {EXPECTED_QUESTIONS} matched questions; "
            f"found {len(question_sets[0])}"
        )
    return reports


def align_reports(reports: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    question_ids = set(next(iter(reports.values()))["question_id"])
    order = [
        question_id
        for question_id in common.simpleqa_full_question_order()
        if question_id in question_ids
    ]
    return {
        slug: df.set_index("question_id").loc[order].reset_index()
        for slug, df in reports.items()
    }


def question_metrics(reports: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        slug: common.report_metrics(df, p=NOMINAL_COVERAGE)
        for slug, df in reports.items()
    }


def includes_idk(df: pd.DataFrame) -> pd.Series:
    return df["candidates"].map(
        lambda candidates: any(
            candidate["grade"] == "not_attempted"
            and candidate["probability"] > 0
            for candidate in candidates
        )
    )


def concentrated_report(df: pd.DataFrame) -> pd.Series:
    return df["candidates"].map(
        lambda candidates: any(
            candidate["grade"] != "not_attempted"
            and candidate["probability"] >= 0.995
            for candidate in candidates
        )
    )


def write_summary(
    reports: dict[str, pd.DataFrame],
    metrics: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    rows = []
    for spec in RULES:
        df = reports[spec.slug]
        q = metrics[spec.slug]
        rows.append(
            {
                "scoring_guidance": spec.label,
                "n_questions": len(q),
                "mean_list_size": q["list_size"].mean(),
                "single_candidate_rate": (q["list_size"] == 1).mean(),
                "concentrated_report_rate": concentrated_report(df).mean(),
                "reports_including_idk_rate": includes_idk(df).mean(),
                "mean_idk_mass": q["idk_mass"].mean(),
                "mean_top_concrete_probability": q[
                    "top_concrete_probability"
                ].mean(),
                "mean_reported_mass": df["log_reported_mass"].mean(),
                "truth_in_list_rate": q["truth_in_list"].mean(),
                "coverage_rate": q["coverage"].mean(),
                "coverage_or_idk_rate": q["coverage_or_idk"].mean(),
                "incorrect_rate": q["incorrect"].mean(),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "FS5A_rule_sensitivity.csv", index=False)


def rate_with_interval(values: pd.Series) -> tuple[float, float, float]:
    successes = int(values.sum())
    n = len(values)
    low, high = common.wilson_interval(successes, n)
    return successes / n, low, high


def mean_with_interval(values: pd.Series) -> tuple[float, float, float]:
    values = values.astype(float).dropna()
    estimate = values.mean()
    half_width = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return estimate, estimate - half_width, estimate + half_width


def make_summary_figure(
    reports: dict[str, pd.DataFrame],
    metrics: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Create the stated-report and graded-outcome panels in Fig. S5A."""
    x = np.arange(len(RULES))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharex=True)
    panels = (
        (
            axes[0],
            "Stated report",
            (
                (
                    "reports including IDK",
                    [
                        rate_with_interval(includes_idk(reports[spec.slug]))
                        for spec in RULES
                    ],
                    "#9334e6",
                ),
                (
                    "mean top-candidate probability",
                    [
                        mean_with_interval(metrics[spec.slug]["top_concrete_probability"])
                        for spec in RULES
                    ],
                    "#e8710a",
                ),
                (
                    "mean reported IDK probability",
                    [
                        mean_with_interval(metrics[spec.slug]["idk_mass"])
                        for spec in RULES
                    ],
                    "#188038",
                ),
            ),
        ),
        (
            axes[1],
            "Graded outcomes",
            (
                (
                    "coverage-or-IDK@0.9",
                    [
                        rate_with_interval(metrics[spec.slug]["coverage_or_idk"])
                        for spec in RULES
                    ],
                    "#188038",
                ),
                (
                    "full-list coverage",
                    [
                        rate_with_interval(metrics[spec.slug]["truth_in_list"])
                        for spec in RULES
                    ],
                    "#1a73e8",
                ),
                (
                    "incorrect@0.9",
                    [
                        rate_with_interval(metrics[spec.slug]["incorrect"])
                        for spec in RULES
                    ],
                    "#d93025",
                ),
            ),
        ),
    )
    ticks = [spec.tick for spec in RULES]
    for ax, title, series in panels:
        for label, estimates, color in series:
            values = np.array([estimate for estimate, _, _ in estimates])
            lows = np.array([low for _, low, _ in estimates])
            highs = np.array([high for _, _, high in estimates])
            ax.errorbar(
                x,
                values,
                yerr=[values - lows, highs - values],
                marker="o",
                ms=4.5,
                lw=1.6,
                capsize=2.5,
                color=color,
                label=label,
            )
        ax.set_xticks(x, ticks, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18, lw=0.6)
        ax.legend(frameon=False, fontsize=8.5, loc="center left")
    axes[0].set_ylabel("Fraction or mean reported probability")

    fig.tight_layout()
    target = out_dir / "FS5A_rule_sensitivity.pdf"
    fig.savefig(target)
    plt.close(fig)
    print(f"wrote {target}")


OUTCOME_ROWS = ("abstain", "correct", "incorrect")
OUTCOME_COLUMNS = ("incorrect", "correct", "abstain")


def report_outcome(candidates: list[dict]) -> str:
    prefix = common.top_p_set(candidates, NOMINAL_COVERAGE)
    if prefix.covers:
        return "correct"
    if prefix.has_idk:
        return "abstain"
    return "incorrect"


def make_outcome_figure(reports: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Compare each alternative prompt with the residual-log prompt."""
    base = reports["residual_log"]
    base_outcomes = {
        question_id: report_outcome(candidates)
        for question_id, candidates in zip(base["question_id"], base["candidates"])
    }

    matrices = []
    for spec in reversed(RULES[1:]):
        matrix = {
            row: {column: 0 for column in OUTCOME_COLUMNS}
            for row in OUTCOME_ROWS
        }
        for question_id, candidates in zip(
            reports[spec.slug]["question_id"], reports[spec.slug]["candidates"]
        ):
            matrix[base_outcomes[question_id]][report_outcome(candidates)] += 1
        matrices.append((spec, matrix))

    vmax = max(
        value
        for _, matrix in matrices
        for row in matrix.values()
        for value in row.values()
    )
    norm = Normalize(vmin=0, vmax=vmax)
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.35 * len(matrices), 4.15))
    csv_rows = []
    for index, (ax, (spec, matrix)) in enumerate(
        zip(np.atleast_1d(axes).ravel(), matrices)
    ):
        values = np.array(
            [[matrix[row][column] for column in OUTCOME_COLUMNS] for row in OUTCOME_ROWS],
            dtype=float,
        )
        rgb = np.asarray(to_rgb(spec.color))
        midpoint = tuple(0.64 * np.ones(3) + 0.36 * rgb)
        color_map = LinearSegmentedColormap.from_list(
            spec.slug, ["#ffffff", midpoint, spec.color]
        )
        ax.imshow(values, cmap=color_map, norm=norm)
        row_totals = values.sum(axis=1)
        for row_index in range(3):
            for column_index in range(3):
                value = values[row_index, column_index]
                percent = (
                    100 * value / row_totals[row_index]
                    if row_totals[row_index]
                    else 0
                )
                rgba = color_map(norm(value))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.6 else "black"
                ax.annotate(
                    f"{value:.0f}",
                    xy=(column_index, row_index),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=11.5,
                    fontweight=700,
                    color=text_color,
                )
                ax.annotate(
                    f"({percent:.1f}%)",
                    xy=(column_index, row_index),
                    xytext=(0, -8),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color=text_color,
                )
                csv_rows.append(
                    {
                        "scoring_guidance": spec.label,
                        "residual_log_outcome": OUTCOME_ROWS[row_index],
                        "alternative_outcome": OUTCOME_COLUMNS[column_index],
                        "count": int(value),
                    }
                )
        ax.set_xticks(range(3), OUTCOME_COLUMNS, fontsize=10)
        ax.set_yticks(range(3))
        if index == 0:
            ax.set_yticklabels(
                [
                    f"{row}\n({int(total)})"
                    for row, total in zip(OUTCOME_ROWS, row_totals)
                ],
                fontsize=10,
            )
            ax.set_ylabel(f"Log top-{NOMINAL_COVERAGE:g} set outcome", fontsize=11)
        else:
            ax.set_yticklabels([])
        ax.set_title(
            {"no_rule": "unspecified rule", "linear": "linear score (improper)"}.get(
                spec.slug, spec.label
            ),
            fontsize=12,
        )

    fig.tight_layout()
    target = out_dir / "FS5B_rule_outcome_tables.pdf"
    fig.savefig(target)
    plt.close(fig)
    pd.DataFrame(csv_rows).to_csv(
        out_dir / "FS5B_rule_outcome_tables.csv", index=False
    )
    print(f"wrote {target}")


def write_paired_contrasts(metrics: dict[str, pd.DataFrame], out_dir: Path) -> None:
    indexed = {
        slug: frame.set_index("question_id") for slug, frame in metrics.items()
    }
    comparisons = [(spec.slug, "residual_log") for spec in RULES[1:]] + [
        ("brier", "quadratic")
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for first_slug, second_slug in comparisons:
        first = indexed[first_slug]
        second = indexed[second_slug]
        for metric in common.REPORT_METRICS:
            differences = (
                first[metric].astype(float) - second[metric].astype(float)
            ).to_numpy()
            differences = differences[~np.isnan(differences)]
            samples = np.array(
                [
                    differences[
                        rng.integers(0, len(differences), len(differences))
                    ].mean()
                    for _ in range(BOOTSTRAP_DRAWS)
                ]
            )
            ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
            rows.append(
                {
                    "comparison": f"{first_slug}_minus_{second_slug}",
                    "metric": metric,
                    "n_questions": len(differences),
                    "difference": differences.mean(),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    pd.DataFrame(rows).to_csv(
        out_dir / "FS5_rule_paired_comparisons.csv", index=False
    )


def run(results_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = align_reports(load_reports(results_dir))
    metrics = question_metrics(reports)
    write_summary(reports, metrics, out_dir)
    write_paired_contrasts(metrics, out_dir)
    make_summary_figure(reports, metrics, out_dir)
    make_outcome_figure(reports, out_dir)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    run(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
