"""Reproduce Fig. 5 and Table S8: calibration of threshold decisions.

Uses retained, graded candidates without further merging or normalization.
An empty candidate list always abstains. Bootstrap samples resample questions;
grouping identical (top probability, correctness) pairs is an exact computational
shortcut for that empirical bootstrap, including empty reports as one category.

Run: python analysis/threshold_calibration.py [--grader gemini]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import common

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260911
MIN_PLOT_ANSWERS = 30
THRESHOLDS = np.unique(np.r_[np.arange(101) / 100, 0.75, 6 / 7])
PENALTY_THRESHOLDS = {3: 0.75, 6: 6 / 7}


def grouped_reports(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (top probability, top correctness) categories and their counts.

    A probability of -1 identifies empty reports. Ties use retained entry order,
    matching common.posthoc_decision. Correctness is read only after selection.
    """
    groups: dict[tuple[float, int], int] = {}
    ids = [r["question_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Expected one retained report per question")
    for record in records:
        concrete = [c for c in common.parse_candidates(record)
                    if c["grade"] != "not_attempted"]
        if concrete:
            top = max(concrete, key=lambda c: c["probability"])
            u = float(top["probability"])
            if not 0 <= u <= 1:
                raise ValueError(f"Invalid top probability: {u}")
            pair = (u, int(top["grade"] == "correct"))
        else:
            pair = (-1., 0)
        groups[pair] = groups.get(pair, 0) + 1
    if not groups:
        raise ValueError("No reports found")
    pairs = sorted(groups)
    return np.asarray(pairs), np.asarray([groups[p] for p in pairs])


def rates(groups: np.ndarray, counts: np.ndarray,
          thresholds: np.ndarray) -> dict[str, np.ndarray]:
    """Rates for one count vector or a batch of bootstrap count vectors."""
    selected = ((groups[:, 0, None] >= thresholds[None, :])
                & (groups[:, 0, None] >= 0)).astype(float)
    answered = counts @ selected
    incorrect = counts @ ((1 - groups[:, 1, None]) * selected)
    expected_incorrect = counts @ ((1 - groups[:, 0, None]) * selected)
    totals = np.sum(counts, axis=-1)[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        observed = np.where(answered > 0, incorrect / answered, np.nan)
        reported = np.where(answered > 0, expected_incorrect / answered, np.nan)
    return {"n_answered": answered, "answer_rate": answered / totals,
            "observed_error": observed, "reported_error": reported,
            "error_gap": observed - reported}


def summarize(records: list[dict], thresholds: np.ndarray, *, draws: int,
              seed: int) -> pd.DataFrame:
    groups, counts = grouped_reports(records)
    estimates = rates(groups, counts, thresholds)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(int(counts.sum()), counts / counts.sum(), size=draws)
    bootstrap = rates(groups, weights, thresholds)
    output = pd.DataFrame({"threshold": thresholds, **estimates})
    output["n_questions"] = int(counts.sum())
    output["n_nonempty"] = int(counts[groups[:, 0] >= 0].sum())
    output["n_answered"] = output["n_answered"].astype(int)
    output["bootstrap_valid_draws"] = np.isfinite(bootstrap["observed_error"]).sum(axis=0)
    for metric in ("answer_rate", "observed_error", "reported_error", "error_gap"):
        intervals = np.full((2, len(thresholds)), np.nan)
        for j in range(len(thresholds)):
            values = bootstrap[metric][:, j]
            values = values[np.isfinite(values)]
            if len(values):
                intervals[:, j] = np.quantile(values, [0.025, 0.975])
        output[f"{metric}_lo"] = intervals[0]
        output[f"{metric}_hi"] = intervals[1]
    return output


def plot_curves(data: pd.DataFrame, target: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    n_models = len(common.MODEL_ORDER)
    fig, axes = plt.subplots(1, n_models, figsize=(3.8 * n_models, 3.8),
                             sharex=True, sharey=True, squeeze=False)
    for index, (ax, key) in enumerate(zip(axes.flat, common.MODEL_ORDER)):
        part = data[data.model_key == key].sort_values("threshold")
        mask = part.n_answered >= MIN_PLOT_ANSWERS
        t = part.threshold.to_numpy()
        color = common.MODEL_COLORS[key]
        observed = np.where(mask, 100 * part.observed_error, np.nan)
        reported = np.where(mask, 100 * part.reported_error, np.nan)
        ax.plot(t, observed, color=color, linewidth=2)
        ax.plot(t, reported, color=color, linewidth=2, linestyle="--")
        ax.fill_between(t, 100 * part.observed_error_lo, 100 * part.observed_error_hi,
                        where=mask, color=color, alpha=.17, linewidth=0)
        ax.plot(t, 100 * (1 - t), color="0.55", linestyle=":", linewidth=1.3)
        ax.set_title(f"{chr(65 + index)}  {common.RUNS[key]['label']}",
                     loc="left", fontweight="bold", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.grid(alpha=.18)
        ax.set_xlabel("Answer threshold, $t$")
        if index == 0:
            ax.set_ylabel("Error among answered questions (%)")
    handles = [Line2D([0], [0], color="0.2", lw=2, label="Observed error"),
               Line2D([0], [0], color="0.2", lw=2, ls="--", label="Mean reported error"),
               Line2D([0], [0], color="0.55", lw=1.3, ls=":", label="Calibrated bound: $1-t$")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .01))
    fig.tight_layout(rect=(0, .11, 1, 1), w_pad=1.2)
    fig.savefig(target, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


def run(results_dir: Path, out_dir: Path, *, include_figure_outputs: bool = True,
        include_table_output: bool = True, draws: int = BOOTSTRAP_DRAWS,
        seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    frames = []
    for index, key in enumerate(common.MODEL_ORDER):
        records = common.load_jsonl(results_dir / common.RUNS[key]["run"]
                                    / "simpleqa_topp_results.jsonl")
        if len(records) != 4326:
            raise ValueError(f"{key}: expected 4,326 questions; found {len(records)}")
        frame = summarize(records, THRESHOLDS, draws=draws, seed=seed + index)
        frame.insert(0, "model", common.RUNS[key]["label"])
        frame.insert(0, "model_key", key)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if include_figure_outputs:
        data.to_csv(out_dir / "F5_ThresholdCalibration.csv", index=False)
        plot_curves(data, out_dir / "F5_ThresholdCalibration.pdf")
    if include_table_output:
        rows = []
        for penalty, threshold in PENALTY_THRESHOLDS.items():
            part = data[data.threshold == threshold].copy()
            part.insert(2, "L", penalty)
            rows.append(part)
        table = pd.concat(rows).sort_values(["model_key", "L"])
        table.to_csv(out_dir / "TableS8_ThresholdCalibration.csv", index=False)
    metadata = {"bootstrap_draws": draws, "bootstrap_seed": seed,
                "seed_offset": "model index in common.MODEL_ORDER",
                "bootstrap_unit": "question (including empty reports)",
                "intervals": "pointwise paired percentile, 95%",
                "plot_min_answered": MIN_PLOT_ANSWERS,
                "processing": "retained graded entries; no further merging or normalization",
                "model_order": list(common.MODEL_ORDER),
                "results_dir": str(results_dir.resolve().relative_to(common.REPO_ROOT)) if results_dir.resolve().is_relative_to(common.REPO_ROOT) else str(results_dir.resolve())}
    (out_dir / "threshold_calibration_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(f"Threshold calibration: {out_dir}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grader", choices=("openai", "gemini"), default="openai")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    out = args.out_dir or root / "results" / ("figures" if args.grader == "openai"
                                             else "figures_gemini")
    run(root / "results" / f"graded_by_{args.grader}", out)


if __name__ == "__main__":
    main()
