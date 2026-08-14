"""Stage 1: regenerate the paper figures built around the residual-rho
SimpleQA elicitation and penalty-sample behavior.

Figure roles:
  - F1: residual-rho frontier versus penalty-prompt point.
  - F1A: same frontier view with accuracy on the y axis.
  - F2: sample-average penalty/log outcome table (formerly F17).
  - F3: cumulative rates as the paired question prefix grows (formerly F14).
  - F3A: cumulative three-way top-p outcome rates.
  - F4: residual-log versus penalty convergence (formerly appendix FA1).
  - F5A: threshold response curves using three-way outcome definitions.
  - F6: distinct responses per question.
  - F7: repeated log-elicitation consistency.
  - F8: question-level replication variability of reported distributions.
  - F9: question-level mean pairwise JSD diagnostics.

Pairing: all contrasts are paired by question_id in the deterministic
SimpleQA order whenever both residual-log and penalty rows are available.

Run:  python analysis/make_figures.py   (see analysis/reproduce_all.py)
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    MODEL_COLORS,
    MODEL_ORDER,
    OUTPUTS_DIR,
    RUNS,
    frontier,
    norm_answer,
    load_jsonl,
    load_run,
    posthoc_decision,
    result_file,
    open_result,
    simpleqa_full_question_order,
    top_p_set,
)

OUT = Path(__file__).resolve().parent.parent / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
ACTIVE_MODEL_ORDER = list(MODEL_ORDER)

NOMINAL_P = 0.9
BOOT_BAND_REPS = 500
BOOT_BAND_MIN_N = 5
BOOT_BAND_STEP = 10
PRIMARY_PENALTY = 3.0
PENALTY_LEVELS = [0.0, 3.0, 6.0]
PENALTY_COLORS = {0.0: "#111827", 3.0: "crimson", 6.0: "#7c3aed"}
PENALTY_BAR_COLORS = {0.0: "#fca5a5", 3.0: "#ef4444", 6.0: "#991b1b"}
PENALTY_LINESTYLES = {0.0: "-.", 3.0: "--", 6.0: ":"}
PENALTY_MARKERS = {0.0: "o", 3.0: "s", 6.0: "^"}
PENALTY_ALPHA = {0.0: 0.62, 3.0: 0.72, 6.0: 0.78}
RULE_STYLE = {"residual": "-", "naive": "--"}
RULE_ALPHA = {"residual": 1.0, "naive": 0.45}
DEFAULT_F7_RUN_DIR = "claudesonnet46_consistency"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# Only these figures appear in the manuscript.  savefig writes just these,
# under their manuscript names and as PDF only, and silently skips every other
# figure so the output directory holds only manuscript artifacts.  The
# concept figure (F1_Concept.pdf) is a hand-authored illustration shipped as
# a static asset in results/figures/.
FIGURE_RENAMES = {
    # F2A_Frontier/F2B_Frontier are written directly by fig_frontier with
    # fixed (non-tight) margins so the stacked rows stay aligned.
    "f2_penalty_log_outcome_table_sample_average": "F3_OutcomeTable_L3",
    "f2_penalty_log_outcome_table_sample_average_L0": "FS1_OutcomeTable_L0",
    "f2_penalty_log_outcome_table_sample_average_L6": "FS1_OutcomeTable_L6",
    "f9_log_consistency_jsd_diagnostics": "FS2_JSD",
    "f3a_cumulative_three_way_rates": "FS3_cumulative",
    "f6_responses_per_question": "FS4_response_counts",
}


def savefig(fig, name: str):
    manuscript_name = FIGURE_RENAMES.get(name)
    if manuscript_name is not None:
        fig.savefig(
            OUT / f"{manuscript_name}.pdf", bbox_inches="tight", pad_inches=0.02
        )
    plt.close(fig)
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


def model_key_for_id(model_id: str) -> str | None:
    for key, cfg in RUNS.items():
        if cfg["model_id"] == model_id:
            return key
    return None


MODEL_ALIASES = {
    "gemini": "gemini35flash",
    "gemini35": "gemini35flash",
    "gemini35flash": "gemini35flash",
    "google/gemini-3.5-flash": "gemini35flash",
    "sonnet": "sonnet46",
    "sonnet46": "sonnet46",
    "claude-sonnet-4-6": "sonnet46",
    "deepseek": "deepseekv32maas",
    "deepseekv32": "deepseekv32maas",
    "deepseekv32maas": "deepseekv32maas",
    "deepseek-ai/deepseek-v3.2-maas": "deepseekv32maas",
    "qwen": "qwen3_235b",
    "qwen3": "qwen3_235b",
    "qwen3_235b": "qwen3_235b",
    "qwen3-235b-a22b-instruct-2507-maas": "qwen3_235b",
    "qwen/qwen3-235b-a22b-instruct-2507-maas": "qwen3_235b",
}


def parse_model_keys(values: list[str]) -> list[str]:
    keys = []
    for value in values:
        for raw_part in str(value).split(","):
            part = raw_part.strip()
            if not part:
                continue
            key = MODEL_ALIASES.get(part, part)
            if key not in RUNS:
                valid = ", ".join(RUNS)
                raise SystemExit(f"Unknown model '{part}'. Valid model keys: {valid}")
            if key not in keys:
                keys.append(key)
    if not keys:
        raise SystemExit("At least one model must be selected.")
    return keys


def configure_run(*, model_keys: list[str], out_dir: Path) -> None:
    global ACTIVE_MODEL_ORDER, OUT
    ACTIVE_MODEL_ORDER = model_keys
    OUT = Path(out_dir)
    OUT.mkdir(parents=True, exist_ok=True)


def model_panel_subplots(*, height: float, sharex: bool = False, sharey: bool = False):
    n = len(ACTIVE_MODEL_ORDER)
    fig_width = max(4.8, 4.35 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_width, height), sharex=sharex, sharey=sharey)
    return fig, np.atleast_1d(axes).ravel()


def penalty_label(penalty: float) -> str:
    return f"L={penalty:g}"


def penalty_tag(penalty: float) -> str:
    return f"L{penalty:g}".replace(".", "p")


def penalty_threshold(penalty: float) -> float:
    return penalty / (1.0 + penalty)


def penalty_run_dir_name(cfg: dict, penalty: float) -> str:
    if np.isclose(penalty, PRIMARY_PENALTY):
        return cfg["residual"]
    return f"{cfg['residual']}_L{penalty:g}"


def load_paired():
    """Return per model:
      'residual', 'naive'            -- elicitation arms paired on the naive
                                        rows, for paired naive-vs-residual
                                        deltas;
      'residual_all'                 -- all available residual-rho log rows;
      'penalty'                      -- deduplicated penalty arm; prefers the
                                        standalone residual-dir penalty file
                                        when present, else falls back to full;
      'residual_penalty'             -- residual elicitation restricted to the
                                        penalty-arm questions.
    All alignments are verified."""
    global ACTIVE_MODEL_ORDER
    out = {}
    loaded_keys = []
    canonical_order = list(simpleqa_full_question_order())
    for key in list(ACTIVE_MODEL_ORDER):
        try:
            dfull = load_run(RUNS[key]["full"]).set_index("question_id", drop=False)
            dnaive = load_run(RUNS[key]["log200"]).set_index("question_id", drop=False)
            dres_all = load_run(RUNS[key]["residual"]).set_index("question_id", drop=False)
        except FileNotFoundError as exc:
            print(f"  {key}: skipped (missing finalized run file: {exc.filename})")
            continue
        # arms may have been extended independently; pair on the (ordered)
        # intersection and keep the full residual arm separately.
        naive_set = set(dnaive["question_id"])
        residual_set = set(dres_all["question_id"])
        common = [q for q in canonical_order if q in residual_set and q in naive_set]
        common.extend(
            q for q in dres_all["question_id"]
            if q in naive_set and q not in set(common)
        )
        if not common:
            print(f"  {key}: skipped (no shared questions between naive and residual)")
            continue
        dres = dres_all.loc[common]
        dnaive = dnaive.loc[common]
        full_set = set(dfull["question_id"])
        fq = [q for q in canonical_order if q in set(common) and q in full_set]
        fq.extend(q for q in common if q in full_set and q not in set(fq))
        if len(fq) != len(dfull):
            print(f"  {key}: skipped (full-run questions are not nested)")
            continue
        penalties = {}
        residual_by_penalty = {}
        for penalty in PENALTY_LEVELS:
            dpen_all = load_penalty_arm(key, penalty)
            if dpen_all.empty:
                if np.isclose(penalty, PRIMARY_PENALTY):
                    print(f"  {key}: skipped (no primary penalty-arm questions available)")
                    penalties = {}
                    break
                print(f"  {key}: penalty {penalty_label(penalty)} rows=0 (not found)")
                continue
            penalty_set = set(dpen_all["question_id"])
            pq = [q for q in canonical_order if q in residual_set and q in penalty_set]
            pq.extend(
                q for q in dres_all["question_id"]
                if q in penalty_set and q not in set(pq)
            )
            if not pq:
                if np.isclose(penalty, PRIMARY_PENALTY):
                    print(f"  {key}: skipped (no paired primary penalty-arm questions)")
                    penalties = {}
                    break
                print(f"  {key}: penalty {penalty_label(penalty)} rows=0 (no paired qids)")
                continue
            dpen = dpen_all.loc[pq]
            penalties[float(penalty)] = dpen
            residual_by_penalty[float(penalty)] = dres_all.loc[pq]
            print(
                f"  {key}: penalty {penalty_label(penalty)} rows={len(dpen)} "
                f"(source={dpen.attrs.get('source')}, raw_rows={dpen.attrs.get('raw_rows')})"
            )
        if PRIMARY_PENALTY not in penalties:
            continue
        dpen = penalties[PRIMARY_PENALTY]
        out[key] = {
            "residual": dres,            # paired with the naive arm
            "residual_all": dres_all,    # full residual arm (may be longer)
            "naive": dnaive,
            "full": dfull.loc[fq],
            "penalty": dpen,
            "penalties": penalties,
            "residual_full": dres.loc[fq],
            "residual_penalty": residual_by_penalty[PRIMARY_PENALTY],
            "residual_by_penalty": residual_by_penalty,
            "naive_full": dnaive.loc[fq],
        }
        loaded_keys.append(key)
    if not out:
        raise SystemExit("No selected models had complete finalized log and penalty rows.")
    ACTIVE_MODEL_ORDER = loaded_keys
    return out


PENALTY_NUMERIC_COLS = [
    "n_samples_requested",
    "penalty_value",
    "penalty_threshold",
    "penalty_answered_samples",
    "penalty_abstain_samples",
    "penalty_correct_samples",
    "penalty_incorrect_samples",
    "penalty_not_attempted_samples",
    "penalty_accuracy_overall",
    "penalty_accuracy_when_answered",
    "penalty_hallucination_rate",
    "penalty_abstention_rate",
    "penalty_total_tokens",
]


def _penalty_df_from_rows(
    rows: list[dict],
    *,
    model_id: str,
    source: str,
    penalty: float | None = None,
) -> pd.DataFrame:
    """Return one penalty row per (model, question_id), keeping the last row.

    The standalone penalty runner is resumable, so an interrupted/restarted
    run can append duplicate question rows.  The generation seeds are fixed;
    for analysis we treat the latest completed row as the canonical one.
    """
    dedup: dict[tuple[str, str], dict] = {}
    raw_rows = 0
    for r in rows:
        if r.get("model") != model_id or not r.get("question_id"):
            continue
        if penalty is not None:
            try:
                row_penalty = float(r.get("penalty_value"))
            except (TypeError, ValueError):
                continue
            if not np.isclose(row_penalty, penalty):
                continue
        raw_rows += 1
        qid = str(r["question_id"])
        row = dict(r)
        row["question_id"] = qid
        dedup[(row["model"], qid)] = row
    df = pd.DataFrame(dedup.values())
    if df.empty:
        return df
    df = df.set_index("question_id", drop=False)
    for col in PENALTY_NUMERIC_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.attrs["source"] = source
    df.attrs["raw_rows"] = raw_rows
    return df


def load_penalty_arm(key: str, penalty: float = PRIMARY_PENALTY) -> pd.DataFrame:
    """Load one penalty arm for a model.

    L=3 keeps the historical behavior: prefer the standalone residual-dir
    penalty file and fall back to the legacy full run.  Other penalty levels
    are read from separate directories, e.g. *_L6.
    """
    cfg = RUNS[key]
    candidates = [
        OUTPUTS_DIR / penalty_run_dir_name(cfg, penalty) / "simpleqa_penalty_results.jsonl",
    ]
    if np.isclose(penalty, PRIMARY_PENALTY):
        candidates.append(OUTPUTS_DIR / cfg["full"] / "simpleqa_topp_results.jsonl")
    for candidate_path in candidates:
        standalone = result_file(candidate_path)
        if standalone is None:
            continue
        df = _penalty_df_from_rows(
            load_jsonl(standalone),
            model_id=cfg["model_id"],
            source=str(standalone.relative_to(OUTPUTS_DIR)),
            penalty=penalty,
        )
        if not df.empty:
            return df
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# F1: frontier, both rules
# ---------------------------------------------------------------------------

def _frontier_decision_matrices(
    df: pd.DataFrame,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-question log-threshold decisions used for frontier bootstraps."""
    outcomes_by_t = [
        df["candidates"].map(lambda cands, t=t: posthoc_decision(cands, t)).to_numpy()
        for t in thresholds
    ]
    outcomes = np.column_stack(outcomes_by_t)
    incorrect = (outcomes == "incorrect").astype(float)
    correct = (outcomes == "correct").astype(float)
    abstain = (outcomes == "abstain").astype(float)
    return incorrect, correct, abstain


def _interp_frontier_on_abstention(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate a monotone threshold frontier onto abstention-rate values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) == 0:
        return np.full_like(x_grid, np.nan, dtype=float)
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    unique_y = np.zeros_like(unique_x, dtype=float)
    for idx in range(len(unique_x)):
        unique_y[idx] = np.nanmean(y[inverse == idx])
    if len(unique_x) == 1:
        return np.full_like(x_grid, unique_y[0], dtype=float)
    return np.interp(x_grid, unique_x, unique_y,
                     left=unique_y[0], right=unique_y[-1])


def bootstrap_frontier_bands(
    df: pd.DataFrame,
    thresholds: np.ndarray,
    x_grid: np.ndarray,
    *,
    seed: int,
    n_boot: int = BOOT_BAND_REPS,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Question-level bootstrap bands for log frontiers.

    The bootstrap resamples questions, recomputes the threshold frontier, and
    interpolates each replicate onto a common abstention-rate grid.
    """
    n = len(df)
    if n == 0:
        empty = (np.full_like(x_grid, np.nan, dtype=float),
                 np.full_like(x_grid, np.nan, dtype=float))
        return {"hallucination_rate": empty, "accuracy_overall": empty}
    incorrect, correct, abstain = _frontier_decision_matrices(df, thresholds)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot) / n
    boot_abst = weights @ abstain
    boot_halluc = weights @ incorrect
    boot_acc = weights @ correct

    out = {}
    for name, values in (
        ("hallucination_rate", boot_halluc),
        ("accuracy_overall", boot_acc),
    ):
        interp = np.vstack([
            _interp_frontier_on_abstention(boot_abst[b], values[b], x_grid)
            for b in range(n_boot)
        ])
        lo, hi = np.nanquantile(interp, [0.025, 0.975], axis=0)
        out[name] = (lo, hi)
    return out


def fig_frontier(paired):
    """Writes F2A (hallucination row) and F2B (accuracy row) as separate
    PDFs with identical fixed margins so the manuscript can stack them under
    LaTeX-set panel letters (FS1 style).  Do not save with a tight bbox:
    the shared margins are what keep the two rows' panels aligned."""
    print("== F1: frontier (residual-rho prompt vs penalty) ==")
    grid = np.linspace(0, 0.95, 96)
    band_rows = []
    n_models = len(ACTIVE_MODEL_ORDER)
    fig_width = max(5.0, 4.05 * n_models)
    row_figs = []
    row_axes = []
    # Row B is taller: it carries the shared legend below its x label.
    for row_height in (3.09, 3.69):
        fig, axes = plt.subplots(
            1,
            n_models,
            figsize=(fig_width, row_height),
            sharey=True,
            squeeze=False,
        )
        row_figs.append(fig)
        row_axes.append(axes[0])
    row_specs = [
        ("hallucination_rate", "penalty_hallucination_rate",
         "empirical_hallucination_rate"),
        ("accuracy_overall", "penalty_accuracy_overall",
         "empirical_accuracy_overall"),
    ]
    for col, key in enumerate(ACTIVE_MODEL_ORDER):
        c = MODEL_COLORS[key]
        dres = paired[key]["residual_penalty"]
        fr = frontier(dres, grid)
        band_x = np.linspace(
            float(fr["abstention_rate"].min()),
            float(fr["abstention_rate"].max()),
            121,
        )
        bands = bootstrap_frontier_bands(
            dres,
            grid,
            band_x,
            seed=91000 + 97 * col,
        )
        dfull = paired[key]["full"]
        for row, (frontier_col, penalty_col, empirical_col) in enumerate(row_specs):
            ax = row_axes[row][col]
            band_lo, band_hi = bands[frontier_col]
            ax.fill_between(
                band_x,
                band_lo,
                band_hi,
                color=c,
                alpha=0.38,
                linewidth=0,
            )
            for xval, lo, hi in zip(band_x, band_lo, band_hi):
                band_rows.append({
                    "model": key,
                    "model_label": RUNS[key]["label"],
                    "metric": frontier_col,
                    "abstention_rate": xval,
                    "lo_95": lo,
                    "hi_95": hi,
                })
            ax.plot(
                fr["abstention_rate"],
                fr[frontier_col],
                color=c,
                lw=2,
                label="thresholded log report frontier",
            )
            for penalty, dpen in paired[key]["penalties"].items():
                ax.scatter(
                    [dpen["penalty_abstention_rate"].mean()],
                    [dpen[penalty_col].mean()],
                    color=PENALTY_COLORS.get(penalty, "crimson"),
                    marker="o",
                    s=52 if np.isclose(penalty, PRIMARY_PENALTY) else 44,
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=5,
                    label=f"penalty {penalty_label(penalty)}",
                )
            if _has_empirical_baseline(dfull):
                ax.scatter(
                    [dfull["empirical_not_attempted_rate"].mean()],
                    [dfull[empirical_col].mean()],
                    color="black",
                    marker="s",
                    s=42,
                    zorder=5,
                    label=f"baseline prompt (≈L=0, n={len(dfull)})",
                )
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 0.90 if row == 0 else 0.65)
            ax.grid(alpha=0.18, lw=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", labelsize=11.5)
            if row == 1:
                ax.set_xlabel("abstention", fontsize=14)
        row_axes[0][col].set_title(
            RUNS[key]["label"],
            fontsize=16,
            color="#111827",
        ).set_path_effects([withStroke(linewidth=0.5, foreground="#111827")])
    row_axes[0][0].set_ylabel("hallucination", fontsize=14)
    row_axes[1][0].set_ylabel("accuracy", fontsize=14)
    pd.DataFrame(band_rows).to_csv(OUT / "F2_Frontier.csv", index=False)
    shared_margins = dict(left=0.042, right=0.995, wspace=0.14)
    # Keep the axes height identical across the two rows so the stacked
    # panels stay aligned: (top-bottom)*height must equal 2.45in in both.
    row_figs[0].subplots_adjust(top=0.8835, bottom=0.0905, **shared_margins)
    row_figs[1].subplots_adjust(top=0.9755, bottom=0.3115, **shared_margins)
    legend_handles, legend_labels = row_axes[1][0].get_legend_handles_labels()
    legend_handles = [
        Line2D([0], [0], color="black", lw=2)
        if label.startswith("thresholded")
        else handle
        for handle, label in zip(legend_handles, legend_labels)
    ]
    row_figs[1].legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5185, 0.01),
        ncol=len(legend_labels),
        frameon=False,
        fontsize=13.5,
        markerscale=1.6,
    )
    for fig, name in zip(row_figs, ("F2A_Frontier", "F2B_Frontier")):
        fig.savefig(OUT / f"{name}.pdf")
        plt.close(fig)
        print(f"  wrote {name}.pdf")


def fig_frontier_accuracy(paired):
    print("== F1A: accuracy frontier (residual-rho prompt vs penalty) ==")
    grid = np.linspace(0, 0.95, 96)
    fig, axes = model_panel_subplots(height=4)
    for ax, key in zip(axes, ACTIVE_MODEL_ORDER):
        c = MODEL_COLORS[key]
        dres = paired[key]["residual_penalty"]
        fr = frontier(dres, grid)
        ax.plot(
            fr["abstention_rate"],
            fr["accuracy_overall"],
            color=c,
            lw=2,
            label="post-hoc rule on elicited dist.",
        )
        dfull = paired[key]["full"]
        for penalty, dpen in paired[key]["penalties"].items():
            ax.scatter(
                [dpen["penalty_abstention_rate"].mean()],
                [dpen["penalty_accuracy_overall"].mean()],
                color=PENALTY_COLORS.get(penalty, "crimson"),
                marker="o",
                s=54 if np.isclose(penalty, PRIMARY_PENALTY) else 46,
                edgecolors="white",
                linewidths=0.6,
                zorder=5,
                label=f"penalty prompt {penalty_label(penalty)}",
            )
        if _has_empirical_baseline(dfull):
            ax.scatter(
                [dfull["empirical_not_attempted_rate"].mean()],
                [dfull["empirical_accuracy_overall"].mean()],
                color="black",
                marker="s",
                s=45,
                zorder=5,
                label=f"baseline prompt (≈L=0, n={len(dfull)})",
            )
        ax.set_title(RUNS[key]["label"])
        ax.set_xlabel("abstention rate")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.18, lw=0.6)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=7, loc="best")
    savefig(fig, "f1_accuracy_frontier")


# ---------------------------------------------------------------------------
# F5: hallucination and abstention as functions of the threshold t
# ---------------------------------------------------------------------------

def fig_threshold_response(paired):
    """Hallucination and abstention rates vs the decision threshold t.

    The elicitation arm yields genuine curves (one report, every t applied
    post hoc); a fixed penalty prompt is constant in t -- horizontal lines.
    Vertical guides mark each penalty prompt's own design point.
    """
    print("== F5: threshold response (rates vs t) ==")
    grid = np.linspace(0, 0.99, 100)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), sharex=True)
    panels = [
        ("hallucination_rate", "penalty_hallucination_rate",
         "hallucination rate (wrong & answered)"),
        ("abstention_rate", "penalty_abstention_rate", "abstention rate"),
        ("accuracy_overall", "penalty_accuracy_overall",
         "accuracy (correct / all questions)"),
        ("accuracy_when_answered", "penalty_accuracy_when_answered",
         "accuracy when answered (correct / answered)"),
    ]
    for key in ACTIVE_MODEL_ORDER:
        c = MODEL_COLORS[key]
        fr = frontier(paired[key]["residual_penalty"], grid)
        for ax, (col, pen_col, _) in zip(axes.flat, panels):
            ax.plot(fr["t"], fr[col], color=c, lw=2,
                    label=RUNS[key]["label"] if col == "hallucination_rate" else None)
            for penalty, dpen in paired[key]["penalties"].items():
                ax.axhline(
                    dpen[pen_col].mean(),
                    color=c,
                    lw=1.4,
                    ls=PENALTY_LINESTYLES.get(penalty, "--"),
                    alpha=PENALTY_ALPHA.get(penalty, 0.65),
                )
    for ax, (_, _, ylab) in zip(axes.flat, panels):
        for penalty in sorted({p for k in ACTIVE_MODEL_ORDER for p in paired[k]["penalties"]}):
            ax.axvline(penalty_threshold(penalty), color="gray", lw=0.9, ls="-.", alpha=0.45)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_xlim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("decision threshold t (answer iff top prob ≥ t)")
    axes[0, 0].annotate("penalty design thresholds", xy=(0.73, 0.97),
                        xycoords="axes fraction", fontsize=7, ha="right", color="gray")
    penalty_handles = [
        Line2D([0], [0],
               color="black",
               lw=1.5,
               ls=PENALTY_LINESTYLES.get(p, "--"),
               label=f"penalty {penalty_label(p)}")
        for p in sorted({p for k in ACTIVE_MODEL_ORDER for p in paired[k]["penalties"]})
    ]
    model_handles, _ = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles=model_handles + penalty_handles, fontsize=7, loc="upper right")
    fig.tight_layout()
    savefig(fig, "f5_threshold_response")


def fig_threshold_three_way_response(paired):
    """Three-way threshold response using the F3A outcome partition."""
    print("== F5A: three-way threshold response (rates vs t) ==")
    grid = np.linspace(0, 0.99, 100)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4), sharex=True, sharey=True)
    panels = [
        ("accuracy_overall", "accuracy_rate", "accuracy"),
        ("hallucination_rate", "hallucination_rate", "hallucination rate"),
        ("abstention_rate", "abstention_rate", "abstention rate"),
    ]
    rows = []
    for key in ACTIVE_MODEL_ORDER:
        c = MODEL_COLORS[key]
        fr = frontier(paired[key]["residual_penalty"], grid)
        for ax, (col, metric, _) in zip(axes, panels):
            ax.plot(
                fr["t"],
                fr[col],
                color=c,
                lw=2,
                label=RUNS[key]["label"] if col == "accuracy_overall" else None,
            )
            for t, value in zip(fr["t"], fr[col]):
                rows.append(
                    {
                        "model": key,
                        "model_label": RUNS[key]["label"],
                        "method": "residual_log_threshold",
                        "penalty_level": np.nan,
                        "threshold": t,
                        "metric": metric,
                        "value": value,
                    }
                )

        for penalty, dpen in paired[key]["penalties"].items():
            halluc, acc, abst = _penalty_three_way_arrays(dpen)
            penalty_values = {
                "accuracy_rate": float(np.mean(acc)),
                "hallucination_rate": float(np.mean(halluc)),
                "abstention_rate": float(np.mean(abst)),
            }
            for ax, (_, metric, _) in zip(axes, panels):
                ax.scatter(
                    [penalty_threshold(penalty)],
                    [penalty_values[metric]],
                    marker=PENALTY_MARKERS.get(penalty, "o"),
                    s=36,
                    color=c,
                    edgecolor="white",
                    linewidth=0.6,
                    alpha=PENALTY_ALPHA.get(penalty, 0.88),
                    zorder=4,
                )
                rows.append(
                    {
                        "model": key,
                        "model_label": RUNS[key]["label"],
                        "method": "native_penalty",
                        "penalty_level": penalty,
                        "threshold": penalty_threshold(penalty),
                        "metric": metric,
                        "value": penalty_values[metric],
                    }
                )

    available_penalties = sorted({p for k in ACTIVE_MODEL_ORDER for p in paired[k]["penalties"]})
    for ax, (_, _, ylab) in zip(axes, panels):
        ax.set_xlabel("decision threshold t")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.18, lw=0.6)

    model_handles = [
        Line2D([0], [0], color=MODEL_COLORS[key], lw=2, label=RUNS[key]["label"])
        for key in ACTIVE_MODEL_ORDER
    ]
    method_handles = [Line2D([0], [0], color="black", lw=2, ls="-", label="residual log")]
    for penalty in available_penalties:
        method_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                marker=PENALTY_MARKERS.get(penalty, "o"),
                markerfacecolor="black",
                markeredgecolor="white",
                markeredgewidth=0.6,
                lw=0,
                label=f"penalty prompt {penalty_label(penalty)}",
            )
        )
    axes[0].legend(handles=model_handles, fontsize=7, loc="best")
    axes[2].legend(
        handles=method_handles,
        fontsize=7,
        loc="best",
        handlelength=4.0,
        handletextpad=0.6,
        borderpad=0.35,
        labelspacing=0.45,
    )
    fig.tight_layout(w_pad=1.5)
    pd.DataFrame(rows).to_csv(OUT / "f5a_threshold_three_way_response.csv", index=False)
    savefig(fig, "f5a_threshold_three_way_response")


# ---------------------------------------------------------------------------
# Shared threshold helpers
# ---------------------------------------------------------------------------

def _log_threshold_inputs(dlog: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-concrete answer probability and grade arrays for threshold rules."""
    top_prob = []
    top_correct = []
    top_incorrect = []
    for candidates in dlog["candidates"]:
        concrete = [c for c in candidates if c["grade"] != "not_attempted"]
        if not concrete:
            top_prob.append(np.nan)
            top_correct.append(False)
            top_incorrect.append(False)
            continue
        top = max(concrete, key=lambda c: c["probability"])
        top_prob.append(float(top["probability"]))
        top_correct.append(top["grade"] == "correct")
        top_incorrect.append(top["grade"] == "incorrect")
    return (
        np.asarray(top_prob, dtype=float),
        np.asarray(top_correct, dtype=bool),
        np.asarray(top_incorrect, dtype=bool),
    )


# ---------------------------------------------------------------------------
# F6: elicited vs empirical, both rules
# ---------------------------------------------------------------------------

def _has_empirical_baseline(df: pd.DataFrame) -> bool:
    return (
        "empirical_accuracy_overall" in df
        and df["empirical_accuracy_overall"].notna().any()
        and "empirical_distribution_json" in df
        and df["empirical_distribution_json"].notna().any()
    )


# ---------------------------------------------------------------------------
# F6: responses provided per question, by method
# ---------------------------------------------------------------------------

def _json_distribution_items(raw) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, float) and np.isnan(raw):
        return []
    try:
        items = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _json_list(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, float) and np.isnan(raw):
        return []
    try:
        items = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return []
    return items if isinstance(items, list) else []


def _positive_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(out):
        return default
    return out


def _distinct_response_count(items: list[dict]) -> int:
    answers = set()
    for item in items:
        if _positive_float(item.get("probability")) <= 0:
            continue
        answer = str(item.get("answer", "")).strip()
        if not answer:
            continue
        grade = str(item.get("grade", ""))
        answers.add((norm_answer(answer), grade))
    return len(answers)


def _penalty_distribution_items(row: pd.Series) -> list[dict]:
    items = _json_distribution_items(row.get("penalty_distribution_json"))
    abstain_samples = row.get("penalty_abstain_samples")
    if abstain_samples is not None:
        n_samples = max(1, int(_positive_float(row.get("n_samples_requested"), 1.0)))
        abstain_prob = _positive_float(abstain_samples) / n_samples
        if abstain_prob > 0:
            items = list(items) + [{
                "answer": "ABSTAIN",
                "probability": abstain_prob,
                "grade": "not_attempted",
            }]
    return items


def fig_responses_per_question(paired):
    print("== F6: responses per question ==")
    fig, axes = model_panel_subplots(height=4.8, sharey=True)
    count_rows = []
    count_data = {}
    for ax, key in zip(axes, ACTIVE_MODEL_ORDER):
        df = paired[key]["residual_penalty"]
        log_counts = df["candidates"].map(_distinct_response_count).to_numpy()
        penalty_counts = {}
        for penalty, dpen in sorted(paired[key]["penalties"].items()):
            counts = dpen.apply(
                lambda row: _distinct_response_count(_penalty_distribution_items(row)),
                axis=1,
            ).to_numpy()
            penalty_counts[penalty] = counts
            for qid, pen_n in zip(dpen["question_id"], counts):
                count_rows.append({
                    "model": key,
                    "model_label": RUNS[key]["label"],
                    "question_id": qid,
                    "method": "penalty",
                    "penalty_level": penalty,
                    "distinct_response_entries": int(pen_n),
                })
        count_data[key] = {"log": log_counts, "penalty": penalty_counts}
        for qid, log_n in zip(df["question_id"], log_counts):
            count_rows.append({
                "model": key,
                "model_label": RUNS[key]["label"],
                "question_id": qid,
                "method": "log",
                "penalty_level": np.nan,
                "distinct_response_entries": int(log_n),
            })

    for ax, key in zip(axes, ACTIVE_MODEL_ORDER):
        c = MODEL_COLORS[key]
        data = count_data[key]
        log_counts = data["log"]
        observed_max = int(max(
            [log_counts.max(initial=0)]
            + [counts.max(initial=0) for counts in data["penalty"].values()]
        ))
        available_penalties = [p for p in PENALTY_LEVELS if p in data["penalty"]]
        series = [("penalty", p, data["penalty"][p]) for p in available_penalties]
        series.append(("log", None, log_counts))
        panel_cap = int(np.ceil(max(
            [np.quantile(counts, 0.99) for _method, _penalty, counts in series]
        )))
        has_overflow = observed_max > panel_cap
        plot_max = panel_cap + int(has_overflow)
        n_series = len(series)
        width = min(0.34, 1.22 / max(1, n_series))
        offsets = (np.arange(n_series) - (n_series - 1) / 2) * width

        def panel_hist(counts: np.ndarray) -> np.ndarray:
            hist = [(counts == k).mean() for k in range(panel_cap + 1)]
            if has_overflow:
                hist.append((counts > panel_cap).mean())
            return np.asarray(hist)

        log_hist = panel_hist(log_counts)
        x = np.arange(plot_max + 1)
        for idx, (method, penalty, counts) in enumerate(series):
            hist = log_hist if method == "log" else panel_hist(counts)
            color = c if method == "log" else PENALTY_BAR_COLORS[penalty]
            alpha = 0.9 if method == "log" else 0.82
            edgecolor = "#111827" if method == "log" else "white"
            linewidth = 0.42 if method == "log" else 0.28
            label = "log report" if method == "log" else f"penalty {penalty_label(penalty)}"
            ax.bar(
                x + offsets[idx],
                hist,
                width=width * 0.98,
                color=color,
                alpha=alpha,
                label=label,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=3,
            )
        ax.set_title(
            RUNS[key]["label"], fontsize=16, color="#111827"
        ).set_path_effects([withStroke(linewidth=0.5, foreground="#111827")])
        ax.set_xlabel("distinct responses per question", fontsize=17)
        if panel_cap <= 15:
            ticks = np.arange(panel_cap + 1)
        else:
            step = max(1, int(np.ceil(panel_cap / 10)))
            ticks = np.arange(0, panel_cap + 1, step)
            if ticks[-1] != panel_cap:
                ticks = np.r_[ticks, panel_cap]
        if has_overflow and ticks[-1] == panel_cap:
            ticks = ticks[:-1]
        tick_labels = [str(k) for k in ticks]
        if has_overflow:
            ticks = np.r_[ticks, plot_max]
            tick_labels.append(f">{panel_cap}")
            ax.axvline(panel_cap + 0.5, color="#9ca3af", lw=0.6,
                       linestyle=":", zorder=2)
        ax.set_xticks(ticks, tick_labels)
        ax.tick_params(axis="x", labelsize=11.5)
        ax.set_xlim(-0.78, plot_max + 0.78)
        ax.grid(axis="y", alpha=0.16, lw=0.55, zorder=0)
        penalty_summary = ", ".join(
            f"{penalty_label(p)}={np.median(data['penalty'][p]):.0f}"
            for p in available_penalties
        )
        print(
            f"  {key}: median distinct responses "
            f"log={np.median(log_counts):.0f}, penalty ({penalty_summary}); "
            f"max log={log_counts.max(initial=0)}, "
            f"max penalty={max((v.max(initial=0) for v in data['penalty'].values()), default=0)}, "
            f"panel cap={panel_cap}{' + overflow' if has_overflow else ''}"
        )
    axes[0].set_ylabel("fraction of questions", fontsize=17)
    legend_handles = [
        Patch(facecolor=PENALTY_BAR_COLORS[p], alpha=0.82, label=f"penalty {penalty_label(p)}")
        for p in PENALTY_LEVELS
        if any(p in count_data[key]["penalty"] for key in count_data)
    ]
    legend_handles.append(Patch(facecolor="#ffffff", edgecolor="#111827",
                                label="log report (model color)"))
    pd.DataFrame(count_rows).to_csv(OUT / "FS4_response_counts.csv", index=False)
    fig.tight_layout()
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=17,
    )
    savefig(fig, "f6_responses_per_question")


# ---------------------------------------------------------------------------
# F4: convergence of residual-log vs penalty estimates
# ---------------------------------------------------------------------------

CONVERGENCE_Z = 1.959963984540054
CONVERGENCE_METRICS = [
    "coverage-or-IDK @ p=0.9",
    "strict coverage @ p=0.9",
    "mean IDK mass",
]


def per_question_convergence_series(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-question diagnostics in stored deterministic question order."""
    sets = df["candidates"].map(lambda c: top_p_set(c, NOMINAL_P))
    oridk = sets.map(lambda s: float(s.covers or s.has_idk)).to_numpy(dtype=float)
    strict = sets.map(lambda s: float(s.covers)).to_numpy(dtype=float)
    q_idk = df["log_idk_mass"].fillna(0).to_numpy(dtype=float)
    return {
        "coverage-or-IDK @ p=0.9": oridk,
        "strict coverage @ p=0.9": strict,
        "mean IDK mass": q_idk,
    }


def penalty_convergence_series(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Same F4 diagnostics computed from the penalty sample distribution."""
    items = df.apply(_penalty_distribution_items, axis=1).tolist()
    sets = [top_p_set(item, NOMINAL_P) for item in items]
    oridk = np.array([float(s.covers or s.has_idk) for s in sets], dtype=float)
    strict = np.array([float(s.covers) for s in sets], dtype=float)
    denom = df["n_samples_requested"].astype(float).to_numpy()
    denom = np.where(denom > 0, denom, 1.0)
    q_idk = (
        df["penalty_abstain_samples"].fillna(0).astype(float).to_numpy()
        + df["penalty_not_attempted_samples"].fillna(0).astype(float).to_numpy()
    ) / denom
    return {
        "coverage-or-IDK @ p=0.9": oridk,
        "strict coverage @ p=0.9": strict,
        "mean IDK mass": q_idk,
    }


def running_mean_band(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Running mean with a normal pointwise 95% band."""
    x = np.asarray(x, dtype=float)
    n = np.arange(1, len(x) + 1)
    if len(x) == 0:
        return n, np.array([]), np.array([])
    mean = np.cumsum(x) / n
    sq = np.cumsum(x**2) / n
    sd = np.sqrt(np.maximum(0.0, sq - mean**2))
    half = CONVERGENCE_Z * sd / np.sqrt(n)
    return n, mean, half


def bootstrap_running_mean_band(
    x: np.ndarray,
    *,
    seed: int,
    n_boot: int = BOOT_BAND_REPS,
    min_n: int = BOOT_BAND_MIN_N,
    step: int = BOOT_BAND_STEP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise question-level bootstrap band for a running mean."""
    x = np.asarray(x, dtype=float)
    n_obs = len(x)
    if n_obs == 0:
        empty = np.array([])
        return empty, empty, empty
    start = min(min_n, n_obs)
    ns = np.unique(np.r_[np.arange(start, n_obs + 1, step), n_obs]).astype(int)
    rng = np.random.default_rng(seed)
    lo = []
    hi = []
    for k in ns:
        take = rng.integers(0, k, size=(n_boot, k))
        vals = x[:k][take].mean(axis=1)
        qlo, qhi = np.quantile(vals, [0.025, 0.975])
        lo.append(qlo)
        hi.append(qhi)
    return ns, np.asarray(lo), np.asarray(hi)


def fig_log_vs_penalty_convergence(paired) -> None:
    print("== F4: residual-log vs penalty convergence ==")
    model_keys = [key for key in ACTIVE_MODEL_ORDER if paired[key]["penalties"]]
    if not model_keys:
        print("  skipped F4: no penalty rows available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax_idx, (ax, metric) in enumerate(zip(axes, CONVERGENCE_METRICS)):
        for key_idx, key in enumerate(model_keys):
            color = MODEL_COLORS[key]
            series = per_question_convergence_series(paired[key]["residual_all"])[metric]
            n, mean, _ = running_mean_band(series)
            m = n >= min(BOOT_BAND_MIN_N, len(series))
            bx, blo, bhi = bootstrap_running_mean_band(
                series,
                seed=31000 + 1000 * ax_idx + 100 * key_idx,
            )
            ax.plot(
                n[m],
                mean[m],
                color=color,
                lw=1.8,
                ls="-",
                label=RUNS[key]["label"] if ax_idx == 0 else None,
            )
            ax.fill_between(bx, blo, bhi, color=color, alpha=0.10, linewidth=0)

            for penalty_idx, penalty in enumerate(sorted(paired[key]["penalties"])):
                pen_series = penalty_convergence_series(paired[key]["penalties"][penalty])[metric]
                pn, pmean, _ = running_mean_band(pen_series)
                pm = pn >= min(BOOT_BAND_MIN_N, len(pen_series))
                pbx, pblo, pbhi = bootstrap_running_mean_band(
                    pen_series,
                    seed=32000 + 1000 * ax_idx + 100 * key_idx + penalty_idx,
                )
                ax.plot(
                    pn[pm],
                    pmean[pm],
                    color=color,
                    lw=1.8,
                    ls=PENALTY_LINESTYLES.get(penalty, "--"),
                    alpha=PENALTY_ALPHA.get(penalty, 0.82),
                )
                ax.fill_between(pbx, pblo, pbhi, color=color, alpha=0.05, linewidth=0)
        ax.set_title(metric, fontsize=10)
        ax.set_xlabel("question count n (deterministic order)")
        ax.set_ylim(-0.03, 1.04)
    axes[0].set_ylabel("running estimate")

    model_handles = [
        Line2D([0], [0], color=MODEL_COLORS[key], lw=2, label=RUNS[key]["label"])
        for key in model_keys
    ]
    method_handles = [Line2D([0], [0], color="black", lw=1.8, ls="-", label="residual log")]
    available_penalties = sorted({p for key in model_keys for p in paired[key]["penalties"]})
    for penalty in available_penalties:
        method_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                lw=1.8,
                ls=PENALTY_LINESTYLES.get(penalty, "--"),
                label=f"penalty {penalty_label(penalty)}",
            )
        )
    axes[0].legend(handles=model_handles, fontsize=8, loc="lower right")
    axes[2].legend(handles=method_handles, fontsize=8, loc="best")

    fig.tight_layout()
    savefig(fig, "f4_log_vs_penalty_convergence")


# ---------------------------------------------------------------------------
# F3: cumulative rates as the paired question prefix grows
# ---------------------------------------------------------------------------

def bootstrap_cumulative_rate_bands(
    halluc: np.ndarray,
    acc: np.ndarray,
    abst: np.ndarray,
    *,
    seed: int,
    n_boot: int = BOOT_BAND_REPS,
    min_n: int = BOOT_BAND_MIN_N,
    step: int = BOOT_BAND_STEP,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Pointwise question-level bootstrap bands for cumulative rate curves."""
    halluc = np.asarray(halluc, dtype=float)
    acc = np.asarray(acc, dtype=float)
    abst = np.asarray(abst, dtype=float)
    n = len(halluc)
    if n == 0:
        empty = (np.array([]), np.array([]), np.array([]))
        return {
            "hallucination_rate": empty,
            "accuracy_overall": empty,
            "abstention_rate": empty,
            "accuracy_when_answered": empty,
        }
    start = min(min_n, n)
    ns = np.unique(np.r_[np.arange(start, n + 1, step), n]).astype(int)
    rng = np.random.default_rng(seed)
    out = {name: [[], []] for name in (
        "hallucination_rate",
        "accuracy_overall",
        "abstention_rate",
        "accuracy_when_answered",
    )}
    for k in ns:
        take = rng.integers(0, k, size=(n_boot, k))
        h_sum = halluc[:k][take].sum(axis=1)
        a_sum = acc[:k][take].sum(axis=1)
        z_mean = abst[:k][take].mean(axis=1)
        ans = h_sum + a_sum
        awa = np.divide(
            a_sum,
            ans,
            out=np.full_like(a_sum, np.nan, dtype=float),
            where=ans > 0,
        )
        boot = {
            "hallucination_rate": h_sum / k,
            "accuracy_overall": a_sum / k,
            "abstention_rate": z_mean,
            "accuracy_when_answered": awa,
        }
        for name, vals in boot.items():
            lo, hi = np.nanquantile(vals, [0.025, 0.975])
            out[name][0].append(lo)
            out[name][1].append(hi)
    return {
        name: (ns, np.asarray(bounds[0]), np.asarray(bounds[1]))
        for name, bounds in out.items()
    }

def cumulative_rates_for_key(paired, key: str, threshold: float = 0.75) -> pd.DataFrame:
    """Cumulative paired rates in deterministic SimpleQA order.

    Residual-log is a one-decision-per-question curve at threshold t.
    Penalty arms contribute their empirical 50-sample accuracy /
    hallucination / abstention rates.
    """
    dlog = paired[key]["residual_penalty"]
    log_outcomes = dlog["candidates"].map(lambda c: posthoc_decision(c, threshold))
    rows = []
    specs = [
        (
            "residual_log_t075",
            (log_outcomes == "incorrect").astype(float).to_numpy(),
            (log_outcomes == "correct").astype(float).to_numpy(),
            (log_outcomes == "abstain").astype(float).to_numpy(),
        ),
    ]
    for penalty, dpen in paired[key]["penalties"].items():
        specs.append(
            (
                f"penalty_native_{penalty_tag(penalty)}",
                dpen["penalty_hallucination_rate"].astype(float).to_numpy(),
                dpen["penalty_accuracy_overall"].astype(float).to_numpy(),
                dpen["penalty_abstention_rate"].astype(float).to_numpy(),
            )
        )
    for method, halluc, acc, abst in specs:
        denom = np.arange(1, len(halluc) + 1)
        cum_halluc = np.cumsum(halluc)
        cum_acc = np.cumsum(acc)
        cum_abst = np.cumsum(abst)
        cum_answered = cum_halluc + cum_acc
        cum_acc_answered = np.divide(
            cum_acc,
            cum_answered,
            out=np.full_like(cum_acc, np.nan, dtype=float),
            where=cum_answered > 0,
        )
        for i, (h, a, z, awa) in enumerate(
            zip(cum_halluc / denom,
                cum_acc / denom,
                cum_abst / denom,
                cum_acc_answered),
            start=1,
        ):
            rows.append(
                {
                    "model": key,
                    "method": method,
                    "threshold": threshold if method.startswith("residual") else np.nan,
                    "n_questions": i,
                    "hallucination_rate": h,
                    "accuracy_overall": a,
                    "abstention_rate": z,
                    "accuracy_when_answered": awa,
                }
            )
    return pd.DataFrame(rows)


def fig_cumulative_rates(paired) -> pd.DataFrame:
    print("== F3: cumulative rates by number of questions ==")
    rows = pd.concat(
        [cumulative_rates_for_key(paired, key) for key in ACTIVE_MODEL_ORDER],
        ignore_index=True,
    )
    rows.to_csv(OUT / "f3_cumulative_rates_by_question.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), sharex=False)
    panels = [
        ("hallucination_rate", "hallucination rate"),
        ("abstention_rate", "abstention rate"),
        ("accuracy_overall", "accuracy"),
        ("accuracy_when_answered", "accuracy when answered"),
    ]
    for ax, (col, ylab) in zip(axes.flat, panels):
        band_seed = 14000
        for model_idx, key in enumerate(ACTIVE_MODEL_ORDER):
            c = MODEL_COLORS[key]
            sub = rows[rows["model"] == key]
            dlog = paired[key]["residual_penalty"]
            log_outcomes = dlog["candidates"].map(lambda cands: posthoc_decision(cands, 0.75))
            method_specs = [
                (
                    "residual_log_t075",
                    "-",
                    0.95,
                    (log_outcomes == "incorrect").astype(float).to_numpy(),
                    (log_outcomes == "correct").astype(float).to_numpy(),
                    (log_outcomes == "abstain").astype(float).to_numpy(),
                )
            ]
            method_specs.extend(
                (
                    f"penalty_native_{penalty_tag(p)}",
                    PENALTY_LINESTYLES.get(p, "--"),
                    PENALTY_ALPHA.get(p, 0.72),
                    paired[key]["penalties"][p]["penalty_hallucination_rate"].astype(float).to_numpy(),
                    paired[key]["penalties"][p]["penalty_accuracy_overall"].astype(float).to_numpy(),
                    paired[key]["penalties"][p]["penalty_abstention_rate"].astype(float).to_numpy(),
                )
                for p in sorted(paired[key]["penalties"])
            )
            for method_idx, (method, ls, alpha, halluc, acc, abst) in enumerate(method_specs):
                d = sub[sub["method"] == method]
                bands = bootstrap_cumulative_rate_bands(
                    halluc,
                    acc,
                    abst,
                    seed=band_seed + 100 * model_idx + method_idx,
                )
                bx, blo, bhi = bands[col]
                ax.fill_between(
                    bx,
                    blo,
                    bhi,
                    color=c,
                    alpha=0.10 if method == "residual_log_t075" else 0.055,
                    linewidth=0,
                )
                ax.plot(
                    d["n_questions"], d[col],
                    color=c, lw=1.8, ls=ls, alpha=alpha,
                )
        ax.set_xlabel("number of questions")
        ax.set_ylabel(ylab)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.18, lw=0.6)

    model_handles = [
        Line2D([0], [0], color=MODEL_COLORS[key], lw=2, label=RUNS[key]["label"])
        for key in ACTIVE_MODEL_ORDER
    ]
    method_handles = [
        Line2D([0], [0], color="black", lw=2, ls="-",
               label="residual-ρ log, t=0.75"),
    ]
    for penalty in sorted({p for k in ACTIVE_MODEL_ORDER for p in paired[k]["penalties"]}):
        method_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                lw=2,
                ls=PENALTY_LINESTYLES.get(penalty, "--"),
                label=f"penalty prompt {penalty_label(penalty)}",
            )
        )
    axes[0, 0].legend(handles=model_handles, fontsize=7, loc="best")
    axes[1, 1].legend(handles=method_handles, fontsize=7, loc="best")
    savefig(fig, "f3_cumulative_rates")
    return rows


def _log_top_p_three_way_outcome(candidates: list[dict], p: float = NOMINAL_P) -> str:
    top = top_p_set(candidates, p)
    if top.covers:
        return "accuracy"
    if not top.members or (top.has_idk and top.n_concrete == 0):
        return "abstention"
    return "hallucination"


def _log_top_p_three_way_arrays(dlog: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outcomes = dlog["candidates"].map(_log_top_p_three_way_outcome)
    halluc = (outcomes == "hallucination").astype(float).to_numpy()
    acc = (outcomes == "accuracy").astype(float).to_numpy()
    abst = (outcomes == "abstention").astype(float).to_numpy()
    return halluc, acc, abst


def _penalty_three_way_arrays(dpen: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    denom = dpen["n_samples_requested"].astype(float).to_numpy()
    denom = np.where(denom > 0, denom, 1.0)
    acc = dpen["penalty_correct_samples"].astype(float).to_numpy() / denom
    halluc = dpen["penalty_incorrect_samples"].astype(float).to_numpy() / denom
    abst = (
        dpen["penalty_abstain_samples"].astype(float).to_numpy()
        + dpen["penalty_not_attempted_samples"].astype(float).to_numpy()
    ) / denom
    return halluc, acc, abst


def cumulative_three_way_rates_for_key(paired, key: str) -> pd.DataFrame:
    """Cumulative three-way outcome rates.

    Log reports are evaluated as top-p sets: truth in the prefix is accuracy,
    an IDK-only prefix is abstention, and every other truth-missing prefix is
    hallucination.  Penalty prompts use their empirical sample partition into
    correct, incorrect, and abstain/not-attempted samples.
    """
    specs = [
        (
            "residual_log_top_p",
            *_log_top_p_three_way_arrays(paired[key]["residual_penalty"]),
        ),
    ]
    for penalty, dpen in paired[key]["penalties"].items():
        specs.append(
            (
                f"penalty_native_{penalty_tag(penalty)}",
                *_penalty_three_way_arrays(dpen),
            )
        )

    rows = []
    for method, halluc, acc, abst in specs:
        denom = np.arange(1, len(halluc) + 1)
        cum_halluc = np.cumsum(halluc) / denom
        cum_acc = np.cumsum(acc) / denom
        cum_abst = np.cumsum(abst) / denom
        for i, (h, a, z) in enumerate(zip(cum_halluc, cum_acc, cum_abst), start=1):
            rows.append(
                {
                    "model": key,
                    "method": method,
                    "n_questions": i,
                    "hallucination_rate": h,
                    "accuracy_rate": a,
                    "abstention_rate": z,
                    "partition_sum": h + a + z,
                }
            )
    return pd.DataFrame(rows)


def fig_cumulative_three_way_rates(paired) -> pd.DataFrame:
    print("== F3A: cumulative three-way top-p outcome rates ==")
    rows = pd.concat(
        [cumulative_three_way_rates_for_key(paired, key) for key in ACTIVE_MODEL_ORDER],
        ignore_index=True,
    )
    rows.to_csv(OUT / "FS3_cumulative.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12.9, 4.65), sharex=False)
    panels = [
        ("accuracy_rate", "accuracy"),
        ("hallucination_rate", "hallucination"),
        ("abstention_rate", "abstention"),
    ]
    band_seed = 15000
    method_specs_by_model = {}
    for model_idx, key in enumerate(ACTIVE_MODEL_ORDER):
        method_specs = [
            (
                "residual_log_top_p",
                "-",
                0.95,
                2.45,
                *_log_top_p_three_way_arrays(paired[key]["residual_penalty"]),
            )
        ]
        method_specs.extend(
            (
                f"penalty_native_{penalty_tag(p)}",
                PENALTY_LINESTYLES.get(p, "--"),
                PENALTY_ALPHA.get(p, 0.72),
                2.05,
                *_penalty_three_way_arrays(paired[key]["penalties"][p]),
            )
            for p in sorted(paired[key]["penalties"])
        )
        enriched_specs = []
        for method_idx, (method, ls, alpha, lw, halluc, acc, abst) in enumerate(method_specs):
            bands = bootstrap_cumulative_rate_bands(
                halluc,
                acc,
                abst,
                seed=band_seed + 100 * model_idx + method_idx,
            )
            enriched_specs.append((method, ls, alpha, lw, bands))
        method_specs_by_model[key] = enriched_specs

    for ax, (col, ylab) in zip(axes, panels):
        for model_idx, key in enumerate(ACTIVE_MODEL_ORDER):
            c = MODEL_COLORS[key]
            sub = rows[rows["model"] == key]
            for method, ls, alpha, lw, bands in method_specs_by_model[key]:
                d = sub[sub["method"] == method]
                band_key = "accuracy_overall" if col == "accuracy_rate" else col
                bx, blo, bhi = bands[band_key]
                ax.fill_between(
                    bx,
                    blo,
                    bhi,
                    color=c,
                    alpha=0.12 if method == "residual_log_top_p" else 0.06,
                    linewidth=0,
                )
                ax.plot(
                    d["n_questions"],
                    d[col],
                    color=c,
                    lw=lw,
                    ls=ls,
                    alpha=alpha,
                )
        ax.set_xlabel("questions", fontsize=14)
        ax.set_ylabel(ylab, fontsize=14)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.18, lw=0.6)
        ax.tick_params(axis="both", labelsize=10.5)

    model_handles = [
        Line2D([0], [0], color=MODEL_COLORS[key], lw=2, label=RUNS[key]["label"])
        for key in ACTIVE_MODEL_ORDER
    ]
    method_handles = [
        Line2D([0], [0], color="black", lw=2.4, ls="-",
               label="log top-p report"),
    ]
    for penalty in sorted({p for k in ACTIVE_MODEL_ORDER for p in paired[k]["penalties"]}):
        method_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                lw=2.4,
                ls=PENALTY_LINESTYLES.get(penalty, "--"),
                label=f"penalty {penalty_label(penalty)}",
            )
        )
    # One shared legend below the panels: models on the top row, line
    # styles (methods) on the bottom row (legend fills column-major).
    if len(model_handles) == len(method_handles):
        combined_handles = [
            handle
            for pair in zip(model_handles, method_handles)
            for handle in pair
        ]
    else:
        combined_handles = model_handles + method_handles
    fig.tight_layout(w_pad=1.5)
    fig.legend(
        handles=combined_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(model_handles),
        frameon=False,
        fontsize=12,
        handlelength=3.6,
        columnspacing=1.4,
        labelspacing=1.0,
    )
    savefig(fig, "f3a_cumulative_three_way_rates")
    return rows


# ---------------------------------------------------------------------------
# F2: penalty-vs-log 3x3 outcome table
# ---------------------------------------------------------------------------

OUTCOME_KEYS = ["abstain", "correct", "incorrect"]
PENALTY_OUTCOME_LABELS = ["abstain", "correct", "incorrect"]
LOG_OUTCOME_LABELS = ["abstain", "correct", "incorrect"]


def _parse_final_text_response(text: str) -> str | None:
    nonempty_lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    final_line_pattern = re.compile(
        r"^\s*(?:#+\s*)?(?:final(?:\s+answer)?|answer)\s*:\s*(.+?)\s*$",
        re.IGNORECASE,
    )
    for line in reversed(nonempty_lines):
        match = final_line_pattern.match(line.strip().strip("*_`"))
        if match:
            value = match.group(1).strip().strip("*_`").strip()
            return value or None
    stripped = str(text).strip()
    return stripped or None


def _is_abstain_answer(answer: str | None) -> bool:
    key = norm_answer(answer or "")
    return (
        not key
        or key in {
            "abstain",
            "i don't know",
            "i do not know",
            "idk",
            "not attempted",
            "unknown",
            "unsure",
        }
        or key.startswith("abstain")
    )


def _log_top_p_outcome(candidates: list[dict], p: float = NOMINAL_P) -> str:
    top = top_p_set(candidates, p)
    if top.covers:
        return "correct"
    if top.has_idk:
        return "abstain"
    return "incorrect"


def _first_penalty_sample_outcome(row: pd.Series) -> str | None:
    preview = _json_list(row.get("penalty_sample_response_preview_json"))
    if not preview:
        return None
    answer = _parse_final_text_response(str(preview[0]))
    if _is_abstain_answer(answer):
        return "abstain"

    answer_key = norm_answer(answer or "")
    for item in _json_distribution_items(row.get("penalty_distribution_json")):
        keys = [norm_answer(item.get("answer", ""))]
        keys.extend(norm_answer(x) for x in item.get("source_answers", []) if x)
        if answer_key in keys:
            grade = str(item.get("grade", "incorrect"))
            if grade == "correct":
                return "correct"
            if grade == "not_attempted":
                return "abstain"
            return "incorrect"

    if answer_key and answer_key == norm_answer(row.get("gold_answer", "")):
        return "correct"
    return "incorrect"


def _penalty_sample_average_weights(row: pd.Series) -> dict[str, float]:
    n = max(1.0, _positive_float(row.get("n_samples_requested"), 1.0))
    abstain = _positive_float(row.get("penalty_abstain_samples"))
    # A small number of non-empty generations can still be graded as
    # not-attempted; for this 3-way table they belong in the abstain row.
    abstain += _positive_float(row.get("penalty_not_attempted_samples"))
    return {
        "abstain": abstain / n,
        "correct": _positive_float(row.get("penalty_correct_samples")) / n,
        "incorrect": _positive_float(row.get("penalty_incorrect_samples")) / n,
    }


def _penalty_log_outcome_tables_for_key(
    paired,
    key: str,
    *,
    mode: str,
    penalty: float,
) -> tuple[pd.DataFrame, list[dict], int]:
    dpen = paired[key]["penalties"][penalty]
    dlog = paired[key]["residual_by_penalty"][penalty]
    table = pd.DataFrame(0.0, index=OUTCOME_KEYS, columns=OUTCOME_KEYS)
    rows = []
    used = 0

    for qid in dlog["question_id"]:
        if qid not in dpen.index:
            continue
        log_outcome = _log_top_p_outcome(dlog.loc[qid, "candidates"])
        if mode == "first_sample":
            penalty_outcome = _first_penalty_sample_outcome(dpen.loc[qid])
            if penalty_outcome is None:
                continue
            table.loc[penalty_outcome, log_outcome] += 1.0
            rows.append(
                {
                    "model": key,
                    "model_label": RUNS[key]["label"],
                    "question_id": qid,
                    "mode": mode,
                    "penalty_level": penalty,
                    "penalty_outcome": penalty_outcome,
                    "log_top_p_outcome": log_outcome,
                    "weight": 1.0,
                }
            )
            used += 1
        elif mode == "sample_average":
            weights = _penalty_sample_average_weights(dpen.loc[qid])
            for penalty_outcome, weight in weights.items():
                table.loc[penalty_outcome, log_outcome] += weight
                rows.append(
                    {
                        "model": key,
                        "model_label": RUNS[key]["label"],
                        "question_id": qid,
                        "mode": mode,
                        "penalty_level": penalty,
                        "penalty_outcome": penalty_outcome,
                        "log_top_p_outcome": log_outcome,
                        "weight": weight,
                    }
                )
            used += 1
        else:
            raise ValueError(f"unknown mode: {mode}")

    return table, rows, used


def _blend_with_white(color: str, weight: float) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple((1 - weight) * np.ones(3) + weight * rgb)


def _model_question_cmap(key: str, color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f"{key}_question_counts",
        ["#ffffff", _blend_with_white(color, 0.36), color],
    )


def _plot_outcome_tables(
    matrices: dict[str, pd.DataFrame],
    used_counts: dict[str, int],
    *,
    mode: str,
    penalty: float,
    name: str,
    model_keys: list[str] | None = None,
) -> None:
    if model_keys is None:
        model_keys = ACTIVE_MODEL_ORDER
    n_models = len(model_keys)
    fig_width = max(5.2, 4.75 * n_models)
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(fig_width, 5.35),
        sharex=False,
        sharey=False,
        gridspec_kw={"wspace": 0.34},
    )
    axes = np.atleast_1d(axes).ravel()
    vmax = max((float(mat.to_numpy().max()) for mat in matrices.values()), default=1.0)
    vmax = max(vmax, 1.0)
    norm = Normalize(vmin=0, vmax=vmax)
    for ax, key in zip(axes, model_keys):
        mat = matrices[key]
        arr = mat.to_numpy(dtype=float)
        base_color = MODEL_COLORS.get(key, "#1a73e8")
        cmap = _model_question_cmap(key, base_color)
        ax.imshow(arr, cmap=cmap, norm=norm)
        row_totals = arr.sum(axis=1)
        for i in range(arr.shape[0]):
            row_total = float(row_totals[i])
            for j in range(arr.shape[1]):
                value = arr[i, j]
                pct = 100 * value / row_total if row_total > 0 else 0.0
                value_text = f"{value:.0f}" if mode == "first_sample" else f"{value:.1f}"
                # Pick text color from the tile's actual luminance so bright
                # ramps (e.g. orange) keep black text even at high counts.
                rgba = cmap(norm(value))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.6 else "black"
                ax.annotate(
                    value_text,
                    xy=(j, i),
                    xycoords="data",
                    xytext=(0, 8.8),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=12.8,
                    fontweight=800,
                    color=text_color,
                )
                ax.annotate(
                    f"({pct:.1f}%)",
                    xy=(j, i),
                    xycoords="data",
                    xytext=(0, -9.6),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=11.6,
                    color=text_color,
                ).set_path_effects(
                    [withStroke(linewidth=0.01, foreground=text_color)]
                )
        # The default font has no true semibold weight, so a hairline stroke
        # around the regular glyphs stands in for one.
        title_effect = [withStroke(linewidth=0.5, foreground="#111827")]
        ax.set_xticks(range(len(OUTCOME_KEYS)), LOG_OUTCOME_LABELS, fontsize=12)
        for label in ax.get_xticklabels():
            label.set_color("#111827")
        ax.set_yticks(range(len(OUTCOME_KEYS)))
        ax.set_yticklabels([])
        for i, (base_label, row_total) in enumerate(zip(PENALTY_OUTCOME_LABELS, row_totals)):
            total_text = f"{row_total:.0f}" if mode == "first_sample" else f"{row_total:.1f}"
            row_label = base_label.replace("\n", " ")
            ax.annotate(
                row_label,
                xy=(-0.028, i),
                xycoords=ax.get_yaxis_transform(),
                xytext=(0, 6),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=12,
                color="#111827",
                clip_on=False,
            )
            ax.annotate(
                total_text,
                xy=(-0.028, i),
                xycoords=ax.get_yaxis_transform(),
                xytext=(0, -9),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=11.2,
                color="#4b5563",
                clip_on=False,
            ).set_path_effects(
                [withStroke(linewidth=0.2, foreground="#4b5563")]
            )
        ax.set_xlabel("log top-p report", labelpad=14, fontsize=14)
        ax.set_title(
            RUNS[key]["label"],
            fontsize=16,
            color="#111827",
            pad=11,
        ).set_path_effects(title_effect)
        ax.tick_params(axis="x", rotation=0)
        # Full border on all four sides (the global rcParams hide top/right).
        for spine in ax.spines.values():
            spine.set_visible(True)
    axes[0].set_ylabel(
        f"penalty {penalty_label(penalty)}", labelpad=61, fontsize=14
    )
    subtle_count_cmap = LinearSegmentedColormap.from_list(
        "subtle_question_counts",
        ["#ffffff", "#d1d5db", "#6b7280"],
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=subtle_count_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.008, pad=0.012, shrink=0.86)
    # No label: the caption explains that shading encodes question counts.
    cbar.ax.tick_params(labelsize=8, length=2.5, width=0.5, colors="#111827")
    cbar.outline.set_visible(False)
    savefig(fig, name)


def _fig_penalty_log_outcome_table_for_penalty(
    paired,
    penalty: float,
    *,
    name: str,
) -> dict[str, pd.DataFrame]:
    outputs = {}
    mode = "sample_average"
    model_keys = [key for key in ACTIVE_MODEL_ORDER if penalty in paired[key]["penalties"]]
    if not model_keys:
        print(f"  skipped F2: no {penalty_label(penalty)} penalty rows")
        return outputs

    matrices = {}
    used_counts = {}
    all_rows = []
    for key in model_keys:
        mat, rows, used = _penalty_log_outcome_tables_for_key(
            paired, key, mode=mode, penalty=penalty
        )
        matrices[key] = mat
        used_counts[key] = used
        all_rows.extend(rows)
        print(f"  {key} ({mode}, {penalty_label(penalty)}): n={used}")
    detail = pd.DataFrame(all_rows)
    csv_name = FIGURE_RENAMES.get(name, name)
    detail.to_csv(OUT / f"{csv_name}_long.csv", index=False)

    summary_rows = []
    for key, mat in matrices.items():
        total = max(1.0, float(mat.to_numpy().sum()))
        for penalty_outcome in OUTCOME_KEYS:
            for log_outcome in OUTCOME_KEYS:
                value = float(mat.loc[penalty_outcome, log_outcome])
                summary_rows.append(
                    {
                        "model": key,
                        "model_label": RUNS[key]["label"],
                        "mode": mode,
                        "penalty_level": penalty,
                        "penalty_outcome": penalty_outcome,
                        "log_top_p_outcome": log_outcome,
                        "count_or_expected_count": value,
                        "percent": 100 * value / total,
                    }
                )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / f"{csv_name}.csv", index=False)
    outputs[f"{mode}_{penalty_tag(penalty)}"] = summary
    _plot_outcome_tables(
        matrices,
        used_counts,
        mode=mode,
        penalty=penalty,
        name=name,
        model_keys=model_keys,
    )
    return outputs


def fig_penalty_log_outcome_tables(paired) -> dict[str, pd.DataFrame]:
    print("== F2: penalty-vs-log sample-average outcome tables ==")
    outputs = {}
    base_name = "f2_penalty_log_outcome_table_sample_average"
    for penalty in PENALTY_LEVELS:
        name = base_name if np.isclose(penalty, PRIMARY_PENALTY) else f"{base_name}_{penalty_tag(penalty)}"
        outputs.update(
            _fig_penalty_log_outcome_table_for_penalty(
                paired,
                penalty,
                name=name,
            )
        )
    return outputs


# ---------------------------------------------------------------------------
# F7: repeated log-elicitation consistency
# ---------------------------------------------------------------------------

def f7_member_key(candidate: dict) -> str:
    if candidate.get("grade") == "not_attempted":
        return "__IDK__"
    text = str(candidate.get("answer", ""))
    return norm_answer(text) or text.strip().lower()


def f7_pairwise_jaccard(sets: list[set[str]]) -> float:
    if len(sets) <= 1:
        return np.nan
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            vals.append(1.0 if not union else len(sets[i] & sets[j]) / len(union))
    return float(np.mean(vals)) if vals else np.nan


def f7_report_distribution(candidates: list[dict]) -> dict[str, float]:
    dist: dict[str, float] = {}
    for candidate in candidates:
        try:
            prob = float(candidate.get("probability", 0.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(prob) or prob <= 0:
            continue
        key = f7_member_key(candidate)
        dist[key] = dist.get(key, 0.0) + prob
    total = sum(dist.values())
    if total > 0:
        dist = {key: value / total for key, value in dist.items()}
    return dist


def f7_distribution_matrix(
    distributions: list[dict[str, float]],
) -> tuple[np.ndarray, list[str]]:
    if not distributions:
        return np.zeros((0, 0), dtype=float), []
    keys = sorted(set().union(*(set(dist) for dist in distributions)))
    if not keys:
        return np.zeros((len(distributions), 0), dtype=float), []
    arr = np.array(
        [[float(dist.get(key, 0.0)) for key in keys] for dist in distributions],
        dtype=float,
    )
    arr[~np.isfinite(arr)] = 0.0
    arr[arr < 0] = 0.0
    row_sums = arr.sum(axis=1, keepdims=True)
    np.divide(arr, row_sums, out=arr, where=row_sums > 0)
    return arr, keys


def f7_entropy_array(arr: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(arr > 0, -arr * np.log2(arr), 0.0)
    return terms.sum(axis=-1)


def f7_pairwise_jsd_matrix_from_array(arr: np.ndarray) -> np.ndarray:
    n = arr.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if arr.shape[1] == 0:
        return np.zeros((n, n), dtype=float)
    h = f7_entropy_array(arr)
    mid = 0.5 * (arr[:, None, :] + arr[None, :, :])
    jsd = f7_entropy_array(mid) - 0.5 * (h[:, None] + h[None, :])
    return np.maximum(0.0, jsd)


def f7_upper_triangle_values(matrix: np.ndarray) -> list[float]:
    if matrix.shape[0] <= 1:
        return []
    tri = np.triu_indices(matrix.shape[0], k=1)
    return matrix[tri].astype(float).tolist()


def f7_pairwise_tv_values(distributions: list[dict[str, float]]) -> list[float]:
    if len(distributions) <= 1:
        return []
    arr, _keys = f7_distribution_matrix(distributions)
    if arr.shape[1] == 0:
        return []
    tv = 0.5 * np.abs(arr[:, None, :] - arr[None, :, :]).sum(axis=2)
    return f7_upper_triangle_values(tv)


def f7_jsd(a: dict[str, float], b: dict[str, float]) -> float:
    arr, _keys = f7_distribution_matrix([a, b])
    if arr.shape[1] == 0:
        return 0.0
    return float(f7_pairwise_jsd_matrix_from_array(arr)[0, 1])


def f7_pairwise_jsd_values(distributions: list[dict[str, float]]) -> list[float]:
    if len(distributions) <= 1:
        return []
    arr, _keys = f7_distribution_matrix(distributions)
    if arr.shape[1] == 0:
        return []
    return f7_upper_triangle_values(f7_pairwise_jsd_matrix_from_array(arr))


def f7_consensus_distribution(distributions: list[dict[str, float]]) -> dict[str, float]:
    if not distributions:
        return {}
    keys = set().union(*(set(dist) for dist in distributions))
    consensus = {
        key: float(np.mean([dist.get(key, 0.0) for dist in distributions]))
        for key in keys
    }
    total = sum(consensus.values())
    if total > 0:
        consensus = {key: value / total for key, value in consensus.items()}
    return consensus


def f7_consensus_jsd_values(distributions: list[dict[str, float]]) -> list[float]:
    if not distributions:
        return []
    arr, _keys = f7_distribution_matrix(distributions)
    if arr.shape[1] == 0:
        return []
    consensus = arr.mean(axis=0)
    total = consensus.sum()
    if total <= 0:
        return []
    consensus = consensus / total
    h = f7_entropy_array(arr)
    h_consensus = float(f7_entropy_array(consensus))
    mid = 0.5 * (arr + consensus[None, :])
    vals = f7_entropy_array(mid) - 0.5 * (h + h_consensus)
    return np.maximum(0.0, vals).astype(float).tolist()


def parse_candidates_for_f7(rec: dict) -> list[dict]:
    raw = rec.get("log_candidates_json")
    if not raw:
        return []
    candidates = json.loads(raw) if isinstance(raw, str) else raw
    out = []
    for candidate in candidates:
        prob = candidate.get("probability")
        if prob is None:
            continue
        out.append(
            {
                "answer": str(candidate.get("answer", "")),
                "probability": float(prob),
                "points": float(candidate.get("points", 0.0)),
                "grade": str(candidate.get("grade", "")),
            }
        )
    return out


def load_log_consistency(run_dir_name: str) -> pd.DataFrame:
    run_dir = OUTPUTS_DIR / run_dir_name
    result_paths = [
        path
        for path in [result_file(run_dir / "simpleqa_log_consistency_results.jsonl")]
        if path is not None
    ]
    if not result_paths:
        raise FileNotFoundError(run_dir / "simpleqa_log_consistency_results.jsonl")

    records_by_key: dict[tuple[str, str, int], dict] = {}
    for path in result_paths:
        with open_result(path) as f:
            for line_i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  F7: ignored malformed in-progress line {line_i} from {path}")
                    continue
                key = (
                    str(rec.get("model")),
                    str(rec.get("question_id")),
                    int(rec.get("repeat_index", 0)),
                )
                records_by_key[key] = rec

    rows = []
    for rec in records_by_key.values():
        candidates = parse_candidates_for_f7(rec)
        top = top_p_set(candidates, NOMINAL_P)
        top_members = sorted({f7_member_key(c) for c in top.members})
        model_key = model_key_for_id(str(rec["model"])) or str(rec["model"])
        rows.append(
            {
                "question_id": rec["question_id"],
                "model": rec["model"],
                "model_key": model_key,
                "repeat_index": int(rec.get("repeat_index", 0)),
                "log_temperature": float(rec.get("log_temperature", np.nan)),
                "candidates": candidates,
                "log_q_true": float(rec.get("log_q_true", np.nan)),
                "log_hallucination_mass": float(rec.get("log_hallucination_mass", np.nan)),
                "log_idk_mass": float(rec.get("log_not_attempted_mass", np.nan)),
                "log_top_answer": rec.get("log_top_answer"),
                "log_top_grade": rec.get("log_top_grade"),
                "log_top_p_set_size": len(top.members),
                "log_top_p_members": top_members,
                "parse_failed": str(rec.get("log_parse_status", "")).startswith("malformed"),
                "log_total_tokens": float(rec.get("log_total_tokens", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def summarize_log_consistency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_key, question_id), group in df.groupby(["model_key", "question_id"], sort=False):
        member_sets = [set(members) for members in group["log_top_p_members"]]
        distributions = [f7_report_distribution(candidates) for candidates in group["candidates"]]
        tv_values = f7_pairwise_tv_values(distributions)
        jsd_values = f7_pairwise_jsd_values(distributions)
        consensus_jsd_values = f7_consensus_jsd_values(distributions)
        top_keys = [
            f7_member_key({"answer": answer, "grade": grade})
            for answer, grade in zip(group["log_top_answer"], group["log_top_grade"])
        ]
        top_counts = pd.Series(top_keys).value_counts()
        rows.append(
            {
                "model_key": model_key,
                "model_label": RUNS.get(model_key, {}).get("label", model_key),
                "question_id": question_id,
                "n_repeats": len(group),
                "mean_log_q_true": group["log_q_true"].mean(),
                "sd_log_q_true": group["log_q_true"].std(ddof=1),
                "mean_log_hallucination_mass": group["log_hallucination_mass"].mean(),
                "sd_log_hallucination_mass": group["log_hallucination_mass"].std(ddof=1),
                "mean_log_idk_mass": group["log_idk_mass"].mean(),
                "sd_log_idk_mass": group["log_idk_mass"].std(ddof=1),
                "mean_top_p_set_size": group["log_top_p_set_size"].mean(),
                "sd_top_p_set_size": group["log_top_p_set_size"].std(ddof=1),
                "mean_top_p_jaccard": f7_pairwise_jaccard(member_sets),
                "mean_top_p_set_distance": 1.0 - f7_pairwise_jaccard(member_sets),
                "mean_pairwise_tv": float(np.mean(tv_values)) if tv_values else np.nan,
                "median_pairwise_tv": float(np.median(tv_values)) if tv_values else np.nan,
                "q90_pairwise_tv": float(np.quantile(tv_values, 0.90)) if tv_values else np.nan,
                "mean_pairwise_jsd": float(np.mean(jsd_values)) if jsd_values else np.nan,
                "median_pairwise_jsd": float(np.median(jsd_values)) if jsd_values else np.nan,
                "q90_pairwise_jsd": float(np.quantile(jsd_values, 0.90)) if jsd_values else np.nan,
                "mean_consensus_jsd": (
                    float(np.mean(consensus_jsd_values)) if consensus_jsd_values else np.nan
                ),
                "median_consensus_jsd": (
                    float(np.median(consensus_jsd_values)) if consensus_jsd_values else np.nan
                ),
                "q90_consensus_jsd": (
                    float(np.quantile(consensus_jsd_values, 0.90)) if consensus_jsd_values else np.nan
                ),
                "top_answer_agreement": float(top_counts.iloc[0] / len(group)) if len(top_counts) else np.nan,
                "top_answer_disagreement": (
                    1.0 - float(top_counts.iloc[0] / len(group)) if len(top_counts) else np.nan
                ),
                "parse_failure_rate": group["parse_failed"].mean(),
                "mean_log_total_tokens": group["log_total_tokens"].mean(),
            }
        )
    return pd.DataFrame(rows)


def f7_band(rows: list[dict], *, metric: str, x_col: str = "repeat_count") -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.groupby([x_col, "metric"], as_index=False)["value"]
        .agg(mean="mean", q25=lambda x: np.quantile(x, 0.25), q75=lambda x: np.quantile(x, 0.75))
    )


def f7_plot_band(
    ax,
    data: pd.DataFrame,
    *,
    metric: str,
    label: str,
    color: str,
    ls: str = "-",
    x_col: str = "repeat_count",
) -> None:
    sub = data[data["metric"] == metric].sort_values(x_col)
    if sub.empty:
        return
    ax.plot(sub[x_col], sub["mean"], color=color, lw=2, ls=ls, label=label)
    ax.fill_between(
        sub[x_col].to_numpy(dtype=float),
        sub["q25"].to_numpy(dtype=float),
        sub["q75"].to_numpy(dtype=float),
        color=color,
        alpha=0.12,
        linewidth=0,
    )


def simpleqa_question_sort_key(question_id: object) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(question_id))
    return (int(match.group(1)) if match else 10**12, str(question_id))


def f7_line_convergence(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcome_rows = []
    agreement_rows = []
    frontier_rows = []
    question_rows = []

    for _qid, group in df.sort_values("repeat_index").groupby("question_id", sort=False):
        outcomes = group["candidates"].map(_log_top_p_three_way_outcome)
        outcome_specs = [
            ("accuracy_rate", outcomes == "accuracy"),
            ("hallucination_rate", outcomes == "hallucination"),
            ("abstention_rate", outcomes == "abstention"),
        ]
        for label, mask in outcome_specs:
            values = mask.astype(float).to_numpy()
            final = float(np.mean(values))
            running = np.cumsum(values) / np.arange(1, len(values) + 1)
            for k, estimate in enumerate(running, start=1):
                outcome_rows.append(
                    {
                        "repeat_count": k,
                        "metric": label,
                        "value": float(estimate - final) ** 2,
                    }
                )

        top_keys = [
            f7_member_key({"answer": answer, "grade": grade})
            for answer, grade in zip(group["log_top_answer"], group["log_top_grade"])
        ]
        member_sets = [set(members) for members in group["log_top_p_members"]]
        top_counter: Counter[str] = Counter(top_keys[:1])
        jaccard_sum = 0.0
        jaccard_pairs = 0
        for k in range(2, len(group) + 1):
            new_idx = k - 1
            top_counter[top_keys[new_idx]] += 1
            new_set = member_sets[new_idx]
            for old_set in member_sets[:new_idx]:
                union = old_set | new_set
                jaccard_sum += 1.0 if not union else len(old_set & new_set) / len(union)
            jaccard_pairs += new_idx
            agreement_rows.append(
                {
                    "repeat_count": k,
                    "metric": "top_answer_majority",
                    "value": float(max(top_counter.values()) / k) if top_counter else np.nan,
                }
            )
            agreement_rows.append(
                {
                    "repeat_count": k,
                    "metric": "top_p_set_jaccard",
                    "value": float(jaccard_sum / jaccard_pairs) if jaccard_pairs else np.nan,
                }
            )

    grid = np.linspace(0, 0.95, 96)
    repeat_ids = sorted(df["repeat_index"].unique())
    hallucination_by_repeat = []
    abstention_by_repeat = []
    for repeat_i in repeat_ids:
        group = df[df["repeat_index"] == repeat_i]
        top_prob, _top_correct, top_incorrect = _log_threshold_inputs(group)
        answered = top_prob[:, None] >= grid[None, :]
        hallucination_by_repeat.append(
            np.mean(answered & top_incorrect[:, None], axis=0)
        )
        abstention_by_repeat.append(np.mean(~answered, axis=0))

    hallucination_mat = np.vstack(hallucination_by_repeat)
    abstention_mat = np.vstack(abstention_by_repeat)
    final_hallucination = hallucination_mat.mean(axis=0)
    final_abstention = abstention_mat.mean(axis=0)
    denom = np.arange(1, len(repeat_ids) + 1, dtype=float)[:, None]
    prefix_hallucination = np.cumsum(hallucination_mat, axis=0) / denom
    prefix_abstention = np.cumsum(abstention_mat, axis=0) / denom
    hallucination_error = np.mean(np.square(prefix_hallucination - final_hallucination), axis=1)
    abstention_error = np.mean(np.square(prefix_abstention - final_abstention), axis=1)
    for k, (h_err, a_err) in enumerate(zip(hallucination_error, abstention_error), start=1):
        frontier_rows.append(
            {
                "repeat_count": k,
                "metric": "hallucination_rate",
                "value": float(h_err),
            }
        )
        frontier_rows.append(
            {
                "repeat_count": k,
                "metric": "abstention_rate",
                "value": float(a_err),
            }
        )

    question_order = sorted(df["question_id"].drop_duplicates(), key=simpleqa_question_sort_key)
    question_pos = {question_id: pos for pos, question_id in enumerate(question_order)}
    for _repeat_i, group in df.groupby("repeat_index", sort=False):
        group = (
            group.assign(question_pos=group["question_id"].map(question_pos))
            .sort_values("question_pos")
        )
        outcomes = group["candidates"].map(_log_top_p_three_way_outcome)
        outcome_specs = [
            ("accuracy_rate", outcomes == "accuracy"),
            ("hallucination_rate", outcomes == "hallucination"),
            ("abstention_rate", outcomes == "abstention"),
        ]
        for label, mask in outcome_specs:
            values = mask.astype(float).to_numpy()
            final = float(np.mean(values))
            running = np.cumsum(values) / np.arange(1, len(values) + 1)
            for k, estimate in enumerate(running, start=1):
                question_rows.append(
                    {
                        "question_count": k,
                        "metric": label,
                        "value": float(estimate - final) ** 2,
                    }
                )

    return (
        f7_band(outcome_rows, metric="value"),
        f7_band(agreement_rows, metric="value"),
        f7_band(frontier_rows, metric="value"),
        f7_band(question_rows, metric="value", x_col="question_count"),
    )


def fig_log_consistency_convergence(
    df: pd.DataFrame,
    *,
    label: str,
    n_questions: int,
    n_repeats: int,
    temperature: float,
) -> None:
    outcomes, agreement, frontier_conv, question_conv = f7_line_convergence(df)
    outcomes.assign(panel="top_p_outcome").to_csv(
        OUT / "f7_log_consistency_convergence_top_p_outcomes.csv",
        index=False,
    )
    agreement.assign(panel="agreement").to_csv(
        OUT / "f7_log_consistency_convergence_agreement.csv",
        index=False,
    )
    frontier_conv.assign(panel="frontier").to_csv(
        OUT / "f7_log_consistency_convergence_frontier.csv",
        index=False,
    )
    question_conv.assign(panel="top_p_outcome_by_question").to_csv(
        OUT / "f7_log_consistency_convergence_top_p_outcomes_by_question.csv",
        index=False,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), sharex=False)

    f7_plot_band(axes[0], outcomes, metric="accuracy_rate", label="accuracy", color="#1a73e8")
    f7_plot_band(
        axes[0],
        outcomes,
        metric="hallucination_rate",
        label="hallucination",
        color="#d97706",
    )
    f7_plot_band(
        axes[0],
        outcomes,
        metric="abstention_rate",
        label="abstention",
        color="#009E73",
    )
    axes[0].set_title("A. Top-0.9 outcomes converge", fontsize=10)
    axes[0].set_xlabel("number of replications")
    axes[0].set_ylabel("MSE vs full-replication mean")
    axes[0].legend(fontsize=7)

    f7_plot_band(
        axes[1],
        frontier_conv,
        metric="hallucination_rate",
        label="hallucination rate",
        color="#1a73e8",
    )
    f7_plot_band(
        axes[1],
        frontier_conv,
        metric="abstention_rate",
        label="abstention rate",
        color="#009E73",
    )
    axes[1].set_title("B. Frontier estimate converges", fontsize=10)
    axes[1].set_xlabel("number of replications")
    axes[1].set_ylabel("MSE vs full-replication frontier")
    axes[1].legend(fontsize=7)

    f7_plot_band(
        axes[2],
        question_conv,
        metric="accuracy_rate",
        label="accuracy",
        color="#1a73e8",
        x_col="question_count",
    )
    f7_plot_band(
        axes[2],
        question_conv,
        metric="hallucination_rate",
        label="hallucination",
        color="#d97706",
        x_col="question_count",
    )
    f7_plot_band(
        axes[2],
        question_conv,
        metric="abstention_rate",
        label="abstention",
        color="#009E73",
        x_col="question_count",
    )
    axes[2].set_title("C. Top-0.9 outcomes by questions", fontsize=10)
    axes[2].set_xlabel("number of questions")
    axes[2].set_ylabel("MSE vs full-question mean")
    axes[2].legend(fontsize=7)

    fig.suptitle(
        f"F7. Log-elicitation replication consistency: "
        f"{n_questions} questions; median {n_repeats} replications",
        y=1.03,
    )
    fig.tight_layout()
    savefig(fig, "f7_log_consistency")


def f7_ecdf_panel(
    ax,
    values,
    *,
    color: str,
    label: str,
    x_label: str,
    x_upper: float,
    linestyle: str = "-",
    median_text_y: float = 0.08,
    median_label: str | None = None,
    annotate_median: bool = True,
) -> None:
    vals = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    vals = vals[(vals >= 0) & (vals <= 1)].sort_values().to_numpy()
    if len(vals) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(x_label)
        ax.set_ylabel("cumulative fraction of questions")
        return
    y = np.arange(1, len(vals) + 1) / len(vals)
    median = float(np.median(vals))
    ax.step(vals, y, where="post", color=color, lw=2.2, ls=linestyle, label=label)
    ax.axvline(median, color=color, lw=1.2, alpha=0.35)
    if annotate_median:
        ax.text(
            0.97,
            median_text_y,
            f"{median_label or label} median {median:.3f}",
            ha="right",
            va="bottom",
            fontsize=8,
            color=color,
            transform=ax.transAxes,
        )
    ax.set_xlim(0, x_upper)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(x_label)
    ax.set_ylabel("cumulative fraction of questions")


def f7_axis_upper(values, *, floor: float) -> float:
    vals = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    vals = vals[(vals >= 0) & (vals <= 1)]
    if vals.empty:
        return floor
    return min(1.0, max(floor, float(vals.quantile(0.99)) * 1.08))


def fig_log_consistency_question_variability(
    summary: pd.DataFrame,
    *,
    label: str,
    n_questions: int,
    n_repeats: int,
    temperature: float,
) -> None:
    detail = summary.sort_values(["model_key", "mean_pairwise_tv"], ascending=[True, False])
    detail.to_csv(OUT / "f8_log_consistency_question_variability.csv", index=False)

    model_keys = list(summary["model_key"].dropna().drop_duplicates())
    if not model_keys:
        print("  skipped F8: no model keys found")
        return

    upper_values = pd.concat(
        [summary["mean_pairwise_jsd"], summary["mean_consensus_jsd"]],
        ignore_index=True,
    )
    jsd_upper = f7_axis_upper(upper_values, floor=0.20)
    n_cols = min(3, len(model_keys))
    n_rows = int(np.ceil(len(model_keys) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.25 * n_cols, 3.2 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for ax_idx, (ax, model_key) in enumerate(zip(axes.ravel(), model_keys)):
        sub = summary[summary["model_key"] == model_key]
        model_label = RUNS.get(model_key, {}).get("label", sub["model_label"].iloc[0])
        model_color = MODEL_COLORS.get(model_key, "#1a73e8")
        pairwise_median = float(np.nanmedian(sub["mean_pairwise_jsd"]))
        consensus_median = float(np.nanmedian(sub["mean_consensus_jsd"]))
        f7_ecdf_panel(
            ax,
            sub["mean_pairwise_jsd"],
            color=model_color,
            label=f"pairwise replications (median {pairwise_median:.3f})",
            x_label="mean JS divergence (bits)",
            x_upper=jsd_upper,
            linestyle="-",
            annotate_median=False,
        )
        f7_ecdf_panel(
            ax,
            sub["mean_consensus_jsd"],
            color="#4b5563",
            label=f"to consensus report (median {consensus_median:.3f})",
            x_label="mean JS divergence (bits)",
            x_upper=jsd_upper,
            linestyle="--",
            annotate_median=False,
        )
        prefix = f"{chr(ord('A') + ax_idx)}. " if len(model_keys) > 1 else ""
        ax.set_title(f"{prefix}{model_label}", fontsize=10)
        ax.grid(alpha=0.18, lw=0.6)
        ax.legend(fontsize=8, loc="lower right")

    for ax in axes.ravel()[len(model_keys):]:
        ax.set_visible(False)

    fig.suptitle(
        f"F8. Question-level replication variability: "
        f"{n_questions} questions; median {n_repeats} replications",
        y=1.03,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig(fig, "f8_log_consistency_question_variability")


def f10_replication_specific_question_jsd(df: pd.DataFrame) -> pd.DataFrame:
    """Mean divergence from each replication to the others, by question."""
    rows = []
    for (model_key, question_id), group in df.groupby(["model_key", "question_id"], sort=False):
        group = group.sort_values("repeat_index")
        repeat_ids = group["repeat_index"].astype(int).to_numpy()
        distributions = [
            f7_report_distribution(row.candidates)
            for row in group.itertuples(index=False)
        ]
        arr, _keys = f7_distribution_matrix(distributions)
        if arr.shape[0] <= 1:
            continue
        jsd = f7_pairwise_jsd_matrix_from_array(arr)
        model_label = RUNS.get(model_key, {}).get("label", model_key)
        for pos, repeat_id in enumerate(repeat_ids):
            mask = np.ones(arr.shape[0], dtype=bool)
            mask[pos] = False
            vals = jsd[pos, mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            rows.append(
                {
                    "model_key": model_key,
                    "model_label": model_label,
                    "question_id": question_id,
                    "repeat_index": int(repeat_id),
                    "mean_pairwise_jsd": float(np.mean(vals)),
                    "n_pairwise": int(vals.size),
                }
            )
    return pd.DataFrame(rows)


def f10_ecdf_summary(
    question_summary: pd.DataFrame,
    *,
    value_col: str = "median",
    grid: np.ndarray | None = None,
    grid_size: int = 301,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    vals = (
        question_summary[value_col]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    vals = vals[(vals >= 0) & (vals <= 1)]
    if vals.size == 0:
        return pd.DataFrame()

    if grid is None:
        x_min = float(vals.min())
        x_max = float(vals.max())
        if np.isclose(x_min, x_max):
            x_max = min(1.0, x_min + 1e-3)
        grid = np.linspace(x_min, x_max, grid_size)
    else:
        grid = np.asarray(grid, dtype=float)

    observed = np.searchsorted(np.sort(vals), grid, side="right") / vals.size
    rng = np.random.default_rng(seed)
    ecdf = np.empty((n_boot, grid.size), dtype=float)
    for b in range(n_boot):
        sample = rng.choice(vals, size=vals.size, replace=True)
        ecdf[b] = np.searchsorted(np.sort(sample), grid, side="right") / sample.size
    quantiles = np.quantile(ecdf, [0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
    return pd.DataFrame(
        {
            "mean_pairwise_jsd": grid,
            "q025": quantiles[0],
            "q25": quantiles[1],
            "bootstrap_median": quantiles[2],
            "q75": quantiles[3],
            "q975": quantiles[4],
            "observed": observed,
            "n_questions": vals.size,
            "n_bootstrap": n_boot,
        }
    )


def f10_question_summary(replication_detail: pd.DataFrame) -> pd.DataFrame:
    if replication_detail.empty:
        return pd.DataFrame()
    summary = (
        replication_detail.groupby(["model_key", "model_label", "question_id"], as_index=False)[
            "mean_pairwise_jsd"
        ]
        .agg(
            median="median",
            q25=lambda x: np.quantile(x, 0.25),
            q75=lambda x: np.quantile(x, 0.75),
            q025=lambda x: np.quantile(x, 0.025),
            q975=lambda x: np.quantile(x, 0.975),
            n_replications="count",
        )
    )
    return summary


def fig_log_consistency_ecdf_tail_questions(
    df: pd.DataFrame,
    *,
    label: str,
    n_questions: int,
    n_repeats: int,
    temperature: float,
    random_seed: int = 0,
) -> None:
    detail = f10_replication_specific_question_jsd(df)
    if detail.empty:
        print("  skipped F9: no replication-specific JSD values found")
        return

    auxiliary_summary = f10_question_summary(detail)
    if auxiliary_summary.empty:
        print("  skipped F9: insufficient JSD summaries")
        return

    question_summary = summarize_log_consistency(df)
    if question_summary.empty:
        print("  skipped F9: insufficient question-level pairwise JSD summaries")
        return

    detail.to_csv(OUT / "f9_log_consistency_replication_question_jsd.csv", index=False)
    auxiliary_summary.to_csv(OUT / "f9_log_consistency_question_tail.csv", index=False)
    question_summary.to_csv(
        OUT / "f9_log_consistency_question_mean_pairwise_jsd.csv",
        index=False,
    )

    x_values = (
        question_summary["mean_pairwise_jsd"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    x_values = x_values[(x_values >= 0) & (x_values <= 1)]
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    if np.isclose(x_min, x_max):
        x_max = min(1.0, x_min + 1e-3)
    x_pad = max(0.005, 0.04 * (x_max - x_min))
    x_left = max(0.0, x_min - x_pad)
    x_right = min(1.0, x_max + x_pad)
    common_grid = np.linspace(x_min, x_max, 301)

    available_model_keys = list(question_summary["model_key"].drop_duplicates())
    model_keys = [key for key in MODEL_ORDER if key in available_model_keys]
    model_keys.extend(key for key in available_model_keys if key not in model_keys)
    ecdf_frames = []
    for model_idx, model_key in enumerate(model_keys):
        model_summary = question_summary[question_summary["model_key"] == model_key]
        model_ecdf = f10_ecdf_summary(
            model_summary,
            value_col="mean_pairwise_jsd",
            grid=common_grid,
            seed=random_seed + model_idx,
        )
        if model_ecdf.empty:
            continue
        model_ecdf.insert(0, "model_key", model_key)
        model_ecdf.insert(
            1,
            "model_label",
            RUNS.get(model_key, {}).get("label", model_key),
        )
        ecdf_frames.append(model_ecdf)
    if not ecdf_frames:
        print("  skipped F9: insufficient model-specific JSD summaries")
        return
    ecdf = pd.concat(ecdf_frames, ignore_index=True)
    ecdf.to_csv(OUT / "FS2_JSD.csv", index=False)

    x_axis_label = "mean pairwise Jensen–Shannon divergence"
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    legend_handles = []
    median_names = {
        "gemini35flash": "Gemini",
        "sonnet46": "Sonnet",
        "deepseekv32maas": "DeepSeek",
        "qwen3_235b": "Qwen",
    }
    median_by_model = {
        model_key: float(
            np.median(
                question_summary.loc[
                    question_summary["model_key"] == model_key,
                    "mean_pairwise_jsd",
                ].dropna()
            )
        )
        for model_key in model_keys
    }
    ax.axhline(
        0.5,
        color="#6b7280",
        lw=0.8,
        linestyle=(0, (3, 3)),
        alpha=0.8,
        zorder=0,
    )
    for model_idx, model_key in enumerate(model_keys):
        model_ecdf = ecdf[ecdf["model_key"] == model_key]
        if model_ecdf.empty:
            continue
        model_summary = question_summary[question_summary["model_key"] == model_key]
        model_label = RUNS.get(model_key, {}).get("label", model_key)
        color = MODEL_COLORS.get(model_key, "#4b5563")
        x = model_ecdf["mean_pairwise_jsd"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            model_ecdf["q025"].to_numpy(dtype=float),
            model_ecdf["q975"].to_numpy(dtype=float),
            color=color,
            alpha=0.10,
            linewidth=0,
        )
        ax.plot(
            x,
            model_ecdf["observed"].to_numpy(dtype=float),
            color=color,
            lw=2.2,
        )
        ecdf_median = median_by_model[model_key]
        ax.text(
            0.98,
            0.70 - 0.052 * model_idx,
            f"{median_names.get(model_key, model_label)}: median {ecdf_median:.3f}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8,
            color=color,
            path_effects=[withStroke(linewidth=2.5, foreground="white")],
            zorder=5,
        )
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2.2, label=model_label)
        )
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("cumulative fraction of questions")
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(
        handles=legend_handles,
        fontsize=8,
        loc="lower right",
        frameon=False,
    )
    fig.tight_layout()
    savefig(fig, "f9_log_consistency_jsd_diagnostics")


def f7_repeat_pairwise_jsd_matrix(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeat_ids = sorted(model_df["repeat_index"].dropna().astype(int).unique())
    repeat_pos = {repeat_id: i for i, repeat_id in enumerate(repeat_ids)}
    sums = np.zeros((len(repeat_ids), len(repeat_ids)), dtype=float)
    counts = np.zeros((len(repeat_ids), len(repeat_ids)), dtype=int)

    for _question_id, group in model_df.groupby("question_id", sort=False):
        local_repeat_ids = [int(row.repeat_index) for row in group.itertuples(index=False)]
        distributions = [
            f7_report_distribution(row.candidates)
            for row in group.itertuples(index=False)
        ]
        if not local_repeat_ids:
            continue
        arr, _keys = f7_distribution_matrix(distributions)
        local_jsd = f7_pairwise_jsd_matrix_from_array(arr)
        idx = np.array([repeat_pos[repeat_id] for repeat_id in local_repeat_ids], dtype=int)
        sums[np.ix_(idx, idx)] += local_jsd
        counts[np.ix_(idx, idx)] += 1

    means = np.full_like(sums, np.nan, dtype=float)
    np.divide(sums, counts, out=means, where=counts > 0)
    rows = [
        {
            "repeat_i": repeat_i,
            "repeat_j": repeat_j,
            "mean_jsd": float(means[i, j]) if counts[i, j] > 0 else np.nan,
            "n_questions": int(counts[i, j]),
        }
        for i, repeat_i in enumerate(repeat_ids)
        for j, repeat_j in enumerate(repeat_ids)
    ]
    long = pd.DataFrame(rows)
    matrix = long.pivot(index="repeat_i", columns="repeat_j", values="mean_jsd").reindex(
        index=repeat_ids,
        columns=repeat_ids,
    )
    return matrix, long


def f7_pairwise_jsd_heatmap_data(
    df: pd.DataFrame,
) -> tuple[list[str], dict[str, pd.DataFrame], pd.DataFrame, float]:
    model_keys = list(df["model_key"].drop_duplicates())
    matrices: dict[str, pd.DataFrame] = {}
    long_rows = []
    vmax_values = []
    for model_key in model_keys:
        model_df = df[df["model_key"] == model_key]
        matrix, long = f7_repeat_pairwise_jsd_matrix(model_df)
        matrices[model_key] = matrix
        long.insert(0, "model_key", model_key)
        long.insert(1, "model_label", RUNS.get(model_key, {}).get("label", model_key))
        long_rows.append(long)
        vals = matrix.to_numpy(dtype=float)
        vmax_values.extend(vals[np.isfinite(vals)].ravel().tolist())

    long = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    vmax = min(1.0, max(0.05, float(np.quantile(vmax_values, 0.99)) if vmax_values else 0.05))
    return model_keys, matrices, long, vmax


def fig_log_consistency_pairwise_jsd_heatmap(
    df: pd.DataFrame,
    *,
    label: str,
    n_questions: int,
    n_repeats: int,
    temperature: float,
) -> None:
    model_keys, matrices, long, vmax = f7_pairwise_jsd_heatmap_data(df)

    if not long.empty:
        long.to_csv(
            OUT / "f9_log_consistency_pairwise_jsd_heatmap.csv",
            index=False,
        )

    fig_width = max(4.2, 3.5 * len(model_keys))
    fig, axes = plt.subplots(
        1,
        len(model_keys),
        figsize=(fig_width, 3.65),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for ax, model_key in zip(axes.ravel(), model_keys):
        matrix = matrices[model_key]
        repeat_labels = matrix.index.to_numpy()
        image = ax.imshow(
            matrix.to_numpy(dtype=float),
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(RUNS.get(model_key, {}).get("label", model_key), fontsize=10)
        ax.set_xlabel("replication index")
        ax.set_ylabel("replication index")
        n_ticks = min(5, len(repeat_labels))
        if n_ticks > 0:
            tick_pos = np.linspace(0, len(repeat_labels) - 1, n_ticks, dtype=int)
            ax.set_xticks(tick_pos, [str(int(repeat_labels[pos])) for pos in tick_pos], fontsize=7)
            ax.set_yticks(tick_pos, [str(int(repeat_labels[pos])) for pos in tick_pos], fontsize=7)

    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, pad=0.03)
        cbar.set_label("mean pairwise JS divergence (bits)")
    fig.suptitle(
        f"F9. Replication-index pairwise Jensen-Shannon heatmap: "
        f"{n_questions} questions; median {n_repeats} replications",
        y=1.03,
    )
    savefig(fig, "f9_log_consistency_pairwise_jsd_heatmap")


def fig_log_consistency(
    run_dir_name: str | None,
    *,
    question_limit: int | None = None,
    repeat_limit: int | None = None,
) -> bool:
    run_dir_names = [
        part.strip()
        for part in str(run_dir_name or "").split(",")
        if part.strip()
    ]
    if not run_dir_names:
        print("== F7: skipped (no log consistency run directory configured) ==")
        return False
    frames = []
    for name in run_dir_names:
        try:
            frames.append(load_log_consistency(name))
        except FileNotFoundError as exc:
            print(f"  F7: skipped missing run {exc.filename}")
    if not frames:
        print("== F7: skipped (no log consistency run files found) ==")
        return False
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        print("== F7: skipped (log consistency file has no rows) ==")
        return False

    if repeat_limit is not None:
        if repeat_limit <= 1:
            raise ValueError("repeat_limit must be at least 2")
        df = df[df["repeat_index"].astype(int) < repeat_limit].copy()

    if question_limit is not None:
        if question_limit <= 0:
            raise ValueError("question_limit must be positive")
        present_model_keys = set(df["model_key"].astype(str))
        missing_models = [
            model_key
            for model_key in ACTIVE_MODEL_ORDER
            if model_key not in present_model_keys
        ]
        if missing_models:
            raise RuntimeError(
                "Missing requested log-consistency models: "
                + ", ".join(missing_models)
            )
        question_order = list(simpleqa_full_question_order())
        if question_limit > len(question_order):
            raise ValueError(
                f"question_limit={question_limit} exceeds the "
                f"{len(question_order)}-question SimpleQA order"
            )
        selected_ids = set(question_order[:question_limit])
        df = df[df["question_id"].astype(str).isin(selected_ids)].copy()
        missing_by_model = {}
        for model_key, model_df in df.groupby("model_key", sort=False):
            present_ids = set(model_df["question_id"].astype(str))
            missing = selected_ids - present_ids
            if missing:
                missing_by_model[model_key] = len(missing)
        if missing_by_model:
            details = ", ".join(
                f"{model_key}: {count} missing"
                for model_key, count in missing_by_model.items()
            )
            raise RuntimeError(
                "The requested shared consistency subset is incomplete ("
                + details
                + ")"
            )
        print(
            f"  F7: restricted every model to the first {question_limit} "
            "questions in the deterministic SimpleQA order"
        )

    if repeat_limit is not None:
        repeat_counts = df.groupby(["model_key", "question_id"]).size()
        incomplete = repeat_counts[repeat_counts != repeat_limit]
        if not incomplete.empty:
            examples = ", ".join(
                f"{model_key}/{question_id}: {count}"
                for (model_key, question_id), count in incomplete.head(5).items()
            )
            raise RuntimeError(
                f"The requested {repeat_limit}-replication subset is incomplete; "
                f"examples: {examples}"
            )
        print(
            f"  F7: restricted every model--question pair to replications "
            f"0--{repeat_limit - 1}"
        )

    print("== F7: repeated log-elicitation consistency ==")
    summary = summarize_log_consistency(df)
    df.drop(columns=["candidates"], errors="ignore").to_csv(
        OUT / "f7_log_consistency_repeats.csv",
        index=False,
    )
    summary.to_csv(OUT / "f7_log_consistency_by_question.csv", index=False)

    label = (
        summary["model_label"].iloc[0]
        if summary["model_label"].nunique() == 1
        else ", ".join(summary["model_label"].drop_duplicates())
    )
    n_questions = summary["question_id"].nunique()
    n_repeats = int(summary["n_repeats"].median())
    temperature = df["log_temperature"].dropna().iloc[0] if df["log_temperature"].notna().any() else np.nan

    fig_log_consistency_convergence(
        df,
        label=label,
        n_questions=n_questions,
        n_repeats=n_repeats,
        temperature=temperature,
    )
    fig_log_consistency_question_variability(
        summary,
        label=label,
        n_questions=n_questions,
        n_repeats=n_repeats,
        temperature=temperature,
    )
    fig_log_consistency_ecdf_tail_questions(
        df,
        label=label,
        n_questions=n_questions,
        n_repeats=n_repeats,
        temperature=temperature,
    )
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate Stage 1 figures with an optional subset of models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_ORDER),
        help=(
            "Model keys or aliases to include. Valid keys: "
            f"{', '.join(RUNS)}. Aliases include gemini, sonnet, "
            "deepseek, and qwen. "
            "Comma-separated values are also accepted."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT,
        help="Directory for generated figures and CSVs.",
    )
    parser.add_argument(
        "--f7-run-dir",
        default=DEFAULT_F7_RUN_DIR,
        help=(
            "Output directory name under outputs/ containing "
            "simpleqa_log_consistency_results.jsonl. Supply comma-separated "
            "directories to combine model ECDFs, or an empty string to skip F7."
        ),
    )
    parser.add_argument(
        "--f7-question-limit",
        type=int,
        default=None,
        help=(
            "Restrict every log-consistency model to the same first N questions "
            "in the deterministic SimpleQA order."
        ),
    )
    parser.add_argument(
        "--f7-repeat-limit",
        type=int,
        default=None,
        help="Restrict every log-consistency question to replication indices below N.",
    )
    parser.add_argument(
        "--only-log-consistency",
        action="store_true",
        help="Generate only the repeated-log consistency outputs, including FS2_JSD.pdf.",
    )
    args = parser.parse_args(argv)
    args.models = parse_model_keys(args.models)
    return args


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    configure_run(model_keys=args.models, out_dir=args.out_dir)
    print("Selected models:", ", ".join(ACTIVE_MODEL_ORDER))
    print("Output directory:", OUT)
    if args.only_log_consistency:
        wrote_f7 = fig_log_consistency(
            args.f7_run_dir,
            question_limit=args.f7_question_limit,
            repeat_limit=args.f7_repeat_limit,
        )
        if not wrote_f7:
            raise SystemExit("No log-consistency figure was generated.")
        print("\nDone. Updated log-consistency outputs in", OUT)
        return
    paired = load_paired()
    fig_frontier(paired)
    fig_frontier_accuracy(paired)
    fig_penalty_log_outcome_tables(paired)
    fig_cumulative_rates(paired)
    fig_cumulative_three_way_rates(paired)
    fig_log_vs_penalty_convergence(paired)
    fig_threshold_three_way_response(paired)
    fig_responses_per_question(paired)
    wrote_f7 = fig_log_consistency(
        args.f7_run_dir,
        question_limit=args.f7_question_limit,
        repeat_limit=args.f7_repeat_limit,
    )
    suffix = "F1-F9" if wrote_f7 else "F1-F6"
    print(f"\nDone. Updated {suffix} figures in", OUT)


if __name__ == "__main__":
    main()
