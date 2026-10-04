"""Generate the paper's figures from the graded results.

Outputs, written to results/figures/ with the CSV data corresponding to each figure:
F3_Frontier.pdf (Fig. 3), F4_OutcomeTable_L3.pdf (Fig. 4),
FS1_OutcomeTable_L0/L6.pdf (Fig. S1), FS2_JSD.pdf (Fig. S2),
FS3_cumulative.pdf (Fig. S3), and F2_ResponseCounts.pdf (Fig. 2).

Run: python analysis/make_figures.py   (see analysis/reproduce_all.py)
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
DEFAULT_CONSISTENCY_RUNS = "claudesonnet46_consistency"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def savefig(fig, name: str):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {name}.pdf")


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


def penalty_run_dir_name(cfg: dict, penalty: float) -> str:
    if np.isclose(penalty, PRIMARY_PENALTY):
        return cfg["run"]
    return f"{cfg['run']}_L{penalty:g}"


def load_paired():
    """Per model: the elicited log run in the canonical SimpleQA order
    ("log"), the penalty arms by level ("penalties"), and the log rows
    aligned to each penalty arm's questions ("log_by_penalty")."""
    global ACTIVE_MODEL_ORDER
    out = {}
    loaded_keys = []
    canonical_order = list(simpleqa_full_question_order())
    for key in list(ACTIVE_MODEL_ORDER):
        try:
            dlog_all = load_run(RUNS[key]["run"]).set_index("question_id", drop=False)
        except FileNotFoundError as exc:
            print(f"  {key}: skipped (missing run file: {exc.filename})")
            continue
        log_set = set(dlog_all["question_id"])
        order = [q for q in canonical_order if q in log_set]
        order.extend(q for q in dlog_all["question_id"] if q not in set(order))
        if not order:
            print(f"  {key}: skipped (no log rows)")
            continue
        dlog = dlog_all.loc[order]
        penalties = {}
        log_by_penalty = {}
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
            pq = [q for q in canonical_order if q in log_set and q in penalty_set]
            pq.extend(
                q for q in dlog_all["question_id"]
                if q in penalty_set and q not in set(pq)
            )
            if not pq:
                if np.isclose(penalty, PRIMARY_PENALTY):
                    print(f"  {key}: skipped (no paired primary penalty-arm questions)")
                    penalties = {}
                    break
                print(f"  {key}: penalty {penalty_label(penalty)} rows=0 (no paired qids)")
                continue
            penalties[float(penalty)] = dpen_all.loc[pq]
            log_by_penalty[float(penalty)] = dlog_all.loc[pq]
            print(
                f"  {key}: penalty {penalty_label(penalty)} rows={len(pq)} "
                f"(source={penalties[float(penalty)].attrs.get('source')})"
            )
        if PRIMARY_PENALTY not in penalties:
            continue
        out[key] = {
            "log": dlog,
            "penalties": penalties,
            "log_by_penalty": log_by_penalty,
        }
        loaded_keys.append(key)
    if not out:
        raise SystemExit("No selected models had complete log and penalty rows.")
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
    """Return one penalty row per (model, question_id), keeping the last.

    The penalty runner is resumable, so a restarted run can append duplicate
    question rows. Seeds are fixed, so the latest completed row is canonical.
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

    L=3 is in the model's main run directory; other levels are read from
    separate directories, e.g. *_L6.
    """
    cfg = RUNS[key]
    path = result_file(
        OUTPUTS_DIR / penalty_run_dir_name(cfg, penalty) / "simpleqa_penalty_results.jsonl"
    )
    if path is None:
        return pd.DataFrame()
    return _penalty_df_from_rows(
        load_jsonl(path),
        model_id=cfg["model_id"],
        source=str(path.relative_to(OUTPUTS_DIR)),
        penalty=penalty,
    )


# Fig. 3: hallucination-abstention frontier
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
    """Interpolate a threshold frontier onto abstention-rate values."""
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


def fig_frontier(paired, relative):
    """Figure 3: aligned A/B frontiers and C relative gains, with one legend."""
    print("== Fig. 3: RBD/EPP frontiers and relative accuracy gains ==")
    grid = np.linspace(0, 0.95, 96)
    band_rows, curve_rows, point_rows = [], [], []
    n_models = len(ACTIVE_MODEL_ORDER)
    fig = plt.figure(figsize=(max(5.0, 4.05 * n_models), 8.55))
    # Separate spacer rows keep A--B compact while allowing B's x-axis labels.
    layout = fig.add_gridspec(5, n_models, height_ratios=[1, .30, 1, .40, .72])
    axes = np.empty((3, n_models), dtype=object)
    for row in range(3):
        for col in range(n_models):
            axes[row, col] = fig.add_subplot(
                layout[2 * row, col], sharey=axes[row, 0] if col else None,
            )
            axes[row, col].tick_params(labelleft=col == 0)
    row_specs = [("hallucination_rate", "penalty_hallucination_rate"),
                 ("accuracy_overall", "penalty_accuracy_overall")]
    for col, key in enumerate(ACTIVE_MODEL_ORDER):
        color = MODEL_COLORS[key]
        reports = paired[key]["log_by_penalty"][PRIMARY_PENALTY]
        fr = frontier(reports, grid)
        curve_rows.extend(fr.assign(model=key, model_label=RUNS[key]["label"]).to_dict("records"))
        band_x = np.linspace(float(fr.abstention_rate.min()), float(fr.abstention_rate.max()), 121)
        bands = bootstrap_frontier_bands(reports, grid, band_x, seed=91000 + 97 * col)
        for row, (metric, epp_metric) in enumerate(row_specs):
            ax = axes[row, col]
            lo, hi = bands[metric]
            ax.fill_between(band_x, lo, hi, color=color, alpha=.38, linewidth=0)
            band_rows.extend(dict(model=key, model_label=RUNS[key]["label"], metric=metric,
                                  abstention_rate=x, lo_95=l, hi_95=h)
                             for x, l, h in zip(band_x, lo, hi))
            ax.plot(fr.abstention_rate, fr[metric], color=color, lw=2, label="RBD")
            for level, epp in paired[key]["penalties"].items():
                x, y = epp.penalty_abstention_rate.mean(), epp[epp_metric].mean()
                ax.scatter([x], [y], color=PENALTY_COLORS[level], marker="o",
                           s=52 if level == PRIMARY_PENALTY else 44, edgecolors="white",
                           linewidths=.6, zorder=5, label=fr"EPP, $L={level:g}$")
                point_rows.append(dict(model=key, L=level, metric=metric, abstention=x, value=y))
            ax.set(xlim=(-.02, 1.02), ylim=(-.02, .90 if row == 0 else .65))
            ax.grid(alpha=.18, lw=.6)
            if row == 1:
                ax.set_xlabel("abstention", fontsize=14)
            ax.tick_params(axis="both", labelsize=11.5)
        axes[0, col].set_title(RUNS[key]["label"], fontsize=16, fontweight="bold", pad=12)
        ax = axes[2, col]
        for pos, (_, r) in enumerate(relative[relative.model == key].sort_values("L").iterrows()):
            ax.errorbar(pos, r.relative_accuracy_gain,
                        yerr=[[r.relative_accuracy_gain-r.ci_low], [r.ci_high-r.relative_accuracy_gain]],
                        fmt="s" if r.L == 3 else "^", markersize=7, color=color, capsize=4, lw=1.6)
            ax.annotate(f"{r.relative_accuracy_gain:.1f}%", (pos, r.relative_accuracy_gain),
                        xytext=(10, 0), textcoords="offset points", va="center", fontsize=13)
        ax.axhline(0, color=".6", ls="--", lw=.9, zorder=0)
        ax.set(xticks=[0, 1], xticklabels=["$L=3$", "$L=6$"], xlim=(-.45, 1.55),
               ylim=(-3, 25), yticks=[0, 10, 20])
        ax.tick_params(axis="both", labelsize=12)
    axes[0, 0].set_ylabel("hallucination", fontsize=14)
    axes[1, 0].set_ylabel("accuracy", fontsize=14)
    axes[2, 0].set_ylabel("Relative gain (%)", fontsize=14)
    fig.subplots_adjust(left=.085, right=.992, bottom=.065, top=.88, hspace=0, wspace=.16)
    for row, letter in enumerate("ABC"):
        fig.text(.008, axes[row, 0].get_position().y1 + .012, letter, fontsize=17, fontweight="bold", va="bottom")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles[0] = Line2D([0], [0], color="black", lw=2)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.535, .995),
               ncol=len(labels), frameon=False, fontsize=13, markerscale=1.3)
    pd.DataFrame(band_rows).to_csv(OUT / "F3_Frontier.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(OUT / "F3_Frontier_curves.csv", index=False)
    pd.DataFrame(point_rows).to_csv(OUT / "F3_EPP_points.csv", index=False)
    fig.savefig(OUT / "F3_Frontier.pdf")
    plt.close(fig)
    print("  wrote F3_Frontier.pdf")


def _has_empirical_baseline(df: pd.DataFrame) -> bool:
    return (
        "empirical_accuracy_overall" in df
        and df["empirical_accuracy_overall"].notna().any()
        and "empirical_distribution_json" in df
        and df["empirical_distribution_json"].notna().any()
    )


# Fig. 2: distinct response entries per question, by method
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


def fig_response_counts(paired):
    print("== Fig. 2: responses per question ==")
    fig, axes = model_panel_subplots(height=4.8, sharey=True)
    count_rows = []
    count_data = {}
    for ax, key in zip(axes, ACTIVE_MODEL_ORDER):
        df = paired[key]["log_by_penalty"][PRIMARY_PENALTY]
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
            label = "RBD" if method == "log" else fr"EPP, $L = {penalty:g}$"
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
        ax.text(0.66, 0.965, "mean", transform=ax.transAxes,
                ha="left", va="top", fontsize=10.5, color="#111827")
        for idx, (method, penalty, counts) in enumerate(series):
            mean_label = "RBD" if method == "log" else f"EPP, {penalty_label(penalty)}"
            ax.text(0.66, 0.965 - 0.068 * (idx + 1),
                    f"{mean_label}: {np.mean(counts):.1f}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=10.5, color="#111827")
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
    axes[0].set_ylabel("proportion of questions", fontsize=17)
    legend_handles = [
        Patch(facecolor=PENALTY_BAR_COLORS[p], alpha=0.82, label=fr"EPP, $L = {p:g}$")
        for p in PENALTY_LEVELS
        if any(p in count_data[key]["penalty"] for key in count_data)
    ]
    legend_handles.append(Patch(facecolor="#ffffff", edgecolor="#111827",
                                label="RBD\n(model color)"))
    pd.DataFrame(count_rows).to_csv(OUT / "F2_ResponseCounts.csv", index=False)
    fig.tight_layout()
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=17,
    )
    savefig(fig, "F2_ResponseCounts")


# Fig. S3: cumulative mean display-content frequencies
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


def cumulative_rates_for_key(paired, key: str) -> pd.DataFrame:
    """Cumulative rates for the three display-content categories.

    For a log report, the top-p prefix is truth-containing, no-concrete-answer,
    or a concrete response without the truth. Error-penalty-prompt samples map to the same categories
    through correct, abstain/not-attempted, and incorrect, respectively.
    """
    specs = [
        (
            "residual_log_top_p",
            *_log_top_p_three_way_arrays(paired[key]["log_by_penalty"][PRIMARY_PENALTY]),
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


def fig_cumulative_rates(paired) -> pd.DataFrame:
    print("== Fig. S3: cumulative mean display-content frequencies ==")
    rows = pd.concat(
        [cumulative_rates_for_key(paired, key) for key in ACTIVE_MODEL_ORDER],
        ignore_index=True,
    )
    rows.to_csv(OUT / "FS3_cumulative.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12.9, 4.65), sharex=False)
    panels = [
        ("accuracy_rate", "correct answer present"),
        ("hallucination_rate", "only incorrect candidate answers"),
        ("abstention_rate", "no candidate answer"),
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
                *_log_top_p_three_way_arrays(paired[key]["log_by_penalty"][PRIMARY_PENALTY]),
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
        ax.set_ylabel(ylab, fontsize=12)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.18, lw=0.6)
        ax.tick_params(axis="both", labelsize=10.5)

    model_handles = [
        Line2D([0], [0], color=MODEL_COLORS[key], lw=2, label=RUNS[key]["label"])
        for key in ACTIVE_MODEL_ORDER
    ]
    method_handles = [
        Line2D([0], [0], color="black", lw=2.4, ls="-",
               label=f"RBD, Top-{NOMINAL_P:g}"),
    ]
    for penalty in sorted({p for k in ACTIVE_MODEL_ORDER for p in paired[k]["penalties"]}):
        method_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                lw=2.4,
                ls=PENALTY_LINESTYLES.get(penalty, "--"),
                label=fr"EPP, $L = {penalty:g}$",
            )
        )
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
    savefig(fig, "FS3_cumulative")
    return rows


# Fig. 4 / Fig. S1: answer-or-abstain responses versus report-based set outcomes
OUTCOME_KEYS = ["abstain", "correct", "incorrect"]
PENALTY_OUTCOME_LABELS = ["abstain", "correct", "incorrect"]
SET_OUTCOME_KEYS = ["coverage", "miscoverage_with_idk", "miscoverage_without_idk"]
SET_OUTCOME_LABELS = ["Coverage", "Miscoverage\nwith IDK", "Miscoverage\nwithout IDK"]


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
    """Legacy raw keys: correct=coverage, abstain=miscoverage with IDK,
    incorrect=miscoverage without IDK. These keys describe sets, not decisions.
    """
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
    dlog = paired[key]["log_by_penalty"][penalty]
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
    # Preserve all three mutually exclusive set outcomes, ordered as displayed.
    matrices = {
        key: pd.DataFrame({
            "coverage": mat.loc[OUTCOME_KEYS, "correct"],
            "miscoverage_with_idk": mat.loc[OUTCOME_KEYS, "abstain"],
            "miscoverage_without_idk": mat.loc[OUTCOME_KEYS, "incorrect"],
        })
        for key, mat in matrices.items()
    }
    cells = []
    for key in model_keys:
        mat = matrices[key]
        for row in OUTCOME_KEYS:
            row_total = float(mat.loc[row].sum())
            for column in SET_OUTCOME_KEYS:
                value = float(mat.loc[row, column])
                cells.append({
                    "model": key, "model_label": RUNS[key]["label"],
                    "mode": mode, "penalty_level": penalty,
                    "penalty_outcome": row, "set_outcome": column,
                    "count_or_expected_count": value,
                    "row_percent": 100 * value / row_total if row_total else 0.0,
                })
    pd.DataFrame(cells).to_csv(OUT / f"{name}_set_outcomes.csv", index=False)
    n_models = len(model_keys)
    fig, axes = plt.subplots(1, n_models, figsize=(max(5.2, 4.6 * n_models), 5.1))
    fig.subplots_adjust(left=.10, right=.947, bottom=.18, top=.88, wspace=.34)
    axes = np.atleast_1d(axes).ravel()
    vmax = max((float(mat.to_numpy().max()) for mat in matrices.values()), default=1.0)
    vmax = max(vmax, 1.0)
    norm = Normalize(vmin=0, vmax=vmax)
    for ax, key in zip(axes, model_keys):
        mat = matrices[key]
        arr = mat[SET_OUTCOME_KEYS].to_numpy(dtype=float)
        base_color = MODEL_COLORS.get(key, "#1a73e8")
        cmap = _model_question_cmap(key, base_color)
        ax.imshow(arr, cmap=cmap, norm=norm)
        # Separate coverage from both miscoverage categories with a white gutter
        # and a thin neutral divider, without altering the cell coordinates.
        ax.axvline(.5, color="white", linewidth=6, zorder=2)
        ax.axvline(.5, color="#64748b", linewidth=.8, zorder=3)
        row_totals = arr.sum(axis=1)
        for i in range(arr.shape[0]):
            row_total = float(row_totals[i])
            for j in range(arr.shape[1]):
                value = arr[i, j]
                pct = 100 * value / row_total if row_total > 0 else 0.0
                value_text = f"{value:.0f}" if mode == "first_sample" else f"{value:.1f}"
                if mode != "first_sample" and 0 < value < 0.05:
                    value_text = "<0.1"
                pct_text = "<0.1%" if 0 < pct < 0.05 else f"{pct:.1f}%"
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
                    fontsize=14.5,
                    fontweight=800,
                    color=text_color,
                )
                ax.annotate(
                    f"({pct_text})",
                    xy=(j, i),
                    xycoords="data",
                    xytext=(0, -9.6),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=13.3,
                    color=text_color,
                ).set_path_effects(
                    [withStroke(linewidth=0.01, foreground=text_color)]
                )
        title_effect = [withStroke(linewidth=0.5, foreground="#111827")]
        ax.set_xticks(
            range(len(SET_OUTCOME_KEYS)), SET_OUTCOME_LABELS,
            fontsize=12.3, fontfamily="Arial" if any(f.name == "Arial" for f in matplotlib.font_manager.fontManager.ttflist) else "DejaVu Sans",
        )
        for label in ax.get_xticklabels():
            label.set_color("#111827")
        ax.set_yticks(range(len(OUTCOME_KEYS)))
        ax.set_yticklabels([])
        for i, (base_label, row_total) in enumerate(zip(PENALTY_OUTCOME_LABELS, row_totals)):
            total_text = f"{row_total:.0f}" if mode == "first_sample" else f"{row_total:.1f}"
            row_label = base_label.replace("\n", " ")
            ax.annotate(
                row_label,
                xy=(-0.035, i),
                xycoords=ax.get_yaxis_transform(),
                xytext=(0, 6),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=13.3,
                color="#111827",
                clip_on=False,
            )
            ax.annotate(
                total_text,
                xy=(-0.035, i),
                xycoords=ax.get_yaxis_transform(),
                xytext=(0, -9),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=12.8,
                color="#4b5563",
                clip_on=False,
            ).set_path_effects(
                [withStroke(linewidth=0.2, foreground="#4b5563")]
            )
        ax.set_xlabel(f"RBD, Top-{NOMINAL_P:g}", labelpad=12, fontsize=14)
        ax.set_title(
            RUNS[key]["label"],
            fontsize=16.5,
            color="#111827",
            pad=11,
        ).set_path_effects(title_effect)
        ax.tick_params(axis="x", rotation=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
    axes[0].set_ylabel(
        f"EPP, {penalty_label(penalty)}", labelpad=66, fontsize=14.5
    )
    subtle_count_cmap = LinearSegmentedColormap.from_list(
        "subtle_question_counts",
        ["#ffffff", "#d1d5db", "#6b7280"],
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=subtle_count_cmap)
    sm.set_array([])
    cax = fig.add_axes([.965, .22, .008, .61])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=8, length=2.5, width=0.5, colors="#111827")
    cbar.outline.set_visible(False)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=.13)
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def _fig_outcome_table_for_penalty(
    paired,
    penalty: float,
    *,
    name: str,
) -> dict[str, pd.DataFrame]:
    outputs = {}
    mode = "sample_average"
    model_keys = [key for key in ACTIVE_MODEL_ORDER if penalty in paired[key]["penalties"]]
    if not model_keys:
        print(f"  skipped: no {penalty_label(penalty)} penalty rows")
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
    detail.to_csv(OUT / f"{name}_long.csv", index=False)

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
    summary.to_csv(OUT / f"{name}.csv", index=False)
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


def fig_outcome_tables(paired) -> dict[str, pd.DataFrame]:
    print("== Fig. 4 / Fig. S1: answer-or-abstain responses vs three set outcomes ==")
    outputs = {}
    for penalty in PENALTY_LEVELS:
        name = ("F4_OutcomeTable_L3" if np.isclose(penalty, PRIMARY_PENALTY)
                else f"FS1_OutcomeTable_{penalty_tag(penalty)}")
        outputs.update(
            _fig_outcome_table_for_penalty(
                paired,
                penalty,
                name=name,
            )
        )
    return outputs


# Fig. S2: elicitation consistency
def member_key(candidate: dict) -> str:
    if candidate.get("grade") == "not_attempted":
        return "__IDK__"
    text = str(candidate.get("answer", ""))
    return norm_answer(text) or text.strip().lower()


def pairwise_jaccard(sets: list[set[str]]) -> float:
    if len(sets) <= 1:
        return np.nan
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            vals.append(1.0 if not union else len(sets[i] & sets[j]) / len(union))
    return float(np.mean(vals)) if vals else np.nan


def report_distribution(candidates: list[dict]) -> dict[str, float]:
    dist: dict[str, float] = {}
    for candidate in candidates:
        try:
            prob = float(candidate.get("probability", 0.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(prob) or prob <= 0:
            continue
        key = member_key(candidate)
        dist[key] = dist.get(key, 0.0) + prob
    total = sum(dist.values())
    if total > 0:
        dist = {key: value / total for key, value in dist.items()}
    return dist


def distribution_matrix(
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


def entropy_array(arr: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(arr > 0, -arr * np.log2(arr), 0.0)
    return terms.sum(axis=-1)


def pairwise_jsd_matrix_from_array(arr: np.ndarray) -> np.ndarray:
    n = arr.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if arr.shape[1] == 0:
        return np.zeros((n, n), dtype=float)
    h = entropy_array(arr)
    mid = 0.5 * (arr[:, None, :] + arr[None, :, :])
    jsd = entropy_array(mid) - 0.5 * (h[:, None] + h[None, :])
    return np.maximum(0.0, jsd)


def upper_triangle_values(matrix: np.ndarray) -> list[float]:
    if matrix.shape[0] <= 1:
        return []
    tri = np.triu_indices(matrix.shape[0], k=1)
    return matrix[tri].astype(float).tolist()


def pairwise_tv_values(distributions: list[dict[str, float]]) -> list[float]:
    if len(distributions) <= 1:
        return []
    arr, _keys = distribution_matrix(distributions)
    if arr.shape[1] == 0:
        return []
    tv = 0.5 * np.abs(arr[:, None, :] - arr[None, :, :]).sum(axis=2)
    return upper_triangle_values(tv)


def pairwise_jsd_values(distributions: list[dict[str, float]]) -> list[float]:
    if len(distributions) <= 1:
        return []
    arr, _keys = distribution_matrix(distributions)
    if arr.shape[1] == 0:
        return []
    return upper_triangle_values(pairwise_jsd_matrix_from_array(arr))


def consensus_jsd_values(distributions: list[dict[str, float]]) -> list[float]:
    if not distributions:
        return []
    arr, _keys = distribution_matrix(distributions)
    if arr.shape[1] == 0:
        return []
    consensus = arr.mean(axis=0)
    total = consensus.sum()
    if total <= 0:
        return []
    consensus = consensus / total
    h = entropy_array(arr)
    h_consensus = float(entropy_array(consensus))
    mid = 0.5 * (arr + consensus[None, :])
    vals = entropy_array(mid) - 0.5 * (h + h_consensus)
    return np.maximum(0.0, vals).astype(float).tolist()


def parse_report_candidates(rec: dict) -> list[dict]:
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
                    print(f"  ignored malformed line {line_i} from {path}")
                    continue
                key = (
                    str(rec.get("model")),
                    str(rec.get("question_id")),
                    int(rec.get("repeat_index", 0)),
                )
                records_by_key[key] = rec

    rows = []
    for rec in records_by_key.values():
        candidates = parse_report_candidates(rec)
        top = top_p_set(candidates, NOMINAL_P)
        top_members = sorted({member_key(c) for c in top.members})
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
        distributions = [report_distribution(candidates) for candidates in group["candidates"]]
        tv_values = pairwise_tv_values(distributions)
        jsd_values = pairwise_jsd_values(distributions)
        consensus_values = consensus_jsd_values(distributions)
        top_keys = [
            member_key({"answer": answer, "grade": grade})
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
                "mean_top_p_jaccard": pairwise_jaccard(member_sets),
                "mean_top_p_set_distance": 1.0 - pairwise_jaccard(member_sets),
                "mean_pairwise_tv": float(np.mean(tv_values)) if tv_values else np.nan,
                "median_pairwise_tv": float(np.median(tv_values)) if tv_values else np.nan,
                "q90_pairwise_tv": float(np.quantile(tv_values, 0.90)) if tv_values else np.nan,
                "mean_pairwise_jsd": float(np.mean(jsd_values)) if jsd_values else np.nan,
                "median_pairwise_jsd": float(np.median(jsd_values)) if jsd_values else np.nan,
                "q90_pairwise_jsd": float(np.quantile(jsd_values, 0.90)) if jsd_values else np.nan,
                "mean_consensus_jsd": (
                    float(np.mean(consensus_values)) if consensus_values else np.nan
                ),
                "median_consensus_jsd": (
                    float(np.median(consensus_values)) if consensus_values else np.nan
                ),
                "q90_consensus_jsd": (
                    float(np.quantile(consensus_values, 0.90)) if consensus_values else np.nan
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


def bootstrap_ecdf_summary(
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


def fig_jsd_ecdf(question_summary: pd.DataFrame, *, random_seed: int = 0) -> None:
    if question_summary.empty:
        print("  skipped Fig. S2: no question-level pairwise JSD summaries")
        return

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
        model_ecdf = bootstrap_ecdf_summary(
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
        print("  skipped Fig. S2: insufficient model-specific JSD summaries")
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
    ax.set_ylabel("cumulative proportion of questions")
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
    savefig(fig, "FS2_JSD")


def fig_consistency(
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
        print("== Fig. S2: skipped (no log consistency run directory configured) ==")
        return False
    frames = []
    for name in run_dir_names:
        try:
            frames.append(load_log_consistency(name))
        except FileNotFoundError as exc:
            print(f"  skipped missing run {exc.filename}")
    if not frames:
        print("== Fig. S2: skipped (no log consistency run files found) ==")
        return False
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        print("== Fig. S2: skipped (log consistency file has no rows) ==")
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
            f"  restricted every model to the first {question_limit} "
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
            f"  restricted every model--question pair to replications "
            f"0--{repeat_limit - 1}"
        )

    print("== Fig. S2: repeated log-elicitation consistency ==")
    summary = summarize_log_consistency(df)
    summary.to_csv(OUT / "FS2_JSD_by_question.csv", index=False)
    fig_jsd_ecdf(summary)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the paper figures, optionally for a subset of models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_ORDER),
        help=f"Model keys or aliases to include (default: all of {', '.join(RUNS)}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT,
        help="Directory for generated figures and CSVs.",
    )
    parser.add_argument(
        "--consistency-runs",
        default=DEFAULT_CONSISTENCY_RUNS,
        help=(
            "Comma-separated run directories with repeated-elicitation results "
            "for Fig. S2, or an empty string to skip it."
        ),
    )
    parser.add_argument(
        "--consistency-question-limit",
        type=int,
        default=None,
        help="Restrict every model to the same first N questions of the SimpleQA order.",
    )
    parser.add_argument(
        "--consistency-repeat-limit",
        type=int,
        default=None,
        help="Restrict every question to replication indices below N.",
    )
    parser.add_argument(
        "--only-consistency",
        action="store_true",
        help="Generate only the Fig. S2 outputs.",
    )
    args = parser.parse_args(argv)
    args.models = parse_model_keys(args.models)
    return args


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    configure_run(model_keys=args.models, out_dir=args.out_dir)
    print("Selected models:", ", ".join(ACTIVE_MODEL_ORDER))
    print("Output directory:", OUT)
    if args.only_consistency:
        wrote = fig_consistency(
            args.consistency_runs,
            question_limit=args.consistency_question_limit,
            repeat_limit=args.consistency_repeat_limit,
        )
        if not wrote:
            raise SystemExit("No consistency figure was generated.")
        return
    paired = load_paired()
    from comparison_uncertainty import main as comparison_intervals
    relative = comparison_intervals(["--outdir", str(OUT)])
    fig_frontier(paired, relative)
    fig_outcome_tables(paired)
    fig_cumulative_rates(paired)
    fig_response_counts(paired)
    fig_consistency(
        args.consistency_runs,
        question_limit=args.consistency_question_limit,
        repeat_limit=args.consistency_repeat_limit,
    )
    print("\nDone. Figures written to", OUT)


if __name__ == "__main__":
    main()
