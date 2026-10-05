"""Retrospective, label-free report features with learned selective decisions.

Runs locally on existing reports and grades; never calls a model or grader API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
import platform

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rich-decision-rule-mpl"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import binom
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import common
import threshold_selection as selection

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026091201
REPEATS = 100
FEATURES = ("u", "idk", "margin", "log_candidates", "listed_total",
            "normalized_entropy", "top_share")
METHODS = ("raw_u", "u_boost", "rich_logistic", "rich_boost")
LABELS = {"raw_u": "Raw top probability", "u_boost": "Top probability: boosting",
          "rich_logistic": "Full report: logistic", "rich_boost": "Full report: boosting"}
COLORS = {"raw_u": "#747474", "u_boost": "#9467bd",
          "rich_logistic": "#d98a22", "rich_boost": "#1675b9"}
BOOST_SETTINGS = dict(max_iter=150, learning_rate=.05, max_leaf_nodes=7,
                      max_depth=3, min_samples_leaf=40, l2_regularization=10.,
                      early_stopping=False)


def report_features(response: str, top_p: float = .9) -> dict:
    """Only the raw response enters; labels, question, and answer key cannot enter."""
    parsed, top, status = selection.raw_selection(response, top_p)
    concrete = [c for c in parsed if not selection.is_open_not_attempted(c["answer"])]
    ps = np.asarray([c["probability"] for c in parsed], dtype=float)
    if np.any(~np.isfinite(ps)) or np.any((ps < 0) | (ps > 1)):
        raise ValueError("Invalid parsed probabilities")
    total = float(ps.sum())
    if total > 1 + 1e-8:
        raise ValueError("Parsed total exceeds one")
    cps = sorted([float(c["probability"]) for c in concrete], reverse=True)
    u = float(parsed[top]["probability"]) if top is not None else -1.
    idk = sum(c["probability"] for c in parsed
              if selection.is_open_not_attempted(c["answer"]))
    positive = ps[ps > 0]
    entropy = 0.
    if len(positive) > 1:
        normalized = positive / total
        entropy = float(-np.sum(normalized * np.log(normalized)) / np.log(len(positive)))
    return {"u": u, "idk": float(idk),
            "margin": u - (cps[1] if len(cps) > 1 else 0.) if cps else 0.,
            "log_candidates": float(np.log1p(len(cps))), "listed_total": total,
            "normalized_entropy": entropy,
            "top_share": u / sum(cps) if cps and sum(cps) > 0 else 0.,
            "eligible": top is not None, "parse_status": status}


def prepare(records: list[dict]) -> pd.DataFrame:
    features = pd.DataFrame([{"question_id": str(r["question_id"]),
                              **report_features(r["raw_log_response"],
                                                float(r.get("simpleqa_top_p", .9)))}
                             for r in records])
    # Correctness is attached only after all raw features have been extracted.
    labels = selection.prepare_records(records)
    result = features.merge(labels[["question_id", "u", "z", "grade"]],
                            on="question_id", validate="one_to_one", suffixes=("", "_check"))
    np.testing.assert_array_equal(result.u, result.u_check)
    return result.drop(columns="u_check").sort_values("question_id").reset_index(drop=True)


def split_indices(frame: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    if frame.question_id.duplicated().any():
        raise ValueError("Duplicate question IDs")
    sorted_positions = np.argsort(frame.question_id.to_numpy(), kind="stable")
    order = sorted_positions[np.random.default_rng(seed).permutation(len(frame))]
    first, second = len(frame) // 2, len(frame) // 2 + len(frame) // 4
    return {"development": order[:first], "certification": order[first:second],
            "test": order[second:]}


def make_estimator(method: str, seed: int):
    if method == "rich_logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000,
                                                                  random_state=seed))
    if method in ("u_boost", "rich_boost"):
        return HistGradientBoostingClassifier(**BOOST_SETTINGS, random_state=seed)
    raise ValueError(method)


def fit_scores(frame: pd.DataFrame, development: np.ndarray, seed: int):
    scores, oof, estimators = {}, {}, {}
    valid = frame.eligible.to_numpy(dtype=bool)
    dev_valid = development[valid[development]]
    z = frame.z.to_numpy(dtype=int)
    fold = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(dev_valid, z[dev_valid]))
    for method in METHODS:
        if method == "raw_u":
            scores[method] = frame.u.to_numpy().copy()
            oof[method] = scores[method][development].copy()
            continue
        columns = ["u"] if method == "u_boost" else list(FEATURES)
        x = frame[columns].to_numpy(dtype=float)
        out = np.full(len(frame), -1.)
        for train, validation in fold:
            model = make_estimator(method, seed)
            model.fit(x[dev_valid[train]], z[dev_valid[train]])
            out[dev_valid[validation]] = model.predict_proba(x[dev_valid[validation]])[:, 1]
        oof[method] = out[development]
        model = make_estimator(method, seed)
        model.fit(x[dev_valid], z[dev_valid])
        prediction = np.full(len(frame), -1.)
        prediction[valid] = model.predict_proba(x[valid])[:, 1]
        scores[method], estimators[method] = prediction, model
    return scores, oof, estimators


def counts(score, z, grid=selection.GRID):
    keep = np.asarray(score)[:, None] >= np.asarray(grid)[None, :]
    return keep.sum(axis=0), (keep * (1 - np.asarray(z))[:, None]).sum(axis=0)


def learn_order(score, z, alpha, grid=selection.GRID):
    n, k = counts(score, z, grid)
    p = binom.cdf(k, n, alpha)
    # Entire order is frozen before looking at certification questions.
    return np.lexsort((np.asarray(grid), -n, p))


def fixed_sequence(score, z, order, alpha, tail, grid=selection.GRID):
    n, k = counts(score, z, grid)
    bounds = selection.upper_binomial(k, n, tail)
    passed, visited = [], []
    for index in order:
        visited.append(int(index))
        if bounds[index] > alpha or n[index] == 0:
            break
        passed.append(int(index))
    chosen = min(passed, key=lambda j: grid[j]) if passed else None
    return chosen, n, k, bounds, visited, passed


def grouped(score, z):
    score, z = np.asarray(score), np.asarray(z)
    levels, inverse = np.unique(-score, return_inverse=True)
    n = np.bincount(inverse, minlength=len(levels))
    errors = np.bincount(inverse, weights=1-z, minlength=len(levels))
    return n, errors


def aurc(score, z):
    """Expected discrete AURC over uniformly random permutations within ties."""
    n, k = grouped(score, z)
    total = int(n.sum())
    if total == 0:
        return np.nan
    end_n, end_k = np.cumsum(n), np.cumsum(k)
    start_n, start_k = end_n - n, end_k - k
    h = np.r_[0., np.cumsum(1 / np.arange(1, total + 1))]
    rate = k / n
    value = np.sum(k + (start_k - rate * start_n) * (h[end_n] - h[start_n])) / total
    return float(value)


def coverage_error(score, z, answer_rate):
    score, z = np.asarray(score), np.asarray(z)
    eligible = score >= 0
    requested = answer_rate * len(score)
    if requested <= 0 or requested > eligible.sum():
        return np.nan
    n, k = grouped(score[eligible], z[eligible])
    error_count = np.interp(requested, np.r_[0, np.cumsum(n)], np.r_[0, np.cumsum(k)])
    return float(error_count / requested)


def metrics(score, z):
    valid = np.asarray(score) >= 0
    s, y = np.asarray(score)[valid], np.asarray(z)[valid]
    return {"n_test": len(score), "n_eligible": int(valid.sum()), "aurc": aurc(s, y),
            "auroc": roc_auc_score(y, s), "brier": brier_score_loss(y, s),
            "log_loss": log_loss(y, np.clip(s, 1e-12, 1-1e-12), labels=[0, 1]),
            **{f"error_at_{int(100*c)}pct": coverage_error(score, z, c) for c in [.1, .2, .4]}}


def evaluate(score, z, threshold):
    return selection.evaluate(pd.DataFrame({"u": score, "z": z}), threshold)


def analyze(frame, seed):
    split = split_indices(frame, seed)
    dev, cal, test = [split[s] for s in ("development", "certification", "test")]
    scores, oof, estimators = fit_scores(frame, dev, seed)
    z = frame.z.to_numpy(dtype=int)
    ranking, decisions, traces = [], [], []
    for method in METHODS:
        score = scores[method]
        ranking.append({"seed": seed, "method": method, **metrics(score[test], z[test])})
        for alpha in selection.TARGETS:
            order = learn_order(oof[method], z[dev], alpha)
            chosen, n, k, bounds, visited, passed = fixed_sequence(
                score[cal], z[cal], order, alpha, selection.DELTA / len(selection.TARGETS))
            for procedure in ("fixed_sequence", "bonferroni"):
                if procedure == "fixed_sequence":
                    index, b = chosen, bounds
                else:
                    b = selection.upper_binomial(k, n, selection.DELTA / len(selection.GRID))
                    qualifying = np.flatnonzero((b <= alpha) & (n > 0))
                    index = int(qualifying[0]) if len(qualifying) else None
                threshold = float(selection.GRID[index]) if index is not None else None
                decisions.append({"seed": seed, "method": method, "procedure": procedure,
                                  "alpha": alpha, "threshold": threshold, "supported": index is not None,
                                  "n_calibration_answered": int(n[index]) if index is not None else 0,
                                  "k_calibration_incorrect": int(k[index]) if index is not None else 0,
                                  "calibration_upper": float(b[index]) if index is not None else np.nan,
                                  **evaluate(score[test], z[test], threshold)})
            if seed == SEED:
                dev_n, dev_k = counts(oof[method], z[dev])
                order_position = np.argsort(order)
                traces.extend({"method": method, "alpha": alpha, "threshold": float(t),
                               "order_position": int(order_position[j]),
                               "n_development_oof": int(dev_n[j]), "k_development_oof": int(dev_k[j]),
                               "n_calibration": int(n[j]), "k_calibration": int(k[j]),
                               "risk_upper": float(bounds[j]), "visited": j in visited, "passed": j in passed}
                              for j, t in enumerate(selection.GRID))
    return pd.DataFrame(ranking), pd.DataFrame(decisions), pd.DataFrame(traces), scores, split, estimators


def bootstrap_primary(score_a, score_b, z, seed, draws=2000):
    valid = (np.asarray(score_a) >= 0) & (np.asarray(score_b) >= 0)
    a, b, y = np.asarray(score_a)[valid], np.asarray(score_b)[valid], np.asarray(z)[valid]
    point = aurc(a, y) - aurc(b, y)
    rng = np.random.default_rng(seed)
    differences = np.empty(draws)
    for j in range(draws):
        idx = rng.integers(0, len(y), len(y))
        differences[j] = aurc(a[idx], y[idx]) - aurc(b[idx], y[idx])
    tail = .05 / len(common.MODEL_ORDER) / 2
    lo, hi = np.quantile(differences, [tail, 1-tail])
    return {"contrast": "rich_boost_minus_u_boost", "aurc_difference": point,
            "lower": lo, "upper": hi, "interval_level": 1-2*tail, "bootstrap_draws": draws}


def summarize(ranking, decisions):
    summaries, contrasts = [], []
    for (key, method, procedure, alpha), g in decisions.groupby(["model_key", "method", "procedure", "alpha"]):
        good = g[g.supported]
        summaries.append({"model_key": key, "method": method, "procedure": procedure, "alpha": alpha,
                          "n_splits": len(g), "n_supported": int(g.supported.sum()),
                          "supported_fraction": g.supported.mean(),
                          "mean_answer_rate_all_splits": g.answer_rate.mean(),
                          "median_answer_rate_all_splits": g.answer_rate.median(),
                          "median_error_supported": good.observed_error.median(),
                          "median_answer_rate_supported": good.answer_rate.median(),
                          "test_error_above_target_supported": int((good.observed_error > alpha).sum())})
    for (key, seed), g in ranking.groupby(["model_key", "seed"]):
        g = g.set_index("method")
        for comparator in ("raw_u", "u_boost", "rich_logistic"):
            contrasts.append({"model_key": key, "seed": seed, "comparator": comparator,
                              **{metric+"_difference": g.loc["rich_boost", metric]-g.loc[comparator, metric]
                                 for metric in ("aurc", "auroc", "error_at_10pct", "error_at_20pct", "error_at_40pct")}})
    return pd.DataFrame(summaries), pd.DataFrame(contrasts)


def plot_primary(predictions, target):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharex=True, sharey=True)
    for ax, key in zip(axes, common.MODEL_ORDER):
        g = predictions[(predictions.model_key == key) & (predictions.split == "test")]
        for method in METHODS:
            coverage = np.arange(1, 101) / 100
            risk = [coverage_error(g[method], g.z, c) for c in coverage]
            ax.plot(100*coverage, 100*np.asarray(risk), color=COLORS[method],
                    label=LABELS[method], lw=2, ls="--" if method == "u_boost" else "-")
        ax.set_title(common.RUNS[key]["label"], loc="left", fontsize=11, fontweight="bold")
        ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Questions answered (%)")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Error among answered questions (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Richer decision rules: primary test split (1,082 questions)", fontsize=13)
    fig.tight_layout(rect=(0, .15, 1, .95))
    fig.savefig(target.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(grader="openai", out_dir=None, repeats=REPEATS):
    out = Path(out_dir) if out_dir else ROOT / "results" / "exploratory" / f"rich_decision_rule_{grader}"
    out.mkdir(parents=True, exist_ok=True)
    tree = ROOT / "results" / f"graded_by_{grader}"
    frames, inputs = {}, []
    for key in common.MODEL_ORDER:
        path = common.result_file(tree / common.RUNS[key]["run"] / "simpleqa_topp_results.jsonl")
        frames[key] = prepare(common.load_jsonl(path))
        inputs.append({"model_key": key, "file": str(path.relative_to(ROOT)),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    ids = frames[common.MODEL_ORDER[0]].question_id.tolist()
    if len(ids) != 4326 or any(f.question_id.tolist() != ids for f in frames.values()):
        raise ValueError("Expected the same 4,326 question IDs in all three models")
    pd.concat([f.assign(model_key=k) for k, f in frames.items()]).to_csv(out / "features.csv", index=False)
    ranks, decisions, traces, predictions, bootstrap = [], [], [], [], []
    with threadpool_limits(limits=1):
        for key, frame in frames.items():
            for offset in range(repeats + 1):
                seed = SEED + offset
                rank, decision, trace, scores, split, estimators = analyze(frame, seed)
                ranks.append(rank.assign(model_key=key))
                decisions.append(decision.assign(model_key=key))
                if offset == 0:
                    traces.append(trace.assign(model_key=key))
                    pred = frame.copy()
                    pred["split"] = ""
                    for name, ix in split.items():
                        pred.loc[ix, "split"] = name
                    for method, score in scores.items():
                        pred[method] = score
                    predictions.append(pred.assign(model_key=key))
                    test = split["test"]
                    bootstrap.append({"model_key": key, **bootstrap_primary(
                        scores["rich_boost"][test], scores["u_boost"][test],
                        frame.z.to_numpy()[test], SEED + 1000)})
                    joblib.dump({"estimators": estimators, "features": FEATURES,
                                 "seed": seed, "development_ids": frame.iloc[split["development"]].question_id.tolist()},
                                out / f"fitted_{key}.joblib")
                if offset % 10 == 0 or offset == repeats:
                    print(f"{key}: completed split {offset}/{repeats}", flush=True)
            # Save progress at each completed model, without changing paper artifacts.
            pd.concat(ranks).to_csv(out / "ranking_all_splits.csv", index=False)
            pd.concat(decisions).to_csv(out / "decisions_all_splits.csv", index=False)
    ranking, decision = pd.concat(ranks), pd.concat(decisions)
    prediction = pd.concat(predictions)
    ranking[ranking.seed == SEED].to_csv(out / "ranking_primary.csv", index=False)
    decision[decision.seed == SEED].to_csv(out / "decisions_primary.csv", index=False)
    prediction.to_csv(out / "predictions_primary.csv", index=False)
    pd.concat(traces).to_csv(out / "testing_traces_primary.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(out / "aurc_paired_bootstrap.csv", index=False)
    repeated_rank = ranking[ranking.seed != SEED] if repeats else ranking
    repeated_decision = decision[decision.seed != SEED] if repeats else decision
    summary, contrasts = summarize(repeated_rank, repeated_decision)
    summary.to_csv(out / "certification_split_summary.csv", index=False)
    contrasts.to_csv(out / "ranking_paired_split_differences.csv", index=False)
    plot_primary(prediction, out / "risk_coverage_primary")
    metadata = {"design": "retrospective benchmark exploration; no external validation",
                "primary_seed": SEED, "additional_splits": repeats,
                "split_n": {name: len(ix) for name, ix in split.items()},
                "features": FEATURES, "methods": METHODS, "primary_method": "rich_boost",
                "boosting_settings": BOOST_SETTINGS, "logistic_C": 1., "oof_folds": 3,
                "targets": selection.TARGETS, "delta": selection.DELTA,
                "fixed_sequence_tail": selection.DELTA / len(selection.TARGETS),
                "confidence_scope": "joint targets, separately per model, fixed rule, and split",
                "grid": selection.GRID.tolist(), "inputs": inputs,
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "software": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved all results to {out}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grader", choices=("openai", "gemini"), default="openai")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()
    if args.repeats < 0:
        parser.error("--repeats must be nonnegative")
    run(args.grader, args.out_dir, args.repeats)
