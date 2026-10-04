"""Select report thresholds with simultaneous bounds, then evaluate a held-out half.

Candidate selection uses raw responses and the existing label-free parser.
Stored grades are attached afterward by reconstructing the original grouping;
every group's probability must be conserved and the selected grade unambiguous.
No model or grading API is called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta

import common

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runner"))
from engine import (canonical_incorrect_answer_display, canonical_stadium_answer,
                    is_open_not_attempted,
                    open_answer_key, parse_simpleqa_topp_response)

# Fixed before computing this experiment; shared question splits across models.
SPLIT_SEED = 20260912
REPEATED_SPLITS = 100
DELTA = .05
TARGETS = (.10, 1 / 7, .20, .25)
GRID = np.unique(np.r_[np.arange(101) / 100, 6 / 7])


def raw_selection(response: str, top_p: float = .9):
    """Return parsed entries, selected index, and parser status; no labels enter."""
    parsed, status = parse_simpleqa_topp_response(response, top_p=top_p)
    parsed = parsed or []
    eligible = [i for i, c in enumerate(parsed)
                if not is_open_not_attempted(c["answer"])]
    top = max(eligible, key=lambda i: parsed[i]["probability"]) if eligible else None
    return parsed, top, status


def recover_top_grade(record: dict, parsed: list[dict], top: int) -> str:
    """Recover the selected entry's stored grade, never its selection score.

    Original grouping could replace a correct answer with the gold string and
    remove single-entry provenance. Resolve this using allowed display mappings
    and conservation of each stored group's probability. Reject any ambiguity in
    the selected grade instead of silently guessing or dropping a question.
    """
    graded = record["log_candidates_json"]
    graded = json.loads(graded) if isinstance(graded, str) else graded
    tol = 1e-8
    gold = open_answer_key(record["gold_answer"])
    capacity = np.array([float(g["probability"]) for g in graded])
    options = []
    for c in parsed:
        answer = c["answer"]
        key = open_answer_key(answer)
        displays = {key, open_answer_key(canonical_incorrect_answer_display(answer))}
        question = record["question"].casefold()
        if "stadium" in question and "world cup" in question:
            stadium = canonical_stadium_answer(answer)
            if stadium:
                displays.add(open_answer_key(stadium[1]))
        possible = []
        for j, g in enumerate(graded):
            sources = g.get("source_answers", [])
            display = open_answer_key(g["answer"])
            if is_open_not_attempted(answer):
                matches = is_open_not_attempted(g["answer"])
            elif sources:
                matches = key in {open_answer_key(a) for a in sources}
            else:
                matches = (display in displays or display == gold
                           or is_open_not_attempted(g["answer"]))
            if matches and c["probability"] <= capacity[j] + tol:
                possible.append(j)
        options.append(possible)
    order = sorted(range(len(parsed)), key=lambda i: (len(options[i]), -parsed[i]["probability"]))
    labels = set()
    assignment = {}

    def search(depth):
        if len(labels) > 1:
            return
        if depth == len(order):
            if np.max(np.abs(capacity), initial=0) <= tol:
                labels.add(graded[assignment[top]]["grade"])
            return
        i = order[depth]
        probability = parsed[i]["probability"]
        for j in options[i]:
            if capacity[j] + tol < probability:
                continue
            capacity[j] -= probability
            assignment[i] = j
            search(depth + 1)
            capacity[j] += probability

    search(0)
    if len(labels) != 1:
        raise ValueError(f"{record['question_id']}: unresolved raw-entry grade: {labels}")
    return labels.pop()


def prepare_records(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        parsed, top, status = raw_selection(record["raw_log_response"],
                                             float(record.get("simpleqa_top_p", .9)))
        u, z, answer, grade = -1., 0, "", "empty"
        if top is not None:
            c = parsed[top]
            u, answer = float(c["probability"]), c["answer"]
            if not np.isfinite(u) or not 0 <= u <= 1:
                raise ValueError(f"Invalid probability for {record['question_id']}: {u}")
            grade = recover_top_grade(record, parsed, top)
            if grade not in {"correct", "incorrect", "not_attempted"}:
                raise ValueError(f"Unexpected grade: {grade}")
            z = int(grade == "correct")
        rows.append({"question_id": str(record["question_id"]), "u": u, "z": z,
                     "answer": answer, "grade": grade, "parse_status": status})
    frame = pd.DataFrame(rows).sort_values("question_id").reset_index(drop=True)
    if frame.question_id.duplicated().any():
        raise ValueError("Expected exactly one report per question")
    return frame


def upper_binomial(k, n, tail: float):
    """One-sided exact binomial bound; zero answers have bound one."""
    k, n = np.broadcast_arrays(np.asarray(k), np.asarray(n))
    if (not 0 < tail < 1 or np.any(k < 0) or np.any(k > n)
            or np.any(n != np.floor(n)) or np.any(k != np.floor(k))):
        raise ValueError("Invalid binomial bound inputs")
    bound = np.ones(k.shape, dtype=float)
    mask = (n > 0) & (k < n)
    bound[mask] = beta.ppf(1 - tail, k[mask] + 1, n[mask] - k[mask])
    return bound


def counts_at(frame: pd.DataFrame, thresholds: np.ndarray):
    selected = frame.u.to_numpy()[:, None] >= thresholds[None, :]
    n = selected.sum(axis=0)
    k = ((1 - frame.z.to_numpy())[:, None] * selected).sum(axis=0)
    return n, k


def calibrate(frame: pd.DataFrame, grid=GRID, delta=DELTA) -> pd.DataFrame:
    grid = np.asarray(grid)
    if len(grid) == 0 or np.any(np.diff(grid) <= 0) or np.any((grid < 0) | (grid > 1)):
        raise ValueError("Threshold grid must be sorted, distinct, and in [0,1]")
    n, k = counts_at(frame, grid)
    return pd.DataFrame({"threshold": grid, "n_calibration": n, "k_calibration": k,
                         "risk_upper": upper_binomial(k, n, delta / len(grid))})


def choose_threshold(bounds: pd.DataFrame, alpha: float) -> pd.Series | None:
    accepted = bounds[(bounds.risk_upper <= alpha) & (bounds.n_calibration > 0)]
    return None if accepted.empty else accepted.loc[accepted.threshold.idxmin()]


def evaluate(frame: pd.DataFrame, threshold: float | None) -> dict:
    if threshold is None:
        n, k = 0, 0
    else:
        nn, kk = counts_at(frame, np.array([threshold]))
        n, k = int(nn[0]), int(kk[0])
    lo = 0. if k == 0 else float(beta.ppf(.025, k, n - k + 1))
    hi = 1. if k == n else float(beta.ppf(.975, k + 1, n - k))
    return {"n_test": len(frame), "n_answered": n, "n_incorrect": k,
            "answer_rate": n / len(frame), "observed_error": k / n if n else np.nan,
            "error_lo": lo if n else np.nan, "error_hi": hi if n else np.nan}


def split_frames(frame: pd.DataFrame, seed: int):
    # Sorting makes membership independent of file order. All models use same IDs.
    frame = frame.sort_values("question_id").reset_index(drop=True)
    order = np.random.default_rng(seed).permutation(len(frame))
    cut = len(frame) // 2
    return frame.iloc[order[:cut]], frame.iloc[order[cut:]]


def analyze_split(frame, seed, *, include_baseline=True):
    calibration, test = split_frames(frame, seed)
    bounds = calibrate(calibration)
    rows = []
    for alpha in TARGETS:
        selection = choose_threshold(bounds, alpha)
        threshold = None if selection is None else float(selection.threshold)
        rows.append({"seed": seed, "alpha": alpha, "method": "selected",
                     "threshold": threshold, "supported": selection is not None,
                     "n_calibration_answered": 0 if selection is None else int(selection.n_calibration),
                     "k_calibration_incorrect": 0 if selection is None else int(selection.k_calibration),
                     "calibration_upper": np.nan if selection is None else float(selection.risk_upper),
                     **evaluate(test, threshold)})
        if include_baseline:
            rows.append({"seed": seed, "alpha": alpha, "method": "fixed",
                         "threshold": 1 - alpha, "supported": False,
                         "n_calibration_answered": np.nan, "k_calibration_incorrect": np.nan,
                         "calibration_upper": np.nan, **evaluate(test, 1 - alpha)})
    return pd.DataFrame(rows), bounds


def latex_table(data: pd.DataFrame) -> str:
    lines = []
    for key in common.MODEL_ORDER:
        part = data[data.model_key == key]
        for index, alpha in enumerate(TARGETS):
            pair = part[np.isclose(part.alpha, alpha)]
            fixed = pair[pair.method == "fixed"].iloc[0]
            selected = pair[pair.method == "selected"].iloc[0]
            name = common.RUNS[key]["label"] if index == 0 else ""
            a = "$1/7$" if alpha == 1/7 else f"${alpha:g}$"
            if selected.supported:
                chosen = f"{selected.threshold:.2f}"
                error = f"{100*selected.observed_error:.1f} [{100*selected.error_lo:.1f}, {100*selected.error_hi:.1f}]"
            else:
                chosen, error = "---", "---"
            lines.append(f"{name} & {a} & {int(fixed.n_answered):,} & {100*fixed.observed_error:.1f} & {chosen} & {int(selected.n_answered):,} & {error} \\\\")
        lines.append(r"\addlinespace")
    return "\n".join(lines) + "\n"


def run(results_dir: Path, out_dir: Path, *, repeats=REPEATED_SPLITS):
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = []
    frames = {}
    for key in common.MODEL_ORDER:
        path = common.result_file(results_dir / common.RUNS[key]["run"] / "simpleqa_topp_results.jsonl")
        if path is None:
            raise FileNotFoundError(key)
        frames[key] = prepare_records(common.load_jsonl(path))
        inputs.append({"model_key": key, "file": str(path.relative_to(results_dir)),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    ids = frames[common.MODEL_ORDER[0]].question_id.tolist()
    if len(ids) != 4326 or any(f.question_id.tolist() != ids for f in frames.values()):
        raise ValueError("Expected the same 4,326 question IDs for every model")
    primary, sweeps, repeated, predictions = [], [], [], []
    for key, frame in frames.items():
        result, bounds = analyze_split(frame, SPLIT_SEED)
        primary.append(result.assign(model_key=key, model=common.RUNS[key]["label"]))
        sweeps.append(bounds.assign(model_key=key))
        cal, _ = split_frames(frame, SPLIT_SEED)
        predictions.append(frame.assign(model_key=key, split=np.where(frame.index.isin(cal.index), "calibration", "test")))
        for offset in range(repeats):
            result, _ = analyze_split(frame, SPLIT_SEED + offset + 1)
            repeated.append(result.assign(model_key=key))
        print(f"Threshold selection: completed {key}", flush=True)
    primary = pd.concat(primary, ignore_index=True)
    primary.to_csv(out_dir / "ThresholdSelectionLegacy.csv", index=False)
    (out_dir / "ThresholdSelectionLegacy_rows.tex").write_text(latex_table(primary))
    pd.concat(sweeps).to_csv(out_dir / "threshold_selection_bounds.csv", index=False)
    pd.concat(predictions).to_csv(out_dir / "threshold_selection_predictions.csv", index=False)
    if repeated:
        repeated = pd.concat(repeated, ignore_index=True)
        repeated.to_csv(out_dir / "threshold_selection_repeated_splits.csv", index=False)
        rows = []
        for (key, alpha), group in repeated[repeated.method == "selected"].groupby(["model_key", "alpha"]):
            successful = group[group.supported]
            row = {"model_key": key, "alpha": alpha, "n_splits": len(group),
                   "n_supported": len(successful), "supported_fraction": group.supported.mean()}
            for metric in ["threshold", "answer_rate", "observed_error"]:
                row.update({metric+"_median": successful[metric].median(),
                            metric+"_q25": successful[metric].quantile(.25),
                            metric+"_q75": successful[metric].quantile(.75)})
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "threshold_selection_split_summary.csv", index=False)
    metadata = {"results_tree": str(results_dir.resolve().relative_to(common.REPO_ROOT)) if results_dir.resolve().is_relative_to(common.REPO_ROOT) else str(results_dir.resolve()),
                "split_seed": SPLIT_SEED, "calibration_n": 2163, "test_n": 2163,
                "repeated_splits": repeats, "repeated_seeds": [SPLIT_SEED+1, SPLIT_SEED+repeats],
                "delta": DELTA, "targets": TARGETS, "grid": GRID.tolist(),
                "bound": "one-sided Clopper-Pearson with tail delta / grid size",
                "confidence_scope": "simultaneous thresholds and targets, separately per model and split",
                "selection": "smallest qualifying grid threshold; otherwise abstain on every question",
                "processing": "raw response; existing label-free parser; fixed non-answer strings; parsed-order ties",
                "evaluation": "grades attached after candidate selection; exact binomial test intervals",
                "design": "retrospective benchmark split; settings fixed before computing this experiment",
                "inputs": inputs}
    (out_dir / "threshold_selection_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return primary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grader", choices=("openai", "gemini"), default="openai")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out = args.out_dir or ROOT / "results" / ("figures" if args.grader == "openai" else "figures_gemini")
    run(ROOT / "results" / f"graded_by_{args.grader}", out)


if __name__ == "__main__":
    main()
