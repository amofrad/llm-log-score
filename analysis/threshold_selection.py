"""Select report thresholds with simultaneous bounds, then evaluate a held-out half.

Candidate selection uses raw responses and the existing label-free parser.
Stored grades are attached afterward by reconstructing the original grouping;
every group's probability must be conserved and the selected grade unambiguous.
Produces Table S9 and the minimum supported error limits quoted in the paper.
No model or grading API is called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.stats import beta

import common


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runner"))
from engine import (canonical_incorrect_answer_display, canonical_stadium_answer,
                    is_open_not_attempted,
                    open_answer_key, parse_simpleqa_topp_response)

# Fixed before computing this experiment; shared question splits across models.
SEED = 20260912
DELTA = .05
TARGETS = (.20, .25, .30, .35, .40)
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


def build_bounds(certification, grid, delta=DELTA):
    return calibrate(certification, grid=grid, delta=delta).rename(
        columns={"n_calibration":"n_cert_answered", "k_calibration":"k_cert_incorrect"})


def lookup(bounds, alpha):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    valid = bounds[(bounds.n_cert_answered>0)&(bounds.risk_upper<=alpha)]
    if valid.empty:
        return None
    return valid.loc[valid.threshold.idxmin()]


def describe_selection(bounds, alpha):
    row = lookup(bounds, alpha)
    return {"alpha":float(alpha), "supported":row is not None,
            "threshold":float(row.threshold) if row is not None else None,
            "certification_upper":float(row.risk_upper) if row is not None else None,
            "n_cert_answered":int(row.n_cert_answered) if row is not None else 0,
            "k_cert_incorrect":int(row.k_cert_incorrect) if row is not None else 0}


def analyze(frame, seed, primary=False):
    calibration, test = split_frames(frame, seed)
    bounds = build_bounds(calibration, GRID)
    info = {"method": "fixed_full", "seed": seed, "grid_size": len(GRID),
            "n_certification": len(calibration), "n_development": 0}
    rows = []
    previous = np.inf
    for alpha in TARGETS:
        chosen = describe_selection(bounds, alpha)
        threshold = chosen["threshold"]
        ordered = threshold if threshold is not None else np.inf
        if ordered > previous:
            raise AssertionError("Relaxing alpha cannot increase the selected threshold")
        previous = ordered
        rows.append({**info, **chosen, **evaluate(test, threshold)})
    predictions = None
    if primary:
        predictions = frame.copy()
        predictions["full_split"] = np.where(
            predictions.question_id.isin(calibration.question_id), "certification", "test")
    return pd.DataFrame(rows), bounds.assign(**info), predictions


def load_inputs(grader):
    frames, inputs = {}, []
    for key in common.MODEL_ORDER:
        path=common.result_file(ROOT/"results"/f"graded_by_{grader}"/common.RUNS[key]["run"]/"simpleqa_topp_results.jsonl")
        if path is None:
            raise FileNotFoundError(key)
        records=common.load_jsonl(path)
        frames[(key,"raw")]=prepare_records(records)[["question_id","u","z"]]
        inputs.append({"file":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    ids=next(iter(frames.values())).question_id.tolist()
    if len(ids)!=4326 or any(f.question_id.tolist()!=ids for f in frames.values()):
        raise ValueError("Expected identical 4,326 question IDs")
    return frames, inputs


def run_analysis(grader="openai", out=None, repeats=100):
    out=Path(out) if out else ROOT/"results"/"intermediate"/f"threshold_selection_{grader}"
    out.mkdir(parents=True,exist_ok=True)
    frames,inputs=load_inputs(grader)
    results,bands,predictions=[],[],[]
    for (key,rep),frame in frames.items():
        for offset in range(repeats+1):
            rows,bound,pred=analyze(frame,SEED+offset,primary=offset==0)
            info={"model_key":key,"representation":rep}
            results.append(rows.assign(**info))
            # Every split's family and bounds are retained, not only successful ones.
            bands.append(bound.assign(**info))
            if offset==0:
                predictions.append(pred.assign(**info))
        print(f"{grader}: {key} / {rep} completed {repeats+1} splits",flush=True)
    all_results=pd.concat(results,ignore_index=True)
    all_bounds=pd.concat(bands,ignore_index=True)
    primary=all_results[all_results.seed==SEED]
    all_results.to_csv(out/"all_results.csv",index=False)
    all_bounds.to_csv(out/"all_bounds.csv",index=False)
    primary[primary.alpha.isin(TARGETS)].to_csv(out/"primary_results.csv",index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/"primary_inputs.csv",index=False)
    source_files=[Path(__file__),Path(__file__).with_name("common.py"),ROOT/"runner"/"engine.py"]
    meta={"grader":grader,"delta":DELTA,"seed":SEED,"additional_splits":repeats,
          "targets":TARGETS,"fixed_grid":GRID.tolist(),
          "methods":["fixed_full"],"primary_method":"fixed_full","primary_representation":"raw",
          "sizes":{"fixed_certification":2163,"test":2163},
          "confidence_scope":"simultaneous thresholds and arbitrary alpha, separately per method, model, representation, and split",
          "design":"retrospective previously examined benchmark; no new external validation",
          "inputs":inputs,"source_hashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
          "software":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__}}
    (out/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(f"Saved {out}",flush=True)



def _primary_row(frame, model, alpha, method):
    rows = frame[(frame.model_key == model) & (frame.representation == "raw")
                 & (frame.method == method) & np.isclose(frame.alpha, alpha, rtol=0, atol=1e-14)]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {model}, {alpha}, {method}; found {len(rows)}")
    return rows.iloc[0]


def analysis_complete(folder):
    return all((folder / name).is_file() for name in (
        "primary_results.csv", "primary_inputs.csv", "all_results.csv", "all_bounds.csv", "metadata.json"))


def run(out_dir=None, recompute=False, primary_grader="openai", analysis_dir=None):
    out_dir = Path(out_dir) if out_dir else ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = Path(analysis_dir) if analysis_dir else ROOT / "results/intermediate"
    minimum_bounds = []
    for grader in ("openai", "gemini"):
        folder = analysis_dir / f"threshold_selection_{grader}"
        if recompute or not analysis_complete(folder):
            run_analysis(grader, folder, repeats=100)
        # Minimum bounds quoted in the SI discussion, including nonqualifying cases.
        bounds = pd.read_csv(folder / "all_bounds.csv", float_precision="round_trip")
        primary_seed = pd.read_csv(folder / "primary_results.csv", usecols=["seed"]).seed.unique()
        assert len(primary_seed) == 1
        bounds = bounds[(bounds.seed == primary_seed[0]) & (bounds.n_cert_answered > 0)]
        for model, group in bounds.groupby("model_key", sort=False):
            best = group.sort_values(["risk_upper", "threshold"]).iloc[0]
            minimum_bounds.append({"grader": grader, "model_key": model,
                                   "minimum_supported_alpha": best.risk_upper,
                                   "threshold": best.threshold,
                                   "n_cert_answered": best.n_cert_answered,
                                   "k_cert_incorrect": best.k_cert_incorrect})
    pd.DataFrame(minimum_bounds).to_csv(out_dir / "threshold_selection_summary.csv", index=False)
    primary = pd.read_csv(analysis_dir / f"threshold_selection_{primary_grader}" / "primary_results.csv",
                          float_precision="round_trip")
    records = []
    for model in common.MODEL_ORDER:
        for alpha in TARGETS:
            row = _primary_row(primary, model, alpha, "fixed_full")
            records.append(row.to_dict() | {"grader": primary_grader})
    pd.DataFrame(records).to_csv(out_dir / "TableS9_RiskControl.csv", index=False)
    print(f"Wrote SI Table S9 to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/figures")
    parser.add_argument("--grader", choices=("openai", "gemini"), default="openai",
                        help="Grader displayed in Table S9.")
    parser.add_argument("--analysis-dir", type=Path,
                        help="Keep intermediate results here; otherwise use a temporary directory.")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute existing intermediate results.")
    args = parser.parse_args()
    if args.analysis_dir is None:
        with tempfile.TemporaryDirectory(prefix="rbd-threshold-") as temporary:
            run(args.out_dir, recompute=True, primary_grader=args.grader,
                analysis_dir=Path(temporary))
    else:
        run(args.out_dir, recompute=args.recompute, primary_grader=args.grader,
            analysis_dir=args.analysis_dir)


if __name__ == "__main__":
    main()
