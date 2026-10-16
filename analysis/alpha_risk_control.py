"""Map an error tolerance to a simultaneously supported answer threshold.

Reproduce the fixed-grid threshold-selection results in Tables S9 and S10.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

import common
import threshold_selection as selection

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260912
DELTA = .05
TARGETS = (.20, .25, .30, .35, .40)


def build_bounds(certification, grid, delta=DELTA):
    return selection.calibrate(certification, grid=grid, delta=delta).rename(
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
    calibration, test = selection.split_frames(frame, seed)
    bounds = build_bounds(calibration, selection.GRID)
    info = {"method": "fixed_full", "seed": seed, "grid_size": len(selection.GRID),
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
        rows.append({**info, **chosen, **selection.evaluate(test, threshold)})
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
        frames[(key,"raw")]=selection.prepare_records(records)[["question_id","u","z"]]
        inputs.append({"file":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    ids=next(iter(frames.values())).question_id.tolist()
    if len(ids)!=4326 or any(f.question_id.tolist()!=ids for f in frames.values()):
        raise ValueError("Expected identical 4,326 question IDs")
    return frames, inputs


def run(grader="openai", out=None, repeats=100):
    out=Path(out) if out else ROOT/"results"/"intermediate"/f"alpha_risk_control_{grader}"
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
    source_files=[Path(__file__),Path(__file__).with_name("threshold_selection.py"),
                  Path(__file__).with_name("common.py"),ROOT/"runner"/"engine.py"]
    meta={"grader":grader,"delta":DELTA,"seed":SEED,"additional_splits":repeats,
          "targets":TARGETS,"fixed_grid":selection.GRID.tolist(),
          "methods":["fixed_full"],"primary_method":"fixed_full","primary_representation":"raw",
          "sizes":{"fixed_certification":2163,"test":2163},
          "confidence_scope":"simultaneous thresholds and arbitrary alpha, separately per method, model, representation, and split",
          "design":"retrospective previously examined benchmark; no new external validation",
          "inputs":inputs,"source_hashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
          "software":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__}}
    (out/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(f"Saved {out}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grader",choices=("openai","gemini"),default="openai")
    parser.add_argument("--out-dir",type=Path)
    parser.add_argument("--repeats",type=int,default=100)
    args=parser.parse_args()
    if args.repeats<0:
        parser.error("--repeats must be nonnegative")
    run(args.grader,args.out_dir,args.repeats)
