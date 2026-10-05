"""Map an error tolerance to a simultaneously supported answer threshold.

Standalone retrospective analysis. Does not modify paper text or figures.
See alpha_risk_control_theory.md for theoretical details.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
import platform
import shutil

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "alpha-risk-control-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

import common
import threshold_selection as old
from calibration_decisions import fit_isotonic, processed_records

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260912
DELTA = .05
ORIGINAL_TARGETS = old.TARGETS
ADDITIONAL_TARGETS = (.30, .35, .40)
TARGETS = ORIGINAL_TARGETS + ADDITIONAL_TARGETS
CURVE = np.unique(np.r_[np.arange(10,161)/200, TARGETS])
METHODS = ("fixed_full", "isotonic_grid", "fixed_half")
LABELS = {"fixed_full":"Fixed grid; 2,163 certification questions",
          "isotonic_grid":"Learned grid; 1,082 certification questions",
          "fixed_half":"Fixed grid; 1,082 certification questions",
          "original":"Unvalidated threshold $1-\\alpha$"}
COLORS = {"fixed_full":"#126dae", "isotonic_grid":"#d47d19", "fixed_half":"#7e5597", "original":".55"}


def split(frame, seed):
    full, test = old.split_frames(frame, seed)
    cut = len(full)//2
    return full, full.iloc[:cut], full.iloc[cut:], test


def learned_grid(development):
    fit = fit_isotonic(development.u, development.z)
    jumps = np.flatnonzero(np.diff(fit.fitted)>0)+1
    return np.unique(np.r_[0., fit.x[jumps]])


def build_bounds(certification, grid, delta=DELTA):
    return old.calibrate(certification, grid=grid, delta=delta).rename(
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
    full, dev, cert, test = split(frame, seed)
    grids = {"fixed_full":old.GRID, "isotonic_grid":learned_grid(dev), "fixed_half":old.GRID}
    bands, rows = [], []
    alphas = CURVE if primary else TARGETS
    for method, grid in grids.items():
        data = full if method=="fixed_full" else cert
        bounds = build_bounds(data, grid)
        info = {"method":method, "seed":seed, "grid_size":len(grid),
                "n_certification":len(data), "n_development":len(dev) if method=="isotonic_grid" else 0}
        bands.append(bounds.assign(**info))
        previous = np.inf
        for alpha in sorted(alphas):
            selection = describe_selection(bounds, alpha)
            threshold = selection["threshold"]
            ordered = threshold if threshold is not None else np.inf
            if ordered>previous:
                raise AssertionError("Relaxing alpha cannot increase the selected threshold")
            previous = ordered
            rows.append({**info, **selection, **old.evaluate(test, threshold)})
    for alpha in sorted(alphas):
        rows.append({"method":"original", "seed":seed, "alpha":alpha,
                     "supported":False, "threshold":1-alpha,
                     **old.evaluate(test,1-alpha)})
    predictions = None
    if primary:
        predictions = frame.copy()
        predictions["full_split"] = np.where(predictions.question_id.isin(full.question_id),"certification","test")
        predictions["learned_split"] = np.where(predictions.question_id.isin(dev.question_id),"development",
                                                np.where(predictions.question_id.isin(cert.question_id),"certification","test"))
    return pd.DataFrame(rows), pd.concat(bands,ignore_index=True), predictions


def load_inputs(grader):
    frames, inputs = {}, []
    for key in common.MODEL_ORDER:
        path=common.result_file(ROOT/"results"/f"graded_by_{grader}"/common.RUNS[key]["run"]/"simpleqa_topp_results.jsonl")
        if path is None:
            raise FileNotFoundError(key)
        records=common.load_jsonl(path)
        frames[(key,"raw")]=old.prepare_records(records)[["question_id","u","z"]]
        frames[(key,"processed")]=processed_records(records)
        inputs.append({"file":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    ids=next(iter(frames.values())).question_id.tolist()
    if len(ids)!=4326 or any(f.question_id.tolist()!=ids for f in frames.values()):
        raise ValueError("Expected identical 4,326 question IDs")
    return frames, inputs


def plot_curves(primary, out):
    plt.rcParams.update({"font.size":10,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    for rep in ("raw","processed"):
        fig, axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey="row")
        for col,key in enumerate(common.MODEL_ORDER):
            part=primary[(primary.model_key==key)&(primary.representation==rep)]
            for method in (*METHODS,"original"):
                g=part[part.method==method].sort_values("alpha")
                linestyle="--" if method=="original" else "-"
                axes[0,col].step(100*g.alpha,g.threshold,where="post",color=COLORS[method],ls=linestyle,lw=1.8,label=LABELS[method])
                axes[1,col].step(100*g.alpha,100*g.answer_rate,where="post",color=COLORS[method],ls=linestyle,lw=1.8)
            axes[0,col].set_title(common.RUNS[key]["label"],loc="left",fontsize=11,fontweight="bold")
            axes[0,col].set_ylim(-.02,1.02)
            axes[1,col].set(xlim=(5,80),ylim=(-2,102),xlabel="Allowed error among answers, $\\alpha$ (%)")
            for row in range(2):
                axes[row,col].grid(alpha=.17)
        axes[0,0].set_ylabel("Selected probability cutoff")
        axes[1,0].set_ylabel("Held-out questions answered (%)")
        h,l=axes[0,0].get_legend_handles_labels()
        fig.legend(h,l,loc="lower center",ncol=2,frameon=False)
        description="Raw reports" if rep=="raw" else "Gold-informed processed reports"
        fig.suptitle(f"{description}: error tolerance to supported threshold",fontsize=13)
        fig.text(.5,.902,"Unsupported cutoffs are omitted; their answer rate is zero. Confidence: 95% per method and model.",ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.12,1,.9))
        fig.savefig(out/f"alpha_threshold_{rep}.png",dpi=170,bbox_inches="tight")
        fig.savefig(out/f"alpha_threshold_{rep}.pdf",bbox_inches="tight")
        plt.close(fig)


def run(grader="openai", out=None, repeats=100):
    out=Path(out) if out else ROOT/"results"/"exploratory"/f"alpha_risk_control_{grader}"
    out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(__file__).with_name("alpha_risk_control_theory.md"),out/"theory.md")
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
    primary_bounds=all_bounds[all_bounds.seed==SEED]
    all_results.to_csv(out/"all_results.csv",index=False)
    all_bounds.to_csv(out/"all_bounds.csv",index=False)
    primary.to_csv(out/"alpha_lookup_curve.csv",index=False)
    primary[primary.alpha.isin(TARGETS)].to_csv(out/"primary_results.csv",index=False)
    primary_bounds.to_csv(out/"primary_bounds.csv",index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/"primary_inputs.csv",index=False)
    minimum=[]
    for names,g in primary_bounds.groupby(["model_key","representation","method"]):
        nonempty=g[g.n_cert_answered>0]
        if len(nonempty):
            best=nonempty.sort_values(["risk_upper","threshold"]).iloc[0]
            minimum.append(dict(zip(["model_key","representation","method"],names)) | {
                "minimum_supported_alpha":best.risk_upper,"threshold_at_minimum":best.threshold,
                "n_cert_answered":best.n_cert_answered,"k_cert_incorrect":best.k_cert_incorrect,
                "grid_size":best.grid_size})
    pd.DataFrame(minimum).to_csv(out/"minimum_supported_alpha.csv",index=False)
    repeated=all_results[(all_results.seed!=SEED)&all_results.method.isin(METHODS)]
    summaries=[]
    for names,g in repeated.groupby(["model_key","representation","method","alpha"]):
        selected=g[g.supported]
        answered=g[g.n_answered>0]
        summaries.append(dict(zip(["model_key","representation","method","alpha"],names)) | {
            "n_splits":len(g),"n_supported":int(g.supported.sum()),"mean_answer_rate":g.answer_rate.mean(),
            "median_threshold_when_supported":selected.threshold.median(),
            "median_error_when_answering":answered.observed_error.median(),
            "n_test_error_exceeds_alpha":int((answered.observed_error>answered.alpha).sum()),
            "median_grid_size":g.grid_size.median()})
    pd.DataFrame(summaries).to_csv(out/"split_summary.csv",index=False)
    plot_curves(primary,out)
    source_files=[Path(__file__),Path(__file__).with_name("threshold_selection.py"),
                  Path(__file__).with_name("calibration_decisions.py"),Path(__file__).with_name("common.py"),ROOT/"runner"/"engine.py"]
    meta={"grader":grader,"delta":DELTA,"seed":SEED,"additional_splits":repeats,
          "targets":TARGETS,"original_targets":ORIGINAL_TARGETS,"additional_targets":ADDITIONAL_TARGETS,
          "target_extension":"User-requested 30%, 35%, 40% after initial results; methods and bounds unchanged",
          "curve_alphas":CURVE.tolist(),"fixed_grid":old.GRID.tolist(),
          "methods":METHODS,"primary_method":"fixed_full","primary_representation":"raw",
          "sizes":{"fixed_certification":2163,"development":1081,"split_certification":1082,"test":2163},
          "confidence_scope":"simultaneous thresholds and arbitrary alpha, separately per method, model, representation, and split",
          "design":"retrospective previously examined benchmark; no new external validation",
          "inputs":inputs,"source_hashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
          "software":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__}}
    (out/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(f"Saved {out}",flush=True)


def query(folder,alpha,model=None,representation="raw",method="fixed_full"):
    bounds=pd.read_csv(Path(folder)/"primary_bounds.csv",float_precision="round_trip")
    bounds=bounds[(bounds.representation==representation)&(bounds.method==method)]
    if model:
        bounds=bounds[bounds.model_key==model]
    if bounds.empty:
        raise ValueError("No matching saved bound table")
    return [{"model":key,"representation":representation,"method":method,
             **describe_selection(g,alpha)} for key,g in bounds.groupby("model_key",sort=False)]


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grader",choices=("openai","gemini"),default="openai")
    parser.add_argument("--out-dir",type=Path)
    parser.add_argument("--repeats",type=int,default=100)
    parser.add_argument("--lookup",type=Path,help="Read saved bounds instead of fitting or evaluating")
    parser.add_argument("--alpha",type=float,default=.25)
    parser.add_argument("--model",choices=common.MODEL_ORDER)
    parser.add_argument("--representation",choices=("raw","processed"),default="raw")
    parser.add_argument("--method",choices=METHODS,default="fixed_full")
    args=parser.parse_args()
    if args.lookup:
        print(json.dumps(query(args.lookup,args.alpha,args.model,args.representation,args.method),indent=2,allow_nan=False))
    else:
        if args.repeats<0:
            parser.error("--repeats must be nonnegative")
        run(args.grader,args.out_dir,args.repeats)
