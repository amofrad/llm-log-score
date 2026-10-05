"""Develop calibration and empirical score optimization as threshold decisions.

Standalone retrospective analysis; does not change manuscript figures or text.
See calibration_decisions_theory.md for theoretical details.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
import tempfile
from pathlib import Path
import platform
import shutil

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "calibration-decisions-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq, minimize
from scipy.special import expit

import common
import threshold_selection as prior

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260912
REPEATS = 100
PENALTIES = (0., 3., 6.)
CURVE_PENALTIES = np.arange(101) / 10
METHODS = ("original", "isotonic_direct", "beta", "always_abstain")
LABELS = {"original": "Original threshold", "isotonic_direct": "Isotonic / direct score",
          "beta": "Smooth beta calibration", "always_abstain": "Always abstain",
          "penalty_prompt": "Error-penalty prompt"}
COLORS = {"original": "#737373", "isotonic_direct": "#126dae", "beta": "#d47d19"}
BETA_EPS = 1e-6
BETA_REG = .001


@dataclass
class IsotonicFit:
    x: np.ndarray
    counts: np.ndarray
    correct: np.ndarray
    fitted: np.ndarray
    blocks: list[dict]


def fit_isotonic(u, z):
    u, z = np.asarray(u, dtype=float), np.asarray(z, dtype=int)
    if len(u) != len(z) or np.any(~np.isfinite(u)) or np.any(~np.isin(z, [0, 1])):
        raise ValueError("Invalid score/correctness pairs")
    valid = u >= 0
    if np.any(u[valid] > 1):
        raise ValueError("Top probabilities must not exceed one")
    x, inverse = np.unique(u[valid], return_inverse=True)
    counts = np.bincount(inverse, minlength=len(x))
    correct = np.bincount(inverse, weights=z[valid], minlength=len(x)).astype(int)
    blocks = []
    for j, (n, k) in enumerate(zip(counts, correct)):
        blocks.append({"start": j, "end": j, "n": int(n), "k": int(k)})
        while len(blocks) > 1:
            a, b = blocks[-2:]
            if a["k"] * b["n"] <= b["k"] * a["n"]:
                break
            blocks[-2:] = [{"start": a["start"], "end": b["end"],
                            "n": a["n"] + b["n"], "k": a["k"] + b["k"]}]
    fitted = np.zeros(len(x))
    for block in blocks:
        fitted[block["start"]:block["end"]+1] = block["k"] / block["n"]
    return IsotonicFit(x, counts, correct, fitted, blocks)


def ratio(penalty):
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("Penalty must be finite and nonnegative")
    f = Fraction(str(float(penalty))).limit_denominator(1_000_000)
    return f.numerator, f.denominator


def isotonic_threshold(fit, penalty):
    numerator, denominator = ratio(penalty)
    for block in fit.blocks:
        if block["k"] * (numerator + denominator) >= numerator * block["n"]:
            return 0. if block["start"] == 0 else float(fit.x[block["start"]])
    return np.inf


def direct_threshold(fit, penalty):
    """Independent suffix-score maximization with exact rational cost comparisons."""
    if len(fit.x) == 0:
        return np.inf
    numerator, denominator = ratio(penalty)
    thresholds = np.unique(np.r_[0., fit.x, np.inf])
    starts = np.searchsorted(fit.x, thresholds, side="left")
    correct = np.r_[np.cumsum(fit.correct[::-1])[::-1], 0][starts]
    incorrect = np.r_[np.cumsum((fit.counts-fit.correct)[::-1])[::-1], 0][starts]
    score = denominator * correct - numerator * incorrect
    return float(thresholds[np.argmax(score)])


def isotonic_probability(fit, u):
    u = np.asarray(u, dtype=float)
    p = np.full(u.shape, -1.)
    if len(fit.x):
        valid = u >= 0
        indices = np.clip(np.searchsorted(fit.x, u[valid], side="right")-1, 0, len(fit.x)-1)
        p[valid] = fit.fitted[indices]
    return p


def beta_design(u):
    p = np.clip(np.asarray(u), BETA_EPS, 1-BETA_EPS)
    return np.column_stack([np.log(p), -np.log1p(-p), np.ones(len(p))])


def beta_objective(coef, x, z):
    linear = x @ coef
    displacement = coef - np.array([1., 1., 0.])
    value = np.mean(np.logaddexp(0, linear)-z*linear) + BETA_REG/2 * (displacement @ displacement)
    gradient = x.T @ (expit(linear)-z) / len(z) + BETA_REG * displacement
    return value, gradient


def fit_beta(u, z):
    valid = np.asarray(u) >= 0
    if not np.any(valid):
        return None
    x = beta_design(np.asarray(u)[valid])
    result = minimize(beta_objective, np.array([1., 1., 0.]), args=(x, np.asarray(z)[valid]),
                      jac=True, method="L-BFGS-B", bounds=[(0., None), (0., None), (None, None)],
                      options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 2000, "maxls": 100})
    if not result.success:
        raise RuntimeError(f"Beta calibration did not converge: {result.message}")
    return result.x


def beta_probability(coef, u):
    u = np.asarray(u, dtype=float)
    p = np.full(u.shape, -1.)
    valid = u >= 0
    if coef is not None:
        p[valid] = expit(beta_design(u[valid]) @ coef)
    return p


def beta_threshold(coef, penalty):
    if coef is None:
        return np.inf
    ratio(penalty)
    target = penalty / (1+penalty)
    def f(u):
        return float(beta_probability(coef, [u])[0]) - target
    if f(0.) >= 0:
        return 0.
    if f(1.) < 0:
        return np.inf
    return float(brentq(f, 0., 1., xtol=1e-14))


def processed_records(records):
    rows = []
    for record in records:
        candidates = [c for c in common.parse_candidates(record) if c["grade"] != "not_attempted"]
        if candidates:
            top = max(candidates, key=lambda c: c["probability"])
            u, z = float(top["probability"]), int(top["grade"] == "correct")
        else:
            u, z = -1., 0
        rows.append({"question_id": str(record["question_id"]), "u": u, "z": z})
    frame = pd.DataFrame(rows).sort_values("question_id").reset_index(drop=True)
    if frame.question_id.duplicated().any():
        raise ValueError("Duplicate questions")
    return frame


def metrics(u, z, threshold, penalty):
    u, z = np.asarray(u), np.asarray(z)
    returned = (u >= 0) & (u >= threshold)
    n = int(returned.sum())
    k = int(np.sum(returned * (1-z)))
    values = returned * (z-penalty*(1-z))
    c, mean_score = n / len(u), float(values.mean())
    raw_implied = float(np.mean(np.where(returned, u-penalty*(1-u), 0)))
    observed = k / n if n else np.nan
    reported = float(np.mean(1-u[returned])) if n else np.nan
    gap = observed-reported if n else np.nan
    np.testing.assert_allclose(raw_implied-mean_score, (1+penalty)*c*gap if n else 0, atol=1e-12)
    return {"n_test": len(u), "n_answered": n, "n_incorrect": k,
            "answer_rate": c, "conditional_error": observed, "mean_score": mean_score,
            "mean_raw_reported_error": reported, "raw_probability_implied_score": raw_implied,
            "score_overstatement": raw_implied-mean_score}, values


def penalty_metrics(frame, penalty):
    c, h = frame[f"penalty_correct_{int(penalty)}"].to_numpy(), frame[f"penalty_incorrect_{int(penalty)}"].to_numpy()
    answered = c+h
    values = c-penalty*h
    return {"n_test": len(frame), "n_answered": float(answered.sum()), "n_incorrect": float(h.sum()),
            "answer_rate": float(answered.mean()),
            "conditional_error": float(h.sum()/answered.sum()) if answered.sum() else np.nan,
            "mean_score": float(values.mean()), "mean_raw_reported_error": np.nan,
            "raw_probability_implied_score": np.nan, "score_overstatement": np.nan}, values


def penalty_record_rates(record, penalty):
    n = record["n_samples_requested"]
    if n != 50 or float(record["penalty_value"]) != penalty:
        raise ValueError("Unexpected penalty condition or sample count")
    counts = [record[f"penalty_{name}_samples"] for name in
              ("correct", "incorrect", "abstain", "not_attempted")]
    if any(k < 0 or int(k) != k for k in counts) or sum(counts) != n:
        raise ValueError("Penalty outcomes do not reconcile to requested samples")
    correct, incorrect = counts[0]/n, counts[1]/n
    np.testing.assert_allclose([correct, incorrect, counts[2]/n],
                              [record["penalty_accuracy_overall"],
                               record["penalty_hallucination_rate"],
                               record["penalty_abstention_rate"]], atol=1e-9)
    # As in the existing three-way analysis, explicit abstentions and graded
    # not_attempted responses both receive zero. Preserve the 50-call denominator.
    return correct, incorrect


def paired_interval(difference, weights, level=.95):
    draws = weights @ np.asarray(difference, dtype=float) / len(difference)
    tail = (1-level)/2
    low, high = np.quantile(draws, [tail, 1-tail])
    return {"difference": float(np.mean(difference)), "lower": low, "upper": high, "level": level}


def analyze(frame, seed, primary=False):
    train, test = prior.split_frames(frame, seed)
    u, z = train.u.to_numpy(), train.z.to_numpy()
    fit, coef = fit_isotonic(u, z), fit_beta(u, z)
    penalties = CURVE_PENALTIES if primary else PENALTIES
    rows, thresholds, contrasts = [], [], []
    weights = None
    if primary:
        weights = np.random.default_rng(SEED+1000).multinomial(len(test), np.full(len(test), 1/len(test)), size=2000)
    previous = -np.inf
    for penalty in penalties:
        t_iso, t_direct = isotonic_threshold(fit, penalty), direct_threshold(fit, penalty)
        if t_iso != t_direct:
            raise AssertionError(f"Isotonic/direct disagreement at L={penalty}: {t_iso}, {t_direct}")
        if t_direct < previous:
            raise AssertionError("Optimal thresholds must be nondecreasing in penalty")
        previous = t_direct
        fitted_thresholds = {"original": penalty/(1+penalty), "isotonic_direct": t_direct,
                             "beta": beta_threshold(coef, penalty), "always_abstain": np.inf}
        values = {}
        for method, threshold in fitted_thresholds.items():
            m, values[method] = metrics(test.u, test.z, threshold, penalty)
            train_m, _ = metrics(train.u, train.z, threshold, penalty)
            rows.append({"seed": seed, "L": penalty, "method": method, "threshold": threshold,
                         "fit_mean_score": train_m["mean_score"], **m})
        thresholds.append({"seed": seed, "L": penalty, "route1_threshold": t_iso,
                           "route2_threshold": t_direct, "beta_threshold": fitted_thresholds["beta"]})
        if penalty in PENALTIES:
            m, values["penalty_prompt"] = penalty_metrics(test, penalty)
            rows.append({"seed": seed, "L": penalty, "method": "penalty_prompt", "threshold": np.nan,
                         "fit_mean_score": np.nan, **m})
            if primary:
                for method in ("isotonic_direct", "beta"):
                    for comparator in ("original", "penalty_prompt", "always_abstain"):
                        level = 1-.05/6 if method == "isotonic_direct" and comparator == "original" and penalty in (3.,6.) else .95
                        contrasts.append({"L": penalty, "method": method, "comparator": comparator,
                                          **paired_interval(values[method]-values[comparator], weights, level)})
    prediction = None
    if primary:
        prediction = frame.copy()
        prediction["split"] = np.where(prediction.question_id.isin(train.question_id), "fit", "test")
        prediction["isotonic_probability"] = isotonic_probability(fit, prediction.u)
        prediction["beta_probability"] = beta_probability(coef, prediction.u)
        for penalty in PENALTIES:
            for method in ("original", "isotonic_direct", "beta"):
                t = next(r["threshold"] for r in rows if r["L"] == penalty and r["method"] == method)
                _, score = metrics(prediction.u, prediction.z, t, penalty)
                prediction[f"score_{method}_L{int(penalty)}"] = score
    return pd.DataFrame(rows), pd.DataFrame(thresholds), pd.DataFrame(contrasts), prediction, fit, coef


def load_inputs(grader):
    frames, inputs = {}, []
    tree = ROOT / "results" / f"graded_by_{grader}"
    def read(path):
        resolved = common.result_file(path)
        if resolved is None:
            raise FileNotFoundError(path)
        inputs.append({"file": str(resolved.relative_to(ROOT)), "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()})
        return common.load_jsonl(resolved)
    for key in common.MODEL_ORDER:
        run = common.RUNS[key]["run"]
        reports = read(tree/run/"simpleqa_topp_results.jsonl")
        representations = {"raw": prior.prepare_records(reports)[["question_id", "u", "z"]],
                           "processed": processed_records(reports)}
        for penalty, suffix in [(0,"_L0"),(3,""),(6,"_L6")]:
            records = read(tree/(run+suffix)/"simpleqa_penalty_results.jsonl")
            rates = []
            for r in records:
                correct, incorrect = penalty_record_rates(r, penalty)
                rates.append({"question_id": str(r["question_id"]), f"penalty_correct_{penalty}": correct,
                              f"penalty_incorrect_{penalty}": incorrect})
            for name in representations:
                before = representations[name]
                merged = before.merge(pd.DataFrame(rates), on="question_id", validate="one_to_one")
                if len(merged) != len(before):
                    raise ValueError("Missing penalty questions")
                representations[name] = merged.sort_values("question_id").reset_index(drop=True)
        for name, frame in representations.items():
            frames[(key, name)] = frame
    ids = next(iter(frames.values())).question_id.tolist()
    if len(ids) != 4326 or any(f.question_id.tolist() != ids for f in frames.values()):
        raise ValueError("Expected identical 4,326 questions")
    return frames, inputs


def plot_primary(data, predictions, out):
    plt.rcParams.update({"font.size":10, "pdf.fonttype":42, "axes.spines.top":False, "axes.spines.right":False})
    for representation in ("raw", "processed"):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey="row")
        for column, key in enumerate(common.MODEL_ORDER):
            part = data[(data.model_key==key) & (data.representation==representation)]
            for method in COLORS:
                g=part[part.method==method].sort_values("L")
                axes[0,column].plot(g.L, g.mean_score, color=COLORS[method], lw=2, label=LABELS[method])
                axes[1,column].plot(g.L,100*g.answer_rate,color=COLORS[method],lw=2)
            g=part[part.method=="penalty_prompt"]
            axes[0,column].scatter(g.L,g.mean_score,color="black",marker="D",s=30,label=LABELS["penalty_prompt"],zorder=5)
            axes[1,column].scatter(g.L,100*g.answer_rate,color="black",marker="D",s=30,zorder=5)
            axes[0,column].axhline(0,color="black",ls=":",lw=1,label="Always abstain: score 0")
            axes[0,column].set_title(common.RUNS[key]["label"],loc="left",fontweight="bold",fontsize=11)
            axes[1,column].set(xlabel="Incorrect-answer penalty, $L$",xlim=(0,10),ylim=(-2,102))
            for row in range(2):
                axes[row,column].grid(alpha=.16)
        axes[0,0].set_ylabel("Mean decision score per question")
        axes[1,0].set_ylabel("Questions answered (%)")
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False)
        description="Raw reports" if representation=="raw" else "Gold-informed processed reports"
        fig.suptitle(f"{description}: primary held-out comparison (2,163 questions)",fontsize=13)
        fig.tight_layout(rect=(0,.11,1,.95))
        fig.savefig(out/f"decision_scores_{representation}.png",dpi=170,bbox_inches="tight")
        fig.savefig(out/f"decision_scores_{representation}.pdf",bbox_inches="tight")
        plt.close(fig)


def run(grader="openai", out_dir=None, repeats=REPEATS):
    out = Path(out_dir) if out_dir else ROOT/"results"/"exploratory"/f"calibration_decisions_{grader}"
    out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(__file__).with_name("calibration_decisions_theory.md"),out/"theory.md")
    frames,inputs=load_inputs(grader)
    all_rows,all_thresholds,all_contrasts,all_predictions,fit_records=[],[],[],[],[]
    for (key,representation),frame in frames.items():
        for offset in range(repeats+1):
            rows,thresholds,contrasts,pred,fit,coef=analyze(frame,SEED+offset,primary=offset==0)
            info={"model_key":key,"representation":representation}
            all_rows.append(rows.assign(**info)); all_thresholds.append(thresholds.assign(**info))
            if offset==0:
                all_contrasts.append(contrasts.assign(**info)); all_predictions.append(pred.assign(**info))
                for block in fit.blocks:
                    fit_records.append({**info,**block,"u_start":fit.x[block["start"]],"u_end":fit.x[block["end"]],"fitted_probability":block["k"]/block["n"]})
                (out/f"fit_{key}_{representation}.json").write_text(json.dumps({"u":fit.x.tolist(),"counts":fit.counts.tolist(),"correct":fit.correct.tolist(),"fitted_probability":fit.fitted.tolist(),"blocks":fit.blocks,"beta_parameters":coef.tolist() if coef is not None else None},indent=2)+"\n")
            if offset%25==0 or offset==repeats:
                print(f"{key} / {representation}: split {offset}/{repeats}",flush=True)
    data,thresholds,predictions=pd.concat(all_rows),pd.concat(all_thresholds),pd.concat(all_predictions)
    primary=data[data.seed==SEED]
    repeated=data[data.seed!=SEED]
    data.to_csv(out/"all_metrics.csv",index=False)
    primary[primary.L.isin(PENALTIES)].to_csv(out/"primary_metrics.csv",index=False)
    primary.to_csv(out/"primary_penalty_curves.csv",index=False)
    thresholds.to_csv(out/"route_equivalence_thresholds.csv",index=False)
    predictions.to_csv(out/"primary_predictions.csv",index=False)
    pd.concat(all_contrasts).to_csv(out/"primary_paired_intervals.csv",index=False)
    pd.DataFrame(fit_records).to_csv(out/"primary_isotonic_blocks.csv",index=False)
    summary=[]
    for names,g in repeated.groupby(["model_key","representation","L","method"]):
        summary.append(dict(zip(["model_key","representation","L","method"],names)) | {
            "n_splits":len(g),"mean_score":g.mean_score.mean(),"median_score":g.mean_score.median(),
            "score_q25":g.mean_score.quantile(.25),"score_q75":g.mean_score.quantile(.75),
            "mean_answer_rate":g.answer_rate.mean(),"median_answer_rate":g.answer_rate.median(),
            "n_all_abstain":int((g.n_answered==0).sum()),"n_positive_test_score":int((g.mean_score>0).sum()),
            "median_error_when_answering":g.loc[g.n_answered>0,"conditional_error"].median()})
    pd.DataFrame(summary).to_csv(out/"split_summary.csv",index=False)
    paired=[]
    for names,g in repeated.groupby(["model_key","representation","L","seed"]):
        g=g.set_index("method")
        for method in ("isotonic_direct","beta"):
            for comparator in ("original","penalty_prompt","always_abstain"):
                paired.append(dict(zip(["model_key","representation","L","seed"],names)) | {
                    "method":method,"comparator":comparator,
                    "score_difference":g.at[method,"mean_score"]-g.at[comparator,"mean_score"],
                    "answer_rate_difference":g.at[method,"answer_rate"]-g.at[comparator,"answer_rate"]})
    pd.DataFrame(paired).to_csv(out/"paired_split_differences.csv",index=False)
    plot_primary(primary,predictions,out)
    plot_calibration_maps(out)
    metadata={"primary_seed":SEED,"additional_splits":repeats,"fit_n":2163,"test_n":2163,
              "tabulated_penalties":PENALTIES,"curve_penalties":CURVE_PENALTIES.tolist(),
              "primary_representation":"raw","representations":["raw","processed"],
              "primary_method":"isotonic_direct","beta_regularizer":BETA_REG,"beta_clip":BETA_EPS,
              "primary_interval_scope":"six raw OpenAI-graded isotonic-vs-original comparisons at L=3,6; other intervals secondary",
              "route1_route2_all_thresholds_equal":bool((thresholds.route1_threshold==thresholds.route2_threshold).all()),
              "design":"retrospective benchmark; no fixed-error certification; overlapping splits are descriptive",
              "inputs":inputs,"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "software":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__}}
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"Saved results to {out}",flush=True)
    return out


def plot_calibration_maps(out):
    for representation in ("raw", "processed"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        for ax, key in zip(axes, common.MODEL_ORDER):
            saved = json.loads((out/f"fit_{key}_{representation}.json").read_text())
            x = np.array(saved["u"])
            # Show only the observed fitting range; endpoint extrapolation is
            # a prediction convention, not observed calibration evidence.
            grid = np.linspace(x[0], x[-1], 500)
            ax.step(x, saved["fitted_probability"], where="post", lw=2,
                    color=COLORS["isotonic_direct"], label="Fitted isotonic correctness")
            ax.plot(grid, beta_probability(saved["beta_parameters"], grid), lw=2,
                    color=COLORS["beta"], label="Fitted smooth correctness")
            ax.plot([0,1], [0,1], color=".6", lw=1, ls=":", label="Reported probability")
            ax.axhline(.75, color=".3", ls="--", lw=.9)
            ax.axhline(6/7, color=".3", ls="--", lw=.9)
            ax.text(.025,.756,"Required at L=3",fontsize=8,color=".25")
            ax.text(.025,6/7+.006,"Required at L=6",fontsize=8,color=".25")
            ax.set(xlim=(0,1),ylim=(0,1.02),xlabel="Reported top probability, $U$")
            ax.set_title(common.RUNS[key]["label"], loc="left", fontsize=11, fontweight="bold")
            ax.grid(alpha=.15)
        axes[0].set_ylabel("Estimated probability of correctness")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False)
        description = "Raw reports" if representation == "raw" else "Gold-informed processed reports"
        fig.suptitle(f"{description}: calibration maps fitted on 2,163 questions",fontsize=13)
        fig.tight_layout(rect=(0,.13,1,.94))
        fig.savefig(out/f"calibration_maps_{representation}.png",dpi=170,bbox_inches="tight")
        fig.savefig(out/f"calibration_maps_{representation}.pdf",bbox_inches="tight")
        plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grader",choices=("openai","gemini"),default="openai")
    parser.add_argument("--out-dir",type=Path)
    parser.add_argument("--repeats",type=int,default=REPEATS)
    args=parser.parse_args()
    if args.repeats<0:
        parser.error("--repeats must be nonnegative")
    run(args.grader,args.out_dir,args.repeats)
