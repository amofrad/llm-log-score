"""Regenerate every table in the paper from a graded result tree.

Steps (run all by default, or select with --steps):
  audits       Tables S1/S2 reliability bins and the Table S3 realized
               log-loss audit (theory_audit_*.csv).
  gaps         Table S5 matched-abstention gaps with the paired
               question-level bootstrap (matched_abstention_gaps.csv).
  containment  The Assumption-1 support-containment audit
               (support_containment.csv).
  tables       LaTeX table bodies for Tables 1, 2, S1, S2, S3, S5 in the
               manuscript's exact formatting (table_bodies.txt).
  headline     Machine-readable headline numbers quoted in the text
               (headline_report.json).
  tokens       Average answer-model token cost per question, Table S4
               (token_costs.csv).
  agreement    Grader-agreement audit between the OpenAI-graded and
               Gemini-graded trees (grader_agreement.csv); this step reads
               both trees and ignores --grader.

Select the result tree with --grader openai|gemini (default: openai, the
paper's primary grading).  Run:  python analysis/make_tables.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "grading"))

import common  # noqa: E402
from common import load_jsonl, load_run, open_result, result_file, top_p_set  # noqa: E402
import make_figures as mf  # noqa: E402
from graders import canonical_incorrect_answer_key  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "results" / "figures"

# (label, RUNS key, log-run dir, log-consistency dir)
MODELS = [
    ("Gemini 3.5 Flash", "gemini35flash", "gemini35flash", "gemini35flash_consistency"),
    ("Claude Sonnet 4.6", "sonnet46", "claudesonnet46", "claudesonnet46_consistency"),
    ("DeepSeek V3.2", "deepseekv32maas", "deepseekv32", "deepseekv32_consistency"),
    ("Qwen3 235B-A22B", "qwen3_235b", "qwen3_235b", "qwen3_235b_consistency"),
]
P = 0.9
THRESH = {0: 0.0, 3: 0.75, 6: 6.0 / 7.0}
PENALTIES = [0.0, 3.0, 6.0]
TOP_BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0001)]
IDK_BINS = [(0.0, 0.05), (0.05, 0.2), (0.2, 0.4), (0.4, 0.7), (0.7, 1.0001)]

# Matched-gap bootstrap configuration (Table S5).
GAP_THRESHOLDS = np.linspace(0, 0.95, 96)  # same grid as the frontier figure
N_BOOT = 2000
SEED = 20260812


def set_results_tree(tree: Path) -> None:
    """Point every loader at one graded result tree."""
    tree = Path(tree)
    common.OUTPUTS_DIR = tree
    mf.OUTPUTS_DIR = tree


def tree_for_grader(grader: str) -> Path:
    return REPO_ROOT / "results" / f"graded_by_{grader}"


# ---------------------------------------------------------------------------
# Audits: reliability bins (Tables S1/S2) and realized log loss (Table S3)
# ---------------------------------------------------------------------------

def question_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        cands = r["candidates"]
        concrete = [c for c in cands if c["grade"] != "not_attempted"]
        idk = sum(c["probability"] for c in cands if c["grade"] == "not_attempted")
        q_true = sum(c["probability"] for c in concrete if c["grade"] == "correct")
        any_correct = any(c["grade"] == "correct" for c in concrete)
        top_prob, top_correct = np.nan, np.nan
        if concrete:
            top = max(concrete, key=lambda c: float(c["probability"]))
            top_prob = float(top["probability"])
            top_correct = float(top["grade"] == "correct")
        # cell containing the adjudicated truth: pooled correct cell if the
        # truth is listed, else the residual cell
        q_py = q_true if any_correct else idk
        # unflagged miss for the top-p display
        ordered = sorted(cands, key=lambda c: float(c["probability"]), reverse=True)
        cum, in_prefix_truth, in_prefix_idk = 0.0, False, False
        for c in ordered:
            cum += float(c["probability"])
            if c["grade"] == "correct":
                in_prefix_truth = True
            if c["grade"] == "not_attempted":
                in_prefix_idk = True
            if cum >= P:
                break
        unflagged_miss = (not in_prefix_truth) and (not in_prefix_idk)
        rows.append({
            "question_id": r["question_id"],
            "top_prob": top_prob,
            "top_correct": top_correct,
            "idk_mass": idk,
            "truth_unlisted": float(not any_correct),
            "q_py": q_py,
            "unflagged_miss": float(unflagged_miss),
        })
    return pd.DataFrame(rows)


def reliability_table(qs: pd.DataFrame, value: str, target: str, bins) -> pd.DataFrame:
    out = []
    sub = qs.dropna(subset=[value])
    for lo, hi in bins:
        m = (sub[value] >= lo) & (sub[value] < hi)
        if m.sum() == 0:
            continue
        out.append({
            "bin_lo": lo, "bin_hi": min(hi, 1.0), "n": int(m.sum()),
            "mean_stated": float(sub.loc[m, value].mean()),
            "empirical": float(sub.loc[m, target].mean()),
        })
    return pd.DataFrame(out)


def dy_stats(qs: pd.DataFrame, weights: np.ndarray) -> dict:
    w = np.asarray(weights, dtype=float)
    q = qs["q_py"].to_numpy(dtype=float)
    miss = qs["unflagged_miss"].to_numpy(dtype=float)
    wsum = w.sum()
    u = float((w * miss).sum() / wsum)
    # each unflagged miss carries D_Y >= log(1/(1-p)) since the truth's cell
    # lies outside a prefix holding stated mass >= p
    bound = u * float(np.log(1.0 / (1.0 - P)))
    finite = q > 0
    dy = np.full_like(q, np.inf)
    dy[finite] = -np.log(q[finite])
    # weighted median of dy (including infinities)
    order = np.argsort(dy)
    cw = np.cumsum(w[order]) / wsum
    median = float(dy[order][np.searchsorted(cw, 0.5)])
    mean_finite = float((w[finite] * dy[finite]).sum() / w[finite].sum())
    frac_inf = float(w[~finite].sum() / wsum)
    return {"u": u, "bound": bound, "median_dy": median,
            "mean_dy_finite": mean_finite, "frac_infinite": frac_inf}


def penalty_incorrect_weights(run_dir: str) -> dict:
    """Per-question incorrect-sample fraction from the native L=3 arm."""
    weights = {}
    for rec in load_jsonl(common.OUTPUTS_DIR / run_dir / "simpleqa_penalty_results.jsonl"):
        n = rec.get("n_samples_requested") or 50
        weights[str(rec["question_id"])] = (rec.get("penalty_incorrect_samples") or 0) / n
    return weights


def audits(out_dir: Path) -> None:
    top_rows, idk_rows, dy_rows = [], [], []
    for label, _key, run, _cons in MODELS:
        df = load_run(run)
        qs = question_stats(df)
        w_inc = penalty_incorrect_weights(run)

        t = reliability_table(qs, "top_prob", "top_correct", TOP_BINS)
        t.insert(0, "model", label)
        top_rows.append(t)

        t = reliability_table(qs, "idk_mass", "truth_unlisted", IDK_BINS)
        t.insert(0, "model", label)
        idk_rows.append(t)

        overall = dy_stats(qs, np.ones(len(qs)))
        overall.update({"model": label, "group": "overall"})
        w = qs["question_id"].map(w_inc).fillna(0.0).to_numpy()
        pen_inc = dy_stats(qs, w)
        pen_inc.update({"model": label, "group": "penalty-incorrect"})
        dy_rows.extend([overall, pen_inc])

    pd.concat(top_rows, ignore_index=True).to_csv(
        out_dir / "theory_audit_top_reliability.csv", index=False)
    pd.concat(idk_rows, ignore_index=True).to_csv(
        out_dir / "theory_audit_residual_reliability.csv", index=False)
    dy_tab = pd.DataFrame(dy_rows)[
        ["model", "group", "u", "bound", "median_dy", "mean_dy_finite", "frac_infinite"]
    ]
    dy_tab.to_csv(out_dir / "theory_audit_realized_log_loss.csv", index=False)
    print("== Realized log loss vs certified bound ==")
    print(dy_tab.round(4).to_string(index=False))


# ---------------------------------------------------------------------------
# Matched-abstention gaps with paired bootstrap (Table S5)
# ---------------------------------------------------------------------------

def _paired_inputs(key: str, penalty: float):
    dlog = load_run(common.RUNS[key]["run"]).set_index("question_id", drop=False)
    dpen = mf.load_penalty_arm(key, penalty)
    shared = dlog.index.intersection(dpen.index)
    dlog = dlog.loc[shared]
    dpen = dpen.loc[shared]
    n = pd.to_numeric(dpen["n_samples_requested"], errors="coerce").fillna(50.0).to_numpy()
    corr = pd.to_numeric(dpen["penalty_correct_samples"], errors="coerce").fillna(0.0).to_numpy() / n
    inc = pd.to_numeric(dpen["penalty_incorrect_samples"], errors="coerce").fillna(0.0).to_numpy() / n
    abst = 1.0 - corr - inc
    log_inc, log_corr, log_abs = mf._frontier_decision_matrices(dlog, GAP_THRESHOLDS)
    return corr, inc, abst, log_inc, log_corr, log_abs


def _gap_stats(w, corr, inc, abst, log_inc, log_corr, log_abs):
    """Weighted (question-frequency) gap statistics for one resample."""
    a_pen = float(w @ abst)
    h_pen = float(w @ inc)
    c_pen = float(w @ corr)
    ab_t = w @ log_abs
    h_t = w @ log_inc
    c_t = w @ log_corr
    grid = np.asarray([a_pen])
    h_log = float(mf._interp_frontier_on_abstention(ab_t, h_t, grid)[0])
    c_log = float(mf._interp_frontier_on_abstention(ab_t, c_t, grid)[0])
    return {
        "pen_abstention": a_pen,
        "pen_hallucination": h_pen,
        "pen_accuracy": c_pen,
        "gap_hallucination": h_pen - h_log,
        "gap_accuracy": c_log - c_pen,
        "min_log_abstention": float(ab_t.min()),
    }


def matched_gaps(out_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for _label, key, _run, _cons in MODELS:
        label = common.RUNS[key]["label"]
        for penalty in PENALTIES:
            corr, inc, abst, log_inc, log_corr, log_abs = _paired_inputs(key, penalty)
            nq = len(corr)
            w0 = np.full(nq, 1.0 / nq)
            point = _gap_stats(w0, corr, inc, abst, log_inc, log_corr, log_abs)
            boot = {"gap_hallucination": [], "gap_accuracy": [],
                    "pen_hallucination": [], "pen_accuracy": [], "pen_abstention": []}
            for _ in range(N_BOOT):
                idx = rng.integers(0, nq, size=nq)
                w = np.bincount(idx, minlength=nq).astype(float) / nq
                s = _gap_stats(w, corr, inc, abst, log_inc, log_corr, log_abs)
                for k in boot:
                    boot[k].append(s[k])
            row = {"model": label, "L": penalty, "n_questions": nq,
                   "matched": point["pen_abstention"] >= point["min_log_abstention"]}
            for k, v in point.items():
                row[k] = v
            for k, vals in boot.items():
                lo, hi = np.quantile(np.asarray(vals), [0.025, 0.975])
                row[f"{k}_lo"] = float(lo)
                row[f"{k}_hi"] = float(hi)
            rows.append(row)
            print(f"{label} L={penalty:g}: pen abst {point['pen_abstention']:.3f} "
                  f"(matched={row['matched']}), "
                  f"gap_hall {point['gap_hallucination']*100:+.2f}pp "
                  f"[{row['gap_hallucination_lo']*100:+.2f}, {row['gap_hallucination_hi']*100:+.2f}], "
                  f"gap_acc {point['gap_accuracy']*100:+.2f}pp "
                  f"[{row['gap_accuracy_lo']*100:+.2f}, {row['gap_accuracy_hi']*100:+.2f}]")
    pd.DataFrame(rows).to_csv(out_dir / "matched_abstention_gaps.csv", index=False)
    print("wrote", out_dir / "matched_abstention_gaps.csv")


# ---------------------------------------------------------------------------
# Support-containment audit (Assumption 1)
# ---------------------------------------------------------------------------

def _log_report_keys(cands: list[dict]) -> tuple[set, bool]:
    keys = set()
    has_correct = False
    for c in cands:
        if c["grade"] == "not_attempted":
            continue
        if c["grade"] == "correct":
            has_correct = True
        keys.add(canonical_incorrect_answer_key(c["answer"]))
    return keys, has_correct


def support_containment(out_dir: Path) -> None:
    """How often an answered penalty-arm sample names an answer that the same
    model's log report also lists: a correct-graded penalty answer is
    contained iff the log report lists any correct-graded candidate (both
    land in the pooled gold cell), and an incorrect-graded penalty answer is
    contained iff its canonical string (or any raw source variant) matches
    the canonical string of a concrete log candidate."""
    rows = []
    for _label, key, _run, _cons in MODELS:
        label = common.RUNS[key]["label"]
        dlog = load_run(common.RUNS[key]["run"]).set_index("question_id", drop=False)
        for penalty in PENALTIES:
            dpen = mf.load_penalty_arm(key, penalty)
            shared = dlog.index.intersection(dpen.index)
            n_ans = n_in = 0.0
            n_ans_c = n_in_c = 0.0
            n_ans_i = n_in_i = 0.0
            for qid in shared:
                cands = dlog.at[qid, "candidates"]
                keys, has_correct = _log_report_keys(cands)
                raw = dpen.at[qid, "penalty_distribution_json"]
                items = json.loads(raw) if isinstance(raw, str) and raw else []
                for item in items:
                    grade = item.get("grade")
                    if grade not in ("correct", "incorrect"):
                        continue
                    count = float(item.get("count") or 0)
                    if count <= 0:
                        continue
                    if grade == "correct":
                        contained = has_correct
                        n_ans_c += count
                        n_in_c += count * contained
                    else:
                        variants = [item.get("answer", "")] + list(item.get("source_answers") or [])
                        contained = any(
                            canonical_incorrect_answer_key(v) in keys for v in variants if v
                        )
                        n_ans_i += count
                        n_in_i += count * contained
                    n_ans += count
                    n_in += count * contained
            rows.append({
                "model": label, "L": penalty,
                "answered_samples": int(n_ans),
                "contained": n_in / n_ans if n_ans else np.nan,
                "contained_correct": n_in_c / n_ans_c if n_ans_c else np.nan,
                "contained_incorrect": n_in_i / n_ans_i if n_ans_i else np.nan,
            })
            r = rows[-1]
            print(f"{label} L={penalty:g}: answered={r['answered_samples']} "
                  f"contained={r['contained']:.3f} "
                  f"(correct {r['contained_correct']:.3f}, incorrect {r['contained_incorrect']:.3f})")
    pd.DataFrame(rows).to_csv(out_dir / "support_containment.csv", index=False)
    print("wrote", out_dir / "support_containment.csv")


# ---------------------------------------------------------------------------
# LaTeX table bodies (Tables 1, 2, S1, S2, S3, S5)
# ---------------------------------------------------------------------------

def _f3(x):
    return f"{x:.3f}"


def _penalty_set_metrics(dpen):
    """Top-0.9 set metrics from the penalty sample distribution.

    The residual cell carries penalty_abstain_samples/n only, and IDK mass
    sums the not_attempted-graded cells.
    """
    covers, has_idk, idk_mass = [], [], []
    for _, r in dpen.iterrows():
        raw = r.get("penalty_distribution_json")
        items = json.loads(raw) if isinstance(raw, str) and raw else []
        cells = [
            {"answer": str(it.get("answer", "")), "grade": str(it.get("grade", "incorrect")),
             "probability": float(it.get("probability", 0.0))}
            for it in items if float(it.get("probability", 0.0)) > 0
        ]
        n = max(1, int(r.get("n_samples_requested") or 1))
        abstain_prob = float(r.get("penalty_abstain_samples") or 0) / n
        if abstain_prob > 0:
            cells.append({"answer": "ABSTAIN", "grade": "not_attempted",
                          "probability": abstain_prob})
        s = top_p_set(cells, P)
        covers.append(s.covers)
        has_idk.append(s.has_idk)
        idk_mass.append(sum(c["probability"] for c in cells if c["grade"] == "not_attempted"))
    covers = np.asarray(covers)
    has_idk = np.asarray(has_idk)
    return {"cov_idk": float((covers | has_idk).mean()),
            "strict": float(covers.mean()),
            "idk_mass": float(np.mean(idk_mass))}


def _penalty_decision_rates(dpen):
    n = pd.to_numeric(dpen["n_samples_requested"], errors="coerce").fillna(50.0)
    corr = pd.to_numeric(dpen["penalty_correct_samples"], errors="coerce").fillna(0.0) / n
    inc = pd.to_numeric(dpen["penalty_incorrect_samples"], errors="coerce").fillna(0.0) / n
    return {"abstain": float((1 - corr - inc).mean()),
            "halluc": float(inc.mean()), "acc": float(corr.mean())}


def table_bodies(out_dir: Path) -> None:
    lines = []

    def emit(text=""):
        lines.append(text)
        print(text)

    logs = {label: load_run(run) for label, _key, run, _cons in MODELS}
    pens = {(label, L): mf.load_penalty_arm(key, float(L))
            for label, key, _run, _cons in MODELS for L in [0, 3, 6]}

    # ---------------- Table 1 ----------------
    t1 = []
    for label, _key, _run, _cons in MODELS:
        df = logs[label]
        block = [f"\\multirow{{6}}{{*}}{{\\modelname{{{label}}}}}"]
        for i, L in enumerate([0, 3, 6]):
            dec = df["candidates"].map(lambda c, t=THRESH[L]: common.posthoc_decision(c, t))
            lg = {"abstain": (dec == "abstain").mean(), "halluc": (dec == "incorrect").mean(),
                  "acc": (dec == "correct").mean()}
            pn = _penalty_decision_rates(pens[(label, L)])
            lead = block[0] if i == 0 else ""
            t1.append(f"{lead} & \\multirow{{2}}{{*}}{{${L}$}} & Log & {_f3(lg['abstain'])} & {_f3(lg['halluc'])} & {_f3(lg['acc'])}\\\\")
            t1.append(f" &  & Pen. & {_f3(pn['abstain'])} & {_f3(pn['halluc'])} & {_f3(pn['acc'])}\\\\")
            if i < 2:
                t1.append("\\cmidrule(lr){2-6}")
        t1.append("\\midrule")
    t1 = t1[:-1]  # drop trailing midrule
    emit("==== TABLE 1 BODY ====")
    emit("\n".join(t1))

    # ---------------- Table 2 ----------------
    t2 = []
    for label, _key, _run, _cons in MODELS:
        df = logs[label]
        covers, has_idk = [], []
        for cands in df["candidates"]:
            s = top_p_set(cands, P)
            covers.append(s.covers)
            has_idk.append(s.has_idk)
        covers = np.asarray(covers); has_idk = np.asarray(has_idk)
        log_row = {"cov_idk": (covers | has_idk).mean(), "strict": covers.mean(),
                   "idk_mass": df["log_idk_mass"].mean()}
        t2.append(f"\\multirow{{4}}{{*}}{{\\modelname{{{label}}}}} & -- & Log & {_f3(log_row['cov_idk'])} & {_f3(log_row['strict'])} & {_f3(log_row['idk_mass'])}\\\\")
        for L in [0, 3, 6]:
            m = _penalty_set_metrics(pens[(label, L)])
            t2.append(f" & {L} & Pen. & {_f3(m['cov_idk'])} & {_f3(m['strict'])} & {_f3(m['idk_mass'])}\\\\")
        t2.append("\\midrule")
    t2 = t2[:-1]
    emit("\n==== TABLE 2 BODY ====")
    emit("\n".join(t2))

    # ---------------- Tables S1/S2 from audit CSVs ----------------
    idk = pd.read_csv(out_dir / "theory_audit_residual_reliability.csv")
    top = pd.read_csv(out_dir / "theory_audit_top_reliability.csv")
    bins_idk = [(0.0, 0.05, "$[0,0.05)$"), (0.05, 0.2, "$[0.05,0.2)$"), (0.2, 0.4, "$[0.2,0.4)$"),
                (0.4, 0.7, "$[0.4,0.7)$"), (0.7, 1.0, "$[0.7,1]$")]
    bins_top = [(0.0, 0.25, "$[0,0.25)$"), (0.25, 0.5, "$[0.25,0.5)$"), (0.5, 0.75, "$[0.5,0.75)$"),
                (0.75, 0.9, "$[0.75,0.9)$"), (0.9, 1.0, "$[0.9,1]$")]
    for name, tab, bins in [("S1 (residual)", idk, bins_idk), ("S2 (top)", top, bins_top)]:
        emit(f"\n==== TABLE {name} BODY ====")
        for lo, hi, lab in bins:
            cells = [lab]
            for label, _key, _run, _cons in MODELS:
                r = tab[(tab.model == label) & (np.isclose(tab.bin_lo, lo))]
                if len(r) == 0:
                    cells += ["--", "--", "--"]
                else:
                    r = r.iloc[0]
                    cells += [f"{int(r['n'])}", f"{r['mean_stated']:.3f}", f"{r['empirical']:.3f}"]
            emit("  & ".join(cells) + "\\\\")

    # ---------------- Table S3 ----------------
    dy = pd.read_csv(out_dir / "theory_audit_realized_log_loss.csv")
    emit("\n==== TABLE S3 BODY ====")
    for label, _key, _run, _cons in MODELS:
        o = dy[(dy.model == label) & (dy.group == "overall")].iloc[0]
        p_ = dy[(dy.model == label) & (dy.group == "penalty-incorrect")].iloc[0]
        emit(f"\\modelname{{{label}}} & {o.u:.3f} & {o.bound:.3f} & {o.median_dy:.2f} & {o.mean_dy_finite:.2f} & {p_.u:.3f} & {p_.bound:.3f} & {p_.median_dy:.2f} & {p_.mean_dy_finite:.2f}\\\\")
    emit("\nS3 4dp bounds: " + str({label: (round(r.u, 4), round(r.bound, 4))
                                    for label, *_ in MODELS
                                    for _, r in dy.iterrows()
                                    if r.model == label and r.group == "overall"}))
    emit("S3 4dp pen-inc: " + str({label: (round(r.u, 4), round(r.bound, 4))
                                   for label, *_ in MODELS
                                   for _, r in dy.iterrows()
                                   if r.model == label and r.group == "penalty-incorrect"}))

    # ---------------- Table S5 ----------------
    g = pd.read_csv(out_dir / "matched_abstention_gaps.csv")
    emit("\n==== TABLE S5 BODY ====")
    for label, _key, _run, _cons in MODELS:
        rows = g[g.model == label]
        emit(f"\\multirow{{3}}{{*}}{{\\modelname{{{label}}}}}")
        for _, r in rows.sort_values("L").iterrows():
            gh = r.gap_hallucination * 100; gl = r.gap_hallucination_lo * 100; gu = r.gap_hallucination_hi * 100
            ah = r.gap_accuracy * 100; al = r.gap_accuracy_lo * 100; au = r.gap_accuracy_hi * 100
            emit(f" & {int(r.L)} & {r.pen_abstention:.3f} & ${gh:.2f}$ $[{gl:.2f}, {gu:.2f}]$ & ${ah:.2f}$ $[{al:.2f}, {au:.2f}]$\\\\")
        emit("\\midrule")

    (out_dir / "table_bodies.txt").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Headline numbers quoted in the running text
# ---------------------------------------------------------------------------

def headline(out_dir: Path) -> None:
    report = {}
    for _label, key, run, _cons in MODELS:
        label = common.RUNS[key]["label"]
        df = load_run(run)
        entry = {"tables": {}}
        for L, t in THRESH.items():
            dec = df["candidates"].map(lambda c, t=t: common.posthoc_decision(c, t))
            entry["tables"][f"log_L{L:g}"] = {
                "abstain": float((dec == "abstain").mean()),
                "halluc": float((dec == "incorrect").mean()),
                "acc": float((dec == "correct").mean()),
            }
        covers, has_idk = [], []
        for cands in df["candidates"]:
            s = top_p_set(cands, P)
            covers.append(s.covers)
            has_idk.append(s.has_idk)
        covers = np.asarray(covers)
        has_idk = np.asarray(has_idk)
        entry["tables"]["set"] = {
            "strict": float(covers.mean()),
            "cov_idk": float((covers | has_idk).mean()),
            "idk_mass": float(df["log_idk_mass"].mean()),
            "no_concrete": int((df["n_concrete"] == 0).sum()),
            "unflagged_miss": float((~covers & ~has_idk).mean()),
        }
        entry["penalty"] = {}
        for L in PENALTIES:
            dpen = mf.load_penalty_arm(key, L)
            entry["penalty"][f"pen_L{L:g}"] = _penalty_decision_rates(dpen)
        report[label] = entry
    with open(out_dir / "headline_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print("== Headline values ==")
    for label, r in report.items():
        s = r["tables"]["set"]
        print(f"{label}: strict={s['strict']:.3f} cov_idk={s['cov_idk']:.3f} "
              f"idk_mass={s['idk_mass']:.3f} no_concrete={s['no_concrete']} "
              f"unflagged={s['unflagged_miss']:.3f}")


# ---------------------------------------------------------------------------
# Token costs (Table S4)
# ---------------------------------------------------------------------------

def token_costs(out_dir: Path) -> None:
    """Average answer-model token cost per question (input plus output).

    Log row: one elicited report per question.  Penalty rows: totals across
    the 50 samples per question at each level; the per-sample row divides
    the L=3 totals by 50."""
    rows = []
    for label, key, run, _cons in MODELS:
        entry = {"model": label}
        df = load_run(run)
        entry["log"] = float(df["log_total_tokens"].mean())
        for L in PENALTIES:
            dpen = mf.load_penalty_arm(key, L)
            entry[f"penalty_L{L:g}"] = float(
                pd.to_numeric(dpen["penalty_total_tokens"], errors="coerce").mean())
        entry["penalty_L3_per_sample"] = entry["penalty_L3"] / 50.0
        rows.append(entry)
        print(f"{label}: log={entry['log']:.0f} "
              f"L0={entry['penalty_L0']:,.0f} L3={entry['penalty_L3']:,.0f} "
              f"L6={entry['penalty_L6']:,.0f} L3/sample={entry['penalty_L3_per_sample']:.0f}")
    pd.DataFrame(rows).to_csv(out_dir / "token_costs.csv", index=False)
    print("wrote", out_dir / "token_costs.csv")


# ---------------------------------------------------------------------------
# Grader-agreement audit between the two shipped trees
# ---------------------------------------------------------------------------

def _iter_grade_pairs(path_a: Path, path_b: Path, field: str):
    """Yield (grade_a, grade_b, question_id, row_index) for matched candidates."""
    with open_result(path_a) as fa, open_result(path_b) as fb:
        for i, (la, lb) in enumerate(zip(fa, fb)):
            la, lb = la.strip(), lb.strip()
            if not la or not lb:
                continue
            ra, rb = json.loads(la), json.loads(lb)
            if str(ra.get("question_id")) != str(rb.get("question_id")):
                raise ValueError(f"Row misalignment at line {i} of {path_a.name}")
            raw_a, raw_b = ra.get(field) or "[]", rb.get(field) or "[]"
            ca = json.loads(raw_a) if isinstance(raw_a, str) else raw_a
            cb = json.loads(raw_b) if isinstance(raw_b, str) else raw_b
            if len(ca) != len(cb):
                raise ValueError(f"Candidate-count mismatch at line {i} of {path_a.name}")
            for item_a, item_b in zip(ca, cb):
                yield (str(item_a.get("grade", "")), str(item_b.get("grade", "")),
                       str(ra.get("question_id")), i)


def grader_agreement(out_dir: Path) -> None:
    """Candidate-grade agreement and top-0.9 coverage flips between the
    OpenAI-graded (primary) and Gemini-graded (second grader) trees."""
    tree_a = tree_for_grader("openai")
    tree_b = tree_for_grader("gemini")
    rows = []
    for label, key, run, cons in MODELS:
        files = [
            (Path(run) / "simpleqa_topp_results.jsonl", "log_candidates_json"),
            (Path(run) / "simpleqa_penalty_results.jsonl", "penalty_distribution_json"),
            (Path(f"{run}_L0") / "simpleqa_penalty_results.jsonl", "penalty_distribution_json"),
            (Path(f"{run}_L6") / "simpleqa_penalty_results.jsonl", "penalty_distribution_json"),
            (Path(cons) / "simpleqa_log_consistency_results.jsonl", "log_candidates_json"),
        ]
        agree = total = 0
        confusion: dict[tuple[str, str], int] = {}
        for rel, field in files:
            pa = result_file(tree_a / rel)
            pb = result_file(tree_b / rel)
            if pa is None or pb is None:
                raise FileNotFoundError(tree_a / rel)
            for ga, gb, _qid, _i in _iter_grade_pairs(pa, pb, field):
                total += 1
                agree += ga == gb
                if ga != gb:
                    confusion[(gb, ga)] = confusion.get((gb, ga), 0) + 1
        # coverage flips on the log run
        da, db = {}, {}
        for tree, store in [(tree_a, da), (tree_b, db)]:
            common.OUTPUTS_DIR = tree
            df = load_run(run)
            for _, r in df.iterrows():
                store[r["question_id"]] = top_p_set(r["candidates"], P).covers
        flips = sum(da[q] != db[q] for q in da)
        rows.append({
            "model": label,
            "candidate_grades_compared": total,
            "agreement": agree / total,
            "coverage_flips": flips,
            "coverage_flip_rate": flips / len(da),
            "disagreements_gemini_to_openai": json.dumps(
                {f"{k[0]}->{k[1]}": v for k, v in sorted(confusion.items())}),
        })
        r = rows[-1]
        print(f"{label}: agreement={r['agreement']:.4f} over {total:,} grades; "
              f"coverage flips {flips}/{len(da)} ({r['coverage_flip_rate']:.4%})")
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "grader_agreement.csv", index=False)
    pooled = tab["candidate_grades_compared"].to_numpy()
    print(f"pooled agreement: {np.average(tab['agreement'], weights=pooled):.4f}")
    print("wrote", out_dir / "grader_agreement.csv")


# ---------------------------------------------------------------------------

STEPS = {
    "audits": audits,
    "gaps": matched_gaps,
    "containment": support_containment,
    "tables": table_bodies,
    "headline": headline,
    "tokens": token_costs,
    "agreement": grader_agreement,
}
DEFAULT_STEPS = ["audits", "gaps", "containment", "tables", "headline", "tokens", "agreement"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grader", choices=["openai", "gemini"], default="openai",
                        help="Which graded result tree to analyze (default: openai).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", default=",".join(DEFAULT_STEPS),
                        help=f"Comma-separated subset of: {', '.join(STEPS)}")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise SystemExit(f"Unknown steps: {unknown}")
    for step in steps:
        set_results_tree(tree_for_grader(args.grader))
        print(f"\n### step: {step} (grader={args.grader}) ###")
        STEPS[step](args.out_dir)


if __name__ == "__main__":
    main()
