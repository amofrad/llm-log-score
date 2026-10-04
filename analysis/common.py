"""Shared loaders and post-hoc decision helpers for the analysis.

Reads the graded simpleqa_topp_results.jsonl files from the runner and
grader; every loader accepts either .jsonl or .jsonl.gz.

A candidate is a dict with keys answer, probability, points, and grade
(correct/incorrect/not_attempted). not_attempted entries are IDK; the rest
are concrete answers. Grades come from grading/graders.py.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = Path(
    os.environ.get("GRADED_RESULTS_DIR", REPO_ROOT / "results" / "graded_by_openai")
)
# size of the SimpleQA 'test' split; the 5 'few_shot' questions are appended after
SIMPLEQA_TEST_SPLIT_SIZE = 4321

RUNS = {
    "gemini35flash": {
        "run": "gemini35flash",
        "model_id": "google/gemini-3.5-flash",
        "label": "Gemini 3.5 Flash",
    },
    "sonnet46": {
        "run": "claudesonnet46",
        "model_id": "claude-sonnet-4-6",
        "label": "Claude Sonnet 4.6",
    },
    "deepseekv32maas": {
        "run": "deepseekv32",
        "model_id": "deepseek-ai/deepseek-v3.2-maas",
        "label": "DeepSeek V3.2",
    },
}

MODEL_ORDER = ["gemini35flash", "sonnet46", "deepseekv32maas"]
MODEL_COLORS = {
    "gemini35flash": "#1a73e8",
    "sonnet46": "#d97706",
    "deepseekv32maas": "#009E73",
}


def norm_answer(s: str) -> str:
    """Normalize an answer string for within-report matching."""
    return " ".join(str(s).lower().replace(".", " ").replace(",", " ").split())


def result_file(path: Path) -> Path | None:
    """Return the path, its .gz variant, or None if neither exists."""
    path = Path(path)
    if path.exists():
        return path
    gz = path.with_name(path.name + ".gz")
    if gz.exists():
        return gz
    return None


def open_result(path: Path):
    """Open a result file for reading, gzip or plain."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path)


@lru_cache(maxsize=1)
def simpleqa_full_question_order() -> tuple[str, ...]:
    """The runner's deterministic question order (shuffle seed 17).

    Result files are resume artifacts and may reorder rows, so running-prefix
    curves follow this order rather than file order.
    """
    question_ids = [f"simpleqa-{i}" for i in range(SIMPLEQA_TEST_SPLIT_SIZE)]
    random.Random(17).shuffle(question_ids)
    question_ids.extend(
        f"simpleqa-{i}" for i in range(SIMPLEQA_TEST_SPLIT_SIZE, SIMPLEQA_TEST_SPLIT_SIZE + 5)
    )
    return tuple(question_ids)


def load_jsonl(path: Path) -> list[dict]:
    resolved = result_file(path)
    if resolved is None:
        raise FileNotFoundError(path)
    records = []
    with open_result(resolved) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_candidates(rec: dict) -> list[dict]:
    """Parse log_candidates_json into candidate dicts, in reported order."""
    raw = rec.get("log_candidates_json")
    if not raw:
        return []
    cands = json.loads(raw) if isinstance(raw, str) else raw
    out = []
    for c in cands:
        prob = c.get("probability")
        if prob is None:
            continue
        out.append(
            {
                "answer": str(c.get("answer", "")),
                "probability": float(prob),
                "points": float(c.get("points", 0.0)),
                "grade": str(c.get("grade", "")),
            }
        )
    return out


def load_run(
    run_dir_name: str, *, results_dir: Path | None = None
) -> pd.DataFrame:
    """Load one run directory into a tidy per-question DataFrame."""
    tree = Path(results_dir) if results_dir is not None else OUTPUTS_DIR
    path = tree / run_dir_name / "simpleqa_topp_results.jsonl"
    records = load_jsonl(path)
    rows = []
    for rec in records:
        cands = parse_candidates(rec)
        concrete = [c for c in cands if c["grade"] != "not_attempted"]
        idk_mass = sum(c["probability"] for c in cands if c["grade"] == "not_attempted")
        q_true = sum(c["probability"] for c in concrete if c["grade"] == "correct")
        hallucination_mass = sum(
            c["probability"] for c in concrete if c["grade"] == "incorrect"
        )
        reported_mass = sum(c["probability"] for c in cands)
        top = max(cands, key=lambda c: c["probability"]) if cands else None
        rows.append(
            {
                "question_id": str(rec["question_id"]),
                "model": rec["model"],
                "category": rec.get("category"),
                "answer_type": rec.get("answer_type"),
                "gold_answer": rec.get("gold_answer"),
                "question": rec.get("question"),
                "candidates": cands,
                "n_candidates": len(cands),
                "n_concrete": len(concrete),
                "log_q_true": q_true,
                "log_hallucination_mass": hallucination_mass,
                "log_idk_mass": idk_mass,
                "log_reported_mass": reported_mass,
                "log_top_answer": top["answer"] if top else None,
                "log_top_grade": top["grade"] if top else None,
                "log_top_prob": top["probability"] if top else np.nan,
                "log_total_tokens": _float(rec.get("log_total_tokens")),
                "log_generation_error": rec.get("log_generation_error"),
                "empirical_accuracy_overall": _float(rec.get("empirical_accuracy_overall")),
                "empirical_hallucination_rate": _float(rec.get("empirical_hallucination_rate")),
                "empirical_not_attempted_rate": _float(rec.get("empirical_not_attempted_rate")),
                "empirical_total_tokens": _float(rec.get("empirical_total_tokens")),
                "penalty_value": _float(rec.get("penalty_value")),
                "penalty_threshold": _float(rec.get("penalty_threshold")),
                "penalty_accuracy_overall": _float(rec.get("penalty_accuracy_overall")),
                "penalty_accuracy_when_answered": _float(rec.get("penalty_accuracy_when_answered")),
                "penalty_hallucination_rate": _float(rec.get("penalty_hallucination_rate")),
                "penalty_abstention_rate": _float(rec.get("penalty_abstention_rate")),
                "penalty_total_tokens": _float(rec.get("penalty_total_tokens")),
                "imported_from": rec.get("imported_from"),
                "log_idk_rule": rec.get("log_idk_rule"),
                "log_idk_rho": _float(rec.get("log_idk_rho")),
                "log_prompt_variant": rec.get("log_prompt_variant"),
                "simpleqa_top_p": _float(rec.get("simpleqa_top_p")),
            }
        )
    df = pd.DataFrame(rows)
    df["run_dir"] = run_dir_name
    return df


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson confidence interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(
        phat * (1 - phat) / n + z * z / (4 * n * n)
    ) / denom
    return (center - half, center + half)


def _float(x):
    if x is None or x == "":
        return np.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


# Confidence sets and post-hoc decisions
def sorted_candidates(cands: list[dict]) -> list[dict]:
    """Stable-sort candidates by reported probability, descending."""
    return sorted(cands, key=lambda c: -c["probability"])


@dataclass
class TopPSet:
    members: list[dict] = field(default_factory=list)
    cum_mass: float = 0.0

    @property
    def covers(self) -> bool:
        return any(c["grade"] == "correct" for c in self.members)

    @property
    def has_idk(self) -> bool:
        return any(c["grade"] == "not_attempted" for c in self.members)

    @property
    def n_concrete(self) -> int:
        return sum(1 for c in self.members if c["grade"] != "not_attempted")


def top_p_set(cands: list[dict], p: float) -> TopPSet:
    """Smallest prefix of probability-sorted candidates with cumulative
    reported mass >= p (or all candidates if total reported mass < p)."""
    s = TopPSet()
    for c in sorted_candidates(cands):
        if s.cum_mass >= p:
            break
        s.members.append(c)
        s.cum_mass += c["probability"]
    return s


REPORT_METRICS = (
    "list_size",
    "idk_mass",
    "top_concrete_probability",
    "truth_in_list",
    "coverage",
    "coverage_or_idk",
    "incorrect",
)


def report_metrics(df: pd.DataFrame, p: float = 0.9) -> pd.DataFrame:
    """Compute the report summaries used by the sensitivity analyses."""
    rows = []
    for _, record in df.iterrows():
        candidates = record["candidates"]
        concrete = [
            candidate
            for candidate in candidates
            if candidate["grade"] != "not_attempted"
        ]
        prefix = top_p_set(candidates, p)
        rows.append(
            {
                "question_id": record["question_id"],
                "list_size": len(concrete),
                "idk_mass": record["log_idk_mass"],
                "top_concrete_probability": max(
                    (candidate["probability"] for candidate in concrete),
                    default=np.nan,
                ),
                "truth_in_list": any(
                    candidate["grade"] == "correct" for candidate in candidates
                ),
                "coverage": prefix.covers,
                "coverage_or_idk": prefix.covers or prefix.has_idk,
                "incorrect": not prefix.covers and not prefix.has_idk,
            }
        )
    return pd.DataFrame(rows)


def posthoc_decision(cands: list[dict], t: float) -> str:
    """Offline penalty-rubric decision on the elicited distribution.

    Answer with the top concrete candidate if its probability >= t (t=0
    always answers), else abstain. Returns correct, incorrect, or abstain.
    """
    concrete = [c for c in cands if c["grade"] != "not_attempted"]
    if not concrete:
        return "abstain"
    top = max(concrete, key=lambda c: c["probability"])
    if t > 0 and top["probability"] < t:
        return "abstain"
    return "correct" if top["grade"] == "correct" else "incorrect"


def frontier(df: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    """Post-hoc decision frontier over a threshold grid.

    For each t: abstention rate, accuracy, hallucination rate,
    accuracy-when-answered, and mean abstention-reward score
    s_t = 1[correct] + t * 1[abstain].
    """
    rows = []
    for t in thresholds:
        outcomes = df["candidates"].map(lambda c, t=t: posthoc_decision(c, t))
        n = len(outcomes)
        n_corr = (outcomes == "correct").sum()
        n_inc = (outcomes == "incorrect").sum()
        n_abs = (outcomes == "abstain").sum()
        answered = n_corr + n_inc
        rows.append(
            {
                "t": t,
                "abstention_rate": n_abs / n,
                "accuracy_overall": n_corr / n,
                "hallucination_rate": n_inc / n,
                "accuracy_when_answered": (n_corr / answered) if answered else np.nan,
                "mean_score_st": (n_corr + t * n_abs) / n,
            }
        )
    return pd.DataFrame(rows)
