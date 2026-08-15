"""Apply a grader to a tree of run results, producing a graded tree.

Reads every run directory under --runs, collects the unique factual
(question, answer) pairs across all uncertainty-report candidates and
penalty-arm samples, grades each pair once with the selected grader, and
writes a parallel tree under --out with the same directory layout and file
schema, all grades replaced, and the derived per-record fields recomputed.
A manifest (GRADING_MANIFEST.json) records the grader and row counts.

    python grading/grade.py --grader openai --runs results/graded_by_gemini \\
        --out /tmp/graded_tree

    python grading/grade.py --grader gemini --runs <tree> --out <tree2> \\
        --gcp-project <project> --gcp-location <location>

Grading never feeds back into generation, so it is a pure post-processing
stage: the input tree's grades (if any) are ignored, and any tree with the
runner's file schema works as input. Mechanical abstentions (the explicit
abstain token and I-don't-know variants) are caught without a model call.
Grades are checkpointed to <out>/grades.jsonl, so an interrupted run resumes
without re-grading.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "runner"))

from graders import (  # noqa: E402
    GeminiGrader,
    OpenAIGrader,
    is_open_not_attempted,
    open_answer_key,
)

RESULT_FILES = (
    "simpleqa_topp_results.jsonl",
    "simpleqa_penalty_results.jsonl",
    "simpleqa_log_consistency_results.jsonl",
)


@dataclass(frozen=True)
class Example:
    question: str
    answer: str


def candidate_field(filename: str) -> str:
    return "penalty_distribution_json" if "penalty_results" in filename else "log_candidates_json"


def open_maybe_gz(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path)


def discover_sources(tree: Path) -> list[Path]:
    sources = []
    for run_dir in sorted(p for p in tree.iterdir() if p.is_dir()):
        for name in RESULT_FILES:
            for candidate in (run_dir / name, run_dir / f"{name}.gz"):
                if candidate.is_file():
                    sources.append(candidate)
                    break
    if not sources:
        raise SystemExit(f"No result files found under {tree}")
    return sources


def iter_records(path: Path):
    with open_maybe_gz(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def item_key(question_id: str, answer: str) -> str:
    return f"{question_id}\x1f{open_answer_key(answer)}"


def collect_items(sources: list[Path]) -> dict[str, dict]:
    """One grading item per unique (question, canonical answer) pair."""
    items: dict[str, dict] = {}
    for path in sources:
        field = candidate_field(path.name)
        for record in iter_records(path):
            example = Example(
                question=str(record["question"]),
                answer=str(record["gold_answer"]),
            )
            raw = record.get(field) or "[]"
            candidates = json.loads(raw) if isinstance(raw, str) else list(raw)
            for cand in candidates:
                answer = str(cand.get("answer", ""))
                if cand.get("grade") == "not_attempted" or is_open_not_attempted(answer):
                    continue
                key = item_key(str(record["question_id"]), answer)
                if key not in items:
                    items[key] = {"example": example, "answer": answer}
                elif (items[key]["example"].question != example.question
                      or items[key]["example"].answer != example.answer):
                    raise ValueError(f"Key collision with differing source text: {key!r}")
    return items


def load_checkpoint(path: Path) -> dict[str, str]:
    grades: dict[str, str] = {}
    if path.is_file():
        for record in iter_records(path):
            grades[record["key"]] = record["grade"]
    return grades


def grade_items(items: dict[str, dict], grader, checkpoint: Path, workers: int) -> dict[str, str]:
    grades = load_checkpoint(checkpoint)
    pending = {k: v for k, v in items.items() if k not in grades}
    print(f"{len(items):,} unique items; {len(grades):,} checkpointed; {len(pending):,} to grade")
    if not pending:
        return grades
    lock = threading.Lock()
    done = 0
    with open(checkpoint, "a") as sink, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(grader.grade, item["example"], item["answer"]): key
            for key, item in pending.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            label = future.result()
            with lock:
                grades[key] = label
                sink.write(json.dumps({"key": key, "grade": label}) + "\n")
                sink.flush()
                done += 1
                if done % 500 == 0:
                    print(f"  graded {done:,}/{len(pending):,}")
    return grades


def update_candidates(record: dict, field: str, grades: dict[str, str]) -> list[dict]:
    raw = record.get(field) or "[]"
    candidates = json.loads(raw) if isinstance(raw, str) else list(raw)
    for candidate in candidates:
        answer = str(candidate.get("answer", ""))
        if candidate.get("grade") == "not_attempted" or is_open_not_attempted(answer):
            candidate["grade"] = "not_attempted"
            continue
        candidate["grade"] = grades[item_key(str(record["question_id"]), answer)]
    record[field] = json.dumps(candidates, ensure_ascii=True)
    return candidates


def refresh_log_fields(record: dict, candidates: list[dict]) -> None:
    concrete = [c for c in candidates if c.get("grade") != "not_attempted"]
    q_true = sum(float(c.get("probability", 0.0)) for c in concrete if c.get("grade") == "correct")
    halluc = sum(float(c.get("probability", 0.0)) for c in concrete if c.get("grade") == "incorrect")
    not_attempted = sum(
        float(c.get("probability", 0.0)) for c in candidates if c.get("grade") == "not_attempted"
    )
    top = max(candidates, key=lambda c: float(c.get("probability", 0.0)), default=None)
    record["log_q_true"] = q_true
    record["log_hallucination_mass"] = halluc
    record["log_not_attempted_mass"] = not_attempted
    record["log_top_answer"] = top.get("answer") if top else None
    record["log_top_grade"] = top.get("grade") if top else None
    record["log_top_prob"] = float(top.get("probability", 0.0)) if top else None


def refresh_penalty_fields(record: dict, candidates: list[dict]) -> None:
    counts = {"correct": 0, "incorrect": 0, "not_attempted": 0}
    for candidate in candidates:
        grade = str(candidate.get("grade", "incorrect"))
        counts[grade if grade in counts else "incorrect"] += int(candidate.get("count", 0) or 0)
    original_abstain = int(record.get("penalty_abstain_samples") or 0)
    requested = int(record.get("n_samples_requested") or (sum(counts.values()) + original_abstain) or 1)
    answered = counts["correct"] + counts["incorrect"]
    record["penalty_grade_counts_json"] = json.dumps(counts, ensure_ascii=True)
    record["penalty_correct_samples"] = counts["correct"]
    record["penalty_incorrect_samples"] = counts["incorrect"]
    record["penalty_not_attempted_samples"] = counts["not_attempted"]
    record["penalty_answered_samples"] = answered
    record["penalty_abstain_samples"] = original_abstain
    record["penalty_accuracy_overall"] = counts["correct"] / requested
    record["penalty_hallucination_rate"] = counts["incorrect"] / requested
    record["penalty_abstention_rate"] = original_abstain / requested
    record["penalty_accuracy_when_answered"] = counts["correct"] / answered if answered else None


def transform_file(source: Path, destination: Path, grades: dict[str, str]) -> int:
    field = candidate_field(source.name.replace(".gz", ""))
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open_maybe_gz(source) as read_handle, gzip.open(destination, "wt") as write_handle:
        for line in read_handle:
            if not line.strip():
                continue
            record = json.loads(line)
            candidates = update_candidates(record, field, grades)
            if field == "log_candidates_json":
                refresh_log_fields(record, candidates)
            else:
                refresh_penalty_fields(record, candidates)
            write_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            rows += 1
    return rows


def build_grader(args):
    if args.grader == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENAI_API_KEY to use the OpenAI grader.")
        return OpenAIGrader(api_key, model=args.grader_model or "gpt-5.6-terra")
    if not (args.gcp_project and args.gcp_location):
        raise SystemExit("--gcp-project and --gcp-location are required for the Gemini grader.")
    from engine import create_gcp_client_for_model  # noqa: E402 (heavy import)

    model = args.grader_model or "google/gemini-3.5-flash"
    client = create_gcp_client_for_model(
        model=model, project=args.gcp_project, location=args.gcp_location
    )
    return GeminiGrader(client, model=model)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grader", choices=["openai", "gemini"], required=True)
    parser.add_argument("--runs", type=Path, required=True,
                        help="Input tree of run directories (grades ignored).")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output tree; must not already contain result files.")
    parser.add_argument("--grader-model", default=None,
                        help="Override the default grader model id.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gcp-project", default=None)
    parser.add_argument("--gcp-location", default=None)
    args = parser.parse_args(argv)

    sources = discover_sources(args.runs)
    print(f"{len(sources)} result files under {args.runs}")
    items = collect_items(sources)
    args.out.mkdir(parents=True, exist_ok=True)
    grader = build_grader(args)
    grades = grade_items(items, grader, args.out / "grades.jsonl", args.workers)

    manifest = {"grader": args.grader, "grader_model": grader.model, "sources": []}
    for source in sources:
        rel = source.relative_to(args.runs)
        rel_out = str(rel).replace(".gz", "") + ".gz"
        rows = transform_file(source, args.out / rel_out, grades)
        manifest["sources"].append({"source": str(rel).replace(".gz", ""), "rows": rows})
        print(f"  wrote {rel_out} ({rows:,} rows)")
    (args.out / "GRADING_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Done. Graded tree at {args.out}")


if __name__ == "__main__":
    main()
