"""Command-line runner for the SimpleQA elicitation experiments. Three
subcommands, all built on engine.py: simpleqa (single-report log elicitation
with an optional penalty arm), consistency (repeated-report runs), and batch
(GCP batch prediction that feeds the shared response cache)."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine  # noqa: E402

# simpleqa subcommand: one log-elicitation call per question (temperature 0),
# scored by the residual-rho rule in the prompt, plus an optional penalty arm
# (--include-penalty). Sampling is deterministic (shuffle seed 17); runs
# resume from the checkpoint in --out-dir.
#
#   GOOGLE_CLOUD_PROJECT=... python run.py simpleqa \
#       --models claude-sonnet-4-6 --idk-rho 0.5 \
#       --include-penalty --penalty 3 --out-dir outputs/claudesonnet46

SAMPLING_SEED = 17
# size of the SimpleQA 'test' split; the 5 'few_shot' questions are appended after
SIMPLEQA_TEST_SPLIT_SIZE = 4321


def parse_simpleqa_splits(value: str) -> list[str]:
    """Parse a split expression, keeping the test split first.

    SimpleQA ships a 4,321-question test split and a 5-question few_shot
    split. For test+few_shot/all, test loads first and few_shot is appended,
    so earlier prefixes stay comparable to the test-only runs.
    """
    normalized = value.strip()
    if normalized in {"all", "full", "test+few_shot", "test,few_shot"}:
        return ["test", "few_shot"]
    parts = [part.strip() for part in normalized.replace(",", "+").split("+")]
    return [part for part in parts if part]


def canonical_simpleqa_question_id(question_id: object) -> str:
    qid = str(question_id)
    prefix = "few_shot-simpleqa-"
    if qid.startswith(prefix):
        suffix = qid[len(prefix):]
        if suffix.isdigit():
            return f"simpleqa-{SIMPLEQA_TEST_SPLIT_SIZE + int(suffix)}"
    return qid


def canonicalize_simpleqa_row_question_id(row: dict) -> dict:
    canonical = canonical_simpleqa_question_id(row.get("question_id", ""))
    if canonical == row.get("question_id"):
        return row
    updated = dict(row)
    updated["question_id"] = canonical
    return updated


def rewrite_jsonl_with_canonical_question_ids(path: Path) -> None:
    if not path.exists():
        return
    rows = []
    changed = False
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        updated = canonicalize_simpleqa_row_question_id(row)
        changed = changed or updated is not row
        rows.append(updated)

    dedup: dict[tuple, dict] = {}
    order: list[tuple] = []
    for row in rows:
        key = (
            row.get("model"),
            row.get("question_id"),
            row.get("seed"),
            row.get("penalty_value"),
        )
        if key not in dedup:
            order.append(key)
        dedup[key] = row
    canonical_rows = [dedup[key] for key in order]
    if len(canonical_rows) != len(rows):
        changed = True

    if changed:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in canonical_rows)
        )
        print(f"[simpleqa ids] normalized question IDs in {path}")


def examples_with_appended_split_ids(
    examples: list["engine.SimpleQAExample"],
    *,
    start_index: int,
) -> list["engine.SimpleQAExample"]:
    out = []
    for offset, example in enumerate(examples):
        out.append(
            engine.SimpleQAExample(
                question_id=f"simpleqa-{start_index + offset}",
                question=example.question,
                answer=example.answer,
                category=example.category,
                answer_type=example.answer_type,
            )
        )
    return out


def load_simpleqa_examples_for_run(args) -> list["engine.SimpleQAExample"]:
    split_names = parse_simpleqa_splits(args.split)
    if not split_names:
        raise SystemExit("--split must name at least one SimpleQA split.")
    if len(split_names) == 1:
        return engine.load_simpleqa(
            split=split_names[0],
            num_samples=args.num_samples,
            seed=SAMPLING_SEED,
            categories=None,
            sample_per_category=None,
        )

    combined: list[engine.SimpleQAExample] = []
    remaining = args.num_samples
    for split in split_names:
        if remaining is not None and remaining <= 0:
            break
        start_index = len(combined)
        split_examples = engine.load_simpleqa(
            split=split,
            num_samples=remaining,
            seed=SAMPLING_SEED,
            categories=None,
            sample_per_category=None,
        )
        if split != "test":
            split_examples = examples_with_appended_split_ids(
                split_examples,
                start_index=start_index,
            )
        combined.extend(split_examples)
        if remaining is not None:
            remaining -= len(split_examples)

    seen: set[str] = set()
    duplicates: list[str] = []
    for example in combined:
        if example.question_id in seen:
            duplicates.append(example.question_id)
        seen.add(example.question_id)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise SystemExit(f"Combined SimpleQA split has duplicate question IDs: {preview}")
    return combined



# Rule bookkeeping

def annotate_results(results_path: Path, *, rho: float, eps: float) -> None:
    """Add log_idk_rule / log_idk_rho / realized residual-score fields."""
    if not results_path.exists():
        return
    rows = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["log_idk_rule"] = "residual"
            row["log_idk_rho"] = rho
            q_true = float(row.get("log_q_true") or 0.0)
            idk_mass = float(row.get("log_not_attempted_mass") or 0.0)
            try:
                candidates = json.loads(row.get("log_candidates_json") or "[]")
            except json.JSONDecodeError:
                candidates = []
            any_correct = any(c.get("grade") == "correct" for c in candidates)
            realized = q_true if any_correct else rho * idk_mass
            row["log_report_residual_score"] = math.log(max(eps, realized))
            row["log_report_residual_shifted_score"] = (
                row["log_report_residual_score"] + engine.LOG_SCORE_SHIFT
            )
            rows.append(row)
    with open(results_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def first_row_rho(path: Path) -> tuple[float | None, bool]:
    """Return (log_idk_rho, has_rows) for the first row of a jsonl file."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    return (row.get("log_idk_rho"), True)
    except FileNotFoundError:
        pass
    return (None, False)


def check_resume_compatibility(
    out_dir: Path, results_path: Path, partial_path: Path, *, rho: float
) -> None:
    """Refuse to resume into an out-dir whose rows or marker carry a different
    rho. A partial checkpoint, written before annotation, has no rho and is
    accepted."""
    marker = out_dir / "_rule_marker.json"
    if marker.exists():
        m = json.loads(marker.read_text())
        found = m.get("idk_rho")
        if found is None or abs(found - rho) > 1e-9:
            raise SystemExit(
                f"Refusing resume into {out_dir}: its marker says rho={found} "
                f"but this run uses rho={rho}. Use a separate --out-dir per rho."
            )
        return
    for path in (results_path, partial_path):
        found, has_rows = first_row_rho(path)
        if found is not None:
            if abs(found - rho) > 1e-9:
                raise SystemExit(
                    f"Refusing resume from {path}: rows were produced with "
                    f"rho={found} but this run uses rho={rho}. "
                    f"Use a separate --out-dir per rho."
                )
            return


def write_rule_marker(out_dir: Path, *, rho: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_rule_marker.json").write_text(
        json.dumps({"log_idk_rule": "residual", "idk_rho": rho}) + "\n"
    )



# Penalty arm (optional; --include-penalty)

def run_penalty_question(
    example, *, model, answer_client, grader_client, grader_model, use_grader,
    mock, seed, penalty, n_samples, temperature, answer_max_tokens,
    empty_content_retries, json_only_prompts, gcp_reasoning_effort,
):
    """Run the penalty arm for one question: sample the answer-or-decline
    rubric n_samples times and grade each sample."""
    penalty_prompt = (
        engine.build_simpleqa_penalty_json_prompt(example, penalty)
        if json_only_prompts
        else engine.build_simpleqa_penalty_prompt(example, penalty)
    )
    penalty_system_message = engine.simpleqa_penalty_system_message(
        strict_json=json_only_prompts
    )
    penalty_prompt_format = "json" if json_only_prompts else "final_line"
    grade_cache: dict[str, str] = {}
    counts: dict[str, int] = {}
    display: dict[str, str] = {}
    preview: list[str] = []
    usage = engine.empty_token_usage_totals()
    abstain_count = 0
    generation_error_count = 0
    empty_retry_count = 0

    for sample_i in range(n_samples):
        sample_seed = seed * 1_000_000 + 200_000 + sample_i
        if mock:
            rng = engine.stable_rng("simpleqa-penalty", example.question_id, model, sample_seed)
            response = (
                f"Final: {engine.ABSTAIN_LABEL}"
                if rng.random() < engine.threshold_from_penalty(penalty)
                else f"Final: {example.answer}"
            )
        else:
            try:
                response, retry_count = engine.generate_with_empty_content_retries(
                    answer_client,
                    model=model,
                    prompt=penalty_prompt,
                    system_message=penalty_system_message,
                    seed=sample_seed,
                    temperature=temperature,
                    max_tokens=answer_max_tokens,
                    empty_content_retries=empty_content_retries,
                )
                empty_retry_count += retry_count
            except engine.EmptyAssistantContentError:
                generation_error_count += 1
                empty_retry_count += max(0, empty_content_retries)
                response = f"Final: {engine.ABSTAIN_LABEL}"
        engine.add_token_usage(
            usage,
            engine.generation_token_usage(
                client=None if mock else answer_client,
                prompt=penalty_prompt,
                system_message=penalty_system_message,
                response=response,
            ),
        )
        if len(preview) < 20:
            preview.append(response)
        parsed = engine.parse_final_text_response(response) or engine.ABSTAIN_LABEL
        answer = engine.normalize_open_answer_text(parsed)
        if (
            engine.open_answer_key(answer) == engine.open_answer_key(engine.ABSTAIN_LABEL)
            or engine.is_open_not_attempted(answer)
        ):
            abstain_count += 1
        else:
            key = engine.open_answer_key(answer)
            if not key:
                abstain_count += 1
                continue
            counts[key] = counts.get(key, 0) + 1
            display.setdefault(key, answer)

    distribution = engine.simpleqa_answer_distribution_from_counts(counts, display, n_samples)
    engine.simpleqa_grade_counts_from_distribution(
        distribution,
        example=example,
        grade_cache=grade_cache,
        grader_client=grader_client,
        grader_model=grader_model,
        use_grader=use_grader,
        mock=mock,
        seed=seed + 17,
    )
    distribution = engine.canonicalize_simpleqa_distribution(distribution, example=example)
    grade_counts = engine.simpleqa_grade_counts_from_graded_distribution(distribution)
    answered = sum(int(row.get("count", 0) or 0) for row in distribution)
    correct = grade_counts.get("correct", 0)
    incorrect = grade_counts.get("incorrect", 0)
    return {
        "question_id": example.question_id,
        "category": example.category,
        "answer_type": example.answer_type,
        "model": model,
        "seed": seed,
        "n_samples_requested": n_samples,
        "gcp_reasoning_effort": engine.effective_gcp_reasoning_effort(
            model, gcp_reasoning_effort
        ),
        "gold_answer": example.answer,
        "question": example.question,
        "penalty_value": penalty,
        "penalty_threshold": engine.threshold_from_penalty(penalty),
        "penalty_prompt_format": penalty_prompt_format,
        "penalty_system_message": penalty_system_message,
        "penalty_distribution_json": json.dumps(distribution, ensure_ascii=True),
        "penalty_grade_counts_json": json.dumps(grade_counts, ensure_ascii=True),
        "penalty_answered_samples": answered,
        "penalty_abstain_samples": abstain_count,
        "penalty_correct_samples": correct,
        "penalty_incorrect_samples": incorrect,
        "penalty_not_attempted_samples": grade_counts.get("not_attempted", 0),
        "penalty_accuracy_overall": correct / max(1, n_samples),
        "penalty_accuracy_when_answered": (correct / answered) if answered else None,
        "penalty_hallucination_rate": incorrect / max(1, n_samples),
        "penalty_abstention_rate": abstain_count / max(1, n_samples),
        "penalty_sample_response_preview_json": json.dumps(preview, ensure_ascii=True),
        "penalty_input_tokens": usage.get("input_tokens"),
        "penalty_output_tokens": usage.get("output_tokens"),
        "penalty_total_tokens": usage.get("total_tokens"),
        "penalty_token_source_counts_json": json.dumps(
            usage.get("source_counts", {}), ensure_ascii=True
        ),
        "penalty_generation_error_count": generation_error_count,
        "penalty_empty_content_retry_count": empty_retry_count,
    }


def print_settings(args, examples, rho, effort, n_pen_done=0, n_pen_jobs=0) -> None:
    """Print the complete effective configuration of this run."""
    bar = "=" * 72
    arms = []
    if not args.skip_log:
        arms.append("log elicitation (residual rule)")
    if args.include_penalty:
        arms.append(f"penalty L={args.penalty:g}")
    print(bar)
    print("SIMPLEQA RUN SETTINGS")
    print(bar)
    print(f"  arms                  : {', '.join(arms)}")
    print(f"  models                : {', '.join(args.models)}")
    fam = ("anthropic rawPredict" if not args.mock and engine.is_gcp_anthropic_model(args.models[0])
           else "openai chat completions" if not args.mock else "mock")
    print(f"  provider              : GCP Vertex ({fam})")
    grader_location = args.grader_gcp_location or args.gcp_location
    print(f"  gcp answer location   : {args.gcp_project} / {args.gcp_location}")
    print(f"  gcp grader location   : {args.gcp_project} / {grader_location}")
    print(f"  gcp reasoning effort  : {effort}")
    if any(engine.is_gcp_deepseek_model(model) for model in args.models):
        print(f"  deepseek thinking     : {'disabled' if args.disable_thinking else 'enabled'} "
              "(GCP chat_template_kwargs)")
    print(f"  JSON-only prompts     : {args.json_only_prompts}")
    print("-" * 72)
    print(f"  dataset / split       : OpenEvals/SimpleQA / {args.split}")
    print(f"  questions             : {len(examples)} "
          f"(deterministic shuffle seed {SAMPLING_SEED}; nested across runs)")
    if examples:
        print(f"  first / last question : {examples[0].question_id} / {examples[-1].question_id}")
    print(f"  generation seed       : {args.seed}")
    print(f"  grader                : {args.grader_model} (use_grader={args.use_grader}, "
          f"temperature=0, max_tokens={engine.DEFAULT_GRADER_MAX_TOKENS})")
    print(
        f"  cache                 : {args.cache_path} "
        f"(cache_only={args.cache_only}, answer_cache_only={args.answer_cache_only})"
    )
    print(f"  out dir               : {args.out_dir}")
    if not args.skip_log:
        print("-" * 72)
        print("  LOG ELICITATION ARM (1 call per question, temperature 0)")
        print(f"    scoring rule        : residual (rho={rho:g})")
        print(f"    top-p               : {args.simpleqa_top_p:g} "
              f"(points must sum to >= {100 * args.simpleqa_top_p:g})")
        print(f"    log max tokens      : {args.log_max_tokens}")
        log_system_messages = sorted({
            engine.simpleqa_log_system_message_for_model(
                model,
                strict_json=args.json_only_prompts,
            )
            for model in args.models
        })
        if len(log_system_messages) == 1:
            print(f"    system message      : {log_system_messages[0]!r}")
        else:
            print(f"    system message      : mixed per model ({len(log_system_messages)} variants)")
    if args.include_penalty:
        print("-" * 72)
        print("  PENALTY ARM (Kalai et al. rubric prompt, sampled per question)")
        print(f"    penalty L           : {args.penalty:g} "
              f"(rubric threshold t = L/(1+L) = "
              f"{engine.threshold_from_penalty(args.penalty):.3f})")
        print(f"    samples per question: {args.penalty_samples}")
        print(f"    temperature         : {args.penalty_temperature:g}")
        print(f"    answer max tokens   : {args.penalty_max_tokens}")
        print(f"    disable thinking    : {args.disable_thinking} (part of cache key; "
              f"matches the original runs)")
        print(f"    per-sample seed     : seed*1_000_000 + 200_000 + i "
              f"(sample 0 -> {args.seed * 1_000_000 + 200_000})")
        penalty_system_message = engine.simpleqa_penalty_system_message(
            strict_json=args.json_only_prompts
        )
        print(f"    response format     : {'JSON object' if args.json_only_prompts else 'Final: line'}")
        print(f"    system message      : {penalty_system_message!r}")
        print(f"    max workers         : {args.max_workers}")
        print(f"    resume              : {n_pen_done} questions already done, "
              f"{n_pen_jobs} to run "
              f"({n_pen_jobs} x {args.penalty_samples} = "
              f"{n_pen_jobs * args.penalty_samples} answer calls; cached ones free)")
        if examples:
            print("    exact penalty prompt for the first question:")
            penalty_prompt_preview = (
                engine.build_simpleqa_penalty_json_prompt(examples[0], args.penalty)
                if args.json_only_prompts
                else engine.build_simpleqa_penalty_prompt(examples[0], args.penalty)
            )
            for line in penalty_prompt_preview.splitlines():
                print(f"    | {line}")
    print(bar, flush=True)


def run_penalty_arm(args, examples, effort) -> None:
    """Run the penalty arm over the given questions, with resume."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "simpleqa_penalty_results.jsonl"
    rewrite_jsonl_with_canonical_question_ids(results_path)
    done = set()
    expected_prompt_format = "json" if args.json_only_prompts else "final_line"
    expected_system_message = engine.simpleqa_penalty_system_message(
        strict_json=args.json_only_prompts
    )
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing_penalty = r.get("penalty_value")
                if existing_penalty is not None and not math.isclose(
                    float(existing_penalty), float(args.penalty), rel_tol=0.0, abs_tol=1e-9
                ):
                    raise SystemExit(
                        f"Refusing resume into {args.out_dir}: "
                        f"{results_path.name} contains penalty L={float(existing_penalty):g}, "
                        f"but this run uses L={float(args.penalty):g}. "
                        "Use a separate --out-dir per penalty level."
                    )
                existing_prompt_format = r.get("penalty_prompt_format") or "final_line"
                existing_system_message = (
                    r.get("penalty_system_message")
                    or engine.OPEN_ANSWER_SYSTEM_MESSAGE
                )
                existing_effort = engine.normalize_gcp_reasoning_effort(
                    r.get("gcp_reasoning_effort")
                )
                expected_effort = engine.effective_gcp_reasoning_effort(
                    str(r.get("model") or args.models[0]), effort
                )
                if existing_effort != expected_effort:
                    raise SystemExit(
                        f"Refusing resume into {args.out_dir}: "
                        f"{results_path.name} contains penalty rows produced with "
                        f"gcp_reasoning_effort={existing_effort!r}, but this run uses "
                        f"gcp_reasoning_effort={expected_effort!r}. Use a separate "
                        "--out-dir or archive the incompatible rows first."
                    )
                if (
                    existing_prompt_format != expected_prompt_format
                    or existing_system_message != expected_system_message
                ):
                    raise SystemExit(
                        f"Refusing resume into {args.out_dir}: "
                        f"{results_path.name} contains penalty rows produced with "
                        f"prompt_format={existing_prompt_format!r}, but this run uses "
                        f"prompt_format={expected_prompt_format!r}. Use a separate "
                        "--out-dir for JSON-only prompt runs so existing results are preserved."
                    )
                done.add((r["model"], r["question_id"]))
    requested = [(m, ex) for m in args.models for ex in examples]
    jobs = [(m, ex) for m, ex in requested if (m, ex.question_id) not in done]
    done_in_scope = len(requested) - len(jobs)
    print_settings(args, examples, args.idk_rho,
                   effort, n_pen_done=done_in_scope, n_pen_jobs=len(jobs))
    if not jobs:
        print("[penalty] nothing to do")
        return

    answer_client = None
    grader_client = None
    if not args.mock:
        grader_location = args.grader_gcp_location or args.gcp_location
        client_kwargs = dict(
            project=args.gcp_project,
            location=args.gcp_location,
            cache=engine.SQLiteCache(args.cache_path),
            request_delay_s=args.request_delay_s,
            max_retries=args.max_retries,
            retry_base_delay_s=args.retry_base_delay_s,
            disable_thinking=args.disable_thinking,
        )
        answer_client = engine.create_gcp_client_for_model(
            model=args.models[0],
            gcp_reasoning_effort=effort,
            cache_only=args.cache_only or args.answer_cache_only,
            **client_kwargs,
        )
        if (
            not args.answer_cache_only
            and grader_location == args.gcp_location
            and engine.is_gcp_anthropic_model(args.grader_model)
            == engine.is_gcp_anthropic_model(args.models[0])
        ):
            grader_client = answer_client
        else:
            grader_client_kwargs = dict(client_kwargs)
            grader_client_kwargs["location"] = grader_location
            grader_client = engine.create_gcp_client_for_model(
                model=args.grader_model,
                gcp_reasoning_effort=None,
                cache_only=args.cache_only,
                **grader_client_kwargs,
            )

    write_lock = threading.Lock()
    completed = 0
    failed = 0
    processed = 0
    errors_path = args.out_dir / "simpleqa_penalty_errors.jsonl"

    def work(job):
        model, ex = job
        return run_penalty_question(
            ex, model=model, answer_client=answer_client,
            grader_client=grader_client, grader_model=args.grader_model,
            use_grader=args.use_grader, mock=args.mock, seed=args.seed,
            penalty=args.penalty, n_samples=args.penalty_samples,
            temperature=args.penalty_temperature,
            answer_max_tokens=args.penalty_max_tokens,
            empty_content_retries=args.empty_content_retries,
            json_only_prompts=args.json_only_prompts,
            gcp_reasoning_effort=effort,
        )

    with open(results_path, "a") as out_f:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(work, job): job for job in jobs}
            for fut in as_completed(futures):
                model, ex = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    processed += 1
                    error_row = {
                        "question_id": ex.question_id,
                        "model": model,
                        "penalty_value": args.penalty,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    with write_lock:
                        with open(errors_path, "a") as error_f:
                            error_f.write(
                                json.dumps(error_row, ensure_ascii=True) + chr(10)
                            )
                            error_f.flush()
                    print(
                        f"[penalty] ERROR {processed}/{len(jobs)} "
                        f"q={ex.question_id} {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                with write_lock:
                    out_f.write(json.dumps(row, ensure_ascii=True) + chr(10))
                    completed += 1
                    processed += 1
                    out_f.flush()
                if args.verbose:
                    print(f"[penalty] {processed}/{len(jobs)} q={row['question_id']} "
                          f"abst={row['penalty_abstention_rate']:.2f} "
                          f"acc={row['penalty_accuracy_overall']:.2f}", flush=True)

    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    import statistics
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    summary_path = args.out_dir / "simpleqa_penalty_summary.csv"
    with open(summary_path, "w") as f:
        f.write("model,n,penalty,mean_penalty_accuracy_overall,"
                "mean_penalty_accuracy_when_answered,mean_penalty_hallucination_rate,"
                "mean_penalty_abstention_rate" + chr(10))
        for m, rs in by_model.items():
            awa = [r["penalty_accuracy_when_answered"] for r in rs
                   if r["penalty_accuracy_when_answered"] is not None]
            f.write(f"{m},{len(rs)},{args.penalty:g},"
                    f"{statistics.mean(r['penalty_accuracy_overall'] for r in rs):.6f},"
                    f"{statistics.mean(awa) if awa else ''},"
                    f"{statistics.mean(r['penalty_hallucination_rate'] for r in rs):.6f},"
                    f"{statistics.mean(r['penalty_abstention_rate'] for r in rs):.6f}"
                    + chr(10))
    print(f"[penalty] done: {len(rows)} rows -> {results_path}")
    if failed:
        print(
            f"[penalty] {failed} question(s) failed and remain eligible for "
            f"resume; details -> {errors_path}",
            file=sys.stderr,
        )
    print(f"[penalty] summary -> {summary_path}")


def _simpleqa_add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models", nargs="+", required=True,
                   help="GCP Vertex model ids (one family per invocation, "
                        "e.g. claude-sonnet-4-6 OR google/gemini-3.5-flash xai/...).")
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of questions. Defaults to all questions in the selected split(s).",
    )
    p.add_argument(
        "--split",
        default="all",
        help=(
            "SimpleQA split. Default all means test+few_shot: the historical shuffled "
            "test order followed by the five few-shot questions."
        ),
    )
    p.add_argument("--idk-rho", type=float, default=0.5,
                   help="Residual discount rho in (0,1).")
    p.add_argument("--simpleqa-top-p", type=float, default=0.9)
    p.add_argument("--log-max-tokens", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0, help="Generation seed (not sampling).")
    p.add_argument("--grader-model", default="google/gemini-3.5-flash")
    p.add_argument("--use-grader", choices=["fallback", "grader", "exact"], default="fallback")
    p.add_argument("--gcp-project",
                   default=os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT"))
    p.add_argument("--gcp-location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    p.add_argument(
        "--grader-gcp-location",
        default=None,
        help="Optional Vertex location for the grader; defaults to --gcp-location.",
    )
    p.add_argument("--gcp-reasoning-effort", default=None,
                   help="e.g. 'low' for gemini runs (must match prior runs for cache reuse).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache-path", type=Path, default=Path("outputs/cache.sqlite"))
    p.add_argument("--cache-only", action="store_true")
    p.add_argument(
        "--answer-cache-only",
        action="store_true",
        help=(
            "Require answer-model generations to be present in the cache while "
            "still allowing live grader calls. Intended for importing GCP batch "
            "predictions without silently falling back to online answer requests."
        ),
    )
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--empty-content-retries", type=int, default=3)
    p.add_argument("--request-delay-s", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=12)
    p.add_argument("--retry-base-delay-s", type=float, default=2.0)
    p.add_argument("--include-penalty", action="store_true",
                   help="Also run the penalty (rubric) arm on the same questions.")
    p.add_argument("--skip-log", action="store_true",
                   help="Skip the log elicitation arm (penalty-only run).")
    p.add_argument("--penalty", type=float, default=3.0, help="Error penalty L.")
    p.add_argument("--penalty-samples", type=int, default=50,
                   help="Penalty samples per question.")
    p.add_argument("--penalty-temperature", type=float, default=1.0,
                   help="Matches the original penalty-arm runs (probed from cache).")
    p.add_argument("--penalty-max-tokens", type=int, default=1024,
                   help="Matches the original penalty-arm runs (probed from cache).")
    p.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True,
                   help="Disable provider thinking where supported; for GCP DeepSeek this "
                        "sends chat_template_kwargs.thinking=false. Part of the cache key.")
    p.add_argument("--json-only-prompts", action=argparse.BooleanOptionalAction, default=True,
                   help="Use strict JSON-only prompts for both log elicitation and "
                        "penalty samples across all answer models.")
    p.add_argument("--max-workers", type=int, default=4,
                   help="Concurrent questions in the penalty arm.")
    p.add_argument("--log-max-workers", type=int, default=1,
                   help="Concurrent questions in the log arm; one preserves serial behavior.")
    p.add_argument("--mock", action="store_true", help="No API calls; deterministic fake data.")
    p.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print per-question progress (disable with --no-verbose).",
    )


def _simpleqa_main(args: argparse.Namespace) -> None:
    rho = args.idk_rho
    if not (0.0 < rho < 1.0):
        raise SystemExit("--idk-rho must lie strictly between 0 and 1.")

    effort = engine.normalize_gcp_reasoning_effort(args.gcp_reasoning_effort)

    # A penalty-only run (--skip-log) must not overwrite the directory's
    # existing rule marker.
    if not args.skip_log:
        results_path = args.out_dir / "simpleqa_topp_results.jsonl"
        partial_path = args.out_dir / "simpleqa_topp_results.partial.jsonl"
        rewrite_jsonl_with_canonical_question_ids(results_path)
        rewrite_jsonl_with_canonical_question_ids(partial_path)
        check_resume_compatibility(args.out_dir, results_path, partial_path, rho=rho)
        write_rule_marker(args.out_dir, rho=rho)

    # dataset (deterministic sampling)
    if args.mock:
        mock_n = args.num_samples if args.num_samples is not None else 20
        examples = [
            engine.SimpleQAExample(
                question_id=f"mock-{i}",
                question=f"Mock question {i}?",
                answer=f"answer-{i}",
                category="Mock",
                answer_type="Other",
            )
            for i in range(mock_n)
        ]
    else:
        examples = load_simpleqa_examples_for_run(args)

    # clients
    answer_client = None
    grader_client = None
    if not args.mock:
        if not args.gcp_project:
            raise SystemExit(
                "Set --gcp-project or GOOGLE_CLOUD_PROJECT."
            )
        families = {engine.is_gcp_anthropic_model(m) for m in args.models}
        if len(families) > 1:
            raise SystemExit(
                "Mixing Anthropic and OpenAI-compatible Vertex models in one "
                "invocation is not supported; run one family at a time."
            )
        grader_location = args.grader_gcp_location or args.gcp_location
        client_kwargs = dict(
            project=args.gcp_project,
            location=args.gcp_location,
            cache=engine.SQLiteCache(args.cache_path),
            request_delay_s=args.request_delay_s,
            max_retries=args.max_retries,
            retry_base_delay_s=args.retry_base_delay_s,
            disable_thinking=args.disable_thinking,
        )
        answer_client = engine.create_gcp_client_for_model(
            model=args.models[0],
            gcp_reasoning_effort=effort,
            cache_only=args.cache_only or args.answer_cache_only,
            **client_kwargs,
        )
        if (
            not args.answer_cache_only
            and grader_location == args.gcp_location
            and engine.is_gcp_anthropic_model(args.grader_model)
            == engine.is_gcp_anthropic_model(args.models[0])
        ):
            grader_client = answer_client
        else:
            grader_client_kwargs = dict(client_kwargs)
            grader_client_kwargs["location"] = grader_location
            grader_client = engine.create_gcp_client_for_model(
                model=args.grader_model,
                gcp_reasoning_effort=None,
                cache_only=args.cache_only,
                **grader_client_kwargs,
            )

    if args.skip_log:
        if not args.include_penalty:
            raise SystemExit("--skip-log without --include-penalty leaves nothing to run.")
        run_penalty_arm(args, examples, effort)
        return

    if args.verbose and not args.include_penalty:
        print_settings(args, examples, rho, effort)

    rows = engine.run_simpleqa_log_only_experiment(
        examples=examples,
        models=args.models,
        answer_client=answer_client,
        grader_client=grader_client,
        grader_model=args.grader_model,
        use_grader=args.use_grader,
        mock=args.mock,
        seed=args.seed,
        log_max_tokens=args.log_max_tokens,
        top_p=args.simpleqa_top_p,
        idk_rho=rho,
        eps=args.eps,
        out_dir=args.out_dir,
        verbose=args.verbose,
        resume_existing_results=True,
        gcp_reasoning_effort=effort,
        empty_content_retries=args.empty_content_retries,
        strict_json_prompts=args.json_only_prompts,
        max_workers=args.log_max_workers,
    )

    annotate_results(results_path, rho=rho, eps=args.eps)
    annotate_results(partial_path, rho=rho, eps=args.eps)
    print(f"Done: {len(rows)} rows (residual rule, rho={rho:g}) -> {results_path}")

    if args.include_penalty:
        run_penalty_arm(args, examples, effort)

# consistency subcommand: repeated SimpleQA log elicitation for test-retest
# consistency, one row per (model, question, repeat), so analysis can measure
# how stable the elicited report is under repeated sampling.
#
#   python run.py consistency --models claude-sonnet-4-6 \
#       --num-samples 20 --log-repeats 50 --out-dir outputs/sonnet_log_consistency

def _consistency_add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--split", default="test")
    parser.add_argument("--log-repeats", type=int, default=50)
    parser.add_argument("--log-temperature", type=float, default=0.0)
    parser.add_argument("--simpleqa-top-p", type=float, default=0.9)
    parser.add_argument("--idk-rho", type=float, default=0.5)
    parser.add_argument("--log-max-tokens", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grader-model", default="google/gemini-3.5-flash")
    parser.add_argument("--use-grader", choices=["fallback", "grader", "exact"], default="fallback")
    parser.add_argument(
        "--gcp-project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT"),
    )
    parser.add_argument("--gcp-location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument(
        "--grader-gcp-location",
        default=None,
        help="Optional Vertex location for the grader; defaults to --gcp-location.",
    )
    parser.add_argument("--gcp-reasoning-effort", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, default=Path("outputs/cache.sqlite"))
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument(
        "--answer-cache-only",
        action="store_true",
        help="Require answer-model cache hits while allowing uncached grader calls.",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--empty-content-retries", type=int, default=3)
    parser.add_argument("--request-delay-s", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=12)
    parser.add_argument("--retry-base-delay-s", type=float, default=2.0)
    parser.add_argument(
        "--resource-exhausted-retries",
        type=int,
        default=0,
        help="Additional retries of one request after provider-level 429 retries are exhausted.",
    )
    parser.add_argument(
        "--resource-exhausted-cooldown-s",
        type=float,
        default=300.0,
        help="Cooldown before resubmitting a job that exhausted provider-level 429 retries.",
    )
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json-only-prompts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)


def log_prompt_for_example(example: engine.SimpleQAExample, *, top_p: float, rho: float) -> str:
    return engine.build_simpleqa_topp_log_prompt(example, top_p, rho)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _consistency_main(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive.")
    if args.log_repeats <= 0:
        raise SystemExit("--log-repeats must be positive.")
    if not (0.0 < args.idk_rho < 1.0):
        raise SystemExit("--idk-rho must lie strictly between 0 and 1.")
    if args.resource_exhausted_retries < 0:
        raise SystemExit("--resource-exhausted-retries must be nonnegative.")
    if args.resource_exhausted_cooldown_s < 0:
        raise SystemExit("--resource-exhausted-cooldown-s must be nonnegative.")
    if not args.mock and not args.gcp_project:
        raise SystemExit("Set --gcp-project or GOOGLE_CLOUD_PROJECT.")

    effort = engine.normalize_gcp_reasoning_effort(args.gcp_reasoning_effort)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.out_dir / "simpleqa_log_consistency_results.partial.jsonl"
    results_path = args.out_dir / "simpleqa_log_consistency_results.jsonl"
    summary_path = args.out_dir / "simpleqa_log_consistency_summary.csv"

    examples = (
        [
            engine.SimpleQAExample(
                question_id=f"mock-{i}",
                question=f"Mock question {i}?",
                answer=f"answer-{i}",
                category="Mock",
                answer_type="Other",
            )
            for i in range(args.num_samples)
        ]
        if args.mock
        else load_simpleqa_examples_for_run(args)
    )

    answer_client = None
    grader_client = None
    if not args.mock:
        families = {engine.is_gcp_anthropic_model(m) for m in args.models}
        if len(families) > 1:
            raise SystemExit("Mixing Anthropic and OpenAI-compatible Vertex models is unsupported.")
        client_kwargs = dict(
            project=args.gcp_project,
            location=args.gcp_location,
            cache=engine.SQLiteCache(args.cache_path),
            cache_only=args.cache_only or args.answer_cache_only,
            request_delay_s=args.request_delay_s,
            max_retries=args.max_retries,
            retry_base_delay_s=args.retry_base_delay_s,
            disable_thinking=args.disable_thinking,
        )
        answer_client = engine.create_gcp_client_for_model(
            model=args.models[0], gcp_reasoning_effort=effort, **client_kwargs
        )
        grader_location = args.grader_gcp_location or args.gcp_location
        if (
            engine.is_gcp_anthropic_model(args.grader_model)
            == engine.is_gcp_anthropic_model(args.models[0])
            and grader_location == args.gcp_location
            and not args.answer_cache_only
        ):
            grader_client = answer_client
        else:
            grader_client_kwargs = dict(client_kwargs)
            grader_client_kwargs["location"] = grader_location
            grader_client_kwargs["cache_only"] = args.cache_only
            grader_client = engine.create_gcp_client_for_model(
                model=args.grader_model,
                gcp_reasoning_effort=None,
                **grader_client_kwargs,
            )

    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str, int]] = set()
    for source in (partial_path, results_path):
        if not source.exists():
            continue
        for row in engine.read_jsonl(source):
            if row.get("log_idk_rule") != "residual":
                raise SystemExit(f"Refusing resume from {source}: log_idk_rule differs.")
            if abs(float(row.get("log_idk_rho")) - args.idk_rho) > 1e-9:
                raise SystemExit(f"Refusing resume from {source}: log_idk_rho differs.")
            if abs(float(row.get("log_temperature")) - args.log_temperature) > 1e-9:
                raise SystemExit(f"Refusing resume from {source}: log_temperature differs.")
            if abs(float(row.get("simpleqa_top_p")) - args.simpleqa_top_p) > 1e-9:
                raise SystemExit(f"Refusing resume from {source}: simpleqa_top_p differs.")
            key = (str(row["model"]), str(row["question_id"]), int(row["repeat_index"]))
            if key not in completed:
                rows.append(row)
                completed.add(key)

    model_order = {model: i for i, model in enumerate(args.models)}
    example_order = {example.question_id: i for i, example in enumerate(examples)}

    def sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
        return (
            model_order.get(str(row.get("model")), len(model_order)),
            example_order.get(str(row.get("question_id")), len(example_order)),
            int(row.get("repeat_index", 0)),
            str(row.get("question_id")),
        )

    total_jobs = len(args.models) * len(examples) * args.log_repeats
    total_questions = len(args.models) * len(examples)
    jobs = []
    question_counter: dict[tuple[str, str], int] = {}
    for model_i, model in enumerate(args.models):
        log_system_message = engine.simpleqa_log_system_message_for_model(
            model,
            strict_json=args.json_only_prompts,
        )
        for example_i, example in enumerate(examples):
            question_counter[(model, example.question_id)] = model_i * len(examples) + example_i + 1
            prompt = log_prompt_for_example(
                example,
                top_p=args.simpleqa_top_p,
                rho=args.idk_rho,
            )
            for repeat_i in range(args.log_repeats):
                key = (model, example.question_id, repeat_i)
                if key in completed:
                    continue
                jobs.append((model, example, repeat_i, prompt, log_system_message))

    if not jobs:
        rows.sort(key=sort_key)
        write_jsonl(partial_path, rows)
        write_jsonl(results_path, rows)
        print(f"Done. Wrote {len(rows)} repeated log rows to {results_path}")
        print(f"Wrote per-question summary to {summary_path}")
        return

    write_lock = threading.Lock()
    new_rows_by_question: dict[tuple[str, str], int] = {}
    skipped_by_question: dict[tuple[str, str], int] = {}
    for model in args.models:
        for example in examples:
            question_key = (model, example.question_id)
            have = sum((model, example.question_id, repeat_i) in completed for repeat_i in range(args.log_repeats))
            skipped_by_question[question_key] = have

    def run_one(job) -> dict[str, Any]:
        model, example, repeat_i, prompt, log_system_message = job
        sample_seed = args.seed * 1_000_000 + repeat_i
        log_generation_error = None
        log_empty_content_retry_count = 0
        if args.mock:
            rng = engine.stable_rng("simpleqa-log-consistency", example.question_id, model, sample_seed)
            true_points = 55 + 35 * rng.random()
            idk_points = max(0.0, min(100.0 - true_points, 15 * rng.random()))
            wrong_points = max(0.0, 100.0 - true_points - idk_points)
            log_response = json.dumps(
                {
                    "answers": [
                        {"answer": example.answer, "points": true_points},
                        {"answer": "plausible distractor", "points": wrong_points},
                        {"answer": engine.UNKNOWN_LABEL, "points": idk_points},
                    ]
                },
                ensure_ascii=True,
            )
        else:
            assert answer_client is not None
            try:
                log_response, log_empty_content_retry_count = engine.generate_with_empty_content_retries(
                    answer_client,
                    model=model,
                    prompt=prompt,
                    system_message=log_system_message,
                    seed=sample_seed,
                    temperature=args.log_temperature,
                    max_tokens=args.log_max_tokens,
                    empty_content_retries=args.empty_content_retries,
                )
            except engine.EmptyAssistantContentError as exc:
                log_generation_error = str(exc)
                log_empty_content_retry_count = max(0, args.empty_content_retries)
                log_response = json.dumps(
                    {"answers": [{"answer": engine.UNKNOWN_LABEL, "points": engine.PROBABILITY_POINT_TOTAL}]},
                    ensure_ascii=True,
                )
        parsed_candidates, log_status = engine.parse_simpleqa_topp_response(
            log_response,
            top_p=args.simpleqa_top_p,
        )
        log_candidates = parsed_candidates or [
            {"answer": engine.UNKNOWN_LABEL, "points": engine.PROBABILITY_POINT_TOTAL, "probability": 1.0}
        ]
        if parsed_candidates is None:
            log_status = f"malformed_{log_status}"
        if log_generation_error is not None:
            log_status = f"generation_empty_content_{log_status}"
        grade_cache: dict[str, str] = {}
        for candidate_i, candidate in enumerate(log_candidates):
            candidate["grade"] = engine.grade_simpleqa_answer_cached(
                example=example,
                answer=str(candidate["answer"]),
                grade_cache=grade_cache,
                grader_client=grader_client,
                grader_model=args.grader_model,
                use_grader=args.use_grader,
                mock=args.mock,
                seed=sample_seed * 1_000 + candidate_i,
            )
        log_candidates = engine.canonicalize_simpleqa_distribution(log_candidates, example=example)
        log_top = max(log_candidates, key=lambda item: float(item.get("probability", 0.0)))
        log_q_true = sum(float(c.get("probability", 0.0)) for c in log_candidates if c.get("grade") == "correct")
        log_reported_mass = sum(float(c.get("probability", 0.0)) for c in log_candidates)
        log_not_attempted_mass = sum(float(c.get("probability", 0.0)) for c in log_candidates if c.get("grade") == "not_attempted")
        log_incorrect_mass = sum(float(c.get("probability", 0.0)) for c in log_candidates if c.get("grade") == "incorrect")
        log_token_usage = engine.generation_token_usage(
            client=None if args.mock else answer_client,
            prompt=prompt,
            system_message=log_system_message,
            response=log_response,
        )
        return {
            "question_id": example.question_id,
            "category": example.category,
            "answer_type": example.answer_type,
            "model": model,
            "repeat_index": repeat_i,
            "seed": sample_seed,
            "base_seed": args.seed,
            "simpleqa_top_p": args.simpleqa_top_p,
            "log_temperature": args.log_temperature,
            "log_idk_rule": "residual",
            "log_idk_rho": args.idk_rho,
            "log_system_message": log_system_message,
            "gcp_reasoning_effort": engine.effective_gcp_reasoning_effort(model, effort),
            "gold_answer": example.answer,
            "question": example.question,
            "log_top_answer": log_top.get("answer"),
            "log_top_prob": log_top.get("probability"),
            "log_top_grade": log_top.get("grade"),
            "log_q_true": log_q_true,
            "log_hallucination_mass": log_incorrect_mass,
            "log_not_attempted_mass": log_not_attempted_mass,
            "log_reported_mass": log_reported_mass,
            "log_unreported_mass": max(0.0, 1.0 - log_reported_mass),
            "log_report_log_score": math.log(max(args.eps, log_q_true)),
            "log_candidates_json": json.dumps(log_candidates, ensure_ascii=True),
            "log_parse_status": log_status,
            "raw_log_response": log_response,
            "log_input_tokens": log_token_usage["input_tokens"],
            "log_output_tokens": log_token_usage["output_tokens"],
            "log_total_tokens": log_token_usage["total_tokens"],
            "log_generation_error": log_generation_error,
            "log_empty_content_retry_count": log_empty_content_retry_count,
        }

    done_new = 0
    resource_exhausted_counts: dict[tuple[str, str, int], int] = {}
    worker_count = max(1, args.max_workers)
    max_pending = worker_count
    print(
        f"[log consistency] starting {len(jobs)} pending jobs "
        f"with {worker_count} workers ({len(completed)} cached/completed)",
        flush=True,
    )
    with open(partial_path, "a") as partial_file:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            job_iter = iter(jobs)
            futures = {}

            def submit_next() -> bool:
                try:
                    job = next(job_iter)
                except StopIteration:
                    return False
                futures[pool.submit(run_one, job)] = job
                return True

            for _ in range(min(max_pending, len(jobs))):
                submit_next()

            while futures:
                finished, _pending = wait(futures, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = futures.pop(future)
                    try:
                        row = future.result()
                    except Exception as exc:  # noqa: BLE001
                        model, example, repeat_index, _prompt, _system_message = job
                        retry_key = (model, example.question_id, repeat_index)
                        message = str(exc).lower()
                        is_resource_exhausted = (
                            "resource_exhausted" in message
                            or "resource exhausted" in message
                            or "429 client error" in message
                        )
                        retry_number = resource_exhausted_counts.get(retry_key, 0) + 1
                        if (
                            is_resource_exhausted
                            and retry_number <= args.resource_exhausted_retries
                        ):
                            resource_exhausted_counts[retry_key] = retry_number
                            print(
                                "[log consistency] provider capacity exhausted for "
                                f"q={example.question_id} repeat={repeat_index}; "
                                f"capacity retry {retry_number}/"
                                f"{args.resource_exhausted_retries} after "
                                f"{args.resource_exhausted_cooldown_s:g}s",
                                flush=True,
                            )
                            if args.resource_exhausted_cooldown_s:
                                time.sleep(args.resource_exhausted_cooldown_s)
                            futures[pool.submit(run_one, job)] = job
                            continue
                        raise
                    key = (str(row["model"]), str(row["question_id"]), int(row["repeat_index"]))
                    question_key = (str(row["model"]), str(row["question_id"]))
                    with write_lock:
                        rows.append(row)
                        completed.add(key)
                        new_rows_by_question[question_key] = new_rows_by_question.get(question_key, 0) + 1
                        done_new += 1
                        partial_file.write(json.dumps(row, ensure_ascii=True) + "\n")
                        partial_file.flush()
                        if args.verbose:
                            question_rows = [
                                existing
                                for existing in rows
                                if (
                                    existing.get("model") == row["model"]
                                    and existing.get("question_id") == row["question_id"]
                                )
                            ]
                            parse_failures = sum(
                                str(existing.get("log_parse_status", "")).startswith("malformed")
                                for existing in question_rows
                            )
                            empty_retries = sum(
                                int(existing.get("log_empty_content_retry_count", 0) or 0)
                                for existing in question_rows
                            )
                            print(
                                f"[log consistency] job={done_new}/{len(jobs)} "
                                f"question={question_counter.get(question_key, '?')}/{total_questions} "
                                f"model={row['model']} q={row['question_id']} "
                                f"repeats={len(question_rows)}/{args.log_repeats} "
                                f"new={new_rows_by_question.get(question_key, 0)} "
                                f"skipped={skipped_by_question.get(question_key, 0)} "
                                f"parse_fail={parse_failures} empty_retries={empty_retries} "
                                f"temp={args.log_temperature:g}",
                                flush=True,
                            )
                    submit_next()

    rows.sort(key=sort_key)
    write_jsonl(results_path, rows)
    summary_rows = []
    for (model, qid), group in pd_groupby(rows, keys=("model", "question_id")):
        summary_rows.append(
            {
                "model": model,
                "question_id": qid,
                "n_repeats": len(group),
                "mean_log_q_true": mean([float(r["log_q_true"]) for r in group]),
                "sd_log_q_true": sample_sd([float(r["log_q_true"]) for r in group]),
                "mean_log_hallucination_mass": mean([float(r["log_hallucination_mass"]) for r in group]),
                "sd_log_hallucination_mass": sample_sd([float(r["log_hallucination_mass"]) for r in group]),
                "mean_log_not_attempted_mass": mean([float(r["log_not_attempted_mass"]) for r in group]),
                "sd_log_not_attempted_mass": sample_sd([float(r["log_not_attempted_mass"]) for r in group]),
                "parse_failure_rate": mean([float(str(r.get("log_parse_status", "")).startswith("malformed")) for r in group]),
            }
        )
    with open(summary_path, "w") as f:
        if summary_rows:
            columns = list(summary_rows[0])
            f.write(",".join(columns) + "\n")
            for row in summary_rows:
                f.write(",".join(str(row.get(col, "")) for col in columns) + "\n")
    print(f"Done. Wrote {len(rows)} repeated log rows to {results_path}")
    print(f"Wrote per-question summary to {summary_path}")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def sample_sd(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def pd_groupby(rows: list[dict[str, Any]], *, keys: tuple[str, ...]):
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        groups.setdefault(key, []).append(row)
    for key in sorted(groups):
        yield key, groups[key]

# batch subcommand: generate answer-model responses via GCP batch prediction,
# inserting them into the same SQLite cache as `run.py simpleqa`; the ordinary
# runner then grades and summarizes as usual.
#
#   python run.py batch prepare --arm log --num-samples 4326 \
#       --work-dir outputs/gptoss120b_batch/log
#   python run.py batch submit --plan .../batch_plan.json
#   # after it succeeds: status / download / import-cache with the same --plan
#
# Then run `run.py simpleqa` with the same generation args,
# --gcp-location us-central1, and --answer-cache-only: answer calls hit the
# cache while grader calls stay live.

DEFAULT_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
DEFAULT_LOCATION = "us-central1"
DEFAULT_BUCKET = os.environ.get("GCS_BATCH_BUCKET")
DEFAULT_MODEL = "openai/gpt-oss-120b-maas"
DEFAULT_PUBLISHER_MODEL = "publishers/openai/models/gpt-oss-120b-maas"
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected an object in {path}:{line_number}.")
        yield row


def _batch_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _gcp_access_token() -> str:
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError as exc:
        raise RuntimeError("Install google-auth to submit GCP batch jobs.") from exc

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return str(credentials.token)


def _api_request(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Install requests to submit GCP batch jobs.") from exc

    response = requests.request(
        method,
        url,
        headers={
            "Authorization": f"Bearer {_gcp_access_token()}",
            "Content-Type": "application/json; charset=utf-8",
        },
        json=payload,
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(
            f"GCP API request failed ({response.status_code}): {response.text[:4000]}"
        )
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected a JSON object from {url}; received {type(data).__name__}.")
    return data


def _run_gcloud(*args: str) -> None:
    command = ["gcloud", *args]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"gcloud command failed: {' '.join(command)}\n{detail}")


def _upload_file_to_gcs(local_path: str, gcs_uri: str) -> None:
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// destination, received {gcs_uri!r}.")
    bucket_and_object = gcs_uri[5:]
    bucket, separator, object_name = bucket_and_object.partition("/")
    if not separator or not bucket or not object_name:
        raise ValueError(f"Expected a GCS object URI, received {gcs_uri!r}.")

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Install requests to upload GCP batch inputs.") from exc

    with Path(local_path).open("rb") as handle:
        response = requests.post(
            f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o",
            headers={
                "Authorization": f"Bearer {_gcp_access_token()}",
                "Content-Type": "application/jsonl",
            },
            params={"uploadType": "media", "name": object_name},
            data=handle,
            timeout=600,
        )
    if not response.ok:
        raise RuntimeError(
            f"GCS upload failed ({response.status_code}): {response.text[:4000]}"
        )


def _load_examples(
    num_samples: int | None,
    split: str,
    question_start: int = 0,
) -> list[engine.SimpleQAExample]:
    if question_start < 0:
        raise ValueError("question_start must be nonnegative")
    loaded_count = None if num_samples is None else question_start + num_samples
    examples = load_simpleqa_examples_for_run(
        SimpleNamespace(num_samples=loaded_count, split=split)
    )
    if question_start > len(examples):
        raise ValueError(
            f"question_start={question_start} exceeds the {len(examples)} loaded questions"
        )
    return examples[question_start:]


def _cache_kwargs(
    *,
    model: str,
    prompt: str,
    system_message: str,
    seed: int,
    temperature: float,
    max_tokens: int,
    project: str,
    location: str,
    disable_thinking: bool,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    """Mirror OpenRouterClient.generate cache-key construction exactly."""
    base_url = engine.gcp_vertex_openai_base_url(project, location)
    request_seed = engine.provider_request_seed(model=model, base_url=base_url, seed=seed)
    request_effort = engine.effective_gcp_reasoning_effort(model, reasoning_effort)
    request_temperature = (
        temperature
        if engine.chat_completion_supports_temperature(
            base_url=base_url, temperature=temperature
        )
        else None
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "system_message": system_message,
        "seed": seed,
        "temperature": temperature,
        "request_temperature": request_temperature,
        "max_tokens": max_tokens,
        "base_url": base_url,
        "disable_thinking": disable_thinking,
        "openrouter_reasoning_effort": None,
        "gcp_reasoning_effort": request_effort,
        "openai_reasoning_effort": None,
        "transport": "openai_chat_completions",
    }
    deepseek_thinking = engine.gcp_deepseek_thinking_cache_value(
        model=model,
        base_url=base_url,
        disable_thinking=disable_thinking,
    )
    if deepseek_thinking is not None:
        kwargs["gcp_deepseek_thinking"] = deepseek_thinking
    if request_temperature == temperature:
        kwargs.pop("request_temperature")
    if request_seed != seed:
        kwargs["request_seed"] = request_seed
    return kwargs


def _request_body(
    *,
    model: str,
    prompt: str,
    system_message: str,
    seed: int,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str | None,
    disable_thinking: bool,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if engine.is_gcp_deepseek_model(model):
        body["chat_template_kwargs"] = {"thinking": not disable_thinking}
    return body


def _generation_specs(
    args: argparse.Namespace,
    examples: list[engine.SimpleQAExample],
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    reasoning_effort = engine.effective_gcp_reasoning_effort(
        args.model,
        engine.normalize_gcp_reasoning_effort(args.reasoning_effort),
    )
    if args.arm == "log":
        system_message = engine.simpleqa_log_system_message_for_model(
            args.model, strict_json=True
        )
        # The ordinary single-report runner uses integer 0, whereas the
        # consistency runner receives a float from argparse. Cache keys retain
        # that distinction, so preserve each caller's exact representation.
        temperature = 0 if args.log_repeats == 1 else float(args.log_temperature)
        max_tokens = args.log_max_tokens
        for question_index, example in enumerate(examples):
            prompt = engine.build_simpleqa_topp_log_prompt(
                example, args.simpleqa_top_p, args.idk_rho
            )
            for repeat_index in range(args.log_repeats):
                sample_seed = (
                    args.seed
                    if args.log_repeats == 1
                    else args.seed * 1_000_000 + repeat_index
                )
                custom_id = (
                    f"log-{question_index:05d}"
                    if args.log_repeats == 1
                    else f"log-{question_index:05d}-{repeat_index:03d}"
                )
                cache_kwargs = _cache_kwargs(
                    model=args.model,
                    prompt=prompt,
                    system_message=system_message,
                    seed=sample_seed,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    project=args.project,
                    location=args.location,
                    disable_thinking=args.disable_thinking,
                    reasoning_effort=reasoning_effort,
                )
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": _request_body(
                        model=args.model,
                        prompt=prompt,
                        system_message=system_message,
                        seed=sample_seed,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                        disable_thinking=args.disable_thinking,
                    ),
                }
                manifest = {
                    "custom_id": custom_id,
                    "cache_key": engine.cache_key_for_kwargs(cache_kwargs),
                    "question_id": example.question_id,
                    "sample_index": repeat_index if args.log_repeats > 1 else None,
                    "repeat_index": repeat_index if args.log_repeats > 1 else None,
                }
                yield request, manifest
        return

    system_message = engine.simpleqa_penalty_system_message(strict_json=True)
    temperature = args.penalty_temperature
    max_tokens = args.penalty_max_tokens
    for question_index, example in enumerate(examples):
        prompt = engine.build_simpleqa_penalty_json_prompt(example, args.penalty)
        for sample_index in range(args.penalty_samples):
            sample_seed = args.seed * 1_000_000 + 200_000 + sample_index
            custom_id = f"pen-{question_index:05d}-{sample_index:03d}"
            cache_kwargs = _cache_kwargs(
                model=args.model,
                prompt=prompt,
                system_message=system_message,
                seed=sample_seed,
                temperature=temperature,
                max_tokens=max_tokens,
                project=args.project,
                location=args.location,
                disable_thinking=args.disable_thinking,
                reasoning_effort=reasoning_effort,
            )
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": _request_body(
                    model=args.model,
                    prompt=prompt,
                    system_message=system_message,
                    seed=sample_seed,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    disable_thinking=args.disable_thinking,
                ),
            }
            manifest = {
                "custom_id": custom_id,
                "cache_key": engine.cache_key_for_kwargs(cache_kwargs),
                "question_id": example.question_id,
                "sample_index": sample_index,
            }
            yield request, manifest


def prepare(args: argparse.Namespace) -> None:
    if args.location == "global":
        raise SystemExit("Batch prediction for managed open models does not support global endpoints.")
    if args.arm == "log" and args.log_repeats <= 0:
        raise SystemExit("--log-repeats must be positive.")
    if args.arm == "penalty" and args.penalty_samples <= 0:
        raise SystemExit("--penalty-samples must be positive.")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive.")
    if args.work_dir.exists() and any(args.work_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"{args.work_dir} is not empty. Use a fresh directory or pass --overwrite."
        )

    examples = _load_examples(args.num_samples, args.split, args.question_start)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    input_dir = args.work_dir / "inputs"
    manifest_dir = args.work_dir / "manifests"
    run_name = args.run_name or args.work_dir.name
    gcs_root = f"gs://{args.bucket}/{args.gcs_prefix.strip('/')}/{run_name}"
    chunks: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    request_count = 0

    def flush_chunk() -> None:
        nonlocal request_rows, manifest_rows
        if not request_rows:
            return
        chunk_index = len(chunks)
        chunk_name = f"chunk-{chunk_index:05d}"
        input_path = (input_dir / f"{chunk_name}.jsonl").resolve()
        manifest_path = (manifest_dir / f"{chunk_name}.jsonl").resolve()
        _batch_write_jsonl(input_path, request_rows)
        _batch_write_jsonl(manifest_path, manifest_rows)
        chunks.append(
            {
                "chunk_name": chunk_name,
                "request_count": len(request_rows),
                "input_path": str(input_path),
                "manifest_path": str(manifest_path),
                "input_uri": f"{gcs_root}/inputs/{chunk_name}.jsonl",
                "output_uri": f"{gcs_root}/outputs/{chunk_name}",
            }
        )
        request_rows = []
        manifest_rows = []

    estimated_input_tokens = 0
    for request, manifest in _generation_specs(args, examples):
        request_rows.append(request)
        manifest_rows.append(manifest)
        request_count += 1
        messages = request["body"]["messages"]
        estimated_input_tokens += sum(
            engine.estimated_token_count(str(message.get("content", "")))
            for message in messages
        )
        if len(request_rows) >= args.chunk_size:
            flush_chunk()
    flush_chunk()

    plan = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arm": args.arm,
        "run_name": run_name,
        "work_dir": str(args.work_dir.resolve()),
        "project": args.project,
        "location": args.location,
        "bucket": args.bucket,
        "gcs_root": gcs_root,
        "model": args.model,
        "publisher_model": args.publisher_model,
        "cache_path": str(args.cache_path.resolve()),
        "configuration": {
            "num_questions": len(examples),
            "question_start": args.question_start,
            "split": args.split,
            "seed": args.seed,
            "reasoning_effort": engine.normalize_gcp_reasoning_effort(
                args.reasoning_effort
            ),
            "disable_thinking": args.disable_thinking,
            "simpleqa_top_p": args.simpleqa_top_p,
            "log_idk_rule": "residual",
            "idk_rho": args.idk_rho,
            "log_max_tokens": args.log_max_tokens,
            "log_repeats": args.log_repeats,
            "log_temperature": args.log_temperature,
            "penalty": args.penalty,
            "penalty_samples": args.penalty_samples,
            "penalty_temperature": args.penalty_temperature,
            "penalty_max_tokens": args.penalty_max_tokens,
        },
        "request_count": request_count,
        "estimated_input_tokens": estimated_input_tokens,
        "batch_input_price_per_million_usd": args.batch_input_price_per_million,
        "estimated_batch_input_cost_usd": (
            estimated_input_tokens
            / 1_000_000
            * args.batch_input_price_per_million
        ),
        "chunks": chunks,
    }
    plan_path = args.work_dir / "batch_plan.json"
    _write_json(plan_path, plan)
    print(f"Prepared {request_count:,} requests in {len(chunks)} chunk(s).")
    print(f"Estimated input tokens: {estimated_input_tokens:,}")
    print(
        "Estimated batch input cost: "
        f"${plan['estimated_batch_input_cost_usd']:.4f} (output cost not included)"
    )
    print(f"Plan: {plan_path}")


def _job_url(project: str, location: str, job_name: str | None = None) -> str:
    host = engine.gcp_vertex_api_host(location)
    base = (
        f"https://{host}/v1/"
        f"projects/{project}/locations/{location}/batchPredictionJobs"
    )
    if job_name:
        return f"https://{host}/v1/{job_name}"
    return base


def submit(args: argparse.Namespace) -> None:
    plan = _load_json(args.plan)
    if plan.get("location") == "global":
        raise SystemExit(
            "Batch prediction for managed open models does not support global "
            "endpoints. This plan cannot be submitted; use the online runner "
            "for a global-only model."
        )
    submitted = 0
    for chunk in plan["chunks"]:
        existing_job = chunk.get("job", {})
        if existing_job.get("name"):
            if not (
                args.retry_failed
                and existing_job.get("state") == "JOB_STATE_FAILED"
            ):
                continue
            chunk.setdefault("job_attempts", []).append(existing_job)
            chunk.pop("job", None)
        if args.max_jobs is not None and submitted >= args.max_jobs:
            break
        print(f"Uploading {chunk['chunk_name']} ({chunk['request_count']:,} requests)...")
        _upload_file_to_gcs(chunk["input_path"], chunk["input_uri"])
        label_model = "".join(
            char if char.isalnum() or char in "-_" else "-"
            for char in str(plan["model"]).lower()
        ).strip("-_")[-63:]
        payload = {
            "displayName": f"{plan['run_name']}-{chunk['chunk_name']}",
            "model": plan["publisher_model"],
            "inputConfig": {
                "instancesFormat": "jsonl",
                "gcsSource": {"uris": [chunk["input_uri"]]},
            },
            "outputConfig": {
                "predictionsFormat": "jsonl",
                "gcsDestination": {"outputUriPrefix": chunk["output_uri"]},
            },
            "labels": {"experiment": "simpleqa", "answer_model": label_model},
        }
        job = _api_request(
            "POST", _job_url(plan["project"], plan["location"]), payload=payload
        )
        chunk["job"] = job
        submitted += 1
        _write_json(args.plan, plan)
        print(f"Submitted {job.get('name')} ({job.get('state')}).")
    if submitted == 0:
        print("No unsubmitted chunks in the selected scope.")
    if args.wait:
        wait_for_jobs(args.plan, args.poll_seconds)


def refresh_status(plan_path: Path) -> dict[str, Any]:
    plan = _load_json(plan_path)
    for chunk in plan["chunks"]:
        name = chunk.get("job", {}).get("name")
        if not name:
            continue
        job = _api_request(
            "GET", _job_url(plan["project"], plan["location"], name)
        )
        chunk["job"] = job
    _write_json(plan_path, plan)
    return plan


def print_status(plan: dict[str, Any]) -> None:
    counts: dict[str, int] = {}
    for chunk in plan["chunks"]:
        state = chunk.get("job", {}).get("state", "NOT_SUBMITTED")
        counts[state] = counts.get(state, 0) + 1
        print(f"{chunk['chunk_name']}: {state}")
        if chunk.get("job", {}).get("error"):
            print(f"  error: {json.dumps(chunk['job']['error'], ensure_ascii=True)}")
    print("Summary: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))


def wait_for_jobs(plan_path: Path, poll_seconds: float) -> None:
    while True:
        plan = refresh_status(plan_path)
        print_status(plan)
        submitted_states = [
            chunk.get("job", {}).get("state")
            for chunk in plan["chunks"]
            if chunk.get("job", {}).get("name")
        ]
        if submitted_states and all(state in TERMINAL_STATES for state in submitted_states):
            return
        time.sleep(poll_seconds)


def status(args: argparse.Namespace) -> None:
    plan = refresh_status(args.plan)
    print_status(plan)


def cancel(args: argparse.Namespace) -> None:
    plan = refresh_status(args.plan)
    cancelled = 0
    for chunk in plan["chunks"]:
        job = chunk.get("job", {})
        name = job.get("name")
        state = job.get("state")
        if not name or state in TERMINAL_STATES:
            continue
        _api_request(
            "POST",
            _job_url(plan["project"], plan["location"], name) + ":cancel",
            payload={},
        )
        cancelled += 1
        print(f"Cancellation requested for {chunk['chunk_name']} ({state}).")
    if cancelled == 0:
        print("No active or queued jobs to cancel.")
    if args.wait and cancelled:
        wait_for_jobs(args.plan, args.poll_seconds)


def download(args: argparse.Namespace) -> None:
    plan = refresh_status(args.plan)
    changed = False
    for chunk in plan["chunks"]:
        state = chunk.get("job", {}).get("state")
        if state != "JOB_STATE_SUCCEEDED":
            print(f"Skipping {chunk['chunk_name']}: state={state or 'NOT_SUBMITTED'}")
            continue
        source_uri = (
            chunk.get("job", {}).get("outputInfo", {}).get("gcsOutputDirectory")
            or chunk["output_uri"]
        )
        output_dir = Path(plan["work_dir"]) / "downloaded" / chunk["chunk_name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        if any(output_dir.rglob("*.jsonl")):
            print(f"Reusing downloaded output in {output_dir}")
        else:
            print(f"Downloading {source_uri} -> {output_dir}")
            _run_gcloud(
                "storage", "cp", "--recursive", source_uri, str(output_dir)
            )
        chunk["download_dir"] = str(output_dir.resolve())
        changed = True
    if changed:
        _write_json(args.plan, plan)


def _as_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _find_custom_id(row: dict[str, Any]) -> str | None:
    direct = row.get("custom_id")
    if direct is not None:
        return str(direct)
    for key in ("instance", "request", "input"):
        nested = _as_json_object(row.get(key))
        if nested is not None:
            found = _find_custom_id(nested)
            if found is not None:
                return found
    return None


def _find_chat_response(row: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(row.get("choices"), list):
        return row
    for key in ("prediction", "response", "body", "output"):
        nested = _as_json_object(row.get(key))
        if nested is not None:
            found = _find_chat_response(nested)
            if found is not None:
                return found
    return None


def _response_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    message = choice.get("message")
    if not isinstance(message, dict):
        return ""
    return str(message.get("content") or "")


def import_cache(args: argparse.Namespace) -> None:
    plan = _load_json(args.plan)
    cache_path = args.cache_path or Path(plan["cache_path"])
    cache = engine.SQLiteCache(cache_path)
    imported = 0
    failed = 0
    missing_outputs: list[str] = []
    report_rows: list[dict[str, Any]] = []

    for chunk in plan["chunks"]:
        manifest_rows = list(_iter_jsonl(Path(chunk["manifest_path"])))
        manifest_by_id = {str(row["custom_id"]): row for row in manifest_rows}
        download_dir_value = chunk.get("download_dir")
        if not download_dir_value:
            missing_outputs.append(chunk["chunk_name"])
            continue
        output_files = sorted(Path(download_dir_value).rglob("*.jsonl"))
        output_rows = [row for path in output_files for row in _iter_jsonl(path)]
        for index, output_row in enumerate(output_rows):
            custom_id = _find_custom_id(output_row)
            if custom_id is None and len(output_rows) == len(manifest_rows):
                custom_id = str(manifest_rows[index]["custom_id"])
            manifest = manifest_by_id.get(str(custom_id)) if custom_id is not None else None
            response = _find_chat_response(output_row)
            content = _response_content(response or {})
            if manifest is None or response is None or not content.strip():
                failed += 1
                report_rows.append(
                    {
                        "chunk_name": chunk["chunk_name"],
                        "custom_id": custom_id,
                        "status": "failed",
                        "has_manifest": manifest is not None,
                        "has_response": response is not None,
                        "has_content": bool(content.strip()),
                        "error": output_row.get("error"),
                    }
                )
                continue
            response = dict(response)
            response["_simpleqa_batch"] = {
                "job_name": chunk.get("job", {}).get("name"),
                "custom_id": custom_id,
                "input_uri": chunk["input_uri"],
            }
            cache.set(
                str(manifest["cache_key"]),
                {"content": content, "raw": response},
            )
            imported += 1
            report_rows.append(
                {
                    "chunk_name": chunk["chunk_name"],
                    "custom_id": custom_id,
                    "question_id": manifest.get("question_id"),
                    "sample_index": manifest.get("sample_index"),
                    "status": "imported",
                }
            )

    report_path = Path(plan["work_dir"]) / "cache_import_report.jsonl"
    _batch_write_jsonl(report_path, report_rows)
    print(f"Imported {imported:,} responses into {cache_path}.")
    if failed:
        print(f"Failed to import {failed:,} output rows; see {report_path}.")
    if missing_outputs:
        print("Chunks without downloaded output: " + ", ".join(missing_outputs))
    if imported != int(plan["request_count"]):
        raise SystemExit(
            f"Imported {imported:,} of {int(plan['request_count']):,} planned requests. "
            "Do not run the analysis without reviewing the import report."
        )


def add_plan_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)

def _batch_add_arguments(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--arm", choices=["log", "penalty"], required=True)
    prepare_parser.add_argument("--work-dir", type=Path, required=True)
    prepare_parser.add_argument("--run-name", default=None)
    prepare_parser.add_argument("--project", default=DEFAULT_PROJECT, required=DEFAULT_PROJECT is None)
    prepare_parser.add_argument("--location", default=DEFAULT_LOCATION)
    prepare_parser.add_argument("--bucket", default=DEFAULT_BUCKET, required=DEFAULT_BUCKET is None)
    prepare_parser.add_argument("--gcs-prefix", default="simpleqa-gptoss120b-batch")
    prepare_parser.add_argument("--model", default=DEFAULT_MODEL)
    prepare_parser.add_argument("--publisher-model", default=DEFAULT_PUBLISHER_MODEL)
    prepare_parser.add_argument("--num-samples", type=int, default=None)
    prepare_parser.add_argument(
        "--question-start",
        type=int,
        default=0,
        help="Skip this many questions in the deterministic run order.",
    )
    prepare_parser.add_argument("--split", default="all")
    prepare_parser.add_argument("--seed", type=int, default=0)
    prepare_parser.add_argument("--reasoning-effort", default="low")
    prepare_parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    prepare_parser.add_argument("--simpleqa-top-p", type=float, default=0.9)
    prepare_parser.add_argument("--idk-rho", type=float, default=0.5)
    prepare_parser.add_argument("--log-max-tokens", type=int, default=4000)
    prepare_parser.add_argument(
        "--log-repeats",
        type=int,
        default=1,
        help="Log reports per question; values above one match the consistency subcommand seeds.",
    )
    prepare_parser.add_argument("--log-temperature", type=float, default=0.0)
    prepare_parser.add_argument("--penalty", type=float, default=3.0)
    prepare_parser.add_argument("--penalty-samples", type=int, default=50)
    prepare_parser.add_argument("--penalty-temperature", type=float, default=1.0)
    prepare_parser.add_argument("--penalty-max-tokens", type=int, default=2048)
    prepare_parser.add_argument("--chunk-size", type=int, default=25_000)
    prepare_parser.add_argument("--batch-input-price-per-million", type=float, default=0.045)
    prepare_parser.add_argument("--cache-path", type=Path, default=Path("outputs/cache.sqlite"))
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.set_defaults(func=prepare)

    submit_parser = subparsers.add_parser("submit")
    add_plan_argument(submit_parser)
    submit_parser.add_argument("--max-jobs", type=int, default=None)
    submit_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Archive and resubmit chunks whose latest job is JOB_STATE_FAILED.",
    )
    submit_parser.add_argument("--wait", action="store_true")
    submit_parser.add_argument("--poll-seconds", type=float, default=60.0)
    submit_parser.set_defaults(func=submit)

    status_parser = subparsers.add_parser("status")
    add_plan_argument(status_parser)
    status_parser.set_defaults(func=status)

    cancel_parser = subparsers.add_parser("cancel")
    add_plan_argument(cancel_parser)
    cancel_parser.add_argument("--wait", action="store_true")
    cancel_parser.add_argument("--poll-seconds", type=float, default=30.0)
    cancel_parser.set_defaults(func=cancel)

    download_parser = subparsers.add_parser("download")
    add_plan_argument(download_parser)
    download_parser.set_defaults(func=download)

    import_parser = subparsers.add_parser("import-cache")
    add_plan_argument(import_parser)
    import_parser.add_argument("--cache-path", type=Path, default=None)
    import_parser.set_defaults(func=import_cache)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified runner for the SimpleQA uncertainty-elicitation experiments.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    simpleqa_parser = subparsers.add_parser(
        "simpleqa",
        help="Single-report log elicitation plus optional penalty arm.",
        description="SimpleQA top-p elicitation under the residual-rho log rule.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _simpleqa_add_arguments(simpleqa_parser)
    simpleqa_parser.set_defaults(func=_simpleqa_main)

    consistency_parser = subparsers.add_parser(
        "consistency",
        help="Repeated log elicitations for test-retest consistency.",
        description="Run repeated SimpleQA log elicitations for consistency plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _consistency_add_arguments(consistency_parser)
    consistency_parser.set_defaults(func=_consistency_main)

    batch_parser = subparsers.add_parser(
        "batch",
        help="GCP batch-prediction bridge (prepare/submit/status/cancel/download/import-cache).",
        description="GCP batch generation bridge for SimpleQA experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _batch_add_arguments(batch_parser)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
