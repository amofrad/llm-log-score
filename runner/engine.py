"""Core engine for the SimpleQA uncertainty-elicitation experiments: API clients, response caching, prompt construction, scoring, and LLM grading. Imported by run.py (data collection) and by the grading and analysis stages."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import sys
import threading
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

# The grading rubric and answer canonicalization are shared with the grading
# stage; grading/graders.py is their single source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "grading"))
from graders import (  # noqa: E402
    GRADER_SYSTEM_MESSAGE,
    canonical_incorrect_answer_display,
    canonical_incorrect_answer_key,
    canonical_stadium_answer,
    extract_short_answer_surface,
    is_open_not_attempted,
    normalize_open_answer_text,
    open_answer_key,
    parse_simpleqa_grade,
    simpleqa_grade_prompt,
)


DEFAULT_PENALTIES = [3.0]

DEFAULT_SYSTEM_MESSAGE = "You are a careful multiple-choice question answering system."

SIMPLEQA_LOG_SYSTEM_MESSAGE = (
    "You are a calibrated factual forecaster. You may reason if needed. "
    "End with exactly one valid JSON object in the required format."
)

SIMPLEQA_JSON_ONLY_SYSTEM_MESSAGE = (
    "You are a calibrated factual question-answering system. Return only one "
    "valid JSON object in the required format. Do not include reasoning, "
    "explanations, Markdown, or any prose before or after the JSON."
)

SIMPLEQA_LOG_JSON_ONLY_SYSTEM_MESSAGE = SIMPLEQA_JSON_ONLY_SYSTEM_MESSAGE

SIMPLEQA_QUADRATIC_SYSTEM_MESSAGE = (
    "You are a calibrated factual forecaster. You may reason if needed. "
    "End with exactly one valid JSON object in the required format."
)

OPEN_ANSWER_SYSTEM_MESSAGE = (
    "You are a careful factual question answering system. If you explain your "
    "reasoning, end with exactly one final line in the required format."
)

SIMPLEQA_PENALTY_JSON_ONLY_SYSTEM_MESSAGE = SIMPLEQA_JSON_ONLY_SYSTEM_MESSAGE

DEFAULT_GRADER_MAX_TOKENS = 128

ABSTAIN_LABEL = "ABSTAIN"

UNKNOWN_LABEL = "UNKNOWN"

DEFAULT_SIMPLEQA_DATASET = "OpenEvals/SimpleQA"

DEFAULT_SIMPLEQA_TOP_K = 5

DEFAULT_SIMPLEQA_TOP_P = 0.9

PROBABILITY_POINT_TOTAL = 100.0

LOG_SCORE_SHIFT = 5.0

PROVIDER_SEED_MAX = 2_147_483_647

RETRY_SEED_STRIDE = 1_000_003


@dataclass(frozen=True)
class SimpleQAExample:
    question_id: str
    question: str
    answer: str
    category: str
    answer_type: str = ""


class SQLiteCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        with self.lock:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at REAL NOT NULL)"
            )
            self.conn.commit()

    def get(self, key: str) -> Any | None:
        with self.lock:
            row = self.conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=True)
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO cache(key, value, created_at) VALUES (?, ?, ?)",
                (key, payload, time.time()),
            )
            self.conn.commit()


def cache_key_for_kwargs(kwargs: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()


def _first_numeric_value(data: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def token_usage_from_raw_response(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    usage = data.get("usage") or data.get("usageMetadata") or data.get("usage_metadata")
    if isinstance(usage, dict):
        input_tokens = _first_numeric_value(
            usage,
            (
                "prompt_tokens",
                "input_tokens",
                "promptTokenCount",
                "prompt_token_count",
                "cache_read_input_tokens",
            ),
        )
        output_tokens = _first_numeric_value(
            usage,
            (
                "completion_tokens",
                "output_tokens",
                "candidatesTokenCount",
                "candidates_token_count",
                "generated_tokens",
            ),
        )
        total_tokens = _first_numeric_value(
            usage,
            ("total_tokens", "totalTokenCount", "total_token_count"),
        )
        if total_tokens is None and (input_tokens is not None or output_tokens is not None):
            total_tokens = int(input_tokens or 0) + int(output_tokens or 0)
        if total_tokens is not None:
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "source": "provider_usage",
            }

    input_tokens = _first_numeric_value(data, ("prompt_eval_count", "prompt_tokens"))
    output_tokens = _first_numeric_value(data, ("eval_count", "completion_tokens"))
    total_tokens = _first_numeric_value(data, ("total_tokens",))
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)
    if total_tokens is None:
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "source": "provider_usage",
    }


def estimated_token_count(text: str | None) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(str(text)) / 4))


def estimated_generation_token_usage(
    *,
    prompt: str,
    system_message: str | None,
    response: str,
    input_multiplier: int = 1,
    output_multiplier: int = 1,
) -> dict[str, Any]:
    input_tokens = (
        estimated_token_count(system_message)
        + estimated_token_count(prompt)
        + 4
    ) * max(0, input_multiplier)
    output_tokens = estimated_token_count(response) * max(0, output_multiplier)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "source": "estimated_chars",
    }


def empty_token_usage_totals() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "source_counts": {},
    }


def add_token_usage(total: dict[str, Any], usage: dict[str, Any] | None) -> None:
    if usage is None:
        return
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if value is None:
            continue
        total[key] = int(total.get(key, 0) or 0) + int(value)
    source = str(usage.get("source") or "unknown")
    source_counts = total.setdefault("source_counts", {})
    source_counts[source] = int(source_counts.get(source, 0)) + 1


def generation_token_usage(
    *,
    client: Any | None,
    prompt: str,
    system_message: str | None,
    response: str,
) -> dict[str, Any]:
    usage = getattr(client, "last_token_usage", None) if client is not None else None
    if isinstance(usage, dict) and usage.get("total_tokens") is not None:
        return usage
    return estimated_generation_token_usage(
        prompt=prompt,
        system_message=system_message,
        response=response,
    )


def retry_after_seconds(response: Any) -> float | None:
    if response is None:
        return None
    retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    try:
        data = response.json()
    except Exception:  # noqa: BLE001
        return None
    metadata = data.get("error", {}).get("metadata", {}) if isinstance(data, dict) else {}
    for key in ["retry_after_seconds", "retry_after_seconds_raw"]:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    nested_headers = metadata.get("headers", {})
    if isinstance(nested_headers, dict):
        value = nested_headers.get("Retry-After") or nested_headers.get("retry-after")
        if value is not None:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                return None
    return None


class EmptyAssistantContentError(RuntimeError):
    """Raised when a provider returns no visible assistant text."""


class RetryableProviderResponseError(RuntimeError):
    """Raised when a provider returns a retryable error in a successful JSON body."""

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


def retry_after_seconds_from_data(data: Any) -> float | None:
    if not isinstance(data, dict):
        return None
    candidates: list[Any] = []
    for key in ("retry_after", "retry_after_seconds", "retryAfter"):
        candidates.append(data.get(key))
    error = data.get("error")
    if isinstance(error, dict):
        for key in ("retry_after", "retry_after_seconds", "retryAfter"):
            candidates.append(error.get(key))
        metadata = error.get("metadata")
        if isinstance(metadata, dict):
            for key in ("retry_after", "retry_after_seconds", "retry_after_seconds_raw"):
                candidates.append(metadata.get(key))
    for value in candidates:
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return None


def retryable_provider_error_message(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    error = data.get("error")
    parts: list[str] = []
    if isinstance(error, str):
        parts.append(error)
    elif isinstance(error, dict):
        for key in ("message", "status", "code", "reason"):
            value = error.get(key)
            if value is not None:
                parts.append(str(value))
    elif error is not None:
        parts.append(str(error))
    for key in ("message", "status"):
        value = data.get(key)
        if value is not None:
            parts.append(str(value))
    message = " ".join(parts).strip()
    lowered = message.lower()
    retryable_fragments = (
        "throttl",
        "too many concurrent",
        "rate limit",
        "rate-limit",
        "temporarily unavailable",
        "try again",
        "overloaded",
        "capacity",
    )
    if message and any(fragment in lowered for fragment in retryable_fragments):
        return message
    return None


def should_retry_exception(exc: BaseException) -> bool:
    if isinstance(exc, EmptyAssistantContentError):
        return False
    if isinstance(exc, RetryableProviderResponseError):
        return True
    response = getattr(exc, "response", None)
    if response is None:
        return True
    return int(response.status_code) in {408, 409, 425, 429, 500, 502, 503, 504}


def exception_with_response_body(exc: BaseException) -> BaseException:
    response = getattr(exc, "response", None)
    if response is None:
        return exc
    body = getattr(response, "text", "")
    body = body[:800].replace("\n", " ")
    if not body:
        return exc
    return RuntimeError(f"{exc}; response_body={body}")


def retry_delay_seconds(exc: BaseException, attempt: int, base_delay_s: float) -> float:
    max_retryable_provider_delay_s = 120.0
    provider_retry_after = getattr(exc, "retry_after_s", None)
    if provider_retry_after is not None:
        return max(0.0, float(provider_retry_after))
    response = getattr(exc, "response", None)
    retry_after = retry_after_seconds(response)
    if retry_after is not None:
        return retry_after
    jitter = random.uniform(0.0, 0.5 * base_delay_s)
    if isinstance(exc, RetryableProviderResponseError):
        delay = max(30.0, base_delay_s * (2**attempt))
        return min(max_retryable_provider_delay_s, delay) + jitter
    return base_delay_s * (2**attempt) + jitter


def is_local_base_url(url: str) -> bool:
    lowered = url.lower()
    return lowered.startswith(
        (
            "http://localhost",
            "http://127.0.0.1",
            "http://0.0.0.0",
            "http://[::1]",
        )
    )


def normalize_gcp_reasoning_effort(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized == "none":
        return "minimal"
    return normalized


def ollama_api_base_url(url: str) -> str:
    base = url.rstrip("/")
    for suffix in ("/v1", "/api"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def is_gcp_vertex_base_url(url: str) -> bool:
    return "aiplatform.googleapis.com" in url.lower()


def is_openai_platform_base_url(url: str) -> bool:
    return "api.openai.com" in url.lower()


def is_gcp_anthropic_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("claude-") or normalized.startswith("anthropic/claude-")


def is_gcp_xai_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("xai/") or normalized.startswith("grok-")


def is_gcp_deepseek_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("deepseek-ai/deepseek-") or normalized.startswith("deepseek/")


def is_gcp_google_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("google/") or normalized.startswith("gemini-")


def effective_gcp_reasoning_effort(model: str, effort: str | None) -> str | None:
    # Google MaaS DeepSeek docs expose reasoning via chat_template_kwargs.thinking,
    # not reasoning_effort. Keep unsupported effort controls out of the payload
    # and row metadata so "none" cannot accidentally become "minimal".
    if is_gcp_deepseek_model(model):
        return None
    return effort


def gcp_deepseek_thinking_value(*, model: str, base_url: str, disable_thinking: bool) -> bool | None:
    if is_gcp_vertex_base_url(base_url) and is_gcp_deepseek_model(model):
        return not disable_thinking
    return None


def gcp_deepseek_thinking_cache_value(
    *, model: str, base_url: str, disable_thinking: bool
) -> str | None:
    value = gcp_deepseek_thinking_value(
        model=model,
        base_url=base_url,
        disable_thinking=disable_thinking,
    )
    if value is None:
        return None
    return "enabled" if value else "disabled"


def simpleqa_log_system_message_for_model(
    model: str,
    *,
    strict_json: bool = False,
    strict_json_deepseek: bool = False,
) -> str:
    if strict_json or (strict_json_deepseek and is_gcp_deepseek_model(model)):
        return SIMPLEQA_LOG_JSON_ONLY_SYSTEM_MESSAGE
    return SIMPLEQA_LOG_SYSTEM_MESSAGE


def simpleqa_penalty_system_message(*, strict_json: bool = False) -> str:
    return SIMPLEQA_PENALTY_JSON_ONLY_SYSTEM_MESSAGE if strict_json else OPEN_ANSWER_SYSTEM_MESSAGE


def provider_request_seed(*, model: str, base_url: str, seed: int | None) -> int | None:
    if seed is None:
        return None
    normalized_seed = int(seed) % PROVIDER_SEED_MAX
    if is_gcp_vertex_base_url(base_url) and is_gcp_xai_model(model):
        return (normalized_seed % (PROVIDER_SEED_MAX - 1)) + 1
    return normalized_seed


def set_chat_completion_token_limit(payload: dict[str, Any], *, base_url: str, max_tokens: int) -> None:
    token_key = "max_completion_tokens" if is_openai_platform_base_url(base_url) else "max_tokens"
    payload[token_key] = max_tokens


def chat_completion_supports_temperature(*, base_url: str, temperature: float) -> bool:
    if not is_openai_platform_base_url(base_url):
        return True
    return abs(float(temperature) - 1.0) <= 1e-12


def gcp_anthropic_model_id(model: str) -> str:
    normalized = model.strip()
    if normalized.startswith("anthropic/"):
        normalized = normalized.split("/", 1)[1]
    return normalized.replace(".", "-")


def gcp_vertex_api_host(location: str) -> str:
    return "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"


def gcp_openapi_model_id(model: str) -> str:
    """Return the publisher-qualified model name required by Vertex OpenAPI."""
    normalized = model.strip()
    if normalized.startswith("publishers/"):
        parts = normalized.split("/")
        if len(parts) == 4 and parts[2] == "models":
            publisher, model_name = parts[1], parts[3]
            return f"{publisher}/{model_name}"
    if "/" not in normalized and normalized.lower().startswith("llama-"):
        return f"meta/{normalized}"
    return normalized


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        cache: SQLiteCache,
        base_url: str = "https://openrouter.ai/api/v1",
        cache_only: bool = False,
        timeout_s: int = 120,
        request_delay_s: float = 0.0,
        max_retries: int = 8,
        retry_base_delay_s: float = 5.0,
        disable_thinking: bool = True,
        gcp_reasoning_effort: str | None = None,
        openai_reasoning_effort: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.cache = cache
        self.base_url = base_url.rstrip("/")
        self.request_base_url = self.base_url
        self.cache_only = cache_only
        self.timeout_s = timeout_s
        self.request_delay_s = request_delay_s
        self.max_retries = max_retries
        self.retry_base_delay_s = retry_base_delay_s
        self.disable_thinking = disable_thinking
        self.gcp_reasoning_effort = gcp_reasoning_effort
        self.openai_reasoning_effort = openai_reasoning_effort
        self.request_lock = threading.Lock()
        self.last_request_at = 0.0
        self.last_token_usage: dict[str, Any] | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if "openrouter.ai" in self.base_url:
            headers["HTTP-Referer"] = "https://github.com/openai/hallucinations-paper-experiments"
            headers["X-Title"] = "MMLU-Pro log scoring vs penalty simulation"
        return headers

    def _wait_for_turn(self) -> None:
        if self.request_delay_s <= 0:
            return
        with self.request_lock:
            now = time.time()
            wait_s = self.request_delay_s - (now - self.last_request_at)
            if wait_s > 0:
                time.sleep(wait_s)
            self.last_request_at = time.time()

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system_message: str | None = DEFAULT_SYSTEM_MESSAGE,
        seed: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int | None = None,
    ) -> str:
        use_ollama_native = is_local_base_url(self.base_url)
        request_seed = provider_request_seed(model=model, base_url=self.base_url, seed=seed)
        request_gcp_reasoning_effort = effective_gcp_reasoning_effort(
            model, self.gcp_reasoning_effort
        )
        request_gcp_deepseek_thinking = gcp_deepseek_thinking_cache_value(
            model=model,
            base_url=self.base_url,
            disable_thinking=self.disable_thinking,
        )
        request_temperature = (
            temperature
            if use_ollama_native
            or chat_completion_supports_temperature(base_url=self.base_url, temperature=temperature)
            else None
        )
        kwargs = {
            "model": model,
            "prompt": prompt,
            "system_message": system_message,
            "seed": seed,
            "temperature": temperature,
            "request_temperature": request_temperature,
            "max_tokens": max_tokens,
            "base_url": self.base_url,
            "disable_thinking": self.disable_thinking,
            "openrouter_reasoning_effort": (
                "none" if self.disable_thinking and "openrouter.ai" in self.base_url else None
            ),
            "gcp_reasoning_effort": (
                request_gcp_reasoning_effort
                if is_gcp_vertex_base_url(self.base_url)
                else None
            ),
            "openai_reasoning_effort": (
                self.openai_reasoning_effort
                if is_openai_platform_base_url(self.base_url)
                else None
            ),
            "transport": "ollama_native" if use_ollama_native else "openai_chat_completions",
        }
        if request_gcp_deepseek_thinking is not None:
            kwargs["gcp_deepseek_thinking"] = request_gcp_deepseek_thinking
        if request_temperature == temperature:
            kwargs.pop("request_temperature")
        if request_seed != seed:
            kwargs["request_seed"] = request_seed
        key = cache_key_for_kwargs(kwargs)
        cached = self.cache.get(key)
        if cached is not None:
            self.last_token_usage = token_usage_from_raw_response(cached.get("raw"))
            return str(cached["content"])
        if self.cache_only:
            raise RuntimeError(f"Cache miss for model={model!r}; cache-only mode is enabled.")

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Install requests or run with --mock.") from exc

        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        if use_ollama_native:
            endpoint = f"{ollama_api_base_url(self.base_url)}/api/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            if seed is not None:
                payload["options"]["seed"] = seed
            if self.disable_thinking:
                payload["think"] = False
        else:
            endpoint = f"{self.request_base_url}/chat/completions"
            payload = {
                "model": (
                    gcp_openapi_model_id(model)
                    if is_gcp_vertex_base_url(self.base_url)
                    else model
                ),
                "messages": messages,
            }
            if request_temperature is not None:
                payload["temperature"] = request_temperature
            set_chat_completion_token_limit(payload, base_url=self.base_url, max_tokens=max_tokens)
            if self.disable_thinking and "openrouter.ai" in self.base_url:
                payload["reasoning"] = {"effort": "none", "exclude": True}
                payload["include_reasoning"] = False
            deepseek_thinking = gcp_deepseek_thinking_value(
                model=model,
                base_url=self.base_url,
                disable_thinking=self.disable_thinking,
            )
            if deepseek_thinking is not None:
                payload["chat_template_kwargs"] = {"thinking": deepseek_thinking}
            if request_gcp_reasoning_effort and is_gcp_vertex_base_url(self.base_url):
                payload["reasoning_effort"] = request_gcp_reasoning_effort
            if self.openai_reasoning_effort and is_openai_platform_base_url(self.base_url):
                payload["reasoning_effort"] = self.openai_reasoning_effort
            if request_seed is not None:
                payload["seed"] = request_seed

        retry_count = self.max_retries if max_retries is None else max_retries
        last_error: Exception | None = None
        for attempt in range(retry_count):
            try:
                self._wait_for_turn()
                response = requests.post(
                    endpoint,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                data = response.json()
                if use_ollama_native:
                    message = data.get("message", {})
                    content = message.get("content") or ""
                    reasoning = message.get("thinking") or ""
                    finish_reason = data.get("done_reason") or data.get("done")
                else:
                    if "choices" not in data:
                        retryable_message = retryable_provider_error_message(data)
                        if retryable_message is not None:
                            raise RetryableProviderResponseError(
                                f"No choices in retryable provider response: {data}",
                                retry_after_s=retry_after_seconds_from_data(data),
                            )
                        raise RuntimeError(f"No choices in response: {data}")
                    choice = data["choices"][0]
                    message = choice.get("message", {})
                    content = message.get("content") or ""
                    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
                    finish_reason = choice.get("finish_reason")
                if not str(content).strip():
                    self.last_token_usage = token_usage_from_raw_response(data)
                    if reasoning:
                        raise EmptyAssistantContentError(
                            "Model returned empty assistant content after producing "
                            f"{len(str(reasoning))} characters of reasoning; "
                            f"finish_reason={finish_reason!r}. Disable thinking or raise max_tokens."
                        )
                    raise EmptyAssistantContentError(
                        f"Model returned empty assistant content; finish_reason={finish_reason!r}."
                    )
                self.last_token_usage = token_usage_from_raw_response(data)
                self.cache.set(key, {"content": content, "raw": data})
                return str(content)
            except Exception as exc:  # noqa: BLE001
                last_error = exception_with_response_body(exc)
                if attempt + 1 == retry_count or not should_retry_exception(exc):
                    break
                time.sleep(retry_delay_seconds(exc, attempt, self.retry_base_delay_s))
        if isinstance(last_error, EmptyAssistantContentError):
            raise last_error
        raise RuntimeError(f"Generation failed for {model}: {last_error}") from last_error

    def first_token_logprobs(
        self,
        *,
        model: str,
        prompt: str,
        system_message: str | None = DEFAULT_SYSTEM_MESSAGE,
        seed: int | None = None,
        temperature: float = 1.0,
        top_logprobs: int = 50,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        use_ollama_native = is_local_base_url(self.base_url)
        request_seed = provider_request_seed(model=model, base_url=self.base_url, seed=seed)
        request_gcp_reasoning_effort = effective_gcp_reasoning_effort(
            model, self.gcp_reasoning_effort
        )
        request_gcp_deepseek_thinking = gcp_deepseek_thinking_cache_value(
            model=model,
            base_url=self.base_url,
            disable_thinking=self.disable_thinking,
        )
        request_temperature = (
            temperature
            if use_ollama_native
            or chat_completion_supports_temperature(base_url=self.base_url, temperature=temperature)
            else None
        )
        kwargs = {
            "model": model,
            "prompt": prompt,
            "system_message": system_message,
            "seed": seed,
            "temperature": temperature,
            "request_temperature": request_temperature,
            "top_logprobs": top_logprobs,
            "base_url": self.base_url,
            "disable_thinking": self.disable_thinking,
            "openrouter_reasoning_effort": (
                "none" if self.disable_thinking and "openrouter.ai" in self.base_url else None
            ),
            "gcp_reasoning_effort": (
                request_gcp_reasoning_effort
                if is_gcp_vertex_base_url(self.base_url)
                else None
            ),
            "openai_reasoning_effort": (
                self.openai_reasoning_effort
                if is_openai_platform_base_url(self.base_url)
                else None
            ),
            "transport": "ollama_native_logprobs" if use_ollama_native else "openai_chat_completions_logprobs",
        }
        if request_gcp_deepseek_thinking is not None:
            kwargs["gcp_deepseek_thinking"] = request_gcp_deepseek_thinking
        if request_temperature == temperature:
            kwargs.pop("request_temperature")
        if request_seed != seed:
            kwargs["request_seed"] = request_seed
        key = cache_key_for_kwargs(kwargs)
        cached = self.cache.get(key)
        if cached is not None:
            return dict(cached)
        if self.cache_only:
            raise RuntimeError(f"Cache miss for model={model!r}; cache-only mode is enabled.")

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Install requests or run with --mock.") from exc

        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        capped_top_logprobs = max(0, min(20, top_logprobs))
        if use_ollama_native:
            endpoint = f"{ollama_api_base_url(self.base_url)}/api/chat"
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": False,
                "think": False if self.disable_thinking else None,
                "logprobs": True,
                "top_logprobs": capped_top_logprobs,
                "options": {
                    "temperature": temperature,
                    "num_predict": 1,
                },
            }
            if payload["think"] is None:
                payload.pop("think")
            if seed is not None:
                payload["options"]["seed"] = seed
        else:
            endpoint = f"{self.request_base_url}/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "logprobs": True,
                "top_logprobs": capped_top_logprobs,
            }
            if request_temperature is not None:
                payload["temperature"] = request_temperature
            set_chat_completion_token_limit(payload, base_url=self.base_url, max_tokens=1)
            if self.disable_thinking and "openrouter.ai" in self.base_url:
                payload["reasoning"] = {"effort": "none", "exclude": True}
                payload["include_reasoning"] = False
            deepseek_thinking = gcp_deepseek_thinking_value(
                model=model,
                base_url=self.base_url,
                disable_thinking=self.disable_thinking,
            )
            if deepseek_thinking is not None:
                payload["chat_template_kwargs"] = {"thinking": deepseek_thinking}
            if request_gcp_reasoning_effort and is_gcp_vertex_base_url(self.base_url):
                payload["reasoning_effort"] = request_gcp_reasoning_effort
            if self.openai_reasoning_effort and is_openai_platform_base_url(self.base_url):
                payload["reasoning_effort"] = self.openai_reasoning_effort
            if request_seed is not None:
                payload["seed"] = request_seed

        def normalize_chat_logprobs(logprobs_payload: Any) -> list[dict[str, Any]]:
            if not logprobs_payload:
                return []
            if isinstance(logprobs_payload, list):
                return [entry for entry in logprobs_payload if isinstance(entry, dict)]
            if not isinstance(logprobs_payload, dict):
                return []
            content_positions = logprobs_payload.get("content")
            if isinstance(content_positions, list):
                return [entry for entry in content_positions if isinstance(entry, dict)]
            if isinstance(content_positions, dict):
                return [content_positions]
            tokens = logprobs_payload.get("tokens")
            token_logprobs = logprobs_payload.get("token_logprobs")
            top_logprobs_payload = logprobs_payload.get("top_logprobs")
            if isinstance(tokens, list) and isinstance(token_logprobs, list):
                normalized: list[dict[str, Any]] = []
                for i, token in enumerate(tokens):
                    top_items: list[dict[str, Any]] = []
                    if isinstance(top_logprobs_payload, list) and i < len(top_logprobs_payload):
                        raw_top = top_logprobs_payload[i]
                        if isinstance(raw_top, dict):
                            top_items = [
                                {"token": str(top_token), "logprob": top_logprob}
                                for top_token, top_logprob in raw_top.items()
                            ]
                        elif isinstance(raw_top, list):
                            top_items = [
                                item for item in raw_top if isinstance(item, dict)
                            ]
                    normalized.append(
                        {
                            "token": str(token),
                            "logprob": token_logprobs[i] if i < len(token_logprobs) else None,
                            "top_logprobs": top_items,
                        }
                    )
                return normalized
            return []

        retry_count = self.max_retries if max_retries is None else max_retries
        last_error: Exception | None = None
        for attempt in range(retry_count):
            try:
                self._wait_for_turn()
                response = requests.post(
                    endpoint,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                data = response.json()
                if use_ollama_native:
                    message = data.get("message", {})
                    content = message.get("content", "")
                    logprobs = data.get("logprobs") or message.get("logprobs") or []
                else:
                    if "choices" not in data:
                        retryable_message = retryable_provider_error_message(data)
                        if retryable_message is not None:
                            raise RetryableProviderResponseError(
                                f"No choices in retryable provider response: {data}",
                                retry_after_s=retry_after_seconds_from_data(data),
                            )
                        raise RuntimeError(f"No choices in response: {data}")
                    choice = data["choices"][0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    logprobs = normalize_chat_logprobs(choice.get("logprobs"))
                result = {
                    "content": content,
                    "logprobs": logprobs,
                    "raw": data,
                }
                self.cache.set(key, result)
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exception_with_response_body(exc)
                if attempt + 1 == retry_count or not should_retry_exception(exc):
                    break
                time.sleep(retry_delay_seconds(exc, attempt, self.retry_base_delay_s))
        raise RuntimeError(f"Logprob generation failed for {model}: {last_error}") from last_error


def gcp_vertex_openai_base_url(project: str, location: str) -> str:
    return (
        "https://aiplatform.googleapis.com/v1/"
        f"projects/{project}/locations/{location}/endpoints/openapi"
    )


def gcp_vertex_openai_request_base_url(project: str, location: str) -> str:
    host = gcp_vertex_api_host(location)
    return (
        f"https://{host}/v1/"
        f"projects/{project}/locations/{location}/endpoints/openapi"
    )


class GCPVertexClient(OpenRouterClient):
    def __init__(
        self,
        *,
        project: str,
        location: str,
        cache: SQLiteCache,
        cache_only: bool = False,
        timeout_s: int = 120,
        request_delay_s: float = 0.0,
        max_retries: int = 8,
        retry_base_delay_s: float = 5.0,
        disable_thinking: bool = True,
        gcp_reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(
            api_key="",
            cache=cache,
            base_url=gcp_vertex_openai_base_url(project, location),
            cache_only=cache_only,
            timeout_s=timeout_s,
            request_delay_s=request_delay_s,
            max_retries=max_retries,
            retry_base_delay_s=retry_base_delay_s,
            disable_thinking=disable_thinking,
            gcp_reasoning_effort=gcp_reasoning_effort,
        )
        # Keep the historical base_url in cache keys while sending online
        # requests through Vertex's location-specific hostname.
        self.request_base_url = gcp_vertex_openai_request_base_url(project, location)
        self.project = project
        self.location = location
        self._credentials: Any | None = None
        self._credentials_lock = threading.Lock()

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._access_token()}",
        }

    def _access_token(self) -> str:
        try:
            import google.auth
            import google.auth.transport.requests
        except ImportError as exc:
            raise RuntimeError(
                "Install google-auth to use --answer-provider gcp. "
                "It is included in requirements.txt."
            ) from exc

        with self._credentials_lock:
            if self._credentials is None:
                self._credentials, _project = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            if not self._credentials.valid or self._credentials.expired:
                self._credentials.refresh(google.auth.transport.requests.Request())
            token = getattr(self._credentials, "token", None)
            if not token:
                raise RuntimeError("Could not obtain a Google Cloud access token.")
            return str(token)


class GCPAnthropicVertexClient:
    def __init__(
        self,
        *,
        project: str,
        location: str,
        cache: SQLiteCache,
        cache_only: bool = False,
        timeout_s: int = 120,
        request_delay_s: float = 0.0,
        max_retries: int = 8,
        retry_base_delay_s: float = 5.0,
        disable_thinking: bool = True,
    ) -> None:
        self.project = project
        self.location = location
        self.cache = cache
        self.cache_only = cache_only
        self.timeout_s = timeout_s
        self.request_delay_s = request_delay_s
        self.max_retries = max_retries
        self.retry_base_delay_s = retry_base_delay_s
        self.disable_thinking = disable_thinking
        self.request_lock = threading.Lock()
        self.last_request_at = 0.0
        self.last_token_usage: dict[str, Any] | None = None
        self._credentials: Any | None = None
        self._credentials_lock = threading.Lock()

    def _wait_for_turn(self) -> None:
        if self.request_delay_s <= 0:
            return
        with self.request_lock:
            now = time.time()
            wait_s = self.request_delay_s - (now - self.last_request_at)
            if wait_s > 0:
                time.sleep(wait_s)
            self.last_request_at = time.time()

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._access_token()}",
        }

    def _access_token(self) -> str:
        try:
            import google.auth
            import google.auth.transport.requests
        except ImportError as exc:
            raise RuntimeError(
                "Install google-auth to use --answer-provider gcp. "
                "It is included in requirements.txt."
            ) from exc

        with self._credentials_lock:
            if self._credentials is None:
                self._credentials, _project = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            if not self._credentials.valid or self._credentials.expired:
                self._credentials.refresh(google.auth.transport.requests.Request())
            token = getattr(self._credentials, "token", None)
            if not token:
                raise RuntimeError("Could not obtain a Google Cloud access token.")
            return str(token)

    def _endpoint(self, model: str) -> str:
        model_id = gcp_anthropic_model_id(model)
        return (
            f"https://{gcp_vertex_api_host(self.location)}/v1/projects/{self.project}"
            f"/locations/{self.location}/publishers/anthropic/models/{model_id}:rawPredict"
        )

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system_message: str | None = DEFAULT_SYSTEM_MESSAGE,
        seed: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int | None = None,
    ) -> str:
        kwargs = {
            "provider": "gcp_vertex_anthropic_raw_predict",
            "model": gcp_anthropic_model_id(model),
            "prompt": prompt,
            "system_message": system_message,
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "project": self.project,
            "location": self.location,
            "disable_thinking": self.disable_thinking,
        }
        key = cache_key_for_kwargs(kwargs)
        cached = self.cache.get(key)
        if cached is not None:
            self.last_token_usage = token_usage_from_raw_response(cached.get("raw"))
            return str(cached["content"])
        if self.cache_only:
            raise RuntimeError(f"Cache miss for model={model!r}; cache-only mode is enabled.")

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Install requests or run with --mock.") from exc

        payload: dict[str, Any] = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
            "temperature": temperature,
        }
        if system_message:
            payload["system"] = system_message

        retry_count = self.max_retries if max_retries is None else max_retries
        last_error: Exception | None = None
        for attempt in range(retry_count):
            try:
                self._wait_for_turn()
                response = requests.post(
                    self._endpoint(model),
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                data = response.json()
                parts = data.get("content", [])
                content = "\n".join(
                    str(part.get("text", ""))
                    for part in parts
                    if isinstance(part, dict) and part.get("type") == "text"
                )
                if not content.strip():
                    stop_reason = data.get("stop_reason")
                    raise RuntimeError(
                        f"No text content in Vertex Claude response; stop_reason={stop_reason!r}; "
                        f"response={data}"
                    )
                self.last_token_usage = token_usage_from_raw_response(data)
                self.cache.set(key, {"content": content, "raw": data})
                return content
            except Exception as exc:  # noqa: BLE001
                last_error = exception_with_response_body(exc)
                if attempt + 1 == retry_count or not should_retry_exception(exc):
                    break
                time.sleep(retry_delay_seconds(exc, attempt, self.retry_base_delay_s))
        raise RuntimeError(f"Generation failed for {model}: {last_error}") from last_error

    def first_token_logprobs(self, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("Vertex Claude does not expose first-token logprobs through this client.")


def create_gcp_client_for_model(
    *,
    model: str,
    project: str,
    location: str,
    cache: SQLiteCache,
    cache_only: bool,
    request_delay_s: float,
    max_retries: int,
    retry_base_delay_s: float,
    disable_thinking: bool,
    gcp_reasoning_effort: str | None,
) -> Any:
    client_class = GCPAnthropicVertexClient if is_gcp_anthropic_model(model) else GCPVertexClient
    kwargs: dict[str, Any] = {
        "project": project,
        "location": location,
        "cache": cache,
        "cache_only": cache_only,
        "request_delay_s": request_delay_s,
        "max_retries": max_retries,
        "retry_base_delay_s": retry_base_delay_s,
        "disable_thinking": disable_thinking,
    }
    if client_class is GCPVertexClient:
        kwargs["gcp_reasoning_effort"] = gcp_reasoning_effort
    client = client_class(**kwargs)
    if client_class is GCPVertexClient and is_gcp_google_model(model):
        # Google-hosted Gemini models use the global OpenAPI hostname even
        # when the experiment records a regional grader location.
        client.request_base_url = gcp_vertex_openai_base_url(project, location)
    return client


def stable_rng(*parts: Any) -> random.Random:
    text = json.dumps(parts, sort_keys=True, default=str)
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)
    return random.Random(seed)


def threshold_from_penalty(penalty: float) -> float:
    if penalty < 0:
        raise ValueError("Penalty must be nonnegative.")
    return penalty / (1.0 + penalty)


def format_simpleqa_question(example: SimpleQAExample) -> str:
    return f"Question: {example.question}"


def build_simpleqa_regular_prompt(example: SimpleQAExample) -> str:
    return f"""{format_simpleqa_question(example)}

[Answer the question.]
End with exactly one final line:
Final: <short answer>
"""


def build_simpleqa_penalty_prompt(example: SimpleQAExample, penalty: float) -> str:
    tau = threshold_from_penalty(penalty)
    scoring_text = (
        f"Scoring: +1 point if your final answer is correct; "
        f"-{penalty:g} points if your final answer is incorrect; "
        f"0 points if your final answer is {ABSTAIN_LABEL}."
    )
    if tau == 0:
        rubric = (
            f"{scoring_text} Since wrong answers have no penalty, never abstain. "
            "Make your best guess if unsure."
        )
    else:
        threshold_text = f"{100 * tau:.0f}%"
        rubric = (
            f"{scoring_text} This means you should answer only if your probability "
            f"of being correct is greater than {threshold_text}; otherwise {ABSTAIN_LABEL}."
        )
    return f"""{format_simpleqa_question(example)}

[{rubric}]
Answer the question. End with exactly one final line:
Final: <short answer or {ABSTAIN_LABEL}>
"""


def build_simpleqa_penalty_json_prompt(example: SimpleQAExample, penalty: float) -> str:
    tau = threshold_from_penalty(penalty)
    scoring_text = (
        f"Scoring: +1 point if your answer is correct; "
        f"-{penalty:g} points if your answer is incorrect; "
        f"0 points if your answer is {ABSTAIN_LABEL}."
    )
    if tau == 0:
        rubric = (
            f"{scoring_text} Since wrong answers have no penalty, never abstain. "
            "Make your best guess if unsure."
        )
    else:
        threshold_text = f"{100 * tau:.0f}%"
        rubric = (
            f"{scoring_text} This means you should answer only if your probability "
            f"of being correct is greater than {threshold_text}; otherwise {ABSTAIN_LABEL}."
        )
    return f"""{format_simpleqa_question(example)}

[{rubric}]

Return exactly one JSON object:
{{"answer": "<answer or {ABSTAIN_LABEL}>"}}
"""


def build_simpleqa_topk_log_prompt(example: SimpleQAExample, top_k: int) -> str:
    return f"""{format_simpleqa_question(example)}

[Instead of giving one answer, report your probability distribution over the
{top_k} most plausible short answers. Assign exactly 100 whole-number points
across those answers. If answer X is correct, your score is
log(points assigned to X / 100) + {LOG_SCORE_SHIFT:g}. To maximize expected
score, assign points in proportion to how likely each answer is correct.]

Return exactly one JSON object:
{{"answers": [{{"answer": "...", "points": ...}}, ...]}}
"""


def build_simpleqa_topp_log_prompt(example: SimpleQAExample, top_p: float) -> str:
    target_points = max(0.0, min(100.0, 100.0 * top_p))
    return f"""{format_simpleqa_question(example)}

[Instead of giving one answer, report a Top-P probability set. List as many
plausible short answers as needed so that the points you list sum to at least
{target_points:g} and at most 100. Use fewer answers when your belief is
concentrated and more answers when your belief is spread out. You may include
the answer "I don't know" and assign it points for probability mass where you
do not know enough to give a factual answer. If answer X is correct, your score
is log(points assigned to X / 100) + {LOG_SCORE_SHIFT:g}. To maximize expected
score, use your actual probabilities.]

Return exactly one JSON object:
{{"answers": [{{"answer": "...", "points": ...}}, ...]}}
"""


def build_simpleqa_topk_quadratic_prompt(example: SimpleQAExample, top_k: int) -> str:
    return f"""{format_simpleqa_question(example)}

[Instead of giving one answer, report your probability distribution over the
{top_k} most plausible short answers. Assign exactly 100 whole-number points
across those answers. If answer X is correct, let q_X be the points assigned
to X divided by 100. Your score is
q_X - 0.5 * sum_j q_j^2,
where the sum is over the answers you list. This quadratic scoring rule is
strictly proper for normalized probability reports, so to maximize expected
score you should report your actual probabilities.]

Return exactly one JSON object:
{{"answers": [{{"answer": "...", "points": ...}}, ...]}}
"""


def build_simpleqa_topp_quadratic_prompt(example: SimpleQAExample, top_p: float) -> str:
    target_points = max(0.0, min(100.0, 100.0 * top_p))
    return f"""{format_simpleqa_question(example)}

[Instead of giving one answer, report a probability distribution over plausible
short answers. Assign exactly 100 whole-number points across the answers you
list. Include the answer "I don't know" or "Other" for probability mass where
you do not know enough to name a concrete factual answer. Your top answers
should contain at least {target_points:g} points of probability mass before any
residual "I don't know" or "Other" mass when possible.

If answer X is correct, let q_X be the points assigned to X divided by 100.
Your score is
q_X - 0.5 * sum_j q_j^2,
where the sum is over the answers you list. This quadratic scoring rule is
strictly proper for normalized probability reports, so to maximize expected
score you should report your actual probabilities.]

Return exactly one JSON object:
{{"answers": [{{"answer": "...", "points": ...}}, ...]}}
"""


def sample_simpleqa_examples(
    rows: list[SimpleQAExample],
    *,
    num_samples: int | None,
    seed: int,
    sample_per_category: int | None,
) -> list[SimpleQAExample]:
    rng = random.Random(seed)
    if sample_per_category is not None:
        by_category: dict[str, list[SimpleQAExample]] = {}
        for ex in rows:
            by_category.setdefault(ex.category, []).append(ex)
        sampled: list[SimpleQAExample] = []
        for _cat, cat_rows in sorted(by_category.items()):
            rng.shuffle(cat_rows)
            sampled.extend(cat_rows[:sample_per_category])
        rng.shuffle(sampled)
        return sampled
    rng.shuffle(rows)
    if num_samples is not None:
        return rows[:num_samples]
    return rows


def load_simpleqa(
    *,
    split: str,
    num_samples: int | None,
    seed: int,
    categories: set[str] | None,
    sample_per_category: int | None,
    dataset_name: str = DEFAULT_SIMPLEQA_DATASET,
) -> list[SimpleQAExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets from requirements.txt or run with --mock.") from exc

    ds = load_dataset(dataset_name, split=split)
    rows: list[SimpleQAExample] = []
    for row_i, row in enumerate(ds):
        metadata = row.get("metadata") if isinstance(row, dict) else None
        metadata = metadata if isinstance(metadata, dict) else {}
        category = str(
            row.get("topic")
            or metadata.get("topic")
            or row.get("category")
            or "unknown"
        )
        if categories and category not in categories:
            continue
        question = str(row.get("problem") or row.get("question") or "").strip()
        answer = str(row.get("answer") or row.get("target") or "").strip()
        if not question or not answer:
            continue
        rows.append(
            SimpleQAExample(
                question_id=str(row.get("id") or row.get("question_id") or f"simpleqa-{row_i}"),
                question=question,
                answer=answer,
                category=category,
                answer_type=str(row.get("answer_type") or metadata.get("answer_type") or ""),
            )
        )

    return sample_simpleqa_examples(
        rows,
        num_samples=num_samples,
        seed=seed,
        sample_per_category=sample_per_category,
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for match in re.finditer(r"{", text):
        start = match.start()
        try:
            obj, end_offset = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            candidates.append((start, start + end_offset, obj))
    if not candidates:
        return None
    top_level_candidates = [
        candidate
        for candidate in candidates
        if not any(
            other_start < candidate[0] and candidate[1] <= other_end
            for other_start, other_end, _other_obj in candidates
        )
    ]
    return (top_level_candidates or candidates)[-1][2]


_SIMPLEQA_TOLERANT_ANSWER_ITEM = re.compile(
    r"""
    \{
      \s*"(?P<answer_key>answer|response|text)"\s*:
      \s*(?P<answer>.*?)
      \s*,\s*"(?P<points_key>points|probability_points|score|probability)"\s*:
      \s*(?P<points>
        [-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?
      )
      \s*
    \}
    """,
    re.DOTALL | re.VERBOSE,
)


def _decode_tolerant_json_answer(token: str) -> tuple[str | None, set[str]]:
    """Recover an answer token without guessing its semantic content."""
    value = token.strip()
    repairs: set[str] = set()
    if not value:
        return None, repairs

    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        inner = value[1:-1]
        if re.search(r'(?<!\\)"', inner):
            repairs.add("embedded_quote")
            inner = re.sub(r'(?<!\\)"', r'\\"', inner)
        if re.search(r'\\(?!["\\/bfnrtu])', inner):
            repairs.add("invalid_escape")
            inner = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", inner)
        try:
            return str(json.loads(f'"{inner}"')), repairs
        except json.JSONDecodeError:
            return value[1:-1], repairs | {"string_fallback"}

    repairs.add("unquoted_answer")
    return value, repairs


def extract_repaired_simpleqa_topp_object(
    text: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Recover the requested answer/points schema after strict JSON failure.

    The fallback only recognizes complete answer/points objects. It preserves
    answer text verbatim apart from decoding string syntax and treats an
    unquoted answer value as text, so it does not infer or correct an answer.
    """
    answers: list[dict[str, Any]] = []
    repairs: set[str] = set()
    for match in _SIMPLEQA_TOLERANT_ANSWER_ITEM.finditer(text):
        answer, answer_repairs = _decode_tolerant_json_answer(match.group("answer"))
        if answer is None:
            continue
        points_key = match.group("points_key")
        answers.append(
            {
                "answer": answer,
                points_key: float(match.group("points")),
            }
        )
        repairs.update(answer_repairs)

    if not answers:
        return None, None
    repair_label = "_".join(sorted(repairs)) or "item_extraction"
    return {"answers": answers}, f"repaired_{repair_label}"


def scaled_log_score(q_true: float, eps: float) -> float:
    if not 0.0 < eps < 1.0:
        raise ValueError("--eps must be strictly between 0 and 1 for scaled log scoring.")
    q = min(1.0, max(eps, float(q_true)))
    return math.log(q / eps) / math.log(1.0 / eps)


def shifted_log_score(q_true: float, eps: float) -> float:
    q = min(1.0, max(eps, float(q_true)))
    return math.log(q) + LOG_SCORE_SHIFT


def parse_final_text_response(text: str) -> str | None:
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    final_line_pattern = re.compile(
        r"^\s*(?:#+\s*)?(?:final(?:\s+answer)?|answer)\s*:\s*(.+?)\s*$",
        re.IGNORECASE,
    )
    for line in reversed(nonempty_lines):
        match = final_line_pattern.match(line.strip().strip("*_`"))
        if match:
            value = match.group(1).strip().strip("*_`").strip()
            return value or None

    obj = extract_json_object(text)
    if obj is not None:
        for key in ["answer", "final_answer", "response", "selected_answer"]:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    stripped = text.strip()
    return stripped or None


def canonical_simpleqa_answer_key(
    *,
    example: SimpleQAExample,
    answer: str,
    grade: str,
) -> tuple[str, str]:
    if grade == "correct":
        return f"correct::{open_answer_key(example.answer)}", normalize_open_answer_text(example.answer)
    if grade == "not_attempted":
        return "not_attempted", "I don't know"
    if "stadium" in example.question.casefold() and "world cup" in example.question.casefold():
        stadium_answer = canonical_stadium_answer(answer)
        if stadium_answer is not None:
            key, display = stadium_answer
            return f"incorrect::{key}", display
    display = canonical_incorrect_answer_display(answer)
    return f"incorrect::{canonical_incorrect_answer_key(answer)}", display


def canonicalize_simpleqa_distribution(
    distribution: list[dict[str, Any]],
    *,
    example: SimpleQAExample,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in distribution:
        answer = str(item.get("answer", ""))
        grade = str(item.get("grade", "incorrect"))
        key, display = canonical_simpleqa_answer_key(example=example, answer=answer, grade=grade)
        probability = float(item.get("probability", 0.0))
        count = int(item.get("count", 0) or 0)
        points = float(item.get("points", probability * PROBABILITY_POINT_TOTAL) or 0.0)
        if key not in merged:
            merged[key] = {
                "answer": display,
                "grade": grade,
                "probability": 0.0,
                "points": 0.0,
                "source_answers": [],
            }
            if "count" in item:
                merged[key]["count"] = 0
        if "count" in item:
            merged[key]["count"] = int(merged[key].get("count", 0)) + count
        merged[key]["probability"] = float(merged[key]["probability"]) + probability
        merged[key]["points"] = float(merged[key]["points"]) + points
        source_answers = merged[key].setdefault("source_answers", [])
        if answer and answer not in source_answers:
            source_answers.append(answer)
        if len(display) < len(str(merged[key]["answer"])) and grade != "correct":
            merged[key]["answer"] = display

    rows = list(merged.values())
    for row in rows:
        row["points"] = float(row.get("probability", 0.0)) * PROBABILITY_POINT_TOTAL
    for row in rows:
        if len(row.get("source_answers", [])) <= 1:
            row.pop("source_answers", None)
    return sorted(rows, key=lambda row: float(row.get("probability", 0.0)), reverse=True)


def simpleqa_grade_counts_from_graded_distribution(
    distribution: list[dict[str, Any]],
) -> dict[str, int]:
    grade_counts = {"correct": 0, "incorrect": 0, "not_attempted": 0}
    for row in distribution:
        grade = str(row.get("grade", "incorrect"))
        count = int(row.get("count", 0) or 0)
        grade_counts[grade] = grade_counts.get(grade, 0) + count
    return grade_counts


def grade_simpleqa_response(
    client: Any,
    *,
    grader_model: str,
    example: SimpleQAExample,
    response: str,
    seed: int,
) -> str:
    if is_open_not_attempted(response):
        return "not_attempted"
    try:
        output = client.generate(
            model=grader_model,
            prompt=simpleqa_grade_prompt(example, response),
            system_message=GRADER_SYSTEM_MESSAGE,
            seed=seed,
            temperature=0,
            max_tokens=DEFAULT_GRADER_MAX_TOKENS,
        )
    except EmptyAssistantContentError:
        return "incorrect"
    return parse_simpleqa_grade(output) or "incorrect"


def exact_simpleqa_grade(example: SimpleQAExample, response: str) -> str:
    if is_open_not_attempted(response):
        return "not_attempted"
    return "correct" if open_answer_key(response) == open_answer_key(example.answer) else "incorrect"


def parse_simpleqa_topp_response(
    response: str,
    *,
    top_p: float,
) -> tuple[list[dict[str, Any]] | None, str]:
    obj = extract_json_object(response)
    repair_status = None
    answer_container_keys = {
        "answers",
        "candidate_answers",
        "responses",
        "probability_points",
    }
    if obj is None or not answer_container_keys.intersection(obj):
        repaired_obj, repair_status = extract_repaired_simpleqa_topp_object(response)
        if repaired_obj is not None:
            obj = repaired_obj
    if obj is None:
        return None, "unparsed"

    raw_answers = (
        obj.get("answers")
        or obj.get("candidate_answers")
        or obj.get("responses")
        or obj.get("probability_points")
    )
    extracted: list[tuple[str, float]] = []
    if isinstance(raw_answers, list):
        for item in raw_answers:
            if isinstance(item, dict):
                answer = item.get("answer") or item.get("response") or item.get("text")
                points_key = None
                points = None
                for key in ("points", "probability_points", "score", "probability"):
                    if key in item and item.get(key) is not None:
                        points_key = key
                        points = item.get(key)
                        break
            else:
                answer = item
                points_key = None
                points = 0.0
            if answer is None:
                continue
            try:
                point_value = float(points)
            except (TypeError, ValueError):
                point_value = 0.0
            if points_key == "probability" and 0.0 <= point_value <= 1.0:
                point_value *= PROBABILITY_POINT_TOTAL
            extracted.append((normalize_open_answer_text(str(answer)), max(0.0, point_value)))
    elif isinstance(raw_answers, dict):
        for answer, points in raw_answers.items():
            try:
                point_value = float(points)
            except (TypeError, ValueError):
                point_value = 0.0
            extracted.append((normalize_open_answer_text(str(answer)), max(0.0, point_value)))

    # Some APIs follow the requested schema but place probabilities in the
    # ``points`` field.  Recover that unambiguous intent when the raw total is
    # below the requested minimum and treating subunit values as probabilities
    # produces a valid point total.  This also handles mixed responses such as
    # 0.4, 0.3, 0.1, and 20.
    raw_total = sum(points for _, points in extracted)
    if raw_total + 1e-9 < PROBABILITY_POINT_TOTAL * top_p:
        probability_scaled = [
            (answer, points * PROBABILITY_POINT_TOTAL if 0.0 < points <= 1.0 else points)
            for answer, points in extracted
        ]
        scaled_total = sum(points for _, points in probability_scaled)
        if scaled_total + 1e-9 >= PROBABILITY_POINT_TOTAL * top_p:
            extracted = probability_scaled

    merged: dict[str, dict[str, Any]] = {}
    for answer, points in extracted:
        key = open_answer_key(answer)
        if not key:
            continue
        if key not in merged:
            merged[key] = {"answer": answer, "points": 0.0}
        merged[key]["points"] += points

    candidates = list(merged.values())
    if not candidates:
        return None, "empty"

    total_points = sum(float(candidate["points"]) for candidate in candidates)
    status = repair_status or "parser"
    if total_points <= 0:
        uniform = min(PROBABILITY_POINT_TOTAL, PROBABILITY_POINT_TOTAL * top_p) / len(candidates)
        for candidate in candidates:
            candidate["points"] = uniform
        total_points = sum(float(candidate["points"]) for candidate in candidates)
        status = "zero_total_uniform"
    elif total_points > PROBABILITY_POINT_TOTAL:
        for candidate in candidates:
            candidate["points"] = PROBABILITY_POINT_TOTAL * float(candidate["points"]) / total_points
        status = f"{status}_renormalized_{total_points:g}"
        total_points = PROBABILITY_POINT_TOTAL

    for candidate in candidates:
        candidate["probability"] = float(candidate["points"]) / PROBABILITY_POINT_TOTAL

    reported_mass = total_points / PROBABILITY_POINT_TOTAL
    if reported_mass + 1e-9 < top_p:
        status = f"{status}_below_top_p_{reported_mass:.3f}"
    return candidates, status


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def grade_simpleqa_answer_cached(
    *,
    example: SimpleQAExample,
    answer: str,
    grade_cache: dict[str, str],
    grader_client: Any | None,
    grader_model: str,
    use_grader: str,
    mock: bool,
    seed: int,
) -> str:
    key = open_answer_key(answer)
    if not key:
        return "not_attempted"
    if key not in grade_cache:
        if mock or use_grader == "never":
            grade_cache[key] = exact_simpleqa_grade(example, answer)
        else:
            if grader_client is None:
                raise RuntimeError("SimpleQA requires a grader unless --mock or --use-grader never is set.")
            grade_cache[key] = grade_simpleqa_response(
                grader_client,
                grader_model=grader_model,
                example=example,
                response=answer,
                seed=seed,
            )
    return grade_cache[key]


def simpleqa_answer_distribution_from_counts(
    counts: dict[str, int],
    display: dict[str, str],
    total: int,
) -> list[dict[str, Any]]:
    denom = max(1, total)
    rows = [
        {"answer": display[key], "count": count, "probability": count / denom}
        for key, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]
    return rows


def simpleqa_grade_counts_from_distribution(
    distribution: list[dict[str, Any]],
    *,
    example: SimpleQAExample,
    grade_cache: dict[str, str],
    grader_client: Any | None,
    grader_model: str,
    use_grader: str,
    mock: bool,
    seed: int,
) -> dict[str, int]:
    grade_counts = {"correct": 0, "incorrect": 0, "not_attempted": 0}
    for row in distribution:
        answer = str(row["answer"])
        grade = grade_simpleqa_answer_cached(
            example=example,
            answer=answer,
            grade_cache=grade_cache,
            grader_client=grader_client,
            grader_model=grader_model,
            use_grader=use_grader,
            mock=mock,
            seed=seed,
        )
        count = int(row["count"])
        grade_counts[grade] = grade_counts.get(grade, 0) + count
        row["grade"] = grade
    return grade_counts


def retry_seed_for_attempt(seed: int | None, attempt: int) -> int | None:
    if seed is None or attempt <= 0:
        return seed
    return (int(seed) + attempt * RETRY_SEED_STRIDE) % PROVIDER_SEED_MAX


def generate_with_empty_content_retries(
    client: Any,
    *,
    model: str,
    prompt: str,
    system_message: str | None,
    seed: int | None,
    temperature: float,
    max_tokens: int,
    empty_content_retries: int,
) -> tuple[str, int]:
    retry_count = max(0, empty_content_retries)
    last_error: EmptyAssistantContentError | None = None
    for attempt in range(retry_count + 1):
        try:
            return (
                client.generate(
                    model=model,
                    prompt=prompt,
                    system_message=system_message,
                    seed=retry_seed_for_attempt(seed, attempt),
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                attempt,
            )
        except EmptyAssistantContentError as exc:
            last_error = exc
            if attempt >= retry_count:
                break
    assert last_error is not None
    raise last_error


def simpleqa_log_only_row_from_source(
    row: dict[str, Any],
    *,
    empty_content_retries: int,
    gcp_reasoning_effort: str | None,
    log_system_message: str | None = None,
    imported_from: str | None = None,
) -> dict[str, Any]:
    model = str(row.get("model") or "")
    request_gcp_reasoning_effort = effective_gcp_reasoning_effort(
        model, gcp_reasoning_effort
    )
    output = {
        "question_id": row.get("question_id"),
        "category": row.get("category"),
        "answer_type": row.get("answer_type"),
        "model": row.get("model"),
        "seed": int(row.get("seed", 0) or 0),
        "n_samples_requested": 0,
        "simpleqa_log_mode": row.get("simpleqa_log_mode", "top-p"),
        "simpleqa_top_p": row.get("simpleqa_top_p"),
        "empty_content_retries": empty_content_retries,
        "gcp_reasoning_effort": request_gcp_reasoning_effort,
        "log_system_message": log_system_message or row.get("log_system_message"),
        "experiment_mode": "simpleqa_log_only",
        "imported_from": imported_from,
        "gold_answer": row.get("gold_answer"),
        "question": row.get("question"),
        "empirical_top_answer": None,
        "empirical_top_prob": None,
        "empirical_top_grade": None,
        "empirical_accuracy_overall": None,
        "empirical_hallucination_rate": None,
        "empirical_not_attempted_rate": None,
        "empirical_distribution_json": None,
        "empirical_grade_counts_json": None,
        "empirical_input_tokens": None,
        "empirical_output_tokens": None,
        "empirical_total_tokens": None,
        "empirical_token_source_counts_json": None,
        "empirical_generation_error_count": None,
        "empirical_empty_content_retry_count": None,
        "penalty_value": None,
        "penalty_threshold": None,
        "penalty_answered_samples": None,
        "penalty_abstain_samples": None,
        "penalty_correct_samples": None,
        "penalty_incorrect_samples": None,
        "penalty_not_attempted_samples": None,
        "penalty_accuracy_overall": None,
        "penalty_accuracy_when_answered": None,
        "penalty_hallucination_rate": None,
        "penalty_abstention_rate": None,
        "penalty_distribution_json": None,
        "penalty_grade_counts_json": None,
        "penalty_input_tokens": None,
        "penalty_output_tokens": None,
        "penalty_total_tokens": None,
        "penalty_token_source_counts_json": None,
        "penalty_generation_error_count": None,
        "penalty_empty_content_retry_count": None,
        "sample_response_preview_json": None,
        "penalty_sample_response_preview_json": None,
    }
    for key in [
        "log_top_answer",
        "log_top_prob",
        "log_top_grade",
        "log_q_true",
        "log_hallucination_mass",
        "log_not_attempted_mass",
        "log_reported_mass",
        "log_unreported_mass",
        "log_report_log_score",
        "log_report_shifted_log_score",
        "log_report_scaled_log_score",
        "log_candidates_json",
        "log_parse_status",
        "raw_log_response",
        "log_input_tokens",
        "log_output_tokens",
        "log_total_tokens",
        "log_token_source_counts_json",
        "log_generation_error",
        "log_empty_content_retry_count",
    ]:
        output[key] = row.get(key)
    return output


def run_simpleqa_log_only_experiment(
    *,
    examples: list[SimpleQAExample],
    models: list[str],
    answer_client: OpenRouterClient | None,
    grader_client: Any | None,
    grader_model: str,
    use_grader: str,
    mock: bool,
    seed: int,
    log_max_tokens: int,
    top_p: float,
    eps: float,
    out_dir: Path,
    verbose: bool,
    plot_examples: int,
    resume_existing_results: bool,
    gcp_reasoning_effort: str | None,
    empty_content_retries: int,
    import_result_paths: list[Path],
    strict_json_prompts: bool = False,
    max_workers: int = 1,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_mode = "top-p"
    output_stem = "simpleqa_topp"
    partial_path = out_dir / f"{output_stem}_results.partial.jsonl"
    results_path = out_dir / f"{output_stem}_results.jsonl"
    summary_path = out_dir / f"{output_stem}_summary.csv"
    total_jobs = len(examples) * len(models)
    log_setting = f"top_p={top_p:g}"
    example_ids = {example.question_id for example in examples}
    example_order = {example.question_id: i for i, example in enumerate(examples)}
    model_order = {model: i for i, model in enumerate(models)}
    model_set = set(models)

    def row_matches_scope(row: dict[str, Any]) -> bool:
        if str(row.get("question_id")) not in example_ids:
            return False
        if str(row.get("model")) not in model_set:
            return False
        if int(row.get("seed", seed) or seed) != seed:
            return False
        if str(row.get("simpleqa_log_mode", log_mode)) != log_mode:
            return False
        row_top_p = row.get("simpleqa_top_p")
        if row_top_p is None or abs(float(row_top_p) - top_p) > 1e-9:
            return False
        row_model = str(row.get("model"))
        expected_gcp_reasoning_effort = effective_gcp_reasoning_effort(
            row_model, gcp_reasoning_effort
        )
        if normalize_gcp_reasoning_effort(row.get("gcp_reasoning_effort")) != expected_gcp_reasoning_effort:
            return False
        return True

    def row_matches_log_only_run(row: dict[str, Any]) -> bool:
        if not row_matches_scope(row):
            return False
        row_model = str(row.get("model"))
        expected_log_system_message = simpleqa_log_system_message_for_model(
            row_model,
            strict_json=strict_json_prompts,
        )
        row_log_system_message = row.get("log_system_message") or SIMPLEQA_LOG_SYSTEM_MESSAGE
        if row_log_system_message != expected_log_system_message:
            return False
        if not row.get("log_candidates_json"):
            return False
        if str(row.get("log_parse_status", "")).startswith("malformed"):
            return False
        return not row.get("log_generation_error")

    def row_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
        return (
            model_order.get(str(row.get("model")), len(model_order)),
            example_order.get(str(row.get("question_id")), len(example_order)),
            str(row.get("question_id")),
        )

    chosen_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    resume_paths = [path for path in [partial_path, results_path] if resume_existing_results and path.exists()]
    source_paths: list[Path] = []
    if resume_existing_results:
        source_paths.extend(resume_paths)
    source_paths.extend(path for path in import_result_paths if path.exists())

    for source_path in source_paths:
        is_resume_path = source_path in resume_paths
        for row in read_jsonl(source_path):
            if is_resume_path:
                row_model = str(row.get("model") or (models[0] if models else ""))
                expected_log_system_message = simpleqa_log_system_message_for_model(
                    row_model,
                    strict_json=strict_json_prompts,
                )
                row_log_system_message = row.get("log_system_message") or SIMPLEQA_LOG_SYSTEM_MESSAGE
                if row_log_system_message != expected_log_system_message:
                    raise SystemExit(
                        f"Refusing to resume into {out_dir}: existing log rows in "
                        f"{source_path.name} were produced with a different log system "
                        "message. Use a separate --out-dir for JSON-only prompt runs "
                        "so existing results are preserved."
                    )
            if row_matches_scope(row) and not row_matches_log_only_run(row):
                row_model = str(row.get("model"))
                expected_log_system_message = simpleqa_log_system_message_for_model(
                    row_model,
                    strict_json=strict_json_prompts,
                )
                row_log_system_message = row.get("log_system_message") or SIMPLEQA_LOG_SYSTEM_MESSAGE
                if row_log_system_message != expected_log_system_message:
                    raise SystemExit(
                        f"Refusing to resume/import log rows from {source_path}: "
                        "the log system message differs from this run. Use a separate "
                        "--out-dir for JSON-only prompt runs so existing results are preserved."
                    )
            if not row_matches_log_only_run(row):
                continue
            key = (str(row["model"]), str(row["question_id"]), int(row.get("seed", seed)))
            chosen_rows[key] = simpleqa_log_only_row_from_source(
                row,
                empty_content_retries=empty_content_retries,
                gcp_reasoning_effort=gcp_reasoning_effort,
                log_system_message=simpleqa_log_system_message_for_model(
                    str(row.get("model")),
                    strict_json=strict_json_prompts,
                ),
                imported_from=str(source_path),
            )

    rows = sorted(chosen_rows.values(), key=row_sort_key)
    completed_keys = set(chosen_rows)
    if source_paths:
        print(
            f"[simpleqa log-only resume] Loaded {len(rows)} clean log rows from "
            f"{len(source_paths)} source file(s); remaining jobs={total_jobs - len(completed_keys)}",
            flush=True,
        )

    def run_job(job: tuple[int, str, SimpleQAExample]) -> tuple[tuple[str, str, int], dict[str, Any]]:
        job_i, model, example = job
        key = (model, example.question_id, seed)
        if verbose:
            print(
                f"[simpleqa log-only] job={job_i}/{total_jobs} model={model} "
                f"question={example.question_id} {log_setting}",
                flush=True,
            )
        grade_cache: dict[str, str] = {}
        log_prompt = build_simpleqa_topp_log_prompt(example, top_p)
        log_system_message = simpleqa_log_system_message_for_model(
            model,
            strict_json=strict_json_prompts,
        )
        log_generation_error = None
        log_empty_content_retry_count = 0
        if mock:
            log_candidates = [
                {"answer": example.answer, "points": 80.0, "probability": 0.8},
                {"answer": UNKNOWN_LABEL, "points": 20.0, "probability": 0.2},
            ]
            log_response = json.dumps({"answers": log_candidates}, ensure_ascii=True)
            log_status = "mock"
        else:
            assert answer_client is not None
            try:
                log_response, log_empty_content_retry_count = generate_with_empty_content_retries(
                    answer_client,
                    model=model,
                    prompt=log_prompt,
                    system_message=log_system_message,
                    seed=seed,
                    temperature=0,
                    max_tokens=log_max_tokens,
                    empty_content_retries=empty_content_retries,
                )
            except EmptyAssistantContentError as exc:
                log_generation_error = str(exc)
                log_empty_content_retry_count = max(0, empty_content_retries)
                log_response = json.dumps(
                    {"answers": [{"answer": UNKNOWN_LABEL, "points": PROBABILITY_POINT_TOTAL}]},
                    ensure_ascii=True,
                )
                if verbose:
                    print(
                        f"[simpleqa log-only warning] model={model} "
                        f"question={example.question_id} empty_content={exc}; "
                        f"retries={empty_content_retries}; using {UNKNOWN_LABEL} distribution",
                        flush=True,
                    )
            parsed_candidates, log_status = parse_simpleqa_topp_response(
                log_response,
                top_p=top_p,
            )
            log_candidates = parsed_candidates or [
                {"answer": UNKNOWN_LABEL, "points": PROBABILITY_POINT_TOTAL, "probability": 1.0}
            ]
            if parsed_candidates is None:
                log_status = f"malformed_{log_status}"
            if log_generation_error is not None:
                log_status = f"generation_empty_content_{log_status}"

        for candidate_i, candidate in enumerate(log_candidates):
            candidate["grade"] = grade_simpleqa_answer_cached(
                example=example,
                answer=str(candidate["answer"]),
                grade_cache=grade_cache,
                grader_client=grader_client,
                grader_model=grader_model,
                use_grader=use_grader,
                mock=mock,
                seed=seed * 1_000 + candidate_i,
            )
        log_candidates = canonicalize_simpleqa_distribution(
            log_candidates,
            example=example,
        )
        log_top = max(log_candidates, key=lambda item: float(item.get("probability", 0.0)))
        log_q_true = sum(
            float(candidate.get("probability", 0.0))
            for candidate in log_candidates
            if candidate.get("grade") == "correct"
        )
        log_reported_mass = sum(float(candidate.get("probability", 0.0)) for candidate in log_candidates)
        log_not_attempted_mass = sum(
            float(candidate.get("probability", 0.0))
            for candidate in log_candidates
            if candidate.get("grade") == "not_attempted"
        )
        log_incorrect_mass = sum(
            float(candidate.get("probability", 0.0))
            for candidate in log_candidates
            if candidate.get("grade") == "incorrect"
        )
        log_unreported_mass = max(0.0, 1.0 - log_reported_mass)
        log_token_usage = generation_token_usage(
            client=None if mock else answer_client,
            prompt=log_prompt,
            system_message=log_system_message,
            response=log_response,
        )
        row = simpleqa_log_only_row_from_source(
            {
                "question_id": example.question_id,
                "category": example.category,
                "answer_type": example.answer_type,
                "model": model,
                "seed": seed,
                "simpleqa_log_mode": log_mode,
                "simpleqa_top_p": top_p,
                "log_system_message": log_system_message,
                "gold_answer": example.answer,
                "question": example.question,
                "log_top_answer": log_top.get("answer"),
                "log_top_prob": log_top.get("probability"),
                "log_top_grade": log_top.get("grade"),
                "log_q_true": log_q_true,
                "log_hallucination_mass": log_incorrect_mass,
                "log_not_attempted_mass": log_not_attempted_mass,
                "log_reported_mass": log_reported_mass,
                "log_unreported_mass": log_unreported_mass,
                "log_report_log_score": math.log(max(eps, log_q_true)),
                "log_report_shifted_log_score": shifted_log_score(log_q_true, eps),
                "log_report_scaled_log_score": scaled_log_score(log_q_true, eps),
                "log_candidates_json": json.dumps(log_candidates, ensure_ascii=True),
                "log_parse_status": log_status,
                "raw_log_response": log_response,
                "log_input_tokens": log_token_usage["input_tokens"],
                "log_output_tokens": log_token_usage["output_tokens"],
                "log_total_tokens": log_token_usage["total_tokens"],
                "log_token_source_counts_json": json.dumps(
                    {str(log_token_usage.get("source") or "unknown"): 1},
                    sort_keys=True,
                ),
                "log_generation_error": log_generation_error,
                "log_empty_content_retry_count": log_empty_content_retry_count,
            },
            empty_content_retries=empty_content_retries,
            gcp_reasoning_effort=gcp_reasoning_effort,
            log_system_message=log_system_message,
        )
        return key, row

    pending_jobs: list[tuple[int, str, SimpleQAExample]] = []
    job_i = 0
    for model in models:
        for example in examples:
            job_i += 1
            key = (model, example.question_id, seed)
            if key in completed_keys:
                if verbose:
                    print(
                        f"[simpleqa log-only resume] skip job={job_i}/{total_jobs} "
                        f"model={model} question={example.question_id}",
                        flush=True,
                    )
                continue
            pending_jobs.append((job_i, model, example))

    max_workers = max(1, int(max_workers))
    checkpoint_every = 1 if max_workers == 1 else max_workers
    completed_since_checkpoint = 0

    def accept_result(result: tuple[tuple[str, str, int], dict[str, Any]]) -> None:
        nonlocal completed_since_checkpoint
        key, row = result
        rows.append(row)
        completed_keys.add(key)
        completed_since_checkpoint += 1
        if completed_since_checkpoint >= checkpoint_every:
            rows.sort(key=row_sort_key)
            write_jsonl(partial_path, rows)
            completed_since_checkpoint = 0

    if max_workers == 1:
        for job in pending_jobs:
            accept_result(run_job(job))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(run_job, job) for job in pending_jobs]
            for future in as_completed(futures):
                accept_result(future.result())

    if completed_since_checkpoint:
        rows.sort(key=row_sort_key)
        write_jsonl(partial_path, rows)

    rows.sort(key=row_sort_key)
    write_jsonl(results_path, rows)
    write_csv(summary_path, summarize_simpleqa_rows(rows, include_quadratic=False))
    make_simpleqa_topk_plots(
        rows,
        out_dir,
        max_examples=plot_examples,
        include_quadratic=False,
    )
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote SimpleQA log-only {log_mode} plots to {out_dir}")
    return rows


def summarize_simpleqa_rows(
    rows: list[dict[str, Any]],
    *,
    include_quadratic: bool = True,
) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    summary: list[dict[str, Any]] = []

    def numeric_values(model_rows: list[dict[str, Any]], key: str) -> list[float]:
        values: list[float] = []
        for row in model_rows:
            value = row.get(key)
            if value in {None, ""}:
                continue
            values.append(float(value))
        return values

    for model, model_rows in sorted(by_model.items()):
        penalty_values = [row.get("penalty_value") for row in model_rows if row.get("penalty_value") is not None]
        penalty_present = bool(penalty_values)
        quadratic_present = include_quadratic and any(row.get("quadratic_candidates_json") for row in model_rows)
        summary.append(
            {
                "model": model,
                "n": len(model_rows),
                "mean_empirical_accuracy_overall": safe_mean(
                    numeric_values(model_rows, "empirical_accuracy_overall")
                ),
                "mean_empirical_hallucination_rate": safe_mean(
                    numeric_values(model_rows, "empirical_hallucination_rate")
                ),
                "mean_empirical_not_attempted_rate": safe_mean(
                    numeric_values(model_rows, "empirical_not_attempted_rate")
                ),
                "mean_log_q_true": safe_mean(numeric_values(model_rows, "log_q_true")),
                "mean_log_hallucination_mass": safe_mean(
                    numeric_values(model_rows, "log_hallucination_mass")
                ),
                "mean_log_not_attempted_mass": safe_mean(
                    numeric_values(model_rows, "log_not_attempted_mass")
                ),
                "mean_log_reported_mass": safe_mean(
                    [
                        float(row.get("log_reported_mass", 1.0))
                        for row in model_rows
                        if row.get("log_reported_mass", 1.0) is not None
                    ]
                ),
                "mean_log_unreported_mass": safe_mean(
                    [
                        float(row.get("log_unreported_mass", 0.0))
                        for row in model_rows
                        if row.get("log_unreported_mass", 0.0) is not None
                    ]
                ),
                "mean_log_shifted_log_score": safe_mean(
                    numeric_values(model_rows, "log_report_shifted_log_score")
                ),
                "mean_quadratic_q_true": (
                    safe_mean(numeric_values(model_rows, "quadratic_q_true"))
                    if quadratic_present
                    else None
                ),
                "mean_quadratic_hallucination_mass": (
                    safe_mean(numeric_values(model_rows, "quadratic_hallucination_mass"))
                    if quadratic_present
                    else None
                ),
                "mean_quadratic_not_attempted_mass": (
                    safe_mean(numeric_values(model_rows, "quadratic_not_attempted_mass"))
                    if quadratic_present
                    else None
                ),
                "mean_quadratic_reported_mass": (
                    safe_mean(
                        [
                            float(row.get("quadratic_reported_mass", 1.0))
                            for row in model_rows
                            if row.get("quadratic_reported_mass", 1.0) is not None
                        ]
                    )
                    if quadratic_present
                    else None
                ),
                "mean_quadratic_unreported_mass": (
                    safe_mean(
                        [
                            float(row.get("quadratic_unreported_mass", 0.0))
                            for row in model_rows
                            if row.get("quadratic_unreported_mass", 0.0) is not None
                        ]
                    )
                    if quadratic_present
                    else None
                ),
                "mean_quadratic_score": (
                    safe_mean(numeric_values(model_rows, "quadratic_report_score"))
                    if quadratic_present
                    else None
                ),
                "penalty": safe_mean([float(value) for value in penalty_values]) if penalty_present else None,
                "mean_penalty_accuracy_overall": (
                    safe_mean(numeric_values(model_rows, "penalty_accuracy_overall"))
                    if penalty_present
                    else None
                ),
                "mean_penalty_accuracy_when_answered": (
                    safe_mean(numeric_values(model_rows, "penalty_accuracy_when_answered"))
                    if penalty_present
                    else None
                ),
                "mean_penalty_hallucination_rate": (
                    safe_mean(numeric_values(model_rows, "penalty_hallucination_rate"))
                    if penalty_present
                    else None
                ),
                "mean_penalty_abstention_rate": (
                    safe_mean(numeric_values(model_rows, "penalty_abstention_rate"))
                    if penalty_present
                    else None
                ),
            }
        )
    if not include_quadratic:
        for row in summary:
            for key in list(row):
                if "quadratic" in key:
                    row.pop(key, None)
    return summary


def make_simpleqa_topk_plots(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    max_examples: int = 8,
    include_quadratic: bool = True,
    distribution_max_questions: int | None = None,
    only_filenames: set[str] | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
    except ImportError:
        print("matplotlib is not installed; skipping SimpleQA plots.", file=sys.stderr)
        return
    if not rows:
        return
    only_filenames = set(only_filenames or [])

    def want(filename: str) -> bool:
        return not only_filenames or filename in only_filenames

    log_modes = {str(row.get("simpleqa_log_mode", "top-k")) for row in rows}
    log_method_label = "Log Top-P" if log_modes == {"top-p"} else "Log Top-K"
    quadratic_method_label = "Quadratic Top-P" if log_modes == {"top-p"} else "Quadratic Top-K"
    log_report_label = f"{log_method_label} report"
    quadratic_report_label = f"{quadratic_method_label} report"
    log_examples_filename = (
        "simpleqa_log_topp_examples.png" if log_modes == {"top-p"} else "simpleqa_log_topk_examples.png"
    )

    summary = summarize_simpleqa_rows(rows, include_quadratic=include_quadratic)
    metric_rows: list[dict[str, Any]] = []
    for row in summary:
        if row.get("mean_empirical_accuracy_overall") is not None:
            metric_rows.append(
                {
                    "model": row["model"],
                    "method": "Empirical",
                    "accuracy": row["mean_empirical_accuracy_overall"],
                    "hallucination": row["mean_empirical_hallucination_rate"],
                    "abstain": row["mean_empirical_not_attempted_rate"],
                }
            )
        if row.get("mean_log_q_true") is not None:
            metric_rows.append(
                {
                    "model": row["model"],
                    "method": log_method_label,
                    "accuracy": row["mean_log_q_true"],
                    "hallucination": row["mean_log_hallucination_mass"],
                    "abstain": float(row["mean_log_not_attempted_mass"] or 0.0)
                    + float(row.get("mean_log_unreported_mass") or 0.0),
                }
            )
        if include_quadratic and row.get("mean_quadratic_q_true") is not None:
            metric_rows.append(
                {
                    "model": row["model"],
                    "method": quadratic_method_label,
                    "accuracy": row["mean_quadratic_q_true"],
                    "hallucination": row["mean_quadratic_hallucination_mass"],
                    "abstain": float(row["mean_quadratic_not_attempted_mass"] or 0.0)
                    + float(row.get("mean_quadratic_unreported_mass") or 0.0),
                }
            )
        if row.get("mean_penalty_accuracy_overall") is not None:
            metric_rows.append(
                {
                    "model": row["model"],
                    "method": "Penalty",
                    "accuracy": row["mean_penalty_accuracy_overall"],
                    "hallucination": row["mean_penalty_hallucination_rate"],
                    "abstain": row["mean_penalty_abstention_rate"],
                }
            )

    methods = [entry["method"] for entry in metric_rows]
    x = list(range(len(metric_rows)))
    width = 0.25
    abstain_label = "I don't know / unreported mass" if log_modes == {"top-p"} else "Abstain / not attempted"
    if metric_rows and want("simpleqa_method_comparison.png"):
        fig, ax = plt.subplots(figsize=(max(8.5, 0.7 * len(metric_rows)), 4.6))
        ax.bar([i - width for i in x], [float(entry["accuracy"] or 0.0) for entry in metric_rows], width, label="Accuracy / q_true", color="#0072B2")
        ax.bar(x, [float(entry["hallucination"] or 0.0) for entry in metric_rows], width, label="Hallucination / incorrect mass", color="#D55E00")
        ax.bar([i + width for i in x], [float(entry["abstain"] or 0.0) for entry in metric_rows], width, label=abstain_label, color="#999999")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{entry['model']}\n{method}" for entry, method in zip(metric_rows, methods)], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Rate or probability mass")
        ax.set_title("SimpleQA open-answer comparison")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "simpleqa_method_comparison.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    if want(log_examples_filename):
        sorted_rows = sorted(rows, key=lambda row: float(row["log_q_true"]))
        selected_rows = sorted_rows[: max(1, min(max_examples, len(sorted_rows)))]
        fig, axes = plt.subplots(len(selected_rows), 1, figsize=(10.8, max(3.6, 2.9 * len(selected_rows))))
        if len(selected_rows) == 1:
            axes = [axes]
        colors = {"correct": "#009E73", "incorrect": "#D55E00", "not_attempted": "#999999", "mixed": "#56B4E9"}
        for ax, row in zip(axes, selected_rows):
            candidates = json.loads(str(row["log_candidates_json"]))
            labels = [
                textwrap.shorten(str(candidate["answer"]), width=34, placeholder="...")
                for candidate in candidates
            ]
            probs = [float(candidate.get("probability", 0.0)) for candidate in candidates]
            grades = [str(candidate.get("grade", "incorrect")) for candidate in candidates]
            ax.barh(list(range(len(candidates))), probs, color=[colors.get(grade, "#999999") for grade in grades])
            ax.set_yticks(list(range(len(candidates))))
            ax.set_yticklabels(labels, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.set_xlabel("Log-reported probability")
            title = (
                f"{row['model']} | q={row['question_id']} | gold: "
                f"{textwrap.shorten(str(row['gold_answer']), width=52, placeholder='...')}"
            )
            ax.set_title(title, fontsize=9)
            ax.grid(axis="x", alpha=0.25)
            question = textwrap.fill(str(row["question"]), width=115)
            raw = textwrap.shorten(str(row["raw_log_response"]).replace("\n", " "), width=260, placeholder="...")
            ax.text(
                0,
                -0.34,
                f"Question: {question}\nRaw log response: {raw}",
                transform=ax.transAxes,
                fontsize=6.5,
                va="top",
            )
        fig.tight_layout(h_pad=2.2)
        fig.savefig(out_dir / log_examples_filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    def response_items_for_method(row: dict[str, Any], method: str) -> list[dict[str, Any]]:
        example = SimpleQAExample(
            question_id=str(row["question_id"]),
            question=str(row["question"]),
            answer=str(row["gold_answer"]),
            category=str(row.get("category", "unknown")),
            answer_type=str(row.get("answer_type", "")),
        )
        if method == "empirical":
            raw_empirical = row.get("empirical_distribution_json")
            if not raw_empirical:
                return []
            items = list(json.loads(str(raw_empirical)))
        elif method == "log":
            raw_log = row.get("log_candidates_json")
            if not raw_log:
                return []
            items = list(json.loads(str(raw_log)))
        elif method == "quadratic":
            raw_quadratic = row.get("quadratic_candidates_json")
            if not raw_quadratic:
                return []
            items = list(json.loads(str(raw_quadratic)))
        elif method == "penalty":
            raw_penalty = row.get("penalty_distribution_json")
            if not raw_penalty:
                return []
            items = list(json.loads(str(raw_penalty)))
            abstain_samples = row.get("penalty_abstain_samples")
            n_samples = max(1, int(row.get("n_samples_requested") or 1))
            if abstain_samples is not None:
                abstain_prob = float(abstain_samples) / n_samples
                if abstain_prob > 0:
                    items.append(
                        {
                            "answer": ABSTAIN_LABEL,
                            "probability": abstain_prob,
                            "grade": "not_attempted",
                        }
                    )
        else:
            raise ValueError(f"Unknown SimpleQA method: {method}")
        canonical_items = canonicalize_simpleqa_distribution(items, example=example)
        if method == "penalty":
            for item in canonical_items:
                if item.get("grade") == "not_attempted":
                    item["answer"] = ABSTAIN_LABEL
        return canonical_items

    def should_show_all_items(row: dict[str, Any], method: str) -> bool:
        return method in {"log", "quadratic"} and str(row.get("simpleqa_log_mode", "top-k")) == "top-p"

    def trimmed_items(items: list[dict[str, Any]], max_items: int = 7) -> list[dict[str, Any]]:
        ordered = sorted(items, key=lambda item: float(item.get("probability", 0.0)), reverse=True)
        if len(ordered) <= max_items:
            return ordered
        shown = ordered[: max_items - 1]
        other_prob = sum(float(item.get("probability", 0.0)) for item in ordered[max_items - 1 :])
        shown.append({"answer": "Other responses", "probability": other_prob, "grade": "mixed"})
        return shown

    def top_probability_mass_prefix(
        items: list[dict[str, Any]],
        *,
        top_p: float,
    ) -> list[dict[str, Any]] | None:
        ordered = sorted(
            [
                item
                for item in items
                if float(item.get("probability", 0.0) or 0.0) > 0
            ],
            key=lambda item: float(item.get("probability", 0.0)),
            reverse=True,
        )
        if not ordered:
            return None
        target_mass = max(0.0, min(1.0, top_p))
        cumulative = 0.0
        prefix = []
        for item in ordered:
            prefix.append(item)
            cumulative += max(0.0, float(item.get("probability", 0.0) or 0.0))
            if cumulative + 1e-12 >= target_mass:
                break
        return prefix

    def correct_in_top_probability_mass(
        items: list[dict[str, Any]],
        *,
        top_p: float,
    ) -> bool | None:
        prefix = top_probability_mass_prefix(items, top_p=top_p)
        if prefix is None:
            return None
        return any(str(item.get("grade", "incorrect")) == "correct" for item in prefix)

    def correct_or_idk_when_correct_missing_in_top_probability_mass(
        items: list[dict[str, Any]],
        *,
        top_p: float,
    ) -> bool | None:
        prefix = top_probability_mass_prefix(items, top_p=top_p)
        if prefix is None:
            return None
        if any(str(item.get("grade", "incorrect")) == "correct" for item in prefix):
            return True
        positive_items = [
            item
            for item in items
            if float(item.get("probability", 0.0) or 0.0) > 0
        ]
        if any(str(item.get("grade", "incorrect")) == "correct" for item in positive_items):
            return False
        return any(
            str(item.get("grade", "incorrect")) == "not_attempted"
            or is_open_not_attempted(str(item.get("answer", "")))
            for item in prefix
        )

    def item_is_not_attempted(item: dict[str, Any]) -> bool:
        return (
            str(item.get("grade", "incorrect")) == "not_attempted"
            or is_open_not_attempted(str(item.get("answer", "")))
        )

    def correct_or_idk_in_top_probability_mass(
        items: list[dict[str, Any]],
        *,
        top_p: float,
    ) -> bool | None:
        prefix = top_probability_mass_prefix(items, top_p=top_p)
        if prefix is None:
            return None
        return any(
            str(item.get("grade", "incorrect")) == "correct" or item_is_not_attempted(item)
            for item in prefix
        )

    def idk_only_in_top_probability_mass(
        items: list[dict[str, Any]],
        *,
        top_p: float,
    ) -> bool | None:
        prefix = top_probability_mass_prefix(items, top_p=top_p)
        if prefix is None:
            return None
        correct_present = any(str(item.get("grade", "incorrect")) == "correct" for item in prefix)
        idk_present = any(item_is_not_attempted(item) for item in prefix)
        return idk_present and not correct_present

    def top_answer_is_correct(items: list[dict[str, Any]], *, top_p: float) -> bool | None:
        del top_p
        positive_items = [
            item
            for item in items
            if float(item.get("probability", 0.0) or 0.0) > 0
        ]
        if not positive_items:
            return None
        top_item = max(positive_items, key=lambda item: float(item.get("probability", 0.0) or 0.0))
        return str(top_item.get("grade", "incorrect")) == "correct"

    method_specs = []
    if any(row.get("empirical_distribution_json") for row in rows):
        method_specs.append(("Empirical samples", "empirical"))
    if any(row.get("log_candidates_json") for row in rows):
        method_specs.append((log_report_label, "log"))
    if any(row.get("penalty_distribution_json") for row in rows):
        method_specs.append(("Penalty samples", "penalty"))
    if include_quadratic and any(row.get("quadratic_candidates_json") for row in rows):
        insert_at = 2 if len(method_specs) >= 2 else len(method_specs)
        method_specs.insert(insert_at, (quadratic_report_label, "quadratic"))
    if not method_specs:
        print("No plottable SimpleQA method distributions found; skipping SimpleQA plots.", file=sys.stderr)
        return
    color_map = {
        "correct": "#009E73",
        "incorrect": "#D55E00",
        "not_attempted": "#999999",
        "mixed": "#56B4E9",
    }

    top_p_values = [
        float(row["simpleqa_top_p"])
        for row in rows
        if row.get("simpleqa_top_p") not in {None, ""}
    ]
    rounded_top_p_values = {round(value, 6) for value in top_p_values}
    coverage_top_p = top_p_values[0] if len(rounded_top_p_values) == 1 else DEFAULT_SIMPLEQA_TOP_P
    models = sorted({str(row["model"]) for row in rows})
    method_colors = {
        "Empirical samples": "#0072B2",
        log_report_label: "#388210",
        quadratic_report_label: "#F0B042",
        "Penalty samples": "#E0435E",
    }

    def make_method_metric_bar_plot(
        *,
        metric_fn: Any,
        title: str,
        ylabel: str,
        filename: str,
        y_upper: float | None = None,
    ) -> None:
        metric_by_method_model: dict[tuple[str, str], tuple[float, int]] = {}
        for method_title, method_key in method_specs:
            for model in models:
                values: list[float] = []
                for row in rows:
                    if str(row["model"]) != model:
                        continue
                    items = response_items_for_method(row, method_key)
                    value = metric_fn(items, top_p=coverage_top_p)
                    if value is not None:
                        values.append(float(value))
                mean_value = safe_mean(values)
                if mean_value is not None:
                    metric_by_method_model[(method_title, model)] = (mean_value, len(values))
        if not metric_by_method_model:
            return

        visible_methods = [
            method_title
            for method_title, _method_key in method_specs
            if any((method_title, model) in metric_by_method_model for model in models)
        ]
        bar_width = min(0.24, 0.78 / max(1, len(visible_methods)))
        x_positions = list(range(len(models)))
        fig, ax = plt.subplots(figsize=(max(7.5, 1.45 * len(models)), 4.5))
        for method_i, method_title in enumerate(visible_methods):
            offset = (method_i - (len(visible_methods) - 1) / 2) * bar_width
            xs = []
            ys = []
            ns = []
            for model_i, model in enumerate(models):
                entry = metric_by_method_model.get((method_title, model))
                if entry is None:
                    continue
                mean_value, n_questions = entry
                xs.append(model_i + offset)
                ys.append(mean_value)
                ns.append(n_questions)
            bars = ax.bar(
                xs,
                ys,
                width=bar_width,
                label=method_title,
                color=method_colors.get(method_title, "#56B4E9"),
            )
            for bar, value, n_questions in zip(bars, ys, ns):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(1.02, value + 0.025),
                    f"{value:.2f}\nn={n_questions}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [textwrap.shorten(model, width=34, placeholder="...") for model in models],
            rotation=20,
            ha="right",
            fontsize=8,
        )
        if y_upper is None:
            all_values = [entry[0] for entry in metric_by_method_model.values()]
            y_upper = max(1.0, max(all_values or [0.0]) * 1.18)
        ax.set_ylim(0, y_upper)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(visible_methods))
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    if want("simpleqa_top_p_coverage_accuracy.png"):
        make_method_metric_bar_plot(
            metric_fn=lambda items, top_p: (
                None
                if correct_in_top_probability_mass(items, top_p=top_p) is None
                else float(bool(correct_in_top_probability_mass(items, top_p=top_p)))
            ),
            title=f"Correct answer contained in top {coverage_top_p:.0%} response mass",
            ylabel="Share of questions",
            filename="simpleqa_top_p_coverage_accuracy.png",
            y_upper=1.08,
        )
    if want("simpleqa_top_answer_accuracy.png"):
        make_method_metric_bar_plot(
            metric_fn=lambda items, top_p: (
                None
                if top_answer_is_correct(items, top_p=top_p) is None
                else float(bool(top_answer_is_correct(items, top_p=top_p)))
            ),
            title="Top-response accuracy",
            ylabel="Share of questions",
            filename="simpleqa_top_answer_accuracy.png",
            y_upper=1.08,
        )
    if want("simpleqa_top_p_coverage_or_idk_accuracy.png"):
        make_method_metric_bar_plot(
            metric_fn=lambda items, top_p: (
                None if correct_or_idk_in_top_probability_mass(items, top_p=top_p) is None
                else float(bool(correct_or_idk_in_top_probability_mass(items, top_p=top_p)))
            ),
            title=(
                f"Correct answer or IDK contained in top {coverage_top_p:.0%} response mass"
            ),
            ylabel="Share of questions",
            filename="simpleqa_top_p_coverage_or_idk_accuracy.png",
            y_upper=1.08,
        )
    if want("simpleqa_top_p_idk_only_rate.png"):
        make_method_metric_bar_plot(
            metric_fn=lambda items, top_p: (
                None if idk_only_in_top_probability_mass(items, top_p=top_p) is None
                else float(bool(idk_only_in_top_probability_mass(items, top_p=top_p)))
            ),
            title=f"IDK-only outcomes in top {coverage_top_p:.0%} response mass",
            ylabel="Share of questions",
            filename="simpleqa_top_p_idk_only_rate.png",
            y_upper=1.08,
        )

    def make_confidence_set_diagnostic_bar_plot() -> None:
        metrics = [
            (
                "Strict coverage",
                lambda items, top_p: correct_in_top_probability_mass(items, top_p=top_p),
                "#009E73",
            ),
            (
                "Correct or IDK",
                lambda items, top_p: correct_or_idk_in_top_probability_mass(items, top_p=top_p),
                "#56B4E9",
            ),
            (
                "IDK only",
                lambda items, top_p: idk_only_in_top_probability_mass(items, top_p=top_p),
                "#999999",
            ),
        ]
        label_entries: list[tuple[str, str]] = []
        for model in models:
            for method_title, method_key in method_specs:
                if any(
                    str(row["model"]) == model and response_items_for_method(row, method_key)
                    for row in rows
                ):
                    label_entries.append((model, method_title))
        if not label_entries:
            return
        x_positions = list(range(len(label_entries)))
        bar_width = min(0.22, 0.78 / len(metrics))
        fig, ax = plt.subplots(figsize=(max(8.5, 0.62 * len(label_entries)), 4.8))
        for metric_i, (metric_label, metric_fn, color) in enumerate(metrics):
            offset = (metric_i - (len(metrics) - 1) / 2) * bar_width
            values = []
            counts = []
            for model, method_title in label_entries:
                method_key = dict(method_specs)[method_title]
                method_values = []
                for row in rows:
                    if str(row["model"]) != model:
                        continue
                    value = metric_fn(response_items_for_method(row, method_key), coverage_top_p)
                    if value is not None:
                        method_values.append(1.0 if value else 0.0)
                values.append(safe_mean(method_values) or 0.0)
                counts.append(len(method_values))
            bars = ax.bar(
                [x + offset for x in x_positions],
                values,
                width=bar_width,
                color=color,
                label=metric_label,
            )
            for bar, value, count in zip(bars, values, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(1.03, value + 0.025),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [
                f"{textwrap.shorten(model, width=28, placeholder='...')}\n{method_title}"
                for model, method_title in label_entries
            ],
            rotation=30,
            ha="right",
            fontsize=7,
        )
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Share of questions")
        ax.set_title(f"Top {coverage_top_p:.0%} confidence-set diagnostics")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
        fig.tight_layout()
        fig.savefig(out_dir / "simpleqa_top_p_confidence_set_diagnostics.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    if want("simpleqa_top_p_confidence_set_diagnostics.png"):
        make_confidence_set_diagnostic_bar_plot()

    def make_cumulative_question_metric_plot(
        *,
        metric_fn: Any,
        ylabel: str,
        title: str,
        filename: str,
        show_nominal_line: bool,
    ) -> None:
        if not models:
            return
        fig, axes = plt.subplots(
            len(models),
            1,
            figsize=(8.4, max(4.6, 3.7 * len(models))),
            squeeze=False,
            sharey=True,
        )
        plotted_any = False
        for ax, model in zip(axes[:, 0], models):
            model_rows = [row for row in rows if str(row["model"]) == model]
            max_n = 0
            for method_title, method_key in method_specs:
                outcomes = []
                for row in model_rows:
                    hit = metric_fn(
                        response_items_for_method(row, method_key),
                        top_p=coverage_top_p,
                    )
                    if hit is not None:
                        outcomes.append(1.0 if hit else 0.0)
                if not outcomes:
                    continue
                plotted_any = True
                max_n = max(max_n, len(outcomes))
                running_total = 0.0
                cumulative_rates = []
                for question_i, outcome in enumerate(outcomes, start=1):
                    running_total += outcome
                    cumulative_rates.append(running_total / question_i)
                xs = list(range(1, len(cumulative_rates) + 1))
                markevery = max(1, len(xs) // 12)
                ax.plot(
                    xs,
                    cumulative_rates,
                    marker="o",
                    markevery=markevery,
                    markersize=3.2,
                    linewidth=1.8,
                    color=method_colors.get(method_title, "#56B4E9"),
                    label=f"{method_title} ({cumulative_rates[-1]:.2f})",
                )
            if show_nominal_line:
                ax.axhline(
                    coverage_top_p,
                    color="#333333",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.65,
                    label=f"Nominal p={coverage_top_p:.0%}",
                )
            ax.set_title(str(model))
            ax.set_xlim(1, max(1, max_n))
            ax.set_ylim(-0.03, 1.05)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Questions")
            ax.grid(True, alpha=0.25)
        if not plotted_any:
            plt.close(fig)
            return
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            unique: dict[str, Any] = {}
            for handle, label in zip(handles, labels):
                unique.setdefault(label, handle)
            fig.legend(
                list(unique.values()),
                list(unique.keys()),
                loc="lower center",
                ncol=min(3, len(unique)),
                frameon=False,
            )
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0.12, 1, 0.94))
        fig.savefig(out_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    if want("simpleqa_top_p_coverage_cumulative_questions.png"):
        make_cumulative_question_metric_plot(
            metric_fn=correct_in_top_probability_mass,
            ylabel="Strict coverage rate",
            title=f"Cumulative top {coverage_top_p:.0%} strict coverage by sample size",
            filename="simpleqa_top_p_coverage_cumulative_questions.png",
            show_nominal_line=True,
        )
    if want("simpleqa_top_p_coverage_or_idk_cumulative_questions.png"):
        make_cumulative_question_metric_plot(
            metric_fn=correct_or_idk_in_top_probability_mass,
            ylabel="Correct-or-IDK rate",
            title=f"Cumulative top {coverage_top_p:.0%} correct-or-IDK success by sample size",
            filename="simpleqa_top_p_coverage_or_idk_cumulative_questions.png",
            show_nominal_line=True,
        )
    if want("simpleqa_top_p_idk_only_cumulative_questions.png"):
        make_cumulative_question_metric_plot(
            metric_fn=idk_only_in_top_probability_mass,
            ylabel="IDK-only rate",
            title=f"Cumulative top {coverage_top_p:.0%} IDK-only rate by sample size",
            filename="simpleqa_top_p_idk_only_cumulative_questions.png",
            show_nominal_line=False,
        )

    def concrete_answer_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            [
                item
                for item in items
                if float(item.get("probability", 0.0) or 0.0) > 0
                and str(item.get("grade", "incorrect")) != "not_attempted"
                and not is_open_not_attempted(str(item.get("answer", "")))
            ],
            key=lambda item: float(item.get("probability", 0.0)),
            reverse=True,
        )

    def concrete_confidence_set_outcome(
        items: list[dict[str, Any]],
        *,
        top_p: float,
    ) -> str | None:
        concrete_items = concrete_answer_items(items)
        if not concrete_items:
            return "no_concrete_set"
        cumulative = 0.0
        prefix = []
        for item in concrete_items:
            prefix.append(item)
            cumulative += max(0.0, float(item.get("probability", 0.0) or 0.0))
            if cumulative + 1e-12 >= top_p:
                break
        covered = any(str(item.get("grade", "incorrect")) == "correct" for item in prefix)
        valid_p_set = cumulative + 1e-12 >= top_p
        if covered and valid_p_set:
            return "covered_valid"
        if covered:
            return "covered_underfilled"
        if valid_p_set:
            return "missed_valid"
        return "underfilled"

    def make_concrete_confidence_set_outcome_plot(
        *,
        selected_rows: list[dict[str, Any]],
        title: str,
        filename: str,
    ) -> None:
        if not selected_rows:
            return
        outcome_specs = [
            ("covered_valid", "Covers, valid p-set", "#009E73"),
            ("covered_underfilled", "Covers, underfilled", "#56B4E9"),
            ("missed_valid", "Misses, valid p-set", "#D55E00"),
            ("underfilled", "Underfilled / abstains", "#999999"),
            ("no_concrete_set", "No concrete answers", "#666666"),
        ]
        selected_models = sorted({str(row["model"]) for row in selected_rows})
        method_labels = [method_title for method_title, _method_key in method_specs]
        x_labels = [
            f"{model}\n{method_title}"
            for model in selected_models
            for method_title in method_labels
        ]
        x_positions = list(range(len(x_labels)))
        bottoms = [0.0 for _ in x_positions]
        fig, ax = plt.subplots(figsize=(max(8.5, 0.65 * len(x_labels)), 4.8))
        for outcome_key, outcome_label, color in outcome_specs:
            values = []
            label_i = 0
            for model in selected_models:
                model_rows = [row for row in selected_rows if str(row["model"]) == model]
                for _method_title, method_key in method_specs:
                    counts = {key: 0 for key, _label, _color in outcome_specs}
                    n = 0
                    for row in model_rows:
                        outcome = concrete_confidence_set_outcome(
                            response_items_for_method(row, method_key),
                            top_p=coverage_top_p,
                        )
                        if outcome is None:
                            continue
                        counts[outcome] = counts.get(outcome, 0) + 1
                        n += 1
                    values.append(counts.get(outcome_key, 0) / n if n else 0.0)
                    label_i += 1
            ax.bar(
                x_positions,
                values,
                bottom=bottoms,
                color=color,
                label=outcome_label,
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share of questions")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    if want("simpleqa_concrete_confidence_set_outcomes.png"):
        make_concrete_confidence_set_outcome_plot(
            selected_rows=rows,
            title=f"Concrete answer confidence-set outcomes at p={coverage_top_p:.0%}",
            filename="simpleqa_concrete_confidence_set_outcomes.png",
        )
    uncertain_rows = [
        row
        for row in rows
        if row.get("penalty_abstention_rate") not in {None, ""}
        and float(row.get("penalty_abstention_rate") or 0.0) >= 0.5
    ]
    if want("simpleqa_penalty_uncertain_confidence_set_outcomes.png"):
        make_concrete_confidence_set_outcome_plot(
            selected_rows=uncertain_rows,
            title=(
                f"Concrete answer confidence-set outcomes on penalty-uncertain "
                f"questions at p={coverage_top_p:.0%}"
            ),
            filename="simpleqa_penalty_uncertain_confidence_set_outcomes.png",
        )

    def preview_output_token_estimate(row: dict[str, Any], key: str, n_samples: int) -> int:
        raw_preview = row.get(key)
        if not raw_preview:
            return 0
        try:
            preview = json.loads(str(raw_preview))
        except json.JSONDecodeError:
            return 0
        if not isinstance(preview, list) or not preview:
            return 0
        preview_tokens = [estimated_token_count(str(response)) for response in preview]
        mean_preview_tokens = safe_mean([float(value) for value in preview_tokens]) or 0.0
        return int(round(mean_preview_tokens * max(0, n_samples)))

    def token_total_from_recorded_or_estimated_row(
        row: dict[str, Any],
        method_key: str,
    ) -> float | None:
        field_prefix = {
            "empirical": "empirical",
            "log": "log",
            "quadratic": "quadratic",
            "penalty": "penalty",
        }[method_key]
        recorded_total = row.get(f"{field_prefix}_total_tokens")
        if recorded_total not in {None, ""}:
            return float(recorded_total)

        example = SimpleQAExample(
            question_id=str(row["question_id"]),
            question=str(row["question"]),
            answer=str(row["gold_answer"]),
            category=str(row.get("category", "unknown")),
            answer_type=str(row.get("answer_type", "")),
        )
        n_samples = max(1, int(row.get("n_samples_requested") or 1))
        if method_key == "empirical":
            prompt = build_simpleqa_regular_prompt(example)
            input_tokens = estimated_generation_token_usage(
                prompt=prompt,
                system_message=OPEN_ANSWER_SYSTEM_MESSAGE,
                response="",
                input_multiplier=n_samples,
                output_multiplier=0,
            )["input_tokens"]
            output_tokens = preview_output_token_estimate(
                row,
                "sample_response_preview_json",
                n_samples,
            )
            return float(input_tokens + output_tokens)
        if method_key == "log":
            if str(row.get("simpleqa_log_mode", "top-k")) == "top-p":
                prompt = build_simpleqa_topp_log_prompt(
                    example,
                    float(row.get("simpleqa_top_p") or DEFAULT_SIMPLEQA_TOP_P),
                )
            else:
                prompt = build_simpleqa_topk_log_prompt(
                    example,
                    int(row.get("simpleqa_top_k") or DEFAULT_SIMPLEQA_TOP_K),
                )
            usage = estimated_generation_token_usage(
                prompt=prompt,
                system_message=SIMPLEQA_LOG_SYSTEM_MESSAGE,
                response=str(row.get("raw_log_response") or ""),
            )
            return float(usage["total_tokens"])
        if method_key == "quadratic":
            if not row.get("quadratic_candidates_json"):
                return None
            if str(row.get("simpleqa_log_mode", "top-k")) == "top-p":
                prompt = build_simpleqa_topp_quadratic_prompt(
                    example,
                    float(row.get("simpleqa_top_p") or DEFAULT_SIMPLEQA_TOP_P),
                )
            else:
                prompt = build_simpleqa_topk_quadratic_prompt(
                    example,
                    int(row.get("simpleqa_top_k") or DEFAULT_SIMPLEQA_TOP_K),
                )
            usage = estimated_generation_token_usage(
                prompt=prompt,
                system_message=SIMPLEQA_QUADRATIC_SYSTEM_MESSAGE,
                response=str(row.get("raw_quadratic_response") or ""),
            )
            return float(usage["total_tokens"])
        if method_key == "penalty":
            if not row.get("penalty_distribution_json"):
                return None
            penalty = float(row.get("penalty_value") or DEFAULT_PENALTIES[0])
            prompt = build_simpleqa_penalty_prompt(example, penalty)
            input_tokens = estimated_generation_token_usage(
                prompt=prompt,
                system_message=OPEN_ANSWER_SYSTEM_MESSAGE,
                response="",
                input_multiplier=n_samples,
                output_multiplier=0,
            )["input_tokens"]
            output_tokens = preview_output_token_estimate(
                row,
                "penalty_sample_response_preview_json",
                n_samples,
            )
            return float(input_tokens + output_tokens)
        return None

    def format_token_count(value: float) -> str:
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}k"
        return f"{value:.0f}"

    def make_row_metric_bar_plot(
        *,
        metric_fn: Any,
        title: str,
        ylabel: str,
        filename: str,
        value_label_fn: Any,
    ) -> None:
        metric_by_method_model: dict[tuple[str, str], tuple[float, int]] = {}
        for method_title, method_key in method_specs:
            for model in models:
                values = [
                    value
                    for row in rows
                    if str(row["model"]) == model
                    for value in [metric_fn(row, method_key)]
                    if value is not None
                ]
                mean_value = safe_mean([float(value) for value in values])
                if mean_value is not None:
                    metric_by_method_model[(method_title, model)] = (mean_value, len(values))
        if not metric_by_method_model:
            return

        visible_methods = [
            method_title
            for method_title, _method_key in method_specs
            if any((method_title, model) in metric_by_method_model for model in models)
        ]
        bar_width = min(0.24, 0.78 / max(1, len(visible_methods)))
        x_positions = list(range(len(models)))
        fig, ax = plt.subplots(figsize=(max(7.5, 1.45 * len(models)), 4.5))
        for method_i, method_title in enumerate(visible_methods):
            offset = (method_i - (len(visible_methods) - 1) / 2) * bar_width
            xs = []
            ys = []
            ns = []
            for model_i, model in enumerate(models):
                entry = metric_by_method_model.get((method_title, model))
                if entry is None:
                    continue
                mean_value, n_questions = entry
                xs.append(model_i + offset)
                ys.append(mean_value)
                ns.append(n_questions)
            bars = ax.bar(
                xs,
                ys,
                width=bar_width,
                label=method_title,
                color=method_colors.get(method_title, "#56B4E9"),
            )
            for bar, value, n_questions in zip(bars, ys, ns):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + max(1.0, max(ys or [1.0]) * 0.025),
                    f"{value_label_fn(value)}\nn={n_questions}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        all_values = [entry[0] for entry in metric_by_method_model.values()]
        ax.set_ylim(0, max(1.0, max(all_values or [0.0]) * 1.20))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [textwrap.shorten(model, width=34, placeholder="...") for model in models],
            rotation=20,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(visible_methods))
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    if want("simpleqa_answer_model_token_cost.png"):
        make_row_metric_bar_plot(
            metric_fn=token_total_from_recorded_or_estimated_row,
            title="Answer-model token cost per question",
            ylabel="Total tokens per question",
            filename="simpleqa_answer_model_token_cost.png",
            value_label_fn=format_token_count,
        )

    def sample_distribution_items(
        items: list[dict[str, Any]],
        *,
        n_samples: int,
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        clean_items = [
            item
            for item in items
            if float(item.get("probability", 0.0) or 0.0) > 0
        ]
        if not clean_items or n_samples <= 0:
            return []
        weights = [float(item.get("probability", 0.0) or 0.0) for item in clean_items]
        total_weight = sum(weights)
        if total_weight <= 0:
            return []
        draws = rng.choices(
            range(len(clean_items)),
            weights=[weight / total_weight for weight in weights],
            k=n_samples,
        )
        sampled: dict[tuple[str, str], dict[str, Any]] = {}
        for draw in draws:
            item = clean_items[draw]
            answer = str(item.get("answer", ""))
            grade = str(item.get("grade", "incorrect"))
            key = (open_answer_key(answer), grade)
            if key not in sampled:
                sampled[key] = {
                    "answer": answer,
                    "grade": grade,
                    "count": 0,
                    "probability": 0.0,
                }
            sampled[key]["count"] = int(sampled[key]["count"]) + 1
        rows_out = list(sampled.values())
        for item in rows_out:
            item["probability"] = int(item["count"]) / n_samples
        return rows_out

    def sampling_budget_values() -> list[int]:
        max_samples = max(
            [int(row.get("n_samples_requested") or 1) for row in rows] or [1]
        )
        candidates = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, max_samples]
        return sorted({value for value in candidates if 1 <= value <= max_samples})

    def sampling_method_token_cost(
        model_rows: list[dict[str, Any]],
        method_key: str,
        sample_count: int,
    ) -> float | None:
        costs = []
        for row in model_rows:
            total_tokens = token_total_from_recorded_or_estimated_row(row, method_key)
            if total_tokens is None:
                continue
            n_samples = max(1, int(row.get("n_samples_requested") or 1))
            costs.append(float(total_tokens) * sample_count / n_samples)
        return safe_mean(costs)

    def simulated_sampling_coverage(
        model_rows: list[dict[str, Any]],
        method_key: str,
        sample_count: int,
        *,
        metric_fn: Any,
        top_p: float,
        repeats: int = 200,
    ) -> tuple[float | None, float | None, float | None]:
        repeat_means = []
        for repeat_i in range(repeats):
            hits = []
            for row in model_rows:
                items = response_items_for_method(row, method_key)
                if not items:
                    continue
                rng = stable_rng(
                    "simpleqa-top-p-sampling-coverage",
                    str(row["model"]),
                    str(row["question_id"]),
                    method_key,
                    sample_count,
                    repeat_i,
                )
                sampled_items = sample_distribution_items(
                    items,
                    n_samples=sample_count,
                    rng=rng,
                )
                hit = metric_fn(sampled_items, top_p=top_p)
                if hit is not None:
                    hits.append(1.0 if hit else 0.0)
            repeat_mean = safe_mean(hits)
            if repeat_mean is not None:
                repeat_means.append(repeat_mean)
        if not repeat_means:
            return None, None, None
        repeat_means.sort()
        mean_value = safe_mean(repeat_means)
        lo = repeat_means[int(0.05 * (len(repeat_means) - 1))]
        hi = repeat_means[int(0.95 * (len(repeat_means) - 1))]
        return mean_value, lo, hi

    def observed_one_call_coverage_and_cost(
        model_rows: list[dict[str, Any]],
        method_key: str,
        *,
        metric_fn: Any,
        top_p: float,
    ) -> tuple[float | None, float | None]:
        hits = []
        costs = []
        for row in model_rows:
            items = response_items_for_method(row, method_key)
            hit = metric_fn(items, top_p=top_p)
            if hit is not None:
                hits.append(1.0 if hit else 0.0)
            cost = token_total_from_recorded_or_estimated_row(row, method_key)
            if cost is not None:
                costs.append(float(cost))
        return safe_mean(hits), safe_mean(costs)

    def make_top_p_coverage_vs_token_cost_plot(
        *,
        metric_fn: Any,
        ylabel: str,
        title: str,
        filename: str,
    ) -> None:
        if not models:
            return
        sample_counts = sampling_budget_values()
        fig, axes = plt.subplots(
            len(models),
            1,
            figsize=(8.2, max(4.6, 3.8 * len(models))),
            squeeze=False,
            sharey=True,
        )
        sampling_specs = [
            ("Empirical samples", "empirical", "#0072B2"),
            ("Penalty samples", "penalty", "#E0435E"),
        ]
        for ax, model in zip(axes[:, 0], models):
            model_rows = [row for row in rows if str(row["model"]) == model]
            for method_title, method_key, color in sampling_specs:
                xs = []
                ys = []
                lows = []
                highs = []
                for sample_count in sample_counts:
                    token_cost = sampling_method_token_cost(
                        model_rows,
                        method_key,
                        sample_count,
                    )
                    mean_cov, lo_cov, hi_cov = simulated_sampling_coverage(
                        model_rows,
                        method_key,
                        sample_count,
                        metric_fn=metric_fn,
                        top_p=coverage_top_p,
                    )
                    if token_cost is None or mean_cov is None:
                        continue
                    xs.append(token_cost)
                    ys.append(mean_cov)
                    lows.append(lo_cov if lo_cov is not None else mean_cov)
                    highs.append(hi_cov if hi_cov is not None else mean_cov)
                if xs:
                    ax.plot(xs, ys, marker="o", linewidth=1.7, color=color, label=method_title)
                    ax.fill_between(xs, lows, highs, color=color, alpha=0.14, linewidth=0)

            one_call_specs = [
                (log_report_label, "log", "#388210"),
            ]
            if include_quadratic and any(row.get("quadratic_candidates_json") for row in model_rows):
                one_call_specs.append((quadratic_report_label, "quadratic", "#F0B042"))
            for method_label, method_key, color in one_call_specs:
                coverage, token_cost = observed_one_call_coverage_and_cost(
                    model_rows,
                    method_key,
                    metric_fn=metric_fn,
                    top_p=coverage_top_p,
                )
                if coverage is None or token_cost is None:
                    continue
                ax.scatter(
                    [token_cost],
                    [coverage],
                    marker="*",
                    s=160,
                    color=color,
                    edgecolor="#333333",
                    linewidth=0.6,
                    label=method_label,
                    zorder=4,
                )
                ax.text(
                    token_cost,
                    min(1.03, coverage + 0.055),
                    f"{format_token_count(token_cost)}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

            ax.axhline(
                coverage_top_p,
                color="#333333",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label=f"Nominal p={coverage_top_p:.0%}",
            )
            ax.set_title(str(model))
            ax.set_ylim(-0.03, 1.05)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            positive_xs = [
                line_x
                for line in ax.lines
                for line_x in line.get_xdata()
                if float(line_x) > 0
            ]
            if positive_xs:
                ax.set_xscale("log")
            ax.set_xlabel("Answer-model tokens per question")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            unique: dict[str, Any] = {}
            for handle, label in zip(handles, labels):
                unique.setdefault(label, handle)
            fig.legend(
                list(unique.values()),
                list(unique.keys()),
                loc="lower center",
                ncol=min(4, len(unique)),
                frameon=False,
            )
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0.10, 1, 0.95))
        fig.savefig(out_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    if want("simpleqa_top_p_coverage_vs_token_cost.png"):
        make_top_p_coverage_vs_token_cost_plot(
            metric_fn=correct_in_top_probability_mass,
            ylabel="Strict coverage",
            title=f"Top {coverage_top_p:.0%} strict coverage versus token budget",
            filename="simpleqa_top_p_coverage_vs_token_cost.png",
        )
    if want("simpleqa_top_p_coverage_or_idk_vs_token_cost.png"):
        make_top_p_coverage_vs_token_cost_plot(
            metric_fn=correct_or_idk_when_correct_missing_in_top_probability_mass,
            ylabel="Coverage or IDK success",
            title=(
                f"Top {coverage_top_p:.0%} correct-or-IDK success versus token budget"
            ),
            filename="simpleqa_top_p_coverage_or_idk_vs_token_cost.png",
        )

    distribution_rows = rows
    if not want("simpleqa_all_method_response_distributions.pdf"):
        return
    if distribution_max_questions is not None and distribution_max_questions > 0:
        n_distribution_rows = min(distribution_max_questions, len(rows))
        if n_distribution_rows < len(rows):
            indices = [
                (i * len(rows)) // n_distribution_rows
                for i in range(n_distribution_rows)
            ]
            distribution_rows = [rows[i] for i in indices]
    questions_per_page = 3
    probability_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    probability_tick_labels = ["0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    for stale_path in out_dir.glob("simpleqa_all_method_response_distributions*.png"):
        stale_path.unlink(missing_ok=True)
    pdf_path = out_dir / "simpleqa_all_method_response_distributions.pdf"
    page_count = math.ceil(len(distribution_rows) / questions_per_page)

    def one_line_metadata_text(value: Any, width: int) -> str:
        compact = re.sub(r"\s+", " ", str(value)).strip()
        return textwrap.shorten(compact, width=width, placeholder="...")

    def add_metadata_line(ax: Any, row: dict[str, Any]) -> None:
        question_id = one_line_metadata_text(row["question_id"], width=42)
        question = one_line_metadata_text(row["question"], width=190)
        gold = one_line_metadata_text(row["gold_answer"], width=86)
        metadata_font_size = 9.0
        question_id_text = f"Question id: {question_id}"
        question_id_textprops = {
            "fontsize": metadata_font_size,
            "fontweight": "bold",
            "color": "#111111",
        }
        question_parts = [
            TextArea(
                question_id_text,
                textprops=question_id_textprops,
            ),
            TextArea(
                f"Question: {question}",
                textprops={
                    "fontsize": metadata_font_size,
                    "fontweight": "bold",
                    "color": "#8B1A1A",
                },
            ),
        ]
        answer_spacer = TextArea(
            question_id_text,
            textprops={**question_id_textprops, "alpha": 0.0},
        )
        correct_answer = TextArea(
            f"Correct answer: {gold}",
            textprops={
                "fontsize": metadata_font_size,
                "fontweight": "bold",
                "color": "#00843D",
            },
        )
        question_line = HPacker(children=question_parts, align="baseline", pad=0, sep=18)
        answer_line = HPacker(
            children=[answer_spacer, correct_answer],
            align="baseline",
            pad=0,
            sep=18,
        )
        packed = VPacker(
            children=[question_line, answer_line],
            align="left",
            pad=0,
            sep=2,
        )
        anchored = AnchoredOffsetbox(
            loc="center left",
            child=packed,
            frameon=False,
            pad=0.0,
            borderpad=0.0,
            bbox_to_anchor=(0.0, 0.52),
            bbox_transform=ax.transAxes,
        )
        ax.add_artist(anchored)

    with PdfPages(pdf_path) as pdf:
        for page_i, start in enumerate(range(0, len(distribution_rows), questions_per_page), start=1):
            page_rows = distribution_rows[start : start + questions_per_page]
            fig = plt.figure(figsize=(17.2, 15.0))
            band_h = 1.0 / questions_per_page
            left = 0.082
            right = 0.018
            col_gap = 0.028
            col_count = len(method_specs)
            col_w = (1.0 - left - right - col_gap * (col_count - 1)) / col_count
            plot_top_pad = 0.026
            plot_text_gap = 0.017
            text_h = 0.058
            band_bottom_pad = 0.017
            axes_grid: list[list[Any]] = []
            text_axes = []
            for row_i in range(len(page_rows)):
                band_top = 1.0 - row_i * band_h
                band_bottom = band_top - band_h
                text_bottom = band_bottom + band_bottom_pad
                plot_bottom = text_bottom + text_h + plot_text_gap
                plot_top = band_top - plot_top_pad
                plot_h = max(0.05, plot_top - plot_bottom)
                axes_grid.append(
                    [
                        fig.add_axes(
                            [
                                left + col_i * (col_w + col_gap),
                                plot_bottom,
                                col_w,
                                plot_h,
                            ]
                        )
                        for col_i in range(len(method_specs))
                    ]
                )
                text_ax = fig.add_axes([left, text_bottom, 1.0 - left - right, text_h])
                text_ax.axis("off")
                text_axes.append(text_ax)
            for row_i, row in enumerate(page_rows):
                for col_i, (method_title, method_key) in enumerate(method_specs):
                    ax = axes_grid[row_i][col_i]
                    raw_items = response_items_for_method(row, method_key)
                    if should_show_all_items(row, method_key):
                        items = sorted(raw_items, key=lambda item: float(item.get("probability", 0.0)), reverse=True)
                    else:
                        items = trimmed_items(raw_items)
                    if not items:
                        ax.text(0.5, 0.5, "Not run", ha="center", va="center", transform=ax.transAxes)
                        ax.set_yticks([])
                        ax.set_xlim(0, 1)
                        ax.set_xticks(probability_ticks)
                        ax.set_xticklabels(probability_tick_labels, fontsize=7)
                        ax.tick_params(axis="x", labelbottom=True)
                        ax.set_title(method_title, fontsize=9)
                        continue
                    labels = [
                        textwrap.shorten(str(item.get("answer", "")), width=34, placeholder="...")
                        for item in items
                    ]
                    probs = [float(item.get("probability", 0.0)) for item in items]
                    grades = [str(item.get("grade", "incorrect")) for item in items]
                    y_positions = list(range(len(items)))
                    ax.barh(
                        y_positions,
                        probs,
                        color=[color_map.get(grade, "#999999") for grade in grades],
                    )
                    label_fontsize = 6.4 if len(items) > 7 else 7
                    value_fontsize = 0.8 * label_fontsize
                    for y_pos, prob in zip(y_positions, probs):
                        value_label = f"{prob:.2f}"
                        if prob >= 0.94:
                            ax.text(
                                prob - 0.015,
                                y_pos,
                                value_label,
                                ha="right",
                                va="center",
                                fontsize=value_fontsize,
                                color="#222222",
                            )
                        else:
                            ax.text(
                                prob + 0.012,
                                y_pos,
                                value_label,
                                ha="left",
                                va="center",
                                fontsize=value_fontsize,
                                color="#222222",
                            )
                    ax.set_yticks(y_positions)
                    ax.set_yticklabels(labels, fontsize=label_fontsize)
                    ax.invert_yaxis()
                    ax.set_xlim(0, 1)
                    ax.set_xticks(probability_ticks)
                    ax.set_xticklabels(probability_tick_labels, fontsize=7)
                    ax.tick_params(axis="x", labelbottom=True)
                    ax.grid(axis="x", alpha=0.25)
                    ax.set_title(method_title, fontsize=9)
                add_metadata_line(text_axes[row_i], row)
            fig.canvas.draw()
            for boundary_i in range(1, questions_per_page):
                y = 1.0 - boundary_i * band_h
                fig.add_artist(
                    plt.Line2D(
                        [0.02, 0.98],
                        [y, y],
                        transform=fig.transFigure,
                        color="#BBBBBB",
                        linewidth=0.8,
                    )
                )
            pdf.savefig(fig)
            plt.close(fig)
