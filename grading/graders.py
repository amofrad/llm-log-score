"""Graders and the grading rubric.

Every response is graded against the SimpleQA gold answer by a language-model
grader. Two are interchangeable:

- OpenAIGrader: GPT-5.6 Terra via the OpenAI API (primary grader).
- GeminiGrader: Gemini via a Vertex client from runner/engine.py (second
  grader, for the grader-agreement audit).

Both use the same rubric: the label-only prompt from simpleqa_grade_prompt,
the GRADER_SYSTEM_MESSAGE system message, temperature 0, and the
parse_simpleqa_grade parser. Abstention-style responses are caught
mechanically by is_open_not_attempted and never sent to a grader.

Answer-string canonicalization also lives here (open_answer_key for
deduplication, canonical_incorrect_answer_key for equivalence-class
matching); the runner engine and analysis stage import it from this module.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

GRADER_SYSTEM_MESSAGE = (
    "You are a strict factual answer grader. "
    "Output only CORRECT, INCORRECT, or NOT_ATTEMPTED."
)
DEFAULT_OPENAI_GRADER_MODEL = "gpt-5.6-terra"
DEFAULT_GEMINI_GRADER_MODEL = "google/gemini-3.5-flash"
DEFAULT_GRADER_MAX_TOKENS = 128


# Rubric
def simpleqa_grade_prompt(example, response: str) -> str:
    """Label-only grading prompt. example needs .question and .answer."""
    return f"""[Grade whether the model answer is equivalent to the target answer.]

Question:
{example.question}

Target answer:
{example.answer}

Model answer:
{response}

Return only one label:
CORRECT - the model answer is semantically equivalent to the target answer.
INCORRECT - the model answer attempts an answer but is not equivalent.
NOT_ATTEMPTED - the model does not answer, says it does not know, or abstains.
"""


def parse_simpleqa_grade(text: str) -> str | None:
    normalized = re.sub(r"[^A-Z_]", "", text.strip().upper())
    if normalized.startswith("CORRECT"):
        return "correct"
    if normalized.startswith("INCORRECT"):
        return "incorrect"
    if normalized.startswith("NOT_ATTEMPTED") or normalized.startswith("NOTATTEMPTED"):
        return "not_attempted"
    return None


# Answer-string normalization and canonicalization
def normalize_open_answer_text(text: str) -> str:
    value = re.sub(r"\s+", " ", text.strip())
    value = value.strip().strip("\"'`*_")
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    return value


def open_answer_key(text: str) -> str:
    value = normalize_open_answer_text(text).casefold()
    value = re.sub(r"^[\"'`*_]+|[\"'`*_.!,;:\s]+$", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def extract_short_answer_surface(text: str) -> str:
    value = normalize_open_answer_text(text)
    value = re.sub(r"^#+\s*answer\s*:?\s*", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"^(?:final\s*:|answer\s*:)\s*", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"\s+each\s+hosted\b.*$", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"\s+during\s+the\s+\d{4}\s+FIFA\s+World\s+Cup\b.*$", "", value, flags=re.IGNORECASE).strip()

    had_match = re.search(
        r"\bhad\s+((?:zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+sons?)\b",
        value,
        re.IGNORECASE,
    )
    if had_match:
        return had_match.group(1).strip()

    dangling_quote_prefix = re.match(
        r"^([^\"“”]{1,90})[\"”]\s*(?:\(|\b(?:was|is|were|are)\b)",
        value,
        re.IGNORECASE,
    )
    if dangling_quote_prefix:
        candidate = dangling_quote_prefix.group(1).strip()
        if candidate:
            return candidate

    if not re.search(r"\bnot\s+named\s+after\b", value, re.IGNORECASE):
        named_after_match = re.search(
            r"\bnamed\s+after\s+(?:the\s+)?([^.;,()]{1,90})",
            value,
            re.IGNORECASE,
        )
        if named_after_match:
            candidate = named_after_match.group(1).strip()
            if candidate:
                return candidate

    quoted_match = re.search(r"[\"“]\s*([^\"”.;]{1,90})\s*(?:[\"”]|[.;]|$)", value)
    if quoted_match:
        quoted = quoted_match.group(1).strip()
        if quoted:
            return quoted

    prefix_match = re.match(
        r"^(.{1,90}?)\s+(?:was|is|were|are)\s+the\s+",
        value,
        re.IGNORECASE,
    )
    if prefix_match:
        candidate = prefix_match.group(1).strip()
        if candidate and not candidate.lower().startswith(("the ", "this ")):
            return candidate

    tail_match = re.search(
        r"\b(?:was|is|were|are|would\s+be|would\s+have\s+been)\s+([^.;]{1,120})(?:[.;]|$)",
        value,
        re.IGNORECASE,
    )
    if tail_match:
        candidate = tail_match.group(1).strip()
        candidate = re.sub(r"^(?:the\s+answer\s+is\s+)", "", candidate, flags=re.IGNORECASE)
        if candidate:
            return candidate

    return value


def canonical_incorrect_answer_key(text: str) -> str:
    value = open_answer_key(extract_short_answer_surface(text))
    value = re.sub(r"^(?:the|a|an)\s+", "", value)
    value = re.sub(
        r"^(?:the\s+)?(?:super\s+)?(?:typhoon|tropical\s+storm|storm|hurricane|cyclone)\s+",
        "",
        value,
    )
    value = re.sub(r"\bzero\s+sons?\b", "0 son", value)
    value = re.sub(r"\bone\s+sons?\b", "1 son", value)
    value = re.sub(r"\btwo\s+sons?\b", "2 sons", value)
    value = re.sub(r"\bthree\s+sons?\b", "3 sons", value)
    value = re.sub(r"\bfour\s+sons?\b", "4 sons", value)
    value = re.sub(r"\bfive\s+sons?\b", "5 sons", value)
    value = re.sub(r"\bsix\s+sons?\b", "6 sons", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or open_answer_key(text)


def canonical_incorrect_answer_display(text: str) -> str:
    value = extract_short_answer_surface(text)
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"^(?:the\s+)?(?:super\s+)?(?:Typhoon|Tropical\s+Storm|Storm|Hurricane|Cyclone)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip().strip("\"'`*_.,;: ")
    return value or normalize_open_answer_text(text)


def canonical_stadium_answer(text: str) -> tuple[str, str] | None:
    value = open_answer_key(extract_short_answer_surface(text))
    aliases = [
        ("fnb/soccer_city", "FNB Stadium / Soccer City", [r"\bfnb stadium\b", r"\bsoccer city\b"]),
        ("cape_town", "Cape Town Stadium", [r"\bcape town stadium\b", r"\bgreen point stadium\b"]),
        ("nelson_mandela_bay", "Nelson Mandela Bay Stadium", [r"\bnelson mandela bay stadium\b"]),
        ("moses_mabhida", "Moses Mabhida Stadium", [r"\bmoses mabhida\b", r"\bdurban stadium\b"]),
        ("ellis_park", "Ellis Park Stadium", [r"\bellis park\b", r"\bcoca-cola park\b"]),
        ("loftus_versfeld", "Loftus Versfeld Stadium", [r"\bloftus versfeld\b"]),
        ("royal_bafokeng", "Royal Bafokeng Stadium", [r"\broyal bafokeng\b"]),
    ]
    found: list[tuple[str, str]] = []
    for key, display, patterns in aliases:
        if any(re.search(pattern, value) for pattern in patterns):
            found.append((key, display))
    if not found:
        return None
    key = "stadiums::" + "|".join(key for key, _display in found)
    display = ", ".join(display for _key, display in found)
    return key, display


def is_open_not_attempted(text: str) -> bool:
    key = open_answer_key(text)
    if not key:
        return True
    markers = {
        "unknown",
        "i don't know",
        "i do not know",
        "idk",
        "dont know",
        "do not know",
        "not sure",
        "unsure",
        "insufficient information",
        "cannot determine",
        "can't determine",
        "cannot answer",
        "can't answer",
        "not enough information",
        "no answer",
        "abstain",
    }
    return key in markers


class OpenAIGrader:
    """GPT-5.6 Terra via the OpenAI Responses API (label-only protocol)."""

    name = "openai"

    def __init__(self, api_key: str, model: str = DEFAULT_OPENAI_GRADER_MODEL):
        self.api_key = api_key
        self.model = model

    def grade(self, example, answer: str) -> str:
        if is_open_not_attempted(answer):
            return "not_attempted"
        body = {
            "model": self.model,
            "reasoning": {"effort": "none"},
            "temperature": 0,
            "max_output_tokens": 16,
            "input": [
                {"role": "system", "content": GRADER_SYSTEM_MESSAGE},
                {"role": "user", "content": simpleqa_grade_prompt(example, answer)},
            ],
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(body).encode(),
            method="POST",
        )
        request.add_header("Authorization", f"Bearer {self.api_key}")
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API HTTP {exc.code}: {detail}") from exc
        texts = []
        for message in payload.get("output") or []:
            for content in message.get("content") or []:
                if content.get("type") == "output_text":
                    texts.append(str(content.get("text", "")))
        label = parse_simpleqa_grade("\n".join(texts))
        if label is None:
            raise ValueError(f"Unparseable grader output: {texts!r}")
        return label


class GeminiGrader:
    """Gemini via a Vertex client created by runner/engine.py.

    The client is injected so this module stays free of GCP dependencies:

        from engine import create_gcp_client_for_model
        client = create_gcp_client_for_model(model=..., project=..., location=...)
        grader = GeminiGrader(client)
    """

    name = "gemini"

    def __init__(self, client, model: str = DEFAULT_GEMINI_GRADER_MODEL):
        self.client = client
        self.model = model

    def grade(self, example, answer: str, seed: int = 0) -> str:
        if is_open_not_attempted(answer):
            return "not_attempted"
        output = self.client.generate(
            model=self.model,
            prompt=simpleqa_grade_prompt(example, answer),
            system_message=GRADER_SYSTEM_MESSAGE,
            seed=seed,
            temperature=0,
            max_tokens=DEFAULT_GRADER_MAX_TOKENS,
        )
        return parse_simpleqa_grade(output) or "incorrect"
