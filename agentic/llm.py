"""Groq chat client, plus the token accounting the benchmark is scored on.

One client, one budget, three pipelines. Every generation in the repo goes
through `complete()`, which is the only place the model name, temperature and
output cap are applied -- the mechanical guarantee behind LOCKED-7.

Token numbers come from the provider's own `usage` block rather than from a
local tokenizer. A local count is an estimate, and for a reasoning model it
cannot see the hidden reasoning tokens at all; `usage` is what
was actually billed, and the benchmark claims token efficiency, so the honest
number is the billed one. `count_tokens` exists only to measure *context* tokens
-- how much retrieved text a pipeline chose to put in front of the model -- where
a consistent approximation across pipelines is what matters, not absolute truth.
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import tiktoken
from groq import APIStatusError, BadRequestError, Groq, RateLimitError

from agentic.config import LLMSettings

# Shared by the chunker, which needs a fixed, reproducible token boundary.
# The served model's tokenizer is not tiktoken's, so this over- or under-counts
# by a few percent. It is applied identically to all three pipelines, which keeps
# the comparison fair; absolute context-token values are reported as approximate.
ENCODING = tiktoken.get_encoding("cl100k_base")

_MAX_ATTEMPTS = 6
_BACKOFF_BASE_S = 2.0

# The longest we will ever wait on one rate-limit response. The provider answers
# a per-minute overage with a short retry-after and an exhausted period with a
# very long one; honouring the long value parks the whole benchmark in a sleep
# with no output. Past this cap the call fails, the runner records it, and the
# wait is visible instead of invisible.
_MAX_RETRY_SLEEP_S = 90.0

# Stay this far clear of the per-minute token allowance. The free tier grants
# 8,000 tokens a minute, which one agentic question can spend on its own, so
# firing calls blind means most are rejected and the run spends its life in
# exponential back-off. Pausing before the allowance runs out turns that into
# steady throughput.
_TOKEN_HEADROOM = 1200

# The provider validates tool-call arguments itself and rejects a malformed
# generation with a 400 rather than returning it. That is a recoverable mistake by
# the model, not a broken request, so it is surfaced as a response the caller can
# answer instead of an exception that ends the investigation.
TOOL_CALL_FAILED = "tool_use_failed"


class QuotaExhausted(RuntimeError):
    """The provider asked for a wait longer than any single call should absorb.

    Distinct from an ordinary failure because the right response is different: no
    later question will fare better, so the run should stop and be resumed once
    the quota resets, rather than marking every remaining question as failed in
    the four minutes it takes to ask them all.
    """

    def __init__(self, wait_s: float):
        self.wait_s = wait_s
        super().__init__(
            f"provider asked for a {wait_s:.0f}s wait ({wait_s / 60:.0f} min); "
            f"the token quota for this period is exhausted"
        )


def count_tokens(text: str) -> int:
    """Approximate token length of retrieved context. See module docstring."""
    return len(ENCODING.encode(text or "", disallowed_special=()))


@dataclass
class Usage:
    """Token and latency accounting for one or more model calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    latency_ms: float = 0.0
    # Time spent sleeping on rate limits. Reported separately because it is the
    # provider's delay, not the agent's, and a wall-clock budget that counts it
    # ends investigations for a reason that has nothing to do with their quality.
    throttled_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: Usage) -> Usage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls
        self.latency_ms += other.latency_ms
        self.throttled_ms += other.throttled_ms
        return self

    def as_dict(self) -> dict[str, float | int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.calls,
            "llm_latency_ms": round(self.latency_ms, 1),
            "throttled_ms": round(self.throttled_ms, 1),
        }


@dataclass
class LLMResponse:
    text: str
    usage: Usage
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""

    @property
    def truncated(self) -> bool:
        """True when the output cap cut the answer off.

        Worth surfacing rather than swallowing: under an equal output budget a
        truncated answer is a real result about that pipeline's verbosity, and
        scoring it as a normal wrong answer would hide the reason.
        """
        return self.finish_reason == "length"


class LLMClient:
    """Thin wrapper over Groq's chat completions.

    Retries only on rate limiting and 5xx, and never on a 4xx that indicates a
    malformed request -- retrying a bad request burns free-tier quota and turns a
    clear error into a slow one.
    """

    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or LLMSettings.from_env()
        self._keys = list(self.settings.all_keys)
        self._key_index = 0
        self._client = self._new_client()
        self._throttled_s = 0.0
        self._remaining_tokens: int | None = None
        self._tokens_reset_s = 0.0

    def _new_client(self) -> Groq:
        # A bounded request timeout, and no SDK-level retries: retrying happens
        # here, where the rate-limit accounting lives, and a hung connection is
        # otherwise indistinguishable from a slow model.
        return Groq(api_key=self._keys[self._key_index], timeout=120.0, max_retries=0)

    @property
    def key_label(self) -> str:
        """Which credential is in use, for logs. Never the key itself."""
        return f"key {self._key_index + 1}/{len(self._keys)}"

    def _next_key(self) -> bool:
        """Switch to the next credential. False when there is none left."""
        if self._key_index + 1 >= len(self._keys):
            return False
        self._key_index += 1
        self._client = self._new_client()
        self._remaining_tokens = None
        return True

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        tools: Iterable[dict[str, Any]] | None = None,
        json_object: bool = False,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """One chat completion under the shared budget.

        `max_output_tokens` is overridable only for internal planning calls,
        which are not the answer generation LOCKED-7 governs; answer generation
        always passes None and therefore always gets the configured cap.
        """
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": max_output_tokens or self.settings.max_output_tokens,
        }
        if self.settings.reasoning_effort:
            # gpt-oss bills its private reasoning as output tokens, so effort is
            # part of the generation budget and must be identical across the three
            # pipelines (LOCKED-7). These answers are short factual strings; high
            # effort buys nothing here and inflates the token numbers we report.
            kwargs["reasoning_effort"] = self.settings.reasoning_effort
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = "auto"
            # One call per turn is the contract the orchestrator states in its
            # prompt and enforces in its loop; a second call in the same turn
            # would be generated, billed, and discarded.
            kwargs["parallel_tool_calls"] = False
        if json_object:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        self._throttled_s = 0.0
        self._await_allowance()
        try:
            completion = self._call_with_retry(kwargs)
        except BadRequestError as exc:
            if _is_tool_call_failure(exc):
                return LLMResponse(
                    text=f"Your tool call was not valid JSON and was rejected: {exc}",
                    usage=Usage(calls=1,
                                latency_ms=(time.perf_counter() - started) * 1000),
                    finish_reason=TOOL_CALL_FAILED,
                )
            raise
        latency_ms = (time.perf_counter() - started) * 1000

        choice = completion.choices[0]
        raw_usage = completion.usage
        usage = Usage(
            input_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
            calls=1,
            latency_ms=latency_ms,
            throttled_ms=self._throttled_s * 1000,
        )
        calls = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
            for tc in (choice.message.tool_calls or [])
        ]
        return LLMResponse(
            text=(choice.message.content or "").strip(),
            usage=usage,
            tool_calls=calls,
            finish_reason=choice.finish_reason or "",
        )

    def _await_allowance(self) -> None:
        """Make room for another call: rotate credentials, or wait.

        Uses the server's own accounting rather than a local estimate, since only
        one of them decides whether a request is rejected.

        With several credentials the per-minute allowance is not a reason to
        sleep -- the next one has its own. Rotating turns a run that is capped at
        one key's 8,000 tokens a minute into one bounded by request latency, which
        is the difference between three hours and one. Sleeping is the fallback
        for when every credential is low at once.
        """
        if self._remaining_tokens is None or self._remaining_tokens > _TOKEN_HEADROOM:
            return
        if self._rotate():
            return
        wait = min(self._tokens_reset_s, _MAX_RETRY_SLEEP_S)
        if wait > 0:
            self._throttled_s += wait
            time.sleep(wait)
        self._remaining_tokens = None

    def _rotate(self) -> bool:
        """Move to the next credential in the ring. False when there is only one."""
        if len(self._keys) < 2:
            return False
        self._key_index = (self._key_index + 1) % len(self._keys)
        self._client = self._new_client()
        self._remaining_tokens = None
        return True

    def _note_limits(self, headers) -> None:
        """Record what the provider says is left of this minute's allowance."""
        try:
            self._remaining_tokens = int(headers.get("x-ratelimit-remaining-tokens"))
            self._tokens_reset_s = _parse_duration(
                headers.get("x-ratelimit-reset-tokens") or "0s")
        except (TypeError, ValueError):
            self._remaining_tokens = None

    def _call_with_retry(self, kwargs: dict[str, Any]):
        last: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                raw = self._client.chat.completions.with_raw_response.create(**kwargs)
                self._note_limits(raw.headers)
                return raw.parse()
            except RateLimitError as exc:
                last = exc
                wait = _retry_after(exc, attempt)
                if wait > _MAX_RETRY_SLEEP_S:
                    # This credential is out of daily quota. Try the next one
                    # before giving up; only when all of them are spent does the
                    # run stop.
                    if self._next_key():
                        continue
                    raise QuotaExhausted(wait) from exc
                self._throttled_s += wait
                time.sleep(wait)
            except APIStatusError as exc:
                if exc.status_code < 500:
                    raise
                last = exc
                wait = _BACKOFF_BASE_S * (2**attempt)
                self._throttled_s += wait
                time.sleep(wait)
        raise RuntimeError(
            f"Groq call failed after {_MAX_ATTEMPTS} attempts: {last}"
        ) from last


def _parse_duration(text: str) -> float:
    """Parse the provider's duration strings: "577ms", "1m26.4s", "12s"."""
    text = (text or "").strip()
    if text.endswith("ms"):
        return float(text[:-2]) / 1000
    total, number = 0.0, ""
    for char in text:
        if char.isdigit() or char == ".":
            number += char
        elif char == "m":
            total += float(number or 0) * 60
            number = ""
        elif char == "s":
            total += float(number or 0)
            number = ""
    return total + (float(number) if number else 0.0)


def _is_tool_call_failure(exc: BadRequestError) -> bool:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return (body.get("error") or {}).get("code") == TOOL_CALL_FAILED
    return TOOL_CALL_FAILED in str(exc)


def _retry_after(exc: Exception, attempt: int) -> float:
    """Honour the server's own Retry-After when it sends one.

    Groq's free tier returns a precise wait; sleeping longer wastes wall clock on
    a 150-question benchmark, and sleeping shorter just earns another 429.
    """
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}).get("retry-after") if response else None
    if header:
        try:
            return float(header) + 0.5
        except ValueError:
            pass
    return _BACKOFF_BASE_S * (2**attempt)
