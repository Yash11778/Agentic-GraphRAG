"""Single source of runtime configuration.

Everything that can differ between a laptop run and the scored run is read here
and nowhere else, so a pipeline can never quietly pick up a different model,
temperature or backend than the other two. That matters because LOCKED-7 in
docs/ARCHITECTURE.md makes an equal generation budget part of the benchmark's
validity: if one pipeline were allowed a larger output cap, its token numbers
would stop being comparable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

CORPUS = ROOT / "hackathon-resources/corpus/corpus.jsonl"
PUBLIC_QUESTIONS = ROOT / "hackathon-resources/questions/eval_public.jsonl"
HIDDEN_QUESTIONS = ROOT / "hackathon-resources/questions/eval_hidden.jsonl"
GRAPH_DIR = ROOT / "data/graph"
VECTOR_DIR = ROOT / "data/vectors"
RESULTS_DIR = ROOT / "data/results"

load_dotenv(ROOT / ".env")


class ConfigError(RuntimeError):
    """Raised when a required setting is absent, rather than failing later inside
    an HTTP call where the cause is much harder to read off a stack trace."""


def _require(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise ConfigError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value


def _int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    return float(raw) if raw else default


@dataclass(frozen=True)
class LLMSettings:
    """The generation budget, shared verbatim by all three pipelines (LOCKED-7)."""

    model: str
    temperature: float
    max_output_tokens: int
    api_key: str
    reasoning_effort: str
    # Additional keys to fall back to when one is out of daily quota. Groq counts
    # its daily token limit per ORGANISATION, not per key, so extra keys minted
    # from one account share one pool and buy nothing; only keys belonging to
    # separate accounts (a teammate's own, with their consent) add quota.
    fallback_keys: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> LLMSettings:
        return cls(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
            temperature=_float("GROQ_TEMPERATURE", 0.0),
            max_output_tokens=_int("GROQ_MAX_OUTPUT_TOKENS", 512),
            api_key=_require("GROQ_API_KEY"),
            reasoning_effort=os.getenv("GROQ_REASONING_EFFORT", "low").strip(),
            fallback_keys=tuple(
                k.strip() for k in (os.getenv("GROQ_API_KEYS") or "").split(",")
                if k.strip()
            ),
        )

    @property
    def all_keys(self) -> tuple[str, ...]:
        """Every key to try, primary first, without duplicates."""
        seen, keys = set(), []
        for key in (self.api_key, *self.fallback_keys):
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
        return tuple(keys)


@dataclass(frozen=True)
class TigerGraphSettings:
    host: str
    secret: str
    graph: str

    @classmethod
    def from_env(cls) -> TigerGraphSettings:
        return cls(
            host=_require("TG_HOST").rstrip("/"),
            secret=_require("TG_PASSWORD"),
            graph=_require("TG_GRAPH"),
        )


def graph_backend_name() -> str:
    """Which backend the tool layer should construct.

    `local` reads data/graph/*.jsonl and needs no network, which is what makes
    iterating on the agent possible while the Savanna workspace is suspended.
    `tigergraph` is the scored path. Both implement one interface, so the choice
    cannot change results -- only where the data is read from.
    """
    name = (os.getenv("GRAPH_BACKEND") or "local").strip().lower()
    if name not in ("local", "tigergraph"):
        raise ConfigError(
            f"GRAPH_BACKEND={name!r} is not valid; use 'local' or 'tigergraph'."
        )
    return name
