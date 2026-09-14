"""What every pipeline returns, and the one place an answer is generated.

All three pipelines end the same way: context in, one short answer out, under the
same model, temperature and output cap (LOCKED-7). Keeping generation here means a
token difference between pipelines can only come from how much context they chose
to retrieve, which is exactly the quantity the benchmark is measuring.
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agentic.llm import LLMClient, Usage, count_tokens
from agentic.tools.tools import Evidence

# The corpus is the only source of truth, and the benchmark scores an exact
# string. Both are stated to the model, because a model that answers from its own
# Olympic knowledge will be right often enough to be dangerous and wrong in
# exactly the cases the corpus was built to test.
SYSTEM_PROMPT = """You answer questions using ONLY the provided context.

Rules:
- The context is the only source of truth. If it disagrees with what you know, the context wins.
- A value the context states is the answer even when it looks implausible. Never discard or "correct" a stored number.
- Answer with the shortest complete answer: a name, a number, or a title. No sentences, no explanation, no units unless the question asks for them.
- For a list of people, give the names exactly as the context writes them.
- When several winners are stored as one string, reproduce that string exactly,
  including the absence of separators between names. The corpus's own spelling is
  the answer; adding commas changes it.
- If the answer is an event, give its full title exactly as the context writes it,
  including the sport and the Games, not just the event name.
- If the context does not contain the answer, reply with exactly: INSUFFICIENT_EVIDENCE
"""

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


@dataclass
class TraceStep:
    """One action, recorded whether it helped or not.

    The fields are the ones the hackathon scores agentic behaviour on: which tool
    ran, what it cost, and whether it moved the investigation forward.
    """

    step_no: int
    agent: str
    tool: str
    retrieval_method: str
    args_digest: str
    status: str
    latency_ms: float
    context_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    chunks_returned: int = 0
    new_evidence_count: int = 0
    # Whether the call returned usable data at all. A count is progress even
    # though it cites no document, so evidence alone cannot measure it.
    produced_data: bool = False
    gap_before: str = ""
    gap_after: str = ""
    strategy_change: bool = False
    note: str = ""


@dataclass
class PipelineResult:
    qid: str
    question: str
    pipeline: str
    answer: str
    status: str                      # ok | insufficient_evidence | error
    route: str = "standard"
    stop_reason: str = ""
    citations: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    context_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    wall_clock_s: float = 0.0
    # Of the wall clock, how much was sleeping on the provider's rate limit.
    # `wall_clock_s - throttled_s` is the time the system itself took.
    throttled_s: float = 0.0
    coverage_expected: int | None = None
    coverage_actual: int | None = None
    trace: list[TraceStep] = field(default_factory=list)
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["total_tokens"] = self.total_tokens
        return row


def render_context(blocks: list[tuple[str, str]], token_budget: int) -> tuple[str, int]:
    """Join labelled context blocks, stopping at the budget.

    Truncation is by whole blocks rather than mid-document: half a document is a
    misleading citation, and the pipelines' token numbers only compare if each one
    is held to the same ceiling.
    """
    kept: list[str] = []
    used = 0
    for label, text in blocks:
        block = f"[{label}]\n{text.strip()}"
        cost = count_tokens(block)
        if used + cost > token_budget and kept:
            break
        kept.append(block)
        used += cost
    return "\n\n".join(kept), used


def generate_answer(llm: LLMClient, question: str, context: str) -> tuple[str, Usage]:
    """One generation call, identical in every pipeline."""
    response = llm.complete([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"},
    ])
    return response.text.strip(), response.usage


def finish(result: PipelineResult, answer: str, usage: Usage, started: float,
           evidence: list[Evidence]) -> PipelineResult:
    """Common tail: record the answer, its cost, and what it was grounded in."""
    result.answer = answer
    result.status = "insufficient_evidence" if answer.strip() == INSUFFICIENT else "ok"
    result.input_tokens += usage.input_tokens
    result.output_tokens += usage.output_tokens
    result.llm_calls += usage.calls
    seen: set[str] = set()
    for item in evidence:
        if item.doc_id and item.doc_id not in seen:
            seen.add(item.doc_id)
    result.citations = list(seen)
    result.evidence = [e.as_dict() for e in evidence]
    result.wall_clock_s = round(time.perf_counter() - started, 3)
    result.throttled_s = round(usage.throttled_ms / 1000, 3)
    return result
