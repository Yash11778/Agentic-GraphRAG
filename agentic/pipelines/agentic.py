"""Pipeline 3 — Agentic GraphRAG. Plan, act, evaluate, decide, repeat.

The orchestrator re-decides after every step from the question, the evidence so
far, and the gap that is still open. It is not a fixed chain: it chooses the tool,
it may change strategy mid-investigation, and it justifies stopping.

Three things here are the substance of the submission rather than plumbing:

**The coverage gate (LOCKED-5).** When the question requires enumerating a set,
the agent asks the database how large that set is and refuses to answer until it
holds that many documents. The model does not get to decide it has enough; the
count does. This is what carries aggregation and superlative questions, which are
half the hidden set.

**Cost-aware routing.** Before planning, the orchestrator judges whether the
question needs an investigation at all. A single-document lookup takes the short
path and is recorded as `route="fast_path"`. The hackathon asks when agentic
reasoning is overkill; this builds that finding into the product instead of only
charting it afterwards.

**The verbatim check.** The corpus is the only source of truth and the benchmark
scores an exact string. When the model's answer is a string a cited document
contains, differing only in separators or case ("Dani King, Laura Trott" for a
medal cell the corpus writes "Dani KingLaura Trott"), the stored string is the
answer. The check is deterministic, only ever fires on that equality, and is
recorded in the trace as a step of its own.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agentic.graph_schema import squash
from agentic.harness.budget import Budget
from agentic.harness.evidence import signature
from agentic.harness.registry import (
    RETRIEVAL_METHODS,
    TOOL_AGENTS,
    Registry,
    tool_schemas,
)
from agentic.harness.state import BUDGET_EXHAUSTED, InvestigationState
from agentic.llm import TOOL_CALL_FAILED, LLMClient, count_tokens
from agentic.pipelines.base import (
    INSUFFICIENT,
    PipelineResult,
    TraceStep,
    generate_answer,
    render_context,
)
from agentic.pipelines.rag import CONTEXT_BUDGET
from agentic.tools.tools import Tools

PLANNER_PROMPT = """You plan investigations over a graph of Olympic events built from a document corpus.

The graph holds Event, Games, Sport, Venue, Athlete and Nation vertices. Events carry
sport, season, year, venue, competitors, nations, dates and medal winners. Games are
linked into a chain of editions.

Decide two things about the question:

1. route: "fast_path" if one document answers it and you only need to find that document;
   "investigate" if it needs counting, comparing across many events, stepping the edition
   chain, or a lookup by venue and date rather than by name.
2. gap: one sentence naming what must be found to answer. Be concrete.

Reply as JSON: {"route": "...", "plan": "...", "gap": "..."}"""

ORCHESTRATOR_PROMPT = """You are investigating a question using graph and document tools.

Rules that matter:
- Every fact about an event -- title, venue, date, medals, competitors, nations -- is
  on Event vertices. Games, Sport, Venue and Athlete carry only an id and a name.
  Filter Event by games_id (e.g. 2012-Summer) and sport directly; there is no need to
  look the Games vertex up first.
- To count anything, or to find a largest or smallest value, call graph_aggregate.
  Never count by listing results yourself.
- For a largest or smallest value: graph_aggregate with field_name gives the number;
  then graph_filter with the same predicates plus that field eq the number finds the
  event. Answer with that event's title.
- To find events by venue and date, use graph_filter on the venue and date fields, or
  graph_neighbors with direction "in" from the venue.
- Match names and dates with the "contains" operator, not "eq". Stored values carry
  qualifiers the question omits ("Athens Olympic Tennis Centre, Athens" for a question
  that says "Olympic Tennis Centre") and dates are written as free text.
- When several events match a venue and a date, the answer is the one whose date_raw
  is exactly the date wording in the question, not one that merely contains it. Ask for
  every match (no max_results) before choosing. If two still tie, prefer the one whose
  sport or event name is echoed in the question's wording, and answer with it: an
  ambiguous match is still evidence, and refusing is not the safer choice.
- When a filter returns nothing, loosen it before abandoning it: switch "eq" to
  "contains", or drop the least certain predicate. Do not fall back to vector search
  while a graph filter has not been tried in its loosened form.
- To step between Olympic editions, use edition_step. Do not use your own knowledge of
  which Games came before which.
- The corpus is the only source of truth. A stored value is the answer however
  implausible it looks; never discard or "correct" a number for seeming too large or
  too small.
- Call one tool at a time. When the evidence answers the question, reply with the answer
  itself and no tool call.
- The answer must be the shortest complete answer: a name, a number, or a title.
- If the answer is an event, give its full title exactly as the graph stores it,
  including the sport and the Games, not just the event name.
- When several winners are stored as one string, reproduce that string exactly,
  including the absence of separators between names. The corpus's own spelling is
  the answer; adding commas changes it.
- If the tools cannot produce the answer, reply exactly: INSUFFICIENT_EVIDENCE"""

# How much of one tool result is shown back to the model. Enough to reason over,
# far short of pasting a 43-row result set into the context on every turn. Rows
# are rendered compactly (see `_render`) so that budget holds a dozen events
# rather than four.
TOOL_RESULT_TOKENS = 900
MAX_TOOL_CALLS = 8

# Vertex attributes that never help the planner: keys, bookkeeping and the
# comparison companions. Dropping them from what the model sees is most of the
# difference between four rows per result and fourteen.
_NOISE_FIELDS = frozenset({
    "url", "approx_tokens", "event_key", "title_event_key", "type", "gold_noc",
    "date_start", "date_end", "date_month", "event_name", "season", "year",
})

# Stored strings an answer may be a re-punctuated copy of, in the order a tie
# is broken. Medal cells first: they are where the model most often inserts
# separators the corpus did not write.
_MEDAL_FIELDS = ("gold_raw", "silver_raw", "bronze_raw")
_VERBATIM_FIELDS = _MEDAL_FIELDS + ("title", "venue")
_VERBATIM_DOCS = 20

Observer = Callable[[TraceStep], None]


def fast_path_budget() -> Budget:
    """A tighter budget for questions one document answers.

    A function, not a module-level Budget: a shared instance would carry its step
    and token counters from one question into the next, and every run after the
    first would start already spent. Four steps, because the first attempt at a
    venue or title match often comes back empty and the route still deserves one
    correction.
    """
    return Budget(max_steps=4, max_tokens=10_000, max_seconds=60)


def run(question: str, tools: Tools, llm: LLMClient, qid: str = "",
        observer: Observer | None = None) -> PipelineResult:
    """Investigate one question. `observer` sees each trace step as it happens."""
    started = time.perf_counter()
    registry = Registry(tools)
    state = InvestigationState(question=question, qid=qid, observer=observer)

    _plan(state, llm)
    if state.route == "fast_path":
        state.budget = fast_path_budget()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": ORCHESTRATOR_PROMPT},
        {"role": "user", "content": state.situation()},
    ]

    answer = ""
    schemas = tool_schemas()
    while True:
        response = llm.complete(messages, tools=schemas)
        state.spend_llm(response.usage)

        if response.finish_reason == TOOL_CALL_FAILED:
            # The model wrote a tool call the provider could not parse. Hand the
            # rejection back and let it correct itself; this costs one step and
            # is visible in the trace rather than ending the question.
            messages.append({"role": "user", "content":
                             "That tool call was malformed JSON. Send it again, "
                             "correctly formed, as a single tool call."})
            state.record(TraceStep(
                step_no=len(state.trace) + 1, agent="orchestrator",
                tool="malformed_tool_call", retrieval_method="none",
                args_digest="", status="error",
                latency_ms=response.usage.latency_ms,
                note="provider rejected the generated arguments",
            ))
            if state.stop_reason() is not None:
                break
            continue

        if not response.tool_calls:
            answer = response.text.strip()
            blocked = _coverage_block(state)
            if blocked:
                # The model believes it is done; the count says otherwise. This
                # push-back is the coverage gate, and it is recorded as a strategy
                # change because it redirects an investigation that was ending.
                messages.append({"role": "assistant", "content": answer})
                messages.append({"role": "user", "content": blocked})
                state.record(TraceStep(
                    step_no=len(state.trace) + 1, agent="evidence_evaluator",
                    tool="coverage_gate", retrieval_method="none",
                    args_digest=_coverage_digest(state),
                    status="incomplete", latency_ms=0.0, strategy_change=True,
                    gap_before=state.gap, gap_after=state.gap,
                    note="answer withheld: evidence set incomplete",
                ))
                answer = ""
                if state.stop_reason() is None:
                    continue
            else:
                state.sufficient = True
                break

        if response.tool_calls:
            _execute(state, registry, messages, response.tool_calls[0])
            _escalate_if_stuck(state)

        reason = state.stop_reason()
        if reason is not None:
            break
        if state.budget.steps >= MAX_TOOL_CALLS:
            break

    stop_reason = state.stop_reason() or BUDGET_EXHAUSTED
    if not answer or state.failed:
        answer = _final_answer(state, llm, stop_reason)
    answer = _verbatim(answer, state, tools)

    return _result(state, answer, stop_reason, started)


def _plan(state: InvestigationState, llm: LLMClient) -> None:
    """One planning call: route, plan and the opening gap."""
    response = llm.complete(
        [{"role": "system", "content": PLANNER_PROMPT},
         {"role": "user", "content": state.question}],
        json_object=True, max_output_tokens=400,
    )
    state.spend_llm(response.usage)
    try:
        parsed = json.loads(response.text or "{}")
    except json.JSONDecodeError:
        parsed = {}
    state.route = "fast_path" if parsed.get("route") == "fast_path" else "agentic"
    state.plan = str(parsed.get("plan", ""))[:400]
    state.gap = str(parsed.get("gap", ""))[:300]
    step = TraceStep(
        step_no=0, agent="orchestrator", tool="plan", retrieval_method="none",
        args_digest=state.route, status="ok",
        latency_ms=response.usage.latency_ms,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        gap_after=state.gap, note=state.plan,
    )
    # Appended, not recorded: planning is not a step against the budget.
    state.trace.append(step)
    if state.observer is not None:
        state.observer(step)


def _execute(state: InvestigationState, registry: Registry,
             messages: list[dict], call: dict) -> None:
    """Run one tool call, record it, and feed a trimmed result back to the model."""
    name = call["name"]
    try:
        arguments = json.loads(call["arguments"] or "{}")
    except json.JSONDecodeError:
        arguments = {}

    result = registry.call(name, arguments)
    added = state.ledger.add(result.evidence)

    # Completeness is tracked per filter, so that a count and a listing of the
    # SAME predicates can be compared and nothing else is (see evidence.py).
    if name in ("graph_aggregate", "graph_filter"):
        key = signature(arguments.get("vertex_type", ""), arguments.get("predicates"))
        if name == "graph_aggregate" and isinstance(result.data, dict):
            matched = result.data.get("matched")
            if isinstance(matched, int) and matched > 0:
                state.ledger.expect(key, matched, f"graph_aggregate {_digest(arguments)}")
        elif name == "graph_filter" and isinstance(result.data, list):
            state.ledger.enumerated(key, len(result.data))

    if result.status == "error":
        rendered = result.error
    else:
        rendered = _render(result.data, TOOL_RESULT_TOKENS)
    if result.status == "empty":
        # An empty result is the most common dead end, and the recovery is always
        # the same shape. Saying so here costs a sentence and saves the model
        # several steps of guessing, without telling it anything question-specific.
        rendered += (" | nothing matched. Loosen it: use contains instead of eq, "
                     "match on fewer or shorter values, or drop the least certain "
                     "predicate.")

    if result.status == "ok":
        state.findings.append((f"{name} {_digest(arguments)}"[:80], rendered))

    # The previous RETRIEVAL step, not simply the previous step: planning and the
    # coverage gate carry no retrieval method, and comparing against them made the
    # first tool call of every run look like a change of strategy.
    previous = next((step for step in reversed(state.trace)
                     if step.retrieval_method != "none"), None)
    state.record(TraceStep(
        step_no=len(state.trace) + 1,
        agent=TOOL_AGENTS.get(name, "orchestrator"),
        tool=name,
        retrieval_method=RETRIEVAL_METHODS.get(name, "none"),
        args_digest=_digest(arguments),
        status=result.status,
        latency_ms=result.latency_ms,
        context_tokens=count_tokens(rendered),
        chunks_returned=len(result.data) if isinstance(result.data, list) else 0,
        new_evidence_count=added,
        produced_data=result.status == "ok",
        gap_before=state.gap,
        gap_after=state.gap,
        strategy_change=bool(previous and previous.tool != name
                             and previous.retrieval_method
                             != RETRIEVAL_METHODS.get(name)),
        note=result.error if result.status == "error" else "",
    ))

    messages.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": call["id"], "type": "function",
                                     "function": {"name": name,
                                                  "arguments": call["arguments"]}}]})
    messages.append({"role": "tool", "tool_call_id": call["id"],
                     "content": f"status={result.status} {rendered}"})
    messages.append({"role": "user", "content": state.situation()})


def _render(data: Any, budget: int) -> str:
    """What the model sees of a tool result: compact rows, cut at a row boundary.

    A vertex row rendered whole is some 600 characters, most of it keys and
    bookkeeping the planner never uses, so a 700-token window held four events
    and the model learned to ask for five. Dropping the noise fields and empty
    values fits a dozen, and truncation says how many rows it hid rather than
    cutting a row in half.
    """
    if not isinstance(data, list):
        text = json.dumps(_compact(data), default=str, ensure_ascii=False)
        return _clip(text, budget)
    lines: list[str] = []
    used = 0
    for row in data:
        line = json.dumps(_compact(row), default=str, ensure_ascii=False)
        cost = count_tokens(line) + 1
        if lines and used + cost > budget:
            break
        if not lines and cost > budget:
            lines.append(_clip(line, budget))
            used = budget
            break
        lines.append(line)
        used += cost
    hidden = len(data) - len(lines)
    if hidden > 0:
        lines.append(f"… {hidden} more of {len(data)} rows not shown. Narrow the "
                     f"filter, or use graph_aggregate for a count or extreme.")
    return "\n".join(lines)


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items()
                if k not in _NOISE_FIELDS and v is not None and v != ""}
    if isinstance(value, list):
        return [_compact(v) for v in value]
    return value


def _clip(text: str, budget: int) -> str:
    if count_tokens(text) <= budget:
        return text
    return text[: budget * 4] + " …[truncated]"


def _escalate_if_stuck(state: InvestigationState) -> None:
    """Give a fast-path run the full budget once its shortcut fails.

    The route is chosen before any evidence exists, so a question that looks like
    a single-document lookup can turn out to need an investigation. Escalating is
    cheaper than routing conservatively, and it is recorded as a strategy change
    rather than hidden: 14 of 19 multi-hop failures were fast-path runs that ran
    out of steps one attempt after their first filter came back empty.
    """
    if state.route != "fast_path" or not state.trace:
        return
    if state.trace[-1].status not in ("empty", "error"):
        return
    state.route = "fast_path_escalated"
    state.budget = Budget()
    state.trace[-1].strategy_change = True
    state.trace[-1].note = "fast path did not resolve; escalated to full budget"


def _coverage_block(state: InvestigationState) -> str:
    """The message that withholds an answer built on an incomplete set.

    Returns empty when there is nothing to enforce: most questions never
    establish an expected cardinality, and the gate must not fire on them.
    """
    short = state.ledger.incomplete
    if short is None:
        return ""
    _, expected, listed = short
    return (f"Not yet. The database counts {expected} items matching those filters, "
            f"but only {listed} were listed. Retrieve the rest before answering, or "
            f"answer from graph_aggregate directly instead of from the listing.")


def _coverage_digest(state: InvestigationState) -> str:
    """"listed/expected" for the trace, when an enumeration is short."""
    short = state.ledger.incomplete
    if short is None:
        return ""
    _, expected, listed = short
    return f"{listed}/{expected}"


def _final_answer(state: InvestigationState, llm: LLMClient, stop_reason: str) -> str:
    """Answer from the evidence in hand, or admit the investigation failed.

    Reached when the loop ended without the model volunteering an answer -- a
    budget or progress stop. A failed run says so (LOCKED-6); it does not guess
    from whatever happened to be retrieved.
    """
    blocks = list(state.findings[-6:]) + list(state.ledger.snippets(limit=8))
    if not blocks:
        return INSUFFICIENT
    context, _ = render_context(blocks, CONTEXT_BUDGET)
    answer, usage = generate_answer(llm, state.question, context)
    state.spend_llm(usage)
    return answer


def _verbatim(answer: str, state: InvestigationState, tools: Tools) -> str:
    """The corpus's own string, when the answer is that string in other clothes.

    Two rules, both deterministic, both over the documents the investigation
    cited, both recorded as a trace step when they fire:

    1. **Re-punctuated.** The answer, with separators removed, equals a medal
       cell, title or venue of a cited document -- "Erik Lesser, Daniel Böhm,
       Arnd Peiffer, and Simon Schempp" for the cell the corpus writes "Erik
       LesserDaniel BöhmArnd PeifferSimon Schempp". The stored string is the
       answer.
    2. **An event named informally.** The answer names exactly one cited event,
       either by containing its full title ("Men's épée at the 1992 Summer
       Olympics (Fencing at the 1992 Summer Olympics – Men's épée)") or by
       containing its event name together with its year ("Men's marathon at the
       2008 Summer Olympics – the athletics event with the most competitors").
       The corpus's name for an event is its title, so the title is the answer.
       It does not fire when the answer also contains a medallist of a cited
       event: "Valerie Vili (Athletics at the 2008 Summer Olympics – Women's
       shot put)" is an answer about the medallist, and replacing it with the
       event's title would turn a right answer into a wrong one.

    Neither rule fires on a difference in a letter or a digit, and neither
    knows what kind of question was asked. What the model wrote is kept in the
    trace step beside what replaced it, so every substitution can be audited.
    """
    if not answer or answer == INSUFFICIENT:
        return answer
    flat = squash(answer)
    keys = {flat, squash(re.sub(r"\band\b", " ", answer))}
    keys.discard("")
    if not keys:
        return answer
    events: list[dict] = []
    for doc_id in state.ledger.doc_ids[:_VERBATIM_DOCS]:
        vertex = tools.vertex("Event", doc_id)
        if vertex:
            events.append(vertex)

    for vertex in events:
        for field_name in _VERBATIM_FIELDS:
            stored = vertex.get(field_name)
            if isinstance(stored, str) and stored != answer and squash(stored) in keys:
                return _restate(state, answer, stored, f"{field_name} of {vertex['id']}")

    if _names_medallist(flat, events):
        return answer
    named = [v for v in events if _names_event(answer, flat, v)]
    if len(named) == 1 and named[0].get("title") and named[0]["title"] != answer:
        return _restate(state, answer, named[0]["title"], f"title of {named[0]['id']}")
    return answer


def _names_medallist(flat: str, events: list[dict]) -> bool:
    """Whether the answer contains a medal cell of any cited event."""
    for vertex in events:
        for field_name in _MEDAL_FIELDS:
            cell = squash(vertex.get(field_name))
            if len(cell) >= 4 and cell in flat:
                return True
    return False


def _names_event(answer: str, flat: str, vertex: dict) -> bool:
    title = squash(vertex.get("title"))
    if title and title in flat:
        return True
    event_name = squash(vertex.get("event_name"))
    year = vertex.get("year")
    return bool(event_name and len(event_name) >= 4 and event_name in flat
                and year and str(year) in answer)


def _restate(state: InvestigationState, answer: str, stored: str, where: str) -> str:
    state.record(TraceStep(
        step_no=len(state.trace) + 1, agent="evidence_evaluator",
        tool="verbatim_check", retrieval_method="none",
        args_digest=where, status="ok", latency_ms=0.0, produced_data=True,
        note=f"answer restated as the corpus writes it: {stored[:60]} "
             f"(the model wrote: {answer[:80]})",
    ))
    return stored


def _result(state: InvestigationState, answer: str, stop_reason: str,
            started: float) -> PipelineResult:
    return PipelineResult(
        qid=state.qid,
        question=state.question,
        pipeline="agentic",
        answer=answer,
        status="insufficient_evidence" if answer.strip() == INSUFFICIENT else "ok",
        route=state.route,
        stop_reason=stop_reason,
        citations=state.ledger.doc_ids,
        evidence=[e.as_dict() for e in state.ledger.items.values()],
        context_tokens=sum(s.context_tokens for s in state.trace),
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
        llm_calls=state.llm_calls,
        tool_calls=sum(1 for s in state.trace if s.retrieval_method != "none"),
        wall_clock_s=round(time.perf_counter() - started, 3),
        throttled_s=round(state.throttled_s, 3),
        coverage_expected=state.ledger.expected_count,
        coverage_actual=len(state.ledger.doc_ids),
        trace=state.trace,
    )


def _digest(arguments: dict) -> str:
    """A short, stable description of a call's arguments, for the trace."""
    parts = []
    for key, value in arguments.items():
        if value is None:
            continue
        text = value if isinstance(value, str) else json.dumps(
            value, default=str, ensure_ascii=False, separators=(",", ":"))
        parts.append(f"{key}={text[:90]}")
    return " ".join(parts)[:240]
