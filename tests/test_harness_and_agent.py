"""The agent harness: budget, evidence ledger, stop criteria, tool dispatch, and
the orchestrator's pure helpers. No model is called anywhere in this file."""
from __future__ import annotations

import time

import pytest

from agentic.harness.budget import Budget
from agentic.harness.evidence import EvidenceLedger, signature
from agentic.harness.registry import Registry, _nullable, _predicates, tool_schemas
from agentic.harness.state import (
    BUDGET_EXHAUSTED,
    COVERAGE_COMPLETE,
    NO_PROGRESS,
    SUFFICIENT,
    UNRECOVERABLE,
    InvestigationState,
)
from agentic.llm import LLMSettings, Usage, _parse_duration
from agentic.pipelines import agentic
from agentic.pipelines.base import INSUFFICIENT, TraceStep, render_context
from agentic.pipelines.graphrag import rank_candidates
from agentic.tools.tools import Evidence


def step(tool="graph_filter", status="ok", method="graph", data=True, docs=0, digest="x"):
    return TraceStep(step_no=1, agent="a", tool=tool, retrieval_method=method,
                     args_digest=digest, status=status, latency_ms=1.0,
                     produced_data=data, new_evidence_count=docs)


# ── budget ───────────────────────────────────────────────────────────────────

def test_budget_soft_and_hard_caps():
    b = Budget(max_steps=4, max_tokens=100, max_seconds=1000)
    assert not b.pressured and not b.exhausted
    b.spend(steps=3)
    assert b.pressured and not b.exhausted
    b.spend(steps=1)
    assert b.exhausted and b.reason() == "step cap 4"


def test_budget_credit_time_gives_back_throttled_seconds():
    b = Budget(max_seconds=10)
    b.started -= 20                      # pretend 20s elapsed
    assert b.exhausted
    b.credit_time(15)
    assert not b.exhausted and b.elapsed == pytest.approx(5, abs=0.5)


# ── evidence ledger ──────────────────────────────────────────────────────────

def test_signature_ignores_predicate_order_and_shape():
    a = signature("Event", [{"field": "year", "operator": "eq", "value": 2012},
                            {"field": "sport", "op": "eq", "value": "Fencing"}])
    b = signature("event", [("sport", "eq", "fencing"), ("year", "eq", "2012")])
    assert a == b


def test_ledger_dedups_and_gates_on_the_same_predicates():
    ledger = EvidenceLedger()
    assert ledger.add([Evidence("Q1", "s", "t"), Evidence("Q1", "s", "t"), Evidence("", "s", "t")]) == 1
    assert ledger.add([Evidence("Q1", "s", "t")]) == 0
    key = signature("Event", [("year", "eq", 2012)])
    other = signature("Event", [("year", "eq", 2008)])
    ledger.expect(key, 5, "count")
    assert ledger.complete is None                    # nothing enumerated yet
    ledger.enumerated(other, 1)
    assert ledger.complete is None                    # a different filter says nothing
    ledger.enumerated(key, 3)
    assert ledger.incomplete == (key, 5, 3)
    assert ledger.complete is False
    assert "short: 3 of 5" in ledger.summary()
    ledger.enumerated(key, 5)
    assert ledger.complete is True and ledger.incomplete is None
    assert ledger.expected_count == 5


# ── state and stop reasons ───────────────────────────────────────────────────

def test_stop_reasons_in_order():
    s = InvestigationState(question="q")
    assert s.stop_reason() is None
    s.record(step(status="empty", data=False))
    s.record(step(status="empty", data=False, digest="y"))
    assert s.stop_reason() == NO_PROGRESS
    s.record(step(status="error", data=False, digest="z"))
    assert s.stop_reason() == UNRECOVERABLE
    assert s.failed
    s.sufficient = True
    assert s.stop_reason() == SUFFICIENT and not s.failed


def test_non_retrieval_steps_do_not_count_as_failures_or_stalls():
    s = InvestigationState(question="q")
    s.record(step(tool="malformed_tool_call", status="error", method="none", data=False))
    s.record(step(tool="coverage_gate", status="incomplete", method="none", data=False))
    assert s.failed_steps == 0 and s.barren_steps == 0
    s.record(step(status="empty", data=False))
    assert s.failed_steps == 1 and s.barren_steps == 1


def test_repeating_a_call_is_standing_still():
    s = InvestigationState(question="q")
    s.record(step(digest="same"))
    assert s.barren_steps == 0
    s.record(step(digest="same"))
    assert s.barren_steps == 1


def test_coverage_complete_beats_budget_and_observer_sees_steps():
    seen = []
    s = InvestigationState(question="q", observer=seen.append)
    s.budget = Budget(max_steps=1)
    key = signature("Event", [("year", "eq", 2012)])
    s.ledger.expect(key, 1, "c")
    s.ledger.enumerated(key, 1)
    s.record(step())
    assert s.stop_reason() == COVERAGE_COMPLETE
    assert len(seen) == 1
    s.ledger.enumerations.clear()
    s.ledger.expectations.clear()
    assert s.stop_reason() == BUDGET_EXHAUSTED


def test_spend_llm_tracks_throttling_separately():
    s = InvestigationState(question="q")
    s.spend_llm(Usage(input_tokens=10, output_tokens=5, calls=1, throttled_ms=2500))
    assert s.throttled_s == 2.5 and s.llm_calls == 1 and s.budget.tokens == 15


# ── registry ─────────────────────────────────────────────────────────────────

def test_predicates_accept_three_shapes():
    want = [("year", "eq", 2012), ("venue", "contains", "Eton")]
    assert _predicates([{"field": "year", "operator": "eq", "value": 2012},
                        {"field": "venue", "op": "contains", "value": "Eton"}]) == want
    assert _predicates([["year", "eq", 2012], ["venue", "contains", "Eton"]]) == want
    assert _predicates('[{"field":"year","operator":"eq","value":2012},'
                       '{"field":"venue","operator":"contains","value":"Eton"}]') == want
    assert _predicates(None) == []
    assert _predicates([{"field": None, "operator": "eq", "value": 1}]) == []


def test_nullable_adds_null_to_type_and_enum():
    s = _nullable({"type": "string", "enum": ["a"]})
    assert s["type"] == ["string", "null"] and s["enum"] == ["a", None]


def test_schemas_name_the_fields_the_tools_accept():
    by_name = {f["function"]["name"]: f for f in tool_schemas()}
    assert set(by_name) == {"vector_search", "doc_fetch", "entity_link", "graph_neighbors",
                            "graph_filter", "graph_aggregate", "edition_step", "medal_lookup"}
    assert "Games: id (e.g. 2012-Summer), name, year, season" in by_name["graph_filter"]["function"]["description"]


def test_registry_dispatch_and_bad_arguments(tools):
    reg = Registry(tools)
    r = reg.call("graph_filter", {"vertex_type": "Event",
                                  "predicates": [{"field": "competitors", "operator": "gt", "value": "63"}]})
    assert r.status == "ok" and [v["id"] for v in r.data] == ["E3", "E4"]
    assert reg.call("no_such_tool", {}).status == "error"
    assert reg.call("graph_filter", {}).status == "error"


# ── orchestrator helpers ─────────────────────────────────────────────────────

def test_render_is_compact_and_cuts_at_rows(backend):
    rows = backend.find("Event", [])
    text = agentic._render(rows, budget=120)
    assert "approx_tokens" not in text and '"url"' not in text
    assert "more of 7 rows not shown" in text
    assert text.count("\n") < len(rows)
    whole = agentic._render(rows, budget=100_000)
    assert whole.count("\n") == len(rows) - 1
    assert '"title": "Fencing at the 2012 Summer Olympics – Women\'s épée"' in whole
    assert agentic._render({"matched": 3, "max": None}, 100) == '{"matched": 3}'


def test_digest_is_short_and_skips_nulls():
    d = agentic._digest({"vertex_type": "Event", "max_results": None,
                         "predicates": [{"field": "venue", "operator": "contains", "value": "x"}]})
    assert d.startswith("vertex_type=Event predicates=")
    assert "max_results" not in d
    assert len(agentic._digest({"a": "y" * 1000})) <= 240


def test_verbatim_snaps_only_on_separator_differences(tools):
    s = InvestigationState(question="q")
    s.ledger.add([Evidence("E7", "s", "graph_filter")])
    assert agentic._verbatim("Dani King, Laura Trott and Joanna Rowsell", s, tools) \
        == "Dani KingLaura TrottJoanna Rowsell"
    assert s.trace[-1].tool == "verbatim_check"
    assert agentic._verbatim("Dani King", s, tools) == "Dani King"          # a real difference
    assert agentic._verbatim("Dani KingLaura TrottJoanna Rowsell", s, tools) \
        == "Dani KingLaura TrottJoanna Rowsell"                           # already verbatim, no step
    assert agentic._verbatim(INSUFFICIENT, s, tools) == INSUFFICIENT
    assert agentic._verbatim("", s, tools) == ""
    s.ledger.add([Evidence("E1", "s", "graph_filter")])
    assert agentic._verbatim("fencing at the 2012 summer olympics - women's epee.", s, tools) \
        == "Fencing at the 2012 Summer Olympics – Women's épée"


def test_verbatim_restates_an_event_named_informally(tools):
    s = InvestigationState(question="q")
    s.ledger.add([Evidence("E3", "s", "graph_filter"), Evidence("E1", "s", "graph_filter")])
    title = "Fencing at the 2008 Summer Olympics – Men's épée"
    # Contains the full title, wrapped in prose.
    assert agentic._verbatim(f"Men's épée at the 2008 Summer Olympics ({title})", s, tools) == title
    # Event name plus year, no title.
    assert agentic._verbatim("Men's épée at the 2008 Summer Olympics – the fencing event "
                             "with the most competitors.", s, tools) == title
    # Event name without the year is not enough.
    assert agentic._verbatim("Men's épée", s, tools) == "Men's épée"
    # Two cited events named at once: ambiguous, left alone.
    both = "Men's épée 2008 and Women's épée 2012"
    assert agentic._verbatim(both, s, tools) == both
    # A person's name never matches an event.
    assert agentic._verbatim("Matteo Tagliariol", s, tools) == "Matteo Tagliariol"


def test_coverage_block_message():
    s = InvestigationState(question="q")
    assert agentic._coverage_block(s) == ""
    key = signature("Event", [("year", "eq", 2012)])
    s.ledger.expect(key, 43, "c")
    s.ledger.enumerated(key, 5)
    assert "counts 43" in agentic._coverage_block(s)
    assert agentic._coverage_digest(s) == "5/43"


def test_fast_path_budget_is_fresh_each_time():
    a, b = agentic.fast_path_budget(), agentic.fast_path_budget()
    a.spend(steps=3)
    assert b.steps == 0 and a.max_steps == 4


# ── base / graphrag / llm helpers ────────────────────────────────────────────

def test_render_context_truncates_by_whole_block():
    blocks = [("a", "one two three"), ("b", "four five six"), ("c", "seven")]
    text, used = render_context(blocks, token_budget=12)
    assert text.startswith("[a]") and "[c]" not in text and used <= 12
    text, _ = render_context(blocks, token_budget=1)
    assert text.startswith("[a]")          # the first block is always kept


def test_rank_candidates_prefers_multiply_reached_then_title_overlap():
    cands = {
        "x": {"vertex": {"id": "x", "title": "Shooting at the 2004 Summer Olympics – Men's trap"}, "reached_by": 1},
        "y": {"vertex": {"id": "y", "title": "Shooting at the 2004 Summer Olympics – Men's skeet"}, "reached_by": 2},
        "z": {"vertex": {"id": "z", "title": "Shooting at the 2004 Summer Olympics – Women's trap"}, "reached_by": 1},
    }
    assert rank_candidates("Who won the men's trap shooting in 2004?", cands) == ["y", "x", "z"]


def test_parse_duration_and_settings_keys(monkeypatch):
    assert _parse_duration("577ms") == pytest.approx(0.577)
    assert _parse_duration("1m26.4s") == pytest.approx(86.4)
    assert _parse_duration("12s") == 12
    assert _parse_duration("") == 0
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setenv("GROQ_API_KEYS", "k2, k1 ,k3")
    assert LLMSettings.from_env().all_keys == ("k1", "k2", "k3")


def test_usage_add_and_dict():
    u = Usage(1, 2, 1, 10.0, 5.0).add(Usage(3, 4, 1, 20.0, 0.0))
    assert (u.input_tokens, u.output_tokens, u.calls, u.total_tokens) == (4, 6, 2, 10)
    assert u.as_dict()["throttled_ms"] == 5.0


def test_trace_step_defaults_are_cheap_to_build():
    t0 = time.perf_counter()
    for _ in range(1000):
        step()
    assert time.perf_counter() - t0 < 1.0
