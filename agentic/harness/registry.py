"""The tool catalogue handed to the model, and the dispatcher that runs its choice.

Two things live here because they must not drift apart: the JSON schemas the model
plans against, and the code that executes what it planned. A schema describing a
parameter the dispatcher ignores is how an agent silently stops doing what its
trace says it did.

The schemas describe *general primitives* (LOCKED-1). None of them mentions a
question type, and `graph_filter`'s predicate list is deliberately open-ended:
composing the right predicates is the model's job and is the behaviour being
scored.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agentic.graph_schema import PLANNER_FIELDS
from agentic.tools.tools import ToolResult, Tools

# Written once from the schema table, so the fields the planner is told about
# are the fields the tool layer accepts.
_FIELD_GUIDE = "; ".join(f"{vtype}: {fields}" for vtype, fields in PLANNER_FIELDS.items())

# Which specialist each tool belongs to, for the trace. The hackathon asks which
# specialised agents were invoked; this is where that answer comes from, rather
# than from a narration the model writes about itself.
TOOL_AGENTS = {
    "vector_search": "similarity_searcher",
    "doc_fetch": "doc_retriever",
    "entity_link": "entity_linker",
    "graph_neighbors": "graph_traverser",
    "graph_filter": "graph_traverser",
    "graph_aggregate": "aggregator",
    "edition_step": "multihop_reasoner",
    "medal_lookup": "multihop_reasoner",
}

RETRIEVAL_METHODS = {
    "vector_search": "vector",
    "doc_fetch": "document",
    "entity_link": "lexical",
    "graph_neighbors": "graph",
    "graph_filter": "graph",
    "graph_aggregate": "graph",
    "edition_step": "graph",
    "medal_lookup": "graph",
}

# Objects rather than positional triples: a tuple-typed `items` array is not
# accepted by the provider's schema validator, and naming the parts makes the
# model's plan readable in the trace without a legend.
_PREDICATES_SCHEMA = {
    "type": "array",
    "description": (
        "Filters combined with AND. String comparison ignores case and accents."
    ),
    "items": {
        "type": "object",
        "properties": {
            "field": {"type": "string"},
            "operator": {"type": "string",
                         "enum": ["eq", "ne", "gt", "gte", "lt", "lte",
                                  "contains", "exists"]},
            "value": {"type": ["string", "number", "boolean"]},
        },
        "required": ["field", "operator", "value"],
    },
}


def _nullable(schema: dict) -> dict:
    """Let an optional parameter be sent as null.

    Models fill in every property they were shown, including the ones they do not
    want, and a bare `null` against a `"type": "string"` is rejected by the
    provider before the call ever reaches us. Accepting null costs nothing and
    removes a whole class of failed generations.
    """
    types = schema["type"]
    schema = dict(schema)
    schema["type"] = ([types, "null"] if isinstance(types, str)
                      else list(types) + ["null"])
    if "enum" in schema:
        schema["enum"] = list(schema["enum"]) + [None]
    return schema


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-style function schemas for every primitive."""
    return [
        _fn("vector_search",
            "Find corpus chunks similar to a query string. Use when the question "
            "names no entity the graph knows, or to locate a document by wording.",
            {"query": {"type": "string"},
             "k": _nullable({"type": "integer",
                             "description": "how many chunks, default 12"})},
            ["query"]),
        _fn("doc_fetch",
            "Fetch the full text of documents by id. Use after another tool has "
            "identified which documents matter.",
            {"doc_ids": {"type": "array", "items": {"type": "string"}}},
            ["doc_ids"]),
        _fn("entity_link",
            "Find graph vertices whose names appear in a piece of text. Returns "
            "type, id and confidence. Usually the first step.",
            {"text": {"type": "string"}},
            ["text"]),
        _fn("graph_neighbors",
            "Walk edges from one vertex. Edge types: AT_GAMES, IN_SPORT, HELD_AT, "
            "WON_GOLD, WON_SILVER, WON_BRONZE, REPRESENTS, PRECEDED_BY. Direction "
            "'in' reaches the vertices pointing AT this one, e.g. all events held "
            "at a venue.",
            {"vertex_type": {"type": "string"},
             "vertex_id": {"type": "string"},
             "edge_types": _nullable({"type": "array", "items": {"type": "string"}}),
             "direction": _nullable({"type": "string",
                                     "enum": ["out", "in", "both"]})},
            ["vertex_type", "vertex_id"]),
        _fn("graph_filter",
            "All vertices of a type matching every predicate. Every fact about an "
            "event (title, venue, date, medals, competitors) is on Event. "
            "Fields by type -- " + _FIELD_GUIDE + ". Omit max_results to see "
            "every match; results are shown compactly.",
            {"vertex_type": {"type": "string"},
             "predicates": _PREDICATES_SCHEMA,
             "max_results": _nullable({"type": "integer"})},
            ["vertex_type", "predicates"]),
        _fn("graph_aggregate",
            "Count the vertices matching predicates, and optionally the sum, max "
            "and min of one numeric field. Always use this to count or to find a "
            "largest or smallest value. Never count by listing.",
            {"vertex_type": {"type": "string"},
             "predicates": _PREDICATES_SCHEMA,
             "field_name": _nullable({
                 "type": "string",
                 "description": "numeric field to reduce, e.g. competitors"})},
            ["vertex_type", "predicates"]),
        _fn("edition_step",
            "Step the Olympic Games chain from one edition to the previous or next "
            "one, using the corpus's own links rather than outside knowledge.",
            {"games_id": {"type": "string", "description": "e.g. 2016-Summer"},
             "direction": _nullable({"type": "string",
                                     "enum": ["previous", "next"]}),
             "steps": _nullable({"type": "integer"})},
            ["games_id"]),
        _fn("medal_lookup",
            "The athletes who won a given medal at one event, with their nation.",
            {"event_id": {"type": "string"},
             "medal": _nullable({"type": "string",
                                 "enum": ["gold", "silver", "bronze"]})},
            ["event_id"]),
    ]


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required},
        },
    }


class Registry:
    """Executes a tool call by name, with the arguments the model produced."""

    def __init__(self, tools: Tools):
        self.tools = tools

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            return ToolResult(data=None, status="error",
                              error=f"no such tool: {name}")
        try:
            return handler(arguments)
        except (TypeError, KeyError, ValueError) as exc:
            # A malformed argument list -- a wrong type, a missing required
            # parameter, an unparseable predicate string -- is the model's
            # mistake, and it can recover from it if told; crashing the
            # investigation over it cannot.
            detail = f"missing argument {exc}" if isinstance(exc, KeyError) else str(exc)
            return ToolResult(data=None, status="error",
                              error=f"bad arguments for {name}: {detail}")

    # ── one method per schema entry ──────────────────────────────────────────
    def _vector_search(self, a: dict) -> ToolResult:
        return self.tools.vector_search(a["query"], k=int(a.get("k") or 12))

    def _doc_fetch(self, a: dict) -> ToolResult:
        return self.tools.doc_fetch(a["doc_ids"])

    def _entity_link(self, a: dict) -> ToolResult:
        return self.tools.entity_link(a["text"])

    def _graph_neighbors(self, a: dict) -> ToolResult:
        return self.tools.graph_neighbors(
            a["vertex_type"], a["vertex_id"],
            a.get("edge_types") or None, a.get("direction") or "out",
        )

    def _graph_filter(self, a: dict) -> ToolResult:
        return self.tools.graph_filter(
            a["vertex_type"], _predicates(a.get("predicates")),
            int(a.get("max_results") or 0),
        )

    def _graph_aggregate(self, a: dict) -> ToolResult:
        return self.tools.graph_aggregate(
            a["vertex_type"], _predicates(a.get("predicates")), a.get("field_name"),
        )

    def _edition_step(self, a: dict) -> ToolResult:
        return self.tools.edition_step(
            a["games_id"], a.get("direction") or "previous", int(a.get("steps") or 1),
        )

    def _medal_lookup(self, a: dict) -> ToolResult:
        return self.tools.medal_lookup(a["event_id"], a.get("medal") or "gold")


def _predicates(raw: Any) -> list[tuple[str, str, Any]]:
    """Normalise the model's predicate list into the backend's triples.

    Models emit this shape three different ways -- nested arrays, objects with
    named keys, or a JSON string -- and all three are accepted here rather than
    rejected, because a plan that was right in substance should not fail on
    punctuation.
    """
    if not raw:
        return []
    if isinstance(raw, str):
        raw = json.loads(raw)
    out = []
    for item in raw:
        if isinstance(item, dict):
            out.append((item.get("field"), item.get("operator") or item.get("op"),
                        item.get("value")))
        else:
            field, op, value = list(item)[:3]
            out.append((field, op, value))
    return [(f, o, v) for f, o, v in out if f and o]
