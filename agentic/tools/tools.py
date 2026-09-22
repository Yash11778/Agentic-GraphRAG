"""The eight retrieval primitives, shared by all three pipelines (LOCKED-3).

Every tool returns the same `ToolResult`: data, the evidence it rests on, what it
cost in context tokens, how long it took, and a status. That uniformity is what
makes the benchmark measurable -- citations, token accounting and trace steps all
fall out of the return type instead of being reconstructed per pipeline.

None of these knows what a question type is (LOCKED-1). `graph_filter` takes
predicates and `graph_aggregate` takes a reduction; deciding which predicates
answer a given question is the orchestrator's job, and that decision is the
agentic behaviour being scored.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agentic.config import CORPUS
from agentic.graph_schema import VERTEX_FIELDS, fold, unknown_fields
from agentic.llm import count_tokens
from agentic.tools.backend import GraphBackend, get_backend

# Entity types whose names a question can plausibly mention, in the order a tie
# is broken: a span that matches both a sport and an athlete is far more likely
# to be the sport, which is the smaller and more distinctive vocabulary.
LINKABLE = (
    ("Sport", "name"),
    ("Venue", "name"),
    ("Games", "name"),
    ("Nation", "code"),
    ("Athlete", "name"),
)

MEDAL_EDGES = {"gold": "WON_GOLD", "silver": "WON_SILVER", "bronze": "WON_BRONZE"}


@dataclass
class Evidence:
    """One retrieved fact, traceable to the document it came from.

    `doc_id` is the unit the benchmark scores grounding against, so every tool
    that can name one does, including the graph tools: an Event vertex id *is* its
    document id, which is what lets a pure graph answer be checked against
    `gold_doc_ids` exactly as a retrieved chunk is.
    """

    doc_id: str
    snippet: str
    source_tool: str
    score: float | None = None

    def as_dict(self) -> dict:
        row = {"doc_id": self.doc_id, "snippet": self.snippet,
               "source_tool": self.source_tool}
        if self.score is not None:
            row["score"] = round(self.score, 4)
        return row


@dataclass
class ToolResult:
    data: Any
    evidence: list[Evidence] = field(default_factory=list)
    tokens: int = 0
    latency_ms: float = 0.0
    status: str = "ok"
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def doc_ids(self) -> list[str]:
        seen, out = set(), []
        for item in self.evidence:
            if item.doc_id and item.doc_id not in seen:
                seen.add(item.doc_id)
                out.append(item.doc_id)
        return out

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "data": self.data,
            "evidence": [e.as_dict() for e in self.evidence],
            "tokens": self.tokens,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
        }


def _timed(fn: Callable[[], tuple]) -> ToolResult:
    """Run a tool body, timing it and accounting for its context cost.

    The body returns `(data, evidence)` and may add a third element to state
    the status itself, for the case where the data is a real value that still
    means "nothing was found" -- a count of zero.

    An exception becomes `status="error"` rather than propagating: a tool failing
    is information the agent should act on -- try another retrieval path -- not a
    reason to abandon the question and lose the trace (LOCKED-6 still applies, in
    that the failure is recorded and never papered over).
    """
    started = time.perf_counter()
    try:
        data, evidence, *rest = fn()
        status = rest[0] if rest else ("ok" if data not in (None, [], {}) else "empty")
        result = ToolResult(data=data, evidence=evidence, status=status)
    except Exception as exc:
        result = ToolResult(data=None, status="error",
                            error=f"{type(exc).__name__}: {exc}")
    result.latency_ms = (time.perf_counter() - started) * 1000
    result.tokens = count_tokens(json.dumps(result.data, default=str))
    return result


class Tools:
    """The tool layer over one graph backend.

    Constructed once per run. The corpus text is memory-mapped lazily because
    only `doc_fetch` needs it, and a graph-only investigation should not pay to
    read 5.5M tokens off disk.
    """

    def __init__(self, backend: GraphBackend | None = None):
        self.backend = backend or get_backend()
        # Vertices already returned by a tool this session. A graph filter hands
        # back whole rows, and the verbatim check then asked the backend for the
        # same rows one by one -- up to twenty round trips of ~270 ms each on
        # Savanna, for data the run already held.
        self._vertices: dict[tuple[str, str], dict] = {}

    def vertex(self, vtype: str, vid: str) -> dict | None:
        """One vertex, from what a tool already returned before the backend."""
        key = (vtype, vid)
        if key not in self._vertices:
            self._vertices[key] = self.backend.get(vtype, vid)
        return self._vertices[key]

    def _remember(self, rows: Iterable[dict]) -> None:
        for row in rows:
            if isinstance(row, dict) and row.get("type") and row.get("id"):
                self._vertices[(row["type"], row["id"])] = row

    @cached_property
    def documents(self) -> dict[str, dict]:
        docs = {}
        with CORPUS.open(encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                docs[doc["doc_id"]] = doc
        return docs

    @cached_property
    def vocabularies(self) -> dict[str, dict[str, str]]:
        """Folded name → vertex id, per linkable vertex type.

        Built once from the graph itself, so the linker can only ever propose
        entities that exist. An entity linker that invents a plausible venue name
        is worse than one that returns nothing.
        """
        vocab: dict[str, dict[str, str]] = {}
        for vtype, field_name in LINKABLE:
            table: dict[str, str] = {}
            for vertex in self.backend.find(vtype, []):
                name = vertex.get(field_name)
                if name:
                    table[fold(name)] = vertex["id"]
            vocab[vtype] = table
        return vocab

    @cached_property
    def _phrase_index(self) -> tuple[dict[str, list[tuple[str, str, str]]], int]:
        """Word-sequence → (type, id, name) candidates, and the longest sequence.

        The linker used to test every vocabulary entry against the question with
        its own regular expression -- some 7,500 patterns, more than the regex
        cache holds, so each call recompiled them all and took ~300ms. Looking
        the question's word n-grams up in a dictionary is a few hundred hashes.
        """
        index: dict[str, list[tuple[str, str, str]]] = {}
        longest = 1
        for vtype, table in self.vocabularies.items():
            for name, vid in table.items():
                key = " ".join(_words(name))
                if len(key) < 3:
                    continue
                index.setdefault(key, []).append((vtype, vid, name))
                longest = max(longest, key.count(" ") + 1)
        return index, min(longest, 12)

    # ── retrieval ────────────────────────────────────────────────────────────
    def vector_search(self, query: str, k: int = 10) -> ToolResult:
        """Top-k corpus chunks by embedding similarity."""
        def run():
            from agentic.tools.embedder import embed_query

            hits = self.backend.vector_search(embed_query(query), k)
            evidence = [
                Evidence(doc_id=h["chunk"]["doc_id"], snippet=h["chunk"]["text"],
                         source_tool="vector_search", score=h["score"])
                for h in hits
            ]
            data = [{"doc_id": h["chunk"]["doc_id"], "chunk_id": h["chunk"]["id"],
                     "score": round(h["score"], 4), "text": h["chunk"]["text"]}
                    for h in hits]
            return data, evidence
        return _timed(run)

    def doc_fetch(self, doc_ids: Iterable[str], max_chars: int = 4000) -> ToolResult:
        """Full documents by id, truncated to a stated length.

        Truncation is reported in the data rather than done silently, because a
        pipeline that answers from a cut-off document should be visibly doing so.
        """
        def run():
            data, evidence = [], []
            for doc_id in doc_ids:
                doc = self.documents.get(doc_id)
                if doc is None:
                    continue
                text = doc["text"]
                data.append({"doc_id": doc_id, "title": doc["title"],
                             "url": doc.get("url", ""),
                             "text": text[:max_chars],
                             "truncated": len(text) > max_chars})
                evidence.append(Evidence(doc_id=doc_id, snippet=doc["title"],
                                         source_tool="doc_fetch"))
            return data, evidence
        return _timed(run)

    def entity_link(self, text: str, limit: int = 8) -> ToolResult:
        """Graph vertices whose names appear in the text, with a confidence.

        Longest match wins and confidence is the share of the question the match
        covers, so "Eton Dorney" beats a stray match on "Eton" and a one-word hit
        inside a long question is correctly reported as weak.

        Matching is on whole words. A bare substring test links the NOC code VEN
        out of the word "event", and a linker that hallucinates Venezuela into
        every question is worse than one that links nothing.
        """
        def run():
            index, longest = self._phrase_index
            words = _words(text)
            folded_len = max(len(fold(text)), 1)
            rank = {vtype: i for i, (vtype, _) in enumerate(LINKABLE)}
            hits: list[dict] = []
            seen: set[tuple[str, str]] = set()
            for start in range(len(words)):
                for size in range(1, min(longest, len(words) - start) + 1):
                    key = " ".join(words[start:start + size])
                    for vtype, vid, name in index.get(key, ()):
                        if (vtype, vid) in seen:
                            continue
                        seen.add((vtype, vid))
                        hits.append({"type": vtype, "id": vid, "name": name,
                                     "confidence": round(len(name) / folded_len, 3)})
            # Longest match first; a tie goes to the more distinctive type.
            hits.sort(key=lambda h: (-len(h["name"]), rank[h["type"]]))
            # Drop a match wholly contained in a longer one already taken:
            # "canoeing" inside "canoe sprint" is not a second entity.
            kept: list[dict] = []
            for hit in hits:
                if not any(hit["name"] in k["name"] for k in kept):
                    kept.append(hit)
            return kept[:limit], []
        return _timed(run)

    # ── graph ────────────────────────────────────────────────────────────────
    def graph_neighbors(self, vertex_type: str, vertex_id: str,
                        edge_types: Iterable[str] | None = None,
                        direction: str = "out") -> ToolResult:
        def run():
            rows = self.backend.neighbors(vertex_type, vertex_id, edge_types, direction)
            self._remember(r["vertex"] for r in rows)
            data = [{"edge": r["edge"]["type"], "direction": r["direction"],
                     "vertex": r["vertex"]} for r in rows]
            evidence = [
                Evidence(doc_id=_doc_id_of(r["vertex"]),
                         snippet=f"{vertex_id} -{r['edge']['type']}- {r['vertex']['id']}",
                         source_tool="graph_neighbors")
                for r in rows if _doc_id_of(r["vertex"])
            ]
            return data, evidence
        return _timed(run)

    def graph_filter(self, vertex_type: str, predicates: list[tuple[str, str, Any]],
                     max_results: int = 0) -> ToolResult:
        """Vertices satisfying every predicate. The workhorse of enumeration."""
        def run():
            _check_fields(vertex_type, predicates)
            rows = self.backend.find(vertex_type, predicates)
            self._remember(rows)
            if max_results:
                rows = rows[:max_results]
            evidence = [
                Evidence(doc_id=_doc_id_of(r), snippet=r.get("title") or r.get("name") or r["id"],
                         source_tool="graph_filter")
                for r in rows if _doc_id_of(r)
            ]
            return rows, evidence
        return _timed(run)

    def graph_aggregate(self, vertex_type: str, predicates: list[tuple[str, str, Any]],
                        field_name: str | None = None) -> ToolResult:
        """Cardinality and numeric reductions, computed in the database.

        This is the tool that keeps arithmetic out of the model (LOCKED-2), and
        `matched` is the expected cardinality the agent's coverage gate compares
        its evidence against before it is allowed to answer (LOCKED-5).
        """
        def run():
            _check_fields(vertex_type, predicates)
            if field_name:
                _check_fields(vertex_type, [(field_name, "exists", True)])
            data = self.backend.aggregate(vertex_type, predicates, field_name)
            ids = data.get("matched_ids") or []
            # The rows the database counted are the evidence for the count. Without
            # them an aggregate answer is correct but uncitable, and scores zero on
            # grounding for a reason that has nothing to do with its quality.
            evidence = [Evidence(doc_id=vid, snippet=f"counted by {vertex_type} aggregate",
                                 source_tool="graph_aggregate")
                        for vid in ids if vid.startswith("Q")]
            # The model is shown the count, not the list: it never needs to read
            # 100 ids to report a number.
            visible = {k: v for k, v in data.items() if k != "matched_ids"}
            # A count of zero is a real number and an empty result at once. It is
            # reported as empty so the planner gets the same "loosen the filter"
            # cue a listing gives, instead of reading 0 as the answer.
            status = "empty" if not visible.get("matched") else "ok"
            return visible, evidence, status
        return _timed(run)

    def edition_step(self, games_id: str, direction: str = "previous",
                     steps: int = 1) -> ToolResult:
        """Walk the Games edition chain, e.g. "the Olympics immediately before 2016".

        Answered from the corpus's own prev/next links rather than from the
        model's memory of Olympic history, which is the point: the README is
        explicit that the corpus, not the world, is ground truth.
        """
        def run():
            current, path = games_id, []
            for _ in range(max(steps, 1)):
                rows = self.backend.neighbors(
                    "Games", current, ["PRECEDED_BY"],
                    "out" if direction == "previous" else "in",
                )
                if not rows:
                    return None, []
                current = rows[0]["vertex"]["id"]
                path.append(current)
            vertex = self.backend.get("Games", current)
            return {"games": vertex, "path": path}, []
        return _timed(run)

    def medal_lookup(self, event_id: str, medal: str = "gold") -> ToolResult:
        """The athletes on one medal edge of one event, with their nation."""
        def run():
            edge = MEDAL_EDGES.get(medal.lower())
            if edge is None:
                raise ValueError(f"unknown medal {medal!r}; use gold, silver or bronze")
            rows = self.backend.neighbors("Event", event_id, [edge], "out")
            athletes = []
            for row in rows:
                athlete = row["vertex"]
                athletes.append({"id": athlete["id"], "name": athlete.get("name"),
                                 "noc": row["edge"].get("noc")})
            event = self.vertex("Event", event_id)
            data = {"event_id": event_id, "medal": medal, "athletes": athletes,
                    "raw": (event or {}).get(f"{medal.lower()}_raw")}
            evidence = [Evidence(doc_id=event_id,
                                 snippet=f"{medal} medal of {(event or {}).get('title', event_id)}",
                                 source_tool="medal_lookup")] if event else []
            return data, evidence
        return _timed(run)


_WORD = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")


def _words(text: str) -> list[str]:
    """The folded words of a string, punctuation between them dropped.

    Both the vocabulary and the question go through this, so "Eton Dorney,
    Buckinghamshire" and "eton dorney buckinghamshire" produce one sequence.
    """
    return _WORD.findall(fold(text))


def _check_fields(vertex_type: str, predicates: Iterable[tuple[str, str, Any]]) -> None:
    """Reject a predicate over a field the vertex type does not carry.

    Raised, so it reaches the planner as `status="error"` with the real field
    list, which it can act on in one step. Left unchecked it was an empty
    result, and the planner spent its remaining steps loosening a filter that
    could never match anything.
    """
    if vertex_type not in VERTEX_FIELDS:
        raise ValueError(f"unknown vertex type {vertex_type!r}; "
                         f"use one of {', '.join(VERTEX_FIELDS)}")
    bad = unknown_fields(vertex_type, [p[0] for p in predicates])
    if bad:
        raise ValueError(f"{vertex_type} has no field {', '.join(map(repr, bad))}; "
                         f"its fields are {', '.join(VERTEX_FIELDS[vertex_type])}")


def _doc_id_of(vertex: dict) -> str:
    """The document a vertex is evidence from, if it is one.

    Event and Document vertices are keyed by their corpus document id; Athlete,
    Venue, Sport and the rest are derived entities with no document of their own,
    and claiming one for them would corrupt the grounding score.
    """
    return vertex["id"] if vertex.get("type") in ("Event", "Document") else ""
