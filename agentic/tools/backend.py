"""Graph backend interface + a local in-memory implementation.

Two backends implement one interface (docs/ARCHITECTURE.md §5): this local one,
which loads data/graph/*.jsonl and needs no network, and the TigerGraph one that
issues the equivalent GSQL. Every tool and both graph pipelines talk to the
interface only, so moving to TigerGraph changes a single constructor call and
cannot change results -- which is also what makes the local backend a valid
oracle for testing the agent while cloud credentials are unavailable.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agentic.graph_schema import (  # noqa: E402  (fold re-exported)
    AGGREGATE_ID_CAP,
    coerce_value,
    fold,
    is_number_token,
)

GRAPH_DIR = ROOT / "data/graph"

# Comparison operators a predicate may use. Deliberately small and total: the
# orchestrator composes these (LOCKED-1), so an unseen question phrasing still
# has to land on one of them rather than on bespoke per-question code.
OPS = {
    "eq":  lambda a, b: a == b,
    "ne":  lambda a, b: a != b,
    "gt":  lambda a, b: a is not None and a > b,
    "gte": lambda a, b: a is not None and a >= b,
    "lt":  lambda a, b: a is not None and a < b,
    "lte": lambda a, b: a is not None and a <= b,
    "in":  lambda a, b: a in b,
    # Every whitespace-separated token of the value must occur in the stored
    # string. A plain substring test is a special case of this (one token), and
    # the extra tolerance is for the corpus's own typography: it writes
    # "Xiaohaituo Bobsleigh and Luge TrackBeijing" and "Beijing Science and
    # TechnologyUniversity Gymnasium", and a question that spaces those
    # correctly must still reach them. A token that is a number matches only as
    # a whole number (see graph_schema.is_number_token). The TigerGraph backend
    # expands one contains predicate into one LIKE, or one whole-number scan,
    # per token, so both read this the same way.
    "contains": lambda a, b: a is not None and _tokens_in(str(b), str(a)),
    "exists":   lambda a, b: (a is not None) == bool(b),
}


def _tokens_in(needle: str, haystack: str) -> bool:
    haystack = haystack.lower()
    for tok in needle.lower().split():
        if is_number_token(tok):
            if not re.search(rf"(?<![0-9]){re.escape(tok)}(?![0-9])", haystack):
                return False
        elif tok not in haystack:
            return False
    return True


class GraphBackend(Protocol):
    def get(self, vtype: str, vid: str) -> dict | None: ...
    def find(self, vtype: str, predicates: list[tuple[str, str, Any]]) -> list[dict]: ...
    def neighbors(self, vtype: str, vid: str, edge_types: Iterable[str] | None,
                  direction: str) -> list[dict]: ...
    def vocabulary(self, vtype: str, field: str) -> list[str]: ...
    def aggregate(self, vtype: str, predicates: list[tuple[str, str, Any]],
                  field: str | None = None) -> dict[str, Any]: ...
    def count(self, vtype: str) -> int: ...
    def vector_search(self, query_vector: list[float], k: int) -> list[dict]: ...


class LocalGraphBackend:
    """In-memory backend over the JSONL the ingestion step writes."""

    def __init__(self, graph_dir: Path = GRAPH_DIR):
        # Vectors load on first use: most tool calls never touch them, and
        # reading 30MB of embeddings to answer a graph query would be waste.
        self._chunks: list[dict] | None = None
        self._vectors = None      # unit-normalised, so a query is one dot product

        self.vertices: dict[str, dict[str, dict]] = defaultdict(dict)
        self.out: dict[tuple[str, str], list[dict]] = defaultdict(list)
        self.inc: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for line in (graph_dir / "vertices.jsonl").open(encoding="utf-8"):
            v = json.loads(line)
            self.vertices[v["type"]][v["id"]] = v
        for line in (graph_dir / "edges.jsonl").open(encoding="utf-8"):
            e = json.loads(line)
            self.out[(e["from_type"], e["from"])].append(e)
            self.inc[(e["to_type"], e["to"])].append(e)

    # ── reads ────────────────────────────────────────────────────────────────
    def get(self, vtype: str, vid: str) -> dict | None:
        return self.vertices.get(vtype, {}).get(vid)

    def find(self, vtype: str, predicates: list[tuple[str, str, Any]]) -> list[dict]:
        """All vertices of a type satisfying every (field, op, value) predicate.

        String comparisons are accent/case folded on both sides; numeric ones are
        not coerced -- a predicate comparing a number against a missing field
        yields False rather than raising, so a partially-populated corpus degrades
        into a smaller result set instead of an exception.
        """
        out = []
        for v in self.vertices.get(vtype, {}).values():
            if all(self._match(v, f, op, val) for f, op, val in predicates):
                out.append(v)
        return out

    @staticmethod
    def _match(v: dict, field: str, op: str, value: Any) -> bool:
        if op not in OPS:
            raise ValueError(f"unknown operator {op!r}; valid: {sorted(OPS)}")
        value = coerce_value(field, value)
        actual = v.get(field)
        if isinstance(value, str) and op in ("eq", "ne", "contains", "in"):
            actual_cmp = fold(actual)
            value_cmp = fold(value)
        else:
            actual_cmp, value_cmp = actual, value
        try:
            return OPS[op](actual_cmp, value_cmp)
        except TypeError:
            return False

    def neighbors(self, vtype: str, vid: str, edge_types=None, direction="out") -> list[dict]:
        """Adjacent vertices, each returned with the edge that reached it."""
        edges: list[dict] = []
        if direction in ("out", "both"):
            edges += self.out.get((vtype, vid), [])
        if direction in ("in", "both"):
            edges += self.inc.get((vtype, vid), [])
        result = []
        for e in edges:
            if edge_types and e["type"] not in edge_types:
                continue
            incoming = e["to_type"] == vtype and e["to"] == vid
            nt, nid = (e["from_type"], e["from"]) if incoming else (e["to_type"], e["to"])
            v = self.get(nt, nid)
            if v:
                result.append({"vertex": v, "edge": e,
                               "direction": "in" if incoming else "out"})
        return result

    def vocabulary(self, vtype: str, field: str) -> list[str]:
        """Distinct values of a field -- the schema card handed to the planner so
        it proposes predicate values that actually exist in the graph."""
        vals = {v.get(field) for v in self.vertices.get(vtype, {}).values()}
        return sorted(str(x) for x in vals if x)

    def count(self, vtype: str) -> int:
        return len(self.vertices.get(vtype, {}))

    def aggregate(self, vtype: str, predicates: list[tuple[str, str, Any]],
                  field: str | None = None) -> dict[str, Any]:
        """Reduce a filtered vertex set to scalars, without materialising it.

        Returns every reduction at once because the agent almost always wants two
        together: `matched` is the cardinality its coverage check compares against
        (LOCKED-5), and max/min answer a superlative directly.

        Rows whose aggregated field is absent are excluded from sum, max and min
        but counted in `matched` -- "how many events are there" and "of those,
        what is the largest stated value" are different questions.
        """
        rows = self.find(vtype, predicates)
        values = []
        if field:
            values = [r.get(field) for r in rows]
            values = [v for v in values if isinstance(v, (int, float))]
        return {
            "matched": len(rows),
            "matched_ids": sorted(r["id"] for r in rows)[:AGGREGATE_ID_CAP],
            "sum": sum(values) if values else None,
            "max": max(values) if values else None,
            "min": min(values) if values else None,
        }

    def vector_search(self, query_vector: list[float], k: int) -> list[dict]:
        """Top-k chunks by cosine similarity, exhaustively.

        A brute-force scan of 19,832 vectors is a few milliseconds of numpy and,
        unlike an approximate index, returns exactly the true top k. That makes
        this an oracle for the TigerGraph vector index rather than a second
        approximation of it.
        """
        import numpy as np

        if self._chunks is None:
            from agentic.ingest.build_vectors import read_local

            self._chunks, raw = read_local()
            # Normalise once at load rather than on every query: the row norms
            # of a 19,832 x 384 matrix were being recomputed per search, which
            # was most of the search.
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            self._vectors = raw / np.where(norms == 0, 1, norms)

        query = np.asarray(query_vector, dtype=np.float32)
        query = query / (np.linalg.norm(query) or 1.0)
        scores = self._vectors @ query
        k = min(k, len(scores))
        # Partial selection then a sort of k items, instead of sorting all rows.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [{"chunk": self._chunks[i], "score": float(scores[i])} for i in top]


def get_backend(name: str | None = None) -> GraphBackend:
    """The backend named by GRAPH_BACKEND, or by an explicit override.

    The one place a backend is chosen. Tools and pipelines call this instead of
    constructing a backend, so switching the whole system between the local
    oracle and Savanna is one environment variable and cannot be done by halves.
    """
    from agentic.config import graph_backend_name

    chosen = (name or graph_backend_name()).lower()
    if chosen == "local":
        return LocalGraphBackend()
    if chosen == "tigergraph":
        from agentic.tools.tigergraph_backend import TigerGraphBackend

        return TigerGraphBackend()
    raise ValueError(f"unknown backend {chosen!r}")
