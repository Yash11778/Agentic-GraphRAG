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
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Protocol

ROOT = Path(__file__).resolve().parents[2]
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
    "contains": lambda a, b: a is not None and str(b).lower() in str(a).lower(),
    "exists":   lambda a, b: (a is not None) == bool(b),
}


def fold(text: str) -> str:
    """Accent- and case-insensitive comparison key.

    Athlete and venue names carry diacritics that a question may or may not
    reproduce ("Kökény" vs "Kokeny"); folding both sides makes lookup robust
    without loosening it into fuzzy matching.
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"\s+", " ", s.lower()).strip()


class GraphBackend(Protocol):
    def get(self, vtype: str, vid: str) -> dict | None: ...
    def find(self, vtype: str, predicates: list[tuple[str, str, Any]]) -> list[dict]: ...
    def neighbors(self, vtype: str, vid: str, edge_types: Iterable[str] | None,
                  direction: str) -> list[dict]: ...
    def vocabulary(self, vtype: str, field: str) -> list[str]: ...


class LocalGraphBackend:
    """In-memory backend over the JSONL the ingestion step writes."""

    def __init__(self, graph_dir: Path = GRAPH_DIR):
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
