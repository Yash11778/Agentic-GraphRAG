"""TigerGraph implementation of the graph backend interface.

Same five reads as the local backend, same argument shapes, same return shapes.
Everything specific to TigerGraph -- the predicate encoding, the absence
sentinels, the folded comparison attributes, reverse edges -- is translated here
and is invisible to the tools above. That is what lets `eval2/validate_backends.py`
assert the two backends are interchangeable, which is the evidence behind
LOCKED-3: if the pipelines share a tool layer, a benchmark difference between them
can only come from control flow.

Reads go through the five installed GSQL queries in `queries.gsql`, so filtering,
counting and argmax execute in the database rather than in Python or in the model
(LOCKED-2).
"""
from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pyTigerGraph as tg

from agentic.config import TigerGraphSettings
from agentic.graph_schema import (
    ABSENT_INT,
    AGGREGATE_ID_CAP,
    EDGE_ATTRS,
    FOLD_SUFFIX,
    INT_FIELDS,
    REVERSE_PREFIX,
    coerce_value,
    fold,
    folded_field,
    reverse_edge,
)

# Predicate operators, split by the type of the attribute they compare. The op
# name sent to GSQL carries the type because a GSQL query cannot inspect an
# attribute's type at run time; see the header of queries.gsql.
_STRING_OPS = {"eq": "s_eq", "ne": "s_ne", "contains": "s_contains"}
_INT_OPS = {"eq": "i_eq", "ne": "i_ne", "gt": "i_gt", "gte": "i_gte",
            "lt": "i_lt", "lte": "i_lte"}


class UnsupportedPredicate(NotImplementedError):
    """Raised for a predicate this backend cannot express in GSQL.

    Raised rather than approximated on purpose. A predicate that silently
    evaluated differently here than in the local backend would make every number
    in the benchmark suspect, and the failure would be invisible.
    """


class TigerGraphBackend:
    """Graph reads backed by Savanna.

    The connection is created once and reused; each read is one installed-query
    call, so the cost of a tool call is one network round trip.
    """

    def __init__(self, settings: TigerGraphSettings | None = None,
                 conn: tg.TigerGraphConnection | None = None):
        self.settings = settings or TigerGraphSettings.from_env()
        if conn is None:
            conn = tg.TigerGraphConnection(
                host=self.settings.host,
                graphname=self.settings.graph,
                gsqlSecret=self.settings.secret,
            )
            conn.getToken(self.settings.secret)
        self.conn = conn

    # ── reads ────────────────────────────────────────────────────────────────
    def get(self, vtype: str, vid: str) -> dict | None:
        rows = self._vertices(
            self.conn.runInstalledQuery("vertex_get", {"vtype": vtype, "vid": vid})
        )
        return rows[0] if rows else None

    def find(self, vtype: str, predicates: list[tuple[str, str, Any]],
             max_results: int = 0) -> list[dict]:
        fields, ops, vals = self._encode(vtype, predicates)
        result = self.conn.runInstalledQuery("vertex_filter", {
            "vtype": vtype, "fields": fields, "ops": ops, "vals": vals,
            "max_results": max_results,
        })
        return self._vertices(result)

    def aggregate(self, vtype: str, predicates: list[tuple[str, str, Any]],
                  field: str | None = None) -> dict[str, Any]:
        """Cardinality and numeric reductions, computed in the database."""
        fields, ops, vals = self._encode(vtype, predicates)
        result = self.conn.runInstalledQuery("vertex_aggregate", {
            "vtype": vtype, "fields": fields, "ops": ops, "vals": vals,
            "agg_op": "all", "agg_field": field or "",
        })
        flat: dict[str, Any] = {}
        for block in result:
            flat.update(block)
        matched = flat.get("matched", 0)
        # GSQL's Max/MinAccum return their identity element when nothing was
        # accumulated. Reporting that as a real maximum would invent a value the
        # corpus never stated, so an empty reduction is None, as it is locally.
        empty = matched == 0 or not field
        return {
            "matched": matched,
            "matched_ids": sorted(flat.get("matched_ids") or [])[:AGGREGATE_ID_CAP],
            "sum": None if empty else flat.get("agg_sum"),
            "max": None if empty or flat.get("agg_max") == ABSENT_INT else flat.get("agg_max"),
            "min": None if empty else flat.get("agg_min"),
        }

    def neighbors(self, vtype: str, vid: str, edge_types: Iterable[str] | None = None,
                  direction: str = "out") -> list[dict]:
        """Adjacent vertices, each with the edge that reached it.

        An "in" traversal is a walk along the declared reverse edge, so both
        directions are one query; the reverse name is stripped again on the way
        out so callers only ever see the schema's own edge types.
        """
        wanted = list(edge_types) if edge_types else []
        if wanted:
            names = []
            if direction in ("out", "both"):
                names += wanted
            if direction in ("in", "both"):
                names += [reverse_edge(t) for t in wanted]
        else:
            names = []

        result = self.conn.runInstalledQuery("vertex_neighbors", {
            "vtype": vtype, "vid": vid, "edge_types": names,
        })
        vertices = {v["id"]: v for v in self._vertices(result)}
        rows = next((b["edges"] for b in result if "edges" in b), [])

        out: list[dict] = []
        for row in rows:
            raw_type = row["edge_type"]
            incoming = raw_type.startswith(REVERSE_PREFIX)
            etype = raw_type[len(REVERSE_PREFIX):] if incoming else raw_type
            if not names and direction != "both":
                if (direction == "in") != incoming:
                    continue
            neighbour = vertices.get(row["to_id"])
            if neighbour is None:
                continue
            edge = {
                "type": etype,
                "from": row["to_id"] if incoming else row["from_id"],
                "to": row["from_id"] if incoming else row["to_id"],
                "from_type": neighbour["type"] if incoming else vtype,
                "to_type": vtype if incoming else neighbour["type"],
            }
            for attr in EDGE_ATTRS.get(etype, ()):
                value = row.get(attr)
                edge[attr] = None if value == ABSENT_INT else value
            out.append({"vertex": neighbour, "edge": edge,
                        "direction": "in" if incoming else "out"})
        return out

    def vocabulary(self, vtype: str, field: str) -> list[str]:
        result = self.conn.runInstalledQuery("vertex_vocabulary",
                                             {"vtype": vtype, "field": field})
        values = next((b["vocab"] for b in result if "vocab" in b), [])
        return sorted(str(v) for v in values if v)

    def count(self, vtype: str) -> int:
        return self.conn.getVertexCount(vtype)

    def vector_search(self, query_vector: list[float], k: int) -> list[dict]:
        """Top-k chunks from the TigerGraph vector index.

        The index returns its hits unordered plus a distance map, so the ranking
        is rebuilt here. Cosine distance is turned into a similarity score so
        both backends report the same number in the same direction.
        """
        result = self.conn.runInstalledQuery("vector_search",
                                             {"qv": list(query_vector), "k": k})
        distances: dict[str, float] = {}
        chunks: dict[str, dict] = {}
        for block in result:
            if "distances" in block:
                distances = block["distances"]
            if "hits" in block:
                for item in block["hits"]:
                    chunks[item["v_id"]] = {
                        "id": item["v_id"],
                        "doc_id": item["attributes"]["doc_id"],
                        "ordinal": item["attributes"]["ordinal"],
                        "text": item["attributes"]["text"],
                    }
        hits = [{"chunk": chunk, "score": 1.0 - float(distances.get(cid, 1.0))}
                for cid, chunk in chunks.items()]
        hits.sort(key=lambda h: -h["score"])
        return hits[:k]

    # ── translation ──────────────────────────────────────────────────────────
    def _encode(self, vtype: str,
                predicates: list[tuple[str, str, Any]]) -> tuple[list, list, list]:
        """(field, op, value) triples → the three parallel lists GSQL takes."""
        fields, ops, vals = [], [], []
        for field, op, value in predicates:
            value = coerce_value(field, value)
            if field in INT_FIELDS:
                gsql_op, gsql_val = self._encode_int(field, op, value)
                fields.append(field)
                ops.append(gsql_op)
                vals.append(gsql_val)
                continue
            column = folded_field(vtype, field) or field
            if op == "contains":
                # One LIKE per token, applied in sequence, which is AND. This is
                # the same reading the local backend gives `contains`: every
                # token of the value occurs somewhere in the stored string.
                for token in fold(value).split() or [""]:
                    fields.append(column)
                    ops.append("s_contains")
                    vals.append(token)
                continue
            gsql_op, gsql_val = self._encode_string(vtype, field, op, value)
            fields.append(column)
            ops.append(gsql_op)
            vals.append(gsql_val)
        return fields, ops, vals

    @staticmethod
    def _encode_int(field: str, op: str, value: Any) -> tuple[str, str]:
        if op in _INT_OPS:
            return _INT_OPS[op], str(int(value))
        if op == "exists":
            if not value:
                # An absent INT is the sentinel, and every integer comparison in
                # queries.gsql excludes it, so "is absent" has no expressible form.
                raise UnsupportedPredicate(
                    f"exists=False on integer field {field!r} cannot be expressed in GSQL"
                )
            return "i_ne", str(ABSENT_INT)
        raise UnsupportedPredicate(f"operator {op!r} on integer field {field!r}")

    @staticmethod
    def _encode_string(vtype: str, field: str, op: str, value: Any) -> tuple[str, str]:
        if op in _STRING_OPS:
            return _STRING_OPS[op], fold(value)
        if op == "exists":
            return ("s_exists", "") if value else ("s_absent", "")
        raise UnsupportedPredicate(f"operator {op!r} on string field {field!r}")

    @staticmethod
    def _vertices(result: list) -> list[dict]:
        """Query output → the same flat dicts the local backend returns.

        Three things are undone here: the folded companion attributes are
        dropped, the absence sentinels become None, and the vertex type and id
        are lifted out of TigerGraph's envelope.
        """
        rows: list[dict] = []
        for block in result:
            if not isinstance(block, dict):
                continue
            for key, value in block.items():
                if key == "edges" or not isinstance(value, list):
                    continue
                for item in value:
                    if not (isinstance(item, dict) and "v_type" in item):
                        continue
                    vertex = {"type": item["v_type"], "id": item["v_id"]}
                    for attr, attr_value in item["attributes"].items():
                        if attr.endswith(FOLD_SUFFIX) or attr == "id":
                            continue
                        if attr in INT_FIELDS:
                            vertex[attr] = None if attr_value == ABSENT_INT else attr_value
                        else:
                            vertex[attr] = attr_value if attr_value != "" else None
                    rows.append(vertex)
        return rows
