"""Apply the GSQL schema and bulk-load data/graph/*.jsonl into TigerGraph.

Run after build_graph.py, once per workspace:

    python agentic/ingest/load_tigergraph.py            # schema if missing, then load
    python agentic/ingest/load_tigergraph.py --recreate # drop the graph first

Loading goes through the REST upsert endpoint in batches rather than a GSQL
loading job over staged CSV. The corpus is small (9,498 vertices, 22,219 edges),
the upsert path needs no file staging on the cloud side, and it reads the exact
same JSONL the local backend reads -- so the two backends cannot diverge because
of a transformation that happened in only one of them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pyTigerGraph as tg

from agentic.config import GRAPH_DIR, TigerGraphSettings
from agentic.graph_schema import (
    ABSENT_INT,
    ABSENT_STR,
    EMBEDDING_DIM,
    FOLD_FIELDS,
    FOLD_SUFFIX,
    INT_FIELDS,
    fold,
)

SCHEMA_FILE = Path(__file__).with_name("schema.gsql")
QUERY_FILE = ROOT / "agentic/tools/queries.gsql"
INSTALLED_QUERIES = (
    "vertex_get", "vertex_filter", "vertex_aggregate",
    "vertex_neighbors", "vertex_vocabulary",
)

BATCH = 1000


def connect(settings: TigerGraphSettings | None = None) -> tg.TigerGraphConnection:
    """An authenticated connection bound to the configured graph."""
    s = settings or TigerGraphSettings.from_env()
    conn = tg.TigerGraphConnection(host=s.host, graphname=s.graph, gsqlSecret=s.secret)
    conn.getToken(s.secret)
    return conn


def graph_exists(conn: tg.TigerGraphConnection, graph: str) -> bool:
    """Whether the graph is listed globally. Must be asked on a connection with
    no graph bound, since LS scoped to a graph describes that graph instead."""
    return f"Graph {graph}(" in conn.gsql("LS")


def has_schema(conn: tg.TigerGraphConnection) -> bool:
    """Whether the schema job has already run against the bound graph."""
    try:
        types = {v["Name"] for v in conn.getSchema(force=True).get("VertexTypes", [])}
    except Exception:
        return False
    return {"Event", "Athlete", "Games"} <= types


def apply_schema(conn: tg.TigerGraphConnection, graph: str) -> None:
    """Run the schema job against an existing, already-bound graph.

    Each schema change on Savanna takes roughly half a minute, so the schema is
    one job, and it is skipped outright when the types already exist -- otherwise
    a rerun fails on "vertex name is used by another object" rather than being a
    no-op, which is a confusing way to say "already done".
    """
    if has_schema(conn):
        print("schema already present, skipping")
        return
    gsql = (SCHEMA_FILE.read_text(encoding="utf-8")
            .replace("{graph}", graph)
            .replace("{dim}", str(EMBEDDING_DIM)))
    print("applying schema (this takes ~40s) ...")
    out = conn.gsql(gsql)
    if "succeeded" not in out and "completes" not in out:
        raise RuntimeError(f"schema change did not succeed:\n{out}")
    print(out.strip().splitlines()[-1])


def install_queries(conn: tg.TigerGraphConnection, graph: str) -> None:
    """Create and install the generic GSQL primitives the tool layer calls.

    Kept in the loader so that one command leaves the workspace fully usable.
    Installation compiles each query and costs a couple of minutes, so it is
    skipped when all five are already installed.
    """
    listing = conn.gsql(f"USE GRAPH {graph}\nLS")
    if all(f"{name}(" in listing and "installed" in listing for name in INSTALLED_QUERIES):
        print("queries already installed, skipping")
        return
    print("installing GSQL queries (this takes ~2 min) ...")
    out = conn.gsql(QUERY_FILE.read_text(encoding="utf-8").replace("{graph}", graph))
    if "failed: 0" not in out:
        raise RuntimeError(f"query installation did not succeed:\n{out[-2000:]}")
    print("  " + [l for l in out.splitlines() if "installation summary" in l][-1].strip())


def clean_attrs(record: dict, drop: set[str], vtype: str = "") -> dict:
    """JSONL record → upsert attributes, with absence explicit and folds precomputed.

    Two transformations, both required for the two backends to agree:
    absent values become the sentinels the backend maps back to None, and every
    field listed in FOLD_FIELDS gets a `_fold` companion carrying the same
    comparison key the local backend computes at query time.
    """
    attrs: dict[str, object] = {}
    for key, value in record.items():
        if key in drop:
            continue
        if value is None:
            attrs[key] = ABSENT_INT if key in INT_FIELDS else ABSENT_STR
        else:
            attrs[key] = value
    for field in FOLD_FIELDS.get(vtype, ()):
        attrs[field + FOLD_SUFFIX] = fold(record.get(field))
    return attrs


def load_vertices(conn: tg.TigerGraphConnection) -> dict[str, int]:
    by_type: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    with (GRAPH_DIR / "vertices.jsonl").open(encoding="utf-8") as f:
        for line in f:
            v = json.loads(line)
            by_type[v["type"]].append(
                (v["id"], clean_attrs(v, {"type", "id"}, v["type"]))
            )

    counts: dict[str, int] = {}
    for vtype, rows in by_type.items():
        total = 0
        for i in range(0, len(rows), BATCH):
            total += conn.upsertVertices(vtype, rows[i:i + BATCH])
        counts[vtype] = total
        print(f"  {vtype:<10} {total:>6}")
    return counts


def load_edges(conn: tg.TigerGraphConnection) -> dict[str, int]:
    # Keyed by the full (source type, edge type, target type) triple because
    # PRECEDED_BY is declared over two different endpoint pairs.
    by_type: dict[tuple[str, str, str], list[tuple[str, str, dict]]] = defaultdict(list)
    with (GRAPH_DIR / "edges.jsonl").open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            key = (e["from_type"], e["type"], e["to_type"])
            attrs = clean_attrs(e, {"type", "from", "to", "from_type", "to_type"})
            by_type[key].append((e["from"], e["to"], attrs))

    counts: dict[str, int] = {}
    for (src, etype, dst), rows in by_type.items():
        total = 0
        for i in range(0, len(rows), BATCH):
            total += conn.upsertEdges(src, etype, dst, rows[i:i + BATCH])
        counts[f"{src}-{etype}->{dst}"] = total
        print(f"  {etype:<13} {src}->{dst:<8} {total:>6}")
    return counts


def verify(conn: tg.TigerGraphConnection, timeout_s: float = 180.0) -> None:
    """Compare loaded counts against the JSONL, and fail loudly on a mismatch.

    A partially loaded graph is the worst failure mode available here: every
    aggregation answer would be quietly, plausibly wrong, and nothing downstream
    would notice.

    The counts are polled rather than read once. An upsert returns as soon as the
    write is accepted, so vertex counts trail it by tens of seconds -- reading
    immediately after loading reports a deficit that resolves itself, which would
    make this check cry wolf on every run.
    """
    expected: dict[str, int] = defaultdict(int)
    with (GRAPH_DIR / "vertices.jsonl").open(encoding="utf-8") as f:
        for line in f:
            expected[json.loads(line)["type"]] += 1

    print("\nverification (waiting for writes to settle):")
    deadline = time.time() + timeout_s
    counts: dict[str, int] = {}
    while True:
        counts = {vtype: conn.getVertexCount(vtype) for vtype in expected}
        if counts == expected or time.time() > deadline:
            break
        time.sleep(10)

    bad = [(t, expected[t], counts[t]) for t in sorted(expected) if counts[t] != expected[t]]
    for vtype in sorted(expected):
        flag = "ok" if counts[vtype] == expected[vtype] else "MISMATCH"
        print(f"  {vtype:<10} expected {expected[vtype]:>6}  in graph {counts[vtype]:>6}  {flag}")
    if bad:
        raise SystemExit(f"load incomplete after {timeout_s:.0f}s: {bad}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recreate", action="store_true",
                    help="drop and recreate the graph before loading")
    args = ap.parse_args()

    settings = TigerGraphSettings.from_env()
    conn = tg.TigerGraphConnection(host=settings.host, gsqlSecret=settings.secret)
    conn.getToken(settings.secret)

    if args.recreate:
        print(f"dropping graph {settings.graph} ...")
        # CASCADE because installed queries depend on the graph and block a
        # plain DROP.
        print(conn.gsql(f"DROP GRAPH {settings.graph} CASCADE").strip())
    if not graph_exists(conn, settings.graph):
        print(f"creating graph {settings.graph} ...")
        print(conn.gsql(f"CREATE GRAPH {settings.graph}()").strip())

    conn = connect(settings)
    apply_schema(conn, settings.graph)
    install_queries(conn, settings.graph)

    started = time.perf_counter()
    print("\nvertices:")
    load_vertices(conn)
    print("\nedges:")
    load_edges(conn)
    verify(conn)
    print(f"\nloaded in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
