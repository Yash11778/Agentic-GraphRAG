"""
Install the LegalCase/CITES schema (legal_schema.gsql) and ingest opinions into
TigerGraph. Two modes:

  Pilot (default): only opinions that appear in citations.csv, i.e. participate
  in at least one real in-corpus citation -- 12,457 opinions, 9,632 edges,
  guaranteed no dangling references. Used for Phase 2 pilot testing.

  --full: ALL opinions in dataset_100m_enriched.jsonl (63,632), not just the
  citation-connected subset -- the other ~51,175 opinions have no CITES edges
  but are still useful as single-hop content and for Basic RAG parity. Edge
  count is unchanged (9,632) since citations.csv was already built against the
  full corpus, not just the pilot subset.

Usage:
    python scripts/ingest_pilot_graph.py --install-schema   # first run only
    python scripts/ingest_pilot_graph.py                    # pilot ingest
    python scripts/ingest_pilot_graph.py --full              # full-scale ingest
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent.resolve()
DATASET_FILE = ROOT / "data" / "raw" / "dataset_100m_enriched.jsonl"
CITATIONS_FILE = ROOT / "data" / "raw" / "citations.csv"
SCHEMA_FILE = Path(__file__).parent / "legal_schema.gsql"

TG_HOST = os.environ["TG_HOST"]
TG_SECRET = os.environ["TG_PASSWORD"]
GRAPH_NAME = os.environ.get("TG_GRAPH", "LegalGraph")

BATCH_SIZE = 200
# 40000 covers the full text of ~100% of opinions (corpus max ≈ 36.4k chars). The old
# 6000 cap head-truncated 35% of opinions at ingestion, so a holding past the cutoff
# could never be retrieved no matter how good retrieval-time selection was.
TEXT_TRUNCATE = 40000


def get_token() -> str:
    resp = requests.post(
        f"{TG_HOST}/gsql/v1/tokens",
        json={"secret": TG_SECRET, "graph": GRAPH_NAME},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["token"]


def install_schema():
    gsql = SCHEMA_FILE.read_text(encoding="utf-8")
    token = get_token()
    print(f"Installing legal schema + queries on {GRAPH_NAME} ...")
    resp = requests.post(
        f"{TG_HOST}/gsql/v1/statements",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "text/plain"},
        data=gsql.encode("utf-8"),
        timeout=300,
    )
    print("Status:", resp.status_code)
    print(resp.text[:3000])
    if resp.status_code not in (200, 201):
        print("ERROR: schema install failed")
        sys.exit(1)
    print("Schema installed successfully.")


def load_citation_graph_node_ids() -> set[str]:
    ids = set()
    with open(CITATIONS_FILE, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for citing, cited in r:
            ids.add(citing)
            ids.add(cited)
    return ids


def load_pilot_opinions(node_ids: set[str]) -> dict:
    opinions = {}
    with open(DATASET_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] in node_ids:
                opinions[row["id"]] = row
    return opinions


def load_all_opinions() -> dict:
    """Every opinion in the corpus, not just the citation-connected subset. The
    ~51k opinions with no CITES edge are still valuable single-hop content and give
    Basic RAG parity (same underlying corpus both pipelines draw from)."""
    opinions = {}
    with open(DATASET_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            opinions[row["id"]] = row
    return opinions


def upsert_vertices(token: str, opinions: dict):
    ids = list(opinions.keys())
    total = len(ids)
    for i in range(0, total, BATCH_SIZE):
        batch_ids = ids[i:i + BATCH_SIZE]
        vertices = {}
        for oid in batch_ids:
            row = opinions[oid]
            vertices[oid] = {
                "case_id":    {"value": oid},  # GSQL can't reference PRIMARY_ID directly in query bodies
                "title":      {"value": (row.get("title") or "")[:500]},
                "case_name":  {"value": (row.get("case_name") or "")[:500]},
                "court_id":   {"value": row.get("court_id") or ""},
                "court_name": {"value": row.get("court_name") or ""},
                "court_type": {"value": row.get("court_type") or ""},
                "year":       {"value": str(row.get("year") or "")},
                "text":       {"value": row["text"][:TEXT_TRUNCATE]},
            }
        payload = {"vertices": {"LegalCase": vertices}}
        resp = requests.post(
            f"{TG_HOST}/restpp/graph/{GRAPH_NAME}",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"  Batch {i}-{i+len(batch_ids)} FAILED: {resp.status_code} {resp.text[:300]}")
        else:
            print(f"  Upserted vertices {i + len(batch_ids)}/{total}", flush=True)


def upsert_edges(token: str, valid_ids: set[str]):
    edges = []
    with open(CITATIONS_FILE, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for citing, cited in r:
            if citing in valid_ids and cited in valid_ids:
                edges.append((citing, cited))

    total = len(edges)
    for i in range(0, total, BATCH_SIZE):
        batch = edges[i:i + BATCH_SIZE]
        cites_map = {}
        for citing, cited in batch:
            cites_map.setdefault(citing, {})[cited] = {}
        payload = {"edges": {"LegalCase": {}}}
        # REST++ upsert edge format: edges.<FromType>.<fromId>.<EdgeType>.<ToType>.<toId>
        edge_payload = {"LegalCase": {}}
        for citing, targets in cites_map.items():
            edge_payload["LegalCase"][citing] = {"CITES": {"LegalCase": {t: {} for t in targets}}}
        resp = requests.post(
            f"{TG_HOST}/restpp/graph/{GRAPH_NAME}",
            headers={"Authorization": f"Bearer {token}"},
            json={"edges": edge_payload},
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"  Edge batch {i}-{i+len(batch)} FAILED: {resp.status_code} {resp.text[:300]}")
        else:
            print(f"  Upserted edges {i + len(batch)}/{total}", flush=True)


def main(install_schema_flag: bool, full: bool):
    if install_schema_flag:
        install_schema()
        return

    if full:
        opinions = load_all_opinions()
        print(f"Full corpus: {len(opinions):,} opinions from {DATASET_FILE}")
    else:
        node_ids = load_citation_graph_node_ids()
        print(f"Pilot slice: {len(node_ids):,} opinion IDs from {CITATIONS_FILE}")
        opinions = load_pilot_opinions(node_ids)
        print(f"Matched {len(opinions):,} opinions in {DATASET_FILE}")
        missing = node_ids - set(opinions.keys())
        if missing:
            print(f"WARNING: {len(missing)} citation-graph node IDs not found in dataset (skipped)")

    token = get_token()

    print("\n--- Upserting LegalCase vertices ---")
    upsert_vertices(token, opinions)

    # Edge count is unchanged between modes: citations.csv was already built against the
    # full corpus, so every edge whose endpoints are present gets loaded either way. In
    # --full mode all 9,632 endpoints are present; in pilot mode the subset is exactly
    # the citation-connected opinions, so the same edges load. upsert is idempotent, so
    # re-running --full over an existing pilot ingest simply adds the ~51k unconnected
    # opinions without duplicating anything.
    print("\n--- Upserting CITES edges ---")
    upsert_edges(token, set(opinions.keys()))

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-schema", action="store_true")
    parser.add_argument("--full", action="store_true",
                        help="Ingest ALL opinions, not just the citation-connected pilot subset")
    args = parser.parse_args()
    main(args.install_schema, args.full)
