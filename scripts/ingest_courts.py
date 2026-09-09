"""
Enrich the knowledge graph with Court vertices and DECIDED_BY edges.

The core citation graph is LegalCase --CITES--> LegalCase. This script adds the
second entity type so the schema reads as a genuine multi-entity knowledge graph
(what a reviewer sees in TigerGraph's Design Schema view):

    LegalCase --CITES--> LegalCase        (which case cites which)
    LegalCase --DECIDED_BY--> Court        (which court decided each case)

Every opinion has a court (100% coverage, 1,413 distinct courts). Court identity is
the court_name (falling back to court_id), with court_type as an attribute.

Run add_court_schema.gsql first (adds the Court vertex + DECIDED_BY edge), then:
    python scripts/ingest_courts.py
"""
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent.resolve()
DATASET_FILE = ROOT / "data" / "raw" / "dataset_100m_enriched.jsonl"

TG_HOST = os.environ["TG_HOST"]
TG_SECRET = os.environ["TG_PASSWORD"]
GRAPH_NAME = os.environ.get("TG_GRAPH", "LegalGraph")
BATCH_SIZE = 1000


def get_token() -> str:
    resp = requests.post(
        f"{TG_HOST}/gsql/v1/tokens",
        json={"secret": TG_SECRET, "graph": GRAPH_NAME},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["token"]


def court_key(row: dict) -> str | None:
    return (row.get("court_name") or row.get("court_id") or "").strip() or None


def load() -> tuple[dict, list]:
    """Return (courts: key->court_type, case_to_court: [(case_id, court_key)])."""
    courts: dict[str, str] = {}
    case_to_court: list[tuple[str, str]] = []
    with open(DATASET_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = court_key(row)
            if not key:
                continue
            courts.setdefault(key, row.get("court_type") or "")
            case_to_court.append((row["id"], key))
    return courts, case_to_court


def upsert_courts(token: str, courts: dict):
    items = list(courts.items())
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        vertices = {
            key: {
                "court_name":  {"value": key[:500]},
                "court_type":  {"value": ctype},
            }
            for key, ctype in batch
        }
        resp = requests.post(
            f"{TG_HOST}/restpp/graph/{GRAPH_NAME}",
            headers={"Authorization": f"Bearer {token}"},
            json={"vertices": {"Court": vertices}},
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"  Court batch {i} FAILED: {resp.status_code} {resp.text[:200]}")
        else:
            print(f"  Upserted courts {i + len(batch)}/{len(items)}", flush=True)


def upsert_decided_by(token: str, case_to_court: list):
    total = len(case_to_court)
    for i in range(0, total, BATCH_SIZE):
        batch = case_to_court[i:i + BATCH_SIZE]
        edge_payload = {"LegalCase": {}}
        for case_id, key in batch:
            edge_payload["LegalCase"].setdefault(case_id, {}).setdefault(
                "DECIDED_BY", {}).setdefault("Court", {})[key] = {}
        resp = requests.post(
            f"{TG_HOST}/restpp/graph/{GRAPH_NAME}",
            headers={"Authorization": f"Bearer {token}"},
            json={"edges": edge_payload},
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"  DECIDED_BY batch {i} FAILED: {resp.status_code} {resp.text[:200]}")
        else:
            print(f"  Upserted DECIDED_BY {i + len(batch)}/{total}", flush=True)


def main():
    courts, case_to_court = load()
    print(f"Courts: {len(courts):,} | DECIDED_BY edges: {len(case_to_court):,}")
    token = get_token()
    print("\n--- Upserting Court vertices ---")
    upsert_courts(token, courts)
    print("\n--- Upserting DECIDED_BY edges ---")
    upsert_decided_by(token, case_to_court)
    print("\nDone.")


if __name__ == "__main__":
    main()
