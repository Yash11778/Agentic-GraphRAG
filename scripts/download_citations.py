"""
Build the citation graph (CITES edges) for our legal corpus from
CourtListener's public bulk citation-map file. No API token needed —
anonymous S3 download, no rate limit.

Keeps only edges where BOTH the citing and cited opinion are in our corpus
(data/raw/dataset_100m_enriched.jsonl, built by build_dataset_from_pile_of_law.py
+ enrich_dataset_metadata.py) -- edges pointing outside our corpus are
dangling for our graph and dropped. This is the actual multi-hop backbone:
CITES edges let GraphRAG traverse "what did case X rely on, and was that
later overruled" chains that flat vector search over Basic RAG cannot do.

Usage:
    python scripts/download_citations.py
"""
import csv
import json
from pathlib import Path

from tqdm import tqdm

from _bulk_csv import latest_bulk_file, stream_csv_dicts, BUCKET

ROOT = Path(__file__).parent.parent.resolve()
OPINIONS_FILE = ROOT / "data" / "raw" / "dataset_100m_enriched.jsonl"
OUT_FILE = ROOT / "data" / "raw" / "citations.csv"


def load_opinion_ids() -> set[str]:
    if not OPINIONS_FILE.exists():
        raise RuntimeError(f"{OPINIONS_FILE} not found — run download_dataset.py and "
                            f"enrich_dataset_metadata.py first")
    ids = set()
    with open(OPINIONS_FILE, encoding="utf-8") as f:
        for line in f:
            ids.add(str(json.loads(line)["id"]))
    return ids


def main():
    keep_ids = load_opinion_ids()
    print(f"Loaded {len(keep_ids):,} opinion IDs from {OPINIONS_FILE}")

    key = latest_bulk_file("citation-map")
    url = f"{BUCKET}/{key}"
    print(f"Streaming: {url}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    scanned = kept = 0
    with open(OUT_FILE, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["citing_opinion_id", "cited_opinion_id"])
        for row in tqdm(stream_csv_dicts(url), desc="Scanning citation-map", unit="rows"):
            scanned += 1
            citing_id = row.get("citing_opinion_id")
            cited_id = row.get("cited_opinion_id")
            if citing_id in keep_ids and cited_id in keep_ids:
                writer.writerow([citing_id, cited_id])
                kept += 1
            if scanned % 2_000_000 == 0:
                print(f"  >> scanned {scanned:,} | kept {kept:,}", flush=True)

    print(f"Scanned {scanned:,} citation-map rows, kept {kept:,} in-corpus edges")
    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
