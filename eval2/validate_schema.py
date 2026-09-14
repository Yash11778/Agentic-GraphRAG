"""Schema-sufficiency check: can the LOCKED graph schema produce every public answer?

This runs BEFORE any of the graph, tools or agent are built. It asks one question:
using only the fields the parser extracts from the gold documents, is each of the
100 public answers derivable? A field the schema omits shows up here as a miss,
which is far cheaper to find now than after ingestion.

This is a ceiling measurement, not a pipeline: it is handed the gold document ids,
so it tests the SCHEMA, not retrieval.

    python eval2/validate_schema.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentic.ingest.parse_infobox import (
    is_olympic_event,
    parse_count,
    parse_infobox,
    parse_title,
)

CORPUS = ROOT / "hackathon-resources/corpus/corpus.jsonl"
PUBLIC = ROOT / "hackathon-resources/questions/eval_public.jsonl"


def norm(s: str) -> str:
    """Answer-comparison normalisation: unicode dashes, case, punctuation, spacing."""
    s = s.replace("–", "-").replace("—", "-").replace("’", "'")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9'\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load():
    docs = {}
    for line in CORPUS.open(encoding="utf-8"):
        d = json.loads(line)
        d["fields"] = parse_infobox(d["text"])
        d["is_event"] = is_olympic_event(d["text"])
        d["parsed_title"] = parse_title(d["title"]) or {}
        docs[d["doc_id"]] = d
    qs = [json.loads(l) for l in PUBLIC.open(encoding="utf-8")]
    return docs, qs


def check_medal(q, gold_docs):
    """lookup / multi_hop / temporal answers that name a medallist."""
    want = norm(q["answer"][0])
    for d in gold_docs:
        for key in ("gold", "silver", "bronze"):
            if key in d["fields"] and want in norm(d["fields"][key]):
                return True, f"{key} of {d['doc_id']}"
    return False, ""


def check_field(q, gold_docs):
    """lookup answers that name a count (nations / competitors)."""
    want = norm(q["answer"][0])
    for d in gold_docs:
        for key in ("nations", "competitors", "win_value"):
            raw = d["fields"].get(key)
            if raw is None:
                continue
            value, _ = parse_count(raw)
            if want == norm(raw) or (value is not None and want == str(value)):
                return True, f"{key} of {d['doc_id']}"
    return False, ""


def check_aggregation(q, gold_docs):
    """'how many X events ... had more than N competitors' -> count over gold set."""
    m = re.search(r"more than (\d+)\s+(competitors|nations)", q["question"], re.I)
    if not m:
        return None, "unparsed-predicate"
    threshold, field = int(m.group(1)), m.group(2).lower()
    hits = 0
    for d in gold_docs:
        value, _ = parse_count(d["fields"].get(field))
        if value is not None and value > threshold:
            hits += 1
    ok = str(hits) == q["answer"][0].strip()
    return ok, f"{field}>{threshold} -> {hits}"


def check_superlative(q, gold_docs):
    """'which X event had the highest number of competitors' -> argmax over gold set."""
    m = re.search(r"(highest|lowest|most|fewest).*?(competitors|nations)", q["question"], re.I)
    field = m.group(2).lower() if m else "competitors"
    lowest = bool(m and m.group(1).lower() in ("lowest", "fewest"))
    best, best_doc = None, None
    for d in gold_docs:
        value, _ = parse_count(d["fields"].get(field))
        if value is None:
            continue
        if best is None or (value < best if lowest else value > best):
            best, best_doc = value, d
    if best_doc is None:
        return False, "no numeric field"
    ok = norm(best_doc["title"]) == norm(q["answer"][0])
    return ok, f"argmax {field}={best} -> {best_doc['title'][:50]}"


CHECKERS = {
    "aggregation": check_aggregation,
    "superlative": check_superlative,
}


def main():
    docs, qs = load()
    n_events = sum(1 for d in docs.values() if d["is_event"])
    print(f"corpus: {len(docs)} docs, {n_events} olympic events\n")

    results = defaultdict(Counter)
    failures = []
    for q in qs:
        gold_docs = [docs[g] for g in q["gold_doc_ids"] if g in docs]
        qtype = q["qtype"]
        if qtype in CHECKERS:
            ok, detail = CHECKERS[qtype](q, gold_docs)
        else:
            ok, detail = check_medal(q, gold_docs)
            if not ok:
                ok, detail = check_field(q, gold_docs)
        results[qtype]["total"] += 1
        if ok is None:
            results[qtype]["unparsed"] += 1
            failures.append((q, detail))
        elif ok:
            results[qtype]["derivable"] += 1
        else:
            results[qtype]["MISS"] += 1
            failures.append((q, detail))

    print(f"{'qtype':<14}{'total':>7}{'derivable':>11}{'miss':>7}{'unparsed':>10}")
    tot = Counter()
    for qtype, c in sorted(results.items()):
        tot.update(c)
        print(f"{qtype:<14}{c['total']:>7}{c['derivable']:>11}{c['MISS']:>7}{c['unparsed']:>10}")
    print(f"{'ALL':<14}{tot['total']:>7}{tot['derivable']:>11}{tot['MISS']:>7}{tot['unparsed']:>10}")
    print(f"\nschema ceiling: {100*tot['derivable']/tot['total']:.1f}%")

    if failures:
        print(f"\n--- {len(failures)} not derivable ---")
        for q, detail in failures[:12]:
            print(f"[{q['qtype']}] {q['qid']} {q['question'][:95]}")
            print(f"    gold={q['answer']}  detail={detail}  n_gold_docs={len(q['gold_doc_ids'])}")


if __name__ == "__main__":
    main()
