"""Backend parity: the local backend and TigerGraph must answer identically.

The benchmark's central claim is that any difference between the three pipelines
comes from control flow, because they share one tool layer (LOCKED-3). That claim
is only worth anything if the tool layer itself returns the same thing wherever it
reads from. This file is the evidence: a battery of calls issued to both backends
and compared value by value.

It is a check, not a pipeline. Nothing here is on the scored path.

    python eval2/validate_backends.py            # summary
    python eval2/validate_backends.py --verbose  # show every difference
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentic.tools.backend import LocalGraphBackend
from agentic.tools.tigergraph_backend import TigerGraphBackend

# Each case is (name, callable taking a backend). Chosen to exercise every
# translation the TigerGraph backend performs: folded string comparison, the
# absence sentinels, integer predicates, reverse-edge traversal, and the two
# endpoint pairs of PRECEDED_BY.
CASES: list[tuple[str, Callable[[Any], Any]]] = [
    # get
    ("get/event", lambda b: b.get("Event", "Q303623")),
    ("get/athlete-accented", lambda b: b.get("Athlete", "roland-kkny")),
    ("get/games", lambda b: b.get("Games", "2012-Summer")),
    ("get/missing", lambda b: b.get("Event", "does-not-exist")),

    # find: string predicates, including accents and case
    ("find/sport-eq", lambda b: b.find("Event", [("sport", "eq", "Shooting"),
                                                 ("year", "eq", 2004)])),
    ("find/venue-accent", lambda b: b.find("Event", [("venue", "eq", "Arena Olímpica do Rio")])),
    ("find/venue-unaccented-query",
     lambda b: b.find("Event", [("venue", "eq", "Arena Olimpica do Rio")])),
    ("find/venue-contains", lambda b: b.find("Event", [("venue", "contains", "Eton")])),
    ("find/title-contains-dash",
     lambda b: b.find("Event", [("title", "contains", "Men's K-2")])),
    ("find/athlete-name", lambda b: b.find("Athlete", [("name", "eq", "Kökény")])),

    # find: the predicate semantics added in the final pass -- a numeric
    # threshold written as a string, and token-wise contains over a venue the
    # corpus writes with a joined word. Both must read identically on both sides.
    ("find/competitors-gt-string",
     lambda b: b.find("Event", [("games_id", "eq", "2014-Winter"),
                                ("sport", "eq", "Alpine skiing"), ("competitors", "gt", "63")])),
    ("find/venue-contains-tokens",
     lambda b: b.find("Event", [("venue", "contains", "Xiaohaituo Bobsleigh and Luge Track Beijing")])),
    ("find/date-contains-tokens-reordered",
     lambda b: b.find("Event", [("venue", "contains", "TechnologyUniversity Gymnasium"),
                                ("date_raw", "contains", "12 August 2008")])),

    # find: integer predicates and absence
    ("find/competitors-gt", lambda b: b.find("Event", [("competitors", "gt", 100)])),
    ("find/competitors-lt", lambda b: b.find("Event", [("competitors", "lt", 5)])),
    ("find/nations-between", lambda b: b.find("Event", [("nations", "gte", 80),
                                                        ("nations", "lte", 85)])),
    ("find/venue-absent", lambda b: b.find("Event", [("venue", "exists", False)])),
    ("find/win-value-exists", lambda b: b.find("Event", [("win_value", "exists", True),
                                                         ("year", "eq", 1988)])),

    # aggregate: the reductions the coverage gate and superlatives rely on
    ("agg/count-shooting", lambda b: b.aggregate("Event", [("sport", "eq", "Shooting"),
                                                           ("year", "eq", 2004),
                                                           ("season", "eq", "Summer"),
                                                           ("competitors", "gt", 37)],
                                                 "competitors")),
    ("agg/max-competitors", lambda b: b.aggregate("Event", [("year", "eq", 2012)], "competitors")),
    ("agg/empty-set", lambda b: b.aggregate("Event", [("year", "eq", 1800)], "competitors")),
    ("agg/no-field", lambda b: b.aggregate("Event", [("season", "eq", "Winter")])),

    # neighbours: out, in, both, filtered and unfiltered
    ("nbr/event-out", lambda b: b.neighbors("Event", "Q303623", None, "out")),
    ("nbr/event-gold", lambda b: b.neighbors("Event", "Q303623", ["WON_GOLD"], "out")),
    ("nbr/venue-in", lambda b: b.neighbors("Venue", "eton-dorney", ["HELD_AT"], "in")),
    ("nbr/athlete-both", lambda b: b.neighbors("Athlete", "roland-kkny", None, "both")),
    ("nbr/games-chain", lambda b: b.neighbors("Games", "2016-Summer", ["PRECEDED_BY"], "out")),
    ("nbr/event-chain-in", lambda b: b.neighbors("Event", "Q303623", ["PRECEDED_BY"], "in")),

    # vocabulary and counts
    ("vocab/sport", lambda b: b.vocabulary("Sport", "name")),
    ("vocab/season", lambda b: b.vocabulary("Games", "season")),
    ("count/event", lambda b: b.count("Event")),
    ("count/athlete", lambda b: b.count("Athlete")),
]


# Similarity retrieval is compared separately and by overlap, not equality. The
# local backend scans every vector exactly; TigerGraph answers from an
# approximate index. Demanding identical top-k would either fail for a reason
# that says nothing about correctness, or force the local side to imitate the
# index's approximation and stop being an oracle.
VECTOR_QUERIES = [
    "Who won the gold medal in the men's K-2 1000 metres canoe sprint in 2012?",
    "biathlon events at the 2018 Winter Olympics",
    "Which venue hosted the Olympic tennis tournament in 2004?",
]
VECTOR_K = 10
VECTOR_MIN_OVERLAP = 8


def check_vectors(local: Any, remote: Any, verbose: bool) -> list[tuple]:
    from agentic.tools.embedder import embed_query

    print(f"\n{'vector query':<44}{'overlap':>9}   status")
    failures = []
    for query in VECTOR_QUERIES:
        vector = embed_query(query)
        local_ids = [h["chunk"]["id"] for h in local.vector_search(vector, VECTOR_K)]
        remote_ids = [h["chunk"]["id"] for h in remote.vector_search(vector, VECTOR_K)]
        overlap = len(set(local_ids) & set(remote_ids))
        ok = overlap >= VECTOR_MIN_OVERLAP
        if not ok:
            failures.append((f"vector/{query[:30]}", local_ids, remote_ids))
        print(f"{query[:42]:<44}{overlap:>4}/{VECTOR_K}   {'ok' if ok else 'TOO LOW'}")
        if verbose:
            print("    local :", local_ids)
            print("    remote:", remote_ids)
    return failures


def canonical(value: Any) -> Any:
    """Order-insensitive form, since neither backend promises a result order."""
    if isinstance(value, list):
        return sorted((canonical(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True,
                                                                             default=str))
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in sorted(value.items())}
    return value


def describe(value: Any) -> str:
    if isinstance(value, list):
        return f"{len(value)} rows"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)[:120]
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true", help="print full differences")
    args = ap.parse_args()

    local = LocalGraphBackend()
    remote = TigerGraphBackend()

    failures = []
    print(f"{'case':<28}{'local':>12}{'tigergraph':>14}   status")
    for name, call in CASES:
        try:
            got_local = canonical(call(local))
            got_remote = canonical(call(remote))
        except Exception as exc:  # a raised difference is still a difference
            failures.append((name, "exception", repr(exc)))
            print(f"{name:<28}{'-':>12}{'-':>14}   ERROR {type(exc).__name__}")
            continue
        same = got_local == got_remote
        if not same:
            failures.append((name, got_local, got_remote))
        print(f"{name:<28}{describe(got_local):>12}{describe(got_remote):>14}"
              f"   {'ok' if same else 'DIFFERS'}")

    failures += check_vectors(local, remote, args.verbose)

    total = len(CASES) + len(VECTOR_QUERIES)
    print(f"\n{total - len(failures)}/{total} checks agree")
    if failures and args.verbose:
        for name, got_local, got_remote in failures:
            print(f"\n=== {name}")
            print("  local     :", json.dumps(got_local, default=str)[:1500])
            print("  tigergraph:", json.dumps(got_remote, default=str)[:1500])
    if failures:
        raise SystemExit(f"{len(failures)} backend differences")


if __name__ == "__main__":
    main()
