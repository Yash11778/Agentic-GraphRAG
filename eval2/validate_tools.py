"""Tool-layer sufficiency: can the graph primitives answer the public set from the
QUESTION ALONE, with no gold document ids?

validate_schema.py proved the schema holds the answers. This proves the composed
primitives can REACH them. The predicate extraction here is regex-based and is a
stand-in for the orchestrator's planner -- this file is a test harness, never part
of a pipeline, so it does not violate LOCKED-1 (no question-type handlers in the
scored path). Its job is to fail loudly if a primitive is missing, before the
agent is built on top.

    python eval2/validate_tools.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentic.ingest.parse_infobox import normalize_event_name
from agentic.tools.backend import LocalGraphBackend, fold

PUBLIC = ROOT / "hackathon-resources/questions/eval_public.jsonl"
B = LocalGraphBackend()

SPORTS = {fold(s): s for s in B.vocabulary("Sport", "name")}
SEASONS = ("Summer", "Winter")


def norm(s) -> str:
    return re.sub(r"[^a-z0-9' -]+", " ", fold(s)).strip()


def find_games(q: str) -> str | None:
    m = re.search(r"\b(\d{4})\s+(Summer|Winter)\s+Olympics", q, re.I)
    return f"{m.group(1)}-{m.group(2).capitalize()}" if m else None


def find_sport(q: str) -> str | None:
    qf = fold(q)
    # Longest match first: "Beach volleyball" must win over "Volleyball".
    for key in sorted(SPORTS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", qf):
            return SPORTS[key]
    return None


def medal_winner(event: dict, medal: str = "gold") -> str | None:
    """The medal cell verbatim -- see the gold_raw comment in build_graph.py."""
    return event.get(f"{medal}_raw")


def asked_medal(q: str) -> str:
    ql = q.lower()
    return "silver" if "silver" in ql else "bronze" if "bronze" in ql else "gold"


# ── per-type resolution, each composed only from backend primitives ───────────
def solve_aggregation(q: str):
    m = re.search(r"more than (\d+)\s+(competitors|nations)", q, re.I)
    if not m:
        return None
    thresh, field = int(m.group(1)), m.group(2).lower()
    preds = [(field, "gt", thresh)]
    games, sport = find_games(q), find_sport(q)
    if games:
        preds.append(("games_id", "eq", games))
    if sport:
        preds.append(("sport", "eq", sport))
    return str(len(B.find("Event", preds)))


def solve_superlative(q: str):
    m = re.search(r"(highest|largest|most|lowest|smallest|fewest)\s+number\s+of\s+"
                  r"(competitors|nations)", q, re.I)
    if not m:
        return None
    lowest = m.group(1).lower() in ("lowest", "smallest", "fewest")
    field = m.group(2).lower()
    preds = [(field, "exists", True)]
    games, sport = find_games(q), find_sport(q)
    if games:
        preds.append(("games_id", "eq", games))
    if sport:
        preds.append(("sport", "eq", sport))
    cands = B.find("Event", preds)
    if not cands:
        return None
    best = (min if lowest else max)(cands, key=lambda v: v[field])
    return best["title"]


def solve_lookup(q: str):
    m = re.search(r"how many (nations|competitors) (?:competed|participated|took part)?"
                  r".*?in\s+(.+?)\s*\??$", q, re.I | re.S)
    if not m:
        return None
    field, title = m.group(1).lower(), m.group(2).strip().rstrip("?")
    hits = B.find("Event", [("title", "eq", title)])
    if not hits:
        hits = [v for v in B.find("Event", [("title", "contains", title[:40])])]
    if not hits:
        return None
    value = hits[0].get(field)
    return str(value) if value is not None else None


def solve_temporal(q: str):
    """'... at the Summer Olympics held immediately before 2016' -> step the chain."""
    m = re.search(r"immediately (before|after) (\d{4})", q, re.I)
    if not m:
        return None
    direction, year = m.group(1).lower(), int(m.group(2))
    season = next((s for s in SEASONS if fold(s) in fold(q)), None)
    if season is None:
        anchor = B.find("Games", [("year", "eq", year)])
        season = anchor[0]["season"] if anchor else "Summer"
    anchor_id = f"{year}-{season}"
    step = B.neighbors("Games", anchor_id, ["PRECEDED_BY"],
                       "out" if direction == "before" else "in")
    if not step:
        return None
    target = step[0]["vertex"]["id"]
    # The event phrase is everything between "the" and "event at the Summer..."
    m2 = re.search(r"(?:medal in|in) the (.+?)\s+(?:event|competition)\b", q, re.I)
    phrase = m2.group(1) if m2 else ""
    sport = find_sport(q)
    preds = [("games_id", "eq", target)]
    if sport:
        preds.append(("sport", "eq", sport))
    cands = B.find("Event", preds)
    if sport:
        phrase = re.sub(rf"\b{re.escape(sport.lower())}\b", "", phrase, flags=re.I)
    phrase_f = normalize_event_name(phrase)
    best = next((c for c in cands
                 if phrase_f in (c.get("event_key"), c.get("title_event_key"))), None)
    if best is None:
        best = next((c for c in cands if phrase_f and
                     (phrase_f in (c.get("event_key") or "")
                      or phrase_f in (c.get("title_event_key") or ""))), None)
    if best is None:
        return None
    return medal_winner(best, asked_medal(q))


def solve_multihop(q: str):
    """'the event held at VENUE on DATE' -> reverse lookup, then the medal edge."""
    m = re.search(r"held at (?:the )?(.+?) on (.+?)(?:\s+at the (\d{4})\s+(Summer|Winter)"
                  r"\s+Olympics)?\s*\??$", q, re.I)
    if not m:
        return None
    venue, date_raw, year, season = (m.group(1).strip(), m.group(2).strip(),
                                     m.group(3), m.group(4))
    preds = [("venue", "eq", venue)]
    if year and season:
        preds.append(("games_id", "eq", f"{year}-{season}"))
    cands = B.find("Event", preds)
    if not cands:
        cands = B.find("Event", [("venue", "contains", venue)])
    date_f = norm(date_raw)
    exact = [c for c in cands if norm(c.get("date_raw") or "") == date_f]
    if not exact:
        # The question may drop or add the year relative to the infobox string.
        stripped = norm(re.sub(r"\b(19|20)\d{2}\b", "", date_raw))
        exact = [c for c in cands
                 if norm(re.sub(r"\b(19|20)\d{2}\b", "", c.get("date_raw") or "")) == stripped]
    if len(exact) != 1:
        return None
    return medal_winner(exact[0], asked_medal(q))


SOLVERS = {"aggregation": solve_aggregation, "superlative": solve_superlative,
           "lookup": solve_lookup, "temporal": solve_temporal, "multi_hop": solve_multihop}


def main():
    qs = [json.loads(l) for l in PUBLIC.open(encoding="utf-8")]
    res = defaultdict(Counter)
    misses = []
    for q in qs:
        got = SOLVERS[q["qtype"]](q["question"])
        want = q["answer"][0]
        ok = got is not None and norm(got) == norm(want)
        res[q["qtype"]]["total"] += 1
        res[q["qtype"]]["ok" if ok else ("none" if got is None else "wrong")] += 1
        if not ok:
            misses.append((q, got))

    print(f"{'qtype':<14}{'total':>7}{'solved':>8}{'wrong':>7}{'unreached':>11}")
    tot = Counter()
    for t, c in sorted(res.items()):
        tot.update(c)
        print(f"{t:<14}{c['total']:>7}{c['ok']:>8}{c['wrong']:>7}{c['none']:>11}")
    print(f"{'ALL':<14}{tot['total']:>7}{tot['ok']:>8}{tot['wrong']:>7}{tot['none']:>11}")
    print(f"\ntool-layer ceiling: {100*tot['ok']/tot['total']:.1f}%")
    for q, got in misses[:15]:
        print(f"\n[{q['qtype']}] {q['qid']}: {q['question'][:100]}")
        print(f"   want={q['answer'][0]!r}  got={got!r}")


if __name__ == "__main__":
    main()
