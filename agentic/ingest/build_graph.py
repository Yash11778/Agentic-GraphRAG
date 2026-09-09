"""corpus.jsonl -> typed property graph (vertices.jsonl + edges.jsonl).

Implements the schema in docs/ARCHITECTURE.md §4. Deterministic (LOCKED-4): no
LLM, and nothing imputed -- a field the source omits is simply absent, so the
coverage numbers printed at the end are the real ones.

The output is backend-neutral on purpose: the same two files feed the local
in-memory backend used for development and the TigerGraph bulk loader used for
the scored run, so switching backends cannot change the data.

    python agentic/ingest/build_graph.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agentic.ingest.parse_infobox import (
    OLYMPIC_MARKER, _NAME_PREFIXES, infobox_type, normalize_event_name, parse_count,
    parse_games, parse_infobox, parse_title,
)

CORPUS = ROOT / "hackathon-resources/corpus/corpus.jsonl"
OUT_DIR = ROOT / "data/graph"

MEDALS = (("gold", "WON_GOLD"), ("silver", "WON_SILVER"), ("bronze", "WON_BRONZE"))

# Wikipedia infoboxes concatenate the members of a winning pair or team without a
# separator ("Rudolf DombiRoland Kökény"): a lower-case letter immediately
# followed by an upper-case one is the join point. Splitting matters because a
# medal edge must point at each athlete individually for athlete-level questions
# to work at all.
_NAME_JOIN_RE = re.compile(r"(?<=[a-zß])(?=[A-ZÁÀÂÄÃÅÇÉÈÊËÍÌÎÏÑÓÒÔÖÕÚÙÛÜÝŽŠĐ])")

# Some medal cells are a country rather than a person (team events), and some
# carry a trailing qualifier. Both are kept verbatim as the athlete name: the
# corpus is ground truth and "correcting" it would break exact-match scoring.


def slug(text: str) -> str:
    """Stable vertex id from a display name."""
    s = re.sub(r"\s+", "-", text.strip().lower())
    return re.sub(r"[^a-z0-9\-]", "", s) or "unknown"


def split_names(raw: str) -> list[str]:
    """Split a medal cell into individual athlete names.

    A split point is rejected when the capital belongs to a surname prefix
    ("MacLennan", "O'Brien"), which otherwise cuts one person into two.
    """
    parts: list[str] = []
    for chunk in re.split(r"\s*,\s*|\s+and\s+", raw.strip()):
        chunk = chunk.strip()
        if not chunk:
            continue
        pieces, last = [], 0
        for m in _NAME_JOIN_RE.finditer(chunk):
            i = m.start()
            tail = chunk[last:i]
            if any(tail.lower().endswith(p) for p in _NAME_PREFIXES):
                continue
            pieces.append(chunk[last:i])
            last = i
        pieces.append(chunk[last:])
        parts.extend(p.strip() for p in pieces if p.strip())
    return parts


def parse_dates(fields: dict) -> dict:
    """Unify `date` / `dates` into a raw string plus day/month bounds.

    Dates in this corpus are display strings ("6 to 8 August", "4–5 August 2012",
    "13 February 2010"), not ISO. The multi_hop questions quote them verbatim, so
    the raw string is what matching keys off; the parsed bounds exist for range
    queries and are left null when the string does not yield them.
    """
    raw = fields.get("date") or fields.get("dates")
    out = {"date_raw": raw, "date_start": None, "date_end": None, "date_month": None}
    if not raw:
        return out
    month = re.search(r"(January|February|March|April|May|June|July|August|September|"
                      r"October|November|December)", raw)
    if month:
        out["date_month"] = month.group(1)
    days = re.findall(r"\b(\d{1,2})\b(?!\d)", raw.replace(",", " "))
    days = [int(d) for d in days if 1 <= int(d) <= 31]
    if days:
        out["date_start"], out["date_end"] = days[0], days[-1]
    return out


def build():
    vertices: dict[tuple[str, str], dict] = {}
    edges: list[dict] = []
    coverage = Counter()
    # (sport, event_name, season) -> {year: event_id}, used to chain editions.
    chains: dict[tuple[str, str, str], dict[int, str]] = defaultdict(dict)

    def add_vertex(vtype: str, vid: str, **attrs):
        key = (vtype, vid)
        if key not in vertices:
            vertices[key] = {"type": vtype, "id": vid, **attrs}
        else:
            for k, v in attrs.items():
                vertices[key].setdefault(k, v)
        return vid

    def add_edge(etype: str, src_type: str, src: str, dst_type: str, dst: str, **attrs):
        edges.append({"type": etype, "from_type": src_type, "from": src,
                      "to_type": dst_type, "to": dst, **attrs})

    n_docs = n_events = 0
    for line in CORPUS.open(encoding="utf-8"):
        doc = json.loads(line)
        n_docs += 1
        fields = parse_infobox(doc["text"])
        is_event = doc["text"].startswith(OLYMPIC_MARKER)

        if not is_event:
            # Distractor documents (films, people, companies) still become Document
            # vertices so vector hits on them can be traced back and counted as the
            # false positives they are.
            add_vertex("Document", doc["doc_id"], title=doc["title"], url=doc.get("url", ""),
                       infobox=infobox_type(doc["text"]) or "", is_event=False,
                       approx_tokens=doc.get("approx_tokens", 0))
            continue

        n_events += 1
        title_parts = parse_title(doc["title"]) or {}
        games = parse_games(fields.get("games")) or (
            {"games_id": title_parts.get("games_id"),
             "year": title_parts.get("year"),
             "season": title_parts.get("season")} if title_parts else None)

        competitors, comp_unit = parse_count(fields.get("competitors"))
        nations, _ = parse_count(fields.get("nations"))
        dates = parse_dates(fields)

        event_id = doc["doc_id"]
        add_vertex(
            "Event", event_id,
            title=doc["title"],
            event_name=fields.get("event") or title_parts.get("event_name", ""),
            # The medal cell verbatim. Gold answers for team events ARE this
            # concatenated string ("Dani KingLaura TrottJoanna Rowsell"), so the
            # raw form is what an exact-match answer must come from; the split
            # Athlete vertices exist for athlete-level traversal, not for answers.
            gold_raw=fields.get("gold"),
            silver_raw=fields.get("silver"),
            bronze_raw=fields.get("bronze"),
            gold_noc=fields.get("goldNOC"),
            event_key=normalize_event_name(
                fields.get("event") or title_parts.get("event_name", "")),
            title_event_key=normalize_event_name(title_parts.get("event_name", "")),
            sport=title_parts.get("sport", ""),
            games_id=(games or {}).get("games_id", ""),
            year=int(games["year"]) if games and games.get("year") else None,
            season=(games or {}).get("season", ""),
            venue=fields.get("venue"),
            competitors=competitors, competitors_unit=comp_unit,
            nations=nations,
            win_value=fields.get("win_value"),
            url=doc.get("url", ""),
            approx_tokens=doc.get("approx_tokens", 0),
            is_event=True,
            **dates,
        )
        for field, present in (("venue", fields.get("venue")), ("competitors", competitors),
                               ("nations", nations), ("date_raw", dates["date_raw"]),
                               ("games", games), ("prev", fields.get("prev")),
                               ("next", fields.get("next"))):
            if present:
                coverage[field] += 1

        if games and games.get("games_id"):
            add_vertex("Games", games["games_id"], year=int(games["year"]),
                       season=games["season"],
                       name=f"{games['year']} {games['season']} Olympics")
            add_edge("AT_GAMES", "Event", event_id, "Games", games["games_id"])

        sport = title_parts.get("sport")
        if sport:
            add_vertex("Sport", slug(sport), name=sport)
            add_edge("IN_SPORT", "Event", event_id, "Sport", slug(sport))

        venue = fields.get("venue")
        if venue:
            add_vertex("Venue", slug(venue), name=venue)
            add_edge("HELD_AT", "Event", event_id, "Venue", slug(venue))

        for field, etype in MEDALS:
            raw = fields.get(field)
            if not raw:
                continue
            noc = fields.get(f"{field}NOC")
            for name in split_names(raw):
                aid = slug(name)
                add_vertex("Athlete", aid, name=name)
                add_edge(etype, "Event", event_id, "Athlete", aid, medal=field, noc=noc or "")
                if noc:
                    add_vertex("Nation", noc.upper(), code=noc.upper())
                    add_edge("REPRESENTS", "Athlete", aid, "Nation", noc.upper(),
                             event_id=event_id, medal=field)

        # Edition chain: keyed on the event identity (sport + event name + season)
        # so "Men's 100 metres" links across Games without colliding with the
        # women's event or with a same-named winter event.
        if title_parts and games:
            key = (title_parts.get("sport", ""),
                   (fields.get("event") or title_parts.get("event_name", "")).lower(),
                   games["season"])
            chains[key][int(games["year"])] = event_id

    # Games edition chain (season-scoped), from the Games vertices themselves.
    games_by_season: dict[str, dict[int, str]] = defaultdict(dict)
    for (vtype, vid), v in vertices.items():
        if vtype == "Games":
            games_by_season[v["season"]][v["year"]] = vid
    for season, by_year in games_by_season.items():
        years = sorted(by_year)
        for prev_year, year in zip(years, years[1:]):
            add_edge("PRECEDED_BY", "Games", by_year[year], "Games", by_year[prev_year],
                     gap_years=year - prev_year)

    # Event edition chain.
    n_chain = 0
    for key, by_year in chains.items():
        years = sorted(by_year)
        for prev_year, year in zip(years, years[1:]):
            add_edge("PRECEDED_BY", "Event", by_year[year], "Event", by_year[prev_year],
                     gap_years=year - prev_year)
            n_chain += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "vertices.jsonl").open("w", encoding="utf-8") as f:
        for v in vertices.values():
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    with (OUT_DIR / "edges.jsonl").open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    by_type = Counter(v["type"] for v in vertices.values())
    by_edge = Counter(e["type"] for e in edges)
    print(f"docs={n_docs}  olympic events={n_events}\n")
    print("vertices:")
    for t, c in by_type.most_common():
        print(f"  {t:<10} {c:>6}")
    print("\nedges:")
    for t, c in by_edge.most_common():
        print(f"  {t:<12} {c:>6}")
    print(f"\nevent-edition links: {n_chain}")
    print("\nfield coverage over events:")
    for field in ("games", "venue", "competitors", "nations", "date_raw", "prev", "next"):
        c = coverage[field]
        print(f"  {field:<12} {c:>6}  ({100*c/n_events:.1f}%)")
    print(f"\nwrote {OUT_DIR}/vertices.jsonl, {OUT_DIR}/edges.jsonl")


if __name__ == "__main__":
    build()
